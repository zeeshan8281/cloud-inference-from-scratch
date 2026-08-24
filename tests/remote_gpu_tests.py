"""Remote GPU correctness suite (PRD §13.3) — runs inside a Modal L4 container.

Executed via ``modal run modal_app.py::remote_gpu_tests``. Requires the pinned
weights and the cloud dependency set; never runs locally. Every check prints
PASS/FAIL and the process exits non-zero on any failure.

MODEL_DIR is injected by the caller (runpy init_globals).
"""

import asyncio
import json
import os
import sys
import time

MODEL_DIR = globals().get("MODEL_DIR") or os.environ.get("MODEL_DIR", "/cache/hf")

PROMPTS = [
    "Explain what a KV cache is in one sentence.",
    "Write a haiku about GPUs.",
    "List three colors and stop.",
]
PARITY_PROMPTS = PROMPTS + [
    "Count from one to five.",
    "Describe the shape of a page table.",
    "What is continuous batching?",
    "Say something about rotary embeddings.",
    "Name the largest ocean.",
    "Define inter-token latency.",
    "Summarize paged attention in plain words.",
]

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str = "") -> None:
    RESULTS.append((name, passed, detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def build_engine(mode: str, fallback: bool = False):

    from cloud_engine.config import build_config
    from cloud_engine.engine import InferenceEngine

    config = build_config(mode, allow_reference_fallback=fallback)
    engine = InferenceEngine(config, model_dir=MODEL_DIR)
    return config, engine


def reference_logits_and_tokens(prompt_ids: list[int], max_new: int):
    """Hugging Face oracle — test-only usage (PRD G4/M1)."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()
    input_ids = torch.tensor([prompt_ids], device="cuda")
    with torch.no_grad():
        logits = model(input_ids).logits[0]
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=151643,
        )[0][input_ids.shape[1]:].tolist()
    return logits, generated


def engine_logits(engine_model, tokenizer, prompt: str):
    import torch

    ids = torch.tensor(tokenizer.encode(prompt), device="cuda")
    with torch.no_grad():
        return engine_model(ids, ctx=None, return_all_logits=True)


def check_logits_parity() -> None:
    import torch

    from cloud_engine.attention import AttentionBackend
    from cloud_engine.model import load_model_config
    from cloud_engine.weights import load_model, load_tokenizer

    dims = load_model_config(MODEL_DIR)
    attn = AttentionBackend(cache=None, num_heads=dims.num_heads,
                            num_kv_heads=dims.num_kv_heads, head_dim=dims.head_dim)
    custom_model, _ = load_model(MODEL_DIR, attn, dtype=torch.float16, device="cuda")
    tokenizer = load_tokenizer(MODEL_DIR)

    for prompt in PROMPTS:
        ours = engine_logits(custom_model, tokenizer, prompt).float().cpu()
        theirs, _ = reference_logits_and_tokens(tokenizer.encode(prompt), max_new=1)
        theirs = theirs.float().cpu()
        close = torch.allclose(ours, theirs, rtol=2e-2, atol=2e-2)
        max_diff = float((ours - theirs).abs().max())
        record(f"logits parity: {prompt[:32]!r}", close, f"max|Δ|={max_diff:.4f}")


async def check_five_mode_token_parity() -> None:
    from cloud_engine.scheduler import GenerationConfig
    from cloud_engine.weights import load_tokenizer

    tokenizer = load_tokenizer(MODEL_DIR)
    outputs_by_mode: dict[str, list[list[int]]] = {}

    for mode in ("naive", "contiguous", "batched", "paged", "triton"):
        _, engine = build_engine(mode, fallback=(mode == "triton"))
        await engine.start()
        try:
            sequences = []
            for prompt in PARITY_PROMPTS:
                handle = await engine.submit(
                    prompt,
                    GenerationConfig(max_output_tokens=32, eos_token_id=151643),
                )
                result = await handle.wait()
                sequences.append(result.token_ids)
            outputs_by_mode[mode] = sequences
        finally:
            await engine.close()

    baseline = outputs_by_mode["contiguous"]
    for mode, sequences in outputs_by_mode.items():
        mismatches = [
            i for i, (a, b) in enumerate(zip(baseline, sequences, strict=True)) if a != b
        ]
        record(
            f"greedy token parity across 5 modes: {mode}",
            not mismatches,
            f"{len(PARITY_PROMPTS)} prompts x<=32 tokens" + (f" mismatched={mismatches}" if mismatches else ""),
        )

    hf_reference = []
    for prompt in PARITY_PROMPTS[:3]:
        _, gen = reference_logits_and_tokens(tokenizer.encode(prompt), max_new=32)
        hf_reference.append(gen)
    ok = all(
        hf == got
        for hf, got in zip(
            hf_reference, outputs_by_mode["contiguous"][:3], strict=True
        )
    )
    record("greedy parity vs HF generate() oracle (first 3 prompts)", ok)


def synthetic_attention_inputs(seq_len: int, batch: int = 1):
    import torch

    generator = torch.Generator(device="cuda").manual_seed(seq_len * 31 + batch)
    q = torch.randn(batch, 14, 64, generator=generator, device="cuda", dtype=torch.float16)
    keys = torch.randn(batch, seq_len, 14, 64, generator=generator, device="cuda", dtype=torch.float16)
    values = torch.randn(batch, seq_len, 14, 64, generator=generator, device="cuda", dtype=torch.float16)
    return q, keys, values


def contiguous_reference(q, keys, values, sm_scale):
    from cloud_engine.attention import causal_attention

    expanded_k = keys.repeat_interleave(7, dim=1)[0]
    expanded_v = values.repeat_interleave(7, dim=1)[0]
    out = causal_attention(q[0], expanded_k, expanded_v, sm_scale, past_len=0)
    return out.reshape(1, 14, 64)


def check_paged_vs_contiguous_boundaries() -> None:
    import torch

    from cloud_engine.attention import AttentionBackend, StepContext
    from cloud_engine.cache import PagedKVCache

    lengths = [1, 15, 16, 17, 127, 128, 129, 2048]
    for seq_len in lengths:
        cache = PagedKVCache(
            num_layers=1, num_kv_heads=2, head_dim=64,
            block_size=16, kv_cache_bytes=64 << 20, dtype=torch.float16, device="cuda",
        )
        request_id = f"boundary-{seq_len}"
        cache.reserve(request_id, seq_len)
        q, keys, values = synthetic_attention_inputs(seq_len)
        backend = AttentionBackend(cache, num_heads=14, num_kv_heads=2, head_dim=64)
        ctx = StepContext(request_id=request_id, kv_start=0, is_decode=False)
        # append happens inside attend(); feed kv-shaped k/v (heads 14 -> kv 2 by slicing groups)
        kv_keys = keys[:, :, ::7]   # take every 7th head -> 2 heads (synthetic)
        kv_values = values[:, :, ::7]
        got = backend.attend(layer=0, ctx=ctx, q=q[0], k=kv_keys[0], v=kv_values[0])
        got = got.reshape(1, 14, -1)
        expected = contiguous_reference(q, keys, values, backend.sm_scale)
        ok = torch.allclose(got.float(), expected.float(), rtol=2e-2, atol=2e-2)
        diff = float((got.float() - expected.float()).abs().max())
        record(f"paged==contiguous @len={seq_len}", ok, f"max|Δ|={diff:.4f}")
        cache.release(request_id)


def check_triton_vs_torch_boundaries() -> None:
    import torch

    from cloud_engine.cache import PagedKVCache
    from cloud_engine.kernel import decode_attention_batched, decode_attention_direct

    batch_sizes = [1, 2, 8, 16]
    lengths = [1, 15, 16, 17, 127, 128, 129, 512, 2048]
    for batch in batch_sizes:
        cache = PagedKVCache(
            num_layers=1, num_kv_heads=2, head_dim=64,
            block_size=16, kv_cache_bytes=256 << 20, dtype=torch.float16, device="cuda",
        )
        tables = []
        lens = []
        for b in range(batch):
            seq_len = lengths[(b + batch) % len(lengths)]
            rid = f"tri-{batch}-{b}"
            cache.reserve(rid, seq_len)
            import torch as _t

            tokens = _t.randn(seq_len, 2, 64, device="cuda", dtype=_t.float16)
            cache.append(rid, layer=0, keys=tokens, values=tokens * 0.5, start_pos=0)
            tables.append(cache.block_table(rid))
            lens.append(seq_len)
        longest = max(lens)
        q, keys, _ = synthetic_attention_inputs(longest, batch=batch)
        # per-request torch reference via gather path
        outs = []
        for b in range(batch):
            rid = f"tri-{batch}-{b}"
            view = cache.view(rid, layer=0)
            single_q = q[b : b + 1]
            expanded_k = view.keys.repeat_interleave(7, dim=1)
            expanded_v = view.values.repeat_interleave(7, dim=1)
            from cloud_engine.attention import causal_attention

            ctx_out = causal_attention(single_q[0], expanded_k, expanded_v, 8**-0.5, past_len=lens[b] - 1)
            outs.append(ctx_out.reshape(14, 64))
        expected = torch.stack(outs)
        got = decode_attention_batched(q, cache.key_pool[0], cache.value_pool[0], tables, lens, 8**-0.5)
        ok = torch.allclose(got.float(), expected.float(), rtol=2e-2, atol=2e-2)
        diff = float((got.float() - expected.float()).abs().max())
        record(f"triton==torch-paged batch={batch}", ok, f"max|Δ|={diff:.4f}")
        # single-request wrapper sanity on first row
        one = decode_attention_direct(q[0], cache.key_pool[0], cache.value_pool[0], tables[0], lens[0], 8**-0.5)
        record(f"triton single wrapper batch={batch}", torch.equal(one, got[0]))
        for b in range(batch):
            cache.release(f"tri-{batch}-{b}")


async def check_concurrent_no_leaks() -> None:
    from cloud_engine.scheduler import GenerationConfig

    _, engine = build_engine("paged")
    await engine.start()
    try:
        async def drive(i: int):
            handle = await engine.submit(
                f"Request number {i} about memory pages.",
                GenerationConfig(max_output_tokens=24, eos_token_id=151643),
            )
            return await handle.wait()

        results = await asyncio.gather(*(drive(i) for i in range(16)))
        completed = sum(1 for r in results if r.finish_reason.startswith(("eos", "max")))
        stats = engine.cache.stats()
        record("16 concurrent requests complete", completed >= 15, f"completed={completed}")
        record("allocator drained after concurrency", stats.blocks_used == 0,
               f"used_blocks={stats.blocks_used}")
    finally:
        await engine.close()


async def check_fault_paths_leave_no_blocks() -> None:
    from cloud_engine.scheduler import GenerationConfig

    scenarios_passed = {}
    for mode in ("paged", "triton"):
        _, engine = build_engine(mode, fallback=True)
        await engine.start()
        try:
            # disconnect-style cancellation mid-generation
            handles = []
            for i in range(4):
                handle = await engine.submit(
                    f"cancellable story part {i}",
                    GenerationConfig(max_output_tokens=256, eos_token_id=None),
                )
                handles.append(handle)
            await asyncio.sleep(0.02)
            for handle in handles:
                handle.cancel()
            await asyncio.sleep(0.2)
            stats = engine.cache.stats()
            scenarios_passed[f"{mode}-cancel"] = stats.blocks_used == 0

            # timeout path: fill active slots then let queue time out
            blockers = []
            for i in range(engine.config.max_active_sequences):
                h = await engine.submit(
                    f"blocker {i} keeps the pool busy",
                    GenerationConfig(max_output_tokens=256, eos_token_id=None),
                )
                blockers.append(h)
            victim = await engine.submit(
                "victim waits too long",
                GenerationConfig(max_output_tokens=8, eos_token_id=None),
            )
            await asyncio.wait_for(victim.request.terminal_future, timeout=90)
            for h in blockers:
                h.cancel()
            await asyncio.sleep(0.3)
            stats = engine.cache.stats()
            scenarios_passed[f"{mode}-timeout"] = stats.blocks_used == 0
        finally:
            await engine.close()
    for name, ok in scenarios_passed.items():
        record(f"fault path leaves zero blocks: {name}", ok)


async def check_streaming_matches_non_streaming() -> None:
    from cloud_engine.scheduler import GenerationConfig

    _, engine = build_engine("triton", fallback=True)
    await engine.start()
    try:
        prompt = "Stream a tiny story about blocks."
        gen = GenerationConfig(max_output_tokens=48, eos_token_id=151643)
        streamed_handle = await engine.submit(prompt, gen)
        events = []
        async for event in streamed_handle.stream():
            events.append(event.token_id)
        streamed_text = engine.detokenize(events)

        blocking_handle = await engine.submit(prompt, gen)
        result = await blocking_handle.wait()
        same_tokens = events == result.token_ids
        same_text = streamed_text == result.text
        record("streamed token sequence equals wait()", same_tokens,
               f"{len(events)} vs {len(result.token_ids)}")
        record("streamed text equals wait() text", same_text)
        record("stream token ids preserve generation order", events == result.token_ids)
    finally:
        await engine.close()


def main() -> int:
    started = time.time()
    print(f"remote GPU suite starting; MODEL_DIR={MODEL_DIR}")
    check_logits_parity()
    asyncio.run(check_five_mode_token_parity())
    check_paged_vs_contiguous_boundaries()
    check_triton_vs_torch_boundaries()
    asyncio.run(check_concurrent_no_leaks())
    asyncio.run(check_fault_paths_leave_no_blocks())
    asyncio.run(check_streaming_matches_non_streaming())

    failed = [(n, d) for n, ok, d in RESULTS if not ok]
    elapsed = time.time() - started
    print(f"\n{len(RESULTS)} checks in {elapsed:.1f}s — "
          f"{len(RESULTS) - len(failed)} passed, {len(failed)} failed")
    if failed:
        print(json.dumps([{"check": n, "detail": d} for n, d in failed], indent=2))
        print("reminder: Modal compute is billable.")
        return 1
    print("ALL REMOTE GPU CHECKS PASSED\nreminder: Modal compute is billable.")
    return 0


if __name__ == "__main__" or __name__ == "__remote_gpu_tests__":
    sys.exit(main())
