"""Ragged L4 correctness and pressure suite for the pinned Qwen2.5-3B model."""

import asyncio
import gc
import json
import os
import sys
import time

MODEL_DIR = globals().get("MODEL_DIR") or os.environ.get("MODEL_DIR", "/cache/hf")
RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str = "") -> None:
    RESULTS.append((name, passed, detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def reference_tokens(prompts: list[str], max_new_tokens: int) -> list[list[int]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16, attn_implementation="eager"
    ).to("cuda").eval()
    outputs = []
    with torch.no_grad():
        for prompt in prompts:
            ids = torch.tensor([tokenizer.encode(prompt)], device="cuda")
            attention_mask = torch.ones_like(ids)
            generated = model.generate(
                ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=151643,
            )
            outputs.append(generated[0][ids.shape[1] :].tolist())
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return outputs


def build_engine(**overrides):
    from cloud_engine.config import build_config
    from cloud_engine.engine import InferenceEngine

    config = build_config("ragged", **overrides)
    return InferenceEngine(config, model_dir=MODEL_DIR)


async def consume_request(request) -> list[int]:
    tokens = []
    while True:
        event = await request.output_queue.get()
        if event.finished:
            return tokens
        if event.token_id is not None:
            tokens.append(event.token_id)


async def check_model_parity_and_real_packing() -> None:
    from cloud_engine.scheduler import GenerationConfig

    prompts = [
        "Explain a KV cache in one sentence.",
        "Name three GPU memory levels.",
        "What is continuous batching?",
        "Define a page table briefly.",
    ]
    expected = reference_tokens(prompts, 8)
    engine = build_engine()
    await engine.start()
    try:
        handles = [
            await engine.submit(
                prompt, GenerationConfig(max_output_tokens=8, eos_token_id=None)
            )
            for prompt in prompts
        ]
        results = await asyncio.gather(*(handle.wait() for handle in handles))
        got = [result.token_ids for result in results]
        snapshot = engine.snapshot_metrics()
        max_requests = snapshot["scheduler"]["max_forward_request_count"]
        record("Qwen2.5-3B ragged tokens equal HF oracle", got == expected)
        record(
            "one transformer invocation contains multiple request IDs",
            max_requests >= 4,
            f"max_request_ids={max_requests}",
        )
        record(
            "ragged run leaves no KV blocks",
            engine.cache.stats().request_blocks_used == 0,
        )
    finally:
        await engine.close()


def check_ragged_kernel_matrix() -> None:
    import torch

    from cloud_engine.attention import causal_attention
    from cloud_engine.cache import PagedKVCache
    from cloud_engine.kernel import ragged_attention_direct

    context_pattern = [0, 15, 16, 17, 127, 128, 511, 1024, 2047, 4000]
    query_pattern = [1, 2, 7, 16]
    worst_diff = 0.0
    for batch in (1, 2, 4, 8, 16):
        cache = PagedKVCache(
            num_layers=1,
            num_kv_heads=2,
            head_dim=128,
            block_size=16,
            kv_cache_bytes=256 << 20,
            dtype=torch.float16,
            device="cuda",
        )
        generator = torch.Generator(device="cuda").manual_seed(10_000 + batch)
        contexts = [context_pattern[i % len(context_pattern)] for i in range(batch)]
        query_lens = [query_pattern[i % len(query_pattern)] for i in range(batch)]
        query_lens = [min(q, 4096 - c) for q, c in zip(query_lens, contexts, strict=True)]
        finals = [c + q for c, q in zip(contexts, query_lens, strict=True)]
        request_ids = [f"ragged-kernel-{batch}-{i}" for i in range(batch)]
        for request_id in request_ids:
            cache.reserve(request_id, 0)
        cache.ensure_capacity_batch(dict(zip(request_ids, finals, strict=True)))

        all_q = []
        all_seq_ids = []
        all_positions = []
        expected = []
        for sequence, (request_id, context, query_len, final_len) in enumerate(
            zip(request_ids, contexts, query_lens, finals, strict=True)
        ):
            keys = torch.randn(
                final_len, 2, 128, generator=generator, device="cuda", dtype=torch.float16
            )
            values = torch.randn(
                final_len, 2, 128, generator=generator, device="cuda", dtype=torch.float16
            )
            cache.append(request_id, 0, keys, values, 0)
            q = torch.randn(
                query_len, 16, 128, generator=generator, device="cuda", dtype=torch.float16
            )
            all_q.append(q)
            all_seq_ids.extend([sequence] * query_len)
            all_positions.extend(range(context, final_len))
            expected.append(
                causal_attention(
                    q,
                    keys.repeat_interleave(8, dim=1),
                    values.repeat_interleave(8, dim=1),
                    128**-0.5,
                    past_len=context,
                ).reshape(query_len, 16, 128)
            )

        width = max(len(cache.block_table(request_id)) for request_id in request_ids)
        tables = torch.zeros(batch, width, dtype=torch.int32, device="cuda")
        for row, request_id in enumerate(request_ids):
            table = cache.block_table(request_id)
            tables[row, : len(table)] = torch.tensor(table, dtype=torch.int32, device="cuda")
        q_flat = torch.cat(all_q)
        got = ragged_attention_direct(
            q_flat,
            cache.key_pool[0],
            cache.value_pool[0],
            tables,
            torch.tensor(all_seq_ids, dtype=torch.int32, device="cuda"),
            torch.tensor(all_positions, dtype=torch.long, device="cuda"),
            128**-0.5,
        )
        expected_flat = torch.cat(expected)
        diff = float((got.float() - expected_flat.float()).abs().max())
        worst_diff = max(worst_diff, diff)
        close = torch.allclose(got.float(), expected_flat.float(), rtol=2e-2, atol=2e-2)
        record(
            f"ragged Triton == Torch batch={batch}",
            close,
            f"queries={sum(query_lens)} max_context={max(finals)} max|Δ|={diff:.5f}",
        )
        for request_id in request_ids:
            cache.release(request_id)
        cache.assert_invariants()
    record("ragged kernel worst absolute error < 0.01", worst_diff < 0.01, f"{worst_diff:.5f}")


def check_serial_triton_3b_envelope() -> None:
    import torch

    from cloud_engine.attention import causal_attention
    from cloud_engine.cache import PagedKVCache
    from cloud_engine.kernel import decode_attention_direct

    length = 4096
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=128,
        block_size=16,
        kv_cache_bytes=32 << 20,
        dtype=torch.float16,
        device="cuda",
    )
    cache.reserve("serial-3b", length)
    generator = torch.Generator(device="cuda").manual_seed(30_000)
    keys = torch.randn(length, 2, 128, generator=generator, device="cuda", dtype=torch.float16)
    values = torch.randn(length, 2, 128, generator=generator, device="cuda", dtype=torch.float16)
    q = torch.randn(1, 16, 128, generator=generator, device="cuda", dtype=torch.float16)
    cache.append("serial-3b", 0, keys, values, 0)
    got = decode_attention_direct(
        q[0], cache.key_pool[0], cache.value_pool[0], cache.block_table("serial-3b"), length, 128**-0.5
    )
    expected = causal_attention(
        q,
        keys.repeat_interleave(8, dim=1),
        values.repeat_interleave(8, dim=1),
        128**-0.5,
        past_len=length - 1,
    ).reshape_as(got)
    diff = float((got.float() - expected.float()).abs().max())
    record(
        "serial Triton supports Qwen2.5-3B at context 4096",
        torch.allclose(got.float(), expected.float(), rtol=2e-2, atol=2e-2),
        f"max|Δ|={diff:.5f}",
    )
    cache.release("serial-3b")


async def check_chunked_prefill_decode_priority() -> None:
    from cloud_engine.scheduler import GenerationConfig
    from cloud_engine.weights import load_tokenizer

    tokenizer = load_tokenizer(MODEL_DIR)
    token_id = tokenizer.encode(" inference", add_special_tokens=False)[0]
    engine = build_engine(max_batched_tokens=512, prefill_chunk_size=256)
    await engine.start()
    try:
        long_request = await engine.scheduler.submit(
            "synthetic-long",
            [token_id] * 4000,
            GenerationConfig(max_output_tokens=16, eos_token_id=None),
        )
        short_request = await engine.scheduler.submit(
            "synthetic-short",
            [token_id] * 8,
            GenerationConfig(max_output_tokens=16, eos_token_id=None),
        )
        long_tokens, short_tokens = await asyncio.gather(
            consume_request(long_request), consume_request(short_request)
        )
        plans = list(engine.runner.forward_plans)
        mixed = [
            plan
            for plan in plans
            if len(plan) >= 2
            and plan[0][0] == short_request.request_id
            and plan[0][2] == 1
            and any(row[0] == long_request.request_id and row[2] <= 256 for row in plan[1:])
        ]
        long_chunks = [
            row[2]
            for plan in plans
            for row in plan
            if row[0] == long_request.request_id and row[1] < 4000
        ]
        record(
            "4000-token prompt advances in bounded chunks",
            bool(long_chunks) and max(long_chunks) <= 256 and len(long_tokens) == 16,
            f"chunks={long_chunks[:20]}",
        )
        record(
            "decode is scheduled before concurrent long prefill",
            bool(mixed) and len(short_tokens) == 16,
            f"mixed_iterations={len(mixed)}",
        )
    finally:
        await engine.close()


async def check_real_pressure_recompute() -> None:
    from cloud_engine.scheduler import GenerationConfig
    from cloud_engine.weights import load_tokenizer

    tokenizer = load_tokenizer(MODEL_DIR)
    token_id = tokenizer.encode(" cache", add_special_tokens=False)[0]
    engine = build_engine(
        kv_cache_bytes=5 << 20,
        prefix_cache_max_blocks=0,
        max_active_sequences=2,
        max_batched_tokens=128,
        prefill_chunk_size=64,
    )
    await engine.start()
    try:
        requests = [
            await engine.scheduler.submit(
                f"pressure-{i}",
                [token_id] * 56,
                GenerationConfig(max_output_tokens=12, eos_token_id=None),
            )
            for i in range(2)
        ]
        outputs = await asyncio.gather(*(consume_request(request) for request in requests))
        snapshot = engine.snapshot_metrics()
        record("undersized pool triggers preemption", snapshot["requests"]["preempted_total"] > 0)
        record("preempted sequence is recomputed", snapshot["tokens"]["recomputed_total"] > 0)
        record(
            "pressure recovery preserves deterministic tokens",
            outputs[0] == outputs[1] and all(len(output) == 12 for output in outputs),
        )
        record(
            "pressure recovery drains allocator",
            engine.cache.stats().request_blocks_used == 0,
            f"used={engine.cache.stats().request_blocks_used}",
        )
        engine.cache.assert_invariants()
    finally:
        await engine.close()


async def check_prefix_cache_reuse() -> None:
    from cloud_engine.scheduler import GenerationConfig

    prompt = "Prefix caching avoids repeated prefill work on shared prompts. " * 24
    engine = build_engine(prefix_cache_max_blocks=64)
    await engine.start()
    try:
        first = await engine.submit(
            prompt, GenerationConfig(max_output_tokens=8, eos_token_id=None)
        )
        first_tokens = (await first.wait()).token_ids
        first_work = sum(
            query_length
            for plan in engine.runner.forward_plans
            for request_id, _, query_length, _ in plan
            if request_id == first.request_id
        )

        second = await engine.submit(
            prompt, GenerationConfig(max_output_tokens=8, eos_token_id=None)
        )
        second_tokens = (await second.wait()).token_ids
        second_work = sum(
            query_length
            for plan in engine.runner.forward_plans
            for request_id, _, query_length, _ in plan
            if request_id == second.request_id
        )
        stats = engine.cache.stats()
        record("prefix-cache hit preserves exact tokens", first_tokens == second_tokens)
        record(
            "prefix-cache hit skips block-aligned prefill",
            second_work + engine.config.block_size <= first_work,
            f"cold_tokens={first_work} warm_tokens={second_work}",
        )
        record("prefix-cache hit is measured", stats.prefix_cache_hits >= 1)
        record(
            "prefix cache retains no request-owned blocks",
            stats.request_blocks_used == 0 and stats.prefix_blocks_used > 0,
            f"request_blocks={stats.request_blocks_used} prefix_blocks={stats.prefix_blocks_used}",
        )
    finally:
        await engine.close()


def main() -> int:
    started = time.time()
    print(f"ragged GPU suite starting; MODEL_DIR={MODEL_DIR}")
    asyncio.run(check_model_parity_and_real_packing())
    check_ragged_kernel_matrix()
    check_serial_triton_3b_envelope()
    asyncio.run(check_chunked_prefill_decode_priority())
    asyncio.run(check_real_pressure_recompute())
    asyncio.run(check_prefix_cache_reuse())
    failed = [(name, detail) for name, ok, detail in RESULTS if not ok]
    elapsed = time.time() - started
    print(
        f"\n{len(RESULTS)} checks in {elapsed:.1f}s — "
        f"{len(RESULTS) - len(failed)} passed, {len(failed)} failed"
    )
    if failed:
        print(json.dumps([{"check": name, "detail": detail} for name, detail in failed], indent=2))
        return 1
    print("ALL RAGGED GPU CHECKS PASSED")
    return 0


if __name__ == "__main__" or __name__ == "__remote_ragged_gpu_tests__":
    sys.exit(main())
