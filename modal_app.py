"""Modal application: all heavy execution lives here (PRD G2).

Local machine: source + this file + the lightweight Modal client. Weights,
Torch, Triton compilation, inference, and benchmarks run remotely on one L4
with max_containers=1 and scale-to-zero.

Cost controls (PRD §11): no schedules, no warm containers, no auto-benchmark
on deploy. Every GPU command prints a billable-compute reminder.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from hashlib import sha256
from pathlib import Path

import modal

PINNED = json.loads(Path(__file__).parent.joinpath("engine_config.json").read_text())
IMAGE_PINS = PINNED["image"]
COMPATIBILITY_CANDIDATE = PINNED["compatibility_candidate"]
QUANTIZATION_PINS = PINNED["quantization"]
MODEL = PINNED["model"]
RAGGED_MODEL = PINNED["ragged_model"]
LLAMA_MODEL = PINNED["llama_model"]
MODAL_CFG = PINNED["modal"]

APP_NAME = MODAL_CFG["app_name"]
VOLUME_NAME = MODAL_CFG["volume_name"]
SECRET_NAME = MODAL_CFG["secret_name"]


def _source_commit() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True
        ).strip()
        digest = sha256()
        paths = [Path(__file__), Path(__file__).parent / "engine_config.json"]
        experiment_runner = Path(__file__).parent / "experiment.py"
        if experiment_runner.is_file():
            paths.append(experiment_runner)
        for directory in ("src", "benchmarks", "tests"):
            paths.extend(sorted((Path(__file__).parent / directory).rglob("*.py")))
        for path in paths:
            digest.update(path.relative_to(Path(__file__).parent).as_posix().encode())
            digest.update(path.read_bytes())
        return f"{commit}+tree-{digest.hexdigest()[:12]}"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


SOURCE_COMMIT = _source_commit()
app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version=IMAGE_PINS["python_version"])
    .pip_install(
        f"torch=={IMAGE_PINS['torch']}",
        f"triton=={IMAGE_PINS['triton']}",
        f"transformers=={IMAGE_PINS['transformers']}",
        f"tokenizers=={IMAGE_PINS['tokenizers']}",
        f"safetensors=={IMAGE_PINS['safetensors']}",
        f"huggingface_hub=={IMAGE_PINS['huggingface_hub']}",
        f"fastapi=={IMAGE_PINS['fastapi']}",
        f"starlette=={IMAGE_PINS['starlette']}",
        f"pydantic=={IMAGE_PINS['pydantic']}",
        f"httpx=={IMAGE_PINS['httpx']}",
        f"uvicorn=={IMAGE_PINS['uvicorn']}",
    )
    .add_local_dir(
        Path(__file__).parent / "src/cloud_engine",
        "/root/cloud_engine",
        copy=True,
    )
    .add_local_dir(
        Path(__file__).parent / "benchmarks",
        "/root/benchmarks",
        copy=True,
    )
    .add_local_dir(
        Path(__file__).parent / "tests",
        "/root/tests",
        copy=True,
    )
    .add_local_dir(
        Path(__file__).parent / "artifacts",
        "/root/artifacts",
        copy=True,
    )
    .add_local_file(
        Path(__file__).parent / "README.md",
        "/root/README.md",
        copy=True,
    )
    .add_local_file(
        Path(__file__).parent / "engine_config.json",
        "/root/engine_config.json",
        copy=True,
    )
    .env(
        {
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "SOURCE_COMMIT": SOURCE_COMMIT,
            "TRITON_CACHE_DIR": "/cache/triton-cache",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

compatibility_image = image.pip_install(
    *(f"{package.replace('_', '-')}=={version}" for package, version in COMPATIBILITY_CANDIDATE.items())
)
quantization_image = compatibility_image.pip_install(
    f"bitsandbytes=={QUANTIZATION_PINS['bitsandbytes']}"
)

vllm_image = (
    modal.Image.debian_slim(python_version=IMAGE_PINS["python_version"])
    .pip_install(
        "vllm==0.10.0",
        "transformers==4.53.2",
        "tokenizers==0.21.2",
        "huggingface_hub==0.33.4",
        f"httpx=={IMAGE_PINS['httpx']}",
    )
    .add_local_dir(
        Path(__file__).parent / "src/cloud_engine",
        "/root/cloud_engine",
        copy=True,
    )
    .add_local_dir(
        Path(__file__).parent / "benchmarks",
        "/root/benchmarks",
        copy=True,
    )
    .add_local_file(
        Path(__file__).parent / "engine_config.json",
        "/root/engine_config.json",
        copy=True,
    )
    .env(
        {
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "SOURCE_COMMIT": SOURCE_COMMIT,
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _gpu_options(**overrides):
    options = dict(
        image=image,
        gpu=MODAL_CFG["gpu"],
        volumes={"/cache": volume},
        timeout=MODAL_CFG["timeout_seconds"],
        scaledown_window=MODAL_CFG["scaledown_window_seconds"],
        max_containers=MODAL_CFG["max_containers"],
        min_containers=MODAL_CFG["min_containers"],
        buffer_containers=MODAL_CFG["buffer_containers"],
    )
    options.update(overrides)
    return options


def _prepare_weights(model: dict | None = None) -> str:
    """Download the pinned snapshot into /cache if absent; return local dir."""
    model = model or MODEL
    os.makedirs("/cache/triton-cache", exist_ok=True)
    from cloud_engine.weights import ensure_weights_downloaded

    path = ensure_weights_downloaded("/cache", model["revision"], model["id"])
    volume.commit()
    print(f"weights ready at {path} ({model['id']} @ {model['revision']})")
    return str(path)


def _build_engine(mode: str, model_dir: str, allow_fallback: bool = False):
    from cloud_engine.config import build_config
    from cloud_engine.engine import InferenceEngine

    config = build_config(mode, allow_reference_fallback=allow_fallback)
    engine = InferenceEngine(config, model_dir=model_dir)
    return config, engine


def _print_run_header(kind: str) -> None:
    try:
        import torch

        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"GPU: {gpu} ({vram:.0f} GiB)")
    except Exception:
        pass
    print(f"task: {kind}")
    print("reminder: Modal cloud compute is billable.\n")


@app.function(**_gpu_options())
def _ensure_weights() -> str:
    """Idempotent weight download; safe to call directly to prewarm the Volume."""
    return _prepare_weights()


@app.function(**_gpu_options())
def _ensure_ragged_weights() -> str:
    """Idempotently cache the pinned Qwen2.5-3B Ragged L4 model."""
    return _prepare_weights(RAGGED_MODEL)


@app.function(**_gpu_options())
def smoke() -> dict:
    """First-run correctness check (PRD §6.1).

    Loads the custom engine in contiguous mode, generates deterministically,
    compares token IDs against a Hugging Face reference (test oracle only),
    then reports TTFT/throughput/peak memory.
    """
    import time

    import torch

    _print_run_header("smoke: contiguous mode parity check")
    model_dir = _prepare_weights()

    prompt = "Explain what a KV cache is in exactly two sentences."
    max_output = 48

    import asyncio

    from cloud_engine.scheduler import GenerationConfig
    from cloud_engine.weights import load_tokenizer

    _, engine_obj = _build_engine("contiguous", model_dir)

    async def _run():
        await engine_obj.start()
        handle = await engine_obj.submit(
            prompt,
            GenerationConfig(max_output_tokens=max_output, eos_token_id=MODEL["eos_token_id"]),
        )
        result = await handle.wait()
        metrics_snapshot = engine_obj.snapshot_metrics()
        await engine_obj.close()
        return result, metrics_snapshot

    started = time.perf_counter()
    result, snapshot = asyncio.run(_run())
    wall_s = time.perf_counter() - started

    tokenizer = load_tokenizer(model_dir)
    reference_ids = _reference_generate(model_dir, tokenizer.encode(prompt), max_output)

    ours = result.token_ids
    match = ours == reference_ids
    print(f"prompt tokens      : {len(tokenizer.encode(prompt))}")
    print(f"output tokens      : {result.output_tokens} (finish={result.finish_reason})")
    print(f"token ids match HF : {match}")
    if not match:
        print(f"ours     : {ours[:24]}{'...' if len(ours) > 24 else ''}")
        print(f"reference: {reference_ids[:24]}{'...' if len(reference_ids) > 24 else ''}")
    ttft = result.ttft_ms or float("nan")
    print(f"TTFT               : {ttft:.1f} ms")
    throughput = result.output_tokens / wall_s
    print(f"output tok/s       : {throughput:.1f} (wall-clock incl. startup)")
    print(
        f"TTFT p50/p95       : {snapshot['latency_ms']['ttft_p50']:.1f}/"
        f"{snapshot['latency_ms']['ttft_p95']:.1f} ms"
    )
    print(f"peak GPU memory    : {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    dashboard = "https://modal.com/apps"
    try:
        call_id = os.environ.get("MODAL_FUNCTION_CALL_ID")
        if call_id:
            dashboard = f"https://modal.com/calls/{call_id}"
    except Exception:
        pass
    print(f"dashboard          : {dashboard}")
    if not match:
        raise SystemExit("SMOKE FAILED: generated token IDs differ from reference")
    print("SMOKE PASSED")
    return {"passed": True, "output_tokens": result.output_tokens}


@app.function(**_gpu_options())
def ragged_smoke() -> dict:
    """Fast packed-runtime proof on the cached 0.5B oracle model."""
    return _execute_ragged_smoke(MODEL, "ragged smoke: real multi-request forward")


def _execute_ragged_smoke(
    model_spec: dict, label: str, quantization: str = "none"
) -> dict:
    import asyncio

    from cloud_engine.config import build_config
    from cloud_engine.engine import InferenceEngine
    from cloud_engine.scheduler import GenerationConfig

    _print_run_header(label)
    model_dir = _prepare_weights(model_spec)
    config = build_config(
        "ragged",
        model_id=model_spec["id"],
        model_revision=model_spec["revision"],
        max_model_len=model_spec["max_model_len"],
        eos_token_id=model_spec["eos_token_id"],
        quantization=quantization,
    )
    engine = InferenceEngine(config, model_dir=model_dir)
    prompts = (
        "Explain KV caching briefly.",
        "Name three GPU memory levels.",
        "Define continuous batching.",
        "What does a page table map?",
    )

    async def _run():
        await engine.start()
        try:
            handles = [
                await engine.submit(
                    prompt,
                    GenerationConfig(max_output_tokens=8, eos_token_id=None),
                )
                for prompt in prompts
            ]
            results = await asyncio.gather(*(handle.wait() for handle in handles))
            snapshot = engine.snapshot_metrics()
            return results, snapshot
        finally:
            await engine.close()

    results, snapshot = asyncio.run(_run())
    max_requests = snapshot["scheduler"]["max_forward_request_count"]
    if max_requests < 2:
        raise RuntimeError(f"packed smoke never shared a forward: max={max_requests}")
    if any(result.output_tokens != 8 for result in results):
        raise RuntimeError("packed smoke returned an incomplete generation")
    from cloud_engine.weights import load_tokenizer

    tokenizer = load_tokenizer(model_dir)
    reference = [
        _reference_generate(
            model_dir, tokenizer.encode(prompt), 8, model_spec["eos_token_id"]
        )
        for prompt in prompts
    ]
    got = [result.token_ids for result in results]
    if got != reference:
        raise RuntimeError(f"ragged token parity failed: got={got}, reference={reference}")
    graph_captures = snapshot["scheduler"]["cuda_graph_captures"]
    graph_replays = snapshot["scheduler"]["cuda_graph_replays"]
    if config.cuda_graph_decode and (graph_captures < 1 or graph_replays < 1):
        raise RuntimeError(
            f"CUDA graph decode was not exercised: captures={graph_captures}, "
            f"replays={graph_replays}"
        )
    print(f"max request IDs in one model invocation: {max_requests}")
    print(f"last packed IDs: {snapshot['scheduler']['last_forward_request_ids']}")
    print("token parity vs Hugging Face: exact")
    print("RAGGED SMOKE PASSED")
    return {
        "passed": True,
        "model": model_spec["id"],
        "model_revision": model_spec["revision"],
        "source_revision": os.environ.get("SOURCE_COMMIT", SOURCE_COMMIT),
        "oracle_sequences": len(reference),
        "tokens_per_sequence": 8,
        "max_forward_request_count": max_requests,
        "cuda_graph_captures": graph_captures,
        "cuda_graph_replays": graph_replays,
    }


@app.function(**_gpu_options())
def llama_ragged_smoke() -> str:
    """Exact packed-runtime parity for the pinned Llama-family model."""
    result = _execute_ragged_smoke(LLAMA_MODEL, "Llama-family ragged parity smoke")
    import torch

    result["gpu"] = torch.cuda.get_device_name(0)
    result["compute_capability"] = list(torch.cuda.get_device_capability())
    return json.dumps(result, indent=2) + "\n"


@app.function(**_gpu_options(image=compatibility_image, timeout=3600))
def compatibility_smoke() -> str:
    """Run packed exact-oracle/CUDA-graph proof on the candidate dependency stack."""
    from importlib.metadata import version

    result = _execute_ragged_smoke(MODEL, "candidate-stack GPU compatibility smoke")
    result["baseline_stack"] = IMAGE_PINS
    result["candidate_stack"] = {
        package: version(package.replace("_", "-"))
        for package in COMPATIBILITY_CANDIDATE
    }
    return json.dumps(result, indent=2) + "\n"


@app.function(**_gpu_options(image=quantization_image, timeout=3600))
def quantization_benchmark() -> str:
    """Paired 3B FP16/LLM.int8 quality, memory, and batch-latency measurement."""
    import asyncio
    import gc
    import math
    import statistics

    import torch

    from cloud_engine.config import build_config
    from cloud_engine.engine import InferenceEngine
    from cloud_engine.scheduler import GenerationConfig

    _print_run_header("paired Qwen2.5-3B FP16 vs LLM.int8 benchmark")
    model_dir = _prepare_weights(RAGGED_MODEL)
    prompts = (
        "Explain why paged KV caches reduce fragmentation.",
        "Give two invariants for continuous batching.",
        "What does a Triton program instance execute?",
        "Describe the purpose of recompute preemption.",
        "Why does online softmax improve attention memory use?",
        "Define time to first token in one sentence.",
        "What problem does prefix caching solve?",
        "Name one risk of an unbounded request queue.",
        "Explain grouped-query attention briefly.",
        "Why pin a model revision in production?",
        "What is a CUDA graph replay?",
        "Describe backpressure for streaming tokens.",
        "Why isolate tenants with admission quotas?",
        "What does an NVTX range provide?",
        "Explain a paged KV block table.",
        "Why benchmark against a correctness oracle?",
    )

    async def run(quantization: str) -> dict:
        torch.cuda.empty_cache()
        baseline_bytes = torch.cuda.memory_allocated()
        config = build_config(
            "ragged",
            model_id=RAGGED_MODEL["id"],
            model_revision=RAGGED_MODEL["revision"],
            max_model_len=RAGGED_MODEL["max_model_len"],
            eos_token_id=RAGGED_MODEL["eos_token_id"],
            prefix_cache_max_blocks=0,
            cuda_graph_decode=False,
            dtype="float16",
            quantization=quantization,
        )
        engine = InferenceEngine(config, model_dir=model_dir)
        await engine.start()
        torch.cuda.synchronize()
        steady_bytes = torch.cuda.memory_allocated() - baseline_bytes
        torch.cuda.reset_peak_memory_stats()

        from cloud_engine.attention import AttentionBackend

        dims = engine.model.dims
        runtime_backend = engine.model.attn_backend
        engine.model.attn_backend = AttentionBackend(
            None, dims.num_heads, dims.num_kv_heads, dims.head_dim
        )
        quality_text = Path("/root/README.md").read_text(encoding="utf-8")
        quality_ids = engine.tokenizer.encode(quality_text)[:2048]
        quality_chunks = [
            quality_ids[start : start + 256]
            for start in range(0, len(quality_ids), 256)
        ]
        teacher_top1 = []
        teacher_nll = 0.0
        teacher_tokens = 0
        with torch.no_grad():
            for token_ids in quality_chunks:
                input_ids = torch.tensor(
                    token_ids,
                    dtype=torch.long,
                    device=engine.device,
                )
                logits = engine.model(input_ids, ctx=None, return_all_logits=True)
                teacher_top1.extend(logits[:-1].argmax(dim=-1).cpu().tolist())
                teacher_nll += float(
                    torch.nn.functional.cross_entropy(
                        logits[:-1], input_ids[1:], reduction="sum"
                    )
                )
                teacher_tokens += input_ids.numel() - 1
        engine.model.attn_backend = runtime_backend

        async def generate(items: tuple[str, ...]) -> list[list[int]]:
            handles = [
                await engine.submit(
                    prompt,
                    GenerationConfig(max_output_tokens=8, eos_token_id=None),
                )
                for prompt in items
            ]
            results = await asyncio.gather(*(handle.wait() for handle in handles))
            torch.cuda.synchronize()
            return [result.token_ids for result in results]

        await generate(tuple(f"warmup: {prompt}" for prompt in prompts))
        timings = []
        outputs = []
        for trial in range(3):
            started = time.perf_counter()
            outputs.append(
                await generate(tuple(f"trial {trial}: {prompt}" for prompt in prompts))
            )
            timings.append((time.perf_counter() - started) * 1000)
        snapshot = engine.snapshot_metrics()["scheduler"]
        result = {
            "steady_gpu_memory_bytes": steady_bytes,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "batch_latency_ms": timings,
            "median_batch_latency_ms": statistics.median(timings),
            "outputs": outputs,
            "teacher_top1": teacher_top1,
            "teacher_nll": teacher_nll,
            "teacher_tokens": teacher_tokens,
            "quality_corpus_sha256": sha256(quality_text.encode()).hexdigest(),
            "max_forward_request_count": snapshot["max_forward_request_count"],
            "cuda_graph_captures": snapshot["cuda_graph_captures"],
            "cuda_graph_replays": snapshot["cuda_graph_replays"],
        }
        await engine.close()
        engine = None
        gc.collect()
        torch.cuda.empty_cache()
        return result

    fp16 = asyncio.run(run("none"))
    int8 = asyncio.run(run("bitsandbytes_int8"))
    fp16_top1 = fp16.pop("teacher_top1")
    int8_top1 = int8.pop("teacher_top1")
    teacher_top1_agreement = sum(
        reference == quantized
        for reference, quantized in zip(fp16_top1, int8_top1, strict=True)
    ) / len(fp16_top1)
    teacher_tokens = fp16.pop("teacher_tokens")
    if teacher_tokens != int8.pop("teacher_tokens"):
        raise RuntimeError("teacher-forced token counts differ")
    fp16_nll = fp16.pop("teacher_nll") / teacher_tokens
    int8_nll = int8.pop("teacher_nll") / teacher_tokens
    perplexity_ratio = math.exp(int8_nll - fp16_nll)
    quality_corpus_sha256 = fp16.pop("quality_corpus_sha256")
    if quality_corpus_sha256 != int8.pop("quality_corpus_sha256"):
        raise RuntimeError("quality corpus hashes differ")
    exact = fp16["outputs"] == int8["outputs"]
    paired = [
        (reference, quantized)
        for reference_trial, quantized_trial in zip(
            fp16["outputs"], int8["outputs"], strict=True
        )
        for reference, quantized in zip(
            reference_trial, quantized_trial, strict=True
        )
    ]
    first_token_agreement = sum(a[0] == b[0] for a, b in paired) / len(paired)
    token_agreement = sum(
        reference == quantized
        for a, b in paired
        for reference, quantized in zip(a, b, strict=True)
    ) / sum(len(a) for a, _ in paired)
    memory_ratio = int8["steady_gpu_memory_bytes"] / fp16["steady_gpu_memory_bytes"]
    latency_ratio = int8["median_batch_latency_ms"] / fp16["median_batch_latency_ms"]
    passed = (
        perplexity_ratio <= 1.05
        and memory_ratio < 0.8
        and latency_ratio <= 2.5
    )
    result = {
        "schema_version": 1,
        "source_revision": os.environ.get("SOURCE_COMMIT", SOURCE_COMMIT),
        "model": RAGGED_MODEL["id"],
        "model_revision": RAGGED_MODEL["revision"],
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "quantization": "bitsandbytes-llm-int8",
        "stack": {
            "torch": torch.__version__,
            "bitsandbytes": QUANTIZATION_PINS["bitsandbytes"],
        },
        "passed": passed,
        "exact_token_parity": exact,
        "teacher_forced_top1_agreement": teacher_top1_agreement,
        "teacher_forced_tokens": teacher_tokens,
        "quality_corpus": "README.md first 2048 tokenizer tokens",
        "quality_corpus_sha256": quality_corpus_sha256,
        "fp16_mean_nll": fp16_nll,
        "int8_mean_nll": int8_nll,
        "perplexity_ratio": perplexity_ratio,
        "first_token_agreement": first_token_agreement,
        "positional_token_agreement": token_agreement,
        "measured_sequences_per_mode": 48,
        "tokens_per_sequence": 8,
        "steady_memory_ratio": memory_ratio,
        "steady_memory_reduction_percent": (1 - memory_ratio) * 100,
        "latency_ratio": latency_ratio,
        "fp16": fp16,
        "int8": int8,
    }
    print(
        f"teacher top1/PPL ratio {teacher_top1_agreement:.1%}/{perplexity_ratio:.4f}; "
        f"generated first/positional {first_token_agreement:.1%}/{token_agreement:.1%}; "
        f"steady memory -{result['steady_memory_reduction_percent']:.1f}%; "
        f"latency ratio {latency_ratio:.2f}x; passed={passed}"
    )
    return json.dumps(result, indent=2) + "\n"


@app.function(**_gpu_options(timeout=1800))
def cuda_graph_benchmark() -> str:
    """Paired L4 decode benchmark with exact-token and replay gates."""
    import asyncio
    import statistics

    import torch

    from cloud_engine.config import build_config
    from cloud_engine.engine import InferenceEngine
    from cloud_engine.scheduler import GenerationConfig

    _print_run_header("paired eager versus CUDA-graph decode benchmark")
    model_dir = _prepare_weights(MODEL)

    async def run() -> dict:
        engines = {}
        for name, enabled in (("eager", False), ("cuda_graph", True)):
            config = build_config(
                "ragged",
                model_id=MODEL["id"],
                model_revision=MODEL["revision"],
                max_model_len=MODEL["max_model_len"],
                eos_token_id=MODEL["eos_token_id"],
                max_active_sequences=4,
                prefix_cache_max_blocks=0,
                cuda_graph_decode=enabled,
            )
            engines[name] = InferenceEngine(config, model_dir=model_dir)
            await engines[name].start()

        async def generate(engine: InferenceEngine, prompts: list[str], tokens: int):
            handles = [
                await engine.submit(
                    prompt, GenerationConfig(max_output_tokens=tokens, eos_token_id=None)
                )
                for prompt in prompts
            ]
            return await asyncio.gather(*(handle.wait() for handle in handles))

        warmup = [f"CUDA graph warmup request {index}." for index in range(4)]
        await generate(engines["eager"], warmup, 8)
        await generate(engines["cuda_graph"], warmup, 8)
        timings = {"eager": [], "cuda_graph": []}
        parity = True
        try:
            for trial in range(3):
                prompts = [
                    f"Trial {trial}, sequence {index}: explain GPU launch overhead."
                    for index in range(4)
                ]
                outputs = {}
                for name in (("eager", "cuda_graph") if trial % 2 == 0 else ("cuda_graph", "eager")):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    outputs[name] = await generate(engines[name], prompts, 32)
                    torch.cuda.synchronize()
                    timings[name].append((time.perf_counter() - started) * 1000)
                parity &= [result.token_ids for result in outputs["eager"]] == [
                    result.token_ids for result in outputs["cuda_graph"]
                ]
            graph_metrics = engines["cuda_graph"].snapshot_metrics()["scheduler"]
        finally:
            await asyncio.gather(*(engine.close() for engine in engines.values()))

        medians = {name: statistics.median(values) for name, values in timings.items()}
        speedup = medians["eager"] / medians["cuda_graph"]
        if not parity or graph_metrics["cuda_graph_replays"] < 1:
            raise RuntimeError(
                f"CUDA graph gate failed: parity={parity}, metrics={graph_metrics}"
            )
        return {
            "schema_version": 1,
            "source_revision": os.environ.get("SOURCE_COMMIT", SOURCE_COMMIT),
            "model": MODEL["id"],
            "model_revision": MODEL["revision"],
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "protocol": {
                "trials": 3,
                "concurrency": 4,
                "output_tokens_per_request": 32,
                "warmup_output_tokens": 8,
                "alternating_order": True,
            },
            "exact_token_parity": parity,
            "cuda_graph_captures": graph_metrics["cuda_graph_captures"],
            "cuda_graph_replays": graph_metrics["cuda_graph_replays"],
            "trial_ms": {name: [round(value, 3) for value in values] for name, values in timings.items()},
            "median_ms": {name: round(value, 3) for name, value in medians.items()},
            "speedup": round(speedup, 4),
        }

    return json.dumps(asyncio.run(run()), indent=2) + "\n"


def _reference_generate(
    model_dir: str,
    prompt_ids: list[int],
    max_new_tokens: int,
    pad_token_id: int = MODEL["eos_token_id"],
) -> list[int]:
    """Hugging Face oracle used ONLY as a test reference (PRD G1/M1)."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    dtype_arg = (
        {"dtype": torch.float16}
        if int(transformers.__version__.split(".")[0]) >= 5
        else {"torch_dtype": torch.float16}
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, attn_implementation="eager", **dtype_arg
    ).to("cuda")
    model.eval()
    input_ids = torch.tensor([prompt_ids], device="cuda")
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=pad_token_id,
        )
    return out[0][input_ids.shape[1] :].tolist()


@app.local_entrypoint()
def benchmark(mode: str = "contiguous", profile: str = "decode", output: str | None = None) -> None:
    """Run a benchmark workload for one engine mode; optionally save JSON locally."""
    if mode not in ("naive", "contiguous", "batched", "paged", "triton", "ragged"):
        raise SystemExit(
            f"mode must be one of naive|contiguous|batched|paged|triton|ragged, got {mode!r}"
        )
    print(f"benchmark: mode={mode} profile={profile}")
    print("reminder: this runs on a billable cloud L4 GPU.\n")
    result = _run_benchmark.remote(mode, profile)
    _print_benchmark_table(result)
    if output:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2))
        print(f"\nraw JSON written to {destination}")


@app.function(**_gpu_options())
def _run_benchmark(mode: str, profile: str) -> dict:
    from benchmarks.run import run_profile

    return json.loads(json.dumps(run_profile(mode, profile)))


@app.function(**_gpu_options(timeout=3600))
def _run_engine_online(
    mode: str,
    arrival_rates: list[float],
    duration_seconds: float,
    slo_ttft_ms: float,
    slo_itl_ms: float,
) -> dict:
    import asyncio

    from benchmarks.online import run_responses_http_sweep
    from cloud_engine.api import create_app
    from cloud_engine.weights import load_tokenizer

    if mode not in ("triton", "ragged"):
        raise ValueError(f"online engine mode must be triton or ragged, got {mode!r}")
    _print_run_header(f"online arrival-rate sweep: {mode} L4")
    model_dir = _prepare_weights(RAGGED_MODEL)
    if mode == "ragged":
        from cloud_engine.config import build_config
        from cloud_engine.engine import InferenceEngine

        config = build_config(mode, prefix_cache_max_blocks=0)
        engine = InferenceEngine(config, model_dir=model_dir)
    else:
        from cloud_engine.config import build_config
        from cloud_engine.engine import InferenceEngine

        config = build_config(
            mode,
            model_id=RAGGED_MODEL["id"],
            model_revision=RAGGED_MODEL["revision"],
            max_model_len=RAGGED_MODEL["max_model_len"],
            eos_token_id=RAGGED_MODEL["eos_token_id"],
        )
        engine = InferenceEngine(config, model_dir=model_dir)
    tokenizer = load_tokenizer(model_dir)

    async def _run():
        await engine.start()
        import httpx
        import uvicorn

        api_key = "benchmark-local-only"
        port = 8001
        server = uvicorn.Server(
            uvicorn.Config(
                create_app(engine, api_key=api_key, model_id=config.model_id),
                host="127.0.0.1",
                port=port,
                log_level="warning",
            )
        )
        server_task = asyncio.create_task(server.serve())
        try:
            while not server.started:
                if server_task.done():
                    await server_task
                await asyncio.sleep(0.05)
            async with httpx.AsyncClient(timeout=600) as client:
                warmup = await client.post(
                    f"http://127.0.0.1:{port}/v1/responses",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": config.model_id,
                        "input": "Warm the packed inference engine.",
                        "max_output_tokens": 8,
                        "temperature": 0,
                    },
                )
                warmup.raise_for_status()
            return await run_responses_http_sweep(
                f"http://127.0.0.1:{port}",
                api_key,
                config.model_id,
                tokenizer,
                arrival_rates,
                duration_seconds,
                slo_ttft_ms,
                slo_itl_ms,
                engine,
                f"{mode}-l4-http",
            )
        finally:
            server.should_exit = True
            await server_task
            await engine.close()

    result = asyncio.run(_run())
    result["metadata"] = {
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "gpu": "NVIDIA L4",
        "source_revision": os.environ.get("SOURCE_COMMIT", SOURCE_COMMIT),
        "max_active_sequences": config.max_active_sequences,
        "max_batched_tokens": config.max_batched_tokens,
        "prefill_chunk_size": config.prefill_chunk_size,
        "kv_cache_bytes": config.kv_cache_bytes,
        "prefix_cache_max_blocks": config.prefix_cache_max_blocks,
    }
    return result


@app.function(
    image=vllm_image,
    gpu=MODAL_CFG["gpu"],
    volumes={"/cache": volume},
    timeout=3600,
    scaledown_window=MODAL_CFG["scaledown_window_seconds"],
    max_containers=1,
)
def _run_vllm_online(
    arrival_rates: list[float],
    duration_seconds: float,
    slo_ttft_ms: float,
    slo_itl_ms: float,
) -> dict:
    import asyncio
    import urllib.request

    from benchmarks.online import run_vllm_http_sweep
    from cloud_engine.weights import ensure_weights_downloaded, load_tokenizer

    _print_run_header("online arrival-rate sweep: vLLM 0.10.0")
    model_dir = ensure_weights_downloaded(
        "/cache", RAGGED_MODEL["revision"], RAGGED_MODEL["id"]
    )
    volume.commit()
    tokenizer = load_tokenizer(model_dir)
    base_url = "http://127.0.0.1:8000"
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model_dir),
        "--served-model-name",
        RAGGED_MODEL["id"],
        "--dtype",
        "half",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "16",
        "--max-num-batched-tokens",
        "2048",
        "--block-size",
        "16",
        "--gpu-memory-utilization",
        "0.85",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--disable-log-requests",
        "--port",
        "8000",
    ]
    server = subprocess.Popen(command)
    try:
        deadline = time.monotonic() + 300
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"vLLM server exited early with code {server.returncode}")
            try:
                with urllib.request.urlopen(f"{base_url}/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except Exception:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "vLLM server did not become healthy within 300 seconds"
                    ) from None
                time.sleep(1)
        result = asyncio.run(
            run_vllm_http_sweep(
                base_url,
                RAGGED_MODEL["id"],
                tokenizer,
                arrival_rates,
                duration_seconds,
                slo_ttft_ms,
                slo_itl_ms,
            )
        )
        result["metadata"] = {
            "model_id": RAGGED_MODEL["id"],
            "model_revision": RAGGED_MODEL["revision"],
            "gpu": "NVIDIA L4",
            "vllm_version": "0.10.0",
            "max_num_seqs": 16,
            "max_num_batched_tokens": 2048,
            "enforce_eager": True,
            "prefix_caching": False,
            "gpu_memory_utilization": 0.85,
            "source_revision": os.environ.get("SOURCE_COMMIT", SOURCE_COMMIT),
        }
        return result
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)


@app.local_entrypoint()
def online_compare(
    output: str = "artifacts/ragged-vllm-online.json",
    rates: str = "0.5,1,2,4",
    duration_seconds: float = 10,
    slo_ttft_ms: float = 1000,
    slo_itl_ms: float = 100,
) -> None:
    """Run identical online workloads on the ragged engine and pinned vLLM."""
    arrival_rates = [float(value) for value in rates.split(",")]
    print(f"online comparison: rates={arrival_rates} duration={duration_seconds}s each")
    print("reminder: this runs three sequential billable L4 jobs.\n")
    serial_triton = _run_engine_online.remote(
        "triton", arrival_rates, duration_seconds, slo_ttft_ms, slo_itl_ms
    )
    ragged = _run_engine_online.remote(
        "ragged", arrival_rates, duration_seconds, slo_ttft_ms, slo_itl_ms
    )
    vllm = _run_vllm_online.remote(
        arrival_rates, duration_seconds, slo_ttft_ms, slo_itl_ms
    )
    result = {
        "protocol": {
            "arrival_process": "fixed interval",
            "arrival_rates_requests_per_second": arrival_rates,
            "duration_seconds_per_rate": duration_seconds,
            "model_id": RAGGED_MODEL["id"],
            "model_revision": RAGGED_MODEL["revision"],
            "gpu": "NVIDIA L4",
            "max_model_len": 4096,
            "max_active_sequences": 16,
            "max_batched_tokens": 2048,
            "temperature": 0,
            "ignore_eos": False,
            "prefix_caching": False,
            "kv_cache_policy": {
                "custom_engine_bytes": PINNED["kv_cache"]["bytes"],
                "vllm_gpu_memory_utilization": 0.85,
            },
            "slo": {"ttft_ms": slo_ttft_ms, "itl_p99_ms": slo_itl_ms},
        },
        "serial_triton": serial_triton,
        "ragged": ragged,
        "vllm": vllm,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    print(f"comparison JSON written to {destination}")


def _print_benchmark_table(result: dict) -> None:
    meta = result["metadata"]
    print(
        f"mode={meta['engine_mode']} profile={meta['profile']} "
        f"gpu={meta.get('gpu_name', 'unknown')} runs={len(result['runs'])}"
    )
    med = result["median"]
    header = f"{'metric':>34} | {'median':>12}"
    print(header)
    print("-" * len(header))
    rows = [
        ("ttft_p50_ms", "ttft_p50_ms"),
        ("ttft_p95_ms", "ttft_p95_ms"),
        ("itl_p50_ms", "itl_p50_ms"),
        ("itl_p95_ms", "itl_p95_ms"),
        ("e2e_p50_ms", "e2e_p50_ms"),
        ("output_tokens_per_second", "output_tok/s"),
        ("completed_requests_per_second", "req/s"),
        ("kv_internal_fragmentation_bytes", "kv_frag_bytes"),
        ("kv_temporary_gather_bytes", "kv_gather_bytes"),
        ("peak_allocated_bytes", "peak_gpu_bytes"),
    ]
    for key, label in rows:
        value = med.get(key)
        if isinstance(value, float):
            print(f"{label:>34} | {value:>12,.1f}")
        elif value is not None:
            print(f"{label:>34} | {value:>12,}")
    print("\nall runs:")
    for index, run in enumerate(result["runs"], start=1):
        print(
            f"  run {index}: out_tok/s={run['aggregate']['output_tokens_per_second']:,.1f} "
            f"ttft_p50={run['latency']['ttft_p50_ms']:.1f}ms"
        )


@app.function(**_gpu_options())
def remote_gpu_tests() -> None:
    """Correctness suite from PRD §13.3 on a real L4."""
    import runpy

    _print_run_header("remote GPU correctness suite")
    model_dir = _prepare_weights()
    tests_dir = Path(__file__).parent / "tests"
    suite = runpy.run_path(
        str(tests_dir / "remote_gpu_tests.py"),
        run_name="__remote_gpu_tests_module__",
        init_globals={"MODEL_DIR": model_dir},
    )
    if suite["main"]():
        raise RuntimeError("remote GPU correctness suite failed")


def _execute_ragged_gpu_suite(label: str) -> str:
    import runpy

    import torch

    _print_run_header(label)
    model_dir = _prepare_weights(RAGGED_MODEL)
    tests_dir = Path(__file__).parent / "tests"
    suite = runpy.run_path(
        str(tests_dir / "remote_ragged_gpu_tests.py"),
        run_name="__ragged_gpu_tests_loaded__",
        init_globals={"MODEL_DIR": model_dir},
    )
    started = time.time()
    if suite["main"]():
        raise RuntimeError("ragged remote GPU correctness suite failed")
    properties = torch.cuda.get_device_properties(0)
    return json.dumps(
        {
            "schema_version": 1,
            "source_revision": os.environ.get("SOURCE_COMMIT", SOURCE_COMMIT),
            "model": RAGGED_MODEL["id"],
            "model_revision": RAGGED_MODEL["revision"],
            "gpu": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability()),
            "total_memory_bytes": properties.total_memory,
            "elapsed_seconds": round(time.time() - started, 3),
            "passed": sum(ok for _, ok, _ in suite["RESULTS"]),
            "failed": sum(not ok for _, ok, _ in suite["RESULTS"]),
            "checks": [
                {"name": name, "passed": ok, "detail": detail}
                for name, ok, detail in suite["RESULTS"]
            ],
        },
        indent=2,
    ) + "\n"


@app.function(**_gpu_options(timeout=3600))
def remote_ragged_gpu_tests() -> str:
    """Qwen2.5-3B packed/ragged correctness and pressure suite on one L4."""
    return _execute_ragged_gpu_suite("ragged L4 GPU correctness suite")


@app.function(**_gpu_options(gpu="A100-40GB", timeout=3600))
def remote_ragged_a100_tests() -> str:
    """The same full suite on a distinct NVIDIA Ampere target."""
    return _execute_ragged_gpu_suite("ragged A100-40GB GPU correctness suite")


@app.function(
    image=image,
    cpu=2,
    timeout=600,
    scaledown_window=MODAL_CFG["scaledown_window_seconds"],
    max_containers=MODAL_CFG["max_containers"],
)
def api_lifecycle_tests() -> str:
    """API schema/lifecycle integration tests in a CPU container (PRD §13.2)."""
    import re
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"],
        cwd="/root",
        capture_output=True,
        text=True,
    )
    print(completed.stdout)
    print(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError("CPU test suite failed")

    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from cloud_engine.api import TenantPolicy, create_app

    engine = SimpleNamespace(
        ready=True,
        config=SimpleNamespace(mode="test"),
        snapshot_metrics=lambda: {"status": "ok"},
    )
    with TestClient(create_app(engine, api_key="test-key", model_id=MODEL["id"])) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/metrics").status_code == 401
        assert client.get("/metrics", headers={"Authorization": "Bearer test-key"}).status_code == 200
        assert client.post("/v1/responses", json={}).status_code == 401
    policies = {
        "admin": TenantPolicy("a" * 32, 2, 4096, True),
        "user": TenantPolicy("u" * 32, 1, 256, False),
    }
    with TestClient(
        create_app(engine, api_key=None, model_id=MODEL["id"], tenant_policies=policies)
    ) as client:
        assert client.get(
            "/metrics", headers={"Authorization": f"Bearer {'a' * 32}"}
        ).status_code == 200
        assert client.get(
            "/metrics", headers={"Authorization": f"Bearer {'u' * 32}"}
        ).status_code == 403
        assert client.post(
            "/v1/responses",
            headers={
                "Authorization": f"Bearer {'u' * 32}",
                "Content-Length": "invalid",
            },
        ).status_code == 400
    print("FastAPI route/auth integration checks passed")
    count = re.search(r"Ran (\d+) tests", completed.stdout + completed.stderr)
    return json.dumps(
        {
            "schema_version": 1,
            "source_revision": os.environ.get("SOURCE_COMMIT", SOURCE_COMMIT),
            "passed": True,
            "unit_tests": int(count.group(1)) if count else None,
            "fastapi_checks": [
                "legacy_auth",
                "admin_metrics",
                "tenant_metrics_denied",
                "malformed_content_length",
            ],
        },
        indent=2,
    ) + "\n"


@app.cls(
    image=image,
    gpu=MODAL_CFG["gpu"],
    volumes={"/cache": volume},
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    timeout=MODAL_CFG["timeout_seconds"],
    scaledown_window=MODAL_CFG["scaledown_window_seconds"],
    max_containers=MODAL_CFG["max_containers"],
    min_containers=MODAL_CFG["min_containers"],
    buffer_containers=MODAL_CFG["buffer_containers"],
)
@modal.concurrent(max_inputs=MODAL_CFG["max_inputs_per_container"])
class ApiServer:
    """Deployed Responses-API service. Mode is fixed server-side."""

    deployed_mode = "ragged"

    @modal.enter()
    async def load(self) -> None:
        self.api_key = os.environ.get("ENGINE_API_KEY")
        tenant_json = os.environ.get("ENGINE_TENANTS_JSON")
        if not self.api_key and not tenant_json:
            raise RuntimeError(
                f"missing ENGINE_API_KEY or ENGINE_TENANTS_JSON in secret "
                f"{SECRET_NAME!r}; refusing to start (fail closed)"
            )
        model_dir = _prepare_weights(RAGGED_MODEL)
        config, self.engine = _build_engine(self.deployed_mode, model_dir)
        await self.engine.start()

        from cloud_engine.api import create_app, parse_tenant_policies

        policies = parse_tenant_policies(tenant_json) if tenant_json else None
        self.app = create_app(
            self.engine,
            api_key=self.api_key,
            model_id=config.model_id,
            tenant_policies=policies,
        )

    @modal.exit()
    async def unload(self) -> None:
        if hasattr(self, "engine"):
            await self.engine.close()

    @modal.asgi_app()
    def serve(self):  # type: ignore[no-untyped-def]
        return self.app


@app.local_entrypoint()
def main(command: str = "help") -> None:
    commands = {
        "smoke": "modal run modal_app.py::smoke",
        "ragged-smoke": "modal run modal_app.py::ragged_smoke",
        "benchmark": "modal run modal_app.py::benchmark --mode <naive|contiguous|batched|paged|triton|ragged>",
        "test-gpu": "modal run modal_app.py::remote_gpu_tests",
        "test-ragged-gpu": "modal run modal_app.py::remote_ragged_gpu_tests",
        "test-cpu": "modal run modal_app.py::api_lifecycle_tests",
        "download-weights": "modal run modal_app.py::_ensure_weights",
        "download-ragged-weights": "modal run modal_app.py::_ensure_ragged_weights",
    }
    if command == "help":
        print("available commands:")
        for name, usage in commands.items():
            print(f"  {name:16} -> {usage}")
        print("\nreminder: GPU commands are billable.")
        return
    mapping = {
        "smoke": smoke,
        "test-gpu": remote_gpu_tests,
        "download-weights": lambda: _ensure_weights.remote(),
    }
    if command in ("smoke", "test-gpu"):
        mapping[command].remote()
    elif command == "download-weights":
        mapping[command]()
    else:
        raise SystemExit(f"unknown command {command!r}; use --command help")
