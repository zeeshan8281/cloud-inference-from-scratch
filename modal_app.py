"""Modal application: all heavy execution lives here (PRD G2).

Local machine: source + this file + the lightweight Modal client. Weights,
Torch, Triton compilation, inference, and benchmarks run remotely on one L4
with max_containers=1 and scale-to-zero.

Cost controls (PRD §11): no schedules, no warm containers, no auto-benchmark
on deploy. Every GPU command prints a billable-compute reminder.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
import sys
import tempfile
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


def _deployed_container_limit() -> int:
    raw = os.environ.get("ENGINE_MAX_CONTAINERS", str(MODAL_CFG["max_containers"]))
    try:
        limit = int(raw)
    except ValueError as exc:
        raise ValueError("ENGINE_MAX_CONTAINERS must be an integer") from exc
    if not 1 <= limit <= MODAL_CFG["distributed_max_containers"]:
        raise ValueError(
            f"ENGINE_MAX_CONTAINERS must be between 1 and "
            f"{MODAL_CFG['distributed_max_containers']}"
        )
    return limit


DEPLOYED_MAX_CONTAINERS = _deployed_container_limit()


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
        controlled_runner = Path(__file__).parent / "experiments/controlled.py"
        if controlled_runner.is_file():
            paths.append(controlled_runner)
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
        f"redis=={IMAGE_PINS['redis']}",
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
        Path(__file__).parent / "experiments/controlled.py",
        "/root/experiments/controlled.py",
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
    .add_local_dir(
        Path(__file__).parent / "ui",
        "/root/ui",
        copy=True,
    )
    .add_local_file(
        Path(__file__).parent / "tokens.css",
        "/root/tokens.css",
        copy=True,
    )
    .add_local_file(
        Path(__file__).parent / "README.md",
        "/root/README.md",
        copy=True,
    )
    .add_local_file(
        Path(__file__).parent / "system-architecture.excalidraw.json",
        "/root/system-architecture.excalidraw.json",
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
        Path(__file__).parent / "experiments/controlled.py",
        "/root/experiments/controlled.py",
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

# One image capable of running both engines: vLLM 0.10.0 pins torch==2.7.1,
# the same torch the custom Triton kernel is pinned against, so both engines
# can share a single container for the sentinel pilot's paired child processes.
sentinel_image = (
    modal.Image.debian_slim(python_version=IMAGE_PINS["python_version"])
    .pip_install(
        f"torch=={IMAGE_PINS['torch']}",
        f"triton=={IMAGE_PINS['triton']}",
        f"safetensors=={IMAGE_PINS['safetensors']}",
        "vllm==0.10.0",
        "transformers==4.53.2",
        "tokenizers==0.21.2",
        "huggingface_hub==0.33.4",
    )
    .add_local_dir(
        Path(__file__).parent / "src/cloud_engine",
        "/root/cloud_engine",
        copy=True,
    )
    .add_local_file(
        Path(__file__).parent / "experiments/controlled.py",
        "/root/experiments/controlled.py",
        copy=True,
    )
    .add_local_file(
        Path(__file__).parent / "experiments/sentinel_pilot.py",
        "/root/experiments/sentinel_pilot.py",
        copy=True,
    )
    .add_local_file(
        Path(__file__).parent / "experiments/sentinel_diagnostics.py",
        "/root/experiments/sentinel_diagnostics.py",
        copy=True,
    )
    .add_local_file(
        Path(__file__).parent / "experiments/protocol_v2.py",
        "/root/experiments/protocol_v2.py",
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


@app.local_entrypoint()
def online_ragged(
    output: str = "artifacts/ragged-vllm-online.json",
    rates: str = "0.5,1,2,4",
    duration_seconds: float = 10,
    slo_ttft_ms: float = 1000,
    slo_itl_ms: float = 100,
) -> None:
    """Rerun only ragged after an engine change, preserving the pinned baseline."""
    destination = Path(output)
    result = json.loads(destination.read_text())
    arrival_rates = [float(value) for value in rates.split(",")]
    ragged = _run_engine_online.remote(
        "ragged", arrival_rates, duration_seconds, slo_ttft_ms, slo_itl_ms
    )
    if ragged["workload_hash"] != result["vllm"]["workload_hash"]:
        raise RuntimeError("ragged and vLLM workloads differ")
    result["ragged"] = ragged
    destination.write_text(json.dumps(result, indent=2))
    print(f"ragged comparison JSON updated at {destination}")


def _controlled_child(implementation: str, operation: str, variant: str) -> dict:
    model_dir = _prepare_weights(RAGGED_MODEL)
    output = Path(tempfile.mkdtemp(prefix="controlled-")) / "result.json"
    subprocess.run(
        [
            sys.executable,
            "/root/experiments/controlled.py",
            implementation,
            operation,
            variant,
            model_dir,
            str(output),
        ],
        check=True,
        env=dict(os.environ, PYTHONPATH="/root"),
    )
    return json.loads(output.read_text())


@app.function(**_gpu_options(timeout=3600))
def _controlled_custom(operation: str, variant: str = "complete") -> dict:
    """Run one clean custom-engine process for correctness or the full matrix."""
    _print_run_header(f"controlled {operation}: custom/{variant}")
    return _controlled_child("custom", operation, variant)


@app.function(
    image=vllm_image,
    gpu=MODAL_CFG["gpu"],
    volumes={"/cache": volume},
    timeout=3600,
    scaledown_window=MODAL_CFG["scaledown_window_seconds"],
    max_containers=1,
)
def _controlled_vllm(operation: str) -> dict:
    """Run one clean vLLM process for correctness or the full matrix."""
    _print_run_header(f"controlled {operation}: vLLM")
    return _controlled_child("vllm", operation, "complete")


def _write_controlled_environment(
    path: Path,
    custom: dict,
    vllm: dict,
    benchmark_results: list[dict] | None = None,
) -> None:
    fields = (
        ("GPU", "gpu"),
        ("GPU memory", "gpu_memory_bytes"),
        ("Driver", "driver"),
        ("CUDA", "cuda"),
        ("PyTorch", "pytorch"),
        ("Python", "python"),
        ("Model", "model"),
        ("Model revision", "model_revision"),
        ("Tokenizer", "tokenizer_class"),
        ("dtype", "dtype"),
    )
    lines = ["# Controlled experiment environment", ""]
    for label, key in fields:
        lines.append(f"{label}: {custom[key]}")
    lines.extend(
        [
            f"Tokenizer package (custom): {custom['tokenizers']}",
            f"Tokenizer package (vLLM): {vllm['tokenizers']}",
            f"Transformers (custom): {custom['transformers']}",
            f"Transformers (vLLM): {vllm['transformers']}",
            f"vLLM version: {vllm['vllm']}",
            f"Correctness source (custom): {custom['repository_commit']}",
            f"Correctness source (vLLM): {vllm['repository_commit']}",
            "",
            "Both runtimes receive the exact input token IDs stored in workloads.jsonl; "
            "client-side tokenization is bypassed during measurement.",
        ]
    )
    if benchmark_results:
        sources: dict[str, set[str]] = {}
        for result in benchmark_results:
            sources.setdefault(result["environment"]["implementation"], set()).add(
                result["environment"]["repository_commit"]
            )
        lines.extend(
            f"Benchmark source ({implementation}): {', '.join(sorted(commits))}"
            for implementation, commits in sorted(sources.items())
        )
    all_sources = {custom["repository_commit"], vllm["repository_commit"]}
    all_sources.update(
        result["environment"]["repository_commit"] for result in benchmark_results or []
    )
    if len(all_sources) == 1:
        lines.insert(2, f"Repository commit: {all_sources.pop()}")
    path.write_text("\n".join(lines) + "\n")


def _write_controlled_summary(root: Path, results: list[dict]) -> None:
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for result in results:
        implementation = result["environment"]["implementation"]
        variant = result["variant"]
        for cell in result["cells"]:
            grouped.setdefault((implementation, variant, cell["cell"]["name"]), []).append(
                cell["summary"]
            )
    destination = root / "summaries/results.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    metrics = (
        "ttft_ms",
        "itl_ms",
        "total_request_latency_ms",
        "output_tokens_per_second",
        "requests_per_second",
        "peak_gpu_memory_bytes",
        "failures",
        "timeouts",
    )
    with destination.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("implementation", "variant", "cell", "restarts", *metrics))
        for key, rows in sorted(grouped.items()):
            medians = []
            for metric in metrics:
                values = [row[metric] for row in rows if row[metric] is not None]
                medians.append(statistics.median(values) if values else "")
            writer.writerow((*key, len(rows), *medians))


@app.local_entrypoint()
def controlled_experiment(
    output: str = "experiments",
    restarts: int = 3,
    variants: str = "complete,no_triton,no_continuous_batching,no_prefix_reuse,no_cuda_graph",
    correctness_only: bool = False,
    resume: bool = False,
) -> None:
    """Correctness-gate, then run the exact matrix and supported ablations."""
    if restarts < 1:
        raise SystemExit("restarts must be positive")
    root = Path(output)
    custom_raw = root / "raw/custom-server"
    vllm_raw = root / "raw/vllm"
    custom_raw.mkdir(parents=True, exist_ok=True)
    vllm_raw.mkdir(parents=True, exist_ok=True)

    print("phase 1/2: exact token-ID correctness gate")
    custom_gate_path = custom_raw / "correctness.json"
    vllm_gate_path = vllm_raw / "correctness.json"
    if resume and custom_gate_path.exists() and vllm_gate_path.exists():
        custom_correctness = json.loads(custom_gate_path.read_text())
        vllm_correctness = json.loads(vllm_gate_path.read_text())
    else:
        custom_correctness = _controlled_custom.remote("correctness", "complete")
        vllm_correctness = _controlled_vllm.remote("correctness")
    (custom_raw / "correctness.json").write_text(json.dumps(custom_correctness, indent=2))
    (vllm_raw / "correctness.json").write_text(json.dumps(vllm_correctness, indent=2))
    _write_controlled_environment(
        root / "environment.md",
        custom_correctness["environment"],
        vllm_correctness["environment"],
    )
    with (root / "workloads.jsonl").open("w") as handle:
        for record in custom_correctness["workloads"]:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    fixed_keys = (
        "gpu",
        "gpu_memory_bytes",
        "driver",
        "cuda",
        "pytorch",
        "python",
        "model",
        "model_revision",
        "tokenizer_class",
        "tokenizers",
        "dtype",
    )
    mismatches = [
        key
        for key in fixed_keys
        if custom_correctness["environment"][key] != vllm_correctness["environment"][key]
    ]
    parity = custom_correctness["outputs"] == vllm_correctness["outputs"]
    same_workload = custom_correctness["workload_hash"] == vllm_correctness["workload_hash"]
    if mismatches or not parity or not same_workload:
        reason = {
            "environment_mismatches": mismatches,
            "exact_token_parity": parity,
            "identical_workload_hash": same_workload,
            "benchmark_started": False,
        }
        (root / "summaries").mkdir(parents=True, exist_ok=True)
        (root / "summaries/correctness-gate.json").write_text(json.dumps(reason, indent=2))
        raise SystemExit(f"correctness gate failed; speed benchmark stopped: {reason}")
    gate = {
        "environment_mismatches": [],
        "exact_token_parity": True,
        "identical_workload_hash": True,
        "custom_crashes": 0,
        "custom_ooms": 0,
        "vllm_crashes": 0,
        "vllm_ooms": 0,
        "benchmark_started": not correctness_only,
    }
    (root / "summaries").mkdir(parents=True, exist_ok=True)
    (root / "summaries/correctness-gate.json").write_text(json.dumps(gate, indent=2))
    if correctness_only:
        print(f"controlled correctness gate passed: {root}")
        return

    selected = [value.strip() for value in variants.split(",") if value.strip()]
    unknown = set(selected) - {"complete", "no_triton", "no_continuous_batching", "no_prefix_reuse", "no_cuda_graph"}
    if unknown:
        raise SystemExit(f"unknown variants: {sorted(unknown)}")
    print("phase 2/2: matrix; each result comes from a clean child process")
    results = []
    for variant in selected:
        for restart in range(1, restarts + 1):
            path = custom_raw / f"{variant}-restart-{restart}.json"
            result = (
                json.loads(path.read_text())
                if resume and path.exists()
                else _controlled_custom.remote("benchmark", variant)
            )
            results.append(result)
            path.write_text(json.dumps(result, indent=2))
    for restart in range(1, restarts + 1):
        path = vllm_raw / f"complete-restart-{restart}.json"
        result = (
            json.loads(path.read_text())
            if resume and path.exists()
            else _controlled_vllm.remote("benchmark")
        )
        results.append(result)
        path.write_text(json.dumps(result, indent=2))
    _write_controlled_environment(
        root / "environment.md",
        custom_correctness["environment"],
        vllm_correctness["environment"],
        results,
    )
    _write_controlled_summary(root, results)
    exclusions = root / "summaries/exclusions.md"
    exclusions.write_text(
        "# Excluded ablations\n\n"
        "- The no-Triton run uses eager execution because the Torch reference backend cannot "
        "replay the Triton-oriented CUDA graph. Its isolated kernel effect must be computed "
        "against `no_cuda_graph`, not against `complete`.\n"
        "- Paged KV off: excluded because the packed runner has no contiguous-KV backend; "
        "using the legacy batched mode would also change the runner, scheduler, and kernel.\n"
        "- Eviction off: excluded because recompute preemption is not independently configurable "
        "and this matrix does not intentionally force KV pressure.\n"
        "- KV block sizes 32 and 64: excluded because the v1 kernel supports only 16.\n"
    )
    print(f"controlled experiment complete: {root}")


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
        env=os.environ | {"REGENERATING_API_LIFECYCLE": "1"},
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
    with TestClient(
        create_app(engine, api_key="test-key", model_id=MODEL["id"], ui_dir="/root/ui")
    ) as client:
        assert client.get("/").status_code == 200
        assert client.get("/ui/styles.css").status_code == 200
        assert client.get("/livez").json() == {"status": "alive"}
        assert client.get("/readyz").status_code == 200
        assert client.get("/healthz").status_code == 200
        assert client.get("/v1/models").status_code == 401
        models = client.get(
            "/v1/models",
            headers={"Authorization": "Bearer test-key", "X-Request-ID": "contract-test"},
        )
        assert models.status_code == 200
        assert models.headers["X-Request-ID"] == "contract-test"
        assert models.json()["data"][0]["id"] == MODEL["id"]
        assert client.get("/metrics").status_code == 401
        assert client.get("/metrics", headers={"Authorization": "Bearer test-key"}).status_code == 200
        preflight = client.options(
            "/v1/responses",
            headers={
                "Origin": "http://127.0.0.1:8765",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
        )
        assert preflight.status_code == 200
        assert preflight.headers["Access-Control-Allow-Origin"] == "http://127.0.0.1:8765"
        assert "authorization" in preflight.headers["Access-Control-Allow-Headers"].lower()
        prometheus = client.get(
            "/metrics/prometheus", headers={"Authorization": "Bearer test-key"}
        )
        assert prometheus.status_code == 200
        assert "cie_tenant_active" in prometheus.text
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
                "liveness_readiness",
                "model_discovery",
                "prometheus_metrics",
                "request_correlation",
                "local_ui_cors",
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
    max_containers=DEPLOYED_MAX_CONTAINERS,
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
        admission_redis_url = os.environ.get("ADMISSION_REDIS_URL")
        if not self.api_key and not tenant_json:
            raise RuntimeError(
                f"missing ENGINE_API_KEY or ENGINE_TENANTS_JSON in secret "
                f"{SECRET_NAME!r}; refusing to start (fail closed)"
            )
        if DEPLOYED_MAX_CONTAINERS > 1 and not admission_redis_url:
            raise RuntimeError(
                "ADMISSION_REDIS_URL is required when ENGINE_MAX_CONTAINERS > 1; "
                "refusing unsafe process-local admission"
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
            ui_dir="/root/ui",
            admission_redis_url=admission_redis_url,
        )

    @modal.exit()
    async def unload(self) -> None:
        if hasattr(self, "engine"):
            await self.engine.close()

    @modal.asgi_app()
    def serve(self):  # type: ignore[no-untyped-def]
        return self.app


def _sentinel_torch_cuda_version() -> str | None:
    try:
        import torch

        return torch.version.cuda
    except Exception:
        return None


def _sentinel_phase_outputs(child_result: dict) -> dict:
    cells = {}
    for cell_name, cell_data in child_result["cells"].items():
        phases = {}
        for phase_name, phase_data in cell_data["phases"].items():
            records = sorted(phase_data["records"], key=lambda record: record["request_index"])
            phases[phase_name] = [record["output_token_ids"] for record in records]
        cells[cell_name] = phases
    return cells


def _run_sentinel_child(
    implementation: str, mode: str, pair: int, model_dir: str, workload_path: Path, workdir: Path
) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    output_path = workdir / "result.json"
    stdout_path = workdir / "stdout.log"
    stderr_path = workdir / "stderr.log"
    with open(stdout_path, "w") as out, open(stderr_path, "w") as err:
        completed = subprocess.run(
            [
                sys.executable,
                "/root/experiments/sentinel_pilot.py",
                implementation,
                mode,
                str(pair),
                model_dir,
                str(workload_path),
                str(output_path),
            ],
            stdout=out,
            stderr=err,
            env=dict(os.environ, PYTHONPATH="/root"),
        )
    child = {
        "implementation": implementation,
        "returncode": completed.returncode,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "crashed": completed.returncode != 0,
        "stop": None,
    }
    if child["crashed"]:
        stderr_text = stderr_path.read_text()
        child["stderr_tail"] = stderr_text[-4000:]
        if "CUDA out of memory" in stderr_text or "OutOfMemoryError" in stderr_text:
            child["oom"] = True
    else:
        payload = json.loads(output_path.read_text())
        if "stop" in payload:
            # An orderly in-process stop rule (e.g. a request timeout), not a
            # crash: the child wrote this itself instead of its full result.
            child["stop"] = payload["stop"]
        else:
            child["result"] = payload
    return child


@app.function(
    image=sentinel_image,
    gpu=MODAL_CFG["gpu"],
    volumes={"/cache": volume},
    timeout=3600,
    scaledown_window=MODAL_CFG["scaledown_window_seconds"],
    max_containers=1,
)
def _sentinel_pair(mode: str, pair: int) -> dict:
    """One paired sentinel-pilot round: two fresh child processes, one custom
    and one vLLM, run sequentially inside this single GPU allocation. Odd
    pairs run custom-then-vLLM; even pairs run the reverse."""
    from transformers import AutoTokenizer

    from experiments.sentinel_pilot import (
        SENTINEL_CELLS,
        StopPilot,
        assert_gpu_identity_stable,
        build_pair_workload,
        check_token_parity,
        gpu_state_snapshot,
    )

    _print_run_header(f"sentinel pilot pair {pair:02d}: {mode}")
    model_dir = _prepare_weights(RAGGED_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    workload = build_pair_workload(mode, pair, SENTINEL_CELLS, tokenizer)

    workdir = Path(tempfile.mkdtemp(prefix=f"sentinel-{mode}-pair{pair:02d}-"))
    workload_path = workdir / "workload.json"
    workload_path.write_text(json.dumps(workload))

    order = "odd" if pair % 2 == 1 else "even"
    implementations = ("custom", "vllm") if order == "odd" else ("vllm", "custom")
    cuda_version = _sentinel_torch_cuda_version()

    result: dict = {
        "mode": mode,
        "pair": pair,
        "order": order,
        "workload_hash": workload["workload_hash"],
        "children": [],
        "stop": None,
    }
    gpu_states = [gpu_state_snapshot(cuda_version)]
    try:
        for position, implementation in enumerate(implementations, start=1):
            child = _run_sentinel_child(
                implementation, mode, pair, model_dir, workload_path, workdir / f"child-{position}"
            )
            child["position"] = position
            result["children"].append(child)
            gpu_states.append(gpu_state_snapshot(cuda_version))
            assert_gpu_identity_stable(gpu_states)
            if child["stop"] is not None:
                raise StopPilot(
                    child["stop"]["kind"],
                    {"implementation": implementation, "position": position, **child["stop"]["detail"]},
                )
            if child["crashed"]:
                raise StopPilot(
                    "oom" if child.get("oom") else "child_crash",
                    {
                        "implementation": implementation,
                        "position": position,
                        "returncode": child["returncode"],
                        "stderr_tail": child["stderr_tail"],
                    },
                )

        by_implementation = {child["implementation"]: child["result"] for child in result["children"]}
        check_token_parity(
            _sentinel_phase_outputs(by_implementation["custom"]),
            _sentinel_phase_outputs(by_implementation["vllm"]),
        )
    except StopPilot as exc:
        result["stop"] = exc.as_dict()
    result["gpu_states"] = gpu_states
    result["workload"] = workload
    return result


@app.local_entrypoint()
def sentinel_pilot(
    output: str = "experiments/sentinel-pilot",
    modes: str = "resource_normalized,complete_system",
    resume: bool = False,
) -> None:
    """Fixed 10-pair, direct-engine closed-batch pilot vs vLLM 0.10.0 on one
    NVIDIA L4. NOT the nine-cell matrix rerun and NOT an HTTP/production
    serving benchmark -- see NEXT_EXPERIMENT_HANDOFF.md."""
    from experiments.sentinel_pilot import PAIRS, build_source_manifest

    root = Path(output)
    selected_modes = [value.strip() for value in modes.split(",") if value.strip()]
    unknown = set(selected_modes) - {"resource_normalized", "complete_system"}
    if unknown:
        raise SystemExit(f"unknown modes: {sorted(unknown)}")

    mode_dirs = {
        "resource_normalized": root / "raw/resource-normalized",
        "complete_system": root / "raw/complete-policy",
    }
    for path in mode_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).parent
    manifest = build_source_manifest(
        repo_root,
        [
            repo_root / "modal_app.py",
            repo_root / "experiments/sentinel_pilot.py",
            repo_root / "experiments/controlled.py",
            repo_root / "engine_config.json",
        ],
    )
    if manifest["dirty"]:
        raise SystemExit("refusing to run the sentinel pilot from a dirty source tree")
    (root / "source-manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"source manifest: commit {manifest['git_commit']} tree {manifest['git_tree']}")

    workloads_path = root / "workloads.jsonl"
    stopped = False
    for mode in selected_modes:
        for pair in range(1, PAIRS + 1):
            path = mode_dirs[mode] / f"pair-{pair:02d}.json"
            if resume and path.exists():
                pair_result = json.loads(path.read_text())
            else:
                pair_result = _sentinel_pair.remote(mode, pair)
                workload = pair_result.pop("workload")
                with workloads_path.open("a") as handle:
                    handle.write(json.dumps(workload, separators=(",", ":")) + "\n")
                path.write_text(json.dumps(pair_result, indent=2))
            print(f"{mode} pair {pair:02d}: order={pair_result['order']} stop={pair_result['stop']}")
            if pair_result["stop"] is not None:
                stopped = True
                break
        if stopped:
            break

    if stopped:
        print(
            "sentinel pilot STOPPED; all evidence retained under "
            f"{root}. No performance claim may be generated from a stopped pilot."
        )
        return
    print(
        f"sentinel pilot pairs complete: {root}\n"
        "run: python3 experiments/sentinel_report.py "
        f"{output} to regenerate summaries/plots"
    )


@app.function(
    image=sentinel_image,
    gpu=MODAL_CFG["gpu"],
    volumes={"/cache": volume},
    timeout=1800,
    max_containers=1,
)
def _sentinel_self_consistency_check(mode: str) -> dict:
    """Diagnostic, not part of the 10-pair protocol: does a single engine
    reproduce its own greedy-decoded output across two fresh, isolated child
    processes given the identical materialized workload? If not, cross-engine
    token mismatches at concurrency > 1 are inherent batched floating-point
    nondeterminism, not something a harness or config change could fix.

    Each of the four runs (custom x2, vLLM x2) is a genuinely fresh
    subprocess -- same isolation the real 10-pair protocol uses -- because
    reusing one process's CUDA allocator across four engine lifecycles left
    stale reserved memory that starved later engines (observed directly: a
    real ValueError/RuntimeError from vLLM's memory profiler, not a finding
    about the mismatch question)."""
    from transformers import AutoTokenizer

    from experiments.sentinel_pilot import SENTINEL_CELLS, build_pair_workload

    _print_run_header(f"sentinel pilot self-consistency check: {mode}")
    model_dir = _prepare_weights(RAGGED_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    workload = build_pair_workload(mode, 9999, SENTINEL_CELLS, tokenizer)
    phase_key = "unique" if mode == "resource_normalized" else "cold"
    check_cells = ("in512-out128-c8", "in1024-out256-c32")

    workdir = Path(tempfile.mkdtemp(prefix=f"sentinel-selfcheck-{mode}-"))
    workload_path = workdir / "workload.json"
    workload_path.write_text(json.dumps(workload))

    def cell_outputs(child_result: dict) -> dict:
        return {
            cell_name: {
                record["request_index"]: record["output_token_ids"]
                for record in child_result["cells"][cell_name]["phases"][phase_key]["records"]
            }
            for cell_name in check_cells
        }

    def diff(first: dict, second: dict) -> list[dict]:
        mismatches = []
        for cell_name in check_cells:
            for index, ids in first[cell_name].items():
                if second[cell_name].get(index) != ids:
                    mismatches.append({"cell": cell_name, "request_index": index})
        return mismatches

    runs: dict[str, list[dict]] = {"custom": [], "vllm": []}
    for implementation in ("custom", "vllm"):
        for attempt in range(2):
            child = _run_sentinel_child(
                implementation, mode, 9999, model_dir, workload_path, workdir / f"{implementation}-{attempt}"
            )
            if child["crashed"]:
                return {"mode": mode, "implementation": implementation, "crashed": True, "child": child}
            runs[implementation].append(cell_outputs(child["result"]))

    custom_mismatches = diff(runs["custom"][0], runs["custom"][1])
    vllm_mismatches = diff(runs["vllm"][0], runs["vllm"][1])
    return {
        "mode": mode,
        "custom_self_consistent": custom_mismatches == [],
        "custom_mismatches": custom_mismatches,
        "vllm_self_consistent": vllm_mismatches == [],
        "vllm_mismatches": vllm_mismatches,
    }


@app.local_entrypoint()
def sentinel_self_consistency_check(mode: str = "resource_normalized") -> None:
    """Diagnostic only. Answers: is a single engine's own greedy decode
    reproducible run-to-run at concurrency > 1, independent of the other
    engine entirely?"""
    result = _sentinel_self_consistency_check.remote(mode)
    print(json.dumps(result, indent=2))


def _run_diagnostic_child(config: dict, workdir: Path) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    config_path = workdir / "config.json"
    output_path = workdir / "output.json"
    config_path.write_text(json.dumps(config))
    stdout_path = workdir / "stdout.log"
    stderr_path = workdir / "stderr.log"
    with open(stdout_path, "w") as out, open(stderr_path, "w") as err:
        try:
            completed = subprocess.run(
                [sys.executable, "/root/experiments/sentinel_diagnostics.py", str(config_path), str(output_path)],
                stdout=out,
                stderr=err,
                env=dict(os.environ, PYTHONPATH="/root"),
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            return {"crashed": True, "stderr_tail": "timed out after 300s"}
    if completed.returncode != 0:
        return {"crashed": True, "stderr_tail": stderr_path.read_text()[-4000:]}
    return {"crashed": False, "result": json.loads(output_path.read_text())}


@app.function(
    image=sentinel_image,
    gpu=MODAL_CFG["gpu"],
    volumes={"/cache": volume},
    timeout=3600,
    max_containers=1,
)
def _sentinel_divergence_diagnostic() -> dict:
    """Bounded root-cause investigation, not the 10-pair protocol. See
    experiments/sentinel_diagnostics.py for the full design rationale."""
    from transformers import AutoTokenizer

    from experiments.sentinel_diagnostics import (
        batch_for_concurrency,
        build_diagnostic_prompts,
        natural_order_c8_batch,
    )

    _print_run_header("sentinel pilot divergence diagnostic")
    model_dir = _prepare_weights(RAGGED_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    prompts = build_diagnostic_prompts(tokenizer)

    workdir = Path(tempfile.mkdtemp(prefix="sentinel-diagnostic-"))
    runs: dict[str, dict] = {}

    def run(name: str, implementation: str, batch_ids: list, target_index: int = 0, **engine_options) -> None:
        config = {
            "implementation": implementation,
            "model_dir": model_dir,
            "batch_ids": batch_ids,
            "target_index": target_index,
            "engine_options": engine_options,
        }
        runs[name] = _run_diagnostic_child(config, workdir / name)

    # Base condition (matches the resource_normalized sentinel-pilot mode
    # exactly) at every concurrency level. Target always at batch position 0.
    for concurrency in (1, 2, 8, 32):
        batch_ids = batch_for_concurrency(prompts, concurrency)
        run(f"custom_base_c{concurrency}", "custom", batch_ids)
        run(f"vllm_base_c{concurrency}", "vllm", batch_ids)

    # The exact original c8 batch in its natural request_index submission
    # order (target at index 1, not reordered to the front). Batch
    # composition/order is itself a candidate cause, not just concurrency.
    natural_batch, natural_target_index = natural_order_c8_batch(prompts)
    run("custom_c8_original_order", "custom", natural_batch, target_index=natural_target_index)
    run("vllm_c8_original_order", "vllm", natural_batch, target_index=natural_target_index)

    # Backend matrix at c8 only (the smallest failing case), target-first order.
    c8_batch = batch_for_concurrency(prompts, 8)
    run("custom_c8_torch_attention", "custom", c8_batch, use_triton_attention=False)
    run("custom_c8_graphs_on", "custom", c8_batch, cuda_graph_decode=True)
    run("custom_c8_prefix_cache_on", "custom", c8_batch, prefix_cache_max_blocks=256)
    run("vllm_c8_graphs_on", "vllm", c8_batch, enforce_eager=False)
    run("vllm_c8_prefix_cache_on", "vllm", c8_batch, enable_prefix_caching=True)

    # Hugging Face Transformers reference, concurrency 1 only (see
    # run_hf_diagnostic's docstring for why: no padding needed at c1, so
    # nothing about HF's own batching can confound the oracle).
    for dtype in ("float16", "bfloat16", "float32"):
        config = {
            "implementation": "hf",
            "model_dir": model_dir,
            "batch_ids": batch_for_concurrency(prompts, 1),
            "dtype": dtype,
        }
        runs[f"hf_c1_{dtype}"] = _run_diagnostic_child(config, workdir / f"hf_c1_{dtype}")

    return {"runs": runs, "target_request": prompts["target"]}


@app.local_entrypoint()
def sentinel_divergence_diagnostic(
    output: str = "experiments/sentinel-pilot/summaries/divergence-diagnostic.json",
) -> None:
    """Bounded root-cause investigation of the concurrency > 1 token
    divergence. Does NOT rerun the 10-pair protocol or relax the correctness
    gate. Writes the raw result for experiments/sentinel_diagnostics.py's
    analysis/report step to consume."""
    result = _sentinel_divergence_diagnostic.remote()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    crashed = [name for name, run in result["runs"].items() if run["crashed"]]
    print(f"divergence diagnostic written to {destination}; {len(crashed)} runs crashed: {crashed}")


@app.function(
    image=sentinel_image,
    gpu=MODAL_CFG["gpu"],
    volumes={"/cache": volume},
    timeout=600,
    max_containers=1,
)
def _sentinel_diagnostic_debug_one() -> dict:
    """Debug-only, not part of any protocol: one cheap custom-engine c1 run,
    to inspect whether the logit-capture hook actually fires."""
    from transformers import AutoTokenizer

    from experiments.sentinel_diagnostics import batch_for_concurrency, build_diagnostic_prompts

    model_dir = _prepare_weights(RAGGED_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    prompts = build_diagnostic_prompts(tokenizer)
    workdir = Path(tempfile.mkdtemp(prefix="sentinel-diagnostic-debug-"))
    config = {
        "implementation": "custom",
        "model_dir": model_dir,
        "batch_ids": batch_for_concurrency(prompts, 1),
        "target_index": 0,
        "engine_options": {},
    }
    return _run_diagnostic_child(config, workdir / "debug")


@app.local_entrypoint()
def sentinel_diagnostic_debug_one() -> None:
    result = _sentinel_diagnostic_debug_one.remote()
    print(json.dumps(result, indent=2))


def _run_protocol_v2_child(config: dict, workdir: Path) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    config_path = workdir / "config.json"
    output_path = workdir / "output.json"
    config_path.write_text(json.dumps(config))
    stdout_path = workdir / "stdout.log"
    stderr_path = workdir / "stderr.log"
    with open(stdout_path, "w") as out, open(stderr_path, "w") as err:
        try:
            completed = subprocess.run(
                [sys.executable, "/root/experiments/protocol_v2.py", str(config_path), str(output_path)],
                stdout=out,
                stderr=err,
                env=dict(os.environ, PYTHONPATH="/root"),
                timeout=1200,
            )
        except subprocess.TimeoutExpired:
            return {"crashed": True, "stderr_tail": "timed out after 1200s"}
    if completed.returncode != 0:
        return {"crashed": True, "stderr_tail": stderr_path.read_text()[-4000:]}
    return {"crashed": False, "result": json.loads(output_path.read_text())}


@app.function(
    image=sentinel_image,
    gpu=MODAL_CFG["gpu"],
    volumes={"/cache": volume},
    timeout=3600,
    max_containers=1,
)
def _protocol_v2_run(split: str, epsilon: float | None) -> dict:
    """Runs one split (calibration or sealed holdout) of Correctness
    Protocol V2 (see CORRECTNESS_PROTOCOL_V2.md): every request in every
    batch is classified via experiments.protocol_v2.classify_request. When
    epsilon is None (calibration), disagreements are left pending and
    experiments.protocol_v2.propose_epsilon is applied at the end."""
    from transformers import AutoTokenizer

    from experiments.protocol_v2 import (
        V2_CELLS,
        build_split_batches,
        classify_request,
        propose_epsilon,
        summarize,
    )

    _print_run_header(f"correctness protocol v2: {split}")
    model_dir = _prepare_weights(RAGGED_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    batches_by_cell = build_split_batches(tokenizer, split)

    # One fresh subprocess per (implementation, cell, batch) -- not one
    # subprocess looping over several engine constructions, which hit a real
    # CUDA OOM after a couple of iterations (PyTorch's allocator did not
    # return the prior engine's memory before the next one allocated).
    workdir = Path(tempfile.mkdtemp(prefix=f"protocol-v2-{split}-"))
    child_results: dict[str, dict] = {}
    for implementation in ("custom", "vllm"):
        for cell in V2_CELLS:
            for batch_index, batch_ids in enumerate(batches_by_cell[cell.name]):
                config = {
                    "implementation": implementation,
                    "model_dir": model_dir,
                    "batch_ids": batch_ids,
                    "engine_options": {},
                }
                name = f"{implementation}_{cell.name}_{batch_index}"
                child_results[name] = _run_protocol_v2_child(config, workdir / name)

    crashed = {name: child for name, child in child_results.items() if child["crashed"]}
    if crashed:
        return {"split": split, "crashed": crashed, "classifications": [], "summary": None}

    classifications = []
    for cell in V2_CELLS:
        num_batches = len(batches_by_cell[cell.name])
        for batch_index in range(num_batches):
            custom_batch = child_results[f"custom_{cell.name}_{batch_index}"]["result"]["result"]
            vllm_batch = child_results[f"vllm_{cell.name}_{batch_index}"]["result"]["result"]
            for custom_request, vllm_request in zip(custom_batch, vllm_batch, strict=True):
                entry = classify_request(custom_request, vllm_request, epsilon, concurrency=cell.concurrency)
                entry["cell"] = cell.name
                entry["batch_index"] = batch_index
                entry["concurrency"] = cell.concurrency
                classifications.append(entry)

    result = {
        "split": split,
        "epsilon": epsilon,
        "crashed": {},
        "classifications": classifications,
        "summary": summarize(classifications),
    }
    if epsilon is None:
        result["proposed_epsilon"] = propose_epsilon(classifications)
    return result


@app.local_entrypoint()
def protocol_v2_calibration(
    output: str = "experiments/sentinel-pilot/summaries/protocol-v2-calibration.json",
) -> None:
    """Correctness Protocol V2, calibration split only. Proposes an epsilon;
    does NOT commit it and does NOT run the sealed holdout set. See
    CORRECTNESS_PROTOCOL_V2.md requirements 6, 7, 9."""
    result = _protocol_v2_run.remote("calibration", None)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    if result["crashed"]:
        print(f"calibration CRASHED: {list(result['crashed'])}")
        return
    print(f"calibration written to {destination}")
    print(json.dumps(result["summary"], indent=2))
    print("proposed epsilon:", json.dumps(result["proposed_epsilon"], indent=2))
    print(
        "This epsilon is NOT committed. Review it, then commit it explicitly "
        "(e.g. into CORRECTNESS_PROTOCOL_V2.md or a companion file) before "
        "running protocol_v2_holdout with it."
    )


@app.local_entrypoint()
def protocol_v2_holdout(
    epsilon: float,
    output: str = "experiments/sentinel-pilot/summaries/protocol-v2-holdout.json",
) -> None:
    """Correctness Protocol V2, sealed holdout split only. `epsilon` must be
    a value already committed to source control (requirement 9) -- this
    entrypoint does not check that itself, it only refuses to derive
    epsilon from the holdout data (epsilon is a required argument, not
    computed here)."""
    result = _protocol_v2_run.remote("holdout", epsilon)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    if result["crashed"]:
        print(f"holdout CRASHED: {list(result['crashed'])}")
        return
    print(f"holdout written to {destination}")
    print(json.dumps(result["summary"], indent=2))
    gate_passes = (
        result["summary"]["hard_failures"] == 0
        and result["summary"]["pending_epsilon"] == 0
    )
    print(f"CORRECTNESS_PROTOCOL_V2.md requirement 10 gate: {'PASSES' if gate_passes else 'DOES NOT PASS'}")
    print(
        "This does not by itself resume the 10-pair performance protocol -- "
        "that remains a separate, explicit decision."
    )


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
