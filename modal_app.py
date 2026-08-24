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
MODEL = PINNED["model"]
RAGGED_MODEL = PINNED["ragged_model"]
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
    import asyncio

    from cloud_engine.config import build_config
    from cloud_engine.engine import InferenceEngine
    from cloud_engine.scheduler import GenerationConfig

    _print_run_header("ragged smoke: real multi-request forward")
    model_dir = _prepare_weights(MODEL)
    config = build_config(
        "ragged",
        model_id=MODEL["id"],
        model_revision=MODEL["revision"],
        max_model_len=MODEL["max_model_len"],
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
        _reference_generate(model_dir, tokenizer.encode(prompt), 8) for prompt in prompts
    ]
    got = [result.token_ids for result in results]
    if got != reference:
        raise RuntimeError(f"ragged token parity failed: got={got}, reference={reference}")
    print(f"max request IDs in one model invocation: {max_requests}")
    print(f"last packed IDs: {snapshot['scheduler']['last_forward_request_ids']}")
    print("token parity vs Hugging Face: exact")
    print("RAGGED SMOKE PASSED")
    return {"passed": True, "max_forward_request_count": max_requests}


def _reference_generate(model_dir: str, prompt_ids: list[int], max_new_tokens: int) -> list[int]:
    """Hugging Face oracle used ONLY as a test reference (PRD G1/M1)."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float16, attn_implementation="eager"
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
            pad_token_id=MODEL["eos_token_id"],
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
        config, engine = _build_engine(mode, model_dir)
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


@app.function(**_gpu_options(timeout=3600))
def remote_ragged_gpu_tests() -> None:
    """Qwen2.5-3B packed/ragged correctness and pressure suite on one L4."""
    import runpy

    _print_run_header("ragged L4 GPU correctness suite")
    model_dir = _prepare_weights(RAGGED_MODEL)
    tests_dir = Path(__file__).parent / "tests"
    suite = runpy.run_path(
        str(tests_dir / "remote_ragged_gpu_tests.py"),
        run_name="__ragged_gpu_tests_loaded__",
        init_globals={"MODEL_DIR": model_dir},
    )
    if suite["main"]():
        raise RuntimeError("ragged remote GPU correctness suite failed")


@app.function(
    image=image,
    cpu=2,
    timeout=600,
    scaledown_window=MODAL_CFG["scaledown_window_seconds"],
    max_containers=MODAL_CFG["max_containers"],
)
def api_lifecycle_tests() -> str:
    """API schema/lifecycle integration tests in a CPU container (PRD §13.2)."""
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

    from cloud_engine.api import create_app

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
    print("FastAPI route/auth integration checks passed")
    return "cpu-tests-passed"


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
        if not self.api_key:
            raise RuntimeError(
                f"missing ENGINE_API_KEY secret {SECRET_NAME!r}; refusing to start (fail closed)"
            )
        model_dir = _prepare_weights(RAGGED_MODEL)
        config, self.engine = _build_engine(self.deployed_mode, model_dir)
        await self.engine.start()

        from cloud_engine.api import create_app

        self.app = create_app(self.engine, api_key=self.api_key, model_id=config.model_id)

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
