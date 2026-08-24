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
from pathlib import Path

import modal

PINNED = json.loads(Path(__file__).parent.joinpath("engine_config.json").read_text())
IMAGE_PINS = PINNED["image"]
MODEL = PINNED["model"]
MODAL_CFG = PINNED["modal"]

APP_NAME = MODAL_CFG["app_name"]
VOLUME_NAME = MODAL_CFG["volume_name"]
SECRET_NAME = MODAL_CFG["secret_name"]


def _source_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True
        ).strip()
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


def _prepare_weights() -> str:
    """Download the pinned snapshot into /cache if absent; return local dir."""
    os.makedirs("/cache/triton-cache", exist_ok=True)
    from cloud_engine.weights import ensure_weights_downloaded

    path = ensure_weights_downloaded("/cache", MODEL["revision"])
    volume.commit()
    print(f"weights ready at {path} (revision {MODEL['revision']})")
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
    if mode not in ("naive", "contiguous", "batched", "paged", "triton"):
        raise SystemExit(f"mode must be one of naive|contiguous|batched|paged|triton, got {mode!r}")
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

    deployed_mode = "triton"

    @modal.enter()
    async def load(self) -> None:
        self.api_key = os.environ.get("ENGINE_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                f"missing ENGINE_API_KEY secret {SECRET_NAME!r}; refusing to start (fail closed)"
            )
        model_dir = _prepare_weights()
        _, self.engine = _build_engine(self.deployed_mode, model_dir)
        await self.engine.start()

        from cloud_engine.api import create_app

        self.app = create_app(self.engine, api_key=self.api_key, model_id=MODEL["id"])

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
        "benchmark": "modal run modal_app.py::benchmark --mode <naive|contiguous|batched|paged|triton>",
        "test-gpu": "modal run modal_app.py::remote_gpu_tests",
        "test-cpu": "modal run modal_app.py::api_lifecycle_tests",
        "download-weights": "modal run modal_app.py::_ensure_weights",
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
