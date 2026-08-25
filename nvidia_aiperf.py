"""Run NVIDIA AIPerf against the deployed Responses API.

Usage: modal run nvidia_aiperf.py
"""

from __future__ import annotations

import json
import subprocess
from hashlib import sha256
from pathlib import Path

import modal

ROOT = Path(__file__).parent
PINNED = json.loads(ROOT.joinpath("engine_config.json").read_text())
MODEL = PINNED["ragged_model"]
SECRET_NAME = PINNED["modal"]["secret_name"]
DEFAULT_URL = "https://zeeshan8281--cloud-inference-lab-apiserver-serve.modal.run"
AIPERF_VERSION = "0.12.0"


def _source_revision() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        digest = sha256()
        for relative in (
            "nvidia_aiperf.py",
            "engine_config.json",
            "src/cloud_engine/api.py",
        ):
            digest.update(relative.encode())
            digest.update(ROOT.joinpath(relative).read_bytes())
        return f"{commit}+tree-{digest.hexdigest()[:12]}"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


SOURCE_REVISION = _source_revision()

app = modal.App("cloud-inference-aiperf")
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    f"aiperf=={AIPERF_VERSION}"
).add_local_file(
    ROOT / "engine_config.json", "/root/engine_config.json", copy=True
).env({"SOURCE_REVISION": SOURCE_REVISION})


@app.function(
    image=image,
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    timeout=1800,
)
def run_aiperf(
    url: str,
    request_count: int,
    concurrency: int,
    input_tokens: int,
    output_tokens: int,
    runs: int,
) -> bytes:
    import os
    import shutil
    import subprocess
    import tempfile

    if min(request_count, concurrency, input_tokens, output_tokens, runs) < 1:
        raise ValueError("counts, concurrency, token lengths, and runs must be positive")
    artifact_dir = Path(tempfile.mkdtemp(prefix="cloud-inference-aiperf-")) / "result"
    api_key = os.environ["ENGINE_API_KEY"]
    command = [
        shutil.which("aiperf") or "aiperf",
        "profile",
        "--random-seed",
        "42",
        "--url",
        url,
        "--model",
        MODEL["id"],
        "--tokenizer",
        MODEL["id"],
        "--tokenizer-revision",
        MODEL["revision"],
        "--endpoint-type",
        "responses",
        "--endpoint",
        "/v1/responses",
        "--streaming",
        "--use-server-token-count",
        "--synthetic-input-tokens-mean",
        str(input_tokens),
        "--synthetic-input-tokens-stddev",
        "0",
        "--output-tokens-mean",
        str(output_tokens),
        "--output-tokens-stddev",
        "0",
        "--request-count",
        str(request_count),
        "--concurrency",
        str(concurrency),
        "--num-profile-runs",
        str(runs),
        "--warmup-request-count",
        "2",
        "--warmup-concurrency",
        str(min(2, concurrency)),
        "--no-server-metrics",
        "--no-gpu-telemetry",
        "--export-level",
        "records",
        "--output-artifact-dir",
        str(artifact_dir),
        "--extra-inputs",
        "temperature:0",
        "--api-key",
        api_key,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=1700)
    safe_output = (completed.stdout + completed.stderr).replace(api_key, "[REDACTED]")
    if completed.returncode:
        raise RuntimeError(f"AIPerf failed ({completed.returncode}):\n{safe_output[-8000:]}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.joinpath("cloud-inference-run.json").write_text(
        json.dumps(
            {
                "aiperf_version": AIPERF_VERSION,
                "model_id": MODEL["id"],
                "model_revision": MODEL["revision"],
                "url": url,
                "request_count": request_count,
                "concurrency": concurrency,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "runs": runs,
                "source_revision": os.environ.get("SOURCE_REVISION", SOURCE_REVISION),
                "console": safe_output,
            },
            indent=2,
        )
        + "\n"
    )
    archive = shutil.make_archive(str(artifact_dir), "zip", artifact_dir)
    return Path(archive).read_bytes()


@app.local_entrypoint()
def main(
    url: str = DEFAULT_URL,
    request_count: int = 20,
    concurrency: int = 4,
    input_tokens: int = 128,
    output_tokens: int = 32,
    runs: int = 3,
    output: str = "artifacts/aiperf-responses.zip",
) -> None:
    print("NVIDIA AIPerf Responses-API benchmark (remote endpoint; CPU load generator).")
    payload = run_aiperf.remote(
        url, request_count, concurrency, input_tokens, output_tokens, runs
    )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    print(f"AIPerf artifact: {destination}")
