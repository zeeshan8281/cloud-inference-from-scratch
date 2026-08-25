"""Capture real L4 NVTX and CUDA-kernel traces.

Usage: modal run nvidia_profile.py
"""

from __future__ import annotations

import json
import subprocess
from hashlib import sha256
from pathlib import Path

import modal

from modal_app import MODAL_CFG, RAGGED_MODEL, SOURCE_COMMIT, image, volume

ROOT = Path(__file__).parent
NSYS_VERSION = "2026.1.3"
NSYS_PACKAGE = "nsight-systems-2026.1.3_2026.1.3.425-1_amd64.deb"
NSYS_SHA256 = "c7309f1c9850f66a9eb95e7215883b8e8e439df6f65ea0cecd81e6b0181a4e83"
NSYS_URL = f"https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/{NSYS_PACKAGE}"

profile_image = image.apt_install("ca-certificates", "wget").run_commands(
    f"wget -q {NSYS_URL} -O /tmp/nsys.deb",
    f"echo '{NSYS_SHA256}  /tmp/nsys.deb' | sha256sum -c -",
    "apt-get update && apt-get install -y --no-install-recommends /tmp/nsys.deb",
    "rm /tmp/nsys.deb && rm -rf /var/lib/apt/lists/*",
).add_local_file(
    ROOT / "modal_app.py", "/root/modal_app.py", copy=True
).env({"PROFILE_SOURCE_REVISION": SOURCE_COMMIT})

app = modal.App("cloud-inference-nsys")


@app.function(
    image=profile_image,
    gpu=MODAL_CFG["gpu"],
    volumes={"/cache": volume},
    timeout=1800,
    scaledown_window=MODAL_CFG["scaledown_window_seconds"],
    max_containers=1,
)
def capture() -> bytes:
    import os
    import shutil
    import tempfile

    from cloud_engine.weights import ensure_weights_downloaded

    model_dir = ensure_weights_downloaded(
        "/cache", RAGGED_MODEL["revision"], RAGGED_MODEL["id"]
    )
    output_dir = Path(tempfile.mkdtemp(prefix="cloud-inference-nsys-"))
    report = output_dir / "ragged-l4"
    torch_trace = output_dir / "ragged-l4.pt.trace.json"
    env = os.environ | {"ENGINE_NVTX": "1", "MODEL_DIR": str(model_dir)}
    command = [
        "nsys",
        "profile",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        "--force-overwrite=true",
        f"--output={report}",
        "python",
        "/root/benchmarks/profile_ragged.py",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, env=env, timeout=1600)
    if completed.returncode:
        raise RuntimeError(
            f"Nsight Systems failed ({completed.returncode}):\n"
            f"{(completed.stdout + completed.stderr)[-8000:]}"
        )
    report_path = report.with_suffix(".nsys-rep")
    if not report_path.is_file():
        raise RuntimeError("Nsight Systems did not create an .nsys-rep artifact")
    stats = subprocess.run(
        [
            "nsys",
            "stats",
            "--report=nvtx_sum,cuda_gpu_kern_sum",
            "--format=csv",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    stats_output = stats.stdout + stats.stderr
    if stats.returncode or "scheduler.execute_batch" not in stats_output:
        raise RuntimeError(f"Nsight report validation failed:\n{stats_output[-8000:]}")
    report.with_suffix(".sqlite").unlink(missing_ok=True)
    torch_run = subprocess.run(
        ["python", "/root/benchmarks/profile_ragged.py"],
        capture_output=True,
        text=True,
        env=env | {"ENGINE_NVTX": "0", "TORCH_TRACE_PATH": str(torch_trace)},
        timeout=1600,
    )
    if torch_run.returncode or not torch_trace.is_file():
        raise RuntimeError(
            f"PyTorch CUDA profiler failed ({torch_run.returncode}):\n"
            f"{(torch_run.stdout + torch_run.stderr)[-8000:]}"
        )
    trace_text = torch_trace.read_text()
    if '"cat": "kernel"' not in trace_text and '"cat":"kernel"' not in trace_text:
        raise RuntimeError("PyTorch trace does not contain CUDA kernel records")
    nsys_has_cuda_kernels = "does not contain CUDA kernel data" not in stats_output
    metadata = {
        "nsight_systems_version": NSYS_VERSION,
        "model_id": RAGGED_MODEL["id"],
        "model_revision": RAGGED_MODEL["revision"],
        "gpu": MODAL_CFG["gpu"],
        "source_revision": os.environ["PROFILE_SOURCE_REVISION"],
        "report_sha256": sha256(report_path.read_bytes()).hexdigest(),
        "torch_trace_sha256": sha256(torch_trace.read_bytes()).hexdigest(),
        "nsight_cuda_kernel_records": nsys_has_cuda_kernels,
        "pytorch_cuda_kernel_records": True,
        "console": completed.stdout + completed.stderr,
        "torch_console": torch_run.stdout + torch_run.stderr,
        "stats": stats_output,
    }
    output_dir.joinpath("cloud-inference-run.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    archive = shutil.make_archive(str(output_dir), "zip", output_dir)
    return Path(archive).read_bytes()


@app.local_entrypoint()
def main(output: str = "artifacts/nsight-ragged-l4.zip") -> None:
    print("NVIDIA Nsight NVTX + PyTorch CUDA-kernel capture on one billable L4.")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(capture.remote())
    print(f"Nsight Systems artifact: {destination}")
