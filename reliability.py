"""Run a correctness-gated concurrent reliability soak on one Modal L4.

Usage: modal run reliability.py --duration-seconds 120
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

from modal_app import RAGGED_MODEL, SOURCE_COMMIT, _gpu_options, _prepare_weights, image

ROOT = Path(__file__).parent
SOAK_SOURCE = f"{SOURCE_COMMIT}+soak-{hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]}"
app = modal.App("cloud-inference-reliability")
soak_image = image.add_local_file(
    ROOT / "modal_app.py", "/root/modal_app.py", copy=True
).env({"SOAK_SOURCE": SOAK_SOURCE})


@app.function(**_gpu_options(image=soak_image, timeout=1800))
async def run_soak(duration_seconds: float, concurrency: int) -> dict:
    import asyncio
    import gc
    import os
    import time

    import torch

    from cloud_engine.config import build_config
    from cloud_engine.engine import InferenceEngine
    from cloud_engine.scheduler import GenerationConfig

    if duration_seconds < 10 or not 1 <= concurrency <= 32:
        raise ValueError("duration must be >=10 seconds and concurrency must be 1..32")
    model_dir = _prepare_weights(RAGGED_MODEL)
    prompts = [
        "Explain paged KV cache ownership, physical block tables, fragmentation, "
        "transactional allocation, and cleanup under request cancellation.",
        "Explain continuous batching, shared token budgets, chunked prefill, queueing, "
        "backpressure, and decode-first scheduling behavior.",
        "Explain prefix caching, block-aligned KV reuse, LRU eviction, cache accounting, "
        "and exact first-token correctness.",
        "Explain recompute preemption, capacity pressure, victim selection, emitted-token "
        "authority, and deterministic recovery.",
    ]
    engine = InferenceEngine(build_config("ragged"), model_dir=model_dir)
    await engine.start()
    started = time.perf_counter()
    deadline = started + duration_seconds
    issued = completed = cancelled = failed = output_tokens = 0

    async def worker(worker_id: int) -> None:
        nonlocal issued, completed, cancelled, failed, output_tokens
        iteration = 0
        while time.perf_counter() < deadline:
            sequence = issued
            issued += 1
            handle = await engine.submit(
                prompts[(worker_id + iteration) % len(prompts)],
                GenerationConfig(max_output_tokens=8, eos_token_id=None),
            )
            if sequence % 11 == 0:
                handle.cancel()
                cancelled += 1
            else:
                try:
                    result = await handle.wait()
                    completed += 1
                    output_tokens += result.output_tokens
                except Exception:
                    failed += 1
            iteration += 1

    try:
        await asyncio.gather(*(worker(index) for index in range(concurrency)))
        for _ in range(1000):
            if not engine.scheduler.active and not engine.scheduler.waiting:
                break
            await asyncio.sleep(0.01)
        stats = engine.cache.stats()
        snapshot = engine.snapshot_metrics()
        engine.cache.assert_invariants()
        checks = {
            "zero_failed_requests": failed == 0,
            "all_issued_terminal": issued == completed + cancelled,
            "zero_request_blocks": stats.request_blocks_used == 0,
            "prefix_cache_exercised": stats.prefix_cache_hits > 0,
            "packed_forward_exercised": snapshot["scheduler"][
                "max_forward_request_count"
            ]
            >= 2,
        }
        if not all(checks.values()):
            raise RuntimeError(f"soak gate failed: {checks}")
        first_run = {
            "duration_seconds": round(time.perf_counter() - started, 3),
            "concurrency": concurrency,
            "issued": issued,
            "completed": completed,
            "cancelled": cancelled,
            "failed": failed,
            "output_tokens": output_tokens,
            "requests_per_second": round(issued / (time.perf_counter() - started), 3),
            "checks": checks,
            "metrics": snapshot,
        }
    finally:
        await engine.close()

    engine = None
    gc.collect()
    torch.cuda.empty_cache()
    restarted = InferenceEngine(build_config("ragged"), model_dir=model_dir)
    await restarted.start()
    try:
        handle = await restarted.submit(
            prompts[0], GenerationConfig(max_output_tokens=8, eos_token_id=None)
        )
        result = await handle.wait()
        restart_stats = restarted.cache.stats()
        restart_ok = result.output_tokens == 8 and restart_stats.request_blocks_used == 0
        if not restart_ok:
            raise RuntimeError("engine restart gate failed")
    finally:
        await restarted.close()

    return {
        "source_revision": os.environ["SOAK_SOURCE"],
        "model_id": RAGGED_MODEL["id"],
        "model_revision": RAGGED_MODEL["revision"],
        "gpu": torch.cuda.get_device_name(0),
        "soak": first_run,
        "restart": {
            "passed": restart_ok,
            "output_tokens": result.output_tokens,
            "request_blocks_after_run": restart_stats.request_blocks_used,
        },
    }


@app.local_entrypoint()
def main(
    duration_seconds: float = 120,
    concurrency: int = 8,
    output: str = "artifacts/reliability-soak-l4.json",
) -> None:
    print("Concurrent cancellation/restart soak on one billable L4.")
    result = run_soak.remote(duration_seconds, concurrency)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["soak"]["checks"], indent=2))
    print(f"artifact: {destination}")
