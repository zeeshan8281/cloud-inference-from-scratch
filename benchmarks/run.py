"""Reproducible benchmark runner (PRD §12).

Protocol per mode: fresh engine, identical pinned weights, two unmeasured
warmup requests, reset peak-memory and metrics, then three measured runs of
the selected workload. Every run is reported alongside the median; nothing is
discarded, retried, or cherry-picked. Output JSON contains counts and hashes
only — never prompts or generated text.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

WORKLOADS_PATH = Path(__file__).parent / "workloads.json"


@dataclass
class WorkloadItem:
    prompt: str
    prompt_tokens: int
    output_tokens: int


def _load_spec() -> dict[str, Any]:
    return json.loads(WORKLOADS_PATH.read_text())


def build_workload(profile: str, tokenizer: Any) -> list[WorkloadItem]:
    """Deterministically compose prompts hitting the profile's token targets."""
    spec = _load_spec()
    if profile not in spec["profiles"]:
        raise ValueError(f"unknown profile {profile!r}; choose from {sorted(spec['profiles'])}")
    settings = spec["profiles"][profile]
    bank = spec["word_bank"]
    rng = random.Random(spec["seed"] + hash(profile) % 100000)

    items: list[WorkloadItem] = []
    for index in range(settings["concurrent_requests"]):
        target_prompt = settings.get("prompt_tokens_target")
        if target_prompt is None:
            lo, hi = settings["prompt_tokens_min"], settings["prompt_tokens_max"]
            target_prompt = lo + (hi - lo) * index // max(1, settings["concurrent_requests"] - 1)
        tolerance = settings.get("prompt_tokens_tolerance", 0)
        out_lo, out_hi = settings["output_tokens_min"], settings["output_tokens_max"]
        output_target = out_lo + (out_hi - out_lo) * index // max(1, settings["concurrent_requests"] - 1)

        words: list[str] = []
        prompt_text = ""
        prompt_len = 0
        while True:
            while prompt_len < target_prompt - (tolerance if tolerance else 0):
                words.append(rng.choice(bank))
                prompt_text = " ".join(words) + "."
                prompt_len = len(tokenizer.encode(prompt_text))
            if tolerance == 0 or abs(prompt_len - target_prompt) <= tolerance:
                break
            # overshoot with strict tolerance: rebuild deterministically
            words = []
            prompt_text = ""
            prompt_len = 0
            for _ in range(max(1, int(target_prompt * 3 / 4))):
                words.append(rng.choice(bank))
                prompt_text = " ".join(words) + "."
                prompt_len = len(tokenizer.encode(prompt_text))
            if abs(prompt_len - target_prompt) <= tolerance + 2:
                break
            rng = random.Random(spec["seed"] + index * 977 + len(words))
        items.append(
            WorkloadItem(
                prompt=prompt_text,
                prompt_tokens=prompt_len,
                output_tokens=output_target,
            )
        )
    return items


def workload_hash(items: list[WorkloadItem]) -> str:
    canonical = json.dumps(
        [[i.prompt_tokens, i.output_tokens] for i in items], separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _percentile(values: list[float], pct: float) -> float:
    import math

    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    low = math.floor(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _metadata(mode: str, profile: str, items: list[WorkloadItem]) -> dict[str, Any]:
    import torch
    import transformers
    from cloud_engine.__version__ import __version__ as pkg_version

    from cloud_engine.config import load_pinned

    pinned = load_pinned()
    return {
        "engine_mode": mode,
        "profile": profile,
        "workload_hash": workload_hash(items),
        "model_id": pinned["model"]["id"],
        "model_revision": pinned["model"]["revision"],
        "package_version": pkg_version,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "triton_version": getattr(__import__("triton"), "__version__", "unknown"),
        "transformers_version": transformers.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "source_revision": os.environ.get("SOURCE_COMMIT", "unpinned-local-image"),
        "note": "cloud compute is billable; results include every run",
    }


async def _execute_run(engine: Any, items: list[WorkloadItem]) -> dict[str, Any]:
    from cloud_engine.scheduler import GenerationConfig

    async def drive(item: WorkloadItem) -> dict[str, Any]:
        handle = await engine.submit(
            item.prompt,
            GenerationConfig(
                max_output_tokens=item.output_tokens,
                eos_token_id=engine.config.eos_token_id,
            ),
        )
        result = await handle.wait()
        async for _ in handle.stream():  # drains any remaining stream events
            pass
        return {
            "ttft_ms": result.ttft_ms or 0.0,
            "e2e_ms": result.e2e_ms,
            "output_tokens": result.output_tokens,
            "input_tokens": result.input_tokens,
            "finish_reason": result.finish_reason,
        }

    wall_start = time.perf_counter()
    records = list(await asyncio.gather(*(drive(item) for item in items)))
    wall_s = time.perf_counter() - wall_start

    ttfts = [r["ttft_ms"] for r in records]
    e2es = [r["e2e_ms"] for r in records]
    total_output = sum(r["output_tokens"] for r in records)
    total_input = sum(r["input_tokens"] for r in records)

    snapshot = engine.snapshot_metrics()
    per_request_itl = [
        (r["e2e_ms"] - r["ttft_ms"]) / max(1, r["output_tokens"] - 1)
        for r in records
        if r["output_tokens"] > 1
    ]
    return {
        "latency": {
            "ttft_p50_ms": round(_percentile(ttfts, 50), 2),
            "ttft_p95_ms": round(_percentile(ttfts, 95), 2),
            "itl_p50_ms": round(_percentile(per_request_itl, 50), 2),
            "itl_p95_ms": round(_percentile(per_request_itl, 95), 2),
            "e2e_p50_ms": round(_percentile(e2es, 50), 2),
            "e2e_p95_ms": round(_percentile(e2es, 95), 2),
        },
        "aggregate": {
            "wall_seconds": round(wall_s, 3),
            "input_tokens_per_second": round(total_input / wall_s, 1),
            "output_tokens_per_second": round(total_output / wall_s, 1),
            "completed_requests_per_second": round(len(records) / wall_s, 2),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
        },
        "scheduler_mean_batch_size_60s": snapshot["scheduler"]["mean_batch_size_60s"],
        "scheduler_max_batch_size_60s": snapshot["scheduler"]["max_batch_size_60s"],
        "kv_internal_fragmentation_bytes": snapshot["kv_cache"].get("internal_fragmentation_bytes", 0),
        "kv_temporary_gather_bytes": snapshot["kv_cache"].get("temporary_gather_bytes", 0),
        "requests_failed": snapshot["requests"]["failed_total"],
        "requests_cancelled": snapshot["requests"]["cancelled_total"],
        "requests_rejected": snapshot["requests"]["rejected_total"],
    }


async def _run_profile_async(mode: str, profile: str) -> dict[str, Any]:
    from cloud_engine.config import build_config, load_pinned
    from cloud_engine.engine import InferenceEngine
    from cloud_engine.weights import ensure_weights_downloaded, load_tokenizer

    model_dir = ensure_weights_downloaded("/cache", load_pinned()["model"]["revision"])
    tokenizer = load_tokenizer(model_dir)
    items = build_workload(profile, tokenizer)

    config = build_config(mode)
    engine = InferenceEngine(config, model_dir=model_dir)
    await engine.start()

    try:
        # two unmeasured warmup requests (PRD §12.2)
        for item in items[:2]:
            handle = await engine.submit(item.prompt, engine.new_generation_config(4))
            await handle.wait()

        runs = []
        for run_index in range(3):
            engine.metrics.reset_runtime()
            engine.reset_peak_memory()
            run = await _execute_run(engine, items)
            run["run_index"] = run_index
            import torch

            run["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            run["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            runs.append(run)
    finally:
        await engine.close()

    def med(key_chain: list[str]) -> float:
        extracted = []
        for run in runs:
            node: Any = run
            for key in key_chain:
                node = node[key]
            extracted.append(float(node))
        return statistics.median(extracted)

    median_view = {
        "ttft_p50_ms": med(["latency", "ttft_p50_ms"]),
        "ttft_p95_ms": med(["latency", "ttft_p95_ms"]),
        "itl_p50_ms": med(["latency", "itl_p50_ms"]),
        "itl_p95_ms": med(["latency", "itl_p95_ms"]),
        "e2e_p50_ms": med(["latency", "e2e_p50_ms"]),
        "output_tokens_per_second": med(["aggregate", "output_tokens_per_second"]),
        "completed_requests_per_second": med(["aggregate", "completed_requests_per_second"]),
        "kv_internal_fragmentation_bytes": med(["kv_internal_fragmentation_bytes"]),
        "kv_temporary_gather_bytes": med(["kv_temporary_gather_bytes"]),
        "peak_allocated_bytes": med(["peak_allocated_bytes"]),
    }
    return {"metadata": _metadata(mode, profile, items), "runs": runs, "median": median_view}


def run_profile(mode: str, profile: str) -> dict[str, Any]:
    return asyncio.run(_run_profile_async(mode, profile))


if __name__ == "__main__":  # direct container debugging convenience
    selected_mode = sys.argv[1] if len(sys.argv) > 1 else "contiguous"
    selected_profile = sys.argv[2] if len(sys.argv) > 2 else "decode"
    print(json.dumps(run_profile(selected_mode, selected_profile), indent=2))
