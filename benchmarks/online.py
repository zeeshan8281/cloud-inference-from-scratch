"""Fixed arrival-rate benchmark shared by the ragged engine and pinned vLLM."""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import asdict
from typing import Any

from .run import _percentile, build_workload, workload_hash


def _latency_summary(records: list[dict[str, Any]]) -> dict[str, float]:
    successful = [record for record in records if not record.get("error")]
    ttft = [record["ttft_ms"] for record in successful]
    e2e = [record["e2e_ms"] for record in successful]
    itl = [sample for record in successful for sample in record["itl_ms"]]
    return {
        "ttft_p50_ms": round(_percentile(ttft, 50), 3),
        "ttft_p95_ms": round(_percentile(ttft, 95), 3),
        "ttft_p99_ms": round(_percentile(ttft, 99), 3),
        "itl_p50_ms": round(_percentile(itl, 50), 3),
        "itl_p95_ms": round(_percentile(itl, 95), 3),
        "itl_p99_ms": round(_percentile(itl, 99), 3),
        "e2e_p50_ms": round(_percentile(e2e, 50), 3),
        "e2e_p95_ms": round(_percentile(e2e, 95), 3),
        "e2e_p99_ms": round(_percentile(e2e, 99), 3),
    }


def _summarize(
    records: list[dict[str, Any]],
    wall_seconds: float,
    queue_samples: list[int],
    slo_ttft_ms: float,
    slo_itl_ms: float,
) -> dict[str, Any]:
    successful = [record for record in records if not record.get("error")]
    total_output = sum(record["output_tokens"] for record in successful)
    good = [
        record
        for record in successful
        if record["ttft_ms"] <= slo_ttft_ms
        and _percentile(record["itl_ms"], 99) <= slo_itl_ms
    ]
    return {
        "latency": _latency_summary(records),
        "throughput": {
            "output_tokens_per_second": round(total_output / wall_seconds, 3),
            "completed_requests_per_second": round(len(successful) / wall_seconds, 3),
            "slo_goodput_requests_per_second": round(len(good) / wall_seconds, 3),
        },
        "requests": {
            "offered": len(records),
            "completed": len(successful),
            "errors": len(records) - len(successful),
            "slo_good": len(good),
        },
        "queue": {
            "available": bool(queue_samples),
            "mean_depth": round(statistics.mean(queue_samples), 3) if queue_samples else None,
            "max_depth": max(queue_samples) if queue_samples else None,
        },
        "wall_seconds": round(wall_seconds, 3),
        "raw_requests": records,
    }


async def _fixed_arrivals(
    arrival_rate: float,
    duration_seconds: float,
    drive: Any,
    queue_depth: Any,
) -> tuple[list[dict[str, Any]], float, list[int]]:
    count = max(1, round(arrival_rate * duration_seconds))
    interval = 1.0 / arrival_rate
    started = time.perf_counter()
    tasks = []
    queue_samples: list[int] = []
    sampling = True

    async def sample_queue() -> None:
        while sampling:
            queue_samples.append(int(queue_depth()))
            await asyncio.sleep(0.01)

    sampler = asyncio.create_task(sample_queue()) if queue_depth is not None else None
    try:
        for index in range(count):
            target = started + index * interval
            await asyncio.sleep(max(0.0, target - time.perf_counter()))
            tasks.append(asyncio.create_task(drive(index, time.perf_counter())))
        records = list(await asyncio.gather(*tasks))
    finally:
        sampling = False
        if sampler is not None:
            await sampler
    return records, time.perf_counter() - started, queue_samples


async def run_engine_sweep(
    engine: Any,
    tokenizer: Any,
    arrival_rates: list[float],
    duration_seconds: float,
    slo_ttft_ms: float,
    slo_itl_ms: float,
    implementation: str = "ragged-l4",
) -> dict[str, Any]:
    from cloud_engine.scheduler import GenerationConfig

    items = build_workload("online", tokenizer)
    results = []
    for rate in arrival_rates:
        engine.metrics.reset_runtime()
        sweep_started = time.perf_counter()

        async def drive(
            index: int,
            submitted_at: float,
            current_rate: float = rate,
            current_start: float = sweep_started,
        ) -> dict[str, Any]:
            item = items[index % len(items)]
            try:
                handle = await engine.submit(
                    item.prompt,
                    GenerationConfig(max_output_tokens=item.output_tokens, eos_token_id=None),
                )
                result = await handle.wait()
                timestamps = handle.request.token_timestamps_ns
                itl = [
                    (right - left) / 1e6
                    for left, right in zip(timestamps, timestamps[1:], strict=False)
                ]
                return {
                    "index": index,
                    "scheduled_offset_ms": round(index / current_rate * 1000, 3),
                    "submit_offset_ms": round((submitted_at - current_start) * 1000, 3),
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "ttft_ms": result.ttft_ms or 0.0,
                    "itl_ms": [round(value, 3) for value in itl],
                    "e2e_ms": result.e2e_ms,
                    "error": None,
                }
            except Exception as exc:
                return {
                    "index": index,
                    "scheduled_offset_ms": round(index / current_rate * 1000, 3),
                    "submit_offset_ms": round((submitted_at - current_start) * 1000, 3),
                    "input_tokens": item.prompt_tokens,
                    "output_tokens": 0,
                    "ttft_ms": 0.0,
                    "itl_ms": [],
                    "e2e_ms": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                }

        records, wall, queue = await _fixed_arrivals(
            rate,
            duration_seconds,
            drive,
            lambda: len(engine.scheduler.waiting),
        )
        summary = _summarize(records, wall, queue, slo_ttft_ms, slo_itl_ms)
        snapshot = engine.snapshot_metrics()
        summary.update(
            {
                "arrival_rate_requests_per_second": rate,
                "preemptions": snapshot["requests"].get("preempted_total", 0),
                "recomputed_tokens": snapshot["tokens"].get("recomputed_total", 0),
                "scheduler": snapshot["scheduler"],
                "kv_cache": snapshot["kv_cache"],
            }
        )
        results.append(summary)
    return {
        "implementation": implementation,
        "workload_hash": workload_hash(items),
        "workload": [asdict(item) | {"prompt": None} for item in items],
        "arrival_rates": arrival_rates,
        "duration_seconds_per_rate": duration_seconds,
        "slo": {"ttft_ms": slo_ttft_ms, "itl_p99_ms": slo_itl_ms},
        "sweeps": results,
    }


async def run_responses_http_sweep(
    base_url: str,
    api_key: str,
    model_id: str,
    tokenizer: Any,
    arrival_rates: list[float],
    duration_seconds: float,
    slo_ttft_ms: float,
    slo_itl_ms: float,
    engine: Any,
    implementation: str,
) -> dict[str, Any]:
    """Measure the custom engine through its authenticated streaming HTTP API."""
    import httpx

    items = build_workload("online", tokenizer)
    results = []
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=600) as client:
        for rate in arrival_rates:
            engine.metrics.reset_runtime()
            sweep_started = time.perf_counter()

            async def drive(
                index: int,
                submitted_at: float,
                current_rate: float = rate,
                current_start: float = sweep_started,
            ) -> dict[str, Any]:
                item = items[index % len(items)]
                first = None
                token_times: list[float] = []
                usage = {"input_tokens": item.prompt_tokens, "output_tokens": 0}
                try:
                    async with client.stream(
                        "POST",
                        f"{base_url}/v1/responses",
                        headers=headers,
                        json={
                            "model": model_id,
                            "input": item.prompt,
                            "max_output_tokens": item.output_tokens,
                            "temperature": 0,
                            "stream": True,
                        },
                    ) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line.startswith("data: ") or line == "data: [DONE]":
                                continue
                            payload = json.loads(line[6:])
                            if payload.get("type") == "response.output_text.delta":
                                stamp = time.perf_counter()
                                first = first or stamp
                                token_times.append(stamp)
                            elif payload.get("type") == "response.completed":
                                usage = payload["response"]["usage"]
                    finished = time.perf_counter()
                    itl = [
                        (right - left) * 1000
                        for left, right in zip(token_times, token_times[1:], strict=False)
                    ]
                    return {
                        "index": index,
                        "scheduled_offset_ms": round(index / current_rate * 1000, 3),
                        "submit_offset_ms": round((submitted_at - current_start) * 1000, 3),
                        "input_tokens": usage["input_tokens"],
                        "output_tokens": usage["output_tokens"],
                        "ttft_ms": ((first or finished) - submitted_at) * 1000,
                        "itl_ms": [round(value, 3) for value in itl],
                        "e2e_ms": (finished - submitted_at) * 1000,
                        "error": None,
                    }
                except Exception as exc:
                    return {
                        "index": index,
                        "scheduled_offset_ms": round(index / current_rate * 1000, 3),
                        "submit_offset_ms": round((submitted_at - current_start) * 1000, 3),
                        "input_tokens": item.prompt_tokens,
                        "output_tokens": 0,
                        "ttft_ms": 0.0,
                        "itl_ms": [],
                        "e2e_ms": 0.0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }

            records, wall, queue = await _fixed_arrivals(
                rate,
                duration_seconds,
                drive,
                lambda: len(engine.scheduler.waiting),
            )
            summary = _summarize(records, wall, queue, slo_ttft_ms, slo_itl_ms)
            snapshot = engine.snapshot_metrics()
            summary.update(
                {
                    "arrival_rate_requests_per_second": rate,
                    "preemptions": snapshot["requests"].get("preempted_total", 0),
                    "recomputed_tokens": snapshot["tokens"].get("recomputed_total", 0),
                    "scheduler": snapshot["scheduler"],
                    "kv_cache": snapshot["kv_cache"],
                }
            )
            results.append(summary)
    return {
        "implementation": implementation,
        "workload_hash": workload_hash(items),
        "workload": [asdict(item) | {"prompt": None} for item in items],
        "arrival_rates": arrival_rates,
        "duration_seconds_per_rate": duration_seconds,
        "slo": {"ttft_ms": slo_ttft_ms, "itl_p99_ms": slo_itl_ms},
        "sweeps": results,
    }


async def run_vllm_http_sweep(
    base_url: str,
    model_id: str,
    tokenizer: Any,
    arrival_rates: list[float],
    duration_seconds: float,
    slo_ttft_ms: float,
    slo_itl_ms: float,
) -> dict[str, Any]:
    import httpx

    items = build_workload("online", tokenizer)
    results = []
    async with httpx.AsyncClient(timeout=600) as client:
        for rate in arrival_rates:
            sweep_started = time.perf_counter()

            async def drive(
                index: int,
                submitted_at: float,
                current_rate: float = rate,
                current_start: float = sweep_started,
            ) -> dict[str, Any]:
                item = items[index % len(items)]
                first = None
                token_times = []
                output_parts = []
                usage = None
                try:
                    async with client.stream(
                        "POST",
                        f"{base_url}/v1/completions",
                        json={
                            "model": model_id,
                            "prompt": item.prompt,
                            "max_tokens": item.output_tokens,
                            "temperature": 0,
                            "stream": True,
                            "stream_options": {"include_usage": True},
                        },
                    ) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line.startswith("data: ") or line == "data: [DONE]":
                                continue
                            payload = json.loads(line[6:])
                            if payload.get("usage"):
                                usage = payload["usage"]
                            choices = payload.get("choices", [])
                            text = choices[0].get("text", "") if choices else ""
                            if text:
                                stamp = time.perf_counter()
                                first = first or stamp
                                token_times.append(stamp)
                                output_parts.append(text)
                    finished = time.perf_counter()
                    itl = [
                        (right - left) * 1000
                        for left, right in zip(token_times, token_times[1:], strict=False)
                    ]
                    return {
                        "index": index,
                        "scheduled_offset_ms": round(index / current_rate * 1000, 3),
                        "submit_offset_ms": round((submitted_at - current_start) * 1000, 3),
                        "input_tokens": usage["prompt_tokens"] if usage else item.prompt_tokens,
                        "output_tokens": usage["completion_tokens"]
                        if usage
                        else len(tokenizer.encode("".join(output_parts), add_special_tokens=False)),
                        "ttft_ms": ((first or finished) - submitted_at) * 1000,
                        "itl_ms": [round(value, 3) for value in itl],
                        "e2e_ms": (finished - submitted_at) * 1000,
                        "error": None,
                    }
                except Exception as exc:
                    return {
                        "index": index,
                        "scheduled_offset_ms": round(index / current_rate * 1000, 3),
                        "submit_offset_ms": round((submitted_at - current_start) * 1000, 3),
                        "input_tokens": item.prompt_tokens,
                        "output_tokens": 0,
                        "ttft_ms": 0.0,
                        "itl_ms": [],
                        "e2e_ms": 0.0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }

            records, wall, queue = await _fixed_arrivals(
                rate, duration_seconds, drive, None
            )
            summary = _summarize(records, wall, queue, slo_ttft_ms, slo_itl_ms)
            summary["arrival_rate_requests_per_second"] = rate
            results.append(summary)
    return {
        "implementation": "vllm-0.10.0",
        "workload_hash": workload_hash(items),
        "workload": [asdict(item) | {"prompt": None} for item in items],
        "arrival_rates": arrival_rates,
        "duration_seconds_per_rate": duration_seconds,
        "slo": {"ttft_ms": slo_ttft_ms, "itl_p99_ms": slo_itl_ms},
        "sweeps": results,
    }
