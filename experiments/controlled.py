"""Controlled one-GPU comparison shared by the custom engine and vLLM."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MATRIX = (
    (128, 128, 1),
    (128, 128, 8),
    (128, 128, 32),
    (512, 128, 1),
    (512, 128, 8),
    (512, 128, 32),
    (1024, 256, 1),
    (1024, 256, 8),
    (1024, 256, 32),
)
CORRECTNESS_CASES = (
    ("short", 16, 16, 1),
    ("long", 1024, 32, 1),
    ("batched", 128, 32, 4),
    ("maximum", 3840, 256, 1),
)
VARIANTS = {
    "complete": {},
    # Torch ragged attention cannot replay the Triton-oriented CUDA graph.
    # Compare this with no_cuda_graph to isolate only the attention backend.
    "no_triton": {"use_triton_attention": False, "cuda_graph_decode": False},
    "no_continuous_batching": {"max_active_sequences": 1},
    "no_prefix_reuse": {"prefix_cache_max_blocks": 0},
    "no_cuda_graph": {"cuda_graph_decode": False},
}


@dataclass(frozen=True)
class Cell:
    input_tokens: int
    output_tokens: int
    concurrency: int

    @property
    def name(self) -> str:
        return f"in{self.input_tokens}-out{self.output_tokens}-c{self.concurrency}"


def _seed_token_ids(tokenizer: Any) -> list[int]:
    ids: list[int] = []
    for text in (" inference", " memory", " scheduler", " token"):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if encoded:
            ids.append(int(encoded[-1]))
    if not ids:
        raise RuntimeError("tokenizer produced no workload seed tokens")
    return ids


def prompt_token_ids(length: int, request_index: int, seed_ids: list[int]) -> list[int]:
    return [seed_ids[(request_index + offset) % len(seed_ids)] for offset in range(length)]


def workload_records(tokenizer: Any) -> list[dict[str, Any]]:
    seeds = _seed_token_ids(tokenizer)
    records = []
    for values in MATRIX:
        cell = Cell(*values)
        for index in range(cell.concurrency):
            ids = prompt_token_ids(cell.input_tokens, index, seeds)
            records.append(
                {
                    "cell": cell.name,
                    "request_index": index,
                    "input_tokens": cell.input_tokens,
                    "output_tokens": cell.output_tokens,
                    "temperature": 0,
                    "ignore_eos": True,
                    "input_token_ids": ids,
                }
            )
    return records


def workload_hash(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _driver_version() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).splitlines()[0]
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}"


def _gpu_used_memory_bytes() -> int:
    try:
        value = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        ).splitlines()[0]
        return int(value.strip()) * 2**20
    except Exception:
        return 0


class GPUMemorySampler:
    """Sample all processes on the isolated benchmark GPU."""

    def __init__(self) -> None:
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, _gpu_used_memory_bytes())
            self._stop.wait(0.1)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_args) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_bytes = max(self.peak_bytes, _gpu_used_memory_bytes())


def environment(model: dict[str, Any], tokenizer: Any, implementation: str) -> dict[str, Any]:
    import tokenizers
    import torch
    import transformers

    properties = torch.cuda.get_device_properties(0)
    result = {
        "implementation": implementation,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": properties.total_memory,
        "driver": _driver_version(),
        "cuda": torch.version.cuda,
        "pytorch": torch.__version__,
        "python": platform.python_version(),
        "model": model["id"],
        "model_revision": model["revision"],
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizers": tokenizers.__version__,
        "transformers": transformers.__version__,
        "dtype": model["dtype"],
        "repository_commit": os.environ.get("SOURCE_COMMIT", "unknown"),
    }
    if implementation == "vllm":
        import vllm

        result["vllm"] = vllm.__version__
    return result


def _summarize(cell: Cell, records: list[dict[str, Any]], wall_s: float, peak: int) -> dict:
    successful = [record for record in records if record["error"] is None]
    output_tokens = sum(record["output_tokens"] for record in successful)
    itls = [value for record in successful for value in record["itl_ms"]]

    def median(key: str) -> float | None:
        values = [record[key] for record in successful]
        return round(statistics.median(values), 3) if values else None

    return {
        "cell": asdict(cell) | {"name": cell.name},
        "records": records,
        "summary": {
            "ttft_ms": median("ttft_ms"),
            "itl_ms": round(statistics.median(itls), 3) if itls else None,
            "total_request_latency_ms": median("e2e_ms"),
            "output_tokens_per_second": round(output_tokens / wall_s, 3),
            "requests_per_second": round(len(successful) / wall_s, 3),
            "peak_gpu_memory_bytes": peak,
            "failures": sum(record["error"] is not None for record in records),
            "timeouts": sum(
                record.get("timeout", False) or record.get("error") == "timed_out"
                for record in records
            ),
            "wall_seconds": round(wall_s, 3),
        },
    }


async def _custom_requests(engine: Any, cell: Cell, seeds: list[int], run_id: str) -> dict:
    import torch

    from cloud_engine.engine import RequestHandle
    from cloud_engine.scheduler import GenerationConfig, RequestState

    async def drive(index: int) -> dict[str, Any]:
        submitted = time.perf_counter()
        request = await engine.scheduler.submit(
            f"{run_id}-{index}",
            prompt_token_ids(cell.input_tokens, index, seeds),
            GenerationConfig(
                max_output_tokens=cell.output_tokens,
                temperature=0,
                eos_token_id=None,
            ),
        )
        stamps: list[float] = []
        timeout = False
        error = None

        async def consume() -> None:
            async for _event in RequestHandle(request, engine).stream():
                stamps.append(time.perf_counter())

        try:
            await asyncio.wait_for(consume(), timeout=900)
            terminal = await request.terminal_future
            if terminal.state is not RequestState.COMPLETED:
                error = terminal.error_detail or terminal.state.value
        except asyncio.TimeoutError:
            timeout = True
            error = "timeout"
            engine.scheduler.cancel(request)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finished = time.perf_counter()
        token_ids = list(request.generated_token_ids)
        return {
            "request_index": index,
            "input_tokens": cell.input_tokens,
            "output_tokens": len(token_ids),
            "output_token_ids_sha256": hashlib.sha256(
                json.dumps(token_ids, separators=(",", ":")).encode()
            ).hexdigest(),
            "ttft_ms": round(((stamps[0] if stamps else finished) - submitted) * 1000, 3),
            "itl_ms": [
                round((right - left) * 1000, 3)
                for left, right in zip(stamps, stamps[1:], strict=False)
            ],
            "e2e_ms": round((finished - submitted) * 1000, 3),
            "error": error,
            "timeout": timeout,
        }

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with GPUMemorySampler() as memory:
        records = await asyncio.gather(*(drive(index) for index in range(cell.concurrency)))
    torch.cuda.synchronize()
    return _summarize(
        cell,
        list(records),
        time.perf_counter() - started,
        max(torch.cuda.max_memory_allocated(), memory.peak_bytes),
    )


async def _custom_generate(engine: Any, cases: tuple[tuple[str, int, int, int], ...]) -> dict:
    from cloud_engine.engine import RequestHandle
    from cloud_engine.scheduler import GenerationConfig

    seeds = _seed_token_ids(engine.tokenizer)
    output: dict[str, list[list[int]]] = {}
    for name, input_tokens, output_tokens, concurrency in cases:
        requests = [
            await engine.scheduler.submit(
                f"correctness-{name}-{index}",
                prompt_token_ids(input_tokens, index, seeds),
                GenerationConfig(max_output_tokens=output_tokens, temperature=0, eos_token_id=None),
            )
            for index in range(concurrency)
        ]
        handles = [RequestHandle(request, engine) for request in requests]
        results = await asyncio.gather(*(handle.wait() for handle in handles))
        output[name] = [result.token_ids for result in results]
    return output


def run_custom(model_dir: str, model: dict[str, Any], operation: str, variant: str) -> dict:
    from cloud_engine.config import build_config
    from cloud_engine.engine import InferenceEngine

    if variant not in VARIANTS:
        raise ValueError(f"unknown controlled experiment variant: {variant}")
    config = build_config("ragged", **VARIANTS[variant])
    engine = InferenceEngine(config, model_dir=model_dir)

    async def run() -> dict:
        await engine.start()
        try:
            metadata = environment(model, engine.tokenizer, "custom-server")
            workloads = workload_records(engine.tokenizer)
            if operation == "correctness":
                return {
                    "environment": metadata,
                    "workloads": workloads,
                    "workload_hash": workload_hash(workloads),
                    "outputs": await _custom_generate(engine, CORRECTNESS_CASES),
                }
            await _custom_generate(engine, (("warmup", 128, 8, 1),) * 2)
            seeds = _seed_token_ids(engine.tokenizer)
            cells = [
                await _custom_requests(engine, Cell(*values), seeds, f"{variant}-{index}")
                for index, values in enumerate(MATRIX)
            ]
            return {
                "environment": metadata,
                "variant": variant,
                "config": VARIANTS[variant],
                "warmup": {"requests": 2, "input_tokens": 128, "output_tokens": 8},
                "workload_hash": workload_hash(workloads),
                "cells": cells,
            }
        finally:
            await engine.close()

    return asyncio.run(run())


def _vllm_engine(model_dir: str, model: dict[str, Any], prefix_reuse: bool):
    from vllm import EngineArgs, LLMEngine

    args = EngineArgs(
        model=model_dir,
        tokenizer=model_dir,
        dtype="half",
        max_model_len=model["max_model_len"],
        max_num_seqs=16,
        max_num_batched_tokens=2048,
        block_size=16,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=prefix_reuse,
        enforce_eager=False,
        disable_log_stats=True,
        trust_remote_code=False,
        seed=0,
    )
    return LLMEngine.from_engine_args(args)


def _vllm_requests(engine: Any, cell: Cell, seeds: list[int], run_id: str) -> dict:
    import torch
    from vllm import SamplingParams

    params = SamplingParams(
        temperature=0,
        top_p=1,
        top_k=-1,
        max_tokens=cell.output_tokens,
        ignore_eos=True,
    )
    started = time.perf_counter()
    state: dict[str, dict[str, Any]] = {}
    for index in range(cell.concurrency):
        request_id = f"{run_id}-{index}"
        state[request_id] = {"index": index, "submitted": time.perf_counter(), "stamps": []}
        engine.add_request(
            request_id,
            {"prompt_token_ids": prompt_token_ids(cell.input_tokens, index, seeds)},
            params,
        )

    torch.cuda.reset_peak_memory_stats()
    with GPUMemorySampler() as memory:
        while engine.has_unfinished_requests():
            if time.perf_counter() - started > 900:
                for request_id in state:
                    engine.abort_request(request_id)
                break
            outputs = engine.step()
            stamp = time.perf_counter()
            for output in outputs:
                item = state[output.request_id]
                token_ids = list(output.outputs[0].token_ids)
                while len(item["stamps"]) < len(token_ids):
                    item["stamps"].append(stamp)
                item["token_ids"] = token_ids
                item["finished"] = output.finished
    torch.cuda.synchronize()
    finished = time.perf_counter()
    records = []
    for item in state.values():
        stamps = item["stamps"]
        token_ids = item.get("token_ids", [])
        timeout = not item.get("finished", False)
        records.append(
            {
                "request_index": item["index"],
                "input_tokens": cell.input_tokens,
                "output_tokens": len(token_ids),
                "output_token_ids_sha256": hashlib.sha256(
                    json.dumps(token_ids, separators=(",", ":")).encode()
                ).hexdigest(),
                "ttft_ms": round(
                    ((stamps[0] if stamps else finished) - item["submitted"]) * 1000, 3
                ),
                "itl_ms": [
                    round((right - left) * 1000, 3)
                    for left, right in zip(stamps, stamps[1:], strict=False)
                ],
                "e2e_ms": round(
                    ((stamps[-1] if stamps else finished) - item["submitted"]) * 1000, 3
                ),
                "error": "timeout" if timeout else None,
                "timeout": timeout,
            }
        )
    return _summarize(
        cell,
        records,
        finished - started,
        max(torch.cuda.max_memory_allocated(), memory.peak_bytes),
    )


def _vllm_generate(
    engine: Any, cases: tuple[tuple[str, int, int, int], ...], seeds: list[int]
) -> dict:
    from vllm import SamplingParams

    output: dict[str, list[list[int]]] = {}
    for name, input_tokens, output_tokens, concurrency in cases:
        params = SamplingParams(
            temperature=0,
            top_p=1,
            top_k=-1,
            max_tokens=output_tokens,
            ignore_eos=True,
        )
        request_ids = []
        for index in range(concurrency):
            request_id = f"correctness-{name}-{index}"
            request_ids.append(request_id)
            engine.add_request(
                request_id,
                {"prompt_token_ids": prompt_token_ids(input_tokens, index, seeds)},
                params,
            )
        completed = {}
        while engine.has_unfinished_requests():
            for result in engine.step():
                if result.finished:
                    completed[result.request_id] = list(result.outputs[0].token_ids)
        output[name] = [completed[request_id] for request_id in request_ids]
    return output


def run_vllm(model_dir: str, model: dict[str, Any], operation: str) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    workloads = workload_records(tokenizer)
    engine = _vllm_engine(model_dir, model, prefix_reuse=True)
    try:
        metadata = environment(model, tokenizer, "vllm")
        seeds = _seed_token_ids(tokenizer)
        if operation == "correctness":
            return {
                "environment": metadata,
                "workloads": workloads,
                "workload_hash": workload_hash(workloads),
                "outputs": _vllm_generate(engine, CORRECTNESS_CASES, seeds),
            }
        _vllm_generate(engine, (("warmup", 128, 8, 1),) * 2, seeds)
        return {
            "environment": metadata,
            "variant": "complete",
            "warmup": {"requests": 2, "input_tokens": 128, "output_tokens": 8},
            "workload_hash": workload_hash(workloads),
            "cells": [
                _vllm_requests(engine, Cell(*values), seeds, f"vllm-{index}")
                for index, values in enumerate(MATRIX)
            ],
        }
    finally:
        shutdown = getattr(engine, "shutdown", None)
        if shutdown is not None:
            shutdown()


def main() -> None:
    if len(sys.argv) != 6:
        raise SystemExit(
            "usage: controlled.py <custom|vllm> <correctness|benchmark> "
            "<variant> <model-dir> <output.json>"
        )
    implementation, operation, variant, model_dir, output = sys.argv[1:]
    pinned = json.loads(Path("/root/engine_config.json").read_text())
    model = pinned["ragged_model"]
    if implementation == "custom":
        result = run_custom(model_dir, model, operation, variant)
    elif implementation == "vllm":
        result = run_vllm(model_dir, model, operation)
    else:
        raise SystemExit(f"unknown implementation: {implementation}")
    Path(output).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
