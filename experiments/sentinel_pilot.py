"""Direct-engine closed-batch sentinel pilot: shared library.

Implements NEXT_EXPERIMENT_HANDOFF.md's fixed 10-pair, direct-engine
closed-batch comparison between the custom engine and vLLM 0.10.0 on one GPU.

This module is import-safe on a machine with no GPU/torch/vllm installed:
every function that needs those packages defers the import to its own body.
Only the pure host-side logic (prompt derivation, GPU-state parsing, warmup
convergence, KV-capacity arithmetic, paired-ratio statistics, the provenance
manifest) is imported eagerly, and all of it is unit tested without a GPU.

Not an HTTP/production-serving benchmark. See NEXT_EXPERIMENT_HANDOFF.md.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.controlled import Cell, GPUMemorySampler, _record_vllm_step

PROTOCOL_VERSION = "sentinel-pilot-v1"

# The three fixed sentinel cells (input tokens, output tokens, offered concurrency).
SENTINEL_CELLS: tuple[Cell, ...] = (
    Cell(128, 128, 1),
    Cell(512, 128, 8),
    Cell(1024, 256, 32),
)

PAIRS = 10
COMPARISON_MODES = ("resource_normalized", "complete_system")

WARMUP_MINIMUM = 3
WARMUP_MAXIMUM = 10
WARMUP_TOLERANCE = 0.03  # last 3 throughputs must span <=3% of their mean

# Two-sided 95% Student-t critical values, keyed by degrees of freedom. Only
# the exact sample sizes this protocol produces are supported: 10 pairs (df=9)
# and the 5-pair odd/even order split (df=4). A pilot that stops early with a
# different pair count must not silently reuse the wrong critical value.
T_CRITICAL_TWO_SIDED_95 = {9: 2.262, 4: 2.776}

GPU_STATE_FIELDS = (
    "uuid",
    "name",
    "pci.bus_id",
    "driver_version",
    "memory.total",
    "memory.used",
    "memory.free",
    "power.draw",
    "power.limit",
    "temperature.gpu",
    "pstate",
    "clocks.sm",
    "clocks.mem",
    "clocks_throttle_reasons.active",
)


class StopPilot(Exception):
    """Raised to halt the pilot immediately under one of its stop rules.

    `reason` is machine-readable and gets persisted verbatim so a stopped
    pilot's evidence is retained and distinguishable from a completed one.
    """

    def __init__(self, kind: str, detail: dict[str, Any]):
        self.kind = kind
        self.detail = detail
        super().__init__(f"{kind}: {detail}")

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "detail": self.detail}


# --------------------------------------------------------------------------
# Deterministic, high-entropy prompt generation
# --------------------------------------------------------------------------


def prompt_seed(mode: str, pair: int, cell_name: str, request_index: int, phase: str) -> str:
    """The recorded seed string a request's token IDs are derived from."""
    return f"{PROTOCOL_VERSION}|{mode}|pair{pair}|{cell_name}|req{request_index}|{phase}"


# A fixed pool of common English words used to build deterministic prompt
# text. Uniformly-random token IDs across the *entire* vocabulary were tried
# first and rejected: they produce out-of-distribution, near-meaningless
# input that pushes the model's next-token distribution unusually flat, so
# greedy decoding hits far more near-tied logits than realistic text does --
# and a near-tied logit is exactly where two engines' different floating-point
# accumulation order (different attention kernels) can pick different tokens.
# This pool still gives combinatorially far more diversity than the four
# repeating tokens it replaces, without that pathology.
_COMMON_WORDS: tuple[str, ...] = (
    "the", "of", "and", "a", "to", "in", "is", "you", "that", "it", "he", "was",
    "for", "on", "are", "as", "with", "his", "they", "at", "be", "this", "have",
    "from", "or", "one", "had", "by", "word", "but", "not", "what", "all", "were",
    "we", "when", "your", "can", "said", "there", "use", "an", "each", "which",
    "she", "do", "how", "their", "if", "will", "up", "other", "about", "out",
    "many", "then", "them", "these", "so", "some", "her", "would", "make", "like",
    "him", "into", "time", "has", "look", "two", "more", "write", "go", "see",
    "number", "no", "way", "could", "people", "my", "than", "first", "water", "been",
    "call", "who", "its", "now", "find", "long", "down", "day", "did", "get",
    "come", "made", "may", "part", "over", "new", "sound", "take", "only", "little",
    "work", "know", "place", "year", "live", "me", "back", "give", "most", "very",
    "after", "thing", "our", "just", "name", "good", "sentence", "man", "think", "say",
    "great", "where", "help", "through", "much", "before", "line", "right", "too", "mean",
    "old", "any", "same", "tell", "boy", "follow", "came", "want", "show", "also",
    "around", "form", "three", "small", "set", "put", "end", "why", "again", "turn",
    "here", "off", "went", "old", "number", "great", "tell", "men", "say", "small",
)


def _sentinel_text(seed: str, word_count: int) -> str:
    words = []
    for position in range(word_count):
        digest = hashlib.sha256(f"{seed}|word|{position}".encode()).digest()
        index = int.from_bytes(digest[:8], "big") % len(_COMMON_WORDS)
        words.append(_COMMON_WORDS[index])
    return " ".join(words)


def sentinel_token_ids(seed: str, length: int, tokenizer: Any) -> list[int]:
    """Deterministic, natural-language-like token IDs derived from `seed`.

    No wall-clock randomness: word choice is `sha256(seed|word|position)`
    reduced mod the fixed word pool, and tokenization happens once here
    during materialization, never during a timed measurement. Two calls with
    the same arguments always return the same IDs.
    """
    word_count = max(length, 8)
    for attempt in range(20):
        text = _sentinel_text(f"{seed}|attempt{attempt}", word_count)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) >= length:
            ids = ids[:length]
            if any(token_id in tokenizer.all_special_ids for token_id in ids):
                raise RuntimeError(f"special token ID present in materialized prompt for seed {seed!r}")
            return ids
        word_count = int(word_count * 1.5) + 8
    raise RuntimeError(f"could not materialize {length} tokens for seed {seed!r}")


def workload_hash(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def materialize_cell_workload(
    mode: str,
    pair: int,
    cell: Cell,
    phase: str,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """Materialize every request's exact input token IDs for one cell/phase,
    before either engine runs. Both engines consume this identical material."""
    records = []
    for index in range(cell.concurrency):
        seed = prompt_seed(mode, pair, cell.name, index, phase)
        records.append(
            {
                "cell": cell.name,
                "phase": phase,
                "request_index": index,
                "input_tokens": cell.input_tokens,
                "output_tokens": cell.output_tokens,
                "seed": seed,
                "input_token_ids": sentinel_token_ids(seed, cell.input_tokens, tokenizer),
            }
        )
    return records


def materialize_warmup_workload(
    mode: str, pair: int, cell: Cell, iteration: int, tokenizer: Any
) -> list[dict[str, Any]]:
    return materialize_cell_workload(mode, pair, cell, phase=f"warmup-{iteration}", tokenizer=tokenizer)


# --------------------------------------------------------------------------
# GPU identity/state capture
# --------------------------------------------------------------------------


def _parse_gpu_state_line(line: str, cuda_version: str | None) -> dict[str, Any]:
    values = [value.strip() for value in line.split(",")]
    if len(values) != len(GPU_STATE_FIELDS):
        raise RuntimeError(f"unexpected nvidia-smi field count in line: {line!r}")
    state = dict(zip(GPU_STATE_FIELDS, values, strict=True))
    state["cuda_version"] = cuda_version
    state["utc_timestamp"] = datetime.now(timezone.utc).isoformat()
    state["monotonic_timestamp"] = time.monotonic()
    return state


def gpu_state_snapshot(cuda_version: str | None = None) -> dict[str, Any]:
    """One GPU identity/state sample: UUID, clocks, memory, throttle reasons,
    plus a UTC and a monotonic timestamp. Requires nvidia-smi on PATH."""
    output = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={','.join(GPU_STATE_FIELDS)}", "--format=csv,noheader"],
        text=True,
    ).strip()
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("nvidia-smi returned no GPU state")
    return _parse_gpu_state_line(lines[0], cuda_version)


def assert_gpu_identity_stable(states: list[dict[str, Any]]) -> None:
    uuids = {state["uuid"] for state in states}
    if len(uuids) > 1:
        raise StopPilot("gpu_uuid_change", {"uuids": sorted(uuids)})


# --------------------------------------------------------------------------
# Shape-matched warmup to convergence
# --------------------------------------------------------------------------


def warmup_to_convergence(
    measure_throughput: Any,
    minimum: int = WARMUP_MINIMUM,
    maximum: int = WARMUP_MAXIMUM,
    tolerance: float = WARMUP_TOLERANCE,
) -> list[float]:
    """Call `measure_throughput()` (a zero-arg callable returning one
    output-tokens/second sample from a fresh disjoint warmup batch) until the
    last three samples span <=`tolerance` of their mean, capped at `maximum`
    calls. Returns every sample collected, in order, for persistence."""
    samples: list[float] = []
    for _ in range(maximum):
        samples.append(float(measure_throughput()))
        if len(samples) >= minimum:
            window = samples[-3:]
            mean = statistics.mean(window)
            span = (max(window) - min(window)) / mean if mean else 0.0
            if span <= tolerance:
                break
    return samples


# --------------------------------------------------------------------------
# KV capacity matching (resource-normalized mode)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class KVCapacityPlan:
    block_size: int
    bytes_per_block: int
    requested_bytes: int
    num_blocks: int
    resolved_bytes: int
    token_capacity: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "block_size": self.block_size,
            "bytes_per_block": self.bytes_per_block,
            "requested_bytes": self.requested_bytes,
            "num_blocks": self.num_blocks,
            "resolved_bytes": self.resolved_bytes,
            "token_capacity": self.token_capacity,
        }


def plan_common_kv_capacity(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    requested_bytes: int,
    element_size: int = 2,  # float16
) -> KVCapacityPlan:
    """One common total-device-memory headroom rule for both engines: the
    standard paged-KV byte formula (2 for K+V) applied to `requested_bytes`,
    producing an identical block count both engines are configured with."""
    bytes_per_block = 2 * num_kv_heads * head_dim * element_size * block_size * num_layers
    num_blocks = requested_bytes // bytes_per_block
    if num_blocks < 1:
        raise ValueError("requested_bytes too small for even one KV block")
    resolved_bytes = num_blocks * bytes_per_block
    return KVCapacityPlan(
        block_size=block_size,
        bytes_per_block=bytes_per_block,
        requested_bytes=requested_bytes,
        num_blocks=num_blocks,
        resolved_bytes=resolved_bytes,
        token_capacity=num_blocks * block_size,
    )


# --------------------------------------------------------------------------
# Primary analysis: paired log-ratio statistics
# --------------------------------------------------------------------------


def paired_ratio_stats(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    """`pairs` is [(custom_throughput, vllm_throughput), ...] for one cell in
    one comparison mode, one row per pair. Computes the primary cross-engine
    analysis: per-pair log ratios, geometric/arithmetic mean and median
    ratios, and (only for a supported sample size) the two-sided 95% paired
    t-interval on the log ratios, transformed back to ratio scale."""
    n = len(pairs)
    log_ratios = [math.log(custom / vllm) for custom, vllm in pairs]
    ratios = [math.exp(value) for value in log_ratios]
    result: dict[str, Any] = {
        "n": n,
        "raw_ratios": ratios,
        "geometric_mean_ratio": math.exp(statistics.mean(log_ratios)) if n else None,
        "arithmetic_mean_ratio": statistics.mean(ratios) if n else None,
        "median_ratio": statistics.median(ratios) if n else None,
    }
    df = n - 1
    if n >= 2 and df in T_CRITICAL_TWO_SIDED_95:
        mean_log = statistics.mean(log_ratios)
        se = statistics.stdev(log_ratios) / math.sqrt(n)
        margin = T_CRITICAL_TWO_SIDED_95[df] * se
        result["ci95_low"] = math.exp(mean_log - margin)
        result["ci95_high"] = math.exp(mean_log + margin)
        result["t_critical"] = T_CRITICAL_TWO_SIDED_95[df]
        result["degrees_of_freedom"] = df
    else:
        result["ci95_low"] = None
        result["ci95_high"] = None
        result["t_critical"] = None
        result["degrees_of_freedom"] = df
    return result


def order_sensitivity_stats(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    """Odd (1-indexed) pairs ran custom-then-vLLM; even pairs ran the reverse.
    Reported separately as an order-sensitivity check, never pooled to pick
    a preferred subset."""
    odd = [pair for index, pair in enumerate(pairs, start=1) if index % 2 == 1]
    even = [pair for index, pair in enumerate(pairs, start=1) if index % 2 == 0]
    return {"odd": paired_ratio_stats(odd), "even": paired_ratio_stats(even)}


# --------------------------------------------------------------------------
# Provenance manifest
# --------------------------------------------------------------------------


def _git(*args: str, cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def build_source_manifest(
    repo_root: Path, benchmark_paths: list[Path], protocol_seed: str = PROTOCOL_VERSION
) -> dict[str, Any]:
    """Machine-readable manifest distinguishing attribution from execution
    attestation: literal commit/tree IDs, a dirty flag, and a SHA-256 for
    every benchmark source path actually used."""
    commit = _git("rev-parse", "HEAD", cwd=repo_root)
    tree = _git("rev-parse", "HEAD^{tree}", cwd=repo_root)
    dirty = bool(_git("status", "--porcelain", cwd=repo_root))
    sources = {}
    for path in benchmark_paths:
        relative = path.relative_to(repo_root).as_posix()
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_seed": protocol_seed,
        "git_commit": commit,
        "git_tree": tree,
        "dirty": dirty,
        "sources": sources,
    }


def hash_file_tree(root: Path) -> dict[str, str]:
    """SHA-256 of every regular file under `root` (used for local model and
    tokenizer files), keyed by path relative to `root`."""
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            hashes[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def resolved_dependency_versions() -> list[str]:
    return sorted(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()
    )


# --------------------------------------------------------------------------
# Correctness gate: exact token-ID parity between the two engines
# --------------------------------------------------------------------------


def check_token_parity(
    custom_cells: dict[str, dict[str, list[list[int]]]],
    vllm_cells: dict[str, dict[str, list[list[int]]]],
) -> None:
    """`*_cells` maps cell name -> phase -> list of per-request output token
    ID lists. Raises StopPilot on any mismatch; the caller is responsible for
    persisting both engines' raw results before this is called so evidence
    from a stopped pilot is retained."""
    mismatches = []
    for cell_name, phases in custom_cells.items():
        other_phases = vllm_cells.get(cell_name, {})
        for phase_name, custom_outputs in phases.items():
            vllm_outputs = other_phases.get(phase_name)
            if vllm_outputs is None:
                mismatches.append({"cell": cell_name, "phase": phase_name, "reason": "missing_phase"})
                continue
            if len(custom_outputs) != len(vllm_outputs):
                mismatches.append(
                    {"cell": cell_name, "phase": phase_name, "reason": "request_count_mismatch"}
                )
                continue
            for index, (custom_ids, vllm_ids) in enumerate(zip(custom_outputs, vllm_outputs, strict=True)):
                if custom_ids != vllm_ids:
                    mismatches.append(
                        {
                            "cell": cell_name,
                            "phase": phase_name,
                            "request_index": index,
                            "reason": "token_mismatch",
                        }
                    )
    if mismatches:
        raise StopPilot("token_mismatch", {"mismatches": mismatches})


# --------------------------------------------------------------------------
# Pair workload materialization (host-side, no GPU needed)
# --------------------------------------------------------------------------

WARMUP_ITERATIONS_MATERIALIZED = WARMUP_MAXIMUM


def build_pair_workload(
    mode: str, pair: int, cells: tuple[Cell, ...], tokenizer: Any
) -> dict[str, Any]:
    """Materialize every request's input token IDs for one (mode, pair)
    before either engine runs. Both engines read this exact file."""
    cell_workloads: dict[str, Any] = {}
    for cell in cells:
        warmups = [
            materialize_warmup_workload(mode, pair, cell, iteration, tokenizer)
            for iteration in range(WARMUP_ITERATIONS_MATERIALIZED)
        ]
        if mode == "resource_normalized":
            phases = {"unique": materialize_cell_workload(mode, pair, cell, "unique", tokenizer)}
        elif mode == "complete_system":
            phases = {
                "cold": materialize_cell_workload(mode, pair, cell, "cold", tokenizer),
                "warm": materialize_cell_workload(mode, pair, cell, "warm", tokenizer),
            }
        else:
            raise ValueError(f"unknown comparison mode: {mode}")
        cell_workloads[cell.name] = {"warmup": warmups, **phases}
    payload = {"mode": mode, "pair": pair, "cells": cell_workloads}
    payload["workload_hash"] = workload_hash([payload])
    return payload


# --------------------------------------------------------------------------
# Model dims / KV plan resolution (needs the model's config.json; no GPU)
# --------------------------------------------------------------------------


def _hf_model_dims(model_dir: str):
    from cloud_engine.model import ModelDims

    config = json.loads((Path(model_dir) / "config.json").read_text())
    return ModelDims.from_hf_config(config)


def resolve_kv_plan(model_dir: str, block_size: int, requested_bytes: int) -> KVCapacityPlan:
    dims = _hf_model_dims(model_dir)
    return plan_common_kv_capacity(
        num_layers=dims.num_layers,
        num_kv_heads=dims.num_kv_heads,
        head_dim=dims.head_dim,
        block_size=block_size,
        requested_bytes=requested_bytes,
    )


# --------------------------------------------------------------------------
# Engine builders (GPU required; imports deferred)
# --------------------------------------------------------------------------


def _build_custom_engine(model_dir: str, mode: str, kv_plan: KVCapacityPlan | None):
    from cloud_engine.config import build_config
    from cloud_engine.engine import InferenceEngine

    if mode == "resource_normalized":
        assert kv_plan is not None
        overrides = dict(
            cuda_graph_decode=False,
            prefix_cache_max_blocks=0,
            kv_cache_bytes=kv_plan.resolved_bytes,
            max_active_sequences=32,
            max_batched_tokens=2048,
            prefill_chunk_size=256,
        )
    elif mode == "complete_system":
        overrides = {}  # the engine's own pinned, documented policy
    else:
        raise ValueError(f"unknown comparison mode: {mode}")
    config = build_config("ragged", **overrides)
    engine = InferenceEngine(config, model_dir=model_dir)
    return config, engine


def _build_vllm_engine(model_dir: str, model: dict[str, Any], mode: str, kv_plan: KVCapacityPlan | None):
    from vllm import EngineArgs, LLMEngine

    common = dict(
        model=model_dir,
        tokenizer=model_dir,
        dtype="half",
        max_model_len=model["max_model_len"],
        block_size=16,
        disable_log_stats=True,
        trust_remote_code=False,
        seed=0,
    )
    if mode == "resource_normalized":
        assert kv_plan is not None
        args = EngineArgs(
            **common,
            max_num_seqs=32,
            max_num_batched_tokens=2048,
            gpu_memory_utilization=0.85,
            enable_prefix_caching=False,
            enforce_eager=True,
            num_gpu_blocks_override=kv_plan.num_blocks,
        )
    elif mode == "complete_system":
        args = EngineArgs(
            **common,
            max_num_seqs=16,
            max_num_batched_tokens=2048,
            gpu_memory_utilization=0.85,
            enable_prefix_caching=True,
            enforce_eager=False,
        )
    else:
        raise ValueError(f"unknown comparison mode: {mode}")
    return LLMEngine.from_engine_args(args)


# --------------------------------------------------------------------------
# Cell/phase execution (GPU required)
# --------------------------------------------------------------------------


def _ttft_itl_e2e(submitted: float, stamps: list[float], finished: float) -> dict[str, Any]:
    return {
        "ttft_ms": round(((stamps[0] if stamps else finished) - submitted) * 1000, 3),
        "itl_ms": [
            round((right - left) * 1000, 3) for left, right in zip(stamps, stamps[1:], strict=False)
        ],
        "e2e_ms": round(((stamps[-1] if stamps else finished) - submitted) * 1000, 3),
    }


async def _run_custom_phase(engine: Any, workload: list[dict[str, Any]], run_id_prefix: str) -> dict:
    import torch

    from cloud_engine.engine import RequestHandle
    from cloud_engine.scheduler import GenerationConfig, RequestState

    async def drive(record: dict[str, Any]) -> dict[str, Any]:
        submitted = time.perf_counter()
        request = await engine.scheduler.submit(
            f"{run_id_prefix}-{record['request_index']}",
            record["input_token_ids"],
            GenerationConfig(
                max_output_tokens=record["output_tokens"], temperature=0, eos_token_id=None
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
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            error = f"{type(exc).__name__}: {exc}"
        finished = time.perf_counter()
        token_ids = list(request.generated_token_ids)
        return {
            "request_index": record["request_index"],
            "output_token_ids": token_ids,
            **_ttft_itl_e2e(submitted, stamps, finished),
            "error": error,
            "timeout": timeout,
        }

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with GPUMemorySampler() as memory:
        records = await asyncio.gather(*(drive(record) for record in workload))
    torch.cuda.synchronize()
    return {
        "records": list(records),
        "wall_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": max(torch.cuda.max_memory_allocated(), memory.peak_bytes),
        "torch_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch_reserved_bytes": torch.cuda.max_memory_reserved(),
        "device_used_bytes_sampled_peak": memory.peak_bytes,
    }


def _run_vllm_phase(engine: Any, workload: list[dict[str, Any]], run_id_prefix: str) -> dict:
    import torch
    from vllm import SamplingParams

    state: dict[str, dict[str, Any]] = {}
    for record in workload:
        request_id = f"{run_id_prefix}-{record['request_index']}"
        state[request_id] = {
            "index": record["request_index"],
            "submitted": time.perf_counter(),
            "stamps": [],
        }
        params = SamplingParams(
            temperature=0, top_p=1, top_k=-1, max_tokens=record["output_tokens"], ignore_eos=True
        )
        engine.add_request(request_id, {"prompt_token_ids": record["input_token_ids"]}, params)

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with GPUMemorySampler() as memory:
        while engine.has_unfinished_requests():
            if time.perf_counter() - started > 900:
                for request_id in state:
                    engine.abort_request(request_id)
                break
            outputs = engine.step()
            stamp = time.perf_counter()
            _record_vllm_step(state, outputs, stamp)
    torch.cuda.synchronize()
    finished = time.perf_counter()
    records = []
    for item in state.values():
        token_ids = item.get("token_ids", [])
        timeout = not item.get("finished", False)
        records.append(
            {
                "request_index": item["index"],
                "output_token_ids": token_ids,
                **_ttft_itl_e2e(item["submitted"], item["stamps"], finished),
                "error": "timeout" if timeout else None,
                "timeout": timeout,
            }
        )
    return {
        "records": records,
        "wall_seconds": finished - started,
        "peak_gpu_memory_bytes": max(torch.cuda.max_memory_allocated(), memory.peak_bytes),
        "torch_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch_reserved_bytes": torch.cuda.max_memory_reserved(),
        "device_used_bytes_sampled_peak": memory.peak_bytes,
    }


def _phase_throughput(phase_result: dict[str, Any]) -> float:
    output_tokens = sum(len(record["output_token_ids"]) for record in phase_result["records"])
    return output_tokens / phase_result["wall_seconds"] if phase_result["wall_seconds"] else 0.0


def _custom_cache_counters(engine: Any) -> dict[str, Any]:
    try:
        snapshot = engine.snapshot_metrics()
        kv = snapshot.get("kv_cache", {})
        return {
            "prefix_cache_hits": kv.get("prefix_cache_hits"),
            "prefix_cache_misses": kv.get("prefix_cache_misses"),
            "blocks_total": kv.get("blocks_total"),
            "blocks_used": kv.get("blocks_used"),
            "unresolved": False,
        }
    except Exception as exc:  # noqa: BLE001
        return {"unresolved": True, "reason": f"{type(exc).__name__}: {exc}"}


def _custom_graph_counters(engine: Any) -> dict[str, Any]:
    try:
        return {
            "cuda_graph_captures": engine.runner.cuda_graph_captures,
            "cuda_graph_replays": engine.runner.cuda_graph_replays,
            "path": "capture_or_replay" if engine.runner.cuda_graph_replays else "eager_fallback",
            "unresolved": False,
        }
    except Exception as exc:  # noqa: BLE001
        return {"unresolved": True, "reason": f"{type(exc).__name__}: {exc}"}


def _vllm_cache_counters(engine: Any) -> dict[str, Any]:
    try:
        from vllm.utils import Device

        rate = engine.scheduler[0].get_prefix_cache_hit_rate(Device.GPU)
        return {
            "cumulative_prefix_cache_hit_rate": rate,
            "note": "vLLM 0.10.0 exposes a cumulative hit rate, not raw hit/miss/copied/evicted "
            "block counts; per-phase counts are unresolved by design, not inferred.",
            "unresolved": False,
        }
    except Exception as exc:  # noqa: BLE001
        return {"unresolved": True, "reason": f"{type(exc).__name__}: {exc}"}


def _vllm_graph_counters(engine: Any, mode: str) -> dict[str, Any]:
    if mode == "resource_normalized":
        return {"path": "eager_fallback", "unresolved": False}
    try:
        executor = engine.model_executor
        runner = executor.driver_worker.model_runner
        graph_runners = runner.graph_runners
        captured_shapes = sum(len(virtual_engine) for virtual_engine in graph_runners)
        return {
            "captured_shapes": captured_shapes,
            "path": "capture_or_replay" if captured_shapes else "eager_fallback",
            "unresolved": False,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "unresolved": True,
            "reason": f"{type(exc).__name__}: {exc}",
            "note": "vLLM does not publicly expose per-request CUDA graph replay counts; "
            "graph path attribution is marked unresolved rather than inferred.",
        }


def _eligible_prefix_tokens(input_tokens: int, block_size: int) -> int:
    return (input_tokens // block_size) * block_size


def _summarize_phase(phase_result: dict[str, Any]) -> dict[str, Any]:
    records = phase_result["records"]
    successful = [record for record in records if record["error"] is None]
    output_tokens = sum(len(record["output_token_ids"]) for record in successful)
    itls = [value for record in successful for value in record["itl_ms"]]

    def median(key: str) -> float | None:
        values = [record[key] for record in successful]
        return round(statistics.median(values), 3) if values else None

    wall = phase_result["wall_seconds"]
    return {
        "ttft_ms": median("ttft_ms"),
        "itl_ms": round(statistics.median(itls), 3) if itls else None,
        "total_request_latency_ms": median("e2e_ms"),
        "output_tokens_per_second": round(output_tokens / wall, 3) if wall else 0.0,
        "requests_per_second": round(len(successful) / wall, 3) if wall else 0.0,
        "failures": sum(record["error"] is not None for record in records),
        "timeouts": sum(
            record.get("timeout", False) or record.get("error") == "timed_out" for record in records
        ),
        "wall_seconds": round(wall, 3),
        "peak_gpu_memory_bytes": phase_result["peak_gpu_memory_bytes"],
        "torch_allocated_bytes": phase_result["torch_allocated_bytes"],
        "torch_reserved_bytes": phase_result["torch_reserved_bytes"],
        "device_used_bytes_sampled_peak": phase_result["device_used_bytes_sampled_peak"],
    }


async def _run_phase(implementation: str, engine: Any, workload: list[dict[str, Any]], run_id: str) -> dict:
    if implementation == "custom":
        phase_result = await _run_custom_phase(engine, workload, run_id)
    elif implementation == "vllm":
        phase_result = _run_vllm_phase(engine, workload, run_id)
    else:
        raise ValueError(f"unknown implementation: {implementation}")
    timed_out = [
        record
        for record in phase_result["records"]
        if record.get("timeout") or record.get("error") == "timed_out"
    ]
    if timed_out:
        raise StopPilot(
            "timeout",
            {
                "run_id": run_id,
                "timed_out_request_indices": [record["request_index"] for record in timed_out],
            },
        )
    return phase_result


def _cache_counters(implementation: str, engine: Any) -> dict[str, Any]:
    return _custom_cache_counters(engine) if implementation == "custom" else _vllm_cache_counters(engine)


def _graph_counters(implementation: str, engine: Any, mode: str) -> dict[str, Any]:
    if implementation == "custom":
        return _custom_graph_counters(engine)
    return _vllm_graph_counters(engine, mode)


async def run_child(
    implementation: str,
    mode: str,
    model_dir: str,
    model: dict[str, Any],
    workload: dict[str, Any],
    kv_plan: KVCapacityPlan | None,
) -> dict[str, Any]:
    """Run one fresh child process's full cell/phase plan for `mode` and
    return everything needed for both the correctness gate and the report:
    per-phase records, summaries, cache/graph counters, and environment."""
    from experiments.controlled import environment as environment_metadata

    engine_config: dict[str, Any] | None = None
    if implementation == "custom":
        config, engine = _build_custom_engine(model_dir, mode, kv_plan)
        await engine.start()
        tokenizer = engine.tokenizer
        engine_config = {
            "cuda_graph_decode": config.cuda_graph_decode,
            "prefix_cache_max_blocks": config.prefix_cache_max_blocks,
            "kv_cache_bytes": config.kv_cache_bytes,
            "max_active_sequences": config.max_active_sequences,
            "max_batched_tokens": config.max_batched_tokens,
            "block_size": config.block_size,
        }
    elif implementation == "vllm":
        from transformers import AutoTokenizer

        engine = _build_vllm_engine(model_dir, model, mode, kv_plan)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        try:
            resolved_num_gpu_blocks = engine.cache_config.num_gpu_blocks
        except Exception as exc:  # noqa: BLE001 - vLLM engine internals vary by version
            resolved_num_gpu_blocks = None
            if mode == "resource_normalized":
                raise StopPilot(
                    "kv_capacity_unresolved",
                    {"reason": f"{type(exc).__name__}: {exc}"},
                ) from exc
        if mode == "resource_normalized" and resolved_num_gpu_blocks is None:
            # V1's multiprocess engine can leave cache_config unpopulated in
            # this process without raising; unable to verify is treated the
            # same as a blocked KV-capacity match, per protocol.
            raise StopPilot(
                "kv_capacity_unresolved",
                {"reason": "engine.cache_config.num_gpu_blocks is None"},
            )
        engine_config = {
            "enforce_eager": mode == "resource_normalized",
            "enable_prefix_caching": mode == "complete_system",
            "block_size": 16,
            "num_gpu_blocks_override": kv_plan.num_blocks if kv_plan is not None else None,
            "resolved_num_gpu_blocks": resolved_num_gpu_blocks,
        }
        if mode == "resource_normalized" and resolved_num_gpu_blocks != kv_plan.num_blocks:
            raise StopPilot(
                "kv_capacity_not_applied",
                {
                    "requested_num_gpu_blocks": kv_plan.num_blocks,
                    "resolved_num_gpu_blocks": resolved_num_gpu_blocks,
                },
            )
    else:
        raise ValueError(f"unknown implementation: {implementation}")

    try:
        metadata = environment_metadata(model, tokenizer, "custom-server" if implementation == "custom" else "vllm")
        cells_out: dict[str, Any] = {}
        for cell_name, cell_workload in workload["cells"].items():
            # Shape-matched warmup to convergence: run each pre-materialized
            # disjoint warmup iteration and score it, stopping once the last
            # three throughputs span <=WARMUP_TOLERANCE of their mean (capped
            # at len(cell_workload["warmup"]) == WARMUP_MAXIMUM iterations).
            warmup_records: list[dict[str, Any]] = []
            samples: list[float] = []
            for iteration_workload in cell_workload["warmup"]:
                phase_result = await _run_phase(
                    implementation, engine, iteration_workload, f"{cell_name}-warmup"
                )
                warmup_records.append(phase_result["records"])
                samples.append(_phase_throughput(phase_result))
                if len(samples) >= WARMUP_MINIMUM:
                    window = samples[-3:]
                    mean = statistics.mean(window)
                    span = (max(window) - min(window)) / mean if mean else 0.0
                    if span <= WARMUP_TOLERANCE:
                        break
            warmup_samples = samples

            phases_out: dict[str, Any] = {}
            if mode == "resource_normalized":
                before = _cache_counters(implementation, engine)
                phase_result = await _run_phase(
                    implementation, engine, cell_workload["unique"], f"{cell_name}-unique"
                )
                after = _cache_counters(implementation, engine)
                phases_out["unique"] = {
                    "records": phase_result["records"],
                    "summary": _summarize_phase(phase_result),
                    "cache_counters_before": before,
                    "cache_counters_after": after,
                    "graph_counters": _graph_counters(implementation, engine, mode),
                    "eligible_prefix_tokens": _eligible_prefix_tokens(
                        cell_workload["unique"][0]["input_tokens"], 16
                    ),
                }
            else:
                cache_before_cold = _cache_counters(implementation, engine)
                cold_result = await _run_phase(
                    implementation, engine, cell_workload["cold"], f"{cell_name}-cold"
                )
                cache_after_cold = _cache_counters(implementation, engine)
                phases_out["cold"] = {
                    "records": cold_result["records"],
                    "summary": _summarize_phase(cold_result),
                    "cache_counters_before": cache_before_cold,
                    "cache_counters_after": cache_after_cold,
                    "graph_counters": _graph_counters(implementation, engine, mode),
                    "eligible_prefix_tokens": _eligible_prefix_tokens(
                        cell_workload["cold"][0]["input_tokens"], 16
                    ),
                }

                cache_before_prime = _cache_counters(implementation, engine)
                prime_result = await _run_phase(
                    implementation, engine, cell_workload["warm"], f"{cell_name}-warm-prime"
                )
                cache_after_prime = _cache_counters(implementation, engine)
                measured_result = await _run_phase(
                    implementation, engine, cell_workload["warm"], f"{cell_name}-warm-measured"
                )
                cache_after_measured = _cache_counters(implementation, engine)
                phases_out["warm"] = {
                    "prime_records": prime_result["records"],
                    "records": measured_result["records"],
                    "summary": _summarize_phase(measured_result),
                    "cache_counters_before_prime": cache_before_prime,
                    "cache_counters_after_prime": cache_after_prime,
                    "cache_counters_after_measured": cache_after_measured,
                    "graph_counters": _graph_counters(implementation, engine, mode),
                    "eligible_prefix_tokens": _eligible_prefix_tokens(
                        cell_workload["warm"][0]["input_tokens"], 16
                    ),
                }

            cells_out[cell_name] = {
                "warmup_throughputs": warmup_samples,
                "warmup_records": warmup_records,
                "phases": phases_out,
            }

        return {
            "protocol_version": PROTOCOL_VERSION,
            "mode": mode,
            "environment": metadata,
            "engine_config": engine_config,
            "kv_plan": kv_plan.as_dict() if kv_plan is not None else None,
            "workload_hash": workload["workload_hash"],
            "cells": cells_out,
        }
    finally:
        if implementation == "custom":
            await engine.close()
        else:
            shutdown = getattr(engine, "shutdown", None)
            if shutdown is not None:
                shutdown()


def main() -> None:
    if len(sys.argv) != 7:
        raise SystemExit(
            "usage: sentinel_pilot.py <custom|vllm> <resource_normalized|complete_system> "
            "<pair> <model-dir> <workload.json> <output.json>"
        )
    implementation, mode, pair_str, model_dir, workload_path, output_path = sys.argv[1:7]
    del pair_str  # pairing/ordering is tracked by the parent, not needed here
    workload = json.loads(Path(workload_path).read_text())
    pinned = json.loads(Path("/root/engine_config.json").read_text())
    model = pinned["ragged_model"]
    kv_plan = None
    if mode == "resource_normalized":
        kv_plan = resolve_kv_plan(model_dir, block_size=16, requested_bytes=pinned["kv_cache"]["bytes"])
    try:
        result = asyncio.run(run_child(implementation, mode, model_dir, model, workload, kv_plan))
    except StopPilot as exc:
        # An orderly, in-process stop rule (e.g. a timed-out request or vLLM
        # not honoring the requested KV capacity) is not a process crash: the
        # parent distinguishes it from a genuine crash by this "stop" key.
        result = {"stop": exc.as_dict()}
    Path(output_path).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
