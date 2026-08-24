"""Pinned engine configuration.

Every value originates in ``engine_config.json`` at a pinned revision.
API clients may never choose mode, model revision, GPU, or cache sizing
(PRD §14: configuration is bound server-side).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

Mode = Literal["naive", "contiguous", "batched", "paged", "triton", "ragged"]

ALL_MODES: tuple[str, ...] = ("naive", "contiguous", "batched", "paged", "triton", "ragged")
SINGLE_SEQUENCE_MODES: tuple[str, ...] = ("naive", "contiguous")

CONFIG_FILENAME = "engine_config.json"
_FALLBACK_DIRS = (Path("/root"), Path("/"))


@dataclass(frozen=True)
class EngineConfig:
    """Immutable engine configuration (PRD §9 required interface)."""

    model_id: str
    model_revision: str
    mode: Mode
    dtype: Literal["float16"]
    max_model_len: int
    max_output_tokens: int
    eos_token_id: int | None
    max_active_sequences: int
    max_queue_size: int
    max_batched_tokens: int
    prefill_chunk_size: int
    queue_timeout_seconds: float
    stream_queue_capacity: int
    slow_consumer_timeout_seconds: float
    block_size: int
    kv_cache_bytes: int
    allow_reference_fallback: bool = False


def find_config_file(start: Path | None = None) -> Path:
    candidates: list[Path] = []
    if start is not None:
        candidates.append(start / CONFIG_FILENAME)
    here = Path(__file__).resolve()
    candidates.extend(parent / CONFIG_FILENAME for parent in [*here.parents])
    candidates.extend(directory / CONFIG_FILENAME for directory in _FALLBACK_DIRS)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{CONFIG_FILENAME} not found; run from the repository root or pass an explicit path"
    )


def load_pinned(path: Path | None = None) -> dict[str, Any]:
    with open(path or find_config_file(), encoding="utf-8") as handle:
        return json.load(handle)


def build_config(
    mode: Mode,
    pinned: dict[str, Any] | None = None,
    **overrides: Any,
) -> EngineConfig:
    """Assemble an EngineConfig from pinned values plus explicit overrides.

    Overrides exist for tests and benchmarks only; deployed services use the
    pinned defaults verbatim so results stay reproducible.
    """
    pinned = pinned or load_pinned()
    model = pinned.get("ragged_model", pinned["model"]) if mode == "ragged" else pinned["model"]
    scheduler = pinned["scheduler"]
    kv = pinned["kv_cache"]
    values = dict(
        model_id=model["id"],
        model_revision=model["revision"],
        mode=mode,
        dtype="float16",
        max_model_len=model["max_model_len"],
        max_output_tokens=model["max_output_tokens"],
        eos_token_id=model.get("eos_token_id"),
        max_active_sequences=scheduler["max_active_sequences"],
        max_queue_size=scheduler["max_queue_size"],
        max_batched_tokens=scheduler["max_batched_tokens"],
        prefill_chunk_size=scheduler.get("prefill_chunk_size", scheduler["max_batched_tokens"]),
        queue_timeout_seconds=scheduler["queue_timeout_seconds"],
        stream_queue_capacity=scheduler["stream_queue_capacity"],
        slow_consumer_timeout_seconds=scheduler["slow_consumer_timeout_seconds"],
        block_size=kv["block_size"],
        kv_cache_bytes=kv["bytes"],
    )
    unknown = set(overrides) - set(values) - {"allow_reference_fallback"}
    if unknown:
        raise ValueError(f"unknown config overrides: {sorted(unknown)}")
    values.update(overrides)
    config = EngineConfig(**values)
    validate_config(config)
    return config


def validate_config(config: EngineConfig) -> None:
    if config.mode not in ALL_MODES:
        raise ValueError(f"mode must be one of {ALL_MODES}, got {config.mode!r}")
    if config.dtype != "float16":
        raise ValueError("v1 supports float16 only (PRD FR1)")
    if config.max_output_tokens < 1:
        raise ValueError("max_output_tokens must be positive")
    if config.max_batched_tokens < 1 or config.prefill_chunk_size < 1:
        raise ValueError("batch token limits must be positive")
    if not 1 <= config.max_model_len <= 2048 * 8:
        raise ValueError("max_model_len out of range")
    if config.block_size != 16:
        # The Triton kernel is compiled against block size 16 (PRD FR7).
        raise ValueError("block_size must be 16 for v1")


def effective_active_limit(config: EngineConfig) -> int:
    """Naive/contiguous modes serve one sequence at a time by design.

    Continuous batching is introduced in ``batched`` mode (PRD G3), so the
    throughput comparison between contiguous and batched is meaningful.
    """
    if config.mode in SINGLE_SEQUENCE_MODES:
        return 1
    return config.max_active_sequences


def with_overrides(config: EngineConfig, **changes: Any) -> EngineConfig:
    return replace(config, **changes)
