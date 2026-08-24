"""Bounded metrics storage (PRD FR11).

All samples live in fixed-capacity rolling windows, so memory use never grows
with process lifetime. Pure standard library: unit-testable without Torch.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from typing import Any

_NS_PER_MS = 1_000_000.0
_DEFAULT_WINDOW_SECONDS = 60.0
_MAX_TRACKED_TOKENS = 200_000  # hard cap; pruned to the rate window continuously


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile on an unsorted copy. Empty input -> 0."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


class _Window:
    """Fixed-capacity, time-pruned sample window."""

    __slots__ = ("samples", "capacity", "window_ns")

    def __init__(self, capacity: int, window_seconds: float) -> None:
        self.samples: deque[tuple[int, float]] = deque(maxlen=capacity)
        self.window_ns = int(window_seconds * 1_000_000_000)

    def add(self, now_ns: int, value: float) -> None:
        self.samples.append((now_ns, value))

    def prune(self, now_ns: int) -> list[float]:
        cutoff = now_ns - self.window_ns
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()
        return [value for _, value in self.samples]


class Metrics:
    """Engine-wide metrics recorder.

    Latency percentiles keep up to ``latency_capacity`` recent samples.
    Throughput/scheduler aggregates are computed over a sliding 60s window.
    KV-cache and GPU figures are latest-observation gauges set by the engine.
    """

    def __init__(
        self,
        latency_capacity: int = 4096,
        window_seconds: float = _DEFAULT_WINDOW_SECONDS,
        clock: Callable[[], int] | None = None,
    ) -> None:
        self._clock = clock or (lambda: time.monotonic_ns())
        self._window_ns = int(window_seconds * 1_000_000_000)
        capacity = latency_capacity
        self._ttft = _Window(capacity, window_seconds)
        self._itl = _Window(capacity, window_seconds)
        self._e2e = _Window(capacity, window_seconds)
        self._batch_sizes = _Window(16_384, window_seconds)
        self._token_times: deque[int] = deque(maxlen=_MAX_TRACKED_TOKENS)
        self.completed_total = 0
        self.failed_total = 0
        self.cancelled_total = 0
        self.rejected_total = 0
        self.timed_out_total = 0
        self.iterations_total = 0
        self.input_tokens_total = 0
        self.output_tokens_total = 0
        self._kv: dict[str, Any] = {}
        self._gpu: dict[str, int] = {}

    def now(self) -> int:
        return self._clock()

    # -- recorders ---------------------------------------------------------
    def record_ttft_ms(self, milliseconds: float, now_ns: int | None = None) -> None:
        self._ttft.add(now_ns if now_ns is not None else self.now(), milliseconds)

    def record_itl_ms(self, milliseconds: float, now_ns: int | None = None) -> None:
        self._itl.add(now_ns if now_ns is not None else self.now(), milliseconds)

    def record_e2e_ms(self, milliseconds: float, now_ns: int | None = None) -> None:
        self._e2e.add(now_ns if now_ns is not None else self.now(), milliseconds)

    def record_iteration(self, batch_size: int, now_ns: int | None = None) -> None:
        self.iterations_total += 1
        self._batch_sizes.add(now_ns if now_ns is not None else self.now(), float(batch_size))

    def record_output_token(self, count: int = 1, now_ns: int | None = None) -> None:
        self.output_tokens_total += count
        stamp = now_ns if now_ns is not None else self.now()
        for _ in range(count):
            self._token_times.append(stamp)

    def record_input_tokens(self, count: int) -> None:
        self.input_tokens_total += count

    def inc_completed(self) -> None:
        self.completed_total += 1

    def inc_failed(self) -> None:
        self.failed_total += 1

    def inc_cancelled(self) -> None:
        self.cancelled_total += 1

    def inc_rejected(self) -> None:
        self.rejected_total += 1

    def inc_timed_out(self) -> None:
        self.timed_out_total += 1

    # -- gauges ------------------------------------------------------------
    def set_kv_stats(self, snapshot: dict[str, Any]) -> None:
        self._kv = dict(snapshot)

    def set_gpu_bytes(self, allocated: int, reserved: int, peak_allocated: int) -> None:
        self._gpu = {
            "allocated_bytes": allocated,
            "reserved_bytes": reserved,
            "peak_allocated_bytes": peak_allocated,
        }

    # -- reporting ----------------------------------------------------------
    def reset_runtime(self) -> None:
        """Clear latency/rate samples and counters (used between benchmark runs).

        Lifetime iteration/token counters reset too: benchmarks want per-run numbers.
        """
        for window in (self._ttft, self._itl, self._e2e, self._batch_sizes):
            window.samples.clear()
        self._token_times.clear()
        self.completed_total = self.failed_total = 0
        self.cancelled_total = self.rejected_total = self.timed_out_total = 0
        self.iterations_total = 0
        self.input_tokens_total = self.output_tokens_total = 0

    def snapshot(self, now_ns: int | None = None) -> dict[str, Any]:
        stamp = now_ns if now_ns is not None else self.now()
        ttft = self._ttft.prune(stamp)
        itl = self._itl.prune(stamp)
        e2e = self._e2e.prune(stamp)
        batches = self._batch_sizes.prune(stamp)
        cutoff = stamp - self._window_ns
        while self._token_times and self._token_times[0] < cutoff:
            self._token_times.popleft()
        tokens_60s = len(self._token_times)
        return {
            "requests": {
                "completed_total": self.completed_total,
                "failed_total": self.failed_total,
                "cancelled_total": self.cancelled_total,
                "rejected_total": self.rejected_total,
                "timed_out_total": self.timed_out_total,
            },
            "latency_ms": {
                "ttft_p50": round(percentile(ttft, 50), 3),
                "ttft_p95": round(percentile(ttft, 95), 3),
                "itl_p50": round(percentile(itl, 50), 3),
                "itl_p95": round(percentile(itl, 95), 3),
                "e2e_p50": round(percentile(e2e, 50), 3),
                "e2e_p95": round(percentile(e2e, 95), 3),
            },
            "tokens": {
                "input_total": self.input_tokens_total,
                "output_total": self.output_tokens_total,
                "output_per_second_60s": round(tokens_60s / 60.0, 2),
            },
            "scheduler": {
                "iterations_total": self.iterations_total,
                "mean_batch_size_60s": round(sum(batches) / len(batches), 3) if batches else 0.0,
                "max_batch_size_60s": int(max(batches)) if batches else 0,
            },
            "kv_cache": dict(self._kv),
            "gpu": dict(self._gpu),
        }
