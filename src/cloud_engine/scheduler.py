"""Request lifecycle, admission control, and the iteration-level scheduler.

One authoritative state machine (PRD FR8):

    waiting -> prefill -> decoding -> completed
                              |-> cancelled
                              |-> timed_out
                              |-> failed

Terminal transitions are irreversible. Every terminal path resolves the
caller's future, closes the stream, records metrics, and releases cache via
the runner. The scheduler knows nothing about HTTP or Torch.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .config import effective_active_limit


class RequestState(Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODING = "decoding"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    FAILED = "failed"


TERMINAL_STATES = frozenset(
    {
        RequestState.COMPLETED,
        RequestState.CANCELLED,
        RequestState.TIMED_OUT,
        RequestState.FAILED,
    }
)


class RejectionReason(Enum):
    QUEUE_FULL = "queue_full"
    KV_CAPACITY = "capacity_exhausted"
    CONTEXT_OVERFLOW = "context_length_exceeded"


class RejectedError(RuntimeError):
    """Admission refusal surfaced to the API layer as 429/503/400."""

    def __init__(self, reason: RejectionReason, detail: str = "") -> None:
        super().__init__(detail or reason.value)
        self.reason = reason


@dataclass
class GenerationConfig:
    max_output_tokens: int
    temperature: float = 0.0
    eos_token_id: int | None = None

    def __post_init__(self) -> None:
        if self.temperature != 0:
            raise ValueError("v1 supports greedy decoding only (temperature=0)")


@dataclass
class StreamEvent:
    """Token published to a request's bounded queue; STOP terminates the stream."""

    token_id: int | None
    finished: bool = False
    finish_reason: str = ""


STOP = StreamEvent(token_id=None, finished=True)


@dataclass
class Request:
    request_id: str
    prompt: str
    prompt_token_ids: list[int]
    config: GenerationConfig
    arrival_ns: int
    state: RequestState = RequestState.WAITING
    generated_token_ids: list[int] = field(default_factory=list)
    output_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=1))
    terminal_future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())

    # engine bookkeeping (not part of the public contract)
    tokens_fed: int = 0
    pending_events: list = field(default_factory=list)
    stalled_since_ns: int | None = None
    first_token_ns: int | None = None
    last_token_ns: int | None = None
    finish_reason: str = ""
    error_detail: str = ""
    rejected: bool = False

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def generated_count(self) -> int:
        return len(self.generated_token_ids)

    def assert_not_terminal(self) -> None:
        if self.is_terminal:
            raise RuntimeError(f"illegal transition from terminal state {self.state}")


def _new_future() -> asyncio.Future:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    return loop.create_future()


class Scheduler:
    """Iteration loop: reap -> decode active -> admit+prefill -> publish."""

    def __init__(
        self,
        config: Any,
        runner: Any,
        metrics: Any,
        clock: Callable[[], int] = time.monotonic_ns,
        idle_sleep_seconds: float = 0.005,
    ) -> None:
        self.config = config
        self.runner = runner
        self.metrics = metrics
        self.clock = clock
        self.idle_sleep_seconds = idle_sleep_seconds
        self.waiting: deque[Request] = deque()
        self.active: list[Request] = []
        self._task: asyncio.Task | None = None
        self._running = False
        self._counter = 0

    # ------------------------------------------------------------------ api
    async def submit(self, prompt: str, prompt_token_ids: list[int], gen_config: GenerationConfig) -> Request:
        total = len(prompt_token_ids) + gen_config.max_output_tokens
        if len(prompt_token_ids) == 0:
            raise RejectedError(RejectionReason.CONTEXT_OVERFLOW, "empty prompt")
        if total > self.config.max_model_len:
            raise RejectedError(RejectionReason.CONTEXT_OVERFLOW)
        if len(self.waiting) >= self.config.max_queue_size:
            raise RejectedError(RejectionReason.QUEUE_FULL)
        self._counter += 1
        request = Request(
            request_id=f"req-{self._counter:08d}",
            prompt=prompt,
            prompt_token_ids=list(prompt_token_ids),
            config=gen_config,
            arrival_ns=self.clock(),
            output_queue=asyncio.Queue(maxsize=self.config.stream_queue_capacity),
            terminal_future=_new_future(),
        )
        self.waiting.append(request)
        return request

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            task, self._task = self._task, None
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                task.cancel()

    async def drain(self) -> None:
        while any(not r.is_terminal for r in (*self.active, *self.waiting)):
            await asyncio.sleep(self.idle_sleep_seconds)

    def cancel(self, request: Request) -> None:
        """Idempotent cancellation from disconnects or the slow-client watchdog."""
        if request.is_terminal:
            return
        request.assert_not_terminal()
        request.state = RequestState.CANCELLED
        request.finish_reason = "cancelled"
        self._finalize(request)

    # ------------------------------------------------------------- internals
    async def _run(self) -> None:
        while self._running:
            now = self.clock()
            progressed = self._reap(now)
            budget = self.config.max_batched_tokens
            processed = 0
            prefill_tokens = 0

            for request in list(self.active):
                if budget <= 0:
                    break
                if request.pending_events:
                    self._drain_pending(request)
                    progressed = True
                    continue  # consumer behind: deliver backlog before new work
                if request.stalled_since_ns is not None:
                    continue  # queue still full this iteration
                token_id = await self._step(request)
                budget -= 1
                processed += 1
                progressed = True
                self._after_token(request, token_id, self.clock())

            limit = effective_active_limit(self.config)
            while self.waiting and len(self.active) < limit and budget > 0:
                candidate = self.waiting[0]
                if len(candidate.prompt_token_ids) > budget:
                    break  # large prompts wait for a fresh iteration (PRD FR5)
                self.waiting.popleft()
                try:
                    self.runner.admit(candidate)
                except Exception as exc:
                    capacity_signal = bool(getattr(exc, "is_capacity_error", False))
                    reason = (
                        RejectionReason.KV_CAPACITY
                        if capacity_signal
                        else RejectionReason.CONTEXT_OVERFLOW
                    )
                    del reason
                    self._fail(candidate, f"admission refused: {exc}", rejected=capacity_signal)
                    continue
                candidate.state = RequestState.PREFILL
                self.active.append(candidate)
                try:
                    token_id = await self._step(candidate)
                except Exception as exc:
                    self._fail(candidate, f"prefill failed: {exc}")
                    continue
                budget -= len(candidate.prompt_token_ids)
                prefill_tokens += len(candidate.prompt_token_ids)
                processed += 1
                progressed = True
                self._after_token(candidate, token_id, self.clock())

            self.metrics.record_iteration(processed)
            await asyncio.sleep(0 if progressed else self.idle_sleep_seconds)

    async def _step(self, request: Request) -> int:
        return await asyncio.to_thread(self.runner.step, request)

    def _reap(self, now: int) -> bool:
        progressed = False
        timeout_ns = int(self.config.queue_timeout_seconds * 1_000_000_000)
        stall_ns = int(self.config.slow_consumer_timeout_seconds * 1_000_000_000)

        for request in list(self.waiting):
            if not request.is_terminal and now - request.arrival_ns > timeout_ns:
                self.waiting.remove(request)
                request.state = RequestState.TIMED_OUT
                request.finish_reason = "timed_out"
                self._finalize(request)
                progressed = True

        for request in list(self.active):
            if request.stalled_since_ns is not None and now - request.stalled_since_ns > stall_ns:
                self.cancel(request)
                progressed = True
        return progressed

    def _after_token(self, request: Request, token_id: int, now: int) -> None:
        request.assert_not_terminal()
        request.generated_token_ids.append(token_id)
        if request.first_token_ns is None:
            request.first_token_ns = now
            self.metrics.record_ttft_ms((now - request.arrival_ns) / 1e6, now)
        else:
            self.metrics.record_itl_ms(
                ((now - (request.last_token_ns or now)) / 1e6), now
            )
        request.last_token_ns = now
        self.metrics.record_output_token(1, now)

        eos_hit = request.config.eos_token_id is not None and token_id == request.config.eos_token_id
        limit_hit = request.generated_count >= request.config.max_output_tokens
        if eos_hit:
            request.finish_reason = "eos_token_reached"
        elif limit_hit:
            request.finish_reason = "max_output_tokens_reached"

        self._publish(request, StreamEvent(token_id=None if eos_hit else token_id))
        if eos_hit or limit_hit:
            request.state = RequestState.COMPLETED
            self._finalize(request)
        else:
            request.state = RequestState.DECODING

    def _publish(self, request: Request, event: StreamEvent) -> None:
        """Backpressure without loss: overflow waits in ``pending_events``.

        A slow consumer never loses tokens; it only stalls its own request,
        and the watchdog cancels stalls older than the configured timeout so
        GPU memory cannot be retained indefinitely (PRD FR9).
        """
        request.pending_events.append(event)
        self._drain_pending(request)

    def _drain_pending(self, request: Request) -> None:
        """Move queued events into the bounded stream queue; track stalls."""
        while request.pending_events:
            try:
                request.output_queue.put_nowait(request.pending_events[0])
            except asyncio.QueueFull:
                if request.stalled_since_ns is None:
                    request.stalled_since_ns = self.clock()
                return
            request.pending_events.pop(0)
        request.stalled_since_ns = None

    def _fail(self, request: Request, detail: str, rejected: bool = False) -> None:
        if request.is_terminal:
            return
        request.rejected = rejected
        request.state = RequestState.FAILED
        request.error_detail = detail
        request.finish_reason = "failed"
        self._finalize(request)

    def _finalize(self, request: Request) -> None:
        """Single terminal funnel: release cache, close stream in order, resolve."""
        if request in self.active:
            self.active.remove(request)
        try:
            self.runner.release(request)
        except Exception as exc:  # release must never break teardown
            request.error_detail = f"release error: {exc}"
        # STOP is appended after any undelivered token events so the stream
        # ends only once every generated token has been handed over.
        request.pending_events.append(STOP)
        self._drain_pending(request)
        if not request.terminal_future.done():
            request.terminal_future.set_result(request)
        if request.state is RequestState.COMPLETED:
            self.metrics.inc_completed()
        elif request.state is RequestState.CANCELLED:
            self.metrics.inc_cancelled()
        elif request.state is RequestState.TIMED_OUT:
            self.metrics.inc_timed_out()
        elif request.state is RequestState.FAILED:
            if request.rejected:
                self.metrics.inc_rejected()
            else:
                self.metrics.inc_failed()
