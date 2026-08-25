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
from .profiling import nvtx_range


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
    terminal_future: asyncio.Future = field(
        default_factory=lambda: asyncio.get_event_loop().create_future()
    )

    # engine bookkeeping (not part of the public contract)
    tokens_fed: int = 0
    pending_events: list = field(default_factory=list)
    stalled_since_ns: int | None = None
    first_token_ns: int | None = None
    last_token_ns: int | None = None
    token_timestamps_ns: list[int] = field(default_factory=list)
    finish_reason: str = ""
    error_detail: str = ""
    rejected: bool = False
    preemption_count: int = 0
    recomputed_tokens: int = 0
    recompute_until: int = 0

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


@dataclass(frozen=True)
class BatchItem:
    """One sequence's token slice in a shared scheduler iteration."""

    request: Request
    token_ids: tuple[int, ...]
    start_pos: int
    sample: bool
    is_decode: bool

    @property
    def query_length(self) -> int:
        return len(self.token_ids)

    @property
    def end_pos(self) -> int:
        return self.start_pos + self.query_length


@dataclass(frozen=True)
class BatchPlan:
    """Decode-first work selected under one shared token budget."""

    items: tuple[BatchItem, ...]

    @property
    def token_count(self) -> int:
        return sum(item.query_length for item in self.items)

    @property
    def request_ids(self) -> tuple[str, ...]:
        return tuple(item.request.request_id for item in self.items)


@dataclass(frozen=True)
class SchedulingCandidate:
    """Read-only scheduling input exposed to experiments."""

    request_id: str
    phase: str
    remaining_tokens: int
    prompt_tokens: int
    generated_tokens: int
    arrival_ns: int


@dataclass(frozen=True)
class PreemptionCandidate:
    """Read-only capacity-pressure input exposed to experiments."""

    request_id: str
    allocated_tokens: int
    tokens_fed: int
    generated_tokens: int
    arrival_ns: int


def decode_first_priority(candidate: SchedulingCandidate) -> tuple[int]:
    """Production policy: decode before prefill; stable sort preserves FIFO."""
    return (candidate.phase != "decode",)


def largest_sequence_preemption(candidate: PreemptionCandidate) -> tuple[int, int, int]:
    """Production policy: reclaim the most resident/recomputed work first."""
    return (candidate.allocated_tokens, candidate.tokens_fed, candidate.arrival_ns)


class Scheduler:
    """Iteration loop: reap -> decode active -> admit+prefill -> publish."""

    def __init__(
        self,
        config: Any,
        runner: Any,
        metrics: Any,
        clock: Callable[[], int] = time.monotonic_ns,
        idle_sleep_seconds: float = 0.005,
        scheduling_priority: Callable[[SchedulingCandidate], tuple[Any, ...]] = decode_first_priority,
        preemption_priority: Callable[
            [PreemptionCandidate], tuple[Any, ...]
        ] = largest_sequence_preemption,
    ) -> None:
        self.config = config
        self.runner = runner
        self.metrics = metrics
        self.clock = clock
        self.idle_sleep_seconds = idle_sleep_seconds
        self.scheduling_priority = scheduling_priority
        self.preemption_priority = preemption_priority
        self.waiting: deque[Request] = deque()
        self.active: list[Request] = []
        self._task: asyncio.Task | None = None
        self._inflight: Request | None = None
        self._inflight_ids: set[str] = set()
        self._running = False
        self._counter = 0

    # ------------------------------------------------------------------ api
    async def submit(
        self, prompt: str, prompt_token_ids: list[int], gen_config: GenerationConfig
    ) -> Request:
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
        if request is not self._inflight and request.request_id not in self._inflight_ids:
            self._finalize(request)

    # ------------------------------------------------------------- internals
    async def _run(self) -> None:
        if callable(getattr(self.runner, "execute_batch", None)):
            await self._run_packed()
            return
        await self._run_serial()

    async def _run_serial(self) -> None:
        while self._running:
            now = self.clock()
            progressed = self._reap(now)
            budget = self.config.max_batched_tokens
            processed = 0
            prefill_tokens = 0

            for request in list(self.active):
                if request.is_terminal:
                    continue
                if budget <= 0:
                    break
                if request.pending_events:
                    self._drain_pending(request)
                    progressed = True
                    continue  # consumer behind: deliver backlog before new work
                if request.stalled_since_ns is not None:
                    continue  # queue still full this iteration
                try:
                    token_id = await self._step(request)
                except Exception as exc:
                    if request.is_terminal:
                        self._finalize(request)
                    else:
                        self._fail(request, f"decode failed: {exc}")
                    continue
                if request.is_terminal:
                    self._finalize(request)
                    continue
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
                    if candidate.is_terminal:
                        self._finalize(candidate)
                    else:
                        self._fail(candidate, f"prefill failed: {exc}")
                    continue
                if candidate.is_terminal:
                    self._finalize(candidate)
                    continue
                budget -= len(candidate.prompt_token_ids)
                prefill_tokens += len(candidate.prompt_token_ids)
                processed += 1
                progressed = True
                self._after_token(candidate, token_id, self.clock())

            self.metrics.record_iteration(processed)
            await asyncio.sleep(0 if progressed else self.idle_sleep_seconds)

    async def _run_packed(self) -> None:
        """Run decode-first packed iterations with chunked prefill."""
        while self._running:
            now = self.clock()
            progressed = self._reap(now)
            progressed = self._drain_active_streams() or progressed
            progressed = self._admit_waiting() or progressed
            with nvtx_range("scheduler.build_batch_plan"):
                plan = self._build_batch_plan()
            if not plan.items:
                await asyncio.sleep(0 if progressed else self.idle_sleep_seconds)
                continue

            try:
                with nvtx_range("scheduler.execute_batch"):
                    outputs = await self._execute_batch(plan)
            except Exception as exc:
                if bool(getattr(exc, "is_capacity_error", False)):
                    if not self._preempt_for_capacity(plan):
                        for item in plan.items:
                            if not item.request.is_terminal:
                                self._fail(
                                    item.request,
                                    f"KV capacity cannot fit one runnable sequence: {exc}",
                                    rejected=True,
                                )
                    await asyncio.sleep(0)
                    continue
                for item in plan.items:
                    if item.request.is_terminal:
                        self._finalize(item.request)
                    else:
                        self._fail(item.request, f"packed forward failed: {exc}")
                await asyncio.sleep(0)
                continue

            stamp = self.clock()
            self.metrics.record_input_tokens(plan.token_count)
            for item in plan.items:
                request = item.request
                if request.is_terminal:
                    self._finalize(request)
                    continue
                token_id = outputs.get(request.request_id)
                recomputed = max(
                    0,
                    min(item.end_pos, request.recompute_until) - item.start_pos,
                )
                if recomputed:
                    request.recomputed_tokens += recomputed
                    self.metrics.record_recomputed_tokens(recomputed)
                if item.sample:
                    if token_id is None:
                        self._fail(request, "packed runner omitted sampled token")
                        continue
                    self._after_token(request, token_id, stamp)
                else:
                    request.state = RequestState.PREFILL
            self.metrics.record_iteration(len(plan.items))
            progressed = True
            await asyncio.sleep(0 if progressed else self.idle_sleep_seconds)

    def _drain_active_streams(self) -> bool:
        progressed = False
        for request in list(self.active):
            if request.pending_events:
                before = len(request.pending_events)
                self._drain_pending(request)
                progressed = progressed or len(request.pending_events) < before
        return progressed

    def _admit_waiting(self) -> bool:
        progressed = False
        limit = effective_active_limit(self.config)
        while self.waiting and len(self.active) < limit:
            candidate = self.waiting[0]
            # A preempted sequence waits for the current residency set to drain;
            # this prevents capacity thrash while preserving eventual recompute.
            if candidate.preemption_count and self.active:
                break
            self.waiting.popleft()
            try:
                self.runner.admit(candidate)
            except Exception as exc:
                capacity_signal = bool(getattr(exc, "is_capacity_error", False))
                self._fail(candidate, f"admission refused: {exc}", rejected=capacity_signal)
                continue
            candidate.state = RequestState.PREFILL
            self.active.append(candidate)
            progressed = True
        return progressed

    def _build_batch_plan(self) -> BatchPlan:
        budget = self.config.max_batched_tokens
        items: list[BatchItem] = []
        eligible = [
            request
            for request in self.active
            if not request.is_terminal
            and not request.pending_events
            and request.stalled_since_ns is None
        ]

        candidates: list[tuple[SchedulingCandidate, Request]] = []
        for request in eligible:
            known = len(request.prompt_token_ids) + request.generated_count
            remaining = known - request.tokens_fed
            if remaining <= 0:
                continue
            phase = (
                "decode"
                if request.state is RequestState.DECODING and remaining == 1
                else "prefill"
            )
            candidates.append(
                (
                    SchedulingCandidate(
                        request_id=request.request_id,
                        phase=phase,
                        remaining_tokens=remaining,
                        prompt_tokens=len(request.prompt_token_ids),
                        generated_tokens=request.generated_count,
                        arrival_ns=request.arrival_ns,
                    ),
                    request,
                )
            )

        for candidate, request in sorted(
            candidates, key=lambda pair: self.scheduling_priority(pair[0])
        ):
            if budget <= 0:
                break
            known_ids = request.prompt_token_ids + request.generated_token_ids
            remaining = len(known_ids) - request.tokens_fed
            if remaining <= 0:
                continue
            limit = 1 if candidate.phase == "decode" else self.config.prefill_chunk_size
            query_length = min(remaining, limit, budget)
            start = request.tokens_fed
            end = start + query_length
            items.append(
                BatchItem(
                    request=request,
                    token_ids=tuple(known_ids[start:end]),
                    start_pos=start,
                    sample=end == len(known_ids),
                    is_decode=query_length == 1 and start >= len(request.prompt_token_ids),
                )
            )
            budget -= query_length
        return BatchPlan(tuple(items))

    async def _execute_batch(self, plan: BatchPlan) -> dict[str, int]:
        self._inflight_ids = set(plan.request_ids)
        try:
            return await asyncio.to_thread(self.runner.execute_batch, plan)
        finally:
            self._inflight_ids.clear()

    def _preempt_for_capacity(self, plan: BatchPlan) -> bool:
        candidates = [request for request in self.active if not request.is_terminal]
        if len(candidates) <= 1:
            return False
        allocated_tokens = getattr(self.runner, "allocated_tokens", lambda _r: 0)
        victim = max(
            candidates,
            key=lambda request: self.preemption_priority(
                PreemptionCandidate(
                    request_id=request.request_id,
                    allocated_tokens=allocated_tokens(request),
                    tokens_fed=request.tokens_fed,
                    generated_tokens=request.generated_count,
                    arrival_ns=request.arrival_ns,
                )
            ),
        )
        if victim in self.active:
            self.active.remove(victim)
        self.runner.release(victim)
        victim.recompute_until = victim.tokens_fed
        victim.tokens_fed = 0
        victim.preemption_count += 1
        victim.state = RequestState.WAITING
        self.waiting.append(victim)
        self.metrics.inc_preempted()
        return True

    async def _step(self, request: Request) -> int:
        self._inflight = request
        try:
            return await asyncio.to_thread(self.runner.step, request)
        finally:
            self._inflight = None

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
            self.metrics.record_itl_ms(((now - (request.last_token_ns or now)) / 1e6), now)
        request.last_token_ns = now
        request.token_timestamps_ns.append(now)
        self.metrics.record_output_token(1, now)

        eos_hit = (
            request.config.eos_token_id is not None and token_id == request.config.eos_token_id
        )
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
        if request in self.waiting:
            self.waiting.remove(request)
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
            now = self.clock()
            self.metrics.record_e2e_ms((now - request.arrival_ns) / 1e6, now)
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
