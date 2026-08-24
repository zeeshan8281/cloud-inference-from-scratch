"""Scheduler tests: FIFO, limits, cancellation, timeout, cleanup — stdlib only.

Uses a deterministic fake runner (no Torch, no GPU) per PRD §13.1/§13.2.
"""

import asyncio
import unittest

from cloud_engine.config import build_config
from cloud_engine.metrics import Metrics
from cloud_engine.scheduler import (
    GenerationConfig,
    RejectedError,
    RejectionReason,
    RequestState,
    Scheduler,
    StreamEvent,
)


class FakeRunner:
    """Deterministic token source with leak tracking."""

    def __init__(self, scripts: dict[str, list[int]] | None = None) -> None:
        self.scripts = scripts or {}
        self.reserved: set[str] = set()
        self.released: list[str] = []
        self.step_calls: list[str] = []
        self.fail_on_step: str | None = None

    def admit(self, request) -> None:
        if request.request_id in self.reserved:
            raise RuntimeError(f"double admit {request.request_id}")
        self.reserved.add(request.request_id)

    def release(self, request) -> None:
        if request.request_id in self.reserved:
            self.reserved.discard(request.request_id)
            self.released.append(request.request_id)

    def step(self, request) -> int:
        self.step_calls.append(request.request_id)
        if self.fail_on_step == request.request_id:
            raise RuntimeError("injected model failure")
        script = self.scripts.get(request.request_id)
        if script:
            index = len([c for c in self.step_calls if c == request.request_id]) - 1
            if index < len(script):
                return script[index]
        # Match real runners: they, not the scheduler, own model-input progress.
        request.tokens_fed += 1
        # default: never emit EOS, so generation runs to max_output_tokens
        return 100 + request.generated_count


def make_scheduler(mode: str = "batched", **overrides) -> Scheduler:
    config = build_config(
        mode,
        pinned={
            "model": {
                "id": "Qwen/Qwen2.5-0.5B",
                "revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
                "dtype": "float16",
                "max_model_len": 2048,
                "max_output_tokens": 256,
                "eos_token_id": 151643,
            },
            "scheduler": {
                "max_active_sequences": overrides.pop("max_active_sequences", 4),
                "max_queue_size": overrides.pop("max_queue_size", 8),
                "max_batched_tokens": overrides.pop("max_batched_tokens", 512),
                "queue_timeout_seconds": overrides.pop("queue_timeout_seconds", 60),
                "stream_queue_capacity": overrides.pop("stream_queue_capacity", 4),
                "slow_consumer_timeout_seconds": overrides.pop(
                    "slow_consumer_timeout_seconds", 10
                ),
            },
            "kv_cache": {"block_size": 16, "bytes": 4294967296},
        },
    )
    return Scheduler(config, FakeRunner(), Metrics(), idle_sleep_seconds=0.001)


async def submit_and_wait(scheduler: Scheduler, prompt: str, tokens: int, **gen_kwargs):
    handle_request = await scheduler.submit(
        prompt, [7] * tokens, GenerationConfig(max_output_tokens=tokens + gen_kwargs.pop("out", 3), eos_token_id=151643, **gen_kwargs)
    )
    await consume_stream(handle_request)
    return await handle_request.terminal_future


async def consume_stream(request) -> list[StreamEvent]:
    events = []
    while True:
        event = await request.output_queue.get()
        events.append(event)
        if event.finished:
            return events


class TestSchedulerBasics(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.scheduler = make_scheduler()
        await self.scheduler.start()

    async def asyncTearDown(self) -> None:
        await self.scheduler.stop()

    async def test_completion_resolves_future_and_releases_runner(self) -> None:
        request = await submit_and_wait(self.scheduler, "hello", 2)
        await asyncio.wait_for(asyncio.shield(request.terminal_future), timeout=5)
        self.assertEqual(request.state, RequestState.COMPLETED)
        self.assertEqual(len(request.generated_token_ids), request.config.max_output_tokens)
        self.assertEqual(request.tokens_fed, request.generated_count)
        self.assertNotIn(request.request_id, self.scheduler.runner.reserved)
        self.assertIn(request.request_id, self.scheduler.runner.released)
        self.assertFalse(request.pending_events)

    async def test_fifo_admission_order(self) -> None:
        order: list[int] = []
        original_admit = self.scheduler.runner.admit

        def spy(request):
            order.append(len(order))
            original_admit(request)

        self.scheduler.runner.admit = spy
        handles = [
            await self.scheduler.submit(
                f"p{i}", [1] * (i + 1), GenerationConfig(max_output_tokens=2, eos_token_id=None)
            )
            for i in range(3)
        ]
        await self.scheduler.drain()
        self.assertEqual(order, [0, 1, 2])
        for handle in handles:
            self.assertEqual(handle.state, RequestState.COMPLETED)

    async def test_active_limit_respected(self) -> None:
        # max_active_sequences=4; submit 6 at once, never more than 4 active
        peak_active = 0

        original_step = self.scheduler.runner.step

        def counting_step(request):
            nonlocal peak_active
            peak_active = max(peak_active, len(self.scheduler.active))
            return original_step(request)

        self.scheduler.runner.step = counting_step
        for i in range(6):
            await self.scheduler.submit(
                f"p{i}", [1], GenerationConfig(max_output_tokens=2, eos_token_id=151643)
            )
        await self.scheduler.drain()
        self.assertLessEqual(peak_active, 4)
        self.assertEqual(peak_active >= 1, True)

    async def test_token_budget_defers_large_prefill(self) -> None:
        await self.scheduler.stop()
        self.scheduler = make_scheduler(max_batched_tokens=10)
        admitted_order: list[str] = []
        original_admit = self.scheduler.runner.admit

        def spy(request):
            admitted_order.append(request.request_id)
            original_admit(request)

        self.scheduler.runner.admit = spy
        await self.scheduler.start()

        # iteration 1: two small prompts fill the budget, the 9-token prompt waits
        first = await self.scheduler.submit(
            "a", [1] * 4, GenerationConfig(max_output_tokens=3, eos_token_id=None)
        )
        second = await self.scheduler.submit(
            "b", [1] * 4, GenerationConfig(max_output_tokens=3, eos_token_id=None)
        )
        big = await self.scheduler.submit(
            "c", [1] * 9, GenerationConfig(max_output_tokens=2, eos_token_id=None)
        )
        await self.scheduler.drain()
        self.assertEqual(first.state, RequestState.COMPLETED)
        self.assertEqual(second.state, RequestState.COMPLETED)
        self.assertEqual(big.state, RequestState.COMPLETED)
        # big was admitted only after both smalls (head-of-line budget deferral)
        self.assertEqual(admitted_order[:2], [first.request_id, second.request_id])
        self.assertEqual(admitted_order[2], big.request_id)

    async def test_context_overflow_rejected_at_submit(self) -> None:
        with self.assertRaises(RejectedError) as caught:
            await self.scheduler.submit(
                "too long",
                [1] * 3000,
                GenerationConfig(max_output_tokens=256, eos_token_id=151643),
            )
        self.assertEqual(caught.exception.reason, RejectionReason.CONTEXT_OVERFLOW)

    async def test_queue_full_rejected(self) -> None:
        scheduler = make_scheduler(max_queue_size=1, queue_timeout_seconds=30)
        # Do not start the loop; nothing drains the waiting queue.
        await scheduler.submit("a", [1], GenerationConfig(max_output_tokens=1, eos_token_id=None))
        with self.assertRaises(RejectedError) as caught:
            await scheduler.submit("b", [1], GenerationConfig(max_output_tokens=1, eos_token_id=None))
        self.assertEqual(caught.exception.reason, RejectionReason.QUEUE_FULL)


class TestTerminalPaths(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_is_idempotent_and_releases(self) -> None:
        scheduler = make_scheduler(queue_timeout_seconds=30)
        await scheduler.start()
        try:
            request = await scheduler.submit(
                "cancel me", [1], GenerationConfig(max_output_tokens=64, eos_token_id=151643)
            )
            await asyncio.sleep(0.02)
            scheduler.cancel(request)
            scheduler.cancel(request)
            scheduler.cancel(request)
            self.assertTrue(request.is_terminal)
            self.assertEqual(request.state, RequestState.CANCELLED)
            await asyncio.sleep(0.05)
            self.assertNotIn(request.request_id, scheduler.runner.reserved)
            snapshot = scheduler.metrics.snapshot()
            self.assertEqual(snapshot["requests"]["cancelled_total"], 1)
        finally:
            await scheduler.stop()

    async def test_waiting_timeout_transitions_to_timed_out(self) -> None:
        scheduler = make_scheduler(
            max_active_sequences=1, queue_timeout_seconds=0.05, max_batched_tokens=4096
        )
        # occupy the single active slot by never letting step finish quickly:
        slow_runner = scheduler.runner

        class BlockingRunner(FakeRunner):
            def step(self, request):
                import time

                time.sleep(0.15)
                return super().step(request)

        scheduler.runner = BlockingRunner()
        await scheduler.start()
        try:
            first = await scheduler.submit(
                "occupy", [1] * 200, GenerationConfig(max_output_tokens=32, eos_token_id=None)
            )
            second = await scheduler.submit(
                "victim", [1] * 200, GenerationConfig(max_output_tokens=32, eos_token_id=None)
            )
            first_consumer = asyncio.create_task(consume_stream(first))
            await asyncio.wait_for(second.terminal_future, timeout=5)
            self.assertEqual(second.state, RequestState.TIMED_OUT)
            await asyncio.wait_for(first.terminal_future, timeout=10)
            await first_consumer
            self.assertEqual(first.state, RequestState.COMPLETED)
            del slow_runner
        finally:
            await scheduler.stop()

    async def test_model_failure_marks_failed_and_releases(self) -> None:
        scheduler = make_scheduler()
        await scheduler.start()
        try:
            scheduler.runner.fail_on_step = "req-00000001"
            request = await scheduler.submit(
                "boom", [1], GenerationConfig(max_output_tokens=4, eos_token_id=151643)
            )
            await asyncio.wait_for(request.terminal_future, timeout=5)
            self.assertEqual(request.state, RequestState.FAILED)
            self.assertNotIn(request.request_id, scheduler.runner.reserved)
            self.assertEqual(scheduler.metrics.snapshot()["requests"]["failed_total"], 1)
        finally:
            await scheduler.stop()

    async def test_capacity_rejection_counts_as_rejected(self) -> None:
        class FullRunner(FakeRunner):
            def admit(self, request):
                exc = RuntimeError("no blocks")
                exc.is_capacity_error = True
                raise exc

        scheduler = make_scheduler()
        scheduler.runner = FullRunner()
        await scheduler.start()
        try:
            request = await scheduler.submit(
                "no room", [1], GenerationConfig(max_output_tokens=2, eos_token_id=151643)
            )
            await asyncio.wait_for(request.terminal_future, timeout=5)
            self.assertEqual(request.state, RequestState.FAILED)
            self.assertTrue(request.rejected)
            self.assertEqual(scheduler.metrics.snapshot()["requests"]["rejected_total"], 1)
        finally:
            await scheduler.stop()


class TestStreamingBackpressure(unittest.IsolatedAsyncioTestCase):
    async def test_slow_consumer_is_cancelled_after_stall_timeout(self) -> None:
        scheduler = make_scheduler(stream_queue_capacity=1, slow_consumer_timeout_seconds=0.1)
        await scheduler.start()
        try:
            request = await scheduler.submit(
                "stall", [1] * 50, GenerationConfig(max_output_tokens=48, eos_token_id=None)
            )
            # consume exactly one event then stop draining
            first = await asyncio.wait_for(request.output_queue.get(), timeout=2)
            self.assertIsInstance(first, StreamEvent)
            await asyncio.sleep(0.5)
            self.assertTrue(request.is_terminal)
            self.assertEqual(request.state, RequestState.CANCELLED)
        finally:
            await scheduler.stop()

    async def test_fast_consumer_sees_every_token_in_order(self) -> None:
        scripts = {"req-00000001": list(range(100, 132))}
        scheduler = make_scheduler(stream_queue_capacity=2)
        scheduler.runner = FakeRunner(scripts)
        await scheduler.start()
        try:
            request = await scheduler.submit(
                "order", [1], GenerationConfig(max_output_tokens=32, eos_token_id=151643)
            )
            received: list[int] = []
            while True:
                event = await asyncio.wait_for(request.output_queue.get(), timeout=5)
                if event.finished:
                    break
                received.append(event.token_id)
            self.assertEqual(received, scripts["req-00000001"][: len(received)])
            self.assertGreaterEqual(len(received), 31)
        finally:
            await scheduler.stop()

    async def test_generated_count_never_exceeds_max_output(self) -> None:
        scheduler = make_scheduler()
        await scheduler.start()
        try:
            for out in (1, 3, 17):
                request = await scheduler.submit(
                    "cap", [1], GenerationConfig(max_output_tokens=out, eos_token_id=None)
                )
                await consume_stream(request)
                await asyncio.wait_for(request.terminal_future, timeout=10)
                self.assertLessEqual(request.generated_count, out)
        finally:
            await scheduler.stop()


class TestGenerationConfigGuard(unittest.TestCase):
    def test_nonzero_temperature_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GenerationConfig(max_output_tokens=4, temperature=0.7)


if __name__ == "__main__":
    unittest.main()
