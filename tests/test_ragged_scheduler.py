"""Pure-Python contracts for packed scheduling, chunking, and recompute preemption."""

import asyncio
import unittest

from cloud_engine.config import build_config
from cloud_engine.metrics import Metrics
from cloud_engine.scheduler import GenerationConfig, RequestState, Scheduler

PINNED = {
    "model": {
        "id": "test/model",
        "revision": "test",
        "dtype": "float16",
        "max_model_len": 128,
        "max_output_tokens": 16,
        "eos_token_id": None,
    },
    "scheduler": {
        "max_active_sequences": 4,
        "max_queue_size": 8,
        "max_batched_tokens": 16,
        "prefill_chunk_size": 4,
        "queue_timeout_seconds": 60,
        "stream_queue_capacity": 8,
        "slow_consumer_timeout_seconds": 10,
    },
    "kv_cache": {"block_size": 16, "bytes": 1024},
}


class FakePackedRunner:
    def __init__(self, token_capacity: int = 10_000) -> None:
        self.token_capacity = token_capacity
        self.resident: dict[str, int] = {}
        self.traces: list[tuple[tuple[str, ...], tuple[int, ...]]] = []

    def admit(self, request) -> None:
        if request.request_id in self.resident:
            raise RuntimeError("duplicate admission")
        self.resident[request.request_id] = 0

    def release(self, request) -> None:
        self.resident.pop(request.request_id, None)

    def allocated_tokens(self, request) -> int:
        return self.resident.get(request.request_id, 0)

    def execute_batch(self, plan):
        required = dict(self.resident)
        for item in plan.items:
            required[item.request.request_id] = max(
                required[item.request.request_id], item.end_pos
            )
        if sum(required.values()) > self.token_capacity:
            error = RuntimeError("fake KV capacity exhausted")
            error.is_capacity_error = True
            raise error

        self.resident = required
        self.traces.append(
            (plan.request_ids, tuple(item.query_length for item in plan.items))
        )
        outputs = {}
        for item in plan.items:
            item.request.tokens_fed = item.end_pos
            if item.sample:
                outputs[item.request.request_id] = 100 + item.request.generated_count
        return outputs


def make_scheduler(runner: FakePackedRunner, **overrides) -> Scheduler:
    scheduler_config = dict(PINNED["scheduler"])
    scheduler_config.update(overrides)
    pinned = {**PINNED, "scheduler": scheduler_config}
    return Scheduler(
        build_config("ragged", pinned=pinned),
        runner,
        Metrics(),
        idle_sleep_seconds=0.001,
    )


async def consume(request) -> list[int]:
    tokens = []
    while True:
        event = await request.output_queue.get()
        if event.finished:
            return tokens
        if event.token_id is not None:
            tokens.append(event.token_id)


class TestRaggedScheduler(unittest.IsolatedAsyncioTestCase):
    async def test_multiple_requests_share_one_forward(self) -> None:
        runner = FakePackedRunner()
        scheduler = make_scheduler(runner)
        requests = [
            await scheduler.submit(
                f"p{i}", [i + 1] * 3, GenerationConfig(max_output_tokens=2)
            )
            for i in range(3)
        ]
        await scheduler.start()
        try:
            results = await asyncio.gather(*(consume(request) for request in requests))
            self.assertEqual(results, [[100, 101]] * 3)
            self.assertTrue(any(len(request_ids) == 3 for request_ids, _ in runner.traces))
            self.assertTrue(all(request.state is RequestState.COMPLETED for request in requests))
        finally:
            await scheduler.stop()

    async def test_prefill_is_chunked_under_shared_token_budget(self) -> None:
        runner = FakePackedRunner()
        scheduler = make_scheduler(runner, max_batched_tokens=5, prefill_chunk_size=4)
        request = await scheduler.submit(
            "long", list(range(11)), GenerationConfig(max_output_tokens=1)
        )
        await scheduler.start()
        try:
            self.assertEqual(await consume(request), [100])
            lengths = [lengths[0] for ids, lengths in runner.traces if ids == (request.request_id,)]
            self.assertEqual(lengths[:3], [4, 4, 3])
        finally:
            await scheduler.stop()

    async def test_pressure_preempts_then_recomputes_without_token_drift(self) -> None:
        runner = FakePackedRunner(token_capacity=10)
        scheduler = make_scheduler(runner, max_batched_tokens=8, prefill_chunk_size=4)
        requests = [
            await scheduler.submit(
                f"p{i}", [i + 1] * 4, GenerationConfig(max_output_tokens=3)
            )
            for i in range(2)
        ]
        await scheduler.start()
        try:
            results = await asyncio.gather(*(consume(request) for request in requests))
            self.assertEqual(results, [[100, 101, 102]] * 2)
            self.assertGreater(sum(request.preemption_count for request in requests), 0)
            self.assertGreater(sum(request.recomputed_tokens for request in requests), 0)
            snapshot = scheduler.metrics.snapshot()
            self.assertGreater(snapshot["requests"]["preempted_total"], 0)
            self.assertGreater(snapshot["tokens"]["recomputed_total"], 0)
            self.assertFalse(runner.resident)
        finally:
            await scheduler.stop()


if __name__ == "__main__":
    unittest.main()
