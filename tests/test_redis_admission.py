"""Real Redis admission tests; enabled when REDIS_TEST_URL is set."""

import asyncio
import os
import unittest
import uuid

from cloud_engine.api import ApiError, RedisTenantGate, TenantPolicy


@unittest.skipUnless(os.environ.get("REDIS_TEST_URL"), "REDIS_TEST_URL not set")
class TestRedisAdmissionIntegration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        import redis.asyncio as redis

        self.clients = [
            redis.from_url(os.environ["REDIS_TEST_URL"], decode_responses=True)
            for _ in range(8)
        ]
        self.namespace = f"cie:test:{uuid.uuid4().hex}"
        policy = {"tenant": TenantPolicy("t" * 32, 2, 4096)}
        self.gates = [
            RedisTenantGate(client, policy, namespace=self.namespace)
            for client in self.clients
        ]

    async def asyncTearDown(self) -> None:
        keys = await self.clients[0].keys(f"{self.namespace}:*")
        if keys:
            await self.clients[0].delete(*keys)
        await asyncio.gather(*(gate.close() for gate in self.gates))

    async def test_concurrency_is_atomic_across_gate_instances(self) -> None:
        async def attempt(gate):
            try:
                return await gate.acquire("tenant", 64)
            except ApiError as exc:
                self.assertEqual(exc.code, "tenant_concurrency_limit")
                return None

        leases = await asyncio.gather(*(attempt(gate) for gate in self.gates))
        accepted = [lease for lease in leases if lease is not None]
        self.assertEqual(len(accepted), 2)
        snapshot = await self.gates[-1].snapshot()
        self.assertEqual(snapshot["tenant"]["active"], 2)
        self.assertEqual(snapshot["tenant"]["reserved_tokens_60s"], 128)
        await asyncio.gather(*(lease.release() for lease in accepted))

    async def test_rollback_is_visible_to_another_instance(self) -> None:
        lease = await self.gates[0].acquire("tenant", 4096)
        await lease.release(rollback_tokens=True)
        replacement = await self.gates[1].acquire("tenant", 4096)
        snapshot = await self.gates[2].snapshot()
        self.assertEqual(snapshot["tenant"]["reserved_tokens_60s"], 4096)
        await replacement.release()


if __name__ == "__main__":
    unittest.main()
