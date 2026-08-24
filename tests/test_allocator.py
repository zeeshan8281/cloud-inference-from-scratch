"""Block allocator unit tests — pure Python, no Torch required (PRD §13.1)."""

import unittest

from cloud_engine.cache import BlockAllocator, CacheCapacityFull, DoubleFreeError


class TestBlockAllocator(unittest.TestCase):
    def setUp(self) -> None:
        self.allocator = BlockAllocator(num_blocks=8, block_size=16)

    def test_blocks_for_rounds_up(self) -> None:
        self.assertEqual(self.allocator.blocks_for(0), 0)
        self.assertEqual(self.allocator.blocks_for(1), 1)
        self.assertEqual(self.allocator.blocks_for(15), 1)
        self.assertEqual(self.allocator.blocks_for(16), 1)
        self.assertEqual(self.allocator.blocks_for(17), 2)
        self.assertEqual(self.allocator.blocks_for(2048), 128)

    def test_allocate_and_free_roundtrip(self) -> None:
        ids = self.allocator.allocate("req-a", 3)
        self.assertEqual(len(ids), 3)
        self.assertEqual(self.allocator.used_count, 3)
        self.assertEqual(self.allocator.free_count, 5)
        self.allocator.free(ids, "req-a")
        self.assertEqual(self.allocator.used_count, 0)
        self.assertEqual(self.allocator.free_count, 8)

    def test_exhaustion_raises_capacity(self) -> None:
        with self.assertRaises(CacheCapacityFull):
            self.allocator.allocate("req-big", 9)
        # exact capacity still succeeds
        ids = self.allocator.allocate("req-exact", 8)
        self.assertEqual(self.allocator.free_count, 0)
        with self.assertRaises(CacheCapacityFull):
            self.allocator.allocate("req-one-more", 1)
        self.allocator.free(ids, "req-exact")

    def test_freed_block_is_reusable(self) -> None:
        first = self.allocator.allocate("req-a", 2)
        self.allocator.free(first, "req-a")
        second = self.allocator.allocate("req-b", 2)
        for block_id in first:
            self.assertIn(block_id, second)

    def test_double_free_protection(self) -> None:
        ids = self.allocator.allocate("req-a", 1)
        self.allocator.free(ids, "req-a")
        with self.assertRaises(DoubleFreeError):
            self.allocator.free(ids, "req-a")

    def test_foreign_free_protection(self) -> None:
        ids = self.allocator.allocate("req-a", 1)
        with self.assertRaises(DoubleFreeError):
            self.allocator.free(ids, "req-b")
        # ownership intact after failed free
        self.assertEqual(self.allocator.owned_by("req-a"), 1)

    def test_duplicate_ids_in_single_call_rejected(self) -> None:
        ids = self.allocator.allocate("req-a", 2)
        with self.assertRaises(DoubleFreeError):
            self.allocator.free([ids[0], ids[0]], "req-a")
        # cleanup so later tests see a consistent state
        self.allocator.free(ids, "req-a")

    def test_ownership_invariants_hold_through_churn(self) -> None:
        live: dict[str, list[int]] = {}
        rng = __import__("random").Random(7)
        owner_seq = 0
        for _ in range(500):
            operation = rng.random()
            if live and operation < 0.45:
                owner = rng.choice(list(live))
                count = min(rng.randint(1, 3), len(live[owner]))
                released = live[owner][:count]
                del live[owner][:count]
                self.allocator.free(released, owner)
                if not live[owner]:
                    del live[owner]
            else:
                want = rng.randint(1, 4)
                try:
                    got = self.allocator.allocate(f"req-{owner_seq}", want)
                    owner_seq += 1
                    live[f"req-{owner_seq - 1}"] = got
                except CacheCapacityFull:
                    pass
            self.allocator.assert_invariants()
            total_owned = sum(len(v) for v in live.values())
            self.assertEqual(self.allocator.used_count, total_owned)
            self.assertEqual(self.allocator.used_count + self.allocator.free_count, 8)

    def test_no_owner_sharing(self) -> None:
        allocated: list[int] = []
        for i in range(4):
            allocated.extend(self.allocator.allocate(f"req-{i}", 2))
        self.assertEqual(len(allocated), len(set(allocated)), "a block has two owners")


if __name__ == "__main__":
    unittest.main()
