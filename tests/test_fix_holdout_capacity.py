"""Actual-capacity validation logic on synthetic metadata; no GPU allocation."""
from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

from experiments.fix_holdout import observed_kv_capacity, validate_kv_capacity


class Pool:
    shape = (2, 3, 16, 1, 4)

    def element_size(self):
        return 2

    def numel(self):
        return 384


class TestFixHoldoutCapacity(unittest.TestCase):
    def setUp(self):
        self.plan = {"num_blocks": 3, "block_size": 16, "resolved_bytes": 1536}
        self.custom = {"implementation": "custom", "num_blocks": 3, "allocator_blocks": 3,
                       "block_size": 16, "pool_shapes": [[2, 3, 16, 1, 4]] * 2,
                       "pool_element_sizes": [2, 2], "reserved_bytes": 1536}
        self.vllm = {"implementation": "vllm", "worker": {"num_gpu_blocks": 3,
                     "tensor_parallel_size": 1, "block_sizes": [16]}}

    def test_actual_custom_tensors_and_worker_rpc_are_observed(self):
        cache = SimpleNamespace(key_pool=Pool(), value_pool=Pool(), num_blocks=3, block_size=16,
                                allocator=SimpleNamespace(total_blocks=3))
        engine = SimpleNamespace(runner=SimpleNamespace(cache=cache))
        self.assertEqual(observed_kv_capacity(engine, "custom"), self.custom)
        calls = []
        engine = SimpleNamespace(collective_rpc=lambda method: calls.append(method) or [self.vllm["worker"]])
        self.assertEqual(observed_kv_capacity(engine, "vllm"), self.vllm)
        self.assertEqual(calls, ["kv_capacity"])

    def test_normalized_capacity_requires_actual_match_not_requested_override(self):
        for evidence in (self.custom, self.vllm):
            validate_kv_capacity(evidence, "resource_normalized", self.plan)
            validate_kv_capacity(evidence, "complete_system", None)
            with self.assertRaisesRegex(ValueError, "not applied"):
                validate_kv_capacity(evidence, "resource_normalized", {**self.plan, "num_blocks": 4})
            with self.assertRaisesRegex(ValueError, "not applied"):
                validate_kv_capacity(evidence, "resource_normalized", None)
        with self.assertRaisesRegex(ValueError, "bytes"):
            validate_kv_capacity(self.custom, "resource_normalized", {**self.plan, "resolved_bytes": 2048})

    def test_inconsistent_custom_pool_and_worker_metadata_fail_closed(self):
        for field, value in [("allocator_blocks", 4), ("pool_shapes", [[2, 4, 16, 1, 4]] * 2),
                             ("pool_element_sizes", [4, 4]), ("reserved_bytes", 1024)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_kv_capacity({**self.custom, field: value}, "resource_normalized", self.plan)
        for field, value in [("num_gpu_blocks", None), ("num_gpu_blocks", True),
                             ("tensor_parallel_size", 2), ("block_sizes", [32])]:
            evidence = copy.deepcopy(self.vllm)
            evidence["worker"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_kv_capacity(evidence, "resource_normalized", self.plan)
