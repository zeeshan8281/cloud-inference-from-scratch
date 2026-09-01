"""GPU-free tests for the sentinel pilot's pure host-side logic: deterministic
prompt derivation, GPU-state parsing, warmup convergence, KV-capacity
arithmetic, paired-ratio statistics, the correctness gate, and the provenance
manifest. Engine execution itself (experiments/sentinel_pilot.py's run_child,
_build_custom_engine, _build_vllm_engine, ...) requires torch/vllm and a real
GPU and is exercised by the actual Modal pilot run, not here.
"""

from __future__ import annotations

import asyncio
import math
import unittest
from pathlib import Path
from unittest import mock

import experiments.sentinel_pilot as sentinel_pilot
from experiments.sentinel_pilot import (
    Cell,
    KVCapacityPlan,
    StopPilot,
    _parse_gpu_state_line,
    _run_phase,
    assert_gpu_identity_stable,
    build_pair_workload,
    build_source_manifest,
    check_token_parity,
    order_sensitivity_stats,
    paired_ratio_stats,
    plan_common_kv_capacity,
    prompt_seed,
    sentinel_token_ids,
    warmup_to_convergence,
    workload_hash,
)

REPO_ROOT = Path(__file__).parents[1]


class TestDeterministicPrompts(unittest.TestCase):
    def test_same_arguments_always_produce_the_same_ids(self) -> None:
        seed = prompt_seed("resource_normalized", 3, "in128-out128-c1", 5, "unique")
        first = sentinel_token_ids(seed, 32, vocab_size=1000, excluded_ids=frozenset())
        second = sentinel_token_ids(seed, 32, vocab_size=1000, excluded_ids=frozenset())
        self.assertEqual(first, second)
        self.assertEqual(len(first), 32)

    def test_different_request_index_or_phase_changes_the_seed_and_ids(self) -> None:
        base = prompt_seed("resource_normalized", 1, "cell", 0, "cold")
        other_index = prompt_seed("resource_normalized", 1, "cell", 1, "cold")
        other_phase = prompt_seed("resource_normalized", 1, "cell", 0, "warm")
        self.assertNotEqual(base, other_index)
        self.assertNotEqual(base, other_phase)
        ids_base = sentinel_token_ids(base, 16, 5000, frozenset())
        ids_other = sentinel_token_ids(other_index, 16, 5000, frozenset())
        self.assertNotEqual(ids_base, ids_other)

    def test_excluded_special_token_ids_never_appear(self) -> None:
        seed = prompt_seed("complete_system", 2, "cell", 0, "cold")
        excluded = frozenset(range(50))  # a deliberately wide, easy-to-hit ban list
        ids = sentinel_token_ids(seed, 64, vocab_size=200, excluded_ids=excluded)
        self.assertTrue(all(token_id not in excluded for token_id in ids))
        self.assertEqual(len(ids), 64)

    def test_high_entropy_not_a_cyclic_repeat_of_a_handful_of_ids(self) -> None:
        seed = prompt_seed("resource_normalized", 1, "in1024-out256-c32", 0, "unique")
        ids = sentinel_token_ids(seed, 1024, vocab_size=150_000, excluded_ids=frozenset())
        self.assertGreater(len(set(ids)), 900)  # cyclic 4-token prompts would give 4


class TestPairWorkloadMaterialization(unittest.TestCase):
    def test_resource_normalized_has_one_unique_phase_and_warmups_are_disjoint_from_it(self) -> None:
        cells = (Cell(128, 128, 1),)
        payload = build_pair_workload(
            "resource_normalized", 1, cells, vocab_size=100_000, excluded_ids=frozenset()
        )
        cell_out = payload["cells"]["in128-out128-c1"]
        self.assertIn("unique", cell_out)
        self.assertNotIn("cold", cell_out)
        warmup_ids = {tuple(r["input_token_ids"]) for batch in cell_out["warmup"] for r in batch}
        unique_ids = {tuple(r["input_token_ids"]) for r in cell_out["unique"]}
        self.assertEqual(warmup_ids & unique_ids, set())

    def test_complete_system_has_disjoint_cold_and_warm_and_warmup_phases(self) -> None:
        cells = (Cell(128, 128, 1),)
        payload = build_pair_workload(
            "complete_system", 1, cells, vocab_size=100_000, excluded_ids=frozenset()
        )
        cell_out = payload["cells"]["in128-out128-c1"]
        cold_ids = {tuple(r["input_token_ids"]) for r in cell_out["cold"]}
        warm_ids = {tuple(r["input_token_ids"]) for r in cell_out["warm"]}
        warmup_ids = {tuple(r["input_token_ids"]) for batch in cell_out["warmup"] for r in batch}
        self.assertEqual(cold_ids & warm_ids, set())
        self.assertEqual(cold_ids & warmup_ids, set())
        self.assertEqual(warm_ids & warmup_ids, set())

    def test_same_pair_and_mode_reproduces_byte_identical_workload(self) -> None:
        cells = (Cell(128, 128, 1), Cell(512, 128, 8))
        first = build_pair_workload("complete_system", 4, cells, 100_000, frozenset())
        second = build_pair_workload("complete_system", 4, cells, 100_000, frozenset())
        self.assertEqual(first, second)

    def test_workload_hash_is_persisted_and_matches_recomputation(self) -> None:
        cells = (Cell(128, 128, 1),)
        payload = build_pair_workload("resource_normalized", 2, cells, 100_000, frozenset())
        stripped = {key: value for key, value in payload.items() if key != "workload_hash"}
        self.assertEqual(payload["workload_hash"], workload_hash([stripped]))


class TestGPUStateParsing(unittest.TestCase):
    def test_parses_every_field_and_stamps_timestamps(self) -> None:
        line = (
            "GPU-abc123, NVIDIA L4, 00000000:00:03.0, 550.90.07, "
            "23034, 1024, 22010, 25.11, 72.00, 45, P0, 1500, 6250, 0x0000000000000000"
        )
        state = _parse_gpu_state_line(line, cuda_version="12.4")
        self.assertEqual(state["uuid"], "GPU-abc123")
        self.assertEqual(state["name"], "NVIDIA L4")
        self.assertEqual(state["pci.bus_id"], "00000000:00:03.0")
        self.assertEqual(state["cuda_version"], "12.4")
        self.assertIn("utc_timestamp", state)
        self.assertIsInstance(state["monotonic_timestamp"], float)

    def test_wrong_field_count_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            _parse_gpu_state_line("GPU-abc123, NVIDIA L4", cuda_version=None)

    def test_uuid_change_raises_stop_pilot(self) -> None:
        line_a = _parse_gpu_state_line(
            "GPU-a, n, b, d, 1, 1, 1, 1, 1, 1, P0, 1, 1, 0x0", None
        )
        line_b = _parse_gpu_state_line(
            "GPU-b, n, b, d, 1, 1, 1, 1, 1, 1, P0, 1, 1, 0x0", None
        )
        with self.assertRaises(StopPilot) as ctx:
            assert_gpu_identity_stable([line_a, line_b])
        self.assertEqual(ctx.exception.kind, "gpu_uuid_change")

    def test_stable_uuid_across_calls_does_not_raise(self) -> None:
        line_a = _parse_gpu_state_line(
            "GPU-a, n, b, d, 1, 1, 1, 1, 1, 1, P0, 1, 1, 0x0", None
        )
        line_a2 = _parse_gpu_state_line(
            "GPU-a, n, b, d, 2, 2, 2, 2, 2, 2, P0, 2, 2, 0x0", None
        )
        assert_gpu_identity_stable([line_a, line_a2])  # must not raise


class TestWarmupConvergence(unittest.TestCase):
    def test_stops_once_last_three_samples_are_within_tolerance(self) -> None:
        samples = iter([10.0, 100.0, 50.2, 50.0, 49.9, 999.0])

        def measure() -> float:
            return next(samples)

        result = warmup_to_convergence(measure, minimum=3, maximum=10, tolerance=0.03)
        self.assertEqual(result, [10.0, 100.0, 50.2, 50.0, 49.9])

    def test_caps_at_maximum_even_without_convergence(self) -> None:
        values = iter([float(i) * 100 for i in range(1, 11)])  # never converges

        def measure() -> float:
            return next(values)

        result = warmup_to_convergence(measure, minimum=3, maximum=5, tolerance=0.03)
        self.assertEqual(len(result), 5)

    def test_never_stops_before_the_minimum_sample_count(self) -> None:
        values = iter([50.0, 50.0])

        def measure() -> float:
            return next(values)

        result = warmup_to_convergence(measure, minimum=3, maximum=2, tolerance=0.03)
        self.assertEqual(len(result), 2)  # capped by maximum, minimum never reached


class TestKVCapacityPlan(unittest.TestCase):
    def test_common_plan_gives_both_engines_the_identical_token_capacity(self) -> None:
        plan = plan_common_kv_capacity(
            num_layers=36, num_kv_heads=2, head_dim=128, block_size=16, requested_bytes=4 * 2**30
        )
        self.assertIsInstance(plan, KVCapacityPlan)
        self.assertEqual(plan.num_blocks * plan.block_size, plan.token_capacity)
        self.assertEqual(plan.num_blocks * plan.bytes_per_block, plan.resolved_bytes)
        self.assertLessEqual(plan.resolved_bytes, plan.requested_bytes)
        self.assertGreater(plan.num_blocks, 0)

    def test_too_small_a_budget_for_one_block_raises(self) -> None:
        with self.assertRaises(ValueError):
            plan_common_kv_capacity(
                num_layers=36, num_kv_heads=2, head_dim=128, block_size=16, requested_bytes=1
            )


class TestPairedRatioStats(unittest.TestCase):
    def test_ten_pairs_report_the_full_required_statistics(self) -> None:
        pairs = [(50.0 + i, 100.0) for i in range(10)]
        stats = paired_ratio_stats(pairs)
        self.assertEqual(stats["n"], 10)
        self.assertEqual(len(stats["raw_ratios"]), 10)
        self.assertAlmostEqual(
            stats["geometric_mean_ratio"],
            math.exp(statistics_mean_log(pairs)),
            places=9,
        )
        self.assertEqual(stats["degrees_of_freedom"], 9)
        self.assertEqual(stats["t_critical"], 2.262)
        self.assertLess(stats["ci95_low"], stats["geometric_mean_ratio"])
        self.assertGreater(stats["ci95_high"], stats["geometric_mean_ratio"])

    def test_identical_ratios_every_pair_give_a_degenerate_but_defined_interval(self) -> None:
        pairs = [(60.0, 100.0)] * 10
        stats = paired_ratio_stats(pairs)
        self.assertAlmostEqual(stats["geometric_mean_ratio"], 0.6, places=9)
        self.assertAlmostEqual(stats["ci95_low"], 0.6, places=6)
        self.assertAlmostEqual(stats["ci95_high"], 0.6, places=6)

    def test_unsupported_sample_size_leaves_the_interval_unset_not_wrong(self) -> None:
        stats = paired_ratio_stats([(1.0, 1.0), (2.0, 1.0), (3.0, 1.0)])  # n=3, df=2
        self.assertIsNone(stats["ci95_low"])
        self.assertIsNone(stats["ci95_high"])
        self.assertIsNone(stats["t_critical"])

    def test_order_sensitivity_splits_odd_and_even_pairs_by_one_indexed_position(self) -> None:
        pairs = [(float(index), 1.0) for index in range(1, 11)]  # pair i -> ratio i
        split = order_sensitivity_stats(pairs)
        self.assertEqual(split["odd"]["n"], 5)
        self.assertEqual(split["even"]["n"], 5)
        self.assertEqual(sorted(round(r) for r in split["odd"]["raw_ratios"]), [1, 3, 5, 7, 9])
        self.assertEqual(sorted(round(r) for r in split["even"]["raw_ratios"]), [2, 4, 6, 8, 10])


def statistics_mean_log(pairs: list[tuple[float, float]]) -> float:
    values = [math.log(c / v) for c, v in pairs]
    return sum(values) / len(values)


class TestCorrectnessGate(unittest.TestCase):
    def test_identical_outputs_do_not_raise(self) -> None:
        custom = {"cell-a": {"cold": [[1, 2, 3], [4, 5]]}}
        vllm = {"cell-a": {"cold": [[1, 2, 3], [4, 5]]}}
        check_token_parity(custom, vllm)  # must not raise

    def test_a_single_token_mismatch_raises_stop_pilot_with_detail(self) -> None:
        custom = {"cell-a": {"cold": [[1, 2, 3]]}}
        vllm = {"cell-a": {"cold": [[1, 2, 9]]}}
        with self.assertRaises(StopPilot) as ctx:
            check_token_parity(custom, vllm)
        self.assertEqual(ctx.exception.kind, "token_mismatch")
        self.assertEqual(ctx.exception.detail["mismatches"][0]["reason"], "token_mismatch")

    def test_missing_phase_on_one_side_raises(self) -> None:
        custom = {"cell-a": {"cold": [[1]], "warm": [[1]]}}
        vllm = {"cell-a": {"cold": [[1]]}}
        with self.assertRaises(StopPilot):
            check_token_parity(custom, vllm)

    def test_mismatched_request_counts_raises(self) -> None:
        custom = {"cell-a": {"cold": [[1], [2]]}}
        vllm = {"cell-a": {"cold": [[1]]}}
        with self.assertRaises(StopPilot):
            check_token_parity(custom, vllm)


class TestRunPhaseStopsOnTimeout(unittest.TestCase):
    """A timed-out request is a stop-rule trigger for this pilot (unlike the
    original nine-cell matrix, which tolerated ablation timeouts)."""

    def test_any_timed_out_record_raises_stop_pilot(self) -> None:
        with mock.patch.object(
            sentinel_pilot,
            "_run_vllm_phase",
            return_value={
                "records": [
                    {"request_index": 0, "timeout": False},
                    {"request_index": 1, "timeout": True},
                ],
                "wall_seconds": 1.0,
            },
        ):
            with self.assertRaises(StopPilot) as ctx:
                asyncio.run(_run_phase("vllm", engine=None, workload=[], run_id="cell-cold"))
        self.assertEqual(ctx.exception.kind, "timeout")
        self.assertEqual(ctx.exception.detail["timed_out_request_indices"], [1])

    def test_a_scheduler_level_timed_out_error_also_raises_stop_pilot(self) -> None:
        # A request can fail via the scheduler's own admission/queue timeout
        # (terminal state != COMPLETED, error set to "timed_out") without the
        # outer asyncio.wait_for ever firing, so `timeout` stays False.
        with mock.patch.object(
            sentinel_pilot,
            "_run_vllm_phase",
            return_value={
                "records": [
                    {"request_index": 0, "timeout": False, "error": None},
                    {"request_index": 1, "timeout": False, "error": "timed_out"},
                ],
                "wall_seconds": 1.0,
            },
        ):
            with self.assertRaises(StopPilot) as ctx:
                asyncio.run(_run_phase("vllm", engine=None, workload=[], run_id="cell-cold"))
        self.assertEqual(ctx.exception.kind, "timeout")
        self.assertEqual(ctx.exception.detail["timed_out_request_indices"], [1])

    def test_no_timeouts_returns_the_phase_result_unchanged(self) -> None:
        expected = {"records": [{"request_index": 0, "timeout": False}], "wall_seconds": 1.0}
        with mock.patch.object(sentinel_pilot, "_run_vllm_phase", return_value=expected):
            result = asyncio.run(_run_phase("vllm", engine=None, workload=[], run_id="cell-cold"))
        self.assertEqual(result, expected)


class TestProvenanceManifest(unittest.TestCase):
    def test_manifest_has_commit_tree_dirty_flag_and_source_hashes(self) -> None:
        manifest = build_source_manifest(REPO_ROOT, [REPO_ROOT / "experiments/sentinel_pilot.py"])
        self.assertEqual(len(manifest["git_commit"]), 40)
        self.assertTrue(manifest["git_tree"])
        self.assertIn("dirty", manifest)
        self.assertIn("experiments/sentinel_pilot.py", manifest["sources"])
        self.assertEqual(len(manifest["sources"]["experiments/sentinel_pilot.py"]), 64)


if __name__ == "__main__":
    unittest.main()
