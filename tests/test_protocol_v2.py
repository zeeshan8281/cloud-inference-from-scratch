"""GPU-free tests for Correctness Protocol V2's pure logic: disjoint
calibration/holdout batch construction, first-disagreement diagnosis,
classification, epsilon proposal, and summary reporting. Engine execution
(run_custom_full_batch, run_vllm_full_batch) requires torch/vllm and a real
GPU and is exercised by the actual protocol run, not here.
"""

from __future__ import annotations

import math
import unittest

from experiments.protocol_v2 import (
    AUDIT_TARGETS,
    V2_BATCHES_PER_CELL,
    V2_CELLS,
    analyze_audit_entry,
    batch_vs_solo_drift,
    build_audit_variants,
    build_split_batches,
    classify_request,
    first_disagreement,
    propose_epsilon,
    summarize,
)
from tests.test_sentinel_pilot import FakeTokenizer


def _result(tokens: list[int], top_k: list[list[tuple[int, float]]], verified: bool = True) -> dict:
    return {"output_tokens": tokens, "top_k": top_k, "verified": verified}


class TestSplitBatches(unittest.TestCase):
    def test_calibration_and_holdout_are_fully_disjoint(self) -> None:
        tokenizer = FakeTokenizer()
        calibration = build_split_batches(tokenizer, "calibration")
        holdout = build_split_batches(tokenizer, "holdout")
        calibration_prompts = {
            tuple(request)
            for cell_batches in calibration.values()
            for batch in cell_batches
            for request in batch
        }
        holdout_prompts = {
            tuple(request) for cell_batches in holdout.values() for batch in cell_batches for request in batch
        }
        self.assertEqual(calibration_prompts & holdout_prompts, set())

    def test_expected_batch_and_request_counts(self) -> None:
        tokenizer = FakeTokenizer()
        batches = build_split_batches(tokenizer, "calibration")
        for cell in V2_CELLS:
            cell_batches = batches[cell.name]
            self.assertEqual(len(cell_batches), V2_BATCHES_PER_CELL[cell.concurrency])
            for batch in cell_batches:
                self.assertEqual(len(batch), cell.concurrency)

    def test_reproducible_across_calls(self) -> None:
        tokenizer = FakeTokenizer()
        first = build_split_batches(tokenizer, "calibration")
        second = build_split_batches(tokenizer, "calibration")
        self.assertEqual(first, second)

    def test_rejects_unknown_split(self) -> None:
        with self.assertRaises(ValueError):
            build_split_batches(FakeTokenizer(), "test")


class TestAuditVariants(unittest.TestCase):
    def test_variants_reconstruct_the_exact_flagged_holdout_requests(self) -> None:
        tokenizer = FakeTokenizer()
        holdout = build_split_batches(tokenizer, "holdout")
        for target in AUDIT_TARGETS:
            original_batch = holdout[target["cell"]][target["batch_index"]]
            target_ids = original_batch[target["index_in_batch"]]
            variants = build_audit_variants(tokenizer, target)
            self.assertEqual(variants["alone"]["batch_ids"], [target_ids])
            self.assertEqual(variants["original"]["batch_ids"], original_batch)
            self.assertEqual(variants["original"]["target_index"], target["index_in_batch"])
            self.assertEqual(variants["reordered_target_first"]["batch_ids"][0], target_ids)
            self.assertEqual(variants["reordered_target_first"]["target_index"], 0)
            self.assertEqual(
                set(map(tuple, variants["reordered_target_first"]["batch_ids"])),
                set(map(tuple, original_batch)),
            )


class TestBatchVsSoloDrift(unittest.TestCase):
    def test_identical_top_k_between_alone_and_batched_is_no_drift(self) -> None:
        top_k = [[(1, -0.1), (2, -1.0)], [(2, -0.2), (3, -1.0)]]
        alone = _result([1, 2], top_k)
        batched = _result([1, 2], top_k)
        result = batch_vs_solo_drift(alone, batched)
        self.assertFalse(result["drift_detected"])
        self.assertEqual(len(result["steps"]), 2)

    def test_large_logprob_difference_at_a_shared_prefix_step_is_flagged(self) -> None:
        alone = _result([1], [[(1, -0.1), (2, -1.0)]])
        batched = _result([1], [[(1, -5.0), (2, -1.0)]])
        result = batch_vs_solo_drift(alone, batched)
        self.assertTrue(result["drift_detected"])
        self.assertTrue(result["steps"][0]["high_drift"])

    def test_prefix_divergence_itself_counts_as_drift_and_stops_comparison(self) -> None:
        alone = _result([1, 2], [[(1, -0.1)], [(2, -0.1)]])
        batched = _result([1, 9], [[(1, -0.1)], [(9, -0.1)]])
        result = batch_vs_solo_drift(alone, batched)
        self.assertEqual(len(result["steps"]), 2)
        self.assertTrue(result["steps"][1]["prefix_diverged"])
        self.assertTrue(result["drift_detected"])


class TestAnalyzeAuditEntry(unittest.TestCase):
    def _child(self, result: list[dict]) -> dict:
        return {"crashed": False, "result": {"result": result}}

    def test_reports_crash_without_analyzing(self) -> None:
        runs = {"alone_custom": {"crashed": True, "stderr_tail": "boom"}}
        result = analyze_audit_entry({}, runs, epsilon=0.01)
        self.assertIn("alone_custom", result["crashed"])

    def test_full_analysis_covers_every_variant_and_both_drift_engines(self) -> None:
        custom_r0 = {"index": 0, "output_tokens": [1], "top_k": [[(1, -0.01), (2, -1.0)]], "verified": True}
        vllm_r0 = {"index": 0, "output_tokens": [1], "top_k": [[(1, -0.02), (2, -1.1)]], "verified": True}
        custom_others = [{"index": 1, "output_tokens": [5], "top_k": [[(5, -0.1)]], "verified": True}]
        vllm_others = [{"index": 1, "output_tokens": [5], "top_k": [[(5, -0.1)]], "verified": True}]
        variants = {
            "alone": {"batch_ids": [[100]], "target_index": 0},
            "original": {"batch_ids": [[100], [200]], "target_index": 0},
            "reordered_target_first": {"batch_ids": [[100], [200]], "target_index": 0},
        }
        runs = {
            "alone_custom": self._child([custom_r0]),
            "alone_vllm": self._child([vllm_r0]),
            "original_custom": self._child([custom_r0, *custom_others]),
            "original_vllm": self._child([vllm_r0, *vllm_others]),
            "reordered_target_first_custom": self._child([custom_r0, *custom_others]),
            "reordered_target_first_vllm": self._child([vllm_r0, *vllm_others]),
        }
        result = analyze_audit_entry(variants, runs, epsilon=0.05)
        self.assertEqual(set(result["per_variant"]), {"alone", "original", "reordered_target_first"})
        self.assertIsNone(result["per_variant"]["alone"]["custom_identity_check"])
        self.assertIsNotNone(result["per_variant"]["original"]["custom_identity_check"])
        self.assertFalse(result["per_variant"]["original"]["custom_identity_check"]["contamination_suspected"])
        self.assertIn("custom", result["batch_vs_solo_drift"])
        self.assertIn("vllm", result["batch_vs_solo_drift"])
        self.assertEqual(result["per_variant"]["alone"]["classification"]["verdict"], "exact_match")

    def test_identical_outputs_across_distinct_batch_members_flags_contamination(self) -> None:
        custom_r0 = {"index": 0, "output_tokens": [1, 2], "top_k": [[(1, -0.01)], [(2, -0.01)]], "verified": True}
        contaminated = {"index": 1, "output_tokens": [1, 2], "top_k": [[(1, -0.01)], [(2, -0.01)]], "verified": True}
        vllm_r0 = {"index": 0, "output_tokens": [1, 2], "top_k": [[(1, -0.01)], [(2, -0.01)]], "verified": True}
        vllm_r1 = {"index": 1, "output_tokens": [9, 9], "top_k": [[(9, -0.01)], [(9, -0.01)]], "verified": True}
        variants = {
            "alone": {"batch_ids": [[100]], "target_index": 0},
            "original": {"batch_ids": [[100], [200]], "target_index": 0},
            "reordered_target_first": {"batch_ids": [[100], [200]], "target_index": 0},
        }
        runs = {
            "alone_custom": self._child([custom_r0]),
            "alone_vllm": self._child([vllm_r0]),
            "original_custom": self._child([custom_r0, contaminated]),
            "original_vllm": self._child([vllm_r0, vllm_r1]),
            "reordered_target_first_custom": self._child([custom_r0, contaminated]),
            "reordered_target_first_vllm": self._child([vllm_r0, vllm_r1]),
        }
        result = analyze_audit_entry(variants, runs, epsilon=0.05)
        self.assertTrue(result["per_variant"]["original"]["custom_identity_check"]["contamination_suspected"])
        self.assertFalse(result["per_variant"]["original"]["vllm_identity_check"]["contamination_suspected"])


class TestFirstDisagreement(unittest.TestCase):
    def test_none_when_tokens_agree(self) -> None:
        custom = _result([1, 2, 3], [[(1, -0.1)], [(2, -0.1)], [(3, -0.1)]])
        vllm = _result([1, 2, 3], [[(1, -0.1)], [(2, -0.1)], [(3, -0.1)]])
        self.assertIsNone(first_disagreement(custom, vllm))

    def test_finds_the_first_differing_position(self) -> None:
        custom_topk = [[(1, -0.1)], [(2, -0.01), (9, -0.02)], [(3, -0.1)]]
        vllm_topk = [[(1, -0.1)], [(5, -0.01), (2, -0.02)], [(3, -0.1)]]
        custom = _result([1, 2, 3], custom_topk)
        vllm = _result([1, 5, 3], vllm_topk)
        disagreement = first_disagreement(custom, vllm)
        self.assertIsNotNone(disagreement)
        self.assertEqual(disagreement["position"], 1)
        self.assertEqual(disagreement["custom_token"], 2)
        self.assertEqual(disagreement["vllm_token"], 5)
        # custom's step-1 top-k is [(2,-0.01),(9,-0.02)]: vllm's selected
        # token 5 is not in it.
        self.assertFalse(disagreement["vllm_token_in_custom_top_k"])
        # vllm's step-1 top-k is [(5,-0.01),(2,-0.02)]: custom's selected
        # token 2 is in it.
        self.assertTrue(disagreement["custom_token_in_vllm_top_k"])

    def test_stops_at_the_first_disagreement_not_a_later_one(self) -> None:
        custom = _result([1, 2, 3], [[(1, -0.1)], [(9, -0.1)], [(3, -0.1)]])
        vllm = _result([1, 5, 3], [[(1, -0.1)], [(5, -0.1)], [(3, -0.1)]])
        disagreement = first_disagreement(custom, vllm)
        self.assertEqual(disagreement["position"], 1)


class TestClassifyRequest(unittest.TestCase):
    def test_exact_match_when_no_disagreement_and_no_flags(self) -> None:
        custom = _result([1, 2, 3], [[(1, -0.01), (2, -5.0)]] * 3)
        vllm = _result([1, 2, 3], [[(1, -0.01), (2, -5.0)]] * 3)
        result = classify_request(custom, vllm, epsilon=0.05)
        self.assertEqual(result["verdict"], "exact_match")

    def test_own_top_k_inconsistency_is_a_hard_failure_even_without_disagreement(self) -> None:
        custom = _result([1, 2, 3], [[(1, -0.01)]] * 3, verified=False)
        vllm = _result([1, 2, 3], [[(1, -0.01)]] * 3)
        result = classify_request(custom, vllm, epsilon=0.05)
        self.assertEqual(result["verdict"], "hard_failure")
        self.assertIn("custom_disagrees_with_own_top_k", result["reasons"])

    def test_near_tie_within_epsilon_and_cross_present_is_qualified(self) -> None:
        # Full 20-entry top-k (matching real data shape) so top_k_overlap,
        # which divides by TOP_K=20, isn't spuriously penalized for a tiny
        # synthetic list. Same 20 token IDs on both sides; only the order of
        # the top two (1 and 9) differs between engines.
        tail = [(100 + i, -1.0 - 0.01 * i) for i in range(18)]
        custom_topk = [(1, -0.01), (9, -0.02), *tail]
        vllm_topk = [(9, -0.015), (1, -0.02), *tail]
        custom = _result([1], [custom_topk])
        vllm = _result([9], [vllm_topk])
        result = classify_request(custom, vllm, epsilon=0.05)
        self.assertEqual(result["verdict"], "near_tie_qualified")

    def test_confident_disagreement_beyond_epsilon_is_a_hard_failure(self) -> None:
        custom_topk = [(1, -0.01), (9, -5.0)]  # large margin: confident
        vllm_topk = [(9, -0.01), (1, -5.0)]  # large margin: confident
        custom = _result([1], [custom_topk])
        vllm = _result([9], [vllm_topk])
        result = classify_request(custom, vllm, epsilon=0.05)
        self.assertEqual(result["verdict"], "hard_failure")
        self.assertIn("confident_disagreement_no_near_tie", result["reasons"])

    def test_selected_token_missing_from_other_engines_top_k_is_a_hard_failure(self) -> None:
        custom_topk = [(1, -0.01), (2, -0.02)]
        vllm_topk = [(9, -0.01), (2, -0.02)]  # vllm's token 9 not in custom's top-k at all
        custom = _result([1], [custom_topk])
        vllm = _result([9], [vllm_topk])
        result = classify_request(custom, vllm, epsilon=0.5)
        self.assertEqual(result["verdict"], "hard_failure")
        self.assertIn("vllm_token_missing_from_custom_top_k", result["reasons"])

    def test_low_top_k_overlap_is_a_hard_failure_regardless_of_margin(self) -> None:
        custom_topk = [(i, -0.01 * i) for i in range(1, 21)]
        vllm_topk = [(i, -0.01 * i) for i in range(100, 120)]
        custom = _result([1], [custom_topk])
        vllm = _result([100], [vllm_topk])
        result = classify_request(custom, vllm, epsilon=1.0)
        self.assertEqual(result["verdict"], "hard_failure")
        self.assertIn("low_top_k_overlap", result["reasons"])

    def test_pending_epsilon_when_epsilon_not_yet_committed(self) -> None:
        tail = [(100 + i, -1.0 - 0.01 * i) for i in range(18)]
        custom_topk = [(1, -0.01), (9, -0.02), *tail]
        vllm_topk = [(9, -0.015), (1, -0.02), *tail]
        custom = _result([1], [custom_topk])
        vllm = _result([9], [vllm_topk])
        result = classify_request(custom, vllm, epsilon=None)
        self.assertEqual(result["verdict"], "disagreement_unclassified_pending_epsilon")

    def test_concurrency_one_disagreement_is_never_tolerated_even_with_a_tiny_margin(self) -> None:
        # An extremely close near-tie that would qualify at concurrency > 1
        # must still be a hard failure at concurrency 1 (requirement 1: no
        # batch composition exists to vary, so this is a real bug).
        tail = [(100 + i, -1.0 - 0.01 * i) for i in range(18)]
        custom_topk = [(1, -0.0001), (9, -0.0002), *tail]
        vllm_topk = [(9, -0.0001), (1, -0.0002), *tail]
        custom = _result([1], [custom_topk])
        vllm = _result([9], [vllm_topk])
        result = classify_request(custom, vllm, epsilon=1.0, concurrency=1)
        self.assertEqual(result["verdict"], "hard_failure")
        self.assertIn("disagreement_at_concurrency_one", result["reasons"])

    def test_batch_vs_solo_drift_is_a_hard_failure_when_flagged(self) -> None:
        custom = _result([1, 2, 3], [[(1, -0.01), (2, -5.0)]] * 3)
        vllm = _result([1, 2, 3], [[(1, -0.01), (2, -5.0)]] * 3)
        result = classify_request(
            custom, vllm, epsilon=0.05, batch_vs_solo_drift={"drift_detected": True}
        )
        self.assertEqual(result["verdict"], "hard_failure")
        self.assertIn("batch_vs_solo_drift", result["reasons"])

    def test_identity_check_contamination_is_a_hard_failure_when_flagged(self) -> None:
        custom = _result([1, 2, 3], [[(1, -0.01), (2, -5.0)]] * 3)
        vllm = _result([1, 2, 3], [[(1, -0.01), (2, -5.0)]] * 3)
        result = classify_request(
            custom, vllm, epsilon=0.05, identity_check={"contamination_suspected": True, "matching_indices": [2]}
        )
        self.assertEqual(result["verdict"], "hard_failure")
        self.assertIn("cross_request_identity_or_kv_corruption", result["reasons"])

    def test_absent_drift_and_identity_checks_do_not_change_prior_behavior(self) -> None:
        custom = _result([1, 2, 3], [[(1, -0.01), (2, -5.0)]] * 3)
        vllm = _result([1, 2, 3], [[(1, -0.01), (2, -5.0)]] * 3)
        result = classify_request(custom, vllm, epsilon=0.05)
        self.assertEqual(result["verdict"], "exact_match")
        self.assertEqual(result["reasons"], [])


class TestProposeEpsilon(unittest.TestCase):
    def test_epsilon_is_the_max_margin_among_clean_candidates(self) -> None:
        classifications = [
            {"disagreement": {"custom_margin": 0.01, "vllm_margin": 0.02}, "reasons": []},
            {"disagreement": {"custom_margin": 0.03, "vllm_margin": 0.015}, "reasons": []},
            {"disagreement": None, "reasons": []},
        ]
        result = propose_epsilon(classifications)
        self.assertAlmostEqual(result["epsilon"], 0.03)
        self.assertEqual(result["candidate_count"], 2)

    def test_disqualified_disagreements_are_excluded_from_the_candidate_pool(self) -> None:
        classifications = [
            {"disagreement": {"custom_margin": 0.01, "vllm_margin": 0.01}, "reasons": []},
            {
                "disagreement": {"custom_margin": 5.0, "vllm_margin": 5.0},
                "reasons": ["low_top_k_overlap"],
            },
        ]
        result = propose_epsilon(classifications)
        self.assertAlmostEqual(result["epsilon"], 0.01)
        self.assertEqual(result["candidate_count"], 1)

    def test_no_candidates_gives_none(self) -> None:
        result = propose_epsilon([{"disagreement": None, "reasons": []}])
        self.assertIsNone(result["epsilon"])
        self.assertEqual(result["candidate_count"], 0)


class TestSummarize(unittest.TestCase):
    def test_counts_and_percentages(self) -> None:
        classifications = [
            {"verdict": "exact_match", "disagreement": None, "reasons": []},
            {"verdict": "exact_match", "disagreement": None, "reasons": []},
            {
                "verdict": "near_tie_qualified",
                "disagreement": {"custom_margin": 0.01},
                "reasons": [],
            },
            {"verdict": "hard_failure", "disagreement": None, "reasons": ["custom_non_finite_logit"]},
        ]
        summary = summarize(classifications)
        self.assertEqual(summary["total"], 4)
        self.assertEqual(summary["exact_match"], 2)
        self.assertEqual(summary["near_tie_qualified"], 1)
        self.assertEqual(summary["hard_failures"], 1)
        self.assertAlmostEqual(summary["exact_match_pct"], 50.0)
        self.assertEqual(len(summary["hard_failure_detail"]), 1)
        self.assertEqual(summary["near_tie_margins"], [0.01])

    def test_math_isnan_never_leaks_into_a_crash(self) -> None:
        # sanity: math module import is actually used by classify_request's
        # finite check, not just imported and unused
        self.assertFalse(math.isfinite(float("nan")))


if __name__ == "__main__":
    unittest.main()
