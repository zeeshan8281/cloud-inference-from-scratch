"""GPU-free tests for the bounded divergence diagnostic: batch construction,
logit/log-prob metrics, and the classification decision rule. Engine
execution itself (run_custom_diagnostic, run_vllm_diagnostic,
run_hf_diagnostic) requires torch/vllm/transformers and a real GPU and is
exercised by the actual diagnostic run, not here.
"""

from __future__ import annotations

import math
import unittest

from experiments.sentinel_diagnostics import (
    NUM_EXTRA_FILLERS,
    TARGET_CELL,
    TARGET_REQUEST_INDEX,
    batch_for_concurrency,
    build_diagnostic_prompts,
    classify_divergence,
    compare_top_k,
    logprobs_from_logits,
    natural_order_c8_batch,
    step_margin,
    top_k_from_logits,
)
from tests.test_sentinel_pilot import FakeTokenizer


class TestBatchConstruction(unittest.TestCase):
    def test_target_is_always_at_position_zero_and_identical_across_sizes(self) -> None:
        prompts = build_diagnostic_prompts(FakeTokenizer())
        for concurrency in (1, 2, 8, 32):
            batch = batch_for_concurrency(prompts, concurrency)
            self.assertEqual(len(batch), concurrency)
            self.assertEqual(batch[0], prompts["target"]["input_token_ids"])

    def test_enough_fillers_exist_for_the_largest_concurrency(self) -> None:
        prompts = build_diagnostic_prompts(FakeTokenizer())
        self.assertGreaterEqual(len(prompts["fillers"]), 31)
        self.assertEqual(len(prompts["fillers"]), 7 + NUM_EXTRA_FILLERS)

    def test_target_matches_the_real_pilot_run_exactly(self) -> None:
        from experiments.sentinel_pilot import materialize_cell_workload

        tokenizer = FakeTokenizer()
        real_cell = materialize_cell_workload("resource_normalized", 1, TARGET_CELL, "unique", tokenizer)
        prompts = build_diagnostic_prompts(tokenizer)
        self.assertEqual(prompts["target"]["input_token_ids"], real_cell[1]["input_token_ids"])

    def test_reproducible_across_calls(self) -> None:
        tokenizer = FakeTokenizer()
        first = build_diagnostic_prompts(tokenizer)
        second = build_diagnostic_prompts(tokenizer)
        self.assertEqual(first, second)

    def test_natural_order_batch_matches_the_real_cell_exactly(self) -> None:
        from experiments.sentinel_pilot import materialize_cell_workload

        tokenizer = FakeTokenizer()
        real_cell = materialize_cell_workload("resource_normalized", 1, TARGET_CELL, "unique", tokenizer)
        prompts = build_diagnostic_prompts(tokenizer)
        batch, target_index = natural_order_c8_batch(prompts)
        self.assertEqual(target_index, TARGET_REQUEST_INDEX)
        self.assertEqual(len(batch), TARGET_CELL.concurrency)
        self.assertEqual(batch, [record["input_token_ids"] for record in real_cell])
        self.assertEqual(batch[target_index], prompts["target"]["input_token_ids"])


class TestLogitMetrics(unittest.TestCase):
    def test_logprobs_from_logits_sum_to_one_in_probability_space(self) -> None:
        logprobs = logprobs_from_logits([1.0, 2.0, 3.0, 0.5])
        total = sum(math.exp(value) for value in logprobs)
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_top_k_from_logits_is_sorted_descending(self) -> None:
        logits = [float(i) for i in range(50)]
        top = top_k_from_logits(logits, k=5)
        self.assertEqual([token_id for token_id, _ in top], [49, 48, 47, 46, 45])
        self.assertEqual(len(top), 5)

    def test_step_margin_is_top1_minus_top2(self) -> None:
        top = [(5, -0.1), (7, -2.3), (9, -5.0)]
        self.assertAlmostEqual(step_margin(top), 2.2, places=9)

    def test_step_margin_none_when_fewer_than_two_entries(self) -> None:
        self.assertIsNone(step_margin([(5, -0.1)]))

    def test_compare_top_k_identical_lists_gives_perfect_agreement(self) -> None:
        top = [(1, -0.1), (2, -1.0), (3, -2.0)]
        metrics = compare_top_k(top, top)
        self.assertAlmostEqual(metrics["max_abs_diff"], 0.0, places=9)
        self.assertAlmostEqual(metrics["cosine_similarity"], 1.0, places=9)
        self.assertAlmostEqual(metrics["top_k_overlap"], 3 / 20)

    def test_compare_top_k_disjoint_lists_give_zero_overlap(self) -> None:
        a = [(1, -0.1), (2, -1.0)]
        b = [(3, -0.1), (4, -1.0)]
        metrics = compare_top_k(a, b)
        self.assertEqual(metrics["top_k_overlap"], 0.0)


class TestClassification(unittest.TestCase):
    def test_custom_diverges_from_hf_while_vllm_matches_is_a_custom_bug(self) -> None:
        result = classify_divergence(
            custom_by_concurrency={1: [1, 2, 3], 8: [1, 2, 3]},
            vllm_by_concurrency={1: [4, 5, 6], 8: [4, 5, 6]},
            hf_c1=[4, 5, 6],
        )
        self.assertEqual(result["verdict"], "custom_engine_correctness_bug")

    def test_vllm_diverges_from_hf_while_custom_matches_is_a_vllm_effect(self) -> None:
        result = classify_divergence(
            custom_by_concurrency={1: [4, 5, 6], 8: [4, 5, 6]},
            vllm_by_concurrency={1: [1, 2, 3], 8: [1, 2, 3]},
            hf_c1=[4, 5, 6],
        )
        self.assertEqual(result["verdict"], "vllm_backend_or_configuration_effect")

    def test_batch_dependent_change_within_an_engine_is_flagged(self) -> None:
        result = classify_divergence(
            custom_by_concurrency={1: [4, 5, 6], 8: [4, 5, 9]},
            vllm_by_concurrency={1: [4, 5, 6], 8: [4, 5, 6]},
            hf_c1=[4, 5, 6],
        )
        self.assertEqual(result["verdict"], "numerical_equivalence_issue_batch_dependent")
        self.assertFalse(result["custom_batch_invariant"])
        self.assertTrue(result["vllm_batch_invariant"])

    def test_everything_agreeing_is_no_divergence(self) -> None:
        result = classify_divergence(
            custom_by_concurrency={1: [4, 5, 6], 8: [4, 5, 6]},
            vllm_by_concurrency={1: [4, 5, 6], 8: [4, 5, 6]},
            hf_c1=[4, 5, 6],
        )
        self.assertEqual(result["verdict"], "no_divergence_observed_in_this_run")

    def test_without_an_hf_reference_falls_back_to_batch_invariance_only(self) -> None:
        result = classify_divergence(
            custom_by_concurrency={1: [4, 5, 6], 8: [4, 5, 9]},
            vllm_by_concurrency={1: [1, 2, 3], 8: [1, 2, 3]},
            hf_c1=None,
        )
        self.assertEqual(result["verdict"], "numerical_equivalence_issue_batch_dependent")
        self.assertFalse(result["custom_matches_hf_at_c1"])
        self.assertFalse(result["vllm_matches_hf_at_c1"])


if __name__ == "__main__":
    unittest.main()
