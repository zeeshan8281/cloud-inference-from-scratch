"""Focused, GPU-free tests for the vLLM per-step timestamp accounting used by
experiments/controlled.py. vLLM's `LLMEngine.step()` returns a batch of
RequestOutput objects with no per-token timestamp, so the harness must record
one wall-clock stamp per step() call and apply it to every token that step
exposed -- including when a single step exposes more than one new token for a
request (chunked prefill, speculative decoding, or a slow test harness poll).
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from experiments.controlled import _record_vllm_step


@dataclass
class FakeCompletionOutput:
    token_ids: list[int]


@dataclass
class FakeRequestOutput:
    request_id: str
    outputs: list[FakeCompletionOutput]
    finished: bool = False


def _fresh_state(*request_ids: str) -> dict[str, dict]:
    return {request_id: {"stamps": []} for request_id in request_ids}


class TestVLLMStepAccounting(unittest.TestCase):
    def test_stamp_is_read_once_after_step_returns_not_per_token(self) -> None:
        state = _fresh_state("r0")
        step_one_outputs = [FakeRequestOutput("r0", [FakeCompletionOutput([11])])]
        step_two_outputs = [FakeRequestOutput("r0", [FakeCompletionOutput([11, 22])])]

        _record_vllm_step(state, step_one_outputs, stamp=100.0)
        _record_vllm_step(state, step_two_outputs, stamp=100.5)

        self.assertEqual(state["r0"]["stamps"], [100.0, 100.5])
        self.assertEqual(state["r0"]["token_ids"], [11, 22])

    def test_multiple_tokens_in_one_step_share_the_single_step_stamp(self) -> None:
        state = _fresh_state("r0")
        outputs = [FakeRequestOutput("r0", [FakeCompletionOutput([1, 2, 3])], finished=True)]

        _record_vllm_step(state, outputs, stamp=42.0)

        self.assertEqual(state["r0"]["stamps"], [42.0, 42.0, 42.0])
        self.assertEqual(state["r0"]["token_ids"], [1, 2, 3])
        self.assertTrue(state["r0"]["finished"])

    def test_second_step_only_stamps_the_newly_exposed_tokens(self) -> None:
        state = _fresh_state("r0")
        _record_vllm_step(
            state, [FakeRequestOutput("r0", [FakeCompletionOutput([1, 2])])], stamp=1.0
        )
        _record_vllm_step(
            state, [FakeRequestOutput("r0", [FakeCompletionOutput([1, 2, 3, 4])])], stamp=2.0
        )

        self.assertEqual(state["r0"]["stamps"], [1.0, 1.0, 2.0, 2.0])

    def test_independent_requests_in_the_same_step_each_get_their_own_stamps(self) -> None:
        state = _fresh_state("r0", "r1")
        outputs = [
            FakeRequestOutput("r0", [FakeCompletionOutput([1])]),
            FakeRequestOutput("r1", [FakeCompletionOutput([9, 10])]),
        ]

        _record_vllm_step(state, outputs, stamp=5.0)

        self.assertEqual(state["r0"]["stamps"], [5.0])
        self.assertEqual(state["r1"]["stamps"], [5.0, 5.0])

    def test_a_step_with_no_new_tokens_for_a_request_appends_nothing(self) -> None:
        state = _fresh_state("r0")
        _record_vllm_step(
            state, [FakeRequestOutput("r0", [FakeCompletionOutput([1])])], stamp=1.0
        )
        # A step where the tracked completion output is unchanged must not
        # fabricate a duplicate stamp for a token that was already counted.
        _record_vllm_step(
            state, [FakeRequestOutput("r0", [FakeCompletionOutput([1])])], stamp=2.0
        )

        self.assertEqual(state["r0"]["stamps"], [1.0])


if __name__ == "__main__":
    unittest.main()
