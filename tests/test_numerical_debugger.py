"""Synthetic unit cases test reporting only; GPU evidence is collected separately."""

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.numerical_debugger import compare_runs, write_report


def capture(tokens=(1, 1, 1)):
    return {
        "crashed": False,
        "result": {
            "config": {"batch_ids": [[7, 8]], "target_index": 0},
            "result": {
                "verified": True,
                "output_tokens": list(tokens),
                "top_k": [[[token, -1.0], [3, -1.5]] for token in tokens],
            },
        },
    }


class DebuggerTests(unittest.TestCase):
    def test_stops_at_first_difference_even_if_later_metrics_invalid(self):
        a, b = capture(), capture((1, 2, 2))
        b["result"]["result"]["top_k"][2] = [[2, float("nan")]]
        result = compare_runs(a, b)
        self.assertEqual(result["first_difference"], 1)
        self.assertEqual(len(result["compared_steps"]), 2)
        self.assertEqual(result["excluded_post_divergence_steps"], 1)

    def test_fail_closed_for_missing_nonfinite_unverified_and_mismatched_inputs(self):
        a = capture()
        for case in (
            "missing",
            "nonfinite",
            "unverified",
            "prefix",
            "empty",
            "wrong_choice",
            "duplicate",
        ):
            b = copy.deepcopy(a)
            r = b["result"]["result"]
            if case == "missing":
                r["top_k"] = []
            if case == "nonfinite":
                r["top_k"][0][0][1] = float("inf")
            if case == "unverified":
                r["verified"] = False
            if case == "prefix":
                b["result"]["config"]["batch_ids"] = [[9]]
            if case == "empty":
                r["output_tokens"] = []
            if case == "wrong_choice":
                r["top_k"][0] = [[3, -0.1], [1, -1.0]]
            if case == "duplicate":
                r["top_k"][0] = [[1, -1.0], [1, -1.0]]
            with self.subTest(case=case):
                self.assertEqual(compare_runs(a, b)["status"], "unavailable")

    def test_length_mismatch_is_not_equality(self):
        self.assertEqual(compare_runs(capture(), capture((1,)))["status"], "incomplete")

    def test_tied_selected_token_is_valid(self):
        a = capture()
        a["result"]["result"]["top_k"][0] = [[3, -1.0], [1, -1.0]]
        self.assertEqual(compare_runs(a, a)["status"], "exact_in_observed_window")

    def test_report_reproducible_and_html_escaped(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root / "raw.json"
            raw.write_text(
                json.dumps(
                    {
                        "runs": {"a": capture(), "b": capture((2, 2, 2))},
                        "comparisons": [{"label": "<script>", "left": "a", "right": "b"}],
                    }
                )
            )
            write_report(raw, root)
            first = (root / "summary.json").read_bytes()
            write_report(raw, root)
            self.assertEqual(first, (root / "summary.json").read_bytes())
            self.assertIn("&lt;script&gt;", (root / "index.html").read_text())
            self.assertNotIn("<script>", (root / "index.html").read_text())


if __name__ == "__main__":
    unittest.main()
