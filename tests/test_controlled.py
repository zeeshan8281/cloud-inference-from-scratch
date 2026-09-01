from __future__ import annotations

import unittest

from experiments.controlled import MATRIX, workload_hash, workload_records


class TestControlledExperiment(unittest.TestCase):
    def test_workload_records_are_exact_and_stable(self) -> None:
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                del add_special_tokens
                return [len(text)]

        first = workload_records(Tokenizer())
        second = workload_records(Tokenizer())
        self.assertEqual(len(first), sum(cell[2] for cell in MATRIX))
        self.assertTrue(all(len(row["input_token_ids"]) == row["input_tokens"] for row in first))
        self.assertEqual(workload_hash(first), workload_hash(second))


if __name__ == "__main__":
    unittest.main()
