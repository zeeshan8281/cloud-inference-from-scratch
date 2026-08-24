from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import unittest
from types import SimpleNamespace

from benchmarks.run import WorkloadItem, _execute_run, workload_hash


class TestBenchmarkReproducibility(unittest.TestCase):
    def test_execute_run_finishes_and_samples_live_cache(self) -> None:
        class Cache:
            gathered_bytes = 7

            def stats(self):
                values = {
                    "reserved_bytes": 100,
                    "occupied_bytes": 60,
                    "internal_fragmentation_bytes": 40,
                }
                return SimpleNamespace(as_metrics=lambda: values)

        class Engine:
            cache = Cache()
            config = SimpleNamespace(eos_token_id=0)

            async def submit(self, _prompt, _config):
                async def wait():
                    await asyncio.sleep(0.01)
                    return SimpleNamespace(
                        ttft_ms=1.0,
                        e2e_ms=2.0,
                        output_tokens=2,
                        input_tokens=1,
                        finish_reason="length",
                    )

                return SimpleNamespace(wait=wait)

            def snapshot_metrics(self):
                return {
                    "scheduler": {"mean_batch_size_60s": 1.0, "max_batch_size_60s": 1},
                    "requests": {"failed_total": 0, "cancelled_total": 0, "rejected_total": 0},
                }

        result = asyncio.run(_execute_run(Engine(), [WorkloadItem("safe", 1, 2)]))
        self.assertEqual(result["kv_peak_reserved_bytes"], 100)
        self.assertEqual(result["kv_internal_fragmentation_bytes"], 40)

    def test_workload_is_stable_across_python_hash_seeds(self) -> None:
        script = """
from benchmarks.run import build_workload, workload_hash
class Tokenizer:
    def encode(self, text):
        return text.split()
print(workload_hash(build_workload('mixed', Tokenizer())))
"""
        hashes = []
        for seed in ("1", "2"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            hashes.append(
                subprocess.check_output([sys.executable, "-c", script], env=env, text=True).strip()
            )
        self.assertEqual(hashes[0], hashes[1])

    def test_workload_hash_covers_prompt_without_exposing_it(self) -> None:
        first = [WorkloadItem("alpha beta", 2, 4)]
        second = [WorkloadItem("gamma beta", 2, 4)]
        self.assertNotEqual(workload_hash(first), workload_hash(second))
        self.assertNotIn("alpha", workload_hash(first))

    def test_online_summary_reports_p99_errors_and_slo_goodput(self) -> None:
        from benchmarks.online import _summarize

        records = [
            {
                "ttft_ms": 100.0,
                "itl_ms": [10.0, 20.0],
                "e2e_ms": 200.0,
                "output_tokens": 3,
                "error": None,
            },
            {
                "ttft_ms": 2000.0,
                "itl_ms": [200.0],
                "e2e_ms": 3000.0,
                "output_tokens": 2,
                "error": None,
            },
            {"output_tokens": 0, "error": "injected"},
        ]
        summary = _summarize(records, 2.0, [0, 1, 2], 1000.0, 100.0)
        self.assertEqual(
            summary["requests"],
            {"offered": 3, "completed": 2, "errors": 1, "slo_good": 1},
        )
        self.assertEqual(summary["throughput"]["output_tokens_per_second"], 2.5)
        self.assertGreater(summary["latency"]["ttft_p99_ms"], 1900)
        self.assertEqual(summary["queue"]["max_depth"], 2)


if __name__ == "__main__":
    unittest.main()
