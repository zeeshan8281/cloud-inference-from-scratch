"""Bounded rolling-metrics tests — standard library only."""

import unittest

from cloud_engine.metrics import Metrics, percentile


class TestMetrics(unittest.TestCase):
    def test_percentile_interpolates_unsorted_values(self) -> None:
        self.assertEqual(percentile([], 50), 0.0)
        self.assertEqual(percentile([30, 10, 20], 50), 20)
        self.assertEqual(percentile([0, 100], 95), 95)

    def test_latency_storage_is_capacity_bounded(self) -> None:
        metrics = Metrics(latency_capacity=3, window_seconds=60)
        for value in range(10):
            metrics.record_ttft_ms(value, now_ns=value)
        self.assertEqual(len(metrics._ttft.samples), 3)
        self.assertEqual(metrics.snapshot(now_ns=10)["latency_ms"]["ttft_p50"], 8)

    def test_old_samples_are_pruned_from_rolling_window(self) -> None:
        metrics = Metrics(window_seconds=1)
        metrics.record_output_token(2, now_ns=0)
        metrics.record_iteration(4, now_ns=0)
        snapshot = metrics.snapshot(now_ns=2_000_000_000)
        self.assertEqual(snapshot["tokens"]["output_per_second_60s"], 0)
        self.assertEqual(snapshot["scheduler"]["mean_batch_size_60s"], 0)

    def test_reset_clears_runtime_values_but_keeps_gauges(self) -> None:
        metrics = Metrics()
        metrics.record_ttft_ms(12)
        metrics.record_output_token(3)
        metrics.inc_completed()
        metrics.set_kv_stats({"blocks_used": 2})
        metrics.reset_runtime()
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["requests"]["completed_total"], 0)
        self.assertEqual(snapshot["tokens"]["output_total"], 0)
        self.assertEqual(snapshot["latency_ms"]["ttft_p50"], 0)
        self.assertEqual(snapshot["kv_cache"]["blocks_used"], 2)


if __name__ == "__main__":
    unittest.main()
