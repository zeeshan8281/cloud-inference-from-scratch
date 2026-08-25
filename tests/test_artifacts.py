"""Committed performance/reliability evidence remains intact and above its gates."""

import hashlib
import json
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).parents[1]
ARTIFACTS = ROOT / "artifacts"


class TestCommittedEvidence(unittest.TestCase):
    def test_aiperf_native_artifact(self) -> None:
        with zipfile.ZipFile(ARTIFACTS / "aiperf-responses.zip") as archive:
            metadata = json.loads(archive.read("cloud-inference-run.json"))
            aggregate = json.loads(
                archive.read("aggregate/profile_export_aiperf_aggregate.json")
            )
            records = [
                name for name in archive.namelist() if name.endswith("profile_export.jsonl")
            ]
        self.assertEqual(metadata["aiperf_version"], "0.12.0")
        self.assertEqual(metadata["runs"], 3)
        self.assertNotIn("unknown", metadata["source_revision"])
        self.assertEqual(len(records), 3)
        self.assertGreaterEqual(
            aggregate["metrics"]["output_token_throughput_avg"]["mean"], 35
        )

    def test_gpu_profile_hashes_and_capabilities(self) -> None:
        with zipfile.ZipFile(ARTIFACTS / "nsight-ragged-l4.zip") as archive:
            metadata = json.loads(archive.read("cloud-inference-run.json"))
            report = archive.read("ragged-l4.nsys-rep")
            torch_trace = archive.read("ragged-l4.pt.trace.json")
        self.assertEqual(hashlib.sha256(report).hexdigest(), metadata["report_sha256"])
        self.assertEqual(
            hashlib.sha256(torch_trace).hexdigest(), metadata["torch_trace_sha256"]
        )
        self.assertIn("scheduler.execute_batch", metadata["stats"])
        self.assertTrue(metadata["pytorch_cuda_kernel_records"])
        self.assertFalse(metadata["nsight_cuda_kernel_records"])

    def test_experiment_correctness_gate(self) -> None:
        result = json.loads(
            (ARTIFACTS / "experiment-short-prefill-first-summary.json").read_text()
        )
        self.assertTrue(all(result["correctness"].values()))
        self.assertEqual(
            result["baseline"]["workload_hash"], result["result"]["workload_hash"]
        )

    def test_reliability_soak_gate(self) -> None:
        result = json.loads((ARTIFACTS / "reliability-soak-l4.json").read_text())
        self.assertTrue(all(result["soak"]["checks"].values()))
        self.assertGreaterEqual(result["soak"]["issued"], 1_900)
        self.assertEqual(result["soak"]["failed"], 0)
        self.assertTrue(result["restart"]["passed"])


if __name__ == "__main__":
    unittest.main()
