"""Committed performance/reliability evidence remains intact and above its gates."""

import hashlib
import json
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).parents[1]
ARTIFACTS = ROOT / "artifacts"


class TestCommittedEvidence(unittest.TestCase):
    def test_api_lifecycle_gate(self) -> None:
        result = json.loads((ARTIFACTS / "api-lifecycle.json").read_text())
        self.assertTrue(result["passed"])
        self.assertGreaterEqual(result["unit_tests"], 66)
        self.assertEqual(
            set(result["fastapi_checks"]),
            {
                "legacy_auth",
                "admin_metrics",
                "tenant_metrics_denied",
                "malformed_content_length",
            },
        )

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

    def test_a100_correctness_gate(self) -> None:
        result = json.loads(
            (ARTIFACTS / "ragged-a100-correctness.json").read_text()
        )
        self.assertEqual(result["gpu"], "NVIDIA A100-SXM4-40GB")
        self.assertEqual(result["compute_capability"], [8, 0])
        self.assertEqual(result["passed"], 20)
        self.assertEqual(result["failed"], 0)
        self.assertTrue(all(check["passed"] for check in result["checks"]))

    def test_current_qwen_correctness_gate(self) -> None:
        result = json.loads(
            (ARTIFACTS / "ragged-l4-correctness.json").read_text()
        )
        self.assertEqual(result["gpu"], "NVIDIA L4")
        self.assertEqual(result["passed"], 20)
        self.assertEqual(result["failed"], 0)
        self.assertTrue(all(check["passed"] for check in result["checks"]))

    def test_llama_family_correctness_gate(self) -> None:
        result = json.loads((ARTIFACTS / "llama-ragged-l4.json").read_text())
        self.assertTrue(result["passed"])
        self.assertTrue(result["model"].startswith("TinyLlama/"))
        self.assertEqual(result["oracle_sequences"], 4)
        self.assertEqual(result["tokens_per_sequence"], 8)
        self.assertGreaterEqual(result["max_forward_request_count"], 4)
        self.assertEqual(result["gpu"], "NVIDIA L4")

    def test_cuda_graph_performance_gate(self) -> None:
        result = json.loads((ARTIFACTS / "cuda-graph-l4.json").read_text())
        self.assertTrue(result["exact_token_parity"])
        self.assertGreaterEqual(result["cuda_graph_captures"], 1)
        self.assertGreaterEqual(result["cuda_graph_replays"], 90)
        self.assertGreaterEqual(result["speedup"], 1.2)

    def test_candidate_stack_compatibility_gate(self) -> None:
        result = json.loads(
            (ARTIFACTS / "compatibility-candidate-l4.json").read_text()
        )
        self.assertTrue(result["passed"])
        self.assertNotIn("unknown", result["source_revision"])
        self.assertEqual(result["candidate_stack"]["torch"], "2.13.0")
        self.assertEqual(result["candidate_stack"]["triton"], "3.7.1")
        self.assertEqual(result["candidate_stack"]["transformers"], "5.15.1")
        self.assertEqual(result["oracle_sequences"], 4)
        self.assertEqual(result["tokens_per_sequence"], 8)
        self.assertGreaterEqual(result["max_forward_request_count"], 4)
        self.assertGreaterEqual(result["cuda_graph_replays"], 1)

    def test_quantized_capacity_gate(self) -> None:
        result = json.loads(
            (ARTIFACTS / "quantization-int8-mlp-qwen3b-l4.json").read_text()
        )
        self.assertTrue(result["passed"])
        self.assertNotIn("unknown", result["source_revision"])
        self.assertEqual(result["gpu"], "NVIDIA L4")
        self.assertEqual(result["quantization"], "bitsandbytes-llm-int8")
        self.assertGreaterEqual(result["teacher_forced_tokens"], 2_000)
        self.assertLessEqual(result["perplexity_ratio"], 1.05)
        self.assertGreaterEqual(result["steady_memory_reduction_percent"], 20)
        self.assertLessEqual(result["latency_ratio"], 2.5)
        self.assertGreaterEqual(result["int8"]["max_forward_request_count"], 16)
        self.assertEqual(result["int8"]["cuda_graph_captures"], 0)

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
