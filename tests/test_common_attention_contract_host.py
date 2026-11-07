"""CPU result-guard tests; synthetic metadata is not GPU validation."""

import ast
import copy
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
NS = {"json": json}
TREE = ast.parse((ROOT / "common_attention_contract_app.py").read_text())
exec(
    compile(
        ast.Module(
            body=[
                n
                for n in TREE.body
                if isinstance(n, ast.FunctionDef) and n.name == "validate_payload"
            ],
            type_ignores=[],
        ),
        "<host>",
        "exec",
    ),
    NS,
)


class HostTests(unittest.TestCase):
    def setUp(self):
        self.expected = {
            "core": {
                "files": {
                    "src/cloud_engine/common_attention.py": "a" * 64,
                    "src/cloud_engine/kernel.py": "b" * 64,
                }
            }
        }
        environment = {
            "model": {},
            "dependencies": {},
            "model_files_sha256": {},
            "source_manifest": self.expected,
            "gpu_before": {"uuid": "simulated"},
            "gpu_after": {"uuid": "simulated"},
        }
        self.fixture = {"states": {"actual-fixture-reference": {"environment": environment}}}
        self.config = {"fixture_path": "synthetic-test"}
        cases = []
        for name, m, reqs in [
            ("prefill", 256, 1),
            ("mixed", 258, 3),
            ("shared_decode", 2, 2),
            ("changed_metadata_replay", 2, 1),
        ]:
            phase = {
                "output_alias": True,
                "output_exact": True,
                "cache_append_and_untouched_exact": True,
                "padding_unchanged": True,
                "output_sha256": "1" * 64,
                "cache_sha256": "2" * 64,
            }
            row = {
                "name": name,
                "tokens": m,
                "requests": reqs,
                "expected_output_sha256": "1" * 64,
                "expected_cache_sha256": "2" * 64,
                "eager": copy.deepcopy(phase),
                "compiled": copy.deepcopy(phase),
                "graph": copy.deepcopy(phase),
                "graph_storage": list(range(1, 10)),
                "query_sha256": "3" * 64,
                "metadata_ids": [0, 1],
                "metadata_positions": [511, 511],
                "write_slots": [511, 527],
            }
            if name == "prefill":
                row["historical_custom_target_exact"] = True
            if name == "changed_metadata_replay":
                row.update(
                    stable_graph_storage=True,
                    graph_storage_before=list(range(1, 10)),
                    graph_storage_after=list(range(1, 10)),
                    query_sha256="4" * 64,
                    metadata_ids=[0, 0],
                    metadata_positions=[510, 511],
                    write_slots=[1022, 1023],
                )
            cases.append(row)
        result = {
            "schema": "common-attention-contract-v1",
            "complete": True,
            "error": None,
            "config": self.config,
            "fixture_sha256": "fixture",
            "policy_id": "policy",
            "attention_id": "attention",
            "installed_callable_active": True,
            "native_builder_class": "FlashAttentionMetadataBuilder",
            "full_attention_model_graph_supported": False,
            "source_sha256": {"common_attention": "a" * 64, "kernel": "b" * 64},
            "precision": {
                "allow_fp16_reduced_precision_reduction": False,
                "allow_tf32": False,
                "float32_matmul_precision": "highest",
            },
            "profiling_none_preserves_alias_cache_output": True,
            "cases": cases,
        }
        self.payload = {"capture": result, "environment": environment, "process": {"returncode": 0}}

    def check(self):
        self.payload["raw_checkpoint"] = json.dumps(self.payload["capture"])
        common = types.ModuleType("cloud_engine.common_attention")
        common.ATTENTION_ID = "attention"
        policy = types.ModuleType("cloud_engine.numerical_policy")
        policy.POLICY_ID = "policy"
        with patch.dict(
            sys.modules,
            {"cloud_engine.common_attention": common, "cloud_engine.numerical_policy": policy},
        ):
            NS["validate_payload"](
                self.payload, self.config, self.expected, self.fixture, "fixture"
            )

    def test_complete_result_accepted(self):
        self.check()

    def test_cache_write_or_padding_failure_rejected(self):
        for key in ("cache_append_and_untouched_exact", "padding_unchanged"):
            with self.subTest(key=key):
                self.setUp()
                self.payload["capture"]["cases"][0]["graph"][key] = False
                with self.assertRaisesRegex(ValueError, "primitive contract failed"):
                    self.check()

    def test_stale_logical_metadata_and_changed_storage_rejected(self):
        changed = self.payload["capture"]["cases"][-1]
        changed["metadata_ids"] = [0, 1]
        with self.assertRaisesRegex(ValueError, "logical metadata"):
            self.check()
        changed["metadata_ids"] = [0, 0]
        changed["graph_storage_after"][0] = 99
        with self.assertRaisesRegex(ValueError, "buffer addresses"):
            self.check()

    def test_missing_case_or_imported_source_rejected(self):
        self.payload["capture"]["cases"].pop()
        with self.assertRaisesRegex(ValueError, "missing primitive"):
            self.check()
        self.setUp()
        self.payload["capture"]["source_sha256"]["kernel"] = "wrong"
        with self.assertRaisesRegex(ValueError, "imported candidate source"):
            self.check()

    def test_historical_target_must_reproduce(self):
        self.payload["capture"]["cases"][0]["historical_custom_target_exact"] = False
        with self.assertRaisesRegex(ValueError, "historical custom"):
            self.check()


if __name__ == "__main__":
    unittest.main()
