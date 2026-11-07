"""Host guard and evidence-retention tests; synthetic children are not GPU validation."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from experiments.protocol_v2 import V2_BATCHES_PER_CELL, V2_CELLS, execution_after
from tests.test_combined_policy_wiring import status

REPO = Path(__file__).parents[1]


def entrypoints(repo=REPO):
    # Load just these functions, avoiding Modal image construction during unit tests.
    wanted = {"_protocol_v2_preflight", "_protocol_v2_child_error", "_protocol_v2_run",
              "protocol_v2_calibration", "protocol_v2_holdout"}
    definitions = []
    for node in ast.parse((REPO / "modal_app.py").read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            node.decorator_list = []
            definitions.append(node)
    namespace = {"Path": Path, "json": json, "os": os, "subprocess": subprocess,
                 "sha256": hashlib.sha256, "tempfile": tempfile,
                 "__file__": str(repo / "modal_app.py"), "SOURCE_COMMIT": "fixture",
                 "_source_commit": lambda: "fixture"}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(REPO / "modal_app.py"), "exec"), namespace)
    return namespace


class TestV2Preflight(unittest.TestCase):
    def test_existing_output_and_invalid_epsilon_stop_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            functions = entrypoints(root)
            remote = mock.Mock()
            functions["_protocol_v2_run"] = types.SimpleNamespace(remote=remote)
            output = root / "retained.json"
            output.write_text("retained")
            with self.assertRaises(SystemExit):
                functions["protocol_v2_holdout"](0.01, str(output))
            self.assertEqual(output.read_text(), "retained")
            for invalid in (float("nan"), float("inf"), -0.1):
                with self.subTest(epsilon=invalid), self.assertRaises(SystemExit):
                    functions["protocol_v2_holdout"](invalid, str(root / "new.json"))
            remote.assert_not_called()

    def test_executed_image_mismatch_is_retained_as_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            functions = entrypoints()
            functions["_protocol_v2_preflight"] = lambda *args, **kwargs: {"source_fingerprint": "frozen"}
            functions["_protocol_v2_run"] = types.SimpleNamespace(remote=lambda *args: {
                "source": "different", "complete": True, "evidence_errors": {}, "children": {"fixture": {}}
            })
            output = Path(directory) / "mismatch.json"
            with self.assertRaises(SystemExit):
                functions["protocol_v2_holdout"](0.012599945068359375, str(output))
            retained = json.loads(output.read_text())
            self.assertFalse(retained["complete"])
            self.assertIn("source", retained["evidence_errors"])
            self.assertIn("fixture", retained["children"])

    def test_actual_committed_document_epsilon_and_clean_revision_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("modal_app.py", "engine_config.json", "CORRECTNESS_PROTOCOL_V2.md"):
                (root / name).write_bytes((REPO / name).read_bytes())
            def git(*args):
                return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.DEVNULL, text=True)
            git("init")
            git("add", ".")
            git("-c", "user.name=Unit Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
            functions = entrypoints(root)
            source = git("rev-parse", "HEAD").strip() + "+tree-fixture"
            functions.update(SOURCE_COMMIT=source, _source_commit=lambda: source)
            preflight = functions["_protocol_v2_preflight"]
            args = (str(root / "new.json"), 0.012599945068359375)
            manifest = preflight(*args, require_frozen=True)
            self.assertFalse(manifest["dirty"])
            self.assertEqual(manifest["committed_epsilon"], args[1])
            self.assertEqual(manifest["epsilon_document_sha256"], hashlib.sha256(
                (root / "CORRECTNESS_PROTOCOL_V2.md").read_bytes()).hexdigest())
            with self.assertRaises(SystemExit):
                preflight(args[0], 0.02, require_frozen=True)
            (root / "engine_config.json").write_text("dirty")
            with self.assertRaises(SystemExit):
                preflight(*args, require_frozen=True)
            git("checkout", "--", "engine_config.json")
            document = root / "CORRECTNESS_PROTOCOL_V2.md"
            document.write_text(document.read_text() + "\n**Committed epsilon: `0.02`**\n")
            git("add", "CORRECTNESS_PROTOCOL_V2.md")
            git("-c", "user.name=Unit Test", "-c", "user.email=test@example.invalid", "commit", "-m", "ambiguous fixture")
            source = git("rev-parse", "HEAD").strip() + "+tree-fixture"
            functions["SOURCE_COMMIT"] = source
            with self.assertRaisesRegex(SystemExit, "exactly one"):
                preflight(*args, require_frozen=True)


def execution_evidence(implementation="custom"):
    snapshot = {"numerical_policy": status(implementation),
                "graphs": [{"capture_count": 0, "replay_count": 0, "observer_active": True}]}
    return {"before": copy.deepcopy(snapshot), "after": copy.deepcopy(snapshot)}


class TestV2Evidence(unittest.TestCase):
    def test_child_records_must_match_configuration_count_identity_and_order(self):
        validate = entrypoints()["_protocol_v2_child_error"]
        config = {"batch_ids": [[1], [2]], "implementation": "custom"}
        child = {"crashed": False, "result": {"config": config, "result": [{"index": i, "execution_evidence": execution_evidence()} for i in range(2)]}}
        self.assertIsNone(validate(child, config))
        for indices in ([], [0], [0, 0], [1, 0], [0, 1, 2]):
            invalid = copy.deepcopy(child)
            invalid["result"]["result"] = [{"index": index} for index in indices]
            with self.subTest(indices=indices):
                self.assertIsNotNone(validate(invalid, config))
        self.assertIsNotNone(validate(child, {"batch_ids": [[3], [4]]}))

    def run_synthetic_split(self, fault=None):
        functions = entrypoints()
        functions.update(_print_run_header=lambda _: None, RAGGED_MODEL={"id": "fixture"},
                         _sentinel_torch_cuda_version=lambda: "fixture")
        batches = {cell.name: [[[i] * cell.input_tokens for i in range(cell.concurrency)]
                              for _ in range(V2_BATCHES_PER_CELL[cell.concurrency])] for cell in V2_CELLS}
        def child(config, directory):
            directory.mkdir(parents=True)
            (directory / "stdout.log").write_text("fixture stdout")
            (directory / "stderr.log").write_text("fixture stderr")
            if fault == "crash" and config["implementation"] == "custom":
                return {"crashed": True, "stderr_tail": "fixture crash"}
            records = [{"index": i, "output_tokens": [i] * 8, "top_k": [[(i, -0.1)]] * 8,
                        "verified": True, "execution_evidence": execution_evidence(config["implementation"])}
                       for i in range(len(config["batch_ids"]))]
            if fault == "missing":
                records = []
            elif fault == "policy":
                records[0]["execution_evidence"]["after"]["numerical_policy"]["wrappers_active"] = False
            elif fault == "after_rpc":
                records[0]["execution_evidence"]["after"] = {"error": "graph RPC failed"}
            return {"crashed": False, "result": {"config": config, "result": records}}
        functions["_run_protocol_v2_child"] = child
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict("sys.modules", {
            "transformers": types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda _: object()))
        }), mock.patch("experiments.protocol_v2.build_split_batches", return_value=batches), \
                mock.patch("experiments.controlled.environment", return_value={"model_revision": "fixture"}), \
                mock.patch("experiments.sentinel_pilot.resolved_dependency_versions", return_value=["fixture"]), \
                mock.patch("experiments.sentinel_pilot.gpu_state_snapshot", return_value={"uuid": "fixture"}):
            (Path(directory) / "config.json").write_text("fixture model")
            functions["tempfile"] = types.SimpleNamespace(mkdtemp=lambda **_: str(Path(directory) / "children"))
            functions["_prepare_weights"] = lambda _: directory
            return functions["_protocol_v2_run"]("holdout", 0.012599945068359375)

    def test_complete_result_retains_all_evidence_but_never_clears_gate(self):
        result = self.run_synthetic_split()
        self.assertEqual(len(result["children"]), 20)
        self.assertEqual(result["summary"]["total"], 107)
        self.assertTrue(result["complete"])
        self.assertEqual(result["model_file_hashes"]["config.json"], hashlib.sha256(b"fixture model").hexdigest())
        self.assertFalse(result["gate_cleared"])
        self.assertEqual(result["evaluation_kind"], "exposed_v2_regression")
        self.assertEqual(result["batch_workloads"].keys(), {cell.name for cell in V2_CELLS})
        self.assertTrue(all(child["logs"]["stdout"] == "fixture stdout" for child in result["children"].values()))

    def test_crashes_and_both_sides_missing_records_retain_every_child(self):
        for fault in ("crash", "missing", "policy", "after_rpc"):
            with self.subTest(fault=fault):
                result = self.run_synthetic_split(fault)
                self.assertEqual(len(result["children"]), 20)
                self.assertFalse(result["complete"])
                self.assertTrue(result["evidence_errors"])
                self.assertIsNone(result["summary"])
                self.assertEqual(result["classifications"], [])

    def test_missing_and_tampered_actual_evidence_rejected(self):
        validate = entrypoints()["_protocol_v2_child_error"]
        for implementation in ("custom", "vllm"):
            config = {"implementation": implementation, "batch_ids": [[1]]}
            record = {"index": 0, "execution_evidence": execution_evidence(implementation)}
            child = {"crashed": False, "result": {"config": config, "result": [record]}}
            self.assertIsNone(validate(child, config))
            mutations = [lambda r: r.pop("execution_evidence"),
                         lambda r: r["execution_evidence"].update(after=None),
                         lambda r: r["execution_evidence"]["after"].update(numerical_policy=None),
                         lambda r: r["execution_evidence"]["after"]["numerical_policy"].update(wrappers_active=False),
                         lambda r: r["execution_evidence"]["before"]["graphs"][0].update(replay_count=1),
                         lambda r: r["execution_evidence"]["after"].update(graphs=[])]
            if implementation == "vllm":
                mutations.append(lambda r: r["execution_evidence"]["after"]["graphs"][0].update(observer_active=False))
                mutations.append(lambda r: r["execution_evidence"]["after"]["numerical_policy"].update(attention=None))
            for mutate in mutations:
                invalid = copy.deepcopy(child)
                mutate(invalid["result"]["result"][0])
                self.assertIsNotNone(validate(invalid, config))

    def test_after_snapshot_failure_is_retained_for_host_rejection(self):
        with mock.patch("experiments.protocol_v2.execution_snapshot", side_effect=RuntimeError("RPC lost")):
            after = execution_after(object(), "vllm")
        self.assertIn("RPC lost", after["error"])
