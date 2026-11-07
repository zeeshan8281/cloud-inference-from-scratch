"""Host-only guard tests with synthetic child results; not GPU validation."""
from __future__ import annotations

import ast
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments.fix_holdout import digest

REPO = Path(__file__).resolve().parents[1]


def entrypoint(remote):
    node = next(node for node in ast.parse((REPO / "residual_probe_app.py").read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "run")
    node.decorator_list = []
    scope = {"Path": Path, "json": json, "capture": SimpleNamespace(remote=remote)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "residual_probe_app", "exec"), scope)
    return scope["run"]


def payload():
    return {"process": {"returncode": 0, "stdout": "retained stdout", "stderr": "retained stderr"},
            "environment": {"gpu_before": {"uuid": "fixture-GPU"}, "gpu_after": {"uuid": "fixture-GPU"}},
            "files": {"output.json": json.dumps({"complete": True, "error": None}), "trace.jsonl": "raw fixture trace"}}


class TestResidualProbeApp(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = json.loads((REPO / "experiments/fix-validation-smoke-01/manifest.json").read_text())
        self.path = self.root / "manifest.json"
        self.path.write_text(json.dumps(self.manifest))

    def test_overwrite_is_refused_before_manifest_read_or_remote(self):
        remote = mock.Mock()
        with self.assertRaisesRegex(SystemExit, "overwrite"):
            entrypoint(remote)(str(self.root), str(self.root / "missing.json"))
        remote.assert_not_called()

    def test_only_exposed_complete_manifest_is_accepted(self):
        for mutation in ("split", "namespace", "checksum", "missing_cell"):
            with self.subTest(mutation=mutation):
                manifest = copy.deepcopy(self.manifest)
                if mutation == "split":
                    manifest["split"] = "holdout"
                elif mutation == "namespace":
                    manifest["namespace"] = "new-prompts"
                elif mutation == "checksum":
                    manifest["workloads_sha256"] = "wrong"
                else:
                    manifest["workloads"]["resource_normalized"].pop()
                    manifest["workloads_sha256"] = digest(manifest["workloads"])
                self.path.write_text(json.dumps(manifest))
                remote = mock.Mock()
                with self.assertRaises((SystemExit, ValueError)):
                    entrypoint(remote)(str(self.root / mutation), str(self.path))
                remote.assert_not_called()

    def test_exact_c8_batch0_selected_and_ten_cases_preserved(self):
        remote = mock.Mock(side_effect=lambda *args: payload())
        destination = self.root / "result"
        with redirect_stdout(io.StringIO()):
            entrypoint(remote)(str(destination), str(self.path))
        expected = next(batch["input_token_ids"] for batch in self.manifest["workloads"]["resource_normalized"]
                        if batch["concurrency"] == 8 and batch["batch_index"] == 0)
        self.assertEqual(remote.call_count, 10)
        self.assertTrue(all(call.args[1] == expected for call in remote.call_args_list))
        self.assertEqual(remote.call_args_list[0].args, ("custom", expected, "batch", True))
        self.assertEqual(len(list(destination.glob("*.json"))), 11)

    def test_failed_missing_malformed_and_incomplete_children_are_retained(self):
        for failure in ("crash", "missing", "malformed", "not_object", "incomplete", "error"):
            with self.subTest(failure=failure):
                child = payload()
                if failure == "crash":
                    child["process"]["returncode"] = 1
                elif failure == "missing":
                    child["files"].pop("output.json")
                else:
                    child["files"]["output.json"] = {
                        "malformed": "{", "not_object": "[]", "incomplete": '{"complete": false}',
                        "error": '{"complete": true, "error": "failure"}'}[failure]
                remote = mock.Mock(return_value=child)
                destination = self.root / failure
                with self.assertRaises(SystemExit):
                    entrypoint(remote)(str(destination), str(self.path))
                self.assertEqual(remote.call_count, 1)
                self.assertEqual(json.loads((destination / "custom-batch-all.json").read_text()), child)

    def test_biased_experiment_runs_one_retained_child(self):
        remote = mock.Mock(return_value=payload())
        destination = self.root / "biased"
        with redirect_stdout(io.StringIO()):
            entrypoint(remote)(str(destination), str(self.path), "biased")
        self.assertEqual(remote.call_count, 1)
        self.assertEqual(remote.call_args.args[0], "biased")
        self.assertTrue((destination / "biased-batch-all.json").exists())
        with self.assertRaisesRegex(SystemExit, "unknown experiment"):
            entrypoint(remote)(str(self.root / "unknown"), str(self.path), "unknown")

    def test_gpu_change_across_children_fails_after_retaining_second(self):
        first, second = payload(), payload()
        second["environment"]["gpu_before"]["uuid"] = "other-GPU"
        second["environment"]["gpu_after"]["uuid"] = "other-GPU"
        remote = mock.Mock(side_effect=[first, second])
        destination = self.root / "changed-gpu"
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(SystemExit, "GPU identity"):
            entrypoint(remote)(str(destination), str(self.path))
        self.assertEqual(remote.call_count, 2)
        self.assertTrue((destination / "vllm-batch-control.json").exists())
