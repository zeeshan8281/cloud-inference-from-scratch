"""Synthetic host guards only; no GPU validation."""
import ast
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

from experiments.fix_holdout import digest

REPO = Path(__file__).resolve().parents[1]


def entrypoint(remote):
    nodes = [item for item in ast.parse((REPO / "warm_prefix_probe_app.py").read_text()).body
             if isinstance(item, ast.FunctionDef) and item.name in ("run", "validate_payload")]
    for node in nodes:
        node.decorator_list = []
    scope = {"Path": Path, "json": json, "__file__": str(REPO / "warm_prefix_probe_app.py"),
             "capture": NS(remote=remote), "diagnostic_sources": lambda _: {"test": "source"}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "warm_app", "exec"), scope)
    return scope["run"]


def payload(candidate="fp32_qkv_and_head"):
    return {"config": {"candidate": candidate}, "process": {"returncode": 0}, "files": {"output.json": json.dumps(
        {"complete": True, "error": None, "numerical_mismatch": True, "schema": "warm-prefix-probe-v1", "config": {"candidate": candidate},
         "cases": {f"{selection}-{phase}": {"candidate_policy": {"candidate": candidate}, "capture": {"records": [{"output_tokens": [1] * 128}] * (8 if selection == "batch" else 1)}}
                   for selection in ("batch", "solo") for phase in ("cold", "warm")}}), "trace.jsonl": "retained"},
        "environment": {"candidate": candidate, "gpu_before": {"uuid": "test-gpu"}, "gpu_after": {"uuid": "test-gpu"},
                        "computational_source": {"test": "core"}, "diagnostic_sources": {"test": "source"},
                        "model": {"id": "test"}, "model_files_sha256": {"weight": "hash"}, "dependencies": {"test": "1"}}}


class TestWarmPrefixApp(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = json.loads((REPO / "experiments/fix-validation-smoke-01/manifest.json").read_text())
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps(self.source))
        patch = mock.patch("experiments.fix_holdout.computational_source", return_value={"test": "core"})
        patch.start()
        self.addCleanup(patch.stop)

    def test_four_sequential_controls_use_complete_exposed_inputs(self):
        remote = mock.Mock(side_effect=lambda *args: payload())
        with redirect_stdout(io.StringIO()):
            entrypoint(remote)(str(self.root / "out"), str(self.manifest))
        self.assertEqual([call.args[1] for call in remote.call_args_list],
                         ["graph_control", "eager_control", "eager_trace", "eager_repeat"])
        expected = next(row for row in self.source["workloads"]["complete_system"]
                        if row["concurrency"] == 8 and row["batch_index"] == 0)
        self.assertTrue(all(call.args[0] == expected for call in remote.call_args_list))

    def test_fixed_candidate_is_forwarded_for_all_four_controls(self):
        selected = "fixed_tile_qkv_fp32_head"
        remote = mock.Mock(side_effect=lambda *args: payload(selected))
        with redirect_stdout(io.StringIO()):
            entrypoint(remote)(str(self.root / "fixed"), str(self.manifest), selected)
        self.assertEqual(remote.call_count, 4)
        self.assertTrue(all(call.args[2] == selected for call in remote.call_args_list))
        remote = mock.Mock()
        with self.assertRaisesRegex(SystemExit, "unknown"):
            entrypoint(remote)(str(self.root / "unknown"), str(self.manifest), "unknown")
        remote.assert_not_called()

    def test_failures_retained_before_stopping(self):
        for kind in ("crash", "incomplete", "source", "gpu", "missing", "environment", "window", "candidate"):
            with self.subTest(kind=kind):
                value = payload()
                if kind == "crash":
                    value["process"]["returncode"] = 1
                elif kind == "incomplete":
                    value["files"]["output.json"] = '{"complete": false}'
                elif kind == "missing":
                    del value["files"]["output.json"]
                elif kind == "environment":
                    value["environment"]["gpu_after"] = {}
                    value["environment"]["environment_error"] = "GPU observation unavailable"
                elif kind == "window":
                    child = json.loads(value["files"]["output.json"])
                    child["cases"]["batch-warm"]["capture"]["records"][0]["output_tokens"] = [1]
                    value["files"]["output.json"] = json.dumps(child)
                elif kind == "candidate":
                    value["environment"]["candidate"] = "wrong"
                elif kind == "source":
                    value["environment"]["diagnostic_sources"] = {}
                else:
                    value["environment"]["gpu_after"]["uuid"] = "other"
                remote = mock.Mock(return_value=value)
                with self.assertRaises(SystemExit):
                    entrypoint(remote)(str(self.root / kind), str(self.manifest))
                self.assertEqual(remote.call_count, 1)
                self.assertEqual(json.loads((self.root / kind / "graph_control.json").read_text()), value)

    def test_overwrite_and_fresh_inputs_rejected_before_dispatch(self):
        remote = mock.Mock()
        with self.assertRaises(SystemExit):
            entrypoint(remote)(str(self.root), str(self.manifest))
        self.source["split"] = "holdout"
        self.source["workloads_sha256"] = digest(self.source["workloads"])
        self.manifest.write_text(json.dumps(self.source))
        with self.assertRaises(SystemExit):
            entrypoint(remote)(str(self.root / "fresh"), str(self.manifest))
        remote.assert_not_called()


class TestDurableWarm(unittest.TestCase):
    def load(self, names, **scope):
        nodes = [node for node in ast.parse((REPO / "warm_prefix_probe_app.py").read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
        for node in nodes:
            node.decorator_list = []
        scope.update(json=json, hashlib=hashlib)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "durable_warm", "exec"), scope)
        return scope

    def test_campaign_commits_raw_before_failure_and_does_not_dispatch_more(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            local = mock.Mock(return_value={"partial": "retained", "process": {"returncode": 1}})
            commits = []
            def commit():
                commits.append(sorted(path.name for path in root.rglob("*.json")))
            def reject(*args):
                self.assertTrue((root / ("a" * 32) / "graph_control.json").exists())
                self.assertIn("graph_control.json", commits[-1])
                raise ValueError("failed child")
            ns = self.load({"campaign"}, Path=lambda value: root if value == "/evidence" else Path(value),
                           evidence_volume=NS(commit=commit), capture=NS(local=local), validate_payload=reject)
            index = ns["campaign"]("a" * 32, {}, "source", "fixed_tile_qkv_fp32_head")
            self.assertEqual(local.call_args.args[2], "fixed_tile_qkv_fp32_head")
            self.assertEqual(index["candidate"], "fixed_tile_qkv_fp32_head")
            self.assertEqual(local.call_count, 1)
            self.assertFalse(index["complete"])
            self.assertIn("failed child", index["error"])
            with self.assertRaises(FileExistsError):
                ns["campaign"]("a" * 32, {}, "source")

    def test_spawn_is_detached_and_call_id_persisted_before_wait(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "out"
            def wait():
                dispatch = json.loads((destination / "dispatch.json").read_text())
                self.assertEqual(dispatch["call_id"], "fc-created")
                self.assertEqual(dispatch["candidate"], "fixed_tile_qkv_fp32_head")
            call = NS(object_id="fc-created", get=wait)
            app_run = mock.Mock(return_value=nullcontext())
            spawn = mock.Mock(return_value=call)
            ns = self.load({"durable"}, Path=Path, __file__=str(REPO / "warm_prefix_probe_app.py"),
                           app=NS(run=app_run, app_id="ap-created"), campaign=NS(spawn=spawn), diagnostic_sources=lambda _: {},
                           recover_artifacts=lambda *args: {"complete": True})
            with mock.patch("experiments.fix_holdout.computational_source", return_value={}), redirect_stdout(io.StringIO()):
                ns["durable"](str(destination), str(REPO / "experiments/fix-validation-smoke-01/manifest.json"), candidate="fixed_tile_qkv_fp32_head")
            app_run.assert_called_once_with(detach=True)
            self.assertEqual(spawn.call_count, 1)
            self.assertEqual(spawn.call_args.args[3], "fixed_tile_qkv_fp32_head")

    def test_get_exception_preserves_saved_handle_and_original_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dispatch = {"call_id": "fc-test", "campaign_id": "c", "app_id": "ap-test"}
            (root / "dispatch.json").write_text(json.dumps(dispatch))
            failure = ConnectionError("client disconnected")
            call = NS(get=mock.Mock(side_effect=failure))
            spawn = mock.Mock()
            ns = self.load({"durable"}, Path=Path,
                           modal=NS(FunctionCall=NS(from_id=lambda _: call)), campaign=NS(spawn=spawn),
                           recover_artifacts=mock.Mock(side_effect=FileNotFoundError("queued")))
            output = io.StringIO()
            with redirect_stdout(output), self.assertRaises(ConnectionError) as caught:
                ns["durable"](temp, "unused", recover=True)
            self.assertIs(caught.exception, failure)
            self.assertEqual(json.loads((root / "dispatch.json").read_text()), dispatch)
            self.assertIn("pending or unavailable", output.getvalue())
            spawn.assert_not_called()

    def test_recovery_uses_saved_call_without_spawn(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "dispatch.json").write_text(json.dumps({"call_id": "fc-test", "campaign_id": "c"}))
            call = NS(get=mock.Mock())
            from_id = mock.Mock(return_value=call)
            spawn = mock.Mock(side_effect=AssertionError("must not respawn"))
            recovered = mock.Mock(return_value={"complete": True, "error": None})
            ns = self.load({"durable"}, Path=Path, modal=NS(FunctionCall=NS(from_id=from_id)),
                           campaign=NS(spawn=spawn), recover_artifacts=recovered)
            with redirect_stdout(io.StringIO()):
                ns["durable"](temp, "unused", recover=True)
            from_id.assert_called_once_with("fc-test")
            call.get.assert_called_once()
            spawn.assert_not_called()
            recovered.assert_called_once_with(root, "c")
