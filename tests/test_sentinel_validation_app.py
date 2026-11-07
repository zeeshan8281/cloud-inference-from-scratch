"""Host orchestration checks only; no fixture result is GPU validation."""
from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import io
import json
import subprocess
import tempfile
import threading
import types
import unittest
import uuid
from dataclasses import asdict
from pathlib import Path
from unittest import mock

from tests.test_combined_policy_wiring import status as policy_status

REPO = Path(__file__).parents[1]


def functions(root):
    definitions = []
    for node in ast.parse((REPO/"sentinel_validation_app.py").read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in {"preflight", "verify_local_source", "verify_pair", "verify_artifact_provenance", "verify_child_records", "verify_holdout_prerequisite", "verify_frozen_lineage", "run", "execute_pilot"}:
            node.decorator_list = []
            definitions.append(node)
    namespace = {"Path": Path, "json": json, "hashlib": hashlib, "subprocess": subprocess, "asdict": asdict,
                 "uuid": uuid, "__file__": str(root/"sentinel_validation_app.py"), "PINNED": {"ragged_model": {"id": "fixture"}}}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(REPO/"sentinel_validation_app.py"), "exec"), namespace)
    return namespace


class SentinelValidationAppTests(unittest.TestCase):
    def test_freeze_precedes_output_creation_and_guards_later_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("sentinel_validation_app.py", "modal_app.py", "engine_config.json", "pyproject.toml", "uv.lock"):
                (root/name).write_text("fixture")
            def git(*args):
                return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
            git("init")
            git("add", ".")
            git("-c", "user.name=Unit Test", "-c", "user.email=unit@example.invalid", "commit", "-m", "fixture")
            source = git("rev-parse", "HEAD") + "+tree-fixture"
            functions_ = functions(root)
            functions_.update(SOURCE_COMMIT=source, _source_commit=lambda: source)
            destination = root/"output"
            manifest = functions_["preflight"](destination)
            self.assertFalse(destination.exists())
            destination.mkdir()
            functions_["verify_local_source"](manifest)
            with self.assertRaises(SystemExit):
                functions_["preflight"](destination)
            (root/"modal_app.py").write_text("changed")
            with self.assertRaises(SystemExit):
                functions_["verify_local_source"](manifest)
            with self.assertRaises(SystemExit):
                functions_["preflight"](root/"other-output")
            self.assertFalse((root/"other-output").exists())

    def test_provenance_and_independent_record_guards(self):
        from experiments.sentinel_pilot import SENTINEL_CELLS, StopPilot
        from tests.test_sentinel_runtime import result_fixture

        namespace = functions(REPO)
        provenance = {"model_files_before": {"weights": "a"*64}, "model_files_after": {"weights": "a"*64},
                      "dependency_versions": ["torch==fixture"]}
        namespace["verify_artifact_provenance"](provenance)
        for field, value in (("model_files_after", {"weights": "b"*64}), ("model_files_before", {}),
                             ("dependency_versions", [])):
            bad = copy.deepcopy(provenance)
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(StopPilot):
                namespace["verify_artifact_provenance"](bad)
        result = result_fixture()
        result["engine_config"] = {"inference_precision": {"allow_fp16_reduced_precision_reduction": False}}
        workload = {"cells": {cell.name: {"unique": [{"request_index": i, "output_tokens": cell.output_tokens}
                        for i in range(cell.concurrency)]} for cell in SENTINEL_CELLS}}
        for cell in result["cells"].values():
            cell["phases"]["unique"]["numerical_policy"] = {"before": policy_status(), "after": policy_status()}
            for record in cell["phases"]["unique"]["records"]:
                record.update(error=None, timeout=False)
        namespace["verify_child_records"](result, workload, "custom")
        for fault in ("duplicate", "error", "timeout", "precision"):
            bad = copy.deepcopy(result)
            records = bad["cells"][SENTINEL_CELLS[1].name]["phases"]["unique"]["records"]
            if fault == "duplicate":
                records[0]["request_index"] = records[1]["request_index"]
            elif fault == "precision":
                bad["engine_config"]["inference_precision"]["allow_fp16_reduced_precision_reduction"] = True
            else:
                records[0][fault] = True
            with self.subTest(fault=fault), self.assertRaises(StopPilot):
                namespace["verify_child_records"](bad, workload, "custom")

    def test_bounded_twenty_pairs_and_cancelled_started_jobs_are_retained(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                namespace = functions(root)
                destination = root / "output"
                calls, lock, cancelled = [], threading.Lock(), threading.Event()
                store_values = {}
                class Store:
                    def __init__(self, values=store_values, event=cancelled):
                        self.values, self.event = values, event
                    def get(self, key):
                        return self.values.get(key)
                    def put(self, key, value, skip_if_exists=False):
                        if skip_if_exists and key in self.values:
                            return False
                        self.values[key] = value
                        if value:
                            self.event.set()
                        return True
                def remote(mode, number, campaign_id, lock=lock, calls=calls, fail=fail, cancelled=cancelled):
                    with lock:
                        calls.append((mode, number))
                    if fail and number != 2:
                        cancelled.wait(timeout=5)
                    stop = {"kind": "token_mismatch" if number == 2 else "campaign_cancelled", "detail": {}} if fail else None
                    return {"mode": mode, "pair": number, "stop": stop,
                            "gpu_states": [{"uuid": f"pair-{mode}-{number}"}], "workload": {"mode": mode, "pair": number},
                            "stdout": "fixture retained", "model_files_before": {"weights": "a" * 64},
                            "model_files_after": {"weights": "a" * 64}, "dependency_versions": ["torch==fixture"]}
                namespace.update(preflight=lambda _: {"source_fingerprint": "fixture"}, verify_local_source=lambda _: None,
                                 verify_holdout_prerequisite=lambda _: {"independently_revalidated": True},
                                 verify_pair=lambda *args: None, pair=types.SimpleNamespace(remote=remote),
                                 modal=types.SimpleNamespace(Dict=types.SimpleNamespace(from_name=lambda *args, **kwargs: Store())))
                with contextlib.redirect_stdout(io.StringIO()):
                    if fail:
                        with self.assertRaises(SystemExit):
                            namespace["run"](str(destination), "fresh-summary")
                    else:
                        namespace["run"](str(destination), "fresh-summary")
                self.assertEqual(len(calls), 4 if fail else 20)
                self.assertEqual(len(list((destination / "raw").rglob("pair-*.json"))), len(calls))
                self.assertEqual((destination / "completion.json").exists(), not fail)
                if fail:
                    self.assertTrue(cancelled.is_set())
                    self.assertTrue(all(mode == "resource_normalized" for mode, _ in calls))
                else:
                    self.assertEqual(set(calls), {(mode, number) for mode in ("resource_normalized", "complete_system") for number in range(1, 11)})
                    completion = json.loads((destination / "completion.json").read_text())
                    self.assertEqual(len(completion["gpu_uuids"]), 20)
                    self.assertEqual(completion["max_concurrent_pairs"], 4)

    def test_missing_or_incomplete_holdout_blocks_before_dispatch(self):
        namespace = functions(REPO)
        with self.assertRaisesRegex(SystemExit, "holdout-summary"):
            namespace["verify_holdout_prerequisite"]("")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(json.dumps({"complete": False, "passes": True, "validation_evidence": {}}))
            with self.assertRaisesRegex(SystemExit, "complete successful"):
                namespace["verify_holdout_prerequisite"](str(path))

    def test_holdout_prerequisite_binds_all_forty_raw_files_and_recomputes_verdict(self):
        from experiments.fix_holdout import MODES, NAMESPACE, digest
        from experiments.sentinel_pilot import SENTINEL_CELLS
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workloads, serial = {}, 0
            for mode in MODES:
                batches = []
                for cell in SENTINEL_CELLS:
                    for batch_index in range({1: 3, 8: 5, 32: 2}[cell.concurrency]):
                        ids = []
                        for _ in range(cell.concurrency):
                            serial += 1
                            ids.append([serial] * cell.input_tokens)
                        batches.append({"cell": cell.name, "batch_index": batch_index,
                            "input_tokens": cell.input_tokens, "output_tokens": cell.output_tokens,
                            "concurrency": cell.concurrency, "input_token_ids": ids,
                            "prompt_hashes": [digest(prompt) for prompt in ids]})
                workloads[mode] = batches
            environment = {"model": {"revision": "fixture"}, "model_files_sha256": {"weights": "a" * 64},
                           "dependencies": ["fixture==1"]}
            bounds = {"namespace": NAMESPACE, "calibration_passes": True, "computational_source_sha256": "source",
                      "runtime_environment_sha256": digest(environment),
                      "bounds": {mode: {engine: 0.1 for engine in ("custom", "vllm")} for mode in MODES}}
            manifest = {"split": "holdout", "namespace": NAMESPACE, "workloads": workloads,
                        "workloads_sha256": digest(workloads), "bounds_sha256": digest(bounds),
                        "excluded_prompt_hashes": [], "computational_source": {"sha256": "source"}}
            (root / "manifest.json").write_text(json.dumps(manifest))
            (root / "bounds.json").write_text(json.dumps(bounds))
            index = {}
            for mode in MODES:
                for engine in ("custom", "vllm"):
                    for number in range(10):
                        path = root / mode / engine / f"batch-{number:02d}.json"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        payload = {"process": {"returncode": 0}, "capture": {"synthetic": True},
                                   "environment": {**environment, "computational_source": {"sha256": "source"},
                                       "gpu_before": {"uuid": f"{mode}-{number}"}, "gpu_after": {"uuid": f"{mode}-{number}"}}}
                        path.write_text(json.dumps(payload))
                        index[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            summary = {"complete": True, "passes": True, "split": "holdout", "namespace": NAMESPACE,
                "validation_evidence": {"split": "holdout", "namespace": NAMESPACE,
                    "manifest_path": "manifest.json", "bounds_path": "bounds.json",
                    "manifest_sha256": digest(manifest), "bounds_sha256": digest(bounds),
                    "computational_source_sha256": "source", "artifact_index": index}}
            summary_path = root / "summary.json"
            summary_path.write_text(json.dumps(summary))
            namespace = functions(REPO)
            namespace["verify_frozen_lineage"] = lambda *args: {"unit": "separately tested with real Git"}
            check = namespace["verify_holdout_prerequisite"]
            with mock.patch("experiments.fix_holdout.computational_source", return_value={"sha256": "source"}), mock.patch(
                    "experiments.fix_holdout.summarize", return_value={"complete": True, "passes": True}) as summarize:
                accepted = check(str(summary_path))
                self.assertEqual(accepted["raw_capture_count"], 40)
                self.assertEqual(accepted["fresh_unique_requests"], 214)
                self.assertEqual(len(summarize.call_args.args[1]), 40)
                summarize.return_value = {"complete": True, "passes": False}
                with self.assertRaisesRegex(SystemExit, "raw revalidation"):
                    check(str(summary_path))
                summarize.return_value = {"complete": True, "passes": True}
                path = root / next(iter(index))
                path.write_text(path.read_text() + " ")
                with self.assertRaisesRegex(SystemExit, "checksum mismatch"):
                    check(str(summary_path))

class FrozenInputLineageTests(unittest.TestCase):
    def test_original_committed_inputs_required_even_when_external_copies_are_consistent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def git(*args):
                return subprocess.check_output(['git', *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
            git('init')
            (root/'CORRECTNESS_FIX_VALIDATION.md').write_text('**Status: FROZEN**')
            manifest, bounds = {'split': 'holdout', 'id': 'original'}, {'bounds': {'fixed': 0.5}}
            for name, value in [('sealed-inputs.json', manifest), ('frozen-limits.json', bounds)]:
                (root/name).write_text(json.dumps(value))
            git('add', '.')
            git('-c', 'user.name=Unit Test', '-c', 'user.email=unit@example.invalid', 'commit', '-m', 'freeze')
            lineage = {'execution_commit': git('rev-parse', 'HEAD'), 'manifest_repo_path': 'sealed-inputs.json',
                       'bounds_repo_path': 'frozen-limits.json'}
            check = functions(root)['verify_frozen_lineage']
            with mock.patch.dict('sys.modules', {'fix_holdout_app': types.SimpleNamespace(frozen_bounds=lambda path: bounds)}):
                self.assertEqual(check({'frozen_input_lineage': lineage}, manifest, bounds), lineage)
                for key, original in [('manifest', manifest), ('bounds', bounds)]:
                    changed = {**original, 'altered': True}
                    with self.subTest(copied=key), self.assertRaisesRegex(SystemExit, 'lineage invalid'):
                        check({'frozen_input_lineage': lineage}, changed if key == 'manifest' else manifest,
                              changed if key == 'bounds' else bounds)
                for name, original in [('sealed-inputs.json', manifest), ('frozen-limits.json', bounds)]:
                    (root/name).write_text(json.dumps({**original, 'uncommitted': True}))
                    with self.subTest(uncommitted=name), self.assertRaisesRegex(SystemExit, 'lineage invalid'):
                        check({'frozen_input_lineage': lineage}, manifest, bounds)
                    (root/name).write_text(json.dumps(original))
                with self.assertRaisesRegex(SystemExit, 'missing original'):
                    check({}, manifest, bounds)
                # A later documentation-only commit preserves the same source/input lineage.
                (root/'notes.md').write_text('later audit')
                git('add', '.')
                git('-c', 'user.name=Unit Test', '-c', 'user.email=unit@example.invalid', 'commit', '-m', 'notes')
                self.assertEqual(check({'frozen_input_lineage': lineage}, manifest, bounds), lineage)
