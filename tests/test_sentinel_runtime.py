"""Real CPU subprocess/error-retention tests; synthetic payloads are not GPU evidence."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments import sentinel_runtime as runtime
from experiments.sentinel_pilot import PROTOCOL_VERSION, SENTINEL_CELLS, StopPilot


def result_fixture():
    cells = {}
    for cell in SENTINEL_CELLS:
        records = [{"request_index": i, "output_token_ids": [i+1]*cell.output_tokens} for i in reversed(range(cell.concurrency))]
        cells[cell.name] = {"warmup_throughputs": [100, 101, 100],
                            "phases": {"unique": {"records": records, "graph_counters": {"unresolved": False, "path": "eager_fallback"}}}}
    return {"protocol_version": PROTOCOL_VERSION, "mode": "resource_normalized", "workload_hash": "fixture",
            "environment": {"model": "fixture", "model_revision": "pinned", "dtype": "float16", "repository_commit": "frozen", "implementation": "custom-server"},
            "cells": cells}


class SentinelRuntimeTests(unittest.TestCase):
    def child(self, script, timeout=10):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = root/"child.py"
            code.write_text(script)
            workload = root/"workload.json"
            workload.write_text("{}")
            with mock.patch.object(runtime, "PILOT_SCRIPT", code), mock.patch.object(runtime, "CHILD_TIMEOUT_SECONDS", timeout):
                return runtime.run_child("custom", "resource_normalized", 1, "fixture-model", workload, root/"child")

    def test_completed_child_retains_full_logs_and_orderly_stop(self):
        child = self.child('''import json,sys
from pathlib import Path
print("stdout retained", flush=True)
print("stderr retained", file=sys.stderr, flush=True)
Path(sys.argv[-1]).write_text(json.dumps({"stop":{"kind":"timeout","detail":{"partial_records":[1]}}}))
''')
        self.assertFalse(child["crashed"])
        self.assertEqual(child["stop"]["detail"]["partial_records"], [1])
        self.assertIn("stdout retained", child["stdout"])
        self.assertIn("stderr retained", child["stderr"])

    def test_timeout_and_invalid_json_retain_evidence(self):
        child = self.child('''import sys,time
print("before timeout", flush=True)
print("stderr before timeout", file=sys.stderr, flush=True)
time.sleep(30)
''', timeout=0.5)
        self.assertTrue(child["crashed"])
        self.assertEqual(child["stop"]["kind"], "child_timeout")
        self.assertIn("before timeout", child["stdout"])
        self.assertIn("stderr before timeout", child["stderr"])
        malformed = self.child('''import sys
from pathlib import Path
Path(sys.argv[-1]).write_text("not json")
''')
        self.assertEqual(malformed["stop"]["kind"], "invalid_child_result")

    def test_full_payload_validation_rejects_protocol_gaps(self):
        good = result_fixture()
        workload = {"mode": "resource_normalized", "workload_hash": "fixture"}
        model = {"id": "fixture", "revision": "pinned", "dtype": "float16"}
        runtime.validate_child_result(good, workload, model, "frozen", "custom")
        self.assertEqual(runtime.phase_outputs(good)[SENTINEL_CELLS[1].name]["unique"][0][0], 1)
        for fault in ("hash", "source", "graph", "warmup", "missing_cell"):
            result = copy.deepcopy(good)
            first = result["cells"][SENTINEL_CELLS[0].name]
            if fault == "hash":
                result["workload_hash"] = "changed"
            elif fault == "source":
                result["environment"]["repository_commit"] = "other"
            elif fault == "graph":
                first["phases"]["unique"]["graph_counters"]["unresolved"] = True
            elif fault == "warmup":
                first["warmup_throughputs"] = [1, 100, 1]*3 + [100]
            else:
                result["cells"].pop(SENTINEL_CELLS[-1].name)
            with self.subTest(fault=fault), self.assertRaises(StopPilot):
                runtime.validate_child_result(result, workload, model, "frozen", "custom")

    def test_source_hash_mapping_matches_local_repository_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"cloud_engine").mkdir()
            (root/"experiments").mkdir()
            for name in ("engine_config.json", "cloud_engine/model.py", "experiments/sentinel_runtime.py"):
                (root/name).write_text("fixture")
            self.assertEqual(set(runtime.executed_source_hashes(root)), {"engine_config.json", "src/cloud_engine/model.py", "experiments/sentinel_runtime.py"})

    def test_cancellation_retains_completed_phase_journal_and_full_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = root / "child.py"
            code.write_text('''import json,sys,time
from pathlib import Path
print("before cancellation", flush=True)
Path(sys.argv[-1]).with_name("partial-phases.json").write_text(json.dumps({"phases":[{"records":[1]}]}))
time.sleep(30)
''')
            workdir = root / "child"
            with mock.patch.object(runtime, "PILOT_SCRIPT", code):
                child = runtime.run_child("custom", "resource_normalized", 1, "fixture", root / "workload", workdir,
                                          cancelled=lambda: (workdir / "partial-phases.json").exists())
            self.assertEqual(child["stop"]["kind"], "campaign_cancelled")
            self.assertEqual(child["partial_checkpoint"]["phases"][0]["records"], [1])
            self.assertIn("before cancellation", child["stdout"])
            self.assertTrue(child["crashed"])

    def test_child_journal_wraps_after_phase_and_retains_orderly_failure(self):
        import asyncio
        import json

        from experiments import sentinel_pilot
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            async def measured(*args):
                if args[3] == "second":
                    raise StopPilot("fixture_stop", {"retained": [2]})
                return {"records": [1], "wall_seconds": 0.01}
            async def body():
                await sentinel_pilot._run_phase("custom", None, [], "first")
                await sentinel_pilot._run_phase("custom", None, [], "second")
            with mock.patch.object(sentinel_pilot, "_run_phase", measured), mock.patch.object(
                    sentinel_pilot, "main", lambda: asyncio.run(body())), mock.patch.object(runtime.sys, "argv", ["child", str(path)]):
                with self.assertRaises(StopPilot):
                    runtime.journaled_child_main()
                self.assertIs(sentinel_pilot._run_phase, measured)
            journal = json.loads(path.with_name("partial-phases.json").read_text())
            self.assertEqual(journal["phases"][0]["run_id"], "first")
            self.assertEqual(journal["stop"]["kind"], "fixture_stop")
            self.assertFalse(journal["complete"])
