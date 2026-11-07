"""Pure frozen-evidence guards; no Modal construction, prompts or GPU execution."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.fix_holdout import MODES, NAMESPACE, digest

REPO = Path(__file__).resolve().parents[1]


class TestFrozenFixBounds(unittest.TestCase):
    def test_bounds_require_exact_complete_committed_calibration_and_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bounds = {mode: {engine: 0.125 for engine in ("custom", "vllm")} for mode in MODES}
            environment = {
                "model": {"revision": "fixture"},
                "model_files_sha256": {"weights": "fixture"},
                "dependencies": {"torch": "fixture"},
            }
            summary = {
                "complete": True,
                "passes": True,
                "calibration_maxima": bounds,
                "provenance": {str(index): environment for index in range(40)},
            }
            path = root / "summary.json"
            path.write_text(json.dumps(summary))
            record = {
                "namespace": NAMESPACE,
                "calibration_passes": True,
                "bounds": bounds,
                "computational_source_sha256": "source",
                "calibration_summary_path": path.name,
                "calibration_summary_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "runtime_environment_sha256": digest(environment),
            }
            nodes = [
                node
                for node in ast.parse((REPO / "fix_holdout_app.py").read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "frozen_bounds"
            ]
            scope = {
                "REPO": root,
                "subprocess": subprocess,
                "math": math,
                "hashlib": hashlib,
                "committed": lambda name: summary if name == path.name else record,
            }
            exec(compile(ast.Module(body=nodes, type_ignores=[]), "frozen_bounds", "exec"), scope)

            def git(args, **kwargs):
                return "" if args[1] == "status" else "**Status: FROZEN**"

            with (
                mock.patch("subprocess.check_output", side_effect=git),
                mock.patch(
                    "experiments.fix_holdout.computational_source",
                    return_value={"sha256": "source"},
                ),
            ):
                self.assertIs(scope["frozen_bounds"]("bounds.json"), record)
                for field, invalid in [
                    ("computational_source_sha256", "different"),
                    ("calibration_summary_sha256", "different"),
                    ("runtime_environment_sha256", "different"),
                ]:
                    with self.subTest(field=field):
                        old = record[field]
                        record[field] = invalid
                        with self.assertRaises(ValueError):
                            scope["frozen_bounds"]("bounds.json")
                        record[field] = old
                summary["passes"] = False
                with self.assertRaises(ValueError):
                    scope["frozen_bounds"]("bounds.json")
                summary["passes"] = True
                summary["provenance"].pop("39")
                with self.assertRaises(ValueError):
                    scope["frozen_bounds"]("bounds.json")
