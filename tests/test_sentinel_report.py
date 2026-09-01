"""experiments/sentinel_report.py: raw-record -> summary/plot arithmetic and
reproducibility, exercised against synthetic pair data (no GPU needed).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).parents[1]
CELLS = ("in128-out128-c1", "in512-out128-c8", "in1024-out256-c32")


def _child_result(implementation: str, throughputs: dict[str, float], phases: tuple[str, ...]) -> dict:
    cells = {}
    for cell in CELLS:
        cells[cell] = {
            "phases": {
                phase: {
                    "records": [
                        {"request_index": 0, "output_token_ids": [1, 2, 3], "error": None, "timeout": False}
                    ],
                    "summary": {
                        "output_tokens_per_second": throughputs[cell],
                        "failures": 0,
                        "timeouts": 0,
                    },
                }
                for phase in phases
            }
        }
    return {"cells": cells, "environment": {"implementation": implementation}}


def _write_mode(
    root: Path, mode: str, subdir: str, phases: tuple[str, ...], custom_tps: float, vllm_tps: float,
    pairs: int = 10, stop_pair: int | None = None,
) -> None:
    directory = root / "raw" / subdir
    directory.mkdir(parents=True, exist_ok=True)
    for pair in range(1, pairs + 1):
        custom = _child_result(
            "custom-server", {cell: custom_tps for cell in CELLS}, phases
        )
        vllm = _child_result("vllm", {cell: vllm_tps for cell in CELLS}, phases)
        payload = {
            "mode": mode,
            "pair": pair,
            "order": "odd" if pair % 2 else "even",
            "workload_hash": "fixture",
            "stop": (
                {"kind": "token_mismatch", "detail": {}} if pair == stop_pair else None
            ),
            "children": [
                {"implementation": "custom", "position": 1, "result": custom},
                {"implementation": "vllm", "position": 2, "result": vllm},
            ],
            "gpu_states": [],
        }
        (directory / f"pair-{pair:02d}.json").write_text(json.dumps(payload))


def _run_report(root: Path, env_extra: dict[str, str] | None = None) -> None:
    subprocess.run(
        [sys.executable, "-m", "experiments.sentinel_report", str(root)],
        check=True,
        cwd=REPO_ROOT,
        env=dict(os.environ, **(env_extra or {})),
    )


class TestGoldenRatioArithmetic(unittest.TestCase):
    def test_geometric_mean_matches_hand_computed_value(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_mode(root, "resource_normalized", "resource-normalized", ("unique",), 60.0, 100.0)
            _run_report(root)
            rows = list(
                csv_rows(root / "summaries/paired-results.csv")
            )
        self.assertEqual(len(rows), 3)  # one row per sentinel cell
        for row in rows:
            self.assertEqual(row["mode"], "resource_normalized")
            self.assertEqual(row["n"], "10")
            self.assertAlmostEqual(float(row["geometric_mean_ratio"]), 0.6, places=9)
            self.assertAlmostEqual(float(row["median_ratio"]), 0.6, places=9)
            self.assertAlmostEqual(float(row["ci95_low"]), 0.6, places=6)
            self.assertAlmostEqual(float(row["ci95_high"]), 0.6, places=6)
            self.assertEqual(row["t_critical"], "2.262")
            self.assertEqual(row["degrees_of_freedom"], "9")


def csv_rows(path: Path):
    import csv

    with path.open(newline="") as handle:
        yield from csv.DictReader(handle)


class TestStoppedPilotExclusion(unittest.TestCase):
    def test_a_stopped_pair_produces_no_claim_and_is_listed_in_exclusions(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_mode(
                root, "resource_normalized", "resource-normalized", ("unique",), 60.0, 100.0,
                stop_pair=5,
            )
            _run_report(root)
            findings = (root / "summaries/findings.md").read_text()
            exclusions = (root / "summaries/exclusions.md").read_text()
            correctness = json.loads((root / "summaries/correctness.json").read_text())
            rows = list(csv_rows(root / "summaries/paired-results.csv"))

        self.assertIn("STOPPED", findings)
        self.assertIn("pair 05", exclusions)
        self.assertTrue(correctness["resource_normalized"]["stopped"])
        self.assertEqual(rows, [])  # no performance claim from a stopped pilot


class TestIncompletePilotDeferral(unittest.TestCase):
    def test_fewer_than_ten_pairs_with_no_stop_defers_the_claim(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_mode(
                root, "resource_normalized", "resource-normalized", ("unique",), 60.0, 100.0,
                pairs=6,
            )
            _run_report(root)
            findings = (root / "summaries/findings.md").read_text()
            rows = list(csv_rows(root / "summaries/paired-results.csv"))

        self.assertIn("Incomplete: 6/10", findings)
        self.assertEqual(rows, [])


class TestCompleteSystemColdWarmSeparation(unittest.TestCase):
    def test_cold_and_warm_are_reported_as_separate_rows(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_mode(
                root, "complete_system", "complete-policy", ("cold", "warm"), 70.0, 100.0
            )
            _run_report(root)
            rows = list(csv_rows(root / "summaries/paired-results.csv"))

        phases = {row["phase"] for row in rows}
        self.assertEqual(phases, {"cold", "warm"})
        self.assertEqual(len(rows), 6)  # 3 cells x 2 phases


class TestReportIsHashSeedInvariant(unittest.TestCase):
    def test_regenerated_summaries_and_plots_are_byte_identical_across_hash_seeds(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_mode(root, "resource_normalized", "resource-normalized", ("unique",), 55.0, 91.0)
            snapshots = {}
            for seed in ("0", "1", "2"):
                _run_report(root, env_extra={"PYTHONHASHSEED": seed})
                snapshots[seed] = {
                    path.relative_to(root).as_posix(): path.read_bytes()
                    for path in sorted(root.rglob("*"))
                    if path.is_file() and path.name != "artifact-manifest.json"
                }
        baseline = snapshots["0"]
        for seed in ("1", "2"):
            self.assertEqual(baseline, snapshots[seed], f"differs at PYTHONHASHSEED={seed}")


if __name__ == "__main__":
    unittest.main()
