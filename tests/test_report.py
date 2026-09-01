"""experiments/report.py must produce byte-identical evidence regardless of
Python's string-hash randomization. A prior version derived plot series order
from a set, so re-running the same raw data could silently reorder the
custom/vLLM bars depending on PYTHONHASHSEED.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import experiments.report as report

ROOT = Path(__file__).parents[1]
REPORT_SCRIPT = ROOT / "experiments" / "report.py"
GENERATED_FILES = (
    ROOT / "experiments/plots/throughput.svg",
    ROOT / "experiments/plots/ablation-throughput.svg",
    ROOT / "experiments/summaries/results.csv",
    ROOT / "experiments/summaries/variation.csv",
    ROOT / "experiments/summaries/findings.md",
)
HASH_SEEDS = ("0", "1", "2")


class TestReportIsHashSeedInvariant(unittest.TestCase):
    def test_regenerated_artifacts_are_byte_identical_across_hash_seeds(self) -> None:
        snapshots = {}
        for seed in HASH_SEEDS:
            subprocess.run(
                [sys.executable, str(REPORT_SCRIPT)],
                check=True,
                cwd=ROOT,
                env=dict(os.environ, PYTHONHASHSEED=seed),
            )
            snapshots[seed] = {path.name: path.read_bytes() for path in GENERATED_FILES}

        baseline_seed = HASH_SEEDS[0]
        for seed in HASH_SEEDS[1:]:
            self.assertEqual(
                snapshots[baseline_seed],
                snapshots[seed],
                f"experiments/report.py output differs between PYTHONHASHSEED="
                f"{baseline_seed} and PYTHONHASHSEED={seed}; display order must not "
                "be derived from set/dict iteration order.",
            )


class TestRawInvalidIsExcluded(unittest.TestCase):
    """Runs stopped or otherwise invalidated per the pilot's stop rules must be
    retained on disk but never enter the aggregate summaries."""

    def test_raw_invalid_directory_is_not_picked_up_by_raw_results(self) -> None:
        with _temporary_raw_tree() as raw_root:
            (raw_root / "custom-server").mkdir(parents=True)
            (raw_root / "vllm").mkdir(parents=True)
            (raw_root / "invalid").mkdir(parents=True)
            valid = raw_root / "custom-server" / "complete-restart-1.json"
            valid.write_text(json.dumps(_golden_result("custom-server", "complete")))
            invalid = raw_root / "invalid" / "complete-restart-99.json"
            invalid.write_text(json.dumps(_golden_result("custom-server", "complete")))
            nested_invalid = raw_root / "custom-server" / "invalid"
            nested_invalid.mkdir()
            (nested_invalid / "complete-restart-98.json").write_text(
                json.dumps(_golden_result("custom-server", "complete"))
            )

            with mock.patch.object(report, "ROOT", raw_root.parent):
                results = report._raw_results()

        paths = [result["_path"] for result in results]
        self.assertEqual(paths, ["raw/custom-server/complete-restart-1.json"])
        self.assertTrue(all("invalid" not in path for path in paths))


def _golden_result(
    implementation: str,
    variant: str,
    *,
    ttft: float = 1.0,
    itl: float = 1.0,
    e2e: float = 1.0,
    tok_s: float = 1.0,
    req_s: float = 1.0,
    peak: int = 1,
    records: list[dict] | None = None,
) -> dict:
    records = records if records is not None else [{"error": None, "timeout": False}]
    return {
        "_path": f"raw/{implementation}/{variant}-golden-fixture.json",
        "environment": {"implementation": implementation},
        "variant": variant,
        "cells": [
            {
                "cell": {"name": "in128-out128-c1"},
                "summary": {
                    "ttft_ms": ttft,
                    "itl_ms": itl,
                    "total_request_latency_ms": e2e,
                    "output_tokens_per_second": tok_s,
                    "requests_per_second": req_s,
                    "peak_gpu_memory_bytes": peak,
                    "wall_seconds": 1.0,
                },
                "records": records,
            }
        ],
    }


class TestRawToSummaryGoldenFixture(unittest.TestCase):
    """Hand-computed expected medians for two synthetic restarts, guarding the
    raw-record -> results.csv arithmetic in report._write_summaries."""

    def test_medians_and_failure_counts_match_hand_computed_values(self) -> None:
        restart_one = _golden_result(
            "custom-server",
            "complete",
            ttft=10.0,
            itl=2.0,
            e2e=100.0,
            tok_s=50.0,
            req_s=5.0,
            peak=1000,
            records=[
                {"error": None, "timeout": False},
                {"error": None, "timeout": False},
                {"error": "timeout", "timeout": True},
            ],
        )
        restart_two = _golden_result(
            "custom-server",
            "complete",
            ttft=30.0,
            itl=4.0,
            e2e=300.0,
            tok_s=70.0,
            req_s=7.0,
            peak=3000,
            records=[
                {"error": None, "timeout": False},
                {"error": None, "timeout": False},
            ],
        )

        with _temporary_summaries_dir() as summaries_dir, \
             mock.patch.object(report, "ROOT", summaries_dir.parent):
            report._write_summaries([restart_one, restart_two])
            rows = report._rows()

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["implementation"], "custom-server")
        self.assertEqual(row["variant"], "complete")
        self.assertEqual(row["cell"], "in128-out128-c1")
        self.assertEqual(row["restarts"], "2")
        self.assertEqual(float(row["ttft_ms"]), 20.0)
        self.assertEqual(float(row["itl_ms"]), 3.0)
        self.assertEqual(float(row["total_request_latency_ms"]), 200.0)
        self.assertEqual(float(row["output_tokens_per_second"]), 60.0)
        self.assertEqual(float(row["requests_per_second"]), 6.0)
        self.assertEqual(float(row["peak_gpu_memory_bytes"]), 2000.0)
        # restart one: 1 failure (the timeout) out of 3 records; restart two: 0.
        self.assertEqual(float(row["failures"]), 0.5)
        self.assertEqual(float(row["timeouts"]), 0.5)


@contextmanager
def _temporary_raw_tree():
    with TemporaryDirectory() as tmp:
        root = Path(tmp) / "raw"
        root.mkdir()
        yield root


@contextmanager
def _temporary_summaries_dir():
    with TemporaryDirectory() as tmp:
        root = Path(tmp) / "summaries"
        root.mkdir()
        (Path(tmp) / "raw").mkdir()
        yield root


if __name__ == "__main__":
    unittest.main()
