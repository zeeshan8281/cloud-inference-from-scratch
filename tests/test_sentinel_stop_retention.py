"""CPU-only orchestration tests; synthetic phases are not GPU validation."""
from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

from experiments import sentinel_pilot as pilot
from tests.test_combined_policy_wiring import status as policy_status


class StopRetentionTests(unittest.TestCase):
    def test_late_stop_preserves_completed_cells_and_current_evidence(self):
        for failure in ("warmup", "warm-measured"):
            engine = SimpleNamespace(tokenizer=object(), start=mock.AsyncMock(), close=mock.AsyncMock())
            config = SimpleNamespace(cuda_graph_decode=True, prefix_cache_max_blocks=1,
                                     kv_cache_bytes=16, max_active_sequences=32, max_batched_tokens=4096, block_size=16)
            request = [{"input_tokens": 16}]
            workload = {"workload_hash": "fixture", "cells": {
                name: {"warmup": [request]*3, "cold": request, "warm": request} for name in ("first", "second")}}
            visits = {}

            async def phase(implementation, engine_, batch, run_id, visits=visits, failure=failure):
                visits[run_id] = visits.get(run_id, 0) + 1
                if run_id == "second-" + failure and (failure != "warmup" or visits[run_id] == 2):
                    raise pilot.StopPilot("fixture_stop", {"phase_result": {"partial_records": ["failed-phase"]}})
                return {"records": [run_id], "graph_counters": {"unit_phase": run_id},
                        "numerical_policy": {"before": policy_status(), "after": policy_status()}}

            with self.subTest(failure=failure), mock.patch.object(pilot, "_build_custom_engine", return_value=(config, engine)), \
                    mock.patch("cloud_engine.precision.precision_status", return_value={"allow_fp16_reduced_precision_reduction": False}), \
                    mock.patch("experiments.controlled.environment", return_value={"source": "fixture"}), \
                    mock.patch.object(pilot, "_run_phase", side_effect=phase), \
                    mock.patch.object(pilot, "_phase_throughput", return_value=100), \
                    mock.patch.object(pilot, "_cache_counters", return_value={}), \
                    mock.patch.object(pilot, "_graph_counters", return_value={}), \
                    mock.patch.object(pilot, "_summarize_phase", return_value={}), \
                    self.assertRaises(pilot.StopPilot) as caught:
                asyncio.run(pilot.run_child("custom", "complete_system", "fixture", {}, workload, None))
            self.assertEqual(caught.exception.kind, "fixture_stop")
            self.assertEqual(caught.exception.detail["phase_result"]["partial_records"], ["failed-phase"])
            retained = caught.exception.detail["retained_execution"]
            self.assertEqual(set(retained["completed_cells"]), {"first"})
            warm = retained["completed_cells"]["first"]["phases"]["warm"]
            self.assertEqual(warm["graph_counters"], {"unit_phase": "first-warm-measured"})
            self.assertEqual(warm["prime_graph_counters"], {"unit_phase": "first-warm-prime"})
            self.assertEqual(warm["numerical_policy"]["after"], policy_status())
            self.assertEqual(warm["prime_numerical_policy"]["after"], policy_status())
            self.assertEqual(retained["current_cell"], "second")
            self.assertEqual(retained["workload_hash"], "fixture")
            self.assertEqual(len(retained["warmup_records"]), 1 if failure == "warmup" else 3)
            if failure == "warmup":
                self.assertEqual(retained["completed_phases"], {})
                self.assertEqual(retained["completed_phase_results"], {})
            else:
                self.assertEqual(set(retained["completed_phases"]), {"cold"})
                self.assertEqual(retained["completed_phase_results"]["warm_prime"]["records"], ["second-warm-prime"])
            engine.close.assert_awaited_once()
