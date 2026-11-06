"""Pure synthetic evidence checks; these do not validate any GPU implementation."""

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

from experiments.fix_holdout import (
    MODES,
    capture_vllm,
    compare_records,
    computational_source,
    digest,
    materialize,
    summarize,
    validate_capture,
    validate_graph_evidence,
    validate_manifest,
    validate_pack,
    validate_record,
)
from experiments.protocol_v2 import V2_BATCHES_PER_CELL, V2_CELLS
from tests.test_combined_policy_wiring import status as policy_status


def record(ids=(3, 4), steps=2):
    return {
        "index": 0,
        "request_id": "actual-0",
        "input_token_ids": list(ids),
        "input_verified": True,
        "initial_computed_tokens": 0,
        "output_tokens": [0] * steps,
        "steps": [
            {"all_finite": True, "argmax": 0, "top_k": [(i, -float(i)) for i in range(20)]}
            for _ in range(steps)
        ],
    }


def capture():
    return {
        "records": [record()],
        "precision": policy_status()["precision"],
        "numerical_policy": {"before": policy_status(), "after": policy_status()},
        "identity": {"failures": [], "pack_checks": 2},
        "graph_counters": {"unresolved": False, "before": {"cuda_graph_captures": 2, "cuda_graph_replays": 5},
                           "after": {"cuda_graph_captures": 2, "cuda_graph_replays": 5},
                           "cuda_graph_captures": 0, "cuda_graph_replays": 0},
    }


def fixture_manifest():
    batches = []
    for cell in V2_CELLS:
        for index in range(V2_BATCHES_PER_CELL[cell.concurrency]):
            ids = [[index] * cell.input_tokens for _ in range(cell.concurrency)]
            batches.append(
                {
                    "cell": cell.name,
                    "batch_index": index,
                    "input_tokens": cell.input_tokens,
                    "output_tokens": cell.output_tokens,
                    "concurrency": cell.concurrency,
                    "input_token_ids": ids,
                    "prompt_hashes": [digest(item) for item in ids],
                }
            )
    return {"split": "calibration", "workloads": {mode: deepcopy(batches) for mode in MODES}}


class Array:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


def packed_fixture():
    request = NS(request_id="r", prompt="label", prompt_token_ids=[3, 4], generated_token_ids=[])
    plan = NS(items=[NS(request=request, token_ids=(3, 4), start_pos=0, end_pos=2, sample=True)])
    cache = NS(
        allocator=NS(_owner={0: "r"}, _free=[1], total_blocks=2),
        _tables={"r": [0]},
        _prefixes={},
        block_size=16,
        block_table=lambda _: [0],
    )
    context = NS(
        block_tables=Array([[0]]),
        positions=Array([0, 1]),
        slot_mapping=Array([0, 1]),
        request_ids=("r",),
        query_ranges=((0, 2),),
    )
    packed = NS(
        context=context,
        input_ids=Array([3, 4]),
        sampled_request_ids=("r",),
        logit_indices=Array([1]),
    )
    return (plan, packed, cache)


class TestFixHoldout(unittest.TestCase):
    def test_capture_requires_actual_identity_precision_and_full_window(self):
        validate_capture(capture(), [[3, 4]], 2, cold=True, mode="resource_normalized")
        for field, value in [("identity", {}), ("precision", []), ("records", []), ("numerical_policy", {})]:
            bad = capture()
            bad[field] = value
            with self.assertRaises(ValueError):
                validate_capture(bad, [[3, 4]], 2, cold=True, mode="resource_normalized")
        bad = record()
        bad["steps"].pop()
        with self.assertRaises(ValueError):
            validate_record(bad, [3, 4], 2, cold=True)
        bad = record()
        bad["steps"][0]["all_finite"] = False
        with self.assertRaises(ValueError):
            validate_record(bad, [3, 4], 2, cold=True)

    def test_postphase_graph_rpc_failure_retains_completed_records(self):
        graph_calls = 0
        def rpc(name, **kwargs):
            nonlocal graph_calls
            if name == "graph_status":
                graph_calls += 1
                if graph_calls == 2:
                    raise RuntimeError("postphase RPC unavailable")
                return [{"observer_active": True, "capture_count": 0, "replay_count": 0}]
            if name == "holdout_evidence":
                return [{"requests": {"r-0": {"input_verified": True}}, "failures": [], "pack_checks": 1}]
        engine = NS(collective_rpc=rpc, add_request=mock.Mock(),
                    has_unfinished_requests=mock.Mock(side_effect=[True, False]),
                    step=lambda: [NS(finished=True, request_id="r-0", outputs=[NS(token_ids=[9, 8])])])
        modules = {"vllm": NS(SamplingParams=lambda **kwargs: kwargs),
                   "cloud_engine.precision": NS(require_vllm_precision=lambda _: [policy_status()["precision"]],
                                              numerical_policy_snapshot=lambda *args: policy_status("vllm"))}
        with mock.patch.dict("sys.modules", modules):
            result = capture_vllm(engine, [[3, 4]], 2, "r")
        self.assertEqual(result["records"][0]["output_tokens"], [9, 8])
        self.assertTrue(result["records"][0]["input_verified"])
        self.assertTrue(result["graph_counters"]["unresolved"])
        self.assertEqual(result["identity"]["failures"], [])
        with self.assertRaises(ValueError):
            validate_capture(result, [[3, 4]], 2, cold=True, mode="resource_normalized")

    def test_phase_graph_deltas_require_actual_replay_and_exact_counters(self):
        eager = capture()["graph_counters"]
        validate_graph_evidence(eager, "resource_normalized")
        with self.assertRaises(ValueError):
            validate_graph_evidence(eager, "complete_system")
        replay = deepcopy(eager)
        replay["after"]["cuda_graph_replays"] += 3
        replay["cuda_graph_replays"] = 3
        validate_graph_evidence(replay, "complete_system")
        with self.assertRaises(ValueError):
            validate_graph_evidence(replay, "resource_normalized")
        for bad in ({}, {**replay, "cuda_graph_replays": 2},
                    {**replay, "cuda_graph_replays": True}, {**replay, "unresolved": True}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_graph_evidence(bad, "complete_system")
        worker = {"unresolved": False,
                  "workers_before": [{"observer_active": True, "capture_count": 5, "replay_count": 7}],
                  "workers_after": [{"observer_active": True, "capture_count": 5, "replay_count": 9}],
                  "worker_deltas": [{"capture_count": 0, "replay_count": 2}]}
        validate_graph_evidence(worker, "complete_system")
        worker["workers_before"][0]["observer_active"] = False
        with self.assertRaises(ValueError):
            validate_graph_evidence(worker, "complete_system")

    def test_comparison_stops_at_first_different_history_and_does_not_widen_bound(self):
        (left, right) = (record(), record())
        right["output_tokens"][0] = 1
        result = compare_records(left, right, bound=0)
        self.assertTrue(not result["exact_tokens"] and result["failures"][0]["position"] == 0)
        right = record()
        right["steps"][0]["top_k"][1] = (1, -1.125)
        self.assertTrue(compare_records(left, right, bound=0.125)["failures"] == [])
        self.assertTrue(
            compare_records(left, right, bound=0.124)["failures"][0]["reason"]
            == "batch_solo_logprob_drift"
        )

    def test_manifest_requires_all_cells_shapes_and_windows(self):
        manifest = fixture_manifest()
        validate_manifest(manifest)
        self.assertTrue(
            sum(batch["concurrency"] for batch in manifest["workloads"][MODES[0]]) == 107
        )
        manifest["workloads"][MODES[0]].pop()
        with self.assertRaisesRegex(ValueError, "incomplete workload cells"):
            validate_manifest(manifest)
        manifest = fixture_manifest()
        manifest["workloads"][MODES[1]][0]["output_tokens"] = 8
        with self.assertRaisesRegex(ValueError, "incorrect workload shape"):
            validate_manifest(manifest)
        self.assertTrue(summarize(fixture_manifest(), {})["passes"] is False)

    def test_fresh_materialization_is_blocked_before_freeze_without_touching_tokenizer(self):
        with self.assertRaisesRegex(ValueError, "frozen protocol"):
            materialize(None, "holdout", MODES[0])

    def test_actual_pack_rejects_foreign_blocks_wrong_inputs_and_alias_writes(self):
        args = packed_fixture()
        self.assertTrue(validate_pack(*args, {"label": [3, 4]})["checked_slots"] == 2)
        args[2].allocator._owner[0] = "other"
        with self.assertRaisesRegex(ValueError, "foreign block"):
            validate_pack(*args, {"label": [3, 4]})
        for field, values in [("input_ids", [3, 9]), ("slot_mapping", [0, 0])]:
            (plan, packed, cache) = packed_fixture()
            setattr(packed if field == "input_ids" else packed.context, field, Array(values))
            with self.assertRaisesRegex(ValueError, "attribution mismatch"):
                validate_pack(plan, packed, cache, {"label": [3, 4]})

    def test_computational_hash_binds_code_not_docs(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tmp_path = Path(temporary.name)
        repo = Path(__file__).resolve().parents[1]
        source = computational_source(repo)
        self.assertTrue("fix_holdout_app.py" in source["files"])
        self.assertTrue("src/cloud_engine/precision.py" in source["files"])
        self.assertTrue(not any(name.endswith(".md") for name in source["files"]))
        for name in source["files"]:
            target = tmp_path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((repo / name).read_bytes())
        self.assertTrue(computational_source(tmp_path) == source)
        (tmp_path / "CORRECTNESS_FIX_VALIDATION.md").write_text("new freeze commit")
        self.assertTrue(computational_source(tmp_path) == source)
        (tmp_path / "src/cloud_engine/precision.py").write_text("changed")
        self.assertTrue(computational_source(tmp_path)["sha256"] != source["sha256"])
