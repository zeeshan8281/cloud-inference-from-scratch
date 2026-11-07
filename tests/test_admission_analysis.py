"""Synthetic integrity tests; these fixtures are not real-GPU evidence."""
import copy
import unittest

from experiments.admission_analysis import SOURCES, analyze, expected_schedules


def fixture(admission="atomic", reorder=False, operator_value=2.0):
    batch = [[100, 101], [200, 201]]
    if reorder:
        batch.reverse()
    index = 0 if reorder else 1
    plans = expected_schedules(batch, admission)
    results = [{"index": i, "request_id": f"diag-{i}", "output_tokens": [ids[0]]*3, "verified": True} for i, ids in enumerate(batch)]
    calls, counts = [], [0, 0]
    for plan in plans:
        rows = []
        for rid in plan:
            i = int(rid[5:])
            token = batch[i][0]
            rows.append({"request_id": rid, "hidden": [1.0, 2.0], "candidate_ids": [token],
                         "native_logits": [1.0], "fp32_projection": [1.0], "permuted_logits": [1.0],
                         "native_argmax": token, "computed_tokens": 0 if not counts[i] else len(batch[i])+counts[i]-1,
                         "output_tokens_before": [token]*counts[i]})
            counts[i] += 1
        calls.append({"shape": [len(rows), 2], "rows": rows})
    events = []
    for i in range(2):
        events.append({"event": "admit", "request_id": f"diag-{i}"})
        if admission == "staged" and i == 0:
            events.append({"event": "step"})
    events.extend([{"event": "step"}]*3)
    target_plan = next(p for p in plans if f"diag-{index}" in p)
    req_ids, lengths = list(target_plan), list(target_plan.values())
    operators = []
    for module, value in (("model.embed_tokens", 1.0), ("model.layers.0", operator_value)):
        operators.append({"module": module, "request_id": f"diag-{index}", "computed_tokens": 0,
                          "output_tokens_before": [], "prompt_length": 2, "target_token_position": 1,
                          "target_input_token": 201, "req_ids": req_ids, "query_lengths": lengths,
                          "target_packed_offset": sum(lengths[:req_ids.index(f'diag-{index}')+1])-1,
                          "total_num_scheduled_tokens": sum(lengths), "precision": {"fixture": True},
                          "vectors": [{"path": "output", "shape": [sum(lengths), 1], "dtype": "fixture", "values": [value]}]})
    return {"exit_code": 0, "gpu": {"uuid": "fixture"}, "source_hashes": dict.fromkeys(SOURCES, "a"*64),
            "probe_sha256": "a"*64, "hook_sha256": "a"*64, "packages": {"fixture": "1"},
            "capture": {"config": {"model": {"id": "fixture", "revision": "fixture"}, "batch_ids": batch,
                                    "target_index": index, "admission": admission, "operator_trace": True},
                        "execution": {"VLLM_ENABLE_V1_MULTIPROCESSING": "0"},
                        "schedules": [{"num_scheduled_tokens": p, "total_num_scheduled_tokens": sum(p.values())} for p in plans],
                        "admission_events": events, "all_results": results, "result": results[index],
                        "projection_calls": calls, "operator_calls": operators}}


class AdmissionAnalysisTests(unittest.TestCase):
    def test_partial_study_and_observed_operator_difference(self):
        result = analyze({"atomic": fixture(), "staged": fixture("staged", operator_value=2.5)})
        self.assertFalse(result["complete_six_case_study"])
        self.assertEqual(len(result["missing"]), 4)
        compared = result["comparisons"][0]
        self.assertTrue(compared["all_requests_equal"])
        first = compared["operators"]["first_observed_unequal"]
        self.assertEqual(first["module"], "model.layers.0")
        self.assertEqual(first["max_abs_difference"], 0.5)

    def test_reordering_uses_prompt_identity(self):
        result = analyze({"atomic": fixture(), "target_first_atomic": fixture(reorder=True)})
        self.assertTrue(result["comparisons"][0]["all_requests_equal"])
        target = next(r for r in result["comparisons"][0]["requests"] if r["is_target"])
        self.assertEqual((target["left_request_id"], target["right_request_id"]), ("diag-1", "diag-0"))

    def test_shared_rotary_alias_is_explicitly_excluded(self):
        a, b = fixture(), fixture("staged")
        for payload in (a, b):
            alias = copy.deepcopy(payload["capture"]["operator_calls"][0])
            alias["module"] = "model.layers.0.self_attn.rotary_emb"
            payload["capture"]["operator_calls"].append(alias)
        result = analyze({"atomic": a, "staged": b})
        self.assertEqual(result["comparisons"][0]["operators"]["excluded_trailing_alias_records"], {"left": 1, "right": 1})
        b["capture"]["operator_calls"][-1]["module"] = "unexplained"
        with self.assertRaises(ValueError):
            analyze({"atomic": a, "staged": b})

    def test_rejects_corrupted_evidence(self):
        for field in ("gpu", "source", "schedule", "prefix", "operator", "crash"):
            bad = copy.deepcopy(fixture("staged"))
            if field == "gpu":
                bad["gpu"]["uuid"] = "other"
            elif field == "source":
                bad["source_hashes"]["operator_probe_hook.py"] = "b"*64
            elif field == "schedule":
                bad["capture"]["schedules"][0]["num_scheduled_tokens"]["diag-0"] = 1
            elif field == "prefix":
                bad["capture"]["projection_calls"][0]["rows"][0]["output_tokens_before"] = [7]
            elif field == "operator":
                bad["capture"]["operator_calls"][0]["target_input_token"] = 999
            else:
                bad["exit_code"] = 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                analyze({"atomic": fixture(), "staged": bad})
