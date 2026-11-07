"""Diagnostic-only worker evidence for the full fixed-policy holdout.

Validates packed inputs and live-request KV address ownership, not the semantic
contents of every cached KV tensor or the scheduler's unavailable free list.
"""
from __future__ import annotations

import hashlib
import json
import math

import torch
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from cloud_engine.precision import precision_status
from cloud_engine.vllm_precision_worker import PrecisionWorker

_state = None
_expected = {}
_active = None
_execute = GPUModelRunner.execute_model
_prepare = GPUModelRunner._prepare_inputs
_project = LogitsProcessor._get_logits


class HoldoutWorker(PrecisionWorker):
    def holdout_reset(self, expected_prompts: dict[str, list[int]]) -> None:
        global _state, _expected
        if not expected_prompts or any(not isinstance(key, str) or not ids or
                any(not isinstance(token, int) or token < 0 for token in ids)
                for key, ids in expected_prompts.items()):
            raise ValueError("Expected nonempty named token-ID prompts")
        assert self.model_runner.vllm_config.parallel_config.tensor_parallel_size == 1
        _expected = {key: list(ids) for key, ids in expected_prompts.items()}
        _state = {"requests": {key: {"steps": [], "input_verified": False,
                                     "initial_computed_tokens": None, "pack_checks": 0}
                               for key in expected_prompts},
                  "failures": [], "pack_checks": 0, "forward_traces": [],
                  "scope": "actual packed input/device tables and live-request KV write ownership; no KV-value/free-list audit"}

    def holdout_evidence(self) -> dict:
        assert _state is not None, "holdout_reset must precede evidence collection"
        return {**_state, "precision": precision_status()}


def execute(self, scheduler_output, *args, **kwargs):
    global _active
    _active = (self, scheduler_output) if _state is not None else None
    try:
        return _execute(self, scheduler_output, *args, **kwargs)
    finally:
        _active = None



class UnsupportedPreemption(ValueError):
    """Worker cached tables cannot certify scheduler ownership after preemption."""


def require_no_preemption(schedule):
    cached = schedule.scheduled_cached_reqs
    resumed = [key for key, flag in zip(cached.req_ids, cached.resumed_from_preemption, strict=True) if flag]
    if resumed:
        raise UnsupportedPreemption(f"Scheduler resumed preempted requests: {resumed}")


def require_writable_owner(block_owners, scheduled_ids):
    if len(block_owners) == 1:
        return
    if any(owner[0] not in scheduled_ids for owner in block_owners):
        raise UnsupportedPreemption("Unscheduled cached owner may retain a freed preemption table; ownership unresolved")
    raise AssertionError("Write into a shared read-only prefix block")


def check_pack(runner, schedule):
    require_no_preemption(schedule)
    req_ids = list(runner.input_batch.req_ids)
    lengths = [int(schedule.num_scheduled_tokens[key]) for key in req_ids]
    total = schedule.total_num_scheduled_tokens
    assert req_ids and len(set(req_ids)) == len(req_ids)
    assert set(req_ids) == set(schedule.num_scheduled_tokens)
    assert all(length > 0 for length in lengths) and sum(lengths) == total
    starts = [int(value) for value in runner.query_start_loc_np[:len(req_ids) + 1]]
    expected_starts = [0]
    for length in lengths:
        expected_starts.append(expected_starts[-1] + length)
    assert starts == expected_starts, "Packed query offsets disagree"
    assert runner.query_start_loc[:len(req_ids) + 1].cpu().tolist() == starts, "Device query offsets disagree"
    expected_lengths = [int(runner.requests[key].num_computed_tokens) + length for key, length in zip(req_ids, lengths, strict=True)]
    assert runner.seq_lens[:len(req_ids)].cpu().tolist() == expected_lengths, "Device attention lengths disagree"
    inputs = runner.input_ids[:total].cpu().tolist()
    positions = runner.positions[:total].cpu().tolist()
    assert inputs == runner.input_ids_cpu[:total].tolist(), "CPU/device packed input mismatch"
    assert positions == runner.positions_np[:total].tolist(), "CPU/device position mismatch"
    writes = []
    for index, (key, length) in enumerate(zip(req_ids, lengths, strict=True)):
        assert key in _expected, f"Unexpected request {key}"
        request = runner.requests[key]
        entry = _state["requests"][key]
        if entry["initial_computed_tokens"] is None:
            entry["initial_computed_tokens"] = int(request.num_computed_tokens)
        assert list(request.prompt_token_ids) == _expected[key], "Worker prompt identity mismatch"
        known = list(request.prompt_token_ids) + list(request.output_token_ids)
        begin, end = starts[index:index + 2]
        offset = int(request.num_computed_tokens)
        assert positions[begin:end] == list(range(offset, offset + length)), "Wrong request positions"
        assert inputs[begin:end] == known[offset:offset + length], "Wrong request input content"
        assert offset + length <= len(known), "Packed tokens exceed known request history"
        writes.extend((key, position) for position in positions[begin:end])
    table_hashes = []
    for group, table in enumerate(runner.input_batch.block_table.block_tables):
        block_size = table.block_size
        block_count = runner.kv_cache_config.num_blocks
        owners = {}
        # Include paused/unscheduled live requests, not only this packed batch.
        for key, request in runner.requests.items():
            known = list(request.prompt_token_ids) + list(request.output_token_ids)
            assert 0 <= request.num_computed_tokens <= len(known), "Invalid live request history length"
            for logical, block in enumerate(request.block_ids[group]):
                assert 0 <= block < block_count, "KV block outside allocated capacity"
                owners.setdefault(block, []).append((key, logical, request.num_computed_tokens, known))
        for index, key in enumerate(req_ids):
            blocks = list(runner.requests[key].block_ids[group])
            assert len(blocks) == int(table.num_blocks_per_row[index]), "KV table length mismatch"
            assert table.block_table_np[index, :len(blocks)].tolist() == blocks, "CPU KV table identity mismatch"
            assert table.block_table[index, :len(blocks)].cpu().tolist() == blocks, "Device KV table identity mismatch"
        slots = table.slot_mapping[:total].cpu().tolist()
        assert slots == table.slot_mapping_np[:total].tolist(), "CPU/device slot map mismatch"
        table_hashes.append({"group": group, "block_size": block_size,
                             "slot_mapping_sha256": hashlib.sha256(json.dumps(slots).encode()).hexdigest(),
                             "block_tables_sha256": hashlib.sha256(json.dumps([runner.requests[key].block_ids[group] for key in req_ids]).encode()).hexdigest()})
        occupied = set()
        for packed_index, (key, position) in enumerate(writes):
            logical, within = divmod(position, block_size)
            blocks = runner.requests[key].block_ids[group]
            assert logical < len(blocks), "KV write outside request block table"
            block = blocks[logical]
            slot = block * block_size + within
            assert slots[packed_index] == slot, "KV write slot belongs to another address"
            assert slot not in occupied, "Two packed writes alias a KV slot"
            require_writable_owner(owners[block], set(req_ids))
            occupied.add(slot)
        for block_owners in owners.values():
            if len(block_owners) <= 1:
                continue
            first_key, first_logical, _, first_known = block_owners[0]
            del first_key
            end = (first_logical + 1) * block_size
            for _, logical, computed, known in block_owners:
                if logical != first_logical or computed < end or known[:end] != first_known[:end]:
                    if any(owner[0] not in set(req_ids) for owner in block_owners):
                        raise UnsupportedPreemption("Shared cached ownership may be stale after preemption")
                    raise AssertionError("Shared KV block is not an identical complete read-only prefix")
    _state["forward_traces"].append({"req_ids": req_ids, "query_lengths": lengths, "query_starts": starts,
                                     "computed_tokens": [int(runner.requests[key].num_computed_tokens) for key in req_ids],
                                     "packed_input_sha256": hashlib.sha256(json.dumps(inputs).encode()).hexdigest(),
                                     "kv_table_hashes": table_hashes})
    _state["pack_checks"] += 1
    for key in req_ids:
        entry = _state["requests"][key]
        entry["pack_checks"] += 1
        entry["input_verified"] = not entry.get("input_failure", False)


def prepare(self, scheduler_output, *args, **kwargs):
    result = _prepare(self, scheduler_output, *args, **kwargs)
    if _active is not None:
        try:
            check_pack(self, scheduler_output)
        except (AssertionError, IndexError, KeyError, TypeError, ValueError) as exc:
            _state["failures"].append({"kind": "unsupported_preemption" if isinstance(exc, UnsupportedPreemption) else "packed_input_or_kv_ownership", "detail": str(exc)})
            for key in self.input_batch.req_ids:
                if key in _state["requests"]:
                    _state["requests"][key]["input_verified"] = False
                    _state["requests"][key]["input_failure"] = True
    return result


def project(self, hidden, head, bias):
    logits = _project(self, hidden, head, bias)
    if _active is None or logits is None:
        return logits
    runner, schedule = _active
    req_ids = list(runner.input_batch.req_ids)
    assert logits.shape[0] == len(req_ids), "Cannot align full-vocabulary evidence"
    finite = torch.isfinite(logits).all(dim=-1).cpu().tolist()
    for index, key in enumerate(req_ids):
        request = runner.requests[key]
        computed = int(request.num_computed_tokens)
        # vLLM computes but discards a sample for incomplete prompt chunks.
        if computed + schedule.num_scheduled_tokens[key] < request.num_tokens:
            continue
        assert key in _state["requests"], "Sampled unexpected request"
        if finite[index]:
            values = torch.log_softmax(logits[index].float(), dim=-1)
            top_values, top_ids = values.topk(20)
            top_k = list(zip(top_ids.cpu().tolist(), top_values.cpu().tolist(), strict=True))
            assert all(math.isfinite(value) for _, value in top_k)
            argmax = int(logits[index].argmax())
        else:
            top_k, argmax = [], None
            _state["failures"].append({"kind": "nonfinite_full_vocab", "request_id": key})
        _state["requests"][key]["steps"].append({
            "argmax": argmax, "top_k": top_k, "all_finite": finite[index],
            "computed_tokens": computed, "output_tokens_before": list(request.output_token_ids),
        })
    return logits


GPUModelRunner.execute_model = execute
GPUModelRunner._prepare_inputs = prepare
LogitsProcessor._get_logits = project
