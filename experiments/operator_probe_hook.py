"""Observe one target's final packed token through embedding and layer zero.

This is not a global first-difference detector: other positions and later
layers are not captured, and synchronous observation can change scheduling.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker

from experiments import projection_hook  # installs the projection observer first


class Observer(projection_hook.Observer):
    """Worker extension whose import enables bounded operator observation."""


_active = None
_in_first_layer = False
_execute = GPUModelRunner.execute_model
_load_model = Worker.load_model


def execute(self, scheduler_output, *args, **kwargs):
    global _active, _in_first_layer
    _active = (self, scheduler_output)
    _in_first_layer = False
    try:
        return _execute(self, scheduler_output, *args, **kwargs)
    finally:
        _active = None
        _in_first_layer = False


def enter_first_layer(module, inputs):
    global _in_first_layer
    assert not _in_first_layer, "Nested first-layer observer invocation"
    _in_first_layer = True


def leave_first_layer(module, inputs, output):
    global _in_first_layer
    try:
        observe("model.layers.0", output)
    finally:
        _in_first_layer = False


def packed_outputs(value, total, path="output"):
    """Retain tuple structure, selecting only outputs with the packed token axis."""
    if isinstance(value, torch.Tensor):
        if value.ndim and value.shape[0] == total:
            yield path, value
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from packed_outputs(item, total, f"{path}[{index}]")


def observe(name, output):
    if _active is None or (name != "model.embed_tokens" and not _in_first_layer):
        return
    runner, schedule = _active
    target = os.environ.get("PROJECTION_TARGET_ID", "diag-1")
    req_ids = list(runner.input_batch.req_ids)
    if target not in req_ids:
        return
    request = runner.requests[target]
    if request.num_computed_tokens != 0:
        return
    assert not request.output_token_ids, "First prefill already has generated history"
    lengths = [schedule.num_scheduled_tokens[req_id] for req_id in req_ids]
    total = schedule.total_num_scheduled_tokens
    assert all(length > 0 for length in lengths) and sum(lengths) == total
    starts = [int(value) for value in runner.query_start_loc_np[:len(req_ids) + 1]]
    expected = [0]
    for length in lengths:
        expected.append(expected[-1] + length)
    assert starts == expected, "Request membership does not match packed query offsets"
    index = req_ids.index(target)
    offset = starts[index + 1] - 1
    assert 0 <= offset < total
    token_position = lengths[index] - 1
    expected_token = request.prompt_token_ids[token_position]
    actual_token = int(runner.input_ids_cpu[offset])
    assert actual_token == expected_token, "Target packed token identity mismatch"
    vectors = []
    for path, tensor in packed_outputs(output, total):
        vector = tensor[offset].detach().float().reshape(-1)
        assert vector.numel() and bool(torch.isfinite(vector).all()), "Nonfinite/empty operator output"
        vectors.append({"path": path, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                        "values": vector.cpu().tolist()})
    if not vectors:
        return
    record = {
        "module": name, "request_id": target, "req_ids": req_ids,
        "query_lengths": lengths, "total_num_scheduled_tokens": total,
        "target_packed_offset": offset, "target_token_position": token_position,
        "target_input_token": actual_token, "prompt_length": len(request.prompt_token_ids),
        "computed_tokens": request.num_computed_tokens,
        "output_tokens_before": list(request.output_token_ids), "vectors": vectors,
        "scope": "last packed token of target first prefill; embedding and layer zero outputs only",
        "precision": {
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
    }
    with Path(os.environ["OPERATOR_TRACE"]).open("a") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")


def load_model(self, *args, **kwargs):
    result = _load_model(self, *args, **kwargs)
    selected = []
    for name, module in self.model_runner.model.named_modules():
        if name in ("model.embed_tokens", "model.layers.0") or name.startswith("model.layers.0."):
            selected.append(name)
            if name == "model.layers.0":
                module.register_forward_pre_hook(enter_first_layer)
                module.register_forward_hook(leave_first_layer)
            else:
                # Shared rotary modules also execute in later layers.
                module.register_forward_hook(
                    lambda module, inputs, output, name=name: observe(name, output)
                )
    assert {"model.embed_tokens", "model.layers.0"} <= set(selected), "Expected Qwen operator modules missing"
    return result


if os.environ.get("OPERATOR_TRACE"):
    GPUModelRunner.execute_model = execute
    Worker.load_model = load_model
