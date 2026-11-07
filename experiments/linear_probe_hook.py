"""Frozen-row GEMM intervention at the first layer's down projection.

The zero-peer matrices are isolated operator probes, not model inference.
Native outputs are retained unchanged; reduction flags are always restored.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from vllm.v1.worker.gpu_worker import Worker

from experiments import operator_probe_hook


class Observer(operator_probe_hook.Observer):
    """Import installs the operator observer and one frozen-input intervention."""


_load_model = Worker.load_model
_captured = False


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def vector_values(tensor):
    vector = tensor.detach().float().reshape(-1)
    assert vector.numel() and bool(torch.isfinite(vector).all()), "Invalid GEMM vector"
    return vector.cpu().tolist()


def observe(module, inputs, output):
    global _captured
    if _captured or operator_probe_hook._active is None:
        return
    runner, schedule = operator_probe_hook._active
    target = os.environ.get("PROJECTION_TARGET_ID", "diag-1")
    req_ids = list(runner.input_batch.req_ids)
    if target not in req_ids or runner.requests[target].num_computed_tokens != 0:
        return
    assert operator_probe_hook._in_first_layer, "Down projection is outside layer zero"
    assert module.tp_size == 1 and module.bias is None, "Probe requires unsharded bias-free Qwen MLP"
    request = runner.requests[target]
    assert not request.output_token_ids, "Expected the first prefill"
    lengths = [schedule.num_scheduled_tokens[req_id] for req_id in req_ids]
    total = schedule.total_num_scheduled_tokens
    assert all(length > 0 for length in lengths) and sum(lengths) == total
    starts = [int(value) for value in runner.query_start_loc_np[:len(req_ids) + 1]]
    expected = [0]
    for length in lengths:
        expected.append(expected[-1] + length)
    assert starts == expected, "Cannot align frozen input row"
    index = req_ids.index(target)
    offset = starts[index + 1] - 1
    source = inputs[0]
    native = output[0] if isinstance(output, tuple) else output
    assert source.ndim == native.ndim == 2 and source.shape[0] == native.shape[0] == total
    assert source.dtype == module.weight.dtype == torch.float16
    assert source.shape[1] == module.weight.shape[1]
    assert native.shape[1] == module.weight.shape[0]
    assert int(runner.input_ids_cpu[offset]) == request.prompt_token_ids[lengths[index] - 1]
    frozen = source[offset].detach().clone()
    input_sha = tensor_sha(frozen)
    weight_sha = tensor_sha(module.weight)
    original_flag = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    record = {
        "module": "model.layers.0.mlp.down_proj", "request_id": target,
        "req_ids": req_ids, "query_lengths": lengths, "native_input_shape": list(source.shape),
        "native_target_offset": offset, "target_token_position": lengths[index] - 1,
        "computed_tokens": request.num_computed_tokens,
        "output_tokens_before": list(request.output_token_ids),
        "input_values": vector_values(frozen), "input_sha256": input_sha,
        "weight_sha256": weight_sha, "weight_shape": list(module.weight.shape),
        "weight_dtype": str(module.weight.dtype),
        "quant_method": type(module.quant_method).__module__ + "." + type(module.quant_method).__qualname__,
        "native_output": vector_values(native[offset]),
        "native_allow_fp16_reduced_precision_reduction": original_flag,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "scope": "same frozen target input and same weight object; all other rows zero; isolated GEMM only",
        "replays": [],
    }
    reference = torch.mv(module.weight.detach().double(), frozen.double())
    assert bool(torch.isfinite(reference).all()), "Nonfinite FP64 reference"
    record["fp64_reference"] = reference.cpu().tolist()
    record["fp64_reference_rounded_fp16"] = vector_values(reference.to(torch.float16))
    record["reference_scope"] = "FP64 dot products of the same FP16 input/weights, not an end-to-end model oracle"
    del reference
    try:
        for reduction in (True, False):
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = reduction
            for rows, row_index in ((4096, 1023), (3585, 512)):
                matrix = torch.zeros((rows, frozen.numel()), device=source.device, dtype=source.dtype)
                matrix[row_index].copy_(frozen)
                assert torch.equal(matrix[row_index], frozen), "Frozen input changed"
                replay = module.quant_method.apply(module, matrix, bias=module.bias)
                assert replay.shape == (rows, module.weight.shape[0])
                record["replays"].append({
                    "input_shape": list(matrix.shape), "target_offset": row_index,
                    "allow_fp16_reduced_precision_reduction": reduction,
                    "input_sha256": tensor_sha(matrix[row_index]), "weight_sha256": weight_sha,
                    "output": vector_values(replay[row_index]),
                })
                del matrix, replay
    finally:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = original_flag
    assert tensor_sha(frozen) == input_sha and tensor_sha(module.weight) == weight_sha
    assert all(replay["input_sha256"] == input_sha for replay in record["replays"])
    assert torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction == original_flag
    record["restored_allow_fp16_reduced_precision_reduction"] = original_flag
    destination = Path(os.environ["LINEAR_TRACE"])
    with destination.open("x") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")
    _captured = True


def load_model(self, *args, **kwargs):
    result = _load_model(self, *args, **kwargs)
    modules = dict(self.model_runner.model.named_modules())
    name = "model.layers.0.mlp.down_proj"
    assert name in modules, "Expected Qwen down projection is missing"
    modules[name].register_forward_hook(observe)
    return result


if os.environ.get("LINEAR_TRACE"):
    assert os.environ.get("OPERATOR_TRACE"), "Operator observer is required for row alignment"
    Worker.load_model = load_model
