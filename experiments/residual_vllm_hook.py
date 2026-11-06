"""Bounded residual observer; synchronous traces may affect admission timing."""
import json
import os
from pathlib import Path

import torch
from vllm.model_executor.layers.logits_processor import LogitsProcessor

from cloud_engine.vllm_precision_worker import PrecisionWorker

if os.environ.get("OPERATOR_TRACE"):
    from experiments import operator_probe_hook as base


class Observer(PrecisionWorker):
    def residual_status(self):
        return {"allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "float32_matmul_precision": torch.get_float32_matmul_precision()}


def append(path, record):
    with Path(os.environ[path]).open('a') as handle:
        handle.write(json.dumps(record, allow_nan=False) + '\n')


def targets():
    return json.loads(os.environ['RESIDUAL_TARGET_IDS'])


def context(runner, schedule, target):
    req_ids = list(runner.input_batch.req_ids)
    request = runner.requests[target]
    lengths = [schedule.num_scheduled_tokens[r] for r in req_ids]
    starts = [int(x) for x in runner.query_start_loc_np[:len(req_ids) + 1]]
    expected = [0]
    for length in lengths:
        assert length > 0
        expected.append(expected[-1] + length)
    assert starts == expected and expected[-1] == schedule.total_num_scheduled_tokens
    index = req_ids.index(target)
    offset = starts[index + 1] - 1
    position = request.num_computed_tokens + lengths[index] - 1
    history = list(request.prompt_token_ids) + list(request.output_token_ids)
    assert position < len(history) and int(runner.input_ids_cpu[offset]) == history[position]
    return {'request_id': target, 'req_ids': req_ids, 'query_lengths': lengths,
            'query_starts': starts, 'total_num_scheduled_tokens': expected[-1],
            'target_packed_offset': offset, 'target_token_position': position,
            'computed_tokens': request.num_computed_tokens,
            'output_tokens_before': list(request.output_token_ids),
            'target_input_token': history[position], 'prompt_length': len(request.prompt_token_ids)}


def observe(name, output):
    if base._active is None or (name != 'model.embed_tokens' and not base._in_first_layer):
        return
    runner, schedule = base._active
    for target in targets():
        if target not in runner.input_batch.req_ids:
            continue
        record = context(runner, schedule, target)
        vectors = []
        for path, tensor in base.packed_outputs(output, schedule.total_num_scheduled_tokens):
            vector = tensor[record['target_packed_offset']].detach().float().reshape(-1)
            assert vector.numel() and bool(torch.isfinite(vector).all())
            vectors.append({'path': path, 'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                            'values': vector.cpu().tolist()})
        if vectors:
            append('OPERATOR_TRACE', {**record, 'module': name, 'vectors': vectors,
                   'scope': 'target last packed row; embedding and layer zero only',
                   'allow_fp16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction})


def project(self, hidden, head, bias):
    logits = base.projection_hook._project(self, hidden, head, bias)
    if base._active is None or logits is None:
        return logits
    runner, schedule = base._active
    ids = list(runner.input_batch.req_ids)
    assert hidden.shape[0] == logits.shape[0] == len(ids)
    for target in targets():
        if target not in ids:
            continue
        index = ids.index(target)
        record = context(runner, schedule, target)
        assert bool(torch.isfinite(logits[index]).all()) and bool(torch.isfinite(hidden[index]).all())
        candidates = sorted(set(logits[index].topk(20).indices.tolist() + [2182, 3015, 264, 705]))
        append('RESIDUAL_PROJECTION_TRACE', {**record, 'hidden_shape': list(hidden.shape),
               'hidden_dtype': str(hidden.dtype), 'logits_dtype': str(logits.dtype),
               'hidden': hidden[index].float().cpu().tolist(), 'candidate_ids': candidates,
               'native_logits': logits[index, candidates].float().cpu().tolist(),
               'native_argmax': int(logits[index].argmax()), 'all_finite': True,
               'emits_token': record['target_token_position'] + 1 == runner.requests[target].num_tokens})
    return logits


if os.environ.get("OPERATOR_TRACE"):
    base.observe = observe
    LogitsProcessor._get_logits = project
