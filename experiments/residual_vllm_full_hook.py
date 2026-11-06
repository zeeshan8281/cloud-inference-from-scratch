"""Full-window diagnostic with bounded all-layer target-row observation.

Control preserves the full schedule collector but installs no tensor hooks.
No observation establishes the globally earliest divergent token.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path

import torch
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker

from experiments import fix_holdout_worker as holdout
from experiments.qkv_policy_worker import QKVPolicyWorker

_load = Worker.load_model
_prepare = GPUModelRunner._prepare_inputs
_project = LogitsProcessor._get_logits
_target = None
_path = None
_schedules = []
_layer = None
_calls = 0
_observe = os.environ.get('RESIDUAL_FULL_OBSERVE') == '1'


def context():
    if not _observe or holdout._active is None or _target is None:
        return None
    runner, schedule = holdout._active
    ids = list(runner.input_batch.req_ids)
    if _target not in ids:
        return None
    request = runner.requests[_target]
    history = list(request.output_token_ids)
    if len(history) >= 2:
        return None
    index = ids.index(_target)
    starts = [int(v) for v in runner.query_start_loc_np[:len(ids)+1]]
    lengths = [int(schedule.num_scheduled_tokens[key]) for key in ids]
    assert starts == [sum(lengths[:i]) for i in range(len(ids)+1)]
    offset = starts[index+1]-1
    position = int(request.num_computed_tokens)+lengths[index]-1
    known = list(request.prompt_token_ids)+history
    assert int(runner.input_ids_cpu[offset]) == known[position]
    return {'request_id': _target, 'req_ids': ids, 'query_starts': starts, 'query_lengths': lengths,
            'forward': len(_schedules)-1, 'packed_tokens': int(schedule.total_num_scheduled_tokens),
            'target_offset': offset, 'target_position': position, 'target_input_token': known[position],
            'computed_tokens': int(request.num_computed_tokens), 'output_tokens_before': history,
            'emits_token': position+1 == request.num_tokens}


def tensors(value, path):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, (tuple, list)):
        for i, item in enumerate(value):
            yield from tensors(item, f'{path}[{i}]')


def vector(value, path, shape):
    value = value.detach().contiguous().cpu()
    assert value.numel() and bool(torch.isfinite(value).all())
    raw = value.view(torch.uint8).numpy().tobytes()
    return {'path': path, 'shape': list(shape), 'row_shape': list(value.shape),
            'dtype': str(value.dtype), 'finite': True, 'sha256': hashlib.sha256(raw).hexdigest(),
            'bytes_base64': base64.b64encode(raw).decode()}


def append(record):
    global _calls
    _calls += 1
    with Path(_path).open('a') as handle:
        handle.write(json.dumps({'call': _calls, **record}, allow_nan=False)+'\n')


def observe(name, value, direction):
    record = context()
    if record is None:
        return
    if name.startswith('model.layers.'):
        if _layer is None:
            return
        # Rotary modules are shared; their first named_modules alias is not
        # evidence of which layer executed them. Active enclosing layer is.
        parts = name.split('.')
        parts[2] = str(_layer)
        name = '.'.join(parts)
    vectors = [vector(t[record['target_offset']], path, t.shape)
               for path, t in tensors(value, direction)
               if t.ndim and t.shape[0] == record['packed_tokens'] and t.is_floating_point()]
    if vectors:
        append({**record, 'operator': name, 'direction': direction, 'vectors': vectors,
                'scope': 'target final packed row only; all model layers'})


def prepare(self, schedule, *args, **kwargs):
    result = _prepare(self, schedule, *args, **kwargs)
    if holdout._active is not None and _target is not None:
        ids = list(self.input_batch.req_ids)
        starts = [int(v) for v in self.query_start_loc_np[:len(ids)+1]]
        _schedules.append({'req_ids': ids, 'query_starts': starts,
                           'items': [{'request_id': key, 'computed_tokens': int(self.requests[key].num_computed_tokens),
                                      'scheduled_tokens': int(schedule.num_scheduled_tokens[key]),
                                      'input_tokens': self.input_ids_cpu[starts[i]:starts[i+1]].tolist(),
                                      'output_tokens_before': list(self.requests[key].output_token_ids)}
                                     for i, key in enumerate(ids)]})
    return result


def project(self, hidden, head, bias):
    logits = _project(self, hidden, head, bias)
    record = context()
    if record is None or logits is None:
        return logits
    index = record['req_ids'].index(_target)
    assert len(record['req_ids']) == hidden.shape[0] == logits.shape[0]
    assert bool(torch.isfinite(logits[index]).all())
    candidates = sorted(set(logits[index].topk(20).indices.tolist()+[2182, 3015, 279]))
    append({**record, 'operator': 'lm_head', 'direction': 'projection',
            'vectors': [vector(hidden[index], 'input', hidden.shape)],
            'logits_dtype': str(logits.dtype), 'all_vocab_finite': True,
            'argmax': int(logits[index].argmax()), 'candidate_ids': candidates,
            'candidate_logits': logits[index, candidates].float().cpu().tolist()})
    return logits


def enter_layer(index, name, inputs):
    global _layer
    assert _layer is None
    _layer = index
    observe(name, inputs, 'input')


def leave_layer(name, output):
    global _layer
    try:
        observe(name, output, 'output')
    finally:
        _layer = None


def load(self, *args, **kwargs):
    result = _load(self, *args, **kwargs)
    if not _observe:
        return result
    selected = []
    for name, module in self.model_runner.model.named_modules():
        if name in ('model.embed_tokens', 'model.norm') or name.startswith('model.layers.'):
            selected.append(name)
            if name.startswith('model.layers.') and len(name.split('.')) == 3:
                index = int(name.split('.')[2])
                module.register_forward_pre_hook(lambda m, x, index=index, name=name: enter_layer(index, name, x))
                module.register_forward_hook(lambda m, x, y, name=name: leave_layer(name, y), always_call=True)
            else:
                module.register_forward_pre_hook(lambda m, x, name=name: observe(name, x, 'input'))
                module.register_forward_hook(lambda m, x, y, name=name: observe(name, y, 'output'))
    assert {'model.embed_tokens', 'model.norm', 'model.layers.0'} <= set(selected)
    return result


class FullResidualWorker(QKVPolicyWorker):
    def residual_trace_reset(self, target: str, path: str):
        global _target, _path, _schedules, _calls, _layer
        _target, _path, _schedules, _calls, _layer = target, path, [], 0, None
        if Path(path).exists():
            raise ValueError('trace path already exists')
        Path(path).touch()
        return {'observe': _observe}

    def residual_trace_evidence(self):
        return {'schedules': _schedules, 'trace_calls': _calls, 'observe': _observe,
                'trace_sha256': hashlib.sha256(Path(_path).read_bytes()).hexdigest()}


GPUModelRunner._prepare_inputs = prepare
Worker.load_model = load
if _observe:
    LogitsProcessor._get_logits = project
