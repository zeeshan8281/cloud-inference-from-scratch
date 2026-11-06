"""Worker-local observer for normal vLLM V1 multiprocessing execution."""
import json
import os
from pathlib import Path

import torch
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class Observer:
    """Importing this worker extension installs the bounded observation hook."""


_active = None
_execute = GPUModelRunner.execute_model
_project = LogitsProcessor._get_logits


def execute(self, *args, **kwargs):
    global _active
    _active = self
    try:
        return _execute(self, *args, **kwargs)
    finally:
        _active = None


def project(self, hidden, head, bias):
    logits = _project(self, hidden, head, bias)
    if _active is None or logits is None:
        return logits
    req_ids = list(_active.input_batch.req_ids)
    assert len(req_ids) == hidden.shape[0], 'Cannot align projection rows to requests'
    shifted = _project(self, hidden.roll(1, dims=0), head, bias).roll(-1, dims=0)
    ids = logits.float().topk(20, dim=-1).indices
    rows = []
    for row, req_id in enumerate(req_ids):
        candidates = sorted(set(ids[row].tolist() + [323, 1401]))
        precise = torch.mv(head.weight[candidates].float(), hidden[row].float())
        if bias is not None:
            precise += bias[candidates].float()
        rows.append({'request_id': req_id, 'hidden': hidden[row].float().cpu().tolist(),
                     'candidate_ids': candidates,
                     'native_logits': logits[row, candidates].float().cpu().tolist(),
                     'fp32_projection': precise.cpu().tolist(),
                     'permuted_logits': shifted[row, candidates].float().cpu().tolist(),
                     'native_argmax': int(logits[row].argmax()),
                     'permuted_argmax': int(shifted[row].argmax()),
                     'computed_tokens': _active.requests[req_id].num_computed_tokens,
                     'output_tokens_before': list(_active.requests[req_id].output_token_ids)})
    record = {'hidden_dtype': str(hidden.dtype), 'logits_dtype': str(logits.dtype),
              'shape': list(hidden.shape), 'rows': rows}
    with Path(os.environ['PROJECTION_TRACE']).open('a') as handle:
        handle.write(json.dumps(record)+'\n')
    return logits


GPUModelRunner.execute_model = execute
LogitsProcessor._get_logits = project
