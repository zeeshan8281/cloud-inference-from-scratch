"""Diagnostic TP1 LM-head candidate with same-hidden native counterfactual."""
import hashlib

import torch
import torch.nn.functional as functional
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)

from experiments import fix_holdout_worker as observer
from experiments.qkv_policy_probe import head_eligible, policy_state, select_output_projection

_original = UnquantizedEmbeddingMethod.apply
_head = None
_tied = None
_state = {}


def reset():
    _state.clear()
    _state.update(calls=0, native_counterfactual_calls=0, emitted_rows=0, all_finite=True,
                  input_dtypes=[], output_dtypes=[], autocast_enabled=False, events=[])


def install(model, runner):
    global _head, _tied
    if runner.vllm_config.parallel_config.tensor_parallel_size != 1:
        raise ValueError('Head diagnostic requires TP1')
    _head, _tied = select_output_projection(model, VocabParallelEmbedding, torch.float16)
    if bool(runner.model_config.hf_config.tie_word_embeddings) != _tied:
        raise ValueError('Output projection identity disagrees with tied-weight config')
    if not isinstance(_head.quant_method, UnquantizedEmbeddingMethod):
        raise ValueError('Expected unquantized LM head')
    reset()
    UnquantizedEmbeddingMethod.apply = apply


def apply(self, layer, hidden, bias=None):
    if layer is not _head:
        return _original(self, layer, hidden, bias)
    if not head_eligible(layer, hidden, VocabParallelEmbedding, torch.float16):
        raise ValueError('Unexpected diagnostic head/input dtype')
    if bias is not None and bias.dtype != torch.float16:
        raise ValueError('Unexpected head bias dtype')
    policy_state()
    if torch.is_autocast_enabled():
        raise ValueError('LM-head candidate forbids autocast')
    result = functional.linear(hidden.float(), layer.weight.float(), None if bias is None else bias.float())
    if result.dtype != torch.float32:
        raise ValueError('LM-head candidate intermediate must be actual FP32')
    _state['calls'] += 1
    _state['input_dtypes'] = [str(hidden.dtype)]
    _state['output_dtypes'] = [str(result.dtype)]
    if observer._active is None:
        return result
    native = _original(self, layer, hidden, bias)
    if native.dtype != torch.float16:
        raise ValueError('Counterfactual native head is not FP16')
    _state['native_counterfactual_calls'] += 1
    runner, schedule = observer._active
    ids = list(runner.input_batch.req_ids)
    if len(ids) != result.shape[0] or native.shape != result.shape:
        raise ValueError('Cannot align candidate/counterfactual rows')
    for index, key in enumerate(ids):
        request = runner.requests[key]
        if request.num_computed_tokens + schedule.num_scheduled_tokens[key] < request.num_tokens:
            continue
        a, b = native[index, :layer.org_vocab_size], result[index, :layer.org_vocab_size]
        finite = bool(torch.isfinite(a).all() & torch.isfinite(b).all())
        _state['all_finite'] &= finite
        if not finite:
            raise ValueError('Nonfinite full-vocabulary head output')
        _state['emitted_rows'] += 1
        na, fa = int(a.argmax()), int(b.argmax())
        nt, ft = int((a == a.max()).sum()), int((b == b.max()).sum())
        if na != fa or nt > 1 or ft > 1:
            candidates = sorted(set(a.topk(5).indices.tolist() + b.topk(5).indices.tolist()))
            _state['events'].append({'request_id': key, 'computed_tokens': request.num_computed_tokens,
                'output_tokens_before': list(request.output_token_ids),
                'hidden_sha256': hashlib.sha256(hidden[index].detach().contiguous().cpu().numpy().tobytes()).hexdigest(),
                'native_argmax': na, 'fp32_argmax': fa, 'native_tie_count': nt, 'fp32_tie_count': ft,
                'candidate_ids': candidates, 'native_logits': a[candidates].float().cpu().tolist(),
                'fp32_logits': b[candidates].cpu().tolist(), 'input_dtype': str(hidden.dtype),
                'weight_dtype': str(layer.weight.dtype), 'native_dtype': str(native.dtype),
                'output_dtype': str(result.dtype), 'all_finite': True})
    return result


def status():
    return {'enabled': True, 'implementation': 'vllm_fp32_output_projection',
            'module_count': int(_head is not None), 'expected_module_count': 1,
            'wrapper_active': UnquantizedEmbeddingMethod.apply is apply,
            'weight_dtype': str(_head.weight.dtype), 'head_class': type(_head).__name__,
            'tied_input_embedding': _tied, **_state,
            'scope': 'actual FP32 output; same-hidden native FP16 counterfactual, original vocabulary TP1 only; events retain changed argmax or ties'}
