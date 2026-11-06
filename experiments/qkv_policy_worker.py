"""Worker-local candidate arithmetic; imported only by the diagnostic probe."""
import os

import torch
from vllm.model_executor.layers.linear import QKVParallelLinear, UnquantizedLinearMethod
from vllm.v1.worker.gpu_worker import Worker

from experiments.fix_holdout_worker import HoldoutWorker
from experiments.qkv_policy_probe import eligible, fp32_linear, policy_state, require_fixed_bias

torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32 = False
policy_state()
_apply = UnquantizedLinearMethod.apply
_load = Worker.load_model
_names = {}
_calls = {}
_expected = None
_metadata = []
_candidate = os.environ.get('CIE_DIAGNOSTIC_CANDIDATE', 'fp32_biased_qkv')
if _candidate not in ('fp32_biased_qkv', 'fp32_qkv_and_head', 'fixed_tile_qkv_fp32_head', 'fixed_tile_linears_fp32_head'):
    raise ValueError('Unknown diagnostic candidate')
_head_policy = None
_all_linears = _candidate == 'fixed_tile_linears_fp32_head'
_fixed_linear = None
if _candidate == 'fixed_tile_qkv_fp32_head':
    from cloud_engine.batch_invariant_linear import batch_invariant_linear
    _fixed_linear = batch_invariant_linear
if _all_linears:
    from cloud_engine.batch_invariant_linear import fixed_tile_linear
    _fixed_linear = fixed_tile_linear


def apply(self, layer, x, bias=None):
    if _all_linears:
        if id(layer) not in _names:
            return _apply(self, layer, x, bias)
        return _fixed_linear(x, layer.weight, bias)
    if not isinstance(layer, QKVParallelLinear):
        return _apply(self, layer, x, bias)
    # A caller may intentionally omit bias (skip_bias_add): retain that contract.
    if bias is None:
        return _apply(self, layer, x, bias)
    if not eligible(layer, x, bias, QKVParallelLinear, torch.float16):
        raise ValueError('Candidate QKV requires actual FP16 input/weight/supplied bias')
    if id(layer) not in _names:
        raise ValueError('Unregistered candidate QKV module')
    if _fixed_linear is not None:
        return _fixed_linear(x, layer.weight, bias)
    _calls[_names[id(layer)]] += 1
    return fp32_linear(layer, x, bias)


def load(self, *args, **kwargs):
    global _expected, _head_policy
    result = _load(self, *args, **kwargs)
    policy_state()
    model = self.model_runner.model
    _expected = (4 if _all_linears else 1) * self.model_runner.model_config.hf_config.num_hidden_layers
    for name, module in model.named_modules():
        selected = isinstance(module, QKVParallelLinear)
        if _all_linears:
            selected = selected or name.endswith('.self_attn.o_proj') or any(name.endswith('.mlp.' + suffix) for suffix in ('gate_up_proj', 'down_proj'))
        if selected:
            require_fixed_bias(module, _candidate)
            if _all_linears and ((module.bias is None) == isinstance(module, QKVParallelLinear)):
                raise ValueError(f'Unexpected pinned transformer bias: {name}')
            valid = (module.weight.dtype == torch.float16 and (module.bias is None or module.bias.dtype == torch.float16)) if _all_linears else eligible(module, module.weight, module.bias, QKVParallelLinear, torch.float16)
            if not valid:
                raise ValueError(f'Unsupported candidate QKV module: {name}')
            if not isinstance(module.quant_method, UnquantizedLinearMethod):
                raise ValueError('Candidate only supports unquantized QKV')
            _metadata.append({'name': name, 'weight_dtype': str(module.weight.dtype), 'bias_dtype': str(module.bias.dtype) if module.bias is not None else None,
                              'skip_bias_add': module.skip_bias_add})
            _names[id(module)] = name
            if _fixed_linear is None:
                _calls[name] = 0
    if len(_names) != _expected or not _names:
        raise ValueError('Candidate QKV module count differs from model layers')
    if _candidate in ('fp32_qkv_and_head', 'fixed_tile_qkv_fp32_head', 'fixed_tile_linears_fp32_head'):
        from experiments import qkv_head_policy
        _head_policy = qkv_head_policy
        _head_policy.install(model, self.model_runner)
    return result


class QKVPolicyWorker(HoldoutWorker):
    def holdout_reset(self, expected_prompts):
        HoldoutWorker.holdout_reset(self, expected_prompts)
        if _head_policy is not None:
            _head_policy.reset()

    def qkv_policy_status(self):
        fixed_policy = None
        if _fixed_linear is not None:
            from cloud_engine.batch_invariant_linear import batch_invariant_linear, kernel_policy
            operator, policy = batch_invariant_linear, kernel_policy
            if _all_linears:
                from cloud_engine.batch_invariant_linear import (
                    fixed_tile_linear,
                    linear_kernel_policy,
                )
                operator, policy = fixed_tile_linear, linear_kernel_policy
            fixed_policy = {**policy(), 'operator_active': _fixed_linear is operator}
        return {'candidate': _candidate, 'precision': policy_state(),
                'fixed_tile_policy': fixed_policy, 'head': _head_policy.status() if _head_policy is not None else {'enabled': False},
                'modules': list(_names.values()), 'module_count': len(_names),
                'expected_module_count': _expected, 'calls': dict(_calls),
                'module_metadata': list(_metadata), 'wrapper_active': UnquantizedLinearMethod.apply is apply,
                'call_scope': ('installation identity only; no hotpath counters' if _fixed_linear is not None else
                               'Python executions including graph capture; replay does not increment'),
                'scope': ('all pinned transformer linears; FP32 head unchanged' if _all_linears else 'only unquantized biased FP16 QKVParallelLinear apply; supplied bias preserved')}


UnquantizedLinearMethod.apply = apply
Worker.load_model = load
