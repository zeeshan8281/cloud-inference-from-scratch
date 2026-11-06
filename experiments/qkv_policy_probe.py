"""Diagnostic-only FP32 biased-QKV candidate; never declares a verified fix."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path


def policy_state():
    import torch
    state = {'allow_fp16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
             'allow_tf32': torch.backends.cuda.matmul.allow_tf32,
             'float32_matmul_precision': torch.get_float32_matmul_precision()}
    if state != {'allow_fp16_reduced_precision_reduction': False, 'allow_tf32': False,
                 'float32_matmul_precision': 'highest'}:
        raise ValueError(f'Candidate arithmetic policy not applied: {state}')
    return state


def eligible(layer, x, bias, target_type, fp16):
    return (isinstance(layer, target_type) and bias is not None
            and x.dtype == fp16 and layer.weight.dtype == fp16 and bias.dtype == fp16)


def head_eligible(layer, hidden, target_type, fp16):
    return isinstance(layer, target_type) and hidden.dtype == fp16 and layer.weight.dtype == fp16


def select_output_projection(model, target_type, fp16):
    head = getattr(model, 'lm_head', None)
    if not isinstance(head, target_type) or head.weight.dtype != fp16:
        raise ValueError('Expected actual FP16 output projection')
    embedding = getattr(getattr(model, 'model', None), 'embed_tokens', None)
    return head, head is embedding


def require_fixed_bias(layer, candidate):
    if candidate in ('fixed_tile_qkv_fp32_head', 'fixed_tile_linears_fp32_head') and getattr(layer, 'skip_bias_add', None) is not False:
        raise ValueError('Fixed-tile QKV requires actual skip_bias_add=False')


def fp32_linear(layer, x, bias):
    import torch.nn.functional as functional
    return functional.linear(x.float(), layer.weight.float(), bias.float()).to(x.dtype)


def install_custom(model, candidate="fp32_biased_qkv"):
    import torch
    all_linears = candidate == "fixed_tile_linears_fp32_head"
    fixed = candidate in ("fixed_tile_qkv_fp32_head", "fixed_tile_linears_fp32_head")
    fixed_linear = None
    if fixed:
        from cloud_engine.batch_invariant_linear import batch_invariant_linear
        fixed_linear = batch_invariant_linear
        if all_linears:
            from cloud_engine.batch_invariant_linear import fixed_tile_linear
            fixed_linear = fixed_tile_linear
    names = []
    calls = {}
    metadata = []
    installed = []
    for name, module in model.named_modules():
        selected = any(name.endswith('.self_attn.' + projection) for projection in ('q_proj', 'k_proj', 'v_proj'))
        if all_linears:
            selected = selected or name.endswith('.self_attn.o_proj') or any(name.endswith('.mlp.' + projection) for projection in ('gate_proj', 'up_proj', 'down_proj'))
        if not selected:
            continue
        biased = any(name.endswith('.self_attn.' + projection) for projection in ('q_proj', 'k_proj', 'v_proj'))
        if all_linears and ((module.bias is None) == biased):
            raise ValueError(f'Unexpected pinned transformer bias: {name}')
        if all_linears and module.bias is None:
            valid = isinstance(module, torch.nn.Linear) and module.weight.dtype == torch.float16
        else:
            valid = eligible(module, module.weight, module.bias, torch.nn.Linear, torch.float16)
        if not valid:
            raise ValueError(f'Unsupported custom QKV module: {name}')
        names.append(name)
        metadata.append({'name': name, 'weight_dtype': str(module.weight.dtype), 'bias_dtype': str(module.bias.dtype) if module.bias is not None else None})
        if fixed:
            def forward(x, module=module):
                return fixed_linear(x, module.weight, module.bias)
        else:
            calls[name] = 0
            def forward(x, module=module, name=name):
                if not eligible(module, x, module.bias, torch.nn.Linear, torch.float16):
                    raise ValueError(f'Custom QKV input changed dtype: {name}')
                calls[name] += 1
                return fp32_linear(module, x, module.bias)
        module.forward = forward
        installed.append((module, forward))
    head = {'enabled': candidate in ('fp32_qkv_and_head', 'fixed_tile_qkv_fp32_head', 'fixed_tile_linears_fp32_head'), 'implementation': 'custom_existing_fp32',
            'module_count': 1, 'expected_module_count': 1, 'wrapper_active': True,
            'weight_dtype': str((model.embed_tokens if model.lm_head is None else model.lm_head).weight.dtype),
            'input_dtypes': [], 'output_dtypes': [], 'calls': 0, 'native_counterfactual_calls': 0,
            'emitted_rows': 0, 'all_finite': True, 'autocast_enabled': False, 'events': [],
            'scope': 'existing FP32 projection unchanged; finiteness separately checked by full holdout observer'}
    def observe_head(module, inputs, output):
        if torch.is_autocast_enabled() or output.dtype != torch.float32:
            raise ValueError('Custom existing LM head did not return FP32 without autocast')
        head['calls'] += 1
        head['emitted_rows'] += output.shape[0]
        head['output_dtypes'] = [str(output.dtype)]
    def observe_norm(module, inputs, output):
        if output.dtype != torch.float16:
            raise ValueError('Custom final norm input to head is not FP16')
        head['input_dtypes'] = [str(output.dtype)]
    head_handle = norm_handle = None
    if candidate in ('fp32_qkv_and_head', 'fixed_tile_qkv_fp32_head', 'fixed_tile_linears_fp32_head'):
        head_handle = model.register_forward_hook(observe_head)
        norm_handle = model.norm.register_forward_hook(observe_norm)
    expected = (7 if all_linears else 3) * model.dims.num_layers
    if len(names) != expected or not names:
        raise ValueError(f'Expected {expected} custom biased QKV modules, got {len(names)}')
    def status():
        if candidate in ('fp32_qkv_and_head', 'fixed_tile_qkv_fp32_head', 'fixed_tile_linears_fp32_head'):
            head['wrapper_active'] = (model._forward_hooks.get(head_handle.id) is observe_head
                                      and model.norm._forward_hooks.get(norm_handle.id) is observe_norm)
        fixed_policy = None
        if fixed:
            from cloud_engine.batch_invariant_linear import batch_invariant_linear, kernel_policy
            operator, policy = batch_invariant_linear, kernel_policy
            if all_linears:
                from cloud_engine.batch_invariant_linear import (
                    fixed_tile_linear,
                    linear_kernel_policy,
                )
                operator, policy = fixed_tile_linear, linear_kernel_policy
            fixed_policy = {**policy(), 'operator_active': fixed_linear is operator}
        return {'candidate': candidate, 'fixed_tile_policy': fixed_policy, 'head': dict(head), 'precision': policy_state(), 'modules': names,
                'module_count': len(names), 'expected_module_count': expected, 'calls': dict(calls),
                'module_metadata': metadata, 'wrapper_active': all(module.forward is forward for module, forward in installed),
                'call_scope': ('installation identity only; no hotpath counters' if fixed else
                               'Python executions including graph capture; replay does not increment'),
                'scope': ('all pinned transformer linear projections; FP32 head unchanged' if all_linears else 'only custom biased FP16 q/k/v; weights and inputs retained FP16')}
    return status


def load_candidate(loader, installer, *args, **kwargs):
    """Install after weights load, before the engine can construct/capture its runner."""
    result = loader(*args, **kwargs)
    return result, installer(result[0])


async def run(config, checkpoint):
    import torch

    from cloud_engine.precision import configure_inference_precision
    from experiments import fix_holdout

    if config.get('candidate') not in ('fp32_biased_qkv', 'fp32_qkv_and_head', 'fixed_tile_qkv_fp32_head', 'fixed_tile_linears_fp32_head'):
        raise ValueError('Explicit diagnostic candidate required')
    os.environ['CIE_DIAGNOSTIC_CANDIDATE'] = config['candidate']
    workload = config['workload']
    if workload['output_tokens'] != 128 or len(workload['input_token_ids']) != 8:
        raise ValueError('Candidate requires exposed c8 full128 window')
    configure_inference_precision()
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    policy_state()
    from cloud_engine import engine as engine_module
    original_loader = engine_module.load_model
    custom_status = None
    def candidate_loader(*args, **kwargs):
        nonlocal custom_status
        result, custom_status = load_candidate(original_loader, lambda model: install_custom(model, config['candidate']), *args, **kwargs)
        return result
    engine_module.load_model = candidate_loader
    original_capture_custom = fix_holdout.capture_custom
    original_capture_vllm = fix_holdout.capture_vllm
    def verify(status):
        fixed = config['candidate'] in ('fixed_tile_qkv_fp32_head', 'fixed_tile_linears_fp32_head')
        if (status['module_count'] != status['expected_module_count'] or status.get('wrapper_active') is not True
                or (not fixed and (not status['calls'] or not all(status['calls'].values())))
                or (fixed and (status['calls'] != {} or status.get('fixed_tile_policy', {}).get('operator_active') is not True))):
            raise ValueError('Candidate did not execute all expected QKV modules')
        if config['candidate'] in ('fp32_qkv_and_head', 'fixed_tile_qkv_fp32_head', 'fixed_tile_linears_fp32_head'):
            head = status['head']
            if (not head['wrapper_active'] or head['module_count'] != 1 or head['calls'] <= 0
                    or head['input_dtypes'] != ['torch.float16'] or head['output_dtypes'] != ['torch.float32']
                    or head['autocast_enabled'] or not head['all_finite']):
                raise ValueError('Actual FP32 head evidence missing')
        return status
    async def custom_capture(*args, **kwargs):
        policy_state()
        result = await original_capture_custom(*args, **kwargs)
        result['qkv_policy'] = verify(custom_status())
        result['qkv_policy']['head']['all_finite'] = all(step['all_finite'] for record in result['records'] for step in record['steps'])
        engine = args[0]
        result['qkv_policy']['graph_counts'] = {'captures': engine.runner.cuda_graph_captures, 'replays': engine.runner.cuda_graph_replays}
        return result
    def vllm_capture(engine, *args, **kwargs):
        before = engine.collective_rpc('qkv_policy_status')
        if len(before) != 1:
            raise ValueError('Expected TP1 candidate worker before phase')
        result = original_capture_vllm(engine, *args, **kwargs)
        states = engine.collective_rpc('qkv_policy_status')
        if len(states) != 1:
            raise ValueError('Expected TP1 candidate worker')
        result['qkv_policy'] = verify(states[0])
        result['qkv_policy']['precision_before_phase'] = before[0]['precision']
        if config['candidate'] in ('fp32_qkv_and_head', 'fixed_tile_qkv_fp32_head', 'fixed_tile_linears_fp32_head'):
            head = result['qkv_policy']['head']
            if head['emitted_rows'] != sum(len(record['output_tokens']) for record in result['records']):
                raise ValueError('Counterfactual rows do not match emitted window')
        result['qkv_policy']['graph_counts'] = engine.collective_rpc('graph_status')
        return result
    fix_holdout.capture_custom, fix_holdout.capture_vllm = custom_capture, vllm_capture
    original_args = None
    if config['implementation'] == 'vllm':
        import vllm
        original_args = vllm.EngineArgs
        def args_factory(**kwargs):
            kwargs['worker_extension_cls'] = 'experiments.qkv_policy_worker.QKVPolicyWorker'
            return original_args(**kwargs)
        vllm.EngineArgs = args_factory
    try:
        return await fix_holdout.run_workload(config, checkpoint)
    finally:
        engine_module.load_model = original_loader
        fix_holdout.capture_custom, fix_holdout.capture_vllm = original_capture_custom, original_capture_vllm
        if original_args is not None:
            vllm.EngineArgs = original_args


if __name__ == '__main__':
    config_file, output = map(Path, sys.argv[1:])
    if output.exists():
        raise SystemExit('Refusing to overwrite candidate evidence')
    result = asyncio.run(run(json.loads(config_file.read_text()), output))
    if not result['complete'] or result['error']:
        raise SystemExit('Candidate capture incomplete; evidence retained')
