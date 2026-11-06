"""Scoped numerical policy draft. Installation evidence is not execution validation."""
from __future__ import annotations

import copy
from types import MethodType

POLICY_ID = 'fixed_tile_linears_fp32_head_fa2_min_query2_v1'
_STATE_ATTRIBUTE = '_cloud_numerical_policy_installation'
_FLOOR = None


def pinned_dimensions(config, custom=False):
    mapping = {'hidden_size': 2048, 'intermediate_size': 11008, 'vocab_size': 151936,
               'num_layers' if custom else 'num_hidden_layers': 36,
               'num_heads' if custom else 'num_attention_heads': 16,
               'num_kv_heads' if custom else 'num_key_value_heads': 2,
               'tie_word_embeddings': True}
    return all(getattr(config, name, None) == value for name, value in mapping.items())


def vllm_scope(model_config, parallel_config):
    config = model_config.hf_config
    return (pinned_dimensions(config) and getattr(config, 'model_type', None) == 'qwen2'
            and parallel_config.tensor_parallel_size == 1
            and getattr(model_config, 'quantization', None) is None
            and str(getattr(model_config, 'dtype', None)) == 'torch.float16')


def floor_metadata(metadata, eligible):
    """Only change the legal query-length upper bound, never real query offsets."""
    if eligible and metadata.num_actual_tokens > 0 and metadata.max_query_len == 1:
        metadata.max_query_len = 2
    return metadata


def metadata_floor_status():
    if _FLOOR is None:
        return {'installed': False, 'active': False, 'pinned_version': None, 'fa_version': None}
    builder, wrapper, version, fa_version = _FLOOR
    return {'installed': True, 'active': builder.build is wrapper,
            'pinned_version': version, 'fa_version': fa_version}


def install_pinned_fa2_metadata_policy():
    global _FLOOR
    if _FLOOR is not None:
        return metadata_floor_status()
    import vllm
    from vllm.v1.attention.backends.flash_attn import (
        FlashAttentionMetadataBuilder,
        get_flash_attn_version,
    )
    version = vllm.__version__
    fa_version = get_flash_attn_version()
    if version != '0.10.0' or fa_version != 2:
        return {'installed': False, 'active': False, 'pinned_version': version, 'fa_version': fa_version}
    original = FlashAttentionMetadataBuilder.build

    def build(self, *args, **kwargs):
        metadata = original(self, *args, **kwargs)
        eligible = vllm_scope(self.model_config, self.parallel_config)
        return floor_metadata(metadata, eligible)

    FlashAttentionMetadataBuilder.build = build
    _FLOOR = (FlashAttentionMetadataBuilder, build, version, fa_version)
    return metadata_floor_status()


def _unsupported(implementation, reason):
    return {'installed': False, 'policy_id': POLICY_ID, 'implementation': implementation,
            'reason': reason, 'metadata_floor': metadata_floor_status()}


def _validate_parameter(module, shape, biased, torch):
    weight = module.weight
    bias = getattr(module, 'bias', None)
    if (tuple(weight.shape) != shape or weight.dtype != torch.float16 or not weight.is_cuda
            or not weight.is_contiguous() or ((bias is not None) != biased)):
        raise ValueError('pinned module weight/bias contract mismatch')
    if bias is not None and (tuple(bias.shape) != (shape[0],) or bias.dtype != torch.float16
                             or bias.device != weight.device or not bias.is_contiguous()):
        raise ValueError('pinned bias storage mismatch')


def _record(model, implementation, wrappers, parameters, expected, links=()):
    # The model owns its record. No global value may keep a GPU model alive.
    setattr(model, _STATE_ATTRIBUTE, (implementation, wrappers, parameters, expected, links))
    return policy_status(model)


def policy_status(model):
    state = getattr(model, _STATE_ATTRIBUTE, None)
    if state is None:
        return _unsupported('unknown', 'policy not installed on this model')
    implementation, wrappers, parameters, expected, links = state
    active = (all(getattr(owner, name) is function for owner, name, function in wrappers)
              and all(owner.quant_method is method for owner, method in links))
    preserved = all(getattr(owner, name) is parameter and str(parameter.dtype) == 'torch.float16'
                    for owner, name, parameter in parameters)
    return {'installed': True, 'policy_id': POLICY_ID, 'implementation': implementation,
            'linears_count': len(wrappers)-1, 'expected_linears': expected,
            'head_installed': active, 'wrappers_active': active, 'parameters_preserved': preserved,
            'storage_dtype': 'torch.float16', 'output_dtype': 'torch.float32',
            'metadata_floor': metadata_floor_status(),
            'evidence_scope': 'actual callable and parameter identity; execution/dtypes validated by capture'}


def install_custom_policy(model):
    if hasattr(model, _STATE_ATTRIBUTE):
        return policy_status(model)
    import torch

    from .batch_invariant_linear import fixed_tile_head, fixed_tile_linear
    if not pinned_dimensions(model.dims, custom=True) or not model.dims.attention_bias:
        return _unsupported('custom', 'outside pinned Qwen3B dimensions/bias scope')
    if not hasattr(model, 'project_logits'):
        raise ValueError('custom model lacks explicit output projection seam')
    head = model.embed_tokens if model.lm_head is None else model.lm_head
    if head.weight.dtype != torch.float16 or not head.weight.is_cuda:
        return _unsupported('custom', 'policy requires FP16 CUDA model')
    if model.lm_head is not None:
        raise ValueError('pinned custom output weight must be tied')
    shapes = {'self_attn.q_proj': ((2048,2048), True), 'self_attn.k_proj': ((256,2048), True),
              'self_attn.v_proj': ((256,2048), True), 'self_attn.o_proj': ((2048,2048), False),
              'mlp.gate_proj': ((11008,2048), False), 'mlp.up_proj': ((11008,2048), False),
              'mlp.down_proj': ((2048,11008), False)}
    selected = []
    for layer in model.layers:
        for path, (shape, biased) in shapes.items():
            parent, name = path.split('.')
            module = getattr(getattr(layer, parent), name)
            if not isinstance(module, torch.nn.Linear):
                return _unsupported('custom', 'quantized or unsupported linear module')
            _validate_parameter(module, shape, biased, torch)
            selected.append(module)
    if len(selected) != 252:
        raise ValueError('custom layer count mismatch')
    _validate_parameter(head, (151936,2048), False, torch)
    from .precision import configure_numerical_policy_precision
    configure_numerical_policy_precision()
    wrappers, parameters = [], []
    for module in selected:
        def forward(x, module=module):
            return fixed_tile_linear(x, module.weight, module.bias)
        module.forward = forward
        wrappers.append((module, 'forward', forward))
        parameters.append((module, 'weight', module.weight))
        if module.bias is not None:
            parameters.append((module, 'bias', module.bias))
    def project_logits(hidden):
        return fixed_tile_head(hidden, head.weight)
    model.project_logits = project_logits
    wrappers.append((model, 'project_logits', project_logits))
    parameters.append((head, 'weight', head.weight))
    return _record(model, 'custom', wrappers, parameters, 252)


def install_vllm_policy(model, model_config, parallel_config):
    if hasattr(model, _STATE_ATTRIBUTE):
        return policy_status(model)
    import torch
    import vllm
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

    from .batch_invariant_linear import fixed_tile_head, fixed_tile_linear
    if vllm.__version__ != '0.10.0' or not vllm_scope(model_config, parallel_config):
        return _unsupported('vllm', 'outside pinned unquantized Qwen3B TP1 scope')
    head = model.lm_head
    if head is not model.model.embed_tokens or not isinstance(head.quant_method, UnquantizedEmbeddingMethod):
        raise ValueError('pinned actual tied output projection mismatch')
    if head.weight.dtype != torch.float16 or not head.weight.is_cuda:
        return _unsupported('vllm', 'policy requires FP16 CUDA model')
    shapes = {'self_attn.qkv_proj': ((2560,2048), True), 'self_attn.o_proj': ((2048,2048), False),
              'mlp.gate_up_proj': ((22016,2048), False), 'mlp.down_proj': ((2048,11008), False)}
    selected = []
    for layer in model.model.layers:
        for path, (shape, biased) in shapes.items():
            parent, name = path.split('.')
            module = getattr(getattr(layer, parent), name)
            if not isinstance(module.quant_method, UnquantizedLinearMethod) or module.skip_bias_add is not False:
                raise ValueError('unsupported quantized or deferred-bias projection')
            _validate_parameter(module, shape, biased, torch)
            selected.append(module)
    if len(selected) != 144:
        raise ValueError('vLLM layer count mismatch')
    _validate_parameter(head, (151936,2048), False, torch)
    from .precision import configure_numerical_policy_precision
    configure_numerical_policy_precision()
    wrappers, parameters, links = [], [], []
    for module in selected:
        method = copy.copy(module.quant_method)
        def apply(self, layer, x, bias=None):
            return fixed_tile_linear(x, layer.weight, bias)
        bound = MethodType(apply, method)
        method.apply = bound
        module.quant_method = method
        links.append((module, method))
        wrappers.extend([(method, 'apply', bound)])
        parameters.append((module, 'weight', module.weight))
        if module.bias is not None:
            parameters.append((module, 'bias', module.bias))
    method = copy.copy(head.quant_method)
    def apply_head(self, layer, hidden, bias=None):
        if bias is not None:
            raise ValueError('pinned output head has no bias')
        return fixed_tile_head(hidden, layer.weight)
    bound = MethodType(apply_head, method)
    method.apply = bound
    head.quant_method = method
    links.append((head, method))
    wrappers.append((method, 'apply', bound))
    parameters.append((head, 'weight', head.weight))
    install_pinned_fa2_metadata_policy()
    return _record(model, 'vllm', wrappers, parameters, 144, links)
