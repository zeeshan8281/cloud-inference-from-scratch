"""Pinned shared ragged attention; installation evidence is not execution proof."""
import math
from types import MethodType

ATTENTION_ID = 'shared_ragged_paged_attention_v1'
_BUILDER = None


def metadata_rows(starts, seq_lens, computed, tokens):
    """Validate CPU scheduling metadata before any device launch; no CUDA reads."""
    if (len(starts) != len(seq_lens)+1 or len(computed) != len(seq_lens)
            or not starts or starts[0] != 0 or starts[-1] != tokens or tokens < 0):
        raise ValueError('invalid common attention query starts')
    rows = []
    for index, length in enumerate(seq_lens):
        count = starts[index+1]-starts[index]
        if (count <= 0 or length < count or length > 4096 or computed[index] != length-count):
            raise ValueError('unsupported common attention logical position bounds')
        rows.append((starts[index], starts[index+1], index, length-count))
    return rows


def prepare_metadata(builder, metadata, common):
    import numpy as np
    import torch
    starts = common.query_start_loc_cpu.tolist()
    lengths = common.seq_lens_cpu.tolist()
    computed = common.num_computed_tokens_cpu.tolist()
    tokens = common.num_actual_tokens
    rows = metadata_rows(starts, lengths, computed, tokens)
    if getattr(builder, 'use_full_cuda_graph', False):
        raise ValueError('pinned FA2 full-attention CUDA graphs remain unsupported')
    capacity = builder.vllm_config.scheduler_config.max_num_batched_tokens
    if (tokens > capacity or builder.block_size != 16 or metadata.num_actual_tokens != tokens
            or metadata.max_seq_len != max(lengths, default=0) or metadata.max_seq_len > 4096
            or metadata.block_table.shape[1]*16 < max(lengths, default=0)):
        raise ValueError('unsupported common attention capacity')
    buffers = getattr(builder, '_cloud_attention_buffers', None)
    if buffers is None:
        cpu_ids = torch.zeros(capacity, dtype=torch.int32, pin_memory=True)
        cpu_pos = torch.zeros(capacity, dtype=torch.int64, pin_memory=True)
        ids = torch.zeros(capacity, dtype=torch.int32, device=builder.device)
        positions = torch.zeros(capacity, dtype=torch.int64, device=builder.device)
        buffers = (cpu_ids, cpu_pos, ids, positions)
        builder._cloud_attention_buffers = buffers
    cpu_ids, cpu_pos, ids, positions = buffers
    if ids.numel() != capacity or ids.device != builder.device:
        raise ValueError('common attention persistent buffer contract changed')
    ids_np, pos_np = cpu_ids.numpy(), cpu_pos.numpy()
    for start, end, index, past in rows:
        ids_np[start:end] = index
        pos_np[start:end] = np.arange(past, past+end-start, dtype=np.int64)
    ids[:tokens].copy_(cpu_ids[:tokens], non_blocking=True)
    positions[:tokens].copy_(cpu_pos[:tokens], non_blocking=True)
    metadata.cloud_query_seq_ids = ids[:tokens]
    metadata.cloud_positions = positions[:tokens]
    metadata.cloud_bounds_validated = True
    return metadata


def install_metadata():
    global _BUILDER
    if _BUILDER is not None:
        return
    import vllm
    from vllm.v1.attention.backends.flash_attn import (
        FlashAttentionMetadataBuilder,
        get_flash_attn_version,
    )
    if vllm.__version__ != '0.10.0' or get_flash_attn_version() != 2:
        raise ValueError('common attention requires pinned vLLM FA2')
    original = FlashAttentionMetadataBuilder.build
    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        from .numerical_policy import vllm_scope
        metadata = original(self, common_prefix_len, common_attn_metadata, fast_build)
        if vllm_scope(self.model_config, self.parallel_config):
            return prepare_metadata(self, metadata, common_attn_metadata)
        return metadata
    FlashAttentionMetadataBuilder.build = build
    _BUILDER = (FlashAttentionMetadataBuilder, build, prepare_metadata, metadata_rows)


def impl_contract(impl):
    names = ('num_heads', 'num_kv_heads', 'head_size', 'scale', 'vllm_flash_attn_version',
             'alibi_slopes', 'sliding_window', 'logits_soft_cap', 'kv_sharing_target_layer_name', 'kv_cache_dtype')
    return tuple(getattr(impl, name) for name in names)


def validate_impl(impl):
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    if (type(impl) is not FlashAttentionImpl or impl.num_heads != 16 or impl.num_kv_heads != 2
            or impl.head_size != 128 or impl.scale != 128**-0.5
            or impl.vllm_flash_attn_version != 2 or impl.alibi_slopes is not None
            or impl.sliding_window != (-1, -1) or impl.logits_soft_cap != 0
            or impl.kv_sharing_target_layer_name is not None or impl.kv_cache_dtype not in ('auto','float16')):
        raise ValueError('unsupported pinned common attention implementation')
    if getattr(impl.forward, '__func__', None) is not FlashAttentionImpl.forward:
        raise ValueError('attention callable already replaced')


def validated_ragged_attention(*args):
    from .kernel import ragged_attention_prevalidated
    return ragged_attention_prevalidated(*args)


def _shared_forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None, output_scale=None):
    import torch
    from vllm.attention.utils.fa_utils import reshape_and_cache_flash
    if output is None or output_scale is not None:
        raise ValueError('common attention requires native output buffer without quantization')
    if attn_metadata is None:
        return output
    n = attn_metadata.num_actual_tokens
    if (getattr(attn_metadata, 'cloud_bounds_validated', False) is not True
            or query.dtype != torch.float16 or key.dtype != torch.float16 or value.dtype != torch.float16
            or output.dtype != torch.float16 or not query.is_cuda
            or query.ndim != 3 or tuple(query.shape[1:]) != (16,128) or query.shape[0] < n
            or key.ndim != 3 or tuple(key.shape[1:]) != (2,128) or key.shape[0] < n
            or value.shape != key.shape or output.shape != query.shape
            or any(t.device != query.device for t in (key, value, kv_cache, output))
            or attn_metadata.slot_mapping.numel() != n
            or kv_cache.dtype != torch.float16 or kv_cache.ndim != 5
            or tuple(kv_cache.shape[2:]) != (16,2,128) or kv_cache.shape[0] != 2
            or attn_metadata.cloud_positions.numel() != n or attn_metadata.cloud_query_seq_ids.numel() != n):
        raise ValueError('invalid actual shared attention operands/metadata')
    if n == 0:
        return output
    key_cache, value_cache = kv_cache.unbind(0)
    reshape_and_cache_flash(key, value, key_cache, value_cache, attn_metadata.slot_mapping,
                            self.kv_cache_dtype, layer._k_scale, layer._v_scale)
    result = validated_ragged_attention(query[:n], key_cache, value_cache, attn_metadata.block_table,
                                        attn_metadata.cloud_query_seq_ids, attn_metadata.cloud_positions, self.scale)
    output[:n].copy_(result)
    return output


def bind_impl(impl):
    """Call only after validating every model instance before any mutation."""
    impl.forward = MethodType(_shared_forward, impl)
    return (impl, impl.forward)


def validate_model(model):
    modules = [layer.self_attn.attn for layer in model.model.layers]
    if len(modules) != 36 or len({id(layer.impl) for layer in modules}) != 36:
        raise ValueError('requires 36 distinct actual attention implementations')
    for layer in modules:
        validate_impl(layer.impl)
    return modules


def install_model(modules):
    from . import kernel
    install_metadata()
    records = [(layer, layer.impl, bind_impl(layer.impl)[1], impl_contract(layer.impl)) for layer in modules]
    return {'implementation': 'vllm', 'records': records, 'kernel': kernel,
            'launch': kernel.ragged_attention_prevalidated, 'internal': kernel._ragged_attention_direct,
            'arithmetic': kernel._ragged_paged_attention_kernel, 'forward': _shared_forward,
            'adapter': validated_ragged_attention}


def custom_record(model):
    from .attention import RaggedTritonAttentionBackend
    if type(model.attn_backend) is not RaggedTritonAttentionBackend:
        return None  # Preserve other custom backends; strict validation rejects unsupported evidence.
    from . import kernel
    return {'implementation': 'custom', 'model': model, 'backend': model.attn_backend,
            'forward': model.attn_backend.attend.__func__, 'kernel': kernel,
            'launch': kernel.ragged_attention_direct, 'internal': kernel._ragged_attention_direct,
            'arithmetic': kernel._ragged_paged_attention_kernel}


def status(record):
    if record is None:
        return {'implementation': ATTENTION_ID, 'active': False, 'count': 0, 'expected_count': 36}
    kernel = record['kernel']
    active = (kernel._ragged_attention_direct is record['internal']
              and kernel._ragged_paged_attention_kernel is record['arithmetic'])
    if record['implementation'] == 'vllm':
        records = record['records']
        active = (active and len(records) == 36 and kernel.ragged_attention_prevalidated is record['launch']
                  and _shared_forward is record['forward'] and validated_ragged_attention is record['adapter']
                  and all(layer.impl is impl and impl.forward is function and impl_contract(impl) == contract
                          for layer, impl, function, contract in records))
        builder_active = (_BUILDER is not None and _BUILDER[0].build is _BUILDER[1]
                          and prepare_metadata is _BUILDER[2] and metadata_rows is _BUILDER[3])
        count = len(records)
    else:
        backend = record['backend']
        active = (active and record['model'].attn_backend is backend
                  and getattr(backend.attend, '__self__', None) is backend
                  and getattr(backend.attend, '__func__', None) is record['forward']
                  and kernel.ragged_attention_direct is record['launch']
                  and backend.allow_reference_fallback is False
                  and (backend.num_heads, backend.num_kv_heads, backend.head_dim, backend.sm_scale) == (16,2,128,1.0 / math.sqrt(128)))
        builder_active, count = None, 36
    return {'implementation': ATTENTION_ID, 'active': active, 'count': count, 'expected_count': 36,
            'metadata_active': builder_active, 'pinned_version': '0.10.0' if record['implementation'] == 'vllm' else None,
            'arithmetic': 'existing_ragged_paged_attention', 'block_size': 16, 'head_dim': 128}
