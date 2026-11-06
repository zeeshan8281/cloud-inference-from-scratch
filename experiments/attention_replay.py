"""Frozen pinned-FA2 replay helpers, used only by diagnostic18.

Snapshots contain live logical slots only; unused cache capacity is never
serialized. Exact full-layout replay is a prerequisite for counterfactuals.
"""
from __future__ import annotations

import base64
import hashlib
from dataclasses import replace


def tensor_layout(tensor):
    return {'shape': list(tensor.shape), 'stride': list(tensor.stride()),
            'storage_offset': tensor.storage_offset(), 'dtype': str(tensor.dtype),
            'pointer_mod256': tensor.data_ptr() % 256}


def clone_layout(tensor, *, zeros=False):
    import torch
    strides = tensor.stride()
    if any(s < 0 for s in strides):
        raise ValueError('negative stride unsupported')
    span = tensor.storage_offset()+1+sum((n-1)*s for n,s in zip(tensor.shape,strides,strict=True))
    if span*tensor.element_size() > 256*1024**2:
        raise ValueError('bounded one-layer replay backing exceeds256MiB')
    backing = torch.empty((span,), device=tensor.device, dtype=tensor.dtype)
    result = backing.as_strided(tensor.shape, strides, tensor.storage_offset())
    if zeros:
        result.zero_()
    else:
        result.copy_(tensor)
    if result.data_ptr()%256 != tensor.data_ptr()%256:
        raise ValueError('cannot preserve input pointer alignment')
    return result


def encode(tensor):
    import torch
    value = tensor.detach().contiguous().cpu()
    if not bool(torch.isfinite(value).all()):
        raise ValueError('nonfinite captured tensor')
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': hashlib.sha256(raw).hexdigest(),
            'bytes_base64': base64.b64encode(raw).decode()}


def logical(cache, blocks, length):
    import torch
    positions = torch.arange(length, device=cache.device)
    physical = blocks[positions//cache.shape[2]].long()
    return cache[:, physical, positions%cache.shape[2], :, :]


def query_plan(starts, lengths, target, total):
    if (len(starts) != len(lengths)+1 or starts[0] != 0 or starts[-1] != total
            or not 0 <= target < len(lengths)):
        raise ValueError('actual query ranges mismatch')
    counts = [b-a for a,b in zip(starts[:-1],starts[1:],strict=True)]
    if any(count <= 0 or count > length for count,length in zip(counts,lengths,strict=True)):
        raise ValueError('invalid query lengths')
    if counts[target] != 1:
        raise ValueError('target must decode one token; other requests may prefill')
    return starts[target], counts


def live_page_plan(tables, lengths, block_size, capacity):
    used = {}
    for blocks,length in zip(tables,lengths,strict=True):
        count = (length+block_size-1)//block_size
        if length <= 0 or len(blocks) != count:
            raise ValueError('incomplete live block table')
        for j,block in enumerate(blocks):
            if not 0 <= block < capacity:
                raise ValueError('live page outside actual cache')
            used[block] = max(used.get(block,0),min(block_size,length-j*block_size))
    return used


def freeze_inputs(impl, layer, query, key, value, cache, metadata, output, context):
    import torch
    if (impl.vllm_flash_attn_version != 2 or metadata.use_cascade or metadata.scheduler_metadata is not None
            or impl.kv_sharing_target_layer_name is not None or impl.alibi_slopes is not None
            or tuple(impl.sliding_window) != (-1,-1) or impl.logits_soft_cap != 0
            or impl.kv_cache_dtype.startswith('fp8')):
        raise ValueError('probe only supports actual nonquantized noncascade FA2 without window/alibi/softcap')
    if cache.dtype != torch.float16 or query.dtype != torch.float16 or cache.shape[0] != 2:
        raise ValueError('expected FP16 Q/pagedKV')
    starts = metadata.query_start_loc.cpu().tolist()
    lengths = metadata.seq_lens.cpu().tolist()
    n = len(context['req_ids'])
    if len(lengths) != n:
        raise ValueError('metadata request count mismatch')
    target = context['req_ids'].index(context['request_id'])
    target_offset, query_counts = query_plan(starts,lengths,target,metadata.num_actual_tokens)
    if (starts != context['query_starts'] or target_offset != context['target_offset']
            or metadata.num_actual_tokens != context['packed_tokens']):
        raise ValueError('attention metadata disagrees with observed worker packing')
    if context['target_position'] != 512 or lengths[target] != 513 or context['output_tokens_before'] != [279]:
        raise ValueError('unexpected target decode identity')
    blocks = metadata.block_table
    block_size = cache.shape[2]
    counts = [(length+block_size-1)//block_size for length in lengths]
    table = clone_layout(blocks, zeros=True)
    tables = [blocks[i,:count].cpu().tolist() for i,count in enumerate(counts)]
    used = live_page_plan(tables,lengths,block_size,cache.shape[1])
    slots = metadata.slot_mapping.cpu().tolist()
    for i,(length,count) in enumerate(zip(lengths,counts,strict=True)):
        table[i,:count].copy_(blocks[i,:count])
        for packed, position in zip(range(starts[i],starts[i+1]),range(length-query_counts[i],length),strict=True):
            expected_slot = tables[i][position//block_size]*block_size+position%block_size
            if slots[packed] != expected_slot:
                raise ValueError('actual append slot does not match block ownership')
    prior = logical(cache,table[target],context['computed_tokens']).clone()
    meta = replace(metadata, query_start_loc=clone_layout(metadata.query_start_loc),
                   seq_lens=clone_layout(metadata.seq_lens), block_table=table,
                   slot_mapping=clone_layout(metadata.slot_mapping))
    return {'query': clone_layout(query), 'key': clone_layout(key), 'value': clone_layout(value),
            'metadata': meta, 'target': target, 'target_offset': target_offset, 'used': used, 'prior': prior,
            'layouts': {name: tensor_layout(tensor) for name,tensor in
                        (('query',query),('key',key),('value',value),('kv_cache',cache),('output',output),
                         ('block_table',blocks),('query_start_loc',metadata.query_start_loc),('seq_lens',metadata.seq_lens),('slot_mapping',metadata.slot_mapping))},
            'scales': {name: encode(getattr(layer,name)) for name in ('_q_scale','_k_scale','_v_scale')}}


def errors(value, reference):
    difference = (value.double()-reference.double()).abs()
    return {'max_abs': float(difference.max()), 'mean_abs': float(difference.mean()),
            'changed_elements': int((value.double() != reference.double()).sum())}


def replay(impl, layer, cache, output, original, frozen, context):
    import torch
    meta, target, used = frozen['metadata'], frozen['target'], frozen['used']
    q,k,v = frozen['query'], frozen['key'], frozen['value']
    offset = frozen['target_offset']
    lengths = meta.seq_lens.cpu().tolist()
    full_target = logical(cache,meta.block_table[target],lengths[target]).clone()
    prior_after = full_target[:,:context['computed_tokens']]
    past_equal = torch.equal(frozen['prior'],prior_after)
    new_equal = torch.equal(full_target[0,-1],k[offset]) and torch.equal(full_target[1,-1],v[offset])
    if not past_equal or not new_equal:
        raise ValueError('native append changed past slots or failed to store incoming newKV')
    # Reconstruct original-shaped/strided storage, populate only live slots.
    # Tail slots in referenced pages and all unused pages are zero, not read out.
    frozen_cache = clone_layout(cache,zeros=True)
    for block,count in used.items():
        frozen_cache[:,block,:count].copy_(cache[:,block,:count])
    live_ids = sorted(used)
    compact = frozen_cache[:,live_ids].clone()
    native = output[:meta.num_actual_tokens].clone()
    full_out = clone_layout(output,zeros=True)
    full_out.fill_(float('nan'))
    original(impl,layer,q,k,v,frozen_cache,meta,output=full_out)
    exact = torch.equal(native,full_out[:meta.num_actual_tokens])
    record = {**context, 'layer_index': 0, 'actual_fa_version': impl.vllm_flash_attn_version,
              'backend_class': type(impl).__module__+'.'+type(impl).__name__,
              'source_scope': 'pinned installed backend; file hashes recorded by worker',
              'layouts': frozen['layouts'], 'precision_scales': frozen['scales'],
              'metadata': {'num_actual_tokens': meta.num_actual_tokens, 'max_query_len': meta.max_query_len,
                           'max_seq_len': meta.max_seq_len, 'max_num_splits': meta.max_num_splits,
                           'query_start_loc': meta.query_start_loc.cpu().tolist(), 'seq_lens': lengths,
                           'block_table_live': [meta.block_table[i,:(length+cache.shape[2]-1)//cache.shape[2]].cpu().tolist() for i,length in enumerate(lengths)],
                           'block_table_shape': list(meta.block_table.shape), 'slot_mapping': meta.slot_mapping.cpu().tolist(),
                           'use_cascade': meta.use_cascade, 'scheduler_metadata': None, 'causal': True,
                           'scale': impl.scale, 'window': list(impl.sliding_window), 'softcap': impl.logits_soft_cap},
              'all_active_query': encode(q[:meta.num_actual_tokens]),
              'all_active_new_key': encode(k[:meta.num_actual_tokens]),
              'all_active_new_value': encode(v[:meta.num_actual_tokens]),
              'live_page_ids': live_ids, 'valid_slots_per_live_page': [used[i] for i in live_ids],
              'live_pages': encode(compact), 'unused_page_contents_serialized': False,
              'target_query': encode(q[offset]), 'target_kv_before': encode(frozen['prior']),
              'target_kv_after': encode(full_target), 'target_new_key': encode(k[offset]), 'target_new_value': encode(v[offset]),
              'past_kv_unchanged': past_equal, 'new_kv_matches_input': new_equal,
              'native_output': encode(native), 'native_target_output': encode(native[offset]),
              'full_replay_output': encode(full_out[:meta.num_actual_tokens]), 'native_replay_exact': exact,
              'counterfactual_executed': False}
    if not exact:
        return record
    full_out.fill_(float('nan'))
    original(impl,layer,q,k,v,frozen_cache,meta,output=full_out)
    record['full_replay_repeat_exact'] = torch.equal(native,full_out[:meta.num_actual_tokens])
    if not record['full_replay_repeat_exact']:
        return record
    # Preserve target view's strides/offset/alignment; change membership only.
    solo_q,solo_k,solo_v = (clone_layout(t[offset:offset+1]) for t in (q,k,v))
    solo_out = clone_layout(output[offset:offset+1],zeros=True)
    solo_table = clone_layout(meta.block_table[target:target+1])
    solo_meta = replace(meta,num_actual_tokens=1,max_query_len=1,max_seq_len=lengths[target],
                        query_start_loc=torch.tensor([0,1],dtype=meta.query_start_loc.dtype,device=q.device),
                        seq_lens=clone_layout(meta.seq_lens[target:target+1]),block_table=solo_table,
                        slot_mapping=clone_layout(meta.slot_mapping[offset:offset+1]))
    retained_max_meta = replace(solo_meta,max_query_len=meta.max_query_len,max_seq_len=meta.max_seq_len)
    solo_out.fill_(float('nan'))
    original(impl,layer,solo_q,solo_k,solo_v,frozen_cache,retained_max_meta,output=solo_out)
    retained_max_output = solo_out.clone()
    solo_out.fill_(float('nan'))
    original(impl,layer,solo_q,solo_k,solo_v,frozen_cache,retained_max_meta,output=solo_out)
    retained_repeat = torch.equal(retained_max_output,solo_out)
    max2_meta = replace(solo_meta,max_query_len=2)
    solo_out.fill_(float('nan'))
    original(impl,layer,solo_q,solo_k,solo_v,frozen_cache,max2_meta,output=solo_out)
    max2_output = solo_out.clone()
    solo_out.fill_(float('nan'))
    original(impl,layer,solo_q,solo_k,solo_v,frozen_cache,max2_meta,output=solo_out)
    max2_repeat = torch.equal(max2_output,solo_out)
    solo_out.fill_(float('nan'))
    original(impl,layer,solo_q,solo_k,solo_v,frozen_cache,solo_meta,output=solo_out)
    first_solo = solo_out.clone()
    solo_out.fill_(float('nan'))
    original(impl,layer,solo_q,solo_k,solo_v,frozen_cache,solo_meta,output=solo_out)
    frozen_after = logical(frozen_cache,solo_table[0],lengths[target])
    invariant_inputs = torch.equal(full_target,frozen_after) and torch.equal(q[offset],solo_q[0])
    if not invariant_inputs:
        raise ValueError('counterfactual changed targetQ or logicalKV')
    query64 = q[offset].double()
    keys64,values64 = (full_target[i].double().repeat_interleave(query64.shape[0]//full_target.shape[2],dim=1) for i in (0,1))
    probabilities = torch.softmax(torch.einsum('hd,shd->hs',query64,keys64)*impl.scale,dim=-1)
    reference = torch.einsum('hs,shd->hd',probabilities,values64)
    native_target = native[offset].reshape_as(reference)
    singleton_target = solo_out[0].reshape_as(reference)
    record.update(counterfactual_executed=True, target_inputs_fixed=invariant_inputs,
                  singleton_repeat_exact=torch.equal(first_solo,solo_out),
                  singleton_max2_output=encode(max2_output[0]), singleton_max2_repeat_exact=max2_repeat,
                  singleton_max2_equals_native=torch.equal(native[offset],max2_output[0]),
                  singleton_max2_fp64_error=errors(max2_output[0].reshape_as(reference),reference),
                  singleton_preserve_global_max_output=encode(retained_max_output[0]),
                  singleton_preserve_global_max_repeat_exact=retained_repeat,
                  singleton_preserve_global_max_equals_native=torch.equal(native[offset],retained_max_output[0]),
                  singleton_preserve_global_max_vs_native=errors(retained_max_output[0].reshape_as(reference),native_target),
                  singleton_preserve_global_max_fp64_error=errors(retained_max_output[0].reshape_as(reference),reference),
                  singleton_max_query_len=1, singleton_max_seq_len=lengths[target],
                  singleton_output=encode(solo_out[0]), singleton_equals_native=torch.equal(native[offset],solo_out[0]),
                  singleton_vs_native=errors(singleton_target,native_target),
                  fp64_reference=encode(reference), native_fp64_error=errors(native_target,reference),
                  singleton_fp64_error=errors(singleton_target,reference),
                  singleton_layouts={name:tensor_layout(t) for name,t in (('query',solo_q),('key',solo_k),('value',solo_v),('output',solo_out),('block_table',solo_table))},
                  scope='frozen exact native replay, then membership counterfactual with identical targetQ/fullKV; does not measure actual native split count')
    return record
