"""Diagnostic18: observe one actual layer0 decode and replay frozen attention."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

from experiments import attention_replay
from experiments import residual_vllm_full_hook as full

_original = FlashAttentionImpl.forward
_seen = set()


def forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None, output_scale=None):
    context = full.context()
    selected = (context is not None and full._layer == 0 and context['target_position'] == 512
                and len(context['output_tokens_before']) == 1)
    if not selected:
        return _original(self,layer,query,key,value,kv_cache,attn_metadata,output,output_scale)
    identity = (full._path,context['request_id'])
    if identity in _seen:
        raise ValueError('duplicate bounded attention capture')
    _seen.add(identity)
    trace = Path(full._path).with_suffix('.attention.jsonl')
    try:
        if output_scale is not None:
            raise ValueError('output quantization unsupported in diagnostic')
        frozen = attention_replay.freeze_inputs(self,layer,query,key,value,kv_cache,attn_metadata,output,context)
        native = _original(self,layer,query,key,value,kv_cache,attn_metadata,output,output_scale)
        record = attention_replay.replay(self,layer,kv_cache,output,_original,frozen,context)
        import vllm.v1.attention.backends.flash_attn as backend
        import vllm.vllm_flash_attn.flash_attn_interface as interface
        record['installed_source_hashes'] = {str(Path(module.__file__)): hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                                             for module in (interface,backend)}
        with trace.open('a') as handle:
            handle.write(json.dumps(record,allow_nan=False)+'\n')
        if not all(record.get(key) is True for key in ('past_kv_unchanged','new_kv_matches_input',
                   'native_replay_exact','full_replay_repeat_exact','counterfactual_executed',
                   'target_inputs_fixed','singleton_repeat_exact','singleton_preserve_global_max_repeat_exact','singleton_max2_repeat_exact')):
            raise ValueError('attention frozen replay integrity prerequisite failed')
        return native
    except Exception as exc:
        with trace.open('a') as handle:
            handle.write(json.dumps({'capture_error':f'{type(exc).__name__}: {exc}',**context})+'\n')
        raise


class FullAttentionWorker(full.FullResidualWorker):
    """Same candidate/holdout collector; optional bounded attention replay."""


if full._observe:
    FlashAttentionImpl.forward = forward
