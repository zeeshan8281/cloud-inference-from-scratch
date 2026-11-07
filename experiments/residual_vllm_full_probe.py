"""128-token fixed-candidate vLLM control/observer on exposed batch and target2."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path


def validate_config(config):
    if (config.get('trace_case') not in ('control', 'trace', 'repeat')
            or config.get('candidate') != 'fixed_tile_qkv_fp32_head'
            or config.get('mode') != 'resource_normalized' or config.get('implementation') != 'vllm'
            or config.get('target_index') != 2 or config.get('output_tokens') != 128):
        raise ValueError('requires fixed-candidate resource full128 target2 diagnostic')
    batch = config['workload']['input_token_ids']
    if (len(batch) != 8 or any(len(ids) != 512 for ids in batch)
            or config['workload'].get('output_tokens') != 128
            or config['workload'].get('concurrency') != 8):
        raise ValueError('requires exact exposed c8/512 workload')
    return batch


def run(config, output):
    batch = validate_config(config)
    if output.exists():
        raise ValueError('refusing checkpoint overwrite')
    os.environ['CIE_DIAGNOSTIC_CANDIDATE'] = config['candidate']
    os.environ['RESIDUAL_FULL_OBSERVE'] = '0' if config['trace_case'] == 'control' else '1'
    result = {'schema': 'full-residual-vllm-v1', 'complete': False, 'error': None,
              'config': config, 'cases': {}, 'scope': 'all128 outputs; all-layer target last-row observations bounded to first2 generated positions, not global earliest divergence'}

    def save():
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, allow_nan=False))
        temporary.replace(output)

    save()
    try:
        import torch
        import vllm

        from cloud_engine.precision import configure_inference_precision
        from experiments.fix_holdout import (
            capture_vllm,
            observed_kv_capacity,
            reset_prefixes,
            validate_capture,
            validate_kv_capacity,
        )
        from experiments.sentinel_pilot import _build_vllm_engine, resolve_kv_plan

        configure_inference_precision()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        original_args = vllm.EngineArgs
        def args_factory(**kwargs):
            kwargs['worker_extension_cls'] = ('experiments.attention_replay_hook.FullAttentionWorker' if config.get('attention_probe')
                                              else 'experiments.residual_vllm_full_hook.FullResidualWorker')
            return original_args(**kwargs)
        vllm.EngineArgs = args_factory
        plan = resolve_kv_plan(config['model_dir'], 16, config['kv_bytes'])
        try:
            engine = _build_vllm_engine(config['model_dir'], config['model'], 'resource_normalized',
                                       plan)
        finally:
            vllm.EngineArgs = original_args
        result['kv_plan'] = asdict(plan)
        result['kv_capacity'] = observed_kv_capacity(engine, 'vllm')
        save()
        validate_kv_capacity(result['kv_capacity'], 'resource_normalized', result['kv_plan'])
        root = Path(__file__).resolve().parent
        result['diagnostic_source_hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                              for p in (root/'residual_vllm_probe.py', Path(__file__), root/'residual_vllm_full_hook.py')}
        if config.get('attention_probe'):
            result['diagnostic_source_hashes'].update({name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                for name in ('attention_replay.py','attention_replay_hook.py','attention_replay_probe.py')})
        for selection, ids, target in (('batch', batch, 2), ('solo', [batch[2]], 0)):
            run_id = 'residual-'+selection
            trace_path = output.with_name(output.stem+'.'+selection+'.operators.jsonl')
            reset_prefixes(engine, 'vllm')
            engine.collective_rpc('residual_trace_reset', args=(f'{run_id}-{target}', str(trace_path)))
            before = engine.collective_rpc('qkv_policy_status')
            capture = capture_vllm(engine, ids, 128, run_id)
            record = {'capture': capture, 'target_local_index': target, 'trace_path': str(trace_path),
                      'candidate_policy_before': before}
            result['cases'][selection] = record
            save()
            evidence = engine.collective_rpc('residual_trace_evidence')
            policy = engine.collective_rpc('qkv_policy_status')
            record['candidate_policy'] = policy
            if len(evidence) != 1 or len(policy) != 1 or len(before) != 1:
                raise ValueError('expected exactly one actual worker')
            record.update(evidence[0])
            raw = trace_path.read_text()
            if hashlib.sha256(raw.encode()).hexdigest() != record['trace_sha256']:
                raise ValueError('worker/driver trace checksum mismatch')
            record['operator_calls'] = [json.loads(line) for line in raw.splitlines() if line]
            if config.get('attention_probe'):
                attention_path = trace_path.with_suffix('.attention.jsonl')
                attention_text = attention_path.read_text() if attention_path.exists() else ''
                record['attention_path'] = str(attention_path)
                record['attention_sha256'] = hashlib.sha256(attention_text.encode()).hexdigest()
                record['attention_calls'] = [json.loads(line) for line in attention_text.splitlines() if line]
            save()
            validate_capture(capture, ids, 128, cold=True, mode='resource_normalized')
            state = policy[0]
            capture['qkv_policy'] = state
            capture['forward_traces'] = record['schedules']
            if (not state['wrapper_active'] or state['module_count'] != state['expected_module_count']
                    or state['module_count'] <= 0 or not state['fixed_tile_policy']['operator_active']
                    or not state['head']['wrapper_active'] or state['head']['output_dtypes'] != ['torch.float32']
                    or state['head']['emitted_rows'] != len(ids)*128 or not state['head']['all_finite']):
                raise ValueError('actual candidate policy/evidence missing')
            if state['precision'] != before[0]['precision']:
                raise ValueError('candidate arithmetic policy changed')
            if record['trace_calls'] != len(record['operator_calls']):
                raise ValueError('trace count mismatch')
            observing = config['trace_case'] != 'control'
            if record['observe'] != observing or bool(record['operator_calls']) != observing:
                raise ValueError('observer/control trace mismatch')
            if config.get('attention_probe'):
                attention = record['attention_calls']
                if len(attention) != int(observing):
                    raise ValueError('wrong bounded attention capture count')
                if observing and not all(attention[0].get(key) is True for key in (
                        'past_kv_unchanged','new_kv_matches_input','native_replay_exact',
                        'full_replay_repeat_exact','counterfactual_executed','target_inputs_fixed','singleton_repeat_exact','singleton_preserve_global_max_repeat_exact','singleton_max2_repeat_exact')):
                    raise ValueError('attention replay prerequisites failed')
            if observing:
                emitted = [row for row in record['operator_calls'] if row['operator'] == 'lm_head' and row['emits_token']]
                tokens = capture['records'][target]['output_tokens']
                if len(emitted) != 2 or [row['argmax'] for row in emitted] != tokens[:2]:
                    raise ValueError('bounded observed logits do not match actual emitted first2 tokens')
                if [row['output_tokens_before'] for row in emitted] != [[], tokens[:1]]:
                    raise ValueError('bounded target histories disagree')
            save()
        result['complete'] = True
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    save()
    return result
