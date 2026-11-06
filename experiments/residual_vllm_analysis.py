"""Analyze retained residual traces; never clears the pilot or holdout gate."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def delta(left, right):
    if len(left) != len(right) or not left:
        raise ValueError('Vector dimensions missing or unequal')
    if not all(math.isfinite(value) for value in left + right):
        raise ValueError('Nonfinite vector')
    errors = [abs(a - b) for a, b in zip(left, right, strict=True)]
    return {'different_components': sum(error != 0 for error in errors),
            'components': len(errors), 'max_abs': max(errors)}


def aligned(left, right):
    for key in ('request_id', 'target_token_position', 'output_tokens_before', 'target_input_token', 'prompt_length'):
        if left[key] != right[key]:
            raise ValueError(f'Unaligned target context: {key}')


def operator_difference(left, right):
    """Only the first differing observed output, never a global causal origin."""
    def flatten(calls):
        return [(call, vector) for call in calls for vector in call['vectors']]
    a, b = flatten(left), flatten(right)
    if not a or len(a) != len(b):
        raise ValueError('Missing or unequal operator coverage')
    first = None
    for (lc, lv), (rc, rv) in zip(a, b, strict=True):
        aligned(lc, rc)
        if (lc['module'], lv['path'], lv['dtype']) != (rc['module'], rv['path'], rv['dtype']):
            raise ValueError('Operator order/path/dtype differs')
        difference = delta(lv['values'], rv['values'])
        if first is None and difference['different_components']:
            first = {'module': lc['module'], 'path': lv['path'], **difference,
                     'left_shape': lv['shape'], 'right_shape': rv['shape']}
    return {'observed_outputs': len(a), 'first_observed_difference': first}


def projection_difference(left, right):
    aligned(left, right)
    if left['hidden_dtype'] != right['hidden_dtype'] or left['logits_dtype'] != right['logits_dtype']:
        raise ValueError('Projection dtype differs')
    a = dict(zip(left['candidate_ids'], left['native_logits'], strict=True))
    b = dict(zip(right['candidate_ids'], right['native_logits'], strict=True))
    candidates = sorted(set(a) & set(b))
    def ranking(record, values):
        ordered = sorted(values.items(), key=lambda item: item[1], reverse=True)
        if len(ordered) < 2 or record['native_argmax'] not in values:
            raise ValueError('Missing native argmax/top-two support')
        return {'native_argmax': record['native_argmax'], 'top_two': ordered[:2],
                'top_two_margin': ordered[0][1] - ordered[1][1]}
    return {'hidden': delta(left['hidden'], right['hidden']),
            'candidate_intersection': candidates,
            'candidate_logits': delta([a[i] for i in candidates], [b[i] for i in candidates]),
            'left': ranking(left, a), 'right': ranking(right, b),
            'left_packed_tokens': left['total_num_scheduled_tokens'],
            'right_packed_tokens': right['total_num_scheduled_tokens']}


def analyze(directory):
    runs, environments = {}, []
    for selection in ('batch', 'solo-2', 'solo-6'):
        for variant in ('control', 'trace', 'repeat'):
            payload = json.loads((directory / f'vllm-{selection}-{variant}.json').read_text())
            if payload['process']['returncode'] != 0:
                raise ValueError('Failed child process')
            run = json.loads(payload['files']['output.json'])
            if run.get('complete') is not True or run.get('error') is not None:
                raise ValueError('Incomplete child')
            states = run['observed_precision']
            if len(states) != 1 or states[0]['allow_fp16_reduced_precision_reduction'] is not False:
                raise ValueError('Actual fixed precision policy missing')
            runs[selection, variant] = run
            environments.append(payload['environment'])
    gpu_ids = {env[time]['uuid'] for env in environments for time in ('gpu_before', 'gpu_after')}
    if len(gpu_ids) != 1 or any(env['source_hashes'] != environments[0]['source_hashes'] for env in environments):
        raise ValueError('GPU identity or executed source differs')
    reproduction, comparisons = [], []
    for selection in ('batch', 'solo-2', 'solo-6'):
        control, trace, repeat = [runs[selection, variant] for variant in ('control', 'trace', 'repeat')]
        for target, result in control['results'].items():
            reproduction.append({'selection': selection, 'request_id': target,
                                 'control_tokens': result['output_tokens'],
                                 'trace_tokens': trace['results'][target]['output_tokens'],
                                 'repeat_tokens': repeat['results'][target]['output_tokens'],
                                 'tokens_equal': result['output_tokens'] == trace['results'][target]['output_tokens'] == repeat['results'][target]['output_tokens']})
    for index in (2, 6):
        target = f'diag-{index}'
        for variant in ('trace', 'repeat'):
            batch, solo = runs['batch', variant], runs[f'solo-{index}', variant]
            if batch['config']['batch_ids'][index] != solo['config']['batch_ids'][index]:
                raise ValueError('Prompt identity changed')
            emitted = [[call for call in run['projection_calls'] if call['request_id'] == target and call['emits_token']] for run in (batch, solo)]
            if any(len(calls) != 2 for calls in emitted):
                raise ValueError('Expected two emitted target steps')
            for position, (left, right) in enumerate(zip(*emitted, strict=True)):
                entry = {'request_id': target, 'variant': variant, 'output_position': position}
                if left['output_tokens_before'] != right['output_tokens_before']:
                    entry.update(comparable=False, reason='generated prefixes differ')
                else:
                    operators = [[call for call in run['operator_calls'] if call['request_id'] == target and call['target_token_position'] == left['target_token_position'] and call['output_tokens_before'] == left['output_tokens_before']] for run in (batch, solo)]
                    entry.update(comparable=True, projection=projection_difference(left, right),
                                 operators=operator_difference(*operators))
                comparisons.append(entry)
    return {'gpu_uuid': next(iter(gpu_ids)), 'reproduction': reproduction, 'comparisons': comparisons,
            'gate_cleared': False, 'scope': 'exposed calibration two-token diagnostic; first observed target-row difference only',
            'limits': ['Layer-zero last-row traces cannot locate the first difference across all positions/layers.',
                       'Matching control tokens does not prove unchanged scheduling or zero observer effect.',
                       'Two-token admission can differ from the 128-token calibration smoke.',
                       'Fresh holdout has not been evaluated by this analysis.']}


if __name__ == '__main__':
    print(json.dumps(analyze(Path(sys.argv[1])), indent=2, allow_nan=False))
