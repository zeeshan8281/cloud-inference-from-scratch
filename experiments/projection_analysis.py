"""Compare actual projection captures at the first generated position only."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def target_capture(payload):
    if payload['exit_code'] or not payload.get('capture'):
        raise ValueError('Probe did not complete')
    capture = payload['capture']
    if capture['result'].get('verified') is not True:
        raise ValueError('Unverified token capture')
    tokens = capture['result']['output_tokens']
    calls = capture['projection_calls']
    index = capture['config']['target_index']
    worker_rows = any('request_id' in row for call in calls for row in call['rows'])
    if worker_rows:
        if any('request_id' not in row for call in calls for row in call['rows']):
            raise ValueError('Mixed identified and positional projection rows')
        rows = [row for call in calls for row in call['rows'] if row['request_id'] == f'diag-{index}']
    else:
        rows = [call['rows'][index] for call in calls[-len(tokens):]]
    if not tokens or len(rows) != len(tokens):
        raise ValueError('Missing aligned projection calls')
    if [row['native_argmax'] for row in rows] != tokens:
        raise ValueError('Captured projection rows do not match emitted tokens')
    for position, row in enumerate(rows):
        if worker_rows and row.get('output_tokens_before') != tokens[:position]:
            raise ValueError('Captured row is not on the emitted shared prefix')
        ids = row['candidate_ids']
        arrays = ['native_logits', 'fp32_projection']
        if 'permuted_argmax' in row:
            arrays.append('permuted_logits')
        if not ids or len(set(ids)) != len(ids) or not {323, 1401, row['native_argmax']} <= set(ids):
            raise ValueError('Invalid or missing candidate IDs')
        if not row['hidden'] or any(len(row[key]) != len(ids) for key in arrays):
            raise ValueError('Invalid projection dimensions')
        if not all(math.isfinite(v) for key in ['hidden', *arrays] for v in row[key]):
            raise ValueError('Nonfinite projection evidence')
        if row['native_logits'][ids.index(row['native_argmax'])] != max(row['native_logits']):
            raise ValueError('Selected token is not maximal on captured support')
    return rows[0]


def compare(left, right):
    if not left['gpu'].get('uuid') or left['gpu'].get('uuid') != right['gpu'].get('uuid'):
        raise ValueError('Physical GPU changed')
    a, b = target_capture(left), target_capture(right)
    for field in ('probe_sha256', 'hook_sha256'):
        if field in left or field in right:
            if not left.get(field) or left.get(field) != right.get(field):
                raise ValueError('Probe source differs')
    configs = [payload['capture']['config'] for payload in (left, right)]
    if configs[0].get('model') != configs[1].get('model'):
        raise ValueError('Model configuration differs')
    prompts = [config['batch_ids'][config['target_index']] for config in configs]
    if prompts[0] != prompts[1]:
        raise ValueError('Target prompts differ')
    if len(a['hidden']) != len(b['hidden']):
        raise ValueError('Hidden shapes differ')
    result = {'gpu_uuid': left['gpu']['uuid'],
              'hidden_exact_equal': a['hidden'] == b['hidden'],
              'hidden_max_abs_difference': max(abs(x-y) for x, y in zip(a['hidden'], b['hidden'], strict=True)),
              'alignment': 'request ID and emitted prefix' if 'request_id' in a and 'request_id' in b else 'legacy positional tail; no request-ID trace',
              'scope': 'first output position, identical prompt, candidate-only FP32 re-projection'}
    for name, row in [('original', a), ('target_first', b)]:
        ids = row['candidate_ids']
        result[name] = {'native_argmax': row['native_argmax'],
                        'native_1401_minus_323': row['native_logits'][ids.index(1401)] - row['native_logits'][ids.index(323)],
                        'fp32_1401_minus_323': row['fp32_projection'][ids.index(1401)] - row['fp32_projection'][ids.index(323)]}
        if 'permuted_argmax' in row:
            result[name]['permuted_argmax'] = row['permuted_argmax']
            result[name]['permuted_candidate_logits_equal'] = row['native_logits'] == row['permuted_logits']
    return result


if __name__ == '__main__':
    root = Path(sys.argv[1])
    result = compare(*(json.loads((root/f'{variant}.json').read_text()) for variant in ('original', 'target_first')))
    (root/'projection-summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
