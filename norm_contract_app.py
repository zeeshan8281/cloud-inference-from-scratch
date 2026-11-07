"""Durable single-GPU native RMSNorm replay; never a full-model acceptance gate."""
import base64
import hashlib
import json
import os
import tempfile
from pathlib import Path

import modal

FILES = ('norm_contract_probe.py',)
if modal.is_local():
    from fix_holdout_app import capture_image
    image = capture_image.add_local_file('experiments/norm_contract_probe.py', '/root/experiments/norm_contract_probe.py', copy=True)
    image = image.add_local_file('norm_contract_app.py', '/root/norm_contract_app.py', copy=True)
else:
    image = None
app = modal.App('cie-normalization-A4-contract')
volume = modal.Volume.from_name('cloud-inference-model-cache', create_if_missing=False)
evidence_volume = modal.Volume.from_name('cie-retained-diagnostics', create_if_missing=True)


def source(root, deployed=False):
    from experiments.fix_holdout import computational_source
    files = ['norm_contract_app.py', 'experiments/norm_contract_probe.py']
    return {'core': computational_source(root, deployed=deployed),
            'diagnostic': {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}}


def check_fixture(fixture, expected):
    if fixture.get('schema') != 'captured-normalization-contract-v1' or set(fixture['states']) != {'custom', 'vllm'}:
        raise ValueError('incorrect historical fixture')
    for state in fixture['states'].values():
        if state['m'] != 1 or state['offset'] != 0:
            raise ValueError('historical production source/layout mismatch')
    if any(fixture['states']['custom']['input'][key] != fixture['states']['vllm']['input'][key]
           for key in ('shape', 'dtype', 'sha256', 'bytes_base64')):
        raise ValueError('historical operands differ')


def verify_historical(fixture):
    for implementation, state in fixture['states'].items():
        raw = Path(state['raw_path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != state['raw_sha256']:
            raise ValueError('historical raw changed')
        payload = json.loads(raw)
        for key in ('source_manifest', 'model', 'model_files_sha256', 'dependencies'):
            if state[key] != payload['environment'][key]:
                raise ValueError('fixture historical environment changed')
        operator = ('model.' if implementation == 'vllm' else '')+'layers.0.input_layernorm'
        records = [json.loads(line) for line in payload['raw_traces']['capture.batch.operators.jsonl'].splitlines()]
        for direction in ('input', 'output'):
            matches = [r for r in records if r.get('operator') == operator and r.get('direction') == direction
                       and r.get('target_position') == 513 and r.get('target_input_token') == 2310
                       and r.get('output_tokens_before') == [279, 2310]]
            if len(matches) != 1 or len(matches[0]['vectors']) != 1:
                raise ValueError('historical selected operand missing/ambiguous')
            row = matches[0]
            if row['packed_tokens'] != 1 or row['target_offset'] != 0:
                raise ValueError('historical layout mismatch')
            actual = row['vectors'][0]
            if any(actual[key] != state[direction][key] for key in ('shape', 'dtype', 'sha256', 'bytes_base64')):
                raise ValueError('fixture operand differs from actual historical trace')
    state = fixture['residual']
    raw = Path(state['raw_path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != state['raw_sha256']:
        raise ValueError('residual historical raw changed')
    payload = json.loads(raw)
    records = [json.loads(line) for line in payload['raw_traces']['capture.batch.operators.jsonl'].splitlines()]
    selected = [r for r in records if r.get('operator') == 'model.layers.0.post_attention_layernorm'
                and r.get('target_position') == 513 and r.get('target_input_token') == 2310
                and r.get('output_tokens_before') == [279,2310]]
    if len(selected) != 2 or selected != state['records']:
        raise ValueError('residual operands differ from actual trace')
    pair = fixture['failed_pair']
    raw = Path(pair['raw_path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != pair['raw_sha256']:
        raise ValueError('historical failed pair changed')
    actual = json.loads(raw)
    if pair['mode'] != 'resource_normalized' or pair['index'] != 1:
        raise ValueError('incorrect historical failed pair selection')
    if hashlib.sha256(json.dumps(pair['workload'],sort_keys=True,separators=(',',':')).encode()).hexdigest() != pair['workload_sha256']:
        raise ValueError('historical workload checksum changed')
    for child in actual['children'].values():
        if child['capture']['config']['workload'] != pair['workload']:
            raise ValueError('workload differs from actual failed pair')




def validate_payload(payload, config, expected, fixture, fixture_sha):
    from cloud_engine.numerical_policy import POLICY_ID
    result, env = payload['capture'], payload['environment']
    if json.loads(payload['raw_checkpoint']) != result:
        raise ValueError('raw checkpoint differs from parsed result')
    proc = payload['process']
    if proc.get('returncode') != 0 or proc.get('timeout') or proc.get('cancelled'):
        raise ValueError('child process failed')
    if result.get('schema') != 'normalization-contract-v1' or result.get('complete') is not True or result.get('error') is not None:
        raise ValueError('normalization contract incomplete')
    if result.get('config') != config or result.get('fixture_sha256') != fixture_sha or result.get('policy_id') != POLICY_ID:
        raise ValueError('fixture/config/policy mismatch')
    if env.get('metadata_error') or env['source_manifest'] != expected:
        raise ValueError('executed source/metadata mismatch')
    if not env['gpu_before'].get('uuid') or env['gpu_before']['uuid'] != env['gpu_after']['uuid']:
        raise ValueError('physical GPU identity missing/changed')
    for state in fixture['states'].values():
        for key in ('model', 'model_files_sha256', 'dependencies'):
            if env[key] != state[key]:
                raise ValueError('historical runtime/model mismatch')
    for name, path in (('policy','src/cloud_engine/numerical_policy.py'),('attention','src/cloud_engine/attention.py'),('opaque','src/cloud_engine/opaque_normalization.py')):
        if result['source_sha256'][name] != expected['core']['files'][path]:
            raise ValueError('micro actual imported source mismatch')
    if result.get('operator') != 'cloud_engine::common_rms_norm' or result.get('installed_callable') != '_norm_forward(opaque_normalization.opaque_rms_norm)':
        raise ValueError('actual opaque normalization binding missing')
    if not result['actual_weight_files'] or any(env['model_files_sha256'].get(name) != value for name,value in result['actual_weight_files'].items()):
        raise ValueError('micro actual weight file mismatch')
    precision = result['precision']
    if (precision.get('allow_fp16_reduced_precision_reduction') is not False
            or precision.get('allow_tf32') is not False or precision.get('float32_matmul_precision') != 'highest'):
        raise ValueError('micro precision policy mismatch')
    pure_bytes = base64.b64decode(fixture['states']['custom']['output']['bytes_base64'],validate=True)
    wanted = {(kind,m) for kind in ('pure','residual') for m in (0,1,16,256,512)}
    actual = set()
    for row in result['cases']:
        key = (row['kind'],row['m'])
        if key in actual:
            raise ValueError('duplicate normative case')
        actual.add(key)
        for name in ('eager_exact_and_aliases','compiled_fullgraph_exact_and_aliases','poisoned_graph_exact_and_aliases'):
            if row.get(name) is not True:
                raise ValueError('normative output/alias/rounding contract failed')
        for name in ('eager_sha256','compiled_sha256','graph_sha256','expected_sha256'):
            value = row[name]
            if not isinstance(value,str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('invalid normative digest')
        if row['kind'] == 'pure' and row['expected_sha256'] != hashlib.sha256(pure_bytes*row['m']).hexdigest():
            raise ValueError('pure result differs from actual historical output')
        for name in ('eager_sha256','compiled_sha256','graph_sha256'):
            if row[name] != row['expected_sha256']:
                raise ValueError('normative output checksum disagreement')
    if actual != wanted or result.get('gate_cleared') is not False:
        raise ValueError('missing normative cases or inappropriate gate clearance')


def child(config, fixture_bytes, model):
    import sys

    import torch

    from experiments.parallel_validation import run_cancellable_process
    from experiments.sentinel_pilot import gpu_state_snapshot, resolved_dependency_versions
    root = Path(tempfile.mkdtemp(prefix='rmsnorm-replay-'))
    (root/'input.json').write_bytes(fixture_bytes)
    config = dict(config, fixture_path=str(root/'input.json'))
    (root/'config.json').write_text(json.dumps(config))
    env = {'source_manifest': source(Path('/root'), True), 'source': os.environ.get('SOURCE_COMMIT'),
           'model': model, 'dependencies': resolved_dependency_versions(),
           'gpu_before': gpu_state_snapshot(torch.version.cuda), 'model_files_sha256': {}}
    for path in sorted(Path(config['model_dir']).rglob('*')):
        if path.is_file() and path.suffix in ('.json', '.safetensors', '.model', '.txt'):
            checksum = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                    checksum.update(chunk)
            env['model_files_sha256'][path.relative_to(config['model_dir']).as_posix()] = checksum.hexdigest()
    process = run_cancellable_process([sys.executable, '/root/experiments/norm_contract_probe.py', str(root/'config.json'), str(root/'capture.json')],
        env=dict(os.environ, PYTHONPATH='/root'), stdout_path=root/'stdout.log', stderr_path=root/'stderr.log',
        cancelled=lambda: False, timeout_seconds=1800)
    checkpoint = (root/'capture.json').read_text() if (root/'capture.json').exists() else None
    try:
        capture = json.loads(checkpoint) if checkpoint else {'complete': False, 'error': 'missing checkpoint'}
    except ValueError as exc:
        capture = {'complete': False, 'error': str(exc)}
    payload = {'environment': env, 'process': process, 'capture': capture, 'raw_checkpoint': checkpoint, 'config': config}
    try:
        env['gpu_after'] = gpu_state_snapshot(torch.version.cuda)
    except Exception as exc:
        env['metadata_error'] = str(exc)
    return payload


@app.function(image=image, gpu='L4', volumes={'/cache': volume, '/evidence': evidence_volume}, timeout=7200,
              startup_timeout=180, max_containers=1, retries=0)
def campaign(campaign_id, fixture_bytes, expected):
    import re

    from cloud_engine.weights import ensure_weights_downloaded
    if re.fullmatch('[0-9a-f]{32}', campaign_id) is None:
        raise ValueError('invalid campaign ID')
    root = Path('/evidence')/campaign_id
    root.mkdir(exist_ok=False)
    index = {'campaign_id': campaign_id, 'complete': False, 'error': None, 'artifacts': {}, 'gate_cleared': False,
             'scope': 'normalization eager/compiled/graph contract; no full-model validation'}
    def commit():
        (root/'index.tmp').write_text(json.dumps(index, allow_nan=False))
        (root/'index.tmp').replace(root/'index.json')
        evidence_volume.commit()
    def retain(name, raw):
        path = root/name
        path.write_bytes(raw)
        index['artifacts'][name] = hashlib.sha256(raw).hexdigest()
        commit()
    retain('input.json', fixture_bytes)
    try:
        fixture = json.loads(fixture_bytes)
        check_fixture(fixture, expected)
        pinned = json.loads(Path('/root/engine_config.json').read_text())
        model = pinned['ragged_model']
        if model != fixture['states']['custom']['model']:
            raise ValueError('pinned model differs from historical model')
        model_dir = str(ensure_weights_downloaded('/cache', model['revision'], model['id']))
        payload = child({'model_dir': model_dir}, fixture_bytes, model)
        retain('micro.json', json.dumps(payload, allow_nan=False).encode())
        validate_payload(payload, payload['config'], expected, fixture, hashlib.sha256(fixture_bytes).hexdigest())
        from experiments.parallel_validation import CANCEL_STORE
        from fix_holdout_app import capture_pair
        store = modal.Dict.from_name(CANCEL_STORE, create_if_missing=True)
        if store.put(campaign_id, False, skip_if_exists=True) is not True:
            raise ValueError('pair cancellation key already exists; no resubmission')
        index['active_stage'] = 'failed-pair-full128'
        commit()
        pair = capture_pair.local('resource_normalized', 1, fixture['failed_pair']['workload'],
                                  campaign_id, expected['core']['sha256'], None)
        retain('pair.json', json.dumps(pair, allow_nan=False).encode())
        if pair.get('stop') is not None or pair.get('error') or set(pair['children']) != {'custom','vllm'}:
            raise ValueError('full128 historical pair failed; retained')
        if pair.get('summary',{}).get('passes') is not True:
            raise ValueError('full128 historical pair did not pass')
        for child_payload in pair['children'].values():
            child_env = child_payload['environment']
            for key in ('model','model_files_sha256','dependencies'):
                if child_env[key] != payload['environment'][key]:
                    raise ValueError('pair and micro runtime/model mismatch')
            if child_env['computational_source'] != expected['core']:
                raise ValueError('pair computational source mismatch')
            for point in ('gpu_before','gpu_after'):
                if child_env[point]['uuid'] != payload['environment']['gpu_before']['uuid']:
                    raise ValueError('pair and micro physical GPU differs')
        index['complete'] = True
    except BaseException as exc:
        index['error'] = f'{type(exc).__name__}: {exc}'
    commit()
    return index


def recover_artifacts(destination, campaign_id):
    from qkv_candidate_app import recover_artifacts as recover
    return recover(destination, campaign_id)


def main(output, fixture_path, recover=False):
    import uuid
    destination = Path(output)
    if recover:
        dispatch = json.loads((destination/'dispatch.json').read_text())
        call = modal.FunctionCall.from_id(dispatch['call_id'])
    else:
        if destination.exists():
            raise ValueError('refusing existing output or resubmission')
        fixture_bytes = Path(fixture_path).read_bytes()
        fixture = json.loads(fixture_bytes)
        expected = source(Path(__file__).resolve().parent)
        check_fixture(fixture, expected)
        verify_historical(fixture)
        destination.mkdir(parents=True)
        campaign_id = uuid.uuid4().hex
        with (destination/'intent.json').open('x') as stream:
            json.dump({'campaign_id': campaign_id, 'source': expected, 'fixture_sha256': hashlib.sha256(fixture_bytes).hexdigest()}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        with app.run(detach=True):
            if source(Path(__file__).resolve().parent) != expected:
                raise ValueError('source changed during image build')
            call = campaign.spawn(campaign_id, fixture_bytes, expected)
            dispatch = {'campaign_id': campaign_id, 'call_id': call.object_id, 'app_id': app.app_id, 'source': expected}
            with (destination/'dispatch.json').open('x') as stream:
                json.dump(dispatch, stream)
                stream.flush()
                os.fsync(stream.fileno())
    try:
        call.get()
    except BaseException:
        try:
            recover_artifacts(destination, dispatch['campaign_id'])
        except Exception as recovery_error:
            print(f'Recovery pending: {recovery_error}; recover saved call without resubmission', flush=True)
        raise
    index = recover_artifacts(destination, dispatch['campaign_id'])
    if not index.get('complete') or index.get('error'):
        raise SystemExit('native replay incomplete; raw retained, no retry or gate clearance')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--fixture', default='input.json')
    parser.add_argument('--recover', action='store_true')
    args = parser.parse_args()
    main(args.output, args.fixture, args.recover)
