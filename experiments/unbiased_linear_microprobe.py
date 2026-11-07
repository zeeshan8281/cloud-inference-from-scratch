"""Replay actual17 unbiased-linear rows; peer rows are explicit repeated inputs.

CONFIG_JSON OUTPUT_JSON: model_dir, fixture_path. No production model changes.
"""
from __future__ import annotations

import base64
import hashlib
import json
import sys
from pathlib import Path

WEIGHTS = {'down_proj': 'model.layers.0.mlp.down_proj.weight',
           'o_proj': 'model.layers.0.self_attn.o_proj.weight'}


def replay_layouts(states):
    actual = {(state['M'],state['target_offset']) for state in states}
    if any(m < 1 or not 0 <= offset < m for m,offset in actual):
        raise ValueError('invalid observed layout')
    return sorted(actual | {(m,offset) for m in (1,8,16,128,256,2048) for offset in (0,m-1)})


def run(config, output):
    import torch
    import torch.nn.functional as functional
    from safetensors import safe_open

    from cloud_engine import batch_invariant_linear as kernel_module
    from cloud_engine.precision import configure_inference_precision

    if output.exists():
        raise ValueError('refusing evidence overwrite')
    source = Path(config['fixture_path'])
    fixture = json.loads(source.read_text())
    if (fixture.get('schema') != 'captured-unbiased-linear-replay-input-v1'
            or {case['name'] for case in fixture['cases']} != set(WEIGHTS)
            or len(fixture['cases']) != 2):
        raise ValueError('requires actual17 two-operator fixture')
    result = {'schema':'fixed-unbiased-linear-microprobe-v1','complete':False,'error':None,
              'fixture_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'fixture_provenance':{key:value for key,value in fixture.items() if key != 'cases'},
              'kernel_policy':kernel_module.linear_kernel_policy(),'operators':{},
              'scope':'actual captured target input and native output; all peer rows explicitly repeat that input, not original peer values'}

    def save():
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(result,allow_nan=False))
        temporary.replace(output)

    def decode(vector):
        raw = base64.b64decode(vector['bytes_base64'],validate=True)
        if vector['dtype'] != 'torch.float16' or hashlib.sha256(raw).hexdigest() != vector['sha256']:
            raise ValueError('actual captured row checksum/dtype mismatch')
        value = torch.frombuffer(bytearray(raw),dtype=torch.float16).clone().cuda()
        if value.numel() != vector['row_shape'][0] or not bool(torch.isfinite(value).all()):
            raise ValueError('actual row shape/nonfinite mismatch')
        return value

    def encode(value):
        value = value.detach().contiguous().cpu()
        if not bool(torch.isfinite(value).all()):
            raise ValueError('nonfinite replay data')
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        return {'shape':list(value.shape),'dtype':str(value.dtype),'sha256':hashlib.sha256(raw).hexdigest(),
                'bytes_base64':base64.b64encode(raw).decode()}

    def metric(actual, reference):
        delta = (actual.double()-reference.double()).abs()
        return {'max_abs':float(delta.max()),'mean_abs':float(delta.mean()),
                'changed_elements':int((actual.double()!=reference.double()).sum())}

    save()
    try:
        configure_inference_precision()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        result['precision'] = {'allow_fp16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                               'allow_tf32':torch.backends.cuda.matmul.allow_tf32,'float32_matmul_precision':torch.get_float32_matmul_precision()}
        result['runtime'] = {'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name()}
        result['source_sha256'] = {str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in (Path(__file__),Path(kernel_module.__file__))}
        parameters, model_hashes = {}, {}
        for path in sorted(Path(config['model_dir']).rglob('*.safetensors')):
            with safe_open(path,framework='pt',device='cpu') as handle:
                keys = set(handle.keys()) & set(WEIGHTS.values())
                if not keys:
                    continue
                relative = path.relative_to(config['model_dir']).as_posix()
                checksum = hashlib.sha256()
                with path.open('rb') as file:
                    for chunk in iter(lambda:file.read(8*1024**2),b''):
                        checksum.update(chunk)
                model_hashes[relative] = checksum.hexdigest()
                if fixture['environment']['model_files_sha256'].get(relative) != model_hashes[relative]:
                    raise ValueError('actual weight file differs from source capture')
                for key in keys:
                    if key in parameters:
                        raise ValueError('duplicate model parameter')
                    parameters[key] = handle.get_tensor(key).to(device='cuda',dtype=torch.float16).contiguous()
        if set(parameters) != set(WEIGHTS.values()):
            raise ValueError('missing actual pinned weights')
        result['actual_weight_file_hashes'] = model_hashes
        for case in fixture['cases']:
            name = case['name']
            if case['weight_name'] != WEIGHTS[name] or case['bias'] is not None or len(case['states']) != 2:
                raise ValueError('unexpected operator/bias fixture')
            rows = [decode(state['input']) for state in case['states']]
            if not all(torch.equal(rows[0],value) for value in rows):
                raise ValueError('captured compared input rows differ')
            row,weight = rows[0],parameters[WEIGHTS[name]]
            if weight.shape != (2048,row.numel()):
                raise ValueError('actual weight/input shape mismatch')
            reference = functional.linear(row.double(),weight.double())
            weight_record = encode(weight)
            weight_record.pop('bytes_base64')
            entry = {'weight_name':WEIGHTS[name],'weight':weight_record,'frozen_input':encode(row),
                     'fp64_reference':encode(reference),'baseline_native_matches':{},'layouts':[],'execution':[]}
            result['operators'][name] = entry
            for state in case['states']:
                matrix = row.repeat(state['M'],1)
                native = functional.linear(matrix,weight)[state['target_offset']]
                captured = decode(state['actual_output'])
                entry['baseline_native_matches'][state['phase']] = {'m':state['M'],'offset':state['target_offset'],
                    'exact':torch.equal(native,captured),'difference':metric(native,captured),
                    'captured':encode(captured),'replay':encode(native)}
            save()
            if not all(value['exact'] for value in entry['baseline_native_matches'].values()):
                raise ValueError('FP16 reducedFalse baseline did not reproduce captured native row')
            for m,offset in replay_layouts(case['states']):
                matrix = row.repeat(m,1)
                baseline = functional.linear(matrix,weight)[offset]
                candidate = kernel_module.fixed_tile_linear(matrix,weight)[offset]
                repeated = kernel_module.fixed_tile_linear(matrix,weight)[offset]
                entry['layouts'].append({'m':m,'offset':offset,'baseline':encode(baseline),'candidate':encode(candidate),
                    'repeat_exact':torch.equal(candidate,repeated),'baseline_fp64_error':metric(baseline,reference),
                    'candidate_fp64_error':metric(candidate,reference),'candidate_rounded_fp64_error':metric(candidate,reference.half())})
            entry['all_layouts_exact'] = len({item['candidate']['sha256'] for item in entry['layouts']}) == 1
            baseline_worst = max(item['baseline_fp64_error']['max_abs'] for item in entry['layouts'])
            candidate_worst = max(item['candidate_fp64_error']['max_abs'] for item in entry['layouts'])
            entry['accuracy'] = {'baseline_worst_max_abs':baseline_worst,'candidate_worst_max_abs':candidate_worst,
                                 'no_worse':candidate_worst<=baseline_worst,'added_tolerance':0}
            compiled = torch.compile(kernel_module.fixed_tile_linear,fullgraph=True,dynamic=True)
            for m in (16,256):
                matrix = row.repeat(m,1)
                eager = kernel_module.fixed_tile_linear(matrix,weight)
                compiled_output = compiled(matrix,weight)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        kernel_module.fixed_tile_linear(matrix,weight)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graphed = kernel_module.fixed_tile_linear(matrix,weight)
                graphed.fill_(float('nan'))
                graph.replay()
                torch.cuda.synchronize()
                entry['execution'].append({'m':m,'compiled_fullgraph_exact':torch.equal(eager,compiled_output),
                                           'cuda_graph_replay_exact':torch.equal(eager,graphed)})
            save()
        entries = list(result['operators'].values())
        result['baseline_reproduces_capture'] = all(v['exact'] for entry in entries for v in entry['baseline_native_matches'].values())
        result['candidate_layouts_exact'] = all(entry['all_layouts_exact'] and all(v['repeat_exact'] for v in entry['layouts']) for entry in entries)
        result['accuracy_no_worse'] = all(entry['accuracy']['no_worse'] for entry in entries)
        result['execution_exact'] = all(v['compiled_fullgraph_exact'] and v['cuda_graph_replay_exact'] for entry in entries for v in entry['execution'])
        result['complete'] = True
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    save()
    return result


if __name__ == '__main__':
    config_file,output = map(Path,sys.argv[1:])
    if not run(json.loads(config_file.read_text()),output)['complete']:
        raise SystemExit('unbiased microprobe incomplete; retain checkpoint')
