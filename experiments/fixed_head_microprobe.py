"""Actual16 head-row replay; repeated peers are synthetic, historical scores partial."""
from __future__ import annotations

import base64
import hashlib
import json
import sys
from pathlib import Path

WEIGHT = 'model.embed_tokens.weight'
VOCAB = 151936


def replay_layouts(states):
    actual = {(state['M'], state['projection_row']) for state in states}
    if actual != {(5, 2), (1, 0)}:
        raise ValueError('requires actual16 head layouts M5/row2 and M1/row0')
    return sorted(actual | {(m, offset) for m in (1, 5, 8, 16, 32) for offset in (0, m-1)})


def validate_fixture(fixture):
    if fixture.get('schema') != 'captured-fp32-head-replay-input-v1':
        raise ValueError('requires actual16 head fixture')
    states = fixture['states']
    replay_layouts(states)
    if len(states) != 2 or {s['selection'] for s in states} != {'batch', 'solo'}:
        raise ValueError('requires both actual layouts')
    for state in states:
        identity = state['head_identity']
        if (len(state['observed_token_ids']) != 20 or len(set(state['observed_token_ids'])) != 20
                or any(not 0 <= token < VOCAB for token in state['observed_token_ids'])
                or identity['head_class'] != 'VocabParallelEmbedding'
                or identity['tied_input_embedding'] is not True
                or identity['weight_dtype'] != 'torch.float16'
                or identity['wrapper_active'] is not True):
            raise ValueError('unexpected observed head identity or score subset')
    if fixture['pinned_model_contract']['expected_checkpoint_weight'] != WEIGHT:
        raise ValueError('unexpected checkpoint weight')


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
    validate_fixture(fixture)
    result = {'schema': 'fixed-fp32-head-microprobe-v1', 'complete': False, 'error': None,
              'fixture_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'fixture_provenance': {k: v for k, v in fixture.items() if k != 'states'},
              'kernel_policy': kernel_module.head_kernel_policy(),
              'historical_reproduction': [], 'layouts': [], 'execution': [],
              'scope': 'Only 20 historical scores were captured. Full vocabulary replay is new evidence. All peer rows repeat the actual target hidden row and are synthetic.'}

    def save():
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, allow_nan=False))
        temporary.replace(output)

    def decode(vector, dtype, size):
        raw = base64.b64decode(vector['bytes_base64'], validate=True)
        if vector['dtype'] != str(dtype) or hashlib.sha256(raw).hexdigest() != vector['sha256']:
            raise ValueError('captured vector checksum/dtype mismatch')
        value = torch.frombuffer(bytearray(raw), dtype=dtype).clone().cuda()
        if value.numel() != size or not bool(torch.isfinite(value).all()):
            raise ValueError('captured vector size/finiteness mismatch')
        return value

    def encode(value):
        value = value.detach().contiguous().cpu()
        if not bool(torch.isfinite(value).all()):
            raise ValueError('nonfinite replay data')
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        return {'shape': list(value.shape), 'dtype': str(value.dtype),
                'sha256': hashlib.sha256(raw).hexdigest(), 'bytes_base64': base64.b64encode(raw).decode()}

    def metric(actual, reference):
        delta = (actual.double()-reference.double()).abs()
        return {'max_abs': float(delta.max()), 'mean_abs': float(delta.mean()),
                'changed_elements': int((delta != 0).sum())}

    save()
    try:
        configure_inference_precision()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        result['precision'] = {'allow_fp16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                               'allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                               'float32_matmul_precision': torch.get_float32_matmul_precision()}
        result['runtime'] = {'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name()}
        result['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (Path(__file__), Path(kernel_module.__file__))}
        weight = None
        result['actual_weight_file_hashes'] = {}
        for path in sorted(Path(config['model_dir']).rglob('*.safetensors')):
            with safe_open(path, framework='pt', device='cpu') as handle:
                if WEIGHT not in handle.keys():
                    continue
                if weight is not None:
                    raise ValueError('duplicate tied embedding parameter')
                relative = path.relative_to(config['model_dir']).as_posix()
                checksum = hashlib.sha256()
                with path.open('rb') as file:
                    for chunk in iter(lambda: file.read(8*1024**2), b''):
                        checksum.update(chunk)
                actual_hash = checksum.hexdigest()
                if fixture['environment']['model_files_sha256'].get(relative) != actual_hash:
                    raise ValueError('weight file differs from historical capture')
                result['actual_weight_file_hashes'][relative] = actual_hash
                weight = handle.get_tensor(WEIGHT).to(device='cuda', dtype=torch.float16).contiguous()
        if weight is None or tuple(weight.shape) != (VOCAB, 2048):
            raise ValueError('actual pinned head weight missing or shape mismatch')
        rows = [decode(s['hidden'], torch.float16, 2048) for s in fixture['states']]
        if not torch.equal(rows[0], rows[1]):
            raise ValueError('historical final hidden rows differ')
        row = rows[0]
        result['frozen_input'] = encode(row)
        # Hash the actual stored operands, without serializing the 622 MB parameter.
        weight_cpu = weight.cpu()
        result['weight'] = {'name': WEIGHT, 'shape': list(weight.shape), 'dtype': str(weight.dtype),
                            'sha256': hashlib.sha256(weight_cpu.numpy().tobytes()).hexdigest()}
        del weight_cpu
        for state in fixture['states']:
            matrix = row.repeat(state['M'], 1)
            native = functional.linear(matrix.float(), weight.float())[state['projection_row']]
            captured = decode(state['observed_fp32_logits'], torch.float32, 20)
            subset = native[state['observed_token_ids']]
            result['historical_reproduction'].append({'selection': state['selection'], 'm': state['M'],
                'offset': state['projection_row'], 'token_ids': state['observed_token_ids'],
                'captured': encode(captured), 'replay': encode(subset), 'exact': torch.equal(subset, captured)})
        result['baseline_reproduces_capture'] = all(v['exact'] for v in result['historical_reproduction'])
        save()
        if not result['baseline_reproduces_capture']:
            raise ValueError('native FP32 replay did not reproduce all historical scores')
        reference = torch.empty(VOCAB, device='cuda', dtype=torch.float64)
        for start in range(0, VOCAB, 4096):
            reference[start:start+4096] = functional.linear(row.double(), weight[start:start+4096].double())
        result['full_fp64_reference'] = encode(reference)
        for m, offset in replay_layouts(fixture['states']):
            matrix = row.repeat(m, 1)
            baseline = functional.linear(matrix.float(), weight.float())[offset]
            candidate = kernel_module.fixed_tile_head(matrix, weight)[offset]
            repeated = kernel_module.fixed_tile_head(matrix, weight)[offset]
            if candidate.dtype != torch.float32:
                raise ValueError('candidate must retain FP32 output')
            result['layouts'].append({'m': m, 'offset': offset, 'baseline': encode(baseline),
                'candidate': encode(candidate), 'repeat_exact': torch.equal(candidate, repeated),
                'baseline_fp64_error': metric(baseline, reference), 'candidate_fp64_error': metric(candidate, reference)})
        result['candidate_layouts_exact'] = len({v['candidate']['sha256'] for v in result['layouts']}) == 1 and all(v['repeat_exact'] for v in result['layouts'])
        baseline_worst = max(v['baseline_fp64_error']['max_abs'] for v in result['layouts'])
        candidate_worst = max(v['candidate_fp64_error']['max_abs'] for v in result['layouts'])
        result['accuracy'] = {'baseline_worst_max_abs': baseline_worst, 'candidate_worst_max_abs': candidate_worst, 'added_tolerance': 0}
        result['accuracy_no_worse'] = candidate_worst <= baseline_worst
        save()
        compiled = torch.compile(kernel_module.fixed_tile_head, fullgraph=True, dynamic=True)
        for m in (1, 5, 32):
            matrix = row.repeat(m, 1)
            eager = kernel_module.fixed_tile_head(matrix, weight)
            compiled_output = compiled(matrix, weight)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    kernel_module.fixed_tile_head(matrix, weight)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graphed = kernel_module.fixed_tile_head(matrix, weight)
            graphed.fill_(float('nan'))
            graph.replay()
            torch.cuda.synchronize()
            result['execution'].append({'m': m, 'compiled_fullgraph_exact': torch.equal(eager, compiled_output),
                                        'cuda_graph_replay_exact': torch.equal(eager, graphed)})
        result['execution_exact'] = all(v['compiled_fullgraph_exact'] and v['cuda_graph_replay_exact'] for v in result['execution'])
        result['complete'] = True
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    save()
    return result


if __name__ == '__main__':
    config_file, output = map(Path, sys.argv[1:])
    if not run(json.loads(config_file.read_text()), output)['complete']:
        raise SystemExit('head microprobe incomplete; retain checkpoint')
