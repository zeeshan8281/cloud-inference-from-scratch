"""Actual captured-row QKV replay; CONFIG_JSON OUTPUT_JSON, no model fix."""
from __future__ import annotations

import base64
import hashlib
import json
import sys
from pathlib import Path


def run(config, output):
    import torch
    import torch.nn.functional as functional
    from safetensors import safe_open

    from cloud_engine.batch_invariant_linear import batch_invariant_linear, kernel_policy
    from cloud_engine.precision import configure_inference_precision

    if output.exists():
        raise ValueError('refusing overwrite')
    fixture_path = Path(config['fixture_path'])
    fixture = json.loads(fixture_path.read_text())
    if fixture['schema'] != 'captured-qkv-replay-input-v1' or fixture['target_position'] != 511:
        raise ValueError('wrong captured fixture')
    result = {'schema': 'fixed-qkv-microprobe-v1', 'complete': False, 'error': None,
              'fixture_sha256': hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
              'fixture_provenance': {k: v for k, v in fixture.items() if k != 'states'},
              'kernel_policy': kernel_policy(), 'projections': {}, 'fused': [], 'execution': []}

    def save():
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, allow_nan=False))
        temporary.replace(output)

    def decode(vector):
        raw = base64.b64decode(vector['bytes_base64'], validate=True)
        if hashlib.sha256(raw).hexdigest() != vector['sha256'] or vector['dtype'] != 'torch.float16':
            raise ValueError('fixture vector hash/dtype mismatch')
        return torch.frombuffer(bytearray(raw), dtype=torch.float16).clone().cuda()

    def encode(tensor):
        tensor = tensor.detach().contiguous().cpu()
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError('nonfinite output')
        raw = tensor.view(torch.uint8).numpy().tobytes()
        return {'dtype': str(tensor.dtype), 'shape': list(tensor.shape),
                'sha256': hashlib.sha256(raw).hexdigest(), 'bytes_base64': base64.b64encode(raw).decode()}

    def metrics(actual, reference):
        delta = (actual.double()-reference.double()).abs()
        return {'changed_elements': int((actual.double() != reference.double()).sum()),
                'max_abs': float(delta.max()), 'mean_abs': float(delta.mean())}

    try:
        configure_inference_precision()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        result['precision'] = {'allow_fp16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                               'allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                               'float32_matmul_precision': torch.get_float32_matmul_precision()}
        result['runtime'] = {'torch': torch.__version__, 'cuda': torch.version.cuda,
                             'gpu': torch.cuda.get_device_name()}
        result['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                   (Path(__file__), Path(__file__).parents[1]/'cloud_engine/batch_invariant_linear.py')}
        rows = {name: decode(state['layers.0.input_layernorm']['vectors'][0]) for name, state in fixture['states'].items()}
        row = rows['solo-cold']
        if row.numel() != 2048 or not all(torch.equal(row, value) for value in rows.values()):
            raise ValueError('captured normalized inputs are not identical2048')
        result['frozen_input'] = encode(row)
        actual_layouts = {name: (state['layers.0.input_layernorm']['packed_tokens'], state['layers.0.input_layernorm']['target_offset']) for name, state in fixture['states'].items()}
        layouts = sorted(set(actual_layouts.values()) | {(m, offset) for m in (1, 8, 16, 128, 256, 2048) for offset in (0, m-1)})
        parameters = {}
        keys = {f'model.layers.0.self_attn.{p}_proj.{suffix}': (p, suffix) for p in ('q', 'k', 'v') for suffix in ('weight', 'bias')}
        for path in sorted(Path(config['model_dir']).rglob('*.safetensors')):
            with safe_open(path, framework='pt', device='cpu') as handle:
                for key in set(handle.keys()) & keys.keys():
                    if keys[key] in parameters:
                        raise ValueError('duplicate pinned parameter')
                    parameters[keys[key]] = handle.get_tensor(key).to(device='cuda', dtype=torch.float16).contiguous()
        if len(parameters) != 6:
            raise ValueError('missing actual layer0 QKV weights')
        candidate_rows = {}
        for projection in ('q', 'k', 'v'):
            weight, bias = parameters[projection, 'weight'], parameters[projection, 'bias']
            reference = functional.linear(row.double(), weight.double(), bias.double())
            entry = {'weight': encode(weight), 'bias': encode(bias), 'fp64_reference': encode(reference),
                     'baseline_native_matches': {}, 'layouts': []}
            # Hashes suffice for large weights; real FP16 bytes stay in pinned model.
            entry['weight'].pop('bytes_base64')
            for name, (m, offset) in actual_layouts.items():
                matrix = row.repeat(m, 1)
                baseline = functional.linear(matrix.float(), weight.float(), bias.float()).half()[offset]
                saved = decode(fixture['states'][name][f'layers.0.self_attn.{projection}_proj']['vectors'][0])
                entry['baseline_native_matches'][name] = {'exact': torch.equal(baseline, saved), 'difference': metrics(baseline, saved)}
            for m, offset in layouts:
                matrix = row.repeat(m, 1)
                baseline = functional.linear(matrix.float(), weight.float(), bias.float()).half()[offset]
                fixed = batch_invariant_linear(matrix, weight, bias)[offset]
                repeated = batch_invariant_linear(matrix, weight, bias)[offset]
                candidate_rows[projection, m, offset] = fixed.clone()
                entry['layouts'].append({'m': m, 'offset': offset, 'baseline': encode(baseline),
                                         'candidate': encode(fixed), 'repeat_exact': torch.equal(fixed, repeated),
                                         'baseline_fp64_error': metrics(baseline, reference),
                                         'candidate_fp64_error': metrics(fixed, reference),
                                         'candidate_rounded_fp64_error': metrics(fixed, reference.half())})
            entry['all_layouts_exact'] = len({item['candidate']['sha256'] for item in entry['layouts']}) == 1
            result['projections'][projection] = entry
            save()
        fused_weight = torch.cat([parameters[p, 'weight'] for p in ('q', 'k', 'v')])
        fused_bias = torch.cat([parameters[p, 'bias'] for p in ('q', 'k', 'v')])
        for m, offset in layouts:
            matrix = row.repeat(m, 1)
            fused = batch_invariant_linear(matrix, fused_weight, fused_bias)[offset]
            separate = torch.cat([candidate_rows[p, m, offset] for p in ('q', 'k', 'v')])
            result['fused'].append({'m': m, 'offset': offset, 'separate_exact': torch.equal(fused, separate), 'difference': metrics(fused, separate)})
        compiled = torch.compile(batch_invariant_linear, fullgraph=True, dynamic=True)
        for m in (16, 256):
            matrix = row.repeat(m, 1)
            eager = batch_invariant_linear(matrix, fused_weight, fused_bias)
            compiled_output = compiled(matrix, fused_weight, fused_bias)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    batch_invariant_linear(matrix, fused_weight, fused_bias)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graphed = batch_invariant_linear(matrix, fused_weight, fused_bias)
            # Poison the captured destination so replay must actually rewrite it.
            graphed.fill_(float('nan'))
            graph.replay()
            torch.cuda.synchronize()
            result['execution'].append({'m': m, 'compiled_fullgraph_exact': torch.equal(eager, compiled_output),
                                        'cuda_graph_replay_exact': torch.equal(eager, graphed)})
        result['baseline_reproduces_capture'] = all(v['exact'] for p in result['projections'].values() for v in p['baseline_native_matches'].values())
        result['candidate_layouts_exact'] = all(p['all_layouts_exact'] and all(x['repeat_exact'] for x in p['layouts']) for p in result['projections'].values())
        result['fused_separate_exact'] = all(x['separate_exact'] for x in result['fused'])
        result['complete'] = True
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    save()
    return result


if __name__ == '__main__':
    config_file, output = map(Path, sys.argv[1:])
    if not run(json.loads(config_file.read_text()), output)['complete']:
        raise SystemExit('microprobe failed; retain checkpoint')
