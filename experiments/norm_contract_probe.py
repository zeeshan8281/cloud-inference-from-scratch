"""Bounded actual normalization callable contract; not full-model acceptance."""
import base64
import hashlib
import json
import sys
from pathlib import Path
from types import MethodType


def decode(record, torch):
    raw = base64.b64decode(record['bytes_base64'], validate=True)
    if record['dtype'] != 'torch.float16' or record['shape'] != [2048] or len(raw) != 4096 or hashlib.sha256(raw).hexdigest() != record['sha256']:
        raise ValueError('invalid actual captured FP16 operand')
    return torch.frombuffer(bytearray(raw), dtype=torch.float16).clone().cuda()


def digest(tensor, torch):
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError('nonfinite output')
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def run(config, output):
    if output.exists():
        raise ValueError('refusing evidence overwrite')
    result = {'schema': 'normalization-contract-v1', 'config': config, 'complete': False, 'error': None,
              'cases': [], 'gate_cleared': False, 'scope': 'actual callable on exposed rows and synthetic repeated rows; no full-model proof'}
    def save():
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, allow_nan=False))
        temporary.replace(output)
    save()
    try:
        import torch
        import vllm
        from cloud_engine.attention import rms_norm
        from cloud_engine.numerical_policy import POLICY_ID, _norm_forward, _normalization_records
        from cloud_engine.opaque_normalization import OPERATOR_NAME, opaque_rms_norm
        from cloud_engine.precision import configure_numerical_policy_precision, precision_status
        from safetensors import safe_open
        from vllm.model_executor.layers.layernorm import RMSNorm
        configure_numerical_policy_precision()
        torch.set_grad_enabled(False)
        fixture_path, model_dir = Path(config['fixture_path']), Path(config['model_dir'])
        fixture = json.loads(fixture_path.read_text())
        model_config = json.loads((model_dir/'config.json').read_text())
        index = json.loads((model_dir/'model.safetensors.index.json').read_text())['weight_map']
        if vllm.__version__ != '0.10.0' or model_config['hidden_size'] != 2048:
            raise ValueError('unsupported pinned runtime/model')
        epsilon = model_config['rms_norm_eps']
        result.update(policy_id=POLICY_ID, fixture_sha256=sha(fixture_path), precision=precision_status(),
                      versions={'torch':torch.__version__, 'vllm':vllm.__version__, 'cuda':torch.version.cuda},
                      epsilon=epsilon, actual_weight_files={}, installed_callable='_norm_forward(opaque_normalization.opaque_rms_norm)', operator=OPERATOR_NAME,
                      source_sha256={'policy':sha(Path(sys.modules[_norm_forward.__module__].__file__)),
                                     'attention':sha(Path(sys.modules[rms_norm.__module__].__file__)),
                                     'opaque':sha(Path(sys.modules['cloud_engine.opaque_normalization'].__file__))})
        pure = decode(fixture['states']['custom']['input'], torch)
        captured_pure = decode(fixture['states']['custom']['output'], torch)
        residual_record = next(r for r in fixture['residual']['records'] if r['direction'] == 'input')
        residual_inputs = {v['path']: decode(v,torch) for v in residual_record['vectors']}
        for kind in ('pure', 'residual'):
            weight_name = fixture['weight_name'] if kind == 'pure' else fixture['residual']['weight_name']
            weight_path = model_dir/index[weight_name]
            weight_sha = sha(weight_path)
            if weight_sha != fixture['states']['custom']['model_files_sha256'][weight_path.name]:
                raise ValueError('actual weight bytes mismatch')
            result['actual_weight_files'][weight_path.name] = weight_sha
            with safe_open(weight_path, framework='pt', device='cpu') as handle:
                weight = handle.get_tensor(weight_name).to(device='cuda',dtype=torch.float16)
            module = RMSNorm(2048,eps=epsilon,dtype=torch.float16).cuda()
            with torch.no_grad():
                module.weight.copy_(weight)
            # Exercise the exact installer's binding path, not a replica.
            _normalization_records([module], 'variance_epsilon', opaque_rms_norm, replace=True)
            if not isinstance(module.forward, MethodType):
                raise ValueError('candidate callable not installed')
            compiled = torch.compile(module.forward, fullgraph=True)
            xrow = pure if kind == 'pure' else residual_inputs['input[0]']
            rrow = None if kind == 'pure' else residual_inputs['input[1]']
            for m in (0, 1, 16, 256, 512):
                source_x = xrow[None,:].repeat(m,1)
                source_r = None if rrow is None else rrow[None,:].repeat(m,1)
                summed = source_x if source_r is None else source_x+source_r
                expected = rms_norm(summed,weight,epsilon)
                if kind == 'pure' and m and not torch.equal(expected,captured_pure[None,:].expand_as(expected)):
                    raise ValueError('pure custom historical output not reproduced')
                x, residual = source_x.clone(), None if source_r is None else source_r.clone()
                def reset(x=x, source_x=source_x, residual=residual, source_r=source_r):
                    x.copy_(source_x)
                    if residual is not None:
                        residual.copy_(source_r)
                def validate(value, x=x, residual=residual, source_x=source_x, summed=summed, expected=expected):
                    y = value if residual is None else value[0]
                    aliases = (y is not x and torch.equal(x,source_x)) if residual is None else (value[0] is x and value[1] is residual and torch.equal(residual,summed))
                    return bool(torch.equal(y,expected) and aliases), digest(y,torch)
                case = {'kind':kind,'m':m,'expected_sha256':digest(expected,torch),
                        'rounded_residual_sha256':None if residual is None else digest(summed,torch),
                        'empty_output':m == 0}
                result['cases'].append(case)
                save()
                reset()
                eager_ok,eager_sha = validate(module.forward(x,residual))
                case.update(eager_exact_and_aliases=eager_ok,eager_sha256=eager_sha)
                save()
                reset()
                compiled_ok,compiled_sha = validate(compiled(x,residual))
                case.update(compiled_fullgraph_exact_and_aliases=compiled_ok,compiled_sha256=compiled_sha)
                save()
                # Input copies are INSIDE graph, so poisoning alias outputs cannot corrupt source operands.
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        reset()
                        compiled(x,residual)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    reset()
                    graph_value = compiled(x,residual)
                graph_y = graph_value if residual is None else graph_value[0]
                graph_y.fill_(float('nan'))
                if residual is not None:
                    residual.fill_(float('nan'))
                graph.replay()
                torch.cuda.synchronize()
                graph_ok,graph_sha = validate(graph_value)
                case.update(poisoned_graph_exact_and_aliases=graph_ok,graph_sha256=graph_sha)
                save()
                if not (eager_ok and compiled_ok and graph_ok):
                    raise ValueError('candidate output/rounding/alias/compiled/graph contract failed')
        result['complete'] = True
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    save()
    return result


if __name__ == '__main__':
    config_path, output_path = map(Path,sys.argv[1:])
    result = run(json.loads(config_path.read_text()),output_path)
    if not result['complete']:
        raise SystemExit('normalization contract incomplete; retained')
