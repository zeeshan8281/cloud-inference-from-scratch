"""Two-token exact-prompt residual probe; controls run without tensor observers."""
import json
import os
import sys
import time
from pathlib import Path


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    output = Path(sys.argv[2])
    if 'trace_case' in config:
        from experiments.residual_vllm_full_probe import run
        if not run(config, output)['complete']:
            raise SystemExit('Full-window residual probe incomplete; retain checkpoint and traces')
        return
    operators = output.with_suffix('.operators.jsonl')
    projections = output.with_suffix('.projection.jsonl')
    assert not any(p.exists() for p in (output, operators, projections)), 'Evidence already exists'
    selection = config['selection']
    indices = list(range(len(config['batch_ids']))) if selection == 'batch' else [int(selection.removeprefix('solo-'))]
    target_ids = [f'diag-{i}' for i in config.get('target_indices', [2, 6]) if i in indices]
    assert target_ids
    for name in ('OPERATOR_TRACE', 'RESIDUAL_PROJECTION_TRACE', 'RESIDUAL_TARGET_IDS'):
        os.environ.pop(name, None)
    if config['observe']:
        os.environ['OPERATOR_TRACE'] = str(operators)
        os.environ['RESIDUAL_PROJECTION_TRACE'] = str(projections)
        os.environ['RESIDUAL_TARGET_IDS'] = json.dumps(target_ids)
    import vllm

    from cloud_engine.precision import require_vllm_precision
    from experiments.sentinel_pilot import _build_vllm_engine, resolve_kv_plan

    original = vllm.EngineArgs
    def arguments(**kwargs):
        kwargs['worker_extension_cls'] = 'experiments.residual_vllm_hook.Observer'
        return original(**kwargs)
    vllm.EngineArgs = arguments
    try:
        engine = _build_vllm_engine(config['model_dir'], config['model'], 'resource_normalized',
                                   resolve_kv_plan(config['model_dir'], 16, config.get('kv_bytes', 4 * 1024**3)))
    finally:
        vllm.EngineArgs = original
    params = vllm.SamplingParams(temperature=0, top_p=1, top_k=-1, max_tokens=2, ignore_eos=True, logprobs=20)
    precision = require_vllm_precision(engine)
    for index in indices:
        engine.add_request(f'diag-{index}', {'prompt_token_ids': config['batch_ids'][index]}, params)
    finished = {}
    started = time.monotonic()
    while engine.has_unfinished_requests():
        assert time.monotonic() - started < 900, 'Residual probe timeout'
        for response in engine.step():
            if response.finished:
                assert response.request_id not in finished and len(response.outputs) == 1
                result = response.outputs[0]
                assert len(result.token_ids) == 2
                finished[response.request_id] = {'output_tokens': list(result.token_ids),
                    'top_k': [[[token, entry.logprob] for token, entry in sorted(step.items(), key=lambda item: item[1].logprob, reverse=True)] for step in result.logprobs]}
    assert set(finished) == {f'diag-{i}' for i in indices}
    calls = [json.loads(line) for line in projections.read_text().splitlines()] if config['observe'] else []
    op_calls = [json.loads(line) for line in operators.read_text().splitlines()] if config['observe'] else []
    for target in target_ids if config['observe'] else []:
        emitted = [call for call in calls if call['request_id'] == target and call['emits_token']]
        assert len(emitted) == 2
        assert [call['native_argmax'] for call in emitted] == finished[target]['output_tokens']
        assert [call['output_tokens_before'] for call in emitted] == [[], finished[target]['output_tokens'][:1]]
        assert any(call['request_id'] == target and call['output_tokens_before'] for call in op_calls)
    output.write_text(json.dumps({'complete': True, 'error': None, 'config': config, 'precision': precision,
                                  'observed_precision': engine.collective_rpc('residual_status'), 'results': finished,
                                  'projection_calls': calls, 'operator_calls': op_calls,
                                  'scope': 'two-token diagnostic; scheduling may differ from 128-token holdout'}, allow_nan=False))


if __name__ == '__main__':
    main()
