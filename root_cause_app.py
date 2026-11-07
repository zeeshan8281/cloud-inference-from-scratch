"""Real-GPU causal probes; never changes the sentinel gate."""
import hashlib
import json
from pathlib import Path

import modal

PROBES = ('projection_probe.py', 'projection_hook.py', 'admission_probe.py', 'operator_probe_hook.py', 'linear_probe_hook.py')
if modal.is_local():
    from modal_app import sentinel_image
    probe_image = sentinel_image
    for filename in PROBES:
        probe_image = probe_image.add_local_file(f'experiments/{filename}', f'/root/experiments/{filename}', copy=True)
else:
    probe_image = None

app = modal.App('cie-root-cause')
volume = modal.Volume.from_name('cloud-inference-model-cache', create_if_missing=False)


@app.function(image=probe_image, gpu='L4', volumes={'/cache': volume}, timeout=600,
              startup_timeout=180, max_containers=1, scaledown_window=300)
def probe(variant: str, experiment: str):
    import os
    import subprocess
    import sys
    import tempfile
    from datetime import datetime, timezone
    from importlib.metadata import version

    import torch
    from transformers import AutoTokenizer

    from cloud_engine.weights import ensure_weights_downloaded
    from experiments.sentinel_diagnostics import build_diagnostic_prompts, natural_order_c8_batch
    from experiments.sentinel_pilot import gpu_state_snapshot

    started = datetime.now(timezone.utc).isoformat()
    allowed = {'worker': ('original_control', 'original', 'target_first', 'target_first_control'),
               'admission': ('atomic', 'staged', 'atomic_repeat', 'staged_repeat',
                             'target_first_atomic', 'target_first_staged'),
               'precision': ('staged_linear', 'atomic_precise', 'staged_precise', 'atomic_precise_repeat', 'staged_precise_repeat', 'target_first_staged_precise')}
    if experiment not in allowed or variant not in allowed[experiment]:
        raise ValueError((experiment, variant))
    pinned = json.loads(Path('/root/engine_config.json').read_text())
    model = pinned['ragged_model']
    model_dir = str(ensure_weights_downloaded('/cache', model['revision'], model['id']))
    prompts = build_diagnostic_prompts(AutoTokenizer.from_pretrained(model_dir))
    batch, index = natural_order_c8_batch(prompts)
    if variant.startswith('target_first'):
        batch, index = [batch[index], *(ids for i, ids in enumerate(batch) if i != index)], 0
    root = Path(tempfile.mkdtemp())
    config = {'model_dir': model_dir, 'model': model, 'batch_ids': batch, 'target_index': index,
              'observe': not variant.endswith('control')}
    script = 'projection_probe.py'
    if experiment in ('admission', 'precision'):
        script = 'admission_probe.py'
        config.update(admission='staged' if 'staged' in variant else 'atomic',
                      operator_trace=variant in ('atomic', 'staged', 'staged_linear', 'atomic_precise', 'staged_precise'),
                      linear_trace=variant == 'staged_linear',
                      fp16_reduced_precision_reduction='precise' not in variant)
    (root/'config.json').write_text(json.dumps(config))
    completed = subprocess.run([sys.executable, f'/root/experiments/{script}', str(root/'config.json'), str(root/'output.json')],
                               capture_output=True, text=True, timeout=480, env=dict(os.environ, PYTHONPATH='/root'))
    return {'gpu': gpu_state_snapshot(torch.version.cuda), 'exit_code': completed.returncode,
            'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
            'packages': {name: version(name) for name in ('torch', 'triton', 'vllm', 'transformers')},
            'stdout': completed.stdout, 'stderr': completed.stderr,
            'source_hashes': {name: hashlib.sha256(Path(f'/root/experiments/{name}').read_bytes()).hexdigest() for name in PROBES},
            'hook_sha256': hashlib.sha256(Path('/root/experiments/projection_hook.py').read_bytes()).hexdigest(),
            'probe_sha256': hashlib.sha256(Path(f'/root/experiments/{script}').read_bytes()).hexdigest(),
            'capture': json.loads((root/'output.json').read_text()) if (root/'output.json').exists() else None}


@app.local_entrypoint()
def run(output: str, experiment: str = 'worker'):
    variants = {'worker': ('original_control', 'original', 'target_first', 'target_first_control'),
                'admission': ('atomic', 'staged', 'atomic_repeat', 'staged_repeat',
                              'target_first_atomic', 'target_first_staged'),
               'precision': ('staged_linear', 'atomic_precise', 'staged_precise', 'atomic_precise_repeat', 'staged_precise_repeat', 'target_first_staged_precise')}
    if experiment not in variants:
        raise ValueError(experiment)
    dest = Path(output)
    dest.mkdir(parents=True, exist_ok=False)
    for variant in variants[experiment]:
        result = probe.remote(variant, experiment)
        (dest/f'{variant}.json').write_text(json.dumps(result, indent=2))
        print(variant, result['exit_code'], flush=True)
        if result['exit_code']:
            raise RuntimeError(result['stderr'][-5000:])
