"""Observe the normal worker without replacing emitted logits or sampling."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Spawn imports the entry script as __mp_main__; keep control runs unobserved.
if __name__ == "experiments.projection_probe":
    from cloud_engine.vllm_precision_worker import PrecisionWorker
    from experiments.projection_hook import Observer

    class FixedObserver(PrecisionWorker, Observer):
        """Apply the fixed worker policy before installing projection observation."""


def main():
    from experiments.sentinel_diagnostics import run_vllm_diagnostic

    config = json.loads(Path(sys.argv[1]).read_text())
    trace = Path(sys.argv[2]).with_suffix('.jsonl')
    os.environ['PROJECTION_TRACE'] = str(trace)
    options = {'worker_extension_cls': 'experiments.projection_probe.FixedObserver'} if config['observe'] else {}
    config['precision_policy'] = 'fixed FP16 reduced-precision reduction disabled; verified by worker RPC'
    result = run_vllm_diagnostic(config['model_dir'], config['model'], options,
                                 config['batch_ids'], config['target_index'])
    captured = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
    if config['observe']:
        assert captured, 'Worker hook did not execute; no projection evidence captured'
    Path(sys.argv[2]).write_text(json.dumps({'config': config, 'result': result, 'projection_calls': captured}))


if __name__ == '__main__':
    main()
