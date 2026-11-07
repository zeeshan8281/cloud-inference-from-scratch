"""Opt-in full128 diagnostic18; existing diagnostic16 defaults are unchanged."""
import json
import sys
from pathlib import Path

from experiments.residual_vllm_full_probe import run

if __name__ == '__main__':
    source, output = map(Path,sys.argv[1:])
    config = json.loads(source.read_text())
    config['attention_probe'] = True
    if not run(config,output)['complete']:
        raise SystemExit('attention replay diagnostic incomplete; retain checkpoint/traces')
