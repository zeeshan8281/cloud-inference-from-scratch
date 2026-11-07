"""Full cold/warm exposed-prefix diagnostic; never a correctness gate pass."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from experiments.custom_residual_probe import phase_plan, run


def configure(config):
    config = dict(config)
    config.update(warm_prefix_protocol=True, mode="complete_system",
                  candidate=config.get("candidate", "fp32_qkv_and_head"),
                  output_tokens=128, cases=["batch", "solo"])
    if len(config["batch_ids"]) != 8 or config["target_index"] != 2 or any(len(ids) != 512 for ids in config["batch_ids"]):
        raise ValueError("requires exact exposed c8/512 target2 workload")
    phase_plan(config)
    return config


if __name__ == "__main__":
    config_file, output = map(Path, sys.argv[1:])
    result = asyncio.run(run(configure(json.loads(config_file.read_text())), output))
    if not result["complete"]:
        raise SystemExit("warm-prefix diagnostic incomplete; retain raw evidence")
