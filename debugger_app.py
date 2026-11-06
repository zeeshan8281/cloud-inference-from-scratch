"""Isolated, checkpointed Modal replay of the stopped sentinel request."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import modal

if modal.is_local():
    from modal_app import sentinel_image
else:
    sentinel_image = None

PINNED = json.loads(Path(__file__).with_name("engine_config.json").read_text())
MODAL_CFG = PINNED["modal"]
RAGGED_MODEL = PINNED["ragged_model"]
volume = modal.Volume.from_name(MODAL_CFG["volume_name"], create_if_missing=True)
app = modal.App("cie-numerical-debugger")


@app.function(image=sentinel_image, gpu=MODAL_CFG["gpu"], volumes={"/cache": volume},
              timeout=300, startup_timeout=120, max_containers=1, scaledown_window=300)
def capture_case(implementation: str, variant: str, prefix: list[int] | None = None) -> dict:
    """One fresh child process; the local caller checkpoints every returned capture."""
    from datetime import datetime, timezone
    from importlib.metadata import version

    import torch
    from transformers import AutoTokenizer

    from cloud_engine.weights import ensure_weights_downloaded
    from experiments.sentinel_diagnostics import (
        build_diagnostic_prompts,
        natural_order_c8_batch,
        run_diagnostic_child,
    )
    from experiments.sentinel_pilot import gpu_state_snapshot

    if implementation not in ("custom", "vllm") or variant not in (
        "original", "target_first", "solo", "repeat", "shared_prefix"
    ):
        raise ValueError("unknown diagnostic case")
    print(f"real GPU capture: {implementation}/{variant}", flush=True)
    model_dir = str(ensure_weights_downloaded("/cache", RAGGED_MODEL["revision"], RAGGED_MODEL["id"]))
    volume.commit()
    prompts = build_diagnostic_prompts(AutoTokenizer.from_pretrained(model_dir))
    batch, index = natural_order_c8_batch(prompts)
    if variant == "target_first":
        batch, index = [batch[index], *(ids for i, ids in enumerate(batch) if i != index)], 0
    elif variant == "solo":
        batch, index = [batch[index]], 0
    elif variant == "shared_prefix":
        if not prefix or not all(isinstance(token, int) and token >= 0 for token in prefix):
            raise ValueError("shared-prefix probe requires observed token IDs")
        batch[index] = [*batch[index], *prefix]
    environment = {"started_utc": datetime.now(timezone.utc).isoformat(),
                   "source": os.environ.get("SOURCE_COMMIT", "unknown"), "model": RAGGED_MODEL,
                   "gpu": gpu_state_snapshot(torch.version.cuda),
                   "packages": {name: version(name) for name in ("torch", "triton", "vllm", "transformers", "tokenizers")},
                   "window_tokens": 3,
                   "scope": "original c8 prompt/order; diagnostic engine settings, fresh subprocesses"}
    config = {"implementation": implementation, "model_dir": model_dir,
              "batch_ids": batch, "target_index": index, "engine_options": {}}
    result = run_diagnostic_child(config, Path(tempfile.mkdtemp(prefix="numerical-debugger-")))
    environment["finished_utc"] = datetime.now(timezone.utc).isoformat()
    return {"environment": environment, "run": result, "target_request": prompts["target"]}


@app.local_entrypoint()
def numerical_debugger(output: str = "experiments/numerical-debugger/run") -> None:
    """Retain every real GPU result and generate JSON, Markdown and HTML reports."""
    from experiments.numerical_debugger import write_report

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    raw = destination / "raw.json"
    if raw.exists():
        raise SystemExit(f"refusing to overwrite retained evidence: {raw}")
    payload = {"runs": {}, "comparisons": [], "case_environments": {}}

    def checkpoint():
        temporary = raw.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        temporary.replace(raw)

    def run(implementation, variant, prefix=None):
        name = f"{implementation}_{variant}"
        result = capture_case.remote(implementation, variant, prefix)
        payload.setdefault("environment", dict(result["environment"]))
        payload.setdefault("target_request", result["target_request"])
        payload["case_environments"][name] = result["environment"]
        payload["runs"][name] = result["run"]
        checkpoint()
        print(f"{name}: {'crashed' if result['run']['crashed'] else 'captured'}", flush=True)

    def compare(label, left, right):
        payload["comparisons"].append({"label": label, "left": left, "right": right})
        checkpoint()

    for variant in ("original", "target_first", "solo", "repeat"):
        for implementation in ("custom", "vllm"):
            run(implementation, variant)
        compare(f"Cross-engine / {variant}", f"custom_{variant}", f"vllm_{variant}")
    for implementation in ("custom", "vllm"):
        compare(f"{implementation} / order sensitivity", f"{implementation}_original", f"{implementation}_target_first")
        compare(f"{implementation} / repeatability", f"{implementation}_original", f"{implementation}_repeat")
    anchor = payload["runs"]["custom_original"]
    if not anchor["crashed"] and anchor["result"]["result"].get("verified"):
        prefix = anchor["result"]["result"]["output_tokens"][:1]
        if prefix:
            payload["shared_prefix_probe"] = {"source": "custom_original", "appended_tokens": prefix,
                                               "cache_state": "fresh re-prefill"}
            for implementation in ("custom", "vllm"):
                run(implementation, "shared_prefix", prefix)
            compare("Cross-engine / shared-prefix re-prefill", "custom_shared_prefix", "vllm_shared_prefix")
    identities = {env["gpu"]["uuid"] for env in payload["case_environments"].values()}
    payload["environment"]["gpu_identity_stable"] = len(identities) == 1
    payload["environment"]["gpu_uuids"] = sorted(identities)
    payload["environment"]["finished_utc"] = result_end = next(reversed(payload["case_environments"].values()))["finished_utc"]
    checkpoint()
    summary = write_report(raw, destination)
    print(f"Completed at {result_end}; {summary['run_count']} runs; {len(summary['crashed_runs'])} crashes. Gate unchanged.")
    if (not payload["environment"]["gpu_identity_stable"] or summary["crashed_runs"]
            or any(row["status"] in ("unavailable", "incomplete") for row in summary["comparisons"])):
        raise SystemExit("diagnostic incomplete or GPU changed; retained evidence for inspection")
