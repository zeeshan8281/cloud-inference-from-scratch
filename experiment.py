"""One-command, correctness-gated scheduling experiment on Modal L4."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import modal

from modal_app import RAGGED_MODEL, SOURCE_COMMIT, _gpu_options, _prepare_weights, image

app = modal.App("cloud-inference-experiments")
experiment_image = image.add_local_file(
    Path(__file__).parent / "modal_app.py", "/root/modal_app.py", copy=True
)


@app.function(**_gpu_options(image=experiment_image))
async def run_experiment(
    source: str,
    rates: str = "1,2,4",
    duration_seconds: float = 5,
) -> dict:
    import asyncio

    from benchmarks.online import run_engine_sweep
    from cloud_engine.config import build_config
    from cloud_engine.engine import InferenceEngine
    from cloud_engine.scheduler import GenerationConfig, decode_first_priority
    from cloud_engine.weights import load_tokenizer

    namespace: dict = {}
    exec(compile(source, "<experiment>", "exec"), namespace)
    priority = namespace.get("priority")
    if not callable(priority):
        raise ValueError("experiment must define priority(candidate)")

    model_dir = _prepare_weights(RAGGED_MODEL)
    tokenizer = load_tokenizer(model_dir)
    engine = InferenceEngine(build_config("ragged"), model_dir=model_dir)
    await engine.start()

    prompts = [
        "Explain paged KV cache in one sentence.",
        "Why does continuous batching improve inference throughput?",
        "Name one latency tradeoff in chunked prefill.",
        "What does a ragged attention kernel avoid?",
    ]

    async def generate_all() -> list[list[int]]:
        async def generate(prompt: str) -> list[int]:
            handle = await engine.submit(prompt, GenerationConfig(max_output_tokens=8))
            return (await handle.wait()).token_ids

        return await asyncio.gather(*(generate(prompt) for prompt in prompts))

    try:
        engine.scheduler.scheduling_priority = decode_first_priority
        baseline_tokens = await generate_all()
        engine.scheduler.scheduling_priority = priority
        experiment_tokens = await generate_all()
        checks = {
            "token_parity": baseline_tokens == experiment_tokens,
            "packed_forward": engine.runner.max_forward_request_count >= 2,
            "zero_live_kv_blocks": engine.cache.stats().occupied_bytes == 0,
        }
        if not all(checks.values()):
            raise RuntimeError(f"correctness gate failed: {checks}")

        parsed_rates = [float(value) for value in rates.split(",")]
        engine.scheduler.scheduling_priority = decode_first_priority
        baseline = await run_engine_sweep(
            engine, tokenizer, parsed_rates, duration_seconds, 500, 100,
            implementation="ragged-l4/default-policy",
        )
        engine.scheduler.scheduling_priority = priority
        experiment = await run_engine_sweep(
            engine, tokenizer, parsed_rates, duration_seconds, 500, 100,
            implementation=f"ragged-l4/{namespace.get('NAME', 'experiment')}",
        )
        checks["zero_failed_requests"] = all(
            sweep["requests"]["errors"] == 0
            for result in (baseline, experiment)
            for sweep in result["sweeps"]
        )
        if not checks["zero_failed_requests"]:
            raise RuntimeError("benchmark produced failed requests")
        return {
            "source_revision": os.environ.get("SOURCE_COMMIT", SOURCE_COMMIT),
            "experiment": {
                "name": namespace.get("NAME", "experiment"),
                "hypothesis": namespace.get("HYPOTHESIS", ""),
                "sha256": hashlib.sha256(source.encode()).hexdigest(),
            },
            "correctness": checks,
            "baseline": baseline,
            "result": experiment,
        }
    finally:
        await engine.close()


@app.local_entrypoint()
def main(
    experiment: str = "experiments/starter.py",
    rates: str = "1,2,4",
    duration_seconds: float = 5,
    output: str = "",
) -> None:
    path = Path(experiment)
    source = path.read_text()
    result = run_experiment.remote(source, rates, duration_seconds)
    destination = Path(output or f"artifacts/experiment-{result['experiment']['name']}.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["correctness"], indent=2))
    print(f"artifact: {destination}")
