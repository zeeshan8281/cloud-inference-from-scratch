"""Small in-process workload for NVIDIA Nsight Systems."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os


async def main() -> None:
    from cloud_engine.config import build_config
    from cloud_engine.engine import InferenceEngine
    from cloud_engine.scheduler import GenerationConfig

    engine = InferenceEngine(
        build_config("ragged"), model_dir=os.environ["MODEL_DIR"]
    )
    await engine.start()
    try:
        trace_path = os.environ.get("TORCH_TRACE_PATH")
        if trace_path:
            import torch

            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
            )
        else:
            profiler = contextlib.nullcontext()
        with profiler:
            handles = [
                await engine.submit(
                    prompt,
                    GenerationConfig(max_output_tokens=8, eos_token_id=None),
                )
                for prompt in (
                    "Explain a KV cache briefly.",
                    "Define continuous batching.",
                    "What is paged attention?",
                    "Why use GPU kernels?",
                )
            ]
            results = await asyncio.gather(*(handle.wait() for handle in handles))
        if trace_path:
            profiler.export_chrome_trace(trace_path)
        snapshot = engine.snapshot_metrics()
        max_requests = snapshot["scheduler"]["max_forward_request_count"]
        blocks_used = engine.cache.stats().blocks_used
        if max_requests < 4 or blocks_used:
            raise RuntimeError(
                f"invalid profile workload: max_requests={max_requests}, blocks_used={blocks_used}"
            )
        print(
            json.dumps(
                {
                    "requests": len(results),
                    "output_tokens": sum(len(result.token_ids) for result in results),
                    "max_forward_request_count": max_requests,
                    "blocks_used_after_run": blocks_used,
                }
            )
        )
    finally:
        await engine.close()


if __name__ == "__main__":
    asyncio.run(main())
