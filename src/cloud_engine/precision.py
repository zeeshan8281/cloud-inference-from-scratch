"""Process-wide FP16 GEMM policy shared by custom and reference workers.

Reduced-precision intermediate reductions can change with packed batch shape.
Disabling them may reduce throughput; it does not promise universal determinism.
The separately installed pinned policy also disables TF32 and requests highest FP32 matmul precision.
BF16, scheduling, sampling, and comparison tolerances are unchanged.
"""
from __future__ import annotations


def configure_inference_precision() -> None:
    import torch

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


def configure_numerical_policy_precision() -> None:
    """Called only after the combined installer validates its supported scope."""
    import torch

    configure_inference_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def precision_status() -> dict:
    import torch

    return {"allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


def require_vllm_precision(engine) -> list[dict[str, bool]]:
    """Query the actual workers, not the driver's requested configuration."""
    states = engine.collective_rpc("precision_status")
    if not states or any(state.get("allow_fp16_reduced_precision_reduction") is not False for state in states):
        raise RuntimeError(f"vLLM worker FP16 reduction policy was not applied: {states}")
    return states


def require_numerical_policy(status: dict, implementation: str) -> dict:
    expected = {"custom": 252, "vllm": 144}.get(implementation)
    if (expected is None or status.get("policy_id") != "fixed_tile_linears_fp32_head_opaque_rmsnorm_fa2_min_query2_v3"
            or status.get("implementation") != implementation or status.get("installed") is not True
            or status.get("linears_count") != expected or status.get("expected_linears") != expected
            or any(status.get(key) is not True for key in ("head_installed", "wrappers_active", "parameters_preserved"))
            or status.get("storage_dtype") != "torch.float16" or status.get("output_dtype") != "torch.float32"):
        raise ValueError("actual shared numerical policy installation missing or changed")
    normalization = status.get("normalization")
    if (not isinstance(normalization, dict)
            or normalization.get("implementation") != "custom_fp32_rmsnorm_fp16_residual"
            or normalization.get("count") != 73 or normalization.get("expected_count") != 73
            or normalization.get("active") is not True or normalization.get("parameter_identity_preserved") is not True):
        raise ValueError("actual common normalization policy missing or changed")
    boundary = ("opaque_custom_op", "cloud_engine::common_rms_norm", True) if implementation == "vllm" else ("eager_torch", None, None)
    if ((normalization.get("compile_boundary"), normalization.get("operator"), normalization.get("operator_active")) != boundary
            or (implementation == "vllm" and normalization.get("operator_active") is not True)):
        raise ValueError("actual normalization compile boundary missing or changed")
    if implementation == "vllm":
        floor = status.get("metadata_floor", {})
        if (floor.get("installed") is not True or floor.get("active") is not True
                or floor.get("pinned_version") != "0.10.0" or floor.get("fa_version") != 2):
            raise ValueError("actual pinned FA2 metadata policy missing or changed")
    if status.get("precision") != {"allow_fp16_reduced_precision_reduction": False, "allow_tf32": False,
                                   "float32_matmul_precision": "highest"}:
        raise ValueError("actual shared arithmetic settings missing or changed")
    return status


def numerical_policy_snapshot(engine, implementation: str) -> dict:
    """Inspect actual installed policy only at phase boundaries, outside timing."""
    if implementation == "custom":
        from cloud_engine.numerical_policy import policy_status

        status = {**policy_status(engine.runner.model), "precision": precision_status()}
    elif implementation == "vllm":
        states = engine.collective_rpc("numerical_policy_status")
        if len(states) != 1:
            raise ValueError("expected actual TP1 numerical policy worker")
        status = states[0]
    else:
        raise ValueError("unknown numerical policy implementation")
    return require_numerical_policy(status, implementation)
