"""Opaque compile boundary around the existing eager RMSNorm arithmetic.

Imported only by the eligible CUDA policy. This introduces no arithmetic kernel.
"""
import torch

from .attention import rms_norm

OPERATOR_NAME = 'cloud_engine::common_rms_norm'


@torch.library.custom_op(OPERATOR_NAME, mutates_args=(), device_types='cuda')
def opaque_rms_norm(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    return rms_norm(x, weight, epsilon).contiguous()


@opaque_rms_norm.register_fake
def _fake_rms_norm(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    return x.new_empty(x.shape)
