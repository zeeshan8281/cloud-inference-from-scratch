"""Attention math and attention/KV backends shared by every engine mode.

The model layer calls ``backend.attend(...)``; the backend decides whether
attention reads a full recomputed sequence (naive), a contiguous per-request
buffer, a gathered paged view (torch reference), or the physical block pool
directly through the Triton kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:
    import torch as _torch
except ImportError:  # pragma: no cover
    _torch = None

from .cache import CacheView


@dataclass
class StepContext:
    """Per-forward-pass context handed down through the model.

    ``request_id``   — cache partition for cached modes (None in naive mode)
    ``kv_start``     — number of tokens already stored before this forward
    ``is_decode``    — True when exactly one token is being processed
    """

    request_id: str | None
    kv_start: int
    is_decode: bool


def rms_norm(hidden: Any, weight: Any, eps: float) -> Any:
    dtype = hidden.dtype
    x = hidden.to(_torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * _torch.rsqrt(variance + eps)
    return (weight.to(_torch.float32) * x).to(dtype)


def build_rope_cache(head_dim: int, max_len: int, theta: float, device: str) -> tuple[Any, Any]:
    inv_freq = 1.0 / (
        theta ** (_torch.arange(0, head_dim, 2, dtype=_torch.float32, device=device) / head_dim)
    )
    positions = _torch.arange(max_len, dtype=_torch.float32, device=device)
    angles = positions[:, None] * inv_freq[None, :]
    emb = _torch.cat((angles, angles), dim=-1)
    return emb.cos(), emb.sin()  # [max_len, head_dim], float32


def rotate_half(x: Any) -> Any:
    half = x.shape[-1] // 2
    return _torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(
    q: Any,
    k: Any,
    cos: Any,
    sin: Any,
    positions: Any,
) -> tuple[Any, Any]:
    """Rotate q/k of shape [seq, heads, head_dim] with RoPE at given positions."""
    cos_row = cos[positions][:, None, :]  # [seq, 1, head_dim]
    sin_row = sin[positions][:, None, :]
    q_out = (q.to(_torch.float32) * cos_row) + (rotate_half(q.to(_torch.float32)) * sin_row)
    k_out = (k.to(_torch.float32) * cos_row) + (rotate_half(k.to(_torch.float32)) * sin_row)
    return q_out.to(q.dtype), k_out.to(k.dtype)


def causal_attention(
    q: Any,
    keys: Any,
    values: Any,
    sm_scale: float,
    past_len: int = 0,
) -> Any:
    """Scaled dot-product causal attention.

    q: [Tq, num_heads, head_dim]; keys/values: [Tk, num_heads, head_dim].
    Returns context [Tq, num_heads * head_dim]. Softmax accumulates in fp32;
    the tensor-core matmuls below accumulate in fp32 internally as well.
    """
    query_len, num_heads, head_dim = q.shape
    key_len = keys.shape[0]
    scores = _torch.einsum("thd,shd->hts", q, keys) * sm_scale  # [H, Tq, Tk]
    positions_q = _torch.arange(query_len, device=q.device) + past_len
    allowed = positions_q[:, None] >= _torch.arange(key_len, device=q.device)[None, :]
    scores = scores.to(_torch.float32).masked_fill(~allowed[None, :, :], float("-inf"))
    probs = _torch.softmax(scores, dim=-1).to(values.dtype)
    context = _torch.einsum("hts,shd->thd", probs, values)
    return context.reshape(query_len, num_heads * head_dim)


class AttentionBackend:
    """Reference backend: torch math over contiguous or gathered views."""

    def __init__(
        self,
        cache: Any,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        self.cache = cache
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sm_scale = 1.0 / math.sqrt(head_dim)
        self.group = num_heads // num_kv_heads

    def attend(self, layer: int, ctx: StepContext, q: Any, k: Any, v: Any) -> Any:
        """q: [Tq, Hq, D] post-RoPE; k/v: [T, KVH, D] pre-cache."""
        if self.cache is None or ctx.request_id is None:
            expanded_k = self._expand(k)
            expanded_v = self._expand(v)
            return causal_attention(q, expanded_k, expanded_v, self.sm_scale, past_len=0)
        self.cache.append(ctx.request_id, layer, k, v, ctx.kv_start)
        view: CacheView = self.cache.view(ctx.request_id, layer)
        expanded_k = self._expand(view.keys)
        expanded_v = self._expand(view.values)
        return causal_attention(q, expanded_k, expanded_v, self.sm_scale, past_len=ctx.kv_start)

    def _expand(self, t: Any) -> Any:
        if self.group == 1:
            return t
        return t.repeat_interleave(self.group, dim=1)


class TritonDecodeAttentionBackend(AttentionBackend):
    """Decode attention straight off the physical pools via the Triton kernel.

    Prefill still uses the gathered torch path (a production engine would use a
    paged prefill kernel; that shortcut is documented in docs/05). Decode never
    reconstructs a logical cache: no ``torch.cat``, no full gather (PRD FR7).
    """

    def __init__(self, *args: Any, allow_reference_fallback: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.allow_reference_fallback = allow_reference_fallback
        self._fallback_warned = False

    def attend(self, layer: int, ctx: StepContext, q: Any, k: Any, v: Any) -> Any:
        assert self.cache is not None and ctx.request_id is not None
        self.cache.append(ctx.request_id, layer, k, v, ctx.kv_start)
        if ctx.is_decode:
            try:
                from .kernel import decode_attention_direct

                seq_len = self.cache.sequence_length(ctx.request_id)
                out = decode_attention_direct(
                    q[0],  # [num_heads, head_dim] single query token
                    self.cache.key_pool[layer],
                    self.cache.value_pool[layer],
                    self.cache.block_table(ctx.request_id),
                    seq_len,
                    self.sm_scale,
                )
                return out.reshape(1, self.num_heads * self.head_dim)
            except Exception as exc:
                if not self.allow_reference_fallback:
                    raise
                if not self._fallback_warned:
                    self._fallback_warned = True
                    print(f"[triton] kernel unavailable ({exc}); using reference gather path")
        return super().attend(layer, ctx, q, k, v)
