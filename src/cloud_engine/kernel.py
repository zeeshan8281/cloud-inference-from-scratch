"""Triton direct-block decode-attention kernel (PRD FR7).

Single-token decode attention that reads K/V straight from the physical block
pools through the request block table: no ``torch.cat``, no logical-cache
reconstruction, no Hugging Face delegation. Online-softmax accumulation over
16-token blocks in fp32.

Supported envelope (fails closed with a precise error outside it):
  fp16 CUDA, head_dim=64, block_size=16, batch<=16, seq_len<=2048,
  num_heads divisible by num_kv_heads (Qwen2.5-0.5B GQA layout).
"""

from __future__ import annotations

from typing import Any

import triton
import triton.language as tl

SUPPORTED_HEAD_DIM = 64
SUPPORTED_BLOCK_SIZE = 16
SUPPORTED_MAX_BATCH = 16
SUPPORTED_MAX_SEQ_LEN = 2048


class KernelUnsupported(RuntimeError):
    """Raised before launch when shapes fall outside the supported envelope."""


def _validate(q: Any, k_pool_layer: Any, v_pool_layer: Any, block_tables: list[list[int]],
              seq_lens: list[int]) -> tuple[int, int, int, int]:
    import torch

    if q.device.type != "cuda":
        raise KernelUnsupported(f"q must live on CUDA, got {q.device}")
    if q.dtype != torch.float16:
        raise KernelUnsupported(f"kernel requires fp16, got {q.dtype}")
    if q.dim() != 3 or q.shape[-1] != SUPPORTED_HEAD_DIM:
        raise KernelUnsupported(
            f"q must be [batch, heads, {SUPPORTED_HEAD_DIM}], got {tuple(q.shape)}"
        )
    if k_pool_layer.dtype != torch.float16 or v_pool_layer.dtype != torch.float16:
        raise KernelUnsupported("KV pools must be fp16")
    if k_pool_layer.shape[1] != SUPPORTED_BLOCK_SIZE:
        raise KernelUnsupported(
            f"block_size must be {SUPPORTED_BLOCK_SIZE}, got {k_pool_layer.shape[1]}"
        )
    if k_pool_layer.shape[-1] != SUPPORTED_HEAD_DIM:
        raise KernelUnsupported(f"pool head_dim must be {SUPPORTED_HEAD_DIM}")
    batch, num_heads, _ = q.shape
    if not 1 <= batch <= SUPPORTED_MAX_BATCH:
        raise KernelUnsupported(f"batch must be 1..{SUPPORTED_MAX_BATCH}, got {batch}")
    kv_heads = k_pool_layer.shape[3]
    if num_heads % kv_heads != 0:
        raise KernelUnsupported(f"num_heads {num_heads} not divisible by kv_heads {kv_heads}")
    if len(block_tables) != batch or len(seq_lens) != batch:
        raise KernelUnsupported("block table and sequence-length counts must match batch")
    for idx, (table, length) in enumerate(zip(block_tables, seq_lens, strict=True)):
        if length < 1 or length > SUPPORTED_MAX_SEQ_LEN:
            raise KernelUnsupported(
                f"sequence {idx} length {length} outside 1..{SUPPORTED_MAX_SEQ_LEN}"
            )
        expected_blocks = -(-length // SUPPORTED_BLOCK_SIZE)
        if len(table) < expected_blocks:
            raise KernelUnsupported(
                f"sequence {idx}: table has {len(table)} blocks, needs at least {expected_blocks}"
            )
    return batch, num_heads, kv_heads, max(map(len, block_tables))


def decode_attention_direct(
    q: Any,
    k_pool_layer: Any,
    v_pool_layer: Any,
    block_table: list[int],
    seq_len: int,
    sm_scale: float,
) -> Any:
    """Convenience single-request wrapper around the batched kernel.

    q: [num_heads, head_dim] fp16 CUDA. Returns [num_heads, head_dim] fp16.
    """
    return decode_attention_batched(
        q.unsqueeze(0), k_pool_layer, v_pool_layer, [block_table], [seq_len], sm_scale
    )[0]


def decode_attention_batched(
    q: Any,
    k_pool_layer: Any,
    v_pool_layer: Any,
    block_tables: list[list[int]],
    seq_lens: list[int],
    sm_scale: float,
) -> Any:
    """Batched single-token decode attention over physical blocks.

    q: [B, Hq, D]; pools: [num_blocks, S, KH, D] for one layer; returns [B, Hq, D].
    """
    import torch
    batch, num_heads, kv_heads, max_blocks = _validate(
        q, k_pool_layer, v_pool_layer, block_tables, seq_lens
    )

    device = q.device
    table_width = max(max_blocks, 1)
    bt = torch.zeros(batch, table_width, dtype=torch.int32, device=device)
    for row, table in enumerate(block_tables):
        bt[row, : len(table)] = torch.tensor(table, dtype=torch.int32, device=device)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)

    out = torch.empty_like(q)

    grid = (batch, num_heads)
    _paged_decode_kernel[grid](
        q,
        bt,
        lens,
        k_pool_layer,
        v_pool_layer,
        out,
        sm_scale,
        num_heads // kv_heads,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        bt.stride(0),
        k_pool_layer.stride(0),
        k_pool_layer.stride(1),
        k_pool_layer.stride(2),
        k_pool_layer.stride(3),
        v_pool_layer.stride(0),
        v_pool_layer.stride(1),
        v_pool_layer.stride(2),
        v_pool_layer.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK=SUPPORTED_BLOCK_SIZE,
        HEAD_DIM=SUPPORTED_HEAD_DIM,
        num_warps=4,
        num_stages=2,
    )
    return out


_paged_decode_kernel_doc = """
Online-softmax decode attention over paged KV blocks.

One program per (sequence, query head). The program walks the sequence's
blocks via the block table, loading 16-key tiles directly from the physical
pool and maintaining running max/normalizer/accumulator in fp32.
"""

@triton.jit
def _paged_decode_kernel(
    Q, BT, LEN, KP, VP, OUT,
    sm_scale,
    GROUP,
    stride_qb, stride_qh, stride_qd,
    stride_btb,
    stride_kn, stride_ks, stride_kh, stride_kd,
    stride_vn, stride_vs, stride_vh, stride_vd,
    stride_ob, stride_oh, stride_od,
    BLOCK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    kv_h = pid_h // GROUP

    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + pid_b * stride_qb + pid_h * stride_qh + offs_d * stride_qd).to(tl.float32)

    seq_len = tl.load(LEN + pid_b)
    num_blocks = tl.cdiv(seq_len, BLOCK)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for block_index in range(num_blocks):
        phys = tl.load(BT + pid_b * stride_btb + block_index)
        offs_s = block_index * BLOCK + tl.arange(0, BLOCK)
        k_ptrs = (
            KP
            + phys.to(tl.int64) * stride_kn
            + offs_s[:, None] * stride_ks
            + kv_h * stride_kh
            + offs_d[None, :] * stride_kd
        )
        k_tile = tl.load(k_ptrs).to(tl.float32)          # [BLOCK, HEAD_DIM]
        scores = tl.sum(k_tile * q[None, :], axis=1) * sm_scale  # [BLOCK]
        valid = offs_s < seq_len
        scores = tl.where(valid, scores, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        probs = tl.exp(scores - m_new)
        l_i = l_i * alpha + tl.sum(probs, axis=0)

        v_ptrs = (
            VP
            + phys.to(tl.int64) * stride_vn
            + offs_s[:, None] * stride_vs
            + kv_h * stride_vh
            + offs_d[None, :] * stride_vd
        )
        v_tile = tl.load(v_ptrs).to(tl.float32)
        acc = acc * alpha + tl.sum(probs[:, None] * v_tile, axis=0)
        m_i = m_new

    result = acc / l_i
    tl.store(
        OUT + pid_b * stride_ob + pid_h * stride_oh + offs_d * stride_od,
        result.to(OUT.dtype.element_ty),
    )
