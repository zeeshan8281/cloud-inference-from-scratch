"""Fixed-tile QKV, optional-bias linears, and an FP32 output projection.

One fixed reduction order for each output tile. Shape invariance is a hypothesis
requiring actual GPU validation, not a promise implied by this module's name.
"""
from __future__ import annotations

BLOCK_M, BLOCK_N, BLOCK_K = 16, 64, 32
NUM_WARPS, NUM_STAGES = 4, 2
OPERATOR_NAME = "cloud_engine::diagnostic_fixed_tile_qkv"
LINEAR_OPERATOR_NAME = "cloud_engine::diagnostic_fixed_tile_linear"
HEAD_OPERATOR_NAME = "cloud_engine::diagnostic_fixed_tile_fp32_head"
HEAD_BLOCK_N, HEAD_BLOCK_K = 4, 2048
HEAD_NUM_WARPS, HEAD_NUM_STAGES = 4, 1


def kernel_policy():
    return {"implementation": "fixed_tile_triton", "operator": OPERATOR_NAME,
            "block_m": BLOCK_M, "block_n": BLOCK_N, "block_k": BLOCK_K,
            "num_warps": NUM_WARPS, "num_stages": NUM_STAGES,
            "input_dtype": "torch.float16", "weight_dtype": "torch.float16",
            "output_dtype": "torch.float16", "accumulation_dtype": "torch.float32",
            "bias_add_dtype": "torch.float32", "split_k": False, "autotune": False,
            "atomics": False, "scope": "pinned unquantized biased Qwen2 QKV diagnostic"}


def validate_contract(x_shape, weight_shape, bias_shape, strides, dtypes, devices):
    """Pure fail-closed pinned-model contract; usable without a CUDA installation."""
    if len(x_shape) != 2 or len(weight_shape) != 2 or len(bias_shape) != 1:
        raise ValueError("fixed QKV requires rank2 input/weight and rank1 bias")
    m, k = x_shape
    n, weight_k = weight_shape
    if m < 1 or k != 2048 or weight_k != k or n not in (256, 2048, 2560) or bias_shape != (n,):
        raise ValueError("unsupported pinned QKV dimensions")
    if strides != ((k, 1), (k, 1), (1,)):
        raise ValueError("fixed QKV requires contiguous row-major operands")
    if dtypes != ("torch.float16",) * 3:
        raise ValueError("fixed QKV requires FP16 input, weight and bias")
    if not devices[0].startswith("cuda") or len(set(devices)) != 1:
        raise ValueError("fixed QKV operands must share one CUDA device")


def linear_kernel_policy():
    return {**kernel_policy(), "operator": LINEAR_OPERATOR_NAME, "bias_optional": True,
            "scope": "pinned Qwen2 QKV/O/gate/up/down diagnostic; excludes output head"}


def validate_linear_contract(x_shape, weight_shape, bias_shape, strides, dtypes, devices):
    if len(x_shape) != 2 or len(weight_shape) != 2:
        raise ValueError("fixed linear requires rank2 input and weight")
    m, k = x_shape
    n, weight_k = weight_shape
    supported = (k == 2048 and n in (256, 2048, 2560, 11008, 22016)) or (k == 11008 and n == 2048)
    if m < 1 or weight_k != k or not supported:
        raise ValueError("unsupported pinned linear dimensions")
    has_bias = bias_shape is not None
    if has_bias and bias_shape != (n,):
        raise ValueError("bias shape must match output width")
    expected_strides = ((k, 1), (k, 1)) + (((1,),) if has_bias else ())
    if strides != expected_strides:
        raise ValueError("fixed linear requires contiguous row-major operands")
    if dtypes != ("torch.float16",) * (3 if has_bias else 2):
        raise ValueError("fixed linear requires FP16 operands")
    if len(devices) != len(dtypes) or not devices[0].startswith("cuda") or len(set(devices)) != 1:
        raise ValueError("fixed linear operands must share one CUDA device")


def head_kernel_policy():
    return {**kernel_policy(), "operator": HEAD_OPERATOR_NAME, "output_dtype": "torch.float32",
            "accumulation_dtype": "torch.float64", "product_dtype": "torch.float32",
            "algorithm": "fp32_products_fp64_tree_sum", "block_m": 1,
            "block_n": HEAD_BLOCK_N, "block_k": HEAD_BLOCK_K,
            "num_warps": HEAD_NUM_WARPS, "num_stages": HEAD_NUM_STAGES,
            "bias_add_dtype": None, "bias": False,
            "scope": "pinned TP1 Qwen2 head; fixed FP64 reduction, no performance claim"}


def validate_head_contract(x_shape, weight_shape, strides, dtypes, devices):
    if (len(x_shape) != 2 or x_shape[0] < 1 or x_shape[1] != 2048
            or weight_shape != (151936, 2048)):
        raise ValueError("head requires actual pinned K2048/N151936")
    if strides != ((2048, 1), (2048, 1)) or dtypes != ("torch.float16",) * 2:
        raise ValueError("head requires contiguous FP16 stored operands")
    if len(devices) != 2 or not devices[0].startswith("cuda") or devices[0] != devices[1]:
        raise ValueError("head operands must share one CUDA device")


# CPU-only tooling may import the contract without Torch/Triton installed. There
# is deliberately no arithmetic fallback masquerading as this diagnostic.
try:
    import torch
    import triton
    import triton.language as tl
except ImportError:
    def batch_invariant_linear(x, weight, bias):
        raise RuntimeError("diagnostic fixed QKV requires Torch2.7 and Triton on CUDA")

    def fixed_tile_linear(x, weight, bias=None):
        raise RuntimeError("diagnostic fixed linear requires Torch2.7 and Triton on CUDA")

    def fixed_tile_head(x, weight):
        raise RuntimeError("diagnostic fixed head requires Torch2.7 and Triton on CUDA")
else:
    @triton.jit(do_not_specialize=["M"])
    def _fixed_qkv(X, W, B, Y, M, N: tl.constexpr, K: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, HAS_BIAS: tl.constexpr = True,
                   OUTPUT_FP32: tl.constexpr = False):
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        cols = tl.program_id(1) * BN + tl.arange(0, BN)
        reduction = tl.arange(0, BK)
        accum = tl.full((BM, BN), 0, tl.float32)
        for block in range(tl.cdiv(K, BK)):
            kk = block * BK + reduction
            x = tl.load(X + rows[:, None] * K + kk[None, :],
                        mask=(rows[:, None] < M) & (kk[None, :] < K), other=0)
            w = tl.load(W + cols[None, :] * K + kk[:, None],
                        mask=(cols[None, :] < N) & (kk[:, None] < K), other=0)
            accum = tl.dot(x, w, accum)
        if HAS_BIAS:
            bias = tl.load(B + cols, mask=cols < N, other=0).to(tl.float32)
            value = accum + bias[None, :]
        else:
            value = accum
        if not OUTPUT_FP32:
            value = value.to(tl.float16)
        tl.store(Y + rows[:, None] * N + cols[None, :], value,
                 mask=(rows[:, None] < M) & (cols[None, :] < N))

    @torch.library.triton_op(OPERATOR_NAME, mutates_args={})
    def batch_invariant_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        validate_contract(tuple(x.shape), tuple(weight.shape), tuple(bias.shape),
                          (x.stride(), weight.stride(), bias.stride()),
                          (str(x.dtype), str(weight.dtype), str(bias.dtype)),
                          (str(x.device), str(weight.device), str(bias.device)))
        m, k = x.shape
        n = weight.shape[0]
        output = torch.empty((m, n), device=x.device, dtype=x.dtype)
        torch.library.wrap_triton(_fixed_qkv)[(triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))](
            x, weight, bias, output, m, n, k, BLOCK_M, BLOCK_N, BLOCK_K,
            num_warps=NUM_WARPS, num_stages=NUM_STAGES)
        return output


    @torch.library.triton_op(LINEAR_OPERATOR_NAME, mutates_args={})
    def fixed_tile_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        operands = (x, weight) if bias is None else (x, weight, bias)
        validate_linear_contract(tuple(x.shape), tuple(weight.shape), None if bias is None else tuple(bias.shape),
                                 tuple(t.stride() for t in operands), tuple(str(t.dtype) for t in operands),
                                 tuple(str(t.device) for t in operands))
        m, k = x.shape
        n = weight.shape[0]
        output = torch.empty((m, n), device=x.device, dtype=x.dtype)
        torch.library.wrap_triton(_fixed_qkv)[(triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))](
            x, weight, weight if bias is None else bias, output, m, n, k, BLOCK_M, BLOCK_N, BLOCK_K,
            HAS_BIAS=bias is not None, num_warps=NUM_WARPS, num_stages=NUM_STAGES)
        return output


    @triton.jit
    def _fixed_head(X, W, Y, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.program_id(1) * BN + tl.arange(0, BN)
        kk = tl.arange(0, K)
        x = tl.load(X + row * K + kk).to(tl.float32)
        w = tl.load(W + cols[:, None] * K + kk[None, :],
                    mask=cols[:, None] < N, other=0).to(tl.float32)
        # Every finite FP16 product is exactly representable in FP32. Only the
        # fixed K reduction uses FP64; weights remain stored/loaded as FP16.
        products = x[None, :] * w
        summed = tl.sum(products.to(tl.float64), axis=1)
        tl.store(Y + row * N + cols, summed.to(tl.float32), mask=cols < N)

    @torch.library.triton_op(HEAD_OPERATOR_NAME, mutates_args={})
    def fixed_tile_head(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        validate_head_contract(tuple(x.shape), tuple(weight.shape), (x.stride(), weight.stride()),
                               (str(x.dtype), str(weight.dtype)), (str(x.device), str(weight.device)))
        m, k = x.shape
        n = weight.shape[0]
        output = torch.empty((m, n), device=x.device, dtype=torch.float32)
        torch.library.wrap_triton(_fixed_head)[(m, triton.cdiv(n, HEAD_BLOCK_N))](
            x, weight, output, n, HEAD_BLOCK_K, HEAD_BLOCK_N,
            num_warps=HEAD_NUM_WARPS, num_stages=HEAD_NUM_STAGES)
        return output
