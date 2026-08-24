"""Custom Qwen2 inference forward path (PRD FR2).

Implements only what Qwen2.5-0.5B needs at inference time: token embedding,
RMSNorm, rotary embeddings, grouped-query causal attention, SwiGLU MLP,
residuals, final norm and the tied LM head. No Hugging Face modules, no
``generate()``, no HF cache classes anywhere on this path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch as _torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("model.py requires torch (cloud image only)") from exc

from .attention import AttentionBackend, StepContext, apply_rope, build_rope_cache, rms_norm


@dataclass(frozen=True)
class ModelDims:
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    attention_bias: bool
    tie_word_embeddings: bool
    eos_token_id: int

    @classmethod
    def from_hf_config(cls, config: dict[str, Any]) -> ModelDims:
        required = {
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "vocab_size",
            "rms_norm_eps",
            "rope_theta",
            "tie_word_embeddings",
            "eos_token_id",
        }
        missing = required - set(config)
        if missing or config.get("model_type") != "qwen2":
            raise ValueError(f"unsupported/missing model config fields: {sorted(missing)}")
        hidden_size = config["hidden_size"]
        num_heads = config["num_attention_heads"]
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size not divisible by num_attention_heads")
        return cls(
            hidden_size=hidden_size,
            num_layers=config["num_hidden_layers"],
            num_heads=num_heads,
            num_kv_heads=config["num_key_value_heads"],
            head_dim=hidden_size // num_heads,
            intermediate_size=config["intermediate_size"],
            vocab_size=config["vocab_size"],
            rms_norm_eps=config["rms_norm_eps"],
            rope_theta=config["rope_theta"],
            attention_bias=config.get("attention_bias", True),
            tie_word_embeddings=config["tie_word_embeddings"],
            eos_token_id=config["eos_token_id"],
        )


class Qwen2MLP(nn.Module):
    def __init__(self, dims: ModelDims, dtype: Any, device: str) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            dims.hidden_size, dims.intermediate_size, bias=False, dtype=dtype, device=device
        )
        self.up_proj = nn.Linear(
            dims.hidden_size, dims.intermediate_size, bias=False, dtype=dtype, device=device
        )
        self.down_proj = nn.Linear(
            dims.intermediate_size, dims.hidden_size, bias=False, dtype=dtype, device=device
        )

    def forward(self, x: Any) -> Any:
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2Attention(nn.Module):
    def __init__(self, dims: ModelDims, dtype: Any, device: str) -> None:
        super().__init__()
        self.q_proj = nn.Linear(
            dims.hidden_size,
            dims.num_heads * dims.head_dim,
            bias=dims.attention_bias,
            dtype=dtype,
            device=device,
        )
        self.k_proj = nn.Linear(
            dims.hidden_size,
            dims.num_kv_heads * dims.head_dim,
            bias=dims.attention_bias,
            dtype=dtype,
            device=device,
        )
        self.v_proj = nn.Linear(
            dims.hidden_size,
            dims.num_kv_heads * dims.head_dim,
            bias=dims.attention_bias,
            dtype=dtype,
            device=device,
        )
        self.o_proj = nn.Linear(
            dims.num_heads * dims.head_dim, dims.hidden_size, bias=False, dtype=dtype, device=device
        )

    def project(self, x: Any) -> tuple[Any, Any, Any]:
        seq_len = x.shape[0]
        q = self.q_proj(x).view(seq_len, -1, self.head_dim)
        k = self.k_proj(x).view(seq_len, -1, self.head_dim)
        v = self.v_proj(x).view(seq_len, -1, self.head_dim)
        return q, k, v


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, dims: ModelDims, dtype: Any, device: str) -> None:
        super().__init__()
        self.input_layernorm = Qwen2RMSNorm(dims, dtype, device)
        self.self_attn = Qwen2Attention(dims, dtype, device)
        self.post_attention_layernorm = Qwen2RMSNorm(dims, dtype, device)
        self.mlp = Qwen2MLP(dims, dtype, device)


class Qwen2RMSNorm(nn.Module):
    def __init__(self, dims: ModelDims, dtype: Any, device: str) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch_empty(dims.hidden_size, dtype=dtype, device=device))
        self.eps = dims.rms_norm_eps

    def forward(self, x: Any) -> Any:
        return rms_norm(x, self.weight, self.eps)


def torch_empty(*args: Any, **kwargs: Any) -> Any:
    return _torch.empty(*args, **kwargs)


class Qwen2CausalLM(nn.Module):
    """Engine-owned Qwen2 model. Module names intentionally mirror HF so the
    explicit loader can map safetensors keys one-to-one."""

    def __init__(
        self, dims: ModelDims, attn_backend: AttentionBackend, dtype: Any, device: str
    ) -> None:
        super().__init__()
        self.dims = dims
        self.attn_backend = attn_backend
        self.embed_tokens = nn.Embedding(
            dims.vocab_size, dims.hidden_size, dtype=dtype, device=device
        )
        self.layers = nn.ModuleList(
            Qwen2DecoderLayer(dims, dtype, device) for _ in range(dims.num_layers)
        )
        self.norm = Qwen2RMSNorm(dims, dtype, device)
        if not dims.tie_word_embeddings:
            raise ValueError("v1 requires tied word embeddings (Qwen2.5-0.5B)")
        self._rope_cos: Any = None
        self._rope_sin: Any = None

    def _ensure_rope(self, device: str) -> None:
        if self._rope_cos is None or self._rope_cos.device != _torch.device(device):
            self._rope_cos, self._rope_sin = build_rope_cache(
                self.dims.head_dim, 32768, self.dims.rope_theta, device
            )

    def forward(
        self,
        input_ids: Any,
        ctx: StepContext | None,
        return_all_logits: bool = False,
    ) -> Any:
        """input_ids: [seq] LongTensor on device. Returns logits [seq', vocab] fp32."""
        device = input_ids.device
        self._ensure_rope(str(device))
        seq_len = input_ids.shape[0]
        first_position = ctx.kv_start if ctx is not None else 0
        positions = _torch.arange(first_position, first_position + seq_len, device=device)
        hidden = self.embed_tokens(input_ids)
        for layer_index, layer in enumerate(self.layers):
            residual = hidden
            attn_in = layer.input_layernorm(hidden)
            q, k, v = layer.self_attn.project(attn_in)
            q, k = apply_rope(q, k, self._rope_cos, self._rope_sin, positions)
            context = self.attn_backend.attend(layer_index, ctx, q, k, v)
            hidden = residual + layer.self_attn.o_proj(context)
            residual = hidden
            mlp_in = layer.post_attention_layernorm(hidden)
            hidden = residual + layer.mlp(mlp_in)
        hidden = self.norm(hidden)
        selected = hidden if return_all_logits else hidden[-1:]
        logits = _torch.mm(
            selected.to(_torch.float32), self.embed_tokens.weight.to(_torch.float32).t()
        )
        return logits


def greedy_sample(logits: Any) -> int:
    """Greedy decoding: argmax over final-position logits in fp32."""
    return int(_torch.argmax(logits[-1].to(_torch.float32)).item())


def load_model_config(model_dir: Path) -> ModelDims:
    with open(model_dir / "config.json", encoding="utf-8") as handle:
        raw = json.load(handle)
    return ModelDims.from_hf_config(raw)
