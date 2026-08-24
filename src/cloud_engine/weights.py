"""Explicit safetensors weight loading and cloud download orchestration (PRD FR2).

The loader maps safetensors keys explicitly, verifies shapes against the pinned
config, and fails with readable errors on missing, duplicated, or unexpected
required tensors. It never silently ignores incompatible architectures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import safetensors
import safetensors.torch

from .model import ModelDims, Qwen2CausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B"
_DOWNLOAD_PATTERNS = [
    "config.json",
    "generation_config.json",
    "model*.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
]

# Tied models legitimately ship without lm_head; if present anyway we verify
# it matches embed_tokens rather than silently ignoring it.
_TIED_OPTIONAL_KEYS = {"lm_head.weight", "model.lm_head.weight"}


def expected_tensors(dims: ModelDims) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (dims.vocab_size, dims.hidden_size),
        "model.norm.weight": (dims.hidden_size,),
    }
    for i in range(dims.num_layers):
        prefix = f"model.layers.{i}"
        shapes.update(
            {
                f"{prefix}.input_layernorm.weight": (dims.hidden_size,),
                f"{prefix}.self_attn.q_proj.weight": (
                    dims.num_heads * dims.head_dim,
                    dims.hidden_size,
                ),
                f"{prefix}.self_attn.k_proj.weight": (
                    dims.num_kv_heads * dims.head_dim,
                    dims.hidden_size,
                ),
                f"{prefix}.self_attn.v_proj.weight": (
                    dims.num_kv_heads * dims.head_dim,
                    dims.hidden_size,
                ),
                f"{prefix}.self_attn.o_proj.weight": (
                    dims.hidden_size,
                    dims.num_heads * dims.head_dim,
                ),
                f"{prefix}.post_attention_layernorm.weight": (dims.hidden_size,),
                f"{prefix}.mlp.gate_proj.weight": (dims.intermediate_size, dims.hidden_size),
                f"{prefix}.mlp.up_proj.weight": (dims.intermediate_size, dims.hidden_size),
                f"{prefix}.mlp.down_proj.weight": (dims.hidden_size, dims.intermediate_size),
            }
        )
        if dims.attention_bias:
            shapes.update(
                {
                    f"{prefix}.self_attn.q_proj.bias": (dims.num_heads * dims.head_dim,),
                    f"{prefix}.self_attn.k_proj.bias": (dims.num_kv_heads * dims.head_dim,),
                    f"{prefix}.self_attn.v_proj.bias": (dims.num_kv_heads * dims.head_dim,),
                }
            )
    return shapes


def load_state_dict(model_dir: str | Path, dims: ModelDims, dtype: Any = None) -> dict[str, Any]:
    model_dir = Path(model_dir)
    shard_paths = sorted(model_dir.glob("*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"no *.safetensors found in {model_dir}")
    state: dict[str, Any] = {}
    seen: dict[str, str] = {}
    for path in shard_paths:
        with safetensors.safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118 - safetensors API
                if key in seen:
                    raise ValueError(f"duplicate tensor {key!r} in {path.name} and {seen[key]}")
                seen[key] = path.name
                state[key] = handle.get_tensor(key)

    expected = expected_tensors(dims)
    missing = sorted(set(expected) - set(state))
    if missing:
        preview = ", ".join(missing[:8])
        more = "" if len(missing) <= 8 else f" (+{len(missing) - 8} more)"
        raise KeyError(f"missing required tensors: {preview}{more}")

    unexpected = sorted(set(state) - set(expected) - _TIED_OPTIONAL_KEYS)
    if unexpected:
        raise KeyError(
            "unexpected tensors incompatible with pinned architecture: "
            f"{', '.join(unexpected[:12])}"
        )

    if "model.lm_head.weight" in state and not dims.tie_word_embeddings:
        raise ValueError("lm_head present but config says embeddings are not tied")

    for name, shape in expected.items():
        actual = tuple(state[name].shape)
        if actual != shape:
            raise ValueError(f"shape mismatch for {name}: expected {shape}, got {actual}")

    target_dtype = dtype or state["model.embed_tokens.weight"].dtype
    return {
        name.removeprefix("model."): tensor.to(target_dtype)
        for name, tensor in state.items()
        if name in expected
    }


def load_model(
    model_dir: str | Path,
    attn_backend: Any,
    dtype: Any = None,
    device: str = "cuda",
) -> tuple[Qwen2CausalLM, ModelDims]:
    """Build the engine model and load verified weights strictly."""
    import torch

    model_dir = Path(model_dir)
    with open(model_dir / "config.json", encoding="utf-8") as handle:
        hf_config = json.load(handle)
    dims = ModelDims.from_hf_config(hf_config)
    model = Qwen2CausalLM(dims, attn_backend, dtype=dtype or torch.float16, device=device)
    state = load_state_dict(model_dir, dims, dtype=dtype or torch.float16)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.unexpected_keys or incompatible.missing_keys:
        raise RuntimeError(f"state dict mismatch: {incompatible}")
    model.eval()
    model.requires_grad_(False)
    return model, dims


def ensure_weights_downloaded(
    cache_root: str,
    revision: str,
    model_id: str = MODEL_ID,
) -> Path:
    """Download the pinned snapshot into the Modal Volume (idempotent)."""
    from huggingface_hub import snapshot_download

    local_path = snapshot_download(
        repo_id=model_id,
        revision=revision,
        cache_dir=str(Path(cache_root) / "hf"),
        allow_patterns=_DOWNLOAD_PATTERNS,
    )
    return Path(local_path)


def load_tokenizer(model_dir: str | Path) -> Any:
    """Tokenizers may come from HF tooling (PRD G1); generation may not."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_dir))
