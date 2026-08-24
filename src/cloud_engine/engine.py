"""InferenceEngine: owns tokenizer, model, cache backend, and scheduler.

Dependency direction (PRD §8): api -> engine -> model -> attention/cache.
The engine is the single authoritative cache owner; API handlers interact
only through ``submit``/``RequestHandle``.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .attention import AttentionBackend, StepContext, TritonDecodeAttentionBackend
from .cache import ContiguousKVCache, PagedKVCache
from .config import EngineConfig
from .metrics import Metrics
from .model import Qwen2CausalLM, greedy_sample, load_model_config
from .scheduler import GenerationConfig, Request, RequestState, Scheduler, StreamEvent
from .weights import load_model, load_tokenizer


@dataclass
class GenerationResult:
    request_id: str
    text: str
    token_ids: list[int]
    input_tokens: int
    output_tokens: int
    finish_reason: str
    ttft_ms: float | None
    e2e_ms: float


class RunnerBase:
    """Scheduler-facing execution interface. step() runs in a worker thread."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config

    def admit(self, request: Request) -> None: ...
    def release(self, request: Request) -> None: ...


class NaiveRunner(RunnerBase):
    """Baseline: recompute prompt + all generated tokens every step (PRD FR3)."""

    def __init__(self, config: EngineConfig, model: Qwen2CausalLM, device: str) -> None:
        super().__init__(config)
        self.model = model
        self.device = device

    def _full_ids(self, request: Request):
        import torch

        ids = list(request.prompt_token_ids)
        if request.tokens_fed > 0:
            ids.extend(request.generated_token_ids[: request.tokens_fed])
        return torch.tensor([ids], dtype=torch.long, device=self.device)[0]

    def step(self, request: Request) -> int:
        input_ids = self._full_ids(request)
        token_id = greedy_sample(self.model(input_ids, ctx=None))
        request.tokens_fed += 1
        return token_id


class CachedRunner(RunnerBase):
    """Prefill once into the KV backend, then decode one token per step."""

    def __init__(
        self,
        config: EngineConfig,
        model: Qwen2CausalLM,
        cache: Any,
        device: str,
    ) -> None:
        super().__init__(config)
        self.model = model
        self.cache = cache
        self.device = device

    def admit(self, request: Request) -> None:
        capacity = len(request.prompt_token_ids) + request.config.max_output_tokens
        self.cache.reserve(request.request_id, capacity)

    def release(self, request: Request) -> None:
        self.cache.release(request.request_id)

    def _ctx(self, request: Request, seq_len: int, is_decode: bool) -> StepContext:
        kv_start = seq_len - (1 if is_decode else len(request.prompt_token_ids))
        return StepContext(
            request_id=request.request_id,
            kv_start=max(kv_start, 0),
            is_decode=is_decode,
        )

    def step(self, request: Request) -> int:
        import torch

        cached_len = request.tokens_fed  # tokens already written to the cache
        if cached_len == 0:
            input_ids = torch.tensor(request.prompt_token_ids, dtype=torch.long, device=self.device)
            ctx = self._ctx(request, len(request.prompt_token_ids), is_decode=False)
            logits = self.model(input_ids, ctx=ctx)
        else:
            next_token = request.generated_token_ids[cached_len - 1]
            input_ids = torch.tensor([next_token], dtype=torch.long, device=self.device)
            total = cached_len + 1
            ctx = self._ctx(request, total, is_decode=True)
            logits = self.model(input_ids, ctx=ctx)
        request.tokens_fed += 1
        return greedy_sample(logits)


class RequestHandle:
    """Consumer-side view of one request: stream tokens or await completion."""

    def __init__(self, request: Request, engine: InferenceEngine) -> None:
        self.request = request
        self._engine = engine

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def state(self) -> RequestState:
        return self.request.state

    async def stream(self) -> AsyncIterator[StreamEvent]:
        """Yield StreamEvents until STOP. Cancels the request on early exit."""
        try:
            while True:
                event = await self.request.output_queue.get()
                if event.finished:
                    return
                yield event
        finally:
            if not self.request.is_terminal:
                self._engine.scheduler.cancel(self.request)

    async def wait(self) -> GenerationResult:
        request = await self.request.terminal_future
        if request.state is not RequestState.COMPLETED:
            raise RuntimeError(
                f"generation {request.state.value}: {request.error_detail or request.finish_reason}"
            )
        now_ns = self._engine.metrics.now()
        e2e_ms = (now_ns - request.arrival_ns) / 1e6
        text = self._engine.detokenize(request.generated_token_ids)
        return GenerationResult(
            request_id=request.request_id,
            text=text,
            token_ids=list(request.generated_token_ids),
            input_tokens=len(request.prompt_token_ids),
            output_tokens=request.generated_count,
            finish_reason=request.finish_reason,
            ttft_ms=(
                (request.first_token_ns - request.arrival_ns) / 1e6
                if request.first_token_ns
                else None
            ),
            e2e_ms=e2e_ms,
        )

    def cancel(self) -> None:
        self._engine.scheduler.cancel(self.request)


class InferenceEngine:
    """Public engine surface used by the API layer and benchmarks."""

    def __init__(
        self,
        config: EngineConfig,
        model_dir: Any | None = None,
        runner_factory: Callable[[EngineConfig], RunnerBase] | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.device = device or ("cuda" if self._cuda_available() else "cpu")
        self._custom_runner_factory = runner_factory
        self.metrics = Metrics()
        self.tokenizer: Any = None
        self.model: Qwen2CausalLM | None = None
        self.cache: Any = None
        self.runner: RunnerBase | None = None
        self.scheduler: Scheduler | None = None
        self.ready = False

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    # ------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        if self.ready:
            return
        await asyncio.to_thread(self._initialize)
        self.scheduler = Scheduler(self.config, self.runner, self.metrics)
        await self.scheduler.start()

    async def close(self) -> None:
        if self.scheduler is not None:
            for request in (*self.scheduler.active, *self.scheduler.waiting):
                self.scheduler.cancel(request)
            await self.scheduler.stop()
        self.ready = False

    def _initialize(self) -> None:
        import torch

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        if self._custom_runner_factory is not None:
            # Test path: deterministic fake model injected without Torch weights.
            self.runner = self._custom_runner_factory(self.config)
            self.scheduler = Scheduler(self.config, self.runner, self.metrics)
            self.ready = True
            return

        if self.model_dir is None:
            raise RuntimeError("model_dir required unless a fake runner is injected")
        dims = load_model_config(self.model_dir)

        if self.config.mode == "naive":
            self.cache = None
            attn_backend = AttentionBackend(
                cache=None,
                num_heads=dims.num_heads,
                num_kv_heads=dims.num_kv_heads,
                head_dim=dims.head_dim,
            )
        elif self.config.mode in ("contiguous", "batched"):
            self.cache = ContiguousKVCache(
                dims.num_layers,
                dims.num_kv_heads,
                dims.head_dim,
                dtype=dtype,
                device=self.device,
            )
            attn_backend = AttentionBackend(
                self.cache, dims.num_heads, dims.num_kv_heads, dims.head_dim
            )
        elif self.config.mode == "paged":
            self.cache = PagedKVCache(
                dims.num_layers,
                dims.num_kv_heads,
                dims.head_dim,
                block_size=self.config.block_size,
                kv_cache_bytes=self.config.kv_cache_bytes,
                dtype=dtype,
                device=self.device,
            )
            attn_backend = AttentionBackend(
                self.cache, dims.num_heads, dims.num_kv_heads, dims.head_dim
            )
        elif self.config.mode == "triton":
            self.cache = PagedKVCache(
                dims.num_layers,
                dims.num_kv_heads,
                dims.head_dim,
                block_size=self.config.block_size,
                kv_cache_bytes=self.config.kv_cache_bytes,
                dtype=dtype,
                device=self.device,
            )
            attn_backend = TritonDecodeAttentionBackend(
                self.cache,
                dims.num_heads,
                dims.num_kv_heads,
                dims.head_dim,
                allow_reference_fallback=self.config.allow_reference_fallback,
            )
        else:  # pragma: no cover - validated in config
            raise ValueError(f"unsupported mode {self.config.mode}")

        self.model, _ = load_model(self.model_dir, attn_backend, dtype=dtype, device=self.device)
        self.tokenizer = load_tokenizer(self.model_dir)
        self.runner = (
            NaiveRunner(self.config, self.model, self.device)
            if self.config.mode == "naive"
            else CachedRunner(self.config, self.model, self.cache, self.device)
        )
        self.ready = True

    # ---------------------------------------------------------------- usage
    async def submit(
        self, prompt: str, gen_config: GenerationConfig | None = None
    ) -> RequestHandle:
        assert self.scheduler is not None and self.ready
        gen_config = gen_config or GenerationConfig(
            max_output_tokens=self.config.max_output_tokens,
            temperature=0.0,
            eos_token_id=self.config.eos_token_id,
        )
        prompt_ids = self.count_token_ids(prompt)
        request = await self.scheduler.submit(prompt, prompt_ids, gen_config)
        return RequestHandle(request, self)

    def count_token_ids(self, text: str) -> list[int]:
        if self.tokenizer is not None:
            return list(self.tokenizer.encode(text))
        return [1] * max(1, len(text) // 4)  # crude stand-in when no tokenizer (tests)

    def detokenize(self, token_ids: list[int]) -> str:
        if self.tokenizer is None or not token_ids:
            return ""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def snapshot_metrics(self) -> dict[str, Any]:
        base = self.metrics.snapshot()
        if self.scheduler is not None:
            base["requests"]["waiting"] = len(self.scheduler.waiting)
            base["requests"]["active"] = len(self.scheduler.active)
        else:
            base["requests"]["waiting"] = 0
            base["requests"]["active"] = 0
        if self.cache is not None:
            base["kv_cache"].update(self.cache.stats().as_metrics())
            try:
                self.metrics.set_kv_stats(base["kv_cache"])
            except Exception:
                pass
        if self.device == "cuda":
            import torch

            self.metrics.set_gpu_bytes(
                allocated=torch.cuda.memory_allocated(),
                reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(),
            )
            base["gpu"] = dict(self.metrics._gpu)
        return base

    def reset_peak_memory(self) -> None:
        if self.device == "cuda":
            import torch

            torch.cuda.reset_peak_memory_stats()

    def new_generation_config(self, max_output_tokens: int) -> GenerationConfig:
        return GenerationConfig(
            max_output_tokens=min(max_output_tokens, self.config.max_output_tokens),
            temperature=0.0,
            eos_token_id=self.config.eos_token_id,
        )


def make_request_id() -> str:
    return f"resp_{uuid.uuid4().hex}"
