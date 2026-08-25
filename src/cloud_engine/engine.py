"""InferenceEngine: owns tokenizer, model, cache backend, and scheduler.

Dependency direction (PRD §8): api -> engine -> model -> attention/cache.
The engine is the single authoritative cache owner; API handlers interact
only through ``submit``/``RequestHandle``.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .attention import (
    AttentionBackend,
    PackedContext,
    RaggedTritonAttentionBackend,
    StepContext,
    TritonDecodeAttentionBackend,
)
from .cache import ContiguousKVCache, PagedKVCache
from .config import EngineConfig
from .metrics import Metrics
from .model import Qwen2CausalLM, greedy_sample, load_model_config
from .profiling import nvtx_range
from .scheduler import (
    BatchPlan,
    GenerationConfig,
    Request,
    RequestState,
    Scheduler,
    SchedulingCandidate,
    StreamEvent,
    decode_first_priority,
)
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


@dataclass(frozen=True)
class PackedBatch:
    """One flat token axis and all device metadata needed by ragged attention."""

    input_ids: Any
    context: PackedContext
    logit_indices: Any
    sampled_request_ids: tuple[str, ...]


class RunnerBase:
    """Scheduler-facing execution interface. step() runs in a worker thread."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config

    def admit(self, request: Request) -> None: ...
    def release(self, request: Request) -> None: ...


class NaiveRunner(RunnerBase):
    """Baseline: recompute prompt + all generated tokens every step (PRD FR3)."""

    def __init__(
        self,
        config: EngineConfig,
        model: Qwen2CausalLM,
        cache: ContiguousKVCache,
        device: str,
    ) -> None:
        super().__init__(config)
        self.model = model
        self.cache = cache
        self.device = device

    def _full_ids(self, request: Request):
        import torch

        ids = list(request.prompt_token_ids)
        if request.tokens_fed > 0:
            ids.extend(request.generated_token_ids[: request.tokens_fed])
        return torch.tensor([ids], dtype=torch.long, device=self.device)[0]

    def step(self, request: Request) -> int:
        import torch

        input_ids = self._full_ids(request)
        logits = self.model(input_ids, ctx=None)
        top_two = torch.topk(logits[-1], 2).values
        # ponytail: replay only FP16 near-ties; use deterministic kernels if this
        # threshold ever needs tuning for another model or accelerator.
        if float(top_two[0] - top_two[1]) < 0.05:
            logits = self._canonical_replay(request)
        token_id = greedy_sample(logits)
        request.tokens_fed += 1
        return token_id

    def _canonical_replay(self, request: Request):
        """Resolve a near-tie with cached arithmetic, retaining no KV state."""
        import torch

        capacity = len(request.prompt_token_ids) + request.tokens_fed
        self.cache.reserve(request.request_id, capacity)
        try:
            prompt_ids = torch.tensor(
                request.prompt_token_ids, dtype=torch.long, device=self.device
            )
            logits = self.model(
                prompt_ids,
                ctx=StepContext(request.request_id, kv_start=0, is_decode=False),
            )
            for index, token_id in enumerate(request.generated_token_ids[: request.tokens_fed]):
                input_ids = torch.tensor([token_id], dtype=torch.long, device=self.device)
                logits = self.model(
                    input_ids,
                    ctx=StepContext(
                        request.request_id,
                        kv_start=len(request.prompt_token_ids) + index,
                        is_decode=True,
                    ),
                )
            return logits
        finally:
            self.cache.release(request.request_id)


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

    def _ctx(self, request: Request, is_decode: bool) -> StepContext:
        kv_start = len(request.prompt_token_ids) + request.tokens_fed - 1 if is_decode else 0
        return StepContext(
            request_id=request.request_id,
            kv_start=kv_start,
            is_decode=is_decode,
        )

    def step(self, request: Request) -> int:
        import torch

        cached_len = request.tokens_fed  # tokens already written to the cache
        if cached_len == 0:
            input_ids = torch.tensor(request.prompt_token_ids, dtype=torch.long, device=self.device)
            ctx = self._ctx(request, is_decode=False)
            logits = self.model(input_ids, ctx=ctx)
        else:
            next_token = request.generated_token_ids[cached_len - 1]
            input_ids = torch.tensor([next_token], dtype=torch.long, device=self.device)
            ctx = self._ctx(request, is_decode=True)
            logits = self.model(input_ids, ctx=ctx)
        request.tokens_fed += 1
        return greedy_sample(logits)


class RaggedRunner(RunnerBase):
    """Pack multiple sequences into one model invocation per scheduler iteration."""

    def __init__(
        self,
        config: EngineConfig,
        model: Qwen2CausalLM,
        cache: PagedKVCache,
        device: str,
    ) -> None:
        super().__init__(config)
        self.model = model
        self.cache = cache
        self.device = device
        self.forward_invocations = 0
        self.max_forward_request_count = 0
        self.forward_traces: deque[tuple[str, ...]] = deque(maxlen=256)
        self.forward_plans: deque[tuple[tuple[str, int, int, bool], ...]] = deque(
            maxlen=256
        )

    def admit(self, request: Request) -> None:
        self.cache.reserve(request.request_id, 0)

    def release(self, request: Request) -> None:
        self.cache.release(request.request_id)

    def allocated_tokens(self, request: Request) -> int:
        return self.cache.allocated_tokens(request.request_id)

    def _pack(self, plan: BatchPlan) -> PackedBatch:
        import torch

        required = {item.request.request_id: item.end_pos for item in plan.items}
        with nvtx_range("ragged.kv.ensure_capacity"):
            self.cache.ensure_capacity_batch(required)

        input_ids: list[int] = []
        positions: list[int] = []
        token_request_ids: list[str] = []
        query_seq_ids: list[int] = []
        query_ranges: list[tuple[int, int]] = []
        query_start_loc = [0]
        context_lens: list[int] = []
        final_lens: list[int] = []
        sampled_request_ids: list[str] = []
        logit_indices: list[int] = []

        for sequence_index, item in enumerate(plan.items):
            start = len(input_ids)
            input_ids.extend(item.token_ids)
            positions.extend(range(item.start_pos, item.end_pos))
            token_request_ids.extend([item.request.request_id] * item.query_length)
            query_seq_ids.extend([sequence_index] * item.query_length)
            end = len(input_ids)
            query_ranges.append((start, end))
            query_start_loc.append(end)
            context_lens.append(item.start_pos)
            final_lens.append(item.end_pos)
            if item.sample:
                sampled_request_ids.append(item.request.request_id)
                logit_indices.append(end - 1)

        tables = [self.cache.block_table(item.request.request_id) for item in plan.items]
        width = max((len(table) for table in tables), default=1)
        block_tables = torch.zeros(
            (len(tables), width), dtype=torch.int32, device=self.device
        )
        for row, table in enumerate(tables):
            if table:
                block_tables[row, : len(table)] = torch.tensor(
                    table, dtype=torch.int32, device=self.device
                )

        slot_mapping = self.cache.slot_mapping(token_request_ids, positions)
        context = PackedContext(
            request_ids=tuple(item.request.request_id for item in plan.items),
            query_ranges=tuple(query_ranges),
            context_lens=tuple(context_lens),
            final_lens=tuple(final_lens),
            positions=torch.tensor(positions, dtype=torch.long, device=self.device),
            query_start_loc=torch.tensor(
                query_start_loc, dtype=torch.int32, device=self.device
            ),
            block_tables=block_tables,
            slot_mapping=torch.tensor(slot_mapping, dtype=torch.long, device=self.device),
            query_seq_ids=torch.tensor(
                query_seq_ids, dtype=torch.int32, device=self.device
            ),
        )
        return PackedBatch(
            input_ids=torch.tensor(input_ids, dtype=torch.long, device=self.device),
            context=context,
            logit_indices=torch.tensor(logit_indices, dtype=torch.long, device=self.device),
            sampled_request_ids=tuple(sampled_request_ids),
        )

    def execute_batch(self, plan: BatchPlan) -> dict[str, int]:
        import torch

        packed = self._pack(plan)
        self.forward_invocations += 1
        self.max_forward_request_count = max(
            self.max_forward_request_count, len(plan.request_ids)
        )
        self.forward_traces.append(plan.request_ids)
        self.forward_plans.append(
            tuple(
                (
                    item.request.request_id,
                    item.start_pos,
                    item.query_length,
                    item.sample,
                )
                for item in plan.items
            )
        )
        phase = "decode" if all(item.is_decode for item in plan.items) else "mixed"
        with nvtx_range(
            f"ragged.forward.{phase}.requests_{len(plan.items)}.tokens_{plan.token_count}"
        ):
            logits = self.model(
                packed.input_ids,
                ctx=packed.context,
                logit_indices=packed.logit_indices,
            )
        with nvtx_range("ragged.kv.commit"):
            self.cache.commit_lengths(
                {item.request.request_id: item.end_pos for item in plan.items}
            )
        for item in plan.items:
            item.request.tokens_fed = item.end_pos
        if not packed.sampled_request_ids:
            return {}
        with nvtx_range("ragged.sample.greedy"):
            token_ids = torch.argmax(logits.to(torch.float32), dim=-1).tolist()
        return dict(zip(packed.sampled_request_ids, token_ids, strict=True))


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
        if not self.request.terminal_future.done():
            async for _ in self.stream():
                pass
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
        scheduling_priority: Callable[[SchedulingCandidate], tuple[Any, ...]] = decode_first_priority,
    ) -> None:
        self.config = config
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.device = device or ("cuda" if self._cuda_available() else "cpu")
        self._custom_runner_factory = runner_factory
        self.scheduling_priority = scheduling_priority
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
        self.scheduler = Scheduler(
            self.config, self.runner, self.metrics, scheduling_priority=self.scheduling_priority
        )
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
            self.scheduler = Scheduler(
                self.config,
                self.runner,
                self.metrics,
                scheduling_priority=self.scheduling_priority,
            )
            self.ready = True
            return

        if self.model_dir is None:
            raise RuntimeError("model_dir required unless a fake runner is injected")
        dims = load_model_config(self.model_dir)

        if self.config.mode == "naive":
            self.cache = ContiguousKVCache(
                dims.num_layers,
                dims.num_kv_heads,
                dims.head_dim,
                dtype=dtype,
                device=self.device,
            )
            attn_backend = AttentionBackend(
                cache=self.cache,
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
        elif self.config.mode in ("triton", "ragged"):
            self.cache = PagedKVCache(
                dims.num_layers,
                dims.num_kv_heads,
                dims.head_dim,
                block_size=self.config.block_size,
                kv_cache_bytes=self.config.kv_cache_bytes,
                dtype=dtype,
                device=self.device,
            )
            backend_type = (
                RaggedTritonAttentionBackend
                if self.config.mode == "ragged"
                else TritonDecodeAttentionBackend
            )
            attn_backend = backend_type(
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
        if self.config.mode == "naive":
            self.runner = NaiveRunner(self.config, self.model, self.cache, self.device)
        elif self.config.mode == "ragged":
            self.runner = RaggedRunner(self.config, self.model, self.cache, self.device)
        else:
            self.runner = CachedRunner(self.config, self.model, self.cache, self.device)
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
        if isinstance(self.runner, RaggedRunner):
            base["scheduler"]["forward_invocations"] = self.runner.forward_invocations
            base["scheduler"]["max_forward_request_count"] = (
                self.runner.max_forward_request_count
            )
            base["scheduler"]["last_forward_request_ids"] = list(
                self.runner.forward_traces[-1] if self.runner.forward_traces else ()
            )
            base["scheduler"]["recent_forward_plans"] = [
                [
                    {
                        "request_id": request_id,
                        "start_pos": start_pos,
                        "query_length": query_length,
                        "sample": sample,
                    }
                    for request_id, start_pos, query_length, sample in plan
                ]
                for plan in self.runner.forward_plans
            ]
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
