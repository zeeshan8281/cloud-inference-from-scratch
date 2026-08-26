"""OpenAI Responses API subset with strict validation (PRD FR10).

Validation helpers are pure standard library so they unit-test on a laptop
without FastAPI installed; ``create_app`` performs the web imports lazily.
Unknown parameters and unsupported input forms are rejected, never ignored.
"""

import hmac
import json
import re
import secrets
import time
import uuid
from asyncio import CancelledError, Lock
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALLOWED_FIELDS = frozenset(
    {
        "model",
        "input",
        "instructions",
        "max_output_tokens",
        "temperature",
        "stream",
        "stream_options",
    }
)
MAX_BODY_BYTES = 64 * 1024
REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9._:-]{1,128}")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.headers = headers or {}

    def body(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": "invalid_request_error",
                "code": self.code,
            }
        }


@dataclass(frozen=True)
class ValidatedRequest:
    model: str
    input: str
    max_output_tokens: int
    temperature: float
    stream: bool


@dataclass(frozen=True)
class TenantPolicy:
    api_key: str
    max_concurrent: int
    tokens_per_minute: int
    metrics: bool = False


def parse_tenant_policies(raw: str) -> dict[str, TenantPolicy]:
    """Validate the secret-backed tenant policy map and fail closed."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("ENGINE_TENANTS_JSON must be valid JSON") from exc
    if not isinstance(data, dict) or not 1 <= len(data) <= 100:
        raise ValueError("tenant policy map must contain 1..100 tenants")
    policies: dict[str, TenantPolicy] = {}
    keys: set[str] = set()
    for tenant, value in data.items():
        if not isinstance(tenant, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", tenant):
            raise ValueError("tenant IDs must be 1..64 safe characters")
        if not isinstance(value, dict) or set(value) - {
            "api_key",
            "max_concurrent",
            "tokens_per_minute",
            "metrics",
        }:
            raise ValueError(f"invalid policy fields for tenant {tenant!r}")
        key = value.get("api_key")
        concurrent = value.get("max_concurrent")
        token_limit = value.get("tokens_per_minute")
        metrics = value.get("metrics", False)
        if not isinstance(key, str) or len(key) < 32 or key in keys:
            raise ValueError("tenant API keys must be unique and at least 32 characters")
        if isinstance(concurrent, bool) or not isinstance(concurrent, int) or not 1 <= concurrent <= 32:
            raise ValueError("max_concurrent must be an integer from 1 to 32")
        if isinstance(token_limit, bool) or not isinstance(token_limit, int) or token_limit < 256:
            raise ValueError("tokens_per_minute must be an integer of at least 256")
        if not isinstance(metrics, bool):
            raise ValueError("metrics must be a boolean")
        policies[tenant] = TenantPolicy(key, concurrent, token_limit, metrics)
        keys.add(key)
    return policies


def authenticate(header_value: str | None, policies: dict[str, TenantPolicy]) -> str | None:
    if not header_value:
        return None
    scheme, _, token = header_value.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    matched = None
    for tenant, policy in policies.items():
        if hmac.compare_digest(token.encode(), policy.api_key.encode()):
            matched = tenant
    return matched


class TenantLease:
    def __init__(
        self,
        gate: Any,
        tenant: str,
        stamp: float,
        tokens: int,
        lease_id: str | None = None,
    ) -> None:
        self._gate = gate
        self.tenant = tenant
        self._stamp = stamp
        self._tokens = tokens
        self._lease_id = lease_id
        self._released = False

    async def release(self, *, rollback_tokens: bool = False) -> None:
        if not self._released:
            self._released = True
            await self._gate.release(self, rollback_tokens=rollback_tokens)


class TenantGate:
    """Per-process tenant concurrency and conservative rolling token budgets."""

    # ponytail: process-local is deployment-global while Modal max_containers=1;
    # move counters to a transactional shared store before horizontal scaling.

    def __init__(self, policies: dict[str, TenantPolicy], clock: Callable[[], float] = time.monotonic):
        self.policies = policies
        self._clock = clock
        self._lock = Lock()
        self._active = {tenant: 0 for tenant in policies}
        self._tokens = {tenant: deque() for tenant in policies}

    async def acquire(self, tenant: str, tokens: int) -> TenantLease:
        async with self._lock:
            now = self._clock()
            history = self._tokens[tenant]
            while history and history[0][0] <= now - 60:
                history.popleft()
            policy = self.policies[tenant]
            if self._active[tenant] >= policy.max_concurrent:
                raise ApiError(
                    429,
                    "tenant_concurrency_limit",
                    "tenant concurrency limit reached",
                    {"Retry-After": "1"},
                )
            if sum(value for _, value in history) + tokens > policy.tokens_per_minute:
                retry_after = max(1, int(61 - (now - history[0][0]))) if history else 60
                raise ApiError(
                    429,
                    "tenant_token_rate_limit",
                    "tenant rolling token budget exceeded",
                    {"Retry-After": str(retry_after)},
                )
            self._active[tenant] += 1
            history.append((now, tokens))
            return TenantLease(self, tenant, now, tokens)

    async def release(self, lease: TenantLease, *, rollback_tokens: bool) -> None:
        async with self._lock:
            self._active[lease.tenant] -= 1
            if rollback_tokens:
                try:
                    self._tokens[lease.tenant].remove((lease._stamp, lease._tokens))
                except ValueError:
                    pass

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            now = self._clock()
            return {
                tenant: {
                    "active": self._active[tenant],
                    "reserved_tokens_60s": sum(
                        tokens for stamp, tokens in self._tokens[tenant] if stamp > now - 60
                    ),
                    "max_concurrent": policy.max_concurrent,
                    "tokens_per_minute": policy.tokens_per_minute,
                }
                for tenant, policy in self.policies.items()
            }


class RedisTenantGate:
    """Atomic cross-container admission and rolling quotas backed by Redis."""

    _ACQUIRE = """
local active = KEYS[1]
local tokens = KEYS[2]
local counts = KEYS[3]
local now = tonumber(ARGV[1])
local active_expiry = tonumber(ARGV[2])
local token_cutoff = tonumber(ARGV[3])
local max_concurrent = tonumber(ARGV[4])
local token_limit = tonumber(ARGV[5])
local requested = tonumber(ARGV[6])
local lease = ARGV[7]
local expired = redis.call('ZRANGEBYSCORE', tokens, '-inf', token_cutoff)
for _, member in ipairs(expired) do redis.call('HDEL', counts, member) end
redis.call('ZREMRANGEBYSCORE', tokens, '-inf', token_cutoff)
redis.call('ZREMRANGEBYSCORE', active, '-inf', now)
local active_count = redis.call('ZCARD', active)
if active_count >= max_concurrent then return {0, 1, active_count, 0} end
local reserved = 0
for _, value in ipairs(redis.call('HVALS', counts)) do reserved = reserved + tonumber(value) end
if reserved + requested > token_limit then return {0, 2, active_count, reserved} end
redis.call('ZADD', active, active_expiry, lease)
redis.call('ZADD', tokens, now, lease)
redis.call('HSET', counts, lease, requested)
redis.call('PEXPIRE', active, active_expiry - now + 60000)
redis.call('PEXPIRE', tokens, 120000)
redis.call('PEXPIRE', counts, 120000)
return {1, 0, active_count + 1, reserved + requested}
"""
    _RELEASE = """
redis.call('ZREM', KEYS[1], ARGV[1])
if ARGV[2] == '1' then
  redis.call('ZREM', KEYS[2], ARGV[1])
  redis.call('HDEL', KEYS[3], ARGV[1])
end
return 1
"""
    _SNAPSHOT = """
local active = KEYS[1]
local tokens = KEYS[2]
local counts = KEYS[3]
local now = tonumber(ARGV[1])
local cutoff = tonumber(ARGV[2])
local expired = redis.call('ZRANGEBYSCORE', tokens, '-inf', cutoff)
for _, member in ipairs(expired) do redis.call('HDEL', counts, member) end
redis.call('ZREMRANGEBYSCORE', tokens, '-inf', cutoff)
redis.call('ZREMRANGEBYSCORE', active, '-inf', now)
local reserved = 0
for _, value in ipairs(redis.call('HVALS', counts)) do reserved = reserved + tonumber(value) end
return {redis.call('ZCARD', active), reserved}
"""

    def __init__(
        self,
        client: Any,
        policies: dict[str, TenantPolicy],
        *,
        lease_seconds: int = 1800,
        clock: Callable[[], float] = time.time,
        namespace: str = "cie:admission",
    ) -> None:
        self.client = client
        self.policies = policies
        self.lease_seconds = lease_seconds
        self._clock = clock
        self.namespace = namespace

    @classmethod
    def from_url(
        cls,
        url: str,
        policies: dict[str, TenantPolicy],
        *,
        lease_seconds: int = 1800,
    ) -> "RedisTenantGate":
        import redis.asyncio as redis

        return cls(
            redis.from_url(url, decode_responses=True, socket_connect_timeout=5),
            policies,
            lease_seconds=lease_seconds,
        )

    def _keys(self, tenant: str) -> tuple[str, str, str]:
        prefix = f"{self.namespace}:{{{tenant}}}"
        return f"{prefix}:active", f"{prefix}:tokens", f"{prefix}:counts"

    async def acquire(self, tenant: str, tokens: int) -> TenantLease:
        now_ms = int(self._clock() * 1000)
        lease_id = secrets.token_hex(16)
        policy = self.policies[tenant]
        try:
            result = await self.client.eval(
                self._ACQUIRE,
                3,
                *self._keys(tenant),
                now_ms,
                now_ms + self.lease_seconds * 1000,
                now_ms - 60_000,
                policy.max_concurrent,
                policy.tokens_per_minute,
                tokens,
                lease_id,
            )
        except Exception as exc:
            raise ApiError(
                503,
                "admission_backend_unavailable",
                "distributed admission backend unavailable",
                {"Retry-After": "1"},
            ) from exc
        if int(result[0]) != 1:
            if int(result[1]) == 1:
                raise ApiError(429, "tenant_concurrency_limit", "tenant concurrency limit reached", {"Retry-After": "1"})
            raise ApiError(429, "tenant_token_rate_limit", "tenant rolling token budget exceeded", {"Retry-After": "60"})
        return TenantLease(self, tenant, now_ms / 1000, tokens, lease_id)

    async def release(self, lease: TenantLease, *, rollback_tokens: bool) -> None:
        try:
            await self.client.eval(
                self._RELEASE,
                3,
                *self._keys(lease.tenant),
                lease._lease_id,
                int(rollback_tokens),
            )
        except Exception as exc:
            raise ApiError(503, "admission_backend_unavailable", "distributed admission backend unavailable") from exc

    async def snapshot(self) -> dict[str, Any]:
        now_ms = int(self._clock() * 1000)
        snapshots: dict[str, Any] = {}
        try:
            for tenant, policy in self.policies.items():
                active, reserved = await self.client.eval(
                    self._SNAPSHOT,
                    3,
                    *self._keys(tenant),
                    now_ms,
                    now_ms - 60_000,
                )
                snapshots[tenant] = {
                    "active": int(active),
                    "reserved_tokens_60s": int(reserved),
                    "max_concurrent": policy.max_concurrent,
                    "tokens_per_minute": policy.tokens_per_minute,
                }
        except Exception as exc:
            raise ApiError(503, "admission_backend_unavailable", "distributed admission backend unavailable") from exc
        return snapshots

    async def close(self) -> None:
        await self.client.aclose()


def _input_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list) or not value:
        raise ApiError(400, "unsupported_input_type", "input must contain text")
    parts: list[str] = []
    for message in value:
        if not isinstance(message, dict) or message.get("role") != "user":
            raise ApiError(400, "unsupported_input_type", "only user text messages are supported")
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
            continue
        if not isinstance(content, list) or not content:
            raise ApiError(400, "unsupported_input_type", "message content must contain text")
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "input_text":
                raise ApiError(400, "unsupported_input_type", "only input_text blocks are supported")
            text = block.get("text")
            if not isinstance(text, str):
                raise ApiError(400, "unsupported_input_type", "input_text.text must be a string")
            parts.append(text)
    return "\n\n".join(parts)


def validate_payload(data: Any, *, model_id: str, default_max_output_tokens: int) -> ValidatedRequest:
    if not isinstance(data, dict):
        raise ApiError(400, "invalid_request", "request body must be a JSON object")
    unknown = sorted(set(data) - ALLOWED_FIELDS)
    if unknown:
        raise ApiError(
            400,
            "unsupported_parameter",
            f"unsupported parameter(s): {', '.join(unknown)}",
        )
    if "model" not in data:
        raise ApiError(400, "invalid_request", "'model' is required")
    if data["model"] != model_id:
        raise ApiError(400, "invalid_model", f"model must be {model_id!r}")

    if "input" not in data:
        raise ApiError(400, "invalid_request", "'input' is required")
    raw_input = _input_text(data["input"])
    instructions = data.get("instructions", "")
    if not isinstance(instructions, str):
        raise ApiError(400, "unsupported_input_type", "instructions must be a string")
    prompt = "\n\n".join(part for part in (instructions.strip(), raw_input.strip()) if part)
    if not prompt:
        raise ApiError(
            400,
            "unsupported_input_type",
            "input must contain non-empty text",
        )

    raw_max = data.get("max_output_tokens", default_max_output_tokens)
    if isinstance(raw_max, bool) or not isinstance(raw_max, int):
        raise ApiError(400, "invalid_max_output_tokens", "max_output_tokens must be an integer")
    if not 1 <= raw_max <= 256:
        raise ApiError(400, "invalid_max_output_tokens", "max_output_tokens must be between 1 and 256")

    raw_temp = data.get("temperature", 0)
    if isinstance(raw_temp, bool) or not isinstance(raw_temp, (int, float)):
        raise ApiError(400, "invalid_temperature", "temperature must be a number")
    if raw_temp != 0:
        raise ApiError(400, "invalid_temperature", "only temperature=0 (greedy) is supported in v1")

    raw_stream = data.get("stream", False)
    if not isinstance(raw_stream, bool):
        raise ApiError(400, "invalid_stream", "stream must be a boolean")
    stream_options = data.get("stream_options")
    if stream_options is not None and stream_options != {"include_usage": True}:
        raise ApiError(
            400,
            "unsupported_parameter",
            "stream_options supports only include_usage=true",
        )

    return ValidatedRequest(
        model=data["model"],
        input=prompt,
        max_output_tokens=raw_max,
        temperature=float(raw_temp),
        stream=raw_stream,
    )


def authorized(header_value: str | None, api_key: str) -> bool:
    """Constant-time bearer-token comparison (PRD §14)."""
    policy = TenantPolicy(api_key, 1, 256, True)
    return authenticate(header_value, {"default": policy}) == "default"


def build_response_object(
    *,
    response_id: str,
    message_id: str,
    model_id: str,
    created_at: int | None = None,
    text: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
    status: str = "in_progress",
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(created_at if created_at is not None else time.time()),
        "status": status,
        "model": model_id,
        "output": [
            {
                "type": "message",
                "id": message_id,
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


def sse_frame(event_type: str, payload: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"


def prometheus_text(snapshot: dict[str, Any]) -> str:
    """Render a bounded metrics snapshot in Prometheus text format."""
    samples: list[tuple[str, str, float]] = []

    def collect(value: Any, path: tuple[str, ...], labels: str = "") -> None:
        if isinstance(value, dict):
            for key in sorted(value):
                if path == () and key == "tenants":
                    for tenant, tenant_values in sorted(value[key].items()):
                        safe_tenant = json.dumps(str(tenant))
                        collect(tenant_values, ("tenant",), f'{{tenant={safe_tenant}}}')
                else:
                    collect(value[key], (*path, str(key)), labels)
        elif isinstance(value, bool):
            samples.append(("_".join(path), labels, float(value)))
        elif isinstance(value, (int, float)):
            samples.append(("_".join(path), labels, float(value)))

    collect(snapshot, ())
    lines: list[str] = []
    declared: set[str] = set()
    for raw_name, labels, value in samples:
        name = "cie_" + re.sub(r"[^a-zA-Z0-9_:]", "_", raw_name)
        rendered = str(int(value)) if value.is_integer() else repr(value)
        if name not in declared:
            lines.append(f"# TYPE {name} gauge")
            declared.add(name)
        lines.append(f"{name}{labels} {rendered}")
    return "\n".join(lines) + "\n"


async def response_event_sequence(
    base_response: dict[str, Any],
    deltas: AsyncIterator[str] | Callable[[], AsyncIterator[str]],
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Ordered Responses-style events with strictly increasing sequence numbers.

    Yields (event_type, payload) tuples; the HTTP layer renders SSE frames.
    Kept pure so event order/numbering is unit-testable without a server.
    """
    sequence = 0
    message_id = base_response["output"][0]["id"]
    yield "response.created", {
        "type": "response.created",
        "sequence_number": sequence,
        "response": {**base_response, "status": "in_progress"},
    }

    iterator = deltas if isinstance(deltas, AsyncIterator) else deltas()
    full_text_parts: list[str] = []
    async for piece in iterator:
        sequence += 1
        full_text_parts.append(piece)
        yield "response.output_text.delta", {
            "type": "response.output_text.delta",
            "sequence_number": sequence,
            "item_id": message_id,
            "output_index": 0,
            "content_index": 0,
            "delta": piece,
        }

    sequence += 1
    final_text = "".join(full_text_parts)
    yield "response.output_text.done", {
        "type": "response.output_text.done",
        "sequence_number": sequence,
        "item_id": message_id,
        "output_index": 0,
        "content_index": 0,
        "text": final_text,
    }

    completed_response = build_response_object(
        response_id=base_response["id"],
        message_id=message_id,
        model_id=base_response["model"],
        created_at=base_response["created_at"],
        text=final_text,
        output_tokens=len(final_text),  # replaced by real usage by the caller below
        status="completed",
    )
    # The caller passes usage through base_response when available.
    if "usage" in base_response:
        completed_response["usage"] = base_response["usage"]
    sequence += 1
    yield "response.completed", {
        "type": "response.completed",
        "sequence_number": sequence,
        "response": completed_response,
    }


def create_app(
    engine: Any,
    *,
    api_key: str | None,
    model_id: str,
    logger: Any = None,
    tenant_policies: dict[str, TenantPolicy] | None = None,
    ui_dir: str | Path | None = None,
    admission_redis_url: str | None = None,
) -> Any:
    """Build the FastAPI application. Requires the cloud dependency set."""
    import logging

    from fastapi import FastAPI, Request
    from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse

    log = logger or logging.getLogger("cloud_engine.api")
    if logger is None:
        log.setLevel(logging.INFO)
    if tenant_policies is None:
        if not api_key:
            raise ValueError("api_key or tenant_policies is required")
        tenant_policies = {
            "default": TenantPolicy(api_key, 32, 1_000_000_000, True)
        }
    gate = (
        RedisTenantGate.from_url(admission_redis_url, tenant_policies)
        if admission_redis_url
        else TenantGate(tenant_policies)
    )

    app = FastAPI(title="Cloud Inference Engine Lab", docs_url=None, redoc_url=None)

    if isinstance(gate, RedisTenantGate):
        @app.on_event("shutdown")
        async def close_admission_backend() -> None:
            await gate.close()

    @app.middleware("http")
    async def operational_headers(request: Request, call_next):  # type: ignore[no-untyped-def]
        supplied = request.headers.get("x-request-id", "")
        request.state.request_id = (
            supplied if REQUEST_ID_PATTERN.fullmatch(supplied) else f"req_{uuid.uuid4().hex}"
        )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response

    if ui_dir is not None:
        assets = Path(ui_dir)

        @app.get("/", include_in_schema=False)
        async def operator_console():  # type: ignore[no-untyped-def]
            return FileResponse(assets / "index.html", media_type="text/html")

        @app.get("/ui/styles.css", include_in_schema=False)
        async def operator_styles():  # type: ignore[no-untyped-def]
            return FileResponse(assets / "styles.css", media_type="text/css")

        @app.get("/ui/app.js", include_in_schema=False)
        async def operator_script():  # type: ignore[no-untyped-def]
            return FileResponse(assets / "app.js", media_type="text/javascript")

        @app.get("/tokens.css", include_in_schema=False)
        async def operator_tokens():  # type: ignore[no-untyped-def]
            return FileResponse(assets.parent / "tokens.css", media_type="text/css")

    def audit_event(
        tenant: str,
        request_id: str,
        status: str,
        input_tokens: int,
        output_tokens: int,
        e2e_ms: float | None = None,
    ) -> None:
        event: dict[str, Any] = {
            "tenant": tenant,
            "request_id": request_id,
            "status": status,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        if e2e_ms is not None:
            event["e2e_ms"] = round(e2e_ms, 3)
        log.info(
            "inference_audit %s",
            json.dumps(event, separators=(",", ":"), sort_keys=True),
        )

    @app.exception_handler(ApiError)
    async def _api_error_handler(_request: Request, exc: ApiError):  # type: ignore[no-untyped-def]
        return JSONResponse(status_code=exc.status, content=exc.body(), headers=exc.headers)

    @app.get("/livez")
    async def livez():  # type: ignore[no-untyped-def]
        return {"status": "alive"}

    @app.get("/healthz")
    @app.get("/readyz")
    async def readyz():  # type: ignore[no-untyped-def]
        if engine.ready:
            return {"status": "ready", "model": model_id, "mode": engine.config.mode}
        return JSONResponse(status_code=503, content={"status": "starting"})

    @app.get("/v1/models")
    async def models(request: Request):  # type: ignore[no-untyped-def]
        tenant = authenticate(request.headers.get("authorization"), tenant_policies)
        if tenant is None:
            return JSONResponse(status_code=401, content={"error": {"code": "authentication_failed"}})
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "cloud-inference-engine-lab",
                }
            ],
        }

    @app.get("/metrics")
    async def metrics(request: Request):  # type: ignore[no-untyped-def]
        tenant = authenticate(request.headers.get("authorization"), tenant_policies)
        if tenant is None:
            return JSONResponse(status_code=401, content={"error": {"code": "authentication_failed"}})
        if not tenant_policies[tenant].metrics:
            return JSONResponse(status_code=403, content={"error": {"code": "permission_denied"}})
        snapshot = engine.snapshot_metrics()
        snapshot["tenants"] = await gate.snapshot()
        return snapshot

    @app.get("/metrics/prometheus")
    async def prometheus_metrics(request: Request):  # type: ignore[no-untyped-def]
        tenant = authenticate(request.headers.get("authorization"), tenant_policies)
        if tenant is None:
            return JSONResponse(status_code=401, content={"error": {"code": "authentication_failed"}})
        if not tenant_policies[tenant].metrics:
            return JSONResponse(status_code=403, content={"error": {"code": "permission_denied"}})
        snapshot = engine.snapshot_metrics()
        snapshot["tenants"] = await gate.snapshot()
        return PlainTextResponse(
            prometheus_text(snapshot),
            media_type="text/plain; version=0.0.4",
        )

    @app.post("/v1/responses")
    async def responses(request: Request):  # type: ignore[no-untyped-def]
        tenant = authenticate(request.headers.get("authorization"), tenant_policies)
        if tenant is None:
            return JSONResponse(status_code=401, content={"error": {"code": "authentication_failed"}})

        declared_length = request.headers.get("content-length")
        if declared_length and not declared_length.isdecimal():
            return JSONResponse(
                status_code=400,
                content={"error": {"code": "invalid_content_length"}},
            )
        if declared_length and int(declared_length) > MAX_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={"error": {"code": "request_too_large", "message": "body exceeds 64 KiB"}},
            )
        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={"error": {"code": "request_too_large", "message": "body exceeds 64 KiB"}},
            )
        try:
            data = json.loads(body or b"{}")
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=400,
                content={"error": {"code": "invalid_json", "message": "body is not valid JSON"}},
            )
        try:
            validated = validate_payload(
                data, model_id=model_id, default_max_output_tokens=engine.config.max_output_tokens
            )
        except ApiError as exc:
            return JSONResponse(status_code=exc.status, content=exc.body())

        prompt_ids = engine.count_token_ids(validated.input)
        if len(prompt_ids) + validated.max_output_tokens > engine.config.max_model_len:
            error = ApiError(
                400,
                "context_length_exceeded",
                f"prompt ({len(prompt_ids)} tokens) plus max_output_tokens exceeds "
                f"{engine.config.max_model_len}",
            )
            return JSONResponse(status_code=error.status, content=error.body())

        from .scheduler import GenerationConfig as EngineGenConfig

        gen_config = EngineGenConfig(
            max_output_tokens=validated.max_output_tokens,
            temperature=0.0,
            eos_token_id=engine.config.eos_token_id,
        )

        lease = await gate.acquire(
            tenant, len(prompt_ids) + validated.max_output_tokens
        )

        try:
            handle = await engine.submit(validated.input, gen_config)
        except CancelledError:
            await lease.release(rollback_tokens=True)
            raise
        except Exception as exc:  # RejectedError and any admission failure
            await lease.release(rollback_tokens=True)
            return _rejection_response(exc)

        created_at = int(time.time())
        response_id = f"resp_{handle.request_id}"
        message_id = f"msg_{handle.request_id}"

        if not validated.stream:
            try:
                result = await handle.wait()
            except Exception:
                log.exception("generation failed request_id=%s", handle.request_id)
                audit_event(
                    tenant,
                    handle.request_id,
                    "failed",
                    len(prompt_ids),
                    handle.request.generated_count,
                )
                return JSONResponse(
                    status_code=500,
                    content={"error": {"code": "generation_failed", "message": "internal error"}},
                )
            finally:
                await lease.release()
            payload = build_response_object(
                response_id=response_id,
                message_id=message_id,
                model_id=model_id,
                created_at=created_at,
                text=result.text,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                status="completed",
            )
            audit_event(
                tenant,
                handle.request_id,
                "completed",
                result.input_tokens,
                result.output_tokens,
                result.e2e_ms,
            )
            return JSONResponse(status_code=200, content=payload)

        async def token_deltas() -> AsyncIterator[str]:
            prefix = ""
            ids_so_far: list[int] = []
            async for stream_event in handle.stream():
                if stream_event.token_id is None:
                    continue
                ids_so_far.append(stream_event.token_id)
                text = engine.detokenize(ids_so_far)
                delta = text[len(prefix):]
                prefix = text
                if delta:
                    yield delta

        async def event_stream() -> AsyncIterator[str]:
            base_response = build_response_object(
                response_id=response_id,
                message_id=message_id,
                model_id=model_id,
                created_at=created_at,
                status="in_progress",
            )
            try:
                async for event_type, payload in response_event_sequence(base_response, token_deltas):
                    if event_type == "response.completed":
                        result = await handle.wait()
                        payload["response"]["output"][0]["content"][0]["text"] = result.text
                        payload["response"]["usage"] = {
                            "input_tokens": result.input_tokens,
                            "output_tokens": result.output_tokens,
                            "total_tokens": result.input_tokens + result.output_tokens,
                        }
                    yield sse_frame(event_type, payload)
                yield "data: [DONE]\n\n"
            finally:
                if not handle.request.is_terminal:
                    engine.scheduler.cancel(handle.request)
                await lease.release()
                result = (
                    await handle.request.terminal_future
                    if handle.request.terminal_future.done()
                    else None
                )
                audit_event(
                    tenant,
                    handle.request_id,
                    handle.request.state.value,
                    len(prompt_ids),
                    result.generated_count if result else 0,
                )

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    def _rejection_response(exc: Exception):  # type: ignore[no-untyped-def]
        from .scheduler import RejectedError, RejectionReason

        if isinstance(exc, RejectedError):
            mapping = {
                RejectionReason.QUEUE_FULL: (429, "queue_full", {}),
                RejectionReason.KV_CAPACITY: (
                    503,
                    "capacity_exhausted",
                    {"Retry-After": "1"},
                ),
                RejectionReason.CONTEXT_OVERFLOW: (400, "context_length_exceeded", {}),
            }
            status, code, headers = mapping[exc.reason]
            return JSONResponse(
                status_code=status,
                headers=headers,
                content={"error": {"code": code, "message": str(exc)}},
            )
        log.exception("submission failed")
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "generation_failed", "message": "internal error"}},
        )

    return app
