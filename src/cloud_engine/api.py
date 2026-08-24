"""OpenAI Responses API subset with strict validation (PRD FR10).

Validation helpers are pure standard library so they unit-test on a laptop
without FastAPI installed; ``create_app`` performs the web imports lazily.
Unknown parameters and unsupported input forms are rejected, never ignored.
"""

import hmac
import json
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

ALLOWED_FIELDS = frozenset({"model", "input", "max_output_tokens", "temperature", "stream"})
MAX_BODY_BYTES = 64 * 1024


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

    raw_input = data["input"]
    if not isinstance(raw_input, str) or not raw_input.strip():
        raise ApiError(
            400,
            "unsupported_input_type",
            "input must be a non-empty string; arrays, files, and message objects are not supported in v1",
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

    return ValidatedRequest(
        model=data["model"],
        input=raw_input,
        max_output_tokens=raw_max,
        temperature=float(raw_temp),
        stream=raw_stream,
    )


def authorized(header_value: str | None, api_key: str) -> bool:
    """Constant-time bearer-token comparison (PRD §14)."""
    if not header_value:
        return False
    scheme, _, token = header_value.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return False
    return hmac.compare_digest(token.encode("utf-8"), api_key.encode("utf-8"))


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


def create_app(engine: Any, *, api_key: str, model_id: str, logger: Any = None) -> Any:
    """Build the FastAPI application. Requires the cloud dependency set."""
    import logging

    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    log = logger or logging.getLogger("cloud_engine.api")

    app = FastAPI(title="Cloud Inference Engine Lab", docs_url=None, redoc_url=None)

    @app.exception_handler(ApiError)
    async def _api_error_handler(_request: Request, exc: ApiError):  # type: ignore[no-untyped-def]
        return JSONResponse(status_code=exc.status, content=exc.body(), headers=exc.headers)

    @app.get("/healthz")
    async def healthz():  # type: ignore[no-untyped-def]
        if engine.ready:
            return {"status": "ready", "model": model_id, "mode": engine.config.mode}
        return JSONResponse(status_code=503, content={"status": "starting"})

    @app.get("/metrics")
    async def metrics(request: Request):  # type: ignore[no-untyped-def]
        if not authorized(request.headers.get("authorization"), api_key):
            return JSONResponse(status_code=401, content={"error": {"code": "authentication_failed"}})
        return engine.snapshot_metrics()

    @app.post("/v1/responses")
    async def responses(request: Request):  # type: ignore[no-untyped-def]
        if not authorized(request.headers.get("authorization"), api_key):
            return JSONResponse(status_code=401, content={"error": {"code": "authentication_failed"}})

        declared_length = request.headers.get("content-length")
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

        try:
            handle = await engine.submit(validated.input, gen_config)
        except Exception as exc:  # RejectedError and any admission failure
            return _rejection_response(exc)

        created_at = int(time.time())
        response_id = f"resp_{handle.request_id}"
        message_id = f"msg_{handle.request_id}"

        if not validated.stream:
            try:
                result = await handle.wait()
            except Exception:
                log.exception("generation failed request_id=%s", handle.request_id)
                return JSONResponse(
                    status_code=500,
                    content={"error": {"code": "generation_failed", "message": "internal error"}},
                )
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
