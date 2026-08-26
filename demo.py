"""One-command walkthrough of the deployed Ragged L4 API.

Run: modal run demo.py
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor

import modal

DEFAULT_URL = "https://zeeshan8281--cloud-inference-lab-apiserver-serve.modal.run"
MODEL = "Qwen/Qwen2.5-3B"

app = modal.App("cloud-inference-demo")


def _load_api_key(environ: Mapping[str, str]) -> str:
    """Use the legacy key or a metrics-enabled tenant key from the Modal Secret."""
    legacy = environ.get("ENGINE_API_KEY", "").strip()
    if legacy:
        return legacy
    raw_policies = environ.get("ENGINE_TENANTS_JSON", "")
    try:
        policies = json.loads(raw_policies)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Modal Secret has no usable API key") from exc
    if not isinstance(policies, dict):
        raise RuntimeError("ENGINE_TENANTS_JSON must be an object")
    for require_metrics in (True, False):
        for policy in policies.values():
            if not isinstance(policy, dict) or bool(policy.get("metrics")) != require_metrics:
                continue
            key = policy.get("api_key")
            if isinstance(key, str) and key.strip():
                return key.strip()
    raise RuntimeError("Modal Secret has no usable API key")


def _request(
    base_url: str,
    path: str,
    *,
    api_key: str | None = None,
    payload: dict | None = None,
) -> urllib.request.Request:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        headers=headers,
        method="POST" if payload is not None else "GET",
    )


def _json_response(request: urllib.request.Request, *, timeout: int = 600) -> dict:
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _expect_error(
    request: urllib.request.Request,
    *,
    status: int,
    code: str,
) -> None:
    try:
        urllib.request.urlopen(request, timeout=30)
    except urllib.error.HTTPError as exc:
        body = json.load(exc)
        assert exc.code == status, body
        assert body["error"]["code"] == code, body
        print(f"{status} {code} (expected)")
        return
    raise AssertionError(f"request unexpectedly succeeded; wanted {status} {code}")


@app.function(
    image=modal.Image.debian_slim(python_version="3.12"),
    secrets=[modal.Secret.from_name("cloud-inference-api")],
    timeout=900,
)
def run_demo(base_url: str, prompt: str, max_output_tokens: int) -> None:
    if not 1 <= max_output_tokens <= 256:
        raise ValueError("max_output_tokens must be between 1 and 256")
    api_key = _load_api_key(os.environ)
    payload = {
        "model": MODEL,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
        "temperature": 0,
    }

    print("\n[1/7] LIVENESS + READINESS")
    print("Warming the L4 container (the first run can take about a minute)...", flush=True)
    ready = _json_response(_request(base_url, "/readyz"))
    live = _json_response(_request(base_url, "/livez"), timeout=30)
    assert live == {"status": "alive"}
    assert ready == {"status": "ready", "model": MODEL, "mode": "ragged"}
    print(json.dumps({"livez": live, "readyz": ready}, indent=2))

    print("\n[2/7] AUTH + CONTRACT GUARDS")
    _expect_error(
        _request(base_url, "/v1/responses", payload=payload),
        status=401,
        code="authentication_failed",
    )
    _expect_error(
        _request(
            base_url,
            "/v1/responses",
            api_key=api_key,
            payload=payload | {"temperature": 0.7},
        ),
        status=400,
        code="invalid_temperature",
    )

    print("\n[3/7] AUTHENTICATED MODEL DISCOVERY")
    models = _json_response(_request(base_url, "/v1/models", api_key=api_key), timeout=30)
    assert [model["id"] for model in models["data"]] == [MODEL]
    print(json.dumps(models, indent=2))

    print("\n[4/7] BLOCKING GENERATION")
    blocking = _json_response(
        _request(base_url, "/v1/responses", api_key=api_key, payload=payload)
    )
    assert blocking["status"] == "completed"
    assert blocking["model"] == MODEL
    print("assistant:", blocking["output"][0]["content"][0]["text"])
    print("usage:", json.dumps(blocking["usage"]))

    print("\n[5/7] LIVE SSE STREAM")
    event_types: list[str] = []
    sequence_numbers: list[int] = []
    completed_usage = None
    print("assistant: ", end="", flush=True)
    with urllib.request.urlopen(
        _request(
            base_url,
            "/v1/responses",
            api_key=api_key,
            payload=payload | {"stream": True},
        ),
        timeout=600,
    ) as response:
        for raw_line in response:
            line = raw_line.decode().rstrip("\r\n")
            if line.startswith("event: "):
                event_types.append(line[7:])
            elif line.startswith("data: ") and line != "data: [DONE]":
                event = json.loads(line[6:])
                sequence_numbers.append(event["sequence_number"])
                if event["type"] == "response.output_text.delta":
                    print(event["delta"], end="", flush=True)
                elif event["type"] == "response.completed":
                    completed_usage = event["response"]["usage"]
    print()
    assert event_types[0] == "response.created"
    assert event_types[-2:] == ["response.output_text.done", "response.completed"]
    assert sequence_numbers == list(range(len(sequence_numbers)))
    print("events:", " -> ".join(dict.fromkeys(event_types)))
    print("usage:", json.dumps(completed_usage))

    before = _json_response(_request(base_url, "/metrics", api_key=api_key), timeout=30)

    print("\n[6/7] CONCURRENT PACKED GENERATION")

    def generate(index: int) -> int:
        concurrent_payload = payload | {
            "input": f"Request {index}: explain one benefit of continuous batching.",
            "max_output_tokens": max(8, max_output_tokens),
        }
        with urllib.request.urlopen(
            _request(
                base_url,
                "/v1/responses",
                api_key=api_key,
                payload=concurrent_payload,
            ),
            timeout=600,
        ) as response:
            return json.load(response)["usage"]["output_tokens"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        parallel_output_tokens = list(pool.map(generate, range(1, 5)))
    assert all(tokens > 0 for tokens in parallel_output_tokens)
    print("four concurrent requests completed:", parallel_output_tokens)

    print("\n[7/7] ENGINE METRICS + BATCHING PROOF")
    metrics = _json_response(_request(base_url, "/metrics", api_key=api_key), timeout=30)
    summary = {
        "requests": {
            key: metrics["requests"][key]
            for key in ("completed_total", "failed_total", "preempted_total")
        },
        "tokens": {
            key: metrics["tokens"][key]
            for key in ("output_total", "recomputed_total", "output_per_second_60s")
        },
        "scheduler": {
            key: metrics["scheduler"][key]
            for key in (
                "iterations_total",
                "mean_batch_size_60s",
                "max_batch_size_60s",
                "forward_invocations",
                "max_forward_request_count",
            )
        },
        "kv_cache": {
            key: metrics["kv_cache"][key]
            for key in (
                "blocks_total",
                "blocks_used",
                "request_blocks_used",
                "prefix_blocks_used",
                "prefix_cache_hits",
                "prefix_cache_misses",
                "utilization",
            )
        },
    }
    completed_delta = (
        metrics["requests"]["completed_total"]
        - before["requests"]["completed_total"]
    )
    assert completed_delta >= 4
    assert summary["scheduler"]["max_forward_request_count"] >= 2
    if blocking["usage"]["input_tokens"] > 16:
        assert summary["kv_cache"]["prefix_cache_hits"] >= 1
    print(json.dumps(summary, indent=2))
    print(f"completed request delta during concurrency stage: {completed_delta}")
    print(
        "\nDEMO PASSED: health, auth, validation, blocking output, ordered SSE, "
        "prefix reuse, multi-request GPU forwards, and metrics are live.\n"
    )


@app.local_entrypoint()
def main(
    prompt: str = (
        "Explain how paged KV caching helps an inference engine, including block "
        "tables, fragmentation, memory reuse, and continuous batching."
    ),
    max_output_tokens: int = 24,
    url: str = DEFAULT_URL,
) -> None:
    run_demo.remote(url, prompt, max_output_tokens)
