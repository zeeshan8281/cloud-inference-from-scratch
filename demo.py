"""One-command walkthrough of the deployed Ragged L4 API.

Run: modal run demo.py
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import modal

DEFAULT_URL = "https://zeeshan8281--cloud-inference-lab-apiserver-serve.modal.run"
MODEL = "Qwen/Qwen2.5-3B"

app = modal.App("cloud-inference-demo")


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


@app.function(
    image=modal.Image.debian_slim(python_version="3.12"),
    secrets=[modal.Secret.from_name("cloud-inference-api")],
    timeout=900,
)
def run_demo(base_url: str, prompt: str, max_output_tokens: int) -> None:
    if not 1 <= max_output_tokens <= 256:
        raise ValueError("max_output_tokens must be between 1 and 256")
    api_key = os.environ["ENGINE_API_KEY"]
    payload = {
        "model": MODEL,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
        "temperature": 0,
    }

    print("\n[1/6] DEPLOYED ENGINE")
    with urllib.request.urlopen(_request(base_url, "/healthz"), timeout=600) as response:
        health = json.load(response)
    print(json.dumps(health, indent=2))
    assert health == {"status": "ready", "model": MODEL, "mode": "ragged"}

    print("\n[2/6] AUTHENTICATION GUARD")
    try:
        urllib.request.urlopen(
            _request(base_url, "/v1/responses", payload=payload), timeout=30
        )
        raise AssertionError("unauthenticated request was accepted")
    except urllib.error.HTTPError as exc:
        assert exc.code == 401
        print("unauthenticated request -> 401 (expected)")

    print("\n[3/6] BLOCKING GENERATION")
    with urllib.request.urlopen(
        _request(base_url, "/v1/responses", api_key=api_key, payload=payload),
        timeout=600,
    ) as response:
        blocking = json.load(response)
    print("assistant:", blocking["output"][0]["content"][0]["text"])
    print("usage:", json.dumps(blocking["usage"]))

    print("\n[4/6] LIVE SSE STREAM")
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

    print("\n[5/6] CONCURRENT PACKED GENERATION")

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
    print("four concurrent requests completed:", parallel_output_tokens)

    print("\n[6/6] ENGINE METRICS")
    with urllib.request.urlopen(
        _request(base_url, "/metrics", api_key=api_key), timeout=30
    ) as response:
        metrics = json.load(response)
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
            for key in ("blocks_total", "blocks_used", "utilization")
        },
    }
    assert summary["scheduler"]["max_forward_request_count"] >= 2
    print(json.dumps(summary, indent=2))
    print("\nDEMO PASSED: deployed Ragged generation, auth, SSE, and metrics are live.\n")


@app.local_entrypoint()
def main(
    prompt: str = "Explain how paged KV caching helps an inference engine.",
    max_output_tokens: int = 24,
    url: str = DEFAULT_URL,
) -> None:
    run_demo.remote(url, prompt, max_output_tokens)
