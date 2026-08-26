"""Zero-dependency client for the deployed inference engine."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen2.5-3B"


def iter_sse_events(lines: Iterable[bytes]) -> Iterator[dict[str, Any]]:
    """Yield JSON SSE data payloads, ignoring comments and the DONE sentinel."""
    for raw_line in lines:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data and data != "[DONE]":
            yield json.loads(data)


def request(
    base_url: str,
    path: str,
    *,
    api_key: str | None = None,
    payload: dict[str, Any] | None = None,
) -> urllib.request.Request:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if payload is not None:
        headers["Content-Type"] = "application/json"
    return urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        headers=headers,
        method="POST" if payload is not None else "GET",
    )


def read_json(req: urllib.request.Request, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(prog="cie", description=__doc__)
    cli.add_argument("--url", default=os.environ.get("ENGINE_URL"), help="deployed engine URL")
    cli.add_argument("--api-key", default=os.environ.get("ENGINE_API_KEY"), help="tenant bearer key")
    commands = cli.add_subparsers(dest="command", required=True)
    commands.add_parser("health", help="check liveness and readiness")
    commands.add_parser("models", help="list deployed models")
    generate = commands.add_parser("generate", help="generate text")
    generate.add_argument("prompt")
    generate.add_argument("--model", default=os.environ.get("ENGINE_MODEL", DEFAULT_MODEL))
    generate.add_argument("--max-output-tokens", type=int, default=64, choices=range(1, 257), metavar="1..256")
    generate.add_argument("--no-stream", action="store_true")
    commands.add_parser("metrics", help="print authenticated JSON metrics")
    return cli


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.url:
        parser().error("--url or ENGINE_URL is required")
    if args.command != "health" and not args.api_key:
        parser().error("--api-key or ENGINE_API_KEY is required")
    try:
        if args.command == "health":
            live = read_json(request(args.url, "/livez"), 30)
            ready = read_json(request(args.url, "/readyz"), 600)
            print(json.dumps({"liveness": live, "readiness": ready}, indent=2))
        elif args.command == "models":
            print(json.dumps(read_json(request(args.url, "/v1/models", api_key=args.api_key), 30), indent=2))
        elif args.command == "metrics":
            print(json.dumps(read_json(request(args.url, "/metrics", api_key=args.api_key), 30), indent=2))
        else:
            streaming = not args.no_stream
            req = request(
                args.url,
                "/v1/responses",
                api_key=args.api_key,
                payload={
                    "model": args.model,
                    "input": args.prompt,
                    "max_output_tokens": args.max_output_tokens,
                    "temperature": 0,
                    "stream": streaming,
                    **({"stream_options": {"include_usage": True}} if streaming else {}),
                },
            )
            if not streaming:
                result = read_json(req, 600)
                print(result["output"][0]["content"][0]["text"])
            else:
                with urllib.request.urlopen(req, timeout=600) as response:
                    for event in iter_sse_events(response):
                        if event.get("type") == "response.output_text.delta":
                            print(event["delta"], end="", flush=True)
                print()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code}: {detail}", file=sys.stderr)
        return 1
    except (OSError, json.JSONDecodeError) as exc:
        print(f"request failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
