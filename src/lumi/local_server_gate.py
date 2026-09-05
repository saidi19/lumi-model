"""Durable black-box gate for a running loopback Lumi API and UI."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .local_api import is_loopback_host

LOCAL_SERVER_GATE_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _get(base_url: str, path: str) -> tuple[bytes, str]:
    with urlopen(f"{base_url}{path}", timeout=10) as response:  # noqa: S310
        return response.read(), response.headers.get_content_type()


def _post(base_url: str, payload: dict[str, Any]) -> tuple[bytes, float]:
    request = Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urlopen(request, timeout=60) as response:  # noqa: S310
        body = response.read()
    return body, time.perf_counter() - started


def _parse_sse(body: bytes) -> tuple[list[dict[str, Any]], bool]:
    events = []
    done = False
    for frame in body.decode().split("\n\n"):
        for line in frame.splitlines():
            if not line.startswith("data: "):
                continue
            value = line[6:]
            if value == "[DONE]":
                done = True
            else:
                events.append(json.loads(value))
    return events, done


def validate_local_server(
    *,
    base_url: str,
    output_path: str | Path,
    nonstream_prompt: str = "The capital of France is",
    stream_prompt: str = "In Python, a list comprehension can",
    max_tokens: int = 8,
) -> dict[str, Any]:
    """Validate real UI, model discovery, JSON completion, and SSE completion."""

    parsed_url = urlsplit(base_url)
    if parsed_url.scheme != "http" or not parsed_url.hostname:
        raise ValueError("base_url must be an http URL")
    if not is_loopback_host(parsed_url.hostname):
        raise ValueError("the local-server gate only permits loopback URLs")
    if parsed_url.path not in ("", "/") or parsed_url.query or parsed_url.fragment:
        raise ValueError("base_url must not include a path, query, or fragment")
    base_url = base_url.rstrip("/")
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite local-server report: {output}")
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")

    health_body, _ = _get(base_url, "/health")
    health = json.loads(health_body)
    if health.get("status") != "ok" or health.get("model_loaded") is not True:
        raise RuntimeError("Lumi health endpoint is not ready with a loaded model")

    models_body, _ = _get(base_url, "/v1/models")
    models = json.loads(models_body)
    if models.get("object") != "list" or len(models.get("data", [])) != 1:
        raise RuntimeError("Lumi models endpoint changed shape")
    model = models["data"][0]
    model_id = model.get("id")
    if not isinstance(model_id, str) or model.get("object") != "model":
        raise RuntimeError("Lumi model descriptor is invalid")

    ui_body, ui_type = _get(base_url, "/")
    if ui_type != "text/html" or b"not an instruction-tuned assistant" not in ui_body:
        raise RuntimeError("Lumi UI or its base-model warning is missing")
    ui_assets = {}
    for asset_path, allowed_types in (
        ("/app.js", {"application/javascript", "text/javascript"}),
        ("/renderer.js", {"application/javascript", "text/javascript"}),
        ("/styles.css", {"text/css"}),
        ("/vendor/katex/katex.min.css", {"text/css"}),
        ("/vendor/katex/katex.min.js", {"application/javascript", "text/javascript"}),
    ):
        asset_body, asset_type = _get(base_url, asset_path)
        if not asset_body or asset_type not in allowed_types:
            raise RuntimeError(f"Lumi UI asset is missing or invalid: {asset_path}")
        ui_assets[asset_path.removeprefix("/")] = {
            "bytes": len(asset_body),
            "content_type": asset_type,
            "sha256": hashlib.sha256(asset_body).hexdigest(),
        }

    common = {"model": model_id, "max_tokens": max_tokens, "temperature": 0}
    nonstream_body, nonstream_seconds = _post(
        base_url,
        {
            **common,
            "messages": [{"role": "user", "content": nonstream_prompt}],
        },
    )
    nonstream = json.loads(nonstream_body)
    choices = nonstream.get("choices", [])
    if (
        nonstream.get("object") != "chat.completion"
        or nonstream.get("model") != model_id
        or len(choices) != 1
        or not isinstance(choices[0].get("message", {}).get("content"), str)
        or choices[0].get("finish_reason") not in {"length", "stop"}
    ):
        raise RuntimeError("non-streaming Lumi completion changed shape")
    usage = nonstream.get("usage", {})
    if usage.get("total_tokens") != usage.get("prompt_tokens", 0) + usage.get(
        "completion_tokens", 0
    ):
        raise RuntimeError("non-streaming Lumi usage counts do not add up")

    stream_body, stream_seconds = _post(
        base_url,
        {
            **common,
            "messages": [{"role": "user", "content": stream_prompt}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    events, done = _parse_sse(stream_body)
    if not done or not events:
        raise RuntimeError("streaming Lumi completion did not terminate with [DONE]")
    choices_events = [event for event in events if event.get("choices")]
    if not choices_events or choices_events[0]["choices"][0].get("delta", {}).get(
        "role"
    ) != "assistant":
        raise RuntimeError("streaming Lumi completion omitted the assistant role chunk")
    finish_reasons = [
        event["choices"][0].get("finish_reason") for event in choices_events
    ]
    if not any(reason in {"length", "stop"} for reason in finish_reasons):
        raise RuntimeError("streaming Lumi completion omitted its finish reason")
    usage_events = [event for event in events if event.get("usage")]
    if len(usage_events) != 1:
        raise RuntimeError("streaming Lumi completion omitted its usage chunk")
    stream_text = "".join(
        event["choices"][0].get("delta", {}).get("content", "")
        for event in choices_events
    )

    report = {
        "base_url": base_url,
        "completed_at": _utc_now(),
        "health": health,
        "model": model,
        "nonstream": {
            "elapsed_seconds": nonstream_seconds,
            "finish_reason": choices[0]["finish_reason"],
            "text": choices[0]["message"]["content"],
            "usage": usage,
        },
        "schema_version": LOCAL_SERVER_GATE_SCHEMA_VERSION,
        "status": "pass",
        "stream": {
            "done": done,
            "elapsed_seconds": stream_seconds,
            "events": len(events),
            "text": stream_text,
            "usage": usage_events[0]["usage"],
        },
        "ui": {
            "assets": ui_assets,
            "bytes": len(ui_body),
            "content_type": ui_type,
            "sha256": hashlib.sha256(ui_body).hexdigest(),
        },
    }
    _write_json_atomic(output, report)
    return report
