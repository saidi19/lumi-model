from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from lumi.local_api import (
    APIError,
    ChatCompletionRequest,
    ChatMessage,
    GenerationEvent,
    create_server,
    is_loopback_host,
    parse_chat_request,
    render_chat_prompt,
    serve_local,
)
from lumi.local_server_gate import validate_local_server


class FakeEngine:
    model_id = "lumi-test"
    created = 1234567890
    context_window = 2048

    @property
    def loaded(self) -> bool:
        return True

    def load(self) -> None:
        return None

    def model_metadata(self) -> dict:
        return {
            "context_window": self.context_window,
            "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
        }

    def stream(self, request: ChatCompletionRequest) -> Iterator[GenerationEvent]:
        assert request.messages[-1].content == "Hello"
        return iter(
            [
                GenerationEvent("Hel", prompt_tokens=4, completion_tokens=1),
                GenerationEvent(
                    "lo",
                    prompt_tokens=4,
                    completion_tokens=2,
                    finish_reason="length",
                    prompt_tokens_per_second=100.0,
                    generation_tokens_per_second=50.0,
                    peak_memory_gb=0.7,
                ),
            ]
        )


@pytest.fixture
def local_server():
    server = create_server(FakeEngine(), host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _post(base_url: str, payload: dict):
    request = Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urlopen(request, timeout=3)  # noqa: S310


def _post_raw(base_url: str, body: bytes, *, content_type: str):
    request = Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    return urlopen(request, timeout=3)  # noqa: S310


def test_parse_chat_request_supports_text_parts_and_current_token_name() -> None:
    request = parse_chat_request(
        {
            "model": "lumi-test",
            "messages": [
                {"role": "system", "content": "Continue text."},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Hel"},
                        {"type": "text", "text": "lo"},
                    ],
                },
            ],
            "max_completion_tokens": 32,
            "seed": 7,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0.3,
            "top_p": 0.9,
        },
        expected_model="lumi-test",
    )

    assert request.messages[-1] == ChatMessage(role="user", content="Hello")
    assert request.max_tokens == 32
    assert request.include_stream_usage is True
    assert request.seed == 7


def test_parse_chat_request_rejects_unsupported_or_ambiguous_inputs() -> None:
    base = {
        "model": "lumi-test",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    with pytest.raises(APIError, match="not available"):
        parse_chat_request({**base, "model": "wrong"}, expected_model="lumi-test")
    with pytest.raises(APIError, match="final message"):
        parse_chat_request(
            {**base, "messages": [{"role": "assistant", "content": "Hello"}]},
            expected_model="lumi-test",
        )
    with pytest.raises(APIError, match="not supported"):
        parse_chat_request({**base, "tools": []}, expected_model="lumi-test")
    with pytest.raises(APIError, match="disagree"):
        parse_chat_request(
            {**base, "max_tokens": 10, "max_completion_tokens": 11},
            expected_model="lumi-test",
        )


def test_render_chat_prompt_is_explicit_base_model_serialization() -> None:
    prompt = render_chat_prompt(
        (
            ChatMessage("system", "Be concise."),
            ChatMessage("user", "Hello"),
            ChatMessage("assistant", "Hi"),
            ChatMessage("user", "Continue"),
        )
    )

    assert prompt == (
        "System:\nBe concise.\n\nUser:\nHello\n\nAssistant:\nHi\n\n"
        "User:\nContinue\n\nAssistant:\n"
    )


def test_models_health_and_ui_routes(local_server: str) -> None:
    with urlopen(f"{local_server}/health", timeout=3) as response:  # noqa: S310
        health = json.load(response)
    assert health == {"model": "lumi-test", "model_loaded": True, "status": "ok"}

    with urlopen(f"{local_server}/v1/models", timeout=3) as response:  # noqa: S310
        models = json.load(response)
    assert models["object"] == "list"
    assert models["data"][0]["id"] == "lumi-test"
    assert models["data"][0]["quantization"]["bits"] == 4

    with urlopen(local_server, timeout=3) as response:  # noqa: S310
        html = response.read().decode()
    assert "Meet Lumi, locally" in html
    assert "not an instruction-tuned assistant" in html


def test_nonstreaming_chat_completion_is_openai_shaped(local_server: str) -> None:
    with _post(
        local_server,
        {
            "model": "lumi-test",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 2,
        },
    ) as response:
        result = json.load(response)

    assert result["object"] == "chat.completion"
    assert result["model"] == "lumi-test"
    assert result["choices"][0]["message"] == {"content": "Hello", "role": "assistant"}
    assert result["choices"][0]["finish_reason"] == "length"
    assert result["usage"] == {
        "completion_tokens": 2,
        "prompt_tokens": 4,
        "total_tokens": 6,
    }
    assert result["lumi_metrics"]["generation_tokens_per_second"] == 50.0


def test_streaming_chat_completion_ends_with_usage_and_done(local_server: str) -> None:
    with _post(
        local_server,
        {
            "model": "lumi-test",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 2,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as response:
        body = response.read().decode()

    assert '"role":"assistant"' in body
    assert '"content":"Hel"' in body
    assert '"content":"lo"' in body
    assert '"finish_reason":"length"' in body
    assert '"total_tokens":6' in body
    assert body.endswith("data: [DONE]\n\n")


def test_http_error_uses_openai_error_shape(local_server: str) -> None:
    with pytest.raises(HTTPError) as captured:
        _post(
            local_server,
            {
                "model": "wrong",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
    assert captured.value.code == 404
    payload = json.loads(captured.value.read())
    assert payload["error"]["code"] == "model_not_found"


def test_chat_endpoint_requires_json_content_type(local_server: str) -> None:
    with pytest.raises(HTTPError) as captured:
        _post_raw(local_server, b"model=lumi-test", content_type="text/plain")
    assert captured.value.code == 415
    payload = json.loads(captured.value.read())
    assert payload["error"]["code"] == "unsupported_media_type"


def test_network_bind_guard_recognizes_loopback_only() -> None:
    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("::1")
    assert is_loopback_host("localhost")
    assert not is_loopback_host("0.0.0.0")
    assert not is_loopback_host("192.168.1.50")

    with pytest.raises(ValueError, match="non-loopback"):
        serve_local(model_path="missing", host="0.0.0.0", port=8080)


def test_black_box_local_server_gate_seals_a_report(local_server: str, tmp_path) -> None:
    output = tmp_path / "local-server-report.json"

    report = validate_local_server(
        base_url=local_server,
        output_path=output,
        nonstream_prompt="Hello",
        stream_prompt="Hello",
        max_tokens=2,
    )

    assert report["status"] == "pass"
    assert report["model"]["id"] == "lumi-test"
    assert report["nonstream"]["text"] == "Hello"
    assert report["stream"]["done"] is True
    assert set(report["ui"]["assets"]) == {"app.js", "styles.css"}
    assert json.loads(output.read_text(encoding="utf-8")) == report
