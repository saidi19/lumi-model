"""Loopback-first OpenAI-compatible HTTP serving for local Lumi MLX artifacts."""

from __future__ import annotations

import ipaddress
import json
import mimetypes
import socket
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from .tokenizer_gate import encode_untrusted_text

DEFAULT_MODEL_ID = "lumi-150m-local"
DEFAULT_MAX_TOKENS = 128
MAX_REQUEST_BYTES = 1024 * 1024
SUPPORTED_ROLES = {"assistant", "developer", "system", "user"}
UI_DIRECTORY = Path(__file__).with_name("local_ui")


class APIError(ValueError):
    """An OpenAI-shaped client error with an HTTP status."""

    def __init__(
        self,
        message: str,
        *,
        status: int = HTTPStatus.BAD_REQUEST,
        code: str | None = "invalid_request_error",
        param: str | None = None,
        error_type: str = "invalid_request_error",
    ) -> None:
        super().__init__(message)
        self.status = int(status)
        self.code = code
        self.param = param
        self.error_type = error_type

    def payload(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": str(self),
                "param": self.param,
                "type": self.error_type,
            }
        }


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class ChatCompletionRequest:
    model: str
    messages: tuple[ChatMessage, ...]
    max_tokens: int
    temperature: float
    top_p: float
    stream: bool
    include_stream_usage: bool
    seed: int | None


@dataclass(frozen=True)
class GenerationEvent:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None = None
    prompt_tokens_per_second: float | None = None
    generation_tokens_per_second: float | None = None
    peak_memory_gb: float | None = None


class ChatEngine(Protocol):
    model_id: str
    created: int
    context_window: int

    @property
    def loaded(self) -> bool: ...

    def load(self) -> None: ...

    def stream(self, request: ChatCompletionRequest) -> Iterator[GenerationEvent]: ...

    def model_metadata(self) -> dict[str, Any]: ...


def _number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise APIError(f"{field} must be a number", param=field)
    return float(value)


def _message_content(value: Any, *, index: int) -> str:
    param = f"messages[{index}].content"
    if isinstance(value, str):
        content = value
    elif isinstance(value, list):
        pieces = []
        for part_index, part in enumerate(value):
            if not isinstance(part, dict):
                raise APIError(
                    "message content parts must be objects",
                    param=f"{param}[{part_index}]",
                )
            part_type = part.get("type")
            if part_type not in {"text", "input_text"} or not isinstance(
                part.get("text"), str
            ):
                raise APIError(
                    "Lumi local v0.1 supports text message content only",
                    param=f"{param}[{part_index}]",
                    code="unsupported_content_type",
                )
            pieces.append(part["text"])
        content = "".join(pieces)
    else:
        raise APIError("message content must be text", param=param)
    if not content.strip():
        raise APIError("message content cannot be empty", param=param)
    return content


def parse_chat_request(payload: Any, *, expected_model: str) -> ChatCompletionRequest:
    """Validate the supported OpenAI chat-completions request subset."""

    if not isinstance(payload, dict):
        raise APIError("request body must be a JSON object")
    model = payload.get("model")
    if not isinstance(model, str) or not model:
        raise APIError("model is required", param="model")
    if model != expected_model:
        raise APIError(
            f"model {model!r} is not available; use {expected_model!r}",
            status=HTTPStatus.NOT_FOUND,
            code="model_not_found",
            param="model",
        )

    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise APIError("messages must be a non-empty array", param="messages")
    messages = []
    for index, raw_message in enumerate(raw_messages):
        if not isinstance(raw_message, dict):
            raise APIError("each message must be an object", param=f"messages[{index}]")
        role = raw_message.get("role")
        if role not in SUPPORTED_ROLES:
            raise APIError(
                "supported roles are system, developer, user, and assistant",
                param=f"messages[{index}].role",
                code="unsupported_role",
            )
        messages.append(
            ChatMessage(
                role=role,
                content=_message_content(raw_message.get("content"), index=index),
            )
        )
    if messages[-1].role != "user":
        raise APIError("the final message must have role 'user'", param="messages")

    if payload.get("n", 1) != 1:
        raise APIError("Lumi local v0.1 supports n=1 only", param="n")
    for unsupported in ("audio", "functions", "logprobs", "stop", "tools"):
        if payload.get(unsupported) not in (None, False):
            raise APIError(
                f"{unsupported} is not supported by Lumi local v0.1",
                param=unsupported,
                code="unsupported_parameter",
            )
    response_format = payload.get("response_format")
    if response_format not in (None, {"type": "text"}):
        raise APIError(
            "only response_format type 'text' is supported",
            param="response_format",
            code="unsupported_parameter",
        )

    max_tokens = payload.get("max_completion_tokens")
    legacy_max_tokens = payload.get("max_tokens")
    if max_tokens is not None and legacy_max_tokens is not None and max_tokens != legacy_max_tokens:
        raise APIError(
            "max_tokens and max_completion_tokens disagree",
            param="max_completion_tokens",
        )
    max_tokens = max_tokens if max_tokens is not None else legacy_max_tokens
    max_tokens = DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise APIError("max_tokens must be a positive integer", param="max_tokens")

    temperature = _number(payload.get("temperature", 0.0), field="temperature")
    if not 0.0 <= temperature <= 2.0:
        raise APIError("temperature must be between 0 and 2", param="temperature")
    top_p = _number(payload.get("top_p", 1.0), field="top_p")
    if not 0.0 < top_p <= 1.0:
        raise APIError("top_p must be in (0, 1]", param="top_p")
    stream = payload.get("stream", False)
    if not isinstance(stream, bool):
        raise APIError("stream must be boolean", param="stream")
    stream_options = payload.get("stream_options") or {}
    if not isinstance(stream_options, dict):
        raise APIError("stream_options must be an object", param="stream_options")
    include_usage = stream_options.get("include_usage", False)
    if not isinstance(include_usage, bool):
        raise APIError(
            "stream_options.include_usage must be boolean",
            param="stream_options.include_usage",
        )
    seed = payload.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise APIError("seed must be an integer", param="seed")

    return ChatCompletionRequest(
        model=model,
        messages=tuple(messages),
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=stream,
        include_stream_usage=include_usage,
        seed=seed,
    )


def render_chat_prompt(messages: tuple[ChatMessage, ...]) -> str:
    """Serialize chat turns without claiming the base model has a chat template."""

    labels = {
        "assistant": "Assistant",
        "developer": "System",
        "system": "System",
        "user": "User",
    }
    sections = [f"{labels[message.role]}:\n{message.content}" for message in messages]
    sections.append("Assistant:\n")
    return "\n\n".join(sections)


class MLXChatEngine:
    """One-model, serialized-generation engine for an MLX Lumi artifact."""

    def __init__(self, model_path: str | Path, *, model_id: str = DEFAULT_MODEL_ID) -> None:
        self.model_path = Path(model_path).resolve()
        self.model_id = model_id
        self.created = int(time.time())
        self.context_window = 0
        self._model = None
        self._tokenizer = None
        self._config: dict[str, Any] | None = None
        self._load_lock = threading.Lock()
        self._generation_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self.loaded:
            return
        with self._load_lock:
            if self.loaded:
                return
            if not self.model_path.is_dir():
                raise FileNotFoundError(self.model_path)

            from mlx_lm import load

            from .mlx_conversion import _verify_checkpoint_checksums

            _verify_checkpoint_checksums(self.model_path)
            model, tokenizer, config = load(
                str(self.model_path),
                lazy=False,
                return_config=True,
            )
            if config.get("model_type") != "lumi" or config.get("model_file") != "lumi_mlx.py":
                raise ValueError("local artifact is not an explicit Lumi MLX model")
            context_window = config.get("max_position_embeddings")
            if not isinstance(context_window, int) or context_window < 2:
                raise ValueError("local artifact has an invalid context window")
            self._model = model
            self._tokenizer = tokenizer
            self._config = config
            self.context_window = context_window
            self.created = int((self.model_path / "config.json").stat().st_mtime)

    def model_metadata(self) -> dict[str, Any]:
        self.load()
        assert self._config is not None
        return {
            "context_window": self.context_window,
            "quantization": self._config.get("quantization"),
        }

    def stream(self, request: ChatCompletionRequest) -> Iterator[GenerationEvent]:
        self.load()
        assert self._model is not None and self._tokenizer is not None
        prompt = render_chat_prompt(request.messages)
        prompt_ids = encode_untrusted_text(self._tokenizer, prompt)
        if len(prompt_ids) + request.max_tokens > self.context_window:
            available = max(self.context_window - len(prompt_ids), 0)
            raise APIError(
                f"prompt uses {len(prompt_ids)} tokens; at most {available} completion "
                "tokens fit in the context window",
                code="context_length_exceeded",
                param="max_tokens",
            )
        return self._generate(prompt_ids, request)

    def _generate(
        self,
        prompt_ids: list[int],
        request: ChatCompletionRequest,
    ) -> Iterator[GenerationEvent]:
        import mlx.core as mx
        from mlx_lm.generate import stream_generate
        from mlx_lm.sample_utils import make_sampler

        assert self._model is not None and self._tokenizer is not None
        with self._generation_lock:
            if request.seed is not None:
                mx.random.seed(request.seed)
            sampler = make_sampler(temp=request.temperature, top_p=request.top_p)
            for response in stream_generate(
                self._model,
                self._tokenizer,
                prompt_ids,
                max_tokens=request.max_tokens,
                sampler=sampler,
            ):
                yield GenerationEvent(
                    text=response.text,
                    prompt_tokens=int(response.prompt_tokens),
                    completion_tokens=int(response.generation_tokens),
                    finish_reason=response.finish_reason,
                    prompt_tokens_per_second=float(response.prompt_tps),
                    generation_tokens_per_second=float(response.generation_tps),
                    peak_memory_gb=float(response.peak_memory),
                )


def _usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _chat_chunk(
    *,
    completion_id: str,
    created: int,
    model: str,
    delta: dict[str, str],
    finish_reason: str | None,
) -> dict[str, Any]:
    return {
        "choices": [
            {
                "delta": delta,
                "finish_reason": finish_reason,
                "index": 0,
                "logprobs": None,
            }
        ],
        "created": created,
        "id": completion_id,
        "model": model,
        "object": "chat.completion.chunk",
        "system_fingerprint": "lumi-local-v0.1",
    }


class LumiHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def create_server(
    engine: ChatEngine,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    ui_directory: str | Path = UI_DIRECTORY,
) -> LumiHTTPServer:
    """Create an injectable local HTTP server without starting its loop."""

    ui_root = Path(ui_directory)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "LumiLocal/0.1"

        def log_message(self, format_string: str, *args: Any) -> None:
            super().log_message(format_string, *args)

        def _headers(
            self,
            *,
            status: int,
            content_type: str,
            content_length: int | None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Permissions-Policy",
                "camera=(), geolocation=(), microphone=()",
            )
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; connect-src 'self'; img-src 'self' data:; "
                "font-src 'self'; script-src 'self'; style-src 'self'; "
                "style-src-attr 'unsafe-inline'",
            )
            if content_length is not None:
                self.send_header("Content-Length", str(content_length))
            self.end_headers()

        def _json(self, value: Any, *, status: int = HTTPStatus.OK) -> None:
            body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
            self._headers(
                status=status,
                content_type="application/json; charset=utf-8",
                content_length=len(body),
            )
            self.wfile.write(body)

        def _error(self, error: APIError) -> None:
            self.close_connection = True
            self._json(error.payload(), status=error.status)

        def _read_json(self) -> Any:
            if self.headers.get_content_type() != "application/json":
                raise APIError(
                    "Content-Type must be application/json",
                    status=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    code="unsupported_media_type",
                )
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                raise APIError("Content-Length is required", status=HTTPStatus.LENGTH_REQUIRED)
            try:
                content_length = int(raw_length)
            except ValueError as error:
                raise APIError("invalid Content-Length") from error
            if content_length < 1:
                raise APIError("request body cannot be empty")
            if content_length > MAX_REQUEST_BYTES:
                raise APIError(
                    "request body exceeds 1 MiB",
                    status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    code="request_too_large",
                )
            try:
                return json.loads(self.rfile.read(content_length))
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise APIError("request body is not valid UTF-8 JSON") from error

        def _asset(self, name: str) -> None:
            path = (ui_root / name).resolve()
            if not path.is_relative_to(ui_root.resolve()):
                self._error(
                    APIError(
                        "not found",
                        status=HTTPStatus.NOT_FOUND,
                        code="not_found",
                    )
                )
                return
            if not path.is_file():
                self._error(
                    APIError(
                        "not found",
                        status=HTTPStatus.NOT_FOUND,
                        code="not_found",
                    )
                )
                return
            body = path.read_bytes()
            media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if media_type.startswith("text/") or media_type in {"application/javascript"}:
                media_type += "; charset=utf-8"
            self._headers(
                status=HTTPStatus.OK,
                content_type=media_type,
                content_length=len(body),
            )
            self.wfile.write(body)

        def _ui_asset_name(self, request_path: str) -> str | None:
            relative = request_path.removeprefix("/")
            if relative in {"app.js", "renderer.js", "styles.css"}:
                return relative
            if relative.startswith("vendor/katex/"):
                return relative
            return None

        def _models(self) -> None:
            metadata = engine.model_metadata()
            self._json(
                {
                    "data": [
                        {
                            "created": engine.created,
                            "id": engine.model_id,
                            "object": "model",
                            "owned_by": "afnan-technologies",
                            **metadata,
                        }
                    ],
                    "object": "list",
                }
            )

        def _stream_chat(
            self,
            request: ChatCompletionRequest,
            events: Iterator[GenerationEvent],
        ) -> None:
            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            self._headers(
                status=HTTPStatus.OK,
                content_type="text/event-stream; charset=utf-8",
                content_length=None,
            )

            def send(value: Any) -> None:
                encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                self.wfile.write(f"data: {encoded}\n\n".encode())
                self.wfile.flush()

            prompt_tokens = 0
            completion_tokens = 0
            send(
                _chat_chunk(
                    completion_id=completion_id,
                    created=created,
                    model=request.model,
                    delta={"content": "", "role": "assistant"},
                    finish_reason=None,
                )
            )
            try:
                for event in events:
                    prompt_tokens = event.prompt_tokens
                    completion_tokens = event.completion_tokens
                    if event.text:
                        send(
                            _chat_chunk(
                                completion_id=completion_id,
                                created=created,
                                model=request.model,
                                delta={"content": event.text},
                                finish_reason=None,
                            )
                        )
                    if event.finish_reason is not None:
                        send(
                            _chat_chunk(
                                completion_id=completion_id,
                                created=created,
                                model=request.model,
                                delta={},
                                finish_reason=event.finish_reason,
                            )
                        )
                if request.include_stream_usage:
                    send(
                        {
                            "choices": [],
                            "created": created,
                            "id": completion_id,
                            "model": request.model,
                            "object": "chat.completion.chunk",
                            "usage": _usage(prompt_tokens, completion_tokens),
                        }
                    )
            except (BrokenPipeError, ConnectionResetError):
                raise
            except Exception:
                send(
                    APIError(
                        "generation failed",
                        status=HTTPStatus.INTERNAL_SERVER_ERROR,
                        code="generation_error",
                        error_type="server_error",
                    ).payload()
                )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True

        def _complete_chat(
            self,
            request: ChatCompletionRequest,
            events: Iterator[GenerationEvent],
        ) -> None:
            text = []
            prompt_tokens = 0
            completion_tokens = 0
            finish_reason = "length"
            metrics = {}
            for event in events:
                text.append(event.text)
                prompt_tokens = event.prompt_tokens
                completion_tokens = event.completion_tokens
                finish_reason = event.finish_reason or finish_reason
                metrics = {
                    "generation_tokens_per_second": event.generation_tokens_per_second,
                    "peak_memory_gb": event.peak_memory_gb,
                    "prompt_tokens_per_second": event.prompt_tokens_per_second,
                }
            self._json(
                {
                    "choices": [
                        {
                            "finish_reason": finish_reason,
                            "index": 0,
                            "logprobs": None,
                            "message": {"content": "".join(text), "role": "assistant"},
                        }
                    ],
                    "created": int(time.time()),
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "lumi_metrics": metrics,
                    "model": request.model,
                    "object": "chat.completion",
                    "system_fingerprint": "lumi-local-v0.1",
                    "usage": _usage(prompt_tokens, completion_tokens),
                }
            )

        def do_GET(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            try:
                if path == "/":
                    self._asset("index.html")
                elif asset_name := self._ui_asset_name(path):
                    self._asset(asset_name)
                elif path == "/health":
                    self._json(
                        {
                            "model": engine.model_id,
                            "model_loaded": engine.loaded,
                            "status": "ok",
                        }
                    )
                elif path == "/v1/models":
                    self._models()
                else:
                    raise APIError(
                        "not found",
                        status=HTTPStatus.NOT_FOUND,
                        code="not_found",
                    )
            except APIError as error:
                self._error(error)
            except Exception:
                self.log_error("unhandled GET failure")
                self._error(
                    APIError(
                        "internal server error",
                        status=HTTPStatus.INTERNAL_SERVER_ERROR,
                        code="internal_error",
                        error_type="server_error",
                    )
                )

        def do_POST(self) -> None:  # noqa: N802
            try:
                if urlsplit(self.path).path != "/v1/chat/completions":
                    raise APIError(
                        "not found",
                        status=HTTPStatus.NOT_FOUND,
                        code="not_found",
                    )
                request = parse_chat_request(self._read_json(), expected_model=engine.model_id)
                events = engine.stream(request)
                if request.stream:
                    self._stream_chat(request, events)
                else:
                    self._complete_chat(request, events)
            except APIError as error:
                self._error(error)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            except Exception:
                self.log_error("unhandled chat failure")
                if not self.wfile.closed:
                    self._error(
                        APIError(
                            "internal server error",
                            status=HTTPStatus.INTERNAL_SERVER_ERROR,
                            code="internal_error",
                            error_type="server_error",
                        )
                    )

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Allow", "GET, POST, OPTIONS")
            self.send_header("Content-Length", "0")
            self.end_headers()

    server_class = LumiHTTPServer
    try:
        if ipaddress.ip_address(host).version == 6:

            class IPv6LumiHTTPServer(LumiHTTPServer):
                address_family = socket.AF_INET6

            server_class = IPv6LumiHTTPServer
    except ValueError:
        pass
    return server_class((host, port), Handler)


def is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def serve_local(
    *,
    model_path: str | Path,
    model_id: str = DEFAULT_MODEL_ID,
    host: str = "127.0.0.1",
    port: int = 8080,
    allow_network: bool = False,
    preload: bool = True,
) -> None:
    """Run the blocking loopback server used by the CLI."""

    if not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    if not is_loopback_host(host) and not allow_network:
        raise ValueError("refusing a non-loopback bind without allow_network=True")
    engine = MLXChatEngine(model_path, model_id=model_id)
    if preload:
        engine.load()
    server = create_server(engine, host=host, port=port)
    actual_host, actual_port = server.server_address[:2]
    print(f"Lumi chat: http://{actual_host}:{actual_port}", flush=True)
    print(f"OpenAI base URL: http://{actual_host}:{actual_port}/v1", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
