"""Fixed Lambda HTTP/1.1 request profile and bounded incremental response framing.

Pure byte handling only; no credentials are read and no transport is performed.
Request bytes and JSON bodies are sensitive transient internal values, not reports.
This profile is narrower than general HTTP or the full Lambda API.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re

HOST = "cloud.lambda.ai"
MAXIMUM_REQUEST_BYTES = 8192
MAXIMUM_HEADER_BYTES = 8192
MAXIMUM_BODY_BYTES = 65536
MAXIMUM_WIRE_BYTES = 81920
MAXIMUM_CHUNKS = 128
ID = re.compile(r"[0-9a-f]{32}\Z", re.ASCII)
HASH = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
LABEL = re.compile(r"[a-z0-9_-]{1,64}\Z", re.ASCII)
IMAGE_ID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z", re.ASCII)
TOKEN = re.compile(rb"[A-Za-z0-9._~+/-]{8,510}={0,2}\Z")
HEADER_NAME = re.compile(rb"[!#$%&'*+.^_`|~0-9A-Za-z-]+\Z")
_FACTORY = object()


class HttpError(ValueError):
    def __init__(self, _value=None):
        super().__init__("http_profile_rejected")


def need(condition):
    if not condition:
        raise HttpError from None


def matches(value, expression):
    return type(value) is str and expression.fullmatch(value) is not None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def validate_launch(body, expected_sha256, region, instance_type):
    """One pinned image/key, no implicit image, quantity, mounts or cloud-init script."""
    need(type(body) is bytes and 0 < len(body) <= 4096)
    need(matches(expected_sha256, HASH) and matches(region, LABEL)
         and matches(instance_type, LABEL))
    need(hmac.compare_digest(hashlib.sha256(body).hexdigest(), expected_sha256))

    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value)
            value[key] = item
        return value

    try:
        value = json.loads(body.decode("ascii"), object_pairs_hook=pairs)
        need(type(value) is dict and set(value) == {
            "region_name", "instance_type_name", "ssh_key_names", "image",
        })
        need(value["region_name"] == region and value["instance_type_name"] == instance_type)
        keys = value["ssh_key_names"]
        need(type(keys) is list and len(keys) == 1 and type(keys[0]) is str
             and 1 <= len(keys[0]) <= 64 and keys[0].strip() == keys[0]
             and all(32 <= ord(character) <= 126 for character in keys[0]))
        image = value["image"]
        need(type(image) is dict and set(image) == {"id"} and matches(image["id"], IMAGE_ID))
        need(canonical(value) == body)
    except Exception:
        raise HttpError from None
    return body


class Request:
    """Cooperative one-shot request object, not an approval or security capability."""

    __slots__ = ("_raw", "_digest", "_used", "operation")

    def __init__(self, operation, raw, *, _factory=None):
        need(_factory is _FACTORY)
        self.operation = operation
        self._raw = raw
        self._digest = hashlib.sha256(raw).digest()
        self._used = False

    def __repr__(self):
        return "<Lambda Request: redacted>"

    def claim(self):
        need(not self._used)
        self._used = True
        raw, self._raw = self._raw, b""
        need(type(raw) is bytes and 0 < len(raw) <= MAXIMUM_REQUEST_BYTES
             and type(self._digest) is bytes and len(self._digest) == 32
             and hmac.compare_digest(hashlib.sha256(raw).digest(), self._digest))
        return raw


def prepare_request(operation, *, token, instance_id=None, launch_body=None,
                    expected_launch_sha256=None, region=None, instance_type=None):
    """Compile only a fixed host/path. Supplied token/pins do NOT authenticate authority."""
    need(type(operation) is str and operation in {"create", "observe", "terminate"})
    need(type(token) is bytes and TOKEN.fullmatch(token) is not None and len(token) <= 512)
    if operation == "create":
        need(instance_id is None)
        body = validate_launch(launch_body, expected_launch_sha256, region, instance_type)
        path = "/api/v1/instance-operations/launch"
    else:
        need(matches(instance_id, ID))
        need(launch_body is expected_launch_sha256 is region is instance_type is None)
        if operation == "observe":
            path, body = "/api/v1/instances/" + instance_id, b""
        else:
            path = "/api/v1/instance-operations/terminate"
            body = canonical({"instance_ids": [instance_id]})
    method = "GET" if operation == "observe" else "POST"
    raw = (f"{method} {path} HTTP/1.1\r\nHost: {HOST}\r\n".encode("ascii")
           + b"Authorization: Bearer " + token
           + b"\r\nAccept: application/json\r\nAccept-Encoding: identity\r\n"
           + b"Content-Type: application/json\r\nConnection: close\r\n"
           + f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
    need(len(raw) <= MAXIMUM_REQUEST_BYTES)
    return Request(operation, raw, _factory=_FACTORY)


class ResponseParser:
    """Accept only a single HTTP/1.1 200 JSON body and a clean stream EOF.

    Content-Length or bounded chunked framing is required, not close-delimited
    bodies. Reject compression, ambiguous duplicate framing, interim responses,
    folded headers, chunk extensions/trailers and every byte after the message.
    No raw header/reason/body is exposed in errors or a diagnostic report.
    """

    def __init__(self):
        self._buffer = bytearray()
        self._body = bytearray()
        self._state = "headers"
        self._remaining = 0
        self._chunks = 0
        self._received = 0
        self._failed = False

    def __repr__(self):
        return "<Lambda ResponseParser: redacted>"

    def _headers(self, raw):
        lines = raw.split(b"\r\n")
        need(1 <= len(lines) <= 65)
        need(re.fullmatch(rb"HTTP/1\.1 200 [\x20-\x7e]{0,128}", lines[0]) is not None)
        fields = {}
        for line in lines[1:]:
            need(0 < len(line) <= 1024 and b":" in line)
            key, value = line.split(b":", 1)
            need(HEADER_NAME.fullmatch(key) is not None)
            key = key.lower()
            need(key not in fields and all(byte == 9 or 32 <= byte <= 126 for byte in value))
            fields[key] = value.strip(b" \t").lower()
        need(re.fullmatch(rb"application/json(?:;[ \t]*charset=utf-8)?",
                          fields.get(b"content-type", b"")) is not None)
        need(fields.get(b"content-encoding", b"identity") == b"identity")
        length, transfer = fields.get(b"content-length"), fields.get(b"transfer-encoding")
        need((length is None) != (transfer is None))
        if transfer is not None:
            need(transfer == b"chunked")
            self._state = "chunk_size"
        else:
            need(re.fullmatch(rb"0|[1-9][0-9]{0,5}", length) is not None)
            self._remaining = int(length)
            need(self._remaining <= MAXIMUM_BODY_BYTES)
            self._state = "length" if self._remaining else "done"

    def feed(self, data):
        try:
            need(not self._failed and self._state != "finished"
                 and type(data) is bytes and 0 < len(data) <= MAXIMUM_WIRE_BYTES)
            self._received += len(data)
            need(self._received <= MAXIMUM_WIRE_BYTES)
            self._buffer.extend(data)
            while True:
                if self._state == "headers":
                    boundary = self._buffer.find(b"\r\n\r\n")
                    if boundary < 0:
                        need(len(self._buffer) <= MAXIMUM_HEADER_BYTES)
                        break
                    need(boundary + 4 <= MAXIMUM_HEADER_BYTES)
                    header = bytes(self._buffer[:boundary])
                    del self._buffer[:boundary + 4]
                    self._headers(header)
                elif self._state in {"length", "chunk_data"}:
                    count = min(self._remaining, len(self._buffer))
                    self._body.extend(self._buffer[:count])
                    del self._buffer[:count]
                    self._remaining -= count
                    need(len(self._body) <= MAXIMUM_BODY_BYTES)
                    if self._remaining:
                        break
                    self._state = "done" if self._state == "length" else "chunk_crlf"
                elif self._state == "chunk_size":
                    boundary = self._buffer.find(b"\r\n")
                    if boundary < 0:
                        need(len(self._buffer) <= 10)
                        break
                    size = bytes(self._buffer[:boundary])
                    need(re.fullmatch(rb"[0-9A-Fa-f]{1,8}", size) is not None)
                    del self._buffer[:boundary + 2]
                    self._remaining = int(size, 16)
                    if self._remaining == 0:
                        self._state = "final_crlf"
                    else:
                        self._chunks += 1
                        need(self._chunks <= MAXIMUM_CHUNKS and
                             len(self._body) + self._remaining <= MAXIMUM_BODY_BYTES)
                        self._state = "chunk_data"
                elif self._state in {"chunk_crlf", "final_crlf"}:
                    if len(self._buffer) < 2:
                        break
                    need(self._buffer[:2] == b"\r\n")
                    del self._buffer[:2]
                    self._state = "done" if self._state == "final_crlf" else "chunk_size"
                elif self._state == "done":
                    need(not self._buffer)
                    break
                else:
                    raise HttpError
        except Exception:
            self._failed = True
            self._buffer.clear()
            self._body.clear()
            raise HttpError from None

    def finish(self):
        """Called only on EOF; framed body alone is not a successful exchange."""
        if self._failed or self._state != "done" or self._buffer:
            self._failed = True
            self._buffer.clear()
            self._body.clear()
            raise HttpError from None
        result = bytes(self._body)
        self._body.clear()
        self._state = "finished"
        return result

    def discard(self):
        self._failed = True
        self._buffer.clear()
        self._body.clear()
