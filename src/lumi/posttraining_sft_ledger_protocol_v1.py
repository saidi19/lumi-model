"""Bounded parent-owned SFT ledger protocol core; no transport or GPU dispatch.

One trusted parent owns one existing Supervisor and every rank's protocol state.
Workers must not construct independent ledgers. The parent supplies already-bound
channels, distributes independently generated rank tokens out of band, calls
``feed`` with received bytes and sends returned ACK frames. It must call
``mark_ack_sent`` only after a complete send, and fail on send errors, EOF or ACK
timeout. Workers confirm the ACK before issuing another operation. Reservations
are charged BEFORE creating an ACK; failures never refund or allow replay.

Tokens authenticate a rank within this parent session, not an OS process or an
owner approval. They are retained only as private session state, never summaries
or errors. No network, files, subprocesses, Torch or CUDA are accessed here. The
future transport must use bounded local IPC, independent deadlines and process
termination. Durable journal/restart recovery, target authentication, cgroups,
and guarded real-worker integration remain absent; this core grants no GPU
permission. A compromised parent or arbitrary in-process code is out of scope.

The core assumes one persistent set of parent-bound rank channels for its entire
three-invocation lifecycle. EOF during an open session is always fatal, including
between invocations. Fresh-process worker channel closure/rebinding needs a
separately reviewed epoch handshake; do not suppress EOF or reuse this session's
tokens across replacement channels. No fresh-process transport adapter exists.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
from dataclasses import dataclass
from typing import Protocol

MAXIMUM_FRAME_BYTES = 4096
REQUEST_SCHEMA = "lumi.sft-ledger-request.v0.1"
ACK_SCHEMA = "lumi.sft-ledger-ack.v0.1"
MAXIMUM_INTEGER = 2**63 - 1
RESOURCE_FIELDS = frozenset((
    "checkpoint_read_bytes", "encoded_binary_read_bytes", "encoded_index_read_bytes",
    "private_write_bytes", "log_bytes", "encoded_index_rows",
))
ERROR_CODES = frozenset((
    "configuration_rejected", "frame_type_rejected", "frame_size_rejected",
    "frame_incomplete", "frame_noncanonical", "frame_schema_rejected",
    "authentication_rejected", "sequence_rejected", "request_pending",
    "operation_rejected", "argument_rejected", "supervisor_rejected",
    "ack_rejected", "ack_not_sent", "ack_send_failed", "ack_timeout",
    "rank_disconnected", "step_coordinator_required", "step_boundary_rejected",
    "step_ack_duplicate", "lifecycle_rejected", "protocol_interrupted",
))


class LedgerProtocolError(ValueError):
    """Fixed allowlisted code, without token, request or exception details."""

    def __init__(self, code: str):
        if type(code) is not str or code not in ERROR_CODES:
            code = "protocol_interrupted"
        self.code = code
        super().__init__(code)


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise LedgerProtocolError(code) from None


def _integer(value: object, *, zero: bool = False) -> bool:
    return type(value) is int and (0 if zero else 1) <= value <= MAXIMUM_INTEGER


def _token(value: object) -> bool:
    return (
        type(value) is str and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical(value: dict) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError, RecursionError, OverflowError):
        raise LedgerProtocolError("frame_schema_rejected") from None


def encode_frame(message: dict) -> bytes:
    """Pure framing helper; encoding is not admission or authorization."""
    _require(type(message) is dict, "frame_schema_rejected")
    payload = _canonical(message)
    _require(0 < len(payload) <= MAXIMUM_FRAME_BYTES, "frame_size_rejected")
    return len(payload).to_bytes(4, "big") + payload


def _decode_payload(payload: bytes) -> dict:
    def pairs(items):
        output = {}
        for key, value in items:
            _require(key not in output, "frame_noncanonical")
            output[key] = value
        return output

    def reject_constant(_value):
        raise LedgerProtocolError("frame_noncanonical")

    try:
        decoded = json.loads(payload, object_pairs_hook=pairs, parse_constant=reject_constant)
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
        raise LedgerProtocolError("frame_noncanonical") from None
    _require(type(decoded) is dict, "frame_schema_rejected")
    _require(_canonical(decoded) == payload, "frame_noncanonical")
    return decoded


class FrameDecoder:
    """Incremental bounded decoder. An error permanently consumes the decoder.

    A read chunk itself is capped at one maximum frame including its header.
    At most 4096 payload bytes are buffered. The IPC layer must use the same read
    bound and propagate decoder errors to its ParentLedgerSession.
    """

    def __init__(self):
        self._buffer = bytearray()
        self._expected = None
        self._failed = False

    def feed(self, chunk: bytes) -> tuple[dict, ...]:
        try:
            _require(not self._failed, "protocol_interrupted")
            _require(type(chunk) is bytes, "frame_type_rejected")
            _require(0 < len(chunk) <= MAXIMUM_FRAME_BYTES + 4, "frame_size_rejected")
            output = []
            cursor = 0
            while cursor < len(chunk):
                target = 4 if self._expected is None else self._expected
                take = min(target - len(self._buffer), len(chunk) - cursor)
                self._buffer.extend(chunk[cursor:cursor + take])
                cursor += take
                if len(self._buffer) != target:
                    continue
                if self._expected is None:
                    self._expected = int.from_bytes(self._buffer, "big")
                    self._buffer.clear()
                    _require(0 < self._expected <= MAXIMUM_FRAME_BYTES, "frame_size_rejected")
                else:
                    output.append(_decode_payload(bytes(self._buffer)))
                    self._buffer.clear()
                    self._expected = None
            return tuple(output)
        except BaseException:
            self._failed = True
            self._buffer.clear()
            self._expected = None
            raise

    def end_of_stream(self) -> None:
        _require(not self._failed, "protocol_interrupted")
        self._failed = True
        partial = bool(self._buffer) or self._expected is not None
        self._buffer.clear()
        self._expected = None
        _require(not partial, "frame_incomplete")


def create_rank_tokens(world_size: int) -> dict[int, str]:
    """Use independent 256-bit random tokens; deliver privately, never log them."""
    _require(type(world_size) is int and 2 <= world_size <= 12, "configuration_rejected")
    return {rank: secrets.token_hex(32) for rank in range(world_size)}


class SupervisorController(Protocol):
    """Trusted parent dependency, not a worker-provided protocol object."""

    _world_size: int

    def begin_invocation(self, run: str, from_step: int, stop_step: int, resume: bool) -> None: ...
    def reserve(self, rank: int, **amounts: int) -> None: ...
    def reserve_optimizer_step(self, from_step: int, to_step: int) -> None: ...
    def finish_optimizer_step(self, to_step: int) -> None: ...
    def observe_memory(self, host_ram_bytes: int, per_gpu_memory_bytes: list[int]) -> None: ...
    def finish_invocation(self, stop_step: int) -> None: ...
    def checkpoint(self) -> None: ...
    def interrupt(self) -> None: ...


@dataclass(frozen=True, slots=True)
class PendingAck:
    rank: int
    sequence: int
    frame: bytes
    sha256: str


@dataclass(slots=True)
class _Transaction:
    ack: PendingAck
    operation: str
    to_step: int | None
    sent: bool = False


class ParentLedgerSession:
    """All rank requests serialize against ONE parent-owned Supervisor.

    Parent lifecycle methods cannot be requested by workers. Every worker request
    is sequenced/authenticated; ACK confirmation binds its exact response digest.
    Optimizer completion waits for confirmations from every bound rank. No
    reservation refund, request retry, rank replacement or session restart API.
    """

    def __init__(self, supervisor: SupervisorController, *, world_size: int, rank_tokens: dict):
        _require(
            type(world_size) is int and 2 <= world_size <= 12
            and getattr(supervisor, "_world_size", None) == world_size,
            "configuration_rejected",
        )
        _require(
            type(rank_tokens) is dict and len(rank_tokens) == world_size
            and all(type(rank) is int for rank in rank_tokens)
            and set(rank_tokens) == set(range(world_size))
            and all(_token(token) for token in rank_tokens.values())
            and len(set(rank_tokens.values())) == world_size,
            "configuration_rejected",
        )
        self._supervisor = supervisor
        self._world_size = world_size
        self._tokens = dict(rank_tokens)
        self._lock = threading.Lock()
        self._decoders = {rank: FrameDecoder() for rank in rank_tokens}
        self._next = dict.fromkeys(rank_tokens, 1)
        self._pending: dict[int, _Transaction] = {}
        self._step = None
        self._step_confirmed: set[int] = set()
        self._failure = None
        self._active = False
        self._closed = False
        self._requests = 0
        self._acks_sent = 0
        self._confirmations = 0
        self._completed_steps = 0
        self._completed_invocations = 0

    def _fail(self, code: str) -> None:
        if self._failure is None:
            self._failure = code if code in ERROR_CODES else "protocol_interrupted"
            self._tokens.clear()
            try:
                self._supervisor.interrupt()
            except BaseException:
                pass
        raise LedgerProtocolError(self._failure) from None

    def _need(self, condition: bool, code: str) -> None:
        if not condition:
            self._fail(code)

    def _healthy(self) -> None:
        if self._failure is not None:
            self._fail(self._failure)
        self._need(not self._closed, "lifecycle_rejected")

    def _rank(self, rank: int) -> None:
        self._need(type(rank) is int and 0 <= rank < self._world_size, "authentication_rejected")

    def _call(self, method: str, *args, **kwargs) -> None:
        try:
            getattr(self._supervisor, method)(*args, **kwargs)
        except BaseException:
            self._fail("supervisor_rejected")

    def begin_invocation(self, run: str, from_step: int, stop_step: int, resume: bool) -> None:
        with self._lock:
            self._healthy()
            self._need(not self._active and not self._pending and self._step is None, "lifecycle_rejected")
            self._call("begin_invocation", run, from_step, stop_step, resume)
            self._active = True

    def observe_memory(self, host_ram_bytes: int, per_gpu_memory_bytes: list[int]) -> None:
        with self._lock:
            self._healthy()
            self._need(self._active, "lifecycle_rejected")
            self._call("observe_memory", host_ram_bytes, per_gpu_memory_bytes)

    def finish_invocation(self, stop_step: int) -> None:
        with self._lock:
            self._healthy()
            self._need(self._active and not self._pending and self._step is None, "lifecycle_rejected")
            self._call("finish_invocation", stop_step)
            self._active = False
            self._completed_invocations += 1
            if self._completed_invocations == 3:
                self._closed = True
                self._tokens.clear()

    def _message(self, channel_rank: int, request: dict) -> PendingAck | None:
        self._need(
            set(request) == {"schema_version", "rank", "sequence", "token", "operation", "arguments"},
            "frame_schema_rejected",
        )
        self._need(request["schema_version"] == REQUEST_SCHEMA, "frame_schema_rejected")
        self._need(
            type(request["rank"]) is int and request["rank"] == channel_rank
            and _token(request["token"])
            and hmac.compare_digest(request["token"], self._tokens[channel_rank]),
            "authentication_rejected",
        )
        sequence, operation, arguments = request["sequence"], request["operation"], request["arguments"]
        self._need(_integer(sequence), "sequence_rejected")
        self._need(type(operation) is str and type(arguments) is dict, "frame_schema_rejected")
        self._call("checkpoint")
        if operation == "confirm_ack":
            transaction = self._pending.get(channel_rank)
            self._need(transaction is not None, "ack_rejected")
            self._need(sequence == transaction.ack.sequence, "sequence_rejected")
            self._need(transaction.sent, "ack_not_sent")
            self._need(
                set(arguments) == {"ack_sha256"} and _token(arguments["ack_sha256"])
                and hmac.compare_digest(arguments["ack_sha256"], transaction.ack.sha256),
                "ack_rejected",
            )
            if transaction.operation == "step_complete":
                self._need(channel_rank not in self._step_confirmed, "step_ack_duplicate")
                self._step_confirmed.add(channel_rank)
            del self._pending[channel_rank]
            self._confirmations += 1
            if self._step is not None and len(self._step_confirmed) == self._world_size:
                self._need(not self._pending, "step_boundary_rejected")
                self._call("finish_optimizer_step", self._step)
                self._completed_steps += 1
                self._step = None
                self._step_confirmed.clear()
            return None
        self._need(channel_rank not in self._pending, "request_pending")
        self._need(sequence == self._next[channel_rank], "sequence_rejected")
        self._need(self._active, "lifecycle_rejected")
        self._next[channel_rank] += 1
        to_step = None
        if operation == "reserve":
            self._need(
                bool(arguments) and set(arguments) <= RESOURCE_FIELDS
                and all(_integer(amount, zero=True) for amount in arguments.values())
                and any(amount > 0 for amount in arguments.values()), "argument_rejected",
            )
            self._call("reserve", channel_rank, **arguments)
        elif operation == "reserve_step":
            self._need(channel_rank == 0, "step_coordinator_required")
            self._need(self._step is None and not self._pending, "step_boundary_rejected")
            self._need(
                set(arguments) == {"from_step", "to_step"}
                and _integer(arguments["from_step"], zero=True)
                and _integer(arguments["to_step"]), "argument_rejected",
            )
            self._call("reserve_optimizer_step", arguments["from_step"], arguments["to_step"])
            self._step = arguments["to_step"]
            self._step_confirmed.clear()
        elif operation == "step_complete":
            self._need(
                set(arguments) == {"to_step"} and _integer(arguments["to_step"]), "argument_rejected",
            )
            self._need(self._step is not None and arguments["to_step"] == self._step, "step_boundary_rejected")
            self._need(channel_rank not in self._step_confirmed, "step_ack_duplicate")
            # Coordinator's reservation ACK must already have been confirmed.
            self._need(not any(tx.operation == "reserve_step" for tx in self._pending.values()), "step_boundary_rejected")
            to_step = self._step
        else:
            self._fail("operation_rejected")
        frame = encode_frame({
            "schema_version": ACK_SCHEMA, "rank": channel_rank, "sequence": sequence,
            "operation": operation, "status": "reserved",
        })
        ack = PendingAck(channel_rank, sequence, frame, hashlib.sha256(frame).hexdigest())
        self._pending[channel_rank] = _Transaction(ack, operation, to_step)
        self._requests += 1
        return ack

    def feed(self, channel_rank: int, chunk: bytes) -> tuple[PendingAck, ...]:
        """Receive a bounded IPC chunk from a parent-bound rank channel.

        Confirmations produce no ACK of their own. Any invalid frame/request
        consumes the whole session; no charged operation is rolled back.
        """
        with self._lock:
            self._healthy()
            self._rank(channel_rank)
            try:
                messages = self._decoders[channel_rank].feed(chunk)
                return tuple(ack for request in messages if (ack := self._message(channel_rank, request)) is not None)
            except LedgerProtocolError as error:
                self._fail(error.code)
            except BaseException:
                self._fail("protocol_interrupted")

    def mark_ack_sent(self, ack: PendingAck) -> None:
        """Trusted transport callback, only after successfully sending all bytes."""
        with self._lock:
            self._healthy()
            self._need(type(ack) is PendingAck, "ack_rejected")
            self._need(
                type(ack.rank) is int and _integer(ack.sequence)
                and type(ack.frame) is bytes and len(ack.frame) <= MAXIMUM_FRAME_BYTES + 4
                and _token(ack.sha256), "ack_rejected",
            )
            self._rank(ack.rank)
            transaction = self._pending.get(ack.rank)
            self._need(transaction is not None and ack == transaction.ack, "ack_rejected")
            self._need(not transaction.sent, "ack_rejected")
            transaction.sent = True
            self._acks_sent += 1

    def ack_send_failed(self, channel_rank: int) -> None:
        with self._lock:
            self._healthy()
            self._rank(channel_rank)
            self._fail("ack_send_failed")

    def check_pending(self) -> None:
        """Call at the independent transport ACK deadline; no internal timer."""
        with self._lock:
            self._healthy()
            self._call("checkpoint")
            self._need(not self._pending, "ack_timeout")

    def disconnect(self, channel_rank: int) -> None:
        with self._lock:
            self._healthy()
            self._rank(channel_rank)
            try:
                self._decoders[channel_rank].end_of_stream()
            except LedgerProtocolError as error:
                self._fail(error.code)
            self._fail("rank_disconnected")

    def summary(self) -> dict:
        """Only aggregate counts and allowlisted codes, never tokens or IDs."""
        with self._lock:
            return {
                "schema_version": "lumi.sft-ledger-protocol-summary.v0.1",
                "state": (
                    "failed" if self._failure else "completed" if self._closed
                    else "active" if self._active else "ready"
                ),
                "failure_code": self._failure,
                "requests_predebited": self._requests,
                "acknowledgements_sent": self._acks_sent,
                "acknowledgements_confirmed": self._confirmations,
                "pending_acknowledgements": len(self._pending),
                "rank_step_confirmations_pending": len(self._step_confirmed),
                "global_optimizer_steps_completed": self._completed_steps,
                "invocations_completed": self._completed_invocations,
                "transport_implemented": False,
                "durable_transactions": False,
                "execution_permitted": False,
                "production_eligible": False,
            }
