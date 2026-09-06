"""One-invocation epoch protocol over a cumulative, parent-owned supervisor.

This is a stdlib-only protocol core, not a transport or a worker launcher. One
fresh session and independently generated rank tokens belong to each actual
worker invocation. The trusted controller owns epoch admission and durable
uniqueness, and MUST keep the same cumulative supervisor across all invocations.
Constructing this object does not reset or replace that supervisor's ledger.

Requests and ACKs bind a 256-bit epoch as well as their rank and sequence. Tokens
authenticate a parent-bound channel, not an OS PID or an approval. A successful
finish permanently seals this session; only then is EOF expected. A failed or
closed session can never call the supervisor again, so late callbacks on a
retired epoch cannot interfere with a newer epoch. No retry, refund, lifecycle
replay, replacement channel, or reset API exists. Reservations precede ACKs;
optimizer completion requires every rank's ACK confirmation.

No network, private input, file, subprocess, ML framework, or GPU is accessed.
Durability and actual process ownership are the controller's responsibilities.
This core never grants execution permission or claims OS PID authentication.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
from dataclasses import dataclass

from .posttraining_sft_ledger_protocol_v1 import (
    ERROR_CODES,
    MAXIMUM_FRAME_BYTES,
    RESOURCE_FIELDS,
    FrameDecoder,
    LedgerProtocolError,
    PendingAck,
    SupervisorController,
    _integer,
    _require,
    _token,
    encode_frame,
)
from .posttraining_sft_ledger_protocol_v1 import (
    create_rank_tokens as create_rank_tokens,
)

REQUEST_SCHEMA = "lumi.sft-epoch-ledger-request.v0.1"
ACK_SCHEMA = "lumi.sft-epoch-ledger-ack.v0.1"
REQUEST_FIELDS = frozenset(
    (
        "schema_version",
        "epoch",
        "rank",
        "sequence",
        "token",
        "operation",
        "arguments",
    )
)
ACK_FIELDS = frozenset(
    (
        "schema_version",
        "epoch",
        "rank",
        "sequence",
        "operation",
        "status",
    )
)


def create_epoch() -> str:
    """Generate a fresh 256-bit epoch; the controller must prevent durable reuse."""
    return secrets.token_hex(32)


@dataclass(slots=True)
class _Transaction:
    ack: PendingAck
    operation: str
    sent: bool = False


class EpochLedgerSession:
    """Serialize bound rank requests for exactly one cumulative invocation.

    Parent lifecycle methods cannot be requested by workers. ``mark_ack_sent``
    is trusted transport input and must follow a successful complete ACK send.
    ``check_pending`` belongs to an independent bounded transport ACK deadline.
    The controller must bind each fresh channel to its own rank and session;
    routing a stale channel to the current session is not permitted.
    """

    def __init__(
        self, supervisor: SupervisorController, *, world_size: int, rank_tokens: dict, epoch: str
    ):
        _require(
            type(world_size) is int
            and 2 <= world_size <= 12
            and getattr(supervisor, "_world_size", None) == world_size,
            "configuration_rejected",
        )
        _require(
            type(rank_tokens) is dict
            and len(rank_tokens) == world_size
            and all(type(rank) is int for rank in rank_tokens)
            and set(rank_tokens) == set(range(world_size))
            and all(_token(token) for token in rank_tokens.values())
            and len(set(rank_tokens.values())) == world_size
            and _token(epoch)
            and epoch not in rank_tokens.values(),
            "configuration_rejected",
        )
        self._supervisor = supervisor
        self._world_size = world_size
        self._epoch = epoch
        self._tokens = dict(rank_tokens)
        self._lock = threading.Lock()
        self._decoders = {rank: FrameDecoder() for rank in rank_tokens}
        self._next = dict.fromkeys(rank_tokens, 1)
        self._pending: dict[int, _Transaction] = {}
        self._step = None
        self._step_confirmed: set[int] = set()
        self._failure = None
        self._active = False
        self._begun = False
        self._closed = False
        self._requests = 0
        self._acks_sent = 0
        self._confirmations = 0
        self._completed_steps = 0
        self._completed_invocations = 0

    def _fail(self, code: str) -> None:
        # A retired session must NEVER interrupt the shared cumulative ledger.
        if self._closed:
            raise LedgerProtocolError("lifecycle_rejected") from None
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
            raise LedgerProtocolError(self._failure) from None
        if self._closed:
            raise LedgerProtocolError("lifecycle_rejected") from None

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
            self._need(
                not self._begun and not self._active and not self._pending and self._step is None,
                "lifecycle_rejected",
            )
            self._begun = True
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
            self._need(
                self._active and not self._pending and self._step is None, "lifecycle_rejected"
            )
            # Seal at a complete frame boundary BEFORE advancing the supervisor.
            try:
                for decoder in self._decoders.values():
                    decoder.end_of_stream()
            except LedgerProtocolError as error:
                self._fail(error.code)
            self._call("finish_invocation", stop_step)
            self._active = False
            self._completed_invocations = 1
            self._closed = True
            self._tokens.clear()

    def _message(self, channel_rank: int, request: dict) -> PendingAck | None:
        self._need(set(request) == REQUEST_FIELDS, "frame_schema_rejected")
        self._need(request["schema_version"] == REQUEST_SCHEMA, "frame_schema_rejected")
        self._need(
            _token(request["epoch"])
            and hmac.compare_digest(request["epoch"], self._epoch)
            and type(request["rank"]) is int
            and request["rank"] == channel_rank
            and _token(request["token"])
            and hmac.compare_digest(request["token"], self._tokens[channel_rank]),
            "authentication_rejected",
        )
        sequence, operation, arguments = (
            request["sequence"],
            request["operation"],
            request["arguments"],
        )
        self._need(_integer(sequence), "sequence_rejected")
        self._need(type(operation) is str and type(arguments) is dict, "frame_schema_rejected")
        self._need(self._active, "lifecycle_rejected")
        self._call("checkpoint")
        if operation == "confirm_ack":
            transaction = self._pending.get(channel_rank)
            self._need(transaction is not None, "ack_rejected")
            self._need(sequence == transaction.ack.sequence, "sequence_rejected")
            self._need(transaction.sent, "ack_not_sent")
            self._need(
                set(arguments) == {"ack_sha256"}
                and _token(arguments["ack_sha256"])
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
        self._next[channel_rank] += 1
        if operation == "reserve":
            self._need(
                bool(arguments)
                and set(arguments) <= RESOURCE_FIELDS
                and all(_integer(amount, zero=True) for amount in arguments.values())
                and any(amount > 0 for amount in arguments.values()),
                "argument_rejected",
            )
            self._call("reserve", channel_rank, **arguments)
        elif operation == "reserve_step":
            self._need(channel_rank == 0, "step_coordinator_required")
            self._need(self._step is None and not self._pending, "step_boundary_rejected")
            self._need(
                set(arguments) == {"from_step", "to_step"}
                and _integer(arguments["from_step"], zero=True)
                and _integer(arguments["to_step"]),
                "argument_rejected",
            )
            self._call("reserve_optimizer_step", arguments["from_step"], arguments["to_step"])
            self._step = arguments["to_step"]
            self._step_confirmed.clear()
        elif operation == "step_complete":
            self._need(
                set(arguments) == {"to_step"} and _integer(arguments["to_step"]),
                "argument_rejected",
            )
            self._need(
                self._step is not None and arguments["to_step"] == self._step,
                "step_boundary_rejected",
            )
            self._need(channel_rank not in self._step_confirmed, "step_ack_duplicate")
            self._need(
                not any(tx.operation == "reserve_step" for tx in self._pending.values()),
                "step_boundary_rejected",
            )
        else:
            self._fail("operation_rejected")
        frame = encode_frame(
            {
                "schema_version": ACK_SCHEMA,
                "epoch": self._epoch,
                "rank": channel_rank,
                "sequence": sequence,
                "operation": operation,
                "status": "reserved",
            }
        )
        ack = PendingAck(channel_rank, sequence, frame, hashlib.sha256(frame).hexdigest())
        self._pending[channel_rank] = _Transaction(ack, operation)
        self._requests += 1
        return ack

    def feed(self, channel_rank: int, chunk: bytes) -> tuple[PendingAck, ...]:
        """Consume a bounded chunk on this epoch's permanently bound rank channel."""
        with self._lock:
            self._healthy()
            self._rank(channel_rank)
            try:
                messages = self._decoders[channel_rank].feed(chunk)
                return tuple(
                    ack
                    for request in messages
                    if (ack := self._message(channel_rank, request)) is not None
                )
            except LedgerProtocolError as error:
                self._fail(error.code)
            except BaseException:
                self._fail("protocol_interrupted")

    def mark_ack_sent(self, ack: PendingAck) -> None:
        """Trusted callback only after a successful complete send of this exact ACK."""
        with self._lock:
            self._healthy()
            self._need(type(ack) is PendingAck, "ack_rejected")
            self._need(
                type(ack.rank) is int
                and _integer(ack.sequence)
                and type(ack.frame) is bytes
                and len(ack.frame) <= MAXIMUM_FRAME_BYTES + 4
                and _token(ack.sha256),
                "ack_rejected",
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
        """Fail on outstanding transactions at the independent transport deadline."""
        with self._lock:
            self._healthy()
            self._call("checkpoint")
            self._need(not self._pending, "ack_timeout")

    def disconnect(self, channel_rank: int) -> None:
        """Accept EOF only after successful finish; retired EOF cannot touch the ledger."""
        with self._lock:
            if self._closed:
                _require(
                    type(channel_rank) is int and 0 <= channel_rank < self._world_size,
                    "authentication_rejected",
                )
                return
            self._healthy()
            self._rank(channel_rank)
            try:
                self._decoders[channel_rank].end_of_stream()
            except LedgerProtocolError as error:
                self._fail(error.code)
            self._fail("rank_disconnected")

    def summary(self) -> dict:
        """Only aggregate counts and fixed codes; never tokens, epochs, or OS PIDs."""
        with self._lock:
            return {
                "schema_version": "lumi.sft-epoch-ledger-protocol-summary.v0.1",
                "state": (
                    "failed"
                    if self._failure
                    else "completed"
                    if self._closed
                    else "active"
                    if self._active
                    else "ready"
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
                "os_pid_authenticated": False,
                "execution_permitted": False,
                "production_eligible": False,
            }
