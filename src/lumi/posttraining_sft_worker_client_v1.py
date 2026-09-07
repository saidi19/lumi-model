"""Bounded worker-side epoch ledger client over one already-private local socket.

The admitted caller supplies a connected AF_UNIX stream capability and fresh,
parent-bound epoch/rank/token. Successful construction transfers socket ownership.
The caller must guarantee exclusive socket/FD ownership and durable epoch/token
uniqueness. This client neither creates a Supervisor nor authenticates OS peers,
owner approval, model/data access, GPU dispatch, or any execution permission.

Each synchronous operation sends one request, validates its exact canonical ACK,
then fully sends its hash-bound confirmation. Frozen epoch ACK confirmation has
no response: successful return alone does NOT prove parent confirmation receipt.
After step_complete, only wait_step_commit or cleanup is allowed. The NEW control
fence is outside the frozen epoch request protocol. The trusted parent sends it
only after processing every rank confirmation, finding no pending ACKs, and
confirming the shared supervisor completed that exact global step. A private
channel authenticates the capability, not parent honesty or a remote OS PID.

Whole request/ACK/confirmation data capacity is prepaid; a bounded ancillary
allowance is also prepaid before each recvmsg, including fragments, never refunded.
All delivered rights are
closed and rejected; this is not a descriptor-transfer or credential protocol.
Short-I/O continuation is bounded by one non-resetting exchange deadline inside
the non-resetting absolute session deadline; exceptions are never retried. A
failure permanently consumes the client and interrupts once. Cleanup closes
only its owned socket and cannot imply epoch completion or successful retirement.
Only fixed errors/aggregate summaries are safe; never log suppressed traceback
contexts, capabilities, request bodies, or raw socket/exception representations.
"""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import os
import socket
import stat
import threading
import time
from contextlib import contextmanager

from . import posttraining_sft_epoch_protocol_v1 as protocol
from .posttraining_sft_peer_credentials_v1 import _close_received_rights

MAXIMUM_INTEGER = 2**63 - 1
ANCILLARY_BYTES = socket.CMSG_SPACE(16)
STEP_FENCE_SCHEMA = "lumi.sft-worker-step-fence.v0.1"
ERROR_CODES = frozenset({
    "invalid_arguments", "invalid_socket", "descriptor_changed", "owner_mismatch",
    "reentry", "terminal_failure", "closed", "clock_rejected", "deadline_exhausted",
    "wire_cap_exceeded", "frame_rejected", "ack_rejected", "step_fence_rejected",
    "step_fence_pending", "io_failed", "interrupted", "close_failed",
})


class WorkerClientError(RuntimeError):
    def __init__(self, code="io_failed"):
        self.code = code if type(code) is str and code in ERROR_CODES else "io_failed"
        super().__init__(self.code)


def _integer(value, *, zero=False):
    return type(value) is int and (0 if zero else 1) <= value <= MAXIMUM_INTEGER


def _socket_identity(sock, *, require_connected=False):
    if type(sock) is not socket.socket or sock.fileno() < 3:
        raise WorkerClientError("invalid_socket") from None
    try:
        fd = sock.fileno()
        info = os.fstat(fd)
        flags = fcntl.fcntl(fd, fcntl.F_GETFD)
        if (sock.family != socket.AF_UNIX or sock.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE)
                != socket.SOCK_STREAM or not stat.S_ISSOCK(info.st_mode)
                or info.st_uid != os.geteuid() or not flags & fcntl.FD_CLOEXEC):
            raise ValueError
        if require_connected:
            sock.getpeername()  # Connected capability; no peer-PID authentication claim.
        # Darwin may clear anonymous socket permission bits after peer EOF. Those
        # bits do not authenticate this inherited capability; socket type does.
        return fd, info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode), info.st_uid, flags
    except BaseException:
        raise WorkerClientError("invalid_socket") from None


class WorkerLedgerClient:
    """One rank/epoch channel, with caller-owned parent/controller enforcement."""

    def __init__(self, sock, *, epoch, rank, token, interrupt, maximum_wire_bytes,
                 deadline_ns, maximum_exchange_ns, clock_ns=time.monotonic_ns):
        if hasattr(self, "_owner"):
            self._fail("invalid_arguments")
        if (not protocol._token(epoch) or not protocol._token(token) or epoch == token
                or type(rank) is not int or not 0 <= rank < 12 or not callable(interrupt)
                or not callable(clock_ns)
                or not all(_integer(value) for value in
                           (maximum_wire_bytes, deadline_ns, maximum_exchange_ns))):
            raise WorkerClientError("invalid_arguments") from None
        identity = _socket_identity(sock, require_connected=True)
        try:
            now = clock_ns()
        except BaseException:
            raise WorkerClientError("clock_rejected") from None
        if not _integer(now, zero=True):
            raise WorkerClientError("clock_rejected") from None
        if now >= deadline_ns:
            raise WorkerClientError("deadline_exhausted") from None
        self._owner = os.getpid(), threading.get_ident()
        self._epoch, self._rank, self._token = epoch, rank, token
        self._interrupt, self._clock = interrupt, clock_ns
        self._maximum_wire = maximum_wire_bytes
        self._deadline, self._maximum_exchange = deadline_ns, maximum_exchange_ns
        self._identity, self._last_clock = identity, now
        self._failure, self._busy, self._interrupted = None, False, False
        self._retired = False
        self._next_sequence, self._pending_step = 1, None
        self._reserved_wire = self._sent_wire = self._received_wire = 0
        self._requests = self._acks = self._confirmations = self._fences = 0
        self._received_rights_closed = 0
        self._sock = sock  # Transfer only after every constructor validation.

    def __repr__(self):
        return "<WorkerLedgerClient private>"

    @property
    def closed(self):
        return self._sock is None

    def _fail(self, code):
        if self._retired:
            # Late callbacks on a cleaned-up epoch cannot interrupt a newer one.
            raise WorkerClientError(code) from None
        if self._failure is None:
            self._failure = code if code in ERROR_CODES else "io_failed"
        if not self._interrupted:
            self._interrupted = True
            try:
                self._interrupt()
            except BaseException:
                pass
        raise WorkerClientError(code) from None

    def _owner_check(self):
        if self._owner != (os.getpid(), threading.get_ident()):
            self._fail("owner_mismatch")
        if self._busy:
            self._fail("reentry")

    def _check_socket(self):
        if self._failure is not None:
            self._fail("terminal_failure")
        try:
            identity = _socket_identity(self._sock)
        except BaseException:
            self._fail("descriptor_changed")
        if identity != self._identity:
            self._fail("descriptor_changed")

    def _now(self):
        try:
            now = self._clock()
        except BaseException:
            self._fail("clock_rejected")
        if not _integer(now, zero=True) or now < self._last_clock:
            self._fail("clock_rejected")
        self._last_clock = now
        if now >= self._deadline:
            self._fail("deadline_exhausted")
        if self._failure is not None:  # A callback cannot swallow a reentrant failure.
            self._fail("terminal_failure")
        return now

    @contextmanager
    def _operation(self):
        self._owner_check()
        if self.closed:
            self._fail("closed")
        if self._failure is not None:
            self._fail("terminal_failure")
        self._busy = True
        try:
            self._check_socket()
            end = min(self._deadline, self._now() + self._maximum_exchange)
            yield end
            self._check_socket()
            if self._now() >= end:
                self._fail("deadline_exhausted")
        except WorkerClientError:
            raise
        except BaseException:
            self._fail("io_failed")
        finally:
            self._busy = False

    def _prepay(self, count):
        if self._reserved_wire + count > self._maximum_wire:
            self._fail("wire_cap_exceeded")
        self._reserved_wire += count

    def _timeout(self, end):
        self._check_socket()
        remaining = end - self._now()
        if remaining <= 0:
            self._fail("deadline_exhausted")
        self._check_socket()  # The trusted clock must not mutate descriptor ownership.
        try:
            self._sock.settimeout(remaining / 1_000_000_000)
        except BaseException:
            self._fail("io_failed")

    def _send(self, frame, end):
        position = 0
        while position < len(frame):
            self._timeout(end)
            try:
                written = self._sock.send(frame[position:])
            except TimeoutError:
                self._fail("deadline_exhausted")
            except BaseException:
                self._fail("io_failed")
            if type(written) is not int or not 0 < written <= len(frame) - position:
                self._fail("io_failed")
            position += written
            self._sent_wire += written
        self._check_socket()

    def _receive_exact_frame(self, expected, end, code):
        raw = bytearray()
        target = 4
        while len(raw) < len(expected):
            self._timeout(end)
            self._prepay(ANCILLARY_BYTES)
            try:
                block, ancillary, flags, _address = self._sock.recvmsg(
                    target - len(raw), ANCILLARY_BYTES, getattr(socket, "MSG_CMSG_CLOEXEC", 0))
            except TimeoutError:
                self._fail("deadline_exhausted")
            except BaseException:
                self._fail("io_failed")
            # Cleanup precedes every other validation, including malformed/truncated
            # frames. Extra kernel-truncated rights are not made available to Python.
            closed, cleanup_failed = _close_received_rights(ancillary)
            self._received_rights_closed += closed
            if cleanup_failed:
                self._fail("close_failed")
            if type(block) is not bytes or not block or len(block) > target - len(raw):
                self._fail("frame_rejected")
            self._received_wire += len(block)
            if ancillary or flags & ~getattr(socket, "MSG_CMSG_CLOEXEC", 0):
                self._fail("frame_rejected")
            raw.extend(block)
            if len(raw) == 4 and target == 4:
                size = int.from_bytes(raw, "big")
                if not 0 < size <= protocol.MAXIMUM_FRAME_BYTES or size + 4 != len(expected):
                    self._fail("frame_rejected")
                target = len(expected)
        self._check_socket()
        if not hmac.compare_digest(bytes(raw), expected):
            self._fail(code)
        return bytes(raw)

    def _exchange(self, operation, arguments, end):
        if self._pending_step is not None:
            self._fail("step_fence_pending")
        if not _integer(self._next_sequence):
            self._fail("invalid_arguments")
        request = {"schema_version": protocol.REQUEST_SCHEMA, "epoch": self._epoch,
                   "rank": self._rank, "sequence": self._next_sequence,
                   "token": self._token, "operation": operation, "arguments": arguments}
        ack = protocol.encode_frame({
            "schema_version": protocol.ACK_SCHEMA, "epoch": self._epoch,
            "rank": self._rank, "sequence": self._next_sequence,
            "operation": operation, "status": "reserved",
        })
        confirmation = protocol.encode_frame({
            **request, "operation": "confirm_ack",
            "arguments": {"ack_sha256": hashlib.sha256(ack).hexdigest()},
        })
        frame = protocol.encode_frame(request)
        self._prepay(len(frame) + len(ack) + len(confirmation))
        self._next_sequence += 1  # This transaction can never be resent/replayed.
        self._send(frame, end)
        self._requests += 1
        self._receive_exact_frame(ack, end, "ack_rejected")
        self._acks += 1
        self._send(confirmation, end)
        self._confirmations += 1

    def reserve(self, rank, **resources):
        with self._operation() as end:
            if (type(rank) is not int or rank != self._rank or not resources
                    or not set(resources) <= protocol.RESOURCE_FIELDS
                    or not all(_integer(value, zero=True) for value in resources.values())
                    or not any(value > 0 for value in resources.values())):
                self._fail("invalid_arguments")
            self._exchange("reserve", resources, end)

    def reserve_step(self, from_step, to_step):
        with self._operation() as end:
            if (self._rank != 0 or not _integer(from_step, zero=True)
                    or not _integer(to_step) or to_step != from_step + 1):
                self._fail("invalid_arguments")
            self._exchange("reserve_step", {"from_step": from_step, "to_step": to_step}, end)

    def step_complete(self, to_step):
        with self._operation() as end:
            if not _integer(to_step):
                self._fail("invalid_arguments")
            self._exchange("step_complete", {"to_step": to_step}, end)
            self._pending_step = to_step

    def wait_step_commit(self, to_step):
        with self._operation() as end:
            if not _integer(to_step) or to_step != self._pending_step:
                self._fail("step_fence_rejected")
            expected = protocol.encode_frame({
                "schema_version": STEP_FENCE_SCHEMA, "epoch": self._epoch,
                "rank": self._rank, "to_step": to_step, "status": "committed",
            })
            self._prepay(len(expected))
            self._receive_exact_frame(expected, end, "step_fence_rejected")
            self._fences += 1
            self._pending_step = None

    def summary(self):
        self._owner_check()
        return {"state": "failed" if self._failure else "closed" if self.closed else "ready",
                "failure_code": self._failure, "wire_bytes_reserved": self._reserved_wire,
                "wire_bytes_sent": self._sent_wire, "wire_bytes_received": self._received_wire,
                "requests_sent": self._requests, "acknowledgements_validated": self._acks,
                "confirmations_sent": self._confirmations, "step_fences_validated": self._fences,
                "received_descriptors_closed": self._received_rights_closed,
                "step_fence_pending": self._pending_step is not None, "closed": self.closed,
                "os_peer_pid_authenticated": False, "os_resource_enforcement": False,
                "real_sft_execution_permitted": False, "production_eligible": False}

    def close(self):
        self._owner_check()
        if self.closed:
            return
        sock, self._sock = self._sock, None  # Revoke before an ambiguous close, never retry.
        self._token = self._epoch = None
        try:
            info = os.fstat(sock.fileno())
            same = ((info.st_dev, info.st_ino) == self._identity[1:3]
                    and stat.S_ISSOCK(info.st_mode))
        except BaseException:
            same = False
        if not same:
            try:
                sock.detach()  # Never close a different reused descriptor/inode.
            except BaseException:
                pass
            self._fail("descriptor_changed")
        try:
            sock.close()
        except BaseException:
            self._fail("close_failed")
        self._retired = True
