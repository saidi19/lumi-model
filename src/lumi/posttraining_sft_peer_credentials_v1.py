"""Bounded Linux per-message credentials for opaque inert-probe frames only.

SO_PASSCRED/SCM_CREDENTIALS checks the kernel-supplied sender (PID, UID, GID)
tuple on EVERY AF_UNIX SOCK_SEQPACKET message. SO_PEERCRED is deliberately not
used: a socketpair's creator credentials do not identify an inherited sender.
The caller must bind the expected tuple using a separately trusted lifecycle.
This is NOT process-generation/pidfd binding. Privileged CAP_SYS_ADMIN,
CAP_SETUID or CAP_SETGID senders can override applicable credential fields;
this component does not establish absence of those privileges.

This single-owner transport neither decodes protocol payloads nor mutates a
ledger, admits workers or authorizes execution. It provides no cgroup/resource
or network containment, private-data/model access or GPU/inference permission.
Counters describe bounded receive capacity, not total kernel/host resource use.
Only the validated frame is returned; summaries/errors contain no credentials,
descriptors, payloads, socket paths or underlying exception text.
"""

from __future__ import annotations

import math
import os
import socket
import struct
import sys
import time

MAXIMUM_PAYLOAD_BYTES = 4096
MAXIMUM_FRAME_BYTES = MAXIMUM_PAYLOAD_BYTES + 4
RECEIVE_DATA_CAPACITY = MAXIMUM_FRAME_BYTES + 1
UCRED_BYTES = 12
DESCRIPTOR_BYTES = 4
ANCILLARY_BUFFER_BYTES = (
    socket.CMSG_SPACE(UCRED_BYTES) + socket.CMSG_SPACE(32)
    if hasattr(socket, "CMSG_SPACE")
    else 0
)
RECEIVE_CAPACITY_BYTES = RECEIVE_DATA_CAPACITY + ANCILLARY_BUFFER_BYTES
DEFAULT_MAXIMUM_MESSAGES = 4
DEFAULT_MAXIMUM_READ_BYTES = DEFAULT_MAXIMUM_MESSAGES * RECEIVE_CAPACITY_BYTES
HARD_MAXIMUM_MESSAGES = 64
HARD_MAXIMUM_READ_BYTES = HARD_MAXIMUM_MESSAGES * RECEIVE_CAPACITY_BYTES
ERROR_CODES = frozenset(
    (
        "linux_credentials_unavailable",
        "credential_configuration_rejected",
        "credential_socket_rejected",
        "credential_pair_failed",
        "credential_receive_failed",
        "credential_ancillary_rejected",
        "credential_mismatch",
        "credential_frame_rejected",
        "credential_deadline_exhausted",
        "credential_budget_exhausted",
        "credential_channel_closed",
        "credential_channel_rejected",
        "credential_cleanup_failed",
    )
)


class PeerCredentialError(ValueError):
    """An allowlisted opaque code, never caller/OS exception details."""

    def __init__(self, code: str):
        if type(code) is not str or code not in ERROR_CODES:
            code = "credential_channel_rejected"
        self.code = code
        super().__init__(code)


def _need(condition: bool, code: str) -> None:
    if not condition:
        raise PeerCredentialError(code) from None


def require_linux() -> None:
    """Reject unsupported platforms/ABIs before creating or operating sockets."""
    _need(
        sys.platform == "linux"
        and all(
            hasattr(socket, name)
            for name in (
                "AF_UNIX",
                "SOCK_SEQPACKET",
                "SO_PASSCRED",
                "SCM_CREDENTIALS",
                "SCM_RIGHTS",
                "MSG_CMSG_CLOEXEC",
                "MSG_TRUNC",
                "MSG_CTRUNC",
                "CMSG_SPACE",
            )
        )
        and hasattr(socket.socket, "recvmsg")
        and struct.calcsize("iII") == UCRED_BYTES
        and struct.calcsize("i") == DESCRIPTOR_BYTES
        and ANCILLARY_BUFFER_BYTES > 0,
        "linux_credentials_unavailable",
    )


def _socket_ready(sock: socket.socket) -> None:
    _need(
        sock.family == socket.AF_UNIX
        and sock.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) == socket.SOCK_SEQPACKET
        and sock.getsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED) == 1
        and sock.get_inheritable() is False,
        "credential_socket_rejected",
    )


def credential_pair() -> tuple[socket.socket, socket.socket]:
    """Create private packet endpoints; both receive credentials before any send.

    Ownership transfers to the caller only on success. Any setup failure closes
    both endpoints, attempting the second close even if the first close fails.
    """
    require_linux()
    endpoints = ()
    failure = ""
    try:
        endpoints = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        for endpoint in endpoints:
            endpoint.set_inheritable(False)
            endpoint.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        for endpoint in endpoints:
            _socket_ready(endpoint)
    except BaseException:
        failure = "credential_pair_failed"
    if failure:
        for endpoint in endpoints:
            try:
                endpoint.close()
            except BaseException:
                failure = "credential_cleanup_failed"
        raise PeerCredentialError(failure) from None
    return endpoints


def _close_received_rights(ancillary: list) -> tuple[int, bool]:
    """Close every delivered rights integer, including on truncated/rejected data.

    Linux closes excess SCM_RIGHTS descriptors that did not fit recvmsg's control
    buffer; every descriptor delivered to Python remains our responsibility.
    Process all recognizable rights entries before validating ANY packet data.
    A malformed entry or failed close never skips cleanup of later entries.
    """
    closed = 0
    failed = False
    seen = set()
    for entry in ancillary:
        if type(entry) not in (tuple, list) or len(entry) != 3:
            failed = True
            continue
        level, kind, data = entry
        if level != socket.SOL_SOCKET or kind != socket.SCM_RIGHTS:
            continue
        if type(data) is not bytes:
            failed = True
            continue
        if len(data) % DESCRIPTOR_BYTES:
            failed = True
        for offset in range(0, len(data) - len(data) % DESCRIPTOR_BYTES, DESCRIPTOR_BYTES):
            descriptor = struct.unpack_from("i", data, offset)[0]
            if descriptor in seen:
                failed = True
                continue
            seen.add(descriptor)
            try:
                os.close(descriptor)
            except BaseException:
                failed = True
            else:
                closed += 1
    return closed, failed


class CredentialChannel:
    """Own one socket after successful construction; a receive error is terminal.

    Each actual receive attempt prepays full data-plus-ancillary buffer capacity
    without refunds, including syscall errors, EOF and malformed packets. No
    stream accumulation, application retry, deadline reset or protocol decode is
    performed. Caller retains ownership if construction fails. Use from one
    trusted owner/thread only; arbitrary in-process mutation is out of scope.
    """

    def __init__(
        self,
        sock: socket.socket,
        *,
        expected_credentials: tuple[int, int, int],
        deadline: float,
        maximum_messages: int = DEFAULT_MAXIMUM_MESSAGES,
        maximum_read_bytes: int = DEFAULT_MAXIMUM_READ_BYTES,
    ):
        require_linux()
        _need(
            type(expected_credentials) is tuple
            and len(expected_credentials) == 3
            and all(type(value) is int for value in expected_credentials)
            and 1 <= expected_credentials[0] <= 2**31 - 1
            and all(0 <= value <= 2**32 - 1 for value in expected_credentials[1:])
            and type(deadline) in (int, float)
            and 0 < deadline <= 2**63 - 1
            and math.isfinite(deadline)
            and type(maximum_messages) is int
            and 1 <= maximum_messages <= HARD_MAXIMUM_MESSAGES
            and type(maximum_read_bytes) is int
            and 1 <= maximum_read_bytes <= HARD_MAXIMUM_READ_BYTES,
            "credential_configuration_rejected",
        )
        failure = ""
        try:
            _socket_ready(sock)
        except BaseException:
            failure = "credential_socket_rejected"
        if failure:
            raise PeerCredentialError(failure) from None
        self._sock = sock
        self._expected_credentials = expected_credentials
        self._deadline = deadline
        self._maximum_messages = maximum_messages
        self._maximum_read_bytes = maximum_read_bytes
        self._verified_messages = 0
        self._receive_attempts = 0
        self._payload_bytes = 0
        self._read_capacity_bytes = 0
        self._received_descriptors_closed = 0
        self._rejected = False
        self._closed = False
        self._close_attempted = False

    def _remaining(self) -> float:
        remaining = self._deadline - time.monotonic()
        _need(math.isfinite(remaining) and remaining > 0, "credential_deadline_exhausted")
        return remaining

    def _receive(self) -> bytes:
        require_linux()
        self._remaining()
        _need(
            self._receive_attempts < self._maximum_messages
            and self._read_capacity_bytes + RECEIVE_CAPACITY_BYTES <= self._maximum_read_bytes,
            "credential_budget_exhausted",
        )
        _socket_ready(self._sock)
        self._receive_attempts += 1
        self._read_capacity_bytes += RECEIVE_CAPACITY_BYTES
        self._sock.settimeout(self._remaining())
        frame, ancillary, flags, _address = self._sock.recvmsg(
            RECEIVE_DATA_CAPACITY, ANCILLARY_BUFFER_BYTES, socket.MSG_CMSG_CLOEXEC
        )
        # Cleanup MUST precede deadline, flags, frame and credential validation.
        closed, cleanup_failed = _close_received_rights(ancillary)
        self._received_descriptors_closed += closed
        _need(not cleanup_failed, "credential_cleanup_failed")
        self._remaining()
        _need(
            self._receive_attempts <= self._maximum_messages
            and self._read_capacity_bytes <= self._maximum_read_bytes,
            "credential_budget_exhausted",
        )
        _need(
            type(flags) is int and flags & ~socket.MSG_CMSG_CLOEXEC == 0,
            "credential_ancillary_rejected",
        )
        _need(type(ancillary) is list and len(ancillary) == 1, "credential_ancillary_rejected")
        entry = ancillary[0]
        _need(type(entry) is tuple and len(entry) == 3, "credential_ancillary_rejected")
        level, kind, data = entry
        _need(
            level == socket.SOL_SOCKET
            and kind == socket.SCM_CREDENTIALS
            and type(data) is bytes
            and len(data) == UCRED_BYTES,
            "credential_ancillary_rejected",
        )
        _need(struct.unpack("iII", data) == self._expected_credentials, "credential_mismatch")
        _need(
            type(frame) is bytes
            and 5 <= len(frame) <= MAXIMUM_FRAME_BYTES
            and int.from_bytes(frame[:4], "big") == len(frame) - 4,
            "credential_frame_rejected",
        )
        self._remaining()
        self._verified_messages += 1
        self._payload_bytes += len(frame) - 4
        return frame

    def receive(self) -> bytes:
        """Return exactly one credential-checked opaque length-prefixed frame."""
        if self._rejected:
            raise PeerCredentialError("credential_channel_rejected") from None
        if self._closed or self._close_attempted:
            self._rejected = True
            raise PeerCredentialError("credential_channel_closed") from None
        failure = ""
        try:
            return self._receive()
        except PeerCredentialError as error:
            failure = error.code
        except TimeoutError:
            failure = "credential_deadline_exhausted"
        except BaseException:
            failure = "credential_receive_failed"
        self._rejected = True
        raise PeerCredentialError(failure) from None

    def close(self) -> None:
        """Close once. A failed close is sticky and is never retried (FD reuse)."""
        if self._close_attempted:
            if not self._closed:
                raise PeerCredentialError("credential_cleanup_failed") from None
            return
        self._close_attempted = True
        failed = False
        try:
            self._sock.close()
        except BaseException:
            failed = True
        if failed:
            self._rejected = True
            raise PeerCredentialError("credential_cleanup_failed") from None
        self._closed = True

    def summary(self) -> dict[str, int | bool]:
        """Aggregate counters only; no payload or sender-identity disclosure."""
        return {
            "verified_messages": self._verified_messages,
            "receive_attempts": self._receive_attempts,
            "payload_bytes": self._payload_bytes,
            "read_capacity_bytes": self._read_capacity_bytes,
            "received_descriptors_closed": self._received_descriptors_closed,
            "rejected": self._rejected,
            "closed": self._closed,
        }
