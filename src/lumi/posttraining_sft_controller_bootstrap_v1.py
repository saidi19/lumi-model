"""Validated, in-memory bootstrap handoff; no issuer, spawn, account or file I/O.

The anonymous pipe is filled with ONE nonblocking write before it is exposed.
Short writes and insufficient capacity reject without starting any child. The
32 KiB schema ceiling is not a promise that every host's pipe can hold that much.
Key/pin consistency is not issuer authentication. Python references and pipe
closure do not prove encryption, secure erasure, crash recovery or permission.
"""

from __future__ import annotations

import fcntl
import os
import stat
import threading

from . import posttraining_sft_controller_process_v1 as process
from . import posttraining_sft_dispatch_authority_v1 as authority
from . import posttraining_sft_lambda_http_v1 as http


class BootstrapError(ValueError):
    def __init__(self, _value=None):
        super().__init__("bootstrap_handoff_rejected")


def need(condition):
    if not condition:
        raise BootstrapError from None


def encode_bootstrap(plan_bytes, plan_sha256, *, authority_bytes, authority_sha256,
                     authority_key, authority_key_sha256, launch_body, token):
    """Pure validation/encoding. The returned bytes ARE SECRET; never log them.

    All identity, key custody and approval provenance must be supplied externally.
    No key/MAC issuance, current-time check or single-attempt claim happens here.
    The child repeats the frozen checks, including its fresh creation window.
    """
    failed = False
    result = None
    value = plan = None
    try:
        plan = authority.verify_authority(
            authority_bytes, authority_sha256, key=authority_key,
            expected_key_sha256=authority_key_sha256, plan_bytes=plan_bytes,
            plan_sha256=plan_sha256, token=token,
        )
        http.validate_launch(launch_body, plan["launch_request_sha256"],
                             plan["region"], plan["instance_type"])
        value = {
            "schema_version": process.BOOTSTRAP_SCHEMA,
            "plan_ascii": plan_bytes.decode("ascii"), "plan_sha256": plan_sha256,
            "authority_ascii": authority_bytes.decode("ascii"),
            "authority_sha256": authority_sha256,
            "authority_key_hex": authority_key.hex(),
            "authority_key_sha256": authority_key_sha256,
            "launch_ascii": launch_body.decode("ascii"), "token_ascii": token.decode("ascii"),
        }
        result = process.canonical(value)
        need(len(result) <= process.MAXIMUM_BOOTSTRAP_BYTES)
        process.decode_bootstrap(result)
    except BaseException:
        failed = True
        result = None
    finally:
        # Drop our references, without claiming cryptographic memory erasure.
        plan_bytes = authority_bytes = authority_key = launch_body = token = None
        value = plan = None
    if failed:
        raise BootstrapError from None
    return result


def _close_once(fd):
    try:
        os.close(fd)
        return True
    except BaseException:
        return False


def _descriptor(fd, access):
    need(type(fd) is int and 3 <= fd < 2**20)
    info = os.fstat(fd)
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    need(stat.S_ISFIFO(info.st_mode) and flags & os.O_ACCMODE == access
         and flags & os.O_NONBLOCK and not os.get_inheritable(fd))
    return info.st_dev, info.st_ino


def _prefill(raw):
    """Return an owned readonly pipe and identity only after full write and EOF."""
    owned = []
    result = None
    failed = False
    try:
        need(os.name == "posix" and type(raw) is bytes
             and 0 < len(raw) <= process.MAXIMUM_BOOTSTRAP_BYTES)
        reader, writer = os.pipe()
        owned.extend((reader, writer))
        need(reader != writer)
        for fd in owned:
            os.set_inheritable(fd, False)
            os.set_blocking(fd, False)
        identity = _descriptor(reader, os.O_RDONLY)
        # os.pipe establishes pairing. Darwin assigns different identities to
        # its two endpoints; only compare the reader with its OWN saved identity.
        _descriptor(writer, os.O_WRONLY)
        written = os.write(writer, raw)
        need(type(written) is int and written == len(raw))
        # Remove BEFORE closing: a failed close is never retried by fd number.
        owned.remove(writer)
        need(_close_once(writer))
        need(_descriptor(reader, os.O_RDONLY) == identity)
        result = reader, identity
        owned.remove(reader)
    except BaseException:
        failed = True
    finally:
        raw = None
        for fd in owned:
            _close_once(fd)
    if failed:
        raise BootstrapError from None
    return result


class BootstrapPipe:
    """Single-owner handoff of one prefilled nonblocking, CLOEXEC read descriptor.

    fileno() BORROWS the descriptor for the frozen spawn_controller call; close()
    then releases the parent's copy. Alternatively take_fd() transfers ownership
    exactly once, and the caller must close it. Neither method starts a child.
    A borrowed fd must not be closed, modified, duplicated or read by other code.
    This cooperative handle is not a sandbox or global attempt-exclusivity token.
    Explicit close is required; no destructor makes delayed fd-reuse decisions.
    """

    __slots__ = ("_pid", "_thread", "_fd", "_identity", "_busy", "_close_ok")

    def __init__(self, plan_bytes, plan_sha256, *, authority_bytes, authority_sha256,
                 authority_key, authority_key_sha256, launch_body, token):
        need(not hasattr(self, "_pid"))
        self._pid = os.getpid()
        self._thread = threading.current_thread()
        self._fd = self._identity = None
        self._busy = True
        self._close_ok = True
        raw = None
        try:
            raw = encode_bootstrap(
                plan_bytes, plan_sha256, authority_bytes=authority_bytes,
                authority_sha256=authority_sha256, authority_key=authority_key,
                authority_key_sha256=authority_key_sha256, launch_body=launch_body,
                token=token,
            )
            self._fd, self._identity = _prefill(raw)
        finally:
            plan_bytes = authority_bytes = authority_key = launch_body = token = raw = None
            self._busy = False

    def __repr__(self):
        return "<BootstrapPipe: redacted>"

    def _owner(self):
        return (os.getpid() == self._pid and threading.current_thread() is self._thread
                and not self._busy)

    def _checked_fd(self):
        need(self._fd is not None)
        failed = False
        try:
            need(_descriptor(self._fd, os.O_RDONLY) == self._identity)
        except BaseException:
            failed = True
        if failed:
            # Ownership has become ambiguous; never close a possibly reused fd.
            self._fd = None
            self._close_ok = False
            raise BootstrapError from None
        return self._fd

    def fileno(self):
        need(self._owner())
        self._busy = True
        try:
            return self._checked_fd()
        finally:
            self._busy = False

    def take_fd(self):
        need(self._owner())
        self._busy = True
        try:
            fd = self._checked_fd()
            self._fd = None
            return fd
        finally:
            self._busy = False

    def close(self):
        """Close only our owned copy, once; True is NOT a child/cloud outcome."""
        if not self._owner():
            return False
        self._busy = True
        try:
            if self._fd is None:
                return self._close_ok
            try:
                fd = self._checked_fd()
            except BaseException:
                return False
            self._fd = None
            self._close_ok = _close_once(fd)
            return self._close_ok
        finally:
            self._busy = False
