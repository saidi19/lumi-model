"""Cooperative, stdlib-only byte reservations over transferred private FDs.

There is no path opening, admission, transport authentication, ML or GPU code.
The caller transfers its FD only after successful construction, must not share
or reuse it, and supplies one trusted, process-shared controller callback. On
failure the future worker MUST interrupt that controller and retire its ranks;
these handles only fail terminally and never refund or retry reservations.

This is logical-byte accounting, not physical I/O measurement, an OS quota,
cross-handle exclusivity, or protection against hostile concurrent FD reuse.
Metadata checks detect observed changes but are not an atomic filesystem lock.
Reopening the same inode under the same descriptor number can evade these
checks; the caller must retain exclusive physical FD ownership throughout.
Readers reserve planned bytes even if an operation subsequently fails. Writers
start with an empty file; log_bytes alone is reserved because the supervisor
already also debits private writes. No raw FD, path, payload or callback is
included in exception args or representations. Underlying exception contexts
can remain attached and MUST NOT be logged by the future worker. Explicit close
is required; no finalizer can accidentally close a later descriptor-number reuse.
"""

from __future__ import annotations

import fcntl
import os
import stat
import sys
import threading
from contextlib import contextmanager

MAX_OPERATION_BYTES = 1_048_576
MAX_FILE_BYTES = 2**63 - 1
READER_RESOURCES = frozenset({
    "checkpoint_read_bytes", "encoded_binary_read_bytes", "encoded_index_read_bytes",
})
WRITER_RESOURCES = frozenset({"private_write_bytes", "log_bytes"})
ERROR_CODES = frozenset({
    "invalid_arguments", "invalid_descriptor", "descriptor_changed", "owner_mismatch",
    "reentry", "closed", "terminal_failure", "reserve_failed", "read_failed", "short_read",
    "write_failed", "short_write", "write_cap_exceeded", "close_failed", "internal_error",
})


class MeteredIOError(RuntimeError):
    """Only an allowlisted fixed code, never an underlying exception message."""

    def __init__(self, code="internal_error"):
        self.code = code if type(code) is str and code in ERROR_CODES else "internal_error"
        super().__init__(self.code)


def _snapshot(fd):
    info = os.fstat(fd)
    return (
        info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_uid,
        info.st_size, info.st_mtime_ns, info.st_ctime_ns,
        fcntl.fcntl(fd, fcntl.F_GETFL), fcntl.fcntl(fd, fcntl.F_GETFD),
    )


def _valid_descriptor(value, access):
    return (
        stat.S_ISREG(value[2]) and stat.S_IMODE(value[2]) == 0o600
        and value[3] == 1 and value[4] == os.geteuid()
        and 0 <= value[5] <= MAX_FILE_BYTES
        and value[8] & os.O_ACCMODE == access
        and not value[8] & os.O_APPEND and value[9] & fcntl.FD_CLOEXEC
    )


def _writer_flags_match(before: int, after: int) -> bool:
    # Darwin's sys/fcntl.h: FWASWRITTEN marks a descriptor once written.
    # Only our completed write may newly set it; all other bits stay exact.
    written_flag = 0x00010000 if sys.platform == "darwin" else 0
    return after in (before, before | written_flag)


class _Metered:
    def _initialize(self, fd, rank, resource, reserve, access, resources, maximum=None):
        if hasattr(self, "_fd"):
            self._fail("invalid_arguments")
        if (type(fd) is not int or fd < 3 or type(rank) is not int or not 0 <= rank < 12
                or type(resource) is not str or resource not in resources or not callable(reserve)):
            raise MeteredIOError("invalid_arguments") from None
        if maximum is not None and (type(maximum) is not int or not 0 < maximum <= MAX_FILE_BYTES):
            raise MeteredIOError("invalid_arguments") from None
        try:
            identity = _snapshot(fd)
        except BaseException:
            raise MeteredIOError("invalid_descriptor") from None
        if not _valid_descriptor(identity, access) or (maximum is not None and identity[5] != 0):
            raise MeteredIOError("invalid_descriptor") from None
        self._rank, self._resource, self._reserve = rank, resource, reserve
        self._owner = (os.getpid(), threading.get_ident())
        self._identity, self._position, self._maximum = identity, 0, maximum
        self._failed, self._busy = False, False
        self._fd = fd  # Ownership transfers only after every constructor check succeeds.

    def __repr__(self):
        return "<MeteredIO private>"

    @property
    def closed(self):
        return self._fd is None

    def _fail(self, code):
        self._failed = True
        raise MeteredIOError(code) from None

    def _owner_check(self):
        if self._owner != (os.getpid(), threading.get_ident()):
            self._fail("owner_mismatch")
        if self._busy:
            self._fail("reentry")

    def _verify(self):
        try:
            current = _snapshot(self._fd)
        except BaseException:
            self._fail("descriptor_changed")
        if current != self._identity:
            self._fail("descriptor_changed")

    @contextmanager
    def _operation(self):
        self._owner_check()
        if self.closed:
            self._fail("closed")
        if self._failed:
            self._fail("terminal_failure")
        self._busy = True
        try:
            self._verify()
            yield
            if self._failed:
                self._fail("terminal_failure")
        except MeteredIOError:
            self._failed = True
            raise
        except BaseException:
            self._fail("internal_error")
        finally:
            self._busy = False

    def _reserve_before(self, amount):
        try:
            result = self._reserve(self._rank, **{self._resource: amount})
        except BaseException:
            self._fail("reserve_failed")
        if self._failed:
            self._fail("terminal_failure")
        if result is not None:
            self._fail("reserve_failed")
        self._verify()  # A trusted callback can still close/change/reuse the descriptor.

    def tell(self):
        with self._operation():
            return self._position

    def close(self):
        self._owner_check()
        if self.closed:
            return
        fd, self._fd = self._fd, None  # Never retry a failed close or reused descriptor.
        try:
            current = _snapshot(fd)
        except BaseException:
            self._fail("descriptor_changed")
        if current[:2] != self._identity[:2] or not stat.S_ISREG(current[2]):
            self._fail("descriptor_changed")
        try:
            os.close(fd)
        except BaseException:
            self._fail("close_failed")


class MeteredReader(_Metered):
    """Owned O_RDONLY private regular FD; logical cursor, bounded pread only."""

    def __init__(self, fd, *, rank, resource, reserve):
        self._initialize(fd, rank, resource, reserve, os.O_RDONLY, READER_RESOURCES)

    def _read(self, size):
        if type(size) is not int or not 0 <= size <= MAX_OPERATION_BYTES:
            self._fail("invalid_arguments")
        amount = min(size, self._identity[5] - self._position)
        if amount == 0:
            return b""
        self._reserve_before(amount)
        try:
            result = os.pread(self._fd, amount, self._position)
        except BaseException:
            self._fail("read_failed")
        if type(result) is not bytes or len(result) != amount:
            self._fail("short_read")
        self._verify()
        self._position += amount
        return result

    def read(self, size):
        with self._operation():
            return self._read(size)

    def readinto(self, buffer):
        with self._operation():
            if type(buffer) is not bytearray or len(buffer) > MAX_OPERATION_BYTES:
                self._fail("invalid_arguments")
            original_length = len(buffer)
            result = self._read(original_length)
            if len(buffer) != original_length:
                self._fail("invalid_arguments")
            buffer[:len(result)] = result
            return len(result)

    def seek(self, offset, whence=os.SEEK_SET):
        with self._operation():
            if type(offset) is not int or type(whence) is not int:
                self._fail("invalid_arguments")
            bases = {os.SEEK_SET: 0, os.SEEK_CUR: self._position, os.SEEK_END: self._identity[5]}
            if whence not in bases:
                self._fail("invalid_arguments")
            position = bases[whence] + offset
            if not 0 <= position <= self._identity[5]:
                self._fail("invalid_arguments")
            self._position = position
            return position


class MeteredWriter(_Metered):
    """Owned empty O_WRONLY private FD; append-only logical pwrite cursor."""

    def __init__(self, fd, *, rank, resource, reserve, maximum_file_bytes):
        if hasattr(self, "_fd"):
            self._fail("invalid_arguments")
        if type(maximum_file_bytes) is not int or not 0 < maximum_file_bytes <= MAX_FILE_BYTES:
            raise MeteredIOError("invalid_arguments") from None
        self._initialize(fd, rank, resource, reserve, os.O_WRONLY,
                         WRITER_RESOURCES, maximum_file_bytes)

    def write(self, data):
        with self._operation():
            if type(data) is not bytes or len(data) > MAX_OPERATION_BYTES:
                self._fail("invalid_arguments")
            amount = len(data)
            if self._position + amount > self._maximum:
                self._fail("write_cap_exceeded")
            if amount == 0:
                return 0
            self._reserve_before(amount)
            before = self._identity
            try:
                written = os.pwrite(self._fd, data, self._position)
            except BaseException:
                self._fail("write_failed")
            if type(written) is not int or written != amount:
                self._fail("short_write")
            try:
                after = _snapshot(self._fd)
            except BaseException:
                self._fail("descriptor_changed")
            if (after[:5] != before[:5] or after[9] != before[9]
                    or not _writer_flags_match(before[8], after[8])
                    or after[5] != before[5] + amount):
                self._fail("descriptor_changed")
            self._identity = after  # Only our successful append may change size/timestamps.
            self._position += amount
            return amount
