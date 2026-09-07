"""Bounded, independent serializer views over one verified metered session.

This module imports no ML library and exposes no descriptor or path interface.
Every content read, including repeated/overlapping reads, passes through the
frozen session. Views own only their cursors; closing one never closes the
session or sibling views. The caller must explicitly close the session.

The call cap bounds logical returned/readinto bytes, not serializer tensor
allocations or total process memory. Readline may read ahead by one bounded
chunk; those bytes are charged and are read/charged again on the next call.
Failures poison the shared session, are nonrefundable and cannot be retried.
Only fixed error args/repr are safe to log, never suppressed exception contexts.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager

from .posttraining_sft_checkpoint_reader_v1 import MANIFEST, CheckpointReadSession
from .posttraining_sft_metered_io_v1 import MAX_FILE_BYTES, MAX_OPERATION_BYTES

ERROR_CODES = frozenset({
    "invalid_arguments", "invalid_session", "not_verified", "owner_mismatch",
    "reentry", "closed", "terminal_failure", "read_cap_exceeded", "read_failed",
    "internal_error",
})


class CheckpointViewError(RuntimeError):
    def __init__(self, code="internal_error"):
        self.code = code if type(code) is str and code in ERROR_CODES else "internal_error"
        super().__init__(self.code)


class CheckpointReadView:
    """File-like bounded slice; no fileno/raw/buffer/name or session ownership."""

    def __init__(self, session, relative, *, offset=0, length=None, maximum_read_call_bytes):
        if hasattr(self, "_session"):
            self._fail("invalid_arguments")
        self._session = session if type(session) is CheckpointReadSession else None
        self._owner = os.getpid(), threading.get_ident()
        self._busy, self._failed, self._closed = False, False, False
        if self._session is None:
            self._fail("invalid_session")
        try:
            self._session_check()
            if not session._verified:
                self._fail("not_verified")
            if (type(relative) is not str or relative == MANIFEST
                    or relative not in session._files or type(offset) is not int
                    or type(maximum_read_call_bytes) is not int
                    or not 0 < maximum_read_call_bytes <= MAX_FILE_BYTES):
                self._fail("invalid_arguments")
            extent = session._files[relative][5]
            if not 0 <= offset <= extent:
                self._fail("invalid_arguments")
            if length is None:
                length = extent - offset
            if type(length) is not int or not 0 <= length <= extent - offset:
                self._fail("invalid_arguments")
            self._relative, self._start, self._length = relative, offset, length
            self._maximum_call, self._position = maximum_read_call_bytes, 0
            self._session_check()
        except CheckpointViewError:
            raise
        except BaseException:
            self._fail("read_failed")

    def __repr__(self):
        return "<CheckpointReadView private>"

    @property
    def closed(self):
        return self._closed

    def _fail(self, code):
        self._failed = True
        if self._session is not None:
            self._session._failed = True
        raise CheckpointViewError(code) from None

    def _owner_check(self):
        if self._owner != (os.getpid(), threading.get_ident()):
            self._fail("owner_mismatch")
        if self._busy:
            self._fail("reentry")

    def _session_check(self):
        if self._failed or self._session._failed:
            self._fail("terminal_failure")
        if self._session.closed:
            self._fail("closed")
        self._session._owner_check()
        self._session._root_check()

    @contextmanager
    def _operation(self):
        self._owner_check()
        if self._closed:
            self._fail("closed")
        self._busy = True
        try:
            self._session_check()
            yield
            self._session_check()
        except CheckpointViewError:
            self._failed = True
            self._session._failed = True
            raise
        except BaseException:
            self._fail("read_failed")
        finally:
            self._busy = False

    def _amount(self, size):
        if type(size) is not int or size < -1:
            self._fail("invalid_arguments")
        remaining = self._length - self._position
        amount = remaining if size == -1 else min(size, remaining)
        if amount > self._maximum_call:
            self._fail("read_cap_exceeded")
        return amount

    def _chunk(self, amount):
        self._session_check()
        chunk = self._session.read_file(self._relative, self._start + self._position, amount)
        if type(chunk) is not bytes or len(chunk) != amount:
            self._fail("read_failed")
        self._session_check()
        return chunk

    def read(self, size=-1):
        with self._operation():
            remaining = self._amount(size)
            chunks = []
            while remaining:
                chunk = self._chunk(min(remaining, MAX_OPERATION_BYTES))
                chunks.append(chunk)
                self._position += len(chunk)
                remaining -= len(chunk)
            return b"".join(chunks)

    def readinto(self, buffer):
        with self._operation():
            if type(buffer) not in (bytearray, memoryview):
                self._fail("invalid_arguments")
            with memoryview(buffer) as original:
                if (original.readonly or original.ndim != 1 or original.itemsize != 1
                        or not original.c_contiguous):
                    self._fail("invalid_arguments")
                amount = self._amount(original.nbytes)
                with original.cast("B") as target:
                    copied = 0
                    while copied < amount:
                        chunk = self._chunk(min(amount - copied, MAX_OPERATION_BYTES))
                        target[copied:copied + len(chunk)] = chunk
                        self._position += len(chunk)
                        copied += len(chunk)
                    return copied

    def readline(self, size=-1):
        with self._operation():
            remaining = self._amount(size)
            chunks = []
            while remaining:
                chunk = self._chunk(min(remaining, MAX_OPERATION_BYTES))
                newline = chunk.find(b"\n")
                consumed = len(chunk) if newline == -1 else newline + 1
                chunks.append(chunk[:consumed])
                self._position += consumed
                remaining -= consumed
                if newline != -1:
                    break
            return b"".join(chunks)

    def seek(self, offset, whence=os.SEEK_SET):
        with self._operation():
            if type(offset) is not int or type(whence) is not int:
                self._fail("invalid_arguments")
            if whence == os.SEEK_SET:
                position = offset
            elif whence == os.SEEK_CUR:
                position = self._position + offset
            elif whence == os.SEEK_END:
                position = self._length + offset
            else:
                self._fail("invalid_arguments")
            if not 0 <= position <= self._length:
                self._fail("invalid_arguments")
            self._position = position
            return position

    def tell(self):
        with self._operation():
            return self._position

    def readable(self):
        with self._operation():
            return True

    def seekable(self):
        with self._operation():
            return True

    def slice(self, offset, length):
        with self._operation():
            if (type(offset) is not int or type(length) is not int
                    or not 0 <= offset <= self._length or not 0 <= length <= self._length - offset):
                self._fail("invalid_arguments")
            return CheckpointReadView(self._session, self._relative, offset=self._start + offset,
                                      length=length, maximum_read_call_bytes=self._maximum_call)

    def close(self):
        self._owner_check()
        self._closed = True
