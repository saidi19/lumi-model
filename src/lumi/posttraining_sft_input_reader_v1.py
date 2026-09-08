"""Exact-prefix input reads over exclusively transferred private descriptors.

No path opening, ML import, admission or execution permission. The caller must
provide one trusted cumulative parent reservation callback supporting the v2
metadata category. Logical content bytes are prepaid, not physical I/O measured.
Index reads stop at the first newline without buffering the next record. Metadata
syscalls, allocations and CPU time still need independent host enforcement.
Only fixed errors and aggregate summaries are safe to log, never contexts/FDs.
"""

from __future__ import annotations

import os

from .posttraining_sft_metered_io_v1 import MAX_OPERATION_BYTES, MeteredIOError, _Metered

RESOURCES = frozenset({
    "encoded_metadata_read_bytes", "encoded_binary_read_bytes", "encoded_index_read_bytes",
})
MAX_READ_CALL = {
    "encoded_metadata_read_bytes": 16 * 1024 * 1024 + 1,
    "encoded_binary_read_bytes": 4095 * 4,
    "encoded_index_read_bytes": 176,
}


class InputReadHandle(_Metered):
    """Own one private O_RDONLY FD; expose no fileno, seek, raw or buffer API."""

    def __init__(self, fd, *, rank, resource, reserve, interrupt):
        if hasattr(self, "_fd"):
            self._fail("invalid_arguments")
        if not callable(interrupt):
            raise MeteredIOError("invalid_arguments") from None
        self._interrupt, self._interrupted, self._retired = interrupt, False, False
        self._failure = None
        self._bytes_read, self._rows_reserved = 0, 0
        self._initialize(fd, rank, resource, reserve, os.O_RDONLY, RESOURCES)

    def __repr__(self):
        return "<InputReadHandle private>"

    def _fail(self, code):
        self._failed = True
        if self._failure is None:
            self._failure = code
        if not self._retired and not self._interrupted:
            self._interrupted = True
            try:
                self._interrupt()
            except BaseException:
                pass
        raise MeteredIOError(self._failure) from None

    def _chunk(self, size):
        amount = min(size, self._identity[5] - self._position)
        if amount == 0:
            return b""
        self._reserve_before(amount)
        try:
            chunk = os.pread(self._fd, amount, self._position)
        except BaseException:
            self._fail("read_failed")
        if type(chunk) is bytes:
            self._bytes_read += len(chunk)  # Include a returned short/changed-file read.
        if type(chunk) is not bytes or len(chunk) != amount:
            self._fail("short_read")
        self._verify()
        self._position += amount
        return chunk

    def read(self, size):
        with self._operation():
            if (self._resource == "encoded_index_read_bytes" or type(size) is not int
                    or not 0 <= size <= MAX_READ_CALL[self._resource]):
                self._fail("invalid_arguments")
            remaining = min(size, self._identity[5] - self._position)
            chunks = []
            while remaining:
                chunk = self._chunk(min(remaining, MAX_OPERATION_BYTES))
                chunks.append(chunk)
                remaining -= len(chunk)
            return b"".join(chunks)

    def readline(self, size):
        with self._operation():
            if (self._resource != "encoded_index_read_bytes" or type(size) is not int
                    or not 1 <= size <= MAX_READ_CALL[self._resource]):
                self._fail("invalid_arguments")
            try:
                result = self._reserve(self._rank, encoded_index_rows=1)
            except BaseException:
                self._fail("reserve_failed")
            if result is not None or self._failed:
                self._fail("reserve_failed")
            self._rows_reserved += 1
            self._verify()
            output = bytearray()
            for _ in range(size):
                chunk = self._chunk(1)
                output.extend(chunk)
                if not chunk or chunk == b"\n":
                    break
            return bytes(output)

    def close(self):
        super().close()
        if not self._failed:
            self._retired = True

    def summary(self):
        return {"bytes_read": self._bytes_read, "index_rows_reserved": self._rows_reserved,
                "closed": self.closed, "failed": self._failed}
