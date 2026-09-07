"""One no-replace rename of verified private staging, without ML or fallback.

The caller authorizes the exact parent FD/names, finishes and closes all staging
writers/finalizers, establishes its rank barrier, and supplies one trusted shared
reservation callback. A copied receipt is a binding, NOT evidence of admission,
durability or a barrier. Parent and staging must remain exclusively controlled.
Name-based native rename cannot atomically bind the previously checked inode.

Only Linux renameat2(RENAME_NOREPLACE) and Darwin renameatx_np(RENAME_EXCL) are
supported. Missing OS/libc/filesystem support fails closed: no rename/replace,
copy, retry, rollback or deletion fallback. The namespace change and parent sync
are not a single transaction. An error after attempting rename may leave a
visible checkpoint; inspect state(), stop the shared controller and retire ranks.
Never infer publication from existence or blindly retry a failed attempt.

Fixed error args/repr and aggregate state are safe to report. Suppressed Python
exception contexts/tracebacks may retain private data and MUST NOT be logged.
This is cooperative logical-byte accounting, not an OS quota, a filesystem lock,
a production release, actual training, or protection from hostile FD reuse.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import os
import re
import stat
import sys
import threading
from contextlib import contextmanager

from .posttraining_sft_checkpoint_reader_v1 import (
    _DIGEST,
    MANIFEST,
    MAX_MANIFEST_BYTES,
    MAX_PAYLOAD_FILES,
    CheckpointReadSession,
    _metadata,
    _private,
)
from .posttraining_sft_metered_io_v1 import MAX_FILE_BYTES

MAX_PARENT_ENTRIES = 1024
_OPEN_DIRECTORY = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
_RECEIPT_KEYS = frozenset({
    "status", "manifest_sha256", "payload_files", "payload_bytes", "checkpoint_read_bytes",
    "private_write_bytes", "reserved_sync_operations", "published", "production_eligible",
})
ERROR_CODES = frozenset({
    "invalid_arguments", "invalid_descriptor", "descriptor_changed", "owner_mismatch",
    "reentry", "closed", "terminal_failure", "already_published", "source_missing",
    "destination_exists", "inventory_mismatch", "structure_cap_exceeded", "receipt_mismatch",
    "read_cap_exceeded", "reserve_failed", "platform_unsupported", "rename_failed",
    "io_failed", "close_failed", "internal_error",
})


class CheckpointPublicationError(RuntimeError):
    def __init__(self, code="internal_error"):
        self.code = code if type(code) is str and code in ERROR_CODES else "internal_error"
        super().__init__(self.code)


def _rename_binding():
    """Resolve only a fixed symbol in this process's libc; no library discovery."""
    if sys.platform == "linux":
        symbol, flag = "renameat2", 1  # Linux uapi RENAME_NOREPLACE.
    elif sys.platform == "darwin":
        symbol, flag = "renameatx_np", 4  # Darwin sys/stdio.h RENAME_EXCL.
    else:
        raise CheckpointPublicationError("platform_unsupported") from None
    try:
        library = ctypes.CDLL(None, use_errno=True)
        function = getattr(library, symbol)
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                             ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
    except BaseException:
        raise CheckpointPublicationError("platform_unsupported") from None
    return library, function, flag  # Keep library alive for the complete native call.


def _native_rename_noreplace(binding, parent_fd, staging_name, checkpoint_name):
    _, function, flag = binding
    ctypes.set_errno(0)
    result = function(parent_fd, staging_name.encode("ascii"), parent_fd,
                      checkpoint_name.encode("ascii"), flag)
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise CheckpointPublicationError("destination_exists") from None
    if error in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
        raise CheckpointPublicationError("platform_unsupported") from None
    raise CheckpointPublicationError("rename_failed") from None


def _valid_receipt(receipt):
    if type(receipt) is not dict or set(receipt) != _RECEIPT_KEYS:
        return False
    integers = ("payload_files", "payload_bytes", "checkpoint_read_bytes",
                "private_write_bytes", "reserved_sync_operations")
    return (
        receipt["status"] == "verified_staging" and type(receipt["status"]) is str
        and type(receipt["manifest_sha256"]) is str
        and _DIGEST.fullmatch(receipt["manifest_sha256"]) is not None
        and receipt["published"] is False and receipt["production_eligible"] is False
        and all(type(receipt[key]) is int and 0 <= receipt[key] <= MAX_FILE_BYTES
                for key in integers)
        and 2 <= receipt["payload_files"] <= MAX_PAYLOAD_FILES
        and 0 < receipt["private_write_bytes"] <= MAX_MANIFEST_BYTES
        and receipt["checkpoint_read_bytes"] == (
            2 * receipt["payload_bytes"] + receipt["private_write_bytes"])
        and receipt["reserved_sync_operations"] >= receipt["payload_files"] + 2
    )


class CheckpointPublisher:
    """One same-parent checkpoint rename, full metered verification on both sides.

    Names match the frozen worker: checkpoint-NNNNNNNN and its hidden UUID-named
    staging sibling. Successful construction transfers only the parent FD. The
    publisher writes no content, syncs that parent once and charges 2*(P+M) reads
    on success. Earlier staging writes/syncs are outside these operation counters.
    """

    def __init__(self, parent_fd, *, staging_name, checkpoint_name, staging_receipt,
                 rank, reserve, maximum_file_bytes, maximum_read_bytes):
        if hasattr(self, "_fd"):
            self._fail("invalid_arguments")
        if (type(parent_fd) is not int or parent_fd < 3
                or type(rank) is not int or not 0 <= rank < 12 or not callable(reserve)
                or type(checkpoint_name) is not str
                or not re.fullmatch(r"checkpoint-[0-9]{8}", checkpoint_name, re.ASCII)
                or type(staging_name) is not str
                or not re.fullmatch(r"\." + re.escape(checkpoint_name)
                                    + r"\.[0-9a-f]{32}\.tmp", staging_name, re.ASCII)
                or not _valid_receipt(staging_receipt)
                or any(type(n) is not int or not 0 < n <= MAX_FILE_BYTES
                       for n in (maximum_file_bytes, maximum_read_bytes))):
            raise CheckpointPublicationError("invalid_arguments") from None
        try:
            info = os.fstat(parent_fd)
            flags = fcntl.fcntl(parent_fd, fcntl.F_GETFL), fcntl.fcntl(parent_fd, fcntl.F_GETFD)
            if (not _private(info, directory=True) or flags[0] & os.O_ACCMODE != os.O_RDONLY
                    or flags[0] & os.O_APPEND or not flags[1] & fcntl.FD_CLOEXEC):
                raise ValueError
        except BaseException:
            raise CheckpointPublicationError("invalid_descriptor") from None
        self._owner = os.getpid(), threading.get_ident()
        self._staging, self._checkpoint = staging_name, checkpoint_name
        self._receipt = dict(staging_receipt)
        self._rank, self._reserve = rank, reserve
        self._maximum_file, self._maximum_read = maximum_file_bytes, maximum_read_bytes
        self._identity, self._flags = _metadata(info), flags
        self._failed, self._busy, self._complete = False, False, False
        self._reads, self._renames, self._syncs, self._phase = 0, 0, 0, "ready"
        self._fd = parent_fd  # Transfer only after every constructor check.

    def __repr__(self):
        return "<CheckpointPublisher private>"

    @property
    def closed(self):
        return self._fd is None

    def _fail(self, code):
        self._failed, self._complete = True, False
        raise CheckpointPublicationError(code) from None

    def _owner_check(self):
        if self._owner != (os.getpid(), threading.get_ident()):
            self._fail("owner_mismatch")
        if self._busy:
            self._fail("reentry")

    def _parent_check(self):
        try:
            info = _metadata(os.fstat(self._fd))
            flags = fcntl.fcntl(self._fd, fcntl.F_GETFL), fcntl.fcntl(self._fd, fcntl.F_GETFD)
        except BaseException:
            self._fail("descriptor_changed")
        if info != self._identity or flags != self._flags:
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
            self._parent_check()
            yield
            self._parent_check()
            if self._failed:
                self._fail("terminal_failure")
        except CheckpointPublicationError:
            self._failed, self._complete = True, False
            raise
        except BaseException:
            self._fail("io_failed")
        finally:
            self._busy = False

    def _inventory(self):
        self._parent_check()
        entries = {}
        with os.scandir(self._fd) as stream:
            for entry in stream:
                if len(entries) >= MAX_PARENT_ENTRIES:
                    self._fail("structure_cap_exceeded")
                entries[entry.name] = _metadata(os.stat(entry.name, dir_fd=self._fd,
                                                        follow_symlinks=False))
        self._parent_check()
        return entries

    def _debit(self, rank, *, checkpoint_read_bytes):
        if self._failed:
            self._fail("terminal_failure")
        self._parent_check()
        if self._reads + checkpoint_read_bytes > self._maximum_read:
            self._fail("read_cap_exceeded")
        self._reads += checkpoint_read_bytes  # Failed reservations/reads are not refunded.
        try:
            result = self._reserve(rank, checkpoint_read_bytes=checkpoint_read_bytes)
        except BaseException:
            self._fail("reserve_failed")
        if self._failed:
            self._fail("terminal_failure")
        if result is not None:
            self._fail("reserve_failed")
        self._parent_check()

    @contextmanager
    def _reader(self, name, expected):
        self._parent_check()
        info = os.stat(name, dir_fd=self._fd, follow_symlinks=False)
        if (not _private(info, directory=True) or _metadata(info) != expected
                or info.st_dev != self._identity[0]):
            self._fail("descriptor_changed")
        fd, reader = os.open(name, _OPEN_DIRECTORY, dir_fd=self._fd), None
        try:
            if _metadata(os.fstat(fd)) != expected:
                self._fail("descriptor_changed")
            reader = CheckpointReadSession(fd, rank=self._rank, reserve=self._debit,
                                            expected_manifest_sha256=self._receipt["manifest_sha256"],
                                            maximum_file_bytes=self._maximum_file,
                                            maximum_read_bytes=self._maximum_read)
            yield reader
            self._parent_check()
        finally:
            if reader is None:
                os.close(fd)
            else:
                reader.close()

    def _match_receipt(self, verified):
        if any(verified[key] != self._receipt[key] for key in ("payload_files", "payload_bytes")):
            self._fail("receipt_mismatch")

    def publish(self):
        with self._operation():
            if self._complete:
                self._fail("already_published")
            binding = _rename_binding()  # Unsupported platforms fail before any content read.
            before = self._inventory()
            if self._checkpoint in before:
                self._fail("destination_exists")
            if self._staging not in before:
                self._fail("source_missing")
            expected = before[self._staging]
            self._phase = "verifying_staging"
            with self._reader(self._staging, expected) as reader:
                manifest = os.stat(MANIFEST, dir_fd=reader._fd, follow_symlinks=False)
                if (not _private(manifest, directory=False)
                        or manifest.st_size != self._receipt["private_write_bytes"]):
                    self._fail("receipt_mismatch")
                if 2 * (self._receipt["payload_bytes"] + manifest.st_size) > self._maximum_read:
                    self._fail("read_cap_exceeded")
                verified = reader.verify()
                self._match_receipt(verified)
                files, directories = dict(reader._files), dict(reader._directories)
            if self._inventory() != before:
                self._fail("descriptor_changed")
            self._renames, self._phase = 1, "rename_attempted"
            # The native no-replace check is atomic even if destination appears now.
            _native_rename_noreplace(binding, self._fd, self._staging, self._checkpoint)
            self._phase = "renamed"
            if self._failed:
                self._fail("terminal_failure")
            after = _metadata(os.fstat(self._fd))
            if after[:5] != self._identity[:5]:
                self._fail("descriptor_changed")
            self._identity = after  # Only our successful same-parent rename may update it.
            observed = self._inventory()
            if set(observed) != (set(before) - {self._staging}) | {self._checkpoint}:
                self._fail("inventory_mismatch")
            if any(observed[name] != value for name, value in before.items()
                   if name != self._staging):
                self._fail("descriptor_changed")
            moved = observed[self._checkpoint]
            if moved[:7] != expected[:7]:  # Rename may change only the moved root's ctime.
                self._fail("descriptor_changed")
            self._syncs = 1  # A failed/ambiguous fsync is never retried.
            os.fsync(self._fd)
            if self._failed:
                self._fail("terminal_failure")
            self._parent_check()
            with self._reader(self._checkpoint, moved) as reader:
                verified = reader.verify()
                self._match_receipt(verified)
                directories[""] = moved
                if (reader._files, reader._directories) != (files, directories):
                    self._fail("descriptor_changed")
            if self._inventory() != observed:
                self._fail("descriptor_changed")
            self._complete, self._phase = True, "complete"
            return {"status": "published_checkpoint",
                    "manifest_sha256": self._receipt["manifest_sha256"],
                    "payload_files": verified["payload_files"],
                    "payload_bytes": verified["payload_bytes"],
                    "checkpoint_read_bytes": self._reads, "private_write_bytes": 0,
                    "rename_attempts": self._renames, "parent_sync_attempts": self._syncs,
                    "published": True, "production_eligible": False}

    def state(self):
        """Raw-text-free failure observation, NOT permission to retry/recover."""
        self._owner_check()
        return {"phase": self._phase, "failed": self._failed, "rename_attempts": self._renames,
                "parent_sync_attempts": self._syncs, "checkpoint_read_bytes": self._reads,
                "publication_verified": self._complete, "closed": self.closed}

    def close(self):
        self._owner_check()
        if self.closed:
            return
        fd, self._fd = self._fd, None
        try:
            info = os.fstat(fd)
            if _metadata(info)[:2] != self._identity[:2] or not stat.S_ISDIR(info.st_mode):
                self._fail("descriptor_changed")
            os.close(fd)
        except CheckpointPublicationError:
            raise
        except BaseException:
            self._fail("close_failed")
