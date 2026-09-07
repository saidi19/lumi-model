"""Cooperative metered checkpoint streams and verified private staging, no ML.

Streams take already-authorized, exclusively owned empty file FDs. Finalization
takes a private directory FD AFTER all payload writers have closed and all ranks
have stopped modifying it. The caller provides their exact receipts and one
shared-controller callback; receipts alone are not authentication or proof that
other code obeyed a ledger. This module neither opens a training entrypoint nor
creates a distributed barrier, publishes/renames a checkpoint, or deletes files.

The reader/byte primitives are version-pinned internal collaborators. Private FD
access below is only for checked, targeted fsync; it is never exposed to a
serializer. Unmodified DCP must not be passed these streams: its fileno fallback
can call host-wide os.sync. This is not a Torch/DCP integration claim.

Failures are terminal and nonrefundable. The caller MUST interrupt the shared
controller and retire every rank. Partial staging evidence is left untouched.
Observed metadata is not an atomic lock or protection from hostile same-inode FD
reuse. Never log suppressed exception contexts/tracebacks: only fixed error args
and repr are raw-text-free. Explicit close is required; there are no finalizers.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import stat
import sys
import threading
from contextlib import contextmanager
from pathlib import PurePosixPath

from .posttraining_sft_checkpoint_reader_v1 import (
    _DIGEST,
    MANIFEST,
    MAX_MANIFEST_BYTES,
    MAX_PAYLOAD_FILES,
    CheckpointReadSession,
    _metadata,
    _parts,
    _private,
)
from .posttraining_sft_metered_io_v1 import MAX_FILE_BYTES, MAX_OPERATION_BYTES, MeteredWriter

_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
_CREATE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
ERROR_CODES = frozenset({
    "invalid_arguments", "invalid_descriptor", "descriptor_changed", "owner_mismatch",
    "reentry", "closed", "terminal_failure", "not_closed", "already_finished",
    "inventory_mismatch", "checksum_mismatch", "write_cap_exceeded", "read_cap_exceeded",
    "sync_cap_exceeded", "reserve_failed", "io_failed", "close_failed", "internal_error",
})


class CheckpointSaveError(RuntimeError):
    def __init__(self, code="internal_error"):
        self.code = code if type(code) is str and code in ERROR_CODES else "internal_error"
        super().__init__(self.code)


def _positive(value):
    return type(value) is int and 0 < value <= MAX_FILE_BYTES


def _created_file_directory_match(before, after):
    # APFS may count the newly created regular child in the directory link count.
    # Permit only that +1 immediately after our own exclusive create on Darwin.
    stable = (0, 1, 2, 4)
    links_match = after[3] == before[3] or (
        sys.platform == "darwin" and after[3] == before[3] + 1)
    return links_match and all(before[index] == after[index] for index in stable)


class _Guard:
    def _initialize_guard(self):
        self._owner = os.getpid(), threading.get_ident()
        self._busy, self._failed, self._closed = False, False, False

    def __repr__(self):
        return "<CheckpointSave private>"

    @property
    def closed(self):
        return self._closed

    def _fail(self, code):
        self._failed = True
        raise CheckpointSaveError(code) from None

    def _owner_check(self):
        if self._owner != (os.getpid(), threading.get_ident()):
            self._fail("owner_mismatch")
        if self._busy:
            self._fail("reentry")

    @contextmanager
    def _operation(self):
        self._owner_check()
        if self._closed:
            self._fail("closed")
        if self._failed:
            self._fail("terminal_failure")
        self._busy = True
        try:
            self._check()
            yield
            self._check()
            if self._failed:
                self._fail("terminal_failure")
        except CheckpointSaveError:
            self._failed = True
            raise
        except BaseException:
            self._fail("io_failed")
        finally:
            self._busy = False


class CheckpointWriteStream(_Guard):
    """Append-only write/tell/flush/close; deliberately no fileno or path API.

    Already-allocated contiguous byte buffers are split into at most 1 MiB
    immutable chunks. The whole call and remaining file cap are checked before
    its first write. Allocating serializer tensors/buffers is outside this API.
    Flush syncs only this file, once per changed position (including empty files).
    """

    def __init__(self, fd, *, rank, reserve, maximum_file_bytes, maximum_write_call_bytes,
                 maximum_sync_operations):
        if hasattr(self, "_writer"):
            self._fail("invalid_arguments")
        if (not callable(reserve) or not all(_positive(n) for n in (
                maximum_file_bytes, maximum_write_call_bytes, maximum_sync_operations))):
            raise CheckpointSaveError("invalid_arguments") from None
        self._initialize_guard()
        self._reserve = reserve
        self._maximum_file, self._maximum_call = maximum_file_bytes, maximum_write_call_bytes
        self._maximum_sync, self._syncs = maximum_sync_operations, 0
        self._synced_position, self._bytes = None, 0
        self._digest = hashlib.sha256()
        try:
            self._writer = MeteredWriter(fd, rank=rank, resource="private_write_bytes",
                                         reserve=self._debit, maximum_file_bytes=maximum_file_bytes)
        except BaseException:
            raise CheckpointSaveError("invalid_descriptor") from None

    def _check(self):
        self._writer._verify()

    def _debit(self, rank, **amounts):
        if self._failed:
            self._fail("terminal_failure")
        self._check()
        try:
            result = self._reserve(rank, **amounts)
        except BaseException:
            self._fail("reserve_failed")
        if self._failed:
            self._fail("terminal_failure")
        if result is not None:
            self._fail("reserve_failed")
        self._check()

    def write(self, data):
        with self._operation():
            if type(data) not in (bytes, bytearray, memoryview):
                self._fail("invalid_arguments")
            with memoryview(data) as view:
                if view.ndim != 1 or view.itemsize != 1 or not view.c_contiguous:
                    self._fail("invalid_arguments")
                size = view.nbytes
                if size > self._maximum_call or self._bytes + size > self._maximum_file:
                    self._fail("write_cap_exceeded")
                for start in range(0, size, MAX_OPERATION_BYTES):
                    chunk = bytes(view[start:start + MAX_OPERATION_BYTES])
                    self._writer.write(chunk)
                    self._digest.update(chunk)
                    self._bytes += len(chunk)
                return size

    def tell(self):
        with self._operation():
            return self._writer.tell()

    def _flush(self):
        self._check()
        if self._synced_position == self._bytes:
            return
        if self._syncs >= self._maximum_sync:
            self._fail("sync_cap_exceeded")
        self._syncs += 1  # No retry or refund if synchronization fails.
        with self._writer._operation():
            os.fsync(self._writer._fd)
            self._writer._verify()
        self._synced_position = self._bytes

    def flush(self):
        with self._operation():
            self._flush()

    def close(self):
        self._owner_check()
        if self._closed:
            return
        self._busy = True
        try:
            try:
                if not self._failed:
                    self._flush()
                    if self._failed:
                        self._fail("terminal_failure")
            finally:
                self._closed = True  # Revoke before any ambiguous close.
                self._writer.close()
        except CheckpointSaveError:
            self._failed = True
            raise
        except BaseException:
            self._fail("io_failed")
        finally:
            self._busy = False

    def receipt(self):
        self._owner_check()
        if self._failed:
            self._fail("terminal_failure")
        if not self._closed:
            self._fail("not_closed")
        return {"bytes": self._bytes, "sha256": self._digest.hexdigest()}


class CheckpointStagingFinalizer(_Guard):
    """Hash/readback/manifest/sync over closed payloads; never publication.

    Payload write accounting belongs to the earlier streams/shared controller.
    This finalizer charges its hash+verification reads and new manifest writes.
    The trusted caller must establish the rank barrier and exact receipt mapping.
    """

    def __init__(self, root_fd, *, rank, reserve, expected_files, maximum_file_bytes,
                 maximum_read_bytes, maximum_manifest_write_bytes, maximum_sync_operations):
        if hasattr(self, "_fd"):
            self._fail("invalid_arguments")
        if (type(root_fd) is not int or root_fd < 3 or type(rank) is not int
                or not 0 <= rank < 12 or not callable(reserve)
                or not all(_positive(n) for n in (maximum_file_bytes, maximum_read_bytes,
                                                maximum_manifest_write_bytes,
                                                maximum_sync_operations))
                or type(expected_files) is not dict
                or not 2 <= len(expected_files) <= MAX_PAYLOAD_FILES):
            raise CheckpointSaveError("invalid_arguments") from None
        entries = {}
        try:
            for name, receipt in expected_files.items():
                parts = _parts(name)
                if (parts[-1] == MANIFEST or type(receipt) is not dict
                        or set(receipt) != {"bytes", "sha256"}
                        or type(receipt["bytes"]) is not int
                        or not 0 <= receipt["bytes"] <= maximum_file_bytes
                        or type(receipt["sha256"]) is not str
                        or not _DIGEST.fullmatch(receipt["sha256"])):
                    raise ValueError
                entries[name] = dict(receipt)
            names = set(entries)
            if any(str(parent) in names
                   for name in names for parent in PurePosixPath(name).parents):
                raise ValueError
            manifest = "".join(f"{entries[name]['sha256']}  {name}\n"
                               for name in sorted(entries, key=PurePosixPath)).encode("ascii")
            if len(manifest) > min(MAX_MANIFEST_BYTES, maximum_manifest_write_bytes):
                raise ValueError
        except BaseException:
            raise CheckpointSaveError("invalid_arguments") from None
        try:
            info = os.fstat(root_fd)
            flags = fcntl.fcntl(root_fd, fcntl.F_GETFL), fcntl.fcntl(root_fd, fcntl.F_GETFD)
            if (not _private(info, directory=True) or flags[0] & os.O_ACCMODE != os.O_RDONLY
                    or flags[0] & os.O_APPEND or not flags[1] & fcntl.FD_CLOEXEC):
                raise ValueError
        except BaseException:
            raise CheckpointSaveError("invalid_descriptor") from None
        self._initialize_guard()
        self._rank, self._reserve = rank, reserve
        self._entries, self._manifest = entries, manifest
        self._pin = hashlib.sha256(manifest).hexdigest()
        self._maximum_file, self._maximum_read = maximum_file_bytes, maximum_read_bytes
        self._maximum_write = maximum_manifest_write_bytes
        self._maximum_sync = maximum_sync_operations
        self._reads, self._writes, self._syncs = 0, 0, 0
        self._identity, self._flags, self._finished = _metadata(info), flags, False
        self._fd = root_fd  # Ownership transfers after all validation.

    def _check(self):
        try:
            value = _metadata(os.fstat(self._fd))
            flags = fcntl.fcntl(self._fd, fcntl.F_GETFL), fcntl.fcntl(self._fd, fcntl.F_GETFD)
        except BaseException:
            self._fail("descriptor_changed")
        if value != self._identity or flags != self._flags:
            self._fail("descriptor_changed")

    def _debit(self, rank, **amounts):
        if self._failed:
            self._fail("terminal_failure")
        self._check()
        reads = amounts.get("checkpoint_read_bytes", 0)
        writes = amounts.get("private_write_bytes", 0)
        if self._reads + reads > self._maximum_read:
            self._fail("read_cap_exceeded")
        if self._writes + writes > self._maximum_write:
            self._fail("write_cap_exceeded")
        self._reads, self._writes = self._reads + reads, self._writes + writes
        try:
            result = self._reserve(rank, **amounts)
        except BaseException:
            self._fail("reserve_failed")
        if self._failed:
            self._fail("terminal_failure")
        if result is not None:
            self._fail("reserve_failed")
        self._check()

    @contextmanager
    def _scanner(self):
        fd, scanner = os.dup(self._fd), None
        try:
            scanner = CheckpointReadSession(fd, rank=self._rank, reserve=self._debit,
                                             expected_manifest_sha256=self._pin,
                                             maximum_file_bytes=self._maximum_file,
                                             maximum_read_bytes=self._maximum_read)
            yield scanner
        finally:
            if scanner is None:
                os.close(fd)
            else:
                scanner.close()

    def _sync(self, fd):
        if self._failed:
            self._fail("terminal_failure")
        self._check()
        if self._syncs >= self._maximum_sync:
            self._fail("sync_cap_exceeded")
        self._syncs += 1
        before = _metadata(os.fstat(fd))
        os.fsync(fd)
        if self._failed:
            self._fail("terminal_failure")
        if _metadata(os.fstat(fd)) != before:
            self._fail("descriptor_changed")
        self._check()

    @contextmanager
    def _directory(self, name, expected):
        parent, owned, prefix = self._fd, [], ""
        try:
            if name:
                for part in _parts(name):
                    prefix = f"{prefix}/{part}" if prefix else part
                    value = _metadata(os.stat(part, dir_fd=parent, follow_symlinks=False))
                    if value != expected[prefix]:
                        self._fail("descriptor_changed")
                    child = os.open(part, _READ_FLAGS | os.O_DIRECTORY, dir_fd=parent)
                    owned.append(child)
                    if _metadata(os.fstat(child)) != value:
                        self._fail("descriptor_changed")
                    parent = child
            yield parent
            self._check()
        finally:
            error = False
            while owned:
                try:
                    os.close(owned.pop())
                except BaseException:
                    error = True
            if error:
                self._fail("close_failed")

    def _write_manifest(self):
        self._check()
        fd, stream = os.open(MANIFEST, _CREATE_FLAGS, 0o600, dir_fd=self._fd), None
        try:
            after = _metadata(os.fstat(self._fd))
            if not _created_file_directory_match(self._identity, after):
                self._fail("descriptor_changed")
            self._identity = after  # Only our observed exclusive create may change root metadata.
            stream = CheckpointWriteStream(fd, rank=self._rank, reserve=self._debit,
                                           maximum_file_bytes=MAX_MANIFEST_BYTES,
                                           maximum_write_call_bytes=MAX_MANIFEST_BYTES,
                                           maximum_sync_operations=1)
            stream.write(self._manifest)
            if self._syncs >= self._maximum_sync:
                self._fail("sync_cap_exceeded")
            self._syncs += 1  # The stream's close performs one targeted fsync.
            stream.close()
        except BaseException:
            if stream is not None:
                stream._failed = True  # Cleanup must not retry sync after another failure.
            raise
        finally:
            if stream is None:
                os.close(fd)
            else:
                stream.close()

    def finish(self):
        with self._operation():
            if self._finished:
                self._fail("already_finished")
            total = sum(value["bytes"] for value in self._entries.values())
            if total * 2 + len(self._manifest) > self._maximum_read:
                self._fail("read_cap_exceeded")
            with self._scanner() as scanner, scanner._operation():
                scanner._files, scanner._directories = scanner._inventory()
                if set(scanner._files) != set(self._entries):
                    self._fail("inventory_mismatch")
                if len(scanner._files) + 1 + len(scanner._directories) > self._maximum_sync:
                    self._fail("sync_cap_exceeded")
                for name, receipt in self._entries.items():
                    remaining = scanner._files[name][5]
                    if remaining != receipt["bytes"]:
                        self._fail("checksum_mismatch")
                    digest = hashlib.sha256()
                    with scanner._reader(name) as reader:
                        while remaining:
                            chunk = reader.read(min(remaining, MAX_OPERATION_BYTES))
                            digest.update(chunk)
                            remaining -= len(chunk)
                        if digest.hexdigest() != receipt["sha256"]:
                            self._fail("checksum_mismatch")
                        with reader._operation():
                            self._sync(reader._fd)
                            reader._verify()
                if scanner._inventory() != (scanner._files, scanner._directories):
                    self._fail("descriptor_changed")
            self._write_manifest()
            with self._scanner() as scanner:
                receipt = scanner.verify()
                # All children are synced before their parents. No host-wide sync.
                for name in sorted(scanner._directories, key=lambda n: (n.count("/"), len(n)),
                                   reverse=True):
                    with self._directory(name, scanner._directories) as fd:
                        self._sync(fd)
                if scanner._inventory() != (scanner._files, scanner._directories):
                    self._fail("descriptor_changed")
            self._finished = True
            return {"status": "verified_staging", "manifest_sha256": self._pin,
                    "payload_files": receipt["payload_files"], "payload_bytes": total,
                    "checkpoint_read_bytes": self._reads, "private_write_bytes": self._writes,
                    "reserved_sync_operations": self._syncs, "published": False,
                    "production_eligible": False}

    def close(self):
        self._owner_check()
        if self._closed:
            return
        fd, self._fd, self._closed = self._fd, None, True
        try:
            info = os.fstat(fd)
            if _metadata(info)[:2] != self._identity[:2] or not stat.S_ISDIR(info.st_mode):
                self._fail("descriptor_changed")
            os.close(fd)
        except CheckpointSaveError:
            raise
        except BaseException:
            self._fail("close_failed")
