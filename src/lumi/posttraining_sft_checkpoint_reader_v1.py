"""Pinned checksum verification and bounded checkpoint slices, without ML.

The trusted caller supplies an authorized, immutable private directory FD and
one shared-controller reservation callback. Successful construction transfers
the FD. Every content read (manifest, hashing, and repeated slices) uses the
metered reader; no payload is deserialized, mapped, or handed to a path-native
loader. Metadata traversal is separately bounded, not charged as content bytes.

This is cooperative accounting, not admission, a filesystem lock, an OS quota,
or a Torch/DCP adapter. On failure the caller must interrupt the shared lifecycle
and retire every rank. Explicit close is required. Same-inode FD reuse and all
hostile concurrent mutations cannot be excluded by metadata observations.
Only fixed error arguments/repr are safe to report: suppressed Python exception
contexts may retain private text and MUST NOT be logged.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import stat
import threading
from contextlib import contextmanager

from .posttraining_sft_metered_io_v1 import MAX_FILE_BYTES, MAX_OPERATION_BYTES, MeteredReader

MAX_MANIFEST_BYTES = 65_536
MAX_PAYLOAD_FILES = 256
MAX_TREE_ENTRIES = 2048
MAX_PATH_BYTES = 256
MAX_PATH_COMPONENTS = 8
MANIFEST = "manifest.sha256"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_COMPONENT = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z", re.ASCII)
_READ_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
ERROR_CODES = frozenset({
    "invalid_arguments", "invalid_descriptor", "descriptor_changed", "owner_mismatch",
    "reentry", "closed", "terminal_failure", "not_verified", "already_verified",
    "invalid_manifest", "manifest_mismatch", "checksum_mismatch", "inventory_mismatch",
    "structure_cap_exceeded", "read_cap_exceeded", "reserve_failed", "io_failed",
    "close_failed", "internal_error",
})


class CheckpointReadError(RuntimeError):
    def __init__(self, code="internal_error"):
        self.code = code if type(code) is str and code in ERROR_CODES else "internal_error"
        super().__init__(self.code)


def _metadata(info):
    # Reads may change atime; they must not change these other attributes.
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_uid,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _private(info, *, directory):
    kind = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    return (kind and info.st_uid == os.geteuid()
            and stat.S_IMODE(info.st_mode) == (0o700 if directory else 0o600)
            and (directory or info.st_nlink == 1))


def _parts(value):
    if type(value) is not str or not 0 < len(value) <= MAX_PATH_BYTES:
        raise CheckpointReadError("invalid_manifest") from None
    parts = value.split("/")
    if (len(parts) > MAX_PATH_COMPONENTS
            or any(part in {".", ".."} or not _COMPONENT.fullmatch(part) for part in parts)):
        raise CheckpointReadError("invalid_manifest") from None
    return parts


def _parse_manifest(data):
    try:
        value = data.decode("ascii")
    except (AttributeError, UnicodeError):
        raise CheckpointReadError("invalid_manifest") from None
    if not value.endswith("\n"):
        raise CheckpointReadError("invalid_manifest") from None
    entries = {}
    for line in value[:-1].split("\n"):
        pair = line.split("  ", 1)
        if len(pair) != 2 or not _DIGEST.fullmatch(pair[0]):
            raise CheckpointReadError("invalid_manifest") from None
        digest, relative = pair
        parts = _parts(relative)
        if parts[-1] == MANIFEST or relative in entries:
            raise CheckpointReadError("invalid_manifest") from None
        entries[relative] = digest
        if len(entries) > MAX_PAYLOAD_FILES:
            raise CheckpointReadError("structure_cap_exceeded") from None
    if len(entries) < 2:
        raise CheckpointReadError("invalid_manifest") from None
    # No string-sort requirement: the frozen producer sorts pathlib paths.
    return entries


class CheckpointReadSession:
    """One verification, then bounded explicit slices of declared files only.

    ``maximum_file_bytes`` bounds each payload; the manifest has its own 64 KiB
    bound. ``maximum_read_bytes`` includes all reads, including verification and
    rereads. Empty payload files and bounded empty private directories are valid.
    This version supports canonical ASCII relative names, not arbitrary paths.
    """

    def __init__(self, root_fd, *, rank, reserve, expected_manifest_sha256,
                 maximum_file_bytes, maximum_read_bytes):
        if hasattr(self, "_fd"):
            self._fail("invalid_arguments")
        if (type(root_fd) is not int or root_fd < 3
                or type(rank) is not int or not 0 <= rank < 12 or not callable(reserve)
                or type(expected_manifest_sha256) is not str
                or not _DIGEST.fullmatch(expected_manifest_sha256)
                or any(type(n) is not int or not 0 < n <= MAX_FILE_BYTES
                       for n in (maximum_file_bytes, maximum_read_bytes))):
            raise CheckpointReadError("invalid_arguments") from None
        try:
            info = os.fstat(root_fd)
            flags = (fcntl.fcntl(root_fd, fcntl.F_GETFL), fcntl.fcntl(root_fd, fcntl.F_GETFD))
            valid = (_private(info, directory=True) and flags[0] & os.O_ACCMODE == os.O_RDONLY
                     and not flags[0] & os.O_APPEND and flags[1] & fcntl.FD_CLOEXEC)
        except BaseException:
            raise CheckpointReadError("invalid_descriptor") from None
        if not valid:
            raise CheckpointReadError("invalid_descriptor") from None
        self._rank, self._reserve = rank, reserve
        self._expected = expected_manifest_sha256
        self._maximum_file, self._maximum_read = maximum_file_bytes, maximum_read_bytes
        self._identity, self._flags = _metadata(info), flags
        self._owner = os.getpid(), threading.get_ident()
        self._files, self._directories = {}, {}
        self._spent, self._payload_bytes = 0, 0
        self._busy, self._failed, self._verified = False, False, False
        self._fd = root_fd  # Transfer only after successful validation.

    def __repr__(self):
        return "<CheckpointReadSession private>"

    @property
    def closed(self):
        return self._fd is None

    def _fail(self, code):
        self._failed = True
        raise CheckpointReadError(code) from None

    def _owner_check(self):
        if self._owner != (os.getpid(), threading.get_ident()):
            self._fail("owner_mismatch")
        if self._busy:
            self._fail("reentry")

    def _root_check(self):
        try:
            current = _metadata(os.fstat(self._fd))
            flags = (fcntl.fcntl(self._fd, fcntl.F_GETFL),
                     fcntl.fcntl(self._fd, fcntl.F_GETFD))
        except BaseException:
            self._fail("descriptor_changed")
        if current != self._identity or flags != self._flags:
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
            self._root_check()
            yield
            self._root_check()
            if self._failed:
                self._fail("terminal_failure")
        except CheckpointReadError:
            self._failed = True
            raise
        except BaseException:
            self._fail("io_failed")
        finally:
            self._busy = False

    def _close_owned(self, fd):
        try:
            os.close(fd)
        except BaseException:
            self._fail("close_failed")

    def _inventory(self):
        files, directories, count = {}, {"": self._identity}, 0

        def visit(fd, prefix):
            nonlocal count
            before = _metadata(os.fstat(fd))
            with os.scandir(fd) as stream:
                for entry in stream:
                    count += 1
                    if count > MAX_TREE_ENTRIES:
                        self._fail("structure_cap_exceeded")
                    relative = f"{prefix}/{entry.name}" if prefix else entry.name
                    _parts(relative)
                    info = os.stat(entry.name, dir_fd=fd, follow_symlinks=False)
                    directory = stat.S_ISDIR(info.st_mode)
                    if not _private(info, directory=directory):
                        self._fail("inventory_mismatch")
                    identity = _metadata(info)
                    if directory:
                        child = os.open(entry.name, _READ_FLAGS | os.O_DIRECTORY, dir_fd=fd)
                        try:
                            if _metadata(os.fstat(child)) != identity:
                                self._fail("descriptor_changed")
                            directories[relative] = identity
                            visit(child, relative)
                        finally:
                            self._close_owned(child)
                    else:
                        limit = MAX_MANIFEST_BYTES if relative == MANIFEST else self._maximum_file
                        if not 0 <= info.st_size <= limit:
                            self._fail("structure_cap_exceeded")
                        files[relative] = identity
                        if len(files) > MAX_PAYLOAD_FILES + 1:
                            self._fail("structure_cap_exceeded")
            if _metadata(os.fstat(fd)) != before:
                self._fail("descriptor_changed")

        visit(self._fd, "")
        self._root_check()
        return files, directories

    def _debit(self, rank, *, checkpoint_read_bytes):
        if self._spent + checkpoint_read_bytes > self._maximum_read:
            self._fail("read_cap_exceeded")
        self._spent += checkpoint_read_bytes  # Non-refundable even if shared reserve fails.
        try:
            result = self._reserve(rank, checkpoint_read_bytes=checkpoint_read_bytes)
        except BaseException:
            self._fail("reserve_failed")
        if self._failed:
            self._fail("terminal_failure")
        if result is not None:
            self._fail("reserve_failed")
        self._root_check()

    @contextmanager
    def _reader(self, relative):
        parts = _parts(relative)
        owned_dirs, chain = [], [(self._fd, self._identity)]
        reader, opened = None, None
        try:
            parent, prefix = self._fd, ""
            for part in parts[:-1]:
                prefix = f"{prefix}/{part}" if prefix else part
                expected = self._directories[prefix]
                if _metadata(os.stat(part, dir_fd=parent, follow_symlinks=False)) != expected:
                    self._fail("descriptor_changed")
                child = os.open(part, _READ_FLAGS | os.O_DIRECTORY, dir_fd=parent)
                owned_dirs.append(child)
                if _metadata(os.fstat(child)) != expected:
                    self._fail("descriptor_changed")
                chain.append((child, expected))
                parent = child
            if (_metadata(os.stat(parts[-1], dir_fd=parent, follow_symlinks=False))
                    != self._files[relative]):
                self._fail("descriptor_changed")
            opened = os.open(parts[-1], _READ_FLAGS, dir_fd=parent)
            if _metadata(os.fstat(opened)) != self._files[relative]:
                self._fail("descriptor_changed")
            reader = MeteredReader(opened, rank=self._rank, resource="checkpoint_read_bytes",
                                   reserve=self._debit)
            opened = None  # MeteredReader now owns the descriptor.
            yield reader
            for fd, expected in chain:
                if _metadata(os.fstat(fd)) != expected:
                    self._fail("descriptor_changed")
        finally:
            # Remove ownership before close; never retry even if cleanup raises.
            try:
                if reader is not None:
                    reader.close()
                elif opened is not None:
                    self._close_owned(opened)
            finally:
                first_error = None
                while owned_dirs:
                    try:
                        self._close_owned(owned_dirs.pop())
                    except CheckpointReadError as error:
                        first_error = error
                if first_error is not None:
                    raise first_error from None

    def verify(self):
        with self._operation():
            if self._verified:
                self._fail("already_verified")
            self._files, self._directories = self._inventory()
            if MANIFEST not in self._files:
                self._fail("inventory_mismatch")
            with self._reader(MANIFEST) as reader:
                data = reader.read(self._files[MANIFEST][5])
            if hashlib.sha256(data).hexdigest() != self._expected:
                self._fail("manifest_mismatch")
            entries = _parse_manifest(data)
            del data
            if set(self._files) != set(entries) | {MANIFEST}:
                self._fail("inventory_mismatch")
            self._payload_bytes = sum(self._files[name][5] for name in entries)
            if self._spent + self._payload_bytes > self._maximum_read:
                self._fail("read_cap_exceeded")
            for relative, expected in entries.items():
                digest = hashlib.sha256()
                remaining = self._files[relative][5]
                with self._reader(relative) as reader:
                    while remaining:
                        chunk = reader.read(min(remaining, MAX_OPERATION_BYTES))
                        digest.update(chunk)
                        remaining -= len(chunk)
                if digest.hexdigest() != expected:
                    self._fail("checksum_mismatch")
            if self._inventory() != (self._files, self._directories):
                self._fail("descriptor_changed")
            self._verified = True
            return {"status": "verified", "payload_files": len(entries),
                    "payload_bytes": self._payload_bytes, "reserved_read_bytes": self._spent,
                    "deserialization_performed": False, "production_eligible": False}

    def read_file(self, relative, offset, size):
        with self._operation():
            if not self._verified:
                self._fail("not_verified")
            if (type(relative) is not str or relative == MANIFEST or relative not in self._files
                    or type(offset) is not int or type(size) is not int
                    or not 0 <= size <= MAX_OPERATION_BYTES or offset < 0
                    or offset + size > self._files[relative][5]):
                self._fail("invalid_arguments")
            with self._reader(relative) as reader:
                reader.seek(offset)
                result = reader.read(size)
            return result

    def close(self):
        self._owner_check()
        if self.closed:
            return
        fd, self._fd = self._fd, None
        try:
            current = os.fstat(fd)
        except BaseException:
            self._fail("descriptor_changed")
        if _metadata(current)[:2] != self._identity[:2] or not stat.S_ISDIR(current.st_mode):
            self._fail("descriptor_changed")
        self._close_owned(fd)
