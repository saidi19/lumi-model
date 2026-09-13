"""Cooperative HF content I/O over owned local descriptors, with no ML imports.

Only content read/write requests are predebited. This is not an OS quota, RSS
limit, deadline interrupt, filesystem lock, or proof that other code obeys the
ledger. The caller owns one quiescent process and must stop after any failure.
Partial outputs are preserved; no retry, overwrite, implicit cleanup, or chmod
of inputs is provided. Only fixed error codes may be logged, never contexts.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import stat
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

CHUNK_BYTES = 1_048_576
MAX_FILES = 4096
MAX_CONTROL_BYTES = 131_072
_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
_DIR_FLAGS = _READ_FLAGS | os.O_DIRECTORY
_CREATE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
_CONTROL_NAMES = frozenset({
    "config.json", "generation_config.json", "model.safetensors.index.json",
    "configuration_lumi.py", "modeling_lumi.py",
})
ERROR_CODES = frozenset({
    "invalid_arguments", "owner_mismatch", "reentry", "closed", "terminal_failure",
    "path_rejected", "input_changed", "output_exists", "output_not_created",
    "output_changed", "name_rejected", "unknown_file", "receipt_mismatch",
    "read_limit_exceeded", "write_limit_exceeded", "file_limit_exceeded",
    "control_limit_exceeded", "budget_check_failed", "read_failed", "short_read",
    "write_failed", "short_write", "checksum_mismatch", "close_failed", "io_failed",
})


class ContentIOError(RuntimeError):
    def __init__(self, code="io_failed"):
        self.code = code if type(code) is str and code in ERROR_CODES else "io_failed"
        super().__init__(self.code)


def _positive(value):
    return type(value) is int and 0 < value <= 2**63 - 1


def _snapshot(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_uid,
            info.st_gid, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _flags(fd):
    return fcntl.fcntl(fd, fcntl.F_GETFL), fcntl.fcntl(fd, fcntl.F_GETFD)


def _write_flags_match(before, after):
    # Darwin may set FWASWRITTEN only after this descriptor's own pwrite.
    written = 0x00010000 if sys.platform == "darwin" else 0
    return after[1] == before[1] and after[0] in {before[0], before[0] | written}


def _path(value):
    if (type(value) is not type(Path()) or not value.is_absolute()
            or ".." in value.parts or len(value.parts) < 2):
        raise ContentIOError("path_rejected") from None
    return value


def _name(value):
    if type(value) is not str:
        raise ContentIOError("name_rejected") from None
    if value in _CONTROL_NAMES or value == "model.safetensors":
        return value
    match = re.fullmatch(r"model-([0-9]{5})-of-([0-9]{5})\.safetensors", value)
    if match and 0 < int(match[1]) <= int(match[2]) <= MAX_FILES:
        return value
    raise ContentIOError("name_rejected") from None


def _close_fd(fd):
    try:
        os.close(fd)
    except BaseException:
        raise ContentIOError("close_failed") from None


def _open_directory(path):
    """Walk every ancestor with O_NOFOLLOW; transfer exactly one owned FD."""
    fd = os.open("/", _DIR_FLAGS)
    try:
        for part in path.parts[1:]:
            child = os.open(part, _DIR_FLAGS, dir_fd=fd)
            old, fd = fd, child
            _close_fd(old)
        result, fd = fd, None
        return result
    finally:
        if fd is not None:
            _close_fd(fd)


class ContentIO:
    """Fresh flat private HF output, explicit ownership, bounded bytes in memory.

    ``inventory`` checks known names and metadata, not content; hashes are write
    receipts. ``read`` independently checks its requested receipt and hashes the
    metered bytes. Sources may be 0400/0600/0644; outputs are always 0600. Python
    serialization buffers and native-library allocations belong to the caller.
    """

    def __init__(self, *, maximum_read_bytes, maximum_write_bytes, maximum_file_bytes,
                 maximum_control_bytes, check_budget):
        if (not all(_positive(n) for n in (maximum_read_bytes, maximum_write_bytes,
                                          maximum_file_bytes, maximum_control_bytes))
                or maximum_control_bytes > MAX_CONTROL_BYTES or not callable(check_budget)):
            raise ContentIOError("invalid_arguments") from None
        self._limits = maximum_read_bytes, maximum_write_bytes
        self._maximum_file = maximum_file_bytes
        self._maximum_control = maximum_control_bytes
        self._check_budget = check_budget
        self._owner = os.getpid(), threading.get_ident()
        self._reads = self._writes = self._checks = 0
        self._busy = self._failed = self._closed = False
        self._root = self._root_fd = self._root_identity = self._root_flags = None
        self._inputs = {}
        self._files = {}

    def __repr__(self):
        return "<ContentIO private>"

    def _fail(self, code):
        self._failed = True
        raise ContentIOError(code) from None

    def _owner_check(self):
        if (self._owner != (os.getpid(), threading.get_ident())
                or threading.current_thread() is not threading.main_thread()):
            self._fail("owner_mismatch")
        if self._busy:
            self._fail("reentry")

    def _tick(self):
        self._checks += 1
        try:
            result = self._check_budget()
        except BaseException:
            self._fail("budget_check_failed")
        if result is not None or self._failed:
            self._fail("budget_check_failed")

    @contextmanager
    def _operation(self):
        self._owner_check()
        if self._closed:
            self._fail("closed")
        if self._failed:
            self._fail("terminal_failure")
        self._busy = True
        try:
            self._tick()
            if self._root_fd is not None:
                self._verify_root()
            yield
            if self._root_fd is not None:
                self._verify_root()
            self._tick()
        except ContentIOError:
            self._failed = True
            raise
        except BaseException:
            self._fail("io_failed")
        finally:
            self._busy = False

    def _debit(self, amount, *, write=False):
        if write:
            if self._writes + amount > self._limits[1]:
                self._fail("write_limit_exceeded")
            self._writes += amount
        else:
            if self._reads + amount > self._limits[0]:
                self._fail("read_limit_exceeded")
            self._reads += amount
        self._tick()  # The reservation is never refunded if this or the syscall fails.

    def _verify_root(self):
        if (_snapshot(os.fstat(self._root_fd)) != self._root_identity
                or _flags(self._root_fd) != self._root_flags):
            self._fail("output_changed")
        fd = _open_directory(self._root)
        try:
            if _snapshot(os.fstat(fd)) != self._root_identity:
                self._fail("output_changed")
        finally:
            _close_fd(fd)

    def _read_fd(self, fd, identity, size, code):
        flags = _flags(fd)
        if (flags[0] & os.O_ACCMODE != os.O_RDONLY or flags[0] & os.O_APPEND
                or not flags[1] & fcntl.FD_CLOEXEC):
            self._fail(code)
        if _snapshot(os.fstat(fd)) != identity:
            self._fail(code)
        raw = bytearray()
        for start in range(0, size, CHUNK_BYTES):
            amount = min(CHUNK_BYTES, size - start)
            self._debit(amount)
            if _snapshot(os.fstat(fd)) != identity or _flags(fd) != flags:
                self._fail(code)
            try:
                part = os.pread(fd, amount, start)
            except BaseException:
                self._fail("read_failed")
            if type(part) is not bytes or len(part) != amount:
                self._fail("short_read")
            raw.extend(part)
            if _snapshot(os.fstat(fd)) != identity or _flags(fd) != flags:
                self._fail(code)
        return bytes(raw)

    def read_control(self, path):
        with self._operation():
            if self._root_fd is not None:
                self._fail("invalid_arguments")
            path = _path(path)
            parent = _open_directory(path.parent)
            fd = None
            try:
                parent_identity = _snapshot(os.fstat(parent))
                fd = os.open(path.name, _READ_FLAGS, dir_fd=parent)
                info = os.fstat(fd)
                if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                        or info.st_uid != os.geteuid()
                        or stat.S_IMODE(info.st_mode) not in {0o400, 0o600, 0o644}):
                    self._fail("path_rejected")
                if not 0 < info.st_size <= self._maximum_control:
                    self._fail("control_limit_exceeded")
                identity = _snapshot(info)
                previous = self._inputs.get(path)
                if previous is not None and previous != identity:
                    self._fail("input_changed")
                raw = self._read_fd(fd, identity, info.st_size, "input_changed")
                if (_snapshot(os.stat(path.name, dir_fd=parent, follow_symlinks=False)) != identity
                        or _snapshot(os.fstat(parent)) != parent_identity):
                    self._fail("input_changed")
                current_parent = _open_directory(path.parent)
                try:
                    if _snapshot(os.fstat(current_parent)) != parent_identity:
                        self._fail("input_changed")
                finally:
                    _close_fd(current_parent)
                self._inputs[path] = identity
                return raw
            finally:
                try:
                    if fd is not None:
                        _close_fd(fd)
                finally:
                    _close_fd(parent)

    def create_output(self, path):
        with self._operation():
            if self._root_fd is not None:
                self._fail("output_exists")
            path = _path(path)
            if any(path == source or path in source.parents for source in self._inputs):
                self._fail("path_rejected")
            parent = _open_directory(path.parent)
            try:
                parent_before = os.fstat(parent)
                current = _open_directory(path.parent)
                try:
                    if _snapshot(os.fstat(current)) != _snapshot(parent_before):
                        self._fail("path_rejected")
                finally:
                    _close_fd(current)
                try:
                    os.mkdir(path.name, mode=0o700, dir_fd=parent)
                except FileExistsError:
                    self._fail("output_exists")
                self._root_fd = os.open(path.name, _DIR_FLAGS, dir_fd=parent)
                self._root = path
                info = os.fstat(self._root_fd)
                if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
                    self._fail("path_rejected")
                os.fchmod(self._root_fd, 0o700)
                self._root_identity = _snapshot(os.fstat(self._root_fd))
                self._root_flags = _flags(self._root_fd)
                os.fsync(parent)
            finally:
                _close_fd(parent)

    def write(self, name, raw):
        with self._operation():
            name = _name(name)
            if self._root_fd is None:
                self._fail("output_not_created")
            if type(raw) is not bytes or not raw:
                self._fail("invalid_arguments")
            if len(raw) > self._maximum_file:
                self._fail("file_limit_exceeded")
            if name in _CONTROL_NAMES and len(raw) > self._maximum_control:
                self._fail("control_limit_exceeded")
            if name in self._files:
                self._fail("output_exists")
            if len(self._files) >= MAX_FILES:
                self._fail("file_limit_exceeded")
            if self._writes + len(raw) > self._limits[1]:
                self._fail("write_limit_exceeded")
            fd = None
            try:
                try:
                    fd = os.open(name, _CREATE_FLAGS, 0o600, dir_fd=self._root_fd)
                except FileExistsError:
                    self._fail("output_exists")
                os.fchmod(fd, 0o600)
                initial = os.fstat(fd)
                if (not stat.S_ISREG(initial.st_mode) or initial.st_nlink != 1
                        or initial.st_uid != os.geteuid() or initial.st_size != 0):
                    self._fail("output_changed")
                identity = _snapshot(initial)
                flags = _flags(fd)
                if (flags[0] & os.O_ACCMODE != os.O_WRONLY or flags[0] & os.O_APPEND
                        or not flags[1] & fcntl.FD_CLOEXEC):
                    self._fail("output_changed")
                self._root_identity = _snapshot(os.fstat(self._root_fd))
                for start in range(0, len(raw), CHUNK_BYTES):
                    part = raw[start:start + CHUNK_BYTES]
                    self._debit(len(part), write=True)
                    if _snapshot(os.fstat(fd)) != identity or _flags(fd) != flags:
                        self._fail("output_changed")
                    try:
                        count = os.pwrite(fd, part, start)
                    except BaseException:
                        self._fail("write_failed")
                    if type(count) is not int or count != len(part):
                        self._fail("short_write")
                    observed = os.fstat(fd)
                    observed_flags = _flags(fd)
                    if (_snapshot(observed)[:6] != identity[:6]
                            or not _write_flags_match(flags, observed_flags)
                            or observed.st_size != start + len(part)):
                        self._fail("output_changed")
                    identity = _snapshot(observed)
                    flags = observed_flags
                os.fsync(fd)
                if (_snapshot(os.fstat(fd)) != identity or _flags(fd) != flags
                        or _snapshot(os.stat(name, dir_fd=self._root_fd,
                                             follow_symlinks=False)) != identity):
                    self._fail("output_changed")
                digest = hashlib.sha256(raw).hexdigest()
                self._files[name] = {"identity": identity, "bytes": len(raw), "sha256": digest}
                os.fsync(self._root_fd)
                return {"path": name, "bytes": len(raw), "sha256": digest}
            finally:
                if fd is not None:
                    _close_fd(fd)

    def _known(self, name):
        name = _name(name)
        if self._root_fd is None:
            self._fail("output_not_created")
        if name not in self._files:
            self._fail("unknown_file")
        value = self._files[name]
        if _snapshot(os.stat(name, dir_fd=self._root_fd,
                             follow_symlinks=False)) != value["identity"]:
            self._fail("output_changed")
        return value

    def read(self, name, *, expected_sha256, expected_bytes):
        with self._operation():
            value = self._known(name)
            if (type(expected_bytes) is not int or expected_bytes != value["bytes"]
                    or type(expected_sha256) is not str
                    or expected_sha256 != value["sha256"]):
                self._fail("receipt_mismatch")
            fd = os.open(name, _READ_FLAGS, dir_fd=self._root_fd)
            try:
                raw = self._read_fd(fd, value["identity"], value["bytes"], "output_changed")
                self._known(name)
                if hashlib.sha256(raw).hexdigest() != expected_sha256:
                    self._fail("checksum_mismatch")
                return raw
            finally:
                _close_fd(fd)

    def inventory(self):
        with self._operation():
            if self._root_fd is None:
                self._fail("output_not_created")
            names = set()
            with os.scandir(self._root_fd) as entries:
                for entry in entries:
                    if len(names) >= MAX_FILES or entry.name not in self._files:
                        self._fail("output_changed")
                    names.add(entry.name)
            if names != set(self._files):
                self._fail("output_changed")
            result = []
            for name in sorted(names):
                value = self._known(name)
                result.append({"path": name, "bytes": value["bytes"], "sha256": value["sha256"]})
            return result

    def summary(self):
        with self._operation():
            return {
                "hf_content_read_bytes_reserved": self._reads,
                "hf_content_write_bytes_reserved": self._writes,
                "hf_content_byte_accounting": "cooperative_predebit_no_refund",
                "physical_io_bytes_measured": False,
                "os_resource_limits_enforced": False,
                "memory_limits_enforced": False,
                "hard_timeout_enforced": False,
            }

    def close(self):
        self._owner_check()
        if self._closed:
            return
        self._closed = True
        fd, self._root_fd = self._root_fd, None
        self._busy = True
        failure = None
        originally_failed = self._failed
        try:
            if fd is not None:
                try:
                    _close_fd(fd)
                except ContentIOError as error:
                    failure = error.code
            # Cleanup is permitted after terminal failure; the parent callback
            # must still be checked after the owned descriptor has been revoked.
            self._checks += 1
            try:
                result = self._check_budget()
                if result is not None or (self._failed and not originally_failed):
                    failure = failure or "budget_check_failed"
            except BaseException:
                failure = failure or "budget_check_failed"
            if failure is not None:
                self._fail(failure)
        finally:
            self._busy = False
