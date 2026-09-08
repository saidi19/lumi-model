"""Exact owned Linux cgroup2 resource lease; no worker launch or ML admission.

The root caller supplies an authenticated borrowed parent descriptor. This
module neither enables parent controllers nor changes delegation/ownership.
An independent guardian must hold separate cleanup authority before a bootstrap
is released. A worker must drop privilege and close cgroup descriptors before ML;
this module alone does not prevent migration, filesystem or network escape.

Only fixed errors and aggregate snapshots may be reported. Suppressed exception
contexts can retain private kernel/path text and must never be logged. Failures
allow only identity-checked cleanup snapshots, one kill and empty-only close;
ambiguous writes and closes are not retried.
"""

from __future__ import annotations

import ctypes
import fcntl
import os
import re
import stat
import sys
import threading

from .posttraining_sft_host_observer_v1 import CGROUP_MAGIC, StatFS

ERROR_CODES = frozenset({
    "unsupported", "invalid_arguments", "invalid_descriptor", "descriptor_changed",
    "owner_mismatch", "already_exists", "already_killed", "closed", "terminal_failure",
    "configuration_rejected", "snapshot_rejected", "not_empty", "io_failed", "close_failed",
})
LIMIT_FIELDS = frozenset({"memory_max_bytes", "pids_max", "cpu_quota_us", "cpu_period_us"})
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
_FILE_FLAGS = os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
_MAX_SCALAR = 2**64 - 1


class CgroupError(RuntimeError):
    def __init__(self, code="io_failed"):
        self.code = code if type(code) is str and code in ERROR_CODES else "io_failed"
        super().__init__(self.code)


def _need(value, code):
    if not value:
        raise CgroupError(code) from None


def _runtime():
    _need(sys.platform == "linux" and os.uname().machine == "x86_64" and os.geteuid() == 0
          and sys.byteorder == "little" and ctypes.sizeof(ctypes.c_void_p) == 8
          and ctypes.sizeof(StatFS) == 120 and StatFS.fsid.offset == 56
          and StatFS.spare.offset == 88, "unsupported")


def _identity(info):
    return info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid


def _filesystem(fd):
    libc = ctypes.CDLL(None, use_errno=True)
    call = libc.fstatfs
    call.argtypes, call.restype = [ctypes.c_int, ctypes.POINTER(StatFS)], ctypes.c_int
    result = StatFS()
    _need(call(fd, ctypes.byref(result)) == 0 and result.kind == CGROUP_MAGIC,
          "invalid_descriptor")


def _private(info, directory):
    return ((stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))
            and info.st_uid == 0 and not stat.S_IMODE(info.st_mode) & 0o022)


def _directory(fd):
    _need(type(fd) is int and fd >= 3, "invalid_descriptor")
    info = os.fstat(fd)
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    required = os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK
    _need(_private(info, True) and flags & os.O_ACCMODE == os.O_RDONLY
          and flags & required == required and not flags & os.O_APPEND
          and fcntl.fcntl(fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC, "invalid_descriptor")
    _filesystem(fd)
    return _identity(info)


def _open_file(directory, name, *, write):
    before = os.stat(name, dir_fd=directory, follow_symlinks=False)
    _need(_private(before, False), "invalid_descriptor")
    fd = os.open(name, (os.O_WRONLY if write else os.O_RDONLY) | _FILE_FLAGS, dir_fd=directory)
    try:
        info = os.fstat(fd)
        _need(_identity(info) == _identity(before)
              and info.st_dev == os.fstat(directory).st_dev, "descriptor_changed")
        _filesystem(fd)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _read(directory, name, maximum=4096):
    fd = _open_file(directory, name, write=False)
    try:
        chunks, size = [], 0
        for _ in range(128):
            chunk = os.read(fd, min(4096, maximum + 1 - size))
            _need(type(chunk) is bytes, "snapshot_rejected")
            size += len(chunk)
            _need(size <= maximum, "snapshot_rejected")
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
        raise CgroupError("snapshot_rejected") from None
    finally:
        os.close(fd)


def _write(directory, name, payload):
    fd = _open_file(directory, name, write=True)
    try:
        _need(os.write(fd, payload) == len(payload), "io_failed")
    finally:
        os.close(fd)


def _number(raw):
    _need(type(raw) is bytes and re.fullmatch(rb"(?:0|[1-9][0-9]{0,19})", raw),
          "snapshot_rejected")
    value = int(raw)
    _need(value <= _MAX_SCALAR, "snapshot_rejected")
    return value


def _scalar(raw):
    _need(type(raw) is bytes and raw.endswith(b"\n"), "snapshot_rejected")
    return _number(raw[:-1])


def _events(raw, required):
    _need(type(raw) is bytes and 0 < len(raw) <= 4096 and raw.endswith(b"\n"),
          "snapshot_rejected")
    lines = raw[:-1].split(b"\n")
    _need(len(lines) <= 64, "snapshot_rejected")
    result = {}
    for line in lines:
        pair = line.split(b" ")
        _need(len(pair) == 2 and re.fullmatch(rb"[a-z][a-z0-9_.]{0,63}", pair[0]),
              "snapshot_rejected")
        key = pair[0].decode("ascii")
        _need(key not in result, "snapshot_rejected")
        result[key] = _number(pair[1])
    _need(required <= result.keys(), "snapshot_rejected")
    return result


def _populated(fd):
    events = _events(_read(fd, "cgroup.events"), {"populated", "frozen"})
    _need(events["populated"] in (0, 1) and events["frozen"] in (0, 1), "snapshot_rejected")
    return events["populated"]


def _pids(raw):
    _need(type(raw) is bytes and len(raw) <= 65_536
          and (raw == b"" or raw.endswith(b"\n")), "snapshot_rejected")
    result = set()
    for line in raw[:-1].split(b"\n") if raw else ():
        pid = _number(line)
        _need(0 < pid < 2**31 and pid not in result, "snapshot_rejected")
        result.add(pid)
    return result


def _settings(limits):
    _need(type(limits) is dict and all(type(key) is str for key in limits)
          and set(limits) == LIMIT_FIELDS, "invalid_arguments")
    bounds = {"memory_max_bytes": (16 * 1024**2, 1024**4), "pids_max": (4, 65_536),
              "cpu_quota_us": (1000, 16_000_000), "cpu_period_us": (1000, 1_000_000)}
    _need(all(type(limits[key]) is int and low <= limits[key] <= high
              for key, (low, high) in bounds.items())
          and limits["cpu_quota_us"] <= limits["cpu_period_us"] * 128, "invalid_arguments")
    return {"memory.max": f"{limits['memory_max_bytes']}\n".encode(), "memory.swap.max": b"0\n",
            "memory.oom.group": b"1\n", "pids.max": f"{limits['pids_max']}\n".encode(),
            "cpu.max": f"{limits['cpu_quota_us']} {limits['cpu_period_us']}\n".encode(),
            "cpu.max.burst": b"0\n"}


class CgroupLease:
    """One root-owned, exclusive empty child of a borrowed cgroup2 parent FD."""

    def __init__(self, parent_fd, *, name, limits):
        if hasattr(self, "_owner"):
            self._fail("invalid_arguments")
        self._owner = os.getpid(), threading.get_ident()
        self._fd, self._parent, self._leaf_identity = None, parent_fd, None
        self._failed, self._closed, self._killed, self._created = False, False, False, False
        try:
            _runtime()
            _need(type(name) is str and re.fullmatch(r"lumi-sft-[a-f0-9]{32}", name),
                  "invalid_arguments")
            self._settings, self._name = _settings(limits), name
            self._parent_identity = _directory(parent_fd)
            try:
                os.mkdir(name, mode=0o700, dir_fd=parent_fd)
            except FileExistsError:
                raise CgroupError("already_exists") from None
            self._created = True
            info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            _need(_private(info, True), "invalid_descriptor")
            self._leaf_identity = _identity(info)
            _need(self._leaf_identity[0] == self._parent_identity[0]
                  and self._leaf_identity[:2] != self._parent_identity[:2], "descriptor_changed")
            self._fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
            self._check()
            _need(_populated(self._fd) == 0 and not _pids(_read(self._fd, "cgroup.procs", 65_536)),
                  "not_empty")
            _need(_read(self._fd, "cgroup.type") == b"domain\n", "configuration_rejected")
            # Require every interface before any limit mutation; no parent enabling.
            for filename in (*self._settings, "cgroup.kill"):
                fd = _open_file(self._fd, filename, write=True)
                os.close(fd)
            self.snapshot()  # Mandatory event interfaces must exist before admission.
            for filename, payload in self._settings.items():
                self._check()
                _write(self._fd, filename, payload)
            self.verify_limits()
        except BaseException as error:
            code = error.code if type(error) is CgroupError else "io_failed"
            self._failed = True
            self._constructor_cleanup()
            raise CgroupError(code) from None

    def __repr__(self):
        return "<CgroupLease private>"

    def _fail(self, code):
        self._failed = True
        raise CgroupError(code) from None

    def _check(self, *, cleanup=False):
        _need(self._owner == (os.getpid(), threading.get_ident()), "owner_mismatch")
        _need(not self._closed and self._fd is not None, "closed")
        _need(cleanup or not self._failed, "terminal_failure")
        _need(cleanup or not self._killed, "already_killed")
        _runtime()
        _need(_directory(self._parent) == self._parent_identity
              and _directory(self._fd) == self._leaf_identity
              and _identity(os.stat(self._name, dir_fd=self._parent, follow_symlinks=False))
              == self._leaf_identity, "descriptor_changed")

    def _constructor_cleanup(self):
        if self._fd is None:
            return
        try:
            self._check(cleanup=True)
            if _populated(self._fd) == 0 and not _pids(_read(self._fd, "cgroup.procs", 65_536)):
                os.rmdir(self._name, dir_fd=self._parent)
        except BaseException:
            pass  # Retain uncertain/nonempty evidence; never kill or recursively delete.
        fd, self._fd = self._fd, None
        try:
            os.close(fd)
        except BaseException:
            pass  # Ownership is removed before a single close attempt.

    def fileno(self):
        try:
            self._check()
            return self._fd
        except BaseException as error:
            self._fail(error.code if type(error) is CgroupError else "io_failed")

    def verify_limits(self):
        try:
            self._check()
            for filename, expected in self._settings.items():
                _need(_read(self._fd, filename) == expected, "configuration_rejected")
            self._check()
        except BaseException as error:
            self._fail(error.code if type(error) is CgroupError else "io_failed")

    def snapshot(self):
        try:
            self._check(cleanup=True)
            memory = _events(_read(self._fd, "memory.events"),
                             {"max", "oom", "oom_kill", "oom_group_kill"})
            pids = _events(_read(self._fd, "pids.events"), {"max"})
            cpu = _events(_read(self._fd, "cpu.stat"),
                          {"usage_usec", "nr_throttled", "throttled_usec"})
            result = {
                "populated": _populated(self._fd),
                "pids_current": _scalar(_read(self._fd, "pids.current")),
                "pids_max_events": pids["max"],
                "memory_current_bytes": _scalar(_read(self._fd, "memory.current")),
                "memory_max_events": memory["max"], "memory_oom_events": memory["oom"],
                "memory_oom_kill_events": memory["oom_kill"],
                "memory_oom_group_kill_events": memory["oom_group_kill"],
                "cpu_usage_usec": cpu["usage_usec"], "cpu_nr_throttled": cpu["nr_throttled"],
                "cpu_throttled_usec": cpu["throttled_usec"],
            }
            self._check(cleanup=True)
            return result
        except BaseException as error:
            self._fail(error.code if type(error) is CgroupError else "io_failed")

    def kill(self):
        try:
            self._check(cleanup=True)
            _need(not self._killed, "already_killed")
            self._killed = True  # Ambiguous/short writes are never attempted again.
            _write(self._fd, "cgroup.kill", b"1\n")
            self._check(cleanup=True)
        except BaseException as error:
            self._fail(error.code if type(error) is CgroupError else "io_failed")

    def close(self):
        if self._closed:
            return
        try:
            self._check(cleanup=True)
            _need(_populated(self._fd) == 0 and not _pids(_read(self._fd, "cgroup.procs", 65_536)),
                  "not_empty")
            self._check(cleanup=True)
            os.rmdir(self._name, dir_fd=self._parent)
            fd, self._fd, self._closed = self._fd, None, True
            os.close(fd)
        except BaseException as error:
            self._fail(error.code if type(error) is CgroupError else "close_failed")


def enter_current_process(group_fd):
    """Attach only this newly forked root bootstrap; borrow FD, never change limits."""
    try:
        _runtime()
        identity = _directory(group_fd)
        _need(_read(group_fd, "cgroup.type") == b"domain\n", "configuration_rejected")
        _write(group_fd, "cgroup.procs", b"0\n")
        _need(os.getpid() in _pids(_read(group_fd, "cgroup.procs", 65_536)),
              "configuration_rejected")
        _need(_directory(group_fd) == identity, "descriptor_changed")
    except BaseException as error:
        raise CgroupError(error.code if type(error) is CgroupError else "io_failed") from None
