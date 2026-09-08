"""Privileged, stdlib-only guardian outside one already-owned workload cgroup.

The caller creates its exact CgroupLease in this guardian process, forks one
blocked bootstrap, attaches it before ML/descendants, and supplies release().
Only after verifying limits, populated state and a live *actual parent* pidfd
does the guardian release that bootstrap. Parent loss or the absolute monotonic
deadline kills the entire owned cgroup, including setsid descendants. A natural
empty group is retired too; it is NOT proof of successful worker exit/results.

Workers must lose all root identities/groups, set no_new_privs and close every
cgroup/pidfd authority before execution. Root-owned non-writable cgroup controls
then remain outside their authority. This is not a filesystem/network sandbox,
GPU memory limiter, billing guarantee, host admission or arbitrary-root-kill
defense. The guardian itself needs an independently bounded outer owner/service.
Kernel scheduling/uninterruptible calls can outlast userspace cleanup deadlines.

Only fixed summaries/errors are public. No process IDs, paths, data or suppressed
exception text may be logged. No retry, deadline reset, external process scan,
numeric-PID signal, global cgroup mutation, or paid-instance action exists here.
"""

from __future__ import annotations

import ctypes
import fcntl
import os
import platform
import select
import signal
import sys
import threading
import time

from .posttraining_sft_cgroup_v1 import CgroupLease
from .posttraining_sft_child_status_v1 import MAXIMUM_STATUS_BYTES, parse_status
from .posttraining_sft_host_observer_v1 import PROC_MAGIC, StatFS

SCHEMA = "lumi.sft-cgroup-guardian.v0.1"
ERROR_CODES = frozenset({
    "invalid_arguments", "runtime_rejected", "parent_binding_rejected", "pidfd_rejected",
    "lease_rejected", "release_rejected", "guardian_failed", "cleanup_failed",
    "privilege_drop_rejected", "reentry",
})
REASONS = frozenset({"workload_empty", "parent_lost", "deadline_exhausted"})
NONCLAIMS = (
    "worker_exit_success_verified", "network_isolation_verified", "filesystem_isolation_verified",
    "disk_write_quota_verified", "gpu_memory_enforcement", "billing_enforcement",
    "guardian_death_containment", "real_sft_execution_permitted", "production_eligible",
)
ML_ROOTS = frozenset({"torch", "numpy", "transformers", "safetensors", "tokenizers",
                      "tensorflow", "jax", "cupy"})
_WATCH_LOCK = threading.Lock()


class GuardianError(RuntimeError):
    def __init__(self, code="guardian_failed"):
        self.code = code if type(code) is str and code in ERROR_CODES else "guardian_failed"
        super().__init__(self.code)


def _need(value, code):
    if not value:
        raise GuardianError(code) from None


def _runtime_check():
    _need(sys.platform == "linux" and platform.machine() == "x86_64"
          and sys.implementation.name == "cpython" and sys.flags.no_site == 1
          and sys.dont_write_bytecode and ctypes.sizeof(ctypes.c_void_p) == 8
          and os.geteuid() == 0 and os.getegid() == 0
          and threading.current_thread() is threading.main_thread()
          and threading.active_count() == 1
          and not any(name.split(".", 1)[0] in ML_ROOTS for name in sys.modules)
          and hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal")
          and hasattr(select, "poll"), "runtime_rejected")


def _pidfd_identity(fd):
    _need(type(fd) is int and fd >= 3, "pidfd_rejected")
    flags = fcntl.fcntl(fd, fcntl.F_GETFD)
    _need(bool(flags & fcntl.FD_CLOEXEC), "pidfd_rejected")
    info = os.fstat(fd)
    return info.st_dev, info.st_ino, info.st_mode, flags


def _parent_alive(fd):
    try:
        # Signal zero checks the pidfd object/liveness without signaling a PID.
        signal.pidfd_send_signal(fd, 0, None, 0)
        return True
    except ProcessLookupError:
        return False


def _populated(lease):
    value = lease.snapshot()
    _need(type(value) is dict and type(value.get("populated")) is int
          and value["populated"] in (0, 1), "lease_rejected")
    return value["populated"]


def _close_pidfd(fd, identity):
    # Revoke caller ownership before entering here. Never retry an ambiguous FD.
    _need(_pidfd_identity(fd) == identity, "pidfd_rejected")
    os.close(fd)


def _retire(lease, *, cleanup_timeout_ms):
    # This routine is invoked once, even on a failed release/binding. It cannot
    # relax/recreate a lease or substitute killpg for an unavailable cgroup.kill.
    cleanup_deadline = time.monotonic_ns() + cleanup_timeout_ms * 1_000_000
    lease.kill()
    while True:
        if _populated(lease) == 0:
            return
        remaining = cleanup_deadline - time.monotonic_ns()
        _need(remaining > 0, "cleanup_failed")
        time.sleep(min(0.01, remaining / 1_000_000_000))


def watch_parent(lease, *, expected_parent_pid, deadline_ns, cleanup_timeout_ms, release):
    """Arm once, release once, retire once; caller then reaps and closes its lease.

    Parent PID is used only to open a generation-bound pidfd while bracketed by
    getppid checks. No bare numeric-PID signaling/reaping is performed. A parent
    lost before arming retires the blocked workload without calling release.
    An invalid argument/runtime does not grant authority to touch any lease.
    """
    _runtime_check()
    _need(type(lease) is CgroupLease
          and type(expected_parent_pid) is int and 1 < expected_parent_pid < 2**31
          and type(deadline_ns) is int and 0 < deadline_ns < 2**63
          and type(cleanup_timeout_ms) is int and 1 <= cleanup_timeout_ms <= 10_000
          and callable(release), "invalid_arguments")
    _need(_WATCH_LOCK.acquire(blocking=False), "reentry")
    fd = identity = None
    owns_lease, bound, released, killed, emptied, closed = False, False, False, False, False, False
    reason, failure = None, None
    try:
        # verify_limits includes lease process/thread/descriptor ownership. A
        # failure here cannot establish authority to kill an arbitrary object.
        lease.verify_limits()
        owns_lease = True
        _need(_populated(lease) == 1, "lease_rejected")
        if time.monotonic_ns() >= deadline_ns:
            reason = "deadline_exhausted"
        elif os.getppid() != expected_parent_pid:
            reason = "parent_lost"
        else:
            try:
                fd = os.pidfd_open(expected_parent_pid, 0)
            except ProcessLookupError:
                reason = "parent_lost"
            if fd is not None:
                identity = _pidfd_identity(fd)
                poller = select.poll()
                poller.register(fd, select.POLLIN | select.POLLHUP | select.POLLERR)
                if os.getppid() != expected_parent_pid or not _parent_alive(fd):
                    reason = "parent_lost"
                else:
                    bound = True
                    if time.monotonic_ns() >= deadline_ns:
                        reason = "deadline_exhausted"
                    else:
                        released = True  # A raised/partial callback is consumed too.
                        _need(release() is None, "release_rejected")
                while reason is None:
                    # Death/deadline take precedence over an apparently empty
                    # workload, so late success cannot erase an exhausted run.
                    _need(_pidfd_identity(fd) == identity, "pidfd_rejected")
                    if os.getppid() != expected_parent_pid or not _parent_alive(fd):
                        reason = "parent_lost"
                    elif time.monotonic_ns() >= deadline_ns:
                        reason = "deadline_exhausted"
                    elif _populated(lease) == 0:
                        reason = "workload_empty"
                    else:
                        remaining_ms = max(1, (deadline_ns - time.monotonic_ns()) // 1_000_000)
                        events = poller.poll(min(50, remaining_ms))
                        for number, flags in events:
                            _need(number == fd and not flags & select.POLLNVAL, "pidfd_rejected")
                            if flags & (select.POLLIN | select.POLLHUP | select.POLLERR):
                                reason = "parent_lost"
        _need(reason in REASONS, "guardian_failed")
    except GuardianError as error:
        failure = error.code
    except BaseException:
        failure = "guardian_failed"
    finally:
        if owns_lease:
            killed = True
            try:
                _retire(lease, cleanup_timeout_ms=cleanup_timeout_ms)
                emptied = True
            except BaseException:
                failure = "cleanup_failed"
        if fd is not None:
            to_close, fd = fd, None
            try:
                if identity is None:
                    # Fresh pidfd_open ownership: no callback or transfer has
                    # occurred before the initial identity capture. Close once.
                    os.close(to_close)
                else:
                    _close_pidfd(to_close, identity)
                closed = True
            except BaseException:
                failure = "cleanup_failed"
        _WATCH_LOCK.release()
    if failure is not None:
        raise GuardianError(failure) from None
    return {
        "schema_version": SCHEMA, "reason": reason, "parent_pidfd_bound": bound,
        "release_called": released, "kill_attempted": killed, "empty_verified": emptied,
        "pidfd_closed": closed, **dict.fromkeys(NONCLAIMS, False),
    }


def _verify_dropped_status(uid, gid):
    """Bounded trusted self-procfs snapshot before ML or any further fork."""
    parent = os.getppid()
    fd = os.open("/proc/self/status", os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        fstatfs = libc.fstatfs
        fstatfs.argtypes, fstatfs.restype = [ctypes.c_int, ctypes.POINTER(StatFS)], ctypes.c_int
        filesystem = StatFS()
        _need(fstatfs(fd, ctypes.byref(filesystem)) == 0 and filesystem.kind == PROC_MAGIC,
              "privilege_drop_rejected")
        chunks, size = [], 0
        for _ in range(128):
            chunk = os.read(fd, min(4096, MAXIMUM_STATUS_BYTES + 1 - size))
            size += len(chunk)
            _need(size <= MAXIMUM_STATUS_BYTES, "privilege_drop_rejected")
            if not chunk:
                break
            chunks.append(chunk)
        else:
            raise GuardianError("privilege_drop_rejected") from None
        parse_status(b"".join(chunks), expected_pid=os.getpid(), expected_ppid=parent,
                     expected_uid=uid, expected_gid=gid)
        _need(os.getppid() == parent, "privilege_drop_rejected")
    finally:
        os.close(fd)


def drop_worker_privileges(uid, gid):
    """Irreversibly leave root in a newly attached single-thread bootstrap.

    Call before ML/threads/descendants, close all cgroup/pidfd authority, then
    exec the independently pinned worker under a fixed environment. This helper
    does not close arbitrary caller FDs or establish filesystem/network isolation.
    It must not be used as evidence that the guardian or source was authorized.
    """
    _runtime_check()
    _need(type(uid) is int and 1 <= uid < 2**31
          and type(gid) is int and 1 <= gid < 2**31, "invalid_arguments")
    try:
        _need(os.getresuid() == (0, 0, 0) and os.getresgid() == (0, 0, 0),
              "privilege_drop_rejected")
        libc = ctypes.CDLL(None, use_errno=True)
        prctl = libc.prctl
        prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                          ctypes.c_ulong, ctypes.c_ulong]
        prctl.restype = ctypes.c_int
        _need(prctl(7, 0, 0, 0, 0) == 0, "privilege_drop_rejected")  # PR_GET_KEEPCAPS
        _need(prctl(38, 1, 0, 0, 0) == 0, "privilege_drop_rejected")  # PR_SET_NO_NEW_PRIVS
        os.setgroups([])
        os.setresgid(gid, gid, gid)
        os.setresuid(uid, uid, uid)
        _need(os.getresuid() == (uid, uid, uid) and os.getresgid() == (gid, gid, gid)
              and os.getgroups() == [] and prctl(39, 0, 0, 0, 0) == 1,
              "privilege_drop_rejected")  # PR_GET_NO_NEW_PRIVS
        _verify_dropped_status(uid, gid)
    except BaseException:
        raise GuardianError("privilege_drop_rejected") from None
