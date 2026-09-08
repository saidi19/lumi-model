"""One root-bootstrap-only Linux network namespace with loopback and no other link.

Attach this process to its owned cgroup before calling. After success the caller
must drop all privilege and close every ambient/inherited networking descriptor
before payload execution. Inherited sockets retain their original namespace;
this helper cannot revoke them, shared-UID proxies, or filesystem IPC access.
It does not provide a filesystem sandbox, host admission, or real-SFT permission.

Setup is process-local and one-shot, including failures. There is no setns,
rollback into the old namespace, retry, route/veth setup, or host sysctl change.
Only fixed codes and a fixed receipt may be logged. Suppressed exception context
can retain private OS text and must never be exposed.
"""

from __future__ import annotations

import ctypes
import fcntl
import os
import platform
import socket
import stat
import struct
import sys
import threading

from .posttraining_sft_host_observer_v1 import PROC_MAGIC, StatFS

SCHEMA = "lumi.sft-netns.v0.1"
ERROR_CODES = frozenset({
    "runtime_rejected", "already_attempted", "owner_mismatch", "reentry",
    "namespace_rejected", "namespace_unchanged", "interface_rejected",
    "setup_failed", "close_failed",
})
NONCLAIMS = (
    "full_network_isolation_verified", "filesystem_isolation_verified",
    "complete_os_sandbox_verified", "real_sft_execution_permitted", "production_eligible",
)
ML_ROOTS = frozenset({"torch", "numpy", "transformers", "safetensors", "tokenizers",
                      "tensorflow", "jax", "cupy"})
CLONE_NEWNET = 0x40000000
NSFS_MAGIC = 0x6E736673
NS_GET_NSTYPE = 0xB703
SIOCGIFFLAGS = 0x8913
SIOCSIFFLAGS = 0x8914
IFF_UP = 0x1
IFF_LOOPBACK = 0x8
_LO_NAME = b"lo" + b"\0" * 14
_IFREQ_SIZE = 40
_MAX_STATUS_BYTES = 16_384
_SETUP_LOCK = threading.Lock()
_ATTEMPT_OWNER = None


class NetworkIsolationError(RuntimeError):
    def __init__(self, code="setup_failed"):
        self.code = code if type(code) is str and code in ERROR_CODES else "setup_failed"
        super().__init__(self.code)


def _need(value, code):
    if not value:
        raise NetworkIsolationError(code) from None


def _runtime():
    _need(sys.platform == "linux" and platform.machine() == "x86_64"
          and sys.implementation.name == "cpython" and sys.byteorder == "little"
          and sys.flags.no_site == 1 and sys.dont_write_bytecode
          and ctypes.sizeof(ctypes.c_void_p) == 8 and ctypes.sizeof(ctypes.c_long) == 8
          and ctypes.sizeof(StatFS) == 120 and StatFS.fsid.offset == 56
          and StatFS.spare.offset == 88
          and hasattr(os, "getresuid") and hasattr(os, "getresgid")
          and os.getresuid() == (0, 0, 0) and os.getresgid() == (0, 0, 0)
          and threading.current_thread() is threading.main_thread()
          and threading.active_count() == 1
          and not any(name.split(".", 1)[0] in ML_ROOTS for name in sys.modules)
          and hasattr(socket, "SOCK_CLOEXEC"), "runtime_rejected")
    _kernel_single_thread()


def _filesystem(fd, expected_magic):
    libc = ctypes.CDLL(None, use_errno=True)
    call = libc.fstatfs
    call.argtypes, call.restype = [ctypes.c_int, ctypes.POINTER(StatFS)], ctypes.c_int
    result = StatFS()
    _need(call(fd, ctypes.byref(result)) == 0 and result.kind == expected_magic,
          "namespace_rejected")


def _namespace_identity(fd):
    _need(type(fd) is int and fd >= 0, "namespace_rejected")
    info = os.fstat(fd)
    _need(stat.S_ISREG(info.st_mode) and info.st_uid == 0 and info.st_gid == 0
          and info.st_ino > 0 and fcntl.fcntl(fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC,
          "namespace_rejected")
    _filesystem(fd, NSFS_MAGIC)
    namespace_type = fcntl.ioctl(fd, NS_GET_NSTYPE, 0)
    _need(type(namespace_type) is int and namespace_type == CLONE_NEWNET,
          "namespace_rejected")
    return info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid


def _close_owned(fd):
    # Caller relinquishes its reference before this single, possibly ambiguous close.
    try:
        os.close(fd)
    except BaseException:
        raise NetworkIsolationError("close_failed") from None


def _kernel_single_thread():
    # Python's thread registry alone cannot account for native/unregistered
    # threads. Read only this task's authenticated procfs status, before setup.
    proc_fd = status_fd = None
    failure = None
    try:
        proc_fd = os.open("/proc", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                          | os.O_CLOEXEC | os.O_NONBLOCK)
        proc_info = os.fstat(proc_fd)
        _need(stat.S_ISDIR(proc_info.st_mode) and proc_info.st_uid == 0
              and proc_info.st_gid == 0
              and fcntl.fcntl(proc_fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC,
              "runtime_rejected")
        _filesystem(proc_fd, PROC_MAGIC)
        status_fd = os.open("self/status", os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
                            | os.O_NONBLOCK, dir_fd=proc_fd)
        status_info = os.fstat(status_fd)
        _need(stat.S_ISREG(status_info.st_mode) and status_info.st_uid == 0
              and status_info.st_gid == 0 and status_info.st_dev == proc_info.st_dev
              and fcntl.fcntl(status_fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC,
              "runtime_rejected")
        _filesystem(status_fd, PROC_MAGIC)
        chunks, size = [], 0
        for _ in range(128):
            chunk = os.read(status_fd, min(4096, _MAX_STATUS_BYTES + 1 - size))
            _need(type(chunk) is bytes, "runtime_rejected")
            size += len(chunk)
            _need(size <= _MAX_STATUS_BYTES, "runtime_rejected")
            if not chunk:
                break
            chunks.append(chunk)
        else:
            raise NetworkIsolationError("runtime_rejected") from None
        raw = b"".join(chunks)
        _need(raw and raw.endswith(b"\n") and b"\0" not in raw, "runtime_rejected")
        values = [line.partition(b":")[2].strip(b" \t") for line in raw.split(b"\n")
                  if line.partition(b":")[0] == b"Threads"]
        _need(values == [b"1"], "runtime_rejected")
    except BaseException as error:
        failure = "close_failed" if (type(error) is NetworkIsolationError
                                    and error.code == "close_failed") else "runtime_rejected"
    finally:
        for fd in (status_fd, proc_fd):
            if fd is not None:
                try:
                    _close_owned(fd)
                except BaseException:
                    failure = "close_failed"
        status_fd = proc_fd = None
    if failure is not None:
        raise NetworkIsolationError(failure) from None


def _open_namespace():
    proc_fd = namespace_fd = None
    failure = None
    try:
        proc_fd = os.open("/proc", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                          | os.O_CLOEXEC | os.O_NONBLOCK)
        info = os.fstat(proc_fd)
        _need(stat.S_ISDIR(info.st_mode) and info.st_uid == 0 and info.st_gid == 0
              and fcntl.fcntl(proc_fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC,
              "namespace_rejected")
        _filesystem(proc_fd, PROC_MAGIC)
        # Both kernel-owned self and ns/net symlinks are intentionally followed.
        # The procfs descriptor and the resulting nsfs object are authenticated.
        namespace_fd = os.open("self/ns/net", os.O_RDONLY | os.O_CLOEXEC,
                               dir_fd=proc_fd)
        _namespace_identity(namespace_fd)
    except BaseException as error:
        failure = error.code if type(error) is NetworkIsolationError else "setup_failed"
    if proc_fd is not None:
        closing, proc_fd = proc_fd, None
        try:
            _close_owned(closing)
        except BaseException:
            failure = "close_failed"
    if failure is not None:
        if namespace_fd is not None:
            closing, namespace_fd = namespace_fd, None
            try:
                _close_owned(closing)
            except BaseException:
                failure = "close_failed"
        raise NetworkIsolationError(failure) from None
    return namespace_fd


def _unshare():
    libc = ctypes.CDLL(None, use_errno=True)
    call = libc.unshare
    call.argtypes, call.restype = [ctypes.c_int], ctypes.c_int
    _need(call(CLONE_NEWNET) == 0, "setup_failed")


def _get_flags(fd):
    request = struct.pack("<16sH22x", _LO_NAME, 0)
    result = fcntl.ioctl(fd, SIOCGIFFLAGS, request)
    _need(type(result) is bytes and len(result) == _IFREQ_SIZE
          and result[:16] == _LO_NAME, "interface_rejected")
    flags = struct.unpack_from("<H", result, 16)[0]
    _need(flags & IFF_LOOPBACK, "interface_rejected")
    return flags


def _only_loopback():
    names = socket.if_nameindex()
    _need(type(names) is list and len(names) == 1 and type(names[0]) is tuple
          and len(names[0]) == 2 and type(names[0][0]) is int
          and 0 < names[0][0] < 2**31 and type(names[0][1]) is str
          and names[0][1] == "lo", "interface_rejected")


def _configure_loopback():
    control = None
    failure = None
    try:
        # Construct only after unshare. Never use a socket from the old namespace.
        control = socket.socket(socket.AF_INET, socket.SOCK_DGRAM | socket.SOCK_CLOEXEC, 0)
        fd = control.fileno()
        _need(type(fd) is int and fd >= 0
              and fcntl.fcntl(fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC, "interface_rejected")
        _only_loopback()
        flags = _get_flags(fd)
        request = struct.pack("<16sH22x", _LO_NAME, flags | IFF_UP)
        result = fcntl.ioctl(fd, SIOCSIFFLAGS, request)
        _need(type(result) is bytes and len(result) == _IFREQ_SIZE
              and result[:16] == _LO_NAME, "interface_rejected")
        _need(_get_flags(fd) & (IFF_LOOPBACK | IFF_UP) == IFF_LOOPBACK | IFF_UP,
              "interface_rejected")
        _only_loopback()
    except BaseException as error:
        failure = error.code if type(error) is NetworkIsolationError else "setup_failed"
    if control is not None:
        closing, control = control, None
        try:
            closing.close()
        except BaseException:
            failure = "close_failed"
    if failure is not None:
        raise NetworkIsolationError(failure) from None


def create_loopback_namespace():
    """Create once before privilege drop; return only after owned FDs are closed.

    A failure after setup begins is terminal in this process and its descendants.
    The caller must exit rather than attempt recovery or execute any payload.
    The old namespace FD is never used to restore prior networking authority.
    """
    global _ATTEMPT_OWNER
    _need(_SETUP_LOCK.acquire(blocking=False), "reentry")
    old_fd = new_fd = None
    failure = None
    try:
        owner = os.getpid(), threading.get_ident()
        if _ATTEMPT_OWNER is not None:
            _need(_ATTEMPT_OWNER == owner, "owner_mismatch")
            raise NetworkIsolationError("already_attempted") from None
        _runtime()
        # Consume before any descriptor setup or native mutation, including a
        # failed unshare. A fork inherits this consumed state and cannot reuse it.
        _ATTEMPT_OWNER = owner
        old_fd = _open_namespace()
        old_identity = _namespace_identity(old_fd)
        _unshare()
        new_fd = _open_namespace()
        new_identity = _namespace_identity(new_fd)
        _need(old_identity[:2] != new_identity[:2], "namespace_unchanged")
        _configure_loopback()
        _need(_namespace_identity(old_fd) == old_identity
              and _namespace_identity(new_fd) == new_identity, "namespace_rejected")
    except BaseException as error:
        failure = error.code if type(error) is NetworkIsolationError else "setup_failed"
    finally:
        for fd in (new_fd, old_fd):
            if fd is not None:
                try:
                    _close_owned(fd)
                except BaseException:
                    failure = "close_failed"
        new_fd = old_fd = None
        _SETUP_LOCK.release()
    if failure is not None:
        raise NetworkIsolationError(failure) from None
    return {"schema": SCHEMA, "namespace_changed": True, "loopback_up": True,
            "only_loopback_interface": True, "namespace_fds_closed": True,
            **dict.fromkeys(NONCLAIMS, False)}
