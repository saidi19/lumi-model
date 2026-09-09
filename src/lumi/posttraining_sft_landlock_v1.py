"""One root-bootstrap-only, fail-closed filesystem Landlock policy.

Call only after the trusted mount/pivot-root allowlist is complete and before
dropping worker privilege. Read-only mounts must enforce metadata restrictions;
Landlock alone does not cover all filesystem operations. ABI 5 does not restrict
pathname UNIX socket lookup, including sockets inside an exposed read-only tree.
Those trees remain trusted inputs. Every inherited authority FD must separately
be closed before payload execution. No real-SFT admission is granted here.

Only fixed codes and the fixed receipt may be logged. Suppressed exception
context can retain private OS text and must never be exposed. Setup is one-shot,
including a failed ABI query: callers must exit after any started failure.
"""

from __future__ import annotations

import ctypes
import fcntl
import os
import re
import stat
import threading

from .posttraining_sft_mount_root_v1 import TMPFS_MAGIC, _filesystem
from .posttraining_sft_netns_v1 import _runtime

SCHEMA = "lumi.sft-landlock.v0.1"
ERROR_CODES = frozenset({
    "runtime_rejected", "paths_rejected", "path_rejected", "device_rejected",
    "already_attempted", "owner_mismatch", "reentry", "abi_rejected",
    "ruleset_rejected", "rule_rejected", "privilege_rejected", "restrict_failed",
    "setup_failed", "close_failed",
})
O_PATH = 0x200000
SYS_CREATE_RULESET, SYS_ADD_RULE, SYS_RESTRICT_SELF = 444, 445, 446
LANDLOCK_CREATE_RULESET_VERSION, LANDLOCK_RULE_PATH_BENEATH = 1, 1
EXECUTE, WRITE_FILE, READ_FILE, READ_DIR = 1, 2, 4, 8
REMOVE_DIR, REMOVE_FILE, MAKE_CHAR, MAKE_DIR = 16, 32, 64, 128
MAKE_REG, MAKE_SOCK, MAKE_FIFO, MAKE_BLOCK = 256, 512, 1024, 2048
MAKE_SYM, REFER, TRUNCATE, IOCTL_DEV = 4096, 8192, 16384, 32768
HANDLED_FS = (1 << 16) - 1
READ_ONLY_FILE = READ_FILE | EXECUTE
READ_ONLY_DIRECTORY = READ_ONLY_FILE | READ_DIR
WRITABLE_DIRECTORY = (READ_FILE | READ_DIR | WRITE_FILE | REMOVE_DIR | REMOVE_FILE
                      | MAKE_DIR | MAKE_REG | MAKE_SOCK | MAKE_FIFO | MAKE_SYM
                      | REFER | TRUNCATE)
DEVICES = {"/dev/null": (1, 3, READ_FILE | WRITE_FILE),
           "/dev/zero": (1, 5, READ_FILE | WRITE_FILE),
           "/dev/random": (1, 8, READ_FILE), "/dev/urandom": (1, 9, READ_FILE)}
_COMPONENT = re.compile(r"[A-Za-z0-9_.+-]{1,255}\Z", re.ASCII)
_SETUP_LOCK = threading.Lock()
_ATTEMPT_OWNER = None


class LandlockError(RuntimeError):
    def __init__(self, code="setup_failed"):
        self.code = code if type(code) is str and code in ERROR_CODES else "setup_failed"
        super().__init__(self.code)


class RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def _need(value, code):
    if not value:
        raise LandlockError(code) from None


def _close(fd):
    try:
        os.close(fd)
    except BaseException:
        raise LandlockError("close_failed") from None


def _paths(read_only_paths, writable_paths, device_paths):
    groups = read_only_paths, writable_paths, device_paths
    _need(all(type(group) is tuple for group in groups), "paths_rejected")
    _need(0 < sum(map(len, groups)) <= 16 and read_only_paths, "paths_rejected")
    combined = sum(groups, ())
    for path in combined:
        _need(type(path) is str and 1 < len(path) <= 4096 and path.startswith("/"),
              "paths_rejected")
        parts = path[1:].split("/")
        _need(len(parts) <= 64 and all(part not in (".", "..") and _COMPONENT.fullmatch(part)
                                      for part in parts), "paths_rejected")
    _need(len(set(combined)) == len(combined), "paths_rejected")
    _need(all(path in DEVICES for path in device_paths), "paths_rejected")
    for left in writable_paths + device_paths:
        others = tuple(path for path in combined if path != left)
        _need(all(not left.startswith(right + "/") and not right.startswith(left + "/")
                  for right in others), "paths_rejected")


def _fd_info(fd):
    _need(type(fd) is int and 0 <= fd < 2**31, "path_rejected")
    info = os.fstat(fd)
    _need(info.st_ino > 0 and info.st_dev >= 0
          and fcntl.fcntl(fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC
          and fcntl.fcntl(fd, fcntl.F_GETFL) & O_PATH, "path_rejected")
    return info


def _open_path(path):
    """Walk only authenticated directory FDs; O_PATH symlink leaves are denied."""
    owned = None
    failure = None
    try:
        owned = os.open("/", O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        _need(stat.S_ISDIR(_fd_info(owned).st_mode), "path_rejected")
        parts = [] if path == "/" else path[1:].split("/")
        for index, part in enumerate(parts):
            child = None
            try:
                flags = O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC
                if index != len(parts) - 1:
                    flags |= os.O_DIRECTORY
                child = os.open(part, flags, dir_fd=owned)
                info = _fd_info(child)
                _need(not stat.S_ISLNK(info.st_mode), "path_rejected")
                if index != len(parts) - 1:
                    _need(stat.S_ISDIR(info.st_mode), "path_rejected")
            except BaseException:
                if child is not None:
                    closing, child = child, None
                    _close(closing)
                raise
            previous, owned = owned, child
            _close(previous)
        result, owned = owned, None
        return result
    except BaseException as error:
        failure = error.code if type(error) is LandlockError else "path_rejected"
    finally:
        if owned is not None:
            closing, owned = owned, None
            try:
                _close(closing)
            except BaseException:
                failure = "close_failed"
    raise LandlockError(failure) from None


def _syscall(number, *arguments):
    call = ctypes.CDLL(None, use_errno=True).syscall
    # syscall is variadic: the fixed syscall number and every variadic value
    # have explicit native-width ctypes. Pointer values must not truncate to int.
    call.argtypes, call.restype = [ctypes.c_long], ctypes.c_long
    return call(number, *arguments)


def _no_new_privileges():
    call = ctypes.CDLL(None, use_errno=True).prctl
    call.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                     ctypes.c_ulong, ctypes.c_ulong]
    call.restype = ctypes.c_int
    _need(call(38, 1, 0, 0, 0) == 0 and call(39, 0, 0, 0, 0) == 1,
          "privilege_rejected")


def _private_root(fd, info):
    _need(stat.S_ISDIR(info.st_mode) and info.st_uid == info.st_gid == 0
          and stat.S_IMODE(info.st_mode) == 0o755, "path_rejected")
    _filesystem(fd, TMPFS_MAGIC, "root_rejected")
    view = os.fstatvfs(fd)
    _need(view.f_flag & 7 == 7 and view.f_frsize == 4096
          and view.f_blocks * view.f_frsize == 1048576, "path_rejected")


def _add_rule(ruleset_fd, path, role):
    path_fd = None
    failure = None
    try:
        path_fd = _open_path(path)
        info = _fd_info(path_fd)
        if role == "private_root_listing":
            _need(path == "/", "path_rejected")
            _private_root(path_fd, info)
            mask = READ_DIR
        elif role == "read_only":
            _need(stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode), "path_rejected")
            mask = READ_ONLY_DIRECTORY if stat.S_ISDIR(info.st_mode) else READ_ONLY_FILE
        elif role == "writable":
            _need(stat.S_ISDIR(info.st_mode), "path_rejected")
            mask = WRITABLE_DIRECTORY
        else:
            major, minor, mask = DEVICES[path]
            _need(stat.S_ISCHR(info.st_mode) and info.st_uid == 0
                  and os.major(info.st_rdev) == major and os.minor(info.st_rdev) == minor,
                  "device_rejected")
        value = PathBeneathAttr(mask, path_fd)
        _need(_syscall(SYS_ADD_RULE, ctypes.c_int(ruleset_fd),
                       ctypes.c_int(LANDLOCK_RULE_PATH_BENEATH), ctypes.byref(value),
                       ctypes.c_uint(0)) == 0, "rule_rejected")
    except BaseException as error:
        failure = error.code if type(error) is LandlockError else "rule_rejected"
    finally:
        if path_fd is not None:
            closing, path_fd = path_fd, None
            try:
                _close(closing)
            except BaseException:
                failure = "close_failed"
    if failure is not None:
        raise LandlockError(failure) from None


def restrict_filesystem(*, read_only_paths, writable_paths, device_paths,
                        allow_private_root_listing=False):
    """Consume one policy attempt; any started failure prohibits payload release."""
    global _ATTEMPT_OWNER
    _need(_SETUP_LOCK.acquire(blocking=False), "reentry")
    ruleset_fd = None
    failure = None
    abi = None
    try:
        owner = os.getpid(), threading.get_ident()
        if _ATTEMPT_OWNER is not None:
            _need(_ATTEMPT_OWNER == owner, "owner_mismatch")
            raise LandlockError("already_attempted") from None
        _paths(read_only_paths, writable_paths, device_paths)
        _need(type(allow_private_root_listing) is bool and
              sum(map(len, (read_only_paths, writable_paths, device_paths)))
              + int(allow_private_root_listing) <= 16, "paths_rejected")
        try:
            _runtime()
        except BaseException:
            raise LandlockError("runtime_rejected") from None
        _need(ctypes.sizeof(RulesetAttr) == 8 and ctypes.sizeof(PathBeneathAttr) == 12
              and PathBeneathAttr.parent_fd.offset == 8, "runtime_rejected")
        _ATTEMPT_OWNER = owner
        abi = _syscall(SYS_CREATE_RULESET, ctypes.c_void_p(), ctypes.c_size_t(0),
                       ctypes.c_uint(LANDLOCK_CREATE_RULESET_VERSION))
        _need(type(abi) is int and 5 <= abi < 2**31, "abi_rejected")
        attributes = RulesetAttr(HANDLED_FS)
        candidate = _syscall(SYS_CREATE_RULESET, ctypes.byref(attributes),
                             ctypes.c_size_t(ctypes.sizeof(attributes)), ctypes.c_uint(0))
        _need(type(candidate) is int and 0 <= candidate < 2**31, "ruleset_rejected")
        ruleset_fd = candidate
        _need(fcntl.fcntl(ruleset_fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC, "ruleset_rejected")
        if allow_private_root_listing:
            # The unchanged secure loader opens each ancestor O_RDONLY|O_DIRECTORY.
            # Grant names-only traversal/listing on our verified private tmpfs root,
            # never READ_FILE/WRITE_FILE/EXECUTE or an ordinary host-root rule.
            _add_rule(ruleset_fd, "/", "private_root_listing")
        for role, paths in (("read_only", read_only_paths), ("writable", writable_paths),
                            ("device", device_paths)):
            for path in paths:
                _add_rule(ruleset_fd, path, role)
        _no_new_privileges()
        _need(_syscall(SYS_RESTRICT_SELF, ctypes.c_int(ruleset_fd), ctypes.c_uint(0)) == 0,
              "restrict_failed")
    except BaseException as error:
        failure = error.code if type(error) is LandlockError else "setup_failed"
    finally:
        if ruleset_fd is not None:
            closing, ruleset_fd = ruleset_fd, None
            try:
                _close(closing)
            except BaseException:
                failure = "close_failed"
        _SETUP_LOCK.release()
    if failure is not None:
        raise LandlockError(failure) from None
    return {"schema": SCHEMA, "abi_version": abi, "filesystem_content_policy_enforced": True,
            "private_root_listing_only": allow_private_root_listing,
            "ptrace_domain_restricted": True, "full_pathname_unix_isolation_verified": False,
            "metadata_policy_alone_sufficient": False, "real_sft_execution_permitted": False,
            "production_eligible": False}
