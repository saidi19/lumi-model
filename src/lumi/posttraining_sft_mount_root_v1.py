"""One-shot root bootstrap into an owned, private, read-only Linux mount tree.

Only a bounded tmpfs scratch area remains writable. Its capacity is not a
cumulative write quota. This helper does not hide procfs process metadata, close
ambient descriptors, apply Landlock, or admit a real training job. The caller
must apply those separate controls, drop privilege, and close inherited FDs
before executing its own generated fixture. Any failure is terminal: exit the
attached child and let its independent guardian clean up. Never retry, setns,
restore the old root, or expose exception context.
"""

from __future__ import annotations

import ctypes
import fcntl
import os
import re
import stat
import threading

from . import posttraining_sft_netns_v1 as netns
from .posttraining_sft_host_observer_v1 import PROC_MAGIC, StatFS

SCHEMA = "lumi.sft-mount-root.v0.1"
ERROR_CODES = frozenset({
    "runtime_rejected", "input_rejected", "already_attempted", "owner_mismatch",
    "reentry", "source_rejected", "root_rejected", "namespace_rejected",
    "namespace_unchanged", "mount_failed", "seal_failed", "pivot_failed",
    "readback_failed", "setup_failed", "close_failed",
})
CLONE_NEWNS = 0x20000
O_PATH = 0x200000
MS_RDONLY, MS_NOSUID, MS_NODEV, MS_NOEXEC = 1, 2, 4, 8
MS_BIND, MS_REC, MS_PRIVATE = 4096, 16384, 1 << 18
AT_RECURSIVE, AT_FDCWD = 0x8000, -100
SYS_MOUNT_SETATTR, SYS_PIVOT_ROOT = 442, 155
TMPFS_MAGIC = 0x01021994
_SETUP_LOCK = threading.Lock()
_ATTEMPT_OWNER = None
_ETC = ("ld.so.cache", "hosts", "nsswitch.conf", "passwd", "group")
_LINKS = {"bin": "usr/bin", "sbin": "usr/sbin", "lib": "usr/lib", "lib64": "usr/lib64"}
_DEVICES = {"null": (1, 3), "zero": (1, 5), "random": (1, 8), "urandom": (1, 9)}
_SAFE = re.compile(r"[A-Za-z0-9._-]+")


class MountRootError(RuntimeError):
    def __init__(self, code="setup_failed"):
        self.code = code if type(code) is str and code in ERROR_CODES else "setup_failed"
        super().__init__(self.code)


class MountAttr(ctypes.Structure):
    _fields_ = [("attr_set", ctypes.c_uint64), ("attr_clr", ctypes.c_uint64),
                ("propagation", ctypes.c_uint64), ("userns_fd", ctypes.c_uint64)]


def _need(value, code):
    if not value:
        raise MountRootError(code) from None


def _path(value):
    _need(type(value) is str and 1 < len(value) <= 4096 and value.startswith("/")
          and not value.endswith("/") and "//" not in value, "input_rejected")
    pieces = value[1:].split("/")
    _need(all(piece not in (".", "..") and _SAFE.fullmatch(piece) for piece in pieces),
          "input_rejected")
    return pieces


def _inputs(root_path, runtime_root, source_root, script_path, uid, gid, scratch_bytes):
    for value in (root_path, runtime_root, source_root, script_path):
        _path(value)
    _need(re.fullmatch(r"/tmp/lumi-sft-root-[a-f0-9]{32}", root_path), "input_rejected")
    _need(re.fullmatch(r"/opt/hostedtoolcache/Python/3\.12\.[0-9]{1,3}/x64", runtime_root),
          "input_rejected")
    _need(source_root.endswith("/src") and source_root.count("/") >= 2
          and script_path == source_root[:-4] + "/scripts/run_sft_worker_run_torch_fixture_v1.py",
          "input_rejected")
    _need(all(type(value) is int and 0 < value < 2**31 for value in (uid, gid)),
          "input_rejected")
    _need(type(scratch_bytes) is int and 8 * 1024**2 <= scratch_bytes <= 256 * 1024**2
          and scratch_bytes % 4096 == 0, "input_rejected")
    # Prevent any bind from covering the generated root, scratch, or another
    # admitted source; normal hosted-runner source/runtime locations are disjoint.
    paths = (root_path, runtime_root, source_root, script_path, "/usr", "/proc", "/etc", "/dev")
    _need(not any(a == b or a.startswith(b + "/") or b.startswith(a + "/")
                  for index, a in enumerate(paths) for b in paths[index + 1:]),
          "input_rejected")
    _need(not source_root.startswith("/tmp/") and not script_path.startswith("/tmp/"),
          "input_rejected")


def _close(fd):
    try:
        os.close(fd)
    except BaseException:
        raise MountRootError("close_failed") from None


def _filesystem(fd, magic, code):
    libc = ctypes.CDLL(None, use_errno=True)
    call = libc.fstatfs
    call.argtypes, call.restype = [ctypes.c_int, ctypes.POINTER(StatFS)], ctypes.c_int
    result = StatFS()
    _need(call(fd, ctypes.byref(result)) == 0 and result.kind == magic, code)


def _pin(path, *, directory, owned):
    """No symlink components; only metadata is inspected, never file contents."""
    current = os.open("/", O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    owned.append(current)
    parts = _path(path)
    for index, part in enumerate(parts):
        is_dir = index < len(parts) - 1 or directory
        flags = O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC | (os.O_DIRECTORY if is_dir else 0)
        next_fd = os.open(part, flags, dir_fd=current)
        owned.append(next_fd)
        info = os.fstat(next_fd)
        _need((stat.S_ISDIR(info.st_mode) if is_dir else stat.S_ISREG(info.st_mode))
              and info.st_ino > 0
              and fcntl.fcntl(next_fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC,
              "source_rejected")
        owned.remove(current)
        _close(current)
        current = next_fd
    return current


def _namespace_identity(fd):
    info = os.fstat(fd)
    _need(stat.S_ISREG(info.st_mode) and info.st_uid == info.st_gid == 0
          and info.st_ino > 0 and fcntl.fcntl(fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC,
          "namespace_rejected")
    _filesystem(fd, netns.NSFS_MAGIC, "namespace_rejected")
    _need(fcntl.ioctl(fd, netns.NS_GET_NSTYPE, 0) == CLONE_NEWNS, "namespace_rejected")
    return info.st_dev, info.st_ino


def _open_namespace(owned):
    proc = _pin("/proc", directory=True, owned=owned)
    info = os.fstat(proc)
    _need(info.st_uid == info.st_gid == 0, "namespace_rejected")
    _filesystem(proc, PROC_MAGIC, "namespace_rejected")
    # These two kernel-owned procfs symlinks are the only intentional follow.
    fd = os.open("self/ns/mnt", os.O_RDONLY | os.O_CLOEXEC, dir_fd=proc)
    owned.append(fd)
    _namespace_identity(fd)
    return fd


def _unshare():
    call = ctypes.CDLL(None, use_errno=True).unshare
    call.argtypes, call.restype = [ctypes.c_int], ctypes.c_int
    _need(call(CLONE_NEWNS) == 0, "mount_failed")


def _mount(source, target, filesystem, flags, data=None):
    call = ctypes.CDLL(None, use_errno=True).mount
    call.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                     ctypes.c_ulong, ctypes.c_char_p]
    call.restype = ctypes.c_int
    def encode(value):
        return None if value is None else value.encode("ascii")
    _need(call(encode(source), encode(target), encode(filesystem), flags, encode(data)) == 0,
          "mount_failed")


def _seal(target, attrs, *, recursive):
    call = ctypes.CDLL(None, use_errno=True).syscall
    call.restype = ctypes.c_long
    value = MountAttr(attrs, 0, 0, 0)
    _need(call(ctypes.c_long(SYS_MOUNT_SETATTR), ctypes.c_int(AT_FDCWD),
               ctypes.c_char_p(target.encode("ascii")),
               ctypes.c_uint(AT_RECURSIVE if recursive else 0), ctypes.byref(value),
               ctypes.c_size_t(ctypes.sizeof(value))) == 0, "seal_failed")


def _pivot():
    call = ctypes.CDLL(None, use_errno=True).syscall
    call.restype = ctypes.c_long
    _need(call(ctypes.c_long(SYS_PIVOT_ROOT), ctypes.c_char_p(b"."),
               ctypes.c_char_p(b".oldroot")) == 0, "pivot_failed")
    os.chdir("/")
    call = ctypes.CDLL(None, use_errno=True).umount2
    call.argtypes, call.restype = [ctypes.c_char_p, ctypes.c_int], ctypes.c_int
    _need(call(b"/.oldroot", 2) == 0, "pivot_failed")
    os.rmdir("/.oldroot")


def _mkdirs(path, made):
    current = ""
    for part in path[1:].split("/"):
        current += "/" + part
        if current not in made:
            os.mkdir(current, 0o755)
            # The guardian deliberately uses umask 077. These newly owned
            # scaffold directories must remain traversable after UID drop.
            os.chmod(current, 0o755)
            made.add(current)


def _bind(fd, target, *, directory, made):
    if directory:
        _mkdirs(target, made)
    else:
        _mkdirs(target.rpartition("/")[0], made)
        created = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                          | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        _close(created)
    _mount(f"/proc/self/fd/{fd}", target, None, MS_BIND | (MS_REC if directory else 0))
    attrs = MS_RDONLY | MS_NOSUID | MS_NODEV
    if target.endswith("/proc"):
        attrs |= MS_NOEXEC
    _seal(target, attrs, recursive=directory)


def _readback(path, required, forbidden=0):
    info = os.statvfs(path)
    _need(info.f_flag & required == required and not info.f_flag & forbidden,
          "readback_failed")
    return info


def enter_mount_root(*, root_path, runtime_root, source_root, script_path, uid, gid,
                     scratch_bytes):
    """Consume one fresh owned root; success still requires caller sandbox gates."""
    global _ATTEMPT_OWNER
    _need(_SETUP_LOCK.acquire(blocking=False), "reentry")
    owned, failure = [], None
    try:
        owner = os.getpid(), threading.get_ident()
        if _ATTEMPT_OWNER is not None:
            _need(_ATTEMPT_OWNER == owner, "owner_mismatch")
            raise MountRootError("already_attempted") from None
        _inputs(root_path, runtime_root, source_root, script_path, uid, gid, scratch_bytes)
        try:
            netns._runtime()
        except BaseException:
            raise MountRootError("runtime_rejected") from None
        _ATTEMPT_OWNER = owner
        temporary_parent = _pin("/tmp", directory=True, owned=owned)
        parent_info = os.fstat(temporary_parent)
        _need(parent_info.st_uid == parent_info.st_gid == 0
              and parent_info.st_mode & stat.S_ISVTX, "root_rejected")
        root_pin = _pin(root_path, directory=True, owned=owned)
        initial = os.fstat(root_pin)
        _need(initial.st_uid == initial.st_gid == 0 and stat.S_IMODE(initial.st_mode) == 0o700,
              "root_rejected")
        listing = os.open(f"/proc/self/fd/{root_pin}", os.O_RDONLY | os.O_DIRECTORY
                          | os.O_CLOEXEC | os.O_NONBLOCK)
        owned.append(listing)
        _need(os.fstat(listing) == initial and os.listdir(listing) == [], "root_rejected")
        for path, value in _LINKS.items():
            _need(os.readlink("/" + path) == value, "source_rejected")
        sources = [("/usr", True), (runtime_root, True), (source_root, True),
                   (script_path, False), *(("/etc/" + item, False) for item in _ETC),
                   ("/proc", True)]
        pins = [(path, is_dir, _pin(path, directory=is_dir, owned=owned))
                for path, is_dir in sources]
        old_namespace = _open_namespace(owned)
        old_identity = _namespace_identity(old_namespace)
        _unshare()
        new_namespace = _open_namespace(owned)
        new_identity = _namespace_identity(new_namespace)
        _need(old_identity != new_identity, "namespace_unchanged")
        _mount(None, "/", None, MS_REC | MS_PRIVATE)
        _mount("tmpfs", root_path, "tmpfs", MS_NOSUID | MS_NODEV,
               "size=1048576,nr_inodes=256,mode=0755")
        # A pre-mount O_PATH FD refers to the old directory, never the new tmpfs.
        new_root = _pin(root_path, directory=True, owned=owned)
        new_info = os.fstat(new_root)
        _filesystem(new_root, TMPFS_MAGIC, "root_rejected")
        _need((new_info.st_dev, new_info.st_ino) != (initial.st_dev, initial.st_ino)
              and new_info.st_uid == new_info.st_gid == 0
              and stat.S_IMODE(new_info.st_mode) == 0o755, "root_rejected")
        # Ancestors pre-exist outside the newly mounted root; never mkdir them.
        made = {root_path}
        made.update(root_path[:index] for index, char in enumerate(root_path) if char == "/")
        _mkdirs(root_path + "/.oldroot", made)
        for path, is_dir, fd in pins:
            _bind(fd, root_path + path, directory=is_dir, made=made)
        for path, value in _LINKS.items():
            os.symlink(value, root_path + "/" + path)
        _mkdirs(root_path + "/tmp", made)
        _mount("tmpfs", root_path + "/tmp", "tmpfs", MS_NOSUID | MS_NODEV | MS_NOEXEC,
               f"size={scratch_bytes},nr_inodes=16384,mode=1777")
        os.mkdir(root_path + "/tmp/shm", 0o1777)
        os.chmod(root_path + "/tmp/shm", 0o1777)
        _mkdirs(root_path + "/dev", made)
        _mount("tmpfs", root_path + "/dev", "tmpfs", MS_NOSUID | MS_NOEXEC,
               "size=65536,nr_inodes=64,mode=0755")
        for name, (major, minor) in _DEVICES.items():
            target = root_path + "/dev/" + name
            os.mknod(target, stat.S_IFCHR | 0o666, os.makedev(major, minor))
            os.chmod(target, 0o666)
        for name, value in {"fd": "/proc/self/fd", "stdin": "/proc/self/fd/0",
                            "stdout": "/proc/self/fd/1", "stderr": "/proc/self/fd/2",
                            "shm": "/tmp/shm"}.items():
            os.symlink(value, root_path + "/dev/" + name)
        _seal(root_path + "/dev", MS_RDONLY | MS_NOSUID | MS_NOEXEC, recursive=False)
        _need(_namespace_identity(old_namespace) == old_identity
              and _namespace_identity(new_namespace) == new_identity, "namespace_rejected")
        os.chdir(root_path)
        _pivot()
        _seal("/", MS_RDONLY | MS_NOSUID | MS_NODEV, recursive=False)
        _readback("/", MS_RDONLY | MS_NOSUID | MS_NODEV)
        for path, _, _ in pins:
            _readback(path, MS_RDONLY | MS_NOSUID | MS_NODEV
                      | (MS_NOEXEC if path == "/proc" else 0))
        _readback("/dev", MS_RDONLY | MS_NOSUID | MS_NOEXEC, MS_NODEV)
        scratch = _readback("/tmp", MS_NOSUID | MS_NODEV | MS_NOEXEC, MS_RDONLY)
        scratch_fd = _pin("/tmp", directory=True, owned=owned)
        _filesystem(scratch_fd, TMPFS_MAGIC, "readback_failed")
        _need(scratch.f_frsize == 4096 and scratch.f_blocks * scratch.f_frsize == scratch_bytes,
              "readback_failed")
    except BaseException as error:
        failure = error.code if type(error) is MountRootError else "setup_failed"
    finally:
        while owned:
            fd = owned.pop()
            try:
                _close(fd)
            except BaseException:
                failure = "close_failed"
        _SETUP_LOCK.release()
    if failure is not None:
        raise MountRootError(failure) from None
    return {"schema": SCHEMA, "readonly_root": True, "readonly_binds": True,
            "old_root_detached": True, "private_propagation": True, "private_tmpfs": True,
            "scratch_capacity_bytes": scratch_bytes, "host_mounts_modified": False,
            "real_sft_execution_permitted": False, "production_eligible": False}
