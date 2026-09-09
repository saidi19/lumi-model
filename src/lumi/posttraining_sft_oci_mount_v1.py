"""One-shot Linux bootstrap into an already exported, accepted OCI rootfs.

Only the supplied rootfs is bind-mounted: no host /usr, /etc, runtime, source,
or device tree is imported. Its existing files and metadata are not changed.
Fresh tmpfs/proc overlays are kernel-generated and not release software. The
caller authenticates the export, then separately supplies cgroup, network,
Landlock, IPC/FD closure, privilege drop and payload release. This helper neither
dispatches a worker nor grants host admission or complete software provenance.
Tmpfs sizes are simultaneous capacity ceilings, not cumulative write quotas.

Rootfs and empty mountpoint must be distinct siblings under one root-owned
0700 private parent. Rootfs itself is root-owned 0755; mountpoint is 0700.
Failure after a started attempt is terminal: the owned child must exit, never
retry or restore the old namespace. Only fixed codes and receipts may be logged.
Never invoke native setup on a Mac; local tests use syscall doubles only.
"""

from __future__ import annotations

import ctypes
import os
import stat
import sys
import threading

from . import posttraining_sft_mount_root_v1 as mount_root
from . import posttraining_sft_netns_v1 as netns

SCHEMA = "lumi.sft-oci-mount.v0.1"
ERROR_CODES = frozenset({
    "runtime_rejected", "input_rejected", "already_attempted", "owner_mismatch",
    "reentry", "source_rejected", "root_rejected", "namespace_rejected",
    "namespace_unchanged", "mount_failed", "seal_failed", "pivot_failed",
    "readback_failed", "setup_failed", "close_failed", "namespace_unshare_failed",
    "bind_mount_failed", "layout_rejected", "device_rejected",
})
REQUIRED_DIRECTORIES = (
    "usr", "usr/local", "usr/local/bin", "usr/local/lib", "usr/local/lib/python3.12",
    "usr/local/lib/python3.12/site-packages", "etc", "opt", "opt/lumi-release",
    "opt/lumi-release/source", "opt/lumi-release/source/src",
    "opt/lumi-release/source/src/lumi", "opt/lumi-release/source/scripts",
    "opt/lumi-release/control", "tmp", "dev", "proc",
)
DEVICES = {"null": (1, 3), "zero": (1, 5), "random": (1, 8), "urandom": (1, 9)}
DEVICE_LINKS = {"fd": "/proc/self/fd", "stdin": "/proc/self/fd/0",
                "stdout": "/proc/self/fd/1", "stderr": "/proc/self/fd/2"}
READONLY_ROOT = 7  # MS_RDONLY | MS_NOSUID | MS_NODEV; execution remains allowed.
READONLY_PROC = 15
READONLY_DEV = 11  # Device nodes must remain usable; /dev is deliberately not nodev.
WRITABLE_TMPFS = 14
SCRATCH_MINIMUM, SCRATCH_MAXIMUM = 8 * 1024**2, 512 * 1024**2
SHM_MINIMUM, SHM_MAXIMUM = 1024**2, 64 * 1024**2
DEV_BYTES = 65536
MNT_DETACH = 2
_SETUP_LOCK = threading.Lock()
_ATTEMPT_OWNER = None


class OCIMountError(RuntimeError):
    def __init__(self, code="setup_failed"):
        self.code = code if type(code) is str and code in ERROR_CODES else "setup_failed"
        super().__init__(self.code)


def _need(value, code):
    if not value:
        raise OCIMountError(code) from None


def _inputs(rootfs_path, mountpoint_path, uid, gid, scratch_bytes, shm_bytes):
    try:
        root_parts, mount_parts = mount_root._path(rootfs_path), mount_root._path(mountpoint_path)
    except BaseException:
        raise OCIMountError("input_rejected") from None
    _need(len(root_parts) >= 3 and len(root_parts) == len(mount_parts)
          and root_parts[:-1] == mount_parts[:-1] and root_parts[-1] != mount_parts[-1],
          "input_rejected")
    _need(all(type(value) is int and 0 < value < 2**31 for value in (uid, gid)),
          "input_rejected")
    for value, minimum, maximum in ((scratch_bytes, SCRATCH_MINIMUM, SCRATCH_MAXIMUM),
                                    (shm_bytes, SHM_MINIMUM, SHM_MAXIMUM)):
        _need(type(value) is int and minimum <= value <= maximum and value % 4096 == 0,
              "input_rejected")
    return rootfs_path.rpartition("/")[0]


def _close(owned, descriptor):
    # Relinquish ownership before an ambiguous close; no repeated close attempt.
    owned.remove(descriptor)
    try:
        os.close(descriptor)
    except BaseException:
        raise OCIMountError("close_failed") from None


def _pin(path, owned):
    if path == "/":
        descriptor = os.open("/", mount_root.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW
                             | os.O_CLOEXEC)
        owned.append(descriptor)
        _need(stat.S_ISDIR(os.fstat(descriptor).st_mode), "root_rejected")
        return descriptor
    return mount_root._pin(path, directory=True, owned=owned)


def _identity(descriptor):
    return mount_root._source_identity(descriptor)


def _owned_directory(descriptor, mode, code):
    info = os.fstat(descriptor)
    _need(stat.S_ISDIR(info.st_mode) and info.st_uid == info.st_gid == 0
          and stat.S_IMODE(info.st_mode) == mode and info.st_ino > 0, code)
    return _identity(descriptor)


def _empty_mountpoint(descriptor, owned):
    listing = os.open(f"/proc/self/fd/{descriptor}", os.O_RDONLY | os.O_DIRECTORY
                      | os.O_CLOEXEC | os.O_NONBLOCK)
    owned.append(listing)
    _need(_identity(listing) == _identity(descriptor) and os.listdir(listing) == [],
          "root_rejected")
    _close(owned, listing)


def _layout(rootfs_path, owned):
    # These are metadata-only, no-symlink checks. Release membership/content,
    # rootfs freshness and export integrity remain the guardian's obligations.
    for relative in REQUIRED_DIRECTORIES:
        descriptor = _pin(rootfs_path + "/" + relative, owned)
        info = os.fstat(descriptor)
        _need(info.st_uid == info.st_gid == 0
              and (stat.S_IMODE(info.st_mode) == 0o1777 if relative == "tmp"
                   else stat.S_IMODE(info.st_mode) in (0o755, 0o555)), "layout_rejected")
        _close(owned, descriptor)
    runtime = mount_root._pin(rootfs_path + "/usr/local/bin/python3.12",
                              directory=False, owned=owned)
    info = os.fstat(runtime)
    _need(stat.S_ISREG(info.st_mode) and info.st_uid == info.st_gid == 0
          and stat.S_IMODE(info.st_mode) == 0o755, "layout_rejected")
    _close(owned, runtime)


def _pivot():
    # Linux man-pages pivot_root(2), NOTES, explicitly specifies this sequence:
    # chdir(new_root); pivot_root(".", "."); umount2(".", MNT_DETACH).
    # https://man7.org/linux/man-pages/man2/pivot_root.2.html
    # The caller has fchdir'd into a pinned new-root mount. Do not chdir("/")
    # between the two syscalls: "." must resolve the stacked old root for detach.
    libc = ctypes.CDLL(None, use_errno=True)
    call = libc.syscall
    call.argtypes, call.restype = [ctypes.c_long], ctypes.c_long
    _need(call(ctypes.c_long(mount_root.SYS_PIVOT_ROOT), ctypes.c_char_p(b"."),
               ctypes.c_char_p(b".")) == 0, "pivot_failed")
    detach = libc.umount2
    detach.argtypes, detach.restype = [ctypes.c_char_p, ctypes.c_int], ctypes.c_int
    _need(detach(b".", MNT_DETACH) == 0, "pivot_failed")
    os.chdir("/")


def _overlay_tmpfs(path, size, inodes, mode, uid, gid):
    mount_root._mount("tmpfs", path, "tmpfs", WRITABLE_TMPFS,
                      f"size={size},nr_inodes={inodes},mode={mode},uid={uid},gid={gid}")


def _overlays(mountpoint_path, uid, gid, scratch_bytes, shm_bytes):
    _overlay_tmpfs(mountpoint_path + "/tmp", scratch_bytes, 16384, "1777", uid, gid)
    mount_root._mount("tmpfs", mountpoint_path + "/dev", "tmpfs",
                      mount_root.MS_NOSUID | mount_root.MS_NOEXEC,
                      f"size={DEV_BYTES},nr_inodes=64,mode=0755,uid=0,gid=0")
    for name, (major, minor) in DEVICES.items():
        target = mountpoint_path + "/dev/" + name
        os.mknod(target, stat.S_IFCHR | 0o666, os.makedev(major, minor))
        os.chmod(target, 0o666)
    for name, target in DEVICE_LINKS.items():
        os.symlink(target, mountpoint_path + "/dev/" + name)
    # Only the newly mounted /dev tmpfs is modified, never the exported rootfs.
    shm_path = mountpoint_path + "/dev/shm"
    os.mkdir(shm_path, 0o1777)
    os.chmod(shm_path, 0o1777)
    _overlay_tmpfs(shm_path, shm_bytes, 1024, "1777", uid, gid)
    mount_root._seal(mountpoint_path + "/dev", READONLY_DEV, recursive=False)
    mount_root._mount("proc", mountpoint_path + "/proc", "proc",
                      mount_root.MS_NOSUID | mount_root.MS_NODEV | mount_root.MS_NOEXEC,
                      "hidepid=2")
    mount_root._seal(mountpoint_path + "/proc", READONLY_PROC, recursive=False)


def _tmpfs_readback(path, expected_bytes, uid, gid, owned):
    view = mount_root._readback(path, WRITABLE_TMPFS, mount_root.MS_RDONLY)
    descriptor = _pin(path, owned)
    mount_root._filesystem(descriptor, mount_root.TMPFS_MAGIC, "readback_failed")
    info = os.fstat(descriptor)
    _need(info.st_uid == uid and info.st_gid == gid and stat.S_IMODE(info.st_mode) == 0o1777
          and view.f_frsize == 4096 and view.f_blocks * view.f_frsize == expected_bytes,
          "readback_failed")
    _close(owned, descriptor)


def _readback(expected_root, uid, gid, scratch_bytes, shm_bytes, owned):
    current = _pin("/", owned)
    _need(_identity(current) == expected_root, "readback_failed")
    _close(owned, current)
    mount_root._readback("/", READONLY_ROOT, mount_root.MS_NOEXEC)
    mount_root._readback("/dev", READONLY_DEV, mount_root.MS_NODEV)
    mount_root._readback("/proc", READONLY_PROC)
    for path, magic in (("/dev", mount_root.TMPFS_MAGIC), ("/proc", mount_root.PROC_MAGIC)):
        descriptor = _pin(path, owned)
        mount_root._filesystem(descriptor, magic, "readback_failed")
        _close(owned, descriptor)
    _tmpfs_readback("/tmp", scratch_bytes, uid, gid, owned)
    _tmpfs_readback("/dev/shm", shm_bytes, uid, gid, owned)
    for name, (major, minor) in DEVICES.items():
        info = os.stat("/dev/" + name, follow_symlinks=False)
        _need(stat.S_ISCHR(info.st_mode) and info.st_uid == info.st_gid == 0
              and stat.S_IMODE(info.st_mode) == 0o666
              and (os.major(info.st_rdev), os.minor(info.st_rdev)) == (major, minor),
              "device_rejected")


def enter_oci_rootfs(rootfs_path, mountpoint_path, *, uid=10001, gid=10001,
                     scratch_bytes=536870912, shm_bytes=67108864):
    """Consume one rootfs mount attempt; return no permission to execute a worker."""
    global _ATTEMPT_OWNER
    _need(_SETUP_LOCK.acquire(blocking=False), "reentry")
    owned, failure = [], None
    try:
        owner = os.getpid(), threading.get_ident()
        if _ATTEMPT_OWNER is not None:
            _need(_ATTEMPT_OWNER == owner, "owner_mismatch")
            raise OCIMountError("already_attempted") from None
        parent = _inputs(rootfs_path, mountpoint_path, uid, gid, scratch_bytes, shm_bytes)
        try:
            netns._runtime()
            _need(sys.version_info[:2] == (3, 12), "runtime_rejected")
        except BaseException:
            raise OCIMountError("runtime_rejected") from None
        _ATTEMPT_OWNER = owner
        parent_fd = _pin(parent, owned)
        parent_identity = _owned_directory(parent_fd, 0o700, "root_rejected")
        original_root = _pin(rootfs_path, owned)
        original_target = _pin(mountpoint_path, owned)
        expected_root = _owned_directory(original_root, 0o755, "root_rejected")
        expected_target = _owned_directory(original_target, 0o700, "root_rejected")
        _need(expected_root[:2] != expected_target[:2], "root_rejected")
        _empty_mountpoint(original_target, owned)
        _layout(rootfs_path, owned)
        old_namespace = mount_root._open_namespace(owned)
        old_identity = mount_root._namespace_identity(old_namespace)
        mount_root._unshare()
        new_namespace = mount_root._open_namespace(owned)
        new_identity = mount_root._namespace_identity(new_namespace)
        _need(new_identity != old_identity, "namespace_unchanged")
        mount_root._mount(None, "/", None, mount_root.MS_REC | mount_root.MS_PRIVATE)
        # Fresh namespace pins are mandatory: old namespace vfsmount references
        # cannot safely be used as bind sources after unshare.
        new_parent = _pin(parent, owned)
        root_pin = _pin(rootfs_path, owned)
        target_pin = _pin(mountpoint_path, owned)
        _need(_identity(new_parent) == parent_identity and _identity(root_pin) == expected_root
              and _identity(target_pin) == expected_target, "source_rejected")
        _empty_mountpoint(target_pin, owned)
        for descriptor in (parent_fd, original_root, original_target, new_parent):
            _close(owned, descriptor)
        # A nonrecursive bind deliberately excludes any pre-existing submounts.
        mount_root._mount(f"/proc/self/fd/{root_pin}", f"/proc/self/fd/{target_pin}", None,
                          mount_root.MS_BIND, failure_code="bind_mount_failed")
        bound_root = _pin(mountpoint_path, owned)
        _need(_identity(bound_root) == expected_root, "source_rejected")
        mount_root._seal(mountpoint_path, READONLY_ROOT, recursive=False)
        _close(owned, root_pin)
        _close(owned, target_pin)
        _overlays(mountpoint_path, uid, gid, scratch_bytes, shm_bytes)
        _need(mount_root._namespace_identity(old_namespace) == old_identity
              and mount_root._namespace_identity(new_namespace) == new_identity,
              "namespace_rejected")
        os.fchdir(bound_root)
        _pivot()
        _readback(expected_root, uid, gid, scratch_bytes, shm_bytes, owned)
    except BaseException as error:
        failure = (error.code if type(error) in (OCIMountError, mount_root.MountRootError)
                   and error.code in ERROR_CODES else "setup_failed")
    finally:
        while owned:
            descriptor = owned[-1]
            try:
                _close(owned, descriptor)
            except BaseException:
                failure = "close_failed"
        _SETUP_LOCK.release()
    if failure is not None:
        raise OCIMountError(failure) from None
    return {
        "schema": SCHEMA, "rootfs_only_bind": True, "readonly_root": True,
        "private_propagation": True, "old_root_detached": True,
        "private_tmpfs": True, "private_shm_tmpfs": True, "fresh_procfs": True,
        "scratch_capacity_bytes": scratch_bytes, "shm_capacity_bytes": shm_bytes,
        "host_software_binds": False, "host_mounts_modified": False,
        "rootfs_content_authenticated": False, "ambient_descriptors_closed": False,
        "complete_os_sandbox_verified": False, "worker_dispatched": False,
        "real_sft_execution_permitted": False, "production_eligible": False,
    }
