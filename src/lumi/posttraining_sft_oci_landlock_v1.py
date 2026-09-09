"""One-shot Landlock policy for an exact guardian-owned, mounted OCI rootfs.

The caller supplies the exported root's pre-mount device/inode/type/owner/group
identity. This successor verifies that exact root and read-only mount flags,
then grants only READ_DIR at /. The old tmpfs-specific helper remains unchanged.
Normal read-only, scratch and device rules reuse its existing typed primitives.

Call after the private mount/pivot stage and before IPC filtering, descriptor
closure, privilege drop and worker release. Root identity is not a content hash
or image-authenticity claim. Landlock does not supply complete metadata, IPC,
hostile-host, inherited-FD or full operating-system isolation. Any started
failure is terminal; the attached child must exit without retry. Log only fixed
codes and receipt fields, never the suppressed underlying exception context.
Mac tests must replace all runtime, filesystem and syscall operations.
"""

from __future__ import annotations

import ctypes
import fcntl
import os
import stat
import threading

from . import posttraining_sft_landlock_v1 as landlock

SCHEMA = "lumi.sft-oci-landlock.v0.1"
ERROR_CODES = frozenset({
    "runtime_rejected", "paths_rejected", "path_rejected", "device_rejected",
    "root_identity_rejected", "root_mount_rejected", "already_attempted",
    "owner_mismatch", "reentry", "abi_rejected", "ruleset_rejected", "rule_rejected",
    "privilege_rejected", "restrict_failed", "setup_failed", "close_failed",
})
_SETUP_LOCK = threading.Lock()
_ATTEMPT_OWNER = None


class OCILandlockError(RuntimeError):
    def __init__(self, code="setup_failed"):
        self.code = code if type(code) is str and code in ERROR_CODES else "setup_failed"
        super().__init__(self.code)


def _need(value, code):
    if not value:
        raise OCILandlockError(code) from None


def _expected_identity(value):
    _need(type(value) is tuple and len(value) == 5
          and all(type(number) is int for number in value)
          and 0 <= value[0] < 2**64 and 0 < value[1] < 2**64
          and value[2] == stat.S_IFDIR and value[3:] == (0, 0),
          "root_identity_rejected")


def _close(descriptor):
    try:
        os.close(descriptor)
    except BaseException:
        raise OCILandlockError("close_failed") from None


def _add_root_rule(ruleset_fd, expected_root_identity):
    descriptor, failure = None, None
    try:
        descriptor = landlock._open_path("/")
        info = landlock._fd_info(descriptor)
        observed = (info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode), info.st_uid, info.st_gid)
        _need(observed == expected_root_identity and stat.S_IMODE(info.st_mode) == 0o755,
              "root_identity_rejected")
        view = os.fstatvfs(descriptor)
        # Bind rootfs need not be tmpfs. Match the actual expected object and
        # readonly/nosuid/nodev flags; executable runtime is intentionally allowed.
        _need(view.f_flag & 7 == 7 and not view.f_flag & 8, "root_mount_rejected")
        attributes = landlock.PathBeneathAttr(landlock.READ_DIR, descriptor)
        _need(landlock._syscall(
            landlock.SYS_ADD_RULE, ctypes.c_int(ruleset_fd),
            ctypes.c_int(landlock.LANDLOCK_RULE_PATH_BENEATH), ctypes.byref(attributes),
            ctypes.c_uint(0),
        ) == 0, "rule_rejected")
    except BaseException as error:
        failure = (error.code if type(error) in (OCILandlockError, landlock.LandlockError)
                   and error.code in ERROR_CODES else "rule_rejected")
    finally:
        if descriptor is not None:
            closing, descriptor = descriptor, None
            try:
                _close(closing)
            except BaseException:
                failure = "close_failed"
    if failure is not None:
        raise OCILandlockError(failure) from None


def restrict_filesystem(*, read_only_paths, writable_paths, device_paths, expected_root_identity):
    """Install one exact OCI-root names-only policy, without authorizing dispatch."""
    global _ATTEMPT_OWNER
    _need(_SETUP_LOCK.acquire(blocking=False), "reentry")
    ruleset_fd, failure, abi = None, None, None
    try:
        owner = os.getpid(), threading.get_ident()
        if _ATTEMPT_OWNER is not None:
            _need(_ATTEMPT_OWNER == owner, "owner_mismatch")
            raise OCILandlockError("already_attempted") from None
        _expected_identity(expected_root_identity)
        landlock._paths(read_only_paths, writable_paths, device_paths)
        _need(sum(map(len, (read_only_paths, writable_paths, device_paths))) + 1 <= 16,
              "paths_rejected")
        try:
            landlock._runtime()
        except BaseException:
            raise OCILandlockError("runtime_rejected") from None
        _need(ctypes.sizeof(landlock.RulesetAttr) == 8
              and ctypes.sizeof(landlock.PathBeneathAttr) == 12
              and landlock.PathBeneathAttr.parent_fd.offset == 8, "runtime_rejected")
        _ATTEMPT_OWNER = owner
        abi = landlock._syscall(landlock.SYS_CREATE_RULESET, ctypes.c_void_p(), ctypes.c_size_t(0),
                                ctypes.c_uint(landlock.LANDLOCK_CREATE_RULESET_VERSION))
        _need(type(abi) is int and 5 <= abi < 2**31, "abi_rejected")
        attributes = landlock.RulesetAttr(landlock.HANDLED_FS)
        candidate = landlock._syscall(landlock.SYS_CREATE_RULESET, ctypes.byref(attributes),
                                      ctypes.c_size_t(ctypes.sizeof(attributes)), ctypes.c_uint(0))
        _need(type(candidate) is int and 0 <= candidate < 2**31, "ruleset_rejected")
        ruleset_fd = candidate
        _need(fcntl.fcntl(ruleset_fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC, "ruleset_rejected")
        _add_root_rule(ruleset_fd, expected_root_identity)
        for role, paths in (("read_only", read_only_paths), ("writable", writable_paths),
                            ("device", device_paths)):
            for path in paths:
                landlock._add_rule(ruleset_fd, path, role)
        landlock._no_new_privileges()
        _need(landlock._syscall(landlock.SYS_RESTRICT_SELF, ctypes.c_int(ruleset_fd),
                                ctypes.c_uint(0)) == 0, "restrict_failed")
    except BaseException as error:
        failure = (error.code if type(error) in (OCILandlockError, landlock.LandlockError)
                   and error.code in ERROR_CODES else "setup_failed")
    finally:
        if ruleset_fd is not None:
            closing, ruleset_fd = ruleset_fd, None
            try:
                _close(closing)
            except BaseException:
                failure = "close_failed"
        _SETUP_LOCK.release()
    if failure is not None:
        raise OCILandlockError(failure) from None
    return {
        "schema": SCHEMA, "abi_version": abi, "filesystem_content_policy_enforced": True,
        "exported_root_identity_verified": True, "root_readonly_mount_verified": True,
        "private_root_listing_only": True, "ptrace_domain_restricted": True,
        "rootfs_content_authenticated": False, "full_pathname_unix_isolation_verified": False,
        "metadata_policy_alone_sufficient": False, "inherited_authority_closed": False,
        "complete_os_sandbox_verified": False, "real_host_admission_verified": False,
        "real_sft_execution_permitted": False, "production_eligible": False,
    }
