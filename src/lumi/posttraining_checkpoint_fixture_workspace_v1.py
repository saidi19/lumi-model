"""One fresh private fixture workspace and bounded metadata-only sibling cleanup.

The Linux/root caller authenticates the private harness parent and must first
reap every descendant, release every native mount, and close/remove its separate
OwnedRootfs. The two cleanup booleans are required caller premises, NOT evidence
of those events. No existing workspace can be adopted through this API.

Cleanup accepts only checkpoint/tokenizer/output/mountpoint top-level names in
the newly created directory. Its complete bounded census precedes deletion;
symlinks, special files, hardlinks, inode aliases and mount boundaries reject.
Authenticated procfs fdinfo mount IDs also reject same-device bind mounts.
File contents are never read. Each bottom-up removal rechecks its exact pinned
inode. Concurrent hostile root mutation is not excluded by this adapter.

Cleanup is one-shot even on failure; partial newly owned evidence is retained,
never retried or recursively force-deleted. Tests use syscall doubles only.
Paths and names remain private controls; only the aggregate receipt is public.
"""

from __future__ import annotations

import os
import re
import stat
import time
import uuid
from pathlib import Path

from . import posttraining_checkpoint_guard_sources_v1 as runtime
from . import posttraining_sft_mount_root_v1 as mount_root
from . import posttraining_sft_oci_rootfs_v1 as rootfs

SCHEMA = "lumi.checkpoint-fixture-workspace-cleanup.v0.1"
TOP_LEVEL = frozenset({"checkpoint", "tokenizer", "output", "mountpoint"})
ROLES = frozenset({"generator", "importer"})
MAXIMUM_NODES, MAXIMUM_DEPTH, MAXIMUM_FILE_BYTES = 512, 16, 16 * 1024**2
MAXIMUM_DIRECTORY_ENTRIES = 2 * MAXIMUM_NODES
MAXIMUM_KERNEL_READ_CALLS = 4096
MAXIMUM_KERNEL_READ_BYTES = MAXIMUM_KERNEL_READ_CALLS * 1026
READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
PIN_FLAGS = mount_root.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC
ERROR_CODES = frozenset({"runtime_rejected", "arguments_rejected", "owner_mismatch",
    "already_attempted", "workspace_closed", "creation_failed", "creation_cleanup_unconfirmed",
    "scope_rejected", "mount_rejected", "bounds_exhausted", "identity_changed",
    "cleanup_failed", "close_failed", "deadline_exhausted"})
_CREATE_TOKEN = object()


class WorkspaceError(ValueError):
    def __init__(self, code="cleanup_failed"):
        self.code = code if type(code) is str and code in ERROR_CODES else "cleanup_failed"
        super().__init__(self.code)


def _need(value, code="scope_rejected"):
    if not value:
        raise WorkspaceError(code) from None


def _runtime():
    try:
        runtime._runtime()
    except BaseException:
        raise WorkspaceError("runtime_rejected") from None


def _deadline(value):
    _need(type(value) is int and 0 < value < 2**63, "arguments_rejected")
    _need(time.monotonic_ns() < value, "deadline_exhausted")


def _identity(info):
    return info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid


def _state(info):
    return (*_identity(info), info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _directory(info, *, private=False):
    _need(stat.S_ISDIR(info.st_mode) and info.st_ino > 0
          and info.st_uid == info.st_gid and info.st_uid in (0, 10001))
    if private:
        _need(info.st_uid == 0 and stat.S_IMODE(info.st_mode) == 0o700)


class OwnedFixtureWorkspace:
    """Construct only with create_workspace; retained descriptors scope every removal."""

    def __init__(self, token, parent, role, name):
        _need(token is _CREATE_TOKEN, "arguments_rejected")
        self._parent, self._role, self._name = parent, role, name
        self._path = parent / name
        self._owner_pid = os.getpid()
        self._fds = []
        self._parent_fd = self._root_fd = self._fdinfo = None
        self._parent_identity = self._root_identity = None
        self._created = self._cleanup_attempted = self._closed = False
        self._kernel_bytes = self._kernel_calls = self._kernel_reserved = 0
        self._directory_entries = 0

    @property
    def path(self):
        _need(os.getpid() == self._owner_pid, "owner_mismatch")
        _need(self._created and not self._closed, "workspace_closed")
        return self._path

    def _keep(self, fd):
        self._fds.append(fd)
        return fd

    def _close(self, fd):
        self._fds.remove(fd)  # Do not repeat an ambiguous close.
        try:
            os.close(fd)
        except BaseException:
            raise WorkspaceError("close_failed") from None

    def _close_all(self):
        failed = False
        while self._fds:
            try:
                self._close(self._fds[-1])
            except BaseException:
                failed = True
        self._closed = True
        _need(not failed, "close_failed")

    def _proc(self):
        proc = self._keep(rootfs._open_absolute("/proc", directory=True))
        _need(os.fstat(proc).st_uid == os.fstat(proc).st_gid == 0, "mount_rejected")
        mount_root._filesystem(proc, mount_root.PROC_MAGIC, "source_rejected")
        # Only this kernel-owned 'self' symlink beneath authenticated procfs is followed.
        fdinfo = self._keep(os.open("self/fdinfo", READ_FLAGS | os.O_DIRECTORY, dir_fd=proc))
        _need(os.fstat(fdinfo).st_uid == os.fstat(fdinfo).st_gid == 0, "mount_rejected")
        mount_root._filesystem(fdinfo, mount_root.PROC_MAGIC, "source_rejected")
        self._close(proc)
        self._fdinfo = fdinfo

    def _kernel_read(self, fd, count):
        self._kernel_calls += 1
        self._kernel_reserved += count
        _need(self._kernel_calls <= MAXIMUM_KERNEL_READ_CALLS
              and self._kernel_reserved <= MAXIMUM_KERNEL_READ_BYTES, "bounds_exhausted")
        raw = os.read(fd, count)
        _need(type(raw) is bytes and len(raw) <= count, "mount_rejected")
        self._kernel_bytes += len(raw)
        return raw

    def _mount_id(self, fd, deadline_ns):
        _deadline(deadline_ns)
        view = self._keep(os.open(str(fd), READ_FLAGS, dir_fd=self._fdinfo))
        try:
            info = os.fstat(view)
            _need(stat.S_ISREG(info.st_mode) and info.st_uid == info.st_gid == 0,
                  "mount_rejected")
            mount_root._filesystem(view, mount_root.PROC_MAGIC, "source_rejected")
            raw = self._kernel_read(view, 1025)
            _need(0 < len(raw) <= 1024 and self._kernel_read(view, 1) == b"", "mount_rejected")
            rows = [row for row in raw.splitlines() if row.startswith(b"mnt_id:")]
            _need(len(rows) == 1 and re.fullmatch(rb"mnt_id:\s*[1-9][0-9]{0,19}", rows[0]),
                  "mount_rejected")
            number = int(rows[0].split(b":", 1)[1])
            _need(0 < number < 2**64, "mount_rejected")
            return number
        finally:
            self._close(view)

    def _scope(self):
        parent = os.fstat(self._parent_fd)
        current = os.fstat(self._root_fd)
        _directory(parent, private=True)
        _directory(current, private=True)
        named = os.stat(self._name, dir_fd=self._parent_fd, follow_symlinks=False)
        _need(_identity(parent) == self._parent_identity
              and _identity(current) == _identity(named) == self._root_identity,
              "identity_changed")

    def _names(self, fd, deadline_ns):
        _deadline(deadline_ns)
        names = []
        with os.scandir(fd) as entries:
            for entry in entries:
                _deadline(deadline_ns)
                self._directory_entries += 1
                _need(self._directory_entries <= MAXIMUM_DIRECTORY_ENTRIES, "bounds_exhausted")
                name = entry.name
                _need(type(name) is str and 0 < len(name) <= 255
                      and name not in (".", "..") and "/" not in name and "\0" not in name)
                _need(len(names) < MAXIMUM_NODES, "bounds_exhausted")
                names.append(name)
        _deadline(deadline_ns)
        _need(len(names) == len(set(names)))
        return set(names)

    def _pin_child(self, parent_fd, name, expected, mount_id, deadline_ns, *, directory=False,
                   stable=False):
        flags = READ_FLAGS | os.O_DIRECTORY if directory else PIN_FLAGS
        fd = self._keep(os.open(name, flags,
                                dir_fd=parent_fd))
        try:
            observed = os.fstat(fd)
            _need((_identity(observed) == expected[:5] if stable else _state(observed) == expected),
                  "identity_changed")
            _need(self._mount_id(fd, deadline_ns) == mount_id, "mount_rejected")
            return fd
        except BaseException:
            self._close(fd)
            raise

    def _census(self, deadline_ns):
        self._scope()
        root_mount = self._mount_id(self._root_fd, deadline_ns)
        _need(root_mount == self._mount_id(self._parent_fd, deadline_ns), "mount_rejected")
        nodes = {(): {"fd": self._root_fd, "parent": self._parent_fd, "name": self._name,
                      "state": _state(os.fstat(self._root_fd)), "directory": True}}
        seen = {nodes[()]["state"][:2]}
        byte_count, pending = 0, [()]
        while pending:
            relative = pending.pop()
            node = nodes[relative]
            names = self._names(node["fd"], deadline_ns)
            if not relative:
                _need(names <= TOP_LEVEL)
            node["children"] = names
            for name in sorted(names):
                _deadline(deadline_ns)
                key = (*relative, name)
                _need(len(nodes) < MAXIMUM_NODES and len(key) <= MAXIMUM_DEPTH,
                      "bounds_exhausted")
                info = os.stat(name, dir_fd=node["fd"], follow_symlinks=False)
                directory = stat.S_ISDIR(info.st_mode)
                _need(directory or stat.S_ISREG(info.st_mode))
                _need(bool(relative) or directory)
                _need(info.st_ino > 0 and info.st_dev == nodes[()]["state"][0]
                      and info.st_uid == info.st_gid and info.st_uid in (0, 10001))
                _need((info.st_dev, info.st_ino) not in seen)
                seen.add((info.st_dev, info.st_ino))
                if not directory:
                    _need(info.st_nlink == 1 and type(info.st_size) is int and info.st_size >= 0)
                    byte_count += info.st_size
                    _need(byte_count <= MAXIMUM_FILE_BYTES, "bounds_exhausted")
                observed = _state(info)
                fd = self._pin_child(node["fd"], name, observed, root_mount, deadline_ns,
                                     directory=directory)
                nodes[key] = {"fd": fd, "parent": node["fd"], "name": name,
                              "state": observed, "directory": directory}
                if directory:
                    pending.append(key)
        # Validate the complete census again before the first removal. A later
        # external mutation still rejects at its exact node, never broadens scope.
        for node in nodes.values():
            _deadline(deadline_ns)
            _need(_state(os.fstat(node["fd"])) == node["state"], "identity_changed")
            _need(self._mount_id(node["fd"], deadline_ns) == root_mount, "mount_rejected")
            if node["directory"]:
                _need(self._names(node["fd"], deadline_ns) == node["children"], "identity_changed")
        return nodes, byte_count, root_mount

    def cleanup(self, *, descendants_reaped, native_mounts_released, deadline_ns):
        _runtime()
        _need(os.getpid() == self._owner_pid, "owner_mismatch")
        _need(not self._cleanup_attempted, "already_attempted")
        _need(self._created and not self._closed, "workspace_closed")
        self._cleanup_attempted = True
        failure, files, directories = None, 0, 0
        try:
            _need(descendants_reaped is True and native_mounts_released is True,
                  "arguments_rejected")
            _deadline(deadline_ns)
            self._proc()
            nodes, byte_count, mount_id = self._census(deadline_ns)
            for key in sorted(nodes, key=lambda item: (len(item), item), reverse=True):
                self._scope()
                node = nodes[key]
                fresh = self._pin_child(node["parent"], node["name"], node["state"],
                    mount_id, deadline_ns, directory=node["directory"], stable=node["directory"])
                try:
                    if node["directory"]:
                        _need(not self._names(fresh, deadline_ns), "identity_changed")
                        _deadline(deadline_ns)
                        os.rmdir(node["name"], dir_fd=node["parent"])
                        directories += 1
                    else:
                        _deadline(deadline_ns)
                        os.unlink(node["name"], dir_fd=node["parent"])
                        files += 1
                finally:
                    self._close(fresh)
        except BaseException as error:
            failure = error.code if type(error) is WorkspaceError else "cleanup_failed"
        finally:
            try:
                self._close_all()
            except BaseException:
                failure = "close_failed"
        if failure:
            raise WorkspaceError(failure) from None
        _deadline(deadline_ns)
        receipt = {"schema": SCHEMA, "status": "removed", "role": self._role,
            "workspace_removed": True, "census_nodes": len(nodes), "removed_files": files,
            "removed_directories": directories, "declared_file_bytes": byte_count,
            "kernel_metadata_bytes_read": self._kernel_bytes,
            "kernel_metadata_read_calls": self._kernel_calls, "source_payload_bytes_read": 0,
            "directory_entries_observed": self._directory_entries,
            "identity_checked_removal": True, "mount_boundary_checks_passed": True,
            "descendants_reaped_premise": True, "native_mounts_released_premise": True,
            "descendant_reaping_verified_by_adapter": False,
            "native_mount_release_verified_by_adapter": False,
            "adopted_existing_directory": False, "external_mutation_excluded": False,
            "retry_authorized": False, "production_eligible": False}
        return validate_cleanup_receipt(receipt, role=self._role)


def validate_cleanup_receipt(value, *, role):
    """Pure closed receipt check, not an independent proof that cleanup occurred."""
    _need(type(role) is str and role in ROLES, "arguments_rejected")
    fixed = {"schema": SCHEMA, "status": "removed", "role": role, "workspace_removed": True,
        "source_payload_bytes_read": 0, "identity_checked_removal": True,
        "mount_boundary_checks_passed": True, "descendants_reaped_premise": True,
        "native_mounts_released_premise": True, "descendant_reaping_verified_by_adapter": False,
        "native_mount_release_verified_by_adapter": False, "adopted_existing_directory": False,
        "external_mutation_excluded": False, "retry_authorized": False,
        "production_eligible": False}
    bounds = {"census_nodes": (1, MAXIMUM_NODES), "removed_files": (0, MAXIMUM_NODES - 1),
        "removed_directories": (1, MAXIMUM_NODES), "declared_file_bytes": (0, MAXIMUM_FILE_BYTES),
        "kernel_metadata_read_calls": (8, MAXIMUM_KERNEL_READ_CALLS),
        "kernel_metadata_bytes_read": (1, MAXIMUM_KERNEL_READ_BYTES),
        "directory_entries_observed": (0, MAXIMUM_DIRECTORY_ENTRIES)}
    _need(type(value) is dict and set(value) == set(fixed) | set(bounds)
          and all(type(value[key]) is type(expected) and value[key] == expected
                  for key, expected in fixed.items())
          and all(type(value[key]) is int and low <= value[key] <= high
                  for key, (low, high) in bounds.items()))
    _need(value["removed_files"] + value["removed_directories"] == value["census_nodes"]
          and value["directory_entries_observed"] == 2 * (value["census_nodes"] - 1)
          and value["kernel_metadata_read_calls"] == 6 * value["census_nodes"] + 2
          and value["kernel_metadata_bytes_read"] <= value["kernel_metadata_read_calls"] * 1026)
    return dict(value)


def create_workspace(parent: Path, *, role, deadline_ns):
    """Create one exclusive child, not a path-adoption or general directory-removal API."""
    _runtime()
    _deadline(deadline_ns)
    _need(type(parent) is type(Path()) and parent.is_absolute()
          and str(parent) != "/" and ".." not in parent.parts and type(role) is str
          and role in ROLES, "arguments_rejected")
    try:
        rootfs._absolute(parent)
    except BaseException:
        raise WorkspaceError("arguments_rejected") from None
    name = "lumi-fixture-workspace-" + role + "-" + uuid.uuid4().hex
    owned = OwnedFixtureWorkspace(_CREATE_TOKEN, parent, role, name)
    failure = None
    try:
        owned._parent_fd = owned._keep(rootfs._open_absolute(parent, directory=True))
        parent_info = os.fstat(owned._parent_fd)
        _directory(parent_info, private=True)
        owned._parent_identity = _identity(parent_info)
        _deadline(deadline_ns)
        os.mkdir(name, 0o700, dir_fd=owned._parent_fd)
        owned._created = True  # Own successful creation before any subsequent operation.
        info = os.stat(name, dir_fd=owned._parent_fd, follow_symlinks=False)
        _directory(info, private=True)
        owned._root_identity = _identity(info)
        owned._root_fd = owned._keep(os.open(name, READ_FLAGS | os.O_DIRECTORY,
                                            dir_fd=owned._parent_fd))
        owned._scope()
        _deadline(deadline_ns)
        return owned
    except BaseException as error:
        failure = error.code if type(error) is WorkspaceError else "creation_failed"
        if owned._created:
            try:
                _need(owned._root_identity is not None, "creation_cleanup_unconfirmed")
                info = os.stat(name, dir_fd=owned._parent_fd, follow_symlinks=False)
                _need(_identity(info) == owned._root_identity, "creation_cleanup_unconfirmed")
                # Empty-only removal: no descendant traversal on failed creation.
                os.rmdir(name, dir_fd=owned._parent_fd)
            except BaseException:
                failure = "creation_cleanup_unconfirmed"
        try:
            owned._close_all()
        except BaseException:
            failure = "creation_cleanup_unconfirmed"
    raise WorkspaceError(failure) from None
