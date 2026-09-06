"""Bounded, self-only Linux containment observations; NEVER an admission receipt.

Reads kernel metadata only. No cgroup/namespace configuration, write test, kill,
subprocess, socket, GPU, model or target-host connection is implemented here.
Finite limits, a writable mount, or a namespace inode do not prove delegation,
exclusive membership, escape prevention, cleanup or network isolation.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import os
import stat
import threading
import time

from . import posttraining_sft_host_snapshot_v1 as snapshot
from . import posttraining_sft_kernel_policy_v1 as runtime

MAXIMUM_SECONDS = 5
MAXIMUM_OPENS = 128
MAXIMUM_FILES = 32
MAXIMUM_READ_CALLS = 128
MAXIMUM_READ_CAPACITY = 524_288
PROC_MAGIC = 0x9FA0
SYS_MAGIC = 0x62656572
CGROUP_MAGIC = 0x63677270
NS_MAGIC = 0x6E736673
NS_GET_NSTYPE = 0xB703
NAMESPACES = {"mnt": 0x00020000, "cgroup": 0x02000000, "net": 0x40000000}
SCALARS = {
    "memory.max": True, "memory.high": True, "memory.current": False,
    "memory.swap.max": True, "memory.oom.group": False,
    "pids.max": True, "pids.current": False,
}
UNVERIFIED = (
    "target_host_bound", "dedicated_subtree_verified", "delegation_verified",
    "aggregate_memory_enforcement_verified", "descendant_cleanup_verified",
    "network_isolation_verified", "real_worker_profile_verified",
    "provider_termination_verified", "execution_authorized",
    "real_sft_execution_permitted", "production_eligible", "full_private_integration_passed",
)
ERROR_CODES = snapshot.ERROR_CODES | runtime.ERROR_CODES | frozenset({
    "host_observation_failed", "host_io_budget_exhausted", "host_deadline_exhausted",
    "host_filesystem_rejected", "host_io_rejected", "host_cleanup_failed",
    "host_snapshot_changed", "host_namespace_rejected", "host_configuration_rejected",
})


class ObservationError(ValueError):
    def __init__(self, code):
        self.code = code if type(code) is str and code in ERROR_CODES else "host_observation_failed"
        super().__init__(self.code)


def need(condition, code):
    if not condition:
        raise ObservationError(code) from None


class StatFS(ctypes.Structure):
    """Linux x86-64 LP64 libc statfs layout; other ABIs are rejected before I/O."""

    _fields_ = [
        ("kind", ctypes.c_long), ("block_size", ctypes.c_long),
        ("blocks", ctypes.c_ulong), ("free_blocks", ctypes.c_ulong),
        ("available_blocks", ctypes.c_ulong), ("files", ctypes.c_ulong),
        ("free_files", ctypes.c_ulong), ("fsid", ctypes.c_int * 2),
        ("name_length", ctypes.c_long), ("fragment_size", ctypes.c_long),
        ("flags", ctypes.c_long), ("spare", ctypes.c_long * 4),
    ]


class Reader:
    """Prepaid read capacities; pinned descriptors; one close attempt per handle.

    Reads continue to EOF within the single reserved capacity, so a short kernel
    read cannot silently hide a trailing mount or malformed record. File size is
    not used: genuine procfs/cgroupfs interfaces commonly report st_size == 0.
    Timers cannot bound a blocking kernel syscall or interpreter startup.
    """

    def __init__(self, deadline):
        self.deadline = deadline
        self.pending = []
        self.opens = self.files = self.read_calls = self.capacity = 0
        self.bytes_read = self.close_attempts = self.closed = self.missing = 0
        self.cleanup_failed = False
        need(ctypes.sizeof(StatFS) == 120 and StatFS.fsid.offset == 56
             and StatFS.spare.offset == 88, "host_filesystem_rejected")
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.fstatfs = self.libc.fstatfs
        self.fstatfs.argtypes = [ctypes.c_int, ctypes.POINTER(StatFS)]
        self.fstatfs.restype = ctypes.c_int

    def check(self):
        need(time.monotonic() < self.deadline, "host_deadline_exhausted")

    def open(self, name, *, parent=None, directory=False, namespace=False):
        self.check()
        need(type(name) is str and (name in {"/proc", "/sys"} if parent is None else
             0 < len(name) <= 255 and "/" not in name and name not in {".", ".."}),
             "host_configuration_rejected")
        need(not namespace or parent is not None and name in NAMESPACES and not directory,
             "host_configuration_rejected")
        need(self.opens < MAXIMUM_OPENS, "host_io_budget_exhausted")
        self.opens += 1
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
        if not namespace:
            flags |= os.O_NOFOLLOW
        if directory:
            flags |= os.O_DIRECTORY
        fd = os.open(name, flags, dir_fd=parent)
        try:
            self.pending.append(fd)  # Own before validation, including allocation failures.
        except BaseException:
            self.close_attempts += 1
            try:
                os.close(fd)
                self.closed += 1
            except BaseException:
                self.cleanup_failed = True
            raise ObservationError("host_io_rejected") from None
        info = os.fstat(fd)
        need(stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode),
             "host_filesystem_rejected")
        self.check()
        return fd

    def filesystem(self, fd, expected, device=None):
        self.check()
        value = StatFS()
        need(self.fstatfs(fd, ctypes.byref(value)) == 0 and value.kind == expected,
             "host_filesystem_rejected")
        info = os.fstat(fd)
        if device is not None:
            need((os.major(info.st_dev), os.minor(info.st_dev)) == device,
                 "host_filesystem_rejected")
        self.check()
        return info.st_dev, info.st_ino

    def close(self, fd):
        need(fd in self.pending, "host_cleanup_failed")
        self.pending.remove(fd)  # A failed close must NEVER retry a possibly reused fd.
        self.close_attempts += 1
        try:
            os.close(fd)
            self.closed += 1
        except BaseException:
            self.cleanup_failed = True
            raise ObservationError("host_cleanup_failed") from None

    def finish(self):
        while self.pending:
            try:
                self.close(self.pending[-1])
            except BaseException:
                self.cleanup_failed = True
        return not self.cleanup_failed

    def directory(self, name, parent, magic, device=None):
        fd = self.open(name, parent=parent, directory=True)
        self.filesystem(fd, magic, device)
        return fd

    def walk(self, parent, parts, device):
        for part in parts:
            parent = self.directory(part, parent, CGROUP_MAGIC, device)
        return parent

    def read(self, name, parent, capacity, magic, *, device=None, optional=False):
        self.check()
        need(type(capacity) is int and 0 < capacity <= 131_072,
             "host_configuration_rejected")
        need(self.files < MAXIMUM_FILES
             and self.capacity + capacity + 1 <= MAXIMUM_READ_CAPACITY,
             "host_io_budget_exhausted")
        self.files += 1
        self.capacity += capacity + 1
        try:
            fd = self.open(name, parent=parent)
        except OSError as error:
            if not optional or error.errno != errno.ENOENT:
                raise
            self.missing += 1
            return None
        try:
            self.filesystem(fd, magic, device)
            chunks, size = [], 0
            while True:
                self.check()
                need(self.read_calls < MAXIMUM_READ_CALLS, "host_io_budget_exhausted")
                self.read_calls += 1
                raw = os.read(fd, min(32_768, capacity + 1 - size))
                self.bytes_read += len(raw)
                size += len(raw)
                need(size <= capacity, "host_io_budget_exhausted")
                if not raw:
                    self.check()
                    return b"".join(chunks)
                chunks.append(raw)
        finally:
            self.close(fd)

    def namespace(self, parent, name):
        fd = self.open(name, parent=parent, namespace=True)
        identity = self.filesystem(fd, NS_MAGIC)
        need(fcntl.ioctl(fd, NS_GET_NSTYPE) == NAMESPACES[name], "host_namespace_rejected")
        self.check()
        # Keep the namespace fd pinned until all second snapshots are checked.
        return identity

    def interface_exists(self, parent, name, device):
        self.check()
        try:
            info = os.stat(name, dir_fd=parent, follow_symlinks=False)
        except OSError as error:
            if error.errno != errno.ENOENT:
                raise
            return False
        need(stat.S_ISREG(info.st_mode)
             and (os.major(info.st_dev), os.minor(info.st_dev)) == device,
             "host_filesystem_rejected")
        return True

    def summary(self):
        return {
            "open_attempts": self.opens, "file_capacity_reservations": self.files,
            "read_capacity_bytes": self.capacity, "read_calls": self.read_calls,
            "metadata_bytes_read": self.bytes_read, "optional_files_missing": self.missing,
            "close_attempts": self.close_attempts, "descriptors_closed": self.closed,
            "pending_descriptors": len(self.pending), "cleanup_complete": not self.cleanup_failed,
        }


def collect(reader):
    proc = reader.directory("/proc", None, PROC_MAGIC)
    own = reader.directory(str(os.getpid()), proc, PROC_MAGIC)
    ns = reader.directory("ns", own, PROC_MAGIC)
    namespaces = {name: reader.namespace(ns, name) for name in NAMESPACES}
    membership = snapshot.parse_membership(reader.read("cgroup", own, 4096, PROC_MAGIC))
    mount = snapshot.parse_mountinfo(reader.read("mountinfo", own, 131_072, PROC_MAGIC))
    device = mount["device"]
    sysroot = reader.directory("/sys", None, SYS_MAGIC)
    fs = reader.directory("fs", sysroot, SYS_MAGIC)
    cgroot = reader.directory("cgroup", fs, CGROUP_MAGIC, device)
    group = reader.walk(cgroot, membership, device)
    identity = reader.filesystem(group, CGROUP_MAGIC, device)
    def read(name):
        return reader.read(name, group, 4096, CGROUP_MAGIC, device=device, optional=True)
    controllers_raw = read("cgroup.controllers")
    enabled_raw = read("cgroup.subtree_control")
    controllers = snapshot.parse_controllers(controllers_raw) if controllers_raw is not None else ()
    enabled = snapshot.parse_controllers(enabled_raw) if enabled_raw is not None else ()
    group_type = read("cgroup.type")
    need(group_type in {None, b"domain\n", b"domain threaded\n",
                        b"domain invalid\n", b"threaded\n"},
         "host_io_rejected")
    events_raw = read("cgroup.events")
    events = snapshot.parse_events(events_raw) if events_raw is not None else {}
    scalar_values = {}
    for name, allow_max in SCALARS.items():
        raw = read(name)
        value = snapshot.parse_scalar(raw, allow_max=allow_max) if raw is not None else None
        if name == "memory.oom.group" and value is not None:
            need(value in {0, 1}, "host_scalar_rejected")
        scalar_values[name.replace(".", "_")] = {"present": raw is not None, "value": value}
    kill_exists = reader.interface_exists(group, "cgroup.kill", device)
    net = reader.directory("net", own, PROC_MAGIC)
    network = snapshot.parse_net_dev(reader.read("dev", net, 16_384, PROC_MAGIC))
    need(snapshot.parse_membership(reader.read("cgroup", own, 4096, PROC_MAGIC)) == membership,
         "host_snapshot_changed")
    need(snapshot.parse_mountinfo(reader.read("mountinfo", own, 131_072, PROC_MAGIC)) == mount,
         "host_snapshot_changed")
    refreshed_root = reader.directory("cgroup", fs, CGROUP_MAGIC, device)
    refreshed = reader.walk(refreshed_root, membership, device)
    need(reader.filesystem(refreshed, CGROUP_MAGIC, device) == identity, "host_snapshot_changed")
    need(all(reader.namespace(ns, name) == identity for name, identity in namespaces.items()),
         "host_snapshot_changed")
    return {
        "cgroup2_filesystem_observed": True, "unified_membership_observed": True,
        "membership_nonroot_in_visible_namespace": bool(membership),
        "mount_read_only_observed": mount["mount_read_only"],
        "current_group_domain_observed": group_type == b"domain\n",
        "memory_controller_available": "memory" in controllers,
        "pids_controller_available": "pids" in controllers,
        "memory_controller_enabled_for_children": "memory" in enabled,
        "pids_controller_enabled_for_children": "pids" in enabled,
        "cgroup_populated": events.get("populated"), "cgroup_frozen": events.get("frozen"),
        "cgroup_kill_interface_observed": kill_exists,
        "namespace_snapshots_checked": len(namespaces), "namespace_endpoints_stable": True,
        "resource_snapshots": scalar_values, "network_snapshot": network,
    }


def observe_self():
    """One read-only observation, with no caller-selected host/PID/path or retry.

    Always leaves every admission/permission field false. Even a complete
    observation is not a receipt accepted by the existing launcher/bootstrap.
    It cannot establish continuous stability or trusted host ownership.
    """
    reader = None
    observation = {}
    failure = None
    deadline = time.monotonic() + MAXIMUM_SECONDS
    try:
        runtime.require_abi()  # Read-only; no install or limit mutation.
        need(threading.current_thread() is threading.main_thread()
             and threading.active_count() == 1,
             "host_configuration_rejected")
        reader = Reader(deadline)
        observation = collect(reader)
        reader.check()
    except BaseException as error:
        code = getattr(error, "code", None)
        failure = code if type(code) is str and code in ERROR_CODES else "host_observation_failed"
    if reader is not None and not reader.finish():
        failure = "host_cleanup_failed"
    if failure is None and time.monotonic() >= deadline:
        failure = "host_deadline_exhausted"
    if failure is not None:
        observation = {}  # Never retain a partially checked snapshot as complete evidence.
    return {
        "schema_version": "lumi.sft-host-containment-observation.v0.1",
        "state": "observed" if failure is None else (
            "unsupported" if failure == "kernel_policy_unavailable" else "failed"
        ),
        "failure_code": failure, "observation_complete": failure is None,
        "observations": observation,
        "accounting": reader.summary() if reader is not None else {
            "open_attempts": 0, "file_capacity_reservations": 0, "read_capacity_bytes": 0,
            "read_calls": 0, "metadata_bytes_read": 0, "optional_files_missing": 0,
            "close_attempts": 0, "descriptors_closed": 0, "pending_descriptors": 0,
            "cleanup_complete": True,
        },
        "required_next_evidence": [
            "exact_new_host_and_boot_binding", "owned_dedicated_delegated_subtree",
            "approved_aggregate_limit_installation_and_readbacks",
            "exclusive_worker_membership_and_escape_prevention",
            "owned_descendant_retirement_and_empty_subtree",
            "owned_network_namespace_policy_and_inherited_fd_boundary",
            "separately_reviewed_real_worker_profile", "provider_termination_adapter",
        ],
        **dict.fromkeys(UNVERIFIED, False),
        "host_mutations": 0, "child_processes_started": 0, "external_network_bytes": 0,
        "model_or_dataset_reads": 0, "private_artifact_writes": 0,
        "gpu_use": 0, "inference_calls": 0,
    }
