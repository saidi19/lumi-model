#!/usr/bin/env python3
"""One exact SDK removal on a fresh trusted GitHub VM, then one frozen setup probe.

Never execute this entrypoint on the Mac. The only deletion root is the fixed
preinstalled Android SDK, never a CLI path. Census/deletion are descriptor-relative
and never follow symlinks. Internal symlink leaves are unlinked, not traversed.
Mounts, hardlinked files, special nodes, identity changes and unbounded trees fail
closed. A trusted disposable VM without concurrent writers is an explicit premise.
No file contents in the SDK are read, and no names/paths/errors enter the report.
No image acquisition, Docker command, model/data access or qualification occurs.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import stat
import sys
import threading
import types
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
TARGET = "/usr/local/lib/android/sdk"
PARTS = ("usr", "local", "lib", "android", "sdk")
PROBE = "scripts/probe_sft_prepared_setup_v1.py"
PROBE_SHA256 = "121104c9011ea8de23b05c0af03640e52b28a6a23d2acdbba75f68169dab5da1"
SCHEMA = "lumi.sdk-capacity-preflight.v0.1"
MAXIMUM_NODES = 250_000
MAXIMUM_DEPTH = 48
MAXIMUM_NAME_BYTES = 32 * 1024**2
MAXIMUM_ALLOCATED_BYTES = 64 * 1024**3
MAXIMUM_CLEANUP_SECONDS = 240
MAXIMUM_REPORT_BYTES = 16384
STAGES = ("runtime", "probe_binding", "target_binding", "census", "removal", "capacity",
          "setup_probe", "complete")
CODES = frozenset({"ok", "capacity_insufficient", "runtime_rejected", "binding_rejected",
                   "target_rejected", "tree_rejected", "budget_exhausted", "identity_changed",
                   "deadline_exhausted", "already_attempted", "removal_unconfirmed",
                   "setup_failed", "operation_failed"})
FALSE_FLAGS = ("gpu_used", "real_model_or_data_access", "software_image_acquired",
               "native_qualification_passed", "training_authorized", "production_eligible",
               "historical_failure_cause_proven", "old_attempt_retried",
               "concurrent_writer_exclusion_verified")
_ATTEMPTED = False


class ReclaimError(ValueError):
    def __init__(self, code="operation_failed"):
        self.code = code if type(code) is str and code in CODES else "operation_failed"
        super().__init__(self.code)


def need(value, code="operation_failed"):
    if not value:
        raise ReclaimError(code) from None


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                       allow_nan=False).encode("ascii") + b"\n")


def identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid)


def file_state(info):
    return (*identity(info), info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def runtime():
    need(sys.platform == "linux" and os.uname().machine == "x86_64"
         and sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 12)
         and sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode
         and os.getresuid() == os.getresgid() == (0, 0, 0)
         and os.environ.get("LUMI_SDK_CAPACITY_CI") == "1"
         and signal.getsignal(signal.SIGALRM) == signal.SIG_DFL
         and signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
         and threading.current_thread() is threading.main_thread()
         and threading.active_count() == 1, "runtime_rejected")


def expired(_signum, _frame):
    raise ReclaimError("deadline_exhausted") from None


def arm_deadline():
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(MAXIMUM_CLEANUP_SECONDS)


def disarm_deadline():
    signal.alarm(0)
    signal.signal(signal.SIGALRM, signal.SIG_DFL)


def load_probe():
    path = ROOT / PROBE
    need(not any(p.is_symlink() for p in (path, *path.parents)), "binding_rejected")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
             and not before.st_mode & 0o022 and 0 < before.st_size <= 32768, "binding_rejected")
        raw = bytearray()
        while len(raw) < before.st_size:
            part = os.read(fd, before.st_size - len(raw))
            need(part, "binding_rejected")
            raw.extend(part)
        need(file_state(before) == file_state(os.fstat(fd)) == file_state(path.lstat())
             and not os.read(fd, 1), "binding_rejected")
        need(hashlib.sha256(raw).hexdigest() == PROBE_SHA256, "binding_rejected")
    finally:
        os.close(fd)
    module = types.ModuleType("_lumi_sdk_capacity_probe_v1")
    module.__file__ = str(path)
    exec(compile(bytes(raw), str(path), "exec"), module.__dict__)
    return module


def check_mounts():
    # st_dev alone cannot detect same-device bind mounts; inspect the kernel table too.
    with open("/proc/self/mountinfo", "rb") as source:
        raw = source.read(1024**2 + 1)
    need(0 < len(raw) <= 1024**2, "target_rejected")
    for line in raw.splitlines():
        fields = line.split()
        need(len(fields) >= 10 and b" - " in line, "target_rejected")
        mount = fields[4]
        # The exact ASCII SDK root has no mountinfo escape characters.
        need(mount != TARGET.encode() and not mount.startswith(TARGET.encode() + b"/"),
             "target_rejected")


def open_target():
    handles = []
    try:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        handles.append(os.open("/", flags))
        for part in PARTS:
            info = os.stat(part, dir_fd=handles[-1], follow_symlinks=False)
            need(stat.S_ISDIR(info.st_mode), "target_rejected")
            child = os.open(part, flags, dir_fd=handles[-1])
            handles.append(child)
            need(identity(info) == identity(os.fstat(child)), "identity_changed")
        info = os.fstat(handles[-1])
        need(info.st_dev == os.stat("/tmp", follow_symlinks=False).st_dev, "target_rejected")
        check_mounts()
        return handles, identity(info)
    except BaseException:
        for fd in reversed(handles):
            os.close(fd)
        raise


def names(fd, count_limit):
    result = []
    with os.scandir(fd) as entries:
        for entry in entries:
            need(len(result) < count_limit, "budget_exhausted")
            name = entry.name
            need(type(name) is str and name not in {"", ".", ".."} and "/" not in name,
                 "tree_rejected")
            result.append(name)
    return sorted(result)


def census(root_fd):
    device = os.fstat(root_fd).st_dev
    nodes, children = {}, {}
    total_names = total_allocated = 0

    def walk(fd, relative):
        nonlocal total_names, total_allocated
        need(len(relative) < MAXIMUM_DEPTH, "budget_exhausted")
        children[relative] = names(fd, MAXIMUM_NODES - len(nodes))
        for name in children[relative]:
            need(len(nodes) < MAXIMUM_NODES, "budget_exhausted")
            path = (*relative, name)
            total_names += sum(len(os.fsencode(part)) + 1 for part in path)
            need(total_names <= MAXIMUM_NAME_BYTES, "budget_exhausted")
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            need(info.st_dev == device, "tree_rejected")
            kind = stat.S_IFMT(info.st_mode)
            need(kind in (stat.S_IFDIR, stat.S_IFREG, stat.S_IFLNK), "tree_rejected")
            need(kind != stat.S_IFREG or info.st_nlink == 1, "tree_rejected")
            total_allocated += info.st_blocks * 512
            need(0 <= total_allocated <= MAXIMUM_ALLOCATED_BYTES, "budget_exhausted")
            nodes[path] = file_state(info)
            if kind == stat.S_IFDIR:
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                dir_fd=fd)
                try:
                    need(identity(os.fstat(child)) == identity(info), "identity_changed")
                    walk(child, path)
                finally:
                    os.close(child)
        need(children[relative] == names(fd, MAXIMUM_NODES), "identity_changed")

    walk(root_fd, ())
    return nodes, children, total_allocated


def remove_tree(root_fd, tree, report):
    nodes, children, _allocated = tree

    def walk(fd, relative):
        need(names(fd, MAXIMUM_NODES) == children[relative], "identity_changed")
        for name in children[relative]:
            path = (*relative, name)
            expected = nodes[path]
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            kind = stat.S_IFMT(expected[2])
            if kind == stat.S_IFDIR:
                need(identity(info) == expected[:5], "identity_changed")
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                dir_fd=fd)
                try:
                    need(identity(os.fstat(child)) == expected[:5], "identity_changed")
                    walk(child, path)
                finally:
                    os.close(child)
                need(identity(os.stat(name, dir_fd=fd, follow_symlinks=False)) == expected[:5],
                     "identity_changed")
                os.rmdir(name, dir_fd=fd)
                report["removed_directories"] += 1
            else:
                need(file_state(info) == expected, "identity_changed")
                # Unlink a symlink leaf; never open its target.
                os.unlink(name, dir_fd=fd)
                report["removed_leaves"] += 1
        need(not names(fd, MAXIMUM_NODES), "identity_changed")

    walk(root_fd, ())


def empty_report():
    return {"schema": SCHEMA, "status": "failed", "stage": "runtime", "code": "operation_failed",
            "census_nodes": 0, "census_allocated_bytes": 0, "removed_leaves": 0,
            "removed_directories": 0, "removal_started": False, "sdk_root_removed": False,
            "sdk_root_absence_verified": False, "capacity_before": None, "capacity_after": None,
            "setup_report": None, "ready_for_unchanged_setup": False,
            **dict.fromkeys(FALSE_FLAGS, False)}


def run():
    global _ATTEMPTED
    report, handles, armed, probe = empty_report(), [], False, None
    try:
        runtime()
        need(not _ATTEMPTED, "already_attempted")
        _ATTEMPTED = True
        report["stage"] = "probe_binding"
        probe = load_probe()
        probe.runtime()
        probe.install_guard()
        armed = True
        arm_deadline()
        report["stage"] = "target_binding"
        handles, bound = open_target()
        report["capacity_before"] = probe.capacity(os.statvfs("/tmp"))
        report["stage"] = "census"
        tree = census(handles[-1])
        report["census_nodes"], report["census_allocated_bytes"] = len(tree[0]), tree[2]
        need(tree[0], "tree_rejected")
        need(identity(os.stat(PARTS[-1], dir_fd=handles[-2], follow_symlinks=False)) == bound,
             "identity_changed")
        check_mounts()
        report["stage"] = "removal"
        report["removal_started"] = True  # A later error/timeout never implies no mutation.
        remove_tree(handles[-1], tree, report)
        need(identity(os.stat(PARTS[-1], dir_fd=handles[-2], follow_symlinks=False)) == bound,
             "identity_changed")
        os.rmdir(PARTS[-1], dir_fd=handles[-2])
        report["sdk_root_removed"] = True
        try:
            os.stat(PARTS[-1], dir_fd=handles[-2], follow_symlinks=False)
        except FileNotFoundError:
            report["sdk_root_absence_verified"] = True
        need(report["sdk_root_absence_verified"], "removal_unconfirmed")
        report["stage"] = "capacity"
        # Release all SDK authority and restore SIGALRM before the frozen probe's own timer.
        for fd in reversed(handles):
            os.close(fd)
        handles = []
        report["capacity_after"] = probe.capacity(os.statvfs("/tmp"))
        disarm_deadline()
        armed = False
        if not report["capacity_after"]["acquisition_headroom_met"]:
            report["status"], report["stage"], report["code"] = (
                "observed", "complete", "capacity_insufficient")
            validate_report(report, probe)
            return report
        report["stage"] = "setup_probe"
        setup = probe.run()
        probe.validate_report(setup)
        report["setup_report"] = setup
        need(setup["status"] == "observed", "setup_failed")
        report["ready_for_unchanged_setup"] = report["setup_report"]["ready_for_unchanged_setup"]
        report["status"], report["stage"] = "observed", "complete"
        report["code"] = "ok" if report["ready_for_unchanged_setup"] else "capacity_insufficient"
    except BaseException as error:
        report["code"] = error.code if type(error) is ReclaimError else "operation_failed"
    finally:
        for fd in reversed(handles):
            try:
                os.close(fd)
            except OSError:
                report["status"], report["code"] = "failed", "operation_failed"
        if armed:
            disarm_deadline()
    if report["status"] == "failed":
        report["ready_for_unchanged_setup"] = False
    validate_report(report, probe)
    return report


def validate_report(value, probe=None):
    need(type(value) is dict and set(value) == set(empty_report()))
    need(value["schema"] == SCHEMA and type(value["schema"]) is str)
    need(type(value["status"]) is str and value["status"] in {"observed", "failed"}
         and type(value["stage"]) is str and value["stage"] in STAGES
         and type(value["code"]) is str and value["code"] in CODES)
    for key in FALSE_FLAGS:
        need(value[key] is False)
    for key in ("removal_started", "sdk_root_removed", "sdk_root_absence_verified",
                "ready_for_unchanged_setup"):
        need(type(value[key]) is bool)
    for key, maximum in (("census_nodes", MAXIMUM_NODES),
                          ("census_allocated_bytes", MAXIMUM_ALLOCATED_BYTES),
                          ("removed_leaves", MAXIMUM_NODES),
                          ("removed_directories", MAXIMUM_NODES)):
        need(type(value[key]) is int and 0 <= value[key] <= maximum)
    removed = value["removed_leaves"] + value["removed_directories"]
    need(removed <= value["census_nodes"])
    need(not (removed or value["sdk_root_removed"]) or value["removal_started"])
    need(not value["sdk_root_absence_verified"] or value["sdk_root_removed"])
    for key in ("capacity_before", "capacity_after"):
        part = value[key]
        if part is not None:
            need(probe is not None and type(part) is dict
                 and set(part) == set(probe.DISK_INTS) | set(probe.DISK_BOOLS))
            need(all(type(part[k]) is int for k in probe.DISK_INTS)
                 and all(type(part[k]) is bool for k in probe.DISK_BOOLS))
            fragment = part["filesystem_fragment_bytes"]
            need(fragment > 0 and part["available_disk_bytes"] % fragment == 0)
            need(part == probe.capacity(types.SimpleNamespace(
                f_bavail=part["available_disk_bytes"] // fragment, f_frsize=fragment)))
    if value["setup_report"] is not None:
        need(probe is not None)
        probe.validate_report(value["setup_report"])
    if value["status"] == "observed":
        need(value["stage"] == "complete" and value["sdk_root_absence_verified"]
             and removed == value["census_nodes"] > 0
             and value["capacity_before"] is not None and value["capacity_after"] is not None)
        if value["setup_report"] is None:
            need(not value["capacity_after"]["acquisition_headroom_met"]
                 and not value["ready_for_unchanged_setup"]
                 and value["code"] == "capacity_insufficient")
        else:
            need(value["setup_report"]["status"] == "observed"
                 and value["ready_for_unchanged_setup"]
                 == value["setup_report"]["ready_for_unchanged_setup"]
                 and value["code"] == ("ok" if value["ready_for_unchanged_setup"]
                                        else "capacity_insufficient"))
    else:
        need(not value["ready_for_unchanged_setup"] and value["code"] not in {"ok",
                                                                               "capacity_insufficient"})
    need(len(canonical(value)) <= MAXIMUM_REPORT_BYTES)
    return value


def main():
    if len(sys.argv) != 2 or sys.argv[1] != "--fresh-sdk-capacity-check":
        result = empty_report()
        result["code"] = "runtime_rejected"
    else:
        result = run()
    raw = canonical(result)
    need(len(raw) <= MAXIMUM_REPORT_BYTES and os.write(1, raw) == len(raw))
    return 0 if result["status"] == "observed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
