#!/usr/bin/env python3
"""Audit all of a fixed ten-child SDK prefix, remove only it, then check setup.

Only the fresh disposable Ubuntu VM entrypoint may operate on the real SDK.
Selected trees are completely censused and reverified before the first mutation.
No payloads, names, paths, or exception text are retained. The SDK root and all
unselected children remain. A trusted VM without concurrent writers is a premise,
not a claimed race-proof filesystem transaction. No acquisition or qualification.
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
HELPER = "scripts/reclaim_sft_android_sdk_v1.py"
HELPER_SHA256 = "724cb2d2a6ff01dff790d77bf88434f2be15b0f43e2def32b9def25af5c74ba5"
SCHEMA = "lumi.sdk-prefix-capacity.v0.1"
MAXIMUM_SHARDS = 10
SHARD_NODES = 250_000
AGGREGATE_NODES = 1_000_000
MAXIMUM_DEPTH = 48
MAXIMUM_NAME_BYTES = 32 * 1024**2
AGGREGATE_NAME_BYTES = 128 * 1024**2
MAXIMUM_ALLOCATED_BYTES = 64 * 1024**3
AGGREGATE_ALLOCATED_BYTES = 64 * 1024**3
MAXIMUM_SECONDS = 240
MAXIMUM_REPORT_BYTES = 16384
STAGES = ("runtime", "binding", "target", "census", "verification", "removal", "capacity",
          "setup_probe", "complete")
BUDGETS = {"shard_nodes": SHARD_NODES, "aggregate_nodes": AGGREGATE_NODES,
           "shard_depth": MAXIMUM_DEPTH - 1, "shard_name_bytes": MAXIMUM_NAME_BYTES,
           "aggregate_name_bytes": AGGREGATE_NAME_BYTES,
           "shard_allocated_bytes": MAXIMUM_ALLOCATED_BYTES,
           "aggregate_allocated_bytes": AGGREGATE_ALLOCATED_BYTES,
           "listing_entries": SHARD_NODES}
CODES = ("ok", "capacity_insufficient", "runtime_rejected", "binding_rejected",
         "target_rejected", "tree_rejected", "identity_changed", "deadline_exhausted",
         "already_attempted", "removal_unconfirmed", "setup_failed", "operation_failed",
         "budget_hit")
FALSE_FLAGS = ("sdk_payload_read", "sdk_root_removed", "gpu_used", "model_or_dataset_access",
               "software_image_acquired", "native_qualification_passed", "training_authorized",
               "production_eligible", "historical_failure_cause_proven", "old_attempt_retried",
               "concurrent_writer_exclusion_verified")
_ATTEMPTED = False


class PrefixError(ValueError):
    def __init__(self, code="operation_failed"):
        self.code = code if type(code) is str and code in CODES else "operation_failed"
        super().__init__(self.code)


def need(value, code="operation_failed"):
    if not value:
        raise PrefixError(code) from None


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                       allow_nan=False).encode("ascii") + b"\n")


def runtime():
    need(sys.platform == "linux" and os.uname().machine == "x86_64"
         and sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 12)
         and sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode
         and os.getresuid() == os.getresgid() == (0, 0, 0)
         and all(os.environ.get(name) == "1" for name in (
             "LUMI_SDK_PREFIX_CAPACITY_CI", "LUMI_PREPARED_SETUP_PREFLIGHT_CI",
             "LUMI_CHECKPOINT_GUARD_QUALIFICATION_CI"))
         and signal.getsignal(signal.SIGALRM) == signal.SIG_DFL
         and signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL
         and signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
         and threading.current_thread() is threading.main_thread()
         and threading.active_count() == 1, "runtime_rejected")


def load_helper():
    path = ROOT / HELPER
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
        after, named = os.fstat(fd), path.lstat()
        fields = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink", "st_size",
                  "st_mtime_ns", "st_ctime_ns")
        need(all(getattr(before, f) == getattr(after, f) == getattr(named, f) for f in fields)
             and not os.read(fd, 1) and hashlib.sha256(raw).hexdigest() == HELPER_SHA256,
             "binding_rejected")
    finally:
        os.close(fd)
    module = types.ModuleType("_lumi_sdk_prefix_helper")
    module.__file__ = str(path)
    exec(compile(bytes(raw), str(path), "exec"), module.__dict__)
    return module


def expired(_signum, _frame):
    raise PrefixError("deadline_exhausted") from None


def arm_deadline():
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(MAXIMUM_SECONDS)


def disarm_deadline():
    signal.alarm(0)
    signal.signal(signal.SIGALRM, signal.SIG_DFL)


def empty_report():
    return {"schema": SCHEMA, "status": "failed", "stage": "runtime", "code": "operation_failed",
            "budget_kind": None, "budget_limit": None, "attempted_value": None,
            "selected_shards": 0, "audited_shards": 0, "deleted_shards": 0,
            "census_nodes": 0, "census_name_bytes": 0, "census_allocated_bytes": 0,
            "removed_leaves": 0, "removed_directories": 0, "audit_complete": False,
            "removal_started": False, "sdk_root_preserved": False,
            "selected_absence_verified": False, "unselected_root_names_unchanged": False,
            "capacity_before": None, "capacity_after": None, "setup_report": None,
            "ready_for_unchanged_setup": False, **dict.fromkeys(FALSE_FLAGS, False)}


def budget(value, limit, kind, report):
    need(type(value) is int and 0 <= value < 2**63, "tree_rejected")
    if value > limit:
        report.update(budget_kind=kind, budget_limit=limit, attempted_value=value)
        raise PrefixError("budget_hit") from None


def names(fd, report):
    result = []
    with os.scandir(fd) as entries:
        for entry in entries:
            budget(len(result) + 1, SHARD_NODES, "listing_entries", report)
            name = entry.name
            need(type(name) is str and name not in {"", ".", ".."} and "/" not in name,
                 "tree_rejected")
            result.append(name)
    return sorted(result)


def census_shard(parent_fd, name, helper, report):
    """Return root state plus frozen-remove_tree-compatible descendant maps."""
    device = os.fstat(parent_fd).st_dev
    nodes, children, totals = {}, {}, [0, 0, 0]

    def visit(fd, leaf, relative):
        budget(totals[0] + 1, SHARD_NODES, "shard_nodes", report)
        budget(report["census_nodes"] + 1, AGGREGATE_NODES, "aggregate_nodes", report)
        length = sum(len(os.fsencode(part)) + 1 for part in (name, *relative))
        budget(totals[1] + length, MAXIMUM_NAME_BYTES, "shard_name_bytes", report)
        budget(report["census_name_bytes"] + length, AGGREGATE_NAME_BYTES,
               "aggregate_name_bytes", report)
        info = os.stat(leaf, dir_fd=fd, follow_symlinks=False)
        kind = stat.S_IFMT(info.st_mode)
        need(info.st_dev == device and kind in (stat.S_IFDIR, stat.S_IFREG, stat.S_IFLNK)
             and (kind != stat.S_IFREG or info.st_nlink == 1), "tree_rejected")
        allocated = info.st_blocks * 512
        need(type(allocated) is int and allocated >= 0, "tree_rejected")
        budget(totals[2] + allocated, MAXIMUM_ALLOCATED_BYTES, "shard_allocated_bytes", report)
        budget(report["census_allocated_bytes"] + allocated, AGGREGATE_ALLOCATED_BYTES,
               "aggregate_allocated_bytes", report)
        totals[0] += 1
        totals[1] += length
        totals[2] += allocated
        report["census_nodes"] += 1
        report["census_name_bytes"] += length
        report["census_allocated_bytes"] += allocated
        expected = helper.file_state(info)
        if relative:
            nodes[relative] = expected
        if kind == stat.S_IFDIR:
            budget(len(relative) + 1, MAXIMUM_DEPTH - 1, "shard_depth", report)
            child = os.open(leaf, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                            dir_fd=fd)
            try:
                need(helper.file_state(os.fstat(child)) == expected, "identity_changed")
                children[relative] = names(child, report)
                for descendant in children[relative]:
                    visit(child, descendant, (*relative, descendant))
                # Re-listing already counted nodes uses the full cap, not the remainder.
                need(children[relative] == names(child, report), "identity_changed")
            finally:
                os.close(child)
        need(helper.file_state(os.stat(leaf, dir_fd=fd, follow_symlinks=False)) == expected,
             "identity_changed")
        return expected

    root_state = visit(parent_fd, name, ())
    return name, root_state, nodes, children, totals[2]


def verify_shard(parent_fd, shard, helper):
    """Full metadata recheck of selected trees only, still before any mutation."""
    name, root_state, nodes, children, _allocated = shard

    def visit(fd, leaf, relative, expected):
        info = os.stat(leaf, dir_fd=fd, follow_symlinks=False)
        need(helper.file_state(info) == expected, "identity_changed")
        if stat.S_ISDIR(info.st_mode):
            child = os.open(leaf, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                            dir_fd=fd)
            try:
                need(helper.file_state(os.fstat(child)) == expected, "identity_changed")
                need(helper.names(child, SHARD_NODES) == children[relative], "identity_changed")
                for descendant in children[relative]:
                    path = (*relative, descendant)
                    visit(child, descendant, path, nodes[path])
                need(helper.names(child, SHARD_NODES) == children[relative], "identity_changed")
            finally:
                os.close(child)
        need(helper.file_state(os.stat(leaf, dir_fd=fd, follow_symlinks=False)) == expected,
             "identity_changed")

    visit(parent_fd, name, (), root_state)


def remove_shard(parent_fd, shard, helper, report):
    name, expected, nodes, children, allocated = shard
    info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    need(helper.file_state(info) == expected, "identity_changed")
    if stat.S_ISDIR(info.st_mode):
        child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                        dir_fd=parent_fd)
        try:
            need(helper.file_state(os.fstat(child)) == expected, "identity_changed")
            helper.remove_tree(child, (nodes, children, allocated), report)
        finally:
            os.close(child)
        # Our own child deletions change parent nlink/mtime, not its bound identity.
        need(helper.identity(os.stat(name, dir_fd=parent_fd, follow_symlinks=False))
             == expected[:5], "identity_changed")
        os.rmdir(name, dir_fd=parent_fd)
        report["removed_directories"] += 1
    else:
        os.unlink(name, dir_fd=parent_fd)
        report["removed_leaves"] += 1
    report["deleted_shards"] += 1


def run():
    global _ATTEMPTED
    report, handles, armed, helper, probe = empty_report(), [], False, None, None
    try:
        runtime()
        need(not _ATTEMPTED, "already_attempted")
        _ATTEMPTED = True
        report["stage"] = "binding"
        helper = load_helper()
        probe = helper.load_probe()
        probe.runtime()
        probe.load_harness().runtime()
        probe.install_guard()
        armed = True
        arm_deadline()
        report["stage"] = "target"
        handles, bound = helper.open_target()
        root_fd = handles[-1]
        report["capacity_before"] = probe.capacity(os.statvfs("/tmp"))
        report["stage"] = "census"
        original = names(root_fd, report)
        selected = original[:MAXIMUM_SHARDS]
        need(selected, "tree_rejected")
        report["selected_shards"] = len(selected)
        shards = []
        for name in selected:
            shards.append(census_shard(root_fd, name, helper, report))
            report["audited_shards"] += 1
        report["stage"] = "verification"
        for shard in shards:
            verify_shard(root_fd, shard, helper)
        need(original == names(root_fd, report), "identity_changed")
        need(helper.identity(os.stat(helper.PARTS[-1], dir_fd=handles[-2],
                                     follow_symlinks=False)) == bound, "identity_changed")
        helper.check_mounts()
        report["audit_complete"] = True
        report["stage"] = "removal"
        report["removal_started"] = True  # Any interruption may leave partial deletion.
        for shard in shards:
            remove_shard(root_fd, shard, helper, report)
        need(names(root_fd, report) == original[MAXIMUM_SHARDS:], "removal_unconfirmed")
        need(helper.identity(os.fstat(root_fd)) == bound
             and helper.identity(os.stat(helper.PARTS[-1], dir_fd=handles[-2],
                                         follow_symlinks=False)) == bound, "identity_changed")
        report.update(sdk_root_preserved=True, selected_absence_verified=True,
                      unselected_root_names_unchanged=True)
        report["stage"] = "capacity"
        while handles:
            os.close(handles.pop())
        report["capacity_after"] = probe.capacity(os.statvfs("/tmp"))
        disarm_deadline()
        armed = False
        if report["capacity_after"]["acquisition_headroom_met"]:
            report["stage"] = "setup_probe"
            setup = probe.run()
            probe.validate_report(setup)  # Never retain unvalidated nested content.
            report["setup_report"] = setup
            need(setup["status"] == "observed", "setup_failed")
            report["ready_for_unchanged_setup"] = setup["ready_for_unchanged_setup"]
        report["status"], report["stage"] = "observed", "complete"
        report["code"] = "ok" if report["ready_for_unchanged_setup"] else "capacity_insufficient"
    except BaseException as error:
        if type(error) is PrefixError:
            report["code"] = error.code
        elif helper is not None and type(error) is helper.ReclaimError:
            report["code"] = error.code if error.code in CODES else "operation_failed"
    finally:
        while handles:
            try:
                os.close(handles.pop())
            except OSError:
                report["status"], report["code"] = "failed", "operation_failed"
        if armed:
            disarm_deadline()
    if report["status"] == "failed":
        report["ready_for_unchanged_setup"] = False
    if report["code"] != "budget_hit":
        report.update(budget_kind=None, budget_limit=None, attempted_value=None)
    validate_report(report, probe)
    return report


def validate_report(value, probe=None):
    need(type(value) is dict and set(value) == set(empty_report()))
    need(type(value["schema"]) is str and value["schema"] == SCHEMA)
    need(type(value["status"]) is str and value["status"] in {"observed", "failed"}
         and type(value["stage"]) is str and value["stage"] in STAGES
         and type(value["code"]) is str and value["code"] in CODES)
    for key in FALSE_FLAGS:
        need(value[key] is False)
    for key in ("audit_complete", "removal_started", "sdk_root_preserved",
                "selected_absence_verified", "unselected_root_names_unchanged",
                "ready_for_unchanged_setup"):
        need(type(value[key]) is bool)
    for key, maximum in (("selected_shards", MAXIMUM_SHARDS), ("audited_shards", MAXIMUM_SHARDS),
                          ("deleted_shards", MAXIMUM_SHARDS), ("census_nodes", AGGREGATE_NODES),
                          ("census_name_bytes", AGGREGATE_NAME_BYTES),
                          ("census_allocated_bytes", AGGREGATE_ALLOCATED_BYTES),
                          ("removed_leaves", AGGREGATE_NODES),
                          ("removed_directories", AGGREGATE_NODES)):
        need(type(value[key]) is int and 0 <= value[key] <= maximum)
    need(value["deleted_shards"] <= value["audited_shards"] <= value["selected_shards"])
    removed = value["removed_leaves"] + value["removed_directories"]
    need(removed <= value["census_nodes"] and removed >= value["deleted_shards"])
    need(not (removed or value["deleted_shards"]) or value["removal_started"])
    need(not value["audit_complete"] or (value["audited_shards"] == value["selected_shards"] > 0
                                        and value["census_nodes"] >= value["selected_shards"]))
    need(not value["removal_started"] or value["audit_complete"])
    complete = (value["deleted_shards"] == value["selected_shards"] > 0
                and removed == value["census_nodes"] and value["removal_started"])
    for key in ("sdk_root_preserved", "selected_absence_verified",
                "unselected_root_names_unchanged"):
        need(not value[key] or complete)
    if value["code"] == "budget_hit":
        kind = value["budget_kind"]
        need(type(kind) is str and kind in BUDGETS
             and type(value["budget_limit"]) is int and value["budget_limit"] == BUDGETS[kind]
             and type(value["attempted_value"]) is int
             and value["budget_limit"] < value["attempted_value"] < 2**63)
    else:
        need(all(value[key] is None for key in ("budget_kind", "budget_limit", "attempted_value")))
    for key in ("capacity_before", "capacity_after"):
        part = value[key]
        if part is not None:
            need(probe is not None and type(part) is dict
                 and set(part) == set(probe.DISK_INTS) | set(probe.DISK_BOOLS))
            need(all(type(part[k]) is int and 0 <= part[k] < 2**63 for k in probe.DISK_INTS)
                 and all(type(part[k]) is bool for k in probe.DISK_BOOLS))
            fragment = part["filesystem_fragment_bytes"]
            need(fragment > 0 and part["available_disk_bytes"] % fragment == 0)
            need(part == probe.capacity(types.SimpleNamespace(
                f_bavail=part["available_disk_bytes"] // fragment, f_frsize=fragment)))
    if value["capacity_after"] is not None:
        need(complete and value["capacity_before"] is not None)
    if value["setup_report"] is not None:
        need(probe is not None and value["capacity_after"] is not None
             and value["capacity_after"]["acquisition_headroom_met"])
        probe.validate_report(value["setup_report"])
    if value["status"] == "observed":
        need(value["stage"] == "complete" and complete and value["sdk_root_preserved"]
             and value["selected_absence_verified"] and value["unselected_root_names_unchanged"]
             and value["capacity_after"] is not None)
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
        need(not value["ready_for_unchanged_setup"]
             and value["code"] not in {"ok", "capacity_insufficient"})
    need(len(canonical(value)) <= MAXIMUM_REPORT_BYTES)
    return value


def main():
    if len(sys.argv) != 2 or sys.argv[1] != "--fresh-sdk-prefix-capacity-check":
        result = empty_report()
        result["code"] = "runtime_rejected"
    else:
        result = run()
    raw = canonical(result)
    need(len(raw) <= MAXIMUM_REPORT_BYTES and os.write(1, raw) == len(raw))
    return 0 if result["status"] == "observed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
