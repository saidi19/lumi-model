#!/usr/bin/env python3
"""Bounded read-only metadata census of exact SDK child shards.

Each immediate child subtree keeps the frozen 250k-node/depth/name/blocks caps;
the aggregate diagnostic is capped at one million nodes. No file payloads,
names, paths or exceptions enter the closed report. No mutation or setup probe
is available in this entrypoint.
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
SCHEMA = "lumi.sdk-sharded-census-diagnostic.v0.1"
SHARD_NODES = 250_000
AGGREGATE_NODES = 1_000_000
AGGREGATE_NAME_BYTES = 128 * 1024**2
AGGREGATE_ALLOCATED_BYTES = 64 * 1024**3
MAXIMUM_DEPTH = 48
MAXIMUM_NAME_BYTES = 32 * 1024**2
MAXIMUM_ALLOCATED_BYTES = 64 * 1024**3
MAXIMUM_SECONDS = 240
MAXIMUM_REPORT_BYTES = 8192
MAXIMUM_INTEGER = 2**63 - 1
CODES = ("operation_failed", "runtime_rejected", "binding_rejected", "target_rejected",
         "identity_changed", "tree_rejected", "deadline_exhausted", "already_attempted",
         "read_only_rejected", "budget_hit", "census_complete")
BUDGETS = ("shard_listing_entries", "shard_nodes", "shard_depth", "shard_relative_name_bytes",
           "shard_allocated_bytes", "aggregate_nodes", "aggregate_relative_name_bytes",
           "aggregate_allocated_bytes")
FALSE_FLAGS = ("sdk_mutation_performed", "sdk_payload_read", "setup_probe_run", "gpu_used",
               "model_or_dataset_access", "software_image_acquired", "native_qualification_passed",
               "training_authorized", "production_eligible", "old_attempt_retried",
               "historical_budget_cause_proven", "concurrent_writer_exclusion_verified")
_ATTEMPTED = False


class CensusError(ValueError):
    def __init__(self, code="operation_failed"):
        self.code = code if type(code) is str and code in CODES else "operation_failed"
        super().__init__(self.code)


def need(value, code="operation_failed"):
    if not value:
        raise CensusError(code) from None


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                       allow_nan=False).encode("ascii") + b"\n")


def runtime():
    need(sys.platform == "linux" and os.uname().machine == "x86_64"
         and sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 12)
         and sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode
         and os.getresuid() == os.getresgid() == (0, 0, 0)
         and os.environ.get("LUMI_SDK_SHARDED_CENSUS_CI") == "1"
         and signal.getsignal(signal.SIGALRM) == signal.SIG_DFL
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
             and not os.read(fd, 1), "binding_rejected")
        need(hashlib.sha256(raw).hexdigest() == HELPER_SHA256, "binding_rejected")
    finally:
        os.close(fd)
    module = types.ModuleType("_lumi_sdk_sharded_helpers")
    module.__file__ = str(path)
    exec(compile(bytes(raw), str(path), "exec"), module.__dict__)
    for key, value in (("MAXIMUM_NODES", SHARD_NODES), ("MAXIMUM_DEPTH", MAXIMUM_DEPTH),
                       ("MAXIMUM_NAME_BYTES", MAXIMUM_NAME_BYTES),
                       ("MAXIMUM_ALLOCATED_BYTES", MAXIMUM_ALLOCATED_BYTES),
                       ("MAXIMUM_CLEANUP_SECONDS", MAXIMUM_SECONDS)):
        need(getattr(module, key) == value, "binding_rejected")
    return module


def audit_guard(event, arguments):
    if event == "open":
        path, mode, flags = arguments
        need(type(flags) is int and not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT
             | os.O_TRUNC | os.O_APPEND), "read_only_rejected")
        need((flags & os.O_DIRECTORY and mode is None)
             or (path == "/proc/self/mountinfo" and mode in {"r", "rb"}),
             "read_only_rejected")
    if (event.startswith(("socket.", "subprocess.", "ctypes."))
            or event in {"os.remove", "os.rmdir", "os.mkdir", "os.rename", "os.link", "os.symlink",
                         "os.chmod", "os.chown", "os.fchmod", "os.fchown", "os.truncate",
                         "os.utime", "os.setxattr", "os.removexattr", "os.fork", "os.forkpty",
                         "os.exec", "os.posix_spawn", "os.spawn", "os.system", "shutil.rmtree"}):
        raise CensusError("read_only_rejected") from None


def install_guard():
    sys.addaudithook(audit_guard)


def expired(_signum, _frame):
    raise CensusError("deadline_exhausted") from None


def arm_deadline():
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(MAXIMUM_SECONDS)


def disarm_deadline():
    signal.alarm(0)
    signal.signal(signal.SIGALRM, signal.SIG_DFL)


def empty_report():
    return {"schema": SCHEMA, "status": "failed", "stage": "runtime", "code": "operation_failed",
            "budget_kind": None, "budget_limit": None, "attempted_value": None,
            "census_complete": False, "shards_started": 0, "shards_completed": 0,
            "visited_nodes": 0, "accepted_relative_name_bytes": 0,
            "accepted_allocated_bytes": 0, "maximum_directory_depth": 0,
            "active_shard_nodes": 0, "active_shard_name_bytes": 0,
            "active_shard_allocated_bytes": 0,
            "largest_shard_listing_entries": 0, **dict.fromkeys(FALSE_FLAGS, False)}


def budget(value, limit, kind, report):
    need(type(value) is int and 0 <= value <= MAXIMUM_INTEGER, "tree_rejected")
    if value > limit:
        report.update(budget_kind=kind, budget_limit=limit, attempted_value=value)
        raise CensusError("budget_hit") from None


def names(fd, count_limit, report):
    result = []
    with os.scandir(fd) as entries:
        for entry in entries:
            budget(len(result) + 1, count_limit, "shard_listing_entries", report)
            name = entry.name
            need(type(name) is str and name not in {"", ".", ".."} and "/" not in name,
                 "tree_rejected")
            result.append(name)
            report["largest_shard_listing_entries"] = max(
                report["largest_shard_listing_entries"], len(result),
            )
    return sorted(result)


def shard_census(parent_fd, name, helper, report):
    """Count the immediate child itself and descendants, without opening leaf payloads."""
    device = os.fstat(parent_fd).st_dev
    report.update(active_shard_nodes=0, active_shard_name_bytes=0, active_shard_allocated_bytes=0)

    def visit(current, leaf, path):
        need(type(leaf) is str and leaf not in {"", ".", ".."} and "/" not in leaf,
             "tree_rejected")
        budget(report["active_shard_nodes"] + 1, SHARD_NODES, "shard_nodes", report)
        budget(report["visited_nodes"] + 1, AGGREGATE_NODES, "aggregate_nodes", report)
        name_bytes = sum(len(os.fsencode(part)) + 1 for part in path)
        budget(report["active_shard_name_bytes"] + name_bytes, MAXIMUM_NAME_BYTES,
               "shard_relative_name_bytes", report)
        budget(report["accepted_relative_name_bytes"] + name_bytes, AGGREGATE_NAME_BYTES,
               "aggregate_relative_name_bytes", report)
        info = os.stat(leaf, dir_fd=current, follow_symlinks=False)
        need(info.st_dev == device, "tree_rejected")
        kind = stat.S_IFMT(info.st_mode)
        need(kind in (stat.S_IFDIR, stat.S_IFREG, stat.S_IFLNK)
             and (kind != stat.S_IFREG or info.st_nlink == 1), "tree_rejected")
        allocated = info.st_blocks * 512
        need(type(allocated) is int and allocated >= 0, "tree_rejected")
        budget(report["active_shard_allocated_bytes"] + allocated, MAXIMUM_ALLOCATED_BYTES,
               "shard_allocated_bytes", report)
        budget(report["accepted_allocated_bytes"] + allocated, AGGREGATE_ALLOCATED_BYTES,
               "aggregate_allocated_bytes", report)
        report["active_shard_nodes"] += 1
        report["active_shard_name_bytes"] += name_bytes
        report["active_shard_allocated_bytes"] += allocated
        report["visited_nodes"] += 1
        report["accepted_relative_name_bytes"] += name_bytes
        report["accepted_allocated_bytes"] += allocated
        if kind == stat.S_IFDIR:
            budget(len(path), MAXIMUM_DEPTH - 1, "shard_depth", report)
            report["maximum_directory_depth"] = max(report["maximum_directory_depth"], len(path))
            child = os.open(leaf, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                            dir_fd=current)
            try:
                need(helper.identity(os.fstat(child)) == helper.identity(info), "identity_changed")
                selected = names(child, SHARD_NODES - report["active_shard_nodes"], report)
                for descendant in selected:
                    visit(child, descendant, (*path, descendant))
                # Rechecking already-counted names must not consume the remaining node allowance.
                need(selected == names(child, SHARD_NODES, report), "identity_changed")
            finally:
                os.close(child)
        need(helper.identity(os.stat(leaf, dir_fd=current, follow_symlinks=False))
             == helper.identity(info), "identity_changed")

    visit(parent_fd, name, (name,))


def run():
    global _ATTEMPTED
    result, handles, armed, helper = empty_report(), [], False, None
    try:
        runtime()
        need(not _ATTEMPTED, "already_attempted")
        _ATTEMPTED = True
        result["stage"] = "binding"
        helper = load_helper()
        install_guard()
        armed = True
        arm_deadline()
        result["stage"] = "target"
        handles, bound = helper.open_target()
        root_fd = handles[-1]
        result["stage"] = "census"
        root_names = names(root_fd, SHARD_NODES, result)
        for ordinal, name in enumerate(root_names, 1):
            result["shards_started"] = ordinal
            shard_census(root_fd, name, helper, result)
            result["shards_completed"] = ordinal
        need(root_names == names(root_fd, SHARD_NODES, result), "identity_changed")
        need(helper.identity(os.stat(helper.PARTS[-1], dir_fd=handles[-2],
                                     follow_symlinks=False)) == bound, "identity_changed")
        helper.check_mounts()
        result.update(status="observed", stage="complete", code="census_complete",
                      census_complete=True)
    except BaseException as error:
        if type(error) is CensusError:
            result["code"] = error.code
        elif helper is not None and type(error) is helper.ReclaimError:
            result["code"] = error.code if error.code in CODES else "operation_failed"
        if result["code"] == "budget_hit":
            result["status"] = "observed"
    finally:
        for fd in reversed(handles):
            try:
                os.close(fd)
            except OSError:
                result.update(status="failed", code="operation_failed", census_complete=False)
        if armed:
            disarm_deadline()
    if result["status"] == "failed":
        result["census_complete"] = False
        result.update(budget_kind=None, budget_limit=None, attempted_value=None)
    validate_report(result)
    return result


def validate_report(value):
    need(type(value) is dict and set(value) == set(empty_report()))
    need(type(value["schema"]) is str and value["schema"] == SCHEMA
         and type(value["status"]) is str and value["status"] in {"observed", "failed"}
         and type(value["stage"]) is str
         and value["stage"] in {"runtime", "binding", "target", "census", "complete"}
         and type(value["code"]) is str and value["code"] in CODES)
    for key in ("census_complete", *FALSE_FLAGS):
        need(type(value[key]) is bool and (key not in FALSE_FLAGS or value[key] is False))
    for key, maximum in (("shards_started", SHARD_NODES), ("shards_completed", SHARD_NODES),
                          ("visited_nodes", AGGREGATE_NODES),
                          ("accepted_relative_name_bytes", AGGREGATE_NAME_BYTES),
                          ("accepted_allocated_bytes", AGGREGATE_ALLOCATED_BYTES),
                          ("active_shard_nodes", SHARD_NODES),
                          ("active_shard_name_bytes", MAXIMUM_NAME_BYTES),
                          ("active_shard_allocated_bytes", MAXIMUM_ALLOCATED_BYTES),
                          ("maximum_directory_depth", MAXIMUM_DEPTH - 1),
                          ("largest_shard_listing_entries", SHARD_NODES)):
        need(type(value[key]) is int and 0 <= value[key] <= maximum)
    need(0 <= value["shards_started"] - value["shards_completed"] <= 1)
    need(value["active_shard_nodes"] <= value["visited_nodes"]
         and value["active_shard_name_bytes"] <= value["accepted_relative_name_bytes"]
         and value["active_shard_allocated_bytes"] <= value["accepted_allocated_bytes"])
    need(value["shards_completed"] <= value["visited_nodes"])
    if value["code"] == "budget_hit":
        need(value["status"] == "observed" and value["stage"] == "census"
             and not value["census_complete"] and type(value["budget_kind"]) is str
             and value["budget_kind"] in BUDGETS
             and type(value["budget_limit"]) is type(value["attempted_value"]) is int
             and 0 <= value["budget_limit"] < value["attempted_value"] <= MAXIMUM_INTEGER)
        expected = {"shard_nodes": SHARD_NODES, "shard_depth": MAXIMUM_DEPTH - 1,
                    "shard_relative_name_bytes": MAXIMUM_NAME_BYTES,
                    "shard_allocated_bytes": MAXIMUM_ALLOCATED_BYTES,
                    "aggregate_nodes": AGGREGATE_NODES,
                    "aggregate_relative_name_bytes": AGGREGATE_NAME_BYTES,
                    "aggregate_allocated_bytes": AGGREGATE_ALLOCATED_BYTES}
        kind, limit, attempted = (value[k] for k in (
            "budget_kind", "budget_limit", "attempted_value"))
        if kind == "shard_listing_entries":
            need(limit in (SHARD_NODES, SHARD_NODES - value["active_shard_nodes"])
                 and attempted == limit + 1)
        else:
            need(limit == expected[kind])
            if kind == "shard_nodes":
                need(value["active_shard_nodes"] == limit and attempted == limit + 1)
            if kind == "aggregate_nodes":
                need(value["visited_nodes"] == limit and attempted == limit + 1)
            if kind == "shard_depth":
                need(attempted == MAXIMUM_DEPTH)
    else:
        need(all(value[key] is None for key in ("budget_kind", "budget_limit", "attempted_value")))
        if value["status"] == "observed":
            need(value["stage"] == "complete" and value["code"] == "census_complete"
                 and value["census_complete"]
                 and value["shards_completed"] == value["shards_started"])
        else:
            need(not value["census_complete"] and value["code"] != "census_complete")
    need(len(canonical(value)) <= MAXIMUM_REPORT_BYTES)
    return value


def main():
    result = run() if sys.argv[1:] == ["--read-only-sharded-sdk-census"] else empty_report()
    if sys.argv[1:] != ["--read-only-sharded-sdk-census"]:
        result["code"] = "runtime_rejected"
    raw = canonical(result)
    need(len(raw) <= MAXIMUM_REPORT_BYTES and os.write(1, raw) == len(raw))
    return 0 if result["status"] == "observed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
