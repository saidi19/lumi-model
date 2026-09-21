#!/usr/bin/env python3
"""One exact-target read-only metadata census; no deletion or setup probe.

Retain only fixed codes and partial aggregate counters. The frozen predecessor
is authenticated and loaded as inert definitions for target-opening helpers;
its census, removal, run, probe and main functions are never invoked. A trusted
fresh disposable GitHub VM without concurrent writers remains an explicit premise.
The census preserves predecessor traversal order and all existing resource caps.
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
SCHEMA = "lumi.sdk-census-diagnostic.v0.1"
MAXIMUM_NODES = 250_000
MAXIMUM_DEPTH = 48
MAXIMUM_NAME_BYTES = 32 * 1024**2
MAXIMUM_ALLOCATED_BYTES = 64 * 1024**3
MAXIMUM_SECONDS = 240
MAXIMUM_REPORT_BYTES = 8192
MAXIMUM_INTEGER = 2**63 - 1
STAGES = ("runtime", "binding", "target", "census", "complete")
CODES = ("operation_failed", "runtime_rejected", "binding_rejected", "target_rejected",
         "identity_changed", "tree_rejected", "deadline_exhausted", "already_attempted",
         "read_only_rejected", "budget_hit", "census_complete")
BUDGETS = ("listing_entries", "nodes", "directory_depth", "relative_name_bytes",
           "allocated_bytes")
COUNTERS = {"visited_nodes": MAXIMUM_NODES, "accepted_relative_name_bytes": MAXIMUM_NAME_BYTES,
            "accepted_allocated_bytes": MAXIMUM_ALLOCATED_BYTES,
            "maximum_directory_depth": MAXIMUM_DEPTH - 1,
            "largest_listing_entries": MAXIMUM_NODES}
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
         and os.environ.get("LUMI_SDK_CENSUS_DIAGNOSTIC_CI") == "1"
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
        fields = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink", "st_size",
                  "st_mtime_ns", "st_ctime_ns")
        after, named = os.fstat(fd), path.lstat()
        need(all(getattr(before, f) == getattr(after, f) == getattr(named, f) for f in fields)
             and not os.read(fd, 1), "binding_rejected")
        need(hashlib.sha256(raw).hexdigest() == HELPER_SHA256, "binding_rejected")
    finally:
        os.close(fd)
    module = types.ModuleType("_lumi_sdk_metadata_target_helpers")
    module.__file__ = str(path)
    exec(compile(bytes(raw), str(path), "exec"), module.__dict__)
    for key in ("MAXIMUM_NODES", "MAXIMUM_DEPTH", "MAXIMUM_NAME_BYTES", "MAXIMUM_ALLOCATED_BYTES"):
        need(getattr(module, key) == globals()[key], "binding_rejected")
    need(module.MAXIMUM_CLEANUP_SECONDS == MAXIMUM_SECONDS, "binding_rejected")
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
                         "os.chmod", "os.chown", "os.truncate", "os.utime", "os.setxattr",
                         "os.removexattr", "os.fork", "os.forkpty", "os.exec", "os.posix_spawn",
                         "os.spawn", "os.system", "shutil.rmtree"}):
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


def budget(value, limit, kind, report):
    need(type(value) is int and 0 <= value <= MAXIMUM_INTEGER, "tree_rejected")
    if value > limit:
        report.update(budget_kind=kind, budget_limit=limit, attempted_value=value)
        raise CensusError("budget_hit") from None


def names(fd, count_limit, report):
    result = []
    with os.scandir(fd) as entries:
        for entry in entries:
            budget(len(result) + 1, count_limit, "listing_entries", report)
            name = entry.name
            need(type(name) is str and name not in {"", ".", ".."} and "/" not in name,
                 "tree_rejected")
            result.append(name)
            report["largest_listing_entries"] = max(report["largest_listing_entries"], len(result))
    return sorted(result)


def census(root_fd, helper, report):
    device = os.fstat(root_fd).st_dev

    def walk(fd, relative):
        budget(len(relative), MAXIMUM_DEPTH - 1, "directory_depth", report)
        report["maximum_directory_depth"] = max(report["maximum_directory_depth"], len(relative))
        selected = names(fd, MAXIMUM_NODES - report["visited_nodes"], report)
        for name in selected:
            budget(report["visited_nodes"] + 1, MAXIMUM_NODES, "nodes", report)
            path = (*relative, name)
            total = report["accepted_relative_name_bytes"] + sum(
                len(os.fsencode(part)) + 1 for part in path)
            budget(total, MAXIMUM_NAME_BYTES, "relative_name_bytes", report)
            report["accepted_relative_name_bytes"] = total
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            need(info.st_dev == device, "tree_rejected")
            kind = stat.S_IFMT(info.st_mode)
            need(kind in (stat.S_IFDIR, stat.S_IFREG, stat.S_IFLNK)
                 and (kind != stat.S_IFREG or info.st_nlink == 1), "tree_rejected")
            total = report["accepted_allocated_bytes"] + info.st_blocks * 512
            budget(total, MAXIMUM_ALLOCATED_BYTES, "allocated_bytes", report)
            report["accepted_allocated_bytes"] = total
            report["visited_nodes"] += 1
            if kind == stat.S_IFDIR:
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                dir_fd=fd)
                try:
                    need(helper.identity(os.fstat(child)) == helper.identity(info),
                         "identity_changed")
                    walk(child, path)
                finally:
                    os.close(child)
        need(selected == names(fd, MAXIMUM_NODES, report), "identity_changed")

    walk(root_fd, ())


def empty_report():
    return {"schema": SCHEMA, "status": "failed", "stage": "runtime", "code": "operation_failed",
            "budget_kind": None, "budget_limit": None, "attempted_value": None,
            "census_complete": False, **dict.fromkeys(COUNTERS, 0),
            **dict.fromkeys(FALSE_FLAGS, False)}


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
        result["stage"] = "census"
        census(handles[-1], helper, result)
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
            result["status"] = "observed"  # Diagnostic success, never a complete census claim.
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
    need(value["schema"] == SCHEMA and type(value["schema"]) is str)
    need(type(value["status"]) is str and value["status"] in {"observed", "failed"}
         and type(value["stage"]) is str and value["stage"] in STAGES
         and type(value["code"]) is str and value["code"] in CODES)
    need(type(value["census_complete"]) is bool)
    for key, maximum in COUNTERS.items():
        need(type(value[key]) is int and 0 <= value[key] <= maximum)
    need(all(value[key] is False for key in FALSE_FLAGS))
    if value["code"] == "budget_hit":
        kind, limit, attempted = (
            value[k] for k in ("budget_kind", "budget_limit", "attempted_value")
        )
        need(value["status"] == "observed" and value["stage"] == "census"
             and not value["census_complete"] and type(kind) is str and kind in BUDGETS
             and type(limit) is type(attempted) is int
             and 0 <= limit < attempted <= MAXIMUM_INTEGER)
        expected = {"nodes": MAXIMUM_NODES, "directory_depth": MAXIMUM_DEPTH - 1,
                    "relative_name_bytes": MAXIMUM_NAME_BYTES,
                    "allocated_bytes": MAXIMUM_ALLOCATED_BYTES}
        if kind == "listing_entries":
            need(limit in (MAXIMUM_NODES, MAXIMUM_NODES - value["visited_nodes"])
                 and attempted == limit + 1)
        else:
            need(limit == expected[kind])
    else:
        need(all(value[k] is None for k in ("budget_kind", "budget_limit", "attempted_value")))
        if value["status"] == "observed":
            need(value["stage"] == "complete" and value["code"] == "census_complete"
                 and value["census_complete"])
        else:
            need(not value["census_complete"] and value["code"] != "census_complete")
    need(len(canonical(value)) <= MAXIMUM_REPORT_BYTES)
    return value


def main():
    if sys.argv[1:] == ["--read-only-sdk-census"]:
        result = run()
    else:
        result = empty_report()
        result["code"] = "runtime_rejected"
    raw = canonical(result)
    need(len(raw) <= MAXIMUM_REPORT_BYTES and os.write(1, raw) == len(raw))
    return 0 if result["status"] == "observed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
