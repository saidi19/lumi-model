#!/usr/bin/env python3
"""One small Linux setup observation, NOT a native qualification or retry.

Never execute the probe on the Mac. It authenticates existing frozen source,
loads definitions, sets this process's subreaper flag without spawning children,
and creates/removes two fresh empty directories. It measures available capacity
before any acquisition. No Docker, archive, model, dataset or pipeline is opened.
Only fixed stage codes, booleans and aggregate byte/count values are emitted.

The checkout/host are trusted premises, not authenticated host admission. Disk
headroom is a current-host observation; metadata overhead and concurrent writes
are not reserved. It cannot establish the old disposable host's failure cause.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import stat
import sys
import tempfile
import threading
import types
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
SCHEMA = "lumi.prepared-setup-preflight.v0.1"
HARNESS = "scripts/run_checkpoint_prepared_qualification_v2.py"
HARNESS_SHA256 = "253bb5d0b113949baecb5ea229590c1f1e2091d9f2e659abf6fbbbe9c3e1f107"
TAR_BYTES = 1_062_113_280
ZIP_BYTES = 1_062_113_436
MINIMUM_FREE_DISK_BYTES = 16 * 1024**3
MAXIMUM_SECONDS = 60
MAXIMUM_REPORT_BYTES = 8192
STAGES = ("runtime", "harness_binding", "source_closure", "base_module", "subreaper",
          "admission_module", "rootfs_binding", "temp_parent", "temp_config", "disk_query",
          "complete")
CODES = frozenset({"ok", "disk_floor_insufficient", "acquisition_headroom_insufficient",
                   "runtime_rejected", "binding_rejected", "step_failed", "already_attempted",
                   "deadline_exhausted", "cleanup_unconfirmed"})
NONCLAIMS = ("native_qualification_passed", "historical_failure_cause_proven",
             "real_model_or_data_access", "software_image_acquired", "gpu_used",
             "training_authorized", "production_eligible", "old_attempt_retried",
             "host_admission_verified", "disk_capacity_reserved")
DISK_INTS = ("available_disk_bytes", "filesystem_fragment_bytes", "retained_tar_allocation_bytes",
             "acquisition_peak_allocation_bytes", "minimum_preacquisition_free_disk_bytes")
DISK_BOOLS = ("disk_floor_met", "acquisition_headroom_met")
_ATTEMPTED = False


class PreflightError(ValueError):
    def __init__(self, code="step_failed"):
        self.code = code if type(code) is str and code in CODES else "step_failed"
        super().__init__(self.code)


def need(value, code="step_failed"):
    if not value:
        raise PreflightError(code) from None


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                       allow_nan=False).encode("ascii") + b"\n")


def runtime():
    need(sys.platform == "linux" and os.uname().machine == "x86_64"
         and sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 12)
         and sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode
         and os.getresuid() == os.getresgid() == (0, 0, 0)
         and signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL
         and signal.getsignal(signal.SIGALRM) == signal.SIG_DFL
         and signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
         and os.environ.get("LUMI_PREPARED_SETUP_PREFLIGHT_CI") == "1"
         and threading.current_thread() is threading.main_thread()
         and threading.active_count() == 1, "runtime_rejected")


def expired(_signum, _frame):
    raise PreflightError("deadline_exhausted") from None


def arm_deadline():
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(MAXIMUM_SECONDS)


def disarm_deadline():
    signal.alarm(0)
    signal.signal(signal.SIGALRM, signal.SIG_DFL)


def audit_guard(event, arguments):
    if (event.startswith("socket.") or event.startswith("subprocess.")
            or event in {"os.fork", "os.forkpty", "os.exec", "os.posix_spawn", "os.spawn",
                         "os.system"}
            or (event == "ctypes.dlopen" and arguments and arguments[0] is not None)):
        raise PreflightError("step_failed") from None


def install_guard():
    # A process-lifetime safety rail; tests exercise the callback without installing it.
    sys.addaudithook(audit_guard)


def identity(info):
    return (info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode),
            stat.S_IMODE(info.st_mode), info.st_uid, info.st_gid)


def load_harness():
    path = ROOT / HARNESS
    need(not any(item.is_symlink() for item in (path, *path.parents)), "binding_rejected")
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
        fields = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_uid", "st_gid", "st_size",
                  "st_mtime_ns", "st_ctime_ns")
        need(all(getattr(before, key) == getattr(after, key) == getattr(named, key)
                 for key in fields) and not os.read(fd, 1), "binding_rejected")
        need(hashlib.sha256(raw).hexdigest() == HARNESS_SHA256, "binding_rejected")
    finally:
        os.close(fd)
    selected = types.ModuleType("_lumi_setup_preflight_harness_v1")
    selected.__file__ = str(path)
    exec(compile(bytes(raw), str(path), "exec"), selected.__dict__)
    return selected


def capacity(space):
    blocks, fragment = space.f_bavail, space.f_frsize
    need(type(blocks) is type(fragment) is int and 0 <= blocks < 2**63
         and 0 < fragment <= 1024**2)
    available = blocks * fragment
    need(available < 2**63)
    tar = ((TAR_BYTES + fragment - 1) // fragment) * fragment
    zipped = ((ZIP_BYTES + fragment - 1) // fragment) * fragment
    minimum = max(tar + zipped, MINIMUM_FREE_DISK_BYTES + tar)
    return {"available_disk_bytes": available, "filesystem_fragment_bytes": fragment,
            "retained_tar_allocation_bytes": tar, "acquisition_peak_allocation_bytes": tar + zipped,
            "minimum_preacquisition_free_disk_bytes": minimum,
            "disk_floor_met": available >= MINIMUM_FREE_DISK_BYTES,
            "acquisition_headroom_met": available >= minimum}


def empty_report():
    return {"schema": SCHEMA, "status": "failed", "stage": "runtime", "code": "step_failed",
            "maximum_seconds": MAXIMUM_SECONDS, "source_bound_files": 0,
            "minimum_postacquisition_free_disk_bytes": MINIMUM_FREE_DISK_BYTES,
            "owned_temp_directories_created": 0, "owned_temp_directories_removed": 0,
            "directory_creation_ambiguous": False,
            "private_owned_cleanup_verified": True, "setup_subreaper_installed": False,
            "ready_for_unchanged_setup": False,
            **dict.fromkeys(DISK_INTS, None), **dict.fromkeys(DISK_BOOLS, False),
            **dict.fromkeys(NONCLAIMS, False)}


def cleanup(owned):
    removed, confirmed = 0, True
    for path, expected in reversed(owned):
        try:
            need(expected is not None and identity(path.lstat()) == expected
                 and expected[2:4] == (stat.S_IFDIR, 0o700), "cleanup_unconfirmed")
            path.rmdir()  # Empty owned directory only; no recursive removal.
            removed += 1
        except BaseException:
            confirmed = False
    return removed, confirmed


def run():
    global _ATTEMPTED
    result, owned, armed = empty_report(), [], False
    try:
        runtime()  # Reject Mac and non-isolated/non-root contexts before any source read/write.
        need(not _ATTEMPTED, "already_attempted")
        _ATTEMPTED = True
        armed = True
        arm_deadline()
        install_guard()
        result["stage"] = "harness_binding"
        harness = load_harness()
        harness.runtime()
        need(harness.MINIMUM_FREE_DISK_BYTES == MINIMUM_FREE_DISK_BYTES
             and harness.ARCHIVE_BYTES == TAR_BYTES, "binding_rejected")
        result["stage"] = "source_closure"
        helper, pipeline, _workspaces, _bundles, count = harness.load_sources()
        need(type(count) is int and count == 78, "binding_rejected")
        result["source_bound_files"] = count
        result["stage"] = "base_module"
        base = helper.load_base()
        result["stage"] = "subreaper"
        base.subreaper()  # Process-local kernel flag only; no child is spawned by this probe.
        result["setup_subreaper_installed"] = True
        result["stage"] = "admission_module"
        admission = helper.load_admission()
        need(admission.SoftwareBinding().image_id == admission.IMAGE_ID, "binding_rejected")
        result["stage"] = "rootfs_binding"
        need(pipeline.addon.rootfs is pipeline.layout.rootfs, "binding_rejected")
        result["stage"] = "temp_parent"
        result["directory_creation_ambiguous"] = True
        parent = Path(tempfile.mkdtemp(prefix="lumi-setup-preflight-v1-", dir="/tmp"))
        owned.append((parent, None))
        owned[-1] = (parent, identity(parent.lstat()))
        result["directory_creation_ambiguous"] = False
        result["stage"] = "temp_config"
        config = parent / "docker-config"
        result["directory_creation_ambiguous"] = True
        config.mkdir(mode=0o700)
        owned.append((config, None))
        owned[-1] = (config, identity(config.lstat()))
        result["directory_creation_ambiguous"] = False
        result["stage"] = "disk_query"
        result.update(capacity(os.statvfs(parent)))
        result["status"] = "observed"
        result["code"] = ("disk_floor_insufficient" if not result["disk_floor_met"] else
                          "acquisition_headroom_insufficient"
                          if not result["acquisition_headroom_met"] else "ok")
        result["stage"] = "complete"
    except BaseException as error:
        result["code"] = error.code if type(error) is PreflightError else "step_failed"
    finally:
        result["owned_temp_directories_created"] = len(owned)
        removed, confirmed = cleanup(owned)
        confirmed = confirmed and not result["directory_creation_ambiguous"]
        result["owned_temp_directories_removed"] = removed
        result["private_owned_cleanup_verified"] = confirmed
        if not confirmed:
            result["status"], result["code"] = "failed", "cleanup_unconfirmed"
        if armed:
            disarm_deadline()
    result["ready_for_unchanged_setup"] = result["status"] == "observed" and result["code"] == "ok"
    validate_report(result)
    return result


def validate_report(value):
    expected = empty_report()
    need(type(value) is dict and set(value) == set(expected))
    for name in ("schema", "maximum_seconds", "minimum_postacquisition_free_disk_bytes"):
        need(type(value[name]) is type(expected[name]) and value[name] == expected[name])
    need(type(value["status"]) is str and value["status"] in {"observed", "failed"}
         and type(value["stage"]) is str and value["stage"] in STAGES
         and type(value["code"]) is str and value["code"] in CODES)
    for name in NONCLAIMS:
        need(value[name] is False)
    for name in (*DISK_BOOLS, "private_owned_cleanup_verified", "setup_subreaper_installed",
                 "ready_for_unchanged_setup", "directory_creation_ambiguous"):
        need(type(value[name]) is bool)
    for name in ("source_bound_files", "owned_temp_directories_created",
                 "owned_temp_directories_removed"):
        need(type(value[name]) is int and value[name] >= 0)
    need(value["source_bound_files"] in (0, 78))
    created, removed = (value[name] for name in (
        "owned_temp_directories_created", "owned_temp_directories_removed"))
    need(0 <= removed <= created <= 2)
    need(value["private_owned_cleanup_verified"]
         == (created == removed and not value["directory_creation_ambiguous"]))
    present = [value[name] is not None for name in DISK_INTS]
    need(all(present) or not any(present))
    if all(present):
        need(all(type(value[name]) is int and 0 <= value[name] < 2**63 for name in DISK_INTS))
        fragment = value["filesystem_fragment_bytes"]
        need(fragment > 0 and value["available_disk_bytes"] % fragment == 0)
        space = types.SimpleNamespace(f_bavail=value["available_disk_bytes"] // fragment,
                                      f_frsize=fragment)
        need(all(value[name] == part for name, part in capacity(space).items()))
    else:
        need(not any(value[name] for name in DISK_BOOLS))
    if value["status"] == "observed":
        code = ("disk_floor_insufficient" if not value["disk_floor_met"] else
                "acquisition_headroom_insufficient" if not value["acquisition_headroom_met"]
                else "ok")
        need(value["stage"] == "complete" and value["code"] == code and all(present)
             and value["source_bound_files"] == 78 and value["setup_subreaper_installed"]
             and created == removed == 2 and value["private_owned_cleanup_verified"])
    else:
        need(value["code"] not in {"ok", "disk_floor_insufficient",
                                   "acquisition_headroom_insufficient"})
    need(value["ready_for_unchanged_setup"]
         == (value["status"] == "observed" and value["code"] == "ok"))
    need(len(canonical(value)) <= MAXIMUM_REPORT_BYTES)
    return value


def main():
    if len(sys.argv) != 2 or sys.argv[1] != "--probe-setup":
        result = empty_report()
        result["code"] = "runtime_rejected"
    else:
        result = run()
    raw = canonical(validate_report(result))
    need(os.write(1, raw) == len(raw))
    # A measured capacity shortfall is a successful diagnostic, not setup readiness.
    return 0 if result["status"] == "observed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
