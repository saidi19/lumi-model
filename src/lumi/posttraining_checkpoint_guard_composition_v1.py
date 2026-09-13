"""One generated-only import under the frozen controller/guardian lifecycle.

This internal API accepts an ALREADY independently authenticated, prepared and
frozen rootfs/source bundle. It neither acquires a rootfs nor establishes origin,
host admission or authorization. It is not a real-checkpoint import interface.
Only a root Linux x86_64 CPython 3.12 -S -B caller can start the native path.
The guardian watches controller loss/deadline; guardian death and hostile root
remain outside the claim. Tmpfs/shm limits are capacities, not write accounting.
"""

from __future__ import annotations

import array
import hashlib
import json
import os
import re
import select
import socket
import stat
import time
import types
import uuid
from pathlib import Path, PurePosixPath

from . import posttraining_checkpoint_guard_bootstrap_v1 as child_bootstrap

BASE_SHA256 = "f91ed0c353a20c309ff3e1bfaab8dbe990b28c36626fdeeb0ac5131e0cf75fad"
SCHEMA = "lumi.checkpoint-prepared-guard-result.v0.1"
WORKER_SCHEMA = "lumi.checkpoint-guard-fixture-result.v0.1"
LIMITS = {"memory_max_bytes": 4 * 1024**3, "pids_max": 256,
          "cpu_quota_us": 200_000, "cpu_period_us": 100_000}
MAXIMUM_SECONDS = 90
MAXIMUM_PACKET_BYTES = 4096
NONCLAIMS = ("real_4b_import", "training_authorized", "production_eligible",
             "host_admission_verified", "runtime_origin_authenticated",
             "guardian_death_cleanup_verified", "hostile_root_containment_verified",
             "complete_os_sandbox_verified", "cumulative_tmpfs_writes_metered")
ERROR_CODES = frozenset({"runtime_rejected", "arguments_rejected", "already_attempted",
                         "source_rejected", "packet_rejected", "worker_rejected",
                         "guard_failed", "cleanup_unconfirmed", "deadline_exhausted"})
_ATTEMPTED = False


class GuardCompositionError(ValueError):
    def __init__(self, code="guard_failed"):
        self.code = code if type(code) is str and code in ERROR_CODES else "guard_failed"
        super().__init__(self.code)


def _need(value, code="guard_failed"):
    if not value:
        raise GuardCompositionError(code) from None


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _pairs(rows):
    result = {}
    for key, value in rows:
        _need(key not in result, "packet_rejected")
        result[key] = value
    return result


def _integer(value):
    _need(len(value) <= 19, "packet_rejected")
    result = int(value)
    _need(0 <= result < 2**63, "packet_rejected")
    return result


def _reject_number(_value):
    raise GuardCompositionError("packet_rejected") from None


def _parse(raw):
    try:
        _need(type(raw) is bytes and 0 < len(raw) <= MAXIMUM_PACKET_BYTES, "packet_rejected")
        value = json.loads(raw, object_pairs_hook=_pairs, parse_int=_integer,
                           parse_float=_reject_number, parse_constant=_reject_number)
        _need(type(value) is dict, "packet_rejected")
        pending, count = [(value, 0)], 0
        while pending:
            item, depth = pending.pop()
            count += 1
            _need(count <= 256 and depth <= 4, "packet_rejected")
            if type(item) is dict:
                _need(all(type(key) is str and len(key) <= 128 for key in item),
                      "packet_rejected")
                pending.extend((part, depth + 1) for part in item.values())
            else:
                _need(type(item) in (str, int, bool), "packet_rejected")
                _need(type(item) is not str or len(item) <= 256, "packet_rejected")
        _need(raw in (_canonical(value), _canonical(value) + b"\n"), "packet_rejected")
        return value
    except GuardCompositionError:
        raise
    except BaseException:
        raise GuardCompositionError("packet_rejected") from None


def _ancillary_close(ancillary):
    # A rejected SCM_RIGHTS packet still transfers descriptors. Consume each
    # received descriptor once; failure stays terminal, never retry a close.
    failed = False
    for level, kind, data in ancillary:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            values = array.array("i")
            values.frombytes(data[:len(data) - len(data) % values.itemsize])
            for descriptor in values:
                try:
                    os.close(descriptor)
                except BaseException:
                    failed = True
    _need(not failed, "packet_rejected")


def _packet(channel, deadline_ns):
    remaining = (deadline_ns - time.monotonic_ns()) / 1_000_000_000
    _need(remaining > 0, "deadline_exhausted")
    _need(select.select([channel], [], [], min(remaining, 30))[0], "packet_rejected")
    raw, ancillary, flags, _address = channel.recvmsg(MAXIMUM_PACKET_BYTES, 256)
    if ancillary:
        _ancillary_close(ancillary)
    _need(not ancillary and flags == 0, "packet_rejected")
    return raw


def _receive(channel, deadline_ns):
    return _parse(_packet(channel, deadline_ns))


def _worker(value, environment):
    fixed = {
        "schema": WORKER_SCHEMA, "status": "passed", "code": "ok", "synthetic_only": True,
        "import_passes": 1, "checkpoint_full_audits": 2, "tokenizer_full_audits": 2,
        "frozen_worker_checksum_verifier_passed": True,
        "fixture_generation_in_import_ledger": False, "real_4b_import": False,
        "training_authorized": False, "production_eligible": False, "gpu_used": False,
        "forward_calls": 0, "optimizer_steps": 0, "host_containment_self_attested": False,
        "private_source_copies_rootfs_immutable": False,
        "hostile_owner_code_immutability_verified": False,
    }
    bounded = {"model_parameters": 4096, "generated_input_bytes": 4 * 1024**2,
               "generated_input_files": 32, "logical_content_read_bytes": 64 * 1024**2,
               "private_write_bytes_reserved": 16 * 1024**2,
               "source_files_verified": 128, "source_bytes_verified": 4 * 1024**2,
               "source_verification_read_bytes": 4 * 1024**2 + 65_536,
               "private_source_copy_bytes_reserved": 4 * 1024**2,
               "private_source_copy_readback_bytes": 4 * 1024**2,
               "private_source_copy_files": 128}
    hashes = {"plan_sha256", "kernel_io_contract_sha256", "kernel_evidence_sha256",
              "package_manifest_sha256", "source_manifest_sha256", "entry_sha256"}
    _need(type(value) is dict and set(value) == set(fixed) | set(bounded) | hashes,
          "worker_rejected")
    _need(all(type(value[key]) is type(expected) and value[key] == expected
              for key, expected in fixed.items()), "worker_rejected")
    _need(all(type(value[key]) is int and 0 < value[key] <= maximum
              for key, maximum in bounded.items()), "worker_rejected")
    _need(all(type(value[key]) is str and re.fullmatch(r"[a-f0-9]{64}", value[key])
              for key in hashes), "worker_rejected")
    _need(value["source_manifest_sha256"] == environment["LUMI_IMPORT_SOURCE_MANIFEST_SHA256"]
          and value["entry_sha256"] == environment["LUMI_IMPORT_ENTRY_SHA256"]
          and 2 <= value["source_files_verified"] == value["private_source_copy_files"]
          and value["source_bytes_verified"] == value["private_source_copy_bytes_reserved"]
          == value["private_source_copy_readback_bytes"]
          and 0 < value["source_verification_read_bytes"] - value["source_bytes_verified"]
          <= 65_536, "worker_rejected")
    return dict(value)


def _state(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
            info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _load_base(path, expected):
    _need(expected == BASE_SHA256, "source_rejected")
    selected = Path(path)
    _need(selected.is_absolute() and ".." not in selected.parts
          and selected.parts[-2:] == ("scripts", "run_sft_cgroup_fixture_v1.py"),
          "source_rejected")
    observations = [(part, part.lstat()) for part in reversed((selected, *selected.parents))]
    _need(all(not stat.S_ISLNK(info.st_mode) for _, info in observations), "source_rejected")
    descriptor = os.open(selected, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        _need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
              and 0 < before.st_size <= 65_536, "source_rejected")
        raw = os.read(descriptor, before.st_size)
        _need(len(raw) == before.st_size and _state(before) == _state(os.fstat(descriptor))
              and hashlib.sha256(raw).hexdigest() == expected, "source_rejected")
        _need(all(_state(info) == _state(part.lstat()) for part, info in observations),
              "source_rejected")
    finally:
        os.close(descriptor)
    selected_module = types.ModuleType("_lumi_checkpoint_guard_base_" + uuid.uuid4().hex)
    selected_module.__file__ = str(selected)
    exec(compile(raw, str(selected), "exec"), selected_module.__dict__)
    selected_module.PACKAGE = selected_module.__name__ + "_components"
    return selected_module


def _guard(base, context, case, channel, controller_pid, _uid, _gid, _runtime, deadline_ns):
    """Runs only in the frozen controller's freshly forked independent guardian."""
    parent_fd = lease = workload_channel = child_channel = None
    killed, close_attempted, removed, cleanup_ok = [0], False, False, True
    result = {"status": "failed", "case": case, "code": "guard_failed",
              "empty_and_removed": False}
    try:
        base.close_except({channel.fileno()})
        base.subreaper()
        parent_fd = os.open("/sys/fs/cgroup", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                            | os.O_CLOEXEC | os.O_NONBLOCK)
        lease = base.module("posttraining_sft_cgroup_v1").CgroupLease(
            parent_fd, name="lumi-sft-" + uuid.uuid4().hex, limits=dict(LIMITS))
        original_kill = lease.kill

        def kill_once():
            _need(killed[0] == 0)
            killed[0] += 1
            return original_kill()

        lease.kill = kill_once
        lease.verify_limits()
        _need(lease.snapshot()["populated"] == 0)
        workload_channel, child_channel = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        group_fd = lease.fileno()
        worker_pid = os.fork()
        if worker_pid == 0:
            try:
                child_bootstrap.bootstrap(group_fd, child_channel, **context)
            finally:
                os._exit(2)
        closing, child_channel = child_channel, None
        closing.close()
        _need(_canonical(_receive(workload_channel, deadline_ns))
              == _canonical(child_bootstrap.READY))
        observed = lease.snapshot()
        _need(observed["populated"] == observed["pids_current"] == 1)
        released = [False]

        def release():
            _need(not released[0])
            released[0] = True
            _need(workload_channel.send(b"G") == 1)
            base.send(channel, {"phase": "armed", "case": case})

        watched = base.module("posttraining_sft_cgroup_guardian_v1").watch_parent(
            lease, expected_parent_pid=controller_pid, deadline_ns=deadline_ns,
            cleanup_timeout_ms=10_000, release=release)
        _need(watched["reason"] == "workload_empty" and killed[0] == 1 and released[0]
              and all(watched[key] is True for key in ("kill_attempted", "empty_verified",
                      "parent_pidfd_bound", "release_called", "pidfd_closed")))
        stop = min(deadline_ns + 10_000_000_000, time.monotonic_ns() + 10_000_000_000)
        statuses = base.wait_statuses(stop)
        _need(worker_pid in statuses and 1 <= len(statuses) <= 4
              and all(type(value) is int and value == 0 for value in statuses.values()))
        observed = lease.snapshot()
        _need(observed["populated"] == observed["pids_current"] == 0)
        worker = _worker(_receive(workload_channel, stop), context["environment"])
        _need(_packet(workload_channel, stop) == b"", "packet_rejected")
        close_attempted = True
        lease.close()
        removed = True
        result = {"status": "passed", "case": case, "reason": "workload_empty",
                  "empty_and_removed": True, "workload_processes_reaped": len(statuses),
                  "kill_attempts": killed[0], "worker": worker}
    except BaseException:
        pass  # No exception, path, PID, source or candidate text enters a receipt.
    finally:
        if lease is not None and not removed:
            try:
                if killed[0] == 0:
                    lease.kill()
                stop = min(deadline_ns + 10_000_000_000, time.monotonic_ns() + 10_000_000_000)
                while lease.snapshot()["populated"] and time.monotonic_ns() < stop:
                    time.sleep(0.02)
                observed = lease.snapshot()
                _need(observed["populated"] == observed["pids_current"] == 0)
                base.wait_statuses(stop)
                _need(not close_attempted)
                close_attempted = True
                lease.close()
                removed = True
            except BaseException:
                cleanup_ok = False
        for closing in (child_channel, workload_channel):
            if closing is not None:
                try:
                    closing.close()
                except BaseException:
                    cleanup_ok = False
        if parent_fd is not None:
            try:
                os.close(parent_fd)
            except BaseException:
                cleanup_ok = False
        if not removed or not cleanup_ok:
            result = {"status": "failed", "case": case, "code": "cleanup_unconfirmed",
                      "empty_and_removed": False}
        elif result["status"] == "failed":
            result["empty_and_removed"] = True
        try:
            base.send(channel, result)
        finally:
            channel.close()
    os._exit(0 if result["status"] == "passed" else 2)


def run_prepared_synthetic_guard(rootfs_path, mountpoint_path, root_identity, environment,
                                 base_script_path, expected_base_sha256, deadline_ns):
    """One attempt, only for a separately authenticated/frozen prepared rootfs.

    The bound base and all imported host-side helper sources must belong to that
    independently verified source bundle. This API cannot prove that premise.
    No source acquisition, arbitrary executable/argv or real import is exposed.
    """
    global _ATTEMPTED
    base = None
    try:
        child_bootstrap._runtime()  # Must reject unsupported hosts before any I/O.
    except BaseException:
        raise GuardCompositionError("runtime_rejected") from None
    try:
        copied_environment = child_bootstrap.validate_environment(environment)
        paths = []
        for path in (rootfs_path, mountpoint_path):
            _need(type(path) is str and 0 < len(path) <= 4096
                  and all(32 <= ord(c) < 127 for c in path), "arguments_rejected")
            parsed = PurePosixPath(path)
            _need(parsed.is_absolute() and str(parsed) == path and ".." not in parsed.parts
                  and 3 <= len(parsed.parts) <= 64 and not path.startswith("//"),
                  "arguments_rejected")
            paths.append(parsed)
        _need(paths[0].parent == paths[1].parent and paths[0] != paths[1]
              and type(root_identity) is tuple and len(root_identity) == 5
              and all(type(value) is int for value in root_identity)
              and 0 <= root_identity[0] < 2**64 and 0 < root_identity[1] < 2**64
              and root_identity[2:] == (stat.S_IFDIR, 0, 0)
              and type(base_script_path) is str and 0 < len(base_script_path) <= 4096
              and type(expected_base_sha256) is str and expected_base_sha256 == BASE_SHA256,
              "arguments_rejected")
        now = time.monotonic_ns()
        _need(type(deadline_ns) is int and now < deadline_ns <= now + 90_000_000_000,
              "arguments_rejected")
    except GuardCompositionError:
        raise
    except BaseException:
        raise GuardCompositionError("arguments_rejected") from None
    _need(not _ATTEMPTED, "already_attempted")
    _ATTEMPTED = True
    try:
        context = {"rootfs_path": rootfs_path, "mountpoint_path": mountpoint_path,
                   "root_identity": root_identity, "environment": copied_environment}
        base = _load_base(base_script_path, expected_base_sha256)
        base.receive = _receive  # Only this fresh module, never process-global patches.

        def guarded(*args):
            return _guard(base, context, *args)

        base.guardian = guarded
        base.subreaper()
        result = base.run_case("worker_run", 10001, 10001, child_bootstrap.RUNTIME, deadline_ns)
        _need(type(result) is dict and set(result) == {
            "status", "case", "reason", "empty_and_removed", "workload_processes_reaped",
            "kill_attempts", "worker"} and result["status"] == "passed"
            and result["case"] == "worker_run" and result["reason"] == "workload_empty"
            and result["empty_and_removed"] is True and type(result["kill_attempts"]) is int
            and result["kill_attempts"] == 1
            and type(result["workload_processes_reaped"]) is int
            and 1 <= result["workload_processes_reaped"] <= 4)
        worker = _worker(result["worker"], copied_environment)
        _need(time.monotonic_ns() < deadline_ns, "deadline_exhausted")
        return {"schema": SCHEMA, "status": "passed", "synthetic_only": True,
                "owned_cgroups_empty_and_removed": 1, "workload_processes_reaped":
                result["workload_processes_reaped"], "kill_attempts": 1,
                "controller_and_deadline_watch_installed_before_release": True,
                "worker_exit_status_verified": True, "worker": worker,
                "memory_max_bytes": LIMITS["memory_max_bytes"], "swap_max_bytes": 0,
                "pids_max": LIMITS["pids_max"], "cpu_quota_us": LIMITS["cpu_quota_us"],
                "cpu_period_us": LIMITS["cpu_period_us"], "maximum_seconds": MAXIMUM_SECONDS,
                "tmpfs_capacity_bytes": 512 * 1024**2, "shm_capacity_bytes": 64 * 1024**2,
                **dict.fromkeys(NONCLAIMS, False)}
    except BaseException as error:
        # run_case kills only its owned controller on failure. Its independent
        # guardian then observes that parent loss; reap before leaving this API.
        # Reaping alone is deliberately not a removal receipt or success claim.
        if base is not None:
            try:
                base.wait_statuses(min(deadline_ns, time.monotonic_ns() + 15_000_000_000))
            except BaseException:
                raise GuardCompositionError("cleanup_unconfirmed") from None
        code = error.code if type(error) is GuardCompositionError else "guard_failed"
        raise GuardCompositionError(code) from None
