"""One generated Linux qualification case; never a real-input import API.

Frozen composition v1 remains unchanged. The caller supplies its independently
admitted/frozen rootfs and authenticated host helper sources. Run each case in a
fresh root child: this module consumes exactly one attempt per process. Only the
two exact intentional pin mismatches and one positive tiny import are accepted.
No native execution occurs at import. Origin, host admission, guardian-death and
hostile-root containment remain unproved; tmpfs capacities are not write meters.
"""

from __future__ import annotations

import os
import socket
import stat
import time
import uuid
from pathlib import PurePosixPath

from . import posttraining_checkpoint_guard_composition_v1 as frozen

CASES = ("wrong_entry", "wrong_manifest", "positive")
SCHEMA = "lumi.checkpoint-guard-qualification-case.v0.1"
BASE_SHA256 = frozen.BASE_SHA256
LIMITS = dict(frozen.LIMITS)
MAXIMUM_SECONDS = frozen.MAXIMUM_SECONDS
ERROR_CODES = frozen.ERROR_CODES | {"case_rejected"}
_ATTEMPTED = False


class QualificationError(ValueError):
    def __init__(self, code="guard_failed"):
        self.code = code if type(code) is str and code in ERROR_CODES else "guard_failed"
        super().__init__(self.code)


def _need(value, code="guard_failed"):
    if not value:
        raise QualificationError(code) from None


def _environment(case, environment):
    _need(type(case) is str and case in CASES, "case_rejected")
    copied = frozen.child_bootstrap.validate_environment(environment)
    _need(all(copied[key] != "0" * 64 for key in frozen.child_bootstrap.HASH_ENV),
          "arguments_rejected")
    if case == "wrong_entry":
        copied["LUMI_IMPORT_ENTRY_SHA256"] = "0" * 64
    elif case == "wrong_manifest":
        copied["LUMI_IMPORT_SOURCE_MANIFEST_SHA256"] = "0" * 64
    return copied


def _worker(value, case, environment, exit_code):
    _need(type(case) is str and case in CASES, "case_rejected")
    _need(type(exit_code) is int and exit_code == (0 if case == "positive" else 1),
          "worker_rejected")
    if case == "positive":
        return frozen._worker(value, environment)
    expected = {"schema": frozen.WORKER_SCHEMA, "status": "failed",
                "code": "entry_binding_rejected" if case == "wrong_entry"
                else "source_binding_rejected", "synthetic_only": True,
                "real_4b_import": False, "training_authorized": False,
                "production_eligible": False}
    _need(type(value) is dict and set(value) == set(expected)
          and all(type(value[key]) is type(wanted) and value[key] == wanted
                  for key, wanted in expected.items()), "worker_rejected")
    return dict(value)


def _guard(base, context, qualification_case, case, channel, controller_pid,
           _uid, _gid, _runtime, deadline_ns):
    """Frozen lifecycle, with only the selected worker receipt/exit expectation changed."""
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
                frozen.child_bootstrap.bootstrap(group_fd, child_channel, **context)
            finally:
                os._exit(2)
        closing, child_channel = child_channel, None
        closing.close()
        _need(frozen._canonical(frozen._receive(workload_channel, deadline_ns))
              == frozen._canonical(frozen.child_bootstrap.READY))
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
        exit_code = 0 if qualification_case == "positive" else 1
        _need(worker_pid in statuses and 1 <= len(statuses) <= 4
              and all(type(value) is int and value == (exit_code << 8 if pid == worker_pid else 0)
                      for pid, value in statuses.items()), "worker_rejected")
        observed = lease.snapshot()
        _need(observed["populated"] == observed["pids_current"] == 0)
        worker = _worker(frozen._receive(workload_channel, stop), qualification_case,
                         context["environment"], exit_code)
        _need(frozen._packet(workload_channel, stop) == b"", "packet_rejected")
        close_attempted = True
        lease.close()
        removed = True
        result = {"status": "passed", "case": case, "reason": "workload_empty",
                  "empty_and_removed": True, "workload_processes_reaped": len(statuses),
                  "kill_attempts": killed[0], "worker_exit_code": exit_code, "worker": worker}
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


def run_prepared_case(case, rootfs_path, mountpoint_path, root_identity, environment,
                      base_script_path, expected_base_sha256, deadline_ns):
    """One case, no retry; genuine pins in, selected deliberate mismatch copied privately.

    Source/rootfs admission and PID-only outer supervision are caller premises.
    This API does not acquire software, broaden an import plan or admit a host.
    """
    global _ATTEMPTED
    base = None
    try:
        frozen.child_bootstrap._runtime()  # Unsupported hosts reject before all I/O.
    except BaseException:
        raise QualificationError("runtime_rejected") from None
    try:
        copied_environment = _environment(case, environment)
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
    except QualificationError:
        raise
    except BaseException:
        raise QualificationError("arguments_rejected") from None
    _need(not _ATTEMPTED, "already_attempted")
    _ATTEMPTED = True
    try:
        context = {"rootfs_path": rootfs_path, "mountpoint_path": mountpoint_path,
                   "root_identity": root_identity, "environment": copied_environment}
        base = frozen._load_base(base_script_path, expected_base_sha256)
        base.receive = frozen._receive  # Only this freshly SHA-verified module instance.

        def guarded(*args):
            return _guard(base, context, case, *args)

        base.guardian = guarded
        base.subreaper()
        result = base.run_case("worker_run", 10001, 10001, frozen.child_bootstrap.RUNTIME,
                               deadline_ns)
        _need(type(result) is dict and set(result) == {
            "status", "case", "reason", "empty_and_removed", "workload_processes_reaped",
            "kill_attempts", "worker_exit_code", "worker"} and result["status"] == "passed"
            and result["case"] == "worker_run" and result["reason"] == "workload_empty"
            and result["empty_and_removed"] is True and type(result["kill_attempts"]) is int
            and result["kill_attempts"] == 1
            and type(result["workload_processes_reaped"]) is int
            and 1 <= result["workload_processes_reaped"] <= 4)
        worker = _worker(result["worker"], case, copied_environment, result["worker_exit_code"])
        _need(time.monotonic_ns() < deadline_ns, "deadline_exhausted")
        return {"schema": SCHEMA, "status": "passed", "synthetic_only": True, "case": case,
                "expects_negative": case != "positive",
                "worker_exit_code": result["worker_exit_code"],
                "owned_cgroups_empty_and_removed": 1, "workload_processes_reaped":
                result["workload_processes_reaped"], "kill_attempts": 1,
                "controller_and_deadline_watch_installed_before_release": True,
                "worker_exit_status_verified": True, "worker": worker,
                "memory_max_bytes": LIMITS["memory_max_bytes"], "swap_max_bytes": 0,
                "pids_max": LIMITS["pids_max"], "cpu_quota_us": LIMITS["cpu_quota_us"],
                "cpu_period_us": LIMITS["cpu_period_us"], "maximum_seconds": MAXIMUM_SECONDS,
                "tmpfs_capacity_bytes": 512 * 1024**2, "shm_capacity_bytes": 64 * 1024**2,
                **dict.fromkeys(frozen.NONCLAIMS, False)}
    except BaseException as error:
        if base is not None:
            try:
                base.wait_statuses(min(deadline_ns, time.monotonic_ns() + 15_000_000_000))
            except BaseException:
                raise QualificationError("cleanup_unconfirmed") from None
        code = (error.code if type(error) in (QualificationError, frozen.GuardCompositionError)
                else "guard_failed")
        raise QualificationError(code) from None
