#!/usr/bin/env python3
"""Privileged generated Linux cgroup fixture; NEVER execute on the local Mac.

Only fixed stdlib payloads and the exactly pinned prior generated CPU WorkerRun
fixture run. No root process imports ML. The runtime path is a typed,
bounded hostedtoolcache Python 3.12 ELF binding, executed only after permanent
UID/GID drop, NO_NEW_PRIVS and closure of every cgroup/pidfd authority descriptor.
The binding verifies bytes/shape, not host authenticity or production admission.

Every lease is created/owned by its independent root guardian. The hierarchy is
outer subreaper -> controller -> guardian subreaper -> dropped-UID workload. The
controller watches its outer parent by pidfd; the guardian watches the controller
and an independent deadline, including after controller/outer loss. Only fresh
owned non-root cgroups are changed. Parent controllers are never enabled/changed.
Only the owning guardian kills, checks empty, reaps descendants and removes its
empty leaf. No recursive deletion or unprivileged killpg cleanup claim exists.

Generated resource acceptance is not a complete OS/network/filesystem sandbox,
real-input permission, GPU proof, real-host admission or production eligibility.
All public output is bounded fixed codes/counts/booleans, never PIDs/paths/logs.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import importlib
import json
import os
import re
import select
import signal
import socket
import stat
import sys
import time
import types
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_lumi_generated_cgroup_fixture_v1"
SCHEMA = "lumi.sft-cgroup-linux-fixture.v0.1"
ACCEPTED_SCRIPT = ROOT / "scripts/run_sft_worker_run_torch_fixture_v1.py"
ACCEPTED_SHA256 = "53c50c92965ed1cef0c9624c57367c31ed7ebfb9d7cfb34257188ad179d333d0"
RUNTIME_PATTERN = r"/opt/hostedtoolcache/Python/3\.12\.[0-9]{1,3}/x64/bin/python3\.12"
MAXIMUM_SECONDS = 260
MAXIMUM_MESSAGE_BYTES = 4096
CLEANUP_SECONDS = 10
STAGE, CURRENT_CASE = "guard", "not_started"
COMPONENT_CODE, REMOVAL_REPORTED = "not_reported", False
CASES = ("normal", "memory", "pids", "cpu", "deadline", "parent_loss", "worker_run")
FAILURE_CODES = frozenset(("fixture_failed", "deadline_exhausted", "cleanup_unconfirmed"))
NONCLAIMS = (
    "real_model_or_data_access", "cuda_used", "production_eligible",
    "real_sft_execution_permitted", "complete_os_sandbox_verified",
    "network_isolation_verified", "filesystem_isolation_verified",
    "real_host_admission_verified", "runtime_origin_authenticated",
    "root_parent_controllers_modified", "packages_installed",
)


class FixtureError(RuntimeError):
    def __init__(self, code="fixture_failed"):
        self.code = code if code in FAILURE_CODES else "fixture_failed"
        super().__init__(self.code)


def require(condition):
    if not condition:
        raise FixtureError()


def module(name):
    if PACKAGE not in sys.modules:
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(ROOT / "src/lumi")]
        sys.modules[PACKAGE] = package
    return importlib.import_module(f"{PACKAGE}.{name}")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def send(channel, value):
    raw = canonical(value)
    require(len(raw) <= MAXIMUM_MESSAGE_BYTES and channel.send(raw) == len(raw))


def receive(channel, deadline_ns):
    remaining = (deadline_ns - time.monotonic_ns()) / 1_000_000_000
    require(remaining > 0 and select.select([channel], [], [], min(remaining, 30))[0])
    raw, _ancillary, flags, _address = channel.recvmsg(MAXIMUM_MESSAGE_BYTES)
    require(raw and not flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC))
    result = json.loads(raw)
    require(type(result) is dict)
    return result


def close_except(kept):
    # Linux procfs only; no descriptor number guesses or inherited authority survive.
    descriptors = os.listdir("/proc/self/fd")
    require(len(descriptors) <= 1024)
    for value in descriptors:
        require(value.isdecimal())
        fd = int(value)
        if fd > 2 and fd not in kept:
            try:
                os.close(fd)
            except OSError as error:
                require(error.errno == errno.EBADF)  # Closed transient listdir descriptor.


def subreaper():
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.restype = ctypes.c_int
    observed = ctypes.c_int()
    require(prctl(36, 1, 0, 0, 0) == 0)
    require(prctl(37, ctypes.byref(observed), 0, 0, 0) == 0 and observed.value == 1)


def verify_file(path, expected, maximum, *, elf=False):
    require(type(path) is str and type(expected) is str
            and re.fullmatch(r"[a-f0-9]{64}", expected) is not None)
    selected = Path(path)
    require(selected.is_absolute() and ".." not in selected.parts)
    for current in reversed((selected, *selected.parents)):
        require(not stat.S_ISLNK(current.lstat().st_mode))
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= maximum)
        hasher, size, first = hashlib.sha256(), 0, True
        while size < before.st_size:
            raw = os.read(fd, min(65_536, before.st_size - size))
            require(raw)
            if first and elf:
                require(raw.startswith(b"\x7fELF"))
            first = False
            hasher.update(raw)
            size += len(raw)
        after = os.fstat(fd)
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                 before.st_ctime_ns) == (after.st_dev, after.st_ino, after.st_size,
                                        after.st_mtime_ns, after.st_ctime_ns))
        require(not os.read(fd, 1) and hasher.hexdigest() == expected)
    finally:
        os.close(fd)


def bootstrap():
    require(sys.platform == "linux" and os.uname().machine == "x86_64")
    require(sys.dont_write_bytecode and sys.flags.isolated and sys.flags.no_site)
    require(sys.argv[1:] == ["--synthetic-only"] and os.geteuid() == os.getuid() == 0)
    require(hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"))
    uid, gid = os.environ.get("SUDO_UID", ""), os.environ.get("SUDO_GID", "")
    require(re.fullmatch(r"[1-9][0-9]{0,9}", uid) is not None)
    require(re.fullmatch(r"[1-9][0-9]{0,9}", gid) is not None)
    uid, gid = int(uid), int(gid)
    require(uid < 2**31 and gid < 2**31)
    # sudo, not user environment, supplies the original non-root numeric IDs.
    os.environ.clear()
    os.environ.update({"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8"})
    require(select.select([0], [], [], 5)[0])
    payload = os.read(0, MAXIMUM_MESSAGE_BYTES + 1)
    require(0 < len(payload) <= MAXIMUM_MESSAGE_BYTES)
    require(select.select([0], [], [], 5)[0] and os.read(0, 1) == b"")
    value = json.loads(payload)
    require(type(value) is dict and set(value) == {"runtime_path", "runtime_sha256"})
    require(type(value["runtime_path"]) is str
            and re.fullmatch(RUNTIME_PATTERN, value["runtime_path"]) is not None)
    verify_file(value["runtime_path"], value["runtime_sha256"], 64 * 1024 * 1024, elf=True)
    verify_file(str(ACCEPTED_SCRIPT), ACCEPTED_SHA256, 65_536)
    return uid, gid, value


def limits_for(case):
    return {
        "memory_max_bytes": 4 * 1024**3 if case == "worker_run" else 64 * 1024**2,
        "pids_max": 256 if case == "worker_run" else 4 if case == "pids" else 16,
        "cpu_quota_us": 10_000 if case == "cpu" else 200_000,
        "cpu_period_us": 100_000,
    }


def wait_statuses(deadline_ns):
    statuses = {}
    while time.monotonic_ns() < deadline_ns:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return statuses
        if pid:
            require(pid not in statuses)
            statuses[pid] = status
        else:
            time.sleep(0.02)
    raise FixtureError("cleanup_unconfirmed")


def pids_payload(channel):
    reader, writer = os.pipe2(os.O_CLOEXEC)
    children, rejected = [], False
    try:
        for _ in range(8):
            try:
                pid = os.fork()
            except OSError as error:
                require(error.errno == errno.EAGAIN)
                rejected = True
                break
            if pid == 0:
                os.close(writer)
                channel.close()
                os.read(reader, 1)
                os._exit(0)
            children.append(pid)
        require(rejected and len(children) == 3)
    finally:
        os.close(writer)
        os.close(reader)
        for pid in children:
            require(os.waitpid(pid, 0)[1] == 0)
    send(channel, {"phase": "done", "pids_rejected": rejected})


def parent_loss_payload(channel):
    previous_group = os.getpgrp()
    child = os.fork()
    if child == 0:
        os.setsid()
        grandchild = os.fork()
        if grandchild:
            os._exit(0)
        send(channel, {"phase": "escaped", "session_escape": os.getpgrp() != previous_group})
        time.sleep(30)
        os._exit(2)
    require(os.waitpid(child, 0)[1] == 0)
    send(channel, {"phase": "double_fork", "intermediate_reaped": True})
    time.sleep(30)


def worker_environment(runtime):
    return {
        "PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8",
        "LD_LIBRARY_PATH": str(Path(runtime).parents[1] / "lib"),
        "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "AWS_EC2_METADATA_DISABLED": "true",
        "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "GLOO_SOCKET_IFNAME": "lo", "TP_SOCKET_IFNAME": "lo",
        "LUMI_WORKER_RUN_TORCH_CI_FIXTURE": "1",
    }


def workload(case, group_fd, channel, uid, gid, runtime, deadline_ns):
    try:
        close_except({group_fd, channel.fileno()})
        module("posttraining_sft_cgroup_v1").enter_current_process(group_fd)
        module("posttraining_sft_cgroup_guardian_v1").drop_worker_privileges(uid, gid)
        os.close(group_fd)
        close_except({channel.fileno()})
        require(os.getresuid() == (uid, uid, uid) and os.getresgid() == (gid, gid, gid))
        require(os.getgroups() == [])
        send(channel, {"phase": "ready", "uid_gid_dropped": True,
                       "authority_fds_closed": True})
        require(select.select([channel], [], [], 10)[0] and channel.recv(1) == b"G")
        if case == "worker_run":
            # This executable is never loaded while root; no root-owned ML import.
            os.dup2(channel.fileno(), 1)
            close_except(set())
            os.execve(runtime, [runtime, "-I", "-B", str(ACCEPTED_SCRIPT), "--synthetic-only"],
                      worker_environment(runtime))
        elif case == "memory":
            allocation = bytearray(96 * 1024**2)
            for index in range(0, len(allocation), 4096):
                allocation[index] = 1
            raise FixtureError()  # A surviving over-limit allocation is never success.
        elif case == "pids":
            pids_payload(channel)
        elif case == "cpu":
            stop, value = time.monotonic_ns() + 550_000_000, 1
            while time.monotonic_ns() < stop:
                value = (value * 33 + 17) % 1_000_003
            require(value >= 0)
            send(channel, {"phase": "done", "busy_loop_completed": True})
        elif case == "parent_loss":
            parent_loss_payload(channel)
        elif case == "deadline":
            time.sleep(30)
            raise FixtureError()
        else:
            require(case == "normal" and time.monotonic_ns() < deadline_ns)
            send(channel, {"phase": "done", "generated_work_completed": True})
        channel.close()
        os._exit(0)
    except BaseException as error:
        for name, error_type in (("posttraining_sft_cgroup_v1", "CgroupError"),
                                 ("posttraining_sft_cgroup_guardian_v1", "GuardianError")):
            try:
                implementation = module(name)
                if (type(error) is getattr(implementation, error_type)
                        and error.code in implementation.ERROR_CODES):
                    send(channel, {"phase": "bootstrap_failed", "component_code": error.code})
            except BaseException:
                pass  # A closed channel/failed exec must never expose exception text.
        os._exit(2)


def drain_messages(channel):
    values = []
    while select.select([channel], [], [], 0)[0]:
        raw, _ancillary, flags, _address = channel.recvmsg(MAXIMUM_MESSAGE_BYTES)
        if not raw:
            break
        require(not flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC) and len(values) < 8)
        observed = json.loads(raw)
        require(type(observed) is dict)
        values.append(observed)
    return values


def verify_case(case, watched, snapshot, statuses, worker_pid, messages):
    require(snapshot["populated"] == snapshot["pids_current"] == 0)
    expected_reason = {"deadline": "deadline_exhausted", "parent_loss": "parent_lost"}
    require(watched["reason"] == expected_reason.get(case, "workload_empty"))
    require(worker_pid in statuses and 1 <= len(statuses) <= 4)
    if case in {"memory", "deadline", "parent_loss"}:
        require(os.WIFSIGNALED(statuses[worker_pid])
                and os.WTERMSIG(statuses[worker_pid]) == signal.SIGKILL)
    else:
        require(statuses[worker_pid] == 0)
    if case == "memory":
        require(snapshot["memory_max_events"] > 0 and snapshot["memory_oom_events"] > 0
                and snapshot["memory_oom_kill_events"] > 0)
    elif case == "pids":
        require(snapshot["pids_max_events"] > 0)
        require(messages == [{"phase": "done", "pids_rejected": True}])
    elif case == "cpu":
        require(snapshot["cpu_nr_throttled"] > 0 and snapshot["cpu_throttled_usec"] > 0
                and snapshot["cpu_usage_usec"] > 0)
        require(messages == [{"phase": "done", "busy_loop_completed": True}])
    elif case == "normal":
        require(messages == [{"phase": "done", "generated_work_completed": True}])
    elif case == "parent_loss":
        require(len(statuses) == 2 and not messages)
    elif case == "worker_run":
        require(len(messages) == 1)
        report = messages[0]
        require(report.get("schema_version") == "lumi.sft-worker-run-fixture.v0.1"
                and report.get("status") == "passed")
        for key, expected in (("invocations_completed", 3), ("rank_workers_verified", 6),
                              ("global_optimizer_steps_completed", 4),
                              ("controlled_rank_optimizer_steps", 8)):
            require(report.get(key) == expected)
        for key in ("reference_model_resume_equal", "reference_adam_resume_equal",
                    "reference_rng_resume_equal", "reference_training_state_resume_equal"):
            require(report.get(key) is True)


def guardian(case, channel, controller_pid, uid, gid, runtime, deadline_ns):
    close_except({channel.fileno()})
    subreaper()
    parent_fd, lease, workload_channel = None, None, None
    cleanup_confirmed, result, kill_attempts = False, None, [0]
    try:
        parent_fd = os.open("/sys/fs/cgroup", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                            | os.O_CLOEXEC | os.O_NONBLOCK)
        group_module = module("posttraining_sft_cgroup_v1")
        lease = group_module.CgroupLease(parent_fd, name=f"lumi-sft-{uuid.uuid4().hex}",
                                         limits=limits_for(case))
        original_kill = lease.kill

        def counted_kill():
            # Real lease method, counted before entry; no retry of an ambiguous write.
            require(kill_attempts[0] == 0)
            kill_attempts[0] += 1
            return original_kill()

        lease.kill = counted_kill
        lease.verify_limits()
        require(lease.snapshot()["populated"] == 0)
        workload_channel, child_channel = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        group_fd = lease.fileno()  # Borrow before fork; lease methods belong to this guardian PID.
        worker_pid = os.fork()
        if worker_pid == 0:
            try:
                workload(case, group_fd, child_channel, uid, gid, runtime, deadline_ns)
            finally:
                os._exit(2)
        child_channel.close()
        ready = receive(workload_channel, deadline_ns)
        if (set(ready) == {"phase", "component_code"}
                and ready["phase"] == "bootstrap_failed"):
            for name, error_type in (("posttraining_sft_cgroup_v1", "CgroupError"),
                                     ("posttraining_sft_cgroup_guardian_v1", "GuardianError")):
                implementation = module(name)
                if (type(ready["component_code"]) is str
                        and ready["component_code"] in implementation.ERROR_CODES):
                    raise getattr(implementation, error_type)(ready["component_code"]) from None
        require(ready == {"phase": "ready", "uid_gid_dropped": True,
                          "authority_fds_closed": True})
        require(lease.snapshot()["populated"] == lease.snapshot()["pids_current"] == 1)

        def release():
            require(workload_channel.send(b"G") == 1)
            if case == "parent_loss":
                events = [receive(workload_channel, deadline_ns),
                          receive(workload_channel, deadline_ns)]
                require({value["phase"] for value in events} == {"escaped", "double_fork"})
                require(all(value.get("session_escape", value.get("intermediate_reaped")) is True
                            for value in events))
                require(lease.snapshot()["pids_current"] == 2)
            send(channel, {"phase": "armed", "case": case})

        watched = module("posttraining_sft_cgroup_guardian_v1").watch_parent(
            lease, expected_parent_pid=controller_pid, deadline_ns=deadline_ns,
            cleanup_timeout_ms=CLEANUP_SECONDS * 1000, release=release,
        )
        require(kill_attempts[0] == 1 and watched["kill_attempted"] is True
                and watched["empty_verified"] is True)
        require(watched["parent_pidfd_bound"] is True and watched["release_called"] is True
                and watched["pidfd_closed"] is True)
        statuses = wait_statuses(time.monotonic_ns() + CLEANUP_SECONDS * 1_000_000_000)
        snapshot = lease.snapshot()  # Account exited tasks after reaping, not an earlier snapshot.
        messages = drain_messages(workload_channel)
        verify_case(case, watched, snapshot, statuses, worker_pid, messages)
        lease.close()
        cleanup_confirmed = True
        result = {"status": "passed", "case": case, "reason": watched["reason"],
                  "empty_and_removed": True, "workload_processes_reaped": len(statuses)}
    except BaseException as error:
        result = {"status": "failed", "case": case, "code": "fixture_failed",
                  "empty_and_removed": False}
        for name, error_type in (("posttraining_sft_cgroup_v1", "CgroupError"),
                                 ("posttraining_sft_cgroup_guardian_v1", "GuardianError")):
            implementation = module(name)
            if (type(error) is getattr(implementation, error_type)
                    and error.code in implementation.ERROR_CODES):
                result["component_code"] = error.code
    finally:
        if lease is not None and not cleanup_confirmed:
            try:
                if kill_attempts[0] == 0:
                    lease.kill()
                stop = time.monotonic_ns() + CLEANUP_SECONDS * 1_000_000_000
                while lease.snapshot()["populated"] and time.monotonic_ns() < stop:
                    time.sleep(0.02)
                require(lease.snapshot()["populated"] == 0)
                wait_statuses(stop)
                lease.close()
                cleanup_confirmed = True
            except BaseException:
                pass
        if workload_channel is not None:
            workload_channel.close()
        if parent_fd is not None:
            os.close(parent_fd)
        if result is None or not cleanup_confirmed:
            result = {"status": "failed", "case": case, "code": "cleanup_unconfirmed",
                      "empty_and_removed": False}
        elif result["status"] == "failed":
            result["empty_and_removed"] = True
        try:
            send(channel, result)
        finally:
            channel.close()
    os._exit(0 if result["status"] == "passed" else 2)


def controller(case, channel, outer_pid, uid, gid, runtime, deadline_ns):
    close_except({channel.fileno()})
    parent_pidfd = os.pidfd_open(outer_pid, 0)
    require(os.getppid() == outer_pid)
    own_pid = os.getpid()
    guard_pid = os.fork()
    if guard_pid == 0:
        try:
            guardian(case, channel, own_pid, uid, gid, runtime, deadline_ns)
        finally:
            os._exit(2)
    channel.close()
    try:
        # An outer SIGKILL still triggers controller loss independently of that outer.
        stop = deadline_ns + CLEANUP_SECONDS * 1_000_000_000
        while time.monotonic_ns() < stop:
            if select.select([parent_pidfd], [], [], 0.05)[0] or os.getppid() != outer_pid:
                os._exit(2)
            pid, status = os.waitpid(guard_pid, os.WNOHANG)
            if pid:
                os._exit(0 if status == 0 else 2)
        os._exit(2)  # Guardian has its own earlier absolute deadline and cleanup bound.
    finally:
        os.close(parent_pidfd)


def run_case(case, uid, gid, runtime, overall_deadline_ns):
    global CURRENT_CASE, COMPONENT_CODE, REMOVAL_REPORTED
    CURRENT_CASE = case
    COMPONENT_CODE, REMOVAL_REPORTED = "not_reported", False
    duration = 220 if case == "worker_run" else 2 if case == "deadline" else 10
    deadline_ns = min(overall_deadline_ns - 15_000_000_000,
                      time.monotonic_ns() + int(duration * 1_000_000_000))
    require(deadline_ns > time.monotonic_ns())
    channel, child_channel = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    outer_pid = os.getpid()
    controller_pid = os.fork()
    if controller_pid == 0:
        try:
            controller(case, child_channel, outer_pid, uid, gid, runtime, deadline_ns)
        finally:
            os._exit(2)
    child_channel.close()
    pidfd = os.pidfd_open(controller_pid, 0)
    signaled, final = False, None
    try:
        receive_deadline = deadline_ns + (CLEANUP_SECONDS + 2) * 1_000_000_000
        while time.monotonic_ns() < receive_deadline:
            if not select.select([channel], [], [], 0.1)[0]:
                continue
            value = receive(channel, receive_deadline)
            require(value.get("case") == case)
            if value.get("phase") == "armed":
                if case == "parent_loss":
                    require(not signaled)
                    signal.pidfd_send_signal(pidfd, signal.SIGKILL, None, 0)
                    signaled = True
                continue
            final = value
            REMOVAL_REPORTED = value.get("empty_and_removed") is True
            observed_code = value.get("component_code")
            for module_name in ("posttraining_sft_cgroup_v1",
                                "posttraining_sft_cgroup_guardian_v1"):
                if type(observed_code) is str and observed_code in module(module_name).ERROR_CODES:
                    COMPONENT_CODE = observed_code
            break
        require(final is not None and final.get("status") == "passed"
                and final.get("empty_and_removed") is True)
        statuses = wait_statuses(min(overall_deadline_ns,
                                    time.monotonic_ns() + CLEANUP_SECONDS * 1_000_000_000))
        require(controller_pid in statuses)
        if case == "parent_loss":
            require(len(statuses) == 2 and signaled and os.WIFSIGNALED(statuses[controller_pid])
                    and os.WTERMSIG(statuses[controller_pid]) == signal.SIGKILL)
            require(all(status == 0 for pid, status in statuses.items() if pid != controller_pid))
        else:
            require(len(statuses) == 1 and statuses[controller_pid] == 0)
        return final
    finally:
        # Only the owned controller pidfd is signaled, never the root cgroup or broad PID set.
        if not signaled and not select.select([pidfd], [], [], 0)[0]:
            try:
                signal.pidfd_send_signal(pidfd, signal.SIGKILL, None, 0)
            except ProcessLookupError:
                pass
        os.close(pidfd)
        channel.close()


def run():
    global STAGE
    uid, gid, bootstrap_value = bootstrap()
    deadline_ns = time.monotonic_ns() + MAXIMUM_SECONDS * 1_000_000_000
    os.umask(0o077)
    subreaper()
    STAGE = "cases"
    reports = []
    for case in CASES:
        reports.append(run_case(case, uid, gid, bootstrap_value["runtime_path"], deadline_ns))
    require("torch" not in sys.modules)
    STAGE = "complete"
    return {
        "schema_version": SCHEMA, "status": "passed", **dict.fromkeys(NONCLAIMS, False),
        "cases_verified": 7, "owned_cgroups_empty_and_removed": 7,
        "normal_completion_verified": True, "memory_oom_enforced": True,
        "pids_limit_enforced": True, "cpu_quota_throttling_verified": True,
        "deadline_cleanup_verified": True, "actual_parent_loss_cleanup_verified": True,
        "double_fork_setsid_descendants_reaped": True, "privileges_dropped_before_payload": True,
        "cgroup_pidfd_authority_closed_before_payload": True,
        "guardian_outside_workload_cgroup": True, "no_root_ml_import": True,
        "pinned_worker_run_fixture_completed": True, "worker_run_invocations_completed": 3,
        "worker_run_rank_workers_verified": 6, "worker_run_global_optimizer_steps_completed": 4,
        "worker_run_rank_optimizer_steps_completed": 8,
        "workload_processes_reaped": sum(value["workload_processes_reaped"] for value in reports),
    }


def main():
    output_fd = os.dup(1)
    with open(os.devnull, "wb") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)

    def interrupted(_signal, _frame):
        raise FixtureError("deadline_exhausted")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    signal.alarm(MAXIMUM_SECONDS)
    try:
        report, code = run(), 0
    except BaseException as error:
        cleanup_confirmed = False
        try:
            # Independent guardians remain root and own their leases while draining.
            wait_statuses(time.monotonic_ns() + 15_000_000_000)
            # Reaping alone is insufficient; require owning-guardian removal receipt too.
            cleanup_confirmed = REMOVAL_REPORTED
        except BaseException:
            pass
        report = {"schema_version": SCHEMA, "status": "failed", "stage": STAGE,
                  "case": CURRENT_CASE, "cleanup_confirmed": cleanup_confirmed,
                  "component_code": COMPONENT_CODE,
                  "code": error.code if type(error) is FixtureError else "fixture_failed",
                  **dict.fromkeys(NONCLAIMS, False)}
        code = 1
    finally:
        signal.alarm(0)
    raw = canonical(report) + b"\n"
    require(len(raw) <= MAXIMUM_MESSAGE_BYTES and os.write(output_fd, raw) == len(raw))
    os.close(output_fd)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
