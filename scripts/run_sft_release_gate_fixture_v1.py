#!/usr/bin/env python3
"""Remote Linux ordering fixture; NEVER import or execute on the local Mac.

Compose the exact accepted IPC fixture, retaining its four positive lifecycle
cases. Add two fresh blocked-WorkerRun rejection cases using the same guardian,
mount, privilege-drop and IPC path. Generated release stand-ins are NOT the
runtime executing WorkerRun. Root remains the system CPython 3.12 with -I -S -B.
All generated files have explicit identity-checked cleanup; no recursive removal.
"""

from __future__ import annotations

import errno
import hashlib
import importlib.util
import os
import select
import signal
import socket
import stat
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "scripts/run_sft_ipc_fixture_v1.py"
FROZEN_SHA256 = "88a5e703621fa796a7f110887ea360a0954a3880038f1620f4c7130634677466"
FROZEN_SIZE = 25_095
SCHEMA = "lumi.sft-release-gate-linux-fixture.v0.1"
PIN = "2c83d57f86fe861979fcc74b342e0981281ab4465437989e3eae3302fa23e112"
PAYLOADS = {
    "dependencies/declared.bin": b"GENERATED-DEPENDENCY-STAND-IN\n",
    "runtime/python3.12": b"\x7fELFGENERATED-NONEXECUTABLE-STAND-IN\n",
    "source/worker.py": b"GENERATED-SOURCE-STAND-IN-NEVER-IMPORT\n",
}
DECLARED = {
    "schema": "lumi.sft-release.v0.1", "platform": "linux-x86_64", "python_abi": "cp312",
    "interpreter": "runtime/python3.12", "entrypoint": "source/worker.py",
    "files": [
        {"path": "dependencies/declared.bin", "role": "dependency", "bytes": 30, "mode": 0o444,
         "sha256": "0d18d4480b007d544f5c9efd8120952e631bc5025850fce635aabe50dcba5a6e"},
        {"path": "runtime/python3.12", "role": "runtime", "bytes": 37, "mode": 0o555,
         "sha256": "035a243913d2821fb7343e92945a5fdb7320f48e6e4816e7ebb08c4b3afac054"},
        {"path": "source/worker.py", "role": "source", "bytes": 39, "mode": 0o444,
         "sha256": "54a3af480da559fe1421caedbd082c0904bf9e1246b108dbcb5a80c61b0c26b6"},
    ],
}
LIMITS = {
    "maximum_manifest_bytes": 600, "maximum_entries": 3, "maximum_file_bytes": 39,
    "maximum_total_bytes": 106, "maximum_tree_entries": 6,
    "maximum_content_read_bytes": 109, "maximum_opens": 64, "maximum_runtime_ms": 1000,
}
NONCLAIMS = (
    "generated_release_is_running_runtime", "runtime_origin_authenticated",
    "dynamic_dependency_completeness_verified", "point_in_time_snapshot_verified",
    "immutable_during_execution", "real_host_admission_verified",
    "real_sft_execution_permitted", "production_eligible",
)
IPC = ORIGINAL_RUN = ORIGINAL_CASE = ORIGINAL_WATCH = ORIGINAL_WAIT = None
ORIGINAL_SEND = ORIGINAL_RECEIVE = ORIGINAL_BOOTSTRAP = None
MODE, TREE, BOOTSTRAP = "positive", None, None
ACTIVE_GATE, GUARDIAN_PID, FINAL_GATE = None, None, None
RETIREMENTS, REAPED, BLOCKED_KILLED, FD_BALANCED = 0, 0, False, False
POSITIVE_GATES, NEGATIVE_GATES, TREES_REMOVED = [], [], 0
RUN_DEADLINE = 0


def base():
    return IPC.base()


def need(value):
    base().require(value)


def load_frozen():
    for path in reversed((FROZEN, *FROZEN.parents)):
        if stat.S_ISLNK(path.lstat().st_mode):
            raise RuntimeError("frozen_binding_failed")
    descriptor = os.open(FROZEN, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        raw = os.read(descriptor, FROZEN_SIZE + 1)
        after = os.fstat(descriptor)
        if (not stat.S_ISREG(before.st_mode) or before.st_size != FROZEN_SIZE
                or len(raw) != FROZEN_SIZE or os.read(descriptor, 1)
                or hashlib.sha256(raw).hexdigest() != FROZEN_SHA256
                or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                    before.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_size,
                                           after.st_mtime_ns, after.st_ctime_ns)):
            raise RuntimeError("frozen_binding_failed")
    finally:
        os.close(descriptor)
    spec = importlib.util.spec_from_file_location("_lumi_release_gate_frozen_ipc_v1", FROZEN)
    if spec is None:
        raise RuntimeError("frozen_binding_failed")
    selected = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(FROZEN), "exec"), selected.__dict__)
    return selected


def identity(path):
    info = os.lstat(path)
    return info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode), info.st_uid, info.st_gid


class GeneratedTree:
    def __init__(self):
        self.path = tempfile.mkdtemp(prefix="lumi-release-ordering-", dir="/tmp")
        self.owned = [(self.path, identity(self.path), True)]
        self.closed = False

    def build(self):
        for directory in ("dependencies", "runtime", "source"):
            path = self.path + "/" + directory
            os.mkdir(path, 0o700)
            self.owned.append((path, identity(path), True))
        for row in DECLARED["files"]:
            path = self.path + "/" + row["path"]
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                                 | os.O_CLOEXEC | os.O_NOFOLLOW, row["mode"])
            try:
                self.owned.append((path, identity(path), False))
                # Explicit chmod defeats the inherited 077 umask without relaxing
                # directory ownership. Identity is registered even if chmod fails.
                os.fchmod(descriptor, row["mode"])
                need(os.write(descriptor, PAYLOADS[row["path"]]) == row["bytes"])
            finally:
                os.close(descriptor)
        if MODE == "tamper":
            path = self.path + "/dependencies/declared.bin"
            descriptor = os.open(path, os.O_WRONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            try:
                need(os.write(descriptor, b"X" * 30) == 30)
            finally:
                os.close(descriptor)

    def close(self):
        global TREES_REMOVED
        need(not self.closed)
        self.closed = True  # Ambiguous deletion is not retried.
        failed = False
        for path, expected, directory in reversed(self.owned):
            try:
                need(identity(path) == expected)
                if directory:
                    os.rmdir(path)
                else:
                    os.unlink(path)
            except BaseException:
                failed = True
        if failed:
            raise base().FixtureError("cleanup_unconfirmed") from None
        TREES_REMOVED += 1


def descriptor_snapshot():
    values = os.listdir("/proc/self/fd")
    need(len(values) <= 1024)
    result = {}
    for text in values:
        need(text.isdecimal())
        descriptor = int(text)
        try:
            info = os.fstat(descriptor)
        except OSError as error:
            need(error.errno == errno.EBADF)  # listdir's own transient descriptor.
        else:
            result[descriptor] = info.st_dev, info.st_ino, info.st_mode
    return result


def watch_parent(lease, *, expected_parent_pid, deadline_ns, cleanup_timeout_ms, release):
    global ACTIVE_GATE, GUARDIAN_PID, FD_BALANCED, RETIREMENTS
    GUARDIAN_PID = os.getpid()
    gate = base().module("posttraining_sft_release_gate_v1")
    ACTIVE_GATE = gate.VerifiedRelease(
        root=TREE.path, raw_manifest=base().canonical(DECLARED),
        expected_sha256="0" * 64 if MODE == "wrong_pin" else PIN,
        limits=LIMITS, expected_uid=0, expected_gid=0,
        expected_parent_pid=expected_parent_pid, deadline_ns=deadline_ns, release=release,
    )
    original_kill = lease.kill

    def retire_once():
        global RETIREMENTS
        RETIREMENTS += 1
        need(RETIREMENTS == 1)
        return original_kill()

    before = descriptor_snapshot()
    lease.kill = retire_once
    try:
        return ORIGINAL_WATCH(
            lease, expected_parent_pid=expected_parent_pid, deadline_ns=deadline_ns,
            cleanup_timeout_ms=cleanup_timeout_ms, release=ACTIVE_GATE,
        )
    finally:
        lease.kill = original_kill
        FD_BALANCED = descriptor_snapshot() == before
        need(FD_BALANCED)


def wait_statuses(deadline_ns):
    global REAPED, BLOCKED_KILLED
    result = ORIGINAL_WAIT(deadline_ns)
    if os.getpid() == GUARDIAN_PID and MODE != "positive":
        need(REAPED == 0 and len(result) == 1)
        status = next(iter(result.values()))
        need(os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGKILL)
        REAPED, BLOCKED_KILLED = 1, True
    return result


def send(channel, value):
    if "empty_and_removed" in value and ACTIVE_GATE is not None:
        summary = ACTIVE_GATE.status()
        value = dict(value, release_gate={
            "gate": summary, "retirements": RETIREMENTS, "descriptor_balance": FD_BALANCED,
            "blocked_worker_reaped": REAPED, "blocked_worker_sigkill": BLOCKED_KILLED,
        })
    return ORIGINAL_SEND(channel, value)


def receive(channel, deadline_ns):
    global FINAL_GATE
    value = ORIGINAL_RECEIVE(channel, deadline_ns)
    if "empty_and_removed" in value:
        need(type(value.get("release_gate")) is dict or value.get("status") == "failed")
        FINAL_GATE = value.get("release_gate")
        value = {key: item for key, item in value.items() if key != "release_gate"}
    return value


def validate_gate(value, *, negative):
    need(type(value) is dict and set(value) == {
        "gate", "retirements", "descriptor_balance", "blocked_worker_reaped",
        "blocked_worker_sigkill",
    })
    need(value["retirements"] == 1 and value["descriptor_balance"] is True)
    summary = value["gate"]
    need(summary["schema"] == "lumi.sft-release-gate.v0.1")
    need(summary["real_sft_execution_permitted"] is False
         and summary["production_eligible"] is False)
    expected = ((1, 0, 0, 0) if negative else (1, 1, 1, 1))
    for field, count in zip(("verification_attempts", "verification_completed",
                              "release_attempts", "release_completed"), expected, strict=True):
        need(type(summary[field]) is int and summary[field] == count)
    need(summary["state"] == ("failed" if negative else "released"))
    need(summary["code"] == ({"wrong_pin": "manifest_hash_mismatch", "tamper": "digest_mismatch"}[
        MODE] if negative else "completed"))
    need(value["blocked_worker_reaped"] == (1 if negative else 0)
         and value["blocked_worker_sigkill"] is negative)


def negative_case(case, uid, gid, runtime, overall_deadline_ns):
    """Expect a failed guardian result, but require actual removal and owned reaps.

    Unlike the old positive run_case, this does not reject the expected failure
    before reaping its controller. The old guardian/worker bodies are unchanged.
    """
    need(case == "worker_run" and MODE in {"wrong_pin", "tamper"})
    deadline_ns = min(overall_deadline_ns - 15_000_000_000,
                      time.monotonic_ns() + 15_000_000_000)
    need(deadline_ns > time.monotonic_ns())
    base().CURRENT_CASE, base().REMOVAL_REPORTED = case, False
    channel, child_channel = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    outer_pid = os.getpid()
    controller_pid = os.fork()
    if controller_pid == 0:
        try:
            base().controller(case, child_channel, outer_pid, uid, gid, runtime, deadline_ns)
        finally:
            os._exit(2)
    child_channel.close()
    pidfd = os.pidfd_open(controller_pid, 0)
    reaped = False
    try:
        final = base().receive(channel, deadline_ns + 12_000_000_000)
        base().REMOVAL_REPORTED = final.get("empty_and_removed") is True
        need(final.get("status") == "failed" and final.get("case") == "worker_run"
             and final.get("code") == "fixture_failed"
             and final.get("component_code") == "guardian_failed"
             and base().REMOVAL_REPORTED)
        validate_gate(FINAL_GATE, negative=True)
        statuses = base().wait_statuses(min(overall_deadline_ns,
                                            time.monotonic_ns() + 10_000_000_000))
        need(set(statuses) == {controller_pid} and os.WIFEXITED(statuses[controller_pid])
             and os.WEXITSTATUS(statuses[controller_pid]) == 2)
        reaped = True
        return final
    finally:
        if not reaped and not select.select([pidfd], [], [], 0)[0]:
            try:
                signal.pidfd_send_signal(pidfd, signal.SIGKILL, None, 0)
            except ProcessLookupError:
                pass
        os.close(pidfd)
        channel.close()


def run_case(case, uid, gid, runtime, deadline_ns):
    global TREE, FINAL_GATE, ACTIVE_GATE, GUARDIAN_PID
    global RETIREMENTS, REAPED, BLOCKED_KILLED, FD_BALANCED
    FINAL_GATE = ACTIVE_GATE = GUARDIAN_PID = None
    RETIREMENTS, REAPED, BLOCKED_KILLED, FD_BALANCED = 0, 0, False, False
    TREE = GeneratedTree()
    try:
        TREE.build()
        if MODE == "positive":
            result = ORIGINAL_CASE(case, uid, gid, runtime, deadline_ns)
            validate_gate(FINAL_GATE, negative=False)
            POSITIVE_GATES.append(FINAL_GATE)
        else:
            result = negative_case(case, uid, gid, runtime, deadline_ns)
            NEGATIVE_GATES.append(FINAL_GATE)
        return result
    except BaseException:
        if type(FINAL_GATE) is dict and type(FINAL_GATE.get("gate")) is dict:
            code = FINAL_GATE["gate"].get("code")
            if (type(code) is str
                    and code in base().module("posttraining_sft_release_gate_v1").ERROR_CODES):
                base().COMPONENT_CODE = "release_gate_" + code
        raise
    finally:
        confirmed = base().REMOVAL_REPORTED
        base().REMOVAL_REPORTED = False
        TREE.close()
        TREE = None
        base().REMOVAL_REPORTED = confirmed


def bootstrap():
    global BOOTSTRAP
    BOOTSTRAP = ORIGINAL_BOOTSTRAP()
    return BOOTSTRAP


def run():
    global ORIGINAL_CASE, ORIGINAL_WATCH, ORIGINAL_WAIT, ORIGINAL_BOOTSTRAP, MODE, RUN_DEADLINE
    # Actual system interpreter preflight, before bootstrap creates any resources.
    base().module("posttraining_sft_release_verifier_v1")._runtime()
    need(sys.flags.isolated == 1 and os.getuid() == os.getgid() == 0)
    need(hashlib.sha256(base().canonical(DECLARED)).hexdigest() == PIN)
    guardian = base().module("posttraining_sft_cgroup_guardian_v1")
    ORIGINAL_CASE, ORIGINAL_WATCH = base().run_case, guardian.watch_parent
    ORIGINAL_WAIT, ORIGINAL_BOOTSTRAP = base().wait_statuses, base().bootstrap
    base().run_case, guardian.watch_parent = run_case, watch_parent
    base().wait_statuses, base().bootstrap = wait_statuses, bootstrap
    RUN_DEADLINE = time.monotonic_ns() + base().MAXIMUM_SECONDS * 1_000_000_000
    report = ORIGINAL_RUN()  # Exact accepted IPC run still requires four successes.
    need(len(POSITIVE_GATES) == IPC.INSTALLATIONS_VERIFIED == TREES_REMOVED == 4)
    uid, gid, binding = BOOTSTRAP
    net = IPC.FS.NET
    # Original host sockets have retired; this new exact owned socket supplies
    # only the inherited-authority sanitization control for the blocked cases.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sentinel:
        sentinel.bind(("127.0.0.1", 0))
        sentinel.listen(1)
        net.HOST_SOCKET_FD = sentinel.fileno()
        net.HOST_SOCKET_IDENTITY = net.socket_identity(sentinel.fileno())
        for selected in ("wrong_pin", "tamper"):
            MODE = selected
            base().run_case("worker_run", uid, gid, binding["runtime_path"], RUN_DEADLINE)
            need(net.socket_identity(sentinel.fileno()) == net.HOST_SOCKET_IDENTITY)
    need(len(NEGATIVE_GATES) == 2 and IPC.INSTALLATIONS_VERIFIED == TREES_REMOVED == 6)
    need("torch" not in sys.modules)
    # A compact public projection keeps the inherited 4096-byte report limit.
    # Every original positive case has already completed all of its assertions.
    selected_fields = (
        "cases_verified", "owned_cgroups_empty_and_removed", "owned_mountpoints_removed",
        "seccomp_installations_verified", "visible_same_uid_sentinel_baselines_verified",
        "owned_visible_unix_sockets_removed", "pinned_worker_run_fixture_completed",
        "worker_run_invocations_completed", "worker_run_rank_workers_verified",
        "worker_run_global_optimizer_steps_completed", "worker_run_rank_optimizer_steps_completed",
        "preserved_ip_filesystem_checks_verified", "deadline_cleanup_verified",
        "actual_parent_loss_cleanup_verified", "double_fork_setsid_descendants_reaped",
        "privileges_dropped_before_payload", "cgroup_pidfd_authority_closed_before_payload",
        "no_root_ml_import",
    )
    report = {key: report[key] for key in selected_fields} | {
        "schema_version": SCHEMA, "status": "passed",
        **dict.fromkeys(IPC.NONCLAIMS, False),
    }
    report.update(
        native_release_gate_context_verified=True, release_before_worker_order_verified=True,
        positive_release_gates_verified=4, rejected_release_gates_verified=2,
        rejected_release_calls=0, rejected_worker_run_invocations=0,
        wrong_pin_rejected=True, same_size_tamper_rejected=True,
        release_gate_total_cgroups_removed=6, release_gate_total_trees_removed=6,
        rejected_blocked_workers_reaped=2, rejected_controllers_reaped=2,
        rejected_cases_seccomp_installations=2, rejected_cases_visible_socket_removals=4,
        release_gate_descriptor_balance_verified=True,
    )
    need(len(base().canonical(report)) + 1 <= base().MAXIMUM_MESSAGE_BYTES)
    return report


def main():
    global IPC, ORIGINAL_RUN, ORIGINAL_SEND, ORIGINAL_RECEIVE
    try:
        if (sys.platform != "linux" or sys.version_info[:2] != (3, 12)
                or not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode
                or os.getuid() != 0):
            raise RuntimeError("runtime_rejected")
        IPC = load_frozen()
    except BaseException:
        os.write(1, b'{"schema_version":"lumi.sft-release-gate-linux-fixture.v0.1",'
                    b'"status":"failed","code":"preflight_rejected"}\n')
        return 1
    ORIGINAL_RUN, ORIGINAL_SEND, ORIGINAL_RECEIVE = IPC.run, IPC.send, IPC.receive
    IPC.run, IPC.send, IPC.receive = run, send, receive
    IPC.SCHEMA = SCHEMA
    IPC.NONCLAIMS = tuple(dict.fromkeys((*IPC.NONCLAIMS, *NONCLAIMS)))
    return IPC.main()


if __name__ == "__main__":
    raise SystemExit(main())
