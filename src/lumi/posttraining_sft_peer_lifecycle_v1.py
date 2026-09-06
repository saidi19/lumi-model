"""Owned pidfd lifecycle and capability snapshots for ONE fixed inert child.

Trusted CPython 3.12/3.13, genuine procfs, one controller thread and no foreign
reaper/native signal mutation are prerequisites. This is not an arbitrary-worker
sandbox: namespace/ptrace/cgroup/network containment and ledger admission remain
absent. Snapshots never establish continuous capability exclusion.
"""

from __future__ import annotations

import hashlib
import math
import os
import select
import signal
import struct
import subprocess
import sys
import sysconfig
import threading
import time
from pathlib import Path

from . import posttraining_sft_child_status_v1 as policy
from . import posttraining_sft_peer_credentials_v1 as credentials
from . import posttraining_sft_peer_identity_probe_v1 as previous

WORKER = Path(__file__).resolve().parents[2] / "scripts/sft_peer_lifecycle_inert_worker_v1.py"
WORKER_SHA256 = "e21a32c4cb0b24aa3f4af45878e0be4a6ecc053052fb39dfd956efb5881709da"
MAXIMUM_SECONDS = 6
CLEANUP_SECONDS = 3
STATUS_CAPACITY = policy.MAXIMUM_STATUS_BYTES + 1
FAULTS = frozenset({"none", "wrong_parent", "exit_after_ready", "stall"})
READY = b"lifecycle-ready-v1"
PAYLOAD = b"lifecycle-inert-message-v1"
ERROR_CODES = frozenset({
    "lifecycle_unavailable", "lifecycle_runtime_rejected", "lifecycle_configuration_rejected",
    "lifecycle_signal_policy_rejected", "lifecycle_source_rejected", "lifecycle_failed",
    "lifecycle_deadline_exhausted", "lifecycle_budget_exhausted", "lifecycle_child_exited",
    "lifecycle_identity_rejected", "lifecycle_frame_rejected", "lifecycle_exit_rejected",
    "lifecycle_cleanup_failed",
})


class LifecycleError(ValueError):
    def __init__(self, code):
        self.code = code if type(code) is str and code in ERROR_CODES else "lifecycle_failed"
        super().__init__(self.code)


def need(condition, code):
    if not condition:
        raise LifecycleError(code) from None


def remaining(deadline):
    value = deadline - time.monotonic()
    need(math.isfinite(value) and value > 0, "lifecycle_deadline_exhausted")
    return value


def require_runtime():
    previous.require_stdlib()
    credentials.require_linux()
    need(
        sys.implementation.name == "cpython" and sys.version_info[:2] in {(3, 12), (3, 13)}
        and sysconfig.get_config_var("HAVE_SIGACTION") == 1
        and all(hasattr(os, name) for name in (
            "pidfd_open", "P_PIDFD", "waitid", "WEXITED", "WNOHANG", "WNOWAIT",
            "CLD_EXITED", "CLD_KILLED", "CLD_DUMPED", "getresuid", "getresgid",
            "O_CLOEXEC", "O_NOFOLLOW",
        ))
        and hasattr(signal, "pidfd_send_signal") and hasattr(select, "poll"),
        "lifecycle_unavailable",
    )
    need(
        threading.current_thread() is threading.main_thread() and threading.active_count() == 1
        and os.getresuid() == (os.getuid(),) * 3 and os.getuid() > 0
        and os.getresgid() == (os.getgid(),) * 3 and os.getgid() > 0,
        "lifecycle_runtime_rejected",
    )


def normalize_child_signal():
    """CPython's supported sigaction setter clears SA_NOCLDWAIT before spawning.

    This is normalization plus handler metadata readback, not a native flags
    query. No other thread/native code may change dispositions or reap children.
    """
    need(signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL,
         "lifecycle_signal_policy_rejected")
    old = signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    need(old == signal.SIG_DFL and signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL,
         "lifecycle_signal_policy_rejected")


def authenticated_source():
    with WORKER.open("rb") as stream:
        source = stream.read(8193)
    need(len(source) <= 8192 and hashlib.sha256(source).hexdigest() == WORKER_SHA256,
         "lifecycle_source_rejected")
    return source


def spawn(endpoint, fault, source):
    need(type(source) is bytes and len(source) <= 8192
         and hashlib.sha256(source).hexdigest() == WORKER_SHA256, "lifecycle_source_rejected")
    need(type(fault) is str and fault in FAULTS, "lifecycle_configuration_rejected")
    return subprocess.Popen(
        [sys.executable, "-S", "-B", "-c", source.decode("utf-8"),
         str(endpoint.fileno()), fault],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        close_fds=True, pass_fds=(endpoint.fileno(),),
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "PYTHONHASHSEED": "0"},
    )


class OwnedChild:
    """Single owner; no Popen poll/wait until after failed binding cleanup only."""

    def __init__(self, child):
        self.child = child
        self.fd = None
        self.owned = False
        self.bound_live = False
        self.reaped = False
        self.close_attempted = False
        self.closed = False
        self.signal_attempts = 0
        self.waitid_calls = 0

    def evidence(self, *, consume=False):
        self.waitid_calls += 1
        flags = os.WEXITED | os.WNOHANG | (0 if consume else os.WNOWAIT)
        result = os.waitid(os.P_PIDFD, self.fd, flags)
        if result is not None:
            need(result.si_pid == self.child.pid and result.si_uid == os.getuid()
                 and result.si_signo == signal.SIGCHLD
                 and result.si_code in (os.CLD_EXITED, os.CLD_KILLED, os.CLD_DUMPED),
                 "lifecycle_identity_rejected")
        return result

    def bind(self):
        need(self.fd is None and not self.reaped, "lifecycle_identity_rejected")
        self.fd = os.pidfd_open(self.child.pid, 0)
        need(type(self.fd) is int and self.fd >= 0 and not os.get_inheritable(self.fd),
             "lifecycle_identity_rejected")
        result = self.evidence()
        # Successful waitid establishes child ownership even if it already exited.
        self.owned = True
        need(result is None, "lifecycle_child_exited")
        self.assert_live()
        self.bound_live = True

    def poll(self, milliseconds):
        poller = select.poll()
        poller.register(self.fd, select.POLLIN | select.POLLHUP | select.POLLERR)
        events = poller.poll(milliseconds)
        need(not events or (len(events) == 1 and events[0][0] == self.fd
                            and events[0][1] == select.POLLIN), "lifecycle_identity_rejected")
        return bool(events)

    def assert_live(self):
        need(self.owned and not self.reaped and not self.close_attempted,
             "lifecycle_identity_rejected")
        need(self.evidence() is None and not self.poll(0), "lifecycle_child_exited")

    def reap(self, deadline, *, require_success):
        need(self.owned and not self.reaped and not self.close_attempted,
             "lifecycle_identity_rejected")
        need(self.poll(math.ceil(1000 * remaining(deadline))), "lifecycle_deadline_exhausted")
        observed = self.evidence()
        need(observed is not None, "lifecycle_exit_rejected")
        actual = self.evidence(consume=True)
        need(actual is not None and actual == observed, "lifecycle_exit_rejected")
        code = actual.si_status if actual.si_code == os.CLD_EXITED else -actual.si_status
        self.child.returncode = code  # Real kernel reap; do not call Popen.wait again.
        self.reaped = True
        if require_success:
            need(actual.si_code == os.CLD_EXITED and code == 0, "lifecycle_exit_rejected")

    def terminate_and_reap(self, deadline):
        if self.reaped:
            return
        need(self.owned and not self.close_attempted, "lifecycle_identity_rejected")
        if self.evidence() is None:
            need(self.signal_attempts == 0, "lifecycle_cleanup_failed")
            self.signal_attempts += 1
            try:
                signal.pidfd_send_signal(self.fd, signal.SIGKILL, None, 0)
            except ProcessLookupError:
                # A concurrent exit is acceptable only with real exit/reap evidence below.
                pass
        self.reap(deadline, require_success=False)

    def close(self):
        if self.fd is None:
            self.closed = True
            return
        need(not self.close_attempted, "lifecycle_cleanup_failed")
        self.close_attempted = True
        os.close(self.fd)
        self.closed = True


class StatusReads:
    """One controller plus two owned-child snapshots; capacity prepaid, no refund."""

    def __init__(self):
        self.attempts = 0
        self.capacity_bytes = 0
        self.closed = 0
        self.cleanup_failed = False

    def read(self, path):
        need(self.attempts < 3, "lifecycle_budget_exhausted")
        self.attempts += 1
        self.capacity_bytes += STATUS_CAPACITY
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        read_failed = False
        raw = b""
        try:
            raw = os.read(descriptor, STATUS_CAPACITY)
        except BaseException:
            read_failed = True
        try:
            os.close(descriptor)
        except BaseException:
            self.cleanup_failed = True
        else:
            self.closed += 1
        need(not self.cleanup_failed, "lifecycle_cleanup_failed")
        need(not read_failed, "lifecycle_failed")
        return raw

    def controller(self):
        return policy.parse_controller_status(
            self.read("/proc/self/status"), expected_pid=os.getpid(),
            expected_ppid=os.getppid(), expected_uid=os.getuid(), expected_gid=os.getgid(),
        )

    def child(self, owner, deadline, *, wrong_parent=False):
        remaining(deadline)
        owner.assert_live()
        raw = self.read(f"/proc/{owner.child.pid}/status")
        owner.assert_live()
        remaining(deadline)
        return policy.parse_status(
            raw, expected_pid=owner.child.pid, expected_ppid=os.getpid() + int(wrong_parent),
            expected_uid=os.getuid(), expected_gid=os.getgid(),
        )


def cleanup(owner, channel, sockets):
    clean = True
    # Close IPC first, allowing the fixed child to exit even if pidfd binding failed.
    for handle in ([channel] if channel is not None else []) + sockets:
        try:
            handle.close()
        except BaseException:
            clean = False
    if owner is not None:
        try:
            if owner.owned:
                owner.terminate_and_reap(time.monotonic() + CLEANUP_SECONDS)
            else:
                # Best effort only; this API can mask ECHILD and never proves a reap.
                clean = False
                owner.child.wait(timeout=CLEANUP_SECONDS)
        except BaseException:
            clean = False
        try:
            owner.close()
        except BaseException:
            clean = False
    return clean


def run_lifecycle_probe(*, fault="none"):
    """Fixed trusted child only; every error denies and no started probe is retried."""
    report = {
        **previous.initial_report(), "schema_version": "lumi.sft-peer-lifecycle-probe.v0.1",
        "lifecycle_probe_passed": False, "controller_status_verified": False,
        "capability_snapshots_verified": 0, "continuous_capability_exclusion": False,
        "capability_bounding_set_empty_snapshots": 0,
        "namespace_containment": False, "ptrace_containment": False,
        "no_numeric_pid_signals": True, "pidfd_closed": False,
    }
    deadline = time.monotonic() + (1 if fault == "stall" else MAXIMUM_SECONDS)
    owner = channel = None
    sockets = []
    reads = StatusReads()
    failure = None
    accepted = False
    try:
        need(type(fault) is str and fault in FAULTS, "lifecycle_configuration_rejected")
        require_runtime()
        reads.controller()
        report["controller_status_verified"] = True
        normalize_child_signal()
        source = authenticated_source()
        report["worker_source_authenticated"] = True
        remaining(deadline)
        parent, endpoint = credentials.credential_pair()
        sockets.extend((parent, endpoint))
        owner = OwnedChild(spawn(endpoint, fault, source))
        report["worker_processes_started"] = 1
        owner.bind()  # No numeric PID read, signal or reaping fallback before this.
        sockets.remove(endpoint)
        try:
            endpoint.close()
        except BaseException:
            raise LifecycleError("lifecycle_cleanup_failed") from None
        channel = credentials.CredentialChannel(
            parent, expected_credentials=(owner.child.pid, os.getuid(), os.getgid()),
            deadline=deadline, maximum_messages=2,
            maximum_read_bytes=2 * credentials.RECEIVE_CAPACITY_BYTES,
        )
        sockets.remove(parent)
        for expected, ack in ((READY, b"G"), (PAYLOAD, b"K")):
            remaining(deadline)
            owner.assert_live()
            packet = channel.receive()
            owner.assert_live()
            need(packet == struct.pack("!I", len(expected)) + expected,
                 "lifecycle_frame_rejected")
            snapshot = reads.child(owner, deadline, wrong_parent=fault == "wrong_parent")
            report["capability_snapshots_verified"] += 1
            report["capability_bounding_set_empty_snapshots"] += int(
                snapshot["capability_bounding_set_empty"]
            )
            owner.assert_live()
            parent.settimeout(remaining(deadline))
            need(parent.send(ack) == 1, "lifecycle_frame_rejected")
        owner.reap(deadline, require_success=True)
        accepted = True
    except (LifecycleError, credentials.PeerCredentialError, policy.StatusError,
            previous.ProbeError) as error:
        failure = error.code
    except BaseException:
        failure = "lifecycle_failed"
    clean = cleanup(owner, channel, sockets)
    clean = clean and not reads.cleanup_failed and failure not in {
        "lifecycle_cleanup_failed", "credential_cleanup_failed",
    }
    # Failed status opens count as attempts, not owned descriptors needing a close.
    report["status_read_attempts"] = reads.attempts
    report["status_read_capacity_bytes"] = reads.capacity_bytes
    report["status_descriptors_closed"] = reads.closed
    report["worker_cleanup_complete"] = clean
    if owner is not None:
        report["process_generation_bound"] = owner.bound_live
        report["worker_processes_reaped"] = int(owner.reaped)
        report["pidfd_closed"] = owner.closed
        report["pidfd_signal_attempts"] = owner.signal_attempts
        report["pidfd_waitid_calls"] = owner.waitid_calls
    if channel is not None:
        report["credential_accounting"] = channel.summary()
    if not clean:
        failure = "lifecycle_cleanup_failed"
    elif time.monotonic() >= deadline:
        failure = "lifecycle_deadline_exhausted"
    if failure is None and accepted and clean:
        report.update({
            "state": "passed", "lifecycle_probe_passed": True,
            "linux_peer_credential_probe_passed": True,
            "kernel_pid_tuple_matches_expected_child": True,
        })
    elif failure in {"linux_credentials_unavailable", "lifecycle_unavailable"}:
        report["state"] = "unsupported"
    elif failure not in {"lifecycle_failed", "lifecycle_cleanup_failed"}:
        report["state"] = "rejected"
    report["failure_code"] = failure
    return report
