"""One hash-pinned inert child exercises a fixed Linux kernel policy.

This is NOT a real-worker sandbox, cgroup controller or training admission path.
Only the new child changes its resource limits/seccomp state. The parent retains
the existing owned-pidfd, per-message credentials and capability-snapshot rules.
No model, private input, GPU, external connection or production action is used.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from . import posttraining_sft_kernel_policy_v1 as kernel_policy
from . import posttraining_sft_peer_lifecycle_v1 as lifecycle

credentials = lifecycle.credentials
ROOT = Path(__file__).resolve().parents[2]
POLICY_SHA256 = "33bf272bd1b97f3ce74a9ecec528c88adce972dd8f31873317f3fe912f2c8c85"
WORKER_SHA256 = "3fe2376fd09ce7f30fdd141f90a8d2e973285375d6e89a762fb50610f6a0855b"
SOURCE_BINDINGS = (
    ("src/lumi/posttraining_sft_kernel_policy_v1.py", 32768, POLICY_SHA256),
    ("scripts/sft_kernel_guard_inert_worker_v1.py", 16384, WORKER_SHA256),
)
READY = b"kernel-guard-ready-v1"
PROOF = b"kernel-guard-proof-v1"
MAXIMUM_SECONDS = 6
FAULTS = frozenset({"none", "exit_after_ready", "stall"})
EXPECTED_DENIAL_CHECKS = 17
ERROR_CODES = frozenset({
    "guard_configuration_rejected", "guard_source_rejected", "guard_snapshot_rejected",
    "guard_frame_rejected", "guard_terminal_rejected", "guard_cleanup_failed", "guard_failed",
}) | (
    lifecycle.ERROR_CODES | credentials.ERROR_CODES | lifecycle.policy.ERROR_CODES
    | kernel_policy.ERROR_CODES
)


class GuardError(ValueError):
    def __init__(self, code):
        self.code = code if type(code) is str and code in ERROR_CODES else "guard_failed"
        super().__init__(self.code)


def need(condition, code):
    if not condition:
        raise GuardError(code) from None


def parse_guard_status(raw: bytes, **expected) -> dict:
    """Keep the old full policy, then require filter mode and a positive filter count.

    These procfs fields do not identify a particular filter. Only together with
    authenticated fixed source, successful own installation and the bounded
    syscall challenges do they support this inert-probe result, not attestation.
    """
    report = lifecycle.policy.parse_status(raw, **expected)
    fields = dict(line.split(b":", 1) for line in raw.splitlines())
    mode = fields.get(b"Seccomp", b"").strip(b" \t")
    count = fields.get(b"Seccomp_filters", b"").strip(b" \t")
    need(mode == b"2" and 1 <= len(count) <= 2 and count.isdigit()
         and not count.startswith(b"0") and 1 <= int(count) <= 64,
         "guard_snapshot_rejected")
    return {**report, "filter_mode_snapshot": True, "filter_count_snapshot": int(count)}


class GuardReads(lifecycle.StatusReads):
    """Unchanged three-read prepaid capacity, same no-follow/one-close I/O helper."""

    def child(self, owner, deadline):
        lifecycle.remaining(deadline)
        owner.assert_live()
        raw = self.read(f"/proc/{owner.child.pid}/status")
        owner.assert_live()
        lifecycle.remaining(deadline)
        return parse_guard_status(
            raw, expected_pid=owner.child.pid, expected_ppid=os.getpid(),
            expected_uid=os.getuid(), expected_gid=os.getgid(),
        )


def authenticate_sources() -> tuple[bytes, bytes]:
    sources = []
    for relative, capacity, expected in SOURCE_BINDINGS:
        with (ROOT / relative).open("rb") as stream:
            raw = stream.read(capacity + 1)
        need(len(raw) <= capacity
             and hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected),
             "guard_source_rejected")
        sources.append(raw)
    return tuple(sources)


def spawn(endpoint, fault, sources):
    """Separate namespaces prevent policy/worker global-name collisions.

    Compile only the authenticated public bytes; do not reopen their paths or
    accept a caller-supplied command. Limits/filter are installed only in this
    new subprocess, after its fixed imports and inherited endpoint setup.
    """
    need(type(fault) is str and fault in FAULTS and type(sources) is tuple and len(sources) == 2,
         "guard_configuration_rejected")
    for raw, (_path, capacity, expected) in zip(sources, SOURCE_BINDINGS, strict=True):
        need(type(raw) is bytes and len(raw) <= capacity
             and hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected),
             "guard_source_rejected")
    code = (
        "import types\n"
        "policy = types.ModuleType('_fixed_inert_kernel_policy')\n"
        f"exec(compile({sources[0]!r}, '<fixed-policy>', 'exec'), policy.__dict__)\n"
        f"exec(compile({sources[1]!r}, '<fixed-worker>', 'exec'), "
        "{'__name__': '__main__', 'kernel_policy': policy})\n"
    )
    return subprocess.Popen(
        [sys.executable, "-S", "-B", "-c", code, str(endpoint.fileno()), fault],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        close_fds=True, pass_fds=(endpoint.fileno(),), start_new_session=True,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
    )


def run_kernel_guard_probe(*, fault: str = "none") -> dict:
    """One new inert child or fail closed; never retry and never launch training.

    Six seconds includes successful cleanup; a stalled fixture gets one second.
    Failure retirement has the existing three-second grace. Blocking OS calls,
    interpreter overhead, total host memory and provider billing are not bounded.
    """
    report = {
        "schema_version": "lumi.sft-kernel-guard-inert-probe.v0.1",
        "state": "failed", "failure_code": None,
        "kernel_guard_probe_passed": False, "worker_sources_authenticated": False,
        "controller_status_verified": False, "worker_processes_started": 0,
        "worker_processes_reaped": 0, "worker_cleanup_complete": False,
        "pidfd_bound_live": False, "pidfd_closed": False, "pidfd_signal_attempts": 0,
        "filter_snapshots_verified": 0, "terminal_receive_attempts": 0,
        "terminal_eof_verified": False, "terminal_descriptors_closed": 0,
        "parent_send_bytes_precharged": 0, "denied_syscall_challenges_passed": 0,
        "address_space_rejection_challenges_passed": 0,
        "positive_anonymous_mapping_challenges_passed": 0,
        "inert_child_hard_limits_readback_verified": False,
        "inert_child_filter_install_and_challenges_verified": False,
        "no_numeric_pid_signals": True,
        "cgroup_memory_enforcement": False, "aggregate_descendant_enforcement": False,
        "os_network_enforcement": False, "namespace_containment": False,
        "ptrace_containment": False, "continuous_privilege_exclusion": False,
        "durable_ledger_integrated": False, "real_worker_sandbox_verified": False,
        "target_gpu_host_verified": False, "guarded_real_worker_ready": False,
        "execution_authorized": False, "production_eligible": False,
        "full_private_integration_passed": False,
        "training_source_or_data_reads": 0, "model_or_checkpoint_reads": 0,
        "private_artifact_writes": 0, "external_network_bytes": 0,
        "gpu_use": 0, "inference_calls": 0,
    }
    owner = channel = None
    sockets = []
    reads = GuardReads()
    accepted = False
    failure = None
    deadline = time.monotonic() + (1 if fault == "stall" else MAXIMUM_SECONDS)
    try:
        need(type(fault) is str and fault in FAULTS, "guard_configuration_rejected")
        lifecycle.require_runtime()
        kernel_policy.require_abi()  # Read-only; install() is NEVER called in the parent.
        reads.controller()
        report["controller_status_verified"] = True
        lifecycle.normalize_child_signal()
        sources = authenticate_sources()
        report["worker_sources_authenticated"] = True
        lifecycle.remaining(deadline)
        parent, endpoint = credentials.credential_pair()
        sockets.extend((parent, endpoint))
        owner = lifecycle.OwnedChild(spawn(endpoint, fault, sources))
        report["worker_processes_started"] = 1
        owner.bind()
        sockets.remove(endpoint)
        try:
            endpoint.close()
        except BaseException:
            raise GuardError("guard_cleanup_failed") from None
        channel = credentials.CredentialChannel(
            parent, expected_credentials=(owner.child.pid, os.getuid(), os.getgid()),
            deadline=deadline, maximum_messages=2,
            maximum_read_bytes=2 * credentials.RECEIVE_CAPACITY_BYTES,
        )
        sockets.remove(parent)
        for expected, acknowledgement in ((READY, b"G"), (PROOF, b"K")):
            owner.assert_live()
            raw = channel.receive()
            owner.assert_live()
            need(raw == len(expected).to_bytes(4, "big") + expected, "guard_frame_rejected")
            reads.child(owner, deadline)
            report["filter_snapshots_verified"] += 1
            owner.assert_live()
            frame = len(acknowledgement).to_bytes(4, "big") + acknowledgement
            report["parent_send_bytes_precharged"] += len(frame)
            parent.settimeout(lifecycle.remaining(deadline))
            need(parent.sendmsg([frame]) == len(frame), "guard_frame_rejected")
            lifecycle.remaining(deadline)
        owner.reap(deadline, require_success=True)
        # Real reap precedes this one-shot EOF check. Empty queued credential
        # packets are not EOF; close every delivered right before any rejection.
        report["terminal_receive_attempts"] += 1
        parent.settimeout(lifecycle.remaining(deadline))
        raw, ancillary, flags, _address = parent.recvmsg(
            credentials.RECEIVE_DATA_CAPACITY, credentials.ANCILLARY_BUFFER_BYTES,
            socket.MSG_CMSG_CLOEXEC,
        )
        closed, failed = credentials._close_received_rights(ancillary)
        report["terminal_descriptors_closed"] += closed
        need(not failed, "guard_cleanup_failed")
        lifecycle.remaining(deadline)
        need(raw == b"" and ancillary == [] and type(flags) is int
             and flags & ~socket.MSG_CMSG_CLOEXEC == 0, "guard_terminal_rejected")
        report["terminal_eof_verified"] = True
        accepted = True
    except BaseException as error:
        code = getattr(error, "code", None)
        failure = code if type(code) is str and code in ERROR_CODES else "guard_failed"
    clean = lifecycle.cleanup(owner, channel, sockets)
    clean &= not reads.cleanup_failed and failure not in {
        "guard_cleanup_failed", "lifecycle_cleanup_failed", "credential_cleanup_failed",
    }
    report["worker_cleanup_complete"] = clean
    report["status_read_attempts"] = reads.attempts
    report["status_read_capacity_bytes"] = reads.capacity_bytes
    report["status_descriptors_closed"] = reads.closed
    if owner is not None:
        report["pidfd_bound_live"] = owner.bound_live
        report["pidfd_closed"] = owner.fd is not None and owner.closed
        report["worker_processes_reaped"] = int(owner.reaped)
        report["pidfd_signal_attempts"] = owner.signal_attempts
    if channel is not None:
        report["credential_accounting"] = channel.summary()
    report["total_receive_capacity_bytes"] = (
        report.get("credential_accounting", {}).get("read_capacity_bytes", 0)
        + report["terminal_receive_attempts"] * credentials.RECEIVE_CAPACITY_BYTES
    )
    if not clean:
        failure = "guard_cleanup_failed"
    elif failure is None and time.monotonic() >= deadline:
        failure = "lifecycle_deadline_exhausted"
    if failure is None and accepted:
        report.update({
            "state": "passed", "kernel_guard_probe_passed": True,
            "denied_syscall_challenges_passed": EXPECTED_DENIAL_CHECKS,
            "address_space_rejection_challenges_passed": 1,
            "positive_anonymous_mapping_challenges_passed": 1,
            "inert_child_hard_limits_readback_verified": True,
            "inert_child_filter_install_and_challenges_verified": True,
        })
    elif failure in {
        "linux_credentials_unavailable", "lifecycle_unavailable", "kernel_policy_unavailable",
    }:
        report["state"] = "unsupported"
    report["failure_code"] = failure
    return report
