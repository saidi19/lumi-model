"""One fixed inert Linux child, not a guarded SFT worker or target-host approval.

This isolates kernel per-message credential checks from the existing durable
ledger. It reads only a bounded, hash-pinned child source file and uses unnamed
local sockets. A process-generation binding, capability policy, actual ledger
integration and Linux resource/network containment remain separate prerequisites.
"""

from __future__ import annotations

import hashlib
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

from . import posttraining_sft_peer_credentials_v1 as credentials

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "scripts/sft_peer_identity_inert_worker_v1.py"
WORKER_SHA256 = "a1fa5ed9a013e0a156d6365f9c53d752967b73402fa802bbaa9a39066badd7c7"
MAXIMUM_SECONDS = 5
CLEANUP_GRACE_SECONDS = 1
FAULTS = frozenset({"none", "unexpected_sender", "rights", "oversized", "stall"})
PAYLOAD = b'{"kind":"inert_identity_probe"}'
FRAME = struct.pack("!I", len(PAYLOAD)) + PAYLOAD
PROBE_CODES = frozenset({
    "probe_configuration_rejected", "probe_runtime_rejected", "probe_source_rejected",
    "probe_deadline_exhausted", "probe_frame_rejected", "probe_worker_failed",
    "probe_cleanup_failed", "probe_failed",
})


class ProbeError(ValueError):
    def __init__(self, code):
        self.code = code if type(code) is str and code in PROBE_CODES else "probe_failed"
        super().__init__(self.code)


def _need(condition, code):
    if not condition:
        raise ProbeError(code) from None


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    _need(remaining > 0, "probe_deadline_exhausted")
    return remaining


def require_stdlib():
    _need(
        sys.flags.no_site == 1 and sys.dont_write_bytecode and not sys.flags.optimize
        and not any(name.split(".")[0] in {
            "torch", "transformers", "numpy", "tokenizers", "safetensors",
        } for name in sys.modules),
        "probe_runtime_rejected",
    )


def authenticate_worker():
    """Authenticate bounded public control source before allocating a socket or child."""
    with WORKER.open("rb") as stream:
        raw = stream.read(8193)
    _need(
        len(raw) <= 8192 and hashlib.sha256(raw).hexdigest() == WORKER_SHA256,
        "probe_source_rejected",
    )
    return raw


def spawn_inert_worker(endpoint, fault, source):
    """Execute exact authenticated bytes, not a mutable path after its hash check."""
    _need(
        type(source) is bytes and len(source) <= 8192
        and hashlib.sha256(source).hexdigest() == WORKER_SHA256,
        "probe_source_rejected",
    )
    _need(type(fault) is str and fault in FAULTS, "probe_configuration_rejected")
    return subprocess.Popen(
        [sys.executable, "-S", "-B", "-c", source.decode("utf-8"),
         str(endpoint.fileno()), fault],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        close_fds=True, pass_fds=(endpoint.fileno(),),
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "PYTHONHASHSEED": "0"},
    )


def cleanup(child, channel, sockets):
    """Attempt every owned cleanup; never signal a process not spawned by this probe."""
    clean, reaped = True, 0
    if child is not None:
        try:
            if child.poll() is None:
                child.kill()
        except BaseException:
            clean = False
        try:
            child.wait(timeout=CLEANUP_GRACE_SECONDS)
            reaped = 1
        except BaseException:
            clean = False
    if channel is not None:
        try:
            channel.close()
        except BaseException:
            clean = False
    for endpoint in sockets:
        try:
            endpoint.close()
        except BaseException:
            clean = False
    return clean, reaped


def initial_report():
    return {
        "schema_version": "lumi.sft-peer-identity-probe.v0.1",
        "state": "failed",
        "failure_code": None,
        "linux_peer_credential_probe_passed": False,
        "worker_processes_started": 0,
        "worker_processes_reaped": 0,
        "worker_cleanup_complete": False,
        "worker_source_authenticated": False,
        "kernel_pid_tuple_matches_expected_child": False,
        "process_generation_bound": False,
        "privileged_sender_capabilities_excluded": False,
        "durable_ledger_integrated": False,
        "os_memory_enforcement": False,
        "os_network_enforcement": False,
        "target_gpu_host_verified": False,
        "guarded_real_worker_ready": False,
        "execution_authorized": False,
        "production_eligible": False,
        "full_private_integration_passed": False,
        "training_source_or_data_reads": 0,
        "model_or_checkpoint_reads": 0,
        "private_artifact_writes": 0,
        "external_network_bytes": 0,
        "gpu_use": 0,
        "inference_calls": 0,
    }


def run_inert_probe(*, fault="none"):
    """One child and one receive, never retry; non-Linux denies before any source read.

    Negative fixtures intentionally reject rather than reporting a production
    PASS. Five seconds includes setup and successful cleanup; failure cleanup
    can use a further one-second reap grace. This is not a blocking-syscall or
    provider billing guarantee. The source checkout/interpreter are trusted.
    """
    report = initial_report()
    deadline = time.monotonic() + MAXIMUM_SECONDS
    child = channel = None
    sockets = []
    received = False
    failure = None
    try:
        require_stdlib()
        _need(type(fault) is str and fault in FAULTS, "probe_configuration_rejected")
        credentials.require_linux()
        source = authenticate_worker()
        report["worker_source_authenticated"] = True
        _remaining(deadline)
        parent, endpoint = credentials.credential_pair()
        sockets.extend((parent, endpoint))
        child = spawn_inert_worker(endpoint, fault, source)
        report["worker_processes_started"] = 1
        sockets.remove(endpoint)
        try:
            endpoint.close()
        except BaseException:
            raise ProbeError("probe_cleanup_failed") from None
        expected_pid = os.getpid() if fault == "unexpected_sender" else child.pid
        channel = credentials.CredentialChannel(
            parent, expected_credentials=(expected_pid, os.getuid(), os.getgid()),
            deadline=min(deadline, time.monotonic() + 0.25) if fault == "stall" else deadline,
            maximum_messages=1, maximum_read_bytes=credentials.RECEIVE_CAPACITY_BYTES,
        )
        sockets.remove(parent)  # Ownership transferred; do not retry a failed channel close.
        parent.settimeout(_remaining(deadline))
        _need(parent.send(b"G") == 1, "probe_worker_failed")
        frame = channel.receive()
        _need(frame == FRAME, "probe_frame_rejected")
        received = True
        parent.settimeout(_remaining(deadline))
        _need(parent.send(b"K") == 1, "probe_worker_failed")
        _need(child.wait(timeout=_remaining(deadline)) == 0, "probe_worker_failed")
    except credentials.PeerCredentialError as error:
        failure = error.code
    except ProbeError as error:
        failure = error.code
    except BaseException:
        failure = "probe_failed"
    clean, reaped = cleanup(child, channel, sockets)
    # Earlier endpoint/received-rights close failures cannot be erased by later
    # successful cleanup of the remaining handles. Never retry the failed close.
    clean = clean and failure not in {"probe_cleanup_failed", "credential_cleanup_failed"}
    report["worker_processes_reaped"] = reaped
    report["worker_cleanup_complete"] = clean
    if not clean:
        failure = "probe_cleanup_failed"
    elif time.monotonic() >= deadline:
        failure = "probe_deadline_exhausted"
    if channel is not None:
        report["credential_accounting"] = channel.summary()
    if failure is None and received and clean and reaped == 1:
        report["state"] = "passed"
        report["linux_peer_credential_probe_passed"] = True
        report["kernel_pid_tuple_matches_expected_child"] = True
    elif failure == "linux_credentials_unavailable":
        report["state"] = "unsupported"
    elif type(failure) is str and failure.startswith("credential_"):
        report["state"] = "rejected"
    report["failure_code"] = failure
    return report
