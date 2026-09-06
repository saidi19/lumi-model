"""Fixed kernel-guarded inert workers attached to the frozen durable ledger.

All four child sources are authenticated before attempt creation. The child
installs the unchanged kernel policy and passes its fixed challenges BEFORE
READY; credentials, live ownership and filter/capability snapshots precede the
parent's token release. This is not arbitrary-worker containment or SFT admission.
The prior ledger, wire protocol, policy, challenges and evidence remain unchanged.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import subprocess
import sys
import time
from pathlib import Path

from . import posttraining_sft_authenticated_ledger_v1 as authenticated
from . import posttraining_sft_kernel_guard_probe_v1 as kernel_guard

previous = authenticated.previous
protocol = authenticated.protocol
credentials = authenticated.credentials
socket = credentials.socket
lifecycle = authenticated.lifecycle
kernel_policy = kernel_guard.kernel_policy
MAXIMUM_SECONDS = authenticated.MAXIMUM_SECONDS
MAXIMUM_RECEIVES = authenticated.MAXIMUM_RECEIVES
MAXIMUM_STATUS_READS = authenticated.MAXIMUM_STATUS_READS
MAXIMUM_SEND_BYTES = authenticated.MAXIMUM_SEND_BYTES
CLEANUP_SECONDS = authenticated.CLEANUP_SECONDS
ROOT = Path(__file__).resolve().parents[2]
WORKER_SHA256 = "628841b0fa3f54ec4487c0612dccdd83a2241f205513b85da0569f84a35569aa"
SOURCE_BINDINGS = (
    *kernel_guard.SOURCE_BINDINGS,
    ("scripts/sft_authenticated_ledger_worker_v1.py", 16384, authenticated.WORKER_SHA256),
    ("scripts/sft_guarded_ledger_worker_v1.py", 16384, WORKER_SHA256),
)
READY = b"guarded-ledger-ready-v1"
FAULTS = authenticated.FAULTS | {"guard_exit_before_ready"}
ERROR_CODES = authenticated.ERROR_CODES | kernel_guard.ERROR_CODES
remaining = authenticated.remaining
cleanup = authenticated.cleanup


class IntegrationError(authenticated.IntegrationError):
    def __init__(self, code):
        self.code = code if type(code) is str and code in ERROR_CODES else "integration_failed"
        ValueError.__init__(self, self.code)


def need(condition, code):
    if not condition:
        raise IntegrationError(code) from None


class Budget(authenticated.Budget):
    """The SAME global 64-read/64-receive budgets, now with guarded child snapshots.

    Only the one-read child parser changes. Capacity stays prepaid cumulatively
    across all six children; failure never refunds or resets a reservation.
    """

    def __init__(self):
        super().__init__()
        self.filter_snapshots = 0

    def status(self, owner=None, *, deadline):
        remaining(deadline)
        need(self.status_reservations < MAXIMUM_STATUS_READS, "budget_exhausted")
        self.status_reservations += 1
        reader = kernel_guard.GuardReads()
        try:
            if owner is None:
                reader.controller()
            else:
                snapshot = reader.child(owner, deadline)
                need(snapshot["filter_mode_snapshot"] is True, "guard_snapshot_rejected")
                self.snapshots += 1
                self.filter_snapshots += 1
                self.empty_bounding_snapshots += int(snapshot["capability_bounding_set_empty"])
            remaining(deadline)
        finally:
            self.status_attempts += reader.attempts
            self.status_closed += reader.closed
            self.status_cleanup_failed |= reader.cleanup_failed

    def summary(self):
        return {**super().summary(), "filter_snapshots_verified": self.filter_snapshots}


def authenticate_sources():
    sources = []
    for relative, capacity, expected in SOURCE_BINDINGS:
        with (ROOT / relative).open("rb") as stream:
            raw = stream.read(capacity + 1)
        need(len(raw) <= capacity
             and hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected),
             "source_rejected")
        sources.append(raw)
    return tuple(sources)


def spawn(endpoint, fault, sources):
    """Execute authenticated bytes in separate namespaces; never reopen paths.

    The frozen workers load as libraries, NOT __main__; only the new fixed
    adapter entrypoint executes. All imports occur before child confinement.
    Neither install() nor any syscall challenge runs in the parent process.
    """
    need(type(fault) is str and fault in FAULTS - {"ack_send_failure"}
         and type(sources) is tuple and len(sources) == 4, "configuration_rejected")
    for raw, (_path, capacity, expected) in zip(sources, SOURCE_BINDINGS, strict=True):
        need(type(raw) is bytes and len(raw) <= capacity
             and hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected),
             "source_rejected")
    code = "import types\n"
    for name, raw in zip(("kernel_policy", "kernel_challenges", "ledger_worker"),
                         sources[:3], strict=True):
        code += f"{name} = types.ModuleType('_fixed_{name}')\n"
        code += f"exec(compile({raw!r}, '<fixed-{name}>', 'exec'), {name}.__dict__)\n"
    code += (
        f"exec(compile({sources[3]!r}, '<fixed-guarded-worker>', 'exec'), "
        "{'__name__': '__main__', 'kernel_policy': kernel_policy, "
        "'kernel_challenges': kernel_challenges, 'ledger_worker': ledger_worker})\n"
    )
    return subprocess.Popen(
        [sys.executable, "-S", "-B", "-c", code, str(endpoint.fileno()), fault],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        close_fds=True, pass_fds=(endpoint.fileno(),), start_new_session=True,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
    )


class Peer(authenticated.Peer):
    """Keep frozen ACK/feed/confirmation/EOF rules; gate token release on guard READY."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.guard_verified = False

    def initialize(self, token):
        # receive() already brackets credentials + guarded status with live pidfd.
        need(self.receive() == len(READY).to_bytes(4, "big") + READY, "frame_rejected")
        self.guard_verified = True
        self.send("initialize", token=token)
        self.expect("hello", token=token)

    def exchange(self, operation, arguments, *, ack_send_failure=False):
        # Identical frozen transaction ordering, with the successor's wider
        # diagnostic allowlist so a rejected guard snapshot keeps its stage code.
        self.send("command", operation=operation, arguments=arguments)
        acknowledgements = self.session.feed(self.rank, self.receive())
        need(len(acknowledgements) == 1, "ledger_rejected")
        ack = acknowledgements[0]
        failure = None
        try:
            need(not ack_send_failure, "transport_failed")
            self.send_raw(ack.frame)
            self.session.mark_ack_sent(ack)
            need(self.session.feed(self.rank, self.receive()) == (), "ledger_rejected")
            self.session.check_pending()
            self.send("continue", sequence=ack.sequence)
        except BaseException as error:
            code = getattr(error, "code", None)
            failure = code if type(code) is str and code in ERROR_CODES else "transport_failed"
        if failure is not None:
            try:
                self.session.ack_send_failed(self.rank)
            except protocol.LedgerProtocolError:
                pass
            raise IntegrationError(failure) from None


def run_synthetic_probe(directory: Path, *, timeout_seconds: int = MAXIMUM_SECONDS,
                        fault: str = "none") -> dict:
    """One exclusive synthetic attempt, with the frozen three-epoch ordering.

    No retry, private inputs, model imports, GPU or target-host action. Twenty
    seconds includes successful cleanup; failure retirement has one shared
    three-second grace. Blocking kernel/filesystem calls are not wall bounded.
    Per-child limits are NOT aggregate host/descendant memory enforcement.
    """
    budget = Budget()
    owners, peers, sockets, epochs = [], [], [], []
    controller = None
    failure = None
    controller_verified = source_verified = False
    journal_summary = {}
    deadline = time.monotonic()
    try:
        need(isinstance(directory, Path) and type(timeout_seconds) is int
             and 1 <= timeout_seconds <= MAXIMUM_SECONDS and type(fault) is str
             and fault in FAULTS, "configuration_rejected")
        deadline += timeout_seconds
        lifecycle.require_runtime()
        kernel_policy.require_abi()  # Read-only; never install confinement in the parent.
        budget.status(deadline=deadline)
        controller_verified = True
        lifecycle.normalize_child_signal()
        sources = authenticate_sources()
        source_verified = True
        controller = previous.DurableSupervisor(directory)
        seen = set()
        for number, (run, start, stop, resume) in enumerate(previous.supervision.SEQUENCE):
            remaining(deadline)
            epoch = protocol.create_epoch()
            tokens = protocol.create_rank_tokens(2)
            fresh = {epoch, *tokens.values()}
            need(len(fresh) == 3 and not fresh.intersection(seen), "frame_rejected")
            seen.update(fresh)
            session = protocol.EpochLedgerSession(
                controller, world_size=2, rank_tokens=tokens, epoch=epoch,
            )
            session.begin_invocation(run, start, stop, resume)
            epochs.append(session)
            current = []
            for rank in range(2):
                parent, endpoint = credentials.credential_pair()
                sockets.extend((parent, endpoint))
                child_fault = fault if number == 0 and rank == 0 else "none"
                if child_fault == "ack_send_failure":
                    child_fault = "none"
                owner = lifecycle.OwnedChild(spawn(endpoint, child_fault, sources))
                owners.append(owner)
                owner.bind()
                sockets.remove(endpoint)
                try:
                    endpoint.close()
                except BaseException:
                    raise IntegrationError("cleanup_failed") from None
                channel = credentials.CredentialChannel(
                    parent, expected_credentials=(owner.child.pid, os.getuid(), os.getgid()),
                    deadline=deadline, maximum_messages=24,
                    maximum_read_bytes=24 * credentials.RECEIVE_CAPACITY_BYTES,
                )
                peer = Peer(owner, channel, session, rank, epoch, budget, deadline)
                peers.append(peer)
                sockets.remove(parent)
                current.append(peer)
                peer.initialize(tokens[rank])
            for peer in current:
                peer.exchange("reserve", {"checkpoint_read_bytes": 1},
                              ack_send_failure=fault == "ack_send_failure"
                              and number == 0 and peer.rank == 0)
            for step in range(start, stop):
                current[0].exchange("reserve_step", {"from_step": step, "to_step": step + 1})
                for peer in current:
                    peer.exchange("step_complete", {"to_step": step + 1})
            session.observe_memory(1, [0, 0])  # Synthetic, NOT host/GPU measurements.
            session.finish_invocation(stop)
            for peer in current:
                peer.sealed = True
                peer.send("stop")
                peer.expect("bye")
            for peer in current:
                peer.send("release")
            for peer in current:
                peer.owner.reap(deadline, require_success=True)
            for peer in current:
                peer.terminal_eof()
                peer.channel.close()
                peer.owner.close()
            remaining(deadline)
            controller.commit_retired_epoch()
            remaining(deadline)
    except BaseException as error:
        code = getattr(error, "code", None)
        failure = code if type(code) is str and code in ERROR_CODES else "integration_failed"
        if controller is not None:
            controller.interrupt()
    clean = cleanup(owners, peers, sockets)
    clean &= not budget.status_cleanup_failed and failure not in {
        "cleanup_failed", "lifecycle_cleanup_failed", "credential_cleanup_failed",
        "guard_cleanup_failed",
    }
    if not clean:
        failure = "cleanup_failed"
    if controller is not None and controller.journal is not None:
        journal = controller.journal
        try:
            if failure is not None and journal.summary()["terminal_event"] is None:
                journal.append("attempt_failed", ())
        except BaseException:
            failure = failure or "journal_rejected"
        try:
            journal.close()
            journal_summary = journal.summary()
            need(journal_summary["closed"] and journal_summary["failure_code"] is None,
                 "journal_rejected")
        except BaseException:
            failure = failure or "journal_rejected"
    if failure is None and time.monotonic() >= deadline:
        failure = "deadline_exhausted"
    if failure is not None and controller is not None:
        controller.interrupt()
    guarded = sum(int(peer.guard_verified) for peer in peers)
    successful = (failure is None and controller is not None
                  and controller.epochs_committed == 3 and guarded == 6)
    if failure is None and not successful:
        failure = "integration_failed"
    summary = controller.supervisor.summary() if controller is not None else {}
    totals = summary.get("resources_reserved", {})
    state = "passed" if successful else "failed"
    if failure in {"linux_credentials_unavailable", "lifecycle_unavailable",
                   "kernel_policy_unavailable"}:
        state = "unsupported"
    return {
        "schema_version": "lumi.sft-guarded-ledger-synthetic-summary.v0.1",
        "state": state, "failure_code": failure,
        "worker_processes_started": len(owners),
        "worker_processes_reaped": sum(int(owner.reaped) for owner in owners),
        "worker_cleanup_complete": clean,
        "pidfds_bound_live": sum(int(owner.bound_live) for owner in owners),
        "pidfds_closed": sum(int(owner.fd is not None and owner.closed) for owner in owners),
        "pidfd_signal_attempts": sum(owner.signal_attempts for owner in owners),
        "epochs_completed": controller.epochs_committed if controller is not None else 0,
        "global_optimizer_steps_completed": summary.get("global_optimizer_steps_completed", 0),
        "requests_predebited": sum(item.summary()["requests_predebited"] for item in epochs),
        "acknowledgements_confirmed": sum(
            item.summary()["acknowledgements_confirmed"] for item in epochs
        ),
        "credential_messages_verified": sum(
            peer.channel.summary()["verified_messages"] for peer in peers
        ),
        "received_descriptors_closed": sum(
            peer.channel.summary()["received_descriptors_closed"]
            + peer.terminal_descriptors_closed for peer in peers
        ),
        "terminal_eof_verified": sum(int(peer.terminal_checked) for peer in peers),
        "checkpoint_bytes_synthetically_reserved": totals.get("checkpoint_read_bytes", 0),
        "journal_capacity_predebited": totals.get("private_write_bytes", 0),
        "journal_write_bytes_charged": controller._journal_charged if controller else 0,
        "journal_records_committed": journal_summary.get("records_committed", 0),
        "journal_terminal_event": journal_summary.get("terminal_event"),
        "journal_closed": journal_summary.get("closed", False),
        "controller_status_verified": controller_verified,
        "worker_source_authenticated": source_verified,
        "accounting": budget.summary(),
        "guarded_worker_readiness_verified": guarded,
        "denied_syscall_challenges_passed": guarded * kernel_guard.EXPECTED_DENIAL_CHECKS,
        "positive_anonymous_mapping_challenges_passed": guarded,
        "address_space_rejection_challenges_passed": guarded,
        "inert_child_policy_and_challenges_verified": successful,
        "guarded_inert_ledger_probe_passed": successful,
        "authenticated_inert_ledger_probe_passed": successful,
        "durable_ledger_integrated": True, "no_numeric_pid_signals": True,
        "continuous_capability_exclusion": False, "privileged_sender_capabilities_excluded": False,
        "real_memory_measurement": False, "os_memory_enforcement": False,
        "cgroup_memory_enforcement": False, "aggregate_descendant_enforcement": False,
        "os_network_enforcement": False, "namespace_containment": False,
        "ptrace_containment": False, "real_worker_sandbox_verified": False,
        "target_gpu_host_verified": False, "guarded_real_worker_ready": False,
        "execution_authorized": False, "real_sft_execution_permitted": False,
        "production_eligible": False, "full_private_integration_passed": False,
        "training_source_or_data_reads": 0, "model_or_checkpoint_reads": 0,
        "external_network_bytes": 0, "gpu_use": 0, "inference_calls": 0,
    }
