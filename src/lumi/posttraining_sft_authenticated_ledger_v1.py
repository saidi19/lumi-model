"""Owned, credential-checked inert workers attached to the frozen durable ledger.

This is a fixed synthetic integration, NOT admission for arbitrary code or SFT.
Capabilities are checked at snapshots, not continuously. Trusted genuine procfs,
supported single-threaded CPython and no foreign reaper/native mutation remain
prerequisites. No dataset, model, CUDA, inference or remote operation is present.
All memory observations and work reservations are explicitly synthetic.
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

from . import posttraining_sft_local_ipc_v1 as previous
from . import posttraining_sft_peer_credentials_v1 as credentials
from . import posttraining_sft_peer_lifecycle_v1 as lifecycle

protocol = previous.protocol
MAXIMUM_SECONDS = 20
CLEANUP_SECONDS = 3
MAXIMUM_RECEIVES = 64
MAXIMUM_STATUS_READS = 64
MAXIMUM_SEND_BYTES = 131_072
MAXIMUM_SOURCE_BYTES = 16_384
WORKER_SHA256 = "c5b3e6e58afaa6a385dc14460e2466873b83792f6ec88589222e29376b020e9e"
READY = b"authenticated-ledger-ready-v1"
FAULTS = frozenset((
    "none", "wrong_epoch", "wrong_rank", "rights", "exit_before_confirmation", "stall",
    "bad_exit", "trailing_after_bye", "ack_send_failure",
))
ERROR_CODES = frozenset((
    "configuration_rejected", "source_rejected", "integration_failed", "budget_exhausted",
    "deadline_exhausted", "frame_rejected", "transport_failed", "cleanup_failed",
    "terminal_rejected", "ledger_rejected", "journal_rejected",
)) | (
    previous.ERROR_CODES | credentials.ERROR_CODES | lifecycle.ERROR_CODES
    | lifecycle.policy.ERROR_CODES | protocol.ERROR_CODES
)


class IntegrationError(ValueError):
    """Only fixed allowlisted codes; never retain exception, identity or message text."""

    def __init__(self, code: str):
        self.code = code if type(code) is str and code in ERROR_CODES else "integration_failed"
        super().__init__(self.code)


def need(condition: bool, code: str) -> None:
    if not condition:
        raise IntegrationError(code) from None


def remaining(deadline: float) -> float:
    value = deadline - time.monotonic()
    need(value > 0, "deadline_exhausted")
    return value


class Budget:
    """One cumulative capacity budget shared across every peer and all epochs.

    Status capacity is reserved before delegating one read to the unchanged
    frozen reader; creating a fresh helper does NOT reset this aggregate limit.
    Failed reservations/operations are never refunded. These exclude Python
    module loading and kernel overhead; they are not total host I/O measurements.
    """

    def __init__(self):
        self.receives = 0
        self.send_bytes = 0
        self.status_reservations = 0
        self.status_attempts = 0
        self.status_closed = 0
        self.status_cleanup_failed = False
        self.snapshots = 0
        self.empty_bounding_snapshots = 0

    def receive(self) -> None:
        need(self.receives < MAXIMUM_RECEIVES, "budget_exhausted")
        self.receives += 1

    def send(self, size: int) -> None:
        need(type(size) is int and 0 < size <= credentials.MAXIMUM_FRAME_BYTES,
             "frame_rejected")
        need(self.send_bytes + size <= MAXIMUM_SEND_BYTES, "budget_exhausted")
        self.send_bytes += size

    def status(self, owner=None, *, deadline: float) -> None:
        remaining(deadline)
        need(self.status_reservations < MAXIMUM_STATUS_READS, "budget_exhausted")
        self.status_reservations += 1
        reader = lifecycle.StatusReads()
        try:
            if owner is None:
                reader.controller()
            else:
                snapshot = reader.child(owner, deadline)
                self.snapshots += 1
                self.empty_bounding_snapshots += int(snapshot["capability_bounding_set_empty"])
            remaining(deadline)
        finally:
            self.status_attempts += reader.attempts
            self.status_closed += reader.closed
            self.status_cleanup_failed |= reader.cleanup_failed

    def summary(self) -> dict:
        return {
            "receive_capacity_reservations": self.receives,
            "receive_capacity_bytes": self.receives * credentials.RECEIVE_CAPACITY_BYTES,
            "send_bytes_precharged": self.send_bytes,
            "status_capacity_reservations": self.status_reservations,
            "status_capacity_bytes": self.status_reservations * lifecycle.STATUS_CAPACITY,
            "status_read_attempts": self.status_attempts,
            "status_descriptors_closed": self.status_closed,
            "capability_snapshots_verified": self.snapshots,
            "capability_bounding_set_empty_snapshots": self.empty_bounding_snapshots,
        }


def authenticated_source() -> bytes:
    path = Path(__file__).resolve().parents[2] / "scripts/sft_authenticated_ledger_worker_v1.py"
    with path.open("rb") as stream:
        raw = stream.read(MAXIMUM_SOURCE_BYTES + 1)
    need(len(raw) <= MAXIMUM_SOURCE_BYTES
         and hmac.compare_digest(hashlib.sha256(raw).hexdigest(), WORKER_SHA256), "source_rejected")
    return raw


def spawn(endpoint: socket.socket, fault: str, source: bytes) -> subprocess.Popen:
    """Execute only the bytes already authenticated, never reopen the script path."""
    need(fault in FAULTS - {"ack_send_failure"}, "configuration_rejected")
    need(type(source) is bytes
         and hmac.compare_digest(hashlib.sha256(source).hexdigest(), WORKER_SHA256),
         "source_rejected")
    return subprocess.Popen(
        [sys.executable, "-S", "-B", "-c", source, str(endpoint.fileno()), fault],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        close_fds=True, pass_fds=(endpoint.fileno(),),
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"}, start_new_session=True,
    )


class Peer:
    """A single parent-owned rank/session binding, never rank supplied by a sender.

    Peer constructor takes ownership of a successfully constructed channel.
    Every ledger feed goes through live ownership, per-packet credentials and a
    bracketed status snapshot. An explicit continue handshake holds the child
    alive while its confirmation is checked. This is not hostile-code isolation.
    """

    def __init__(self, owner, channel, session, rank: int, epoch: str, budget: Budget,
                 deadline: float):
        self.owner = owner
        self.channel = channel
        self.session = session
        self.rank = rank
        self.epoch = epoch
        self.budget = budget
        self.deadline = deadline
        self.sealed = False
        self.terminal_checked = False
        self.terminal_attempted = False
        self.terminal_descriptors_closed = 0

    def send_raw(self, raw: bytes) -> None:
        need(type(raw) is bytes and 5 <= len(raw) <= credentials.MAXIMUM_FRAME_BYTES
             and int.from_bytes(raw[:4], "big") == len(raw) - 4, "frame_rejected")
        self.owner.assert_live()
        self.budget.send(len(raw))
        self.channel._sock.settimeout(remaining(self.deadline))
        need(self.channel._sock.sendmsg([raw]) == len(raw), "transport_failed")
        remaining(self.deadline)

    def send(self, kind: str, **fields) -> None:
        value = previous._control(kind, self.epoch, self.rank, **fields)
        self.send_raw(protocol.encode_frame(value))

    def receive(self) -> bytes:
        remaining(self.deadline)
        self.owner.assert_live()
        self.budget.receive()
        raw = self.channel.receive()
        self.owner.assert_live()
        self.budget.status(self.owner, deadline=self.deadline)
        self.owner.assert_live()
        remaining(self.deadline)
        return raw

    def expect(self, kind: str, **fields) -> None:
        raw = self.receive()
        expected = protocol.encode_frame(previous._control(kind, self.epoch, self.rank, **fields))
        need(hmac.compare_digest(raw, expected), "frame_rejected")

    def initialize(self, token: str) -> None:
        need(self.receive() == len(READY).to_bytes(4, "big") + READY, "frame_rejected")
        self.send("initialize", token=token)
        self.expect("hello", token=token)

    def exchange(self, operation: str, arguments: dict, *, ack_send_failure: bool = False) -> None:
        self.send("command", operation=operation, arguments=arguments)
        # DurableSupervisor appends AND fsyncs the predebit before feed returns.
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
            # This interrupts the frozen ledger and NEVER refunds an accepted debit.
            try:
                self.session.ack_send_failed(self.rank)
            except protocol.LedgerProtocolError:
                pass
            # In particular, do not erase descriptor/status cleanup failures.
            raise IntegrationError(failure) from None

    def terminal_eof(self) -> None:
        """After seal + real reap, distinguish EOF from even a zero-byte packet.

        On Linux SEQPACKET, an actual queued empty packet still carries PASSCRED
        ancillary data. Accept only empty data AND empty ancillary data at EOF.
        The same cumulative receive capacity covers this one-shot terminal read.
        """
        need(self.sealed and self.owner.reaped and not self.terminal_attempted,
             "terminal_rejected")
        self.terminal_attempted = True
        self.budget.receive()
        sock = self.channel._sock
        sock.settimeout(remaining(self.deadline))
        raw, ancillary, flags, _address = sock.recvmsg(
            credentials.RECEIVE_DATA_CAPACITY, credentials.ANCILLARY_BUFFER_BYTES,
            socket.MSG_CMSG_CLOEXEC,
        )
        closed, failed = credentials._close_received_rights(ancillary)
        self.terminal_descriptors_closed += closed
        need(not failed, "cleanup_failed")
        remaining(self.deadline)
        need(raw == b"" and ancillary == [] and type(flags) is int
             and flags & ~socket.MSG_CMSG_CLOEXEC == 0, "terminal_rejected")
        self.session.disconnect(self.rank)
        self.terminal_checked = True


def cleanup(owners: list, peers: list[Peer], sockets: list) -> bool:
    """Retire every handle before failure-journal writes; never retry failed close.

    The three-second cleanup deadline is shared, not renewed per child. If an
    ownership bind failed, a bounded Popen wait is best effort, never reap proof.
    No numeric-PID signal or worker replacement is attempted.
    """
    clean = True
    deadline = time.monotonic() + CLEANUP_SECONDS
    while sockets:
        sock = sockets.pop()
        try:
            sock.close()
        except BaseException:
            clean = False
    for peer in peers:
        try:
            peer.channel.close()  # Frozen helper never retries an attempted close.
        except BaseException:
            clean = False
    for owner in owners:
        try:
            if not owner.owned:
                clean = False
                owner.child.wait(timeout=remaining(deadline))
            elif not owner.reaped:
                owner.terminate_and_reap(deadline)
        except BaseException:
            clean = False
        try:
            if not owner.close_attempted:
                owner.close()
            clean &= owner.closed
        except BaseException:
            clean = False
    return clean


def run_synthetic_probe(directory: Path, *, timeout_seconds: int = MAXIMUM_SECONDS,
                        fault: str = "none") -> dict:
    """One exclusive attempt: three fixed epochs or stop on the first failure.

    No retry, data/ML imports, arbitrary executable, target-host operation or real
    training permission. Mac/unsupported Linux is denied before attempt creation.
    Blocking OS/filesystem calls and interpreter overhead are not deadline-bounded.
    """
    budget = Budget()
    owners, peers, sockets, epochs = [], [], [], []
    controller = None
    failure = None
    controller_verified = source_verified = False
    journal_summary = {}
    deadline = time.monotonic()  # Invalid options still have a bounded cleanup path.
    try:
        need(isinstance(directory, Path) and type(timeout_seconds) is int
             and 1 <= timeout_seconds <= MAXIMUM_SECONDS and type(fault) is str
             and fault in FAULTS, "configuration_rejected")
        deadline += timeout_seconds
        lifecycle.require_runtime()
        budget.status(deadline=deadline)
        controller_verified = True
        lifecycle.normalize_child_signal()
        source = authenticated_source()
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
            # Consume the exclusive attempt marker before the first child exists.
            session.begin_invocation(run, start, stop, resume)
            epochs.append(session)
            current = []
            for rank in range(2):
                parent, endpoint = credentials.credential_pair()
                sockets.extend((parent, endpoint))
                child_fault = fault if number == 0 and rank == 0 else "none"
                if child_fault == "ack_send_failure":
                    child_fault = "none"
                owner = lifecycle.OwnedChild(spawn(endpoint, child_fault, source))
                owners.append(owner)
                owner.bind()
                sockets.remove(endpoint)  # Remove before close; an error cannot retry its FD.
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
                peer.expect("bye")  # Child waits for release, so status can be checked live.
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
    # Cleanup precedes any failure journal append/fsync. Close errors stay sticky.
    clean = cleanup(owners, peers, sockets)
    clean &= not budget.status_cleanup_failed and failure not in {
        "cleanup_failed", "lifecycle_cleanup_failed", "credential_cleanup_failed",
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
    successful = failure is None and controller is not None and controller.epochs_committed == 3
    if failure is None and not successful:
        failure = "integration_failed"
    summary = controller.supervisor.summary() if controller is not None else {}
    totals = summary.get("resources_reserved", {})
    state = "passed" if successful else "failed"
    if failure in {"linux_credentials_unavailable", "lifecycle_unavailable"}:
        state = "unsupported"
    return {
        "schema_version": "lumi.sft-authenticated-ledger-synthetic-summary.v0.1",
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
        "authenticated_inert_ledger_probe_passed": successful,
        "durable_ledger_integrated": True, "no_numeric_pid_signals": True,
        "continuous_capability_exclusion": False, "privileged_sender_capabilities_excluded": False,
        "real_memory_measurement": False, "os_memory_enforcement": False,
        "os_network_enforcement": False, "namespace_containment": False,
        "ptrace_containment": False,
        "target_gpu_host_verified": False, "guarded_real_worker_ready": False,
        "execution_authorized": False, "real_sft_execution_permitted": False,
        "production_eligible": False, "full_private_integration_passed": False,
        "training_source_or_data_reads": 0, "external_network_bytes": 0,
        "gpu_use": 0, "inference_calls": 0,
    }
