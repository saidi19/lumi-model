"""Actual local IPC and durable predebits, exclusively for inert synthetic workers.

This integrates ONE cumulative Supervisor, a prepaid append-only attempt journal,
and three fresh sets of child processes. It never imports the lumi package, ML,
training code, data, receipts or launch contracts. The standalone child script
is fixed; there is no caller-supplied executable, command or worker callback.

An inherited AF_UNIX socketpair is a private channel capability, NOT verified OS
peer-PID identity. This is not a production supervisor, CUDA admission adapter,
OS resource sandbox, billing guard, authenticated approval or restart recovery.
All observed memory values and work debits below are explicitly synthetic.
Existing attempt journals, including partial ones, are never reopened or retried.
"""

from __future__ import annotations

import hashlib
import hmac
import socket
import subprocess
import sys
import time
from pathlib import Path

from . import posttraining_sft_attempt_journal_v1 as journal_module
from . import posttraining_sft_epoch_protocol_v1 as protocol
from . import posttraining_sft_supervisor_v1 as supervision

CONTROL_SCHEMA = "lumi.sft-synthetic-ipc-control.v0.1"
JOURNAL_BUDGET = 65_536
MAXIMUM_SECONDS = 20
MESSAGE_SECONDS = 2
MAXIMUM_WIRE_BYTES = 131_072
RESOURCE_ORDER = (
    "checkpoint_read_bytes",
    "encoded_binary_read_bytes",
    "encoded_index_read_bytes",
    "private_write_bytes",
    "log_bytes",
    "encoded_index_rows",
)
FAULTS = frozenset(
    (
        "none",
        "wrong_epoch",
        "partial_request",
        "exit_before_confirmation",
        "stall",
        "partial_ack",
        "bad_exit",
        "trailing_after_bye",
    )
)
ERROR_CODES = frozenset(
    (
        "configuration_rejected",
        "stdlib_runtime_required",
        "attempt_unavailable",
        "deadline_exhausted",
        "ipc_failed",
        "ipc_frame_rejected",
        "ipc_authentication_rejected",
        "ipc_budget_exhausted",
        "worker_failed",
        "worker_cleanup_failed",
        "ledger_rejected",
        "journal_rejected",
        "synthetic_probe_failed",
    )
)


class ProbeError(ValueError):
    """Never retain worker output, paths, capabilities or underlying exception text."""

    def __init__(self, code: str):
        self.code = code if code in ERROR_CODES else "synthetic_probe_failed"
        super().__init__(self.code)


def _need(condition: bool, code: str) -> None:
    if not condition:
        raise ProbeError(code) from None


def require_stdlib() -> None:
    _need(
        sys.flags.no_site == 1
        and sys.dont_write_bytecode
        and not sys.flags.optimize
        and not any(
            name.split(".")[0] in {"torch", "transformers", "numpy", "tokenizers", "safetensors"}
            for name in sys.modules
        ),
        "stdlib_runtime_required",
    )


def synthetic_inputs() -> tuple[dict, dict]:
    """Fixed two-rank metadata only; these are not real host or training limits."""
    counts = {
        "sequential_invocations": 3,
        "global_optimizer_steps": 4,
        "rank_optimizer_steps": 8,
        "logical_sample_events": 8,
        "maximum_unique_train_records": 8,
        "rank_local_input_loads": 6,
        "maximum_encoded_index_rows": 48,
        "maximum_selected_binary_bytes": 48 * 32 * 8,
        "maximum_selected_index_bytes": 48 * 176,
        "includes_checkpoint_hashing_or_final_digest": False,
        "is_total_io_or_cost_budget": False,
    }
    caps = {
        "maximum_combined_runtime_seconds": MAXIMUM_SECONDS,
        "maximum_cost_microusd": 1000,
        "total_host_hourly_rate_microusd": 60_000,
        "maximum_host_ram_bytes": 4096,
        "maximum_per_gpu_memory_bytes": 2048,
        "maximum_private_write_bytes": 2 * JOURNAL_BUDGET,
        "maximum_checkpoint_read_bytes": 4096,
        "maximum_encoded_binary_read_bytes": counts["maximum_selected_binary_bytes"],
        "maximum_encoded_index_read_bytes": counts["maximum_selected_index_bytes"],
        "maximum_log_bytes": 256,
        "maximum_invocations": 3,
        "maximum_global_optimizer_steps": 4,
        "maximum_external_network_bytes": 0,
        "maximum_inference_calls": 0,
    }
    return caps, counts


class DurableSupervisor:
    """Serialize through EpochLedgerSession; persist each accepted predebit before ACK.

    The entire journal capacity is reserved once, before creating its first byte.
    This deliberately over-reserves (never refunds) and covers writes before,
    between and after invocations when Supervisor.reserve is otherwise closed.
    append's debit callback checks each attempted write fits that prepaid budget.
    A failed disk operation can consume work without a record; the exclusive
    attempt marker still prevents any automatic retry or crash-resume.
    """

    _world_size = 2

    def __init__(self, directory: Path):
        caps, counts = synthetic_inputs()
        self.supervisor = supervision.Supervisor(caps, counts, world_size=2)
        self._directory = directory
        self.journal = None
        self._journal_charged = 0
        self._started = 0
        self._awaiting_cleanup = False
        self.epochs_committed = 0

    def _debit_journal(self, amount: int) -> None:
        _need(type(amount) is int and amount > 0, "journal_rejected")
        self._journal_charged += amount
        _need(self._journal_charged <= JOURNAL_BUDGET, "journal_rejected")

    def _record(self, event: str, counters: tuple[int, ...] = ()) -> None:
        failed = False
        try:
            _need(self.journal is not None, "journal_rejected")
            self.journal.append(event, counters)
        except BaseException:
            failed = True
        if failed:
            self.interrupt()
            raise ProbeError("journal_rejected") from None

    def begin_invocation(self, run: str, from_step: int, stop_step: int, resume: bool) -> None:
        if self._awaiting_cleanup:
            self.interrupt()
            raise ProbeError("worker_cleanup_failed") from None
        self.supervisor.begin_invocation(run, from_step, stop_step, resume)
        if self.journal is None:
            self.supervisor.reserve(0, private_write_bytes=JOURNAL_BUDGET)
            failed = False
            try:
                self.journal = journal_module.create_new(
                    self._directory,
                    debit=self._debit_journal,
                    maximum_bytes=JOURNAL_BUDGET,
                )
            except BaseException:
                failed = True
            if failed:
                self.interrupt()
                raise ProbeError("attempt_unavailable") from None
        self._started += 1
        self._record("epoch_started", (self._started, from_step, stop_step))

    def reserve(self, rank: int, **amounts: int) -> None:
        self.supervisor.reserve(rank, **amounts)
        self._record("resource_reserved", tuple(amounts.get(key, 0) for key in RESOURCE_ORDER))

    def reserve_optimizer_step(self, from_step: int, to_step: int) -> None:
        self.supervisor.reserve_optimizer_step(from_step, to_step)
        self._record("step_reserved", (from_step, to_step))

    def finish_optimizer_step(self, to_step: int) -> None:
        self.supervisor.finish_optimizer_step(to_step)
        self._record("step_completed", (to_step,))

    def observe_memory(self, host_ram_bytes: int, per_gpu_memory_bytes: list[int]) -> None:
        self.supervisor.observe_memory(host_ram_bytes, per_gpu_memory_bytes)

    def finish_invocation(self, stop_step: int) -> None:
        self.supervisor.finish_invocation(stop_step)
        self._awaiting_cleanup = True

    def commit_retired_epoch(self) -> None:
        """Trusted parent only, AFTER protocol seal, expected EOF and all reaps."""
        _need(self._awaiting_cleanup, "worker_cleanup_failed")
        self._record("epoch_completed", (self._started,))
        self.epochs_committed += 1
        self._awaiting_cleanup = False
        if self._started == 3:
            self._record("attempt_completed", (3, 4))

    def checkpoint(self) -> None:
        self.supervisor.checkpoint()

    def interrupt(self) -> None:
        try:
            self.supervisor.interrupt()
        except supervision.SupervisorError:
            pass


class Channel:
    """One parent-bound socket, bounded frames and a non-resetting total deadline."""

    def __init__(self, sock: socket.socket, deadline: float, budget: list[int]):
        self.sock = sock
        self.deadline = deadline
        self.budget = budget

    def _time(self, message_deadline: float) -> None:
        remaining = min(self.deadline, message_deadline) - time.monotonic()
        _need(remaining > 0, "deadline_exhausted")
        self.sock.settimeout(remaining)

    def _charge(self, count: int) -> None:
        self.budget[0] += count
        _need(self.budget[0] <= MAXIMUM_WIRE_BYTES, "ipc_budget_exhausted")

    def send_raw(self, raw: bytes) -> None:
        _need(type(raw) is bytes and 4 < len(raw) <= 4100, "ipc_frame_rejected")
        self._charge(len(raw))
        self._time(time.monotonic() + MESSAGE_SECONDS)
        failure = None
        try:
            self.sock.sendall(raw)
        except TimeoutError:
            failure = "deadline_exhausted"
        except OSError:
            failure = "ipc_failed"
        if failure is not None:
            raise ProbeError(failure) from None

    def send(self, value: dict) -> None:
        self.send_raw(protocol.encode_frame(value))

    def receive(self) -> tuple[bytes, dict]:
        end = time.monotonic() + MESSAGE_SECONDS
        raw = bytearray()
        target = 4
        while True:
            self._time(end)
            failure = None
            try:
                block = self.sock.recv(target - len(raw))
            except TimeoutError:
                failure = "deadline_exhausted"
            except OSError:
                failure = "ipc_failed"
            if failure is not None:
                raise ProbeError(failure) from None
            _need(bool(block), "ipc_failed")
            self._charge(len(block))
            raw.extend(block)
            if len(raw) != target:
                continue
            if target == 4:
                size = int.from_bytes(raw, "big")
                _need(0 < size <= 4096, "ipc_frame_rejected")
                target += size
            else:
                break
        try:
            decoded = protocol.FrameDecoder().feed(bytes(raw))
            _need(len(decoded) == 1, "ipc_frame_rejected")
            return bytes(raw), decoded[0]
        except protocol.LedgerProtocolError:
            raise ProbeError("ipc_frame_rejected") from None


def _control(kind: str, epoch: str, rank: int, **fields) -> dict:
    return {"schema_version": CONTROL_SCHEMA, "kind": kind, "epoch": epoch, "rank": rank, **fields}


def _matches_control(value: dict, kind: str, epoch: str, rank: int, **fields) -> bool:
    # Canonical comparison also rejects bool/int equivalence and extra fields.
    return hmac.compare_digest(
        protocol.encode_frame(value),
        protocol.encode_frame(_control(kind, epoch, rank, **fields)),
    )


def spawn_inert_worker(sock: socket.socket, fault: str) -> subprocess.Popen:
    """Fixed stdlib worker only. Secrets go over private FD, never argv/env/logs."""
    require_stdlib()
    _need(fault in FAULTS, "configuration_rejected")
    script = Path(__file__).resolve().parents[2] / "scripts" / "sft_synthetic_ipc_worker_v1.py"
    return subprocess.Popen(
        [sys.executable, "-S", "-B", str(script), str(sock.fileno()), fault],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        pass_fds=(sock.fileno(),),
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        start_new_session=True,
    )


def _cleanup(children: list[subprocess.Popen], sockets: list[socket.socket]) -> tuple[bool, int]:
    """Best effort across ALL handles; never surface raw OS exception text."""
    clean = True
    reaped = 0
    for sock in sockets:
        try:
            sock.close()
        except BaseException:
            clean = False
    for child in children:
        try:
            if child.poll() is None:
                child.kill()
        except ProcessLookupError:
            pass
        except BaseException:
            clean = False
    for child in children:
        try:
            child.wait(timeout=1)
            reaped += 1
        except BaseException:
            clean = False
    return clean, reaped


def _exchange(
    channel: Channel,
    session,
    rank: int,
    epoch: str,
    operation: str,
    arguments: dict,
    *,
    partial_ack: bool = False,
) -> None:
    channel.send(_control("command", epoch, rank, operation=operation, arguments=arguments))
    raw, _value = channel.receive()
    acknowledgements = session.feed(rank, raw)
    _need(len(acknowledgements) == 1, "ledger_rejected")
    ack = acknowledgements[0]
    failed = False
    try:
        if partial_ack:
            # Fixed synthetic fault, after durable predebit and before full ACK.
            partial = ack.frame[: len(ack.frame) // 2]
            channel._charge(len(partial))
            channel._time(time.monotonic() + MESSAGE_SECONDS)
            channel.sock.sendall(partial)
            raise ProbeError("ipc_failed")
        channel.send_raw(ack.frame)
        session.mark_ack_sent(ack)
        confirmation, _value = channel.receive()
        _need(session.feed(rank, confirmation) == (), "ledger_rejected")
        session.check_pending()
    except BaseException:
        failed = True
    if failed:
        session.ack_send_failed(rank)


def run_synthetic_probe(
    directory: Path, *, timeout_seconds: int = MAXIMUM_SECONDS, fault: str = "none"
) -> dict:
    """Run exactly 3 epochs/6 inert children, or stop at the first failed epoch.

    No retry, worker replacement within an epoch, data, ML, GPU, TCP/IP sockets,
    training dispatch or external authorization is accepted by this API.
    Reports contain aggregate counts, fixed state codes and explicit limitations.
    """
    require_stdlib()
    _need(type(directory) is Path or isinstance(directory, Path), "configuration_rejected")
    _need(
        type(timeout_seconds) is int and 1 <= timeout_seconds <= MAXIMUM_SECONDS,
        "configuration_rejected",
    )
    _need(type(fault) is str and fault in FAULTS, "configuration_rejected")
    deadline = time.monotonic() + timeout_seconds
    controller = DurableSupervisor(directory)
    children: list[subprocess.Popen] = []
    sockets: list[socket.socket] = []
    epochs = []
    seen = set()
    wire_budget = [0]
    failure = None
    spawned = 0
    reaped = 0
    clean = True
    try:
        for number, (run, start, stop, resume) in enumerate(supervision.SEQUENCE):
            _need(time.monotonic() < deadline, "deadline_exhausted")
            epoch = protocol.create_epoch()
            tokens = protocol.create_rank_tokens(2)
            fresh = {epoch, *tokens.values()}
            _need(len(fresh) == 3 and not fresh.intersection(seen), "ipc_authentication_rejected")
            seen.update(fresh)
            session = protocol.EpochLedgerSession(
                controller, world_size=2, rank_tokens=tokens, epoch=epoch
            )
            session.begin_invocation(run, start, stop, resume)
            epochs.append(session)
            channels = []
            for rank in range(2):
                parent, child_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
                sockets.append(parent)
                try:
                    child_fault = fault if number == 0 and rank == 0 else "none"
                    child = spawn_inert_worker(child_end, child_fault)
                    children.append(child)
                    spawned += 1
                finally:
                    child_end.close()
                channel = Channel(parent, deadline, wire_budget)
                channels.append(channel)
                channel.send(_control("initialize", epoch, rank, token=tokens[rank]))
            for rank, channel in enumerate(channels):
                _raw, hello = channel.receive()
                _need(
                    _matches_control(hello, "hello", epoch, rank, token=tokens[rank]),
                    "ipc_authentication_rejected",
                )
            for rank, channel in enumerate(channels):
                _exchange(
                    channel,
                    session,
                    rank,
                    epoch,
                    "reserve",
                    {"checkpoint_read_bytes": 1},
                    partial_ack=fault == "partial_ack" and number == 0 and rank == 0,
                )
            for step in range(start, stop):
                _exchange(
                    channels[0],
                    session,
                    0,
                    epoch,
                    "reserve_step",
                    {"from_step": step, "to_step": step + 1},
                )
                for rank, channel in enumerate(channels):
                    _exchange(channel, session, rank, epoch, "step_complete", {"to_step": step + 1})
            # Seal the protocol while children wait for stop. EOF before this point
            # is never silently postponed. Durable epoch success waits for cleanup.
            session.observe_memory(1, [0, 0])  # Synthetic, NOT a real RAM measurement.
            session.finish_invocation(stop)
            for rank, channel in enumerate(channels):
                channel.send(_control("stop", epoch, rank))
                _raw, bye = channel.receive()
                _need(_matches_control(bye, "bye", epoch, rank), "ipc_authentication_rejected")
                # A clean exit AND EOF are required; extra trailing traffic is rejected.
                channel._time(time.monotonic() + MESSAGE_SECONDS)
                trailing = channel.sock.recv(1)
                channel._charge(len(trailing))
                _need(trailing == b"", "ipc_frame_rejected")
                session.disconnect(rank)
            while children:
                child = children[0]
                remaining = deadline - time.monotonic()
                _need(remaining > 0, "deadline_exhausted")
                _need(child.wait(timeout=remaining) == 0, "worker_failed")
                children.pop(0)
                reaped += 1
            for sock in sockets:
                sock.close()
            sockets.clear()
            _need(time.monotonic() < deadline, "deadline_exhausted")
            controller.commit_retired_epoch()
            _need(time.monotonic() < deadline, "deadline_exhausted")
    except BaseException as error:
        failure = (
            error.code
            if isinstance(error, ProbeError)
            else (
                "ledger_rejected"
                if isinstance(error, protocol.LedgerProtocolError)
                else "synthetic_probe_failed"
            )
        )
        controller.interrupt()
    finally:
        # Kill/reap BEFORE any failure-journal fsync, which itself can stall.
        clean, newly_reaped = _cleanup(children, sockets)
        reaped += newly_reaped
        if not clean:
            failure = "worker_cleanup_failed"
            controller.interrupt()
        if controller.journal is not None:
            if failure is not None:
                try:
                    if controller.journal.summary()["terminal_event"] is None:
                        controller.journal.append("attempt_failed", ())
                except BaseException:
                    pass
            controller.journal.close()
            if controller.journal.summary()["failure_code"] is not None:
                failure = failure or "journal_rejected"
                controller.interrupt()
        if failure is None and time.monotonic() >= deadline:
            failure = "deadline_exhausted"
            controller.interrupt()
    summary = controller.supervisor.summary()
    return {
        "schema_version": "lumi.sft-local-ipc-synthetic-summary.v0.1",
        "state": "passed" if failure is None else "failed",
        "failure_code": failure,
        "worker_processes_started": spawned,
        "worker_processes_reaped": reaped,
        "worker_cleanup_complete": clean,
        "epochs_completed": controller.epochs_committed,
        "global_optimizer_steps_completed": summary["global_optimizer_steps_completed"],
        "requests_predebited": sum(item.summary()["requests_predebited"] for item in epochs),
        "acknowledgements_confirmed": sum(
            item.summary()["acknowledgements_confirmed"] for item in epochs
        ),
        "checkpoint_bytes_synthetically_reserved": summary["resources_reserved"][
            "checkpoint_read_bytes"
        ],
        "journal_capacity_predebited": summary["resources_reserved"]["private_write_bytes"],
        "journal_write_bytes_charged": controller._journal_charged,
        "local_ipc_wire_bytes_charged": wire_budget[0],
        "transport_implemented": True,
        "durable_predebit_journal": True,
        "fresh_process_epochs": True,
        "channel_capability_authentication": True,
        "os_peer_pid_authenticated": False,
        "real_memory_measurement": False,
        "os_memory_enforcement": False,
        "os_network_enforcement": False,
        "training_source_or_data_reads": 0,
        "external_network_bytes": 0,
        "gpu_use": 0,
        "inference_calls": 0,
        "real_sft_execution_permitted": False,
        "production_eligible": False,
    }


def synthetic_worker(fd: int, fault: str) -> int:
    """Standalone child's closed command loop: IPC only, never data or ML calls."""
    require_stdlib()
    _need(type(fd) is int and fd >= 3 and fault in FAULTS, "configuration_rejected")
    with socket.socket(fileno=fd) as sock:
        channel = Channel(sock, time.monotonic() + MAXIMUM_SECONDS, [0])
        _raw, initial = channel.receive()
        _need(
            set(initial) == {"schema_version", "kind", "epoch", "rank", "token"},
            "ipc_frame_rejected",
        )
        epoch, rank, token = initial["epoch"], initial["rank"], initial["token"]
        _need(type(rank) is int and rank in {0, 1}, "ipc_authentication_rejected")
        _need(
            all(
                type(value) is str
                and len(value) == 64
                and all(char in "0123456789abcdef" for char in value)
                for value in (epoch, token)
            ),
            "ipc_authentication_rejected",
        )
        _need(
            _matches_control(initial, "initialize", epoch, rank, token=token), "ipc_frame_rejected"
        )
        channel.send(_control("hello", epoch, rank, token=token))
        sequence = 0
        while True:
            _raw, command = channel.receive()
            if _matches_control(command, "stop", epoch, rank):
                channel.send(_control("bye", epoch, rank))
                if fault == "trailing_after_bye":
                    channel._charge(1)
                    sock.sendall(b"x")
                return 2 if fault == "bad_exit" else 0
            _need(
                set(command)
                == {"schema_version", "kind", "epoch", "rank", "operation", "arguments"},
                "ipc_frame_rejected",
            )
            operation, arguments = command["operation"], command["arguments"]
            _need(
                _matches_control(
                    command, "command", epoch, rank, operation=operation, arguments=arguments
                ),
                "ipc_frame_rejected",
            )
            _need(
                operation in {"reserve", "reserve_step", "step_complete"}
                and type(arguments) is dict,
                "ipc_frame_rejected",
            )
            sequence += 1
            request = {
                "schema_version": protocol.REQUEST_SCHEMA,
                "epoch": epoch,
                "rank": rank,
                "sequence": sequence,
                "token": token,
                "operation": operation,
                "arguments": arguments,
            }
            if fault == "wrong_epoch":
                request["epoch"] = "0" * 64 if epoch != "0" * 64 else "1" * 64
            if fault == "stall":
                # Wait for parent EOF or independent timeout; never spin or do work.
                channel.receive()
                return 2
            raw = protocol.encode_frame(request)
            if fault == "partial_request":
                sock.sendall(raw[:5])
                return 2
            channel.send_raw(raw)
            ack_raw, ack = channel.receive()
            expected = {
                "schema_version": protocol.ACK_SCHEMA,
                "epoch": epoch,
                "rank": rank,
                "sequence": sequence,
                "operation": operation,
                "status": "reserved",
            }
            _need(
                hmac.compare_digest(protocol.encode_frame(ack), protocol.encode_frame(expected)),
                "ipc_authentication_rejected",
            )
            if fault == "exit_before_confirmation":
                return 2
            # No synthetic work is considered admitted before this exact full ACK.
            confirmation = {
                **request,
                "epoch": epoch,
                "operation": "confirm_ack",
                "arguments": {"ack_sha256": hashlib.sha256(ack_raw).hexdigest()},
            }
            channel.send(confirmation)
