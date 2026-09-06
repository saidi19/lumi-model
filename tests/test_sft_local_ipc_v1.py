"""Real local child processes, fixed synthetic counters; no training/private inputs."""

from __future__ import annotations

import functools
import importlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_test_sft_local_ipc_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src" / "lumi")]
sys.modules[PACKAGE] = package
ipc = importlib.import_module(PACKAGE + ".posttraining_sft_local_ipc_v1")


def isolated(method):
    """Also execute (not skip) under general pytest, in a clean -S -B child."""

    @functools.wraps(method)
    def run(self):
        if sys.flags.no_site == 1 and sys.dont_write_bytecode:
            return method(self)
        completed = subprocess.run(
            [
                sys.executable,
                "-S",
                "-B",
                str(Path(__file__).resolve()),
                type(self).__name__ + "." + method.__name__,
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=30,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )
        self.assertEqual(completed.returncode, 0, "isolated synthetic IPC case failed")

    return run


class LocalIPCTests(unittest.TestCase):
    def setUp(self):
        # Resolve macOS /var alias before enforcing the journal's no-symlink walk.
        self.temporary = tempfile.TemporaryDirectory(prefix="lumi-synthetic-ipc-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.directory.chmod(0o700)

    def report(self, **kwargs):
        value = ipc.run_synthetic_probe(self.directory, **kwargs)
        self.assertFalse(value["production_eligible"])
        self.assertFalse(value["real_sft_execution_permitted"])
        self.assertFalse(value["os_peer_pid_authenticated"])
        self.assertFalse(value["real_memory_measurement"])
        self.assertFalse(value["os_memory_enforcement"])
        self.assertFalse(value["os_network_enforcement"])
        self.assertEqual(value["training_source_or_data_reads"], 0)
        self.assertEqual(value["external_network_bytes"], 0)
        self.assertEqual(value["gpu_use"], 0)
        self.assertEqual(value["inference_calls"], 0)
        self.assertTrue(value["worker_cleanup_complete"])
        self.assertEqual(value["worker_processes_started"], value["worker_processes_reaped"])
        self.assertLessEqual(value["journal_write_bytes_charged"], ipc.JOURNAL_BUDGET)
        self.assertLessEqual(value["local_ipc_wire_bytes_charged"], ipc.MAXIMUM_WIRE_BYTES)
        text = json.dumps(value)
        self.assertNotIn(str(self.directory), text)
        self.assertNotIn("Traceback", text)
        return value

    @isolated
    def test_three_fresh_epochs_one_cumulative_ledger_and_durable_journal(self):
        value = self.report()
        self.assertEqual(value["state"], "passed", value["failure_code"])
        self.assertEqual(value["epochs_completed"], 3)
        self.assertEqual(value["worker_processes_started"], 6)
        self.assertEqual(value["global_optimizer_steps_completed"], 4)
        self.assertEqual(value["requests_predebited"], 18)
        self.assertEqual(value["acknowledgements_confirmed"], 18)
        self.assertEqual(value["checkpoint_bytes_synthetically_reserved"], 6)
        self.assertEqual(value["journal_capacity_predebited"], ipc.JOURNAL_BUDGET)
        raw = (self.directory / "attempt.journal").read_bytes()
        self.assertEqual(len(raw), value["journal_write_bytes_charged"])
        self.assertEqual((self.directory / "attempt.journal").stat().st_mode & 0o777, 0o600)
        verified = ipc.journal_module.verify_bytes(raw)
        self.assertIsNotNone(verified)
        for secret_field in (b"token", b"rank", b"pid", b"path", b"exception"):
            self.assertNotIn(secret_field, raw)

    @isolated
    def test_existing_attempt_is_untouched_and_spawns_nothing(self):
        first = self.report()
        self.assertEqual(first["state"], "passed")
        prior = (self.directory / "attempt.journal").read_bytes()
        second = self.report()
        self.assertEqual(second["state"], "failed")
        self.assertEqual(second["worker_processes_started"], 0)
        self.assertEqual(prior, (self.directory / "attempt.journal").read_bytes())

    @isolated
    def test_partial_ack_stops_without_refund_or_second_epoch(self):
        value = self.report(fault="partial_ack")
        self.assertEqual(value["state"], "failed")
        self.assertEqual(value["epochs_completed"], 0)
        self.assertEqual(value["worker_processes_started"], 2)
        self.assertEqual(value["checkpoint_bytes_synthetically_reserved"], 1)
        self.assertEqual(value["acknowledgements_confirmed"], 0)

    @isolated
    def test_worker_exit_after_ack_preserves_predebit(self):
        value = self.report(fault="exit_before_confirmation")
        self.assertEqual(value["state"], "failed")
        self.assertEqual(value["epochs_completed"], 0)
        self.assertEqual(value["checkpoint_bytes_synthetically_reserved"], 1)
        self.assertEqual(value["acknowledgements_confirmed"], 0)

    @isolated
    def test_bad_epoch_is_rejected_before_resource_debit(self):
        value = self.report(fault="wrong_epoch")
        self.assertEqual(value["state"], "failed")
        self.assertEqual(value["checkpoint_bytes_synthetically_reserved"], 0)
        self.assertEqual(value["epochs_completed"], 0)

    @isolated
    def test_partial_request_cannot_advance_or_retry(self):
        value = self.report(fault="partial_request")
        self.assertEqual(value["state"], "failed")
        self.assertEqual(value["checkpoint_bytes_synthetically_reserved"], 0)
        self.assertEqual(value["worker_processes_started"], 2)
        prior = (self.directory / "attempt.journal").read_bytes()
        self.assertEqual(self.report()["worker_processes_started"], 0)
        self.assertEqual((self.directory / "attempt.journal").read_bytes(), prior)

    @isolated
    def test_stalled_worker_hits_independent_deadline_and_is_reaped(self):
        started = time.monotonic()
        value = self.report(fault="stall", timeout_seconds=1)
        self.assertEqual(value["state"], "failed")
        self.assertEqual(value["failure_code"], "deadline_exhausted")
        self.assertLess(time.monotonic() - started, 4)
        self.assertEqual(value["epochs_completed"], 0)

    @isolated
    def test_bad_exit_prevents_durable_epoch_success(self):
        value = self.report(fault="bad_exit")
        self.assertEqual(value["state"], "failed")
        self.assertEqual(value["epochs_completed"], 0)
        raw = (self.directory / "attempt.journal").read_bytes()
        self.assertNotIn(b'"event":"epoch_completed"', raw)
        self.assertNotIn(b'"event":"attempt_completed"', raw)
        self.assertIn(b'"event":"attempt_failed"', raw)

    @isolated
    def test_extra_byte_after_bye_is_charged_and_rejected(self):
        charges = []
        charge = ipc.Channel._charge

        def observe(channel, count):
            charges.append(count)
            return charge(channel, count)

        with patch.object(ipc.Channel, "_charge", observe):
            value = self.report(fault="trailing_after_bye")
        self.assertEqual(value["state"], "failed")
        self.assertEqual(value["epochs_completed"], 0)
        self.assertIn(1, charges)
        self.assertEqual(sum(charges), value["local_ipc_wire_bytes_charged"])

    @isolated
    def test_eof_follows_protocol_seal_and_durable_success_follows_all_reaps(self):
        spawned = []
        spawn = ipc.spawn_inert_worker
        disconnect = ipc.protocol.EpochLedgerSession.disconnect
        commit = ipc.DurableSupervisor.commit_retired_epoch

        def create(*args):
            child = spawn(*args)
            spawned.append(child)
            return child

        def eof(session, rank):
            self.assertEqual(session.summary()["state"], "completed")
            return disconnect(session, rank)

        def retire(controller):
            self.assertTrue(all(child.poll() is not None for child in spawned))
            return commit(controller)

        with (
            patch.object(ipc, "spawn_inert_worker", create),
            patch.object(ipc.protocol.EpochLedgerSession, "disconnect", eof),
            patch.object(ipc.DurableSupervisor, "commit_retired_epoch", retire),
        ):
            self.assertEqual(self.report()["state"], "passed")

    @isolated
    def test_journal_close_failure_cannot_report_pass(self):
        close = ipc.journal_module.AttemptJournal.close

        def fail_close(handle):
            close(handle)
            handle._failure = "journal_close_failed"

        with patch.object(ipc.journal_module.AttemptJournal, "close", fail_close):
            value = self.report()
        self.assertEqual(value["state"], "failed")
        self.assertEqual(value["failure_code"], "journal_rejected")

    @isolated
    def test_cleanup_attempts_every_handle_and_hides_os_error_text(self):
        calls = []

        class Broken:
            def close(self):
                calls.append("close")
                raise OSError("SYNTHETIC-PRIVATE-ERROR")

            def poll(self):
                calls.append("poll")
                return None

            def kill(self):
                calls.append("kill")
                raise OSError("SYNTHETIC-PRIVATE-ERROR")

            def wait(self, timeout):
                calls.append("wait")
                raise OSError("SYNTHETIC-PRIVATE-ERROR")

        self.assertEqual(ipc._cleanup([Broken(), Broken()], [Broken(), Broken()]), (False, 0))
        self.assertEqual(calls.count("close"), 2)
        self.assertEqual(calls.count("kill"), 2)
        self.assertEqual(calls.count("wait"), 2)

    @isolated
    def test_wrapper_error_does_not_retain_underlying_exception_context(self):
        controller = ipc.DurableSupervisor(self.directory)

        class Broken:
            def append(self, *args):
                raise OSError("SYNTHETIC-PRIVATE-ERROR")

        controller.journal = Broken()
        with self.assertRaises(ipc.ProbeError) as caught:
            controller._record("epoch_started")
        self.assertIsNone(caught.exception.__context__)
        self.assertEqual(str(caught.exception), "journal_rejected")

    @isolated
    def test_deadline_after_terminal_write_still_prevents_pass(self):
        commit = ipc.DurableSupervisor.commit_retired_epoch
        monotonic = time.monotonic
        expired = [False]

        def retire(controller):
            commit(controller)
            expired[0] = controller.epochs_committed == 3

        with (
            patch.object(ipc.DurableSupervisor, "commit_retired_epoch", retire),
            patch.object(ipc.time, "monotonic", lambda: monotonic() + (60 if expired[0] else 0)),
        ):
            value = self.report()
        self.assertEqual(value["state"], "failed")
        self.assertEqual(value["failure_code"], "deadline_exhausted")
        # A terminal record is evidence of work-stage completion, not overall success.
        self.assertIn(
            b'"event":"attempt_completed"', (self.directory / "attempt.journal").read_bytes()
        )

    @isolated
    def test_failure_record_is_written_after_worker_cleanup(self):
        children = []
        spawn = ipc.spawn_inert_worker
        append = ipc.journal_module.AttemptJournal.append

        def create(*args):
            child = spawn(*args)
            children.append(child)
            return child

        def record(handle, event, counters=()):
            if event == "attempt_failed":
                self.assertTrue(all(child.poll() is not None for child in children))
            return append(handle, event, counters)

        with (
            patch.object(ipc, "spawn_inert_worker", create),
            patch.object(ipc.journal_module.AttemptJournal, "append", record),
        ):
            self.assertEqual(self.report(fault="exit_before_confirmation")["state"], "failed")

    @isolated
    def test_worker_rejects_ack_for_wrong_epoch(self):
        send = ipc.Channel.send_raw

        def wrong_ack(channel, raw):
            message = ipc.protocol.FrameDecoder().feed(raw)[0]
            if message.get("schema_version") == ipc.protocol.ACK_SCHEMA:
                message["epoch"] = "0" * 64
                raw = ipc.protocol.encode_frame(message)
            return send(channel, raw)

        with patch.object(ipc.Channel, "send_raw", wrong_ack):
            value = self.report()
        self.assertEqual(value["state"], "failed")
        self.assertEqual(value["checkpoint_bytes_synthetically_reserved"], 1)
        self.assertEqual(value["acknowledgements_confirmed"], 0)

    @isolated
    def test_journal_fsync_happens_before_every_reservation_ack(self):
        events = []
        fsync = os.fsync
        send = ipc.Channel.send_raw

        def disk_sync(fd):
            result = fsync(fd)
            events.append("sync")
            return result

        def transmit(channel, raw):
            decoded = ipc.protocol.FrameDecoder().feed(raw)[0]
            if decoded.get("schema_version") == ipc.protocol.ACK_SCHEMA:
                events.append("ack")
                self.assertIn("sync", events[:-1])
                # step_complete ACK alone promises receipt, not optimizer completion.
                if decoded["operation"] in {"reserve", "reserve_step"}:
                    self.assertEqual(events[-2], "sync")
            return send(channel, raw)

        with (
            patch.object(ipc.journal_module.os, "fsync", disk_sync),
            patch.object(ipc.Channel, "send_raw", transmit),
        ):
            self.assertEqual(self.report()["state"], "passed")
        self.assertEqual(events.count("ack"), 18)

    @isolated
    def test_journal_failure_before_spawn_consumes_marker(self):
        with patch.object(
            ipc.journal_module.os, "fsync", side_effect=OSError("SYNTHETIC-PRIVATE-TEXT")
        ):
            value = self.report()
        self.assertEqual(value["state"], "failed")
        self.assertEqual(value["worker_processes_started"], 0)
        self.assertNotIn("SYNTHETIC-PRIVATE-TEXT", json.dumps(value))
        self.assertTrue((self.directory / "attempt.journal").exists())
        self.assertEqual(self.report()["worker_processes_started"], 0)

    @isolated
    def test_inherited_channel_eof_causes_child_exit(self):
        parent, child_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        child = ipc.spawn_inert_worker(child_end, "none")
        child_end.close()
        try:
            channel = ipc.Channel(parent, time.monotonic() + 5, [0])
            epoch, token = "a" * 64, "b" * 64
            channel.send(ipc._control("initialize", epoch, 0, token=token))
            _raw, value = channel.receive()
            self.assertTrue(ipc._matches_control(value, "hello", epoch, 0, token=token))
            parent.close()
            self.assertEqual(child.wait(timeout=3), 2)
        finally:
            parent.close()
            if child.poll() is None:
                child.kill()
            child.wait(timeout=2)

    @isolated
    def test_child_exec_has_no_secret_argv_or_environment_and_only_one_fd(self):
        popen = subprocess.Popen
        seen = []

        def spawn(argv, **kwargs):
            self.assertEqual(argv[1:3], ["-S", "-B"])
            self.assertEqual(Path(argv[3]).name, "sft_synthetic_ipc_worker_v1.py")
            self.assertEqual(set(kwargs["env"]), {"PATH", "LC_ALL"})
            self.assertEqual(len(kwargs["pass_fds"]), 1)
            self.assertTrue(kwargs["close_fds"])
            self.assertEqual(kwargs["stdout"], subprocess.DEVNULL)
            self.assertEqual(kwargs["stderr"], subprocess.DEVNULL)
            seen.append(1)
            return popen(argv, **kwargs)

        with patch.object(ipc.subprocess, "Popen", spawn):
            self.assertEqual(self.report()["state"], "passed")
        self.assertEqual(len(seen), 6)

    @isolated
    def test_only_af_unix_socketpairs_are_created(self):
        create = socket.socketpair
        families = []

        def pair(family, kind):
            families.append(family)
            return create(family, kind)

        with patch.object(ipc.socket, "socketpair", pair):
            self.assertEqual(self.report()["state"], "passed")
        self.assertEqual(families, [socket.AF_UNIX] * 6)

    @isolated
    def test_admission_and_bootstrap_real_worker_entries_remain_closed(self):
        admission = importlib.import_module(PACKAGE + ".posttraining_sft_worker_admission_v1")
        with self.assertRaises(admission.AdmissionDenied):
            admission.run_worker()
        completed = subprocess.run(
            [
                sys.executable,
                "-S",
                "-B",
                str(ROOT / "scripts/sft_guarded_bootstrap_v1.py"),
                "--execute",
            ],
            capture_output=True,
            timeout=5,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(json.loads(completed.stdout)["failure_code"], "execute_disabled")
        self.assertNotIn("lumi", sys.modules)
        for name in ("torch", "transformers", "numpy", "tokenizers", "safetensors"):
            self.assertNotIn(name, sys.modules)

    @isolated
    def test_invalid_configuration_creates_nothing(self):
        for timeout in (0, True, 21):
            with self.assertRaises(ipc.ProbeError):
                ipc.run_synthetic_probe(self.directory, timeout_seconds=timeout)
        with self.assertRaises(ipc.ProbeError):
            ipc.run_synthetic_probe(self.directory, fault="arbitrary-command")
        self.assertEqual(list(self.directory.iterdir()), [])

    @isolated
    def test_partial_frame_and_wire_budget_are_bounded(self):
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            channel = ipc.Channel(parent, time.monotonic() + 2, [0])
            child.sendall((2**32 - 1).to_bytes(4, "big"))
            with self.assertRaises(ipc.ProbeError):
                channel.receive()
            channel.budget[0] = ipc.MAXIMUM_WIRE_BYTES
            with self.assertRaises(ipc.ProbeError):
                channel.send({"synthetic": True})
        finally:
            parent.close()
            child.close()


if __name__ == "__main__":
    unittest.main()
