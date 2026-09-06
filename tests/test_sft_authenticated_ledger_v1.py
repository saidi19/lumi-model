"""Synthetic orchestration and mandatory Linux inert-worker integration cases."""

from __future__ import annotations

import functools
import hashlib
import importlib
import json
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_test_sft_authenticated_ledger_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
probe = importlib.import_module(PACKAGE + ".posttraining_sft_authenticated_ledger_v1")
NONCLAIMS = (
    "continuous_capability_exclusion", "privileged_sender_capabilities_excluded",
    "real_memory_measurement", "os_memory_enforcement", "os_network_enforcement",
    "namespace_containment", "ptrace_containment", "target_gpu_host_verified",
    "guarded_real_worker_ready", "execution_authorized", "real_sft_execution_permitted",
    "production_eligible", "full_private_integration_passed",
)


def isolated(method):
    @functools.wraps(method)
    def run(self):
        if sys.flags.no_site == 1 and sys.dont_write_bytecode:
            return method(self)
        result = subprocess.run(
            [sys.executable, "-S", "-B", str(Path(__file__).resolve()),
             type(self).__name__ + "." + method.__name__],
            stdin=subprocess.DEVNULL, capture_output=True, timeout=30,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )
        self.assertEqual(result.returncode, 0, (result.stdout + result.stderr)[-4000:].decode())
    return run


class Transcript:
    """Pure mock worker transcript; the actual frozen parent ledger/journal run."""

    def __init__(self, sock, **_kwargs):
        self._sock = sock
        self.queue = [len(probe.READY).to_bytes(4, "big") + probe.READY]
        self.verified = 0
        self.closed = self.close_attempted = False
        self.sequence = 0
        self.request = None
        sock.sendmsg.side_effect = self.sent
        sock.recvmsg.return_value = (b"", [], 0, None)

    def sent(self, buffers):
        raw = buffers[0]
        value = json.loads(raw[4:])
        kind = value.get("kind")
        if kind == "initialize":
            self.initial = value
            self.queue.append(probe.protocol.encode_frame({**value, "kind": "hello"}))
        elif kind == "command":
            self.sequence += 1
            self.request = {
                "schema_version": probe.protocol.REQUEST_SCHEMA,
                "epoch": value["epoch"], "rank": value["rank"], "sequence": self.sequence,
                "token": self.initial["token"], "operation": value["operation"],
                "arguments": value["arguments"],
            }
            self.queue.append(probe.protocol.encode_frame(self.request))
        elif value["schema_version"] == probe.protocol.ACK_SCHEMA:
            self.queue.append(probe.protocol.encode_frame({
                **self.request, "operation": "confirm_ack",
                "arguments": {"ack_sha256": hashlib.sha256(raw).hexdigest()},
            }))
        elif kind == "stop":
            self.queue.append(probe.protocol.encode_frame({**value, "kind": "bye"}))
        return len(raw)

    def receive(self):
        raw = self.queue.pop(0)
        self.verified += 1
        return raw

    def close(self):
        if self.close_attempted:
            if not self.closed:
                raise probe.credentials.PeerCredentialError("credential_cleanup_failed")
            return
        self.close_attempted = True
        self._sock.close()
        self.closed = True

    def summary(self):
        return {"verified_messages": self.verified, "received_descriptors_closed": 0}


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="lumi-authenticated-ledger-synthetic-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()
        self.directory.chmod(0o700)

    def boundaries(self, value):
        for key in NONCLAIMS:
            self.assertIs(value[key], False, key)
        for key in ("training_source_or_data_reads", "external_network_bytes", "gpu_use",
                    "inference_calls"):
            self.assertEqual(value[key], 0)
        self.assertLessEqual(value["accounting"]["receive_capacity_reservations"], 64)
        self.assertLessEqual(value["accounting"]["status_capacity_reservations"], 64)
        self.assertLessEqual(value["accounting"]["send_bytes_precharged"], 131072)
        self.assertLessEqual(value["journal_write_bytes_charged"], 65536)
        encoded = json.dumps(value)
        for secret in (str(self.directory), "SECRET", "Traceback", "NoNewPrivs", '"token"'):
            self.assertNotIn(secret, encoded)

    def fake_environment(self):
        self.owners, self.transcripts, self.endpoints = [], [], []
        self.enterContext(patch.object(probe.lifecycle, "require_runtime"))
        self.enterContext(patch.object(probe.lifecycle, "normalize_child_signal"))
        self.enterContext(patch.object(probe.socket, "MSG_CMSG_CLOEXEC", 1 << 30, create=True))
        self.clock = self.enterContext(patch.object(probe.time, "monotonic", return_value=100))
        reader = Mock(attempts=1, closed=1, cleanup_failed=False)
        reader.child.return_value = {"capability_bounding_set_empty": False}
        self.readers = self.enterContext(patch.object(probe.lifecycle, "StatusReads",
                                                     return_value=reader))

        def pair():
            parent, endpoint = Mock(), Mock()
            self.endpoints.append((parent, endpoint))
            return parent, endpoint

        def owner(child):
            item = Mock(child=child, fd=71, owned=True, bound_live=True, reaped=False,
                        closed=False, close_attempted=False, signal_attempts=0, assert_live=Mock())
            item.reap.side_effect = lambda *_args, **_kwargs: setattr(item, "reaped", True)
            item.terminate_and_reap.side_effect = lambda *_args: setattr(item, "reaped", True)

            def close():
                item.close_attempted = True
                item.closed = True
            item.close.side_effect = close
            self.owners.append(item)
            return item

        def channel(sock, **kwargs):
            item = Transcript(sock, **kwargs)
            self.transcripts.append(item)
            return item

        self.spawn = self.enterContext(patch.object(probe, "spawn", return_value=Mock(pid=12345)))
        self.owner_factory = self.enterContext(patch.object(probe.lifecycle, "OwnedChild",
                                                            side_effect=owner))
        self.pair = self.enterContext(patch.object(probe.credentials, "credential_pair",
                                                 side_effect=pair))
        self.channel_factory = self.enterContext(patch.object(
            probe.credentials, "CredentialChannel", side_effect=channel,
        ))

    def run_fake(self, **kwargs):
        value = probe.run_synthetic_probe(self.directory, **kwargs)
        self.boundaries(value)
        return value

    def assert_success(self, value):
        self.assertEqual(value["state"], "passed", value["failure_code"])
        self.assertTrue(value["worker_cleanup_complete"])
        self.assertTrue(value["authenticated_inert_ledger_probe_passed"])
        for key, expected in {
            "worker_processes_started": 6, "worker_processes_reaped": 6, "pidfds_bound_live": 6,
            "pidfds_closed": 6, "epochs_completed": 3, "global_optimizer_steps_completed": 4,
            "requests_predebited": 18, "acknowledgements_confirmed": 18,
            "credential_messages_verified": 54, "terminal_eof_verified": 6,
            "checkpoint_bytes_synthetically_reserved": 6, "journal_capacity_predebited": 65536,
            "journal_records_committed": 22, "received_descriptors_closed": 0,
        }.items():
            self.assertEqual(value[key], expected, key)
        self.assertEqual(value["journal_terminal_event"], "attempt_completed")
        self.assertTrue(value["journal_closed"])
        accounting = value["accounting"]
        self.assertEqual(accounting["receive_capacity_reservations"], 60)
        self.assertEqual(accounting["status_read_attempts"], 55)
        self.assertEqual(accounting["status_descriptors_closed"], 55)
        self.assertEqual(accounting["capability_snapshots_verified"], 54)
        self.assertEqual(accounting["status_capacity_bytes"], 55 * probe.lifecycle.STATUS_CAPACITY)
        raw = (self.directory / "attempt.journal").read_bytes()
        self.assertEqual(len(raw), value["journal_write_bytes_charged"])
        self.assertIsNotNone(probe.previous.journal_module.verify_bytes(raw))
        self.assertEqual((self.directory / "attempt.journal").stat().st_mode & 0o777, 0o600)
        for secret in (b"token", b"rank", b"pid", b"path", b"exception", b"NoNewPrivs"):
            self.assertNotIn(secret, raw)

    @isolated
    def test_mock_full_transcript_through_real_durable_ledger(self):
        self.fake_environment()
        self.assert_success(self.run_fake())
        for owner in self.owners:
            owner.bind.assert_called_once()
            owner.reap.assert_called_once()
            owner.close.assert_called_once()
            owner.child.wait.assert_not_called()
        for parent, endpoint in self.endpoints:
            parent.close.assert_called_once()
            endpoint.close.assert_called_once()

    @isolated
    def test_epoch_commit_waits_for_every_reap_eof_and_handle_close(self):
        self.fake_environment()
        original = probe.previous.DurableSupervisor.commit_retired_epoch

        def commit(controller):
            self.assertTrue(all(owner.reaped and owner.closed for owner in self.owners))
            self.assertTrue(all(item.closed for item in self.transcripts))
            self.assertTrue(all(parent.recvmsg.call_count == 1 for parent, _ in self.endpoints))
            original(controller)
        with patch.object(probe.previous.DurableSupervisor, "commit_retired_epoch", commit):
            self.assert_success(self.run_fake())

    @isolated
    def test_exclusive_existing_marker_denies_before_spawn_and_remains_unchanged(self):
        self.fake_environment()
        marker = self.directory / "attempt.journal"
        marker.write_bytes(b"synthetic consumed attempt")
        value = self.run_fake()
        self.assertEqual(value["state"], "failed")
        self.spawn.assert_not_called()
        self.assertEqual(marker.read_bytes(), b"synthetic consumed attempt")

    @isolated
    def test_ack_send_failure_preserves_debit_and_stops_before_second_epoch(self):
        self.fake_environment()
        value = self.run_fake(fault="ack_send_failure")
        self.assertEqual(value["state"], "failed")
        self.assertEqual(value["checkpoint_bytes_synthetically_reserved"], 1)
        self.assertEqual(value["requests_predebited"], 1)
        self.assertEqual(value["acknowledgements_confirmed"], 0)
        self.assertEqual(value["epochs_completed"], 0)
        self.assertEqual(value["worker_processes_started"], 2)
        self.assertEqual(value["journal_terminal_event"], "attempt_failed")

    @isolated
    def test_journal_reservation_fsync_failure_cannot_send_ack(self):
        self.fake_environment()
        original = probe.previous.journal_module.os.fsync
        calls = []

        def reject(descriptor):
            calls.append(1)
            # Initial file + directory, epoch start, then first resource reservation.
            if len(calls) == 4:
                raise OSError("SECRET")
            return original(descriptor)
        with patch.object(probe.previous.journal_module.os, "fsync", reject):
            value = self.run_fake()
        self.assertEqual(len(calls), 4)
        self.assertEqual(value["state"], "failed")
        self.assertEqual(value["checkpoint_bytes_synthetically_reserved"], 1)
        self.assertEqual(value["acknowledgements_confirmed"], 0)
        self.assertEqual(self.transcripts[0]._sock.sendmsg.call_count, 2)  # initialize, command

    @isolated
    def test_endpoint_close_failure_is_sticky_without_close_retry(self):
        self.fake_environment()
        original = self.pair.side_effect

        def pair():
            parent, endpoint = original()
            endpoint.close.side_effect = OSError("SECRET")
            return parent, endpoint
        self.pair.side_effect = pair
        value = self.run_fake()
        self.assertEqual(value["failure_code"], "cleanup_failed")
        self.assertFalse(value["worker_cleanup_complete"])
        self.assertEqual(value["epochs_completed"], 0)
        for parent, endpoint in self.endpoints:
            parent.close.assert_called_once()
            endpoint.close.assert_called_once()

    @isolated
    def test_channel_close_failure_blocks_commit_and_is_not_retried(self):
        self.fake_environment()
        original = self.pair.side_effect

        def pair():
            parent, endpoint = original()
            parent.close.side_effect = OSError("SECRET")
            return parent, endpoint
        self.pair.side_effect = pair
        value = self.run_fake()
        self.assertEqual(value["failure_code"], "cleanup_failed")
        self.assertEqual(value["epochs_completed"], 0)
        for parent, endpoint in self.endpoints:
            parent.close.assert_called_once()
            endpoint.close.assert_called_once()

    @isolated
    def test_pidfd_close_failure_blocks_commit_without_retry(self):
        self.fake_environment()
        original = self.owner_factory.side_effect

        def owner(child):
            item = original(child)

            def close():
                item.close_attempted = True
                raise OSError("SECRET")
            item.close.side_effect = close
            return item
        self.owner_factory.side_effect = owner
        value = self.run_fake()
        self.assertEqual(value["failure_code"], "cleanup_failed")
        self.assertEqual(value["epochs_completed"], 0)
        self.assertEqual(value["pidfds_closed"], 0)
        for owner in self.owners:
            owner.close.assert_called_once()

    @isolated
    def test_confirmation_descriptor_cleanup_error_cannot_be_hidden_by_ack_failure(self):
        self.fake_environment()
        original = Transcript.receive

        def receive(channel):
            if channel.verified == 3:
                raise probe.credentials.PeerCredentialError("credential_cleanup_failed")
            return original(channel)
        with patch.object(Transcript, "receive", receive):
            value = self.run_fake()
        self.assertEqual(value["failure_code"], "cleanup_failed")
        self.assertFalse(value["worker_cleanup_complete"])
        self.assertEqual(value["checkpoint_bytes_synthetically_reserved"], 1)
        self.assertEqual(value["acknowledgements_confirmed"], 0)

    @isolated
    def test_source_authentication_failure_precedes_journal_or_spawn(self):
        self.fake_environment()
        with patch.object(probe, "authenticated_source", side_effect=
                          probe.IntegrationError("source_rejected")):
            value = self.run_fake()
        self.assertEqual(value["failure_code"], "source_rejected")
        self.spawn.assert_not_called()
        self.assertEqual(list(self.directory.iterdir()), [])

    @isolated
    def test_channel_constructor_failure_keeps_both_endpoint_closes_owned(self):
        self.fake_environment()
        self.channel_factory.side_effect = OSError("SECRET")
        value = self.run_fake()
        self.assertEqual(value["state"], "failed")
        for parent, endpoint in self.endpoints:
            parent.close.assert_called_once()
            endpoint.close.assert_called_once()
        self.owners[0].terminate_and_reap.assert_called_once()

    @isolated
    def test_peer_constructor_failure_keeps_parent_close_owned(self):
        self.fake_environment()
        with patch.object(probe, "Peer", side_effect=MemoryError("SECRET")):
            value = self.run_fake()
        self.assertEqual(value["state"], "failed")
        for parent, endpoint in self.endpoints:
            parent.close.assert_called_once()
            endpoint.close.assert_called_once()

    @isolated
    def test_failed_pidfd_open_has_zero_pidfds_closed_and_no_reap_claim(self):
        self.fake_environment()
        original = self.owner_factory.side_effect

        def unbound(child):
            owner = original(child)
            owner.fd = None
            owner.owned = owner.bound_live = False
            owner.bind.side_effect = OSError("SECRET")
            return owner
        self.owner_factory.side_effect = unbound
        value = self.run_fake()
        self.assertEqual(value["failure_code"], "cleanup_failed")
        self.assertEqual(value["pidfds_closed"], 0)
        self.assertEqual(value["worker_processes_reaped"], 0)
        self.owners[0].child.wait.assert_called_once()
        self.owners[0].terminate_and_reap.assert_not_called()

    @isolated
    def test_late_final_deadline_does_not_turn_terminal_journal_into_pass(self):
        self.fake_environment()
        original = probe.previous.journal_module.AttemptJournal.close

        def late(journal):
            original(journal)
            self.clock.return_value = 121
        with patch.object(probe.previous.journal_module.AttemptJournal, "close", late):
            value = self.run_fake()
        self.assertEqual(value["journal_terminal_event"], "attempt_completed")
        self.assertEqual(value["failure_code"], "deadline_exhausted")
        self.assertFalse(value["authenticated_inert_ledger_probe_passed"])

    @isolated
    def test_journal_close_failure_denies_even_after_three_retired_epochs(self):
        self.fake_environment()
        original = probe.previous.journal_module.AttemptJournal.close

        def failed(journal):
            original(journal)  # No test-fixture descriptor leak.
            raise OSError("SECRET")
        with patch.object(probe.previous.journal_module.AttemptJournal, "close", failed):
            value = self.run_fake()
        self.assertEqual(value["epochs_completed"], 3)
        self.assertEqual(value["failure_code"], "journal_rejected")
        self.assertFalse(value["authenticated_inert_ledger_probe_passed"])

    @isolated
    def test_cleanup_precedes_failure_journal_append(self):
        self.fake_environment()
        original = probe.previous.journal_module.AttemptJournal.append

        def append(journal, event, counters=()):
            if event == "attempt_failed":
                self.assertTrue(all(owner.reaped and owner.closed for owner in self.owners))
                self.assertTrue(all(item.closed for item in self.transcripts))
            return original(journal, event, counters)
        with patch.object(probe.previous.journal_module.AttemptJournal, "append", append):
            self.assertEqual(self.run_fake(fault="ack_send_failure")["state"], "failed")

    @isolated
    def test_unsupported_gate_runs_before_any_read_spawn_or_journal(self):
        with (
            patch.object(probe.lifecycle, "require_runtime", side_effect=
                         probe.lifecycle.LifecycleError("lifecycle_unavailable")),
            patch.object(probe.Budget, "status") as status,
            patch.object(probe, "authenticated_source") as source,
            patch.object(probe, "spawn") as spawn,
            patch.object(probe.previous, "DurableSupervisor") as controller,
        ):
            value = self.run_fake()
        self.assertEqual(value["state"], "unsupported")
        for operation in (status, source, spawn, controller):
            operation.assert_not_called()
        self.assertEqual(list(self.directory.iterdir()), [])

    @isolated
    def test_invalid_options_are_rejected_before_runtime(self):
        with patch.object(probe.lifecycle, "require_runtime") as runtime:
            for options in ({"timeout_seconds": True}, {"timeout_seconds": 0},
                            {"timeout_seconds": 21}, {"fault": "SECRET"}):
                self.assertEqual(self.run_fake(**options)["failure_code"], "configuration_rejected")
            runtime.assert_not_called()

    def live(self, fault="none"):
        value = probe.run_synthetic_probe(self.directory, fault=fault,
                                          timeout_seconds=1 if fault == "stall" else 20)
        self.boundaries(value)
        if sys.platform != "linux":
            self.assertEqual(value["state"], "unsupported")
            self.assertEqual(value["accounting"]["status_read_attempts"], 0)
            self.assertEqual(value["worker_processes_started"], 0)
            self.assertEqual(list(self.directory.iterdir()), [])
            return None
        self.assertTrue(value["worker_cleanup_complete"], value["failure_code"])
        self.assertEqual(value["worker_processes_started"], value["worker_processes_reaped"])
        self.assertEqual(value["worker_processes_started"], value["pidfds_closed"])
        return value

    def failed_live(self, fault, debit):
        value = self.live(fault)
        if value is not None:
            self.assertEqual(value["state"], "failed")
            self.assertEqual(value["epochs_completed"], 0)
            self.assertEqual(value["worker_processes_started"], 2)
            self.assertEqual(value["checkpoint_bytes_synthetically_reserved"], debit)
            self.assertEqual(value["journal_terminal_event"], "attempt_failed")
            raw = (self.directory / "attempt.journal").read_bytes()
            self.assertNotIn(b'"event":"epoch_completed"', raw)
            self.assertNotIn(b'"event":"attempt_completed"', raw)
        return value

    @isolated
    def test_live_linux_three_epochs_or_nonlinux_denial(self):
        value = self.live()
        if value is not None:
            self.assert_success(value)

    @isolated
    def test_live_wrong_epoch_is_rejected_before_debit(self):
        self.failed_live("wrong_epoch", 0)

    @isolated
    def test_live_wrong_rank_is_rejected_before_debit(self):
        self.failed_live("wrong_rank", 0)

    @isolated
    def test_live_descriptor_rights_are_closed_and_rejected(self):
        value = self.failed_live("rights", 0)
        if value is not None:
            self.assertEqual(value["received_descriptors_closed"], 1)

    @isolated
    def test_live_ack_send_failure_does_not_refund(self):
        self.failed_live("ack_send_failure", 1)

    @isolated
    def test_live_exit_before_confirmation_does_not_refund(self):
        value = self.failed_live("exit_before_confirmation", 1)
        if value is not None:
            self.assertEqual(value["acknowledgements_confirmed"], 0)

    @isolated
    def test_live_stall_stops_without_replacement(self):
        start = time.monotonic()
        self.failed_live("stall", 0)
        self.assertLess(time.monotonic() - start, 5)

    @isolated
    def test_live_nonzero_exit_blocks_epoch_commit(self):
        self.failed_live("bad_exit", 2)

    @isolated
    def test_live_trailing_packet_after_bye_blocks_epoch_commit(self):
        self.failed_live("trailing_after_bye", 2)


if __name__ == "__main__":
    unittest.main()
