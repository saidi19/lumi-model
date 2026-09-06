"""Pure mocked authenticated-ledger boundaries; no worker, socket or private I/O."""

from __future__ import annotations

import hashlib
import importlib
import io
import struct
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_test_sft_authenticated_ledger_primitives_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
probe = importlib.import_module(PACKAGE + ".posttraining_sft_authenticated_ledger_v1")
EPOCH = "a" * 64
TOKENS = {0: "b" * 64, 1: "c" * 64}


class Base(unittest.TestCase):
    def setUp(self):
        self.clock = self.enterContext(patch.object(probe.time, "monotonic", return_value=10))
        self.enterContext(patch.object(probe.socket, "MSG_CMSG_CLOEXEC", 0x40000000, create=True))
        self.enterContext(patch.object(probe.socket, "SCM_CREDENTIALS", 2, create=True))

    def assert_code(self, code, action):
        with self.assertRaises(probe.IntegrationError) as caught:
            action()
        self.assertEqual(caught.exception.code, code)

    def peer(self):
        owner = Mock(owned=True, reaped=False, close_attempted=False, closed=False,
                     assert_live=Mock())
        channel = Mock()
        channel._sock.sendmsg.side_effect = lambda parts: len(parts[0])
        budget = probe.Budget()
        budget.status = Mock()
        return probe.Peer(owner, channel, Mock(), 0, EPOCH, budget, 30)


class BudgetTests(Base):
    def test_all_64_receive_capacities_are_cumulative_and_exhaustion_never_refunds(self):
        budget = probe.Budget()
        for _epoch in range(4):
            for _rank in range(2):
                for _packet in range(8):
                    budget.receive()
        for _attempt in range(2):
            self.assert_code("budget_exhausted", budget.receive)
        self.assertEqual(budget.receives, 64)
        self.assertEqual(budget.summary()["receive_capacity_bytes"],
                         64 * probe.credentials.RECEIVE_CAPACITY_BYTES)

    def test_send_size_and_aggregate_budget_are_strict_and_cumulative(self):
        budget = probe.Budget()
        for value in (True, False, 0, -1, 1.0, "5", probe.credentials.MAXIMUM_FRAME_BYTES + 1):
            with self.subTest(value=value):
                self.assert_code("frame_rejected", lambda value=value: budget.send(value))
        self.assertEqual(budget.send_bytes, 0)
        while budget.send_bytes < probe.MAXIMUM_SEND_BYTES:
            budget.send(min(probe.credentials.MAXIMUM_FRAME_BYTES,
                            probe.MAXIMUM_SEND_BYTES - budget.send_bytes))
        self.assert_code("budget_exhausted", lambda: budget.send(1))
        self.assertEqual(budget.send_bytes, probe.MAXIMUM_SEND_BYTES)

    def test_status_capacity_is_precharged_across_64_fresh_helpers(self):
        budget = probe.Budget()
        owner = Mock()
        readers = []

        def factory():
            self.assertEqual(budget.status_reservations, len(readers) + 1)
            reader = Mock(attempts=1, closed=1, cleanup_failed=False)
            reader.child.return_value = {"capability_bounding_set_empty": len(readers) % 2 == 0}
            readers.append(reader)
            return reader

        with patch.object(probe.lifecycle, "StatusReads", side_effect=factory) as create:
            budget.status(deadline=30)
            for _ in range(63):
                budget.status(owner, deadline=30)
            self.assert_code("budget_exhausted", lambda: budget.status(owner, deadline=30))
        self.assertEqual(create.call_count, 64)
        readers[0].controller.assert_called_once_with()
        readers[0].child.assert_not_called()
        for reader in readers[1:]:
            reader.child.assert_called_once_with(owner, 30)
            reader.controller.assert_not_called()
        self.assertEqual((budget.status_reservations, budget.status_attempts, budget.status_closed),
                         (64, 64, 64))
        self.assertEqual((budget.snapshots, budget.empty_bounding_snapshots), (63, 31))

    def test_failed_status_keeps_reserved_capacity_and_actual_cleanup_counts(self):
        budget = probe.Budget()
        bad = Mock(attempts=1, closed=0, cleanup_failed=True)
        bad.child.side_effect = probe.lifecycle.LifecycleError("lifecycle_cleanup_failed")
        good = Mock(attempts=1, closed=1, cleanup_failed=False)
        good.child.return_value = {"capability_bounding_set_empty": True}
        with patch.object(probe.lifecycle, "StatusReads", side_effect=[bad, good]):
            with self.assertRaises(probe.lifecycle.LifecycleError):
                budget.status(Mock(), deadline=30)
            self.assertEqual((budget.status_reservations, budget.status_attempts), (1, 1))
            budget.status(Mock(), deadline=30)
        self.assertEqual((budget.status_reservations, budget.status_attempts, budget.status_closed),
                         (2, 2, 1))
        self.assertTrue(budget.status_cleanup_failed)
        self.assertEqual((budget.snapshots, budget.empty_bounding_snapshots), (1, 1))

    def test_expired_before_status_does_not_reserve_or_create_reader(self):
        budget = probe.Budget()
        with patch.object(probe.lifecycle, "StatusReads") as create:
            self.assert_code("deadline_exhausted", lambda: budget.status(deadline=10))
        create.assert_not_called()
        self.assertEqual(budget.status_reservations, 0)

    def test_expired_after_status_keeps_the_completed_attempt_and_reservation(self):
        budget = probe.Budget()
        reader = Mock(attempts=1, closed=1, cleanup_failed=False)
        self.clock.side_effect = [10, 31]
        with patch.object(probe.lifecycle, "StatusReads", return_value=reader):
            self.assert_code("deadline_exhausted", lambda: budget.status(deadline=30))
        self.assertEqual((budget.status_reservations, budget.status_attempts, budget.status_closed),
                         (1, 1, 1))


class PeerTests(Base):
    def test_receive_requires_live_identity_credentials_and_snapshot_in_that_order(self):
        peer = self.peer()
        events = []
        peer.owner.assert_live.side_effect = lambda: events.append("live")
        peer.channel.receive.side_effect = lambda: events.append("credentials") or b"frame"
        peer.budget.status.side_effect = lambda *args, **kwargs: events.append("status")
        original = peer.budget.receive

        def reserve():
            events.append("precharge")
            original()

        with patch.object(peer.budget, "receive", side_effect=reserve):
            self.assertEqual(peer.receive(), b"frame")
        self.assertEqual(events, ["live", "precharge", "credentials", "live", "status", "live"])
        self.assertEqual(peer.budget.receives, 1)
        peer.budget.status.assert_called_once_with(peer.owner, deadline=30)
        peer.session.feed.assert_not_called()

    def test_failed_identity_or_credentials_or_status_never_reaches_ledger(self):
        for stage in ("before", "credentials", "after", "status", "final"):
            with self.subTest(stage=stage):
                peer = self.peer()
                error = probe.lifecycle.LifecycleError("lifecycle_child_exited")
                if stage in {"before", "after", "final"}:
                    peer.owner.assert_live.side_effect = {
                        "before": [error], "after": [None, error], "final": [None, None, error],
                    }[stage]
                elif stage == "credentials":
                    peer.channel.receive.side_effect = probe.credentials.PeerCredentialError(
                        "credential_mismatch",
                    )
                else:
                    peer.budget.status.side_effect = error
                with self.assertRaises((probe.lifecycle.LifecycleError,
                                        probe.credentials.PeerCredentialError)):
                    peer.receive()
                peer.session.feed.assert_not_called()
                self.assertEqual(peer.budget.receives, int(stage != "before"))
                if stage in {"before", "credentials", "after"}:
                    peer.budget.status.assert_not_called()

    def test_receive_budget_exhaustion_happens_before_credential_syscall(self):
        peer = self.peer()
        peer.budget.receives = 64
        self.assert_code("budget_exhausted", peer.receive)
        peer.channel.receive.assert_not_called()
        peer.budget.status.assert_not_called()

    def test_send_rejects_malformed_frame_before_any_transport_or_precharge(self):
        for frame in (b"", b"1234", b"\0\0\0\x02x", bytearray(b"\0\0\0\x01x"),
                      b"\0\0\x10\x01" + b"x" * 4097):
            with self.subTest(length=len(frame)):
                peer = self.peer()
                self.assert_code("frame_rejected",
                                 lambda peer=peer, frame=frame: peer.send_raw(frame))
                peer.owner.assert_live.assert_not_called()
                peer.channel._sock.sendmsg.assert_not_called()
                self.assertEqual(peer.budget.send_bytes, 0)

    def test_failed_send_keeps_its_prepaid_bytes(self):
        raw = b"\0\0\0\x01x"
        for failure in (OSError("synthetic"), 0):
            with self.subTest(failure=type(failure).__name__):
                peer = self.peer()
                if isinstance(failure, BaseException):
                    peer.channel._sock.sendmsg.side_effect = failure
                    expected = OSError
                else:
                    peer.channel._sock.sendmsg.side_effect = None
                    peer.channel._sock.sendmsg.return_value = failure
                    expected = probe.IntegrationError
                with self.assertRaises(expected):
                    peer.send_raw(raw)
                self.assertEqual(peer.budget.send_bytes, len(raw))
                peer.channel._sock.sendmsg.assert_called_once_with([raw])

    def transaction(self):
        peer = self.peer()
        events = []
        controller = Mock(_world_size=2)
        controller.reserve.side_effect = lambda *args, **kwargs: events.append("durable_predebit")
        session = probe.protocol.EpochLedgerSession(
            controller, world_size=2, rank_tokens=TOKENS, epoch=EPOCH,
        )
        session.begin_invocation("synthetic", 0, 1, False)
        peer.session = session
        request = {
            "schema_version": probe.protocol.REQUEST_SCHEMA, "epoch": EPOCH, "rank": 0,
            "sequence": 1, "token": TOKENS[0], "operation": "reserve",
            "arguments": {"checkpoint_read_bytes": 1},
        }
        ack = probe.protocol.encode_frame({
            "schema_version": probe.protocol.ACK_SCHEMA, "epoch": EPOCH, "rank": 0,
            "sequence": 1, "operation": "reserve", "status": "reserved",
        })
        confirmation = {**request, "operation": "confirm_ack",
                        "arguments": {"ack_sha256": hashlib.sha256(ack).hexdigest()}}
        frames = iter([probe.protocol.encode_frame(request),
                       probe.protocol.encode_frame(confirmation)])

        def receive():
            events.append("credentials")
            return next(frames)

        def send(parts):
            value = probe.protocol.FrameDecoder().feed(parts[0])[0]
            events.append("send_" + value.get("kind", "ack"))
            return len(parts[0])

        peer.channel.receive.side_effect = receive
        peer.channel._sock.sendmsg.side_effect = send
        peer.budget.status.side_effect = lambda *args, **kwargs: events.append("status")
        return peer, controller, events, request

    def test_predebit_precedes_ack_and_confirmed_ack_precedes_continue(self):
        peer, controller, events, _request = self.transaction()
        with patch.object(peer.session, "mark_ack_sent", wraps=peer.session.mark_ack_sent) as mark:
            peer.exchange("reserve", {"checkpoint_read_bytes": 1})
        self.assertEqual(events, [
            "send_command", "credentials", "status", "durable_predebit", "send_ack",
            "credentials", "status", "send_continue",
        ])
        mark.assert_called_once()
        controller.reserve.assert_called_once_with(0, checkpoint_read_bytes=1)
        summary = peer.session.summary()
        self.assertEqual(tuple(summary[key] for key in (
            "requests_predebited", "acknowledgements_sent", "acknowledgements_confirmed",
            "pending_acknowledgements",
        )), (1, 1, 1, 0))
        self.assertEqual(peer.budget.receives, 2)

    def test_ack_send_failure_is_sticky_and_does_not_refund_accepted_reservation(self):
        peer, controller, events, request = self.transaction()
        with self.assertRaises(probe.IntegrationError) as caught:
            peer.exchange("reserve", {"checkpoint_read_bytes": 1}, ack_send_failure=True)
        self.assertEqual(caught.exception.code, "transport_failed")
        self.assertEqual(events, ["send_command", "credentials", "status", "durable_predebit"])
        before = peer.session.summary()
        with self.assertRaises(probe.protocol.LedgerProtocolError) as repeated:
            peer.session.feed(0, probe.protocol.encode_frame(request))
        self.assertEqual(repeated.exception.code, "ack_send_failed")
        self.assertEqual(peer.session.summary(), before)
        self.assertEqual((before["requests_predebited"], before["acknowledgements_sent"],
                          before["pending_acknowledgements"]), (1, 0, 1))
        controller.reserve.assert_called_once_with(0, checkpoint_read_bytes=1)
        controller.interrupt.assert_called_once_with()
        self.assertEqual(peer.budget.receives, 1)

    def test_bad_confirmation_fails_without_continue_and_keeps_debit(self):
        peer, controller, events, request = self.transaction()
        bad = {**request, "operation": "confirm_ack", "arguments": {"ack_sha256": "0" * 64}}
        peer.channel.receive.side_effect = [probe.protocol.encode_frame(request),
                                            probe.protocol.encode_frame(bad)]
        with self.assertRaises(probe.IntegrationError) as caught:
            peer.exchange("reserve", {"checkpoint_read_bytes": 1})
        self.assertEqual(caught.exception.code, "ack_rejected")
        self.assertNotIn("send_continue", events)
        self.assertEqual(peer.session.summary()["requests_predebited"], 1)
        self.assertEqual(peer.session.summary()["acknowledgements_confirmed"], 0)
        controller.reserve.assert_called_once_with(0, checkpoint_read_bytes=1)

    def test_request_credential_or_status_failure_cannot_reach_real_epoch_ledger(self):
        for stage in ("credentials", "status"):
            with self.subTest(stage=stage):
                peer, controller, _events, _request = self.transaction()
                if stage == "credentials":
                    peer.channel.receive.side_effect = probe.credentials.PeerCredentialError(
                        "credential_mismatch",
                    )
                else:
                    peer.budget.status.side_effect = probe.lifecycle.LifecycleError(
                        "lifecycle_child_exited",
                    )
                with self.assertRaises((probe.credentials.PeerCredentialError,
                                        probe.lifecycle.LifecycleError)):
                    peer.exchange("reserve", {"checkpoint_read_bytes": 1})
                controller.reserve.assert_not_called()
                self.assertEqual(peer.session.summary()["requests_predebited"], 0)
                self.assertEqual(peer.budget.receives, 1)

    def test_confirmation_cleanup_failure_stays_visible_without_refund_or_continue(self):
        peer, controller, events, request = self.transaction()
        peer.channel.receive.side_effect = [
            probe.protocol.encode_frame(request),
            probe.credentials.PeerCredentialError("credential_cleanup_failed"),
        ]
        self.assert_code("credential_cleanup_failed",
                         lambda: peer.exchange("reserve", {"checkpoint_read_bytes": 1}))
        self.assertEqual(peer.session.summary()["failure_code"], "ack_send_failed")
        self.assertEqual(peer.session.summary()["requests_predebited"], 1)
        self.assertEqual(peer.session.summary()["acknowledgements_confirmed"], 0)
        controller.reserve.assert_called_once_with(0, checkpoint_read_bytes=1)
        self.assertNotIn("send_continue", events)


class TerminalTests(Base):
    def ready(self):
        peer = self.peer()
        peer.sealed = True
        peer.owner.reaped = True
        peer.channel._sock.recvmsg.return_value = (b"", [], 0, None)
        return peer

    def test_eof_requires_both_sealed_session_and_real_reaped_owner(self):
        for sealed, reaped in ((False, False), (False, True), (True, False)):
            with self.subTest(sealed=sealed, reaped=reaped):
                peer = self.ready()
                peer.sealed, peer.owner.reaped = sealed, reaped
                self.assert_code("terminal_rejected", peer.terminal_eof)
                self.assertFalse(peer.terminal_attempted)
                self.assertEqual(peer.budget.receives, 0)
                peer.channel._sock.recvmsg.assert_not_called()

    def test_clean_eof_is_one_shot_and_only_then_disconnects_retired_rank(self):
        for flags in (0, probe.socket.MSG_CMSG_CLOEXEC):
            with self.subTest(flags=flags):
                peer = self.ready()
                peer.channel._sock.recvmsg.return_value = (b"", [], flags, None)
                peer.terminal_eof()
                self.assertTrue(peer.terminal_checked)
                self.assertTrue(peer.terminal_attempted)
                self.assertEqual(peer.budget.receives, 1)
                peer.session.disconnect.assert_called_once_with(0)
                self.assert_code("terminal_rejected", peer.terminal_eof)
                peer.channel._sock.recvmsg.assert_called_once_with(
                    probe.credentials.RECEIVE_DATA_CAPACITY,
                    probe.credentials.ANCILLARY_BUFFER_BYTES, probe.socket.MSG_CMSG_CLOEXEC,
                )
                peer.owner.assert_live.assert_not_called()
                peer.budget.status.assert_not_called()

    def test_empty_credential_packet_data_and_unexpected_flags_are_not_eof(self):
        credential = (probe.socket.SOL_SOCKET, probe.socket.SCM_CREDENTIALS,
                      struct.pack("iII", 12345, 1000, 1000))
        for raw, ancillary, flags in ((b"", [credential], 0), (b"x", [], 0),
                                      (b"", [], probe.socket.MSG_TRUNC), (b"", [], True)):
            with self.subTest(raw_length=len(raw), ancillary_count=len(ancillary), flags=flags):
                peer = self.ready()
                peer.channel._sock.recvmsg.return_value = (raw, ancillary, flags, None)
                self.assert_code("terminal_rejected", peer.terminal_eof)
                self.assert_code("terminal_rejected", peer.terminal_eof)
                self.assertEqual(peer.budget.receives, 1)
                self.assertFalse(peer.terminal_checked)
                peer.channel._sock.recvmsg.assert_called_once()
                peer.session.disconnect.assert_not_called()

    def test_rights_are_all_closed_before_flags_or_packet_rejection(self):
        peer = self.ready()
        ancillary = [(probe.socket.SOL_SOCKET, probe.socket.SCM_RIGHTS, struct.pack("ii", 71, 72))]
        peer.channel._sock.recvmsg.return_value = (b"x", ancillary, probe.socket.MSG_CTRUNC, None)
        with patch.object(probe.credentials.os, "close") as close:
            self.assert_code("terminal_rejected", peer.terminal_eof)
        self.assertEqual(close.call_args_list, [call(71), call(72)])
        self.assertEqual(peer.terminal_descriptors_closed, 2)
        peer.session.disconnect.assert_not_called()

    def test_failed_descriptor_close_does_not_skip_later_rights_or_retry(self):
        peer = self.ready()
        ancillary = [(probe.socket.SOL_SOCKET, probe.socket.SCM_RIGHTS,
                      struct.pack("iii", 71, 72, 71))]
        peer.channel._sock.recvmsg.return_value = (b"", ancillary, probe.socket.MSG_CTRUNC, None)
        with patch.object(probe.credentials.os, "close", side_effect=[OSError(), None]) as close:
            self.assert_code("cleanup_failed", peer.terminal_eof)
            self.assert_code("terminal_rejected", peer.terminal_eof)
        self.assertEqual(close.call_args_list, [call(71), call(72)])
        self.assertEqual(peer.terminal_descriptors_closed, 1)

    def test_receive_exception_is_consumed_once_without_disconnect(self):
        peer = self.ready()
        peer.channel._sock.recvmsg.side_effect = OSError("synthetic")
        with self.assertRaises(OSError):
            peer.terminal_eof()
        self.assert_code("terminal_rejected", peer.terminal_eof)
        self.assertEqual(peer.budget.receives, 1)
        peer.channel._sock.recvmsg.assert_called_once()
        peer.session.disconnect.assert_not_called()

    def test_terminal_budget_failure_is_sticky_and_does_not_read(self):
        peer = self.ready()
        peer.budget.receives = 64
        self.assert_code("budget_exhausted", peer.terminal_eof)
        self.assert_code("terminal_rejected", peer.terminal_eof)
        self.assertEqual(peer.budget.receives, 64)
        peer.channel._sock.recvmsg.assert_not_called()

    def test_terminal_deadline_failure_keeps_capacity_but_never_reads(self):
        peer = self.ready()
        self.clock.return_value = 30
        self.assert_code("deadline_exhausted", peer.terminal_eof)
        self.assert_code("terminal_rejected", peer.terminal_eof)
        self.assertEqual(peer.budget.receives, 1)
        peer.channel._sock.recvmsg.assert_not_called()

    def test_delivered_rights_are_closed_before_post_receive_deadline_failure(self):
        peer = self.ready()
        ancillary = [(probe.socket.SOL_SOCKET, probe.socket.SCM_RIGHTS, struct.pack("ii", 71, 72))]
        peer.channel._sock.recvmsg.return_value = (b"", ancillary, probe.socket.MSG_CTRUNC, None)
        self.clock.side_effect = [10, 30]
        with patch.object(probe.credentials.os, "close") as close:
            self.assert_code("deadline_exhausted", peer.terminal_eof)
        self.assertEqual(close.call_args_list, [call(71), call(72)])
        self.assertEqual(peer.terminal_descriptors_closed, 2)
        peer.session.disconnect.assert_not_called()

    def test_failed_retired_disconnect_cannot_report_terminal_success_or_retry(self):
        peer = self.ready()
        peer.session.disconnect.side_effect = probe.protocol.LedgerProtocolError(
            "rank_disconnected",
        )
        with self.assertRaises(probe.protocol.LedgerProtocolError):
            peer.terminal_eof()
        self.assert_code("terminal_rejected", peer.terminal_eof)
        self.assertFalse(peer.terminal_checked)
        peer.session.disconnect.assert_called_once_with(0)
        peer.channel._sock.recvmsg.assert_called_once()


class CleanupAndSourceTests(Base):
    def channel(self, sock):
        with patch.object(probe.credentials, "require_linux"), \
                patch.object(probe.credentials, "_socket_ready"):
            return probe.credentials.CredentialChannel(
                sock, expected_credentials=(12345, 1000, 1000), deadline=30,
            )

    def owner(self, descriptor, *, reaped=True):
        owner = probe.lifecycle.OwnedChild(Mock(pid=12345))
        owner.fd = descriptor
        owner.owned = True
        owner.reaped = reaped
        return owner

    def test_cleanup_attempts_every_handle_despite_failures_and_never_retries_close(self):
        raw_bad, raw_good = Mock(), Mock()
        raw_bad.close.side_effect = OSError("synthetic")
        sockets = [raw_bad, raw_good]
        bad_sock, good_sock = Mock(), Mock()
        bad_sock.close.side_effect = OSError("synthetic")
        peers = [Mock(channel=self.channel(bad_sock)), Mock(channel=self.channel(good_sock))]
        owners = [self.owner(71), self.owner(72)]
        with patch.object(probe.lifecycle.os, "close", side_effect=[OSError(), None]) as close:
            self.assertFalse(probe.cleanup(owners, peers, sockets))
            self.assertFalse(probe.cleanup(owners, peers, sockets))
        self.assertEqual(sockets, [])
        for sock in (raw_bad, raw_good, bad_sock, good_sock):
            sock.close.assert_called_once_with()
        self.assertEqual(close.call_args_list, [call(71), call(72)])
        self.assertTrue(owners[0].close_attempted)
        self.assertFalse(owners[0].closed)
        self.assertTrue(owners[1].closed)

    def test_cleanup_shares_deadline_and_closes_owned_handles_after_reap_errors(self):
        first, second = self.owner(71, reaped=False), self.owner(72, reaped=False)
        first.terminate_and_reap = Mock(side_effect=probe.lifecycle.LifecycleError(
            "lifecycle_deadline_exhausted",
        ))
        second.terminate_and_reap = Mock(
            side_effect=lambda deadline: setattr(second, "reaped", True),
        )
        with patch.object(probe.lifecycle.os, "close") as close:
            self.assertFalse(probe.cleanup([first, second], [], []))
        first.terminate_and_reap.assert_called_once_with(13)
        second.terminate_and_reap.assert_called_once_with(13)
        self.assertEqual(close.call_args_list, [call(71), call(72)])

    def test_unbound_popen_wait_is_best_effort_not_owned_reap_proof(self):
        owner = probe.lifecycle.OwnedChild(Mock(pid=12345))
        self.assertFalse(probe.cleanup([owner], [], []))
        owner.child.wait.assert_called_once_with(timeout=3)
        self.assertFalse(owner.owned)
        self.assertFalse(owner.reaped)
        self.assertTrue(owner.closed)
        owner.child.kill.assert_not_called()

    def test_fixed_public_worker_hash_and_spawn_use_authenticated_bytes_only(self):
        source = probe.authenticated_source()
        self.assertLessEqual(len(source), probe.MAXIMUM_SOURCE_BYTES)
        self.assertEqual(hashlib.sha256(source).hexdigest(), probe.WORKER_SHA256)
        endpoint = Mock(fileno=Mock(return_value=71))
        with patch.object(probe.subprocess, "Popen") as popen, patch.object(Path, "open") as reopen:
            probe.spawn(endpoint, "none", source)
        reopen.assert_not_called()
        args, kwargs = popen.call_args
        self.assertEqual(args[0], [sys.executable, "-S", "-B", "-c", source, "71", "none"])
        self.assertEqual(kwargs["pass_fds"], (71,))
        self.assertTrue(kwargs["close_fds"])
        self.assertTrue(kwargs["start_new_session"])
        self.assertEqual(kwargs["env"], {"PATH": "/usr/bin:/bin", "LC_ALL": "C"})

    def test_changed_or_oversized_worker_source_is_rejected_after_one_bounded_read(self):
        for raw in (b"changed", b"x" * (probe.MAXIMUM_SOURCE_BYTES + 1)):
            with self.subTest(size=len(raw)):
                stream = Mock(wraps=io.BytesIO(raw))
                stream.__enter__ = Mock(return_value=stream)
                stream.__exit__ = Mock(return_value=False)
                with patch.object(Path, "open", return_value=stream) as opened:
                    self.assert_code("source_rejected", probe.authenticated_source)
                opened.assert_called_once_with("rb")
                stream.read.assert_called_once_with(probe.MAXIMUM_SOURCE_BYTES + 1)

    def test_invalid_spawn_source_or_fault_cannot_execute_a_process(self):
        endpoint = Mock()
        with patch.object(probe.subprocess, "Popen") as popen:
            for fault in ("unknown", "ack_send_failure"):
                self.assert_code("configuration_rejected",
                                 lambda fault=fault: probe.spawn(endpoint, fault, b"x"))
            for source in (b"changed", "not bytes", bytearray(b"x")):
                self.assert_code("source_rejected",
                                 lambda source=source: probe.spawn(endpoint, "none", source))
        popen.assert_not_called()
        endpoint.fileno.assert_not_called()


if __name__ == "__main__":
    unittest.main()
