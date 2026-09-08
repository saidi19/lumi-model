"""Tiny stdlib socketpair tests; no files, child processes, ML or external network."""

from __future__ import annotations

import array
import hashlib
import importlib
import json
import os
import socket
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_synthetic_sft_worker_client_v2"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
client_module = importlib.import_module(PACKAGE + ".posttraining_sft_worker_client_v2")
protocol = client_module.protocol
EPOCH, TOKENS = "a" * 64, {0: "b" * 64, 1: "c" * 64}


class Clock:
    def __init__(self):
        self.now, self.increment = 1000, 0

    def __call__(self):
        result = self.now
        self.now += self.increment
        return result


def transaction(operation="reserve", arguments=None, *, rank=0, sequence=1):
    if arguments is None:
        arguments = {"checkpoint_read_bytes": 3}
    request = {"schema_version": protocol.REQUEST_SCHEMA, "epoch": EPOCH, "rank": rank,
               "sequence": sequence, "token": TOKENS[rank], "operation": operation,
               "arguments": arguments}
    ack_value = {"schema_version": protocol.ACK_SCHEMA, "epoch": EPOCH, "rank": rank,
                 "sequence": sequence, "operation": operation, "status": "reserved"}
    ack = protocol.encode_frame(ack_value)
    confirmation = {**request, "operation": "confirm_ack",
                    "arguments": {"ack_sha256": hashlib.sha256(ack).hexdigest()}}
    return protocol.encode_frame(request), ack, protocol.encode_frame(confirmation)


def fence(*, rank=0, step=1):
    return protocol.encode_frame({"schema_version": client_module.STEP_FENCE_SCHEMA,
                                  "epoch": EPOCH, "rank": rank, "to_step": step,
                                  "status": "committed"})


def receive_frame(sock):
    sock.settimeout(1)
    raw = bytearray()
    target = 4
    while len(raw) < target:
        block = sock.recv(target - len(raw))
        if not block:
            raise AssertionError("synthetic_eof")
        raw.extend(block)
        if len(raw) == 4 and target == 4:
            size = int.from_bytes(raw, "big")
            if not 0 < size <= 4096:
                raise AssertionError("synthetic_frame")
            target += size
    return bytes(raw)


class WorkerClientTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.clients, self.sockets, self.interrupts = [], [], []

    def tearDown(self):
        for client in self.clients:
            try:
                client.close()
            except client_module.WorkerClientError:
                pass
        for sock in self.sockets:
            sock.close()

    def pair(self):
        left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sockets.extend((left, right))
        return left, right

    def client(self, *, rank=0, sock=None, **changes):
        if sock is None:
            sock, server = self.pair()
        else:
            server = None
        options = {"epoch": EPOCH, "rank": rank, "token": TOKENS.get(rank, "d" * 64),
                   "interrupt": lambda: self.interrupts.append(True),
                   "maximum_wire_bytes": 65_536, "deadline_ns": 10_000_000_000,
                   "maximum_exchange_ns": 1_000_000_000, "clock_ns": self.clock}
        options.update(changes)
        client = client_module.WorkerLedgerClient(sock, **options)
        self.clients.append(client)
        return client, server

    def error(self, code, operation, *args, **kwargs):
        with self.assertRaises(client_module.WorkerClientError) as caught:
            operation(*args, **kwargs)
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(caught.exception.args, (code,))
        self.assertNotIn(EPOCH, repr(caught.exception))
        self.assertNotIn(TOKENS[0], repr(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)

    def test_metadata_request_uses_v2_frames_and_exact_confirmation(self):
        client, server = self.client()
        request, ack, confirmation = transaction(arguments={"encoded_metadata_read_bytes": 19})
        server.sendall(ack)
        client.reserve(0, encoded_metadata_read_bytes=19)
        self.assertEqual(receive_frame(server), request)
        self.assertEqual(receive_frame(server), confirmation)
        self.assertEqual(json.loads(request[4:])["arguments"], {"encoded_metadata_read_bytes": 19})
        self.assertEqual(json.loads(request[4:])["schema_version"],
                         "lumi.sft-epoch-ledger-request.v0.2")
        self.assertEqual(client.summary()["schema_version"],
                         "lumi.sft-worker-ledger-client-summary.v0.2")

    def test_metadata_v1_ack_rejected_without_confirmation(self):
        client, server = self.client()
        request, ack, _ = transaction(arguments={"encoded_metadata_read_bytes": 7})
        old_ack = json.loads(ack[4:])
        old_ack["schema_version"] = "lumi.sft-epoch-ledger-ack.v0.1"
        server.sendall(protocol.encode_frame(old_ack))
        self.error("ack_rejected", client.reserve, 0, encoded_metadata_read_bytes=7)
        self.assertEqual(receive_frame(server), request)
        self.assertEqual(client.summary()["confirmations_sent"], 0)
        self.assertEqual(client.summary()["acknowledgements_validated"], 0)

    def test_v1_step_fence_rejected_after_v2_complete_ack(self):
        client, server = self.client()
        _, ack, _ = transaction(operation="step_complete", arguments={"to_step": 1})
        server.sendall(ack)
        client.step_complete(1)
        receive_frame(server)
        receive_frame(server)
        old_fence = json.loads(fence()[4:])
        old_fence["schema_version"] = "lumi.sft-worker-step-fence.v0.1"
        server.sendall(protocol.encode_frame(old_fence))
        self.error("step_fence_rejected", client.wait_step_commit, 1)
        self.assertTrue(client.summary()["step_fence_pending"])
        self.assertEqual(client.summary()["step_fences_validated"], 0)

    def test_metadata_resource_invalid_values_fail_before_socket_send(self):
        for value in (True, False, -1, 1.0, "1", 2**63, 0):
            with self.subTest(value=value):
                client, _ = self.client()
                self.error("invalid_arguments", client.reserve, 0,
                           encoded_metadata_read_bytes=value)
                self.assertEqual(client.summary()["wire_bytes_sent"], 0)

    def test_misnamed_metadata_resource_is_not_remapped(self):
        client, _ = self.client()
        self.error("invalid_arguments", client.reserve, 0, metadata_read_bytes=1)
        self.assertEqual(client.summary()["wire_bytes_reserved"], 0)

    def test_metadata_exchange_capacity_prepaid_before_first_byte(self):
        frames = transaction(arguments={"encoded_metadata_read_bytes": 3})
        client, _ = self.client(maximum_wire_bytes=sum(map(len, frames)) - 1)
        self.error("wire_cap_exceeded", client.reserve, 0, encoded_metadata_read_bytes=3)
        self.assertEqual(client.summary()["wire_bytes_sent"], 0)

    def test_zero_metadata_with_positive_binary_preserves_frozen_resource_semantics(self):
        client, server = self.client()
        resources = {"encoded_metadata_read_bytes": 0, "encoded_binary_read_bytes": 2}
        request, ack, confirmation = transaction(arguments=resources)
        server.sendall(ack)
        client.reserve(0, **resources)
        self.assertEqual(receive_frame(server), request)
        self.assertEqual(receive_frame(server), confirmation)

    def test_v1_client_resource_and_fence_are_not_mutated_by_v2_import(self):
        old = importlib.import_module(PACKAGE + ".posttraining_sft_worker_client_v1")
        self.assertNotIn("encoded_metadata_read_bytes", old.protocol.RESOURCE_FIELDS)
        self.assertEqual(old.STEP_FENCE_SCHEMA, "lumi.sft-worker-step-fence.v0.1")
        self.assertEqual(client_module.STEP_FENCE_SCHEMA, "lumi.sft-worker-step-fence.v0.2")

    def test_both_rank_metadata_reservations_reach_same_v2_parent_before_ack(self):
        local = importlib.import_module(PACKAGE + ".posttraining_sft_local_ipc_v1")
        supervision = importlib.import_module(PACKAGE + ".posttraining_sft_supervisor_v2")
        caps, counts = local.synthetic_inputs()
        caps["maximum_encoded_metadata_read_bytes"] = 12
        supervisor = supervision.Supervisor(caps, counts, world_size=2, clock_ns=lambda: 0)
        session = protocol.EpochLedgerSession(supervisor, world_size=2,
                                               rank_tokens=TOKENS, epoch=EPOCH)
        session.begin_invocation("reference", 0, 2, False)
        pairs = [self.client(rank=rank, deadline_ns=time.monotonic_ns() + 2_000_000_000,
                             clock_ns=time.monotonic_ns) for rank in range(2)]
        errors, observed = [], []

        def parent():
            try:
                for rank in range(2):
                    server = pairs[rank][1]
                    (ack,) = session.feed(rank, receive_frame(server))
                    observed.append(supervisor.summary()["resources_reserved"][
                        "encoded_metadata_read_bytes"])
                    server.sendall(ack.frame)
                    session.mark_ack_sent(ack)
                    if session.feed(rank, receive_frame(server)):
                        raise AssertionError("synthetic_confirmation")
                session.check_pending()
            except BaseException:
                errors.append("synthetic_parent_failed")

        thread = threading.Thread(target=parent, daemon=True)
        thread.start()
        try:
            pairs[0][0].reserve(0, encoded_metadata_read_bytes=5)
            pairs[1][0].reserve(1, encoded_metadata_read_bytes=7)
        finally:
            thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(observed, [5, 12])
        self.assertEqual(session.summary()["acknowledgements_confirmed"], 2)
        self.assertEqual(session.summary()["pending_acknowledgements"], 0)
        resources = supervisor.summary()["resources_reserved"]
        self.assertEqual(resources["encoded_metadata_read_bytes"], 12)
        self.assertEqual(resources["encoded_index_read_bytes"], 0)
        self.assertEqual(resources["encoded_binary_read_bytes"], 0)

    def test_exact_ack_confirmation_and_prepaid_both_direction_accounting(self):
        client, server = self.client()
        request, ack, confirmation = transaction()
        server.sendall(ack)
        self.assertIsNone(client.reserve(0, checkpoint_read_bytes=3))
        self.assertEqual(receive_frame(server), request)
        self.assertEqual(receive_frame(server), confirmation)
        summary = client.summary()
        expected_capacity = (sum(map(len, (request, ack, confirmation)))
                             + 2 * client_module.ANCILLARY_BYTES)
        self.assertEqual(summary["wire_bytes_reserved"], expected_capacity)
        self.assertEqual(summary["wire_bytes_received"], len(ack))
        self.assertEqual(summary["wire_bytes_sent"], len(request) + len(confirmation))
        self.assertEqual(summary["confirmations_sent"], 1)
        self.assertEqual(summary["step_fences_validated"], 0)
        self.assertNotIn(EPOCH, str(summary))
        self.assertNotIn(TOKENS[0], repr(client))

    def test_wire_limit_rejected_before_any_request_or_read(self):
        frames = transaction()
        client, server = self.client(maximum_wire_bytes=sum(map(len, frames)) - 1)
        self.error("wire_cap_exceeded", client.reserve, 0, checkpoint_read_bytes=3)
        self.assertEqual(client.summary()["wire_bytes_sent"], 0)
        server.setblocking(False)
        with self.assertRaises(BlockingIOError):
            server.recv(1)
        self.assertEqual(self.interrupts, [True])

    def test_partial_send_receive_continuations_keep_one_transaction(self):
        client, server = self.client()
        request, ack, confirmation = transaction()
        for sock in (client._sock, server):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
        send, recvmsg = socket.socket.send, socket.socket.recvmsg
        received, peer_errors, send_sizes, receive_chunks = [], [], [], []

        def short_send(sock, data, *args):
            if sock is not client._sock:
                return send(sock, data, *args)
            written = send(sock, data[:7], *args)
            send_sizes.append(written)
            return written

        def short_receive(sock, size, *args):
            if sock is not client._sock:
                return recvmsg(sock, size, *args)
            result = recvmsg(sock, min(size, 3), *args)
            receive_chunks.append(result[0])
            return result

        def drain_peer():
            try:
                # A real peer consumes the request before acknowledging it and
                # keeps consuming while the worker transmits confirmation. A
                # prequeued ACK plus an undrained queue is not a valid harness.
                received.append(receive_frame(server))
                if received[0] != request:
                    raise AssertionError("synthetic_request_mismatch")
                server.sendall(ack)
                received.append(receive_frame(server))
            except BaseException:
                peer_errors.append("synthetic_peer_failed")
                try:
                    server.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

        with patch.object(socket.socket, "send", short_send), \
                patch.object(socket.socket, "recvmsg", short_receive):
            peer = threading.Thread(target=drain_peer, daemon=True)
            peer.start()
            try:
                self.assertIsNone(client.reserve(0, checkpoint_read_bytes=3))
            finally:
                peer.join(timeout=1)  # Same existing one-second transport limit.
        self.assertFalse(peer.is_alive())
        self.assertEqual(peer_errors, [])
        self.assertEqual(received, [request, confirmation])
        self.assertEqual(b"".join(receive_chunks), ack)
        self.assertTrue(all(0 < length <= 7 for length in send_sizes))
        self.assertTrue(all(0 < len(chunk) <= 3 for chunk in receive_chunks))
        self.assertGreater(len(send_sizes), 2)
        self.assertGreater(len(receive_chunks), 2)
        summary = client.summary()
        for counter in ("requests_sent", "acknowledgements_validated", "confirmations_sent"):
            self.assertEqual(summary[counter], 1)
        self.assertEqual(summary["wire_bytes_sent"], len(request) + len(confirmation))
        self.assertEqual(sum(send_sizes), summary["wire_bytes_sent"])
        self.assertEqual(summary["wire_bytes_received"], len(ack))
        self.assertEqual(summary["wire_bytes_reserved"],
                         sum(map(len, (request, ack, confirmation)))
                         + len(receive_chunks) * client_module.ANCILLARY_BYTES)
        self.assertEqual(self.interrupts, [])

    def test_fragmented_undrained_socket_queue_fills_without_blocking(self):
        sender, receiver = self.pair()
        for sock in (sender, receiver):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
        sender.setblocking(False)
        sent, blocked = 0, False
        for _ in range(4096):  # At most 28 KiB; no sleeps or blocking send.
            try:
                sent += sender.send(b"S" * 7)
            except BlockingIOError:
                blocked = True
                break
        self.assertTrue(blocked, "synthetic_undrained_queue_did_not_fill")
        self.assertGreater(sent, 0)
        receiver.setblocking(False)
        drained = 0
        while drained < sent:
            block = receiver.recv(min(4096, sent - drained))
            self.assertTrue(block)
            self.assertEqual(block, b"S" * len(block))
            drained += len(block)
        self.assertEqual(drained, sent)
        self.assertEqual(sender.send(b"S" * 7), 7)

    def test_ancillary_capacity_is_prepaid_before_each_receive_fragment(self):
        frames = transaction()
        maximum = sum(map(len, frames)) + client_module.ANCILLARY_BYTES
        client, server = self.client(maximum_wire_bytes=maximum)
        server.sendall(frames[1])
        self.error("wire_cap_exceeded", client.reserve, 0, checkpoint_read_bytes=3)
        self.assertEqual(client.summary()["wire_bytes_reserved"], maximum)
        self.assertEqual(client.summary()["wire_bytes_received"], 4)
        self.assertEqual(client.summary()["acknowledgements_validated"], 0)

    def test_partial_send_exception_is_not_retried_or_refunded(self):
        client, server = self.client()
        send, calls = socket.socket.send, []

        def fail_after_prefix(sock, data, *args):
            calls.append(True)
            if len(calls) == 1:
                return send(sock, data[:5], *args)
            raise OSError("SYNTHETIC-PRIVATE")

        with patch.object(socket.socket, "send", fail_after_prefix):
            self.error("io_failed", client.reserve, 0, checkpoint_read_bytes=3)
            reserved = client.summary()["wire_bytes_reserved"]
            self.error("terminal_failure", client.reserve, 0, checkpoint_read_bytes=3)
        self.assertEqual(len(calls), 2)
        self.assertEqual(client.summary()["wire_bytes_reserved"], reserved)
        self.assertGreater(reserved, 5)
        self.assertEqual(server.recv(5), transaction()[0][:5])
        self.assertEqual(self.interrupts, [True])

    def test_wrong_epoch_rank_sequence_operation_and_status_ack_are_rejected(self):
        for key, value in (("epoch", "d" * 64), ("rank", 1), ("sequence", 2),
                           ("operation", "reserve_step"), ("status", "rejected")):
            with self.subTest(field=key):
                client, server = self.client()
                _, ack, _ = transaction()
                payload = protocol.FrameDecoder().feed(ack)[0]
                payload[key] = value
                wrong = protocol.encode_frame(payload)
                server.sendall(wrong)
                code = "ack_rejected" if len(wrong) == len(ack) else "frame_rejected"
                self.error(code, client.reserve, 0, checkpoint_read_bytes=3)
                self.assertEqual(client.summary()["confirmations_sent"], 0)

    def test_oversized_header_and_partial_eof_are_terminal(self):
        for payload in (b"\xff" * 4, transaction()[1][:5]):
            with self.subTest(bytes=len(payload)):
                client, server = self.client()
                server.sendall(payload)
                server.shutdown(socket.SHUT_WR)
                self.error("frame_rejected", client.reserve, 0, checkpoint_read_bytes=3)
                self.assertLessEqual(client.summary()["wire_bytes_received"], len(payload))
                self.assertEqual(client.summary()["confirmations_sent"], 0)

    def test_ancillary_descriptor_rights_are_not_accepted(self):
        client, server = self.client()
        extra, _ = self.pair()
        server.sendmsg([transaction()[1]], [(socket.SOL_SOCKET, socket.SCM_RIGHTS,
                                            array.array("i", [extra.fileno()]))])
        self.error("frame_rejected", client.reserve, 0, checkpoint_read_bytes=3)
        self.assertEqual(client.summary()["confirmations_sent"], 0)
        self.assertEqual(client.summary()["received_descriptors_closed"], 1)

    def test_strict_resource_rank_and_numeric_arguments(self):
        for rank, arguments in ((True, {"private_write_bytes": 1}),
                                (1, {"private_write_bytes": 1}), (0, {}),
                                (0, {"private_write_bytes": True}),
                                (0, {"private_write_bytes": -1}),
                                (0, {"private_write_bytes": 0}),
                                (0, {"source_text": 1}),
                                (0, {"private_write_bytes": 2**63})):
            with self.subTest(rank=rank, keys=tuple(arguments)):
                client, _ = self.client()
                self.error("invalid_arguments", client.reserve, rank, **arguments)
                self.assertEqual(client.summary()["wire_bytes_reserved"], 0)

    def test_reserve_step_requires_rank_zero_and_exact_next_step(self):
        for rank, start, stop in ((1, 0, 1), (0, True, 1), (0, 0, 2), (0, 0, False)):
            client, _ = self.client(rank=rank)
            self.error("invalid_arguments", client.reserve_step, start, stop)

    def test_constructor_failure_does_not_adopt_or_interrupt(self):
        for changes in ({"epoch": "bad"}, {"token": EPOCH}, {"rank": True},
                        {"maximum_wire_bytes": True}, {"deadline_ns": 0},
                        {"maximum_exchange_ns": 0}, {"interrupt": None}):
            sock, _ = self.pair()
            self.error("invalid_arguments", self.client, sock=sock, **changes)
            self.assertGreaterEqual(sock.fileno(), 3)
        self.assertEqual(self.interrupts, [])

    def test_nonprivate_socket_type_unconnected_and_inheritable_are_rejected(self):
        disconnected = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        datagram, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.sockets.extend((disconnected, datagram, peer))
        inheritable, _ = self.pair()
        os.set_inheritable(inheritable.fileno(), True)
        for sock in (disconnected, datagram, inheritable, object()):
            self.error("invalid_socket", self.client, sock=sock)

    def test_no_reinitialization_reset_and_only_one_interrupt(self):
        client, server = self.client()
        self.error("invalid_arguments", client.__init__, server, epoch=EPOCH, rank=0,
                   token=TOKENS[0], interrupt=lambda: None, maximum_wire_bytes=1,
                   deadline_ns=10000, maximum_exchange_ns=1, clock_ns=self.clock)
        self.error("terminal_failure", client.reserve, 0, checkpoint_read_bytes=1)
        self.assertEqual(self.interrupts, [True])
        self.assertGreaterEqual(server.fileno(), 3)

    def test_absolute_deadline_and_regressing_clock_fail_before_io(self):
        for now, code in ((10_000_000_000, "deadline_exhausted"), (999, "clock_rejected")):
            client, _ = self.client()
            self.clock.now = now
            self.error(code, client.reserve, 0, checkpoint_read_bytes=1)
            self.assertEqual(client.summary()["wire_bytes_reserved"], 0)
            self.clock.now = 1000

    def test_exchange_deadline_is_not_reset_between_request_ack_and_confirmation(self):
        client, server = self.client(maximum_exchange_ns=100)
        server.sendall(transaction()[1])
        self.clock.increment = 25
        self.error("deadline_exhausted", client.reserve, 0, checkpoint_read_bytes=3)
        self.assertEqual(client.summary()["requests_sent"], 1)
        self.assertEqual(client.summary()["acknowledgements_validated"], 1)
        self.assertEqual(client.summary()["confirmations_sent"], 0)

    def test_clock_reentry_cannot_be_swallowed(self):
        client, _ = self.client()
        armed = [True]

        def reenter():
            if armed[0]:
                armed[0] = False
                try:
                    client.reserve(0, checkpoint_read_bytes=1)
                except client_module.WorkerClientError:
                    pass
            return 1000

        client._clock = reenter
        self.error("terminal_failure", client.reserve, 0, checkpoint_read_bytes=1)
        self.assertEqual(client.summary()["failure_code"], "reentry")
        self.assertEqual(client.summary()["wire_bytes_sent"], 0)

    def test_clock_descriptor_mutation_is_detected_before_send(self):
        client, _ = self.client()

        def mutate():
            os.set_inheritable(client._sock.fileno(), True)
            return 1000

        client._clock = mutate
        self.error("descriptor_changed", client.reserve, 0, checkpoint_read_bytes=1)
        self.assertEqual(client.summary()["wire_bytes_sent"], 0)
        client.close()  # Same owned inode may be cleaned despite changed CLOEXEC.

    def test_thread_and_pid_ownership_reject_before_io(self):
        for index in (0, 1):
            client, _ = self.client()
            owner = list(client._owner)
            owner[index] += 1
            client._owner = tuple(owner)
            self.error("owner_mismatch", client.reserve, 0, checkpoint_read_bytes=1)
            client._owner = os.getpid(), threading.get_ident()
            self.assertEqual(client.summary()["wire_bytes_sent"], 0)

    def test_step_fence_required_before_another_operation(self):
        for operation in (lambda client: client.reserve(0, private_write_bytes=1),
                          lambda client: client.reserve_step(1, 2),
                          lambda client: client.step_complete(2)):
            client, server = self.client()
            server.sendall(transaction("step_complete", {"to_step": 1})[1])
            client.step_complete(1)
            self.assertTrue(client.summary()["step_fence_pending"])
            self.error("step_fence_pending", operation, client)

    def test_exact_fence_allows_next_request_and_advances_sequence(self):
        client, server = self.client()
        first = transaction("step_complete", {"to_step": 1})
        second = transaction(sequence=2)
        server.sendall(first[1] + fence() + second[1])
        client.step_complete(1)
        self.assertIsNone(client.wait_step_commit(1))
        self.assertFalse(client.summary()["step_fence_pending"])
        client.reserve(0, checkpoint_read_bytes=3)
        for expected in (first[0], first[2], second[0], second[2]):
            self.assertEqual(receive_frame(server), expected)
        summary = client.summary()
        self.assertEqual(summary["step_fences_validated"], 1)
        self.assertEqual(summary["wire_bytes_reserved"],
                         sum(map(len, first + second)) + len(fence())
                         + 6 * client_module.ANCILLARY_BYTES)

    def test_fence_without_completion_or_wrong_step_is_rejected(self):
        client, _ = self.client()
        self.error("step_fence_rejected", client.wait_step_commit, 1)
        client, server = self.client()
        server.sendall(transaction("step_complete", {"to_step": 1})[1])
        client.step_complete(1)
        self.error("step_fence_rejected", client.wait_step_commit, 2)

    def test_wrong_bound_fence_cannot_release_pending_step(self):
        client, server = self.client()
        server.sendall(transaction("step_complete", {"to_step": 1})[1] + fence(rank=1))
        client.step_complete(1)
        self.error("step_fence_rejected", client.wait_step_commit, 1)
        self.assertTrue(client.summary()["step_fence_pending"])
        self.assertEqual(client.summary()["step_fences_validated"], 0)

    def test_fence_wire_cap_reserved_before_read(self):
        frames = transaction("step_complete", {"to_step": 1})
        client, server = self.client(maximum_wire_bytes=sum(map(len, frames))
                                     + 2 * client_module.ANCILLARY_BYTES)
        server.sendall(frames[1] + fence())
        client.step_complete(1)
        self.error("wire_cap_exceeded", client.wait_step_commit, 1)
        self.assertEqual(client.summary()["wire_bytes_received"], len(frames[1]))

    def test_close_is_idempotent_and_revokes_capabilities(self):
        client, server = self.client()
        client.close()
        client.close()
        self.assertTrue(client.closed)
        self.assertIsNone(client._token)
        self.assertIsNone(client._epoch)
        self.assertEqual(server.recv(1), b"")
        self.error("closed", client.reserve, 0, checkpoint_read_bytes=1)
        self.assertEqual(self.interrupts, [])
        self.assertEqual(client.summary()["state"], "closed")

    def test_failed_close_is_not_retried(self):
        client, _ = self.client()
        owned = client._sock
        close = socket.socket.close
        calls = []

        def fail_close(sock):
            calls.append(sock is owned)
            raise OSError("SYNTHETIC-PRIVATE")

        with patch.object(socket.socket, "close", fail_close):
            self.error("close_failed", client.close)
            client.close()
        self.assertEqual(calls, [True])
        self.assertEqual(self.interrupts, [True])
        close(owned)  # Fixture reclaims the mock-preserved socket; client did not retry.

    def test_reused_descriptor_is_detached_not_closed(self):
        client, _ = self.client()
        old = client._sock
        target_fd = old.fileno()
        replacement, _ = self.pair()
        os.dup2(replacement.fileno(), target_fd, inheritable=False)
        self.error("descriptor_changed", client.close)
        self.assertEqual(old.fileno(), -1)
        self.assertEqual(os.fstat(target_fd).st_ino, os.fstat(replacement.fileno()).st_ino)
        os.close(target_fd)  # The fixture owns the explicitly installed replacement duplicate.

    def test_real_shared_epoch_completes_only_after_two_rank_confirmations_then_fences(self):
        local = importlib.import_module(PACKAGE + ".posttraining_sft_local_ipc_v1")
        caps, counts = local.synthetic_inputs()
        caps["maximum_encoded_metadata_read_bytes"] = 1024
        supervision = importlib.import_module(PACKAGE + ".posttraining_sft_supervisor_v2")
        supervisor = supervision.Supervisor(caps, counts, world_size=2, clock_ns=lambda: 0)
        session = protocol.EpochLedgerSession(supervisor, world_size=2,
                                               rank_tokens=TOKENS, epoch=EPOCH)
        session.begin_invocation("reference", 0, 2, False)
        pairs = [self.client(rank=rank, deadline_ns=time.monotonic_ns() + 2_000_000_000,
                             clock_ns=time.monotonic_ns) for rank in range(2)]
        errors, completed_after_first = [], []

        def parent():
            try:
                for index, rank in enumerate((0, 0, 1)):
                    server = pairs[rank][1]
                    acknowledgements = session.feed(rank, receive_frame(server))
                    if len(acknowledgements) != 1:
                        raise AssertionError("synthetic_ack_count")
                    ack = acknowledgements[0]
                    server.sendall(ack.frame)
                    session.mark_ack_sent(ack)
                    if session.feed(rank, receive_frame(server)):
                        raise AssertionError("synthetic_confirmation")
                    if index == 1:
                        completed_after_first.append(session.summary()[
                            "global_optimizer_steps_completed"])
                session.check_pending()
                if session.summary()["global_optimizer_steps_completed"] != 1:
                    raise AssertionError("synthetic_step_incomplete")
                for rank, (_, server) in enumerate(pairs):
                    server.sendall(fence(rank=rank))
            except BaseException:
                errors.append("synthetic_parent_failed")

        thread = threading.Thread(target=parent, daemon=True)
        thread.start()
        try:
            pairs[0][0].reserve_step(0, 1)
            pairs[0][0].step_complete(1)
            pairs[1][0].step_complete(1)
            for client, _ in pairs:
                client.wait_step_commit(1)
        finally:
            thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(completed_after_first, [0])
        self.assertEqual(session.summary()["global_optimizer_steps_completed"], 1)
        self.assertEqual(supervisor.summary()["global_optimizer_steps_completed"], 1)
        self.assertEqual(session.summary()["pending_acknowledgements"], 0)
        self.assertTrue(all(client.summary()["step_fences_validated"] == 1
                            for client, _ in pairs))


if __name__ == "__main__":
    unittest.main()
