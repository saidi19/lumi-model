"""Synthetic platform-independent credential transport tests; no real OS sockets.

Load only the new stdlib module in an isolated namespace. Every socket, received
descriptor and monotonic clock is a double; no workers, files, model, ML package,
private data, network or GPU are opened by these tests.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "_test_sft_peer_credentials_core_v1",
    ROOT / "src" / "lumi" / "posttraining_sft_peer_credentials_v1.py",
)
peer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = peer
SPEC.loader.exec_module(peer)

EXPECTED = (12345, 23456, 34567)
OPAQUE_PAYLOAD = b"\xff\x00opaque-no-protocol\xfe"


def frame(payload=OPAQUE_PAYLOAD):
    return len(payload).to_bytes(4, "big") + payload


def credential_entry(credentials=EXPECTED):
    return (peer.socket.SOL_SOCKET, peer.socket.SCM_CREDENTIALS, struct.pack("iII", *credentials))


def rights_entry(*descriptors):
    return (
        peer.socket.SOL_SOCKET,
        peer.socket.SCM_RIGHTS,
        b"".join(struct.pack("i", descriptor) for descriptor in descriptors),
    )


def packet(data=None, ancillary=None, flags=0):
    return (
        frame() if data is None else data,
        [credential_entry()] if ancillary is None else ancillary,
        flags,
        None,
    )


class FakeSocket:
    def __init__(self, packets=None):
        self.family = peer.socket.AF_UNIX
        self.kind = peer.socket.SOCK_SEQPACKET
        self.passcred = 1
        self.inheritable = False
        self.packets = [packet()] if packets is None else list(packets)
        self.reads = []
        self.options = []
        self.timeouts = []
        self.close_calls = 0
        self.close_error = None
        self.setup_error = None
        self.receive_hook = None

    def getsockopt(self, level, option):
        self.options.append((level, option))
        if level != peer.socket.SOL_SOCKET:
            raise AssertionError("unexpected option level")
        if option == peer.socket.SO_TYPE:
            return self.kind
        if option == peer.socket.SO_PASSCRED:
            return self.passcred
        raise AssertionError("SO_PEERCRED or unsupported socket query")

    def setsockopt(self, level, option, value):
        if self.setup_error:
            raise self.setup_error
        if (level, option, value) != (peer.socket.SOL_SOCKET, peer.socket.SO_PASSCRED, 1):
            raise AssertionError("unexpected socket setup")
        self.passcred = value

    def get_inheritable(self):
        return self.inheritable

    def set_inheritable(self, value):
        self.inheritable = value

    def settimeout(self, value):
        self.timeouts.append(value)

    def recvmsg(self, *arguments):
        self.reads.append(arguments)
        if self.receive_hook:
            self.receive_hook()
        result = self.packets.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    def close(self):
        self.close_calls += 1
        if self.close_error:
            raise self.close_error


class PeerCredentialTests(unittest.TestCase):
    def setUp(self):
        # Linux values are synthetic feature doubles, never live syscall inputs.
        for name, value in {
            "SO_PASSCRED": 16,
            "SCM_CREDENTIALS": 2,
            "SOCK_SEQPACKET": 5,
            "MSG_CMSG_CLOEXEC": 0x40000000,
            "MSG_TRUNC": 0x20,
            "MSG_CTRUNC": 0x08,
        }.items():
            self.enterContext(patch.object(peer.socket, name, value, create=True))
        self.enterContext(patch.object(peer.sys, "platform", "linux"))
        self.clock = self.enterContext(patch.object(peer.time, "monotonic", return_value=100.0))
        self.socketpair = self.enterContext(
            patch.object(
                peer.socket, "socketpair", side_effect=AssertionError("real socket blocked")
            )
        )
        self.real_socket = self.enterContext(patch.object(peer.socket, "socket", autospec=True))
        self.descriptor_close = self.enterContext(patch.object(peer.os, "close"))

    def channel(self, sock=None, **kwargs):
        if sock is None:
            sock = FakeSocket()
        options = {"expected_credentials": EXPECTED, "deadline": 101.0}
        options.update(kwargs)
        result = peer.CredentialChannel(sock, **options)
        return result, sock

    def reject(self, channel, code):
        with self.assertRaises(peer.PeerCredentialError) as raised:
            channel.receive()
        self.assertEqual(raised.exception.code, code)
        self.assertEqual(str(raised.exception), code)
        self.assertIsNone(raised.exception.__context__)
        self.assertTrue(channel.summary()["rejected"])
        before = channel.summary()
        with self.assertRaises(peer.PeerCredentialError) as again:
            channel.receive()
        self.assertEqual(again.exception.code, "credential_channel_rejected")
        self.assertEqual(channel.summary(), before)

    def test_supported_platform_abi_only(self):
        peer.require_linux()
        self.socketpair.assert_not_called()
        self.real_socket.assert_not_called()

    def test_nonlinux_rejects_before_pair_or_channel_socket_work(self):
        sock = FakeSocket()
        with patch.object(peer.sys, "platform", "darwin"):
            for action in (peer.require_linux, peer.credential_pair, lambda: self.channel(sock)):
                with self.subTest(action=action):
                    with self.assertRaises(peer.PeerCredentialError) as raised:
                        action()
                    self.assertEqual(raised.exception.code, "linux_credentials_unavailable")
        self.assertEqual(sock.options, [])
        self.assertEqual(sock.close_calls, 0)
        self.socketpair.assert_not_called()

    def test_missing_linux_features_fail_closed(self):
        for name in (
            "SO_PASSCRED", "SCM_CREDENTIALS", "SOCK_SEQPACKET", "MSG_CMSG_CLOEXEC",
            "MSG_TRUNC", "MSG_CTRUNC", "SCM_RIGHTS", "CMSG_SPACE",
        ):
            with self.subTest(feature=name):
                value = getattr(peer.socket, name)
                delattr(peer.socket, name)
                try:
                    with self.assertRaises(peer.PeerCredentialError):
                        peer.credential_pair()
                finally:
                    setattr(peer.socket, name, value)
        self.socketpair.assert_not_called()

    def test_wrong_native_layout_fails_before_pair(self):
        for size in (4, 8, 16):
            with self.subTest(size=size), patch.object(peer.struct, "calcsize", return_value=size):
                with self.assertRaises(peer.PeerCredentialError):
                    peer.credential_pair()
        self.socketpair.assert_not_called()

    def test_pair_enables_both_noninheritable_endpoints(self):
        endpoints = (FakeSocket(), FakeSocket())
        for endpoint in endpoints:
            endpoint.passcred = 0
            endpoint.inheritable = True
        self.socketpair.side_effect = None
        self.socketpair.return_value = endpoints
        self.assertEqual(peer.credential_pair(), endpoints)
        self.socketpair.assert_called_once_with(peer.socket.AF_UNIX, peer.socket.SOCK_SEQPACKET)
        for endpoint in endpoints:
            self.assertEqual(endpoint.passcred, 1)
            self.assertIs(endpoint.inheritable, False)
            self.assertEqual(endpoint.reads, [])
            self.assertEqual(endpoint.close_calls, 0)

    def test_pair_setup_failure_closes_both(self):
        for failed_endpoint in (0, 1):
            with self.subTest(endpoint=failed_endpoint):
                endpoints = (FakeSocket(), FakeSocket())
                endpoints[failed_endpoint].setup_error = OSError("sensitive setup detail")
                self.socketpair.side_effect = None
                self.socketpair.return_value = endpoints
                with self.assertRaises(peer.PeerCredentialError) as raised:
                    peer.credential_pair()
                self.assertEqual(raised.exception.code, "credential_pair_failed")
                self.assertIsNone(raised.exception.__context__)
                self.assertEqual([item.close_calls for item in endpoints], [1, 1])

    def test_pair_close_failure_still_attempts_other_endpoint(self):
        endpoints = (FakeSocket(), FakeSocket())
        endpoints[0].setup_error = OSError("setup")
        endpoints[0].close_error = OSError("cleanup")
        self.socketpair.side_effect = None
        self.socketpair.return_value = endpoints
        with self.assertRaises(peer.PeerCredentialError) as raised:
            peer.credential_pair()
        self.assertEqual(raised.exception.code, "credential_cleanup_failed")
        self.assertEqual([item.close_calls for item in endpoints], [1, 1])

    def test_pair_creation_error_is_opaque(self):
        self.socketpair.side_effect = OSError("secret path and descriptor")
        with self.assertRaises(peer.PeerCredentialError) as raised:
            peer.credential_pair()
        self.assertEqual(str(raised.exception), "credential_pair_failed")
        self.assertIsNone(raised.exception.__context__)

    def test_opaque_frame_returned_without_decoding(self):
        channel, sock = self.channel()
        self.assertEqual(channel.receive(), frame())
        self.assertEqual(sock.reads, [(
            peer.RECEIVE_DATA_CAPACITY, peer.ANCILLARY_BUFFER_BYTES, peer.socket.MSG_CMSG_CLOEXEC,
        )])
        self.assertEqual(sock.timeouts, [1.0])
        self.assertEqual(channel.summary(), {
            "verified_messages": 1,
            "receive_attempts": 1,
            "payload_bytes": len(OPAQUE_PAYLOAD),
            "read_capacity_bytes": peer.RECEIVE_CAPACITY_BYTES,
            "received_descriptors_closed": 0,
            "rejected": False,
            "closed": False,
        })
        self.descriptor_close.assert_not_called()

    def test_smallest_and_largest_frames(self):
        for payload in (b"\x00", b"x" * peer.MAXIMUM_PAYLOAD_BYTES):
            with self.subTest(size=len(payload)):
                channel, _ = self.channel(FakeSocket([packet(frame(payload))]))
                self.assertEqual(channel.receive(), frame(payload))

    def test_linux_cloexec_echo_flag_is_allowed(self):
        channel, _ = self.channel(FakeSocket([packet(flags=peer.socket.MSG_CMSG_CLOEXEC)]))
        self.assertEqual(channel.receive(), frame())

    def test_wrong_pid_uid_and_gid_each_reject(self):
        for index in range(3):
            wrong = list(EXPECTED)
            wrong[index] += 1
            channel, _ = self.channel(FakeSocket([packet(ancillary=[credential_entry(wrong)])]))
            self.reject(channel, "credential_mismatch")
            self.assertEqual(channel.summary()["verified_messages"], 0)

    def test_credentials_required_again_on_each_packet(self):
        sock = FakeSocket([packet(), packet(ancillary=[])])
        channel, _ = self.channel(sock)
        channel.receive()
        self.reject(channel, "credential_ancillary_rejected")
        self.assertEqual(len(sock.reads), 2)
        self.assertEqual(channel.summary()["verified_messages"], 1)

    def test_missing_duplicate_unknown_and_malformed_ancillary_reject(self):
        valid = credential_entry()
        cases = [
            [], [valid, valid], [(9999, valid[1], valid[2])],
            [(valid[0], 9999, valid[2])], [(valid[0], valid[1], b"x" * 11)],
            [(valid[0], valid[1], b"x" * 13)], [(valid[0], valid[1], bytearray(valid[2]))],
            [valid, (9999, 9999, b"unknown")],
        ]
        for ancillary in cases:
            with self.subTest(case=cases.index(ancillary)):
                channel, _ = self.channel(FakeSocket([packet(ancillary=ancillary)]))
                self.reject(channel, "credential_ancillary_rejected")

    def test_unexpected_and_truncation_flags_reject(self):
        for flag in (peer.socket.MSG_TRUNC, peer.socket.MSG_CTRUNC, 0x01, -1, True):
            with self.subTest(flag=flag):
                channel, _ = self.channel(FakeSocket([packet(flags=flag)]))
                self.reject(channel, "credential_ancillary_rejected")

    def test_invalid_frame_boundaries_eof_prefix_and_multiple_frames(self):
        cases = [
            b"", b"\x00", b"\x00\x00\x00\x00", b"\x00\x00\x00\x02x",
            b"\x00\x00\x00\x01xx", b"\xff\xff\xff\xffx",
            frame(b"x" * (peer.MAXIMUM_PAYLOAD_BYTES + 1)), frame() + frame(),
            bytearray(frame()),
        ]
        for data in cases:
            with self.subTest(size=len(data)):
                channel, sock = self.channel(FakeSocket([packet(data)]))
                self.reject(channel, "credential_frame_rejected")
                self.assertEqual(len(sock.reads), 1)
                self.assertEqual(
                    channel.summary()["read_capacity_bytes"], peer.RECEIVE_CAPACITY_BYTES
                )

    def test_rights_are_closed_before_rejection(self):
        channel, _ = self.channel(FakeSocket([packet(ancillary=[
            credential_entry(), rights_entry(41, 42), rights_entry(43),
        ])]))
        self.reject(channel, "credential_ancillary_rejected")
        self.assertEqual(
            [call.args[0] for call in self.descriptor_close.call_args_list], [41, 42, 43]
        )
        self.assertEqual(channel.summary()["received_descriptors_closed"], 3)

    def test_truncated_packet_closes_all_delivered_rights(self):
        for flag in (peer.socket.MSG_TRUNC, peer.socket.MSG_CTRUNC):
            with self.subTest(flag=flag):
                self.descriptor_close.reset_mock()
                channel, _ = self.channel(FakeSocket([packet(
                    data=b"", ancillary=[rights_entry(51, 52), credential_entry()], flags=flag,
                )]))
                self.reject(channel, "credential_ancillary_rejected")
                self.assertEqual(self.descriptor_close.call_count, 2)
                self.assertEqual(channel.summary()["received_descriptors_closed"], 2)

    def test_deadline_after_receive_still_closes_rights(self):
        sock = FakeSocket([packet(ancillary=[rights_entry(61, 62)])])
        sock.receive_hook = lambda: setattr(self.clock, "return_value", 102.0)
        channel, _ = self.channel(sock)
        self.reject(channel, "credential_deadline_exhausted")
        self.assertEqual(self.descriptor_close.call_count, 2)
        self.assertEqual(channel.summary()["read_capacity_bytes"], peer.RECEIVE_CAPACITY_BYTES)

    def test_partial_rights_integer_closes_complete_prefix_and_later_entries(self):
        first = rights_entry(71)
        malformed = (first[0], first[1], first[2] + b"x")
        channel, _ = self.channel(FakeSocket([packet(ancillary=[malformed, rights_entry(72)])]))
        self.reject(channel, "credential_cleanup_failed")
        self.assertEqual([call.args[0] for call in self.descriptor_close.call_args_list], [71, 72])

    def test_malformed_ancillary_does_not_skip_later_rights(self):
        channel, _ = self.channel(FakeSocket([packet(ancillary=[(), rights_entry(73)])]))
        self.reject(channel, "credential_cleanup_failed")
        self.descriptor_close.assert_called_once_with(73)

    def test_close_error_does_not_skip_later_received_descriptors(self):
        self.descriptor_close.side_effect = [OSError("sensitive fd error"), None, None]
        channel, _ = self.channel(FakeSocket([packet(ancillary=[rights_entry(81, 82, 83)])]))
        self.reject(channel, "credential_cleanup_failed")
        self.assertEqual(self.descriptor_close.call_count, 3)
        self.assertEqual(channel.summary()["received_descriptors_closed"], 2)

    def test_repeated_synthetic_descriptor_not_closed_twice(self):
        channel, _ = self.channel(FakeSocket([packet(ancillary=[rights_entry(91, 91, 92)])]))
        self.reject(channel, "credential_cleanup_failed")
        self.assertEqual([call.args[0] for call in self.descriptor_close.call_args_list], [91, 92])

    def test_error_timeout_and_interruption_each_spend_one_capacity_without_retry(self):
        for error, code in (
            (OSError("sensitive details"), "credential_receive_failed"),
            (TimeoutError("worker timeout"), "credential_deadline_exhausted"),
            (InterruptedError("interrupted"), "credential_receive_failed"),
            (KeyboardInterrupt(), "credential_receive_failed"),
        ):
            with self.subTest(kind=type(error).__name__):
                channel, sock = self.channel(FakeSocket([error]))
                self.reject(channel, code)
                self.assertEqual(len(sock.reads), 1)
                self.assertEqual(channel.summary()["receive_attempts"], 1)
                self.assertEqual(
                    channel.summary()["read_capacity_bytes"], peer.RECEIVE_CAPACITY_BYTES
                )

    def test_message_budget_is_cumulative_and_checked_before_next_receive(self):
        channel, sock = self.channel(FakeSocket([packet(), packet()]), maximum_messages=1)
        channel.receive()
        self.reject(channel, "credential_budget_exhausted")
        self.assertEqual(len(sock.reads), 1)

    def test_read_capacity_not_actual_payload_controls_budget(self):
        channel, sock = self.channel(
            FakeSocket([packet(frame(b"x")), packet(frame(b"y"))]),
            maximum_read_bytes=2 * peer.RECEIVE_CAPACITY_BYTES - 1,
        )
        channel.receive()
        self.reject(channel, "credential_budget_exhausted")
        self.assertEqual(channel.summary()["payload_bytes"], 1)
        self.assertEqual(channel.summary()["read_capacity_bytes"], peer.RECEIVE_CAPACITY_BYTES)
        self.assertEqual(len(sock.reads), 1)

    def test_underfunded_first_receive_never_reads(self):
        channel, sock = self.channel(maximum_read_bytes=peer.RECEIVE_CAPACITY_BYTES - 1)
        self.reject(channel, "credential_budget_exhausted")
        self.assertEqual(channel.summary()["receive_attempts"], 0)
        self.assertEqual(sock.reads, [])

    def test_budget_is_rechecked_after_receive(self):
        for field in ("_maximum_messages", "_maximum_read_bytes"):
            with self.subTest(field=field):
                channel, sock = self.channel()
                sock.receive_hook = lambda field=field, channel=channel: setattr(channel, field, 0)
                self.reject(channel, "credential_budget_exhausted")
                self.assertEqual(channel.summary()["verified_messages"], 0)
                self.assertEqual(len(sock.reads), 1)

    def test_nonfinite_clock_cannot_accept_frame(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(clock=value):
                channel, sock = self.channel()
                self.clock.return_value = value
                self.reject(channel, "credential_deadline_exhausted")
                self.assertEqual(sock.reads, [])
                self.clock.return_value = 100.0

    def test_charge_precedes_socket_receive_and_timeout_operation(self):
        channel, sock = self.channel()

        def observe_charge():
            self.assertEqual(channel.summary()["receive_attempts"], 1)
            self.assertEqual(channel.summary()["read_capacity_bytes"], peer.RECEIVE_CAPACITY_BYTES)

        sock.receive_hook = observe_charge
        channel.receive()
        channel, sock = self.channel()
        sock.settimeout = Mock(side_effect=OSError("timeout setup failed"))
        self.reject(channel, "credential_receive_failed")
        self.assertEqual(channel.summary()["read_capacity_bytes"], peer.RECEIVE_CAPACITY_BYTES)
        self.assertEqual(sock.reads, [])

    def test_expired_deadline_never_reads_or_resets(self):
        channel, sock = self.channel(deadline=100.0)
        self.reject(channel, "credential_deadline_exhausted")
        self.assertEqual(sock.reads, [])
        self.assertEqual(channel.summary()["receive_attempts"], 0)

    def test_deadline_is_rechecked_after_receive_before_acceptance(self):
        sock = FakeSocket()
        sock.receive_hook = lambda: setattr(self.clock, "return_value", 101.0)
        channel, _ = self.channel(sock)
        self.reject(channel, "credential_deadline_exhausted")
        self.assertEqual(channel.summary()["verified_messages"], 0)

    def test_absolute_deadline_does_not_reset_between_messages(self):
        channel, sock = self.channel(FakeSocket([packet(), packet()]))
        channel.receive()
        self.clock.return_value = 100.75
        channel.receive()
        self.assertEqual(sock.timeouts, [1.0, 0.25])
        self.clock.return_value = 101.0
        self.reject(channel, "credential_deadline_exhausted")
        self.assertEqual(len(sock.reads), 2)

    def test_socket_properties_checked_at_construction_and_before_each_read(self):
        for attribute, invalid in (
            ("family", peer.socket.AF_INET), ("kind", peer.socket.SOCK_STREAM),
            ("passcred", 0), ("inheritable", True),
        ):
            with self.subTest(attribute=attribute):
                sock = FakeSocket()
                setattr(sock, attribute, invalid)
                with self.assertRaises(peer.PeerCredentialError):
                    self.channel(sock)
                self.assertEqual(sock.close_calls, 0)  # Caller still owns failed construction.
                sock = FakeSocket([packet(), packet()])
                channel, _ = self.channel(sock)
                channel.receive()
                setattr(sock, attribute, invalid)
                self.reject(channel, "credential_socket_rejected")
                self.assertEqual(len(sock.reads), 1)

    def test_socket_query_error_is_opaque_and_sticky(self):
        channel, sock = self.channel()
        sock.getsockopt = Mock(side_effect=OSError("private path"))
        self.reject(channel, "credential_receive_failed")
        self.assertEqual(sock.reads, [])

    def test_strict_expected_credentials_configuration(self):
        for identity in (
            None, list(EXPECTED), (), EXPECTED + (1,), (True, 0, 0), (0, 0, 0),
            (-1, 0, 0), (2**31, 0, 0), (1, -1, 0), (1, 0, 2**32), (1, 1.0, 0),
        ):
            with self.subTest(identity=identity):
                with self.assertRaises(peer.PeerCredentialError) as raised:
                    self.channel(expected_credentials=identity)
                self.assertEqual(raised.exception.code, "credential_configuration_rejected")

    def test_strict_deadline_and_budget_configuration(self):
        invalid = {
            "deadline": (None, True, "101", float("nan"), float("inf"), -1.0, 0, 10**400),
            "maximum_messages": (None, True, 1.0, 0, -1, peer.HARD_MAXIMUM_MESSAGES + 1),
            "maximum_read_bytes": (None, True, 1.0, 0, -1, peer.HARD_MAXIMUM_READ_BYTES + 1),
        }
        for field, values in invalid.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    with self.assertRaises(peer.PeerCredentialError) as raised:
                        self.channel(**{field: value})
                    self.assertEqual(raised.exception.code, "credential_configuration_rejected")

    def test_close_is_idempotent_and_read_after_close_is_sticky(self):
        channel, sock = self.channel()
        channel.close()
        channel.close()
        self.assertEqual(sock.close_calls, 1)
        self.assertTrue(channel.summary()["closed"])
        self.reject(channel, "credential_channel_closed")
        self.assertEqual(sock.reads, [])

    def test_socket_close_failure_is_sticky_without_close_retry(self):
        channel, sock = self.channel()
        sock.close_error = OSError("secret close details")
        for _ in range(2):
            with self.assertRaises(peer.PeerCredentialError) as raised:
                channel.close()
            self.assertEqual(raised.exception.code, "credential_cleanup_failed")
            self.assertIsNone(raised.exception.__context__)
        self.assertEqual(sock.close_calls, 1)
        self.assertFalse(channel.summary()["closed"])
        self.reject(channel, "credential_channel_rejected")

    def test_rejected_channel_can_still_close_owned_socket(self):
        channel, sock = self.channel(FakeSocket([packet(ancillary=[])]))
        self.reject(channel, "credential_ancillary_rejected")
        channel.close()
        self.assertTrue(channel.summary()["closed"])
        self.assertTrue(channel.summary()["rejected"])
        self.assertEqual(sock.close_calls, 1)

    def test_summary_is_detached_and_contains_only_aggregate_values(self):
        channel, _ = self.channel()
        channel.receive()
        report = channel.summary()
        self.assertTrue(all(type(value) in (int, bool) for value in report.values()))
        text = json.dumps(report)
        for forbidden in (*map(str, EXPECTED), "opaque-no-protocol", "Traceback", "path", "pid"):
            self.assertNotIn(forbidden, text)
        report["verified_messages"] = 999
        self.assertEqual(channel.summary()["verified_messages"], 1)

    def test_error_constructor_cannot_echo_unknown_or_unhashable_input(self):
        for value in ("secret message", [], {}, None, 123):
            self.assertEqual(str(peer.PeerCredentialError(value)), "credential_channel_rejected")


if __name__ == "__main__":
    unittest.main()
