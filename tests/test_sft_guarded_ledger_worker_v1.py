"""Pure mocks for fixed source composition; never install limits or seccomp."""

from __future__ import annotations

import ast
import struct
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def source_module(name, relative, **supplied):
    module = types.ModuleType(name)
    module.__dict__.update(supplied)
    raw = (ROOT / relative).read_bytes()
    exec(compile(raw, "<synthetic-source-composition>", "exec"), module.__dict__)
    return module


policy = source_module("_guarded_ledger_policy", "src/lumi/posttraining_sft_kernel_policy_v1.py")
guard = source_module("_guarded_ledger_challenges", "scripts/sft_kernel_guard_inert_worker_v1.py")
ledger = source_module("_guarded_ledger_protocol", "scripts/sft_authenticated_ledger_worker_v1.py")
worker = source_module(
    "_guarded_ledger_wrapper", "scripts/sft_guarded_ledger_worker_v1.py",
    kernel_policy=policy, kernel_challenges=guard, ledger_worker=ledger,
)


class CompositionTests(unittest.TestCase):
    def test_separate_source_namespaces_preserve_frozen_definitions(self):
        self.assertIs(worker.kernel_policy, policy)
        self.assertIs(worker.kernel_challenges, guard)
        self.assertIs(worker.ledger_worker, ledger)
        self.assertIsNot(worker.Channel, ledger.Channel)
        self.assertIs(worker.Channel.__bases__[0], ledger.Channel)
        self.assertIs(worker.Channel.receive, ledger.Channel.receive)
        self.assertIs(worker.Channel.send, ledger.Channel.send)
        self.assertIs(worker.Channel.before_io, ledger.Channel.before_io)
        self.assertEqual(ledger.READY, b"authenticated-ledger-ready-v1")
        self.assertEqual(guard.READY, b"kernel-guard-ready-v1")
        self.assertEqual(worker.READY, b"guarded-ledger-ready-v1")

    def test_source_parses_and_main_exits_without_printing_or_systemexit(self):
        raw = (ROOT / "scripts/sft_guarded_ledger_worker_v1.py").read_bytes()
        tree = ast.parse(raw)
        compile(tree, "<synthetic-wrapper>", "exec")
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertFalse(any(isinstance(node.func, ast.Name) and node.func.id in {
            "print", "open", "eval", "exec", "SystemExit", "__import__",
        } for node in calls))
        self.assertEqual(len([node for node in calls if isinstance(node.func, ast.Attribute)
                              and node.func.attr == "_exit"]), 1)

    def test_frozen_budgets_and_faults_preserved(self):
        self.assertEqual(worker.MAXIMUM_SECONDS, ledger.MAXIMUM_SECONDS)
        self.assertEqual(worker.MAXIMUM_MESSAGES, ledger.MAXIMUM_MESSAGES)
        self.assertEqual(worker.FAULTS - ledger.FAULTS, {"guard_exit_before_ready"})
        self.assertEqual(guard.EXPECTED_DENIAL_CHECKS, 17)
        self.assertEqual(guard.EXPECTED_ALLOWED_MEMORY_CHECKS, 1)
        self.assertEqual(guard.EXPECTED_MEMORY_CHECKS, 1)


class InstallationContractTests(unittest.TestCase):
    def test_exact_installation_contract_passes(self):
        self.assertTrue(worker.installed_exactly(dict(worker.EXPECTED_INSTALLATION)))

    def test_every_missing_or_changed_field_rejected(self):
        for key, expected in worker.EXPECTED_INSTALLATION.items():
            missing = dict(worker.EXPECTED_INSTALLATION)
            del missing[key]
            with self.subTest(key=key, change="missing"):
                self.assertFalse(worker.installed_exactly(missing))
            for wrong in (None, "wrong", 0, 1, True, False, 6.0, [], {}):
                if type(wrong) is type(expected) and wrong == expected:
                    continue
                altered = {**worker.EXPECTED_INSTALLATION, key: wrong}
                with self.subTest(key=key, type_name=type(wrong).__name__):
                    self.assertFalse(worker.installed_exactly(altered))

    def test_extra_fields_or_wrong_container_types_rejected(self):
        for value in (None, [], (), True, {**worker.EXPECTED_INSTALLATION, "extra": True}):
            with self.subTest(type_name=type(value).__name__):
                self.assertFalse(worker.installed_exactly(value))

        class DerivedDict(dict):
            pass

        self.assertFalse(worker.installed_exactly(DerivedDict(worker.EXPECTED_INSTALLATION)))


class ChannelTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(worker.time, "monotonic", return_value=100))
        self.enterContext(patch.object(worker.socket, "SCM_CREDENTIALS", 2, create=True))
        self.enterContext(patch.object(worker.socket, "MSG_CMSG_CLOEXEC", 1 << 30, create=True))
        self.stream = Mock()
        self.stream.fileno.return_value = 77
        self.stream.sendmsg.side_effect = lambda buffers, _ancillary: len(buffers[0])
        self.channel = worker.Channel(self.stream, 120, (1234, 1001, 1001))

    def packets(self):
        return [call.args[0][0] for call in self.stream.sendmsg.call_args_list]

    def test_first_ready_translated_exactly_once(self):
        self.channel.send_payload(ledger.READY)
        self.channel.send_payload(ledger.READY)
        self.assertEqual(self.packets(), [ledger.frame(worker.READY), ledger.frame(ledger.READY)])
        self.assertEqual(self.channel.sent, 2)

    def test_first_payload_must_be_exact_frozen_ready_without_rights(self):
        for payload, rights in ((b"wrong", False), (worker.READY, False),
                                (bytearray(ledger.READY), False), (ledger.READY, True),
                                (ledger.READY, 0)):
            with self.subTest(type_name=type(payload).__name__, rights=rights):
                channel = worker.Channel(self.stream, 120, (1234, 1001, 1001))
                with self.assertRaises(ValueError):
                    channel.send_payload(payload, rights=rights)
        self.stream.sendmsg.assert_not_called()

    def test_first_failed_transport_is_not_retranslated_on_later_direct_call(self):
        self.stream.sendmsg.side_effect = OSError
        with self.assertRaises(OSError):
            self.channel.send_payload(ledger.READY)
        self.stream.sendmsg.side_effect = lambda buffers, _ancillary: len(buffers[0])
        self.channel.send_payload(ledger.READY)
        self.assertEqual(self.packets(), [ledger.frame(worker.READY), ledger.frame(ledger.READY)])
        self.assertEqual(self.channel.sent, 2)

    def test_later_canonical_json_and_rights_delegate_unchanged(self):
        self.channel.send_payload(ledger.READY)
        value = {"schema_version": ledger.REQUEST_SCHEMA, "synthetic": 1}
        self.channel.send(value, rights=True)
        call = self.stream.sendmsg.call_args
        self.assertEqual(call.args[0], [ledger.frame(ledger.canonical(value))])
        self.assertEqual(call.args[1], [(worker.socket.SOL_SOCKET, worker.socket.SCM_RIGHTS,
                                        struct.pack("i", 77))])

    def test_frozen_send_limit_includes_first_ready(self):
        self.channel.send_payload(ledger.READY)
        for _index in range(31):
            self.channel.send_payload(b"synthetic")
        with self.assertRaises(ValueError):
            self.channel.send_payload(b"over budget")
        self.assertEqual(self.stream.sendmsg.call_count, 32)

    def test_frozen_credential_receive_and_decoder_are_unchanged(self):
        value = {"synthetic": 1}
        raw = ledger.frame(ledger.canonical(value))
        credentials = (worker.socket.SOL_SOCKET, worker.socket.SCM_CREDENTIALS,
                       struct.pack("iII", 1234, 1001, 1001))
        self.stream.recvmsg.return_value = (raw, [credentials], 0, None)
        self.assertEqual(self.channel.receive(), (raw, value))
        self.assertEqual(self.channel.received, 1)
        self.assertEqual(self.stream.settimeout.call_args.args, (3,))


class MainTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.enterContext(patch.object(worker.sys, "platform", "linux"))
        self.enterContext(patch.object(worker.sys, "flags", types.SimpleNamespace(
            no_site=1, optimize=0,
        )))
        self.enterContext(patch.object(worker.sys, "dont_write_bytecode", True))
        self.arguments = self.enterContext(patch.object(
            worker.sys, "argv", ["worker", "77", "none"],
        ))
        self.clock = self.enterContext(patch.object(worker.time, "monotonic", return_value=100))
        self.enterContext(patch.object(worker.os, "getppid", return_value=1234))
        self.enterContext(patch.object(worker.os, "getuid", return_value=1001))
        self.enterContext(patch.object(worker.os, "getgid", return_value=1001))
        self.enterContext(patch.object(worker.socket, "SO_PASSCRED", 16, create=True))
        self.stream = MagicMock(family=worker.socket.AF_UNIX)
        self.stream.__enter__.return_value = self.stream
        self.stream.__exit__.side_effect = lambda *_args: self.events.append("close") or False
        self.stream.getsockopt.side_effect = [worker.socket.SOCK_SEQPACKET, 1]
        self.stream.fileno.return_value = 77
        self.socket = self.enterContext(patch.object(worker.socket, "socket",
                                                     side_effect=self.make_socket))
        self.abi = self.enterContext(patch.object(policy, "require_abi",
                                                  side_effect=lambda: self.events.append("abi")))
        self.install = self.enterContext(patch.object(
            policy, "install", side_effect=self.install_guard,
        ))
        self.challenges = Mock()
        self.challenges.run.side_effect = self.run_challenges
        self.prepare = self.enterContext(patch.object(
            guard, "Challenges", side_effect=self.prepare_guard,
        ))
        self.original_channel = worker.Channel
        self.channel = self.enterContext(patch.object(
            worker, "Channel", side_effect=self.make_channel,
        ))
        self.conversation = self.enterContext(patch.object(ledger, "conversation",
                                                           side_effect=self.start_conversation))

    def make_socket(self, **kwargs):
        self.assertEqual(kwargs, {"fileno": 77})
        self.events.append("socket")
        return self.stream

    def prepare_guard(self, supplied):
        self.assertIs(supplied, policy)
        self.events.append("ctypes_ready")
        return self.challenges

    def make_channel(self, *arguments):
        self.assertEqual(arguments, (self.stream, 120, (1234, 1001, 1001)))
        self.stream.set_inheritable.assert_called_once_with(False)
        self.events.append("channel")
        return self.original_channel(*arguments)

    def install_guard(self, descriptor):
        self.assertEqual(descriptor, 77)
        self.assertEqual(self.events[-1], "channel")
        self.no_packets()
        self.events.append("install")
        return dict(worker.EXPECTED_INSTALLATION)

    def run_challenges(self):
        self.assertEqual(self.events[-1], "install")
        self.no_packets()
        self.events.append("challenges")

    def start_conversation(self, channel, fault):
        self.assertIsInstance(channel, self.original_channel)
        self.assertIn(fault, ledger.FAULTS)
        self.assertEqual(self.events[-1], "challenges")
        self.events.append("conversation")
        return 0

    def no_packets(self):
        self.stream.sendmsg.assert_not_called()
        self.stream.recvmsg.assert_not_called()

    def assert_no_ledger(self):
        self.no_packets()
        self.conversation.assert_not_called()

    def test_exact_guard_before_conversation_order_and_socket_cleanup(self):
        self.assertEqual(worker.main(), 0)
        self.assertEqual(self.events, [
            "abi", "ctypes_ready", "socket", "channel", "install", "challenges",
            "conversation", "close",
        ])
        self.assertEqual(self.channel.call_count, 1)
        self.assertEqual(self.install.call_count, 1)
        self.assertEqual(self.challenges.run.call_count, 1)
        self.assertEqual(self.stream.__exit__.call_count, 1)

    def test_guard_exit_fault_is_after_install_and_challenges_before_ready(self):
        self.arguments[2] = worker.GUARD_EXIT_FAULT
        self.assertEqual(worker.main(), 2)
        self.assertEqual(self.events[-3:], ["install", "challenges", "close"])
        self.assert_no_ledger()

    def test_every_original_fault_is_forwarded_without_translation(self):
        for fault in sorted(ledger.FAULTS):
            with self.subTest(fault=fault):
                self.arguments[2] = fault
                self.events.clear()
                self.stream.getsockopt.side_effect = [worker.socket.SOCK_SEQPACKET, 1]
                self.stream.set_inheritable.reset_mock()
                self.assertEqual(worker.main(), 0)
                self.assertEqual(self.conversation.call_args.args[1], fault)

    def test_install_exception_is_opaque_and_blocks_challenges_and_ready(self):
        self.install.side_effect = OSError("SECRET installation exception")
        self.assertEqual(worker.main(), 2)
        self.assert_no_ledger()
        self.challenges.run.assert_not_called()
        self.stream.__exit__.assert_called_once()

    def test_each_bad_installation_field_blocks_challenges_and_ready(self):
        for key in worker.EXPECTED_INSTALLATION:
            with self.subTest(key=key):
                self.install.side_effect = None
                self.install.return_value = {**worker.EXPECTED_INSTALLATION, key: None}
                self.stream.getsockopt.side_effect = [worker.socket.SOCK_SEQPACKET, 1]
                self.stream.set_inheritable.reset_mock()
                self.assertEqual(worker.main(), 2)
        self.assert_no_ledger()
        self.challenges.run.assert_not_called()

    def test_challenge_exception_blocks_ready_and_token_read(self):
        self.challenges.run.side_effect = ValueError("SECRET challenge exception")
        self.assertEqual(worker.main(), 2)
        self.assert_no_ledger()
        self.stream.__exit__.assert_called_once()

    def test_guard_deadline_exhaustion_blocks_ready_and_token_read(self):
        self.clock.side_effect = [100, 120]
        self.assertEqual(worker.main(), 2)
        self.assertEqual(self.events[-3:], ["install", "challenges", "close"])
        self.assert_no_ledger()

    def test_abi_failure_precedes_socket_install_and_challenge_preparation(self):
        self.abi.side_effect = ValueError("SECRET runtime exception")
        self.assertEqual(worker.main(), 2)
        self.socket.assert_not_called()
        self.install.assert_not_called()
        self.prepare.assert_not_called()
        self.assert_no_ledger()

    def test_preparation_failure_never_installs_or_sends(self):
        self.prepare.side_effect = OSError("SECRET preparation exception")
        self.assertEqual(worker.main(), 2)
        self.socket.assert_not_called()
        self.install.assert_not_called()
        self.assert_no_ledger()

    def test_wrong_socket_contract_never_installs(self):
        self.stream.getsockopt.side_effect = [worker.socket.SOCK_STREAM]
        self.assertEqual(worker.main(), 2)
        self.install.assert_not_called()
        self.assert_no_ledger()
        self.stream.__exit__.assert_called_once()

    def test_socket_noninherit_failure_never_installs(self):
        self.stream.set_inheritable.side_effect = OSError("SECRET descriptor exception")
        self.assertEqual(worker.main(), 2)
        self.install.assert_not_called()
        self.assert_no_ledger()
        self.stream.__exit__.assert_called_once()

    def test_cleanup_failure_denies_even_successful_conversation(self):
        self.stream.__exit__.side_effect = OSError("SECRET close exception")
        self.assertEqual(worker.main(), 2)
        self.assertEqual(self.conversation.call_count, 1)

    def test_conversation_exception_returns_failure_and_closes_once(self):
        self.conversation.side_effect = ValueError("SECRET ledger exception")
        self.assertEqual(worker.main(), 2)
        self.stream.__exit__.assert_called_once()

    def test_nonlinux_denies_before_abi_socket_or_install(self):
        worker.sys.platform = "unsupported"
        self.assertEqual(worker.main(), 2)
        self.abi.assert_not_called()
        self.socket.assert_not_called()
        self.install.assert_not_called()

    def test_bad_argument_shapes_deny_before_abi_socket_or_install(self):
        for arguments in (["worker"], ["worker", "2", "none"], ["worker", "1048577", "none"],
                          ["worker", "77", "unknown"], ["worker", "+77", "none"],
                          ["worker", "７７", "none"], ["worker", "77", "none", "extra"]):
            with self.subTest(length=len(arguments)):
                self.arguments[:] = arguments
                self.assertEqual(worker.main(), 2)
        self.abi.assert_not_called()
        self.socket.assert_not_called()
        self.install.assert_not_called()


if __name__ == "__main__":
    unittest.main()
