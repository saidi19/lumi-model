"""Pure mock worker tests: never install a filter, lower limits, or make syscalls."""

from __future__ import annotations

import errno
import importlib.util
import struct
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "_test_sft_kernel_guard_worker_v1", ROOT / "scripts/sft_kernel_guard_inert_worker_v1.py",
)
assert SPEC is not None and SPEC.loader is not None
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class WorkerChannelTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(worker.socket, "SCM_CREDENTIALS", 2, create=True))
        self.enterContext(patch.object(worker.socket, "MSG_CMSG_CLOEXEC", 1 << 30, create=True))
        self.clock = self.enterContext(patch.object(worker.time, "monotonic", return_value=100))
        self.stream = Mock()
        self.credentials = (1234, 1001, 1001)
        self.ancillary = [(worker.socket.SOL_SOCKET, worker.socket.SCM_CREDENTIALS,
                           struct.pack("iII", *self.credentials))]
        self.stream.recvmsg.return_value = (worker.frame(worker.GO), self.ancillary, 0, None)
        self.stream.sendmsg.side_effect = lambda buffers, _ancillary: len(buffers[0])
        self.channel = worker.Channel(self.stream, 106, self.credentials)

    def test_exact_two_send_receive_budget(self):
        for _index in range(2):
            self.channel.send(worker.READY)
            self.assertEqual(self.channel.receive(), worker.GO)
        with self.assertRaises(ValueError):
            self.channel.send(worker.READY)
        with self.assertRaises(ValueError):
            self.channel.receive()
        self.assertEqual(self.stream.sendmsg.call_count, 2)
        self.assertEqual(self.stream.recvmsg.call_count, 2)

    def test_absolute_deadline_does_not_reset(self):
        self.channel.send(worker.READY)
        self.clock.return_value = 105
        self.assertEqual(self.channel.receive(), worker.GO)
        self.assertEqual(self.stream.settimeout.call_args.args, (1,))
        self.clock.return_value = 106
        with self.assertRaises(ValueError):
            self.channel.send(worker.PROOF)
        self.assertEqual(self.channel.sent, 2)
        self.assertEqual(self.stream.sendmsg.call_count, 1)

    def test_partial_send_is_terminal_for_conversation(self):
        self.stream.sendmsg.return_value = 1
        self.stream.sendmsg.side_effect = None
        with self.assertRaises(ValueError):
            self.channel.send(worker.READY)
        self.assertEqual(self.channel.sent, 1)

    def test_receive_failure_keeps_precharge(self):
        self.stream.recvmsg.side_effect = OSError
        with self.assertRaises(OSError):
            self.channel.receive()
        self.assertEqual(self.channel.received, 1)

    def test_credential_pid_uid_gid_must_match(self):
        for values in ((4321, 1001, 1001), (1234, 1002, 1001), (1234, 1001, 1002)):
            with self.subTest(values=values):
                channel = worker.Channel(self.stream, 106, self.credentials)
                self.stream.recvmsg.return_value = (
                    worker.frame(worker.GO),
                    [(worker.socket.SOL_SOCKET, worker.socket.SCM_CREDENTIALS,
                      struct.pack("iII", *values))], 0, None,
                )
                with self.assertRaises(ValueError):
                    channel.receive()

    def test_flags_and_frames_fail_closed(self):
        cases = (
            (b"", 0), (b"G", 0), (b"\x00\x00\x00\x02G", 0),
            (worker.frame(worker.GO), 8), (worker.frame(worker.GO), True),
            (b"\x00\x00\x00\x41" + b"x" * 65, 0),
        )
        for raw, flags in cases:
            with self.subTest(raw_length=len(raw), flags=flags):
                self.stream.recvmsg.return_value = (raw, self.ancillary, flags, None)
                with self.assertRaises(ValueError):
                    worker.Channel(self.stream, 106, self.credentials).receive()

    def test_missing_or_additional_ancillary_rejected(self):
        for ancillary in ([], self.ancillary * 2, [(0, 0, b"")]):
            with self.subTest(count=len(ancillary)):
                self.stream.recvmsg.return_value = (worker.frame(worker.GO), ancillary, 0, None)
                with self.assertRaises(ValueError):
                    worker.Channel(self.stream, 106, self.credentials).receive()

    def test_rights_closed_before_flag_or_deadline_rejection(self):
        rights = (worker.socket.SOL_SOCKET, worker.socket.SCM_RIGHTS, struct.pack("ii", 77, 78))
        self.stream.recvmsg.return_value = (b"", [rights, *self.ancillary], 8, None)
        self.clock.side_effect = [100, 107]
        with patch.object(worker.os, "close") as close, self.assertRaises(ValueError):
            self.channel.receive()
        self.assertEqual([call.args for call in close.call_args_list], [(77,), (78,)])

    def test_rights_cleanup_attempts_every_unique_descriptor_once(self):
        rights = [(worker.socket.SOL_SOCKET, worker.socket.SCM_RIGHTS,
                   struct.pack("iii", 77, 77, 78) + b"x")]
        with patch.object(worker.os, "close", side_effect=[OSError, None]) as close:
            self.assertFalse(worker.close_rights(rights))
        self.assertEqual([call.args for call in close.call_args_list], [(77,), (78,)])

    def test_payload_size_and_type_are_closed(self):
        for payload in (b"", b"x" * 65, "G", bytearray(b"G")):
            with self.subTest(type_name=type(payload).__name__):
                with self.assertRaises(ValueError):
                    worker.frame(payload)
        self.assertEqual(len(worker.frame(b"x" * 64)), 68)


class WorkerConversationTests(unittest.TestCase):
    def setUp(self):
        self.channel = Mock()
        self.channel.receive.side_effect = [worker.GO, worker.RELEASE]
        self.challenges = Mock()

    def test_fixed_success_transcript(self):
        self.assertEqual(worker.conversation(self.channel, self.challenges, "none"), 0)
        self.assertEqual([call.args for call in self.channel.send.call_args_list],
                         [(worker.READY,), (worker.PROOF,)])
        self.challenges.run.assert_called_once_with()
        self.assertEqual(self.channel.receive.call_count, 2)

    def test_no_proof_on_challenge_failure(self):
        self.challenges.run.side_effect = ValueError
        with self.assertRaises(ValueError):
            worker.conversation(self.channel, self.challenges, "none")
        self.channel.send.assert_called_once_with(worker.READY)
        self.assertEqual(self.channel.receive.call_count, 1)

    def test_no_challenge_before_exact_go(self):
        self.channel.receive.side_effect = [b"bad"]
        with self.assertRaises(ValueError):
            worker.conversation(self.channel, self.challenges, "none")
        self.challenges.run.assert_not_called()

    def test_release_must_be_exact(self):
        self.channel.receive.side_effect = [worker.GO, b"bad"]
        with self.assertRaises(ValueError):
            worker.conversation(self.channel, self.challenges, "none")

    def test_exit_after_ready_sends_no_proof(self):
        self.assertEqual(worker.conversation(self.channel, self.challenges, "exit_after_ready"), 2)
        self.channel.send.assert_called_once_with(worker.READY)
        self.channel.receive.assert_not_called()
        self.challenges.run.assert_not_called()

    def test_stall_is_bounded_by_second_channel_receive(self):
        self.assertEqual(worker.conversation(self.channel, self.challenges, "stall"), 2)
        self.assertEqual(self.channel.receive.call_count, 2)
        self.channel.send.assert_called_once_with(worker.READY)
        self.challenges.run.assert_not_called()


class WorkerChallengeTests(unittest.TestCase):
    def setUp(self):
        names = (
            "socket", "connect", "openat", "clone3", "execve", "unshare", "setns", "ptrace",
            "prlimit64", "io_uring_setup", "sendmsg", "recvmsg", "read", "write", "dup3",
        )
        self.policy = SimpleNamespace(SYSCALLS=dict(zip(names, range(len(names)), strict=True)))
        self.libc = Mock()
        self.enterContext(patch.object(worker.ctypes, "CDLL", return_value=self.libc))
        self.challenges = worker.Challenges(self.policy)
        self.libc.syscall.return_value = -1
        self.libc.mmap.side_effect = [4096, self.challenges.map_failed]
        self.libc.munmap.return_value = 0
        self.errno = self.enterContext(patch.object(
            worker.ctypes, "get_errno", side_effect=[errno.EPERM] * 17 + [errno.ENOMEM],
        ))
        self.enterContext(patch.object(worker.ctypes, "set_errno"))

    def test_fixed_typed_syscalls_and_memory_challenges(self):
        self.challenges.run()
        self.assertEqual(self.libc.syscall.call_count, worker.EXPECTED_DENIAL_CHECKS)
        self.assertEqual(self.libc.syscall.argtypes, [worker.ctypes.c_long] * 7)
        self.assertEqual(self.libc.mmap.call_count, 2)
        self.assertEqual([call.args[1] for call in self.libc.mmap.call_args_list],
                         [4096, 129 * 1024 * 1024])
        self.libc.munmap.assert_called_once_with(4096, 4096)
        for call in self.libc.syscall.call_args_list:
            self.assertEqual(len(call.args), 7)

    def test_successful_socket_regression_closes_descriptor_and_denies(self):
        self.libc.syscall.return_value = 77
        with patch.object(worker.os, "close") as close, self.assertRaises(ValueError):
            self.challenges.run()
        close.assert_called_once_with(77)
        self.libc.mmap.assert_not_called()
        self.assertEqual(self.libc.syscall.call_count, 1)

    def test_socket_cleanup_failure_does_not_retry(self):
        self.libc.syscall.return_value = 77
        with patch.object(worker.os, "close", side_effect=OSError) as close:
            with self.assertRaises(OSError):
                self.challenges.run()
        close.assert_called_once_with(77)
        self.libc.mmap.assert_not_called()

    def test_wrong_denial_errno_fails_before_memory(self):
        self.errno.side_effect = [errno.EBADF]
        with self.assertRaises(ValueError):
            self.challenges.run()
        self.libc.mmap.assert_not_called()

    def test_small_mapping_must_succeed(self):
        self.libc.mmap.side_effect = [self.challenges.map_failed]
        with self.assertRaises(ValueError):
            self.challenges.run()
        self.libc.munmap.assert_not_called()

    def test_small_mapping_cleanup_must_succeed(self):
        self.libc.munmap.return_value = -1
        with self.assertRaises(ValueError):
            self.challenges.run()
        self.assertEqual(self.libc.mmap.call_count, 1)

    def test_overlimit_mapping_regression_is_unmapped_and_denied(self):
        self.libc.mmap.side_effect = [4096, 8192]
        with self.assertRaises(ValueError):
            self.challenges.run()
        self.assertEqual([call.args for call in self.libc.munmap.call_args_list],
                         [(4096, 4096), (8192, worker.MEMORY_CHALLENGE_BYTES)])

    def test_overlimit_mapping_requires_enomem_not_only_failure(self):
        self.errno.side_effect = [errno.EPERM] * 18
        with self.assertRaises(ValueError):
            self.challenges.run()


class WorkerSetupTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.policy = Mock(LIMITS={
            worker.resource.RLIMIT_AS: (worker.ADDRESS_SPACE_BYTES, worker.ADDRESS_SPACE_BYTES),
        })
        self.policy.require_abi.side_effect = lambda: self.events.append("abi")
        self.policy.install.side_effect = self.install
        self.enterContext(patch.dict(worker.__dict__, {"kernel_policy": self.policy}))
        self.enterContext(patch.object(worker.sys, "platform", "linux"))
        self.enterContext(patch.object(worker.sys, "flags", SimpleNamespace(no_site=1, optimize=0)))
        self.enterContext(patch.object(worker.sys, "dont_write_bytecode", True))
        self.enterContext(patch.object(worker.sys, "argv", ["worker", "77", "none"]))
        self.enterContext(patch.object(worker.os, "getppid", return_value=1234))
        self.enterContext(patch.object(worker.os, "getuid", return_value=1001))
        self.enterContext(patch.object(worker.os, "getgid", return_value=1001))
        self.enterContext(patch.object(worker.socket, "SO_PASSCRED", 16, create=True))
        self.stream = MagicMock(family=worker.socket.AF_UNIX)
        self.stream.__enter__.return_value = self.stream
        self.stream.getsockopt.side_effect = [worker.socket.SOCK_SEQPACKET, 1]
        self.stream.fileno.return_value = 77
        self.socket = self.enterContext(patch.object(worker.socket, "socket",
                                                     return_value=self.stream))
        self.challenges = self.enterContext(patch.object(
            worker, "Challenges", side_effect=lambda _policy: self.events.append("libc_ready"),
        ))
        self.conversation = self.enterContext(patch.object(
            worker, "conversation", side_effect=lambda *_args: self.events.append("ready") or 0,
        ))

    def install(self, _fd):
        self.events.append("install")
        return {
            "policy_installed": True, "resource_limits_verified": 6,
            "no_new_privileges": True, "seccomp_filter_mode_verified": True,
        }

    def test_libc_preparation_precedes_install_and_ready_follows_install(self):
        self.assertEqual(worker.main(), 0)
        self.assertEqual(self.events, ["abi", "libc_ready", "install", "ready"])
        self.policy.install.assert_called_once_with(77)
        self.stream.set_inheritable.assert_called_once_with(False)
        self.stream.__exit__.assert_called_once()

    def test_unverified_install_never_sends_ready(self):
        self.policy.install.side_effect = lambda _fd: {"policy_installed": False}
        self.assertEqual(worker.main(), 2)
        self.conversation.assert_not_called()
        self.stream.__exit__.assert_called_once()

    def test_unsupported_runtime_denies_before_socket_or_install(self):
        worker.sys.platform = "unsupported"
        self.assertEqual(worker.main(), 2)
        self.socket.assert_not_called()
        self.policy.install.assert_not_called()
        self.challenges.assert_not_called()

    def test_wrong_resource_contract_denies_before_socket(self):
        self.policy.LIMITS[worker.resource.RLIMIT_AS] = (1, 1)
        self.assertEqual(worker.main(), 2)
        self.socket.assert_not_called()
        self.policy.install.assert_not_called()


if __name__ == "__main__":
    unittest.main()
