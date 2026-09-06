"""Synthetic parent boundaries and mandatory Linux fixed-child kernel checks."""

from __future__ import annotations

import functools
import hashlib
import importlib
import io
import json
import struct
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_test_sft_kernel_guard_probe_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
probe = importlib.import_module(PACKAGE + ".posttraining_sft_kernel_guard_probe_v1")
NONCLAIMS = (
    "cgroup_memory_enforcement", "aggregate_descendant_enforcement", "os_network_enforcement",
    "namespace_containment", "ptrace_containment", "continuous_privilege_exclusion",
    "durable_ledger_integrated", "real_worker_sandbox_verified", "target_gpu_host_verified",
    "guarded_real_worker_ready", "execution_authorized", "production_eligible",
    "full_private_integration_passed",
)


def isolated(method):
    @functools.wraps(method)
    def run(self):
        if sys.flags.no_site == 1 and sys.dont_write_bytecode:
            return method(self)
        result = subprocess.run(
            [sys.executable, "-S", "-B", str(Path(__file__).resolve()),
             type(self).__name__ + "." + method.__name__],
            stdin=subprocess.DEVNULL, capture_output=True, timeout=15,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )
        self.assertEqual(result.returncode, 0, (result.stdout + result.stderr)[-4000:].decode())
    return run


def status_bytes(**changes):
    fields = {
        "Pid": "12345", "Tgid": "12345", "PPid": "23456", "TracerPid": "0",
        "Threads": "1", "Uid": "34567\t34567\t34567\t34567",
        "Gid": "45678\t45678\t45678\t45678", "NSpid": "12345", "NStgid": "12345",
        "CapInh": "0000000000000000", "CapPrm": "0000000000000000",
        "CapEff": "0000000000000000", "CapAmb": "0000000000000000",
        "CapBnd": "0000000000000001", "NoNewPrivs": "1",
        "Seccomp": "2", "Seccomp_filters": "2",
    }
    fields.update(changes)
    return "".join(f"{key}:\t{value}\n" for key, value in fields.items()).encode()


class SnapshotTests(unittest.TestCase):
    def parse(self, raw):
        return probe.parse_guard_status(raw, expected_pid=12345, expected_ppid=23456,
                                        expected_uid=34567, expected_gid=45678)

    def test_valid_filter_snapshot_keeps_old_policy_and_no_attestation_claim(self):
        for count in ("1", "2", "64"):
            value = self.parse(status_bytes(Seccomp_filters=count))
            self.assertTrue(value["child_status_policy_passed"])
            self.assertTrue(value["filter_mode_snapshot"])
            self.assertEqual(value["filter_count_snapshot"], int(count))
            self.assertFalse(value["permanent_privilege_containment"])
            self.assertFalse(value["execution_authorized"])
            self.assertNotIn("12345", json.dumps(value))

    def test_missing_disabled_strict_or_malformed_filter_metadata_denies(self):
        for raw in (
            status_bytes().replace(b"Seccomp:\t2\n", b""),
            status_bytes().replace(b"Seccomp_filters:\t2\n", b""),
            *(status_bytes(Seccomp=value) for value in ("0", "1", "3", "02", "2 2")),
            *(status_bytes(Seccomp_filters=value)
              for value in ("0", "01", "65", "100", "-1", "2 2", "x")),
        ):
            with self.subTest(size=len(raw)), self.assertRaises(probe.GuardError) as caught:
                self.parse(raw)
            self.assertEqual(caught.exception.code, "guard_snapshot_rejected")

    def test_duplicates_and_old_identity_privilege_or_thread_policy_still_deny(self):
        for raw in (
            status_bytes() + b"Seccomp:\t2\n",
            status_bytes() + b"Seccomp_filters:\t2\n",
            status_bytes(PPid="1"), status_bytes(Threads="2"), status_bytes(NoNewPrivs="0"),
            status_bytes(CapEff="0000000000000001"), status_bytes(TracerPid="1"),
        ):
            with self.subTest(size=len(raw)), self.assertRaises(probe.lifecycle.policy.StatusError):
                self.parse(raw)

    def test_bracketed_snapshot_uses_unchanged_read_capacity_and_closes_once(self):
        reads = probe.GuardReads()
        owner = Mock(child=Mock(pid=12345), assert_live=Mock())
        with (
            patch.object(probe.time, "monotonic", return_value=10),
            patch.object(probe.os, "getpid", return_value=23456),
            patch.object(probe.os, "getuid", return_value=34567),
            patch.object(probe.os, "getgid", return_value=45678),
            patch.object(probe.os, "open", return_value=71) as opened,
            patch.object(probe.os, "read", return_value=status_bytes()) as read,
            patch.object(probe.os, "close") as close,
        ):
            value = reads.child(owner, 20)
        self.assertTrue(value["filter_mode_snapshot"])
        self.assertEqual(owner.assert_live.call_count, 2)
        self.assertEqual((reads.attempts, reads.closed, reads.capacity_bytes),
                         (1, 1, probe.lifecycle.STATUS_CAPACITY))
        opened.assert_called_once_with("/proc/12345/status",
                                       probe.os.O_RDONLY | probe.os.O_CLOEXEC | probe.os.O_NOFOLLOW)
        read.assert_called_once_with(71, probe.lifecycle.STATUS_CAPACITY)
        close.assert_called_once_with(71)


class ParentTests(unittest.TestCase):
    def boundaries(self, value):
        for key in NONCLAIMS:
            self.assertIs(value[key], False, key)
        for key in ("training_source_or_data_reads", "model_or_checkpoint_reads",
                    "private_artifact_writes", "external_network_bytes", "gpu_use",
                    "inference_calls"):
            self.assertEqual(value[key], 0)
        self.assertLessEqual(value["worker_processes_started"], 1)
        self.assertLessEqual(value["status_read_attempts"], 3)
        self.assertLessEqual(value["status_read_capacity_bytes"],
                             3 * probe.lifecycle.STATUS_CAPACITY)
        self.assertLessEqual(value["total_receive_capacity_bytes"],
                             3 * probe.credentials.RECEIVE_CAPACITY_BYTES)
        self.assertLessEqual(value["parent_send_bytes_precharged"], 10)
        for secret in ("SECRET", "Traceback", "12345", "NoNewPrivs", '"token"'):
            self.assertNotIn(secret, json.dumps(value))

    def fake(self):
        self.enterContext(patch.object(probe.lifecycle, "require_runtime"))
        self.enterContext(patch.object(probe.kernel_policy, "require_abi"))
        self.install = self.enterContext(patch.object(probe.kernel_policy, "install"))
        self.normalize = self.enterContext(patch.object(probe.lifecycle, "normalize_child_signal"))
        self.source = self.enterContext(patch.object(probe, "authenticate_sources",
                                                    return_value=(b"policy", b"worker")))
        self.clock = self.enterContext(patch.object(probe.time, "monotonic", return_value=100))
        self.enterContext(patch.object(probe.socket, "MSG_CMSG_CLOEXEC", 1 << 30, create=True))
        self.enterContext(patch.object(probe.socket, "SCM_CREDENTIALS", 2, create=True))
        self.parent, self.endpoint = Mock(), Mock()
        self.parent.sendmsg.side_effect = lambda buffers: len(buffers[0])
        self.parent.recvmsg.return_value = (b"", [], 0, None)
        self.enterContext(patch.object(probe.credentials, "credential_pair",
                                      return_value=(self.parent, self.endpoint)))
        self.owner = Mock(child=Mock(pid=12345), fd=71, owned=True, bound_live=True,
                          reaped=False, closed=False, close_attempted=False,
                          signal_attempts=0, assert_live=Mock())
        self.owner.reap.side_effect = lambda *_a, **_kw: setattr(self.owner, "reaped", True)
        self.owner.terminate_and_reap.side_effect = lambda *_a: setattr(self.owner, "reaped", True)
        self.owner.close.side_effect = lambda: setattr(self.owner, "closed", True)
        self.spawn = self.enterContext(patch.object(probe, "spawn", return_value=self.owner.child))
        self.enterContext(patch.object(probe.lifecycle, "OwnedChild", return_value=self.owner))
        self.reads = Mock(attempts=3, closed=3, capacity_bytes=3 * probe.lifecycle.STATUS_CAPACITY,
                          cleanup_failed=False)
        self.reads.child.return_value = {"filter_mode_snapshot": True, "filter_count_snapshot": 2}
        self.enterContext(patch.object(probe, "GuardReads", return_value=self.reads))
        self.channel = Mock()
        self.channel.receive.side_effect = [len(raw).to_bytes(4, "big") + raw
                                            for raw in (probe.READY, probe.PROOF)]
        self.channel.summary.return_value = {
            "read_capacity_bytes": 2 * probe.credentials.RECEIVE_CAPACITY_BYTES,
            "verified_messages": 2, "received_descriptors_closed": 0,
        }
        self.construct = self.enterContext(patch.object(probe.credentials, "CredentialChannel",
                                                       return_value=self.channel))

    def report(self, **kwargs):
        value = probe.run_kernel_guard_probe(**kwargs)
        self.boundaries(value)
        return value

    @isolated
    def test_mock_success_checks_two_snapshots_reap_eof_and_cleanup(self):
        self.fake()
        value = self.report()
        self.assertEqual(value["state"], "passed", value["failure_code"])
        self.assertTrue(value["kernel_guard_probe_passed"])
        self.assertTrue(value["worker_cleanup_complete"])
        self.assertEqual(value["filter_snapshots_verified"], 2)
        self.assertEqual(value["denied_syscall_challenges_passed"], 17)
        self.assertEqual(value["address_space_rejection_challenges_passed"], 1)
        self.assertEqual(value["positive_anonymous_mapping_challenges_passed"], 1)
        self.assertEqual(value["worker_processes_reaped"], 1)
        self.assertTrue(value["pidfd_closed"])
        self.assertTrue(value["terminal_eof_verified"])
        self.assertEqual(value["parent_send_bytes_precharged"], 10)
        self.owner.bind.assert_called_once()
        self.owner.reap.assert_called_once()
        self.owner.close.assert_called_once()
        self.channel.close.assert_called_once()
        self.endpoint.close.assert_called_once()
        self.parent.close.assert_not_called()  # Channel owns the endpoint.
        self.install.assert_not_called()  # No caller/parent confinement mutation.

    @isolated
    def test_policy_and_snapshot_validation_precede_each_parent_ack(self):
        self.fake()
        events = []
        self.reads.child.side_effect = lambda *_a: events.append("snapshot")
        self.parent.sendmsg.side_effect = lambda frames: events.append("ack") or len(frames[0])
        self.owner.reap.side_effect = lambda *_a, **_kw: (
            events.append("reap"), setattr(self.owner, "reaped", True),
        )
        self.parent.recvmsg.side_effect = lambda *_a: events.append("eof") or (b"", [], 0, None)
        self.assertEqual(self.report()["state"], "passed")
        self.assertEqual(events, ["snapshot", "ack", "snapshot", "ack", "reap", "eof"])

    @isolated
    def test_missing_filter_snapshot_denies_before_permission(self):
        self.fake()
        self.reads.child.side_effect = probe.GuardError("guard_snapshot_rejected")
        value = self.report()
        self.assertEqual(value["failure_code"], "guard_snapshot_rejected")
        self.parent.sendmsg.assert_not_called()
        self.assertEqual(value["denied_syscall_challenges_passed"], 0)

    @isolated
    def test_ready_or_proof_mismatch_cannot_pass(self):
        self.fake()
        self.channel.receive.side_effect = [len(probe.READY).to_bytes(4, "big") + probe.READY,
                                            b"\0\0\0\x01x"]
        value = self.report()
        self.assertEqual(value["failure_code"], "guard_frame_rejected")
        self.assertEqual(self.parent.sendmsg.call_count, 1)
        self.assertEqual(value["filter_snapshots_verified"], 1)
        self.assertEqual(value["denied_syscall_challenges_passed"], 0)

    @isolated
    def test_short_ack_is_charged_and_cannot_advance(self):
        self.fake()
        self.parent.sendmsg.side_effect = None
        self.parent.sendmsg.return_value = 1
        value = self.report()
        self.assertEqual(value["failure_code"], "guard_frame_rejected")
        self.assertEqual(value["parent_send_bytes_precharged"], 5)
        self.assertEqual(self.channel.receive.call_count, 1)
        self.parent.recvmsg.assert_not_called()

    @isolated
    def test_failed_endpoint_close_is_sticky_and_not_retried(self):
        self.fake()
        self.endpoint.close.side_effect = OSError("SECRET")
        value = self.report()
        self.assertEqual(value["failure_code"], "guard_cleanup_failed")
        self.assertFalse(value["worker_cleanup_complete"])
        self.endpoint.close.assert_called_once()
        self.parent.close.assert_called_once()
        self.construct.assert_not_called()

    @isolated
    def test_channel_construction_failure_keeps_parent_owned_for_cleanup(self):
        self.fake()
        self.construct.side_effect = OSError("SECRET")
        value = self.report()
        self.assertEqual(value["state"], "failed")
        self.parent.close.assert_called_once()
        self.endpoint.close.assert_called_once()
        self.owner.close.assert_called_once()

    @isolated
    def test_unbound_child_cannot_claim_verified_reap_or_pidfd_close(self):
        self.fake()
        self.owner.fd = None
        self.owner.owned = self.owner.bound_live = False
        self.owner.bind.side_effect = OSError("SECRET")
        value = self.report()
        self.assertEqual(value["failure_code"], "guard_cleanup_failed")
        self.assertFalse(value["pidfd_closed"])
        self.assertEqual(value["worker_processes_reaped"], 0)
        self.owner.child.wait.assert_called_once()
        self.owner.terminate_and_reap.assert_not_called()

    @isolated
    def test_pidfd_close_failure_denies_after_valid_proof(self):
        self.fake()
        self.owner.close.side_effect = OSError("SECRET")
        value = self.report()
        self.assertEqual(value["failure_code"], "guard_cleanup_failed")
        self.assertFalse(value["kernel_guard_probe_passed"])
        self.owner.close.assert_called_once()

    @isolated
    def test_channel_cleanup_failure_is_not_hidden_by_reap_success(self):
        self.fake()
        self.channel.close.side_effect = OSError("SECRET")
        value = self.report()
        self.assertEqual(value["failure_code"], "guard_cleanup_failed")
        self.assertEqual(value["worker_processes_reaped"], 1)
        self.assertFalse(value["kernel_guard_probe_passed"])
        self.channel.close.assert_called_once()

    @isolated
    def test_descriptor_receive_cleanup_failure_remains_sticky(self):
        self.fake()
        self.channel.receive.side_effect = probe.credentials.PeerCredentialError(
            "credential_cleanup_failed",
        )
        value = self.report()
        self.assertEqual(value["failure_code"], "guard_cleanup_failed")
        self.assertFalse(value["worker_cleanup_complete"])
        self.parent.sendmsg.assert_not_called()

    @isolated
    def test_empty_credential_packet_is_not_eof(self):
        self.fake()
        self.parent.recvmsg.return_value = (b"", [(probe.socket.SOL_SOCKET,
                                                 probe.socket.SCM_CREDENTIALS,
                                                 struct.pack("iII", 12345, 34567, 45678))], 0, None)
        value = self.report()
        self.assertEqual(value["failure_code"], "guard_terminal_rejected")
        self.assertFalse(value["terminal_eof_verified"])
        self.assertEqual(value["terminal_receive_attempts"], 1)

    @isolated
    def test_terminal_rights_close_precedes_flags_data_and_deadline_rejection(self):
        self.fake()
        rights = [(probe.socket.SOL_SOCKET, probe.socket.SCM_RIGHTS, struct.pack("ii", 81, 82))]

        def terminal(*_args):
            self.clock.return_value = 107
            return b"x", rights, probe.socket.MSG_CTRUNC, None
        self.parent.recvmsg.side_effect = terminal
        with patch.object(probe.os, "close") as close:
            value = self.report()
        self.assertEqual(close.call_args_list, [call(81), call(82)])
        self.assertEqual(value["terminal_descriptors_closed"], 2)
        self.assertEqual(value["failure_code"], "lifecycle_deadline_exhausted")
        self.assertFalse(value["kernel_guard_probe_passed"])

    @isolated
    def test_bad_exit_cannot_reach_terminal_read(self):
        self.fake()
        self.owner.reap.side_effect = probe.lifecycle.LifecycleError("lifecycle_exit_rejected")
        value = self.report()
        self.assertEqual(value["failure_code"], "lifecycle_exit_rejected")
        self.parent.recvmsg.assert_not_called()

    @isolated
    def test_deadline_after_cleanup_denies_overall_success(self):
        self.fake()

        def close():
            self.owner.closed = True
            self.clock.return_value = 107
        self.owner.close.side_effect = close
        value = self.report()
        self.assertEqual(value["failure_code"], "lifecycle_deadline_exhausted")
        self.assertTrue(value["terminal_eof_verified"])
        self.assertFalse(value["kernel_guard_probe_passed"])

    @isolated
    def test_unsupported_runtime_denies_before_source_or_controller_read(self):
        with (
            patch.object(probe.lifecycle, "require_runtime", side_effect=
                         probe.lifecycle.LifecycleError("lifecycle_unavailable")),
            patch.object(probe.GuardReads, "controller") as controller,
            patch.object(probe, "authenticate_sources") as source,
            patch.object(probe, "spawn") as spawn,
            patch.object(probe.kernel_policy, "install") as install,
        ):
            value = self.report()
        self.assertEqual(value["state"], "unsupported")
        for operation in (controller, source, spawn, install):
            operation.assert_not_called()

    @isolated
    def test_unsupported_abi_denies_before_controller_read_or_spawn(self):
        with (
            patch.object(probe.lifecycle, "require_runtime"),
            patch.object(probe.kernel_policy, "require_abi", side_effect=
                         probe.kernel_policy.PolicyError("kernel_policy_unavailable")),
            patch.object(probe.GuardReads, "controller") as controller,
            patch.object(probe, "spawn") as spawn,
        ):
            value = self.report()
        self.assertEqual(value["state"], "unsupported")
        controller.assert_not_called()
        spawn.assert_not_called()

    @isolated
    def test_source_hash_failure_precedes_socket_and_worker_creation(self):
        self.fake()
        self.source.side_effect = probe.GuardError("guard_source_rejected")
        value = self.report()
        self.assertEqual(value["failure_code"], "guard_source_rejected")
        self.spawn.assert_not_called()

    def test_exact_sources_are_authenticated_and_not_reopened_during_spawn(self):
        sources = probe.authenticate_sources()
        for raw, (_path, capacity, expected) in zip(sources, probe.SOURCE_BINDINGS, strict=True):
            self.assertLessEqual(len(raw), capacity)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), expected)
        with patch.object(Path, "open") as opened, patch.object(probe.subprocess, "Popen") as spawn:
            probe.spawn(Mock(fileno=Mock(return_value=71)), "none", sources)
        opened.assert_not_called()
        args, kwargs = spawn.call_args
        self.assertEqual(args[0][:4], [sys.executable, "-S", "-B", "-c"])
        self.assertEqual(args[0][-2:], ["71", "none"])
        self.assertIn("types.ModuleType", args[0][4])
        self.assertIn("'kernel_policy': policy", args[0][4])
        self.assertEqual(kwargs["pass_fds"], (71,))
        self.assertTrue(kwargs["close_fds"])
        self.assertEqual(kwargs["env"], {"PATH": "/usr/bin:/bin", "LC_ALL": "C"})

    def test_changed_and_oversized_source_are_bounded_and_rejected(self):
        for raw in (b"changed", b"x" * 32769):
            stream = Mock(wraps=io.BytesIO(raw))
            stream.__enter__ = Mock(return_value=stream)
            stream.__exit__ = Mock(return_value=False)
            with (patch.object(Path, "open", return_value=stream),
                  self.assertRaises(probe.GuardError)):
                probe.authenticate_sources()
            stream.read.assert_called_once_with(32769)

    def test_exact_bootstrap_compiles_both_namespaces_and_denies_invalid_args_before_mutation(self):
        sources = probe.authenticate_sources()
        with patch.object(probe.subprocess, "Popen") as spawn:
            probe.spawn(Mock(fileno=Mock(return_value=71)), "none", sources)
        code = spawn.call_args.args[0][4]
        with (
            patch.object(sys, "argv", ["-c"]),
            patch.object(probe.os, "_exit") as exit_child,
            patch.object(probe.socket, "socket") as create_socket,
            patch.object(probe.kernel_policy.resource, "setrlimit") as set_limits,
            patch.object(probe.kernel_policy.ctypes, "CDLL") as libc,
        ):
            exec(compile(code, "<synthetic-bootstrap>", "exec"), {})
        exit_child.assert_called_once_with(2)
        for operation in (create_socket, set_limits, libc):
            operation.assert_not_called()

    def test_untrusted_spawn_bytes_or_fault_never_launch(self):
        with patch.object(probe.subprocess, "Popen") as spawn:
            for sources, fault in (((b"x", b"y"), "none"), ((b"x",), "none"),
                                   ((b"x", b"y"), "unknown")):
                with self.assertRaises(probe.GuardError):
                    probe.spawn(Mock(), fault, sources)
            spawn.assert_not_called()

    def live(self, fault="none"):
        value = self.report(fault=fault)
        if sys.platform != "linux":
            self.assertEqual(value["state"], "unsupported")
            self.assertEqual(value["worker_processes_started"], 0)
            self.assertEqual(value["status_read_attempts"], 0)
            return None
        self.assertTrue(value["worker_cleanup_complete"], value["failure_code"])
        self.assertEqual(value["worker_processes_started"], 1, value["failure_code"])
        self.assertEqual(value["worker_processes_reaped"], 1, value["failure_code"])
        self.assertTrue(value["pidfd_closed"])
        return value

    @isolated
    def test_live_linux_kernel_policy_or_nonlinux_denial(self):
        value = self.live()
        if value is not None:
            self.assertEqual(value["state"], "passed", value["failure_code"])
            self.assertEqual(value["filter_snapshots_verified"], 2)
            self.assertEqual(value["denied_syscall_challenges_passed"], 17)
            self.assertEqual(value["address_space_rejection_challenges_passed"], 1)
            self.assertEqual(value["positive_anonymous_mapping_challenges_passed"], 1)
            self.assertTrue(value["terminal_eof_verified"])

    @isolated
    def test_live_early_exit_denies_proof_and_reaps(self):
        value = self.live("exit_after_ready")
        if value is not None:
            self.assertEqual(value["state"], "failed")
            self.assertEqual(value["denied_syscall_challenges_passed"], 0)

    @isolated
    def test_live_stall_denies_proof_and_reaps_without_retry(self):
        value = self.live("stall")
        if value is not None:
            self.assertEqual(value["state"], "failed")
            self.assertEqual(value["denied_syscall_challenges_passed"], 0)


if __name__ == "__main__":
    unittest.main()
