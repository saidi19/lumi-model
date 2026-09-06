"""Synthetic lifecycle guards plus fixed inert Linux fixtures; no training data."""

from __future__ import annotations

import functools
import importlib
import io
import json
import os
import subprocess
import sys
import types
import unittest
from collections import namedtuple
from pathlib import Path
from unittest.mock import Mock, call, patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_test_sft_peer_lifecycle_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
probe = importlib.import_module(PACKAGE + ".posttraining_sft_peer_lifecycle_v1")
Record = namedtuple("Record", "si_pid si_uid si_signo si_status si_code")
NONCLAIMS = (
    "privileged_sender_capabilities_excluded", "continuous_capability_exclusion",
    "durable_ledger_integrated", "os_memory_enforcement", "os_network_enforcement",
    "namespace_containment", "ptrace_containment", "target_gpu_host_verified",
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


def status_bytes(pid=12345, ppid=23456):
    fields = {
        "Pid": str(pid), "Tgid": str(pid), "PPid": str(ppid), "TracerPid": "0",
        "Threads": "1", "Uid": "34567\t34567\t34567\t34567",
        "Gid": "45678\t45678\t45678\t45678", "NSpid": str(pid), "NStgid": str(pid),
        "CapInh": "0000000000000000", "CapPrm": "0000000000000000",
        "CapEff": "0000000000000000", "CapAmb": "0000000000000000",
        "CapBnd": "0000000000000001", "NoNewPrivs": "1",
    }
    return "".join(f"{key}:\t{value}\n" for key, value in fields.items()).encode()


class PrimitiveTests(unittest.TestCase):
    def setUp(self):
        for name, value in {
            "P_PIDFD": 3, "WEXITED": 4, "WNOHANG": 1, "WNOWAIT": 0x1000000,
            "CLD_EXITED": 1, "CLD_KILLED": 2, "CLD_DUMPED": 3,
        }.items():
            self.enterContext(patch.object(probe.os, name, value, create=True))
        self.uid = self.enterContext(patch.object(probe.os, "getuid", return_value=34567))
        self.enterContext(patch.object(probe.os, "getgid", return_value=45678))
        self.enterContext(patch.object(probe.os, "getpid", return_value=23456))
        self.enterContext(patch.object(probe.os, "getppid", return_value=22222))
        self.clock = self.enterContext(patch.object(probe.time, "monotonic", return_value=100))
        self.pidfd = self.enterContext(patch.object(
            probe.os, "pidfd_open", return_value=71, create=True,
        ))
        self.inheritable = self.enterContext(patch.object(
            probe.os, "get_inheritable", return_value=False,
        ))
        self.waitid = self.enterContext(patch.object(probe.os, "waitid", return_value=None,
                                                    create=True))
        self.close_fd = self.enterContext(patch.object(probe.os, "close"))
        self.open_fd = self.enterContext(patch.object(probe.os, "open", return_value=91))
        self.read_fd = self.enterContext(patch.object(
            probe.os, "read", return_value=status_bytes(),
        ))
        self.send_signal = self.enterContext(patch.object(probe.signal, "pidfd_send_signal",
                                                         create=True))
        self.numeric_signal = self.enterContext(patch.object(probe.os, "kill"))
        self.poller = Mock(poll=Mock(return_value=[]))
        self.enterContext(patch.object(probe.select, "poll", return_value=self.poller))
        self.child = Mock(pid=12345, returncode=None)
        self.owner = probe.OwnedChild(self.child)

    def record(self, code=1, status=0, **changes):
        return Record(12345, 34567, probe.signal.SIGCHLD, status, code)._replace(**changes)

    def assert_code(self, code, action):
        with self.assertRaises(probe.LifecycleError) as caught:
            action()
        self.assertEqual(caught.exception.code, code)

    def test_bind_is_owned_unreaped_and_has_no_popen_poll_wait_or_numeric_signal(self):
        self.owner.bind()
        self.assertTrue(self.owner.bound_live)
        self.assertTrue(self.owner.owned)
        self.pidfd.assert_called_once_with(12345, 0)
        self.assertEqual(self.waitid.call_args_list, [call(3, 71, 0x1000005)] * 2)
        for method in (self.child.poll, self.child.wait, self.child.kill, self.numeric_signal):
            method.assert_not_called()

    def test_already_exited_child_is_owned_but_not_accepted_as_live(self):
        self.waitid.return_value = self.record(status=3)
        self.assert_code("lifecycle_child_exited", self.owner.bind)
        self.assertTrue(self.owner.owned)
        self.assertFalse(self.owner.bound_live)

    def test_nonchild_or_failed_acquisition_cannot_bind(self):
        for error in (ChildProcessError("SECRET"), OSError("SECRET")):
            self.owner = probe.OwnedChild(self.child)
            self.waitid.side_effect = error
            with self.assertRaises(type(error)):
                self.owner.bind()
            self.assertFalse(self.owner.owned)
        self.send_signal.assert_not_called()

    def test_inheritable_pidfd_is_rejected_before_waitid(self):
        self.inheritable.return_value = True
        self.assert_code("lifecycle_identity_rejected", self.owner.bind)
        self.waitid.assert_not_called()

    def test_mismatched_kernel_records_are_rejected(self):
        for changes in ({"si_pid": 1}, {"si_uid": 1}, {"si_signo": 0}, {"si_code": 99}):
            self.owner = probe.OwnedChild(self.child)
            self.waitid.return_value = self.record(**changes)
            self.assert_code("lifecycle_identity_rejected", self.owner.bind)
            self.assertFalse(self.owner.owned)

    def test_exit_between_waitid_and_poll_rejects_binding(self):
        self.poller.poll.return_value = [(71, probe.select.POLLIN)]
        self.assert_code("lifecycle_child_exited", self.owner.bind)
        self.assertFalse(self.owner.bound_live)

    def test_unexpected_poll_events_fail_closed(self):
        for events in ([(71, probe.select.POLLHUP)], [(72, probe.select.POLLIN)],
                       [(71, probe.select.POLLERR)], [(71, 1), (71, 1)]):
            self.poller.poll.return_value = events
            self.assert_code("lifecycle_identity_rejected", lambda: self.owner.poll(0))

    def test_success_requires_observed_and_consumed_kernel_exit_records(self):
        self.owner.bind()
        self.waitid.reset_mock()
        self.waitid.return_value = self.record()
        self.poller.poll.return_value = [(71, probe.select.POLLIN)]
        self.owner.reap(102, require_success=True)
        self.assertEqual(self.waitid.call_args_list, [call(3, 71, 0x1000005), call(3, 71, 5)])
        self.assertTrue(self.owner.reaped)
        self.assertEqual(self.child.returncode, 0)
        self.child.wait.assert_not_called()

    def test_nonzero_exit_is_reaped_but_cannot_pass(self):
        self.owner.bind()
        self.waitid.return_value = self.record(status=3)
        self.poller.poll.return_value = [(71, probe.select.POLLIN)]
        self.assert_code(
            "lifecycle_exit_rejected", lambda: self.owner.reap(102, require_success=True),
        )
        self.assertTrue(self.owner.reaped)
        self.assertEqual(self.child.returncode, 3)

    def test_echild_or_changed_exit_record_never_proves_success(self):
        self.owner.bind()
        self.poller.poll.return_value = [(71, probe.select.POLLIN)]
        self.waitid.side_effect = [self.record(), ChildProcessError("SECRET")]
        with self.assertRaises(ChildProcessError):
            self.owner.reap(102, require_success=True)
        self.assertFalse(self.owner.reaped)
        self.waitid.side_effect = [self.record(), self.record(status=1)]
        self.assert_code(
            "lifecycle_exit_rejected", lambda: self.owner.reap(102, require_success=True),
        )
        self.assertFalse(self.owner.reaped)

    def test_cleanup_signals_only_pidfd_once_and_reaps_real_record(self):
        self.owner.bind()
        self.waitid.side_effect = [None, self.record(2, 9), self.record(2, 9)]
        self.poller.poll.return_value = [(71, probe.select.POLLIN)]
        self.owner.terminate_and_reap(102)
        self.owner.terminate_and_reap(102)
        self.send_signal.assert_called_once_with(71, probe.signal.SIGKILL, None, 0)
        self.numeric_signal.assert_not_called()
        self.assertEqual(self.child.returncode, -9)

    def test_pidfd_signal_failure_is_not_retried(self):
        self.owner.bind()
        self.send_signal.side_effect = PermissionError("SECRET")
        with self.assertRaises(PermissionError):
            self.owner.terminate_and_reap(102)
        self.assert_code("lifecycle_cleanup_failed", lambda: self.owner.terminate_and_reap(102))
        self.send_signal.assert_called_once()

    def test_exit_signal_race_still_requires_real_reap(self):
        self.owner.bind()
        self.send_signal.side_effect = ProcessLookupError()
        self.waitid.side_effect = [None, self.record(status=2), self.record(status=2)]
        self.poller.poll.return_value = [(71, probe.select.POLLIN)]
        self.owner.terminate_and_reap(102)
        self.assertTrue(self.owner.reaped)

    def test_pidfd_close_failure_never_retries_descriptor(self):
        self.owner.bind()
        self.close_fd.side_effect = OSError("SECRET")
        with self.assertRaises(OSError):
            self.owner.close()
        self.assert_code("lifecycle_cleanup_failed", self.owner.close)
        self.close_fd.assert_called_once_with(71)
        self.assertFalse(self.owner.closed)

    def test_status_capacity_prepaid_with_exact_three_attempt_cap(self):
        reads = probe.StatusReads()
        for _ in range(3):
            reads.read("/SYNTHETIC/status")
        self.assert_code("lifecycle_budget_exhausted", lambda: reads.read("/SYNTHETIC/status"))
        self.assertEqual(reads.capacity_bytes, 3 * probe.STATUS_CAPACITY)
        self.assertEqual(self.read_fd.call_count, 3)
        self.assertEqual(reads.closed, 3)
        self.open_fd.assert_called_with(
            "/SYNTHETIC/status", os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )

    def test_status_open_failure_is_charged_without_close(self):
        self.open_fd.side_effect = OSError("SECRET")
        reads = probe.StatusReads()
        with self.assertRaises(OSError):
            reads.read("/SYNTHETIC/status")
        self.assertEqual(reads.capacity_bytes, probe.STATUS_CAPACITY)
        self.close_fd.assert_not_called()

    def test_status_read_failure_closes_once_and_does_not_retain_exception_text(self):
        self.read_fd.side_effect = OSError("SECRET")
        reads = probe.StatusReads()
        self.assert_code("lifecycle_failed", lambda: reads.read("/SYNTHETIC/status"))
        self.close_fd.assert_called_once_with(91)
        self.assertEqual(reads.closed, 1)
        self.assertFalse(reads.cleanup_failed)

    def test_status_read_and_close_failures_keep_cleanup_failure_sticky(self):
        for read_error in (None, OSError("SECRET")):
            reads = probe.StatusReads()
            self.read_fd.side_effect = read_error
            self.close_fd.side_effect = OSError("SECRET")
            self.assert_code(
                "lifecycle_cleanup_failed", lambda reads=reads: reads.read("/SYNTHETIC/status"),
            )
            self.assertTrue(reads.cleanup_failed)
            self.assertEqual(reads.closed, 0)
        self.assertEqual(self.close_fd.call_count, 2)

    def test_status_child_snapshot_bracketed_by_live_pidfd(self):
        owner = Mock(child=self.child, assert_live=Mock())
        events = []
        owner.assert_live.side_effect = lambda: events.append("live")
        self.read_fd.side_effect = lambda *args: events.append("read") or status_bytes()
        probe.StatusReads().child(owner, 102)
        self.assertEqual(events, ["live", "read", "live"])
        self.open_fd.assert_called_once_with(
            "/proc/12345/status", os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )

    def test_child_exit_during_snapshot_cannot_reach_parser(self):
        owner = Mock(child=self.child, assert_live=Mock())
        owner.assert_live.side_effect = [None, probe.LifecycleError("lifecycle_child_exited")]
        with patch.object(probe.policy, "parse_status") as parse:
            self.assert_code(
                "lifecycle_child_exited", lambda: probe.StatusReads().child(owner, 102),
            )
            parse.assert_not_called()
        self.close_fd.assert_called_once_with(91)

    def test_cleanup_failure_still_closes_remaining_handles_and_pidfd(self):
        self.owner.bind()
        first, second = Mock(), Mock()
        first.close.side_effect = OSError("SECRET")
        self.send_signal.side_effect = PermissionError("SECRET")
        self.assertFalse(probe.cleanup(self.owner, first, [second]))
        second.close.assert_called_once()
        self.close_fd.assert_called_once_with(71)
        self.assertFalse(self.owner.reaped)

    def test_failed_binding_has_no_numeric_signal_or_verified_popen_reap(self):
        self.child.wait.return_value = 0
        self.assertFalse(probe.cleanup(self.owner, None, []))
        self.child.wait.assert_called_once_with(timeout=probe.CLEANUP_SECONDS)
        self.send_signal.assert_not_called()
        self.numeric_signal.assert_not_called()
        self.assertFalse(self.owner.reaped)

    def test_nonfinite_or_expired_deadline_denies_before_status_read(self):
        for deadline in (99, 100, float("nan"), float("inf")):
            owner = Mock(child=self.child, assert_live=Mock())
            self.assert_code(
                "lifecycle_deadline_exhausted",
                lambda owner=owner, deadline=deadline: probe.StatusReads().child(owner, deadline),
            )
            owner.assert_live.assert_not_called()
        self.open_fd.assert_not_called()

    def test_reap_timeout_never_consumes_exit_or_claims_success(self):
        self.owner.bind()
        self.waitid.reset_mock()
        self.assert_code(
            "lifecycle_deadline_exhausted", lambda: self.owner.reap(102, require_success=True),
        )
        self.waitid.assert_not_called()
        self.assertFalse(self.owner.reaped)


class ProbeTests(unittest.TestCase):
    def boundaries(self, value):
        for key in NONCLAIMS:
            self.assertIs(value[key], False, key)
        for key in ("training_source_or_data_reads", "model_or_checkpoint_reads",
                    "private_artifact_writes", "external_network_bytes", "gpu_use",
                    "inference_calls"):
            self.assertEqual(value[key], 0)
        self.assertLessEqual(value["status_read_attempts"], 3)
        self.assertLessEqual(value["status_read_capacity_bytes"], 3 * probe.STATUS_CAPACITY)
        for forbidden in ("SECRET", str(ROOT), '"pid"', '"uid"', '"gid"'):
            self.assertNotIn(forbidden, json.dumps(value))

    @isolated
    def test_fixed_source_is_hash_bound_bounded_and_executed_from_bytes(self):
        source = probe.authenticated_source()
        with patch.object(probe.subprocess, "Popen") as spawn:
            probe.spawn(Mock(fileno=Mock(return_value=17)), "none", source)
            self.assertEqual(spawn.call_args.args[0], [
                sys.executable, "-S", "-B", "-c", source.decode(), "17", "none",
            ])
            self.assertEqual(spawn.call_args.kwargs["pass_fds"], (17,))
            self.assertIs(spawn.call_args.kwargs["close_fds"], True)
            self.assertEqual(spawn.call_args.kwargs["env"], {
                "PATH": "/usr/bin:/bin", "LC_ALL": "C", "PYTHONHASHSEED": "0",
            })
        for raw in (b"SECRET", b"x" * 8193):
            with patch.object(Path, "open", return_value=io.BytesIO(raw)):
                with self.assertRaises(probe.LifecycleError):
                    probe.authenticated_source()
        with patch.object(probe.subprocess, "Popen") as spawn:
            for fault, raw in (("none", b"SECRET"), ("unknown", source)):
                with self.assertRaises(probe.LifecycleError):
                    probe.spawn(Mock(), fault, raw)
            spawn.assert_not_called()

    @isolated
    def test_signal_normalization_requires_default_before_and_after(self):
        with (
            patch.object(probe.signal, "getsignal", return_value=probe.signal.SIG_DFL),
            patch.object(probe.signal, "signal", return_value=probe.signal.SIG_DFL) as setter,
        ):
            probe.normalize_child_signal()
            setter.assert_called_once_with(probe.signal.SIGCHLD, probe.signal.SIG_DFL)
        for before, returned, after in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            with (
                patch.object(probe.signal, "getsignal", side_effect=[before, after]),
                patch.object(probe.signal, "signal", return_value=returned) as setter,
            ):
                with self.assertRaises(probe.LifecycleError):
                    probe.normalize_child_signal()
                if before:
                    setter.assert_not_called()

    @isolated
    def test_unsupported_runtime_denies_before_procfs_source_sockets_or_child(self):
        with (
            patch.object(probe.credentials, "require_linux", side_effect=
                         probe.credentials.PeerCredentialError("linux_credentials_unavailable")),
            patch.object(probe.StatusReads, "controller") as controller,
            patch.object(probe, "authenticated_source") as source,
            patch.object(probe.credentials, "credential_pair") as pair,
            patch.object(probe, "spawn") as spawn,
        ):
            value = probe.run_lifecycle_probe()
            self.assertEqual(value["state"], "unsupported")
            for action in (controller, source, pair, spawn):
                action.assert_not_called()
            self.boundaries(value)

    @isolated
    def test_unknown_fault_denies_before_runtime_and_reads(self):
        for fault in (None, True, [], "unknown"):
            with patch.object(probe, "require_runtime") as runtime:
                value = probe.run_lifecycle_probe(fault=fault)
                self.assertEqual(value["failure_code"], "lifecycle_configuration_rejected")
                runtime.assert_not_called()

    @isolated
    def test_ml_runtime_denies_before_linux_probe(self):
        with (
            patch.dict(sys.modules, {"torch": types.ModuleType("torch")}),
            patch.object(probe.credentials, "require_linux") as linux,
            patch.object(probe.StatusReads, "controller") as controller,
        ):
            value = probe.run_lifecycle_probe()
            self.assertEqual(value["failure_code"], "probe_runtime_rejected")
            linux.assert_not_called()
            controller.assert_not_called()

    @isolated
    def test_unsupported_cpython_version_denies_before_controller_read(self):
        with (
            patch.object(probe.credentials, "require_linux"),
            patch.object(probe.sys, "version_info", (3, 14)),
            patch.object(probe.StatusReads, "controller") as controller,
        ):
            value = probe.run_lifecycle_probe()
            self.assertEqual(value["failure_code"], "lifecycle_unavailable")
            controller.assert_not_called()

    @isolated
    def test_native_sigaction_support_required_before_controller_read(self):
        with (
            patch.object(probe.credentials, "require_linux"),
            patch.object(probe.sysconfig, "get_config_var", return_value=0),
            patch.object(probe.StatusReads, "controller") as controller,
        ):
            value = probe.run_lifecycle_probe()
            self.assertEqual(value["failure_code"], "lifecycle_unavailable")
            controller.assert_not_called()

    def fake_run(self, *, endpoint_close=False, snapshot_error=False, cleanup_error=False,
                 receive_error=None, early_gate=None, late=False):
        parent, endpoint = Mock(), Mock()
        parent.send.return_value = 1
        if endpoint_close:
            endpoint.close.side_effect = OSError("SECRET")
        owner = Mock(
            child=Mock(pid=12345), bound_live=True, owned=True, reaped=False, closed=False,
            signal_attempts=0, waitid_calls=5, assert_live=Mock(),
        )
        owner.reap.side_effect = lambda *args, **kwargs: setattr(owner, "reaped", True)
        owner.terminate_and_reap.side_effect = lambda *args: setattr(owner, "reaped", True)
        owner.close.side_effect = lambda: setattr(owner, "closed", True)
        if cleanup_error:
            owner.close.side_effect = OSError("SECRET")
        reads = Mock(attempts=3, capacity_bytes=3 * probe.STATUS_CAPACITY, closed=3,
                     cleanup_failed=False)
        reads.child.return_value = {"capability_bounding_set_empty": False}
        if snapshot_error:
            reads.child.side_effect = probe.policy.StatusError("child_status_capability_rejected")
        channel = Mock()
        channel.receive.side_effect = receive_error or [
            len(value).to_bytes(4, "big") + value for value in (probe.READY, probe.PAYLOAD)
        ]
        channel.summary.return_value = {"receive_attempts": 2, "verified_messages": 2}
        with (
            patch.object(probe, "require_runtime"),
            patch.object(probe, "normalize_child_signal", side_effect=early_gate) as normalize,
            patch.object(probe, "authenticated_source", return_value=b"synthetic") as source,
            patch.object(probe, "spawn", return_value=owner.child) as spawn,
            patch.object(probe, "OwnedChild", return_value=owner),
            patch.object(probe, "StatusReads", return_value=reads),
            patch.object(probe.credentials, "credential_pair", return_value=(parent, endpoint)),
            patch.object(probe.credentials, "CredentialChannel", return_value=channel) as construct,
            patch.object(probe.time, "monotonic", return_value=100) as clock,
        ):
            if late:
                owner.close.side_effect = lambda: setattr(clock, "return_value", 107)
            value = probe.run_lifecycle_probe()
            if early_gate:
                normalize.assert_called_once()
                source.assert_not_called()
                spawn.assert_not_called()
            else:
                spawn.assert_called_once()
                owner.bind.assert_called_once()
                endpoint.close.assert_called_once()
                if endpoint_close:
                    construct.assert_not_called()
                    parent.close.assert_called_once()
                else:
                    parent.close.assert_not_called()
                    channel.close.assert_called_once()
            self.boundaries(value)
        return value

    @isolated
    def test_mocked_success_requires_two_snapshots_reap_and_cleanup(self):
        value = self.fake_run()
        self.assertEqual(value["state"], "passed")
        self.assertTrue(value["lifecycle_probe_passed"])
        self.assertEqual(value["capability_snapshots_verified"], 2)
        self.assertEqual(value["worker_processes_reaped"], 1)
        self.assertTrue(value["pidfd_closed"])

    @isolated
    def test_no_pass_after_early_gate_snapshot_receive_cleanup_or_deadline_failure(self):
        for options, code in (
            ({"early_gate": OSError("SECRET")}, "lifecycle_failed"),
            ({"snapshot_error": True}, "child_status_capability_rejected"),
            ({"endpoint_close": True}, "lifecycle_cleanup_failed"),
            ({"cleanup_error": True}, "lifecycle_cleanup_failed"),
            ({"late": True}, "lifecycle_deadline_exhausted"),
            ({"receive_error": probe.credentials.PeerCredentialError(
                "credential_cleanup_failed")}, "lifecycle_cleanup_failed"),
        ):
            value = self.fake_run(**options)
            self.assertEqual(value["failure_code"], code)
            self.assertFalse(value["lifecycle_probe_passed"])
            if code == "lifecycle_cleanup_failed":
                self.assertFalse(value["worker_cleanup_complete"])

    def live(self, fault):
        value = probe.run_lifecycle_probe(fault=fault)
        self.boundaries(value)
        if sys.platform != "linux":
            self.assertEqual(value["state"], "unsupported")
            self.assertEqual(value["status_read_attempts"], 0)
            self.assertEqual(value["worker_processes_started"], 0)
            return None
        self.assertTrue(value["worker_cleanup_complete"], value["failure_code"])
        self.assertEqual(value["worker_processes_started"], 1, value["failure_code"])
        self.assertEqual(value["worker_processes_reaped"], 1, value["failure_code"])
        self.assertTrue(value["pidfd_closed"])
        return value

    @isolated
    def test_live_linux_success_or_nonlinux_denial(self):
        value = self.live("none")
        if value is not None:
            self.assertEqual(value["state"], "passed", value["failure_code"])
            self.assertTrue(value["process_generation_bound"])
            self.assertEqual(value["capability_snapshots_verified"], 2)
            self.assertEqual(value["status_read_attempts"], 3)
            self.assertEqual(value["credential_accounting"]["verified_messages"], 2)
            self.assertEqual(value["pidfd_signal_attempts"], 0)

    @isolated
    def test_live_linux_wrong_parent_policy_rejection_or_nonlinux_denial(self):
        value = self.live("wrong_parent")
        if value is not None:
            self.assertEqual(value["failure_code"], "child_status_identity_rejected")
            self.assertFalse(value["lifecycle_probe_passed"])
            self.assertEqual(value["capability_snapshots_verified"], 0)

    @isolated
    def test_live_linux_early_exit_rejection_or_nonlinux_denial(self):
        value = self.live("exit_after_ready")
        if value is not None:
            self.assertNotEqual(value["state"], "passed")
            self.assertIsNotNone(value["failure_code"])
            self.assertFalse(value["lifecycle_probe_passed"])

    @isolated
    def test_live_linux_stall_cleanup_or_nonlinux_denial(self):
        value = self.live("stall")
        if value is not None:
            self.assertEqual(value["failure_code"], "lifecycle_deadline_exhausted")
            self.assertFalse(value["lifecycle_probe_passed"])
            self.assertEqual(value["capability_snapshots_verified"], 1)


if __name__ == "__main__":
    unittest.main()
