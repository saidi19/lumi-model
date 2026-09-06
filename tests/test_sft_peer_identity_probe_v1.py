"""Synthetic lifecycle guards plus real inert Linux cases (non-Linux denies)."""

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
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_test_sft_peer_identity_probe_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
probe = importlib.import_module(PACKAGE + ".posttraining_sft_peer_identity_probe_v1")
credentials = probe.credentials
NONCLAIMS = (
    "process_generation_bound", "privileged_sender_capabilities_excluded",
    "durable_ledger_integrated", "os_memory_enforcement", "os_network_enforcement",
    "target_gpu_host_verified", "guarded_real_worker_ready", "execution_authorized",
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
            stdin=subprocess.DEVNULL, capture_output=True, timeout=12,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )
        self.assertEqual(result.returncode, 0, "isolated inert identity case failed")
    return run


class PeerIdentityProbeTests(unittest.TestCase):
    def assert_boundaries(self, report):
        for name in NONCLAIMS:
            self.assertIs(report[name], False, name)
        for name in (
            "training_source_or_data_reads", "model_or_checkpoint_reads",
            "private_artifact_writes", "external_network_bytes", "gpu_use", "inference_calls",
        ):
            self.assertEqual(report[name], 0, name)
        text = json.dumps(report)
        self.assertNotIn("SYNTHETIC_SECRET", text)
        self.assertNotIn(str(ROOT), text)
        self.assertNotIn('"pid"', text)
        self.assertNotIn('"uid"', text)
        self.assertNotIn('"gid"', text)

    @isolated
    def test_worker_control_is_hash_bound_and_capped(self):
        source = probe.authenticate_worker()
        self.assertLessEqual(len(source), 8192)
        for raw in (b"SYNTHETIC_SECRET", b"x" * 8193):
            with (
                self.subTest(size=len(raw)),
                patch.object(Path, "open", return_value=io.BytesIO(raw)),
            ):
                with self.assertRaisesRegex(probe.ProbeError, "^probe_source_rejected$"):
                    probe.authenticate_worker()

    @isolated
    def test_spawn_uses_exact_authenticated_bytes_and_one_explicit_descriptor(self):
        source = probe.authenticate_worker()
        endpoint = Mock(fileno=Mock(return_value=17))
        with patch.object(probe.subprocess, "Popen") as spawn:
            probe.spawn_inert_worker(endpoint, "none", source)
            spawn.assert_called_once()
            arguments = spawn.call_args.args[0]
            self.assertEqual(arguments, [
                sys.executable, "-S", "-B", "-c", source.decode("utf-8"), "17", "none",
            ])
            options = spawn.call_args.kwargs
            self.assertEqual(options["pass_fds"], (17,))
            self.assertIs(options["close_fds"], True)
            self.assertEqual(options["env"], {
                "PATH": "/usr/bin:/bin", "LC_ALL": "C", "PYTHONHASHSEED": "0",
            })
            self.assertEqual(options["stdout"], subprocess.DEVNULL)
            self.assertEqual(options["stderr"], subprocess.DEVNULL)
        for fault, raw in (
            ("wrong", source), ("none", b"SYNTHETIC_SECRET"), ("none", source + b"x"),
        ):
            with self.subTest(fault=fault), patch.object(probe.subprocess, "Popen") as spawn:
                with self.assertRaises(probe.ProbeError):
                    probe.spawn_inert_worker(endpoint, fault, raw)
                spawn.assert_not_called()

    @isolated
    def test_runtime_rejects_ml_or_optimization_before_linux_and_source_checks(self):
        for name in ("torch", "transformers", "numpy", "tokenizers", "safetensors"):
            with (
                self.subTest(name=name), patch.dict(sys.modules, {name: types.ModuleType(name)}),
                patch.object(credentials, "require_linux") as linux,
                patch.object(probe, "authenticate_worker") as authenticate,
            ):
                value = probe.run_inert_probe()
                self.assertEqual(value["failure_code"], "probe_runtime_rejected")
                linux.assert_not_called()
                authenticate.assert_not_called()
                self.assert_boundaries(value)
        with patch.object(sys, "flags", types.SimpleNamespace(no_site=1, optimize=1)):
            self.assertEqual(probe.run_inert_probe()["failure_code"], "probe_runtime_rejected")

    @isolated
    def test_unsupported_kernel_denies_before_control_reads_sockets_or_children(self):
        with (
            patch.object(credentials, "require_linux",
                         side_effect=credentials.PeerCredentialError("linux_credentials_unavailable")),
            patch.object(probe, "authenticate_worker") as authenticate,
            patch.object(credentials, "credential_pair") as pair,
            patch.object(probe, "spawn_inert_worker") as spawn,
        ):
            value = probe.run_inert_probe()
            self.assertEqual(value["state"], "unsupported")
            self.assertEqual(value["worker_processes_started"], 0)
            self.assertFalse(value["linux_peer_credential_probe_passed"])
            authenticate.assert_not_called()
            pair.assert_not_called()
            spawn.assert_not_called()
            self.assert_boundaries(value)

    def mocked_run(
        self, *, receive_error=None, close_error=False, late=False, spawn_error=False,
        endpoint_close_error=False,
    ):
        source = probe.authenticate_worker()
        parent, endpoint = Mock(), Mock()
        parent.send.return_value = 1
        if endpoint_close_error:
            endpoint.close.side_effect = OSError("SYNTHETIC_SECRET")
        child = Mock(pid=1234, poll=Mock(return_value=0), wait=Mock(return_value=0))
        summary = {
            "verified_messages": 1, "receive_attempts": 1, "payload_bytes": len(probe.PAYLOAD),
            "read_capacity_bytes": credentials.RECEIVE_CAPACITY_BYTES,
            "received_descriptors_closed": 0, "rejected": False, "closed": True,
        }
        channel = Mock(receive=Mock(return_value=probe.FRAME), summary=Mock(return_value=summary))
        channel.receive.side_effect = receive_error
        if close_error:
            channel.close.side_effect = OSError("SYNTHETIC_SECRET")
        with (
            patch.object(credentials, "require_linux"),
            patch.object(probe, "authenticate_worker", return_value=source) as authenticate,
            patch.object(credentials, "credential_pair", return_value=(parent, endpoint)) as pair,
            patch.object(probe, "spawn_inert_worker", return_value=child,
                         side_effect=OSError("SYNTHETIC_SECRET") if spawn_error else None) as spawn,
            patch.object(credentials, "CredentialChannel", return_value=channel) as construct,
            patch.object(probe.time, "monotonic", return_value=10) as clock,
        ):
            if late:
                channel.close.side_effect = lambda: setattr(clock, "return_value", 20)
            value = probe.run_inert_probe()
            authenticate.assert_called_once_with()
            pair.assert_called_once_with()
            spawn.assert_called_once_with(endpoint, "none", source)
            if not spawn_error and not endpoint_close_error:
                construct.assert_called_once_with(
                    parent, expected_credentials=(1234, os.getuid(), os.getgid()), deadline=15,
                    maximum_messages=1, maximum_read_bytes=credentials.RECEIVE_CAPACITY_BYTES,
                )
                channel.receive.assert_called_once_with()
                channel.close.assert_called_once_with()
                parent.close.assert_not_called()  # Transferred ownership is not closed twice.
            else:
                construct.assert_not_called()
                parent.close.assert_called_once_with()
            endpoint.close.assert_called_once_with()
            self.assert_boundaries(value)
        return value

    @isolated
    def test_mocked_success_requires_cleanup_and_makes_no_worker_admission_claim(self):
        value = self.mocked_run()
        self.assertEqual(value["state"], "passed")
        self.assertIsNone(value["failure_code"])
        self.assertTrue(value["worker_cleanup_complete"])
        self.assertEqual(value["worker_processes_started"], value["worker_processes_reaped"])

    @isolated
    def test_close_failure_late_deadline_and_spawn_failure_never_publish_pass(self):
        for options, code in (
            ({"close_error": True}, "probe_cleanup_failed"),
            ({"late": True}, "probe_deadline_exhausted"),
            ({"spawn_error": True}, "probe_failed"),
        ):
            with self.subTest(code=code):
                value = self.mocked_run(**options)
                self.assertEqual(value["state"], "failed")
                self.assertEqual(value["failure_code"], code)
                self.assertFalse(value["linux_peer_credential_probe_passed"])

    @isolated
    def test_earlier_close_failures_cannot_be_erased_by_later_successful_cleanup(self):
        for options in (
            {"endpoint_close_error": True},
            {"receive_error": credentials.PeerCredentialError("credential_cleanup_failed")},
        ):
            with self.subTest(options=tuple(options)):
                value = self.mocked_run(**options)
                self.assertEqual(value["failure_code"], "probe_cleanup_failed")
                self.assertEqual(value["state"], "failed")
                self.assertFalse(value["worker_cleanup_complete"])
                self.assertFalse(value["linux_peer_credential_probe_passed"])
                self.assertEqual(value["worker_processes_reaped"], 1)

    @isolated
    def test_mocked_rejection_and_opaque_syscall_failure_stop_after_one_receive(self):
        for error, state, code in (
            (credentials.PeerCredentialError("credential_mismatch"),
             "rejected", "credential_mismatch"),
            (OSError("SYNTHETIC_SECRET"), "failed", "probe_failed"),
        ):
            with self.subTest(code=code):
                value = self.mocked_run(receive_error=error)
                self.assertEqual((value["state"], value["failure_code"]), (state, code))
                self.assertFalse(value["linux_peer_credential_probe_passed"])

    @isolated
    def test_cleanup_attempts_each_owned_handle_even_after_other_cleanup_errors(self):
        child = Mock(
            poll=Mock(side_effect=OSError("secret")), wait=Mock(side_effect=OSError("secret")),
        )
        channel, first, second = Mock(), Mock(), Mock()
        channel.close.side_effect = OSError("secret")
        first.close.side_effect = OSError("secret")
        self.assertEqual(probe.cleanup(child, channel, [first, second]), (False, 0))
        child.wait.assert_called_once_with(timeout=1)
        for handle in (channel, first, second):
            handle.close.assert_called_once_with()

    @isolated
    def test_unknown_fault_denies_before_source_read_or_process_launch(self):
        for fault in (None, True, "restart", "SYNTHETIC_SECRET"):
            with (
                self.subTest(fault=fault),
                patch.object(probe, "authenticate_worker") as authenticate,
            ):
                value = probe.run_inert_probe(fault=fault)
                self.assertEqual(value["failure_code"], "probe_configuration_rejected")
                self.assertEqual(value["worker_processes_started"], 0)
                authenticate.assert_not_called()

    def live(self, fault):
        value = probe.run_inert_probe(fault=fault)
        self.assert_boundaries(value)
        if sys.platform != "linux":
            self.assertEqual(value["state"], "unsupported")
            self.assertFalse(value["worker_source_authenticated"])
            self.assertFalse(value["linux_peer_credential_probe_passed"])
            self.assertEqual(value["worker_processes_started"], 0)
            return None
        self.assertTrue(value["worker_source_authenticated"])
        self.assertTrue(value["worker_cleanup_complete"], value["failure_code"])
        self.assertEqual(value["worker_processes_started"], 1)
        self.assertEqual(value["worker_processes_reaped"], 1)
        counts = value["credential_accounting"]
        self.assertEqual(counts["receive_attempts"], 1)
        self.assertEqual(counts["read_capacity_bytes"], credentials.RECEIVE_CAPACITY_BYTES)
        self.assertTrue(counts["closed"])
        return value

    @isolated
    def test_live_linux_sender_credentials_or_nonlinux_denial(self):
        value = self.live("none")
        if value is not None:
            self.assertEqual(value["state"], "passed", value["failure_code"])
            self.assertTrue(value["kernel_pid_tuple_matches_expected_child"])
            self.assertEqual(value["credential_accounting"]["verified_messages"], 1)
            self.assertEqual(value["credential_accounting"]["payload_bytes"], len(probe.PAYLOAD))

    @isolated
    def test_live_linux_child_is_not_socketpair_creator_or_nonlinux_denial(self):
        value = self.live("unexpected_sender")
        if value is not None:
            self.assertEqual((value["state"], value["failure_code"]),
                             ("rejected", "credential_mismatch"))
            self.assertEqual(value["credential_accounting"]["verified_messages"], 0)

    @isolated
    def test_live_linux_received_rights_are_closed_and_rejected_or_nonlinux_denial(self):
        value = self.live("rights")
        if value is not None:
            self.assertEqual((value["state"], value["failure_code"]),
                             ("rejected", "credential_ancillary_rejected"))
            self.assertEqual(value["credential_accounting"]["received_descriptors_closed"], 1)
            self.assertEqual(value["credential_accounting"]["verified_messages"], 0)

    @isolated
    def test_live_linux_oversized_packet_rejected_or_nonlinux_denial(self):
        value = self.live("oversized")
        if value is not None:
            self.assertEqual((value["state"], value["failure_code"]),
                             ("rejected", "credential_frame_rejected"))
            self.assertEqual(value["credential_accounting"]["verified_messages"], 0)

    @isolated
    def test_live_linux_stalled_worker_deadline_and_reap_or_nonlinux_denial(self):
        value = self.live("stall")
        if value is not None:
            self.assertEqual((value["state"], value["failure_code"]),
                             ("rejected", "credential_deadline_exhausted"))
            self.assertEqual(value["credential_accounting"]["verified_messages"], 0)


if __name__ == "__main__":
    unittest.main()
