"""Admission boundary and genuine synthetic CPU SFT integration; no private data."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts/run_sft_offline_integration_v1.py"
MODULE = REPOSITORY / "src/lumi/posttraining_sft_offline_cpu_v1.py"
EXECUTABLE = Path(sys.executable)


def load_module(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


integration = load_module("synthetic_offline_integration_tests", MODULE)
admission = load_module(
    "synthetic_offline_admission_tests",
    REPOSITORY / "src/lumi/posttraining_sft_cpu_admission_v1.py",
)


class AdmissionBoundaryTests(unittest.TestCase):
    def test_public_keyboard_interrupt_is_fixed_text(self):
        script = load_module("synthetic_offline_public_script", SCRIPT)
        gate = Mock()
        gate.fixed_request.return_value = {}
        gate.canonical.return_value = b"{}"
        gate.digest.return_value = "a" * 64
        runtime = Mock()
        runtime.run_integration.side_effect = KeyboardInterrupt()
        with (
            patch("builtins.print") as output,
            patch.object(script.importlib, "import_module", side_effect=[gate, runtime]),
        ):
            self.assertEqual(script.main(["--run-synthetic-cpu"]), 1)
        runtime.run_integration.assert_called_once()
        output.assert_called_once_with('{"status":"FAIL_synthetic_cpu_integration"}', flush=True)

    def test_import_is_stdlib_only_in_fresh_interpreter(self):
        command = (
            "import runpy,sys; runpy.run_path(sys.argv[1]); "
            "assert not any(k.split('.')[0] in "
            "{'torch','transformers','numpy','safetensors','tokenizers','lumi'} "
            "for k in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-S", "-B", "-c", command, str(MODULE)],
            capture_output=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_public_cli_rejects_all_other_forms(self):
        for argv in (
            [],
            ["--execute"],
            ["--device", "cuda"],
            ["--run-synthetic-cpu", "--device", "cpu"],
            ["--run-synthetic-cpu", "/SYNTHETIC-NOT-A-DATA-PATH"],
            ["--help"],
            ["--run-synthetic-cpu", "--run-synthetic-cpu"],
        ):
            with self.subTest(argv=argv):
                result = subprocess.run(
                    [sys.executable, "-S", "-B", str(SCRIPT), *argv],
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(result.returncode, 1)
                self.assertEqual(
                    json.loads(result.stdout), {"status": "FAIL_synthetic_cpu_integration"}
                )
                self.assertEqual(result.stderr, b"")

    def test_bad_runtime_flags_reject_before_dispatch(self):
        result = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "--run-synthetic-cpu"],
            capture_output=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stderr, b"")

    def test_invalid_request_rejected_before_temp_or_spawn(self):
        changed = {**admission.fixed_request(), "device": "cuda"}
        raw = admission.canonical(changed)
        fake_admission = Mock()
        fake_admission.pre_ml_runtime_check.return_value = None
        fake_admission.validate_request.side_effect = admission.AdmissionError()
        with (
            patch.object(integration, "_admission", return_value=fake_admission),
            patch.object(integration.tempfile, "TemporaryDirectory") as temporary,
            patch.object(integration, "_spawn") as spawn,
        ):
            with self.assertRaises(admission.AdmissionError):
                integration.run_integration(
                    raw, hashlib.sha256(raw).hexdigest(), repository_root=REPOSITORY
                )
            temporary.assert_not_called()
            spawn.assert_not_called()

    def test_actual_pre_ml_guard_precedes_request_and_runtime_discovery(self):
        fake_admission = Mock()
        fake_admission.pre_ml_runtime_check.side_effect = admission.AdmissionError()
        with (
            patch.object(integration, "_admission", return_value=fake_admission),
            patch.object(integration, "_repository") as repository,
        ):
            with self.assertRaises(admission.AdmissionError):
                integration.run_integration(b"{}", "a" * 64, repository_root=REPOSITORY)
            fake_admission.validate_request.assert_not_called()
            repository.assert_not_called()

    def test_sigchld_auto_reaping_or_custom_handler_denied_before_spawn(self):
        for disposition in (signal.SIG_IGN, lambda number, frame: None):
            with (
                patch.object(integration, "_admit", return_value=admission.fixed_request()),
                patch.object(integration.signal, "getsignal", return_value=disposition),
                patch.object(integration.tempfile, "TemporaryDirectory") as temporary,
                patch.object(integration, "_spawn") as spawn,
            ):
                with self.assertRaises(integration.IntegrationError):
                    integration.run_integration(b"{}", "a" * 64, repository_root=REPOSITORY)
                temporary.assert_not_called()
                spawn.assert_not_called()

    def test_worker_admission_precedes_root_path_inspection(self):
        with (
            patch.object(integration, "_admit", side_effect=admission.AdmissionError()),
            patch.object(integration, "Path") as paths,
        ):
            with self.assertRaises(admission.AdmissionError):
                integration._validated_control({"request": {}, "request_sha256": "a" * 64})
            paths.assert_not_called()

    def test_child_rejects_before_ml(self):
        reader, writer = os.pipe()
        os.write(writer, b"{}")
        os.close(writer)
        with (
            patch.object(integration, "_validated_control", side_effect=admission.AdmissionError()),
            patch.object(integration, "_initialize_ml") as ml,
            patch.object(integration.os, "dup", return_value=writer),
            patch.object(integration.os, "dup2"),
            patch.object(integration.os, "open", return_value=writer),
            patch.object(integration.os, "close"),
        ):
            with self.assertRaises(SystemExit):
                integration._child_main(reader)
            ml.assert_not_called()
        os.close(reader)

    def test_only_fixed_local_synthetic_training_arguments(self):
        bindings = {
            key: "a" * 64
            for key in (
                "source",
                "inventory",
                "manifest_file",
                "manifest_internal",
                "metadata",
                "seal",
            )
        }
        for stage, expected in enumerate(integration.SCHEDULE):
            arguments = integration._training_arguments(Path("/SYNTHETIC"), bindings, stage)
            self.assertEqual(arguments["strategy"], "ddp")
            self.assertEqual(arguments["precision"], "fp32")
            self.assertIsNone(arguments["authorization_path"])
            self.assertEqual(arguments["encoded_shards_record_limit"], 4)
            self.assertEqual(arguments["target_steps"], 2)
            self.assertEqual(
                (arguments["output"].name, arguments["stop_after_step"], arguments["resume"]),
                expected,
            )

    def test_offline_environment_replaces_gpu_and_runtime_paths(self):
        with patch.dict(
            os.environ,
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "HTTPS_PROXY": "SYNTHETIC",
                "PYTHONPATH": "/SYNTHETIC",
                "RANK": "8",
                "MASTER_ADDR": "example.invalid",
            },
        ):
            environment = integration._environment(Path("/SYNTHETIC"))
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
        self.assertEqual(environment["TRANSFORMERS_OFFLINE"], "1")
        for name in ("HTTPS_PROXY", "PYTHONPATH", "RANK", "MASTER_ADDR"):
            self.assertNotIn(name, environment)
        for name in ("HOME", "AWS_ACCESS_KEY_ID", "SSH_AUTH_SOCK", "OPENAI_API_KEY"):
            self.assertNotIn(name, environment)

    def test_python_network_guard_allows_only_loopback(self):
        for event, args in (
            ("socket.connect", (None, ("127.0.0.1", 1234))),
            ("socket.bind", (None, ("::1", 1234))),
            ("socket.getaddrinfo", ("localhost", 1234)),
        ):
            integration._deny_python_external_network(event, args)
        for event, args in (
            ("socket.connect", (None, ("192.0.2.1", 443))),
            ("socket.bind", (None, ("0.0.0.0", 0))),
            ("socket.getaddrinfo", ("example.invalid", 443)),
        ):
            with self.assertRaises(integration.IntegrationError):
                integration._deny_python_external_network(event, args)

    def test_runtime_path_rejects_unexpected_interpreter_before_paths(self):
        with (
            patch.object(integration.sys, "executable", "/SYNTHETIC/python"),
            patch.object(integration.sys, "path", []) as paths,
        ):
            with self.assertRaises(integration.IntegrationError):
                integration._runtime_paths()
            self.assertEqual(paths, [])

    def test_mask_adapter_rejects_unknown_signatures(self):
        def wrong_optional_default(
            config,
            inputs_embeds,
            attention_mask,
            past_key_values,
            position_ids=None,
            or_mask_function=None,
            and_mask_function=None,
            block_sequence_ids=None,
            layer_idx=None,
            allow_is_causal_skip=False,
        ):
            return None

        for function in (
            lambda **kwargs: None,
            lambda config, inputs_embeds: None,
            wrong_optional_default,
        ):
            with self.assertRaises(integration.IntegrationError):
                integration._make_cpu_mask_adapter(function, None)

    def test_report_rejects_arbitrary_fields_before_forwarding(self):
        for report in (
            {},
            {
                "schema_version": integration.SCHEMA,
                "status": "PASS_synthetic_cpu_ddp_fresh_resume",
                "production_eligible": False,
                "raw_text": "SYNTHETIC-SECRET",
            },
        ):
            with self.assertRaises(integration.IntegrationError):
                integration._validate_report(report)

    def test_catchable_signal_handlers_restore(self):
        events = []

        def install(number, handler):
            events.append((number, handler))
            return "SYNTHETIC-PREVIOUS"

        with patch.object(integration.signal, "signal", side_effect=install):
            with integration._catchable_signal_cleanup() as cancellation:
                self.assertEqual(len(events), 3)
                events[0][1](signal.SIGTERM, None)
                self.assertTrue(cancellation["cancelled"])
        self.assertEqual(len(events), 6)
        self.assertTrue(all(item[1] == "SYNTHETIC-PREVIOUS" for item in events[3:]))

    def test_first_late_and_double_signal_cannot_interrupt_owned_cleanup(self):
        handlers = {}
        events = []
        process = Mock(pid=654321)
        process.wait.side_effect = lambda **kwargs: events.append("wait")

        def install(number, handler):
            handlers[number] = handler
            return signal.SIG_DFL

        def kill(_pid, number):
            events.append(number)
            if number == signal.SIGTERM:
                handlers[signal.SIGINT](signal.SIGINT, None)
                handlers[signal.SIGTERM](signal.SIGTERM, None)

        with (
            patch.object(integration.signal, "signal", side_effect=install),
            patch.object(integration.os, "killpg", side_effect=kill),
        ):
            with integration._catchable_signal_cleanup() as cancellation:
                integration._stop_owned_session(process)
                self.assertTrue(cancellation["cancelled"])
        self.assertEqual(events, [signal.SIGTERM, signal.SIGKILL, "wait"])

    def test_latched_cancellation_rejects_before_pipe_wait(self):
        process = Mock()
        with patch.object(integration.select, "select") as select_call:
            with self.assertRaises(integration.IntegrationError):
                integration._read_runner_report(process, 10, cancellation={"cancelled": True})
        select_call.assert_not_called()

    def test_pipe_is_bounded_and_no_followup_write(self):
        process = Mock()
        with patch.object(integration.subprocess, "Popen", return_value=process) as spawn:
            self.assertIs(
                integration._spawn(
                    {"synthetic_root": "/SYNTHETIC"}, new_session=False, stdout=subprocess.DEVNULL
                ),
                process,
            )
            arguments, options = spawn.call_args
        self.assertEqual(arguments[0][1:4], ["-S", "-B", "-c"])
        self.assertNotIn("torch.distributed.run", arguments[0])
        self.assertFalse(options["start_new_session"])
        self.assertEqual(options["stderr"], subprocess.DEVNULL)

    def test_pipe_short_write_denies_spawn(self):
        with (
            patch.object(integration.os, "write", return_value=0),
            patch.object(integration.subprocess, "Popen") as spawn,
        ):
            with self.assertRaises(integration.IntegrationError):
                integration._spawn(
                    {"synthetic_root": "/SYNTHETIC"}, new_session=False, stdout=subprocess.DEVNULL
                )
            spawn.assert_not_called()

    def test_size_postcheck_rejects_overage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.bin"
            path.write_bytes(b"SYNTHETIC")
            with self.assertRaises(integration.IntegrationError):
                integration._postcheck_size(Path(directory), {"maximum_private_write_bytes": 1})

    def test_size_postcheck_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "synthetic.bin").write_bytes(b"SYNTHETIC")
            (root / "linked.bin").symlink_to(root / "synthetic.bin")
            with self.assertRaises(integration.IntegrationError):
                integration._tree_bytes(root)

    def test_owned_session_cleanup_targets_recorded_group(self):
        process = Mock(pid=654321)
        order = Mock()
        order.attach_mock(process.wait, "wait")
        with patch.object(integration.os, "killpg") as kill:
            order.attach_mock(kill, "killpg")
            integration._stop_owned_session(process)
        self.assertEqual(kill.call_args_list[0].args, (654321, signal.SIGTERM))
        self.assertEqual(kill.call_args_list[1].args, (654321, signal.SIGKILL))
        self.assertEqual(process.wait.call_count, 1)
        self.assertEqual([call[0] for call in order.mock_calls], ["killpg", "killpg", "wait"])

    def test_report_deadline_does_not_reap_before_cleanup(self):
        process = Mock()
        with (
            patch.object(integration.time, "monotonic", return_value=10),
            patch.object(integration.select, "select") as select_call,
        ):
            with self.assertRaises(TimeoutError):
                integration._read_runner_report(process, 9)
        select_call.assert_not_called()
        process.wait.assert_not_called()
        process.poll.assert_not_called()
        process.communicate.assert_not_called()

    def test_report_pipe_read_is_bounded_and_does_not_reap(self):
        process = Mock()
        with (
            patch.object(integration.time, "monotonic", return_value=0),
            patch.object(integration.select, "select", return_value=([process.stdout], [], [])),
            patch.object(integration.os, "read", side_effect=[b'{"synthetic":true}', b""]),
        ):
            self.assertEqual(integration._read_runner_report(process, 1), b'{"synthetic":true}')
        process.wait.assert_not_called()
        process.poll.assert_not_called()
        process.communicate.assert_not_called()

    def test_report_overflow_fails_without_reaping(self):
        process = Mock()
        with (
            patch.object(integration.time, "monotonic", return_value=0),
            patch.object(integration.select, "select", return_value=([process.stdout], [], [])),
            patch.object(integration.os, "read", return_value=b"x" * 4097),
        ):
            with self.assertRaises(integration.IntegrationError):
                integration._read_runner_report(process, 1)
        process.wait.assert_not_called()

    def test_actual_owned_stdlib_session_is_stopped_on_timeout(self):
        process = subprocess.Popen(
            [sys.executable, "-S", "-B", "-c", "import signal; signal.pause()"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            with self.assertRaises(TimeoutError):
                integration._read_runner_report(process, time.monotonic() + 0.1)
        finally:
            integration._stop_owned_session(process)
            process.stdout.close()
        self.assertIn(process.returncode, (-signal.SIGTERM, -signal.SIGKILL))

    def test_rank_launches_share_outer_group_and_have_no_retries(self):
        processes = [Mock(), Mock()]
        for process in processes:
            process.wait.return_value = 0
            process.poll.return_value = 0
        with patch.object(integration, "_spawn", side_effect=processes) as spawn:
            integration._launch_stage(admission.fixed_request(), Path("/SYNTHETIC"), {}, 0)
        self.assertEqual(spawn.call_count, 2)
        self.assertTrue(all(call.kwargs["new_session"] is False for call in spawn.call_args_list))

    def test_failed_rank_stops_peer_without_retry(self):
        first, peer = Mock(), Mock()
        first.wait.return_value, first.poll.return_value = 1, 1
        peer.wait.return_value, peer.poll.return_value = 0, None
        with patch.object(integration, "_spawn", side_effect=[first, peer]) as spawn:
            with self.assertRaises(integration.IntegrationError):
                integration._launch_stage(admission.fixed_request(), Path("/SYNTHETIC"), {}, 0)
        self.assertEqual(spawn.call_count, 2)
        peer.kill.assert_called_once_with()


class RealSyntheticCpuTrainingTests(unittest.TestCase):
    def test_real_ddp_training_and_fresh_resume(self):
        self.assertTrue(
            EXECUTABLE.is_file(), "The executing interpreter is required; no runtime/GPU fallback"
        )
        result = subprocess.run(
            [str(EXECUTABLE), "-S", "-B", str(SCRIPT), "--run-synthetic-cpu"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=195,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout.decode() + result.stderr.decode())
        self.assertEqual(result.stderr, b"")
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "PASS_synthetic_cpu_ddp_fresh_resume")
        child_report = {
            key: value
            for key, value in report.items()
            if key
            not in {"elapsed_seconds", "temporary_cleanup_completed", "wall_watchdog_seconds"}
        }
        self.assertEqual(integration._validate_report(child_report), child_report)
        for changed in (
            {"raw_text": "SYNTHETIC-SECRET"},
            {"model_changed": False},
            {"production_eligible": True},
            {"world_size": True},
            {"runtime_torch_version": [3, 0, 0]},
            {"runtime_transformers_version": [5, 2]},
            {"runtime_torch_version": [float("nan"), 10, 0]},
            {"checked_tree_bytes_high_water": 67_108_865},
            {"source_parameters": False},
            {"source_parameters": 20_000},
        ):
            with self.subTest(report_mutation=list(changed)):
                with self.assertRaises(integration.IntegrationError):
                    integration._validate_report({**child_report, **changed})
        for field in (
            "synthetic_only",
            "model_changed",
            "exact_model",
            "exact_optimizer",
            "exact_rng",
            "exact_stable_state",
            "exact_stable_metrics",
            "finite_loss_and_gradient",
            "supervised_token_accounting_verified",
            "causal_padding_mask_oracle_verified",
            "temporary_cleanup_completed",
            "admission_before_ml",
        ):
            self.assertIs(report[field], True, field)
        for field in (
            "production_eligible",
            "private_or_gpu_execution_enabled",
            "cuda_initialized",
            "hard_aggregate_write_enforcement",
            "memory_enforcement",
            "external_network_metering",
            "parent_death_guardian",
        ):
            self.assertIs(report[field], False, field)
        for field, expected in (
            ("world_size", 2),
            ("sequential_invocations", 3),
            ("rank_processes", 6),
            ("global_optimizer_steps", 4),
            ("rank_optimizer_steps", 8),
            ("unique_selected_records", 4),
            ("checkpoints_verified", 4),
            ("optimizer_rng_pairs_verified", 4),
            ("inference_calls", 0),
            ("wall_watchdog_seconds", 180),
        ):
            self.assertEqual(report[field], expected, field)
        self.assertGreater(report["source_parameters"], 0)
        self.assertLess(report["source_parameters"], 20_000)
        self.assertLessEqual(report["checked_tree_bytes_high_water"], 67_108_864)
        self.assertNotIn(str(REPOSITORY), result.stdout.decode())
        self.assertNotIn("lumi-SYNTHETIC-CPU-v1-", result.stdout.decode())


if __name__ == "__main__":
    unittest.main()
