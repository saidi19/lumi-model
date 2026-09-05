"""Synthetic CI-route validation only; no ML, private artifacts or child processes."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_sft_control_ci_v1.py"
SPEC = importlib.util.spec_from_file_location("synthetic_sft_control_ci", SCRIPT)
ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ci)


def valid_counts(count: int) -> dict:
    return {
        "run": count, "failures": 0, "errors": 0, "skipped": 0,
        "expected_failures": 0, "unexpected_successes": 0,
    }


def valid_report() -> dict:
    return ci.make_report({name: valid_counts(count) for name, count in ci.SUITES})


class ControlCITests(unittest.TestCase):
    def test_exact_seven_paths_and_total_are_fixed(self):
        self.assertEqual(len(ci.SUITES), 7)
        self.assertEqual(sum(count for _, count in ci.SUITES), 273)
        self.assertEqual(len({name for name, _ in ci.SUITES}), 7)
        self.assertNotIn("tests/test_sft_control_ci_v1.py", dict(ci.SUITES))

    def test_list_json_does_not_import_or_run_tests(self):
        output = io.StringIO()
        with (
            patch.object(ci, "require_startup"),
            patch.object(ci, "execute_suites", side_effect=AssertionError("must not run")),
            patch.object(ci, "run_fresh_process", side_effect=AssertionError("must not spawn")),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(ci.main(["--list-json"]), 0)
        self.assertEqual(json.loads(output.getvalue()), [name for name, _ in ci.SUITES])

    def test_unexpected_arguments_rejected_before_any_suite_or_child(self):
        for arguments in (["--module", "private.module"], ["--execute"], ["--list-json", "extra"]):
            with (
                self.subTest(arguments=arguments),
                patch.object(ci, "require_startup"),
                patch.object(ci, "execute_suites", side_effect=AssertionError("must not run")),
                patch.object(ci, "run_fresh_process", side_effect=AssertionError("must not spawn")),
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                self.assertEqual(ci.main(arguments), 1)
                self.assertEqual(
                    json.loads(output.getvalue())["failure_code"], "unexpected_arguments"
                )

    def test_startup_rejects_missing_flags(self):
        for flags, bytecode in ((0, True), (1, False)):
            with (
                patch.object(ci.sys, "flags") as observed,
                patch.object(ci.sys, "dont_write_bytecode", bytecode),
                self.assertRaisesRegex(ci.ControlCIError, "use_python_minus_S_minus_B"),
            ):
                observed.no_site = flags
                ci.require_startup()

    def test_startup_rejects_preloaded_ml(self):
        with (
            patch.object(ci.sys, "flags") as observed,
            patch.object(ci.sys, "dont_write_bytecode", True),
            patch.dict(sys.modules, {"torch.synthetic": object()}),
            self.assertRaisesRegex(ci.ControlCIError, "stdlib_only_startup_required"),
        ):
            observed.no_site = 1
            ci.require_startup()

    def test_parent_rejects_preloaded_lumi_before_spawning(self):
        with (
            patch.object(ci, "require_startup"),
            patch.dict(sys.modules, {"lumi": object()}),
            self.assertRaisesRegex(ci.ControlCIError, "parent_must_not_import_lumi"),
        ):
            ci.require_parent_startup()

    def test_full_aggregate_report_passes(self):
        report = valid_report()
        self.assertEqual(ci.validate_report(json.dumps(report).encode()), report)
        self.assertEqual(report["tests_run"], 273)
        self.assertFalse(report["gpu_execution_authorized"])

    def test_wrong_suite_count_or_missing_suite_fails(self):
        results = {name: valid_counts(count) for name, count in ci.SUITES}
        results.pop(ci.SUITES[0][0])
        with self.assertRaisesRegex(ci.ControlCIError, "suite_set_changed"):
            ci.make_report(results)

    def test_wrong_test_count_fails(self):
        with self.assertRaisesRegex(ci.ControlCIError, "test_count_changed"):
            ci.validate_counts(valid_counts(272), 273)

    def test_failure_error_skip_and_expected_failure_all_fail(self):
        for field in ("failures", "errors", "skipped", "expected_failures", "unexpected_successes"):
            with self.subTest(field=field), self.assertRaises(ci.ControlCIError):
                ci.validate_counts({**valid_counts(1), field: 1}, 1)

    def test_counter_types_are_strict(self):
        for value in (True, 1.0, "1", -1):
            with self.subTest(value=value), self.assertRaises(ci.ControlCIError):
                ci.validate_counts({**valid_counts(1), "run": value}, 1)

    def test_report_rejects_unknown_fields_and_readiness_claims(self):
        for update in (
            {"raw_text": "PRIVATE"}, {"gpu_execution_authorized": True}, {"suite_count": True},
        ):
            with self.subTest(update=update), self.assertRaises(ci.ControlCIError):
                ci.validate_report(json.dumps({**valid_report(), **update}).encode())

    def test_invalid_and_oversized_reports_rejected(self):
        for raw in (b"", b"PRIVATE invalid JSON", b"x" * (ci.MAXIMUM_REPORT_BYTES + 1)):
            with self.subTest(size=len(raw)), self.assertRaises(ci.ControlCIError):
                ci.validate_report(raw)

    def test_duplicate_report_keys_rejected(self):
        with self.assertRaises(ci.ControlCIError):
            ci.validate_report(b'{"tests":{},"tests":{}}')

    def test_aggregate_result_never_formats_exception_text(self):
        result = ci.AggregateResult()
        synthetic = unittest.FunctionTestCase(lambda: None)
        error = (RuntimeError, RuntimeError("PRIVATE"), None)
        result.addError(synthetic, error)
        result.addFailure(synthetic, error)
        result.addSkip(synthetic, "PRIVATE")
        self.assertEqual(result.errors, [None])
        self.assertEqual(result.failures, [None])
        self.assertEqual(result.skipped, [None])
        self.assertNotIn("PRIVATE", json.dumps(ci.counters(result)))

    def test_child_environment_is_exact_not_inherited(self):
        root = Path("/synthetic/repository")
        with patch.dict(ci.os.environ, {"AWS_SESSION_TOKEN": "PRIVATE", "PYTHONPATH": "/private"}):
            environment = ci.child_environment(root)
        self.assertNotIn("AWS_SESSION_TOKEN", environment)
        self.assertEqual(
            environment["PYTHONPATH"], "/synthetic/repository/src:/synthetic/repository/tests"
        )
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "")

    def test_child_command_is_fixed_fresh_and_bounded(self):
        response = subprocess.CompletedProcess([], 0, json.dumps(valid_report()).encode())
        root = Path("/synthetic/repository")
        with (
            patch.object(ci, "require_parent_startup"),
            patch.object(ci.subprocess, "run", return_value=response) as run,
        ):
            report = ci.run_fresh_process(root)
        command = run.call_args.args[0]
        self.assertEqual(command[1:3], ["-S", "-B"])
        self.assertEqual(command[-1], ci.CHILD_MODE)
        self.assertEqual(run.call_args.kwargs["timeout"], 120)
        self.assertIs(run.call_args.kwargs["shell"], False)
        self.assertEqual(report["tests_run"], 273)

    def test_child_timeout_and_spawn_failure_are_sanitized(self):
        for error, code in (
            (subprocess.TimeoutExpired(["PRIVATE"], 120), "control_suite_timeout"),
            (OSError("PRIVATE"), "control_child_start_failed"),
        ):
            with (
                self.subTest(code=code),
                patch.object(ci, "require_parent_startup"),
                patch.object(ci.subprocess, "run", side_effect=error),
                self.assertRaisesRegex(ci.ControlCIError, "^" + code + "$"),
            ):
                ci.run_fresh_process(Path("/synthetic/repository"))

    def test_nonzero_child_exit_fails_without_printing_output(self):
        response = subprocess.CompletedProcess([], 1, b"PRIVATE")
        with (
            patch.object(ci, "require_parent_startup"),
            patch.object(ci.subprocess, "run", return_value=response),
            self.assertRaisesRegex(ci.ControlCIError, "control_child_failed"),
        ):
            ci.run_fresh_process(Path("/synthetic/repository"))


if __name__ == "__main__":
    unittest.main()
