"""Synthetic failure-report tests; no child process or routed control test runs."""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "synthetic_control_failure_diagnostics", ROOT / "scripts/run_sft_control_ci_v1.py"
)
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


def counts(run, **changes):
    return {
        "run": run,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "expected_failures": 0,
        "unexpected_successes": 0,
        **changes,
    }


def failure():
    suite, expected = cli.SUITES[0]
    return {
        "schema_version": cli.SCHEMA,
        "status": "STOP",
        "failure_code": "control_tests_failed",
        "diagnostic": {
            "suite": suite,
            "expected": expected,
            "counters": counts(expected, failures=1),
            "cases": [{"test_index": 1, "kind": "failure", "category": "assertion"}],
            "truncated": False,
        },
    }


def encoded(value):
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


class UnformattableError(Exception):
    def __str__(self):
        raise AssertionError("exception formatting forbidden")

    __repr__ = __str__


class UnformattableTest:
    failureException = AssertionError

    def id(self):
        raise AssertionError("test identity access forbidden")

    def __str__(self):
        raise AssertionError("test formatting forbidden")

    __repr__ = __str__


class ControlFailureDiagnosticTests(unittest.TestCase):
    def assert_invalid(self, value):
        with self.assertRaises(cli.ControlCIError) as caught:
            cli.validate_failure(encoded(value))
        self.assertEqual(str(caught.exception), "child_failure_invalid")
        self.assertIsNone(caught.exception.diagnostic)

    def test_aggregate_never_formats_test_exception_or_skip_reason(self):
        result = cli.AggregateResult()
        test = UnformattableTest()
        error = UnformattableError("SYNTHETIC_PRIVATE_TEXT")
        result.startTest(test)
        result.addError(test, (type(error), error, object()))
        result.addSkip(test, error)
        self.assertEqual(result.errors, [None])
        self.assertEqual(result.skipped, [None])
        self.assertEqual(result.diagnostics, [
            {"test_index": 1, "kind": "error", "category": "other"},
            {"test_index": 1, "kind": "skip", "category": "none"},
        ])
        text = encoded(result.diagnostics).decode()
        self.assertNotIn("SYNTHETIC_PRIVATE_TEXT", text)
        self.assertNotIn("Unformattable", text)

    def test_all_result_kinds_and_fixed_categories_preserve_old_counters(self):
        result = cli.AggregateResult()
        test = UnformattableTest()
        result.startTest(test)
        assertion = (AssertionError, AssertionError("SYNTHETIC"), None)
        os_error = (OSError, OSError("SYNTHETIC"), None)
        timeout = (TimeoutError, TimeoutError("SYNTHETIC"), None)
        result.addFailure(test, assertion)
        result.addError(test, os_error)
        result.addError(test, timeout)
        result.addExpectedFailure(test, assertion)
        result.addUnexpectedSuccess(test)
        result.addSubTest(test, UnformattableTest(), assertion)
        result.addSubTest(test, UnformattableTest(), os_error)
        result.addSubTest(test, UnformattableTest(), None)
        self.assertEqual(cli.counters(result), counts(
            1, failures=2, errors=3, expected_failures=1, unexpected_successes=1
        ))
        self.assertEqual([(case["kind"], case["category"]) for case in result.diagnostics], [
            ("failure", "assertion"), ("error", "os_error"), ("error", "timeout"),
            ("expected_failure", "assertion"), ("unexpected_success", "none"),
            ("subtest_failure", "assertion"), ("subtest_error", "os_error"),
        ])

    def test_suite_setup_error_can_use_index_zero(self):
        result = cli.AggregateResult()
        result.addError(UnformattableTest(), (OSError, OSError("SYNTHETIC"), None))
        value = failure()
        value["failure_code"] = "test_count_changed"
        value["diagnostic"]["counters"] = cli.counters(result)
        value["diagnostic"]["cases"] = result.diagnostics
        self.assertEqual(cli.validate_failure(encoded(value)), value["diagnostic"])
        self.assertEqual(result.diagnostics[0]["test_index"], 0)

    def test_subprocess_timeout_is_classified_without_formatting(self):
        class UnformattableTimeout(cli.subprocess.TimeoutExpired):
            def __str__(self):
                raise AssertionError("exception formatting forbidden")

            __repr__ = __str__

        result = cli.AggregateResult()
        result.startTest(UnformattableTest())
        error = UnformattableTimeout("SYNTHETIC_PRIVATE_COMMAND", 1)
        result.addError(UnformattableTest(), (type(error), error, None))
        self.assertEqual(result.diagnostics, [
            {"test_index": 1, "kind": "error", "category": "timeout"},
        ])
        self.assertNotIn("SYNTHETIC_PRIVATE_COMMAND", encoded(result.diagnostics).decode())

    def test_detail_is_bounded_but_every_failure_remains_counted(self):
        result = cli.AggregateResult()
        result.startTest(UnformattableTest())
        for _ in range(40):
            result.addFailure(UnformattableTest(), (AssertionError, None, None))
        self.assertEqual(len(result.diagnostics), 32)
        self.assertEqual(len(result.failures), 40)
        self.assertTrue(result.truncated)
        value = failure()
        value["failure_code"] = "test_count_changed"
        value["diagnostic"].update(
            counters=cli.counters(result), cases=result.diagnostics, truncated=result.truncated
        )
        self.assertEqual(cli.validate_failure(encoded(value)), value["diagnostic"])

    def test_valid_failure_returns_only_the_allowlisted_diagnostic(self):
        value = failure()
        self.assertEqual(cli.validate_failure(encoded(value)), value["diagnostic"])

    def test_unknown_or_missing_fields_are_rejected_at_every_level(self):
        for level in ("envelope", "diagnostic", "counters", "case"):
            for operation in ("extra", "missing"):
                value = failure()
                targets = {
                    "envelope": value,
                    "diagnostic": value["diagnostic"],
                    "counters": value["diagnostic"]["counters"],
                    "case": value["diagnostic"]["cases"][0],
                }
                target = targets[level]
                if operation == "extra":
                    target["raw_exception"] = "SYNTHETIC_PRIVATE_TEXT"
                else:
                    del target[next(iter(target))]
                self.assert_invalid(value)

    def test_envelope_and_suite_identity_are_strict(self):
        for field, bad in (
            ("schema_version", "unknown"), ("status", "PASS"),
            ("failure_code", "unknown"), ("failure_code", []),
        ):
            value = failure()
            value[field] = bad
            self.assert_invalid(value)
        for field, bad in (
            ("suite", "tests/test_unselected.py"), ("suite", 0), ("expected", True),
            ("expected", cli.SUITES[0][1] + 1), ("truncated", 0), ("cases", {}),
        ):
            value = failure()
            value["diagnostic"][field] = bad
            self.assert_invalid(value)

    def test_counter_types_limits_and_false_success_are_rejected(self):
        for field in counts(1):
            for bad in (True, -1, 1.0, "1", None, 1_000_001):
                value = failure()
                value["diagnostic"]["counters"][field] = bad
                self.assert_invalid(value)
        value = failure()
        value["diagnostic"]["counters"] = counts(cli.SUITES[0][1])
        value["diagnostic"]["cases"] = []
        self.assert_invalid(value)
        value = failure()
        value["diagnostic"]["counters"]["run"] += 1
        self.assert_invalid(value)

    def test_case_indices_types_kinds_and_categories_are_strict(self):
        for field, bad in (
            ("test_index", -1), ("test_index", True), ("test_index", 1.0),
            ("test_index", "1"), ("test_index", cli.SUITES[0][1] + 1),
            ("kind", "SYNTHETIC_PRIVATE_TEXT"), ("kind", []),
            ("category", "SYNTHETIC_EXCEPTION_CLASS"), ("category", {}),
        ):
            value = failure()
            value["diagnostic"]["cases"][0][field] = bad
            self.assert_invalid(value)
        value = failure()
        value["diagnostic"]["cases"] = [None]
        self.assert_invalid(value)

    def test_case_totals_and_truncation_must_match_counters(self):
        for change in ("missing", "duplicate", "wrong_kind", "wrong_truncation", "oversized"):
            value = failure()
            detail = value["diagnostic"]
            if change == "missing":
                detail["cases"] = []
            elif change == "duplicate":
                detail["cases"] *= 2
            elif change == "wrong_kind":
                detail["cases"][0]["kind"] = "error"
            elif change == "wrong_truncation":
                detail["truncated"] = True
            else:
                detail["cases"] *= 33
                detail["counters"]["failures"] = 33
                detail["truncated"] = True
            self.assert_invalid(value)

    def test_malformed_duplicate_oversized_and_nonbytes_reports_are_rejected(self):
        raw = encoded(failure())
        duplicate = b'{"status":"STOP",' + raw[1:]
        for malformed in (
            b"", b"not json", b"\xff", duplicate, raw[:-1],
            b" " * (cli.MAXIMUM_REPORT_BYTES + 1), raw.decode(), bytearray(raw),
        ):
            with self.assertRaises(cli.ControlCIError) as caught:
                cli.validate_failure(malformed)
            self.assertEqual(str(caught.exception), "child_failure_invalid")

    def test_nonzero_child_forwards_validated_details_but_stays_failed(self):
        value = failure()
        completed = types.SimpleNamespace(returncode=1, stdout=encoded(value))
        with (
            patch.object(cli, "require_parent_startup"),
            patch.object(cli.subprocess, "run", return_value=completed) as child,
            self.assertRaises(cli.ControlCIError) as caught,
        ):
            cli.run_fresh_process(ROOT)
        self.assertEqual(str(caught.exception), "control_child_failed")
        self.assertEqual(caught.exception.diagnostic, value["diagnostic"])
        self.assertEqual(child.call_args.kwargs["timeout"], 120)
        self.assertEqual(child.call_args.args[0][1:3], ["-S", "-B"])
        self.assertFalse(child.call_args.kwargs["shell"])

    def test_nonzero_child_discards_invalid_raw_output(self):
        changed = failure()
        changed["diagnostic"]["raw_text"] = "SYNTHETIC_PRIVATE_TEXT"
        for raw in (b"SYNTHETIC_PRIVATE_TRACEBACK", encoded(changed), b"x" * 16_385):
            with (
                patch.object(cli, "require_parent_startup"),
                patch.object(cli.subprocess, "run", return_value=types.SimpleNamespace(
                    returncode=1, stdout=raw
                )),
                self.assertRaises(cli.ControlCIError) as caught,
            ):
                cli.run_fresh_process(ROOT)
            self.assertEqual(str(caught.exception), "control_child_failed")
            self.assertIsNone(caught.exception.diagnostic)

    def test_pass_schema_counts_flags_and_validation_are_unchanged(self):
        report = cli.make_report({suite: counts(expected) for suite, expected in cli.SUITES})
        self.assertEqual(report, {
            "schema_version": cli.SCHEMA,
            "status": "PASS_fixed_synthetic_control_suites",
            "tests": {suite: counts(expected) for suite, expected in cli.SUITES},
            "tests_run": 273,
            "suite_count": 7,
            "gpu_execution_authorized": False,
            "private_artifact_integration_tested": False,
            "retained_test_output_bytes": 0,
        })
        with (
            patch.object(cli, "require_parent_startup"),
            patch.object(cli.subprocess, "run", return_value=types.SimpleNamespace(
                returncode=0, stdout=encoded(report)
            )),
        ):
            self.assertEqual(cli.run_fresh_process(ROOT), report)
        changed = copy.deepcopy(report)
        changed["diagnostic"] = failure()["diagnostic"]
        with self.assertRaises(cli.ControlCIError):
            cli.validate_report(encoded(changed))
        with self.assertRaises(cli.ControlCIError):
            cli.validate_report(encoded(failure()))

    def test_main_failure_envelope_remains_stop_with_sanitized_details(self):
        detail = failure()["diagnostic"]
        capture = io.StringIO()
        with (
            patch.object(cli, "require_startup"),
            patch.object(cli, "execute_suites", side_effect=cli.ControlCIError(
                "control_tests_failed", detail
            )),
            contextlib.redirect_stdout(capture),
        ):
            result = cli.main([cli.CHILD_MODE])
        self.assertEqual(result, 1)
        self.assertEqual(json.loads(capture.getvalue()), failure())

    def test_execute_suites_attaches_only_the_fixed_current_suite(self):
        suite = Mock()
        suite.countTestCases.return_value = cli.SUITES[0][1]

        def fail(result):
            result.testsRun = cli.SUITES[0][1]
            result.addFailure(UnformattableTest(), (AssertionError, None, None))

        suite.run.side_effect = fail
        module = types.ModuleType("synthetic_no_test_body")
        with (
            patch.object(cli, "require_startup"),
            patch.object(cli.importlib.util, "spec_from_file_location", return_value=Mock()),
            patch.object(cli.importlib.util, "module_from_spec", return_value=module),
            patch.object(cli.unittest.defaultTestLoader, "loadTestsFromModule", return_value=suite),
            patch.dict(sys.modules, {}),
            self.assertRaises(cli.ControlCIError) as caught,
        ):
            cli.execute_suites(ROOT)
        self.assertEqual(str(caught.exception), "control_tests_failed")
        detail = caught.exception.diagnostic
        self.assertEqual(detail["suite"], cli.SUITES[0][0])
        self.assertEqual(detail["expected"], cli.SUITES[0][1])
        self.assertEqual(detail["cases"], [{
            "test_index": cli.SUITES[0][1], "kind": "failure", "category": "assertion"
        }])
        suite.run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
