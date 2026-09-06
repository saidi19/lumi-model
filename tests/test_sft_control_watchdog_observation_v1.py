"""Synthetic traceback observations only; no control suites, children or artifact reads."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "synthetic_control_watchdog_observation", ROOT / "scripts/run_sft_control_ci_v1.py"
)
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


class UnformattableAssertion(AssertionError):
    def __str__(self):
        raise AssertionError("formatting forbidden")

    __repr__ = __str__


class UnformattableTest:
    failureException = AssertionError

    def id(self):
        raise AssertionError("test identity forbidden")

    def __str__(self):
        raise AssertionError("test formatting forbidden")

    __repr__ = __str__


def lifecycle(**changes):
    return {
        "reason": "deadline_exhausted_before_spawn",
        "child_spawned": False,
        "worker_status_reported": False,
        "exit_code": None,
        "guardian_exit_code": None,
        "elapsed_milliseconds": 150,
        "group_term_attempted": False,
        "group_kill_attempted": False,
        "direct_child_reaped": False,
        **changes,
    }


def fail_with_result(result, error):
    raise error


def fail_without_result(value, error):
    raise error


def observed(value, *, codes=None, function=fail_with_result, kind="failure"):
    if codes is None:
        codes = (fail_with_result.__code__,)
    aggregate = cli.AggregateResult(codes)
    test = UnformattableTest()
    aggregate.startTest(test)
    try:
        function(value, UnformattableAssertion("SYNTHETIC_PRIVATE_EXCEPTION"))
    except UnformattableAssertion:
        err = sys.exc_info()
        if kind == "failure":
            aggregate.addFailure(test, err)
        elif kind == "subtest_failure":
            aggregate.addSubTest(test, UnformattableTest(), err)
        elif kind == "error":
            aggregate.addError(test, err)
        elif kind == "expected_failure":
            aggregate.addExpectedFailure(test, err)
    return aggregate


def envelope(observation, *, suite=cli.WATCHDOG_SUITE):
    expected = dict(cli.SUITES)[suite]
    return {
        "schema_version": cli.SCHEMA,
        "status": "STOP",
        "failure_code": "control_tests_failed",
        "diagnostic": {
            "suite": suite,
            "expected": expected,
            "counters": {
                "run": expected, "failures": 1, "errors": 0, "skipped": 0,
                "expected_failures": 0, "unexpected_successes": 0,
            },
            "cases": [{
                "test_index": 7, "kind": "failure", "category": "assertion",
                "watchdog": observation,
            }],
            "truncated": False,
        },
    }


def encoded(value):
    return json.dumps(value, separators=(",", ":")).encode()


class WatchdogObservationTests(unittest.TestCase):
    def assert_omitted(self, value, **options):
        aggregate = observed(value, **options)
        self.assertEqual(aggregate.failures, [None])
        self.assertEqual(aggregate.diagnostics, [{
            "test_index": 1, "kind": "failure", "category": "assertion",
        }])

    def assert_invalid(self, value):
        with self.assertRaises(cli.ControlCIError) as caught:
            cli.validate_failure(encoded(value))
        self.assertEqual(str(caught.exception), "child_failure_invalid")
        self.assertIsNone(caught.exception.diagnostic)

    def test_exact_allowed_code_extracts_only_fixed_lifecycle_fields(self):
        source = lifecycle(
            schema_version="SYNTHETIC_PRIVATE_SCHEMA", raw_text=UnformattableTest(),
            unknown_field="SYNTHETIC_PRIVATE_OUTPUT",
        )
        with (
            patch.object(Path, "read_text", side_effect=AssertionError("read forbidden")),
            patch.object(Path, "read_bytes", side_effect=AssertionError("read forbidden")),
            patch("builtins.open", side_effect=AssertionError("read forbidden")),
        ):
            aggregate = observed(source)
        self.assertEqual(aggregate.failures, [None])
        self.assertEqual(aggregate.diagnostics[0]["watchdog"], lifecycle())
        self.assertNotIn("SYNTHETIC_PRIVATE", encoded(aggregate.diagnostics).decode())
        self.assertNotIn("Unformattable", encoded(aggregate.diagnostics).decode())

    def test_unchanged_default_disables_observation(self):
        self.assert_omitted(lifecycle(), codes=())
        self.assertEqual(cli.AggregateResult().watchdog_codes, ())

    def test_unrecognized_code_and_missing_result_are_ignored(self):
        self.assert_omitted(lifecycle(), codes=(fail_without_result.__code__,))
        self.assert_omitted(
            lifecycle(), function=fail_without_result, codes=(fail_without_result.__code__,)
        )

    def test_equivalent_but_not_identical_code_is_not_allowed(self):
        copied_code = fail_with_result.__code__.replace()
        self.assertEqual(copied_code, fail_with_result.__code__)
        self.assertIsNot(copied_code, fail_with_result.__code__)
        self.assert_omitted(lifecycle(), codes=(copied_code,))

    def test_invalid_or_oversized_allowlist_disables_observation(self):
        for codes in (
            [fail_with_result.__code__], (object(),), ("SYNTHETIC_PRIVATE",),
            (fail_with_result.__code__,) * (cli.MAXIMUM_WATCHDOG_CODES + 1),
        ):
            self.assert_omitted(lifecycle(), codes=codes)

    def test_bounded_traceback_scan_does_not_reach_deep_allowed_frame(self):
        def recurse(depth, value, error):
            if depth:
                return recurse(depth - 1, value, error)
            return fail_with_result(value, error)

        def deep(value, error):
            return recurse(cli.MAXIMUM_WATCHDOG_TRACEBACK_FRAMES, value, error)

        self.assert_omitted(lifecycle(), function=deep)

    def test_missing_nonplain_or_oversized_local_results_are_ignored(self):
        class NonplainDict(dict):
            def __getitem__(self, key):
                raise AssertionError("mapping access forbidden")

        missing = lifecycle()
        del missing["reason"]
        oversized = lifecycle()
        oversized.update({str(index): None for index in range(cli.MAXIMUM_WATCHDOG_RESULT_FIELDS)})
        for value in (None, [], UnformattableTest(), NonplainDict(lifecycle()), missing, oversized):
            self.assert_omitted(value)

    def test_unknown_reason_wrong_types_and_out_of_range_values_are_omitted(self):
        invalid = [
            ("reason", "SYNTHETIC_PRIVATE"), ("reason", "x" * 100_000),
            ("reason", UnformattableTest()), ("reason", None),
            ("elapsed_milliseconds", -1), ("elapsed_milliseconds", True),
            ("elapsed_milliseconds", 1.0), ("elapsed_milliseconds", "150"),
            ("elapsed_milliseconds", cli.MAXIMUM_WATCHDOG_ELAPSED_MILLISECONDS + 1),
        ]
        for name in cli.WATCHDOG_BOOLEAN_FIELDS:
            invalid.extend((name, value) for value in (0, 1, None, "false"))
        for name in ("exit_code", "guardian_exit_code"):
            invalid.extend((name, value) for value in (-256, 256, True, 1.0, "1"))
        for field, value in invalid:
            self.assert_omitted(lifecycle(**{field: value}))

    def test_all_fixed_reasons_and_primitive_boundaries_are_valid(self):
        for reason in cli.WATCHDOG_REASONS:
            value = lifecycle(
                reason=reason, exit_code=-255, guardian_exit_code=255,
                elapsed_milliseconds=cli.MAXIMUM_WATCHDOG_ELAPSED_MILLISECONDS,
                **dict.fromkeys(cli.WATCHDOG_BOOLEAN_FIELDS, True),
            )
            self.assertEqual(observed(value).diagnostics[0]["watchdog"], value)
            self.assertEqual(cli.validate_failure(encoded(envelope(value))), envelope(value)[
                "diagnostic"
            ])
        value = lifecycle(elapsed_milliseconds=0, exit_code=0, guardian_exit_code=0)
        self.assertTrue(cli.valid_watchdog_observation(value))

    def test_observation_fault_cannot_prevent_original_failure_recording(self):
        with patch.object(cli, "watchdog_observation", side_effect=UnformattableAssertion()):
            self.assert_omitted(lifecycle())
        with patch.object(cli, "valid_watchdog_observation", side_effect=UnformattableAssertion()):
            self.assert_omitted(lifecycle())

    def test_only_failure_and_subtest_failure_receive_observations(self):
        aggregate = observed(lifecycle(), kind="subtest_failure")
        self.assertEqual(aggregate.failures, [None])
        self.assertEqual(aggregate.diagnostics[0]["kind"], "subtest_failure")
        self.assertEqual(aggregate.diagnostics[0]["watchdog"], lifecycle())
        for kind in ("error", "expected_failure"):
            aggregate = observed(lifecycle(), kind=kind)
            self.assertNotIn("watchdog", aggregate.diagnostics[0])
        aggregate.addSkip(UnformattableTest(), UnformattableTest())
        aggregate.addUnexpectedSuccess(UnformattableTest())
        self.assertTrue(all("watchdog" not in case for case in aggregate.diagnostics))

    def test_parent_rejects_observations_on_other_suites(self):
        for suite, _ in cli.SUITES:
            if suite != cli.WATCHDOG_SUITE:
                self.assert_invalid(envelope(lifecycle(), suite=suite))

    def test_parent_rejects_missing_extra_and_nonplain_observation_fields(self):
        for field in cli.WATCHDOG_OBSERVATION_FIELDS:
            value = lifecycle()
            del value[field]
            self.assert_invalid(envelope(value))
        self.assert_invalid(envelope(lifecycle(raw_text="SYNTHETIC_PRIVATE")))
        for value in (None, [], "SYNTHETIC_PRIVATE", 0):
            self.assert_invalid(envelope(value))

    def test_parent_rejects_invalid_observation_types_and_bounds(self):
        for field, bad in (
            ("reason", "SYNTHETIC_PRIVATE"), ("reason", []), ("reason", False),
            ("child_spawned", 0), ("worker_status_reported", "false"),
            ("group_term_attempted", 1), ("group_kill_attempted", None),
            ("direct_child_reaped", {}), ("exit_code", True), ("exit_code", -256),
            ("guardian_exit_code", 256), ("guardian_exit_code", "-9"),
            ("elapsed_milliseconds", -1), ("elapsed_milliseconds", 1.0),
            ("elapsed_milliseconds", cli.MAXIMUM_WATCHDOG_ELAPSED_MILLISECONDS + 1),
        ):
            self.assert_invalid(envelope(lifecycle(**{field: bad})))

    def test_parent_rejects_watchdog_data_for_nonfailure_kinds(self):
        for kind, counter in (
            ("error", "errors"), ("subtest_error", "errors"), ("skip", "skipped"),
            ("expected_failure", "expected_failures"),
            ("unexpected_success", "unexpected_successes"),
        ):
            value = envelope(lifecycle())
            detail = value["diagnostic"]
            detail["cases"][0]["kind"] = kind
            detail["counters"].update(failures=0, **{counter: 1})
            self.assert_invalid(value)

    def test_invalid_traceback_is_ignored_without_formatting(self):
        aggregate = cli.AggregateResult((fail_with_result.__code__,))
        aggregate.startTest(UnformattableTest())
        aggregate.addFailure(
            UnformattableTest(), (UnformattableAssertion, UnformattableAssertion(), object())
        )
        self.assertNotIn("watchdog", aggregate.diagnostics[0])
        self.assertIsNone(cli.watchdog_observation(None, (fail_with_result.__code__,)))

    def test_diagnostics_remain_bounded_and_original_failures_still_count(self):
        aggregate = cli.AggregateResult((fail_with_result.__code__,))
        aggregate.startTest(UnformattableTest())
        for _ in range(cli.MAXIMUM_FAILURE_CASES + 1):
            try:
                fail_with_result(lifecycle(), UnformattableAssertion())
            except UnformattableAssertion:
                aggregate.addFailure(UnformattableTest(), sys.exc_info())
        self.assertEqual(len(aggregate.failures), cli.MAXIMUM_FAILURE_CASES + 1)
        self.assertEqual(len(aggregate.diagnostics), cli.MAXIMUM_FAILURE_CASES)
        self.assertTrue(aggregate.truncated)
        self.assertLess(len(encoded(aggregate.diagnostics)), cli.MAXIMUM_REPORT_BYTES)

    def test_traceback_and_code_allowlists_use_real_runtime_types(self):
        self.assertIs(type(fail_with_result.__code__), types.CodeType)
        self.assertEqual(observed(lifecycle()).diagnostics[0]["watchdog"], lifecycle())


if __name__ == "__main__":
    unittest.main()
