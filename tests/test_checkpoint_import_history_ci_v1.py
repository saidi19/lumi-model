"""Synthetic history-route guards; no ML, network, private data or child execution."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_checkpoint_import_history_ci_v1.py"
SPEC = importlib.util.spec_from_file_location("synthetic_checkpoint_history_ci", SCRIPT)
ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ci)


def valid_counts(count: int, historical: int) -> dict:
    return {
        "run": count, "passed": count - historical, "historical_failures": historical,
        "failures": 0, "errors": 0, "skipped": 0, "declared_expected_failures": 0,
        "unexpected_successes": 0, "historical_unexpected_successes": 0,
    }


def valid_report() -> dict:
    return ci.make_report({
        path: valid_counts(count, historical) for path, _, count, historical in ci.SUITES
    })


def fixed_case():
    return Mock(id=Mock(return_value=sorted(ci.EXPECTED_HISTORY)[0]))


def matching_error(root: Path):
    code = SimpleNamespace(co_name="convert_and_verify", co_filename=str(root / ci.HISTORY_ORIGIN))
    trace = SimpleNamespace(tb_next=None, tb_frame=SimpleNamespace(f_code=code), tb_lineno=212)
    return ValueError, ValueError(ci.HISTORY_MESSAGE), trace


class CheckpointHistoryCITests(unittest.TestCase):
    def test_fixed_suite_and_binding_contract(self):
        self.assertEqual(len(ci.SUITES), 2)
        self.assertEqual([item[2:] for item in ci.SUITES], [(14, 2), (17, 0)])
        self.assertEqual(len(ci.BINDINGS), 6)
        self.assertEqual(len(ci.EXPECTED_HISTORY), 2)
        self.assertTrue(all(identifier.startswith("checkpoint_history_v1.")
                            for identifier in ci.EXPECTED_HISTORY))

    def test_explicit_historical_failures_are_not_passes(self):
        report = valid_report()
        self.assertEqual(ci.validate_report(json.dumps(report).encode()), report)
        self.assertEqual(report["tests_run"], 31)
        self.assertEqual(report["tests_passed"], 29)
        self.assertEqual(report["expected_historical_failures"], 2)
        self.assertFalse(report["all_tests_passed"])
        self.assertFalse(report["v1_bfloat16_contract_passed"])
        self.assertEqual(report["v2_successor_tests_passed"], 17)

    def test_named_original_failure_is_required_and_not_generic_xfail(self):
        root = Path("/synthetic")
        result = ci.HistoryResult(root)
        result.addError(fixed_case(), matching_error(root))
        self.assertEqual(len(result.history_seen), 1)
        self.assertEqual(result.passed, 0)
        self.assertEqual(result.errors, [])
        self.assertEqual(result.expectedFailures, [])

    def test_unexpected_original_success_is_a_contract_failure(self):
        result = ci.HistoryResult(Path("/synthetic"))
        result.addSuccess(fixed_case())
        self.assertEqual(result.history_successes, 1)
        self.assertEqual(result.passed, 0)

    def test_expected_error_requires_exact_builtin_type_message_and_origin(self):
        root = Path("/synthetic")
        original = matching_error(root)
        errors = (
            (RuntimeError, RuntimeError(ci.HISTORY_MESSAGE), original[2]),
            (ValueError, ValueError("PRIVATE"), original[2]),
            (ValueError, ValueError(ci.HISTORY_MESSAGE), None),
            (ValueError, ValueError(ci.HISTORY_MESSAGE, "PRIVATE"), original[2]),
        )
        for error in errors:
            with self.subTest(error_type=error[0]):
                self.assertFalse(ci.is_historical_error(error, root))
        for field, value in (("co_name", "another_function"), ("co_filename", "/private.py")):
            error = matching_error(root)
            setattr(error[2].tb_frame.f_code, field, value)
            self.assertFalse(ci.is_historical_error(error, root))
        error = matching_error(root)
        error[2].tb_lineno = 211
        self.assertFalse(ci.is_historical_error(error, root))

    def test_matching_origin_must_be_innermost_frame(self):
        root = Path("/synthetic")
        error = matching_error(root)
        outer = SimpleNamespace(tb_next=error[2])
        self.assertTrue(ci.is_historical_error((error[0], error[1], outer), root))
        error[2].tb_next = matching_error(Path("/wrong"))[2]
        self.assertFalse(ci.is_historical_error(error, root))

    def test_same_error_on_another_test_is_not_accepted(self):
        root = Path("/synthetic")
        for identifier in ("unknown", "checkpoint_history_v2." + sorted(ci.EXPECTED_HISTORY)[0]):
            result = ci.HistoryResult(root)
            result.addError(Mock(id=Mock(return_value=identifier)), matching_error(root))
            self.assertEqual(result.errors, [None])
            self.assertEqual(result.history_seen, set())

    def test_duplicate_expected_error_is_not_accepted(self):
        root = Path("/synthetic")
        result = ci.HistoryResult(root)
        result.addError(fixed_case(), matching_error(root))
        result.addError(fixed_case(), matching_error(root))
        self.assertEqual(len(result.history_seen), 1)
        self.assertEqual(result.errors, [None])

    def test_all_other_outcomes_retain_no_exception_or_test_text(self):
        result = ci.HistoryResult(Path("/synthetic"))
        case = Mock(id=Mock(return_value="PRIVATE"))
        error = RuntimeError, RuntimeError("PRIVATE"), None
        result.addError(case, error)
        result.addFailure(case, error)
        result.addSkip(case, "PRIVATE")
        result.addExpectedFailure(case, error)
        result.addUnexpectedSuccess(case)
        result.addSubTest(case, case, error)
        self.assertEqual(result.errors, [None, None])
        self.assertNotIn("PRIVATE", json.dumps(ci.counters(result)))
        self.assertFalse(result.wasSuccessful())

    def test_unexpected_outcome_fields_fail_even_with_valid_pass_totals(self):
        for field in ci.COUNTER_FIELDS - {"run", "passed", "historical_failures"}:
            with self.subTest(field=field), self.assertRaises(ci.HistoryCIError):
                ci.validate_counts({**valid_counts(14, 2), field: 1}, 14, 2)

    def test_missing_history_wrong_run_or_wrong_pass_count_fails(self):
        for field in ("historical_failures", "run", "passed"):
            with self.subTest(field=field), self.assertRaises(ci.HistoryCIError):
                ci.validate_counts({**valid_counts(14, 2), field: 0}, 14, 2)

    def test_successor_failure_cannot_hide_behind_legacy_contract(self):
        results = valid_report()["tests"]
        results[ci.SUITES[1][0]] = valid_counts(17, 2)
        with self.assertRaises(ci.HistoryCIError):
            ci.make_report(results)

    def test_counter_schema_and_types_are_strict(self):
        for value in (True, 1.0, "14", -1):
            with self.subTest(value=value), self.assertRaises(ci.HistoryCIError):
                ci.validate_counts({**valid_counts(14, 2), "run": value}, 14, 2)
        with self.assertRaises(ci.HistoryCIError):
            ci.validate_counts({**valid_counts(14, 2), "raw_text": "PRIVATE"}, 14, 2)

    def test_suite_set_cannot_change(self):
        results = valid_report()["tests"]
        results.pop(ci.SUITES[0][0])
        with self.assertRaises(ci.HistoryCIError):
            ci.make_report(results)

    def test_report_cannot_claim_all_passed_or_real_model_readiness(self):
        for update in (
            {"all_tests_passed": True}, {"tests_passed": 31}, {"real_model_import_tested": True},
            {"expected_historical_failures": False}, {"raw_text": "PRIVATE"},
        ):
            with self.subTest(update=update), self.assertRaises(ci.HistoryCIError):
                ci.validate_report(json.dumps({**valid_report(), **update}).encode())

    def test_invalid_duplicate_or_oversized_report_fails(self):
        for raw in (b"", b"PRIVATE", b'{"tests":{},"tests":{}}',
                    b"x" * (ci.MAXIMUM_REPORT_BYTES + 1)):
            with self.subTest(size=len(raw)), self.assertRaises(ci.HistoryCIError):
                ci.validate_report(raw)

    def test_bindings_validate_synthetic_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fixture.py").write_bytes(b"synthetic")
            bindings = {"fixture.py": hashlib.sha256(b"synthetic").hexdigest()}
            with patch.object(ci, "BINDINGS", bindings):
                ci.verify_bindings(root)

    def test_atime_change_does_not_masquerade_as_source_mutation(self):
        before = SimpleNamespace(
            st_dev=1, st_ino=2, st_mode=3, st_nlink=1, st_size=9,
            st_mtime_ns=10, st_ctime_ns=11, st_atime_ns=12,
        )
        after = SimpleNamespace(**{**vars(before), "st_atime_ns": 13})
        self.assertEqual(ci.file_identity(before), ci.file_identity(after))
        for field in vars(before).keys() - {"st_atime_ns"}:
            altered = SimpleNamespace(**{**vars(before), field: getattr(before, field) + 1})
            self.assertNotEqual(ci.file_identity(before), ci.file_identity(altered))

    def test_missing_or_changed_binding_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(ci, "BINDINGS", {"fixture.py": "a" * 64}):
                with self.assertRaisesRegex(ci.HistoryCIError, "bound_file_unavailable"):
                    ci.verify_bindings(root)
                (root / "fixture.py").write_bytes(b"synthetic")
                with self.assertRaisesRegex(ci.HistoryCIError, "bound_file_hash_changed"):
                    ci.verify_bindings(root)

    def test_symlink_or_nonregular_binding_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "folder").mkdir()
            (root / "link.py").symlink_to(root / "folder")
            for path in ("folder", "link.py"):
                with patch.object(ci, "BINDINGS", {path: "a" * 64}):
                    with self.assertRaises(ci.HistoryCIError):
                        ci.verify_bindings(root)

    def test_network_audit_rejects_only_fixed_network_events(self):
        for event in ci.NETWORK_EVENTS:
            with (
                self.subTest(event=event),
                self.assertRaisesRegex(ci.HistoryCIError, "network_forbidden"),
            ):
                ci.deny_network(event, ("PRIVATE",))
        ci.deny_network("open", ("synthetic",))

    def test_child_environment_does_not_inherit_credentials_or_gpu_visibility(self):
        inherited = {"AWS_SESSION_TOKEN": "PRIVATE", "CUDA_VISIBLE_DEVICES": "0"}
        with patch.dict(ci.os.environ, inherited):
            environment = ci.child_environment(Path("/synthetic"))
        self.assertNotIn("AWS_SESSION_TOKEN", environment)
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
        self.assertEqual(environment["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(environment["PYTHONPATH"], "/synthetic/src")

    def test_child_is_fixed_fresh_bounded_and_verifies_bindings_twice(self):
        response = subprocess.CompletedProcess([], 0, json.dumps(valid_report()).encode())
        with (
            patch.object(ci, "require_startup"),
            patch.object(ci, "verify_bindings") as verify,
            patch.object(ci.subprocess, "run", return_value=response) as run,
        ):
            self.assertEqual(ci.run_fresh_process(Path("/synthetic")), valid_report())
        self.assertEqual(verify.call_count, 2)
        self.assertEqual(run.call_args.args[0][1], "-B")
        self.assertNotIn("-S", run.call_args.args[0])
        self.assertEqual(run.call_args.args[0][-1], ci.CHILD_MODE)
        self.assertEqual(run.call_args.kwargs["timeout"], 180)
        self.assertIs(run.call_args.kwargs["shell"], False)

    def test_child_preserves_symlinked_venv_interpreter_for_ml_dependencies(self):
        response = subprocess.CompletedProcess([], 0, json.dumps(valid_report()).encode())
        with tempfile.TemporaryDirectory() as directory:
            launcher = Path(directory) / "venv-python"
            launcher.symlink_to(ci.sys.executable)
            with (
                patch.object(ci.sys, "executable", str(launcher)),
                patch.object(ci, "require_startup"),
                patch.object(ci, "verify_bindings"),
                patch.object(ci.subprocess, "run", return_value=response) as run,
            ):
                ci.run_fresh_process(Path("/synthetic"))
            self.assertEqual(run.call_args.args[0][0], str(launcher))
            self.assertNotEqual(run.call_args.args[0][0], str(launcher.resolve()))

    def test_child_error_timeout_and_nonzero_exit_are_sanitized(self):
        for error, code in (
            (subprocess.TimeoutExpired(["PRIVATE"], 180), "history_suite_timeout"),
            (OSError("PRIVATE"), "history_child_start_failed"),
        ):
            with (
                self.subTest(code=code), patch.object(ci, "require_startup"),
                patch.object(ci, "verify_bindings"),
                patch.object(ci.subprocess, "run", side_effect=error),
                self.assertRaisesRegex(ci.HistoryCIError, "^" + code + "$"),
            ):
                ci.run_fresh_process(Path("/synthetic"))
        with (
            patch.object(ci, "require_startup"), patch.object(ci, "verify_bindings"),
            patch.object(ci.subprocess, "run", return_value=(
                subprocess.CompletedProcess([], 1, b"PRIVATE")
            )),
            self.assertRaisesRegex(ci.HistoryCIError, "history_child_failed"),
        ):
            ci.run_fresh_process(Path("/synthetic"))

    def test_list_only_never_imports_suites_or_spawns(self):
        with (
            patch.object(ci, "require_startup"),
            patch.object(ci, "execute_suites", side_effect=AssertionError("must not run")),
            patch.object(ci, "run_fresh_process", side_effect=AssertionError("must not spawn")),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(ci.main(["--list-json"]), 0)
        self.assertEqual(json.loads(output.getvalue()), [item[0] for item in ci.SUITES])

    def test_arbitrary_arguments_and_private_errors_never_echo(self):
        with (
            patch.object(ci, "require_startup"),
            patch.object(ci, "run_fresh_process", side_effect=AssertionError("PRIVATE")),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(ci.main(["--module", "PRIVATE"]), 1)
            self.assertEqual(ci.main([]), 1)
        self.assertNotIn("PRIVATE", output.getvalue())

    def test_startup_rejects_missing_bytecode_flag(self):
        with patch.object(ci.sys, "dont_write_bytecode", False):
            with self.assertRaisesRegex(ci.HistoryCIError, "use_python_minus_B"):
                ci.require_startup()


if __name__ == "__main__":
    unittest.main()
