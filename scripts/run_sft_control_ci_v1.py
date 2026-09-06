#!/usr/bin/env python3
"""Run the fixed synthetic SFT control suites in a fresh stdlib-only process.

No private plans, evidence publishers, model inputs or GPU workers are selected.
The existing watchdog tests create real, bounded synthetic child processes.
This runner is a CI test route, not an OS sandbox or permission to execute SFT.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

SUITES = (
    ("tests/test_sft_source_bundle_v1.py", 26),
    ("tests/test_sft_supervisor_v1.py", 51),
    ("tests/test_sft_worker_admission_v1.py", 36),
    ("tests/test_sft_launch_contract_v1.py", 54),
    ("tests/test_process_watchdog_v1.py", 33),
    ("tests/test_sft_guarded_bootstrap_v1.py", 30),
    ("tests/test_sft_ledger_protocol_v1.py", 43),
)
EXPECTED_TESTS = 273
TIMEOUT_SECONDS = 120
MAXIMUM_REPORT_BYTES = 16_384
MAXIMUM_FAILURE_CASES = 32
CHILD_MODE = "--fixed-suite-child"
SCHEMA = "lumi.sft-control-ci.v0.1"
FORBIDDEN_MODULES = ("site", "torch", "transformers", "tokenizers", "safetensors", "numpy")


class ControlCIError(ValueError):
    """Internal callers provide fixed codes, never exception or source text."""

    def __init__(self, code: str, diagnostic: dict | None = None):
        super().__init__(code)
        self.diagnostic = diagnostic


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ControlCIError(code) from None


def require_startup() -> None:
    require(sys.flags.no_site == 1 and sys.dont_write_bytecode, "use_python_minus_S_minus_B")
    require(
        not any(
            name == prefix or name.startswith(prefix + ".")
            for name in sys.modules
            for prefix in FORBIDDEN_MODULES
        ),
        "stdlib_only_startup_required",
    )


class DiscardText:
    def write(self, value: str) -> int:
        return len(value)

    def flush(self) -> None:
        pass


class AggregateResult(unittest.TestResult):
    """Keep fixed counters instead of traceback strings, test IDs or reasons."""

    def __init__(self):
        super().__init__()
        self.diagnostics = []
        self.truncated = False

    def _record(self, kind, err=None):
        if len(self.diagnostics) >= MAXIMUM_FAILURE_CASES:
            self.truncated = True
            return
        category = "none"
        if err is not None:
            category = "other"
            for expected, value in (
                (AssertionError, "assertion"),
                ((TimeoutError, subprocess.TimeoutExpired), "timeout"),
                (OSError, "os_error"),
            ):
                if issubclass(err[0], expected):
                    category = value
                    break
        self.diagnostics.append({"test_index": self.testsRun, "kind": kind, "category": category})

    def addFailure(self, test, err) -> None:
        self.failures.append(None)
        self._record("failure", err)

    def addError(self, test, err) -> None:
        self.errors.append(None)
        self._record("error", err)

    def addSkip(self, test, reason) -> None:
        self.skipped.append(None)
        self._record("skip")

    def addExpectedFailure(self, test, err) -> None:
        self.expectedFailures.append(None)
        self._record("expected_failure", err)

    def addUnexpectedSuccess(self, test) -> None:
        self.unexpectedSuccesses.append(None)
        self._record("unexpected_success")

    def addSubTest(self, test, subtest, err) -> None:
        if err is not None:
            target = self.failures if issubclass(err[0], test.failureException) else self.errors
            target.append(None)
            self._record("subtest_failure" if target is self.failures else "subtest_error", err)


def counters(result: unittest.TestResult) -> dict:
    return {
        "run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "expected_failures": len(result.expectedFailures),
        "unexpected_successes": len(result.unexpectedSuccesses),
    }


def validate_counts(values: dict, expected: int) -> None:
    fields = {
        "run",
        "failures",
        "errors",
        "skipped",
        "expected_failures",
        "unexpected_successes",
    }
    require(type(values) is dict and set(values) == fields, "test_counter_schema")
    require(
        all(type(value) is int and value >= 0 for value in values.values()), "test_counter_type"
    )
    require(values["run"] == expected, "test_count_changed")
    require(
        all(value == 0 for key, value in values.items() if key != "run"), "control_tests_failed"
    )


def make_report(results: dict) -> dict:
    require(
        type(results) is dict and set(results) == {name for name, _ in SUITES}, "suite_set_changed"
    )
    for name, expected in SUITES:
        validate_counts(results[name], expected)
    total = sum(value["run"] for value in results.values())
    require(total == EXPECTED_TESTS, "total_test_count_changed")
    return {
        "schema_version": SCHEMA,
        "status": "PASS_fixed_synthetic_control_suites",
        "tests": results,
        "tests_run": total,
        "suite_count": len(SUITES),
        "gpu_execution_authorized": False,
        "private_artifact_integration_tested": False,
        "retained_test_output_bytes": 0,
    }


def validate_report(raw: bytes) -> dict:
    require(type(raw) is bytes and 0 < len(raw) <= MAXIMUM_REPORT_BYTES, "child_report_size")

    def unique(pairs):
        output = {}
        for key, value in pairs:
            require(key not in output, "child_report_duplicate_key")
            output[key] = value
        return output

    try:
        report = json.loads(raw, object_pairs_hook=unique)
    except (ValueError, UnicodeError, RecursionError):
        raise ControlCIError("child_report_invalid_json") from None
    require(type(report) is dict and "tests" in report, "child_report_schema")
    expected = make_report(report["tests"])
    require(
        json.dumps(report, sort_keys=True, allow_nan=False)
        == json.dumps(expected, sort_keys=True, allow_nan=False),
        "child_report_schema",
    )
    return expected


def validate_failure(raw: bytes) -> dict:
    """Validate only allowlisted aggregate failure details; never return raw output."""
    require(type(raw) is bytes and 0 < len(raw) <= MAXIMUM_REPORT_BYTES, "child_failure_invalid")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "child_failure_invalid")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=unique)
        require(
            type(value) is dict
            and set(value)
            == {
                "schema_version",
                "status",
                "failure_code",
                "diagnostic",
            },
            "child_failure_invalid",
        )
        require(
            value["schema_version"] == SCHEMA
            and value["status"] == "STOP"
            and value["failure_code"] in {"control_tests_failed", "test_count_changed"},
            "child_failure_invalid",
        )
        detail = value["diagnostic"]
        require(
            type(detail) is dict
            and set(detail)
            == {
                "suite",
                "expected",
                "counters",
                "cases",
                "truncated",
            },
            "child_failure_invalid",
        )
        suites = dict(SUITES)
        require(type(detail["suite"]) is str and detail["suite"] in suites, "child_failure_invalid")
        expected = suites[detail["suite"]]
        require(
            type(detail["expected"]) is int and detail["expected"] == expected,
            "child_failure_invalid",
        )
        counts = detail["counters"]
        require(
            type(counts) is dict
            and set(counts)
            == {
                "run",
                "failures",
                "errors",
                "skipped",
                "expected_failures",
                "unexpected_successes",
            },
            "child_failure_invalid",
        )
        require(
            all(type(n) is int and 0 <= n <= 1_000_000 for n in counts.values())
            and counts["run"] <= expected,
            "child_failure_invalid",
        )
        require(
            counts["run"] != expected or any(n for key, n in counts.items() if key != "run"),
            "child_failure_invalid",
        )
        require(
            type(detail["cases"]) is list
            and len(detail["cases"]) <= MAXIMUM_FAILURE_CASES
            and type(detail["truncated"]) is bool,
            "child_failure_invalid",
        )
        total_failures = sum(n for key, n in counts.items() if key != "run")
        require(
            len(detail["cases"]) == min(total_failures, MAXIMUM_FAILURE_CASES)
            and detail["truncated"] is (total_failures > MAXIMUM_FAILURE_CASES),
            "child_failure_invalid",
        )
        observed = {key: 0 for key in counts if key != "run"}
        kind_counter = {
            "failure": "failures",
            "subtest_failure": "failures",
            "error": "errors",
            "subtest_error": "errors",
            "skip": "skipped",
            "expected_failure": "expected_failures",
            "unexpected_success": "unexpected_successes",
        }
        for case in detail["cases"]:
            require(
                type(case) is dict and set(case) == {"test_index", "kind", "category"},
                "child_failure_invalid",
            )
            require(
                type(case["test_index"]) is int and 0 <= case["test_index"] <= counts["run"],
                "child_failure_invalid",
            )
            require(
                case["kind"]
                in {
                    "failure",
                    "error",
                    "skip",
                    "expected_failure",
                    "unexpected_success",
                    "subtest_failure",
                    "subtest_error",
                }
                and case["category"] in {"assertion", "os_error", "timeout", "other", "none"},
                "child_failure_invalid",
            )
            observed[kind_counter[case["kind"]]] += 1
        require(all(observed[key] <= counts[key] for key in observed), "child_failure_invalid")
        return detail
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError):
        raise ControlCIError("child_failure_invalid") from None


def execute_suites(root: Path) -> dict:
    """Child-only fixed selection. The parent supplies the 120-second deadline."""
    require_startup()
    results = {}
    sink = DiscardText()
    for relative, expected in SUITES:
        module_name = "sft_control_ci_" + Path(relative).stem
        spec = importlib.util.spec_from_file_location(module_name, root / relative)
        require(spec is not None and spec.loader is not None, "test_module_unavailable")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            spec.loader.exec_module(module)
            suite = unittest.defaultTestLoader.loadTestsFromModule(module)
            require(suite.countTestCases() == expected, "test_discovery_count_changed")
            result = AggregateResult()
            suite.run(result)
        results[relative] = counters(result)
        try:
            validate_counts(results[relative], expected)
        except ControlCIError as error:
            raise ControlCIError(
                str(error),
                {
                    "suite": relative,
                    "expected": expected,
                    "counters": results[relative],
                    "cases": result.diagnostics,
                    "truncated": result.truncated,
                },
            ) from None
    require_startup()
    return make_report(results)


def child_environment(root: Path) -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "PYTHONPATH": os.pathsep.join(str(root / part) for part in ("src", "tests")),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }


def require_parent_startup() -> None:
    require_startup()
    require("lumi" not in sys.modules, "parent_must_not_import_lumi")


def run_fresh_process(root: Path) -> dict:
    require_parent_startup()
    try:
        completed = subprocess.run(
            [
                str(Path(sys.executable).resolve()),
                "-S",
                "-B",
                str(Path(__file__).resolve()),
                CHILD_MODE,
            ],
            cwd=root,
            env=child_environment(root),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=TIMEOUT_SECONDS,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired:
        raise ControlCIError("control_suite_timeout") from None
    except OSError:
        raise ControlCIError("control_child_start_failed") from None
    if completed.returncode != 0:
        diagnostic = None
        try:
            diagnostic = validate_failure(completed.stdout)
        except ControlCIError:
            pass
        raise ControlCIError("control_child_failed", diagnostic)
    return validate_report(completed.stdout)


def main(argv: list[str] | None = None) -> int:
    diagnostic = None
    try:
        require_startup()
        arguments = sys.argv[1:] if argv is None else argv
        require(arguments in ([], ["--list-json"], [CHILD_MODE]), "unexpected_arguments")
        if arguments == ["--list-json"]:
            print(json.dumps([name for name, _ in SUITES]))
            return 0
        root = Path(__file__).resolve().parents[1]
        report = execute_suites(root) if arguments == [CHILD_MODE] else run_fresh_process(root)
        print(json.dumps(report, sort_keys=True))
        return 0
    except ControlCIError as error:
        code = str(error)
        diagnostic = error.diagnostic
    except BaseException:
        code = "control_ci_internal_error"
    output = {"schema_version": SCHEMA, "status": "STOP", "failure_code": code}
    if diagnostic is not None:
        output["diagnostic"] = diagnostic
    print(json.dumps(output))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
