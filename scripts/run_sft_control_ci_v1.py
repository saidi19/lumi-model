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
import stat
import subprocess
import sys
import tempfile
import types
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
WATCHDOG_SUITE = "tests/test_process_watchdog_v1.py"
MAXIMUM_WATCHDOG_TRACEBACK_FRAMES = 32
MAXIMUM_WATCHDOG_CODES = 64
MAXIMUM_WATCHDOG_RESULT_FIELDS = 32
MAXIMUM_WATCHDOG_ELAPSED_MILLISECONDS = 1_000_000
WATCHDOG_REASONS = frozenset({
    "child_completed", "child_failed", "parent_interrupted", "deadline_exhausted",
    "deadline_exhausted_before_spawn", "spawn_failed", "stream_budget_exhausted",
    "guardian_failed", "supervisor_internal_error", "cleanup_failed",
})
WATCHDOG_BOOLEAN_FIELDS = frozenset({
    "child_spawned", "worker_status_reported", "group_term_attempted",
    "group_kill_attempted", "direct_child_reaped",
})
WATCHDOG_OBSERVATION_FIELDS = WATCHDOG_BOOLEAN_FIELDS | {
    "reason", "exit_code", "guardian_exit_code", "elapsed_milliseconds",
}
CHILD_MODE = "--fixed-suite-child"
SCHEMA = "lumi.sft-control-ci.v0.1"
FORBIDDEN_MODULES = ("site", "torch", "transformers", "tokenizers", "safetensors", "numpy")
WATCHDOG_TMPFS = "/dev/shm"
MINIMUM_TMPFS_BYTES = 16 * 1024 * 1024
MINIMUM_TMPFS_INODES = 64
MAXIMUM_MOUNTINFO_BYTES = 262_144
WATCHDOG_STORAGE_FAILURE_CODES = frozenset({
    "watchdog_tmpfs_mount_invalid", "watchdog_tmpfs_root_invalid",
    "watchdog_tmpfs_capacity_insufficient", "watchdog_tmpfs_private_directory_invalid",
    "watchdog_tmpfs_unavailable",
})


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


def valid_watchdog_observation(value: object) -> bool:
    """Only fixed reason codes and bounded primitive lifecycle observations."""
    return (
        type(value) is dict
        and set(value) == WATCHDOG_OBSERVATION_FIELDS
        and type(value["reason"]) is str
        and len(value["reason"]) <= 64
        and value["reason"] in WATCHDOG_REASONS
        and all(type(value[name]) is bool for name in WATCHDOG_BOOLEAN_FIELDS)
        and all(
            value[name] is None or type(value[name]) is int and -255 <= value[name] <= 255
            for name in ("exit_code", "guardian_exit_code")
        )
        and type(value["elapsed_milliseconds"]) is int
        and 0 <= value["elapsed_milliseconds"] <= MAXIMUM_WATCHDOG_ELAPSED_MILLISECONDS
    )


def watchdog_observation(err: tuple, allowed_codes: tuple) -> dict | None:
    """Inspect bounded exact-code frames; never format errors or read result files."""
    try:
        if type(err) is not tuple or len(err) != 3:
            return None
        traceback = err[2]
        for _ in range(MAXIMUM_WATCHDOG_TRACEBACK_FRAMES):
            if type(traceback) is not types.TracebackType:
                return None
            if any(traceback.tb_frame.f_code is code for code in allowed_codes):
                value = traceback.tb_frame.f_locals.get("result")
                if type(value) is not dict or len(value) > MAXIMUM_WATCHDOG_RESULT_FIELDS:
                    return None
                observation = {name: value[name] for name in WATCHDOG_OBSERVATION_FIELDS}
                return observation if valid_watchdog_observation(observation) else None
            traceback = traceback.tb_next
    except BaseException:
        # Optional diagnostics must not interfere with recording the real failure.
        pass
    return None


class AggregateResult(unittest.TestResult):
    """Keep counters and allowlisted observations, never traceback text or test IDs."""

    def __init__(self, watchdog_codes: tuple = ()):
        super().__init__()
        self.diagnostics = []
        self.truncated = False
        self.watchdog_codes = (
            watchdog_codes
            if type(watchdog_codes) is tuple
            and len(watchdog_codes) <= MAXIMUM_WATCHDOG_CODES
            and all(type(code) is types.CodeType for code in watchdog_codes)
            else ()
        )

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
        case = {"test_index": self.testsRun, "kind": kind, "category": category}
        self.diagnostics.append(case)
        if self.watchdog_codes and kind in {"failure", "subtest_failure"}:
            try:
                observation = watchdog_observation(err, self.watchdog_codes)
                if observation is not None:
                    case["watchdog"] = observation
            except BaseException:
                pass

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
                type(case) is dict
                and set(case) in (
                    {"test_index", "kind", "category"},
                    {"test_index", "kind", "category", "watchdog"},
                ),
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
            if "watchdog" in case:
                require(
                    detail["suite"] == WATCHDOG_SUITE
                    and case["kind"] in {"failure", "subtest_failure"}
                    and valid_watchdog_observation(case["watchdog"]),
                    "child_failure_invalid",
                )
            observed[kind_counter[case["kind"]]] += 1
        require(all(observed[key] <= counts[key] for key in observed), "child_failure_invalid")
        return detail
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError):
        raise ControlCIError("child_failure_invalid") from None


def validate_tmpfs_mountinfo(raw: bytes, device: int) -> None:
    """Require the kernel-reported tmpfs at the fixed synthetic-storage mount."""
    require(0 < len(raw) <= MAXIMUM_MOUNTINFO_BYTES, "watchdog_tmpfs_mount_invalid")
    identity = f"{os.major(device)}:{os.minor(device)}".encode("ascii")
    matches = []
    for line in raw.splitlines():
        before, separator, after = line.partition(b" - ")
        fields, filesystem = before.split(), after.split()
        if len(fields) >= 6 and fields[4] == WATCHDOG_TMPFS.encode("ascii"):
            matches.append(
                bool(separator) and bool(filesystem)
                and fields[2] == identity and filesystem[0] == b"tmpfs"
            )
    require(matches == [True], "watchdog_tmpfs_mount_invalid")


def require_watchdog_tmpfs() -> int:
    """Inspect, never mount or resize, Linux's existing shared-memory filesystem."""
    fd = os.open(WATCHDOG_TMPFS, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        info = os.fstat(fd)
        require(
            stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o1777
            and info.st_uid == 0,
            "watchdog_tmpfs_root_invalid",
        )
        with open("/proc/self/mountinfo", "rb") as stream:
            validate_tmpfs_mountinfo(stream.read(MAXIMUM_MOUNTINFO_BYTES + 1), info.st_dev)
        space = os.fstatvfs(fd)
        require(
            space.f_bavail * space.f_frsize >= MINIMUM_TMPFS_BYTES
            and space.f_favail >= MINIMUM_TMPFS_INODES,
            "watchdog_tmpfs_capacity_insufficient",
        )
        return info.st_dev
    finally:
        os.close(fd)


@contextlib.contextmanager
def watchdog_test_storage():
    """Only the synthetic watchdog suite uses private tmpfs on Linux.

    Real fsync calls, source bytes, assertions and watchdog deadlines stay intact.
    This removes disk-flush contention from short process-control fixtures, not
    from production execution. It does not test crash-persistent storage.
    """
    if sys.platform != "linux":
        yield
        return
    try:
        device = require_watchdog_tmpfs()
        with tempfile.TemporaryDirectory(
            prefix="lumi-watchdog-ci-", dir=WATCHDOG_TMPFS
        ) as directory:
            info = os.stat(directory, follow_symlinks=False)
            require(
                stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o700
                and info.st_uid == os.geteuid() and info.st_dev == device,
                "watchdog_tmpfs_private_directory_invalid",
            )
            previous = tempfile.tempdir
            tempfile.tempdir = directory
            try:
                yield
            finally:
                tempfile.tempdir = previous
    except OSError:
        raise ControlCIError("watchdog_tmpfs_unavailable") from None


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
            result = AggregateResult(
                tuple(
                    getattr(module.WatchdogTests, name).__code__
                    for name in unittest.defaultTestLoader.getTestCaseNames(module.WatchdogTests)
                ) if relative == WATCHDOG_SUITE else ()
            )
            storage = (
                watchdog_test_storage() if relative == WATCHDOG_SUITE else contextlib.nullcontext()
            )
            with storage:
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


def storage_failure_code(raw: bytes) -> str | None:
    """Forward only fixed storage-admission STOP codes, never child output text."""
    if type(raw) is not bytes or not 0 < len(raw) <= MAXIMUM_REPORT_BYTES:
        return None

    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "child_failure_invalid")
            value[key] = item
        return value

    try:
        value = json.loads(raw, object_pairs_hook=unique)
        if (
            type(value) is dict and set(value) == {"schema_version", "status", "failure_code"}
            and value["schema_version"] == SCHEMA and value["status"] == "STOP"
            and type(value["failure_code"]) is str
            and value["failure_code"] in WATCHDOG_STORAGE_FAILURE_CODES
        ):
            return value["failure_code"]
    except (ValueError, TypeError, UnicodeError, RecursionError):
        pass
    return None


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
        storage_code = storage_failure_code(completed.stdout)
        if storage_code is not None:
            raise ControlCIError(storage_code)
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
