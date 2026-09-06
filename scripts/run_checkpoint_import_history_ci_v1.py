#!/usr/bin/env python3
"""Reproduce two frozen v1 BF16 failures and require the v2 successor to pass.

This is an explicit history contract, not an all-tests-passed claim or a generic
xfail route. Only the two pinned, tiny CPU suites run. They use temporary,
synthetic tensors and prohibit model forward calls; no real model import or SFT
is authorized. Offline settings and the Python network audit guard are defense
in depth, not an OS sandbox. Existing source and tests remain byte-for-byte intact.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
import unittest
from pathlib import Path

SCHEMA = "lumi.checkpoint-import-history-ci.v0.1"
CHILD_MODE = "--fixed-history-child"
TIMEOUT_SECONDS = 180
MAXIMUM_REPORT_BYTES = 16_384
SUITES = (
    ("tests/test_sft_checkpoint_import_torch_v1.py", "checkpoint_history_v1", 14, 2),
    ("tests/test_sft_checkpoint_import_torch_v2.py", "checkpoint_history_v2", 17, 0),
)
BINDINGS = {
    "src/lumi/posttraining_checkpoint_import_v1.py":
        "65963fc4110267ec42ecad1ad114d6ca927664e93318fc026d0af06bd1ec1063",
    "src/lumi/posttraining_checkpoint_import_v2.py":
        "8a395f26c9b16d29a39d4fbf9877dd6836d8e7eb1640fa0c9d24ec42f46972d6",
    "src/lumi/posttraining_checkpoint_import_torch_v1.py":
        "776e438b3d8b06740b66f9a8203f7a9b9ab1ce6e90e785e2304be6228c836065",
    "src/lumi/posttraining_checkpoint_import_torch_v2.py":
        "01e301ed152a6e81ae9ef6acdb25b71a3e3778148114228591051c2e25e0ab6f",
    "tests/test_sft_checkpoint_import_torch_v1.py":
        "72f236b91138c6dd5940be0ce69d003dcf5f75e19be233f89a9d10db35fa6551",
    "tests/test_sft_checkpoint_import_torch_v2.py":
        "407908e5a7041d57acee1684ced026d619466825f9bc319fa7a7f71f3e5dff57",
}
EXPECTED_HISTORY = frozenset({
    "checkpoint_history_v1.CheckpointImportTorchTests.test_real_bfloat16_roundtrip",
    "checkpoint_history_v1.CheckpointImportTorchTests."
    "test_real_adapter_bfloat16_and_worker_consumer",
})
HISTORY_ORIGIN = "src/lumi/posttraining_checkpoint_import_torch_v1.py"
HISTORY_MESSAGE = "HF reload tensor mismatch"
COUNTER_FIELDS = frozenset({
    "run", "passed", "historical_failures", "failures", "errors", "skipped",
    "declared_expected_failures", "unexpected_successes", "historical_unexpected_successes",
})
NETWORK_EVENTS = frozenset({
    "socket.connect", "socket.bind", "socket.getaddrinfo", "socket.gethostbyaddr",
    "socket.gethostbyname", "socket.sendto", "socket.sendmsg",
})


class HistoryCIError(ValueError):
    """Only fixed codes are allowed in user-visible failures."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise HistoryCIError(code) from None


def require_startup() -> None:
    require(sys.dont_write_bytecode, "use_python_minus_B")
    require("lumi" not in sys.modules, "fresh_lumi_import_required")


def file_identity(value) -> tuple:
    # Reading source may legitimately change atime; mutation fields must not change.
    return (
        value.st_dev, value.st_ino, value.st_mode, value.st_nlink, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns,
    )


def verify_bindings(root: Path) -> None:
    for relative, digest in BINDINGS.items():
        path = root
        for part in Path(relative).parts:
            path = path / part
            require(not path.is_symlink(), "bound_path_symlink")
        try:
            before = path.stat()
            require(stat.S_ISREG(before.st_mode), "bound_file_type")
            require(0 < before.st_size <= 1_048_576, "bound_file_size")
            with path.open("rb") as stream:
                require(
                    file_identity(os.fstat(stream.fileno())) == file_identity(before),
                    "bound_file_changed",
                )
                raw = stream.read(1_048_577)
            require(file_identity(path.stat()) == file_identity(before), "bound_file_changed")
        except OSError:
            raise HistoryCIError("bound_file_unavailable") from None
        require(hashlib.sha256(raw).hexdigest() == digest, "bound_file_hash_changed")


def deny_network(event: str, arguments: tuple) -> None:
    if event in NETWORK_EVENTS:
        raise HistoryCIError("network_forbidden") from None


class DiscardText:
    def write(self, value: str) -> int:
        return len(value)

    def flush(self) -> None:
        pass


@contextlib.contextmanager
def silence_test_output():
    """Suppress Python and native FD output; only the final aggregate is printed."""
    saved = (os.dup(1), os.dup(2))
    try:
        with open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            with (
                contextlib.redirect_stdout(DiscardText()),
                contextlib.redirect_stderr(DiscardText()),
            ):
                yield
    finally:
        for destination, descriptor in zip((1, 2), saved, strict=True):
            os.dup2(descriptor, destination)
            os.close(descriptor)


def is_historical_error(err, root: Path) -> bool:
    if err[0] is not ValueError or type(err[1]) is not ValueError:
        return False
    if err[1].args != (HISTORY_MESSAGE,):
        return False
    trace = err[2]
    while trace is not None and trace.tb_next is not None:
        trace = trace.tb_next
    return bool(
        trace is not None
        and trace.tb_frame.f_code.co_name == "convert_and_verify"
        and Path(trace.tb_frame.f_code.co_filename) == root / HISTORY_ORIGIN
        and trace.tb_lineno == 212
    )


class HistoryResult(unittest.TestResult):
    """Retain counters and two allowlisted IDs, never tracebacks or raw messages."""

    def __init__(self, root: Path):
        super().__init__()
        self.root = root
        self.passed = 0
        self.history_seen = set()
        self.history_successes = 0

    def addSuccess(self, test) -> None:
        if test.id() in EXPECTED_HISTORY:
            self.history_successes += 1
        else:
            self.passed += 1

    def addError(self, test, err) -> None:
        identifier = test.id()
        if (
            identifier in EXPECTED_HISTORY
            and identifier not in self.history_seen
            and is_historical_error(err, self.root)
        ):
            self.history_seen.add(identifier)
        else:
            self.errors.append(None)

    def addFailure(self, test, err) -> None:
        self.failures.append(None)

    def addSkip(self, test, reason) -> None:
        self.skipped.append(None)

    def addExpectedFailure(self, test, err) -> None:
        self.expectedFailures.append(None)

    def addUnexpectedSuccess(self, test) -> None:
        self.unexpectedSuccesses.append(None)

    def addSubTest(self, test, subtest, err) -> None:
        if err is not None:
            self.errors.append(None)


def counters(result: HistoryResult) -> dict:
    return {
        "run": result.testsRun,
        "passed": result.passed,
        "historical_failures": len(result.history_seen),
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "declared_expected_failures": len(result.expectedFailures),
        "unexpected_successes": len(result.unexpectedSuccesses),
        "historical_unexpected_successes": result.history_successes,
    }


def validate_counts(values: dict, expected: int, historical: int) -> None:
    require(type(values) is dict and set(values) == COUNTER_FIELDS, "test_counter_schema")
    require(
        all(type(value) is int and value >= 0 for value in values.values()), "test_counter_type"
    )
    require(values["run"] == expected, "test_count_changed")
    require(values["passed"] == expected - historical, "passing_test_count_changed")
    require(values["historical_failures"] == historical, "historical_reproduction_count_changed")
    require(
        all(value == 0 for key, value in values.items()
            if key not in {"run", "passed", "historical_failures"}),
        "unexpected_test_outcome",
    )


def make_report(results: dict) -> dict:
    require(
        type(results) is dict and set(results) == {item[0] for item in SUITES}, "suite_set_changed"
    )
    for relative, _, expected, historical in SUITES:
        validate_counts(results[relative], expected, historical)
    return {
        "schema_version": SCHEMA,
        "status": "PASS_explicit_history_and_successor_contract",
        "tests": results,
        "tests_run": 31,
        "tests_passed": 29,
        "expected_historical_failures": 2,
        "all_tests_passed": False,
        "v1_bfloat16_contract_passed": False,
        "v2_successor_tests_passed": 17,
        "bound_source_and_test_files": len(BINDINGS),
        "gpu_execution_authorized": False,
        "real_model_import_tested": False,
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
        raise HistoryCIError("child_report_invalid_json") from None
    require(type(report) is dict and "tests" in report, "child_report_schema")
    expected = make_report(report["tests"])
    require(
        json.dumps(report, sort_keys=True, allow_nan=False)
        == json.dumps(expected, sort_keys=True, allow_nan=False),
        "child_report_schema",
    )
    return expected


def execute_suites(root: Path) -> dict:
    require_startup()
    verify_bindings(root)
    sys.addaudithook(deny_network)
    results = {}
    with silence_test_output():
        for relative, module_name, expected, historical in SUITES:
            spec = importlib.util.spec_from_file_location(module_name, root / relative)
            require(spec is not None and spec.loader is not None, "test_module_unavailable")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            suite = unittest.defaultTestLoader.loadTestsFromModule(module)
            require(suite.countTestCases() == expected, "test_discovery_count_changed")
            result = HistoryResult(root)
            suite.run(result)
            results[relative] = counters(result)
            validate_counts(results[relative], expected, historical)
    verify_bindings(root)
    return make_report(results)


def child_environment(root: Path) -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "PYTHONPATH": str(root / "src"),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }


def run_fresh_process(root: Path) -> dict:
    require_startup()
    verify_bindings(root)
    try:
        completed = subprocess.run(
            [str(Path(sys.executable).absolute()), "-B", str(Path(__file__).resolve()), CHILD_MODE],
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
        raise HistoryCIError("history_suite_timeout") from None
    except OSError:
        raise HistoryCIError("history_child_start_failed") from None
    require(completed.returncode == 0, "history_child_failed")
    report = validate_report(completed.stdout)
    verify_bindings(root)
    return report


def main(argv: list[str] | None = None) -> int:
    try:
        require_startup()
        arguments = sys.argv[1:] if argv is None else argv
        require(arguments in ([], ["--list-json"], [CHILD_MODE]), "unexpected_arguments")
        if arguments == ["--list-json"]:
            print(json.dumps([item[0] for item in SUITES]))
            return 0
        root = Path(__file__).resolve().parents[1]
        report = execute_suites(root) if arguments == [CHILD_MODE] else run_fresh_process(root)
        print(json.dumps(report, sort_keys=True))
        return 0
    except HistoryCIError as error:
        code = str(error)
    except BaseException:
        code = "history_ci_internal_error"
    print(json.dumps({"schema_version": SCHEMA, "status": "STOP", "failure_code": code}))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
