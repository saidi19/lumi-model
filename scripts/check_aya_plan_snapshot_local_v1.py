"""One original Aya plan test in an authenticated, read-only source projection.

Run with the existing trusted Python using -S -B. Runtime self-check uses only a
synthetic assertion. The real mode hashes exact bindings, never follows private
inventory paths, and never imports the complete curation or model stack.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import os
import signal
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "configs/lumi-aya-plan-snapshot-inputs-v0.1.json"
SPEC_SHA256 = "a519b0e427077db6c39d0f94409e0def5965cdb85335d13fbece3d5ff81a020c"
IO_PATH = ROOT / "scripts/check_safety_output_replay_local_v1.py"
IO_SHA256 = "58112cb86ff50dcb494c820934b183a1beb2b903c7612c58d52032d48841e8e9"
TEST_PATH = ROOT / "tests/test_aya_bengali_repair_preflight.py"
TEST_NAME = "test_aya_repair_plan_is_exact_local_and_non_authorizing"
MODULE_NAME = "test_aya_bengali_repair_preflight"
NODEID = "tests/test_aya_bengali_repair_preflight.py::" + TEST_NAME
MAX_FILES, MAX_BYTES, MAX_SECONDS = 17, 10_433_360, 30


class QuietText:
    """Discard Python output while satisfying pytest's terminal stream API."""

    encoding = "utf-8"

    def write(self, value):
        return len(value)

    def flush(self):
        pass

    def isatty(self):
        return False


def control_bytes(path, expected, maximum):
    with path.open("rb") as handle:
        data = handle.read(maximum + 1)
    if len(data) > maximum or hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("control_binding_rejected")
    return data


def io_primitives():
    source = control_bytes(IO_PATH, IO_SHA256, 65_536)
    module = types.ModuleType("_aya_plan_bound_io")
    module.__file__ = str(IO_PATH)
    exec(compile(source, str(IO_PATH), "exec"), module.__dict__)
    module.MAX_FILES, module.MAX_BYTES = MAX_FILES, MAX_BYTES
    return module


def read_spec():
    return json.loads(control_bytes(SPEC_PATH, SPEC_SHA256, 32_768))


def network_denial(io):
    def guard(event, _args):
        if event.startswith("socket.") or event in {
            "subprocess.Popen",
            "os.system",
            "os.fork",
            "os.forkpty",
            "os.posix_spawn",
            "os.exec",
        }:
            raise io.Stop("boundary_violation")

    sys.addaudithook(guard)


def warm_runtime(io):
    sys.path.insert(0, str(io.SITE))
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ.pop("PYTEST_ADDOPTS", None)
    os.environ.pop("PYTEST_PLUGINS", None)
    import pytest

    package_name = "_lumi_aya_plan_entry"
    io.require(package_name not in sys.modules, "runtime_rejected")
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "src/lumi")]
    sys.modules[package_name] = package
    core = importlib.import_module(package_name + ".posttraining_aya_plan_snapshot_v1")
    return pytest, core


class OneNode:
    """Accept exactly one requested node and its three passing lifecycle reports."""

    def __init__(self, io):
        self.io = io
        self.collected = 0
        self.phases = {}

    def pytest_collection_modifyitems(self, items):
        self.io.require(len(items) == 1 and items[0].nodeid == NODEID, "reproduction_rejected")
        self.collected = len(items)

    def pytest_runtest_logreport(self, report):
        self.io.require(
            report.nodeid == NODEID
            and report.when in {"setup", "call", "teardown"}
            and report.when not in self.phases,
            "reproduction_rejected",
        )
        self.phases[report.when] = report.outcome

    def passed(self):
        return self.collected == 1 and self.phases == {
            "setup": "passed",
            "call": "passed",
            "teardown": "passed",
        }


def run_projected_test(io, pytest, module):
    io.require(MODULE_NAME not in sys.modules, "runtime_rejected")
    io.require(
        module.__name__ == MODULE_NAME and module.__file__ == str(TEST_PATH), "runtime_rejected"
    )
    plugin = OneNode(io)
    sys.modules[MODULE_NAME] = module
    try:
        result = pytest.main(
            [
                "-c",
                "/dev/null",
                "--rootdir",
                str(ROOT),
                "--noconftest",
                "--assert=plain",
                "-p",
                "no:cacheprovider",
                "-p",
                "no:logging",
                "-p",
                "no:faulthandler",
                "-s",
                "-q",
                "-o",
                "addopts=",
                str(TEST_PATH) + "::" + TEST_NAME,
            ],
            plugins=[plugin],
        )
        io.require(result == 0 and plugin.passed(), "reproduction_rejected")
        return {"collected": plugin.collected, "phases": dict(plugin.phases)}
    finally:
        if sys.modules.get(MODULE_NAME) is module:
            del sys.modules[MODULE_NAME]


def synthetic_module():
    module = types.ModuleType(MODULE_NAME)
    module.__file__ = str(TEST_PATH)
    exec(
        compile(
            "def " + TEST_NAME + "():\n    assert 2 + 2 == 4\n",
            "<synthetic-aya-plan-runtime>",
            "exec",
        ),
        module.__dict__,
    )
    return module


def main(argv=None):
    output = {
        "schema_version": "lumi.aya-plan-snapshot-local.v0.1",
        "status": "STOP",
        "original_test_body_executed": False,
        "snapshot_backed_pytest_nodes_passed": 0,
        "ordinary_checkout_test_passed": False,
        "full_private_integration_passed": False,
        "execution_authorized": False,
        "production_eligible": False,
    }
    io = inputs = boundary = result = historical = None
    stage = "control_validation"
    started = time.monotonic()

    def deadline(_number, _frame):
        if io is None:
            raise TimeoutError("deadline_exceeded")
        raise io.Stop("deadline_exceeded")

    try:
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(MAX_SECONDS)
        io = io_primitives()
        io.require(
            sys.flags.no_site and sys.dont_write_bytecode and not sys.flags.optimize,
            "runtime_rejected",
        )
        arguments = list(sys.argv[1:] if argv is None else argv)
        io.require(
            arguments in (["--check-runtime"], ["--verify-original-plan-test"]),
            "arguments_rejected",
        )
        spec = read_spec()
        real = arguments == ["--verify-original-plan-test"]
        if real:
            stage = "historical_source_acquisition"
            historical = io.history_bytes(spec["bindings"]["filter_implementation"])
        network_denial(io)
        stage = "runtime_startup"
        with (
            contextlib.redirect_stdout(QuietText()),
            contextlib.redirect_stderr(QuietText()),
        ):
            pytest, core = warm_runtime(io)
            boundary = io.Boundary()
            boundary.roots += ("/dev/null",)
            sys.addaudithook(boundary.guard)
            if real:
                stage = "bound_input_authentication"
                inputs = io.BoundInputs(boundary)
                payloads = {"filter_implementation": historical}
                for role, binding in spec["bindings"].items():
                    if role not in payloads:
                        payloads[role] = inputs.read(binding["path"], binding)
                stage = "authenticated_source_projection"
                module, snapshot = core.build_test_module(payloads, spec["bindings"], str(ROOT))
            else:
                module, snapshot = synthetic_module(), None
            stage = "pytest_execution"
            phases = run_projected_test(io, pytest, module)
            if inputs is not None:
                stage = "input_preservation"
                inputs.verify()
            io.require(time.monotonic() - started < MAX_SECONDS, "deadline_exceeded")
            io.require(boundary.blocked == 0, "boundary_violation")
            result = {
                "status": "PASS_AUTHENTICATED_AYA_PLAN_TEST_ONLY" if real else "PASS_RUNTIME_ONLY",
                "original_test_body_executed": real,
                "snapshot_backed_pytest_nodes_passed": int(real),
                "original_module_fully_imported": False,
                "bound_plan_objects_authenticated": 15 if real else 0,
                "input_files_preserved": real,
                "pytest_lifecycle": phases,
            }
            if snapshot is not None:
                result["snapshot_accounting"] = snapshot.summary()
    except BaseException as error:
        output["failure_code"] = (
            error.code if io is not None and isinstance(error, io.Stop) else "audit_failed"
        )
    finally:
        try:
            if inputs is not None:
                inputs.close()
        except BaseException:
            output["failure_code"] = "input_cleanup_failed"
        try:
            signal.alarm(0)
        except BaseException:
            output["failure_code"] = "deadline_cleanup_failed"
    if time.monotonic() - started >= MAX_SECONDS:
        output["failure_code"] = "deadline_exceeded"
    if result is not None and "failure_code" not in output:
        output.update(result)
    output.update(
        {
            "last_stage": stage,
            "bound_input_opens": inputs.opens if inputs is not None else 0,
            "bound_input_bytes_read": inputs.bytes_read if inputs is not None else 0,
            "historical_source_bytes_read": len(historical) if historical is not None else 0,
            "blocked_operation_attempts": boundary.blocked if boundary is not None else 0,
            "source_rows_read": 0,
            "candidate_rows_read": 0,
            "inventory_paths_followed": 0,
            "tokenizer_loaded": False,
            "input_files_written": 0,
            "network_used": False,
            "gpu_used": False,
            "inference_called": False,
        }
    )
    print(json.dumps(output, sort_keys=True))
    return 0 if output["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
