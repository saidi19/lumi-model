"""One bounded existing-receipt check, not owner approval or candidate replay.

Use the trusted local runtime with -S -B. Only --verify-original-receipt-test
opens the exact existing receipt. Runtime checks use a synthetic assertion.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import signal
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "configs/lumi-aya-receipt-snapshot-inputs-v0.1.json"
SPEC_SHA256 = "155a4608c1188f1f7de7603893d821f7f715f53c733ff62167455d3fdeb9e4bf"
LIFECYCLE_PATH = ROOT / "scripts/check_aya_plan_snapshot_local_v1.py"
LIFECYCLE_SHA256 = "8a4bbf47954c870b0b9ca1274fe483f3f1bfd154349fddc5fa7ef8a1c8aa13ab"
MODULE_NAME = "test_aya_bengali_owner_review"
TEST_NAME = "test_actual_owner_receipt_is_complete_and_nonpromoting"
TEST_PATH = ROOT / "tests/test_aya_bengali_owner_review.py"
NODEID = "tests/test_aya_bengali_owner_review.py::" + TEST_NAME
MAX_FILES, MAX_BYTES, MAX_SECONDS = 4, 45_471, 30
RECEIPT_BYTES = 1667
REAL_MODE = "--verify-original-receipt-test"
READ_ROLES = ("review_implementation", "hash_helper_source", "test_source", "receipt")
CORE_PACKAGE = "_lumi_aya_receipt_entry"
CONTROL_SOURCES = (
    (
        "posttraining_safety_replay_snapshot_v1",
        "src/lumi/posttraining_safety_replay_snapshot_v1.py",
        "16031f300c4fb023a85ecfded74b54333a1fc66bb55cd762e08b9f8c8e95478a",
    ),
    (
        "posttraining_aya_plan_snapshot_v1",
        "src/lumi/posttraining_aya_plan_snapshot_v1.py",
        "c4307277e4521d0a66ae4c09407f9bc4a02ee62a5c97703711ca74f4dcf35be8",
    ),
    (
        "posttraining_aya_receipt_snapshot_v1",
        "src/lumi/posttraining_aya_receipt_snapshot_v1.py",
        "64d5524ba3fccce891992fb668369febf89b3a0c707ef85a62115cfbc19ad8db",
    ),
)


def control_bytes(path, expected, maximum):
    with path.open("rb") as stream:
        payload = stream.read(maximum + 1)
    if len(payload) > maximum or hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("control_binding_rejected")
    return payload


def lifecycle_helpers():
    """A fresh hash-checked helper namespace; its old main is never invoked.

    Only the one-node identity is configured for this successor. This does not
    modify the prior helper file, its installed namespace or any test statement.
    """
    source = control_bytes(LIFECYCLE_PATH, LIFECYCLE_SHA256, 65_536)
    module = types.ModuleType("_aya_receipt_pytest_lifecycle")
    module.__file__ = str(LIFECYCLE_PATH)
    exec(compile(source, str(LIFECYCLE_PATH), "exec"), module.__dict__)
    module.MODULE_NAME, module.TEST_NAME = MODULE_NAME, TEST_NAME
    module.TEST_PATH, module.NODEID = TEST_PATH, NODEID
    return module


def read_spec():
    return json.loads(control_bytes(SPEC_PATH, SPEC_SHA256, 16_384))


def authenticated_core(io):
    io.require(
        not any(
            name == CORE_PACKAGE or name.startswith(CORE_PACKAGE + ".") for name in sys.modules
        ),
        "runtime_rejected",
    )
    authenticated = [
        (suffix, relative, control_bytes(ROOT / relative, digest, 65_536))
        for suffix, relative, digest in CONTROL_SOURCES
    ]
    installed = {}
    try:
        package = types.ModuleType(CORE_PACKAGE)
        package.__path__ = []  # No fallback imports from mutable source paths.
        installed[CORE_PACKAGE] = package
        sys.modules[CORE_PACKAGE] = package
        for suffix, relative, source in authenticated:
            name = CORE_PACKAGE + "." + suffix
            module = types.ModuleType(name)
            module.__file__, module.__package__ = str(ROOT / relative), CORE_PACKAGE
            installed[name] = module
            sys.modules[name] = module
            exec(compile(source, module.__file__, "exec"), module.__dict__)
        return module
    except BaseException:
        for name, module in reversed(tuple(installed.items())):
            if sys.modules.get(name) is module:
                del sys.modules[name]
        raise


def warm_runtime(io):
    sys.path.insert(0, str(io.SITE))
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ.pop("PYTEST_ADDOPTS", None)
    os.environ.pop("PYTEST_PLUGINS", None)
    import pytest

    return pytest, authenticated_core(io)


def initial_report():
    return {
        "schema_version": "lumi.aya-receipt-snapshot-local.v0.1",
        "status": "STOP",
        "original_test_body_executed": False,
        "snapshot_backed_pytest_nodes_passed": 0,
        "ordinary_checkout_test_passed": False,
        "original_module_fully_imported": False,
        "owner_decisions_reproduced": False,
        "human_authorship_independently_verified": False,
        "candidate_sample_reconstructed": False,
        "full_private_integration_passed": False,
        "execution_authorized": False,
        "production_eligible": False,
    }


def main(argv=None):
    output = initial_report()
    io = inputs = boundary = result = None
    stage = "control_validation"
    started = time.monotonic()

    def deadline(_number, _frame):
        if io is None:
            raise TimeoutError("deadline_exceeded")
        raise io.Stop("deadline_exceeded")

    try:
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(MAX_SECONDS)
        lifecycle = lifecycle_helpers()
        io = lifecycle.io_primitives()
        io.MAX_FILES, io.MAX_BYTES, io.MAX_SECONDS = MAX_FILES, MAX_BYTES, MAX_SECONDS
        io.require(
            sys.flags.no_site and sys.dont_write_bytecode and not sys.flags.optimize,
            "runtime_rejected",
        )
        arguments = list(sys.argv[1:] if argv is None else argv)
        io.require(arguments in (["--check-runtime"], [REAL_MODE]), "arguments_rejected")
        real = arguments == [REAL_MODE]
        spec = read_spec()
        io.require(
            spec["nodeid"] == NODEID and set(spec["bindings"]) == set(READ_ROLES)
            and sum(binding["bytes"] for binding in spec["bindings"].values()) == MAX_BYTES
            and spec["bindings"]["receipt"]["bytes"] == RECEIPT_BYTES,
            "spec_rejected",
        )
        lifecycle.network_denial(io)
        stage = "runtime_startup"
        with (
            contextlib.redirect_stdout(lifecycle.QuietText()),
            contextlib.redirect_stderr(lifecycle.QuietText()),
        ):
            pytest, core = warm_runtime(io)
            boundary = io.Boundary()
            boundary.roots += ("/dev/null",)
            sys.addaudithook(boundary.guard)
            if real:
                stage = "bound_input_authentication"
                inputs = io.BoundInputs(boundary)
                payloads = {
                    role: inputs.read(spec["bindings"][role]["path"], spec["bindings"][role])
                    for role in READ_ROLES
                }
                stage = "authenticated_receipt_projection"
                module, snapshot = core.build_test_module(payloads, spec["bindings"], str(ROOT))
                module.pytest = pytest
            else:
                module, snapshot = lifecycle.synthetic_module(), None
            stage = "pytest_execution"
            phases = lifecycle.run_projected_test(io, pytest, module)
            if inputs is not None:
                stage = "input_preservation"
                inputs.verify()
                io.require(
                    inputs.opens == MAX_FILES and inputs.bytes_read == MAX_BYTES, "read_budget"
                )
                summary = snapshot.summary()
                io.require(
                    summary["virtual_opens"] == 1
                    and summary["logical_read_bytes_debited"] == RECEIPT_BYTES
                    and summary["bound_blob_count"] == 1
                    and summary["filesystem_access_performed"] is False,
                    "reproduction_rejected",
                )
            io.require(time.monotonic() - started < MAX_SECONDS, "deadline_exceeded")
            io.require(boundary.blocked == 0, "boundary_violation")
            result = {
                "status": (
                    "PASS_AUTHENTICATED_AYA_RECEIPT_TEST_ONLY" if real else "PASS_RUNTIME_ONLY"
                ),
                "original_test_body_executed": real,
                "snapshot_backed_pytest_nodes_passed": int(real),
                "input_files_preserved": real,
                "pytest_lifecycle": phases,
            }
            if snapshot is not None:
                result["snapshot_accounting"] = summary
                result["receipt_claim_accepted_records"] = 32
                result["receipt_claim_rejected_records"] = 0
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
    output.update({
        "last_stage": stage,
        "bound_input_opens": inputs.opens if inputs is not None else 0,
        "bound_input_bytes_read": inputs.bytes_read if inputs is not None else 0,
        "blocked_operation_attempts": boundary.blocked if boundary is not None else 0,
        "source_rows_read": 0,
        "candidate_rows_read": 0,
        "tokenizer_loaded": False,
        "input_files_written": 0,
        "network_used": False,
        "gpu_used": False,
        "inference_called": False,
    })
    print(json.dumps(output, sort_keys=True))
    return 0 if output["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
