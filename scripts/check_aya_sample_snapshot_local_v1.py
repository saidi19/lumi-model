"""One bounded reconstruction of the existing Aya sample; no artifact writes.

Use the trusted local runtime with -S -B. Only --verify-original-sample-test
reads existing private objects. --check-runtime uses a synthetic assertion.
The receipt checker is loaded only for authenticated lifecycle helpers; its
main and its private receipt check are never called.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import signal
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "configs/lumi-aya-sample-snapshot-inputs-v0.1.json"
SPEC_SHA256 = "fc08077b48e780d76813875a84efe7439f39b20dc17bafa554d091ee5f7b107d"
HELPER_PATH = ROOT / "scripts/check_aya_receipt_snapshot_local_v1.py"
HELPER_SHA256 = "8daa322e30f361b873d98b594df0f2f43043b28b2d885412518c9391c9b643ee"
MODULE_NAME = "test_aya_bengali_owner_review"
TEST_NAME = "test_actual_plan_rebuilds_exact_sample"
TEST_PATH = ROOT / "tests/test_aya_bengali_owner_review.py"
NODEID = "tests/test_aya_bengali_owner_review.py::" + TEST_NAME
MAX_FILES, MAX_BYTES, MAX_SECONDS = 10, 6_551_911, 30
CANDIDATE_BYTES, PRIVATE_BYTES = 5_016_358, 6_480_319
SNAPSHOT_OPENS, SNAPSHOT_BYTES = 9, 11_523_272
REAL_MODE = "--verify-original-sample-test"
READ_ROLES = (
    "plan", "review_implementation", "file_hash_source", "sample_helper_source",
    "test_source", "receipt_finalizer", "materialization_report",
    "owner_review_packet", "attribution_manifest", "accepted_records",
)
CORE_PACKAGE = "_lumi_aya_sample_entry"
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
        "posttraining_aya_sample_snapshot_v1",
        "src/lumi/posttraining_aya_sample_snapshot_v1.py",
        "4a6e092dec8fb6520ba3ddf7c10002238b1078e9fd53f75dd633d932324be993",
    ),
)


def control_bytes(path, expected, maximum):
    with path.open("rb") as stream:
        payload = stream.read(maximum + 1)
    if len(payload) > maximum or hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("control_binding_rejected")
    return payload


def helpers():
    """Configure a fresh hash-authenticated helper namespace, never an old main."""
    source = control_bytes(HELPER_PATH, HELPER_SHA256, 65_536)
    module = types.ModuleType("_aya_sample_control_helpers")
    module.__file__ = str(HELPER_PATH)
    exec(compile(source, str(HELPER_PATH), "exec"), module.__dict__)
    module.MODULE_NAME, module.TEST_NAME = MODULE_NAME, TEST_NAME
    module.TEST_PATH, module.NODEID = TEST_PATH, NODEID
    module.CORE_PACKAGE, module.CONTROL_SOURCES = CORE_PACKAGE, CONTROL_SOURCES
    return module


def read_spec():
    return json.loads(control_bytes(SPEC_PATH, SPEC_SHA256, 16_384))


def initial_report():
    return {
        "schema_version": "lumi.aya-sample-snapshot-local.v0.1",
        "status": "STOP",
        "original_test_body_executed": False,
        "snapshot_backed_pytest_nodes_passed": 0,
        "ordinary_checkout_test_passed": False,
        "original_module_fully_imported": False,
        "owner_decisions_reproduced": False,
        "new_owner_approval_created": False,
        "human_authorship_independently_verified": False,
        "candidate_sample_reconstructed": False,
        "new_candidate_constructed": False,
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
        helper = helpers()
        lifecycle = helper.lifecycle_helpers()
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
            spec["schema_version"] == "lumi.aya-sample-snapshot-inputs.v0.1"
            and spec["nodeid"] == NODEID and set(spec["bindings"]) == set(READ_ROLES)
            and sum(binding["bytes"] for binding in spec["bindings"].values()) == MAX_BYTES
            and spec["bindings"]["accepted_records"]["bytes"] == CANDIDATE_BYTES
            and sum(spec["bindings"][role]["bytes"] for role in READ_ROLES[-3:])
            == PRIVATE_BYTES,
            "spec_rejected",
        )
        lifecycle.network_denial(io)
        stage = "runtime_startup"
        with (
            contextlib.redirect_stdout(lifecycle.QuietText()),
            contextlib.redirect_stderr(lifecycle.QuietText()),
        ):
            pytest, core = helper.warm_runtime(io)
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
                stage = "authenticated_sample_projection"
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
                    summary["virtual_opens"] == SNAPSHOT_OPENS
                    and summary["logical_read_bytes_debited"] == SNAPSHOT_BYTES
                    and summary["bound_blob_count"] == 7
                    and summary["filesystem_access_performed"] is False,
                    "reproduction_rejected",
                )
            io.require(time.monotonic() - started < MAX_SECONDS, "deadline_exceeded")
            io.require(boundary.blocked == 0, "boundary_violation")
            result = {
                "status": (
                    "PASS_AUTHENTICATED_AYA_SAMPLE_TEST_ONLY" if real else "PASS_RUNTIME_ONLY"
                ),
                "original_test_body_executed": real,
                "snapshot_backed_pytest_nodes_passed": int(real),
                "input_files_preserved": real,
                "candidate_sample_reconstructed": real,
                "pytest_lifecycle": phases,
            }
            if snapshot is not None:
                result.update({
                    "snapshot_accounting": summary,
                    "candidate_records_validated": 1478,
                    "sample_records_validated": 32,
                    "original_sample_identity_matched": True,
                })
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
        "upstream_source_rows_read": 0,
        "candidate_rows_parsed": (
            1478 if output["candidate_sample_reconstructed"]
            else None if inputs is not None else 0
        ),
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
