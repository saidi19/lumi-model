"""Compare existing Aya successor and receipt in one bounded two-test run.

Use the trusted local runtime with -S -B. Neither mode performs a promotion.
Real mode reads the exact existing evidence once; runtime mode reads none.
Expected registry/receipt values are computed in memory only and never written.
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
SPEC_PATH = ROOT / "configs/lumi-aya-promotion-snapshot-inputs-v0.1.json"
SPEC_SHA256 = "dd6d15dc9979da8f3e30da4a0dab97ae0502ae760b1a22e1bd79d2e7539baecb"
HELPER_PATH = ROOT / "scripts/check_aya_receipt_snapshot_local_v1.py"
HELPER_SHA256 = "8daa322e30f361b873d98b594df0f2f43043b28b2d885412518c9391c9b643ee"
MODULE_NAME = "test_aya_source_promotion"
TEST_NAMES = (
    "test_successor_changes_only_the_existing_aya_entry",
    "test_promotion_receipt_is_reproducible_raw_free_and_nonproduction",
)
TEST_PATH = ROOT / "tests/test_aya_source_promotion.py"
NODEIDS = tuple("tests/test_aya_source_promotion.py::" + name for name in TEST_NAMES)
MAX_FILES, MAX_BYTES, MAX_SECONDS = 14, 1_524_967, 30
ATTRIBUTION_BYTES, EVIDENCE_BYTES = 1_387_842, 1_397_494
SNAPSHOT_OPENS, SNAPSHOT_BYTES = 35, 4_413_108
REAL_MODE = "--verify-original-promotion-tests"
READ_ROLES = (
    "plan", "data_contract", "predecessor_registry", "promotion_implementation",
    "promotion_runner", "source_registry_implementation", "hash_helper_source", "test_source",
    "policy_source", "successor_registry", "materialization_report", "owner_review_receipt",
    "promotion_receipt", "attribution_manifest",
)
PHYSICAL_READ_PATHS = {
    "source_registry_implementation": "tests/fixtures/aya_source_registry_9f20652.py.txt",
}
CORE_PACKAGE = "_lumi_aya_promotion_entry"
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
        "posttraining_aya_promotion_lifecycle_v1",
        "src/lumi/posttraining_aya_promotion_lifecycle_v1.py",
        "2fccc2111a6973b5df27e8006d6a17cad2d1ef5fbdb0e673ba6de91f5695e79c",
    ),
    (
        "posttraining_aya_promotion_snapshot_v1",
        "src/lumi/posttraining_aya_promotion_snapshot_v1.py",
        "1a02ace306d1513ea82abdec3dc910abad63610d4d23f8c2580f487bc09a1307",
    ),
)


def control_bytes(path, expected, maximum):
    with path.open("rb") as stream:
        payload = stream.read(maximum + 1)
    if len(payload) > maximum or hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("control_binding_rejected")
    return payload


def helpers():
    """Reuse authenticated IO/runtime helpers without calling any prior main."""
    source = control_bytes(HELPER_PATH, HELPER_SHA256, 65_536)
    module = types.ModuleType("_aya_promotion_control_helpers")
    module.__file__ = str(HELPER_PATH)
    exec(compile(source, str(HELPER_PATH), "exec"), module.__dict__)
    module.CORE_PACKAGE, module.CONTROL_SOURCES = CORE_PACKAGE, CONTROL_SOURCES
    return module


def read_spec():
    return json.loads(control_bytes(SPEC_PATH, SPEC_SHA256, 16_384))


def warm_runtime(helper, io):
    pytest, core = helper.warm_runtime(io)
    lifecycle = sys.modules[CORE_PACKAGE + ".posttraining_aya_promotion_lifecycle_v1"]
    io.require(
        lifecycle.MODULE_NAME == MODULE_NAME and lifecycle.TEST_NAMES == TEST_NAMES
        and lifecycle.NODEIDS == NODEIDS,
        "runtime_rejected",
    )
    return pytest, core, lifecycle


def initial_report():
    return {
        "schema_version": "lumi.aya-promotion-snapshot-local.v0.1",
        "status": "STOP",
        "original_test_bodies_executed": False,
        "snapshot_backed_pytest_nodes_passed": 0,
        "ordinary_checkout_tests_passed": False,
        "original_modules_fully_imported": False,
        "owner_decisions_reproduced": False,
        "new_owner_approval_created": False,
        "human_authorship_independently_verified": False,
        "existing_successor_matched": False,
        "existing_promotion_receipt_matched": False,
        "historical_registry_validators_executed": False,
        "current_source_replaced": False,
        "candidate_promotion_performed": False,
        "source_registry_changed": False,
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
        shared = helper.lifecycle_helpers()
        io = shared.io_primitives()
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
            spec["schema_version"] == "lumi.aya-promotion-snapshot-inputs.v0.1"
            and spec["nodeids"] == list(NODEIDS)
            and set(spec["bindings"]) == set(READ_ROLES)
            and spec["physical_read_paths"] == PHYSICAL_READ_PATHS
            and sum(binding["bytes"] for binding in spec["bindings"].values()) == MAX_BYTES
            and spec["bindings"]["attribution_manifest"]["bytes"] == ATTRIBUTION_BYTES
            and sum(spec["bindings"][role]["bytes"] for role in READ_ROLES[-4:])
            == EVIDENCE_BYTES,
            "spec_rejected",
        )
        shared.network_denial(io)
        stage = "runtime_startup"
        with (
            contextlib.redirect_stdout(shared.QuietText()),
            contextlib.redirect_stderr(shared.QuietText()),
        ):
            pytest, core, lifecycle = warm_runtime(helper, io)
            boundary = io.Boundary()
            boundary.roots += ("/dev/null",)
            sys.addaudithook(boundary.guard)
            if real:
                stage = "bound_input_authentication"
                inputs = io.BoundInputs(boundary)
                payloads = {
                    role: inputs.read(
                        PHYSICAL_READ_PATHS.get(role, spec["bindings"][role]["path"]),
                        spec["bindings"][role],
                    )
                    for role in READ_ROLES
                }
                stage = "authenticated_promotion_projection"
                module, snapshot = core.build_test_module(payloads, spec["bindings"], str(ROOT))
                module.pytest = pytest
            else:
                module, snapshot = lifecycle.synthetic_module(str(ROOT)), None
            stage = "pytest_execution"
            phases = lifecycle.run_projected_tests(io, pytest, module, str(ROOT))
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
                    and summary["bound_blob_count"] == 11
                    and summary["filesystem_access_performed"] is False,
                    "reproduction_rejected",
                )
            io.require(time.monotonic() - started < MAX_SECONDS, "deadline_exceeded")
            io.require(boundary.blocked == 0, "boundary_violation")
            result = {
                "status": (
                    "PASS_AUTHENTICATED_AYA_PROMOTION_TESTS_ONLY" if real else "PASS_RUNTIME_ONLY"
                ),
                "original_test_bodies_executed": real,
                "snapshot_backed_pytest_nodes_passed": 2 if real else 0,
                "input_files_preserved": real,
                "existing_successor_matched": real,
                "existing_promotion_receipt_matched": real,
                "historical_registry_validators_executed": real,
                "pytest_lifecycle": phases,
            }
            if snapshot is not None:
                result.update({
                    "snapshot_accounting": summary,
                    "predecessor_source_count_verified": 10,
                    "successor_source_count_verified": 10,
                    "existing_registry_changed_entries_verified": 1,
                    "existing_receipt_claim_candidate_promotions": 1,
                    "existing_receipt_claim_production_promotions": 0,
                    "attribution_records_validated": 1478,
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
