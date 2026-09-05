"""Zero-data-read recovery audit for the completed v0.2 tiny CPU SFT smoke."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import posttraining_sft_cpu_correctness_smoke_v2 as failed_smoke
from . import posttraining_sft_runtime_v3 as executed_runtime
from . import posttraining_sft_runtime_v4 as corrected_runtime

PLAN_SCHEMA_VERSION = "lumi.sft-cpu-correctness-smoke-audit-recovery-plan.v0.3"
REPORT_SCHEMA_VERSION = "lumi.sft-cpu-correctness-smoke-audit-recovery-report.v0.3"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-cpu-correctness-smoke-audit-recovery-verification.v0.3"
)
PLAN_STATUS = "authorized_zero_data_read_existing_attempt_audit_recovery_v0_3"
REPORT_STATUS = "PASS_recovered_completed_cpu_smoke_without_training_retry_v0_3"
VERIFICATION_STATUS = "PASS_verified_zero_retry_cpu_smoke_audit_recovery_v0_3"

EXPECTED_AUTHORIZATION_TEXT = "mama kaj resume kor"
EXPECTED_AUTHORIZATION_BYTES = 19
EXPECTED_AUTHORIZATION_SHA256 = (
    "f69f2743757ca53698ec5d217e6aa074c0d02296798414d26ced0656f733f1d3"
)
EXPECTED_PLAN_PATH = Path(
    "configs/lumi-4b-sft-cpu-correctness-smoke-audit-recovery-v0.3.json"
)
EXPECTED_FAILED_ATTEMPT = {
    "checkpoint_schema_extra_field": "checkpoint_schema_version",
    "checkpoint_schema_extra_value": "lumi.sft-cpu-canary-checkpoint.v0.3",
    "failure_code": "resumed_top_state_checkpoint_schema_propagation",
    "file_bytes": 54_879_851,
    "inventory_entries": 53,
    "inventory_sha256": (
        "a5e1d1c775896ff5af0ca8ce566824e340774b9e1937eedc069646f85daad1b7"
    ),
    "private_root": "artifacts/lumi-sft-cpu-smoke-v0.2/attempt-001/private",
    "stable_metrics_sha256": (
        "1c815ec237bcb71fbc1446b54b57f48e26f5bf5753e1950d40f398a55772686b"
    ),
}
EXPECTED_FINAL_STATE = {
    "loss_sequence": [11.087256037394205, 11.105231468494122],
    "records_seen": 4,
    "status": "complete",
    "step": 2,
    "supervised_tokens_seen": 153,
    "tokens_seen": 219,
}
EXPECTED_READ_AUDIT = {
    "binary_bytes_read": 1_752,
    "encoded_index_row_reads": 4,
    "file_opens": 4,
    "fully_hashed_shards": 0,
    "index_bytes_read": 644,
    "metadata_inventory_verified_before_selected_content_read": True,
    "opened_shards": 1,
    "records_loaded": 4,
    "unbuffered_selected_content_reads": True,
}
EXPECTED_CHECKPOINT_AUDIT = {
    "final_model_semantic_sha256": (
        "96616825fe436416adf8b236459b0466408962ef78c00ed5c759d3fc77ea8aa8"
    ),
    "final_optimizer_semantic_sha256": (
        "1b0fb265d2d433f3a8768a15ee679de6d4c9e7d82fa24e107be443783626d9ea"
    ),
    "final_rng_semantic_sha256": (
        "f3d739372dfd5c8cead5b6fc9a176e3c64fae7b20864f78c8776d717bcef379b"
    ),
    "step_checkpoint_manifest_sha256": {
        "1": "76707e78ecdc1097548be0c1b7af22e4f87f48facde16fe651aa85814fbd4656",
        "2": "356cb49ec7eedaa4b968a28247020ebf789e55e8db344e5493836f7fd9e26a60",
    },
}
EXPECTED_LIMITS = {
    "maximum_additional_encoded_binary_bytes_read": 0,
    "maximum_additional_encoded_index_bytes_read": 0,
    "maximum_additional_encoded_index_rows_read": 0,
    "maximum_additional_private_write_bytes": 2_097_152,
    "maximum_additional_training_sample_consumptions": 0,
    "maximum_candidate_or_source_row_reads": 0,
    "maximum_existing_attempt_private_bytes_read": 536_870_912,
    "maximum_gpu_use": 0,
    "maximum_inference_calls": 0,
    "maximum_network_bytes": 0,
    "maximum_optimizer_updates": 0,
    "maximum_retries_of_failed_attempt": 0,
    "maximum_runtime_minutes": 10,
    "maximum_test_object_reads": 0,
}
EXPECTED_OUTPUTS = {
    "private_root": (
        "artifacts/lumi-sft-cpu-smoke-audit-recovery-v0.3/attempt-001/private"
    ),
    "recovery_marker": (
        "artifacts/lumi-sft-cpu-smoke-audit-recovery-v0.3/attempt-001/private/"
        "audit-recovery-started.json"
    ),
    "report": (
        "evidence/lumi-4b-sft-cpu-correctness-smoke-audit-recovery-v0.3.json"
    ),
    "verification": (
        "evidence/lumi-4b-sft-cpu-correctness-smoke-audit-recovery-"
        "verification-v0.3.json"
    ),
}
EXPECTED_FIXED_BINDINGS = {
    "corrected_runtime_test_v4": {
        "path": "tests/test_posttraining_sft_runtime_v4.py",
        "sha256": "bcb1eef143ab0d9f8e6a1af2a772db81d8cb1cdc66fde0c8f1f724607ddce9bc",
    },
    "corrected_runtime_v4": {
        "path": "src/lumi/posttraining_sft_runtime_v4.py",
        "sha256": "5782357e8f6e3a734d93c493d3ad1801fc3af81eeb2fd34d494541cad991ed27",
    },
    "corrected_runner_v4": {
        "path": "scripts/run_sft_cpu_canary_v4.py",
        "sha256": "2790b7fbb80160c11cd39d1b3b245e46a2443eabc87830616550186cb093abc7",
    },
    "executed_plan_v0_2": {
        "path": "configs/lumi-4b-sft-cpu-correctness-smoke-v0.2.json",
        "sha256": "2d9c6bbf9e90b60c7ceb24b58f6da3101e18a80227aecc7013f41b9d3e37e1ac",
    },
    "executed_runtime_v3": {
        "path": "src/lumi/posttraining_sft_runtime_v3.py",
        "sha256": "335bf09b5ca2df962848beb8dd50cf59114924deb44564c9fcf65d7d0f7b0b8d",
    },
    "failed_auditor_v0_2": {
        "path": "src/lumi/posttraining_sft_cpu_correctness_smoke_v2.py",
        "sha256": "1eaf9fc7b0ba4001c46469a6340b956c8dad9aded24d9775c17c24083d8092c1",
    },
    "model_implementation": {
        "path": "src/lumi/modeling_lumi.py",
        "sha256": "bf7b908d282a752132918dddad648fd8435d3f8ad9db41eac1c2e64715926fad",
    },
}
EXPECTED_CODE_PATHS = {
    "implementation": "src/lumi/posttraining_sft_cpu_audit_recovery_v3.py",
    "runner": "scripts/recover_sft_cpu_correctness_smoke_audit_v3.py",
    "test": "tests/test_sft_cpu_correctness_smoke_audit_recovery_v3.py",
    "verifier": "scripts/verify_sft_cpu_correctness_smoke_audit_recovery_v3.py",
}

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_STABLE_STATE_FIELDS = frozenset(failed_smoke._STABLE_STATE_FIELDS)
_RECOVERY_MARKER = {
    "failed_attempt_inventory_sha256": EXPECTED_FAILED_ATTEMPT["inventory_sha256"],
    "schema_version": "lumi.sft-cpu-correctness-smoke-audit-recovery-started.v0.3",
    "status": "started_no_retry",
}
_MAXIMUM_EVIDENCE_FILE_BYTES = 65_536
_REPORT_CHECKS = {
    "checkpoint_manifests_exact_across_paths": True,
    "diagnosed_extra_field_is_exact_and_singular": True,
    "existing_attempt_bytes_preserved": True,
    "final_model_optimizer_and_rng_exact": True,
    "physical_read_caps_from_executed_run_met": True,
    "runtime_v4_schema_leak_regression_fixed": True,
    "stable_metrics_and_loss_sequence_exact": True,
    "training_paths_both_complete": True,
}
_REPORT_AUTHORIZATION_BOUNDARY = {
    "additional_encoded_content_reads": 0,
    "additional_optimizer_updates": 0,
    "additional_training_sample_consumptions": 0,
    "failed_attempt_mutated": False,
    "gpu_used": False,
    "inference_calls": 0,
    "network_bytes": 0,
    "production_sft_started": False,
    "retry_of_failed_attempt": False,
}
_REPORT_DECISION = {
    "original_v0_2_orchestrator_passed": False,
    "original_v0_2_training_completed": True,
    "recovery_audit_passed": True,
    "recovery_reason_code": EXPECTED_FAILED_ATTEMPT["failure_code"],
}
_EXECUTED_RESOURCE_ACCOUNTING = {
    "dataset_loads": 3,
    "encoded_binary_bytes_read": EXPECTED_READ_AUDIT["binary_bytes_read"] * 3,
    "encoded_index_bytes_read": EXPECTED_READ_AUDIT["index_bytes_read"] * 3,
    "encoded_index_rows_read": 12,
    "optimizer_updates": 4,
    "training_sample_consumptions": 8,
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    return failed_smoke.sha256_file(path)


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key in CPU-smoke recovery input")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_bytes().decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant: {item}")
            ),
        )
    except UnicodeDecodeError as error:
        raise ValueError("CPU-smoke recovery JSON is not UTF-8") from error
    if not isinstance(value, dict):
        raise ValueError("CPU-smoke recovery JSON root is not an object")
    return value


def _resolve_bound_file(root: Path, name: str, binding: Mapping[str, Any]) -> None:
    if set(binding) != {"path", "sha256"}:
        raise ValueError(f"CPU-smoke recovery binding shape changed: {name}")
    relative = binding.get("path")
    digest = binding.get("sha256")
    if (
        not isinstance(relative, str)
        or Path(relative).is_absolute()
        or not isinstance(digest, str)
        or _SHA256_PATTERN.fullmatch(digest) is None
    ):
        raise ValueError(f"CPU-smoke recovery binding is invalid: {name}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"CPU-smoke recovery binding path is invalid: {name}")
    if sha256_file(path) != digest:
        raise ValueError(f"CPU-smoke recovery binding changed: {name}")


def validate_plan(plan: Mapping[str, Any], *, repository_root: Path) -> None:
    expected_top = {
        "authorization_bytes",
        "authorization_sha256",
        "authorization_text",
        "bindings",
        "failed_attempt",
        "limits",
        "outputs",
        "production_eligible",
        "raw_training_text_retained",
        "schema_version",
        "status",
    }
    if (
        set(plan) != expected_top
        or plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("status") != PLAN_STATUS
        or plan.get("authorization_text") != EXPECTED_AUTHORIZATION_TEXT
        or plan.get("authorization_bytes") != EXPECTED_AUTHORIZATION_BYTES
        or plan.get("authorization_sha256") != EXPECTED_AUTHORIZATION_SHA256
        or plan.get("failed_attempt") != EXPECTED_FAILED_ATTEMPT
        or plan.get("limits") != EXPECTED_LIMITS
        or plan.get("outputs") != EXPECTED_OUTPUTS
        or plan.get("production_eligible") is not False
        or plan.get("raw_training_text_retained") is not False
    ):
        raise ValueError("CPU-smoke audit-recovery plan contract changed")
    authorization = EXPECTED_AUTHORIZATION_TEXT.encode("utf-8")
    if (
        len(authorization) != EXPECTED_AUTHORIZATION_BYTES
        or hashlib.sha256(authorization).hexdigest()
        != EXPECTED_AUTHORIZATION_SHA256
    ):
        raise RuntimeError("compiled CPU-smoke recovery authorization changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != (
        set(EXPECTED_FIXED_BINDINGS) | set(EXPECTED_CODE_PATHS)
    ):
        raise ValueError("CPU-smoke audit-recovery binding inventory changed")
    for name, expected in EXPECTED_FIXED_BINDINGS.items():
        if bindings.get(name) != expected:
            raise ValueError(f"CPU-smoke fixed recovery binding changed: {name}")
    for name, path in EXPECTED_CODE_PATHS.items():
        binding = bindings.get(name)
        if not isinstance(binding, Mapping) or binding.get("path") != path:
            raise ValueError(f"CPU-smoke recovery code path changed: {name}")
    for name, binding in bindings.items():
        assert isinstance(binding, Mapping)
        _resolve_bound_file(repository_root, str(name), binding)


def load_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan_path = Path(path).resolve()
    if plan_path != (root / EXPECTED_PLAN_PATH).resolve():
        raise ValueError("CPU-smoke audit-recovery plan path changed")
    plan = _read_json(plan_path)
    validate_plan(plan, repository_root=root)
    return plan


def _private_tree_snapshot(root: Path) -> list[dict[str, Any]]:
    root_info = root.lstat()
    if (
        stat.S_ISLNK(root_info.st_mode)
        or not stat.S_ISDIR(root_info.st_mode)
        or stat.S_IMODE(root_info.st_mode) != 0o700
        or root_info.st_uid != os.geteuid()
    ):
        raise ValueError("failed CPU-smoke private root metadata changed")
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        info = path.lstat()
        relative = path.relative_to(root).as_posix()
        mode = f"{stat.S_IMODE(info.st_mode):04o}"
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("failed CPU-smoke attempt contains a symlink")
        if stat.S_ISDIR(info.st_mode):
            if mode != "0700" or info.st_uid != os.geteuid():
                raise ValueError("failed CPU-smoke directory metadata changed")
            entries.append({"kind": "directory", "mode": mode, "path": relative})
        elif stat.S_ISREG(info.st_mode):
            if mode != "0600" or info.st_uid != os.geteuid() or info.st_nlink != 1:
                raise ValueError("failed CPU-smoke file metadata changed")
            entries.append(
                {
                    "bytes": info.st_size,
                    "kind": "file",
                    "mode": mode,
                    "path": relative,
                    "sha256": sha256_file(path),
                }
            )
        else:
            raise ValueError("failed CPU-smoke attempt contains a special file")
    return entries


def _snapshot_summary(root: Path) -> dict[str, Any]:
    entries = _private_tree_snapshot(root)
    return {
        "file_bytes": sum(item.get("bytes", 0) for item in entries),
        "inventory_entries": len(entries),
        "inventory_sha256": sha256_value(entries),
    }


def _normalized_completed_states(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    uninterrupted = _read_json(
        root / "uninterrupted" / executed_runtime.SFT_CANARY_STATE_FILENAME
    )
    resumed = _read_json(root / "resumed" / executed_runtime.SFT_CANARY_STATE_FILENAME)
    extra = EXPECTED_FAILED_ATTEMPT["checkpoint_schema_extra_field"]
    if set(uninterrupted) != _STABLE_STATE_FIELDS:
        raise ValueError("uninterrupted CPU-smoke state shape changed")
    if set(resumed) != (_STABLE_STATE_FIELDS | {extra}):
        raise ValueError("resumed CPU-smoke state failure signature changed")
    if resumed.get(extra) != EXPECTED_FAILED_ATTEMPT["checkpoint_schema_extra_value"]:
        raise ValueError("resumed CPU-smoke checkpoint schema value changed")
    normalized = {key: value for key, value in resumed.items() if key != extra}
    if uninterrupted != normalized:
        raise ValueError("CPU-smoke states differ beyond the diagnosed schema leak")
    expected = EXPECTED_FINAL_STATE
    if any(
        uninterrupted.get(key) != value
        for key, value in expected.items()
        if key != "loss_sequence"
    ):
        raise ValueError("completed CPU-smoke state totals changed")
    read_audit = uninterrupted.get("run_specification", {}).get("data", {}).get(
        "read_audit"
    )
    if read_audit != EXPECTED_READ_AUDIT:
        raise ValueError("completed CPU-smoke physical read audit changed")
    return uninterrupted, normalized


def audit_existing_attempt(*, repository_root: Path) -> dict[str, Any]:
    attempt_root = repository_root / EXPECTED_FAILED_ATTEMPT["private_root"]
    before = _snapshot_summary(attempt_root)
    if before != {
        key: EXPECTED_FAILED_ATTEMPT[key]
        for key in ("file_bytes", "inventory_entries", "inventory_sha256")
    }:
        raise ValueError("failed CPU-smoke attempt inventory changed")
    failed_smoke._require_exact_output_inventory(attempt_root)
    state, _normalized = _normalized_completed_states(attempt_root)
    left_metrics = failed_smoke._stable_metrics(
        attempt_root / "uninterrupted" / executed_runtime.SFT_CANARY_METRICS_FILENAME
    )
    right_metrics = failed_smoke._stable_metrics(
        attempt_root / "resumed" / executed_runtime.SFT_CANARY_METRICS_FILENAME
    )
    if (
        left_metrics != right_metrics
        or [item["loss"] for item in left_metrics]
        != EXPECTED_FINAL_STATE["loss_sequence"]
        or sha256_value(left_metrics)
        != EXPECTED_FAILED_ATTEMPT["stable_metrics_sha256"]
    ):
        raise ValueError("completed CPU-smoke stable metrics changed")
    checkpoint_audit = failed_smoke._checkpoint_pair_audit(
        attempt_root / "uninterrupted",
        attempt_root / "resumed",
    )
    if checkpoint_audit != EXPECTED_CHECKPOINT_AUDIT:
        raise ValueError("completed CPU-smoke checkpoint equality audit changed")
    source = attempt_root / "source"
    source_manifest_sha256 = executed_runtime._verify_checksum_manifest(
        source,
        expected_manifest_sha256=failed_smoke.EXPECTED_SOURCE_CHECKPOINT[
            "manifest.sha256"
        ],
        require_private=True,
    )
    after = _snapshot_summary(attempt_root)
    if after != before:
        raise ValueError("failed CPU-smoke attempt changed during recovery audit")
    return {
        "checkpoint_audit": checkpoint_audit,
        "existing_attempt_snapshot": after,
        "final_state": {
            key: state[key]
            for key in (
                "records_seen",
                "run_fingerprint",
                "status",
                "step",
                "supervised_tokens_seen",
                "tokens_seen",
            )
        },
        "loss_sequence": [item["loss"] for item in left_metrics],
        "read_audit_per_load": dict(EXPECTED_READ_AUDIT),
        "source_checkpoint_manifest_sha256": source_manifest_sha256,
        "stable_metrics_sha256": sha256_value(left_metrics),
    }


def _build_report(
    plan: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    plan_file_sha256: str,
) -> dict[str, Any]:
    report = {
        "authorization_boundary": dict(_REPORT_AUTHORIZATION_BOUNDARY),
        "checks": dict(_REPORT_CHECKS),
        "decision": dict(_REPORT_DECISION),
        "evidence": dict(audit),
        "executed_resource_accounting": dict(_EXECUTED_RESOURCE_ACCOUNTING),
        "plan_file_sha256": plan_file_sha256,
        "production_eligible": False,
        "raw_training_text_retained": False,
        "recovery_resource_accounting": {
            "additional_encoded_binary_bytes_read": 0,
            "additional_encoded_index_bytes_read": 0,
            "additional_encoded_index_rows_read": 0,
            "additional_optimizer_updates": 0,
            "additional_training_sample_consumptions": 0,
            "gpu_use": 0,
            "inference_calls": 0,
            "network_bytes": 0,
            "existing_attempt_private_bytes_read_upper_bound": plan["limits"][
                "maximum_existing_attempt_private_bytes_read"
            ],
        },
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": REPORT_STATUS,
    }
    failed_smoke.assert_raw_text_free(report)
    report["report_sha256"] = sha256_value(report)
    return report


def _finalize_report_size(report: dict[str, Any], *, marker_bytes: int) -> int:
    size = 0
    resources = report["recovery_resource_accounting"]
    for _ in range(16):
        resources["recovery_marker_bytes"] = marker_bytes
        resources["recovery_report_bytes"] = size
        resources["verification_bytes_reserved"] = _MAXIMUM_EVIDENCE_FILE_BYTES
        resources["recovery_private_write_bytes_upper_bound"] = (
            marker_bytes + size + _MAXIMUM_EVIDENCE_FILE_BYTES
        )
        unsigned = dict(report)
        unsigned.pop("report_sha256", None)
        report["report_sha256"] = sha256_value(unsigned)
        observed = len(failed_smoke.json_file_bytes(report))
        if observed == size:
            if observed > _MAXIMUM_EVIDENCE_FILE_BYTES:
                raise ValueError("CPU-smoke recovery report exceeds its file bound")
            return observed
        size = observed
    raise RuntimeError("CPU-smoke recovery report size did not converge")


def run_audit_recovery(
    *,
    plan_path: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan_path = Path(plan_path).resolve()
    plan = load_plan(plan_path, repository_root=root)
    started = time.monotonic()
    report_path = root / plan["outputs"]["report"]
    verification_path = root / plan["outputs"]["verification"]
    if os.path.lexists(report_path) or os.path.lexists(verification_path):
        raise FileExistsError("CPU-smoke recovery evidence already exists")
    private_root = root / plan["outputs"]["private_root"]
    corrected_runtime._ensure_private_directory(private_root)
    if any(private_root.iterdir()):
        raise FileExistsError("CPU-smoke audit recovery already started; retry forbidden")
    marker = root / plan["outputs"]["recovery_marker"]
    corrected_runtime._write_json_exclusive(marker, _RECOVERY_MARKER)
    audit = audit_existing_attempt(repository_root=root)
    report = _build_report(
        plan,
        audit,
        plan_file_sha256=sha256_file(plan_path),
    )
    marker_bytes = marker.stat().st_size
    _finalize_report_size(report, marker_bytes=marker_bytes)
    if (
        report["recovery_resource_accounting"][
            "recovery_private_write_bytes_upper_bound"
        ]
        > plan["limits"]["maximum_additional_private_write_bytes"]
    ):
        raise ValueError("CPU-smoke audit-recovery private-write cap exceeded")
    if time.monotonic() - started > plan["limits"]["maximum_runtime_minutes"] * 60:
        raise TimeoutError("CPU-smoke audit-recovery runtime cap exceeded")
    failed_smoke._private_write_exclusive(report_path, report)
    return report


def verify_audit_recovery(
    *,
    plan_path: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan_path = Path(plan_path).resolve()
    plan = load_plan(plan_path, repository_root=root)
    started = time.monotonic()
    marker = root / plan["outputs"]["recovery_marker"]
    marker_bytes = failed_smoke._private_evidence_size(marker)
    if _read_json(marker) != _RECOVERY_MARKER:
        raise ValueError("CPU-smoke recovery marker changed")
    report_path = root / plan["outputs"]["report"]
    report_bytes = failed_smoke._private_evidence_size(report_path)
    report = _read_json(report_path)
    unsigned_report = dict(report)
    report_internal = unsigned_report.pop("report_sha256", None)
    recovery_resources = report.get("recovery_resource_accounting")
    if (
        set(report)
        != {
            "authorization_boundary",
            "checks",
            "decision",
            "evidence",
            "executed_resource_accounting",
            "plan_file_sha256",
            "production_eligible",
            "raw_training_text_retained",
            "recovery_resource_accounting",
            "report_sha256",
            "schema_version",
            "status",
        }
        or report_internal != sha256_value(unsigned_report)
        or report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("status") != REPORT_STATUS
        or report.get("production_eligible") is not False
        or report.get("raw_training_text_retained") is not False
        or report.get("authorization_boundary")
        != _REPORT_AUTHORIZATION_BOUNDARY
        or report.get("checks") != _REPORT_CHECKS
        or report.get("decision") != _REPORT_DECISION
        or report.get("executed_resource_accounting")
        != _EXECUTED_RESOURCE_ACCOUNTING
        or report.get("plan_file_sha256") != sha256_file(plan_path)
        or not isinstance(recovery_resources, Mapping)
        or recovery_resources.get("additional_encoded_binary_bytes_read") != 0
        or recovery_resources.get("additional_encoded_index_bytes_read") != 0
        or recovery_resources.get("additional_encoded_index_rows_read") != 0
        or recovery_resources.get("additional_optimizer_updates") != 0
        or recovery_resources.get("additional_training_sample_consumptions") != 0
        or recovery_resources.get("gpu_use") != 0
        or recovery_resources.get("inference_calls") != 0
        or recovery_resources.get("network_bytes") != 0
        or recovery_resources.get("recovery_marker_bytes") != marker_bytes
        or recovery_resources.get("recovery_report_bytes") != report_bytes
        or recovery_resources.get("verification_bytes_reserved")
        != _MAXIMUM_EVIDENCE_FILE_BYTES
        or recovery_resources.get("recovery_private_write_bytes_upper_bound")
        != marker_bytes + report_bytes + _MAXIMUM_EVIDENCE_FILE_BYTES
        or recovery_resources.get("existing_attempt_private_bytes_read_upper_bound")
        != plan["limits"]["maximum_existing_attempt_private_bytes_read"]
    ):
        raise ValueError("CPU-smoke recovery report changed or failed")
    audit = audit_existing_attempt(repository_root=root)
    if report.get("evidence") != audit:
        raise ValueError("CPU-smoke recovery evidence no longer matches attempt")
    verification = {
        "checks": {
            "all_recovery_code_and_input_bindings_match": True,
            "existing_attempt_byte_inventory_unchanged": True,
            "no_additional_data_read_training_or_retry": True,
            "original_failure_disclosed": True,
            "recovered_cpu_smoke_equality_verified": True,
            "runtime_v4_regression_fix_bound": True,
        },
        "comparison": {
            **dict(audit["checkpoint_audit"]),
            "failed_attempt_inventory_sha256": audit[
                "existing_attempt_snapshot"
            ]["inventory_sha256"],
            "run_fingerprint": audit["final_state"]["run_fingerprint"],
            "stable_metrics_sha256": audit["stable_metrics_sha256"],
        },
        "plan_file_sha256": sha256_file(plan_path),
        "production_eligible": False,
        "raw_training_text_retained": False,
        "report_file_sha256": sha256_file(report_path),
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": VERIFICATION_STATUS,
    }
    failed_smoke.assert_raw_text_free(verification)
    verification["verification_sha256"] = sha256_value(verification)
    verification_path = root / plan["outputs"]["verification"]
    payload_bytes = len(failed_smoke.json_file_bytes(verification))
    written = marker_bytes + report_bytes + payload_bytes
    if written > plan["limits"]["maximum_additional_private_write_bytes"]:
        raise ValueError("CPU-smoke recovery verification write cap exceeded")
    if time.monotonic() - started > plan["limits"]["maximum_runtime_minutes"] * 60:
        raise TimeoutError("CPU-smoke recovery verification runtime cap exceeded")
    failed_smoke._private_write_exclusive(verification_path, verification)
    return verification
