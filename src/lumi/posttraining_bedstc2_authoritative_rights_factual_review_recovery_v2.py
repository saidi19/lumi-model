"""Zero-network recovery for the completed Be-DSTC2 factual pass-1 assessment.

The failed v0.1 runner completed its raw-document replay and factual assessment,
then durably wrote the acquisition verification before raising ``AttributeError``
while resolving ``review.sha256_bytes`` for report serialization.  This module
binds that control-flow witness and materializes a fresh v0.2 report without
opening any acquired raw document.  A separate runner performs the one remaining
independent document pass.
"""

from __future__ import annotations

import ast
import json
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import lumi.posttraining_bedstc2_authoritative_rights_acquisition_v1 as acquisition
import lumi.posttraining_bedstc2_authoritative_rights_factual_review_v1 as review_v1
from .posttraining_bedstc2_upstream_rights_review_v1 import (
    file_sha256,
    finish_marker,
    read_bound_bytes,
    resolve_repo_path,
    sha256_bytes,
    sha256_value,
    start_marker,
    strict_json_bytes,
)

PLAN_SCHEMA_VERSION = (
    "lumi.sft-bedstc2-authoritative-rights-factual-review-recovery-plan.v0.2"
)
REPORT_SCHEMA_VERSION = (
    "lumi.sft-bedstc2-authoritative-rights-factual-review-report.v0.2"
)
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-bedstc2-authoritative-rights-factual-review-verification.v0.2"
)
PLAN_STATUS = "authorized_zero_network_pass1_report_materialization_recovery_v0.2"
REPORT_STATUS = (
    "STOP_public_documents_acquired_rights_chain_unresolved_pending_qualified_review"
)
VERIFICATION_STATUS = (
    "PASS_verified_STOP_public_documents_acquired_rights_chain_unresolved_v0.2"
)

EXPECTED_PLAN_PATH = Path(
    "configs/lumi-4b-sft-bedstc2-authoritative-rights-factual-review-recovery-v0.2.json"
)
EXPECTED_AUTHORIZATION_TEXT = "I approve the bounded zero-network v0.2 recovery."
EXPECTED_AUTHORIZATION_SHA256 = (
    "cea79a22db662a471972c27f2acdf3c92655bb064e7f7fd96006ab19da7bfcbc"
)
EXPECTED_CODE_PATHS = {
    "implementation": (
        "src/lumi/posttraining_bedstc2_authoritative_rights_factual_review_recovery_v2.py"
    ),
    "runner": (
        "scripts/recover_sft_bedstc2_authoritative_rights_factual_report_v2.py"
    ),
    "verifier": (
        "scripts/verify_sft_bedstc2_authoritative_rights_factual_review_v2.py"
    ),
    "test": (
        "tests/test_bedstc2_authoritative_rights_factual_review_recovery_v2.py"
    ),
}
EXPECTED_OUTPUT = {
    "report_path": (
        "evidence/lumi-4b-sft-bedstc2-authoritative-rights-factual-review-v0.2.json"
    ),
    "verification_path": (
        "evidence/lumi-4b-sft-bedstc2-authoritative-rights-factual-review-"
        "verification-v0.2.json"
    ),
}
EXPECTED_LIMITS = {
    "successor_processes": 2,
    "factual_document_passes_total": 2,
    "prior_completed_document_passes": 1,
    "recovery_raw_document_reads": 0,
    "independent_pass2_document_reads": 8,
    "maximum_independent_pass2_document_bytes": 16_777_216,
    "maximum_evidence_write_bytes": 1_048_576,
    "maximum_seconds_per_process": 300,
    "network_bytes": 0,
    "network_calls": 0,
    "gpu_use": 0,
    "inference_calls": 0,
    "source_line_reads": 0,
    "candidate_row_reads": 0,
    "baseline_row_reads": 0,
    "test_object_reads": 0,
}
EXPECTED_RECOVERY_CONTRACT = {
    "failed_v0_1_outputs_preserved_byte_for_byte": True,
    "failed_v0_1_verification_output_absent": True,
    "failed_v0_1_pass_not_retried": True,
    "pass1_assessment_completion_proved_by_control_flow_and_finished_verification": True,
    "recovery_materialization_reads_raw_documents": False,
    "exactly_one_remaining_independent_document_pass_authorized": True,
    "network_reacquisition_or_outreach_authorized": False,
    "promotion_seal_training_or_registry_mutation_authorized": False,
    "production_eligible": False,
}
EXPECTED_INPUTS: dict[str, dict[str, Any]] = {
    "acquisition_marker": {
        "path": (
            "data/private/lumi-sft-bedstc2-authoritative-rights-documents-v0.1/"
            "attempt-001/.acquisition-started-v0.1"
        ),
        "bytes": 60,
        "sha256": "ddee9c9bba5c56fc1075b22f57f524a9d34ac2c91086165ba2433871c64be98c",
    },
    "acquisition_plan": {
        "path": str(acquisition.EXPECTED_PLAN_PATH),
        "bytes": 5_869,
        "sha256": "9209a4c177830133612f2d7bf3b6a234b7fb2cf212f8188128ce0451302a8d90",
    },
    "failed_review_plan": {
        "path": str(review_v1.EXPECTED_PLAN_PATH),
        "bytes": 3_550,
        "sha256": "7979949a989db8750db43d131c1bd85d2c32e0561db6025247e139b0f2aec3c7",
    },
    "failed_review_implementation": {
        "path": "src/lumi/posttraining_bedstc2_authoritative_rights_factual_review_v1.py",
        "bytes": 15_708,
        "sha256": "3795c3362a5a0ea92182111b0a0d3bcbe641ac5875e6c753643414e0815f76f9",
    },
    "failed_review_runner": {
        "path": "scripts/review_sft_bedstc2_authoritative_rights_facts_v1.py",
        "bytes": 5_672,
        "sha256": "2f5be0848c83a21b7c694cd2225e4e2f1bfd93509c3dc426312f50f7a7daaf6b",
    },
    "private_manifest": {
        "path": str(acquisition.EXPECTED_OUTPUT["private_manifest_path"]),
        "bytes": 3_541,
        "sha256": "52b4aa021e1099f7a69661f9ac85f951d44a17db53361ce9888020c64959ef3d",
        "seal_key": "manifest_sha256",
        "internal_sha256": "9a3faf3850702541d2fde9fcd40041e2c9038187aa4f847380b5d0f04a9123c2",
    },
    "acquisition_receipt": {
        "path": str(acquisition.EXPECTED_OUTPUT["receipt_path"]),
        "bytes": 4_318,
        "sha256": "e5b55524128be166681b3e415d551431e4337ed424ef49acfa950e0b303217b0",
        "seal_key": "receipt_sha256",
        "internal_sha256": "3b94ebee422936871a9fd8af09b9f6938c31e1bf647851f5674de10e5689b72e",
    },
    "acquisition_verification": {
        "path": str(acquisition.EXPECTED_OUTPUT["verification_path"]),
        "bytes": 837,
        "sha256": "6378f52c675b9a648c7e94f8dba38864ebf50dfaf5295efb05f8b89238525ec1",
        "seal_key": "verification_sha256",
        "internal_sha256": "417a312abd6ecb193cc16414fa757e4acaf3be3e501f941106da470034ca44f7",
    },
    "failed_report_marker": {
        "path": str(review_v1.EXPECTED_OUTPUT["report_path"]),
        "bytes": 63,
        "sha256": "c7c0930d375e3565b8d72e7d816389604e3f603ad4552d0345f7deb2f1c019b3",
    },
}
EXPECTED_OBSERVATIONS = dict(review_v1.EXPECTED_OBSERVATIONS)
EXPECTED_CRITERIA = dict(review_v1.EXPECTED_CRITERIA)
EXPECTED_REASON_CODES = list(review_v1.EXPECTED_REASON_CODES)
EXPECTED_NEXT_GATE = dict(review_v1.EXPECTED_NEXT_GATE)
EXPECTED_ACQUISITION_ACCOUNTING = {
    "documents": 8,
    "http_requests_started": 8,
    "http_responses_observed": 8,
    "inbound_bytes": 2_167_236,
    "network_retries": 0,
}
EXPECTED_ACQUISITION_VERIFICATION_CHECKS = {
    "all_document_sizes_and_hashes_replayed": True,
    "exact_document_inventory_replayed": True,
    "network_used_during_verification": False,
    "private_manifest_and_receipt_seals_verified": True,
    "production_boundary_preserved": True,
}
REPORT_MARKER = b'{"started":"bedstc2_authoritative_rights_factual_recovery_v0.2"}\n'
VERIFICATION_MARKER = (
    b'{"started":"bedstc2_authoritative_rights_factual_verification_v0.2"}\n'
)


def canonical_json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _seal(value: dict[str, Any], key: str) -> dict[str, Any]:
    value[key] = sha256_value(
        {name: item for name, item in value.items() if name != key}
    )
    return value


def _read_bound(root: Path, name: str) -> bytes:
    binding = EXPECTED_INPUTS[name]
    return read_bound_bytes(
        resolve_repo_path(root, str(binding["path"])),
        expected_bytes=int(binding["bytes"]),
        expected_sha256=str(binding["sha256"]),
    )


def _bound_json(root: Path, name: str) -> dict[str, Any]:
    value = strict_json_bytes(_read_bound(root, name))
    if not isinstance(value, dict):
        raise ValueError(f"bound recovery JSON root changed: {name}")
    binding = EXPECTED_INPUTS[name]
    seal_key = binding.get("seal_key")
    if seal_key is not None:
        unsigned = {key: item for key, item in value.items() if key != seal_key}
        if (
            value.get(seal_key) != binding["internal_sha256"]
            or sha256_value(unsigned) != binding["internal_sha256"]
        ):
            raise ValueError(f"bound recovery JSON seal changed: {name}")
    return value


def _call_name(node: ast.Call) -> str:
    value: ast.expr = node.func
    parts: list[str] = []
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return ".".join(reversed(parts))


def validate_failed_runner_witness(runner_payload: bytes, module_payload: bytes) -> dict[str, int]:
    try:
        runner_tree = ast.parse(runner_payload.decode("utf-8", errors="strict"))
        module_tree = ast.parse(module_payload.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, SyntaxError) as error:
        raise ValueError("failed-runner witness source changed") from error
    locations: dict[str, list[int]] = {}
    for node in ast.walk(runner_tree):
        if isinstance(node, ast.Call):
            locations.setdefault(_call_name(node), []).append(node.lineno)
    expected = {
        "review.start_marker": 52,
        "acquisition.start_marker": 53,
        "acquisition.replay_acquired_inventory": 58,
        "review.assess_payloads": 67,
        "acquisition.finish_marker": 68,
        "review.sha256_bytes": 89,
        "review.finish_marker": 113,
    }
    if any(locations.get(name) != [line] for name, line in expected.items()):
        raise ValueError("failed-runner control-flow witness changed")
    if not (
        expected["review.start_marker"]
        < expected["acquisition.start_marker"]
        < expected["acquisition.replay_acquired_inventory"]
        < expected["review.assess_payloads"]
        < expected["acquisition.finish_marker"]
        < expected["review.sha256_bytes"]
        < expected["review.finish_marker"]
    ):
        raise ValueError("failed-runner control-flow order changed")
    module_bindings: set[str] = set()
    for node in module_tree.body:
        if isinstance(node, ast.ImportFrom):
            module_bindings.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module_bindings.add(node.name)
    if "sha256_bytes" in module_bindings:
        raise ValueError("failed missing-attribute witness no longer holds")
    return expected


def validate_completion_witness(root: Path) -> dict[str, Any]:
    marker = _read_bound(root, "failed_report_marker")
    if marker != review_v1.REPORT_MARKER:
        raise ValueError("failed v0.1 report marker changed")
    call_lines = validate_failed_runner_witness(
        _read_bound(root, "failed_review_runner"),
        _read_bound(root, "failed_review_implementation"),
    )
    manifest = _bound_json(root, "private_manifest")
    receipt = _bound_json(root, "acquisition_receipt")
    verification = _bound_json(root, "acquisition_verification")
    if (
        manifest.get("document_count") != 8
        or manifest.get("document_bytes") != 2_167_236
        or manifest.get("production_eligible") is not False
        or receipt.get("status") != acquisition.RECEIPT_STATUS
        or receipt.get("accounting") != EXPECTED_ACQUISITION_ACCOUNTING
        or receipt.get("documents") != manifest.get("documents")
        or receipt.get("production_eligible") is not False
        or verification.get("status") != acquisition.VERIFICATION_STATUS
        or verification.get("checks") != EXPECTED_ACQUISITION_VERIFICATION_CHECKS
        or verification.get("documents") != 8
        or verification.get("document_bytes") != 2_167_236
        or verification.get("production_eligible") is not False
        or verification.get("qualified_review_pending") is not True
    ):
        raise ValueError("pass-1 completion evidence changed")
    return {
        "call_lines": call_lines,
        "manifest": manifest,
        "receipt": receipt,
        "verification": verification,
    }


def load_plan(path: str | Path, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    resolved = resolve_repo_path(root, path)
    if resolved != resolve_repo_path(root, EXPECTED_PLAN_PATH):
        raise ValueError("v0.2 recovery plan path changed")
    info = resolved.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("v0.2 recovery plan metadata changed")
    value = strict_json_bytes(resolved.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("v0.2 recovery plan root changed")
    if (
        set(value)
        != {
            "schema_version", "status", "authorization_text", "authorization_sha256",
            "limits", "inputs", "recovery_contract", "output", "implementation",
            "runner", "verifier", "test",
        }
        or value.get("schema_version") != PLAN_SCHEMA_VERSION
        or value.get("status") != PLAN_STATUS
        or value.get("authorization_text") != EXPECTED_AUTHORIZATION_TEXT
        or value.get("authorization_sha256") != EXPECTED_AUTHORIZATION_SHA256
        or sha256_bytes(EXPECTED_AUTHORIZATION_TEXT.encode("utf-8"))
        != EXPECTED_AUTHORIZATION_SHA256
        or value.get("limits") != EXPECTED_LIMITS
        or value.get("inputs") != EXPECTED_INPUTS
        or value.get("recovery_contract") != EXPECTED_RECOVERY_CONTRACT
        or value.get("output") != EXPECTED_OUTPUT
    ):
        raise ValueError("v0.2 recovery plan contract changed")
    for name, expected_path in EXPECTED_CODE_PATHS.items():
        binding = value.get(name)
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "sha256"}
            or binding.get("path") != expected_path
            or file_sha256(resolve_repo_path(root, expected_path)) != binding.get("sha256")
        ):
            raise ValueError("v0.2 recovery code binding changed")
    for name in EXPECTED_INPUTS:
        _read_bound(root, name)
    failed_verification = resolve_repo_path(
        root, review_v1.EXPECTED_OUTPUT["verification_path"]
    )
    if failed_verification.exists() or failed_verification.is_symlink():
        raise ValueError("failed v0.1 verification output unexpectedly exists")
    for output in EXPECTED_OUTPUT.values():
        if resolve_repo_path(root, output).parent != resolve_repo_path(root, "evidence"):
            raise ValueError("v0.2 recovery output path changed")
    return value


def assemble_report(plan_file_sha256: str, witness: Mapping[str, Any]) -> dict[str, Any]:
    call_lines = witness.get("call_lines")
    manifest = witness.get("manifest")
    receipt = witness.get("receipt")
    verification = witness.get("verification")
    if (
        call_lines
        != {
            "review.start_marker": 52,
            "acquisition.start_marker": 53,
            "acquisition.replay_acquired_inventory": 58,
            "review.assess_payloads": 67,
            "acquisition.finish_marker": 68,
            "review.sha256_bytes": 89,
            "review.finish_marker": 113,
        }
        or not isinstance(manifest, Mapping)
        or not isinstance(receipt, Mapping)
        or not isinstance(verification, Mapping)
    ):
        raise ValueError("v0.2 report completion witness changed")
    value = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": REPORT_STATUS,
        "bindings": {
            "recovery_plan_file_sha256": plan_file_sha256,
            "acquisition_marker_file_sha256": EXPECTED_INPUTS["acquisition_marker"]["sha256"],
            "failed_review_plan_file_sha256": EXPECTED_INPUTS["failed_review_plan"]["sha256"],
            "failed_review_runner_file_sha256": EXPECTED_INPUTS["failed_review_runner"]["sha256"],
            "failed_review_implementation_file_sha256": EXPECTED_INPUTS["failed_review_implementation"]["sha256"],
            "failed_report_marker_file_sha256": EXPECTED_INPUTS["failed_report_marker"]["sha256"],
            "private_manifest_file_sha256": EXPECTED_INPUTS["private_manifest"]["sha256"],
            "private_manifest_internal_sha256": manifest["manifest_sha256"],
            "acquisition_receipt_file_sha256": EXPECTED_INPUTS["acquisition_receipt"]["sha256"],
            "acquisition_receipt_internal_sha256": receipt["receipt_sha256"],
            "acquisition_verification_file_sha256": EXPECTED_INPUTS["acquisition_verification"]["sha256"],
            "acquisition_verification_internal_sha256": verification["verification_sha256"],
        },
        "recovery": {
            "failed_v0_1_report_marker_preserved": True,
            "failed_v0_1_verification_output_absent": True,
            "pass1_document_replay_and_assessment_completed_before_failure": True,
            "failure_stage": "report_serialization_missing_review_sha256_bytes_attribute",
            "assessment_call_line": call_lines["review.assess_payloads"],
            "finished_acquisition_verification_line": call_lines["acquisition.finish_marker"],
            "failing_attribute_call_line": call_lines["review.sha256_bytes"],
            "raw_documents_read_during_recovery": 0,
            "network_bytes_during_recovery": 0,
        },
        "document_accounting": {
            "documents_in_completed_pass1": 8,
            "document_bytes_in_completed_pass1": 2_167_236,
            "factual_document_passes_completed": 1,
            "remaining_independent_document_passes": 1,
        },
        "observations": EXPECTED_OBSERVATIONS,
        "rights_assessment": {
            "criteria": EXPECTED_CRITERIA,
            "criteria_satisfied": 0,
            "criteria_unresolved": 5,
        },
        "reason_codes": EXPECTED_REASON_CODES,
        "document_text_or_snippets_retained": False,
        "legal_conclusion_made": False,
        "qualified_review_pending": True,
        "production_eligible": False,
        "promotion_seal_or_training_performed": False,
        "next_gate": EXPECTED_NEXT_GATE,
        "report_sha256": "",
    }
    return _seal(value, "report_sha256")


def validate_report(value: Mapping[str, Any]) -> None:
    bindings = value.get("bindings")
    expected_bound_values = {
        "acquisition_marker_file_sha256": EXPECTED_INPUTS["acquisition_marker"]["sha256"],
        "failed_review_plan_file_sha256": EXPECTED_INPUTS["failed_review_plan"]["sha256"],
        "failed_review_runner_file_sha256": EXPECTED_INPUTS["failed_review_runner"]["sha256"],
        "failed_review_implementation_file_sha256": EXPECTED_INPUTS["failed_review_implementation"]["sha256"],
        "failed_report_marker_file_sha256": EXPECTED_INPUTS["failed_report_marker"]["sha256"],
        "private_manifest_file_sha256": EXPECTED_INPUTS["private_manifest"]["sha256"],
        "private_manifest_internal_sha256": EXPECTED_INPUTS["private_manifest"]["internal_sha256"],
        "acquisition_receipt_file_sha256": EXPECTED_INPUTS["acquisition_receipt"]["sha256"],
        "acquisition_receipt_internal_sha256": EXPECTED_INPUTS["acquisition_receipt"]["internal_sha256"],
        "acquisition_verification_file_sha256": EXPECTED_INPUTS["acquisition_verification"]["sha256"],
        "acquisition_verification_internal_sha256": EXPECTED_INPUTS["acquisition_verification"]["internal_sha256"],
    }
    expected_recovery = {
        "failed_v0_1_report_marker_preserved": True,
        "failed_v0_1_verification_output_absent": True,
        "pass1_document_replay_and_assessment_completed_before_failure": True,
        "failure_stage": "report_serialization_missing_review_sha256_bytes_attribute",
        "assessment_call_line": 67,
        "finished_acquisition_verification_line": 68,
        "failing_attribute_call_line": 89,
        "raw_documents_read_during_recovery": 0,
        "network_bytes_during_recovery": 0,
    }
    expected_accounting = {
        "documents_in_completed_pass1": 8,
        "document_bytes_in_completed_pass1": 2_167_236,
        "factual_document_passes_completed": 1,
        "remaining_independent_document_passes": 1,
    }
    if (
        value.get("schema_version") != REPORT_SCHEMA_VERSION
        or value.get("status") != REPORT_STATUS
        or not isinstance(bindings, Mapping)
        or set(bindings) != {"recovery_plan_file_sha256", *expected_bound_values}
        or not isinstance(bindings.get("recovery_plan_file_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", bindings["recovery_plan_file_sha256"]) is None
        or any(bindings.get(key) != item for key, item in expected_bound_values.items())
        or value.get("recovery") != expected_recovery
        or value.get("document_accounting") != expected_accounting
        or value.get("observations") != EXPECTED_OBSERVATIONS
        or value.get("rights_assessment")
        != {
            "criteria": EXPECTED_CRITERIA,
            "criteria_satisfied": 0,
            "criteria_unresolved": 5,
        }
        or value.get("reason_codes") != EXPECTED_REASON_CODES
        or value.get("recovery", {}).get("raw_documents_read_during_recovery") != 0
        or value.get("document_accounting", {}).get("factual_document_passes_completed") != 1
        or value.get("document_text_or_snippets_retained") is not False
        or value.get("legal_conclusion_made") is not False
        or value.get("production_eligible") is not False
        or value.get("qualified_review_pending") is not True
        or value.get("promotion_seal_or_training_performed") is not False
        or value.get("next_gate") != EXPECTED_NEXT_GATE
        or value.get("report_sha256")
        != sha256_value({key: item for key, item in value.items() if key != "report_sha256"})
    ):
        raise ValueError("v0.2 factual recovery report changed")


def assemble_verification(
    report: Mapping[str, Any], *, report_file_sha256: str, document_bytes: int
) -> dict[str, Any]:
    validate_report(report)
    if document_bytes != 2_167_236:
        raise ValueError("v0.2 independent document byte count changed")
    value = {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": VERIFICATION_STATUS,
        "checks": {
            "pass1_completion_witness_replayed_without_document_reads": True,
            "second_independent_document_pass_completed": True,
            "all_document_hashes_replayed_in_pass2": True,
            "factual_observations_independently_reproduced": True,
            "rights_criteria_remain_five_of_five_unresolved": True,
            "zero_network_gpu_inference_promotion_seal_or_training": True,
            "production_boundary_preserved": True,
        },
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": report["report_sha256"],
        "document_accounting": {
            "pass1_documents": 8,
            "pass1_document_bytes": 2_167_236,
            "recovery_materialization_document_reads": 0,
            "pass2_documents_read": 8,
            "pass2_document_bytes_read": document_bytes,
            "total_factual_document_passes": 2,
            "network_bytes": 0,
        },
        "production_eligible": False,
        "qualified_review_pending": True,
        "verification_sha256": "",
    }
    return _seal(value, "verification_sha256")


def read_report(path: Path) -> tuple[dict[str, Any], bytes]:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("v0.2 report metadata changed")
    payload = read_bound_bytes(
        path, expected_bytes=info.st_size, expected_sha256=file_sha256(path)
    )
    value = strict_json_bytes(payload)
    if not isinstance(value, dict) or canonical_json_file_bytes(value) != payload:
        raise ValueError("v0.2 report encoding changed")
    validate_report(value)
    return value, payload


def write_marker_payload(path: Path, marker: bytes, value: Mapping[str, Any]) -> bytes:
    payload = canonical_json_file_bytes(value)
    identity = start_marker(path, marker)
    finish_marker(path, marker, payload, identity)
    return payload


__all__ = [name for name in globals() if name.isupper()] + [
    "assemble_report", "assemble_verification", "canonical_json_file_bytes",
    "load_plan", "read_report", "validate_completion_witness",
    "validate_failed_runner_witness", "validate_report", "write_marker_payload",
]
