"""Two-pass, zero-network factual review of acquired Be-DSTC2 rights documents."""

from __future__ import annotations

import json
import socket
import stat
import urllib.request
from contextlib import contextmanager
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import lumi.posttraining_bedstc2_authoritative_rights_acquisition_v1 as acquisition
from .posttraining_bedstc2_upstream_rights_review_v1 import (
    file_sha256,
    finish_marker,
    resolve_repo_path,
    sha256_value,
    start_marker,
    strict_json_bytes,
)

PLAN_SCHEMA_VERSION = (
    "lumi.sft-bedstc2-authoritative-rights-factual-review-plan.v0.1"
)
REPORT_SCHEMA_VERSION = (
    "lumi.sft-bedstc2-authoritative-rights-factual-review-report.v0.1"
)
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-bedstc2-authoritative-rights-factual-review-verification.v0.1"
)
PLAN_STATUS = "authorized_two_pass_zero_network_rights_factual_review"
REPORT_STATUS = (
    "STOP_public_documents_acquired_rights_chain_unresolved_pending_qualified_review"
)
VERIFICATION_STATUS = (
    "PASS_verified_STOP_public_documents_acquired_rights_chain_unresolved"
)
EXPECTED_PLAN_PATH = Path(
    "configs/lumi-4b-sft-bedstc2-authoritative-rights-factual-review-v0.1.json"
)
EXPECTED_ACQUISITION_PLAN_SHA256 = (
    "9209a4c177830133612f2d7bf3b6a234b7fb2cf212f8188128ce0451302a8d90"
)
EXPECTED_CODE_PATHS = {
    "implementation": (
        "src/lumi/posttraining_bedstc2_authoritative_rights_factual_review_v1.py"
    ),
    "runner": "scripts/review_sft_bedstc2_authoritative_rights_facts_v1.py",
    "verifier": (
        "scripts/verify_sft_bedstc2_authoritative_rights_factual_review_v1.py"
    ),
    "test": "tests/test_bedstc2_authoritative_rights_factual_review_v1.py",
}
EXPECTED_OUTPUT = {
    "report_path": (
        "evidence/lumi-4b-sft-bedstc2-authoritative-rights-factual-review-v0.1.json"
    ),
    "verification_path": (
        "evidence/lumi-4b-sft-bedstc2-authoritative-rights-factual-review-"
        "verification-v0.1.json"
    ),
}
EXPECTED_LIMITS = {
    "passes": 2,
    "document_reads_per_pass": 8,
    "maximum_document_bytes_per_pass": 16_777_216,
    "maximum_document_bytes_total": 33_554_432,
    "network_bytes": 0,
    "network_calls": 0,
    "gpu_use": 0,
    "inference_calls": 0,
    "source_line_reads": 0,
    "candidate_row_reads": 0,
    "baseline_row_reads": 0,
    "test_object_reads": 0,
    "maximum_seconds_per_pass": 300,
}
EXPECTED_CRITERIA = {
    "upstream_dstc2_licensor_or_rightsholder_authoritatively_identified": False,
    "upstream_dstc2_license_or_permission_bound_to_exact_objects": False,
    "code_mixed_bengali_object_scope_and_relicense_authority_bound": False,
    "derivative_translation_commercial_and_ml_use_grant_bound": False,
    "complete_attribution_notice_and_reciprocity_duties_bound": False,
}
EXPECTED_REASON_CODES = [
    "public_download_or_research_availability_is_not_a_commercial_rights_grant",
    "repository_license_scope_over_inherited_dstc2_content_not_established",
    "downstream_license_does_not_establish_upstream_relicense_authority",
    "exact_bengali_object_scope_not_bound_to_authoritative_terms",
    "qualified_rights_owner_or_legal_approval_absent",
]
EXPECTED_OBSERVATIONS = {
    "official_microsoft_event_page_acquired": True,
    "dstc2_acl_anthology_landing_page_acquired": True,
    "organizer_associated_exact_revision_readme_acquired": True,
    "organizer_associated_exact_revision_gpl3_text_acquired": True,
    "dstc2_primary_paper_pdf_acquired": True,
    "code_mixed_dialog_paper_pdf_acquired": True,
    "code_mixed_dialog_exact_revision_readme_attributes_dstc2": True,
    "code_mixed_dialog_exact_revision_apache2_text_acquired": True,
    "authoritative_corpus_specific_dstc2_grant_document_in_inventory": False,
    "rights_owner_written_confirmation_in_inventory": False,
    "qualified_legal_approval_in_inventory": False,
}
EXPECTED_NEXT_GATE = {
    "action": "qualified_rights_owner_confirmation_or_legal_review",
    "separate_authorization_required": True,
    "promotion_seal_or_training_authorized": False,
}
REPORT_MARKER = b'{"started":"bedstc2_authoritative_rights_factual_review_v0.1"}\n'
VERIFICATION_MARKER = (
    b'{"started":"bedstc2_authoritative_rights_factual_verification_v0.1"}\n'
)


def canonical_json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


@contextmanager
def network_disabled() -> Any:
    """Fail closed if either factual pass accidentally attempts network access."""

    original_create_connection = socket.create_connection
    original_getaddrinfo = socket.getaddrinfo
    original_urlopen = urllib.request.urlopen

    def blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("network is disabled during rights factual review")

    socket.create_connection = blocked
    socket.getaddrinfo = blocked
    urllib.request.urlopen = blocked
    try:
        yield
    finally:
        socket.create_connection = original_create_connection
        socket.getaddrinfo = original_getaddrinfo
        urllib.request.urlopen = original_urlopen


def _seal(value: dict[str, Any], key: str) -> dict[str, Any]:
    value[key] = sha256_value({name: item for name, item in value.items() if name != key})
    return value


def load_plan(path: str | Path, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    resolved = resolve_repo_path(root, path)
    if resolved != resolve_repo_path(root, EXPECTED_PLAN_PATH):
        raise ValueError("rights factual-review plan path changed")
    info = resolved.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("rights factual-review plan metadata changed")
    value = strict_json_bytes(resolved.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("rights factual-review plan root changed")
    if (
        set(value)
        != {
            "schema_version", "status", "authorization_sha256", "acquisition_plan",
            "limits", "assessment_contract", "output", "implementation", "runner",
            "verifier", "test",
        }
        or value.get("schema_version") != PLAN_SCHEMA_VERSION
        or value.get("status") != PLAN_STATUS
        or value.get("authorization_sha256") != acquisition.EXPECTED_AUTHORIZATION_SHA256
        or value.get("limits") != EXPECTED_LIMITS
        or value.get("output") != EXPECTED_OUTPUT
        or value.get("assessment_contract")
        != {
            "expected_status": REPORT_STATUS,
            "criteria": EXPECTED_CRITERIA,
            "observations": EXPECTED_OBSERVATIONS,
            "reason_codes": EXPECTED_REASON_CODES,
            "production_eligible": False,
            "qualified_review_pending": True,
        }
        or value.get("acquisition_plan")
        != {
            "path": str(acquisition.EXPECTED_PLAN_PATH),
            "sha256": EXPECTED_ACQUISITION_PLAN_SHA256,
        }
    ):
        raise ValueError("rights factual-review plan contract changed")
    if file_sha256(resolve_repo_path(root, acquisition.EXPECTED_PLAN_PATH)) != EXPECTED_ACQUISITION_PLAN_SHA256:
        raise ValueError("authoritative-rights acquisition plan changed")
    for name, expected_path in EXPECTED_CODE_PATHS.items():
        binding = value.get(name)
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "sha256"}
            or binding.get("path") != expected_path
            or file_sha256(resolve_repo_path(root, expected_path)) != binding.get("sha256")
        ):
            raise ValueError("rights factual-review code binding changed")
    return value


def _decode_text(payload: bytes, identity: str) -> str:
    try:
        return payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError(f"strict UTF-8 document changed: {identity}") from error


def assess_payloads(payloads: list[bytes]) -> dict[str, Any]:
    if len(payloads) != len(acquisition.EXPECTED_DOCUMENTS):
        raise ValueError("rights factual document count changed")
    by_id = {
        spec["id"]: payload
        for spec, payload in zip(acquisition.EXPECTED_DOCUMENTS, payloads, strict=True)
    }
    microsoft = _decode_text(by_id["microsoft_dstc_event"], "microsoft").casefold()
    dstc2_landing = _decode_text(
        by_id["dstc2_acl_anthology_landing"], "dstc2-landing"
    ).casefold()
    matthen_readme = _decode_text(
        by_id["matthen_dstc_readme_exact_revision"], "matthen-readme"
    ).casefold()
    matthen_license = _decode_text(
        by_id["matthen_dstc_license_exact_revision"], "matthen-license"
    ).casefold()
    code_readme = _decode_text(
        by_id["code_mixed_dialog_readme_exact_revision"], "code-mixed-readme"
    ).casefold()
    code_license = _decode_text(
        by_id["code_mixed_dialog_license_exact_revision"], "code-mixed-license"
    ).casefold()
    gates = {
        "official_microsoft_event_page_acquired": (
            "dialog state tracking challenge" in microsoft
        ),
        "dstc2_acl_anthology_landing_page_acquired": (
            "second dialog state tracking challenge" in dstc2_landing
            or "dialog state tracking challenge" in dstc2_landing
        ),
        "organizer_associated_exact_revision_readme_acquired": (
            "dialog state tracking" in matthen_readme and "dstc2" in matthen_readme
        ),
        "organizer_associated_exact_revision_gpl3_text_acquired": (
            "gnu general public license" in matthen_license
            and "version 3" in matthen_license
        ),
        "dstc2_primary_paper_pdf_acquired": by_id["dstc2_primary_paper"].startswith(
            b"%PDF-"
        ),
        "code_mixed_dialog_paper_pdf_acquired": by_id[
            "code_mixed_dialog_paper"
        ].startswith(b"%PDF-"),
        "code_mixed_dialog_exact_revision_readme_attributes_dstc2": (
            "dstc2" in code_readme and "code-mixed" in code_readme
        ),
        "code_mixed_dialog_exact_revision_apache2_text_acquired": (
            "apache license" in code_license and "version 2.0" in code_license
        ),
        "authoritative_corpus_specific_dstc2_grant_document_in_inventory": False,
        "rights_owner_written_confirmation_in_inventory": False,
        "qualified_legal_approval_in_inventory": False,
    }
    if gates != EXPECTED_OBSERVATIONS:
        failed = sorted(name for name, result in gates.items() if result != EXPECTED_OBSERVATIONS[name])
        raise ValueError(f"rights factual identity gates changed: {','.join(failed)}")
    return {
        "observations": gates,
        "criteria": dict(EXPECTED_CRITERIA),
        "criteria_satisfied": 0,
        "criteria_unresolved": 5,
    }


def assemble_report(
    assessment: Mapping[str, Any],
    *,
    acquisition_plan_sha256: str,
    manifest_file_sha256: str,
    manifest_internal_sha256: str,
    receipt_file_sha256: str,
    receipt_internal_sha256: str,
    acquisition_verification_file_sha256: str,
    acquisition_verification_internal_sha256: str,
    document_bytes: int,
) -> dict[str, Any]:
    if assessment != {
        "observations": EXPECTED_OBSERVATIONS,
        "criteria": EXPECTED_CRITERIA,
        "criteria_satisfied": 0,
        "criteria_unresolved": 5,
    }:
        raise ValueError("rights factual assessment changed")
    value = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": REPORT_STATUS,
        "bindings": {
            "acquisition_plan_file_sha256": acquisition_plan_sha256,
            "private_manifest_file_sha256": manifest_file_sha256,
            "private_manifest_internal_sha256": manifest_internal_sha256,
            "acquisition_receipt_file_sha256": receipt_file_sha256,
            "acquisition_receipt_internal_sha256": receipt_internal_sha256,
            "acquisition_verification_file_sha256": acquisition_verification_file_sha256,
            "acquisition_verification_internal_sha256": acquisition_verification_internal_sha256,
        },
        "document_accounting": {
            "documents_read": len(acquisition.EXPECTED_DOCUMENTS),
            "document_bytes_read": document_bytes,
            "passes_completed_by_this_report": 1,
            "network_bytes": 0,
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
    seal = value.get("report_sha256")
    if (
        value.get("schema_version") != REPORT_SCHEMA_VERSION
        or value.get("status") != REPORT_STATUS
        or value.get("observations") != EXPECTED_OBSERVATIONS
        or value.get("rights_assessment")
        != {"criteria": EXPECTED_CRITERIA, "criteria_satisfied": 0, "criteria_unresolved": 5}
        or value.get("reason_codes") != EXPECTED_REASON_CODES
        or value.get("production_eligible") is not False
        or value.get("qualified_review_pending") is not True
        or value.get("document_text_or_snippets_retained") is not False
        or seal != sha256_value({key: item for key, item in value.items() if key != "report_sha256"})
    ):
        raise ValueError("rights factual report changed")


def assemble_verification(
    report: Mapping[str, Any],
    *,
    report_file_sha256: str,
    document_bytes: int,
) -> dict[str, Any]:
    validate_report(report)
    value = {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": VERIFICATION_STATUS,
        "checks": {
            "second_independent_document_pass_completed": True,
            "all_document_hashes_replayed": True,
            "factual_observations_independently_reproduced": True,
            "rights_criteria_remain_five_of_five_unresolved": True,
            "no_network_gpu_inference_or_training": True,
            "production_boundary_preserved": True,
        },
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": report["report_sha256"],
        "document_accounting": {
            "documents_read_this_pass": len(acquisition.EXPECTED_DOCUMENTS),
            "document_bytes_read_this_pass": document_bytes,
            "total_factual_passes": 2,
            "network_bytes": 0,
        },
        "production_eligible": False,
        "qualified_review_pending": True,
        "verification_sha256": "",
    }
    return _seal(value, "verification_sha256")


def read_report(path: Path) -> tuple[dict[str, Any], bytes]:
    return acquisition.read_json_file(path)


def write_marker_payload(path: Path, marker: bytes, value: Mapping[str, Any]) -> bytes:
    payload = canonical_json_file_bytes(value)
    identity = start_marker(path, marker)
    finish_marker(path, marker, payload, identity)
    return payload


__all__ = [name for name in globals() if name.isupper()] + [
    "assemble_report", "assemble_verification", "assess_payloads",
    "canonical_json_file_bytes", "load_plan", "network_disabled", "read_report", "validate_report",
    "write_marker_payload",
]
