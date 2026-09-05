"""Fail-closed rights-map audit for the reviewed OpenR1 visible-final candidate."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .posttraining_first_party import file_sha256
from .posttraining_openr1_full_run import report_sha256, sha256_value

PLAN_SCHEMA_VERSION = "lumi.sft-openr1-rights-map-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-openr1-rights-map-report.v0.1"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-openr1-rights-map-verification.v0.1"

EXPECTED_AUTHORIZATION = {
    "external_api_spend_authorized": False,
    "paid_gpu_authorized": False,
    "production_dataset_promotion_authorized": False,
    "production_dataset_seal_authorized": False,
    "production_sft_authorized": False,
    "purpose": "exact_reviewed_openr1_problem_source_rights_map_audit_only",
}
EXPECTED_COUNTS = {
    "amc_aime": 46,
    "aops_forum": 183,
    "cn_contest": 405,
    "inequalities": 16,
    "number_theory": 5,
    "olympiads": 1_996,
    "olympiads_ref": 5,
}
EXPECTED_DATASET_DECLARATIONS = {
    "numinamath_1_5": {
        "declared_license": "apache-2.0",
        "description": (
            "competition problems collected from online exam PDFs, discussion forums, "
            "official olympiad sites, and manually curated data partners"
        ),
        "url": "https://huggingface.co/datasets/AI-MO/NuminaMath-1.5",
    },
    "openr1_math_220k": {
        "declared_license": "apache-2.0",
        "derived_problem_source": "AI-MO/NuminaMath-1.5",
        "url": "https://huggingface.co/datasets/open-r1/OpenR1-Math-220k",
    },
}
EXPECTED_RIGHTS_MAP = {
    "amc_aime": {
        "decision": "quarantine_pending_documented_downstream_rights",
        "evidence_urls": [
            "https://maa.org/student-programs/amc/maa-american-mathematics-competitions-policies/",
            "https://maa.org/maa-press/",
        ],
        "reason_codes": [
            "competition_owner_permission_not_documented_for_lumi",
            "dataset_level_license_does_not_identify_upstream_grant",
        ],
        "records": 46,
    },
    "aops_forum": {
        "decision": "quarantine_pending_documented_downstream_rights",
        "evidence_urls": ["https://artofproblemsolving.com/company/tos"],
        "reason_codes": [
            "forum_terms_do_not_document_a_downstream_grant_to_lumi",
            "commercial_exploitation_restriction_requires_resolution",
        ],
        "records": 183,
    },
    "cn_contest": {
        "decision": "quarantine_pending_documented_downstream_rights",
        "evidence_urls": [
            "https://huggingface.co/datasets/AI-MO/NuminaMath-1.5"
        ],
        "reason_codes": [
            "data_partner_identity_and_rights_chain_not_documented_per_row"
        ],
        "records": 405,
    },
    "inequalities": {
        "decision": "quarantine_pending_documented_downstream_rights",
        "evidence_urls": [
            "https://huggingface.co/datasets/AI-MO/NuminaMath-1.5"
        ],
        "reason_codes": [
            "data_partner_identity_and_rights_chain_not_documented_per_row"
        ],
        "records": 16,
    },
    "number_theory": {
        "decision": "quarantine_pending_documented_downstream_rights",
        "evidence_urls": [
            "https://huggingface.co/datasets/AI-MO/NuminaMath-1.5"
        ],
        "reason_codes": [
            "data_partner_identity_and_rights_chain_not_documented_per_row"
        ],
        "records": 5,
    },
    "olympiads": {
        "decision": "quarantine_pending_documented_downstream_rights",
        "evidence_urls": [
            "https://huggingface.co/datasets/AI-MO/NuminaMath-1.5"
        ],
        "reason_codes": [
            "exact_competition_owner_and_license_not_mapped_per_row"
        ],
        "records": 1_996,
    },
    "olympiads_ref": {
        "decision": "quarantine_pending_documented_downstream_rights",
        "evidence_urls": [
            "https://huggingface.co/datasets/AI-MO/NuminaMath-1.5"
        ],
        "reason_codes": [
            "official_site_access_does_not_establish_reuse_permission_per_row"
        ],
        "records": 5,
    },
}
EXPECTED_LIMITS = {"maximum_records": 2_656, "maximum_runtime_minutes": 60}
EXPECTED_BINDINGS = frozenset(
    {
        "attribution_manifest",
        "audit_implementation",
        "audit_runner",
        "audit_verifier",
        "dataset_readme",
        "owner_review_receipt",
        "visible_final_records",
        "visible_final_report",
        "visible_final_verification",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a SHA-256")
    return value


def resolve_plan_binding(
    repository_root: str | Path,
    binding: Mapping[str, Any],
) -> Path:
    root = Path(repository_root).resolve()
    workspace = root.parent.resolve()
    relative = binding.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("OpenR1 rights binding path is invalid")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError("OpenR1 rights binding escaped the workspace")
    return resolved


def load_rights_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = _load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported OpenR1 rights-map plan")
    if plan.get("status") != "authorized_local_rights_audit_no_promotion":
        raise ValueError("OpenR1 rights-map status changed")
    if plan.get("observed_at") != "2026-08-06":
        raise ValueError("OpenR1 rights-map date changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("OpenR1 rights-map authorization changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("OpenR1 rights-map limits changed")
    if plan.get("dataset_declarations") != EXPECTED_DATASET_DECLARATIONS:
        raise ValueError("OpenR1 rights-map dataset declarations changed")
    if plan.get("rights_map") != EXPECTED_RIGHTS_MAP:
        raise ValueError("OpenR1 rights-map decisions changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("OpenR1 rights-map binding inventory changed")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"OpenR1 rights binding is invalid: {name}")
        expected = _require_sha256(binding.get("sha256"), f"bindings.{name}.sha256")
        resolved = resolve_plan_binding(root, binding)
        if not resolved.is_file() or file_sha256(resolved) != expected:
            raise ValueError(f"OpenR1 rights binding changed: {name}")
    output = plan.get("output")
    if not isinstance(output, Mapping):
        raise ValueError("OpenR1 rights-map output is missing")
    report_path = output.get("report_path")
    verification_path = output.get("verification_path")
    if not isinstance(report_path, str) or not report_path.startswith("evidence/"):
        raise ValueError("OpenR1 rights report path is invalid")
    if not isinstance(verification_path, str) or not verification_path.startswith(
        "evidence/"
    ):
        raise ValueError("OpenR1 rights verification path is invalid")
    if output.get("production_promotion_permitted") is not False:
        raise ValueError("OpenR1 rights audit cannot promote data")
    return plan


def audit_attributions(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(rows) != EXPECTED_LIMITS["maximum_records"]:
        raise ValueError("OpenR1 rights attribution count changed")
    counts: Counter[str] = Counter()
    rights_status: Counter[str] = Counter()
    record_ids: set[str] = set()
    for row in rows:
        record_id = row.get("record_id")
        source = row.get("upstream_problem_source")
        status = row.get("rights_status")
        if not isinstance(record_id, str) or not record_id or record_id in record_ids:
            raise ValueError("OpenR1 rights attribution identity changed")
        if source not in EXPECTED_COUNTS:
            raise ValueError("OpenR1 rights attribution source changed")
        if status != "unresolved_quarantined":
            raise ValueError("OpenR1 rights attribution status changed")
        record_ids.add(record_id)
        counts[str(source)] += 1
        rights_status[str(status)] += 1
    if dict(sorted(counts.items())) != EXPECTED_COUNTS:
        raise ValueError("OpenR1 upstream source distribution changed")
    inventory = [
        {
            "decision": EXPECTED_RIGHTS_MAP[source]["decision"],
            "reason_codes": EXPECTED_RIGHTS_MAP[source]["reason_codes"],
            "records": count,
            "source": source,
        }
        for source, count in sorted(counts.items())
    ]
    return {
        "inventory": inventory,
        "inventory_sha256": sha256_value(inventory),
        "records": len(rows),
        "rights_cleared_records": 0,
        "rights_status_counts": dict(sorted(rights_status.items())),
        "source_labels": len(counts),
        "unresolved_records": len(rows),
    }


def build_rights_report(
    plan: Mapping[str, Any],
    *,
    audit: Mapping[str, Any],
    plan_file_sha256: str,
) -> dict[str, Any]:
    _require_sha256(plan_file_sha256, "plan_file_sha256")
    report: dict[str, Any] = {
        "authorization": plan["authorization"],
        "checks": {
            "dataset_level_license_declarations_recorded": True,
            "owner_review_receipt_bound": True,
            "per_source_rights_chain_required": True,
            "raw_training_text_retained": False,
            "unresolved_sources_fail_closed": True,
        },
        "dataset_declarations": plan["dataset_declarations"],
        "legal_note": (
            "Internal provenance and licensing risk screen only; not legal advice. "
            "Dataset-level license metadata was not treated as proof of upstream "
            "third-party rights for each problem statement."
        ),
        "observed_at": "2026-08-06",
        "plan_file_sha256": plan_file_sha256,
        "production_eligible": False,
        "promotion_eligible_records": 0,
        "remaining_blockers": [
            "documented_upstream_rights_or_owner_risk_acceptance_required",
            "source_registry_promotion_not_authorized",
            "production_dataset_seal_and_sft_not_authorized",
        ],
        "rights_map": plan["rights_map"],
        "schema_version": REPORT_SCHEMA_VERSION,
        "source_audit": dict(audit),
        "status": "verified_inventory_all_openr1_problem_sources_quarantined",
    }
    report["report_sha256"] = report_sha256(report)
    return report


def validate_rights_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported OpenR1 rights report")
    if report.get("report_sha256") != report_sha256(report):
        raise ValueError("OpenR1 rights report seal changed")
    if report.get("production_eligible") is not False:
        raise ValueError("OpenR1 rights report crossed production eligibility")
    if report.get("promotion_eligible_records") != 0:
        raise ValueError("OpenR1 rights report promoted unresolved records")
    if report.get("rights_map") != EXPECTED_RIGHTS_MAP:
        raise ValueError("OpenR1 rights report map changed")
    if report.get("checks", {}).get("raw_training_text_retained") is not False:
        raise ValueError("OpenR1 rights report retained raw training text")


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite OpenR1 rights artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)

