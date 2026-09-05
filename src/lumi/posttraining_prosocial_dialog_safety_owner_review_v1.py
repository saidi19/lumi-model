"""Materialize and verify the human-owner decision for ProsocialDialog."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .posttraining_prosocial_dialog_safety_curation_v1 import file_sha256, sha256_value

PLAN_SCHEMA_VERSION = "lumi.sft-prosocial-dialog-safety-owner-review-plan.v0.1"
DECISIONS_SCHEMA_VERSION = "lumi.sft-prosocial-dialog-safety-owner-decisions.v0.1"
RECEIPT_SCHEMA_VERSION = "lumi.sft-prosocial-dialog-safety-owner-review-receipt.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-prosocial-dialog-safety-owner-review-verification.v0.1"
)
EXPECTED_OWNER_REQUEST = "Accept all 32 rows"
EXPECTED_OWNER_REQUEST_SHA256 = (
    "c23f6a164834290b989d3a4898c8b451d6ef490f0b96197b0ed60ca825319da3"
)
EXPECTED_SAMPLE_SHA256 = (
    "e0b140526c7f1c781323e89716e216f6ec9098fa2182f0b3a5c886a4a2a72d2b"
)
EXPECTED_AUTHORIZATION = {
    "automated_or_ai_owner_approval_authorized": False,
    "candidate_registry_change_authorized": False,
    "dataset_seal_authorized": False,
    "download_authorized": False,
    "external_api_authorized": False,
    "gpu_authorized": False,
    "human_owner_decision_materialization_authorized": True,
    "inference_authorized": False,
    "network_authorized": False,
    "production_promotion_authorized": False,
    "sft_training_authorized": False,
}
_UTC = re.compile(
    r"^20[0-9]{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])T"
    r"(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z$"
)


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite owner-review artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_plan(path: str | Path, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = _load_json(path)
    if not isinstance(plan, dict) or plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("ProsocialDialog owner-review plan schema changed")
    if plan.get("status") != "authorized_human_owner_review_finalization":
        raise ValueError("ProsocialDialog owner review is not authorized")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("ProsocialDialog owner-review authorization widened")
    if plan.get("owner_request_text") != EXPECTED_OWNER_REQUEST:
        raise ValueError("ProsocialDialog owner request changed")
    if plan.get("owner_request_sha256") != EXPECTED_OWNER_REQUEST_SHA256:
        raise ValueError("ProsocialDialog owner request hash changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("ProsocialDialog owner-review bindings missing")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"invalid ProsocialDialog owner-review binding: {name}")
        resolved = (root / str(binding.get("path"))).resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError(f"ProsocialDialog owner-review binding missing: {name}")
        if file_sha256(resolved) != binding.get("sha256"):
            raise ValueError(f"ProsocialDialog owner-review binding changed: {name}")
    return plan


def build_receipt(
    plan: Mapping[str, Any], *, plan_file_sha256: str, repository_root: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    bindings = plan["bindings"]
    sample = _load_json(root / bindings["owner_sample"]["path"])
    if not isinstance(sample, list) or len(sample) != 32:
        raise ValueError("ProsocialDialog owner sample must contain exactly 32 rows")
    if sha256_value(sample) != EXPECTED_SAMPLE_SHA256:
        raise ValueError("ProsocialDialog owner sample content changed")
    sample_ids = {str(row.get("record_id")) for row in sample}
    if len(sample_ids) != 32:
        raise ValueError("ProsocialDialog owner sample IDs are not unique")

    decisions = _load_json(root / bindings["decisions"]["path"])
    if not isinstance(decisions, Mapping) or decisions.get("schema_version") != DECISIONS_SCHEMA_VERSION:
        raise ValueError("ProsocialDialog owner decision schema changed")
    if decisions.get("owner_request_sha256") != EXPECTED_OWNER_REQUEST_SHA256:
        raise ValueError("ProsocialDialog owner decision request binding changed")
    if decisions.get("sample_sha256") != EXPECTED_SAMPLE_SHA256:
        raise ValueError("ProsocialDialog owner decision sample binding changed")
    if decisions.get("reviewer_alias") != "owner-01":
        raise ValueError("ProsocialDialog owner reviewer changed")
    reviewed_at = decisions.get("reviewed_at")
    if not isinstance(reviewed_at, str) or _UTC.fullmatch(reviewed_at) is None:
        raise ValueError("ProsocialDialog owner decision timestamp is invalid")
    rows = decisions.get("records")
    if not isinstance(rows, list) or len(rows) != 32:
        raise ValueError("ProsocialDialog owner decisions must cover 32 rows")
    decision_ids: set[str] = set()
    for decision in rows:
        if not isinstance(decision, Mapping) or set(decision) != {"decision", "note", "record_id"}:
            raise ValueError("ProsocialDialog owner decision fields changed")
        if decision.get("decision") != "accept" or decision.get("note") != "":
            raise ValueError("ProsocialDialog owner decision is not exact all-accept")
        record_id = str(decision.get("record_id"))
        if record_id not in sample_ids or record_id in decision_ids:
            raise ValueError("ProsocialDialog owner decision ID changed or repeated")
        decision_ids.add(record_id)
    if decision_ids != sample_ids:
        raise ValueError("ProsocialDialog owner decision set is incomplete")

    report = _load_json(root / bindings["curation_report"]["path"])
    verification = _load_json(root / bindings["curation_verification"]["path"])
    if report.get("status") != "PASS_nonproduction_candidate_owner_review_pending":
        raise ValueError("ProsocialDialog curation report status changed")
    if report.get("report_sha256") != bindings["curation_report"].get("internal_sha256"):
        raise ValueError("ProsocialDialog curation report internal hash changed")
    if verification.get("status") != "PASS_independent_nonproduction_candidate_verification":
        raise ValueError("ProsocialDialog curation verification status changed")
    if verification.get("verification_sha256") != bindings["curation_verification"].get(
        "internal_sha256"
    ):
        raise ValueError("ProsocialDialog curation verification internal hash changed")
    if report["owner_review"]["packet_sha256"] != EXPECTED_SAMPLE_SHA256:
        raise ValueError("ProsocialDialog report owner sample changed")
    if report["tokens"]["accepted_records"] != 29_264:
        raise ValueError("ProsocialDialog candidate row count changed")
    if report["tokens"]["train_supervised_tokens"] != 899_998:
        raise ValueError("ProsocialDialog train token count changed")
    if report["tokens"]["validation_supervised_tokens"] != 59_999:
        raise ValueError("ProsocialDialog validation token count changed")

    receipt: dict[str, Any] = {
        "accepted_records": 32,
        "all_sampled_rows_accepted": True,
        "authorization": EXPECTED_AUTHORIZATION,
        "bindings": {
            "candidate_file_sha256": bindings["candidate"]["sha256"],
            "curation_report_file_sha256": bindings["curation_report"]["sha256"],
            "curation_report_internal_sha256": bindings["curation_report"][
                "internal_sha256"
            ],
            "curation_verification_file_sha256": bindings["curation_verification"][
                "sha256"
            ],
            "curation_verification_internal_sha256": bindings["curation_verification"][
                "internal_sha256"
            ],
            "decisions_file_sha256": bindings["decisions"]["sha256"],
            "owner_request_sha256": EXPECTED_OWNER_REQUEST_SHA256,
            "owner_sample_file_sha256": bindings["owner_sample"]["sha256"],
            "owner_sample_internal_sha256": EXPECTED_SAMPLE_SHA256,
            "plan_file_sha256": plan_file_sha256,
            "reviewed_record_ids_sha256": sha256_value(sorted(decision_ids)),
        },
        "candidate": {
            "records": 29_264,
            "train_records": 27_463,
            "train_supervised_tokens": 899_998,
            "validation_records": 1_801,
            "validation_supervised_tokens": 59_999,
        },
        "next_gate": {
            "action": "bounded_immutable_source_registry_v0_19_promotion_then_full_mix_reaudit",
            "dataset_seal_or_sft_authorized": False,
            "owner_review_complete": True,
            "registry_change_authorized": False,
        },
        "owner_review_claim": (
            "mixed_provenance_source_human_owner_quality_reviewed_"
            "assistant_responses_human_crowdworker_written"
        ),
        "production_eligible": False,
        "raw_training_text_retained": False,
        "receipt_sha256": "",
        "rejected_record_ids": [],
        "rejected_records": 0,
        "reviewed_at": reviewed_at,
        "reviewer_alias": "owner-01",
        "sample_records": 32,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "owner_accepted_all_32_prosocial_dialog_rows",
    }
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = sha256_value(receipt)
    return receipt


def build_verification(
    receipt: Mapping[str, Any], *, receipt_file_sha256: str
) -> dict[str, Any]:
    unsigned = dict(receipt)
    internal = unsigned.pop("receipt_sha256", None)
    if internal != sha256_value(unsigned):
        raise ValueError("ProsocialDialog owner-review receipt hash changed")
    if receipt.get("accepted_records") != 32 or receipt.get("rejected_records") != 0:
        raise ValueError("ProsocialDialog owner-review outcome changed")
    verification: dict[str, Any] = {
        "checks": {
            "all_32_owner_decisions_are_accept": True,
            "candidate_and_curation_evidence_bound": True,
            "exact_32_row_sample_bound": True,
            "no_raw_training_text_in_receipt": True,
            "no_registry_seal_inference_gpu_network_or_sft": True,
            "owner_review_is_human_claim_not_automated_claim": True,
        },
        "raw_training_text_retained": False,
        "receipt_file_sha256": receipt_file_sha256,
        "receipt_internal_sha256": internal,
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "PASS_owner_accepted_all_32_prosocial_dialog_rows",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification


def finalize(plan_path: str | Path, repository_root: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(repository_root).resolve()
    plan = load_plan(plan_path, root)
    receipt = build_receipt(
        plan, plan_file_sha256=file_sha256(plan_path), repository_root=root
    )
    receipt_path = root / plan["output"]["receipt_path"]
    verification_path = root / plan["output"]["verification_path"]
    _write_json_atomic(receipt_path, receipt)
    verification = build_verification(
        receipt, receipt_file_sha256=file_sha256(receipt_path)
    )
    _write_json_atomic(verification_path, verification)
    return receipt, verification


__all__ = ["build_receipt", "build_verification", "finalize", "load_plan"]
