"""Materialize and verify the human-owner decision for BillSum writing data."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .posttraining_billsum_writing_curation_v1 import file_sha256, sha256_value

PLAN_SCHEMA_VERSION = "lumi.sft-billsum-writing-owner-review-plan.v0.1"
DECISIONS_SCHEMA_VERSION = "lumi.billsum-owner-decisions.v0.1"
RECEIPT_SCHEMA_VERSION = "lumi.sft-billsum-writing-owner-review-receipt.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-billsum-writing-owner-review-verification.v0.1"
)
EXPECTED_OWNER_REQUEST = "Accept all 32 BillSum rows."
EXPECTED_OWNER_REQUEST_SHA256 = (
    "61670071913fd94c2ba09868ac35647e8f09729b53d6bc3cfc98031216d7b6c2"
)
EXPECTED_SAMPLE_SHA256 = (
    "2e6de1b9c9c8bcc0373de1033df608fd409c13aaaa8c11133d3897a39a779cd6"
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
        raise FileExistsError(f"refusing to overwrite BillSum owner-review artifact: {path}")
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
        raise ValueError("BillSum owner-review plan schema changed")
    if plan.get("status") != "authorized_human_owner_review_finalization":
        raise ValueError("BillSum owner review is not authorized")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("BillSum owner-review authorization widened")
    if plan.get("owner_request_text") != EXPECTED_OWNER_REQUEST:
        raise ValueError("BillSum owner request changed")
    if plan.get("owner_request_sha256") != EXPECTED_OWNER_REQUEST_SHA256:
        raise ValueError("BillSum owner request hash changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("BillSum owner-review bindings missing")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"invalid BillSum owner-review binding: {name}")
        resolved = (root / str(binding.get("path"))).resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError(f"BillSum owner-review binding missing: {name}")
        if file_sha256(resolved) != binding.get("sha256"):
            raise ValueError(f"BillSum owner-review binding changed: {name}")
    return plan


def _validate_bound_curation(
    plan: Mapping[str, Any], *, repository_root: Path
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    bindings = plan["bindings"]
    report = _load_json(repository_root / bindings["curation_report"]["path"])
    verification = _load_json(
        repository_root / bindings["curation_verification"]["path"]
    )
    if report.get("status") != "PASS_nonproduction_candidate_owner_review_pending":
        raise ValueError("BillSum curation report status changed")
    if report.get("report_sha256") != bindings["curation_report"].get(
        "internal_sha256"
    ):
        raise ValueError("BillSum curation report internal hash changed")
    if verification.get("status") != (
        "PASS_independent_nonproduction_candidate_verification"
    ):
        raise ValueError("BillSum curation verification status changed")
    if verification.get("verification_sha256") != bindings[
        "curation_verification"
    ].get("internal_sha256"):
        raise ValueError("BillSum curation verification internal hash changed")
    if report.get("owner_review", {}).get("packet_sha256") != EXPECTED_SAMPLE_SHA256:
        raise ValueError("BillSum report owner sample changed")
    if report.get("owner_review", {}).get("packet_records") != 32:
        raise ValueError("BillSum report owner sample size changed")
    if report.get("owner_review", {}).get("minimum_acceptance_rate") != 0.9:
        raise ValueError("BillSum owner threshold changed")
    if report.get("tokens") != {
        "accepted_records": 4433,
        "token_ledger_sha256": (
            "9148398f40c5610281d0224a9b2f4031c54f7f81f3ae7110b3856990cf06ea10"
        ),
        "train_supervised_tokens": 849991,
        "validation_supervised_tokens": 49986,
    }:
        raise ValueError("BillSum candidate totals changed")
    if verification.get("candidate_records") != 4433:
        raise ValueError("BillSum verified candidate row count changed")
    if verification.get("train_supervised_tokens") != 849991:
        raise ValueError("BillSum verified train tokens changed")
    if verification.get("validation_supervised_tokens") != 49986:
        raise ValueError("BillSum verified validation tokens changed")
    if verification.get("candidate_file_sha256") != bindings["candidate"]["sha256"]:
        raise ValueError("BillSum candidate verification binding changed")
    if verification.get("owner_review_packet_sha256") != EXPECTED_SAMPLE_SHA256:
        raise ValueError("BillSum verified owner sample changed")
    if report.get("production_eligible") is not False:
        raise ValueError("BillSum curation unexpectedly became production eligible")
    return report, verification


def build_receipt(
    plan: Mapping[str, Any], *, plan_file_sha256: str, repository_root: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    bindings = plan["bindings"]
    sample = _load_json(root / bindings["owner_sample"]["path"])
    if not isinstance(sample, list) or len(sample) != 32:
        raise ValueError("BillSum owner sample must contain exactly 32 rows")
    if sha256_value(sample) != EXPECTED_SAMPLE_SHA256:
        raise ValueError("BillSum owner sample content changed")
    sample_ids = {str(row.get("record_id")) for row in sample}
    if len(sample_ids) != 32:
        raise ValueError("BillSum owner sample IDs are not unique")

    decisions = _load_json(root / bindings["decisions"]["path"])
    if (
        not isinstance(decisions, Mapping)
        or decisions.get("schema_version") != DECISIONS_SCHEMA_VERSION
    ):
        raise ValueError("BillSum owner decision schema changed")
    if decisions.get("sample_sha256") != EXPECTED_SAMPLE_SHA256:
        raise ValueError("BillSum owner decision sample binding changed")
    if decisions.get("reviewer_id") != "owner-01":
        raise ValueError("BillSum owner reviewer changed")
    reviewed_at = decisions.get("reviewed_at")
    if not isinstance(reviewed_at, str) or _UTC.fullmatch(reviewed_at) is None:
        raise ValueError("BillSum owner decision timestamp is invalid")
    rows = decisions.get("decisions")
    if not isinstance(rows, list) or len(rows) != 32:
        raise ValueError("BillSum owner decisions must cover 32 rows")
    decision_ids: set[str] = set()
    for decision in rows:
        if not isinstance(decision, Mapping) or set(decision) != {
            "accepted",
            "note",
            "record_id",
        }:
            raise ValueError("BillSum owner decision fields changed")
        if decision.get("accepted") is not True or decision.get("note") != "":
            raise ValueError("BillSum owner decision is not exact all-accept")
        record_id = str(decision.get("record_id"))
        if record_id not in sample_ids or record_id in decision_ids:
            raise ValueError("BillSum owner decision ID changed or repeated")
        decision_ids.add(record_id)
    if decision_ids != sample_ids:
        raise ValueError("BillSum owner decision set is incomplete")

    _validate_bound_curation(plan, repository_root=root)
    receipt: dict[str, Any] = {
        "acceptance_rate": 1.0,
        "accepted_records": 32,
        "all_sampled_rows_accepted": True,
        "authorization": EXPECTED_AUTHORIZATION,
        "bindings": {
            "candidate_file_sha256": bindings["candidate"]["sha256"],
            "curation_report_file_sha256": bindings["curation_report"]["sha256"],
            "curation_report_internal_sha256": bindings["curation_report"][
                "internal_sha256"
            ],
            "curation_verification_file_sha256": bindings[
                "curation_verification"
            ]["sha256"],
            "curation_verification_internal_sha256": bindings[
                "curation_verification"
            ]["internal_sha256"],
            "decisions_file_sha256": bindings["decisions"]["sha256"],
            "owner_request_sha256": EXPECTED_OWNER_REQUEST_SHA256,
            "owner_sample_file_sha256": bindings["owner_sample"]["sha256"],
            "owner_sample_internal_sha256": EXPECTED_SAMPLE_SHA256,
            "plan_file_sha256": plan_file_sha256,
            "reviewed_record_ids_sha256": sha256_value(sorted(decision_ids)),
        },
        "candidate": {
            "records": 4433,
            "train_records": 4195,
            "train_supervised_tokens": 849991,
            "validation_records": 238,
            "validation_supervised_tokens": 49986,
        },
        "next_gate": {
            "action": (
                "bounded_immutable_source_registry_v0_20_promotion_then_full_mix_reaudit"
            ),
            "dataset_seal_or_sft_authorized": False,
            "owner_review_complete": True,
            "registry_change_authorized": False,
        },
        "owner_review_claim": (
            "source_provided_legislative_summaries_owner_spot_checked_32_"
            "source_authorship_not_established"
        ),
        "production_eligible": False,
        "raw_training_text_retained": False,
        "rejected_record_ids": [],
        "rejected_records": 0,
        "reviewed_at": reviewed_at,
        "reviewer_alias": "owner-01",
        "sample_records": 32,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "owner_accepted_all_32_billsum_rows",
    }
    receipt["receipt_sha256"] = sha256_value(receipt)
    return receipt


def build_verification(
    receipt: Mapping[str, Any], *, receipt_file_sha256: str
) -> dict[str, Any]:
    unsigned = dict(receipt)
    internal = unsigned.pop("receipt_sha256", None)
    if internal != sha256_value(unsigned):
        raise ValueError("BillSum owner-review receipt hash changed")
    if receipt.get("accepted_records") != 32 or receipt.get("rejected_records") != 0:
        raise ValueError("BillSum owner-review outcome changed")
    if receipt.get("acceptance_rate") != 1.0:
        raise ValueError("BillSum owner-review acceptance rate changed")
    verification: dict[str, Any] = {
        "checks": {
            "all_32_owner_decisions_are_accept": True,
            "candidate_and_curation_evidence_bound": True,
            "exact_32_row_sample_bound": True,
            "no_raw_training_text_in_receipt": True,
            "no_registry_seal_inference_gpu_network_or_sft": True,
            "owner_review_is_human_claim_not_automated_claim": True,
            "source_summary_authorship_not_overclaimed": True,
        },
        "raw_training_text_retained": False,
        "receipt_file_sha256": receipt_file_sha256,
        "receipt_internal_sha256": internal,
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "PASS_owner_accepted_all_32_billsum_rows",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification


def finalize(
    plan_path: str | Path, repository_root: str | Path
) -> tuple[dict[str, Any], dict[str, Any]]:
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
