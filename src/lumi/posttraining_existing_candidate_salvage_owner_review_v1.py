"""Finalize the human-owner review delta for locally salvaged Aya MIT rows."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PLAN_SCHEMA_VERSION = "lumi.sft-existing-candidate-salvage-owner-review-plan.v0.1"
DECISIONS_SCHEMA_VERSION = (
    "lumi.sft-existing-candidate-salvage-owner-decisions.v0.1"
)
RECEIPT_SCHEMA_VERSION = (
    "lumi.sft-existing-candidate-salvage-owner-review-receipt.v0.1"
)
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-existing-candidate-salvage-owner-review-verification.v0.1"
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
EXPECTED_OWNER_REQUEST = "Accept all 6 Aya MIT rows"
EXPECTED_OWNER_REQUEST_SHA256 = (
    "6382c3e3b4f4821aeba3fe2b30ccd79f533386e8fd0d6ce23fc14f3b0022959c"
)
EXPECTED_SAMPLE_SHA256 = (
    "33ce063184a9a9f067672d2060c575e192c3f018e8d05d08692f6054694c944a"
)
EXPECTED_IDS = {
    "closure-coherelabs_aya_bengali_closure_canary-f6eeaab09ce24fd4605c612a",
    "closure-coherelabs_aya_bengali_closure_canary-42340b621fc74df73b3f8a6b",
    "closure-coherelabs_aya_bengali_closure_canary-c51677f0c7e22fc925da827f",
    "closure-coherelabs_aya_bengali_closure_canary-83e9ab13e45974f7350bc67c",
    "closure-coherelabs_aya_bengali_closure_canary-c50520c5f6aec3c0a1669df9",
    "closure-coherelabs_aya_bengali_closure_canary-2b78b9553eef190cc3ebb3bd",
}
_UTC = re.compile(
    r"^20[0-9]{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])T"
    r"(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z$"
)


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def compact_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def sha256_value(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(compact_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_plan(path: str | Path, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = _load_json(Path(path).resolve())
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("salvage owner-review plan schema changed")
    if plan.get("status") != "authorized_human_owner_delta_review_finalization":
        raise ValueError("salvage owner-review is not authorized")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("salvage owner-review authorization widened")
    if plan.get("owner_request_text") != EXPECTED_OWNER_REQUEST:
        raise ValueError("owner request text changed")
    if plan.get("owner_request_sha256") != EXPECTED_OWNER_REQUEST_SHA256:
        raise ValueError("owner request hash changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, dict) or len(bindings) != 8:
        raise ValueError("salvage owner-review bindings changed")
    for name, binding in bindings.items():
        if not isinstance(binding, dict):
            raise ValueError(f"invalid binding: {name}")
        resolved = (root / str(binding.get("path"))).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"binding escaped repository: {name}")
        if not resolved.is_file() or file_sha256(resolved) != binding.get("sha256"):
            raise ValueError(f"binding changed: {name}")
    return plan


def build_receipt(
    plan: Mapping[str, Any], plan_file_sha256: str, repository_root: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    resolved = {
        name: (root / binding["path"]).resolve()
        for name, binding in plan["bindings"].items()
    }
    sample = _load_json(resolved["sample"])
    rows = sample.get("rows")
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("owner-review sample shape changed")
    if hashlib.sha256(compact_json_bytes(rows)).hexdigest() != EXPECTED_SAMPLE_SHA256:
        raise ValueError("owner-review sample content hash changed")
    sample_ids = {str(row.get("record_id")) for row in rows}
    if sample_ids != EXPECTED_IDS:
        raise ValueError("owner-review sample IDs changed")

    decisions = _load_json(resolved["decisions"])
    if decisions.get("schema_version") != DECISIONS_SCHEMA_VERSION:
        raise ValueError("owner decisions schema changed")
    if decisions.get("owner_request_sha256") != EXPECTED_OWNER_REQUEST_SHA256:
        raise ValueError("owner decisions request binding changed")
    if decisions.get("sample_sha256") != EXPECTED_SAMPLE_SHA256:
        raise ValueError("owner decisions sample binding changed")
    if decisions.get("reviewer_alias") != "owner-01":
        raise ValueError("owner decisions reviewer changed")
    reviewed_at = decisions.get("reviewed_at")
    if not isinstance(reviewed_at, str) or _UTC.fullmatch(reviewed_at) is None:
        raise ValueError("owner decisions timestamp is invalid")
    records = decisions.get("records")
    if not isinstance(records, list) or len(records) != 6:
        raise ValueError("owner decisions do not cover six rows")
    observed_ids: set[str] = set()
    for decision in records:
        if not isinstance(decision, Mapping) or set(decision) != {
            "decision",
            "note",
            "record_id",
        }:
            raise ValueError("owner decision fields changed")
        if decision["decision"] != "accept" or decision["note"] != "":
            raise ValueError("owner decision is not the exact all-accept decision")
        record_id = str(decision["record_id"])
        if record_id not in EXPECTED_IDS or record_id in observed_ids:
            raise ValueError("owner decision record ID changed or repeated")
        observed_ids.add(record_id)
    if observed_ids != EXPECTED_IDS:
        raise ValueError("owner decision set is incomplete")

    materialization = _load_json(resolved["materialization_report"])
    materialization_verification = _load_json(
        resolved["materialization_verification"]
    )
    if materialization.get("status") != (
        "materialized_private_nonproduction_salvage_owner_review_pending"
    ):
        raise ValueError("materialization status changed")
    if materialization["owner_review_delta"]["sample_sha256"] != EXPECTED_SAMPLE_SHA256:
        raise ValueError("materialization sample binding changed")
    if materialization_verification.get("status") != (
        "PASS_private_salvage_materialized_owner_review_pending"
    ):
        raise ValueError("materialization verification did not pass")

    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "owner_accepted_all_six_aya_mit_delta_rows",
        "reviewed_at": reviewed_at,
        "reviewer_alias": "owner-01",
        "accepted_records": 6,
        "rejected_records": 0,
        "rejected_record_ids": [],
        "all_sampled_rows_accepted": True,
        "bindings": {
            "plan_file_sha256": plan_file_sha256,
            "owner_request_sha256": EXPECTED_OWNER_REQUEST_SHA256,
            "sample_file_sha256": plan["bindings"]["sample"]["sha256"],
            "sample_sha256": EXPECTED_SAMPLE_SHA256,
            "decisions_file_sha256": plan["bindings"]["decisions"]["sha256"],
            "materialization_report_file_sha256": plan["bindings"][
                "materialization_report"
            ]["sha256"],
            "materialization_verification_file_sha256": plan["bindings"][
                "materialization_verification"
            ]["sha256"],
            "aya_mit_candidate_file_sha256": materialization["output"][
                "aya_mit_candidate_file_sha256"
            ],
        },
        "owner_review_claim": (
            "upstream_human_source_plus_human_owner_quality_reviewed_"
            "not_first_party_authored"
        ),
        "authorization": EXPECTED_AUTHORIZATION,
        "next_gate": {
            "action": "project_registry_v0_18_and_run_full_cross_source_hygiene_audit",
            "owner_review_complete": True,
            "registry_change_authorized": False,
            "dataset_seal_or_sft_authorized": False,
        },
        "raw_training_text_retained": False,
        "production_eligible": False,
    }
    receipt["receipt_sha256"] = sha256_value(receipt)
    return receipt


def build_verification(
    receipt: Mapping[str, Any], receipt_file_sha256: str
) -> dict[str, Any]:
    unsigned = dict(receipt)
    internal = unsigned.pop("receipt_sha256")
    if sha256_value(unsigned) != internal:
        raise ValueError("owner-review receipt hash does not verify")
    if receipt.get("accepted_records") != 6 or receipt.get("rejected_records") != 0:
        raise ValueError("owner-review outcome changed")
    verification: dict[str, Any] = {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "PASS_owner_accepted_all_six_aya_mit_delta_rows",
        "receipt_file_sha256": receipt_file_sha256,
        "receipt_internal_sha256": internal,
        "checks": {
            "exact_six_row_sample_bound": True,
            "all_six_owner_decisions_are_accept": True,
            "materialization_and_verification_bound": True,
            "owner_review_is_human_claim_not_automated_claim": True,
            "no_raw_training_text_in_receipt": True,
            "no_registry_seal_inference_gpu_network_or_sft": True,
        },
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification
