"""Fail-closed human-owner review finalization for the Aya Bengali candidate."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .posttraining_first_party import file_sha256
from .posttraining_oasst1_full_run import deterministic_owner_sample, sha256_value

PLAN_SCHEMA_VERSION = "lumi.sft-aya-bengali-owner-review-plan.v0.1"
DECISIONS_SCHEMA_VERSION = "lumi.sft-aya-bengali-owner-review-decisions.v0.1"
RECEIPT_SCHEMA_VERSION = "lumi.sft-aya-bengali-owner-review-receipt.v0.1"

EXPECTED_AUTHORIZATION = {
    "ai_or_automated_owner_approval_authorized": False,
    "external_api_spend_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_dataset_promotion_authorized": False,
    "production_dataset_seal_authorized": False,
    "production_sft_authorized": False,
    "purpose": "private_human_owner_quality_spot_check_only",
}
EXPECTED_CANDIDATE = {
    "accepted_records": 1478,
    "attribution_file_sha256": ("fe9f01999986d400c70922826729f72de5b2b7b30d9ef0a07e6eef287cce5188"),
    "candidate_file_sha256": ("169cc09f53e2b799058328ab3be946f151f1b2143b7288a757be0006a5901aee"),
    "materialization_report_file_sha256": (
        "9658cb37fe659a65d7dd68d8f7d5529b4240ff96c6c747cf61f40db29274c05c"
    ),
    "materialization_report_internal_sha256": (
        "a2f08b1594a2b6fd9def1f43a48e072cbe0f107f4bc7ea43b391d6066ddc6ade"
    ),
    "owner_review_packet_file_sha256": (
        "11ffcd9ef313df6bc7516a4c95e19098fd02d206ad9da576d06c9836054e3bb2"
    ),
    "preflight_receipt_sha256": (
        "6d7c20b19f6b8154c58f51c59521d1e0045dbcbe3e06baa10ac1766c0f6800b2"
    ),
    "sample_record_ids_sha256": (
        "d4939a8d02d652123555346b02a96e8758f2efb53a12ceac9ac67b423036bc29"
    ),
    "source_revision": "f9ea04583f02a8f86404ff6c58bf75fe637df8a2",
    "supervised_tokens": 577453,
}
EXPECTED_LIMITS = {"records": 32}
EXPECTED_OUTPUT = {
    "decision_filename": "lumi-aya-owner-review-decisions-v0.1.json",
    "production_promotion_permitted": False,
    "receipt_contains_raw_training_text": False,
}
EXPECTED_BINDINGS = frozenset(
    {
        "accepted_records",
        "attribution_manifest",
        "materialization_report",
        "owner_review_packet",
        "receipt_finalizer",
        "review_implementation",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UTC_TIMESTAMP = re.compile(
    r"^20[0-9]{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])T"
    r"(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z$"
)
_BANNED_REVIEWER_MARKERS = {"chatgpt", "claude", "codex", "gemini"}


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
        raise ValueError("Aya owner-review binding path is invalid")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError("Aya owner-review binding escapes workspace")
    return resolved


def load_owner_review_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Load the exact Aya owner-review plan and verify every local binding."""

    root = Path(repository_root).resolve()
    plan = _load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported Aya owner-review plan")
    if plan.get("status") != "human_owner_review_ready_no_promotion":
        raise ValueError("Aya owner-review status changed")
    if plan.get("observed_at") != "2026-08-06":
        raise ValueError("Aya owner-review date changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("Aya owner-review authorization changed")
    if plan.get("candidate") != EXPECTED_CANDIDATE:
        raise ValueError("Aya owner-review candidate changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("Aya owner-review limits changed")
    if plan.get("output") != EXPECTED_OUTPUT:
        raise ValueError("Aya owner-review output boundary changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("Aya owner-review binding set changed")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"Aya owner-review binding must be an object: {name}")
        expected = _require_sha256(binding.get("sha256"), f"bindings.{name}.sha256")
        resolved = resolve_plan_binding(root, binding)
        if not resolved.is_file() or file_sha256(resolved) != expected:
            raise ValueError(f"Aya owner-review binding changed: {name}")
    return plan


def load_exact_owner_sample(
    candidate_path: str | Path,
    *,
    materialization_report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Rebuild the deterministic 32-row sample from the exact private candidate."""

    unsigned_report = {
        key: value for key, value in materialization_report.items() if key != "report_sha256"
    }
    if materialization_report.get("report_sha256") != sha256_value(unsigned_report):
        raise ValueError("Aya materialization report seal changed")
    if (
        materialization_report.get("report_sha256")
        != EXPECTED_CANDIDATE["materialization_report_internal_sha256"]
    ):
        raise ValueError("Aya materialization report binding changed")
    if materialization_report.get("candidate") != {
        "materialized": True,
        "records": EXPECTED_CANDIDATE["accepted_records"],
        "supervised_tokens": EXPECTED_CANDIDATE["supervised_tokens"],
    }:
        raise ValueError("Aya materialized candidate metrics changed")
    if materialization_report.get("production", {}).get("production_eligible") is not False:
        raise ValueError("Aya candidate unexpectedly claims production eligibility")

    rows = [
        json.loads(line)
        for line in Path(candidate_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != EXPECTED_CANDIDATE["accepted_records"]:
        raise ValueError("Aya private candidate row count changed")
    sample = [
        dict(row)
        for row in deterministic_owner_sample(
            rows,
            maximum_records=EXPECTED_LIMITS["records"],
        )
    ]
    sample_ids = sorted(str(row.get("record_id")) for row in sample)
    if len(set(sample_ids)) != EXPECTED_LIMITS["records"]:
        raise ValueError("Aya owner-review sample identity changed")
    if sha256_value(sample_ids) != EXPECTED_CANDIDATE["sample_record_ids_sha256"]:
        raise ValueError("Aya owner-review sample hash changed")
    return sample


def validate_owner_decisions(
    value: Mapping[str, Any],
    *,
    sample_rows: Sequence[Mapping[str, Any]],
    decision_file_sha256: str,
    plan_file_sha256: str,
    reviewed_at: str,
) -> dict[str, Any]:
    """Validate owner decisions and return a raw-text-free non-promoting receipt."""

    _require_sha256(decision_file_sha256, "decision_file_sha256")
    _require_sha256(plan_file_sha256, "plan_file_sha256")
    if not isinstance(reviewed_at, str) or _UTC_TIMESTAMP.fullmatch(reviewed_at) is None:
        raise ValueError("Aya owner reviewed_at must be exact UTC ISO-8601")
    if set(value) != {"bindings", "records", "reviewer_alias", "schema_version"}:
        raise ValueError("Aya owner decision envelope changed")
    if value.get("schema_version") != DECISIONS_SCHEMA_VERSION:
        raise ValueError("unsupported Aya owner decision schema")
    if value.get("bindings") != {
        "attribution_file_sha256": EXPECTED_CANDIDATE["attribution_file_sha256"],
        "candidate_file_sha256": EXPECTED_CANDIDATE["candidate_file_sha256"],
        "preflight_receipt_sha256": EXPECTED_CANDIDATE["preflight_receipt_sha256"],
        "sample_record_ids_sha256": EXPECTED_CANDIDATE["sample_record_ids_sha256"],
    }:
        raise ValueError("Aya owner decision bindings changed")
    reviewer = value.get("reviewer_alias")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("Aya owner reviewer alias is required")
    normalized_reviewer = reviewer.strip().casefold()
    if any(marker in normalized_reviewer for marker in _BANNED_REVIEWER_MARKERS):
        raise ValueError("an AI system cannot claim Aya owner review")

    decisions = value.get("records")
    if not isinstance(decisions, list) or len(decisions) != len(sample_rows):
        raise ValueError("Aya owner decisions must cover every sampled row")
    expected_ids = {str(row["record_id"]) for row in sample_rows}
    observed_ids: set[str] = set()
    accepted_ids: list[str] = []
    rejected_ids: list[str] = []
    for decision in decisions:
        if not isinstance(decision, Mapping) or set(decision) != {
            "decision",
            "note",
            "record_id",
        }:
            raise ValueError("Aya owner decision fields changed")
        record_id = decision.get("record_id")
        if not isinstance(record_id, str) or record_id not in expected_ids:
            raise ValueError("Aya owner decision contains an unknown record")
        if record_id in observed_ids:
            raise ValueError("Aya owner decision contains a duplicate record")
        choice = decision.get("decision")
        note = decision.get("note")
        if choice not in {"accept", "reject"}:
            raise ValueError("Aya owner decision must be accept or reject")
        if not isinstance(note, str) or len(note) > 2000:
            raise ValueError("Aya owner decision note is invalid")
        observed_ids.add(record_id)
        (accepted_ids if choice == "accept" else rejected_ids).append(record_id)
    if observed_ids != expected_ids:
        raise ValueError("Aya owner decisions do not match the sample")

    status = (
        "owner_quality_sample_passed_source_promotion_still_separate"
        if not rejected_ids
        else "owner_quality_sample_rejected_rows_require_repair"
    )
    receipt: dict[str, Any] = {
        "accepted_records": len(accepted_ids),
        "all_sampled_rows_accepted": not rejected_ids,
        "authorization": EXPECTED_AUTHORIZATION,
        "bindings": {
            "decision_file_sha256": decision_file_sha256,
            "materialization_report_internal_sha256": EXPECTED_CANDIDATE[
                "materialization_report_internal_sha256"
            ],
            "plan_file_sha256": plan_file_sha256,
            "sample_record_ids_sha256": EXPECTED_CANDIDATE["sample_record_ids_sha256"],
        },
        "owner_review_claim": (
            "upstream_human_source_plus_human_owner_quality_reviewed_not_first_party_authored"
        ),
        "production_eligible": False,
        "production_promotions": 0,
        "raw_training_text_retained": False,
        "rejected_record_ids": sorted(rejected_ids),
        "rejected_records": len(rejected_ids),
        "remaining_blockers": [
            "source_registry_promotion_not_authorized",
            "complete_10m_to_30m_learning_mix_not_ready",
            "production_dataset_seal_and_sft_not_authorized",
        ],
        "reviewed_at": reviewed_at,
        "reviewer_alias": reviewer.strip(),
        "sample_records": len(sample_rows),
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": status,
    }
    receipt["receipt_sha256"] = sha256_value(receipt)
    return receipt


def validate_owner_review_receipt(receipt: Mapping[str, Any]) -> None:
    """Verify the raw-text-free seal and all closed production gates."""

    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported Aya owner-review receipt")
    seal = _require_sha256(receipt.get("receipt_sha256"), "receipt_sha256")
    if (
        sha256_value({key: value for key, value in receipt.items() if key != "receipt_sha256"})
        != seal
    ):
        raise ValueError("Aya owner-review receipt seal is invalid")
    if receipt.get("raw_training_text_retained") is not False:
        raise ValueError("Aya owner-review receipt retained raw training text")
    if receipt.get("production_eligible") is not False:
        raise ValueError("Aya owner review crossed production eligibility")
    if receipt.get("production_promotions") != 0:
        raise ValueError("Aya owner review crossed a promotion boundary")
