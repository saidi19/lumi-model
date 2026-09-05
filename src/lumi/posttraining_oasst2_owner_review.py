"""Finalize the human owner review for the verified OASST2 candidate."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .posttraining_oasst2_curation import file_sha256, sha256_value

PLAN_SCHEMA_VERSION = "lumi.sft-oasst2-owner-review-plan.v0.1"
DECISIONS_SCHEMA_VERSION = "lumi.oasst2-owner-decisions.v0.1"
RECEIPT_SCHEMA_VERSION = "lumi.sft-oasst2-owner-review-receipt.v0.1"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-oasst2-owner-review-verification.v0.1"
SOURCE_REVISION = "179dd21fc55192153d94adb0e0ce8f69e222bf75"

EXPECTED_AUTHORIZATION = {
    "candidate_registry_promotion_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "gpu_authorized": False,
    "network_authorized": False,
    "owner_review_finalization_authorized": True,
    "sft_training_authorized": False,
    "unrelated_access_authorized": False,
}
EXPECTED_CANDIDATE = {
    "records": 6_426,
    "train_records": 6_136,
    "train_supervised_tokens": 2_221_091,
    "validation_records": 290,
    "validation_supervised_tokens": 96_436,
}
EXPECTED_BINDING_PATHS = {
    "accepted_records": (
        "../lumi-artifacts/4b-sft-oasst2-ready-tree-curation-v0.1/"
        "16394e94d0eb/attempt-001/curated/accepted-records.jsonl"
    ),
    "curation_plan": "configs/lumi-4b-sft-oasst2-ready-tree-curation-v0.1.json",
    "curation_report": "evidence/lumi-4b-sft-oasst2-ready-tree-curation-v0.1.json",
    "curation_verification": (
        "evidence/lumi-4b-sft-oasst2-ready-tree-curation-verification-v0.1.json"
    ),
    "decisions": "../../../Downloads/lumi-oasst2-owner-decisions-v0.1.json",
    "finalizer": "scripts/finalize_oasst2_owner_review.py",
    "implementation": "src/lumi/posttraining_oasst2_owner_review.py",
    "owner_sample": (
        "../lumi-artifacts/4b-sft-oasst2-ready-tree-curation-v0.1/"
        "16394e94d0eb/attempt-001/review/owner-sample.jsonl"
    ),
    "test": "tests/test_oasst2_owner_review.py",
    "verifier": "scripts/verify_oasst2_owner_review.py",
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UTC_TIMESTAMP = re.compile(
    r"^20[0-9]{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])T"
    r"(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z$"
)
_AI_REVIEWER = re.compile(
    r"(?:^|[-_ ])(?:ai|anthropic|chatgpt|claude|codex|gemini|openai)(?:$|[-_ ])",
    re.IGNORECASE,
)


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            rows.append(value)
    return rows


def resolve_binding(
    repository_root: str | Path,
    binding: Mapping[str, Any],
    *,
    name: str,
) -> Path:
    root = Path(repository_root).resolve()
    expected_path = EXPECTED_BINDING_PATHS.get(name)
    if binding.get("path") != expected_path:
        raise ValueError(f"OASST2 owner-review binding path changed: {name}")
    resolved = (root / str(expected_path)).resolve()
    allowed_root = Path.home().resolve() if name == "decisions" else root.parent.resolve()
    if not resolved.is_relative_to(allowed_root):
        raise ValueError(f"OASST2 owner-review binding escapes its boundary: {name}")
    if not resolved.is_file():
        raise FileNotFoundError(f"OASST2 owner-review binding missing: {name}")
    expected_sha256 = _require_sha256(binding.get("sha256"), f"bindings.{name}.sha256")
    if file_sha256(resolved) != expected_sha256:
        raise ValueError(f"OASST2 owner-review binding changed: {name}")
    return resolved


def load_owner_review_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    plan = read_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported OASST2 owner-review plan")
    if plan.get("observed_at") != "2026-08-08":
        raise ValueError("OASST2 owner-review observation date changed")
    if plan.get("source_revision") != SOURCE_REVISION:
        raise ValueError("OASST2 owner-review source revision changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("OASST2 owner-review authorization changed")
    if plan.get("candidate") != EXPECTED_CANDIDATE:
        raise ValueError("OASST2 owner-review candidate counts changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(EXPECTED_BINDING_PATHS):
        raise ValueError("OASST2 owner-review binding inventory changed")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"OASST2 owner-review binding is invalid: {name}")
        resolve_binding(repository_root, binding, name=name)
    output = plan.get("output")
    expected_output_keys = {
        "private_receipt_path",
        "raw_training_text_in_git",
        "repository_receipt_path",
        "repository_verification_path",
    }
    if not isinstance(output, Mapping) or set(output) != expected_output_keys:
        raise ValueError("OASST2 owner-review output inventory changed")
    if output.get("raw_training_text_in_git") is not False:
        raise ValueError("OASST2 owner review cannot retain raw text in Git")
    return plan


def validate_bound_candidate(
    plan: Mapping[str, Any],
    *,
    repository_root: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    bindings = plan["bindings"]
    sample = read_jsonl(
        resolve_binding(repository_root, bindings["owner_sample"], name="owner_sample")
    )
    if len(sample) != 32:
        raise ValueError("OASST2 owner sample must contain exactly 32 rows")
    sample_sha256 = sha256_value(sample)
    if sample_sha256 != _require_sha256(
        bindings["owner_sample"].get("internal_sha256"),
        "bindings.owner_sample.internal_sha256",
    ):
        raise ValueError("OASST2 owner sample content changed")

    report = read_json(
        resolve_binding(repository_root, bindings["curation_report"], name="curation_report")
    )
    verification = read_json(
        resolve_binding(
            repository_root,
            bindings["curation_verification"],
            name="curation_verification",
        )
    )
    if report.get("report_sha256") != bindings["curation_report"].get("internal_sha256"):
        raise ValueError("OASST2 curation report internal hash changed")
    if verification.get("verification_sha256") != bindings["curation_verification"].get(
        "internal_sha256"
    ):
        raise ValueError("OASST2 curation verification internal hash changed")
    if report.get("status") != (
        "curated_nonproduction_short_of_planning_target_owner_review_pending"
    ):
        raise ValueError("OASST2 curation report status changed")
    if report.get("production", {}).get("production_eligible") is not False:
        raise ValueError("OASST2 curation report crossed production eligibility")
    if verification.get("status") != "verified_nonproduction_owner_review_pending":
        raise ValueError("OASST2 curation verification status changed")
    if verification.get("production_eligible") is not False:
        raise ValueError("OASST2 verification crossed production eligibility")
    curated = verification.get("curated")
    if not isinstance(curated, Mapping):
        raise ValueError("OASST2 curated verification summary changed")
    expected = EXPECTED_CANDIDATE
    observed = {
        "records": curated.get("accepted_records"),
        "train_records": report.get("curation", {}).get("split_counts", {}).get("train"),
        "train_supervised_tokens": curated.get("train_supervised_tokens"),
        "validation_records": report.get("curation", {}).get("split_counts", {}).get("validation"),
        "validation_supervised_tokens": curated.get("validation_supervised_tokens"),
    }
    if observed != expected:
        raise ValueError("OASST2 candidate count or token binding changed")
    if report.get("owner_review", {}).get("sample_sha256") != sample_sha256:
        raise ValueError("OASST2 report owner sample changed")
    return sample, report, verification


def validate_owner_decisions(
    value: Mapping[str, Any],
    *,
    sample_rows: list[Mapping[str, Any]],
    plan: Mapping[str, Any],
    decision_file_sha256: str,
    plan_file_sha256: str,
) -> dict[str, Any]:
    expected_fields = {
        "decisions",
        "reviewed_at",
        "reviewer_id",
        "sample_sha256",
        "schema_version",
    }
    if set(value) != expected_fields:
        raise ValueError("OASST2 owner-decision fields changed")
    if value.get("schema_version") != DECISIONS_SCHEMA_VERSION:
        raise ValueError("unsupported OASST2 owner-decision schema")
    sample_sha256 = sha256_value(sample_rows)
    if value.get("sample_sha256") != sample_sha256:
        raise ValueError("OASST2 owner-decision sample SHA-256 changed")
    reviewer = value.get("reviewer_id")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("OASST2 owner reviewer alias is required")
    if _AI_REVIEWER.search(reviewer.strip()) is not None:
        raise ValueError("an AI alias cannot claim OASST2 owner review")
    reviewed_at = value.get("reviewed_at")
    if not isinstance(reviewed_at, str) or _UTC_TIMESTAMP.fullmatch(reviewed_at) is None:
        raise ValueError("OASST2 reviewed_at must be UTC ISO-8601")

    decisions = value.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != len(sample_rows):
        raise ValueError("OASST2 decisions must cover every sampled row")
    expected_ids = {str(row["record_id"]) for row in sample_rows}
    observed_ids: set[str] = set()
    accepted_ids: list[str] = []
    rejected_ids: list[str] = []
    for decision in decisions:
        if not isinstance(decision, Mapping) or set(decision) != {
            "accepted",
            "note",
            "record_id",
        }:
            raise ValueError("OASST2 decision fields changed")
        record_id = decision.get("record_id")
        if not isinstance(record_id, str) or record_id not in expected_ids:
            raise ValueError("OASST2 decision contains an unknown record ID")
        if record_id in observed_ids:
            raise ValueError("OASST2 decision contains a duplicate record ID")
        observed_ids.add(record_id)
        accepted = decision.get("accepted")
        if not isinstance(accepted, bool):
            raise ValueError("OASST2 accepted value must be boolean")
        note = decision.get("note")
        if not isinstance(note, str) or len(note) > 2_000:
            raise ValueError("OASST2 decision note is invalid")
        (accepted_ids if accepted else rejected_ids).append(record_id)
    if observed_ids != expected_ids:
        raise ValueError("OASST2 decisions do not match sampled record IDs")

    bindings = plan["bindings"]
    receipt: dict[str, Any] = {
        "accepted_records": len(accepted_ids),
        "all_sampled_rows_accepted": not rejected_ids,
        "authorization": EXPECTED_AUTHORIZATION,
        "bindings": {
            "accepted_records_file_sha256": bindings["accepted_records"]["sha256"],
            "decision_file_sha256": _require_sha256(decision_file_sha256, "decision_file_sha256"),
            "decision_internal_sha256": sha256_value(value),
            "owner_sample_file_sha256": bindings["owner_sample"]["sha256"],
            "owner_sample_internal_sha256": sample_sha256,
            "plan_file_sha256": _require_sha256(plan_file_sha256, "plan_file_sha256"),
            "report_file_sha256": bindings["curation_report"]["sha256"],
            "report_internal_sha256": bindings["curation_report"]["internal_sha256"],
            "reviewed_record_ids_sha256": sha256_value(sorted(observed_ids)),
            "verification_file_sha256": bindings["curation_verification"]["sha256"],
            "verification_internal_sha256": bindings["curation_verification"]["internal_sha256"],
        },
        "candidate": dict(plan["candidate"]),
        "owner_review_claim": (
            "upstream_crowdsourced_ranked_conversations_human_owner_quality_reviewed"
        ),
        "planning_target_reached": False,
        "production_eligible": False,
        "production_promotions": 0,
        "raw_training_text_retained": False,
        "rejected_record_ids": sorted(rejected_ids),
        "rejected_records": len(rejected_ids),
        "remaining_blockers": [
            "oasst2_candidate_registry_promotion_not_authorized",
            "first_party_multilingual_safety_tool_lane_not_materialized",
            "production_dataset_seal_not_authorized",
            "production_sft_not_authorized",
        ],
        "reviewed_at": reviewed_at,
        "reviewer_id": reviewer.strip(),
        "sample_records": len(sample_rows),
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "source_revision": SOURCE_REVISION,
        "status": (
            "owner_quality_sample_passed_promotion_gate_separate"
            if not rejected_ids
            else "owner_quality_sample_rejected_rows_require_repair"
        ),
    }
    receipt["receipt_sha256"] = sha256_value(receipt)
    return receipt


def validate_owner_review_receipt(receipt: Mapping[str, Any]) -> None:
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported OASST2 owner-review receipt")
    seal = _require_sha256(receipt.get("receipt_sha256"), "receipt_sha256")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if sha256_value(unsigned) != seal:
        raise ValueError("OASST2 owner-review receipt seal is invalid")
    if receipt.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("OASST2 owner-review receipt authorization changed")
    if receipt.get("candidate") != EXPECTED_CANDIDATE:
        raise ValueError("OASST2 owner-review candidate changed")
    if receipt.get("raw_training_text_retained") is not False:
        raise ValueError("OASST2 owner-review receipt retained raw text")
    if receipt.get("production_eligible") is not False:
        raise ValueError("OASST2 owner review crossed production eligibility")
    if receipt.get("production_promotions") != 0:
        raise ValueError("OASST2 owner review crossed promotion boundary")
    accepted = receipt.get("accepted_records")
    rejected = receipt.get("rejected_records")
    if not isinstance(accepted, int) or not isinstance(rejected, int):
        raise ValueError("OASST2 owner-review counts are invalid")
    if accepted + rejected != 32:
        raise ValueError("OASST2 owner-review sample count changed")


__all__ = [
    "DECISIONS_SCHEMA_VERSION",
    "EXPECTED_AUTHORIZATION",
    "EXPECTED_CANDIDATE",
    "PLAN_SCHEMA_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "VERIFICATION_SCHEMA_VERSION",
    "load_owner_review_plan",
    "read_json",
    "read_jsonl",
    "resolve_binding",
    "validate_bound_candidate",
    "validate_owner_decisions",
    "validate_owner_review_receipt",
]
