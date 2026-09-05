"""Fail-closed owner review for the repaired OpenR1 visible-final sample."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .posttraining_first_party import file_sha256
from .posttraining_openr1_full_run import report_sha256, sha256_value

PLAN_SCHEMA_VERSION = "lumi.sft-openr1-visible-final-owner-review-plan.v0.1"
DECISIONS_SCHEMA_VERSION = "lumi.ai-assisted-owner-decisions.v0.3"
RECEIPT_SCHEMA_VERSION = "lumi.sft-openr1-visible-final-owner-review-receipt.v0.1"

EXPECTED_AUTHORIZATION = {
    "ai_or_automated_owner_approval_authorized": False,
    "external_api_spend_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_dataset_promotion_authorized": False,
    "production_dataset_seal_authorized": False,
    "production_sft_authorized": False,
    "purpose": "private_human_owner_visible_final_quality_spot_check_only",
}
EXPECTED_CANDIDATE = {
    "accepted_records_file_sha256": (
        "7d460d213461d1968411ae47cd04fb27b2feca8f5692b21cd5c3a345caa7d349"
    ),
    "owner_review_html_file_sha256": (
        "1f26bb65ea847695f0bdd5d19aad66316aa415f59f7704b26a25b3085b689c5f"
    ),
    "owner_sample_file_sha256": (
        "9c428847e8a4b1197240001474739927bb19dbe5c42cb1e18faec7d0e7478b29"
    ),
    "owner_sample_internal_sha256": (
        "3696da931e516755b5f59af21dbcfce133725f574f77799bdf2687e16eb06541"
    ),
    "records": 2_656,
    "repair_plan_file_sha256": (
        "fe1a62f62db39ba22c7b82f7560332b942acbfb6d27e99d13da9b70538d4691c"
    ),
    "report_file_sha256": (
        "75cb34f15e4b9df6d4552b98ce807cf5e6779bb8d8022e6d80be21455f14a6ed"
    ),
    "report_internal_sha256": (
        "728f7f5e4c5f74652d6febb8480d34f4b133e96e06ef13b2ddeb216d5e169d52"
    ),
    "style_rejected_records": 32,
    "supervised_tokens": 1_170_745,
    "verification_file_sha256": (
        "37997999b8665bd5d4c4ac965d27776dc71a855a474f2258aa4796722dffab27"
    ),
    "verification_internal_sha256": (
        "1e45c56b06cbc8aad6bdec3b210d36d402dc099f07c4d6f2c92619a93ac0b4b0"
    ),
}
EXPECTED_LIMITS = {"records": 32, "train_records": 16, "validation_records": 16}
EXPECTED_OUTPUT = {
    "decision_filename": "lumi-openr1-visible-final-owner-decisions-v0.1.json",
    "production_promotion_permitted": False,
    "receipt_contains_raw_training_text": False,
}
EXPECTED_BINDINGS = frozenset(
    {
        "accepted_records",
        "owner_review_html",
        "owner_sample",
        "receipt_finalizer",
        "repair_plan",
        "review_implementation",
        "visible_final_report",
        "visible_final_verification",
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
        raise ValueError("OpenR1 owner-review binding path is invalid")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError("OpenR1 owner-review binding escaped the workspace")
    return resolved


def load_owner_review_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = _load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported OpenR1 owner-review plan")
    if plan.get("status") != "human_owner_visible_final_review_ready_no_promotion":
        raise ValueError("OpenR1 owner-review status changed")
    if plan.get("observed_at") != "2026-08-06":
        raise ValueError("OpenR1 owner-review date changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("OpenR1 owner-review authorization changed")
    if plan.get("candidate") != EXPECTED_CANDIDATE:
        raise ValueError("OpenR1 owner-review candidate changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("OpenR1 owner-review limits changed")
    if plan.get("output") != EXPECTED_OUTPUT:
        raise ValueError("OpenR1 owner-review output boundary changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("OpenR1 owner-review binding inventory changed")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"OpenR1 owner-review binding is invalid: {name}")
        expected = _require_sha256(binding.get("sha256"), f"bindings.{name}.sha256")
        resolved = resolve_plan_binding(root, binding)
        if not resolved.is_file() or file_sha256(resolved) != expected:
            raise ValueError(f"OpenR1 owner-review binding changed: {name}")
    private_root = plan.get("private_output_root")
    receipt_path = plan.get("receipt_path")
    if not isinstance(private_root, str) or not private_root:
        raise ValueError("OpenR1 owner-review private output is missing")
    if not isinstance(receipt_path, str) or not receipt_path.startswith("evidence/"):
        raise ValueError("OpenR1 owner-review receipt path is invalid")
    private_resolved = (root / private_root).resolve()
    if private_resolved.is_relative_to(root) or not private_resolved.is_relative_to(
        root.parent
    ):
        raise ValueError("OpenR1 owner-review private output escaped its boundary")
    return plan


def validate_owner_sample(
    sample_rows: Sequence[Mapping[str, Any]],
    *,
    report: Mapping[str, Any],
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    if report.get("report_sha256") != report_sha256(report):
        raise ValueError("OpenR1 visible-final report seal changed")
    if report.get("report_sha256") != EXPECTED_CANDIDATE["report_internal_sha256"]:
        raise ValueError("OpenR1 visible-final report binding changed")
    if report.get("status") != "repaired_nonproduction_visible_final_review_pending":
        raise ValueError("OpenR1 visible-final report status changed")
    if report.get("candidate") != {
        "hidden_reasoning_records": EXPECTED_CANDIDATE["records"],
        "records": EXPECTED_CANDIDATE["records"],
        "supervised_tokens": EXPECTED_CANDIDATE["supervised_tokens"],
        "train_records": 2_522,
        "validation_records": 134,
    }:
        raise ValueError("OpenR1 visible-final candidate metrics changed")
    owner_review = report.get("owner_review")
    if not isinstance(owner_review, Mapping) or owner_review.get("sample_sha256") != (
        EXPECTED_CANDIDATE["owner_sample_internal_sha256"]
    ):
        raise ValueError("OpenR1 visible-final owner sample report binding changed")
    if owner_review.get("prior_raw_trace_decisions_carried_forward") is not False:
        raise ValueError("OpenR1 raw-trace decisions were carried forward")

    unsigned_verification = {
        key: value for key, value in verification.items() if key != "verification_sha256"
    }
    if verification.get("verification_sha256") != sha256_value(unsigned_verification):
        raise ValueError("OpenR1 visible-final verification seal changed")
    if verification.get("verification_sha256") != EXPECTED_CANDIDATE[
        "verification_internal_sha256"
    ]:
        raise ValueError("OpenR1 visible-final verification binding changed")
    if verification.get("status") != "verified_nonproduction_visible_final_review_pending":
        raise ValueError("OpenR1 visible-final verification status changed")
    if verification.get("production_eligible") is not False:
        raise ValueError("OpenR1 visible-final candidate crossed production")

    rows = list(sample_rows)
    if len(rows) != EXPECTED_LIMITS["records"]:
        raise ValueError("OpenR1 owner-review sample count changed")
    if sha256_value(rows) != EXPECTED_CANDIDATE["owner_sample_internal_sha256"]:
        raise ValueError("OpenR1 owner-review sample content changed")
    record_ids: set[str] = set()
    split_counts: Counter[str] = Counter()
    for row in rows:
        if set(row) != {
            "language_mode",
            "messages",
            "record_id",
            "skill_id",
            "split",
        }:
            raise ValueError("OpenR1 owner-review row fields changed")
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id or record_id in record_ids:
            raise ValueError("OpenR1 owner-review record identity changed")
        if row.get("language_mode") != "english":
            raise ValueError("OpenR1 owner-review language changed")
        if row.get("skill_id") != "reasoning_math_science_visible_final":
            raise ValueError("OpenR1 owner-review skill changed")
        messages = row.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) != 2
            or [message.get("role") for message in messages] != ["user", "assistant"]
        ):
            raise ValueError("OpenR1 owner-review message shape changed")
        visible = str(messages[-1].get("content", ""))
        if "<think>" in visible.casefold() or "</think>" in visible.casefold():
            raise ValueError("OpenR1 owner-review retained visible reasoning tags")
        record_ids.add(record_id)
        split_counts[str(row.get("split"))] += 1
    if split_counts != Counter(
        {
            "train": EXPECTED_LIMITS["train_records"],
            "validation": EXPECTED_LIMITS["validation_records"],
        }
    ):
        raise ValueError("OpenR1 owner-review split coverage changed")
    return {
        "records": len(rows),
        "sample_sha256": sha256_value(rows),
        "train_records": split_counts["train"],
        "validation_records": split_counts["validation"],
        "visible_think_tag_records": 0,
    }


def validate_owner_decisions(
    value: Mapping[str, Any],
    *,
    sample_rows: Sequence[Mapping[str, Any]],
    decision_file_sha256: str,
    plan_file_sha256: str,
) -> dict[str, Any]:
    _require_sha256(decision_file_sha256, "decision_file_sha256")
    _require_sha256(plan_file_sha256, "plan_file_sha256")
    if set(value) != {
        "decisions",
        "reviewed_at",
        "reviewer_id",
        "sample_sha256",
        "schema_version",
    }:
        raise ValueError("OpenR1 owner decision envelope changed")
    if value.get("schema_version") != DECISIONS_SCHEMA_VERSION:
        raise ValueError("unsupported OpenR1 owner decision schema")
    sample_sha256 = sha256_value(list(sample_rows))
    if value.get("sample_sha256") != sample_sha256:
        raise ValueError("OpenR1 owner decision sample changed")
    reviewer = value.get("reviewer_id")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("OpenR1 owner reviewer alias is required")
    normalized_reviewer = reviewer.strip().casefold()
    if any(marker in normalized_reviewer for marker in _BANNED_REVIEWER_MARKERS):
        raise ValueError("an AI system cannot claim OpenR1 owner review")
    reviewed_at = value.get("reviewed_at")
    if not isinstance(reviewed_at, str) or _UTC_TIMESTAMP.fullmatch(reviewed_at) is None:
        raise ValueError("OpenR1 owner reviewed_at must be exact UTC ISO-8601")
    decisions = value.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != len(sample_rows):
        raise ValueError("OpenR1 owner decisions must cover every sampled row")
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
            raise ValueError("OpenR1 owner decision fields changed")
        record_id = decision.get("record_id")
        if not isinstance(record_id, str) or record_id not in expected_ids:
            raise ValueError("OpenR1 owner decision contains an unknown record")
        if record_id in observed_ids:
            raise ValueError("OpenR1 owner decision contains a duplicate record")
        accepted = decision.get("accepted")
        note = decision.get("note")
        if not isinstance(accepted, bool):
            raise ValueError("OpenR1 owner decision must be boolean")
        if not isinstance(note, str) or len(note) > 2_000:
            raise ValueError("OpenR1 owner decision note is invalid")
        observed_ids.add(record_id)
        (accepted_ids if accepted else rejected_ids).append(record_id)
    if observed_ids != expected_ids:
        raise ValueError("OpenR1 owner decisions do not match the sample")
    status = (
        "owner_visible_final_sample_passed_rights_and_promotion_still_separate"
        if not rejected_ids
        else "owner_visible_final_sample_rejected_rows_require_repair"
    )
    receipt: dict[str, Any] = {
        "accepted_records": len(accepted_ids),
        "all_sampled_rows_accepted": not rejected_ids,
        "authorization": EXPECTED_AUTHORIZATION,
        "bindings": {
            "decision_file_sha256": decision_file_sha256,
            "owner_sample_internal_sha256": sample_sha256,
            "plan_file_sha256": plan_file_sha256,
            "report_internal_sha256": EXPECTED_CANDIDATE["report_internal_sha256"],
            "verification_internal_sha256": EXPECTED_CANDIDATE[
                "verification_internal_sha256"
            ],
        },
        "owner_review_claim": (
            "upstream_synthetic_verified_math_plus_human_owner_visible_final_"
            "quality_reviewed_not_first_party_authored"
        ),
        "production_eligible": False,
        "production_promotions": 0,
        "raw_training_text_retained": False,
        "rejected_record_ids": sorted(rejected_ids),
        "rejected_records": len(rejected_ids),
        "remaining_blockers": [
            "upstream_problem_source_rights_map_unresolved",
            "product_markdown_math_renderer_not_yet_verified",
            "source_registry_promotion_not_authorized",
            "complete_sft_learning_mix_not_ready",
            "production_dataset_seal_and_sft_not_authorized",
        ],
        "reviewed_at": reviewed_at,
        "reviewer_id": reviewer.strip(),
        "sample_records": len(sample_rows),
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": status,
    }
    receipt["receipt_sha256"] = sha256_value(receipt)
    return receipt


def validate_owner_review_receipt(receipt: Mapping[str, Any]) -> None:
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported OpenR1 owner-review receipt")
    seal = _require_sha256(receipt.get("receipt_sha256"), "receipt_sha256")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if sha256_value(unsigned) != seal:
        raise ValueError("OpenR1 owner-review receipt seal is invalid")
    if receipt.get("raw_training_text_retained") is not False:
        raise ValueError("OpenR1 owner-review receipt retained raw training text")
    if receipt.get("production_eligible") is not False:
        raise ValueError("OpenR1 owner review crossed production eligibility")
    if receipt.get("production_promotions") != 0:
        raise ValueError("OpenR1 owner review crossed a promotion boundary")

