"""Fail-closed templates for Lumi-owned human-authored SFT records."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

AUTHORING_PLAN_SCHEMA_VERSION = "lumi.first-party-authoring-plan.v0.1"
AUTHORING_ROW_SCHEMA_VERSION = "lumi.first-party-authoring-row.v0.1"
AUTHORING_AUDIT_SCHEMA_VERSION = "lumi.first-party-authoring-audit.v0.1"
EXPECTED_SKILLS = frozenset(
    {
        "calculator_tool_routing",
        "greeting_and_small_talk",
        "identity_and_capability_boundaries",
        "multi_turn_context_and_correction",
        "simple_arithmetic_model_answer",
        "typo_recovery_banglish",
        "typo_recovery_bengali_script",
        "typo_recovery_english",
    }
)
EXPECTED_AUTHORIZATION = {
    "external_distribution_authorized": False,
    "human_authoring_template_authorized": True,
    "model_generated_training_text_authorized": False,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "raw_training_text_in_git": False,
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contains_forbidden_payload_field(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key in {"assistant_response", "examples", "messages", "prompt", "user_message"}:
                return True
            if _contains_forbidden_payload_field(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_payload_field(item) for item in value)
    return False


def load_first_party_authoring_plan(path: str | Path) -> dict[str, Any]:
    """Load a bounded metadata-only authoring plan and reject hidden text payloads."""

    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("first-party authoring plan must be an object")
    if value.get("schema_version") != AUTHORING_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported first-party authoring plan schema")
    if value.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("first-party authoring authorization boundary changed")
    if value.get("status") != "bounded_blank_template_ready_no_authored_records":
        raise ValueError("first-party authoring plan status changed")
    if _contains_forbidden_payload_field(value):
        raise ValueError("first-party authoring plan may not contain raw prompt or response text")

    source = value.get("source_binding")
    expected_source = {
        "dataset_id": "Afnan-Technologies-Inc/lumi-owned-sft-v0.1",
        "license_id": "lumi-owned",
        "source_id": "lumi_owned_sft_v0_1",
    }
    if source != expected_source:
        raise ValueError("first-party source binding changed")

    coverage = value.get("coverage")
    if not isinstance(coverage, list) or not coverage:
        raise ValueError("first-party authoring coverage must be non-empty")
    skills = [item.get("skill_id") for item in coverage if isinstance(item, Mapping)]
    if len(skills) != len(coverage) or set(skills) != EXPECTED_SKILLS:
        raise ValueError("first-party authoring skill coverage changed")
    if len(skills) != len(set(skills)):
        raise ValueError("first-party authoring skill IDs must be unique")

    train_total = 0
    validation_total = 0
    for item in coverage:
        if not isinstance(item, Mapping):
            raise ValueError("first-party coverage entry must be an object")
        for field in ("task_category", "skill_id"):
            if not isinstance(item.get(field), str) or not item[field]:
                raise ValueError(f"first-party coverage {field} is invalid")
        modes = item.get("language_modes")
        if not isinstance(modes, list) or not modes or not set(modes).issubset(
            {"bn", "en", "multilingual"}
        ):
            raise ValueError("first-party coverage language modes are invalid")
        checks = item.get("acceptance_checks")
        if not isinstance(checks, list) or not checks or any(
            not isinstance(check, str) or not check for check in checks
        ):
            raise ValueError("first-party acceptance checks are invalid")
        train_total += _positive_count(item.get("train_records"), "train_records")
        validation_total += _positive_count(
            item.get("validation_records"), "validation_records"
        )

    totals = value.get("totals")
    if totals != {
        "train_records": train_total,
        "validation_records": validation_total,
    }:
        raise ValueError("first-party authoring totals do not match coverage")
    if totals != {"train_records": 240, "validation_records": 80}:
        raise ValueError("first-party authoring pilot bounds changed")

    field_contract = value.get("field_contract")
    if not isinstance(field_contract, Mapping):
        raise ValueError("first-party field contract is missing")
    if field_contract.get("required_text_fields") != [
        "user_message",
        "assistant_response",
    ]:
        raise ValueError("first-party authoring text fields changed")
    if field_contract.get("template_text_fields_must_start_blank") is not True:
        raise ValueError("first-party text fields must start blank")
    return value


def _positive_count(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def build_blank_authoring_template(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build blank rows for human writing without generating training text."""

    rows: list[dict[str, Any]] = []
    for item in plan["coverage"]:
        modes = item["language_modes"]
        split_counts = (
            ("train", "train_records"),
            ("validation", "validation_records"),
        )
        for split, count_field in split_counts:
            for index in range(item[count_field]):
                rows.append(
                    {
                        "assistant_response": "",
                        "author_declaration": {
                            "author_id": "",
                            "authored_at": "",
                            "human_written_from_scratch": None,
                            "rights_confirmed": None,
                        },
                        "independent_review": {
                            "accepted": None,
                            "quality_score": None,
                            "reviewer_id": "",
                            "safety_review_passed": None,
                        },
                        "language": modes[index % len(modes)],
                        "record_id": f"{item['skill_id']}-{split}-{index:04d}",
                        "schema_version": AUTHORING_ROW_SCHEMA_VERSION,
                        "skill_id": item["skill_id"],
                        "split": split,
                        "task_category": item["task_category"],
                        "user_message": "",
                    }
                )
    return rows


def validate_completed_authoring_row(
    row: Mapping[str, Any],
    *,
    minimum_quality_score: float = 0.75,
) -> dict[str, Any]:
    """Require human authorship, independent review, rights, safety, and non-empty text."""

    if row.get("schema_version") != AUTHORING_ROW_SCHEMA_VERSION:
        raise ValueError("unsupported first-party authoring row schema")
    for field in ("record_id", "skill_id", "split", "task_category", "language"):
        if not isinstance(row.get(field), str) or not row[field]:
            raise ValueError(f"completed first-party row has invalid {field}")
    for field in ("user_message", "assistant_response"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(f"completed first-party row has empty {field}")

    declaration = row.get("author_declaration")
    review = row.get("independent_review")
    if not isinstance(declaration, Mapping) or not isinstance(review, Mapping):
        raise ValueError("completed first-party row lacks author or review evidence")
    author_id = declaration.get("author_id")
    reviewer_id = review.get("reviewer_id")
    if not isinstance(author_id, str) or not author_id:
        raise ValueError("completed first-party row lacks author ID")
    if not isinstance(reviewer_id, str) or not reviewer_id:
        raise ValueError("completed first-party row lacks reviewer ID")
    if author_id == reviewer_id:
        raise ValueError("first-party author and reviewer must differ")
    if declaration.get("human_written_from_scratch") is not True:
        raise ValueError("first-party row lacks original human authorship declaration")
    if declaration.get("rights_confirmed") is not True:
        raise ValueError("first-party row lacks rights declaration")
    if review.get("accepted") is not True or review.get("safety_review_passed") is not True:
        raise ValueError("first-party row has not passed independent review")
    score = review.get("quality_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("first-party row quality score is invalid")
    if float(score) < minimum_quality_score or float(score) > 1:
        raise ValueError("first-party row quality score is outside the accepted range")
    return json.loads(json.dumps(row, ensure_ascii=False))


def _blank_authoring_row(row: Mapping[str, Any]) -> bool:
    return (
        row.get("user_message") == ""
        and row.get("assistant_response") == ""
        and row.get("author_declaration")
        == {
            "author_id": "",
            "authored_at": "",
            "human_written_from_scratch": None,
            "rights_confirmed": None,
        }
        and row.get("independent_review")
        == {
            "accepted": None,
            "quality_score": None,
            "reviewer_id": "",
            "safety_review_passed": None,
        }
    )


def _partial_row_invalid_reasons(
    row: Mapping[str, Any],
    *,
    minimum_quality_score: float,
) -> tuple[str, ...]:
    """Return only hard-invalid reasons; incomplete evidence remains in progress."""

    reasons: list[str] = []
    for field in ("user_message", "assistant_response"):
        if not isinstance(row.get(field), str):
            reasons.append(f"{field}_type")

    declaration = row.get("author_declaration")
    if not isinstance(declaration, Mapping):
        reasons.append("author_declaration_type")
    else:
        for field in ("author_id", "authored_at"):
            if not isinstance(declaration.get(field), str):
                reasons.append(f"author_declaration_{field}_type")
        if declaration.get("human_written_from_scratch") not in {None, True}:
            reasons.append("human_authorship_rejected")
        if declaration.get("rights_confirmed") not in {None, True}:
            reasons.append("rights_rejected")

    review = row.get("independent_review")
    if not isinstance(review, Mapping):
        reasons.append("independent_review_type")
    else:
        if not isinstance(review.get("reviewer_id"), str):
            reasons.append("reviewer_id_type")
        if review.get("accepted") not in {None, True}:
            reasons.append("review_rejected")
        if review.get("safety_review_passed") not in {None, True}:
            reasons.append("safety_review_rejected")
        score = review.get("quality_score")
        if score is not None and (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not minimum_quality_score <= float(score) <= 1
        ):
            reasons.append("quality_score_invalid")

    if isinstance(declaration, Mapping) and isinstance(review, Mapping):
        author_id = declaration.get("author_id")
        reviewer_id = review.get("reviewer_id")
        if author_id and author_id == reviewer_id:
            reasons.append("author_reviewer_not_independent")
    return tuple(sorted(set(reasons)))


def _incomplete_row_reasons(row: Mapping[str, Any]) -> tuple[str, ...]:
    reasons: list[str] = []
    for field in ("user_message", "assistant_response"):
        value = row.get(field)
        if isinstance(value, str) and not value.strip():
            reasons.append(f"missing_{field}")

    declaration = row.get("author_declaration")
    if isinstance(declaration, Mapping):
        if not declaration.get("author_id"):
            reasons.append("missing_author_id")
        if not declaration.get("authored_at"):
            reasons.append("missing_authored_at")
        if declaration.get("human_written_from_scratch") is not True:
            reasons.append("missing_human_authorship_declaration")
        if declaration.get("rights_confirmed") is not True:
            reasons.append("missing_rights_declaration")

    review = row.get("independent_review")
    if isinstance(review, Mapping):
        if review.get("accepted") is not True:
            reasons.append("missing_review_acceptance")
        if review.get("quality_score") is None:
            reasons.append("missing_quality_score")
        if not review.get("reviewer_id"):
            reasons.append("missing_reviewer_id")
        if review.get("safety_review_passed") is not True:
            reasons.append("missing_safety_review")
    return tuple(sorted(set(reasons)))


def audit_first_party_authoring_rows(
    rows: list[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit the exact bounded pilot without retaining prompt or response text."""

    expected_rows = build_blank_authoring_template(plan)
    expected_by_id = {row["record_id"]: row for row in expected_rows}
    observed_by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("first-party authoring row must be an object")
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or record_id not in expected_by_id:
            raise ValueError("first-party authoring rows contain an unknown record ID")
        if record_id in observed_by_id:
            raise ValueError("first-party authoring rows contain a duplicate record ID")
        observed_by_id[record_id] = row
    if set(observed_by_id) != set(expected_by_id):
        raise ValueError("first-party authoring rows do not contain the exact planned IDs")

    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    accepted_skill_counts: Counter[str] = Counter()
    accepted_split_counts: Counter[str] = Counter()
    accepted_language_counts: Counter[str] = Counter()
    minimum_quality_score = float(plan["quality_policy"]["minimum_quality_score"])
    immutable_fields = (
        "schema_version",
        "skill_id",
        "split",
        "task_category",
        "language",
    )

    for record_id in sorted(expected_by_id):
        row = observed_by_id[record_id]
        expected = expected_by_id[record_id]
        if any(row.get(field) != expected[field] for field in immutable_fields):
            raise ValueError("first-party authoring row immutable metadata changed")

        if _blank_authoring_row(row):
            status_counts["blank"] += 1
            continue

        invalid_reasons = _partial_row_invalid_reasons(
            row,
            minimum_quality_score=minimum_quality_score,
        )
        if invalid_reasons:
            status_counts["invalid"] += 1
            reason_counts.update(invalid_reasons)
            continue

        incomplete_reasons = _incomplete_row_reasons(row)
        if incomplete_reasons:
            status_counts["in_progress"] += 1
            reason_counts.update(incomplete_reasons)
            continue

        validate_completed_authoring_row(
            row,
            minimum_quality_score=minimum_quality_score,
        )
        status_counts["accepted"] += 1
        accepted_skill_counts[row["skill_id"]] += 1
        accepted_split_counts[row["split"]] += 1
        accepted_language_counts[row["language"]] += 1

    for status in ("accepted", "blank", "in_progress", "invalid"):
        status_counts[status] += 0
    authoring_complete = status_counts == Counter(
        {"accepted": len(expected_rows), "blank": 0, "in_progress": 0, "invalid": 0}
    )
    report: dict[str, Any] = {
        "accepted_language_counts": dict(sorted(accepted_language_counts.items())),
        "accepted_skill_counts": dict(sorted(accepted_skill_counts.items())),
        "accepted_split_counts": dict(sorted(accepted_split_counts.items())),
        "authorization": {
            "paid_gpu_authorized": False,
            "production_sft_authorized": False,
        },
        "authoring_complete": authoring_complete,
        "expected_rows": len(expected_rows),
        "observed_rows": len(rows),
        "plan_sha256": hashlib.sha256(canonical_json_bytes(plan)).hexdigest(),
        "raw_training_text_retained": False,
        "reason_counts": dict(sorted(reason_counts.items())),
        "rows_sha256": hashlib.sha256(canonical_json_bytes(rows)).hexdigest(),
        "schema_version": AUTHORING_AUDIT_SCHEMA_VERSION,
        "status_counts": dict(sorted(status_counts.items())),
    }
    report["report_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    return report
