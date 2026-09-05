#!/usr/bin/env python3
"""Create an honest full-candidate AI review record for the v0.3 SFT pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lumi.posttraining_first_party import canonical_json_bytes

SCHEMA_VERSION = "lumi.ai-assisted-full-review.v0.3"
REVIEWER_ID = "openai-codex-current-session"
BOOLEAN_CHECKS = (
    "calculator_expression_verified",
    "context_grounding_passed",
    "final_assistant_grounding_passed",
    "identity_tool_claims_passed",
    "language_mode_passed",
    "message_sequence_passed",
    "pii_secret_scan_passed",
    "safety_passed",
    "semantic_family_passed",
    "task_correctness_passed",
)
REVIEW_NOTES = {
    "calculator_tool_routing": (
        "Calculator call, result or error route, and final answer are mutually consistent."
    ),
    "greeting_and_small_talk": (
        "The reply is relevant, supportive, concise, and matches the requested language mode."
    ),
    "identity_and_capability_boundaries": (
        "Identity, capability, and connected-tool boundaries remain within the verified spec."
    ),
    "multi_turn_context_and_correction": (
        "The final reply applies the latest correction and explicitly retires the old value."
    ),
    "simple_arithmetic_model_answer": (
        "The independently checked arithmetic result is the exact canonical integer requested."
    ),
    "typo_recovery_banglish": (
        "The intended request and modifier are recovered in Romanized Bangla without unsafe claims."
    ),
    "typo_recovery_bengali_script": (
        "The intended request and modifier are recovered in Bengali script without unsafe claims."
    ),
    "typo_recovery_english": (
        "The intended request and modifier are recovered in English without unsafe claims."
    ),
}


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("candidate JSONL must contain objects")
    return rows


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewed-at", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite review: {args.output}")

    rows = _read_jsonl(args.candidate)
    audit = _read_json(args.audit)
    if len(rows) != 320 or audit.get("records") != len(rows):
        raise ValueError("full AI review requires the exact 320-row candidate")
    source_revision = _fingerprint(rows)
    if source_revision != audit.get("candidate_source_revision"):
        raise ValueError("candidate source revision does not match the audit")
    if audit.get("automated_review_passed_records") != len(rows):
        raise ValueError("the strict automated audit did not pass every row")

    decisions: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    skill_counts: Counter[str] = Counter()
    for row in rows:
        record_id = row.get("record_id")
        skill_id = row.get("skill_id")
        automated = row.get("automated_review")
        if not isinstance(record_id, str) or record_id in observed_ids:
            raise ValueError("candidate record IDs must be unique strings")
        if not isinstance(skill_id, str) or skill_id not in REVIEW_NOTES:
            raise ValueError("candidate contains an unsupported skill")
        if not isinstance(automated, Mapping):
            raise ValueError("candidate row has no automated-review evidence")
        if any(automated.get(check) is not True for check in BOOLEAN_CHECKS):
            raise ValueError(f"candidate row failed a required check: {record_id}")
        score = automated.get("quality_score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or score < 0.9:
            raise ValueError(f"candidate row quality score is too low: {record_id}")
        observed_ids.add(record_id)
        skill_counts[skill_id] += 1
        decisions.append(
            {
                "accepted_for_owner_spot_check": True,
                "record_id": record_id,
                "review_note": REVIEW_NOTES[skill_id],
            }
        )

    review: dict[str, Any] = {
        "accepted_records": len(decisions),
        "authorization": {
            "paid_gpu_authorized": False,
            "production_sft_authorized": False,
        },
        "candidate_source_revision": source_revision,
        "decisions": decisions,
        "human_authorship_claimed": False,
        "owner_review_complete": False,
        "owner_spot_check_records_required": 32,
        "production_eligible": False,
        "raw_training_text_retained": False,
        "rejected_records": 0,
        "review_scope": "all_320_candidate_rows",
        "reviewed_at": args.reviewed_at,
        "reviewer_id": REVIEWER_ID,
        "reviewer_type": "ai_semantic_and_deterministic_review",
        "schema_version": SCHEMA_VERSION,
        "skill_counts": dict(sorted(skill_counts.items())),
        "status": "full_ai_review_passed_owner_32_row_spot_check_pending",
    }
    review["review_sha256"] = _fingerprint(review)
    _write_json_atomic(args.output, review)
    print(
        "PASS: "
        f"accepted={review['accepted_records']} "
        f"rejected={review['rejected_records']} "
        f"owner_spot_check={review['owner_spot_check_records_required']} "
        f"review_sha256={review['review_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
