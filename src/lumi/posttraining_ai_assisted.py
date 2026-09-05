"""Fail-closed preparation for Lumi-owned AI-assisted SFT candidate records."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .data_quality import normalize_for_matching
from .posttraining_capabilities import parse_single_integer_answer
from .posttraining_data import validate_message_sequence
from .posttraining_first_party import (
    build_blank_authoring_template,
    canonical_json_bytes,
    file_sha256,
    load_first_party_authoring_plan,
)

AI_ASSISTED_PLAN_SCHEMA_VERSION = "lumi.ai-assisted-sft-pilot.v0.1"
AI_ASSISTED_ROW_SCHEMA_VERSION = "lumi.ai-assisted-sft-row.v0.1"
AI_ASSISTED_AUDIT_SCHEMA_VERSION = "lumi.ai-assisted-sft-audit.v0.1"
AI_ASSISTED_MANIFEST_SCHEMA_VERSION = "lumi.ai-assisted-sft-manifest.v0.1"
AI_ASSISTED_EVIDENCE_SCHEMA_VERSION = "lumi.ai-assisted-sft-candidate-evidence.v0.1"
EXPECTED_AUTHORIZATION = {
    "external_distribution_authorized": False,
    "external_model_api_spend_authorized": False,
    "human_authorship_claim_authorized": False,
    "model_generated_training_text_authorized": True,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "raw_training_text_in_git": False,
}
_BENGALI = re.compile(r"[\u0980-\u09ff]")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\s().-]*){10,15}(?!\d)")
_SECRET = re.compile(
    r"(?:BEGIN (?:[A-Z ]+ )?PRIVATE KEY|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|"
    r"\b(?:api[_-]?key|access[_-]?token|secret|password)\s*[:=])",
    re.IGNORECASE,
)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def load_ai_assisted_pilot_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the bounded AI-assisted plan and its coverage-only binding."""

    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("AI-assisted pilot plan must be an object")
    if value.get("schema_version") != AI_ASSISTED_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported AI-assisted pilot plan schema")
    if value.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("AI-assisted pilot authorization boundary changed")
    if value.get("status") != "candidate_generation_authorized_owner_spot_check_pending":
        raise ValueError("AI-assisted pilot status changed")

    binding = value.get("coverage_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("AI-assisted coverage binding is missing")
    relative_path = binding.get("path")
    if relative_path != "configs/lumi-4b-first-party-authoring-pilot-v0.1.json":
        raise ValueError("AI-assisted coverage path changed")
    if binding.get("use") != (
        "record_ids_skill_language_and_split_counts_only_not_human_authorship_policy"
    ):
        raise ValueError("AI-assisted coverage use boundary changed")
    coverage_path = Path(repository_root) / relative_path
    if file_sha256(coverage_path) != binding.get("sha256"):
        raise ValueError("AI-assisted coverage binding changed")
    coverage_plan = load_first_party_authoring_plan(coverage_path)

    generation = value.get("generation_policy")
    if not isinstance(generation, Mapping):
        raise ValueError("AI-assisted generation policy is missing")
    if generation.get("method") != "ai_generated_with_deterministic_task_templates":
        raise ValueError("AI-assisted generation method changed")
    generators = generation.get("allowed_generator_ids")
    if generators != ["openai-codex-current-session"]:
        raise ValueError("AI-assisted generator allowlist changed")
    if generation.get("generator_outputs_must_not_be_claimed_human_written") is not True:
        raise ValueError("AI-assisted human-authorship boundary changed")
    if generation.get("rights_confirmation_required") is not True:
        raise ValueError("AI-assisted rights policy changed")

    review = value.get("automated_review")
    if not isinstance(review, Mapping):
        raise ValueError("AI-assisted automated review policy is missing")
    score = review.get("minimum_quality_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("AI-assisted minimum quality score is invalid")
    if not 0.75 <= float(score) <= 1:
        raise ValueError("AI-assisted minimum quality score is outside safe bounds")
    _required_string(review.get("reviewer_id"), "automated_review.reviewer_id")

    owner_gate = value.get("owner_gate")
    if not isinstance(owner_gate, Mapping):
        raise ValueError("AI-assisted owner gate is missing")
    if owner_gate != {
        "dataset_level_signoff_required": True,
        "independent_second_human_required": False,
        "per_row_human_review_required": False,
        "sample_records": 32,
        "sampling": "four_records_per_skill_with_train_validation_coverage_v0.1",
        "status": "pending",
    }:
        raise ValueError("AI-assisted owner gate changed")
    source = value.get("source_binding")
    if source != {
        "dataset_id": "Afnan-Technologies-Inc/lumi-ai-assisted-sft-v0.1",
        "license_id": "lumi-owned",
        "source_id": "lumi_ai_assisted_sft_v0_1",
    }:
        raise ValueError("AI-assisted source binding changed")
    return value, coverage_plan


def expected_candidate_metadata(coverage_plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the exact 320-row metadata contract reused from the coverage pilot."""

    rows = build_blank_authoring_template(coverage_plan)
    return {
        row["record_id"]: {
            "language": row["language"],
            "record_id": row["record_id"],
            "skill_id": row["skill_id"],
            "split": row["split"],
            "task_category": row["task_category"],
        }
        for row in rows
    }


def load_ai_assisted_candidate_evidence(path: str | Path) -> dict[str, Any]:
    """Validate the raw-text-free bridge from private candidate artifacts to planning."""

    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("AI-assisted candidate evidence must be an object")
    if value.get("schema_version") != AI_ASSISTED_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("unsupported AI-assisted candidate evidence schema")
    if value.get("status") != "machine_validated_candidate_waiting_for_owner_spot_check":
        raise ValueError("AI-assisted candidate evidence status changed")
    if value.get("authorization") != {
        "paid_gpu_authorized": False,
        "production_sft_authorized": False,
    }:
        raise ValueError("AI-assisted candidate evidence authorization changed")
    if value.get("production_eligible") is not False:
        raise ValueError("AI-assisted candidate may not claim production eligibility")
    candidate = value.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("AI-assisted candidate evidence is missing candidate counts")
    for field in (
        "automated_review_passed_records",
        "normalized_unique_records",
        "records",
        "sft_filter_accepted_records",
    ):
        if candidate.get(field) != 320:
            raise ValueError(f"AI-assisted candidate evidence count changed: {field}")
    if candidate.get("sft_filter_rejected_records") != 0:
        raise ValueError("AI-assisted SFT filter has rejected records")
    for field in (
        "candidate_manifest_sha256",
        "candidate_source_revision",
        "candidate_sft_jsonl_sha256",
        "machine_audit_report_sha256",
        "sft_filter_report_sha256",
    ):
        if not isinstance(candidate.get(field), str) or _SHA256.fullmatch(
            candidate[field]
        ) is None:
            raise ValueError(f"AI-assisted candidate evidence hash is invalid: {field}")
    owner_gate = value.get("owner_gate")
    if not isinstance(owner_gate, Mapping):
        raise ValueError("AI-assisted candidate owner gate is missing")
    if set(owner_gate) != {"complete", "sample_records", "sample_sha256", "status"}:
        raise ValueError("AI-assisted candidate owner gate fields changed")
    if (
        owner_gate.get("complete") is not False
        or owner_gate.get("sample_records") != 32
        or owner_gate.get("status") != "pending"
    ):
        raise ValueError("AI-assisted candidate owner gate changed")
    sample_sha256 = owner_gate["sample_sha256"]
    if not isinstance(sample_sha256, str) or _SHA256.fullmatch(sample_sha256) is None:
        raise ValueError("AI-assisted owner sample SHA-256 is invalid")
    private_artifact = value.get("private_artifact")
    if not isinstance(private_artifact, Mapping):
        raise ValueError("AI-assisted private artifact pointer is missing")
    if private_artifact.get("raw_training_text_in_this_evidence") is not False:
        raise ValueError("AI-assisted evidence may not contain raw training text")
    _required_string(private_artifact.get("path"), "private_artifact.path")
    return value


def _message_text(messages: list[Mapping[str, str]]) -> str:
    return "\n".join(message["content"] for message in messages)


def _assistant_text(messages: list[Mapping[str, str]]) -> str:
    return "\n".join(
        message["content"] for message in messages if message["role"] == "assistant"
    )


def _validate_task_evidence(row: Mapping[str, Any], messages: list[Mapping[str, str]]) -> None:
    skill_id = row["skill_id"]
    evidence = row.get("task_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("AI-assisted row lacks task evidence")
    assistant_text = _assistant_text(messages)

    if skill_id == "greeting_and_small_talk":
        if "Afnan Technologies" in assistant_text:
            raise ValueError("ordinary greeting repeats the company introduction")
    elif skill_id in {
        "typo_recovery_banglish",
        "typo_recovery_bengali_script",
        "typo_recovery_english",
    }:
        if evidence.get("intent_recovered") is not True:
            raise ValueError("typo recovery evidence is missing")
    elif skill_id == "simple_arithmetic_model_answer":
        expected = evidence.get("expected_integer")
        if isinstance(expected, bool) or not isinstance(expected, int):
            raise ValueError("arithmetic expected integer is invalid")
        if parse_single_integer_answer(messages[-1]["content"]) != expected:
            raise ValueError("arithmetic response does not match expected integer")
    elif skill_id == "calculator_tool_routing":
        if [message["role"] for message in messages] != [
            "user",
            "tool",
            "tool_result",
            "assistant",
        ]:
            raise ValueError("calculator route must contain an exact tool trajectory")
        expected = str(evidence.get("expected_result"))
        if expected not in messages[2]["content"] or expected not in messages[3]["content"]:
            raise ValueError("calculator result evidence does not match the trajectory")
    elif skill_id == "multi_turn_context_and_correction":
        if len(messages) < 4 or evidence.get("context_retained") is not True:
            raise ValueError("multi-turn context evidence is missing")
    elif skill_id == "identity_and_capability_boundaries":
        if "Lumi" not in assistant_text or "Afnan Technologies Inc." not in assistant_text:
            raise ValueError("identity response lacks the canonical identity")
        if evidence.get("verified_claims_only") is not True:
            raise ValueError("identity claim evidence is missing")
    else:  # pragma: no cover - coverage is closed by the bound plan
        raise ValueError(f"unsupported AI-assisted skill: {skill_id}")


def validate_ai_assisted_candidate_row(
    row: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one AI-generated row without turning it into production evidence."""

    if row.get("schema_version") != AI_ASSISTED_ROW_SCHEMA_VERSION:
        raise ValueError("unsupported AI-assisted row schema")
    for field in ("record_id", "skill_id", "split", "task_category", "language"):
        if row.get(field) != expected.get(field):
            raise ValueError(f"AI-assisted row immutable metadata changed: {field}")

    raw_messages = row.get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError("AI-assisted messages must be an array")
    messages: list[Mapping[str, str]] = []
    for message in raw_messages:
        if not isinstance(message, Mapping):
            raise ValueError("AI-assisted message must be an object")
        if set(message) != {"content", "role"}:
            raise ValueError("AI-assisted message fields changed")
        if not isinstance(message.get("role"), str) or not isinstance(
            message.get("content"), str
        ):
            raise ValueError("AI-assisted message types are invalid")
        messages.append(message)
    validate_message_sequence(messages)
    combined_text = _message_text(messages)
    if _EMAIL.search(combined_text) or _PHONE.search(combined_text) or _SECRET.search(
        combined_text
    ):
        raise ValueError("AI-assisted row contains PII or secret-like text")

    language = row["language"]
    contains_bengali = _BENGALI.search(combined_text) is not None
    if language == "bn" and not contains_bengali:
        raise ValueError("Bengali row lacks Bengali script")
    if language == "en" and contains_bengali:
        raise ValueError("English row unexpectedly contains Bengali script")

    generation = row.get("generation")
    if not isinstance(generation, Mapping):
        raise ValueError("AI-assisted generation evidence is missing")
    if generation.get("generator_id") not in plan["generation_policy"][
        "allowed_generator_ids"
    ]:
        raise ValueError("AI-assisted generator is not allowed")
    if generation.get("method") != plan["generation_policy"]["method"]:
        raise ValueError("AI-assisted generation method changed")
    if generation.get("human_written_from_scratch") is not False:
        raise ValueError("AI-assisted row must not claim human authorship")
    if generation.get("rights_confirmed") is not True:
        raise ValueError("AI-assisted row lacks rights confirmation")
    _required_string(generation.get("generated_at"), "generation.generated_at")

    review = row.get("automated_review")
    if not isinstance(review, Mapping):
        raise ValueError("AI-assisted automated review evidence is missing")
    if review.get("reviewer_id") != plan["automated_review"]["reviewer_id"]:
        raise ValueError("AI-assisted automated reviewer changed")
    for field in (
        "identity_tool_claims_passed",
        "language_mode_passed",
        "message_sequence_passed",
        "pii_secret_scan_passed",
        "safety_passed",
        "task_correctness_passed",
    ):
        if review.get(field) is not True:
            raise ValueError(f"AI-assisted automated review failed: {field}")
    score = review.get("quality_score")
    minimum = float(plan["automated_review"]["minimum_quality_score"])
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not minimum <= float(score) <= 1
    ):
        raise ValueError("AI-assisted automated quality score is invalid")

    owner = row.get("owner_review")
    if owner != {
        "accepted": None,
        "reviewed_at": "",
        "reviewer_id": "",
        "status": "pending",
    }:
        raise ValueError("AI-assisted row may not claim completed owner review")
    _validate_task_evidence(row, messages)
    return json.loads(json.dumps(row, ensure_ascii=False))


def candidate_source_revision(rows: list[Mapping[str, Any]]) -> str:
    """Hash the exact ordered AI-assisted source rows before SFT transformation."""

    return _fingerprint(rows)


def audit_ai_assisted_candidate_rows(
    rows: list[Mapping[str, Any]],
    *,
    coverage_plan: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit exact membership, provenance, review evidence, and normalized uniqueness."""

    expected_by_id = expected_candidate_metadata(coverage_plan)
    observed_by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("AI-assisted row must be an object")
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or record_id not in expected_by_id:
            raise ValueError("AI-assisted rows contain an unknown record ID")
        if record_id in observed_by_id:
            raise ValueError("AI-assisted rows contain a duplicate record ID")
        observed_by_id[record_id] = row
    if set(observed_by_id) != set(expected_by_id):
        raise ValueError("AI-assisted rows do not contain the exact planned IDs")

    normalized: set[str] = set()
    skill_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    for record_id in sorted(expected_by_id):
        row = validate_ai_assisted_candidate_row(
            observed_by_id[record_id],
            expected=expected_by_id[record_id],
            plan=plan,
        )
        conversation = [
            {
                "content": normalize_for_matching(message["content"]),
                "role": message["role"],
            }
            for message in row["messages"]
        ]
        normalized_sha256 = _fingerprint(conversation)
        if normalized_sha256 in normalized:
            raise ValueError("AI-assisted rows contain a normalized duplicate conversation")
        normalized.add(normalized_sha256)
        skill_counts[row["skill_id"]] += 1
        split_counts[row["split"]] += 1
        language_counts[row["language"]] += 1

    report: dict[str, Any] = {
        "authorization": {
            "paid_gpu_authorized": False,
            "production_sft_authorized": False,
        },
        "automated_review_passed_records": len(rows),
        "candidate_source_revision": candidate_source_revision(rows),
        "language_counts": dict(sorted(language_counts.items())),
        "normalized_unique_records": len(normalized),
        "owner_review": {
            "complete": False,
            "sample_records": plan["owner_gate"]["sample_records"],
            "status": "pending",
        },
        "plan_sha256": _fingerprint(plan),
        "production_eligible": False,
        "raw_training_text_retained": False,
        "records": len(rows),
        "schema_version": AI_ASSISTED_AUDIT_SCHEMA_VERSION,
        "skill_counts": dict(sorted(skill_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
    }
    report["report_sha256"] = _fingerprint(report)
    return report


def select_owner_review_sample(
    rows: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select three train and one validation row per skill for owner inspection."""

    by_skill: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_skill.setdefault(str(row["skill_id"]), []).append(row)
    selected: list[dict[str, Any]] = []
    for skill_id in sorted(by_skill):
        train = sorted(
            (row for row in by_skill[skill_id] if row["split"] == "train"),
            key=lambda row: (row["language"], row["record_id"]),
        )
        validation = sorted(
            (row for row in by_skill[skill_id] if row["split"] == "validation"),
            key=lambda row: (row["language"], row["record_id"]),
        )
        if len(train) < 3 or not validation:
            raise ValueError("AI-assisted owner sample coverage is incomplete")
        selected.extend(
            json.loads(json.dumps(row, ensure_ascii=False))
            for row in (*train[:3], validation[0])
        )
    if len(selected) != 32:
        raise AssertionError("AI-assisted owner sample size changed")
    return selected


def to_sft_candidate_records(
    rows: list[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    source_revision: str,
) -> list[dict[str, Any]]:
    """Transform validated rows into candidate SFT records with explicit AI provenance."""

    source = plan["source_binding"]
    transform_sha256 = _fingerprint(
        {
            "schema": "lumi.ai-assisted-to-sft.v0.1",
            "source_revision": source_revision,
        }
    )
    records: list[dict[str, Any]] = []
    for row in rows:
        messages = json.loads(json.dumps(row["messages"], ensure_ascii=False))
        records.append(
            {
                "language": row["language"],
                "messages": messages,
                "provenance": {
                    "consent_basis": "first_party_rights_declaration",
                    "content_sha256": _fingerprint(messages),
                    "generation_method": "ai_generated",
                    "generator_id": row["generation"]["generator_id"],
                    "human_review_status": "owner_spot_check_pending",
                    "transform_sha256": transform_sha256,
                },
                "quality": {
                    "pii_scan": "passed",
                    "review_method": "deterministic_automated_review_owner_pending",
                    "review_status": "accepted",
                    "score": row["automated_review"]["quality_score"],
                    "secret_scan": "passed",
                },
                "record_id": row["record_id"],
                "safety": {
                    "policy_version": "lumi-safety-v0.1",
                    "risk_labels": [],
                },
                "schema_version": "lumi.sft.v0.1",
                "source": {
                    "dataset_id": source["dataset_id"],
                    "document_id": row["record_id"],
                    "license_id": source["license_id"],
                    "revision": source_revision,
                    "source_id": source["source_id"],
                    "uri": f"private://{source['dataset_id']}/{row['record_id']}",
                },
                "split": row["split"],
                "task_category": row["task_category"],
            }
        )
    return records


def build_ai_assisted_manifest(
    *,
    audit: Mapping[str, Any],
    candidate_records: list[Mapping[str, Any]],
    owner_sample: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a raw-text-free candidate manifest that cannot authorize production."""

    manifest: dict[str, Any] = {
        "authorization": {
            "paid_gpu_authorized": False,
            "production_sft_authorized": False,
        },
        "audit_report_sha256": audit["report_sha256"],
        "candidate_records": len(candidate_records),
        "candidate_sft_jsonl_sha256": _fingerprint(candidate_records),
        "candidate_source_revision": audit["candidate_source_revision"],
        "owner_review_complete": False,
        "owner_sample_records": len(owner_sample),
        "owner_sample_sha256": _fingerprint(owner_sample),
        "production_eligible": False,
        "raw_training_text_retained": False,
        "schema_version": AI_ASSISTED_MANIFEST_SCHEMA_VERSION,
        "status": "machine_validated_candidate_waiting_for_owner_spot_check",
    }
    manifest["manifest_sha256"] = _fingerprint(manifest)
    return manifest
