"""Fail-closed English, Bengali, and Banglish punctuation quality gate."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PUNCTUATION_GATE_SCHEMA_VERSION = "lumi.punctuation-quality-gate.v0.1"
PUNCTUATION_REPORT_SCHEMA_VERSION = "lumi.punctuation-quality-report.v0.1"

LANGUAGE_MODES = (
    "english",
    "bengali_script",
    "banglish",
)
PUNCTUATION_OPERATIONS = (
    "missing_terminal_mark",
    "wrong_terminal_mark",
    "comma_and_clause_spacing",
    "quotation_and_bracket_pairing",
    "list_and_enumeration_punctuation",
    "sentence_boundary_and_run_on",
)
CONTENT_CONTEXTS = (
    "plain_prose",
    "code_or_command",
    "url_or_path",
    "decimal_version_or_math",
)
PROTECTED_CONTEXTS = frozenset(CONTENT_CONTEXTS) - {"plain_prose"}

EXPECTED_AUTHORIZATION = {
    "dataset_seal_authorized": False,
    "external_api_spend_authorized": False,
    "model_inference_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_release_authorized": False,
    "production_sft_authorized": False,
    "purpose": "metadata_only_punctuation_contract_and_future_post_sft_gate",
    "raw_prompt_or_response_rows_in_git": False,
}
EXPECTED_TRAINING_DATA_PREREQUISITES = {
    "allowed_provenance": ["human_reviewed", "ai_assisted_owner_reviewed"],
    "assistant_response_tokens_only_loss": True,
    "aya_owner_sample_is_naturalness_evidence_not_punctuation_proof": True,
    "examples_must_preserve_code_commands_urls_paths_decimals_versions_and_math": True,
    "meaning_preservation_required": True,
    "owner_review_required": True,
    "private_raw_rows_required": True,
    "train_validation_semantic_isolation_required": True,
}
EXPECTED_THRESHOLDS = {
    "human_naturalness_approval_rate": 0.95,
    "meaning_preservation_rate": 1.0,
    "overall_punctuation_accuracy": 0.95,
    "per_language_punctuation_accuracy": 0.90,
    "protected_span_preservation_rate": 1.0,
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class PunctuationLane:
    """One metadata-only coverage cell; it contains no prompt or answer text."""

    case_id: str
    content_context: str
    expected_action: str
    language_mode: str
    punctuation_operation: str


def load_punctuation_quality_plan(path: str | Path) -> dict[str, Any]:
    """Load the gate and reject authorization, coverage, or threshold drift."""

    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("punctuation-quality plan must be an object")
    if value.get("schema_version") != PUNCTUATION_GATE_SCHEMA_VERSION:
        raise ValueError("unsupported punctuation-quality schema")
    if value.get("status") != "gate_defined_not_executed":
        raise ValueError("punctuation-quality gate status changed")
    if value.get("observed_at") != "2026-08-06":
        raise ValueError("punctuation-quality gate date changed")
    if value.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("punctuation-quality authorization boundary changed")
    if value.get("training_data_seal_prerequisites") != (EXPECTED_TRAINING_DATA_PREREQUISITES):
        raise ValueError("punctuation training-data prerequisites changed")

    matrix = value.get("metadata_lane_matrix")
    if not isinstance(matrix, Mapping):
        raise ValueError("punctuation metadata matrix is missing")
    if matrix.get("language_modes") != list(LANGUAGE_MODES):
        raise ValueError("punctuation language modes changed")
    if matrix.get("punctuation_operations") != list(PUNCTUATION_OPERATIONS):
        raise ValueError("punctuation operations changed")
    if matrix.get("content_contexts") != list(CONTENT_CONTEXTS):
        raise ValueError("punctuation content contexts changed")
    expected_lanes = len(LANGUAGE_MODES) * len(PUNCTUATION_OPERATIONS) * len(CONTENT_CONTEXTS)
    if matrix.get("expected_lanes") != expected_lanes or expected_lanes != 72:
        raise ValueError("punctuation metadata lane bound changed")
    if matrix.get("future_prompts_per_lane") != 4:
        raise ValueError("punctuation prompt replication bound changed")
    if matrix.get("minimum_post_sft_model_cases") != 288:
        raise ValueError("punctuation model-case bound changed")

    behavior = value.get("model_behavior_contract")
    if not isinstance(behavior, Mapping) or behavior != {
        "bengali_terminal_style": (
            "use_dari_for_declarative_bengali_prose_and_question_or_exclamation_"
            "marks_when_semantically_required"
        ),
        "banglish_style": "natural_latin_script_punctuation_without_over_normalization",
        "correct_surrounding_prose_without_mutating_protected_spans": True,
        "english_style": "standard_context_appropriate_english_punctuation",
        "preserve_user_meaning_and_tone": True,
    }:
        raise ValueError("punctuation model behavior contract changed")

    evaluation = value.get("post_sft_evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("punctuation post-SFT evaluation is missing")
    if evaluation.get("thresholds") != EXPECTED_THRESHOLDS:
        raise ValueError("punctuation quality thresholds changed")
    if evaluation.get("actual_model_inference_required") is not True:
        raise ValueError("punctuation gate must require model inference")
    if evaluation.get("human_naturalness_review_required") is not True:
        raise ValueError("punctuation gate must require human naturalness review")
    if evaluation.get("private_raw_outputs_required") is not True:
        raise ValueError("punctuation raw outputs must remain privately reviewable")
    if evaluation.get("raw_outputs_in_git_or_summary_report") is not False:
        raise ValueError("punctuation reports may not contain raw model outputs")
    if evaluation.get("response_sha256_required") is not True:
        raise ValueError("punctuation response hashes are required")
    return value


def build_punctuation_lanes(plan: Mapping[str, Any]) -> list[PunctuationLane]:
    """Build all 72 raw-text-free coverage cells deterministically."""

    matrix = plan["metadata_lane_matrix"]
    lanes: list[PunctuationLane] = []
    combinations = itertools.product(
        matrix["language_modes"],
        matrix["punctuation_operations"],
        matrix["content_contexts"],
    )
    for index, (language, operation, context) in enumerate(combinations):
        action = (
            "correct_punctuation_and_preserve_meaning"
            if context == "plain_prose"
            else "preserve_protected_span_and_correct_surrounding_prose"
        )
        lanes.append(
            PunctuationLane(
                case_id=f"punctuation-{index:03d}",
                content_context=context,
                expected_action=action,
                language_mode=language,
                punctuation_operation=operation,
            )
        )
    if len(lanes) != matrix["expected_lanes"]:
        raise AssertionError("punctuation lane matrix size changed")
    return lanes


def score_punctuation_outcomes(
    lanes: list[PunctuationLane],
    observed: Mapping[str, Mapping[str, bool]],
) -> dict[str, Any]:
    """Score private-review labels only; raw model text is never accepted."""

    case_ids = [lane.case_id for lane in lanes]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("punctuation case IDs must be unique")
    if set(observed) != set(case_ids):
        raise ValueError("punctuation outcomes must exactly match lane IDs")

    expected_fields = {
        "human_naturalness_approved",
        "meaning_preserved",
        "protected_span_preserved",
        "punctuation_correct",
    }
    by_language: dict[str, list[bool]] = defaultdict(list)
    punctuation_results: list[bool] = []
    naturalness_results: list[bool] = []
    meaning_results: list[bool] = []
    protected_results: list[bool] = []
    failed_lanes = 0
    for lane in lanes:
        labels = observed[lane.case_id]
        if set(labels) != expected_fields or any(
            not isinstance(value, bool) for value in labels.values()
        ):
            raise ValueError("punctuation outcome labels changed")
        punctuation_results.append(labels["punctuation_correct"])
        naturalness_results.append(labels["human_naturalness_approved"])
        meaning_results.append(labels["meaning_preserved"])
        by_language[lane.language_mode].append(labels["punctuation_correct"])
        if lane.content_context in PROTECTED_CONTEXTS:
            protected_results.append(labels["protected_span_preserved"])
        lane_passed = (
            labels["punctuation_correct"]
            and labels["human_naturalness_approved"]
            and labels["meaning_preserved"]
            and (
                labels["protected_span_preserved"] or lane.content_context not in PROTECTED_CONTEXTS
            )
        )
        failed_lanes += int(not lane_passed)

    def rate(values: list[bool]) -> float:
        return sum(values) / len(values)

    per_language = {
        language: {
            "cases": len(results),
            "punctuation_accuracy": rate(results),
        }
        for language, results in sorted(by_language.items())
    }
    rates = {
        "human_naturalness_approval_rate": rate(naturalness_results),
        "meaning_preservation_rate": rate(meaning_results),
        "overall_punctuation_accuracy": rate(punctuation_results),
        "protected_span_preservation_rate": rate(protected_results),
    }
    thresholds_met = (
        rates["human_naturalness_approval_rate"]
        >= EXPECTED_THRESHOLDS["human_naturalness_approval_rate"]
        and rates["meaning_preservation_rate"] >= EXPECTED_THRESHOLDS["meaning_preservation_rate"]
        and rates["overall_punctuation_accuracy"]
        >= EXPECTED_THRESHOLDS["overall_punctuation_accuracy"]
        and rates["protected_span_preservation_rate"]
        >= EXPECTED_THRESHOLDS["protected_span_preservation_rate"]
        and all(
            result["punctuation_accuracy"]
            >= EXPECTED_THRESHOLDS["per_language_punctuation_accuracy"]
            for result in per_language.values()
        )
    )
    report: dict[str, Any] = {
        "by_language": per_language,
        "failed_lanes": failed_lanes,
        "lanes": len(lanes),
        "model_response_text_accepted": False,
        "observed_values_are_review_labels_only": True,
        "quality_thresholds_met": thresholds_met,
        "rates": rates,
        "schema_version": PUNCTUATION_REPORT_SCHEMA_VERSION,
        "thresholds": EXPECTED_THRESHOLDS,
    }
    report["report_sha256"] = _fingerprint(report)
    return report


def build_punctuation_contract_audit(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Audit the oracle contract without making a model-capability claim."""

    lanes = build_punctuation_lanes(plan)
    labels = {
        lane.case_id: {
            "human_naturalness_approved": True,
            "meaning_preserved": True,
            "protected_span_preserved": True,
            "punctuation_correct": True,
        }
        for lane in lanes
    }
    report = score_punctuation_outcomes(lanes, labels)
    report.update(
        {
            "future_model_cases_required": (
                len(lanes) * plan["metadata_lane_matrix"]["future_prompts_per_lane"]
            ),
            "model_inference_executed": False,
            "model_quality_gate_passed": False,
            "oracle_outcomes_meet_thresholds": report.pop("quality_thresholds_met"),
            "paid_gpu_used": False,
            "plan_canonical_sha256": _fingerprint(plan),
            "report_kind": "policy_oracle_audit_not_model_quality_evidence",
        }
    )
    report.pop("report_sha256")
    report["report_sha256"] = _fingerprint(report)
    return report
