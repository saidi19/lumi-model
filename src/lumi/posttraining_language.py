"""Fail-closed response-language routing contract for future Lumi SFT evaluation."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LANGUAGE_ROUTING_SCHEMA_VERSION = "lumi.language-routing-gate.v0.1"
LANGUAGE_ROUTING_REPORT_SCHEMA_VERSION = "lumi.language-routing-contract-report.v0.1"
RESPONSE_MODES = frozenset(
    {"english", "bengali_script", "banglish", "code_switching", "other_language"}
)
INPUT_MODES = RESPONSE_MODES | {"undetermined"}
EXPECTED_AUTHORIZATION = {
    "model_inference_authorized": False,
    "paid_gpu_authorized": False,
    "production_release_authorized": False,
    "purpose": "metadata_only_response_language_contract_and_future_post_sft_gate",
    "raw_prompt_rows_in_git": False,
}
EXPECTED_SELECTION_PRIORITY = [
    "explicit_user_response_language",
    "latest_unambiguous_user_message",
    "established_conversation_language",
    "english_product_default",
]


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
class LanguageRoutingCase:
    case_id: str
    established_conversation_mode: str | None
    expected_response_mode: str
    explicit_response_mode: str | None
    latest_user_mode: str


def load_language_routing_plan(path: str | Path) -> dict[str, Any]:
    """Load the metadata-only language contract and reject scope or policy drift."""

    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("language-routing plan must be an object")
    if value.get("schema_version") != LANGUAGE_ROUTING_SCHEMA_VERSION:
        raise ValueError("unsupported language-routing schema")
    if value.get("status") != "gate_defined_not_executed":
        raise ValueError("language-routing gate status changed")
    if value.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("language-routing authorization boundary changed")

    contract = value.get("model_behavior_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("language-routing model behavior contract is missing")
    if contract.get("default_mode") != "english":
        raise ValueError("language-routing default mode changed")
    if contract.get("selection_priority") != EXPECTED_SELECTION_PRIORITY:
        raise ValueError("language-routing selection priority changed")
    if set(contract.get("supported_routing_modes", ())) != INPUT_MODES:
        raise ValueError("language-routing mode set changed")
    if set(contract.get("release_core_modes", ())) != {
        "english",
        "bengali_script",
        "banglish",
        "code_switching",
    }:
        raise ValueError("language-routing release core changed")
    if contract.get("other_language_release_status") != (
        "best_effort_until_language_specific_data_and_frozen_evaluation_pass"
    ):
        raise ValueError("other-language support claim changed")
    detection = contract.get("detection")
    if not isinstance(detection, Mapping) or detection != {
        "primary": "model_semantic_language_inference",
        "server_side_detector_role": (
            "evaluation_and_observability_not_forced_translation"
        ),
        "short_or_ambiguous_input_uses_conversation_context": True,
        "technical_tokens_do_not_force_language_switch": True,
    }:
        raise ValueError("language-routing detection policy changed")

    matrix = value.get("metadata_case_matrix")
    if not isinstance(matrix, Mapping):
        raise ValueError("language-routing metadata matrix is missing")
    explicit_modes = matrix.get("explicit_response_modes")
    latest_modes = matrix.get("latest_user_modes")
    established_modes = matrix.get("established_conversation_modes")
    expected_optional_modes = {None, *RESPONSE_MODES}
    if not isinstance(explicit_modes, list) or set(explicit_modes) != expected_optional_modes:
        raise ValueError("language-routing explicit modes changed")
    if not isinstance(latest_modes, list) or set(latest_modes) != INPUT_MODES:
        raise ValueError("language-routing latest-user modes changed")
    if not isinstance(established_modes, list) or set(established_modes) != (
        expected_optional_modes
    ):
        raise ValueError("language-routing conversation modes changed")
    expected_cases = len(explicit_modes) * len(latest_modes) * len(established_modes)
    if matrix.get("expected_cases") != expected_cases or expected_cases != 216:
        raise ValueError("language-routing metadata case bound changed")

    evaluation = value.get("post_sft_evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("language-routing evaluation contract is missing")
    if evaluation.get("deterministic_contract_required_pass_rate") != 1.0:
        raise ValueError("language-routing contract matrix must pass exactly")
    if evaluation.get("explicit_user_override_required_pass_rate") != 1.0:
        raise ValueError("explicit language override must pass exactly")
    if evaluation.get("raw_model_responses_retained") is not False:
        raise ValueError("language-routing reports may not retain model responses")
    if evaluation.get("response_sha256_required") is not True:
        raise ValueError("language-routing response hashes must be retained")
    return value


def select_response_language_mode(
    *,
    explicit_response_mode: str | None,
    latest_user_mode: str,
    established_conversation_mode: str | None,
) -> str:
    """Apply explicit request, latest message, conversation, then default precedence."""

    if explicit_response_mode is not None and explicit_response_mode not in RESPONSE_MODES:
        raise ValueError("explicit response mode is invalid")
    if latest_user_mode not in INPUT_MODES:
        raise ValueError("latest user mode is invalid")
    if (
        established_conversation_mode is not None
        and established_conversation_mode not in RESPONSE_MODES
    ):
        raise ValueError("established conversation mode is invalid")
    if explicit_response_mode is not None:
        return explicit_response_mode
    if latest_user_mode != "undetermined":
        return latest_user_mode
    if established_conversation_mode is not None:
        return established_conversation_mode
    return "english"


def build_language_routing_cases(plan: Mapping[str, Any]) -> list[LanguageRoutingCase]:
    """Build the exact metadata-only precedence matrix; no prompts are generated."""

    matrix = plan["metadata_case_matrix"]
    cases: list[LanguageRoutingCase] = []
    combinations = itertools.product(
        matrix["explicit_response_modes"],
        matrix["latest_user_modes"],
        matrix["established_conversation_modes"],
    )
    for index, (explicit, latest, established) in enumerate(combinations):
        cases.append(
            LanguageRoutingCase(
                case_id=f"language-routing-{index:03d}",
                established_conversation_mode=established,
                expected_response_mode=select_response_language_mode(
                    explicit_response_mode=explicit,
                    latest_user_mode=latest,
                    established_conversation_mode=established,
                ),
                explicit_response_mode=explicit,
                latest_user_mode=latest,
            )
        )
    if len(cases) != matrix["expected_cases"]:
        raise AssertionError("language-routing matrix size changed after validation")
    return cases


def score_language_routing_modes(
    cases: list[LanguageRoutingCase],
    observed_modes: Mapping[str, str],
) -> dict[str, Any]:
    """Score language labels only; model response text is never accepted or retained."""

    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("language-routing case IDs must be unique")
    if set(observed_modes) != set(case_ids):
        raise ValueError("observed language modes must exactly match routing case IDs")

    by_expected_mode: dict[str, list[bool]] = defaultdict(list)
    explicit_outcomes: list[bool] = []
    result_counts: Counter[str] = Counter()
    for case in cases:
        observed = observed_modes[case.case_id]
        if observed not in RESPONSE_MODES:
            raise ValueError("observed response mode is invalid")
        passed = observed == case.expected_response_mode
        by_expected_mode[case.expected_response_mode].append(passed)
        result_counts["passed" if passed else "failed"] += 1
        if case.explicit_response_mode is not None:
            explicit_outcomes.append(passed)

    cases_count = len(cases)
    report: dict[str, Any] = {
        "by_expected_mode": {
            mode: {
                "cases": len(outcomes),
                "pass_rate": sum(outcomes) / len(outcomes),
                "passed": sum(outcomes),
            }
            for mode, outcomes in sorted(by_expected_mode.items())
        },
        "cases": cases_count,
        "contract_pass_rate": result_counts["passed"] / cases_count,
        "explicit_override_cases": len(explicit_outcomes),
        "explicit_override_pass_rate": sum(explicit_outcomes) / len(explicit_outcomes),
        "model_response_text_accepted": False,
        "observed_values_are_language_labels_only": True,
        "result_counts": dict(sorted(result_counts.items())),
        "schema_version": LANGUAGE_ROUTING_REPORT_SCHEMA_VERSION,
    }
    report["report_sha256"] = _fingerprint(report)
    return report


def build_language_routing_contract_audit(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Prove the frozen precedence matrix without claiming model inference occurred."""

    cases = build_language_routing_cases(plan)
    report = score_language_routing_modes(
        cases,
        {case.case_id: case.expected_response_mode for case in cases},
    )
    report.update(
        {
            "model_inference_executed": False,
            "paid_gpu_used": False,
            "plan_canonical_sha256": _fingerprint(plan),
            "report_kind": "policy_oracle_contract_audit_not_model_quality_evidence",
        }
    )
    report.pop("report_sha256")
    report["report_sha256"] = _fingerprint(report)
    return report
