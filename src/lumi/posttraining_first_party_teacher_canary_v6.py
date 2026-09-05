"""Deterministic-anchor completion of the local Qwen teacher canary."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import posttraining_first_party_teacher_canary as base
from . import posttraining_first_party_teacher_canary_v2 as retry_v2
from . import posttraining_first_party_teacher_canary_v4 as schema_v4
from . import posttraining_first_party_teacher_canary_v5 as corrected_v5
from .posttraining_first_party import file_sha256

ANCHOR_COMPLETION_CONTRACT = dict(schema_v4.SCHEMA_COMPLETION_CONTRACT)
PROMPT_CONTRACT = "lumi.first-party-teacher-prompt.v0.6"
ANCHOR_SUBSTITUTION_METHOD = "standalone_bengali_anchor_first_exact_v1"

ANCHOR_SUBSTITUTIONS: dict[str, tuple[str, str]] = {
    "kilometer_to_meters": ("মিটার", "meter"),
    "days_in_week": ("সপ্তাহ", "week"),
    "triangle_sides": ("ত্রিভুজ", "triangle"),
    "water_freezing_celsius": ("সেলসিয়াস", "Celsius"),
    "friendly_greeting": ("শুভেচ্ছা", "greeting"),
    "punctuation_repair": ("যতিচিহ্ন", "punctuation"),
    "three_day_study_plan": ("পরিকল্পনা", "study plan"),
    "polite_rewrite": ("পুনর্লিখন", "rewrite"),
    "benign_study_help": ("কুইজ", "quiz"),
    "credential_privacy": ("পাসওয়ার্ড", "password"),
    "urgent_medical_caution": ("জরুরি", "emergency"),
    "phishing_request": ("ফিশিং", "phishing"),
    "calculator_addition": ("হিসাব", "calculate"),
    "calculator_multiplication": ("হিসাব", "calculate"),
    "calculator_division": ("হিসাব", "calculate"),
    "list_sort": ("সাজানো", "sort"),
    "meeting_summary": ("সারাংশ", "summary"),
    "release_summary": ("সারাংশ", "summary"),
    "plant_care_summary": ("সারাংশ", "summary"),
    "project_summary": ("সারাংশ", "summary"),
}


def anchor_seed(record_id: str, attempt_number: int) -> int:
    if not 1 <= attempt_number <= ANCHOR_COMPLETION_CONTRACT["maximum_attempts_per_record"]:
        raise ValueError("anchor-completion attempt number is outside authorization")
    material = f"{record_id}:anchor-completion-attempt:{attempt_number}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def _standalone_anchor_pattern(anchor: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![\u0980-\u09ff]){re.escape(anchor)}(?![\u0980-\u09ff])")


def anchor_output_schema(scenario: base.Scenario) -> dict[str, Any]:
    schema = copy.deepcopy(base.OUTPUT_SCHEMA)
    schema["properties"]["scenario_id"] = {
        "type": "string",
        "enum": [scenario.scenario_id],
    }
    schema["properties"]["safety_label"] = {
        "type": "string",
        "enum": [scenario.safety_label],
    }
    return schema


def anchor_prompt(scenario: base.Scenario, language_mode: str) -> str:
    if language_mode != "natural_code_switching":
        return corrected_v5.corrected_prompt(scenario, language_mode)
    prompt = retry_v2.build_retry_prompt(scenario, language_mode)
    anchor, _ = ANCHOR_SUBSTITUTIONS[scenario.scenario_id]
    instruction = (
        "Write user_prompt and assistant_response as complete, concise, natural Bengali-"
        "script strings. Each field must independently contain the exact standalone "
        f"Bengali anchor {json.dumps(anchor, ensure_ascii=False)} exactly once. The anchor "
        "must be a separate word, not part of a longer Bengali word. Do not use any Latin "
        "letters except ASCII literals explicitly required by the scenario. Do not "
        "translate, inflect, join, omit, or repeat the anchor. Preserve every required "
        "literal exactly in both fields. The runtime will perform one deterministic anchor "
        "substitution per field; write no placeholders, prefixes, suffix objects, or "
        "meta-commentary. Use correct Bengali punctuation."
    )
    prompt = prompt.replace(base.LANGUAGE_INSTRUCTIONS[language_mode], instruction)
    if instruction not in prompt:
        raise AssertionError("deterministic-anchor instruction was not installed")
    return prompt


def anchor_payload(
    record_id: str,
    scenario: base.Scenario,
    language_mode: str,
    *,
    attempt_number: int,
) -> dict[str, Any]:
    payload = base.ollama_payload(record_id, scenario, language_mode)
    payload["prompt"] = anchor_prompt(scenario, language_mode)
    payload["options"]["seed"] = anchor_seed(record_id, attempt_number)
    if language_mode == "natural_code_switching":
        payload["format"] = anchor_output_schema(scenario)
    return payload


def _remove_allowed_ascii_literals(text: str, scenario: base.Scenario) -> str:
    remaining = text
    for literal in sorted(scenario.required_literals, key=len, reverse=True):
        remaining = remaining.replace(literal, "")
    return remaining


def substitute_anchor_output(
    output: Mapping[str, Any],
    *,
    scenario: base.Scenario,
) -> tuple[dict[str, Any], list[dict[str, str]], dict[str, Any]]:
    expected_fields = {
        "assistant_response",
        "safety_label",
        "scenario_id",
        "tool_call",
        "tool_result",
        "user_prompt",
    }
    if set(output) != expected_fields:
        raise ValueError("anchor teacher JSON envelope fields changed")
    anchor, english = ANCHOR_SUBSTITUTIONS[scenario.scenario_id]
    pattern = _standalone_anchor_pattern(anchor)
    originals: dict[str, str] = {}
    replacements: dict[str, str] = {}
    for field in ("user_prompt", "assistant_response"):
        text = output.get(field)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"anchor teacher {field} is empty")
        if base._BENGALI.search(text) is None:  # noqa: SLF001
            raise ValueError(f"anchor teacher {field} lacks Bengali script")
        if base._LATIN.search(_remove_allowed_ascii_literals(text, scenario)) is not None:  # noqa: SLF001
            raise ValueError(f"anchor teacher {field} contains unbound Latin text")
        matches = list(pattern.finditer(text))
        if len(matches) != 1:
            raise ValueError(f"anchor teacher {field} must contain one standalone anchor")
        replaced, count = pattern.subn(english, text, count=1)
        if count != 1:
            raise ValueError(f"anchor substitution count changed for {field}")
        originals[field] = text.strip()
        replacements[field] = replaced.strip()

    flattened = dict(output)
    flattened.update(replacements)
    messages = base._validate_output(  # noqa: SLF001
        flattened,
        scenario=scenario,
        language_mode="natural_code_switching",
    )
    substitution = {
        "anchor_bengali": anchor,
        "assistant_after": replacements["assistant_response"],
        "assistant_before": originals["assistant_response"],
        "english_schema_constant": english,
        "method": ANCHOR_SUBSTITUTION_METHOD,
        "substitutions_per_field": 1,
        "user_after": replacements["user_prompt"],
        "user_before": originals["user_prompt"],
    }
    return flattened, messages, substitution


def _make_row(
    record_id: str,
    scenario: base.Scenario,
    language_mode: str,
    messages: list[dict[str, str]],
    *,
    attempt_number: int,
    substitution: Mapping[str, Any] | None,
) -> dict[str, Any]:
    composed = language_mode == "natural_code_switching"
    anchor, english = ANCHOR_SUBSTITUTIONS[scenario.scenario_id]
    return {
        "schema_version": base.ROW_SCHEMA_VERSION,
        "record_id": record_id,
        "source_id": "lumi_first_party_multilingual_safety_tool_v0_6",
        "task": scenario.task,
        "skill_id": scenario.scenario_id,
        "language_mode": language_mode,
        "split": "validation" if scenario.variant == 3 else "train",
        "messages": messages,
        "provenance": {
            "ai_assisted": True,
            "anchor_bengali": anchor if composed else None,
            "banglish_marker_contract": corrected_v5.BANGLISH_MARKER_CONTRACT
            if language_mode == "banglish"
            else None,
            "deterministic_anchor_substitution": composed,
            "english_schema_constant": english if composed else None,
            "freeform_manual_rewriting": False,
            "generation_attempt": attempt_number,
            "human_authored": False,
            "inference_location": "mac_local_ollama_metal",
            "model_file_sha256": base.MODEL_SHA256,
            "model_id": base.MODEL_ID,
            "model_revision": base.MODEL_REVISION,
            "ollama_model": base.OLLAMA_MODEL,
            "prompt_contract": PROMPT_CONTRACT,
            "quantization": "Q4_K_M",
            "seed": anchor_seed(record_id, attempt_number),
            "self_hosted": True,
            "substitution_method": ANCHOR_SUBSTITUTION_METHOD if composed else None,
            "substitution_source_internal_sha256": base.sha256_value(substitution)
            if substitution is not None
            else None,
        },
        "rights_status": "private_research_candidate_pending_owner_review",
    }


def generate_anchor_attempt(
    record_id: str,
    scenario: base.Scenario,
    language_mode: str,
    *,
    attempt_number: int,
    endpoint: str,
    rejection_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = anchor_payload(
        record_id,
        scenario,
        language_mode,
        attempt_number=attempt_number,
    )
    response: dict[str, Any] | None = None
    substitution: dict[str, Any] | None = None
    started = time.monotonic()
    try:
        response = base._post_local_json(endpoint, payload)  # noqa: SLF001
        if response.get("done") is not True or response.get("done_reason") != "stop":
            raise ValueError("local teacher did not finish with stop")
        if response.get("thinking") not in (None, ""):
            raise ValueError("local teacher returned hidden thinking")
        raw = response.get("response")
        if not isinstance(raw, str) or base._HIDDEN_REASONING.search(raw):  # noqa: SLF001
            raise ValueError("local teacher raw response contains hidden reasoning")
        try:
            output = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError("local teacher response is not JSON") from error
        if not isinstance(output, dict):
            raise ValueError("local teacher response root is not an object")
        if language_mode == "natural_code_switching":
            _, messages, substitution = substitute_anchor_output(output, scenario=scenario)
        else:
            messages = corrected_v5.validate_corrected_output(
                output,
                scenario=scenario,
                language_mode=language_mode,
            )
    except Exception as error:
        base.write_json(
            rejection_path,
            {
                "attempt_number": attempt_number,
                "error_message": str(error),
                "error_type": type(error).__name__,
                "record_id": record_id,
                "request": payload,
                "response": response,
                "schema_version": "lumi.sft-first-party-teacher-rejected-output.v0.5",
                "status": "rejected_private_diagnostic_only",
            },
        )
        raise

    elapsed = time.monotonic() - started
    row = _make_row(
        record_id,
        scenario,
        language_mode,
        messages,
        attempt_number=attempt_number,
        substitution=substitution,
    )
    telemetry = {
        "anchor_substitution": substitution,
        "attempt_number": attempt_number,
        "banglish_marker_contract": corrected_v5.BANGLISH_MARKER_CONTRACT
        if language_mode == "banglish"
        else None,
        "elapsed_seconds": round(elapsed, 6),
        "eval_count": response.get("eval_count"),
        "eval_duration_ns": response.get("eval_duration"),
        "load_duration_ns": response.get("load_duration"),
        "prompt_eval_count": response.get("prompt_eval_count"),
        "prompt_eval_duration_ns": response.get("prompt_eval_duration"),
        "record_id": record_id,
        "request_sha256": base.sha256_value(payload),
        "response_envelope_sha256": base.sha256_value(response),
    }
    return row, telemetry


def _validate_anchor_plan(plan: dict[str, Any], *, artifact: Path) -> None:
    if plan.get("schema_completion") != ANCHOR_COMPLETION_CONTRACT:
        raise ValueError("anchor-completion contract changed")
    carry = plan.get("carry_forward")
    if not isinstance(carry, dict):
        raise ValueError("anchor-completion carry-forward binding is missing")
    family_root = artifact.parent
    expected = {
        "candidate_rows_path": str(
            family_root / "attempt-003" / "private" / "candidate-rows.partial.jsonl"
        ),
        "candidate_rows_sha256": "108b126c1a32fe281637b9e7ccaa2b12189a688d6d8056d612737ec44cc21095",
        "failure_report_path": str(
            family_root / "attempt-005" / "evidence" / "failure-report.json"
        ),
        "failure_report_sha256": "42850104cbf1f315838f8813959e87defeef8ac9013bcd04e9832061bcbf8bd0",
        "records": 3,
        "telemetry_path": str(
            family_root / "attempt-003" / "private" / "generation-telemetry.partial.jsonl"
        ),
        "telemetry_sha256": "cf3deb7325d64b756082830d9135a8633ee0c79d64c0436189fff57c0b35f757",
    }
    if carry != expected:
        raise ValueError("anchor-completion carry-forward contract changed")
    for field in ("candidate_rows", "failure_report", "telemetry"):
        if file_sha256(carry[f"{field}_path"]) != carry[f"{field}_sha256"]:
            raise ValueError(f"anchor-completion carry-forward changed: {field}")


@contextmanager
def _patched_v4_runtime() -> Iterator[None]:
    original_validate = schema_v4._validate_schema_completion_plan
    original_generate = schema_v4.generate_schema_completion_attempt
    schema_v4._validate_schema_completion_plan = _validate_anchor_plan
    schema_v4.generate_schema_completion_attempt = generate_anchor_attempt
    try:
        yield
    finally:
        schema_v4._validate_schema_completion_plan = original_validate
        schema_v4.generate_schema_completion_attempt = original_generate


def run_anchor_completion(
    *,
    plan_path: str | Path,
    repository_root: str | Path,
    artifact_root: str | Path,
    endpoint: str = "http://127.0.0.1:11434/api/generate",
) -> dict[str, Any]:
    with _patched_v4_runtime():
        report = schema_v4.run_schema_completion(
            plan_path=plan_path,
            repository_root=repository_root,
            artifact_root=artifact_root,
            endpoint=endpoint,
        )
    report["schema_completion"]["composition_method"] = ANCHOR_SUBSTITUTION_METHOD
    report["schema_completion"]["deterministic_anchor_substitutions"] = 20
    report["schema_completion"]["pre_substitution_text_retained_privately"] = True
    base.write_json(Path(artifact_root) / "evidence" / "run-report.json", report)
    return report


def _verify_banglish_row(row: Mapping[str, Any]) -> None:
    natural = [
        str(message["content"])
        for message in row.get("messages", [])
        if message.get("role") in {"user", "assistant"}
    ]
    if len(natural) != 2:
        raise ValueError("anchor-completion Banglish natural-message count changed")
    for text in natural:
        markers = set(re.findall(r"[a-z]+", text.casefold())) & set(
            corrected_v5.CORRECTED_BANGLISH_MARKERS
        )
        if not markers:
            raise ValueError("anchor-completion Banglish row lacks a curated marker")


def _verify_anchor_substitution(
    row: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    *,
    scenario: base.Scenario,
) -> None:
    anchor, english = ANCHOR_SUBSTITUTIONS[scenario.scenario_id]
    provenance = row.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("anchor-completion provenance is missing")
    expected = {
        "anchor_bengali": anchor,
        "deterministic_anchor_substitution": True,
        "english_schema_constant": english,
        "freeform_manual_rewriting": False,
        "human_authored": False,
        "prompt_contract": PROMPT_CONTRACT,
        "substitution_method": ANCHOR_SUBSTITUTION_METHOD,
    }
    for field, value in expected.items():
        if provenance.get(field) != value:
            raise ValueError(f"anchor-completion provenance changed: {field}")
    substitution = telemetry.get("anchor_substitution")
    if not isinstance(substitution, Mapping):
        raise ValueError("anchor-completion private substitution evidence is missing")
    if provenance.get("substitution_source_internal_sha256") != base.sha256_value(substitution):
        raise ValueError("anchor-completion substitution evidence binding changed")
    if (
        substitution.get("anchor_bengali") != anchor
        or substitution.get("english_schema_constant") != english
    ):
        raise ValueError("anchor-completion substitution constants changed")
    pattern = _standalone_anchor_pattern(anchor)
    natural = {
        message["role"]: str(message["content"])
        for message in row.get("messages", [])
        if message.get("role") in {"user", "assistant"}
    }
    for field, role in (("user", "user"), ("assistant", "assistant")):
        before = str(substitution[f"{field}_before"])
        after = str(substitution[f"{field}_after"])
        expected_after, count = pattern.subn(english, before, count=1)
        if count != 1 or after != expected_after or natural.get(role) != after:
            raise ValueError("anchor-completion changed text beyond exact substitution")


def verify_anchor_completion(
    *,
    plan_path: str | Path,
    repository_root: str | Path,
    artifact_root: str | Path,
) -> dict[str, Any]:
    artifact = Path(artifact_root).resolve()
    verification = base.verify_run(
        plan_path=plan_path,
        repository_root=repository_root,
        artifact_root=artifact,
    )
    plan = base.load_canary_plan(plan_path, repository_root=repository_root)
    _validate_anchor_plan(plan, artifact=artifact)
    report = base._json_object(artifact / "evidence" / "run-report.json")
    rows = base._read_jsonl(artifact / "private" / "candidate-rows.jsonl")
    telemetry = base._read_jsonl(artifact / "private" / "generation-telemetry.jsonl")
    ledger_path = artifact / "private" / "additional-attempt-ledger.jsonl"
    ledger = base._read_jsonl(ledger_path)
    carry_rows = base._read_jsonl(Path(plan["carry_forward"]["candidate_rows_path"]))
    if rows[:3] != carry_rows or len(rows) != 80 or len(telemetry) != 80:
        raise ValueError("anchor-completion carry-forward or final counts changed")
    if not 77 <= len(ledger) <= ANCHOR_COMPLETION_CONTRACT["additional_inference_attempts"]:
        raise ValueError("anchor-completion attempt count is invalid")
    if sum(item.get("status") == "accepted" for item in ledger) != 77:
        raise ValueError("anchor-completion accepted-attempt count changed")
    if base.sha256_value(ledger) != report.get("schema_completion", {}).get(
        "attempt_ledger_internal_sha256"
    ):
        raise ValueError("anchor-completion attempt ledger changed")
    if file_sha256(ledger_path) != report.get("attempt_ledger_file_sha256"):
        raise ValueError("anchor-completion attempt ledger file changed")

    scenarios = {scenario.scenario_id: scenario for scenario in base.SCENARIOS}
    anchor_rows = 0
    for row, item_telemetry in zip(rows[3:], telemetry[3:], strict=True):
        provenance = row.get("provenance", {})
        if provenance.get("prompt_contract") != PROMPT_CONTRACT:
            raise ValueError("anchor-completion prompt contract changed")
        if row.get("language_mode") == "banglish":
            _verify_banglish_row(row)
        elif row.get("language_mode") == "natural_code_switching":
            scenario = scenarios[str(row["skill_id"])]
            _verify_anchor_substitution(row, item_telemetry, scenario=scenario)
            anchor_rows += 1
    if anchor_rows != 20:
        raise ValueError("anchor-completion code-switch row count changed")
    verification["checks"]["anchor_substitution_exact_and_only_edit"] = True
    verification["checks"]["carry_forward_original_three_rows"] = True
    verification["checks"]["corrected_banglish_marker_gate"] = True
    verification["checks"]["private_pre_substitution_evidence_bound"] = True
    verification["attempt_ledger_file_sha256"] = file_sha256(ledger_path)
    base.write_json(artifact / "evidence" / "verification-report.json", verification)
    return verification
