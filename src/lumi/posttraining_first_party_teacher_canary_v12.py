"""Banglish safety-prompt-corrected continuation of the local Qwen teacher canary."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import posttraining_first_party_teacher_canary as base
from . import posttraining_first_party_teacher_canary_v4 as schema_v4
from . import posttraining_first_party_teacher_canary_v5 as corrected_v5
from . import posttraining_first_party_teacher_canary_v9 as terminal_v9
from . import posttraining_first_party_teacher_canary_v10 as english_v10
from . import posttraining_first_party_teacher_canary_v11 as terminal_v11
from .posttraining_first_party import file_sha256

CONTINUATION_CONTRACT = {
    "additional_inference_attempts": 132,
    "maximum_attempts_per_record": 3,
    "maximum_network_bytes": 0,
    "maximum_retained_new_records": 44,
    "maximum_retained_supervised_tokens_total": 24_000,
    "maximum_runtime_minutes": 120,
    "total_retained_records": 80,
}
PROMPT_CONTRACT = "lumi.first-party-teacher-prompt.v0.12"
BANGLISH_SAFETY_PROMPT_METHOD = "explicit_label_bound_Banglish_safety_boundary_v1"
REFUSAL_BOUNDARY = "parbo na"
MEDICAL_ESCALATION = "joruri"
REFUSAL_FORMAT_EXAMPLE = {
    "assistant_response": (
        "Ami private data ber korte help korte parbo na. Safe ebong legal option niye "
        "help korte pari."
    ),
    "user_prompt": "Tumi ki onno karo private data ber korte parba?",
}
MEDICAL_FORMAT_EXAMPLE = {
    "assistant_response": (
        "Eta joruri hote pare. Ekhoni local emergency help nao ebong kacher biswasto kauke bolo."
    ),
    "user_prompt": "Amar shash nite koshto hocche, ekhon ki korbo?",
}

_V11_MAKE_ROW = terminal_v11._make_row  # noqa: SLF001
_V11_PAYLOAD = terminal_v11.continuation_payload


def _label_instruction(scenario: base.Scenario) -> tuple[str, Mapping[str, str]]:
    if scenario.safety_label == "refuse":
        return (
            "This is a refusal-mode scenario. assistant_response must contain the exact "
            f"consecutive Banglish words {REFUSAL_BOUNDARY} as an explicit refusal boundary, "
            "briefly explain the boundary, and offer only a safe alternative. Do not comply "
            "with the harmful or privacy-invasive request.",
            REFUSAL_FORMAT_EXAMPLE,
        )
    if scenario.safety_label == "medical_caution":
        return (
            "This is an urgent medical-caution scenario. assistant_response must contain the "
            f"exact Banglish word {MEDICAL_ESCALATION}, urge immediate local emergency help, "
            "and tell the user to contact a trusted nearby person. Do not diagnose or minimize "
            "the symptoms.",
            MEDICAL_FORMAT_EXAMPLE,
        )
    raise ValueError("Banglish safety prompt requested for an unsupported safety label")


def continuation_prompt(scenario: base.Scenario, language_mode: str) -> str:
    prompt = terminal_v11.continuation_prompt(scenario, language_mode)
    if language_mode != "banglish" or scenario.safety_label == "safe":
        return prompt
    label_instruction, example = _label_instruction(scenario)
    instruction = (
        f"{corrected_v5.STRICT_BANGLISH_INSTRUCTION} {label_instruction} Both natural fields "
        "must still independently contain at least one curated romanized Bengali marker. "
        "Return no language label or meta-commentary. Follow this format style without copying "
        f"its topic: {json.dumps(example, ensure_ascii=False, sort_keys=True)}."
    )
    prompt = prompt.replace(corrected_v5.STRICT_BANGLISH_INSTRUCTION, instruction)
    if instruction not in prompt:
        raise AssertionError("Banglish safety correction was not installed")
    return prompt


def continuation_payload(
    record_id: str,
    scenario: base.Scenario,
    language_mode: str,
    *,
    attempt_number: int,
) -> dict[str, Any]:
    payload = _V11_PAYLOAD(
        record_id,
        scenario,
        language_mode,
        attempt_number=attempt_number,
    )
    payload["prompt"] = continuation_prompt(scenario, language_mode)
    return payload


def _make_row(
    record_id: str,
    scenario: base.Scenario,
    language_mode: str,
    messages: list[dict[str, str]],
    *,
    attempt_number: int,
    terminal_composition: Mapping[str, Any] | None,
    banglish_composition: Mapping[str, Any] | None,
) -> dict[str, Any]:
    row = _V11_MAKE_ROW(
        record_id,
        scenario,
        language_mode,
        messages,
        attempt_number=attempt_number,
        terminal_composition=terminal_composition,
        banglish_composition=banglish_composition,
    )
    targeted = language_mode == "banglish" and scenario.safety_label != "safe"
    row["source_id"] = "lumi_first_party_multilingual_safety_tool_v0_12"
    provenance = row["provenance"]
    provenance["prompt_contract"] = PROMPT_CONTRACT
    provenance["Banglish_safety_prompt_method"] = (
        BANGLISH_SAFETY_PROMPT_METHOD if targeted else None
    )
    provenance["Banglish_safety_prompt_label"] = scenario.safety_label if targeted else None
    return row


@contextmanager
def _patched_v11_generation() -> Iterator[None]:
    original_payload = terminal_v11.continuation_payload
    original_make_row = terminal_v11._make_row  # noqa: SLF001
    terminal_v11.continuation_payload = continuation_payload
    terminal_v11._make_row = _make_row  # noqa: SLF001
    try:
        yield
    finally:
        terminal_v11.continuation_payload = original_payload
        terminal_v11._make_row = original_make_row  # noqa: SLF001


def generate_continuation_attempt(
    record_id: str,
    scenario: base.Scenario,
    language_mode: str,
    *,
    attempt_number: int,
    endpoint: str,
    rejection_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        with _patched_v11_generation():
            return terminal_v11.generate_continuation_attempt(
                record_id,
                scenario,
                language_mode,
                attempt_number=attempt_number,
                endpoint=endpoint,
                rejection_path=rejection_path,
            )
    except Exception:
        if rejection_path.is_file():
            rejection = base._json_object(rejection_path)  # noqa: SLF001
            rejection["schema_version"] = "lumi.sft-first-party-teacher-rejected-output.v0.12"
            base.write_json(rejection_path, rejection)
        raise


def _validate_continuation_plan(plan: dict[str, Any], *, artifact: Path) -> None:
    if plan.get("schema_completion") != CONTINUATION_CONTRACT:
        raise ValueError("Banglish-safety continuation contract changed")
    carry = plan.get("carry_forward")
    if not isinstance(carry, dict):
        raise ValueError("Banglish-safety carry-forward binding is missing")
    family_root = artifact.parent
    expected = {
        "candidate_rows_path": str(
            family_root / "attempt-011" / "private" / "candidate-rows.partial.jsonl"
        ),
        "candidate_rows_sha256": (
            "3a29f8b0236f85bcb6b511ff4b174eb94378bbda1f4dc1476d1f2746b0c53b18"
        ),
        "failure_report_path": str(
            family_root / "attempt-011" / "evidence" / "failure-report.json"
        ),
        "failure_report_sha256": (
            "b8ff30722cd2cfda6c820b5f9701c78e39de3c37e4d414c6cfeecbb1284384ba"
        ),
        "failure_verification_path": str(
            family_root / "attempt-011" / "evidence" / "failure-verification.json"
        ),
        "failure_verification_sha256": (
            "8e783dd03fbfc6ca874229e430650aefa016f2e07655227838ddef946e77f637"
        ),
        "records": 36,
        "telemetry_path": str(
            family_root / "attempt-011" / "private" / "generation-telemetry.partial.jsonl"
        ),
        "telemetry_sha256": ("e78bba29bc20a7ae1c2f613c567604398ee74667174579074963c91c1f240023"),
    }
    if carry != expected:
        raise ValueError("Banglish-safety carry-forward contract changed")
    for field in ("candidate_rows", "failure_report", "failure_verification", "telemetry"):
        if file_sha256(carry[f"{field}_path"]) != carry[f"{field}_sha256"]:
            raise ValueError(f"Banglish-safety carry-forward changed: {field}")


def _initialize_continuation(
    plan: dict[str, Any],
    *,
    artifact: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    private = artifact / "private"
    partial_rows = private / "candidate-rows.partial.jsonl"
    partial_telemetry = private / "generation-telemetry.partial.jsonl"
    ledger_path = private / "additional-attempt-ledger.jsonl"
    present = (partial_rows.exists(), partial_telemetry.exists())
    if any(present) and not all(present):
        raise ValueError("Banglish-safety partial row/telemetry pair is incomplete")
    carry_rows = base._read_jsonl(Path(plan["carry_forward"]["candidate_rows_path"]))  # noqa: SLF001
    carry_telemetry = base._read_jsonl(  # noqa: SLF001
        Path(plan["carry_forward"]["telemetry_path"])
    )
    if len(carry_rows) != 36 or len(carry_telemetry) != 36:
        raise ValueError("Banglish-safety carry-forward count changed")
    if not any(present):
        base.write_jsonl(partial_rows, carry_rows)
        base.write_jsonl(partial_telemetry, carry_telemetry)
    rows = base._read_jsonl(partial_rows)  # noqa: SLF001
    telemetry = base._read_jsonl(partial_telemetry)  # noqa: SLF001
    ledger = base._read_jsonl(ledger_path)  # noqa: SLF001
    if rows[:36] != carry_rows or telemetry[:36] != carry_telemetry:
        raise ValueError("Banglish-safety exact carry-forward changed")
    expected_ids = base.expected_record_ids()
    if [str(row.get("record_id")) for row in rows] != list(expected_ids[: len(rows)]):
        raise ValueError("Banglish-safety rows are not an exact expected prefix")
    if len(rows) != len(telemetry) or not 36 <= len(rows) <= 80:
        raise ValueError("Banglish-safety row and telemetry counts differ")
    if len(ledger) > CONTINUATION_CONTRACT["additional_inference_attempts"]:
        raise ValueError("Banglish-safety attempt ledger exceeds authorization")
    return rows, telemetry, ledger


@contextmanager
def _patched_v4_runtime() -> Iterator[None]:
    original_contract = schema_v4.SCHEMA_COMPLETION_CONTRACT
    original_validate = schema_v4._validate_schema_completion_plan
    original_initialize = schema_v4._initialize_or_load
    original_generate = schema_v4.generate_schema_completion_attempt
    schema_v4.SCHEMA_COMPLETION_CONTRACT = CONTINUATION_CONTRACT
    schema_v4._validate_schema_completion_plan = _validate_continuation_plan
    schema_v4._initialize_or_load = _initialize_continuation
    schema_v4.generate_schema_completion_attempt = generate_continuation_attempt
    try:
        yield
    finally:
        schema_v4.SCHEMA_COMPLETION_CONTRACT = original_contract
        schema_v4._validate_schema_completion_plan = original_validate
        schema_v4._initialize_or_load = original_initialize
        schema_v4.generate_schema_completion_attempt = original_generate


def run_banglish_safety_continuation(
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
    telemetry = base._read_jsonl(  # noqa: SLF001
        Path(artifact_root) / "private" / "generation-telemetry.jsonl"
    )
    new_telemetry = telemetry[36:]
    terminal_compositions = [
        item["terminal_composition"]
        for item in new_telemetry
        if item.get("terminal_composition") is not None
    ]
    banglish_compositions = [
        item["banglish_terminal_composition"]
        for item in new_telemetry
        if item.get("banglish_terminal_composition") is not None
    ]
    terminal_replacements = sum(
        composition.get(f"{field}_removed_terminal") is not None
        for composition in terminal_compositions
        for field in ("user", "assistant")
    )
    banglish_appends = sum(
        composition.get(f"{field}_appended_terminal") is not None
        for composition in banglish_compositions
        for field in ("user_prompt", "assistant_response")
    )
    report["carry_forward"]["records"] = 36
    report["schema_completion"]["additional_records_retained"] = 44
    report["schema_completion"]["Banglish_safety_prompt_method"] = BANGLISH_SAFETY_PROMPT_METHOD
    report["schema_completion"]["Banglish_terminal_appends"] = banglish_appends
    report["schema_completion"]["English_prompt_method"] = english_v10.ENGLISH_PROMPT_METHOD
    report["schema_completion"]["terminal_punctuation_replacements"] = terminal_replacements
    report["generation"]["natural_code_switching_temperature"] = (
        terminal_v9.NATURAL_CODE_SWITCH_TEMPERATURE
    )
    base.write_json(Path(artifact_root) / "evidence" / "run-report.json", report)
    return report


def _verify_code_switch_terminal(
    row: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    *,
    scenario: base.Scenario,
) -> int:
    compatible = copy.deepcopy(row)
    compatible["provenance"]["prompt_contract"] = terminal_v11.PROMPT_CONTRACT
    return terminal_v11._verify_terminal_composition(  # noqa: SLF001
        compatible,
        telemetry,
        scenario=scenario,
    )


def verify_banglish_safety_continuation(
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
    _validate_continuation_plan(plan, artifact=artifact)
    report = base._json_object(artifact / "evidence" / "run-report.json")  # noqa: SLF001
    rows = base._read_jsonl(artifact / "private" / "candidate-rows.jsonl")  # noqa: SLF001
    telemetry = base._read_jsonl(  # noqa: SLF001
        artifact / "private" / "generation-telemetry.jsonl"
    )
    ledger_path = artifact / "private" / "additional-attempt-ledger.jsonl"
    ledger = base._read_jsonl(ledger_path)  # noqa: SLF001
    carry_rows = base._read_jsonl(  # noqa: SLF001
        Path(plan["carry_forward"]["candidate_rows_path"])
    )
    if rows[:36] != carry_rows or len(rows) != 80 or len(telemetry) != 80:
        raise ValueError("Banglish-safety carry-forward or final counts changed")
    if not 44 <= len(ledger) <= CONTINUATION_CONTRACT["additional_inference_attempts"]:
        raise ValueError("Banglish-safety attempt count is invalid")
    if sum(item.get("status") == "accepted" for item in ledger) != 44:
        raise ValueError("Banglish-safety accepted-attempt count changed")
    if base.sha256_value(ledger) != report.get("schema_completion", {}).get(
        "attempt_ledger_internal_sha256"
    ):
        raise ValueError("Banglish-safety attempt ledger changed")
    if file_sha256(ledger_path) != report.get("attempt_ledger_file_sha256"):
        raise ValueError("Banglish-safety attempt ledger file changed")

    scenarios = {scenario.scenario_id: scenario for scenario in base.SCENARIOS}
    banglish_appends = 0
    english_rows = 0
    medical_rows = 0
    refusal_rows = 0
    terminal_replacements = 0
    for row, item_telemetry in zip(rows[36:], telemetry[36:], strict=True):
        provenance = row.get("provenance", {})
        if provenance.get("prompt_contract") != PROMPT_CONTRACT:
            raise ValueError("Banglish-safety prompt contract changed")
        scenario = scenarios[str(row["skill_id"])]
        language_mode = row.get("language_mode")
        if language_mode == "banglish":
            banglish_appends += terminal_v11._verify_banglish_composition(  # noqa: SLF001
                row,
                item_telemetry,
            )
            targeted = scenario.safety_label != "safe"
            expected_method = BANGLISH_SAFETY_PROMPT_METHOD if targeted else None
            if provenance.get("Banglish_safety_prompt_method") != expected_method:
                raise ValueError("Banglish-safety prompt provenance changed")
            natural = {
                message["role"]: str(message["content"])
                for message in row.get("messages", [])
                if message.get("role") in {"user", "assistant"}
            }
            assistant = natural.get("assistant", "").casefold()
            if scenario.safety_label == "refuse":
                if re.search(r"\bparbo\s+na\b", assistant) is None:
                    raise ValueError("Banglish refusal row lacks the bound boundary")
                refusal_rows += 1
            elif scenario.safety_label == "medical_caution":
                if re.search(r"\bjoruri\b", assistant) is None:
                    raise ValueError("Banglish medical row lacks the bound escalation marker")
                medical_rows += 1
        elif language_mode == "natural_code_switching":
            terminal_replacements += _verify_code_switch_terminal(
                row,
                item_telemetry,
                scenario=scenario,
            )
        elif language_mode == "english":
            english_rows += 1
            if provenance.get("english_prompt_method") != english_v10.ENGLISH_PROMPT_METHOD:
                raise ValueError("English prompt provenance changed")
            natural = [
                str(message["content"])
                for message in row.get("messages", [])
                if message.get("role") in {"user", "assistant"}
            ]
            if any(base._BENGALI.search(text) is not None for text in natural):  # noqa: SLF001
                raise ValueError("English row contains Bengali script")

    if report.get("carry_forward", {}).get("records") != 36:
        raise ValueError("Banglish-safety report carry count changed")
    if banglish_appends != report.get("schema_completion", {}).get("Banglish_terminal_appends"):
        raise ValueError("Banglish-safety terminal append count changed")
    if terminal_replacements != report.get("schema_completion", {}).get(
        "terminal_punctuation_replacements"
    ):
        raise ValueError("code-switch terminal replacement count changed")
    if refusal_rows != 2 or medical_rows != 1:
        raise ValueError("Banglish safety-label coverage changed")

    verification["checks"]["carry_forward_exact_thirty_six_rows"] = True
    verification["checks"]["Banglish_refusal_boundary_exact"] = True
    verification["checks"]["Banglish_medical_escalation_exact"] = True
    verification["checks"]["Banglish_terminal_exact_append_only"] = True
    verification["checks"]["English_mode_zero_Bengali"] = True
    verification["checks"]["code_switch_terminal_normalization_exact_and_only_edit"] = True
    verification["attempt_ledger_file_sha256"] = file_sha256(ledger_path)
    verification["Banglish_terminal_appends"] = banglish_appends
    verification["English_rows_verified"] = english_rows
    verification["Banglish_refusal_rows_verified"] = refusal_rows
    verification["Banglish_medical_rows_verified"] = medical_rows
    verification["terminal_punctuation_replacements"] = terminal_replacements
    base.write_json(artifact / "evidence" / "verification-report.json", verification)
    return verification
