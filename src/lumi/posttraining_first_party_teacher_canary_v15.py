"""Task-anchored Banglish continuation of the local Qwen teacher canary."""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import posttraining_first_party_teacher_canary as base
from . import posttraining_first_party_teacher_canary_v4 as schema_v4
from . import posttraining_first_party_teacher_canary_v5 as corrected_v5
from . import posttraining_first_party_teacher_canary_v11 as terminal_v11
from . import posttraining_first_party_teacher_canary_v13 as safety_v13
from . import posttraining_first_party_teacher_canary_v14 as punctuation_v14
from .posttraining_first_party import file_sha256

CONTINUATION_CONTRACT = {
    "additional_inference_attempts": 84,
    "maximum_attempts_per_record": 3,
    "maximum_network_bytes": 0,
    "maximum_retained_new_records": 28,
    "maximum_retained_supervised_tokens_total": 24_000,
    "maximum_runtime_minutes": 120,
    "total_retained_records": 80,
}
CARRY_RECORDS = 52
NEW_RECORDS = 28
NEW_BANGLISH_ROWS = 7
NEW_NATURAL_CODE_SWITCHING_ROWS = 7
PROMPT_CONTRACT = "lumi.first-party-teacher-prompt.v0.15"
BANGLISH_STYLE_METHOD = "task_example_plus_exact_per_field_marker_v1"
BANGLISH_TEMPERATURE = 0.2

_V14_CONTINUATION_PAYLOAD = punctuation_v14.continuation_payload
_V14_CONTINUATION_PROMPT = punctuation_v14.continuation_prompt
_V14_GENERATE = punctuation_v14.generate_continuation_attempt
_V14_MAKE_ROW = punctuation_v14._make_row  # noqa: SLF001
_CORRECTED_VALIDATE = corrected_v5.validate_corrected_output

BANGLISH_TASK_CONTRACTS: dict[str, dict[str, Any]] = {
    "tool_api": {
        "assistant_marker": "hobe",
        "example": {
            "assistant_response": "Hishab korle uttor 56 hobe.",
            "user_prompt": "8 ar 7 gun korle uttor koto hobe?",
        },
        "user_marker": "hobe",
    },
    "writing_summarization": {
        "assistant_marker": "ami",
        "example": {
            "assistant_response": "Hae, ami mul point gulo short kore dichhi.",
            "user_prompt": "Eta choto kore summary kore dibe?",
        },
        "user_marker": "eta",
    },
}


def _task_contract(scenario: base.Scenario) -> Mapping[str, Any] | None:
    return BANGLISH_TASK_CONTRACTS.get(scenario.task)


def continuation_prompt(scenario: base.Scenario, language_mode: str) -> str:
    prompt = _V14_CONTINUATION_PROMPT(scenario, language_mode)
    contract = _task_contract(scenario)
    if language_mode != "banglish" or contract is None:
        return prompt
    instruction = (
        "Banglish style contract: write natural romanized Bangla with Latin letters only. "
        "user_prompt must contain the exact standalone word "
        f"{json.dumps(contract['user_marker'])}; "
        "assistant_response must contain the exact standalone word "
        f"{json.dumps(contract['assistant_marker'])}. Follow this JSON field style without "
        "copying its topic, numbers, answer, or literals: "
        f"{json.dumps(contract['example'], ensure_ascii=False, sort_keys=True)}. Preserve the "
        "current scenario, required literals, tool_call, and tool_result exactly."
    )
    corrected = safety_v13._insert_system_instruction(prompt, instruction)  # noqa: SLF001
    if instruction not in corrected:
        raise AssertionError("task-anchored Banglish correction was not installed")
    return corrected


def continuation_payload(
    record_id: str,
    scenario: base.Scenario,
    language_mode: str,
    *,
    attempt_number: int,
) -> dict[str, Any]:
    payload = _V14_CONTINUATION_PAYLOAD(
        record_id,
        scenario,
        language_mode,
        attempt_number=attempt_number,
    )
    payload["prompt"] = continuation_prompt(scenario, language_mode)
    if language_mode == "banglish" and _task_contract(scenario) is not None:
        payload["options"]["temperature"] = BANGLISH_TEMPERATURE
    return payload


def _has_exact_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text.casefold()) is not None


def validate_task_anchored_banglish(
    output: Mapping[str, Any],
    *,
    scenario: base.Scenario,
    language_mode: str,
) -> list[dict[str, str]]:
    messages = _CORRECTED_VALIDATE(
        output,
        scenario=scenario,
        language_mode=language_mode,
    )
    contract = _task_contract(scenario)
    if language_mode != "banglish" or contract is None:
        return messages
    for field in ("user_prompt", "assistant_response"):
        value = output.get(field)
        marker = str(contract["user_marker" if field == "user_prompt" else "assistant_marker"])
        if not isinstance(value, str) or not _has_exact_word(value, marker):
            raise ValueError(f"Banglish {field} lacks exact task-bound marker: {marker}")
    return messages


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
    row = _V14_MAKE_ROW(
        record_id,
        scenario,
        language_mode,
        messages,
        attempt_number=attempt_number,
        terminal_composition=terminal_composition,
        banglish_composition=banglish_composition,
    )
    row["source_id"] = "lumi_first_party_multilingual_safety_tool_v0_15"
    provenance = row["provenance"]
    provenance["prompt_contract"] = PROMPT_CONTRACT
    contract = _task_contract(scenario)
    targeted = language_mode == "banglish" and contract is not None
    provenance["Banglish_style_method"] = BANGLISH_STYLE_METHOD if targeted else None
    provenance["Banglish_user_marker"] = contract["user_marker"] if targeted else None
    provenance["Banglish_assistant_marker"] = contract["assistant_marker"] if targeted else None
    provenance["Banglish_temperature"] = BANGLISH_TEMPERATURE if targeted else None
    return row


@contextmanager
def _patched_v14_generation() -> Iterator[None]:
    original_payload = punctuation_v14.continuation_payload
    original_make_row = punctuation_v14._make_row  # noqa: SLF001
    original_validate = corrected_v5.validate_corrected_output
    punctuation_v14.continuation_payload = continuation_payload
    punctuation_v14._make_row = _make_row  # noqa: SLF001
    corrected_v5.validate_corrected_output = validate_task_anchored_banglish
    try:
        yield
    finally:
        punctuation_v14.continuation_payload = original_payload
        punctuation_v14._make_row = original_make_row  # noqa: SLF001
        corrected_v5.validate_corrected_output = original_validate


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
        with _patched_v14_generation():
            return _V14_GENERATE(
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
            rejection["schema_version"] = "lumi.sft-first-party-teacher-rejected-output.v0.15"
            base.write_json(rejection_path, rejection)
        raise


def _validate_continuation_plan(plan: dict[str, Any], *, artifact: Path) -> None:
    if plan.get("schema_completion") != CONTINUATION_CONTRACT:
        raise ValueError("task-anchored Banglish continuation contract changed")
    carry = plan.get("carry_forward")
    if not isinstance(carry, dict):
        raise ValueError("task-anchored Banglish carry-forward binding is missing")
    family_root = artifact.parent
    expected = {
        "candidate_rows_path": str(
            family_root / "attempt-014" / "private" / "candidate-rows.partial.jsonl"
        ),
        "candidate_rows_sha256": (
            "7796a7b8a69b21192d1491d03a10defe8a93cd5f59c2611c65ad46c8ca742e93"
        ),
        "failure_report_path": str(
            family_root / "attempt-014" / "evidence" / "failure-report.json"
        ),
        "failure_report_sha256": (
            "305585784a00f2537545c98965b3b521d5c2276b3dfad5659985262414f2547d"
        ),
        "failure_verification_path": str(
            family_root / "attempt-014" / "evidence" / "failure-verification.json"
        ),
        "failure_verification_sha256": (
            "990fa36dc35cd8520915d0f6e4dc8524ca5412db6732b05216eb7a11521723fa"
        ),
        "records": CARRY_RECORDS,
        "telemetry_path": str(
            family_root / "attempt-014" / "private" / "generation-telemetry.partial.jsonl"
        ),
        "telemetry_sha256": ("2d34faac2e78f2ef17c3f529d5dc1fc3e2e3dee621cc16502dd96ade6c2b78ee"),
    }
    if carry != expected:
        raise ValueError("task-anchored Banglish carry-forward contract changed")
    for field in ("candidate_rows", "failure_report", "failure_verification", "telemetry"):
        if file_sha256(carry[f"{field}_path"]) != carry[f"{field}_sha256"]:
            raise ValueError(f"task-anchored Banglish carry-forward changed: {field}")


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
        raise ValueError("task-anchored Banglish partial row/telemetry pair is incomplete")
    carry_rows = base._read_jsonl(Path(plan["carry_forward"]["candidate_rows_path"]))  # noqa: SLF001
    carry_telemetry = base._read_jsonl(  # noqa: SLF001
        Path(plan["carry_forward"]["telemetry_path"])
    )
    if len(carry_rows) != CARRY_RECORDS or len(carry_telemetry) != CARRY_RECORDS:
        raise ValueError("task-anchored Banglish carry-forward count changed")
    if not any(present):
        base.write_jsonl(partial_rows, carry_rows)
        base.write_jsonl(partial_telemetry, carry_telemetry)
    rows = base._read_jsonl(partial_rows)  # noqa: SLF001
    telemetry = base._read_jsonl(partial_telemetry)  # noqa: SLF001
    ledger = base._read_jsonl(ledger_path)  # noqa: SLF001
    if rows[:CARRY_RECORDS] != carry_rows or telemetry[:CARRY_RECORDS] != carry_telemetry:
        raise ValueError("task-anchored Banglish exact carry-forward changed")
    expected_ids = base.expected_record_ids()
    if [str(row.get("record_id")) for row in rows] != list(expected_ids[: len(rows)]):
        raise ValueError("task-anchored Banglish rows are not an exact expected prefix")
    if len(rows) != len(telemetry) or not CARRY_RECORDS <= len(rows) <= 80:
        raise ValueError("task-anchored Banglish row and telemetry counts differ")
    if len(ledger) > CONTINUATION_CONTRACT["additional_inference_attempts"]:
        raise ValueError("task-anchored Banglish attempt ledger exceeds authorization")
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


def run_task_anchored_banglish_continuation(
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
    rows = base._read_jsonl(Path(artifact_root) / "private" / "candidate-rows.jsonl")  # noqa: SLF001
    new_rows = rows[CARRY_RECORDS:]
    report["carry_forward"]["records"] = CARRY_RECORDS
    report["schema_completion"]["additional_records_retained"] = NEW_RECORDS
    report["schema_completion"]["Banglish_style_method"] = BANGLISH_STYLE_METHOD
    report["schema_completion"]["Banglish_temperature"] = BANGLISH_TEMPERATURE
    report["schema_completion"]["task_anchored_Banglish_rows"] = sum(
        row.get("provenance", {}).get("Banglish_style_method") == BANGLISH_STYLE_METHOD
        for row in new_rows
    )
    report["schema_completion"]["natural_code_switching_rows"] = sum(
        row.get("language_mode") == "natural_code_switching" for row in new_rows
    )
    base.write_json(Path(artifact_root) / "evidence" / "run-report.json", report)
    return report


def _verify_new_code_switch_terminal(
    row: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    *,
    scenario: base.Scenario,
) -> int:
    compatible = copy.deepcopy(row)
    compatible["provenance"]["prompt_contract"] = punctuation_v14.PROMPT_CONTRACT
    return punctuation_v14._verify_terminal_composition(  # noqa: SLF001
        compatible,
        telemetry,
        scenario=scenario,
    )


def verify_task_anchored_banglish_continuation(
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
    carry_telemetry = base._read_jsonl(  # noqa: SLF001
        Path(plan["carry_forward"]["telemetry_path"])
    )
    if (
        rows[:CARRY_RECORDS] != carry_rows
        or telemetry[:CARRY_RECORDS] != carry_telemetry
        or len(rows) != 80
        or len(telemetry) != 80
    ):
        raise ValueError("task-anchored Banglish carry-forward or final counts changed")
    if not NEW_RECORDS <= len(ledger) <= CONTINUATION_CONTRACT["additional_inference_attempts"]:
        raise ValueError("task-anchored Banglish attempt count is invalid")
    if sum(item.get("status") == "accepted" for item in ledger) != NEW_RECORDS:
        raise ValueError("task-anchored Banglish accepted-attempt count changed")
    if base.sha256_value(ledger) != report.get("schema_completion", {}).get(
        "attempt_ledger_internal_sha256"
    ):
        raise ValueError("task-anchored Banglish attempt ledger changed")
    if file_sha256(ledger_path) != report.get("attempt_ledger_file_sha256"):
        raise ValueError("task-anchored Banglish attempt ledger file changed")

    scenarios = {scenario.scenario_id: scenario for scenario in base.SCENARIOS}
    coverage: Counter[tuple[str, str]] = Counter()
    new_banglish = 0
    new_code_switch = 0
    banglish_terminal_appends = 0
    code_switch_terminal_replacements = 0
    for index, (row, item_telemetry) in enumerate(zip(rows, telemetry, strict=True)):
        scenario = scenarios[str(row["skill_id"])]
        language_mode = str(row["language_mode"])
        provenance = row.get("provenance", {})
        if index >= CARRY_RECORDS:
            if provenance.get("prompt_contract") != PROMPT_CONTRACT:
                raise ValueError("task-anchored Banglish prompt contract changed")
            contract = _task_contract(scenario)
            targeted = language_mode == "banglish" and contract is not None
            expected_method = BANGLISH_STYLE_METHOD if targeted else None
            if provenance.get("Banglish_style_method") != expected_method:
                raise ValueError("task-anchored Banglish provenance changed")
            if targeted:
                natural = {
                    message["role"]: str(message["content"])
                    for message in row.get("messages", [])
                    if message.get("role") in {"user", "assistant"}
                }
                if not _has_exact_word(natural["user"], str(contract["user_marker"])):
                    raise ValueError("task-anchored Banglish user marker changed")
                if not _has_exact_word(
                    natural["assistant"],
                    str(contract["assistant_marker"]),
                ):
                    raise ValueError("task-anchored Banglish assistant marker changed")
                if (
                    provenance.get("Banglish_user_marker") != contract["user_marker"]
                    or provenance.get("Banglish_assistant_marker") != contract["assistant_marker"]
                    or provenance.get("Banglish_temperature") != BANGLISH_TEMPERATURE
                ):
                    raise ValueError("task-anchored Banglish marker binding changed")
                banglish_terminal_appends += terminal_v11._verify_banglish_composition(  # noqa: SLF001
                    row,
                    item_telemetry,
                )
                new_banglish += 1
        if scenario.safety_label != "safe":
            assistant = next(
                str(message["content"])
                for message in row.get("messages", [])
                if message.get("role") == "assistant"
            )
            safety_v13.validate_language_bound_safety(
                assistant,
                scenario.safety_label,
                language_mode=language_mode,
            )
            coverage[(language_mode, scenario.safety_label)] += 1
        if index >= CARRY_RECORDS and language_mode == "natural_code_switching":
            code_switch_terminal_replacements += _verify_new_code_switch_terminal(
                row,
                item_telemetry,
                scenario=scenario,
            )
            new_code_switch += 1

    expected_coverage = Counter(
        {
            ("banglish", "medical_caution"): 1,
            ("banglish", "refuse"): 2,
            ("bengali_script", "medical_caution"): 1,
            ("bengali_script", "refuse"): 2,
            ("english", "medical_caution"): 1,
            ("english", "refuse"): 2,
            ("natural_code_switching", "medical_caution"): 1,
            ("natural_code_switching", "refuse"): 2,
        }
    )
    if coverage != expected_coverage:
        raise ValueError("language-bound safety coverage changed")
    if new_banglish != NEW_BANGLISH_ROWS:
        raise ValueError("task-anchored Banglish row count changed")
    if new_code_switch != NEW_NATURAL_CODE_SWITCHING_ROWS:
        raise ValueError("remaining code-switch row count changed")
    schema_report = report.get("schema_completion", {})
    if report.get("carry_forward", {}).get("records") != CARRY_RECORDS:
        raise ValueError("task-anchored Banglish report carry count changed")
    if schema_report.get("task_anchored_Banglish_rows") != NEW_BANGLISH_ROWS:
        raise ValueError("task-anchored Banglish report row count changed")
    if schema_report.get("natural_code_switching_rows") != NEW_NATURAL_CODE_SWITCHING_ROWS:
        raise ValueError("remaining code-switch report row count changed")

    verification["checks"]["carry_forward_exact_fifty_two_rows"] = True
    verification["checks"]["all_seven_remaining_Banglish_rows_task_anchored"] = True
    verification["checks"]["Banglish_exact_per_field_markers"] = True
    verification["checks"]["Banglish_temperature_point_two"] = True
    verification["checks"]["all_twelve_non_safe_rows_language_bound"] = True
    verification["checks"]["internal_punctuation_preserved_byte_exact"] = True
    verification["attempt_ledger_file_sha256"] = file_sha256(ledger_path)
    verification["Banglish_rows_verified"] = new_banglish
    verification["Banglish_terminal_appends"] = banglish_terminal_appends
    verification["natural_code_switching_rows_verified"] = new_code_switch
    verification["code_switch_terminal_replacements"] = code_switch_terminal_replacements
    base.write_json(artifact / "evidence" / "verification-report.json", verification)
    return verification
