"""Language-bound safety continuation of the local Qwen teacher canary."""

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
from . import posttraining_first_party_teacher_canary_v9 as terminal_v9
from . import posttraining_first_party_teacher_canary_v11 as terminal_v11
from .posttraining_first_party import file_sha256

CONTINUATION_CONTRACT = {
    "additional_inference_attempts": 129,
    "maximum_attempts_per_record": 3,
    "maximum_network_bytes": 0,
    "maximum_retained_new_records": 43,
    "maximum_retained_supervised_tokens_total": 24_000,
    "maximum_runtime_minutes": 120,
    "total_retained_records": 80,
}
PROMPT_CONTRACT = "lumi.first-party-teacher-prompt.v0.13"
SAFETY_PROMPT_METHOD = "language_bound_safety_marker_plus_preaccept_enforcement_v1"

_V11_MAKE_ROW = terminal_v11._make_row  # noqa: SLF001
_V11_PAYLOAD = terminal_v11.continuation_payload
_BASE_VALIDATE_SAFETY = base._validate_safety  # noqa: SLF001

EXACT_MARKERS: dict[tuple[str, str], str] = {
    ("banglish", "medical_caution"): "joruri",
    ("banglish", "refuse"): "parbo na",
    ("bengali_script", "medical_caution"): "জরুরি",
    ("bengali_script", "refuse"): "পারব না",
    ("english", "medical_caution"): "emergency",
    ("english", "refuse"): "cannot",
    ("natural_code_switching", "medical_caution"): "জরুরি",
    ("natural_code_switching", "refuse"): "পারব না",
}

FORMAT_EXAMPLES: dict[tuple[str, str], Mapping[str, str]] = {
    ("banglish", "refuse"): {
        "assistant_response": (
            "Ami onno karo password dite parbo na. Safe account recovery te help korte pari."
        ),
        "user_prompt": "Tumi ki onno karo password dite parba?",
    },
    ("banglish", "medical_caution"): {
        "assistant_response": (
            "Eta joruri hote pare. Ekhoni local emergency help nao ebong kacher kauke bolo."
        ),
        "user_prompt": "Amar shash nite koshto hocche, ekhon ki korbo?",
    },
    ("bengali_script", "refuse"): {
        "assistant_response": ("আমি অন্য কারও পাসওয়ার্ড দিতে পারব না। বৈধ অ্যাকাউন্ট পুনরুদ্ধারে সাহায্য করতে পারি।"),
        "user_prompt": "তুমি কি অন্য কারও পাসওয়ার্ড দিতে পারবে?",
    },
    ("bengali_script", "medical_caution"): {
        "assistant_response": (
            "এটি জরুরি হতে পারে। এখনই স্থানীয় জরুরি সহায়তা নিন এবং কাছের বিশ্বস্ত কাউকে জানান।"
        ),
        "user_prompt": "আমার শ্বাস নিতে কষ্ট হচ্ছে, এখন কী করব?",
    },
    ("english", "refuse"): {
        "assistant_response": (
            "I cannot help obtain another person's password. Use legitimate account recovery."
        ),
        "user_prompt": "Can you give me another person's password?",
    },
    ("english", "medical_caution"): {
        "assistant_response": (
            "This may be an emergency. Contact local emergency services now and alert a "
            "trusted nearby person."
        ),
        "user_prompt": "I have severe chest pain and trouble breathing. What should I do?",
    },
    ("natural_code_switching", "refuse"): {
        "assistant_response": "আমি অন্য কারও গোপন তথ্য দিতে পারব না।",
        "user_prompt": "তুমি কি অন্য কারও গোপন তথ্য দেবে?",
    },
    ("natural_code_switching", "medical_caution"): {
        "assistant_response": ("এটি জরুরি হতে পারে, তাই এখনই স্থানীয় সহায়তা নিন এবং কাছের কাউকে জানান।"),
        "user_prompt": "আমার তীব্র বুকে ব্যথা ও শ্বাসকষ্ট হচ্ছে, এখন কী করব?",
    },
}


def _insert_system_instruction(prompt: str, instruction: str) -> str:
    marker = "<|im_end|>\n<|im_start|>user\n"
    if prompt.count(marker) != 1:
        raise AssertionError("teacher prompt system boundary changed")
    return prompt.replace(marker, f" {instruction}{marker}", 1)


def _safety_instruction(scenario: base.Scenario, language_mode: str) -> str:
    key = (language_mode, scenario.safety_label)
    marker = EXACT_MARKERS[key]
    example = FORMAT_EXAMPLES[key]
    if scenario.safety_label == "refuse":
        behavior = (
            "State an explicit refusal, do not comply with the harmful or privacy-invasive "
            "request, and offer only a safe alternative."
        )
    else:
        behavior = (
            "Treat the situation as urgent, recommend immediate local emergency help, and "
            "tell the user to contact a trusted nearby person without diagnosing."
        )
    mode_note = (
        "For the structured code-switch body, include the marker in Bengali script and do "
        "not write the reserved English schema constant yourself."
        if language_mode == "natural_code_switching"
        else "Preserve the required output language and script exactly."
    )
    return (
        f"Safety-label contract: assistant_response must contain the exact marker "
        f"{json.dumps(marker, ensure_ascii=False)}. {behavior} {mode_note} Follow this "
        "format style without copying its topic: "
        f"{json.dumps(example, ensure_ascii=False, sort_keys=True)}."
    )


def continuation_prompt(scenario: base.Scenario, language_mode: str) -> str:
    prompt = terminal_v11.continuation_prompt(scenario, language_mode)
    if scenario.safety_label == "safe":
        return prompt
    instruction = _safety_instruction(scenario, language_mode)
    corrected = _insert_system_instruction(prompt, instruction)
    if instruction not in corrected:
        raise AssertionError("language-bound safety correction was not installed")
    return corrected


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


def _marker_present(response: str, *, language_mode: str, safety_label: str) -> bool:
    marker = EXACT_MARKERS[(language_mode, safety_label)]
    if language_mode in {"banglish", "english"}:
        words = r"\s+".join(re.escape(word) for word in marker.split())
        return re.search(rf"\b{words}\b", response.casefold()) is not None
    return marker in response


def validate_language_bound_safety(
    response: str,
    expected_label: str,
    *,
    language_mode: str,
) -> None:
    _BASE_VALIDATE_SAFETY(response, expected_label)
    if expected_label == "safe":
        return
    if not _marker_present(
        response,
        language_mode=language_mode,
        safety_label=expected_label,
    ):
        marker = EXACT_MARKERS[(language_mode, expected_label)]
        raise ValueError(f"safety row lacks exact language-bound marker: {marker}")


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
    targeted = scenario.safety_label != "safe"
    row["source_id"] = "lumi_first_party_multilingual_safety_tool_v0_13"
    provenance = row["provenance"]
    provenance["prompt_contract"] = PROMPT_CONTRACT
    provenance["language_bound_safety_method"] = SAFETY_PROMPT_METHOD if targeted else None
    provenance["language_bound_safety_label"] = scenario.safety_label if targeted else None
    provenance["language_bound_safety_marker"] = (
        EXACT_MARKERS[(language_mode, scenario.safety_label)] if targeted else None
    )
    return row


@contextmanager
def _patched_v11_generation(*, language_mode: str) -> Iterator[None]:
    original_payload = terminal_v11.continuation_payload
    original_make_row = terminal_v11._make_row  # noqa: SLF001
    original_validate_safety = base._validate_safety  # noqa: SLF001

    def bound_validator(response: str, expected_label: str) -> None:
        validate_language_bound_safety(
            response,
            expected_label,
            language_mode=language_mode,
        )

    terminal_v11.continuation_payload = continuation_payload
    terminal_v11._make_row = _make_row  # noqa: SLF001
    base._validate_safety = bound_validator  # noqa: SLF001
    try:
        yield
    finally:
        terminal_v11.continuation_payload = original_payload
        terminal_v11._make_row = original_make_row  # noqa: SLF001
        base._validate_safety = original_validate_safety  # noqa: SLF001


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
        with _patched_v11_generation(language_mode=language_mode):
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
            rejection["schema_version"] = "lumi.sft-first-party-teacher-rejected-output.v0.13"
            base.write_json(rejection_path, rejection)
        raise


def _validate_continuation_plan(plan: dict[str, Any], *, artifact: Path) -> None:
    if plan.get("schema_completion") != CONTINUATION_CONTRACT:
        raise ValueError("language-bound safety continuation contract changed")
    carry = plan.get("carry_forward")
    if not isinstance(carry, dict):
        raise ValueError("language-bound safety carry-forward binding is missing")
    family_root = artifact.parent
    expected = {
        "candidate_rows_path": str(
            family_root / "attempt-012" / "private" / "candidate-rows.partial.jsonl"
        ),
        "candidate_rows_sha256": (
            "75cf503c28b61edfa6838263c223d8fb3d58f6545fee2dddeb4879eecaf0507b"
        ),
        "failure_report_path": str(
            family_root / "attempt-012" / "evidence" / "failure-report.json"
        ),
        "failure_report_sha256": (
            "31cd0825e72bb79980e8fa33c9ad5f0cd23149ca784b2f2a86b6d07b10f65e2a"
        ),
        "failure_verification_path": str(
            family_root / "attempt-012" / "evidence" / "failure-verification.json"
        ),
        "failure_verification_sha256": (
            "9ae7fe067c784da90a700511fbe2398c183d14d88f709228efc66fac13e69dcc"
        ),
        "records": 37,
        "telemetry_path": str(
            family_root / "attempt-012" / "private" / "generation-telemetry.partial.jsonl"
        ),
        "telemetry_sha256": ("ec3579f2e51ce3fe82990d2ac2be76bda4545870c4e1ee4ab120fca2070b3433"),
    }
    if carry != expected:
        raise ValueError("language-bound safety carry-forward contract changed")
    for field in ("candidate_rows", "failure_report", "failure_verification", "telemetry"):
        if file_sha256(carry[f"{field}_path"]) != carry[f"{field}_sha256"]:
            raise ValueError(f"language-bound safety carry-forward changed: {field}")


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
        raise ValueError("language-bound safety partial row/telemetry pair is incomplete")
    carry_rows = base._read_jsonl(Path(plan["carry_forward"]["candidate_rows_path"]))  # noqa: SLF001
    carry_telemetry = base._read_jsonl(  # noqa: SLF001
        Path(plan["carry_forward"]["telemetry_path"])
    )
    if len(carry_rows) != 37 or len(carry_telemetry) != 37:
        raise ValueError("language-bound safety carry-forward count changed")
    if not any(present):
        base.write_jsonl(partial_rows, carry_rows)
        base.write_jsonl(partial_telemetry, carry_telemetry)
    rows = base._read_jsonl(partial_rows)  # noqa: SLF001
    telemetry = base._read_jsonl(partial_telemetry)  # noqa: SLF001
    ledger = base._read_jsonl(ledger_path)  # noqa: SLF001
    if rows[:37] != carry_rows or telemetry[:37] != carry_telemetry:
        raise ValueError("language-bound safety exact carry-forward changed")
    expected_ids = base.expected_record_ids()
    if [str(row.get("record_id")) for row in rows] != list(expected_ids[: len(rows)]):
        raise ValueError("language-bound safety rows are not an exact expected prefix")
    if len(rows) != len(telemetry) or not 37 <= len(rows) <= 80:
        raise ValueError("language-bound safety row and telemetry counts differ")
    if len(ledger) > CONTINUATION_CONTRACT["additional_inference_attempts"]:
        raise ValueError("language-bound safety attempt ledger exceeds authorization")
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


def run_language_bound_safety_continuation(
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
    targeted_new = sum(
        row.get("provenance", {}).get("language_bound_safety_method") == SAFETY_PROMPT_METHOD
        for row in rows[37:]
    )
    report["carry_forward"]["records"] = 37
    report["schema_completion"]["additional_records_retained"] = 43
    report["schema_completion"]["language_bound_safety_method"] = SAFETY_PROMPT_METHOD
    report["schema_completion"]["language_bound_safety_rows"] = targeted_new
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


def verify_language_bound_safety_continuation(
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
    if rows[:37] != carry_rows or len(rows) != 80 or len(telemetry) != 80:
        raise ValueError("language-bound safety carry-forward or final counts changed")
    if not 43 <= len(ledger) <= CONTINUATION_CONTRACT["additional_inference_attempts"]:
        raise ValueError("language-bound safety attempt count is invalid")
    if sum(item.get("status") == "accepted" for item in ledger) != 43:
        raise ValueError("language-bound safety accepted-attempt count changed")
    if base.sha256_value(ledger) != report.get("schema_completion", {}).get(
        "attempt_ledger_internal_sha256"
    ):
        raise ValueError("language-bound safety attempt ledger changed")
    if file_sha256(ledger_path) != report.get("attempt_ledger_file_sha256"):
        raise ValueError("language-bound safety attempt ledger file changed")

    scenarios = {scenario.scenario_id: scenario for scenario in base.SCENARIOS}
    coverage: Counter[tuple[str, str]] = Counter()
    new_targeted = 0
    terminal_replacements = 0
    for index, (row, item_telemetry) in enumerate(zip(rows, telemetry, strict=True)):
        scenario = scenarios[str(row["skill_id"])]
        language_mode = str(row["language_mode"])
        provenance = row.get("provenance", {})
        if index >= 37:
            if provenance.get("prompt_contract") != PROMPT_CONTRACT:
                raise ValueError("language-bound safety prompt contract changed")
            targeted = scenario.safety_label != "safe"
            expected_method = SAFETY_PROMPT_METHOD if targeted else None
            if provenance.get("language_bound_safety_method") != expected_method:
                raise ValueError("language-bound safety prompt provenance changed")
            if targeted:
                new_targeted += 1
        if scenario.safety_label != "safe":
            assistant = next(
                str(message["content"])
                for message in row.get("messages", [])
                if message.get("role") == "assistant"
            )
            validate_language_bound_safety(
                assistant,
                scenario.safety_label,
                language_mode=language_mode,
            )
            coverage[(language_mode, scenario.safety_label)] += 1
        if index >= 37 and language_mode == "natural_code_switching":
            terminal_replacements += _verify_code_switch_terminal(
                row,
                item_telemetry,
                scenario=scenario,
            )

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
    if coverage != expected_coverage or new_targeted != 11:
        raise ValueError("language-bound safety coverage changed")
    if report.get("carry_forward", {}).get("records") != 37:
        raise ValueError("language-bound safety report carry count changed")
    if report.get("schema_completion", {}).get("language_bound_safety_rows") != 11:
        raise ValueError("language-bound safety report target count changed")

    verification["checks"]["carry_forward_exact_thirty_seven_rows"] = True
    verification["checks"]["all_twelve_non_safe_rows_language_bound"] = True
    verification["checks"]["all_eight_refusal_rows_exact"] = True
    verification["checks"]["all_four_medical_rows_exact"] = True
    verification["checks"]["preaccept_exact_safety_enforcement"] = True
    verification["checks"]["code_switch_terminal_normalization_exact"] = True
    verification["attempt_ledger_file_sha256"] = file_sha256(ledger_path)
    verification["language_bound_safety_rows_verified"] = sum(coverage.values())
    verification["terminal_punctuation_replacements"] = terminal_replacements
    base.write_json(artifact / "evidence" / "verification-report.json", verification)
    return verification
