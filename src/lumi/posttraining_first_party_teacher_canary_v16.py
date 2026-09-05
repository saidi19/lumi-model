"""Structured-body continuation for the remaining Banglish writing rows."""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import posttraining_first_party_teacher_canary as base
from . import posttraining_first_party_teacher_canary_v4 as schema_v4
from . import posttraining_first_party_teacher_canary_v5 as corrected_v5
from . import posttraining_first_party_teacher_canary_v11 as terminal_v11
from . import posttraining_first_party_teacher_canary_v14 as punctuation_v14
from . import posttraining_first_party_teacher_canary_v15 as task_v15
from .posttraining_first_party import file_sha256

CONTINUATION_CONTRACT = {
    "additional_inference_attempts": 48,
    "maximum_attempts_per_record": 3,
    "maximum_network_bytes": 0,
    "maximum_retained_new_records": 16,
    "maximum_retained_supervised_tokens_total": 24_000,
    "maximum_runtime_minutes": 120,
    "total_retained_records": 80,
}
CARRY_RECORDS = 64
NEW_RECORDS = 16
NEW_STRUCTURED_BANGLISH_ROWS = 4
NEW_NATURAL_CODE_SWITCHING_ROWS = 4
PROMPT_CONTRACT = "lumi.first-party-teacher-prompt.v0.16"
WRITING_BODY_COMPOSITION_METHOD = "structured_bodies_plus_fixed_natural_wrappers_v1"
WRITING_BODY_TEMPERATURE = 0.2
USER_WRAPPER = "Eta ektu short summary koro: "
ASSISTANT_WRAPPER = "Ami short summary dichhi: "
SUPPORTED_BODY_TERMINALS = frozenset({".", "?", "!"})

_V15_PAYLOAD = task_v15.continuation_payload
_V15_GENERATE = task_v15.generate_continuation_attempt
_V15_MAKE_ROW = task_v15._make_row  # noqa: SLF001

WRITING_BODY_INSTRUCTIONS = {
    "meeting_summary": (
        "user_prompt.body must be only a short source note saying that the meeting is at "
        "10:00 in Room B and ID must be brought. assistant_response.body must be only a "
        "concise summary of those same three facts."
    ),
    "release_summary": (
        "user_prompt.body must be only a short source note saying that version 2.4 adds "
        "dark mode and offline search. assistant_response.body must be only a concise "
        "summary of those same facts."
    ),
    "plant_care_summary": (
        "user_prompt.body must be only a short source note saying to water 3 times/week, "
        "use indirect light, and check soil Friday. assistant_response.body must be only a "
        "concise summary of those same three facts."
    ),
    "project_summary": (
        "user_prompt.body must be only a short source note saying that Phase Alpha is due "
        "12 May and Nadia owns the review. assistant_response.body must be only a concise "
        "summary of those same facts."
    ),
}

BODY_STYLE_EXAMPLE = {
    "assistant_response": {"body": "Kal 9:00 e office meeting hobe."},
    "user_prompt": {"body": "Note e kal 9:00 e office meeting ache."},
}


def _targeted(scenario: base.Scenario, language_mode: str) -> bool:
    return language_mode == "banglish" and scenario.scenario_id in WRITING_BODY_INSTRUCTIONS


def writing_body_schema(scenario: base.Scenario) -> dict[str, Any]:
    component = {
        "type": "object",
        "properties": {"body": {"type": "string"}},
        "required": ["body"],
        "additionalProperties": False,
    }
    schema = copy.deepcopy(base.OUTPUT_SCHEMA)
    schema["properties"]["scenario_id"] = {
        "type": "string",
        "enum": [scenario.scenario_id],
    }
    schema["properties"]["safety_label"] = {
        "type": "string",
        "enum": [scenario.safety_label],
    }
    schema["properties"]["tool_call"] = {"type": "null"}
    schema["properties"]["tool_result"] = {"type": "null"}
    schema["properties"]["user_prompt"] = copy.deepcopy(component)
    schema["properties"]["assistant_response"] = copy.deepcopy(component)
    return schema


def continuation_prompt(scenario: base.Scenario, language_mode: str) -> str:
    prompt = punctuation_v14.continuation_prompt(scenario, language_mode)
    if not _targeted(scenario, language_mode):
        return task_v15.continuation_prompt(scenario, language_mode)
    instruction = (
        "For this writing row, return user_prompt and assistant_response as objects with "
        "exactly one field named body. Write both body strings in natural romanized Bangla "
        "using Latin letters only; do not use Bengali script. user_prompt.body is only the "
        "source note, not a question or request. assistant_response.body is only its concise "
        "summary. Each body must independently contain a natural romanized Bengali word, "
        "preserve every required literal exactly, and end with ., ?, or !. Do not add the "
        "words Eta or Ami merely to satisfy style; the runtime will add fixed natural "
        "wrappers. Do not copy the example topic or literals. Nested-field style example: "
        f"{json.dumps(BODY_STYLE_EXAMPLE, ensure_ascii=False, sort_keys=True)}."
    )
    if corrected_v5.STRICT_BANGLISH_INSTRUCTION not in prompt:
        raise AssertionError("predecessor Banglish instruction was not found")
    prompt = prompt.replace(corrected_v5.STRICT_BANGLISH_INSTRUCTION, instruction, 1)
    prompt = prompt.replace(
        "Generate an original user prompt and a direct, concise assistant response.",
        "Generate an original source-note body and its direct, concise summary body.",
        1,
    )
    old_instruction = f"Instruction: {scenario.instruction}"
    new_instruction = (
        f"Structured writing instruction: {WRITING_BODY_INSTRUCTIONS[scenario.scenario_id]}"
    )
    if old_instruction not in prompt:
        raise AssertionError("writing scenario instruction was not found")
    prompt = prompt.replace(old_instruction, new_instruction, 1)
    prompt = prompt.replace(
        "End the assistant response with proper punctuation.",
        "End both body strings with supported terminal punctuation.",
        1,
    )
    if instruction not in prompt or new_instruction not in prompt:
        raise AssertionError("structured writing correction was not installed")
    return prompt


def continuation_payload(
    record_id: str,
    scenario: base.Scenario,
    language_mode: str,
    *,
    attempt_number: int,
) -> dict[str, Any]:
    payload = _V15_PAYLOAD(
        record_id,
        scenario,
        language_mode,
        attempt_number=attempt_number,
    )
    if not _targeted(scenario, language_mode):
        return payload
    payload["prompt"] = continuation_prompt(scenario, language_mode)
    payload["format"] = writing_body_schema(scenario)
    payload["options"]["temperature"] = WRITING_BODY_TEMPERATURE
    return payload


def _body(value: Any, *, field: str, scenario: base.Scenario) -> str:
    if not isinstance(value, Mapping) or set(value) != {"body"}:
        raise ValueError(f"structured writing {field} must contain only body")
    body = value.get("body")
    if not isinstance(body, str) or len(body) < 8:
        raise ValueError(f"structured writing {field} body is too short")
    if body != body.strip():
        raise ValueError(f"structured writing {field} body has surrounding whitespace")
    if any(character in body for character in ("\n", "\r", "\t")):
        raise ValueError(f"structured writing {field} body contains control whitespace")
    if base._BENGALI.search(body) is not None:  # noqa: SLF001
        raise ValueError(f"structured writing {field} body contains Bengali script")
    if base._LATIN.search(body) is None:  # noqa: SLF001
        raise ValueError(f"structured writing {field} body lacks Latin text")
    if body[-1] not in SUPPORTED_BODY_TERMINALS:
        raise ValueError(f"structured writing {field} body lacks supported terminal punctuation")
    if not any(
        task_v15._has_exact_word(body, marker)  # noqa: SLF001
        for marker in corrected_v5.CORRECTED_BANGLISH_MARKERS
    ):
        raise ValueError(f"structured writing {field} body lacks romanized Bengali markers")
    for literal in scenario.required_literals:
        if literal not in body:
            raise ValueError(f"structured writing {field} body lacks required literal: {literal}")
    return body


def compose_writing_bodies(
    output: Mapping[str, Any],
    *,
    scenario: base.Scenario,
) -> tuple[dict[str, Any], list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    expected_fields = {
        "assistant_response",
        "safety_label",
        "scenario_id",
        "tool_call",
        "tool_result",
        "user_prompt",
    }
    if set(output) != expected_fields:
        raise ValueError("structured writing teacher JSON envelope fields changed")
    if output.get("scenario_id") != scenario.scenario_id:
        raise ValueError("structured writing scenario binding changed")
    if output.get("safety_label") != scenario.safety_label:
        raise ValueError("structured writing safety label changed")
    if output.get("tool_call") is not None or output.get("tool_result") is not None:
        raise ValueError("structured writing tool fields must remain null")

    user_body = _body(output.get("user_prompt"), field="user_prompt", scenario=scenario)
    assistant_body = _body(
        output.get("assistant_response"),
        field="assistant_response",
        scenario=scenario,
    )
    if user_body == assistant_body:
        raise ValueError("structured writing note and summary bodies are identical")
    if "?" in assistant_body:
        raise ValueError("structured writing summary body is a question")

    flattened = dict(output)
    flattened["user_prompt"] = f"{USER_WRAPPER}{user_body}"
    flattened["assistant_response"] = f"{ASSISTANT_WRAPPER}{assistant_body}"
    normalized, messages, terminal_composition = terminal_v11.compose_banglish_terminal_output(
        flattened,
        scenario=scenario,
    )
    if any(
        terminal_composition[f"{field}_action"] != "preserve_existing_terminal"
        for field in ("user_prompt", "assistant_response")
    ):
        raise ValueError("structured writing body terminal was unexpectedly modified")
    composition = {
        "assistant_body_sha256": base.sha256_value(assistant_body),
        "assistant_final_sha256": base.sha256_value(normalized["assistant_response"]),
        "assistant_wrapper": ASSISTANT_WRAPPER,
        "body_fields_preserved_byte_exact": True,
        "method": WRITING_BODY_COMPOSITION_METHOD,
        "user_body_sha256": base.sha256_value(user_body),
        "user_final_sha256": base.sha256_value(normalized["user_prompt"]),
        "user_wrapper": USER_WRAPPER,
    }
    return normalized, messages, terminal_composition, composition


def _make_row(
    record_id: str,
    scenario: base.Scenario,
    language_mode: str,
    messages: list[dict[str, str]],
    *,
    attempt_number: int,
    terminal_composition: Mapping[str, Any] | None,
    banglish_composition: Mapping[str, Any] | None,
    writing_body_composition: Mapping[str, Any] | None,
) -> dict[str, Any]:
    row = _V15_MAKE_ROW(
        record_id,
        scenario,
        language_mode,
        messages,
        attempt_number=attempt_number,
        terminal_composition=terminal_composition,
        banglish_composition=banglish_composition,
    )
    row["source_id"] = "lumi_first_party_multilingual_safety_tool_v0_16"
    provenance = row["provenance"]
    provenance["prompt_contract"] = PROMPT_CONTRACT
    provenance["writing_body_composition_method"] = (
        WRITING_BODY_COMPOSITION_METHOD if writing_body_composition is not None else None
    )
    provenance["writing_body_composition_internal_sha256"] = (
        base.sha256_value(writing_body_composition)
        if writing_body_composition is not None
        else None
    )
    provenance["writing_body_temperature"] = (
        WRITING_BODY_TEMPERATURE if writing_body_composition is not None else None
    )
    return row


def _targeted_generation_attempt(
    record_id: str,
    scenario: base.Scenario,
    language_mode: str,
    *,
    attempt_number: int,
    endpoint: str,
    rejection_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = continuation_payload(
        record_id,
        scenario,
        language_mode,
        attempt_number=attempt_number,
    )
    response: dict[str, Any] | None = None
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
        _, messages, banglish_composition, writing_composition = compose_writing_bodies(
            output,
            scenario=scenario,
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
                "schema_version": "lumi.sft-first-party-teacher-rejected-output.v0.16",
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
        terminal_composition=None,
        banglish_composition=banglish_composition,
        writing_body_composition=writing_composition,
    )
    telemetry = {
        "attempt_number": attempt_number,
        "banglish_marker_contract": corrected_v5.BANGLISH_MARKER_CONTRACT,
        "banglish_terminal_composition": banglish_composition,
        "elapsed_seconds": round(elapsed, 6),
        "eval_count": response.get("eval_count"),
        "eval_duration_ns": response.get("eval_duration"),
        "load_duration_ns": response.get("load_duration"),
        "prompt_eval_count": response.get("prompt_eval_count"),
        "prompt_eval_duration_ns": response.get("prompt_eval_duration"),
        "record_id": record_id,
        "request_sha256": base.sha256_value(payload),
        "response_envelope_sha256": base.sha256_value(response),
        "terminal_composition": None,
        "writing_body_composition": writing_composition,
    }
    return row, telemetry


def generate_continuation_attempt(
    record_id: str,
    scenario: base.Scenario,
    language_mode: str,
    *,
    attempt_number: int,
    endpoint: str,
    rejection_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if _targeted(scenario, language_mode):
        return _targeted_generation_attempt(
            record_id,
            scenario,
            language_mode,
            attempt_number=attempt_number,
            endpoint=endpoint,
            rejection_path=rejection_path,
        )
    return _V15_GENERATE(
        record_id,
        scenario,
        language_mode,
        attempt_number=attempt_number,
        endpoint=endpoint,
        rejection_path=rejection_path,
    )


def _validate_continuation_plan(plan: dict[str, Any], *, artifact: Path) -> None:
    if plan.get("schema_completion") != CONTINUATION_CONTRACT:
        raise ValueError("structured writing continuation contract changed")
    carry = plan.get("carry_forward")
    if not isinstance(carry, dict):
        raise ValueError("structured writing carry-forward binding is missing")
    family_root = artifact.parent
    expected = {
        "candidate_rows_path": str(
            family_root / "attempt-015" / "private" / "candidate-rows.partial.jsonl"
        ),
        "candidate_rows_sha256": (
            "ba6650bf74a20f6e4dc06a1e9a6a0f3330d8e0b865316406d7d273d205e84fd1"
        ),
        "failure_report_path": str(
            family_root / "attempt-015" / "evidence" / "failure-report.json"
        ),
        "failure_report_sha256": (
            "79aba9a92a622211149cffb001b32e8e1dd0d7295173f8c405347263d9125578"
        ),
        "failure_verification_path": str(
            family_root / "attempt-015" / "evidence" / "failure-verification.json"
        ),
        "failure_verification_sha256": (
            "1a8f353a8688669cc2ab19ec3f5eab598716002b31c8d1983d44d85419fcc5af"
        ),
        "records": CARRY_RECORDS,
        "telemetry_path": str(
            family_root / "attempt-015" / "private" / "generation-telemetry.partial.jsonl"
        ),
        "telemetry_sha256": ("02c8393fc9e2c710260838d2c86f0758b64b45ce21820e3c0f135ba723aebaa2"),
    }
    if carry != expected:
        raise ValueError("structured writing carry-forward contract changed")
    for field in ("candidate_rows", "failure_report", "failure_verification", "telemetry"):
        if file_sha256(carry[f"{field}_path"]) != carry[f"{field}_sha256"]:
            raise ValueError(f"structured writing carry-forward changed: {field}")


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
        raise ValueError("structured writing partial row/telemetry pair is incomplete")
    carry_rows = base._read_jsonl(Path(plan["carry_forward"]["candidate_rows_path"]))  # noqa: SLF001
    carry_telemetry = base._read_jsonl(  # noqa: SLF001
        Path(plan["carry_forward"]["telemetry_path"])
    )
    if len(carry_rows) != CARRY_RECORDS or len(carry_telemetry) != CARRY_RECORDS:
        raise ValueError("structured writing carry-forward count changed")
    if not any(present):
        base.write_jsonl(partial_rows, carry_rows)
        base.write_jsonl(partial_telemetry, carry_telemetry)
    rows = base._read_jsonl(partial_rows)  # noqa: SLF001
    telemetry = base._read_jsonl(partial_telemetry)  # noqa: SLF001
    ledger = base._read_jsonl(ledger_path)  # noqa: SLF001
    if rows[:CARRY_RECORDS] != carry_rows or telemetry[:CARRY_RECORDS] != carry_telemetry:
        raise ValueError("structured writing exact carry-forward changed")
    expected_ids = base.expected_record_ids()
    if [str(row.get("record_id")) for row in rows] != list(expected_ids[: len(rows)]):
        raise ValueError("structured writing rows are not an exact expected prefix")
    if len(rows) != len(telemetry) or not CARRY_RECORDS <= len(rows) <= 80:
        raise ValueError("structured writing row and telemetry counts differ")
    if len(ledger) > CONTINUATION_CONTRACT["additional_inference_attempts"]:
        raise ValueError("structured writing attempt ledger exceeds authorization")
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


def run_structured_writing_continuation(
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
    report["schema_completion"]["writing_body_composition_method"] = WRITING_BODY_COMPOSITION_METHOD
    report["schema_completion"]["structured_Banglish_writing_rows"] = sum(
        row.get("provenance", {}).get("writing_body_composition_method")
        == WRITING_BODY_COMPOSITION_METHOD
        for row in new_rows
    )
    report["schema_completion"]["natural_code_switching_rows"] = sum(
        row.get("language_mode") == "natural_code_switching" for row in new_rows
    )
    base.write_json(Path(artifact_root) / "evidence" / "run-report.json", report)
    return report


def _verify_writing_composition(
    row: Mapping[str, Any],
    telemetry: Mapping[str, Any],
) -> None:
    composition = telemetry.get("writing_body_composition")
    if not isinstance(composition, Mapping):
        raise ValueError("structured writing composition telemetry is missing")
    if composition.get("method") != WRITING_BODY_COMPOSITION_METHOD:
        raise ValueError("structured writing composition method changed")
    if composition.get("user_wrapper") != USER_WRAPPER:
        raise ValueError("structured writing user wrapper changed")
    if composition.get("assistant_wrapper") != ASSISTANT_WRAPPER:
        raise ValueError("structured writing assistant wrapper changed")
    natural = {
        message["role"]: str(message["content"])
        for message in row.get("messages", [])
        if message.get("role") in {"user", "assistant"}
    }
    for role, prefix, body_hash_key, final_hash_key in (
        ("user", USER_WRAPPER, "user_body_sha256", "user_final_sha256"),
        (
            "assistant",
            ASSISTANT_WRAPPER,
            "assistant_body_sha256",
            "assistant_final_sha256",
        ),
    ):
        text = natural[role]
        if not text.startswith(prefix):
            raise ValueError(f"structured writing {role} wrapper changed")
        body = text[len(prefix) :]
        if base.sha256_value(body) != composition.get(body_hash_key):
            raise ValueError(f"structured writing {role} body changed")
        if base.sha256_value(text) != composition.get(final_hash_key):
            raise ValueError(f"structured writing {role} final field changed")
    provenance = row.get("provenance", {})
    if provenance.get("writing_body_composition_internal_sha256") != base.sha256_value(composition):
        raise ValueError("structured writing provenance binding changed")


def verify_structured_writing_continuation(
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
        raise ValueError("structured writing carry-forward or final counts changed")
    if not NEW_RECORDS <= len(ledger) <= CONTINUATION_CONTRACT["additional_inference_attempts"]:
        raise ValueError("structured writing attempt count is invalid")
    if sum(item.get("status") == "accepted" for item in ledger) != NEW_RECORDS:
        raise ValueError("structured writing accepted-attempt count changed")
    if base.sha256_value(ledger) != report.get("schema_completion", {}).get(
        "attempt_ledger_internal_sha256"
    ):
        raise ValueError("structured writing attempt ledger changed")
    if file_sha256(ledger_path) != report.get("attempt_ledger_file_sha256"):
        raise ValueError("structured writing attempt ledger file changed")

    scenarios = {scenario.scenario_id: scenario for scenario in base.SCENARIOS}
    structured_rows = 0
    code_switch_rows = 0
    for index, (row, item_telemetry) in enumerate(zip(rows, telemetry, strict=True)):
        if index < CARRY_RECORDS:
            continue
        scenario = scenarios[str(row["skill_id"])]
        language_mode = str(row["language_mode"])
        if _targeted(scenario, language_mode):
            provenance = row.get("provenance", {})
            if row.get("source_id") != "lumi_first_party_multilingual_safety_tool_v0_16":
                raise ValueError("structured writing source binding changed")
            if provenance.get("prompt_contract") != PROMPT_CONTRACT:
                raise ValueError("structured writing prompt contract changed")
            if (
                provenance.get("writing_body_composition_method") != WRITING_BODY_COMPOSITION_METHOD
                or provenance.get("writing_body_temperature") != WRITING_BODY_TEMPERATURE
            ):
                raise ValueError("structured writing provenance changed")
            _verify_writing_composition(row, item_telemetry)
            if terminal_v11._verify_banglish_composition(row, item_telemetry) != 0:  # noqa: SLF001
                raise ValueError("structured writing terminal was modified")
            structured_rows += 1
        elif row.get("source_id") != "lumi_first_party_multilingual_safety_tool_v0_15":
            raise ValueError("non-target predecessor source binding changed")
        if language_mode == "natural_code_switching":
            task_v15._verify_new_code_switch_terminal(  # noqa: SLF001
                row,
                item_telemetry,
                scenario=scenario,
            )
            code_switch_rows += 1

    if structured_rows != NEW_STRUCTURED_BANGLISH_ROWS:
        raise ValueError("structured writing row count changed")
    if code_switch_rows != NEW_NATURAL_CODE_SWITCHING_ROWS:
        raise ValueError("remaining code-switch row count changed")
    schema_report = report.get("schema_completion", {})
    if report.get("carry_forward", {}).get("records") != CARRY_RECORDS:
        raise ValueError("structured writing report carry count changed")
    if schema_report.get("structured_Banglish_writing_rows") != structured_rows:
        raise ValueError("structured writing report row count changed")
    if schema_report.get("natural_code_switching_rows") != code_switch_rows:
        raise ValueError("remaining code-switch report row count changed")

    verification["checks"]["carry_forward_exact_sixty_four_rows"] = True
    verification["checks"]["all_four_remaining_Banglish_writing_rows_structured"] = True
    verification["checks"]["generated_body_fields_preserved_byte_exact"] = True
    verification["checks"]["fixed_natural_wrappers_verified"] = True
    verification["checks"]["all_non_target_generation_paths_preserved_v0_15"] = True
    verification["attempt_ledger_file_sha256"] = file_sha256(ledger_path)
    verification["structured_Banglish_writing_rows_verified"] = structured_rows
    verification["natural_code_switching_rows_verified"] = code_switch_rows
    base.write_json(artifact / "evidence" / "verification-report.json", verification)
    return verification
