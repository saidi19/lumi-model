"""English-mode-corrected continuation of the local Qwen teacher canary."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import posttraining_first_party_teacher_canary as base
from . import posttraining_first_party_teacher_canary_v4 as schema_v4
from . import posttraining_first_party_teacher_canary_v5 as corrected_v5
from . import posttraining_first_party_teacher_canary_v8 as suffix_v8
from . import posttraining_first_party_teacher_canary_v9 as terminal_v9
from .posttraining_first_party import file_sha256

CONTINUATION_CONTRACT = {
    "additional_inference_attempts": 100,
    "maximum_attempts_per_record": 3,
    "maximum_network_bytes": 0,
    "maximum_retained_new_records": 62,
    "maximum_retained_supervised_tokens_total": 24_000,
    "maximum_runtime_minutes": 120,
    "total_retained_records": 80,
}
PROMPT_CONTRACT = "lumi.first-party-teacher-prompt.v0.10"
ENGLISH_PROMPT_METHOD = "neutral_task_alias_plus_english_only_example_v1"
ENGLISH_FORMAT_EXAMPLE = {
    "assistant_response": "Hello! I am doing well, thank you.",
    "user_prompt": "Hello, how are you?",
}


def continuation_seed(record_id: str, attempt_number: int) -> int:
    if not 1 <= attempt_number <= CONTINUATION_CONTRACT["maximum_attempts_per_record"]:
        raise ValueError("English-corrected attempt number is outside authorization")
    material = f"{record_id}:english-corrected-continuation-attempt:{attempt_number}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def continuation_prompt(scenario: base.Scenario, language_mode: str) -> str:
    prompt = terminal_v9.terminal_prompt(scenario, language_mode)
    if language_mode != "english":
        return prompt
    instruction = (
        "Write user_prompt and assistant_response in natural English only. Every word must "
        "use Latin script; use zero Bengali characters. The task alias below describes the "
        "scenario, not the output language. Use correct English spelling and punctuation. "
        "Return no translation, language label, or meta-commentary. Follow this format style "
        f"without copying its topic: {json.dumps(ENGLISH_FORMAT_EXAMPLE, sort_keys=True)}."
    )
    prompt = prompt.replace(base.LANGUAGE_INSTRUCTIONS[language_mode], instruction)
    old_task = f"Task: {scenario.task}"
    new_task = f"Task alias: {scenario.scenario_id}"
    if old_task not in prompt:
        raise AssertionError("English task metadata was not found")
    prompt = prompt.replace(old_task, new_task, 1)
    if instruction not in prompt or old_task in prompt:
        raise AssertionError("English correction was not installed")
    return prompt


def continuation_payload(
    record_id: str,
    scenario: base.Scenario,
    language_mode: str,
    *,
    attempt_number: int,
) -> dict[str, Any]:
    payload = base.ollama_payload(record_id, scenario, language_mode)
    payload["prompt"] = continuation_prompt(scenario, language_mode)
    payload["options"]["seed"] = continuation_seed(record_id, attempt_number)
    if language_mode == "natural_code_switching":
        payload["format"] = terminal_v9.terminal_output_schema(scenario)
        payload["options"]["temperature"] = terminal_v9.NATURAL_CODE_SWITCH_TEMPERATURE
    return payload


def _make_row(
    record_id: str,
    scenario: base.Scenario,
    language_mode: str,
    messages: list[dict[str, str]],
    *,
    attempt_number: int,
    composition: Mapping[str, Any] | None,
) -> dict[str, Any]:
    row = terminal_v9._make_row(  # noqa: SLF001
        record_id,
        scenario,
        language_mode,
        messages,
        attempt_number=attempt_number,
        composition=composition,
    )
    row["source_id"] = "lumi_first_party_multilingual_safety_tool_v0_10"
    provenance = row["provenance"]
    provenance["english_prompt_method"] = (
        ENGLISH_PROMPT_METHOD if language_mode == "english" else None
    )
    provenance["prompt_contract"] = PROMPT_CONTRACT
    provenance["seed"] = continuation_seed(record_id, attempt_number)
    return row


def generate_continuation_attempt(
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
    composition: dict[str, Any] | None = None
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
            _, messages, composition = terminal_v9.compose_terminal_output(
                output,
                scenario=scenario,
            )
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
                "schema_version": "lumi.sft-first-party-teacher-rejected-output.v0.10",
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
        composition=composition,
    )
    telemetry = {
        "attempt_number": attempt_number,
        "banglish_marker_contract": corrected_v5.BANGLISH_MARKER_CONTRACT
        if language_mode == "banglish"
        else None,
        "elapsed_seconds": round(elapsed, 6),
        "english_prompt_method": ENGLISH_PROMPT_METHOD if language_mode == "english" else None,
        "eval_count": response.get("eval_count"),
        "eval_duration_ns": response.get("eval_duration"),
        "load_duration_ns": response.get("load_duration"),
        "prompt_eval_count": response.get("prompt_eval_count"),
        "prompt_eval_duration_ns": response.get("prompt_eval_duration"),
        "record_id": record_id,
        "request_sha256": base.sha256_value(payload),
        "response_envelope_sha256": base.sha256_value(response),
        "terminal_composition": composition,
    }
    return row, telemetry


def _validate_continuation_plan(plan: dict[str, Any], *, artifact: Path) -> None:
    if plan.get("schema_completion") != CONTINUATION_CONTRACT:
        raise ValueError("English-corrected continuation contract changed")
    carry = plan.get("carry_forward")
    if not isinstance(carry, dict):
        raise ValueError("English-corrected carry-forward binding is missing")
    family_root = artifact.parent
    expected = {
        "candidate_rows_path": str(
            family_root / "attempt-009" / "private" / "candidate-rows.partial.jsonl"
        ),
        "candidate_rows_sha256": "1a01afd93f58900040ab778e29d26074fba3496b0db35a1a3644f3efc456de4b",
        "failure_report_path": str(
            family_root / "attempt-009" / "evidence" / "failure-report.json"
        ),
        "failure_report_sha256": "72aa2ca47819d4710742a02d68b51a7fa0c116758d15fd689dba714a889b1ce1",
        "failure_verification_path": str(
            family_root / "attempt-009" / "evidence" / "failure-verification.json"
        ),
        "failure_verification_sha256": (
            "04541cdb558326398811017f7812cc75b6fdf00a5303eb023dd76fad0bc75b56"
        ),
        "records": 18,
        "telemetry_path": str(
            family_root / "attempt-009" / "private" / "generation-telemetry.partial.jsonl"
        ),
        "telemetry_sha256": "1f3d66b178b12469f423e1155b2bd703b2cdb95387898122dcff9785eb712fc6",
    }
    if carry != expected:
        raise ValueError("English-corrected carry-forward contract changed")
    for field in ("candidate_rows", "failure_report", "failure_verification", "telemetry"):
        if file_sha256(carry[f"{field}_path"]) != carry[f"{field}_sha256"]:
            raise ValueError(f"English-corrected carry-forward changed: {field}")


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
        raise ValueError("English-corrected partial row/telemetry pair is incomplete")
    carry_rows = base._read_jsonl(Path(plan["carry_forward"]["candidate_rows_path"]))  # noqa: SLF001
    carry_telemetry = base._read_jsonl(  # noqa: SLF001
        Path(plan["carry_forward"]["telemetry_path"])
    )
    if len(carry_rows) != 18 or len(carry_telemetry) != 18:
        raise ValueError("English-corrected carry-forward count changed")
    if not any(present):
        base.write_jsonl(partial_rows, carry_rows)
        base.write_jsonl(partial_telemetry, carry_telemetry)
    rows = base._read_jsonl(partial_rows)  # noqa: SLF001
    telemetry = base._read_jsonl(partial_telemetry)  # noqa: SLF001
    ledger = base._read_jsonl(ledger_path)  # noqa: SLF001
    if rows[:18] != carry_rows or telemetry[:18] != carry_telemetry:
        raise ValueError("English-corrected exact carry-forward changed")
    expected_ids = base.expected_record_ids()
    if [str(row.get("record_id")) for row in rows] != list(expected_ids[: len(rows)]):
        raise ValueError("English-corrected rows are not an exact expected prefix")
    if len(rows) != len(telemetry) or not 18 <= len(rows) <= 80:
        raise ValueError("English-corrected row and telemetry counts differ")
    if len(ledger) > CONTINUATION_CONTRACT["additional_inference_attempts"]:
        raise ValueError("English-corrected attempt ledger exceeds authorization")
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


def run_english_corrected_continuation(
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
    new_telemetry = telemetry[18:]
    compositions = [
        item["terminal_composition"]
        for item in new_telemetry
        if item.get("terminal_composition") is not None
    ]
    replacement_count = sum(
        composition.get(f"{field}_removed_terminal") is not None
        for composition in compositions
        for field in ("user", "assistant")
    )
    report["carry_forward"]["records"] = 18
    report["schema_completion"]["additional_records_retained"] = 62
    report["schema_completion"]["composition_method"] = terminal_v9.TERMINAL_COMPOSITION_METHOD
    report["schema_completion"]["english_prompt_method"] = ENGLISH_PROMPT_METHOD
    report["schema_completion"]["terminal_punctuation_replacements"] = replacement_count
    report["generation"]["natural_code_switching_temperature"] = (
        terminal_v9.NATURAL_CODE_SWITCH_TEMPERATURE
    )
    base.write_json(Path(artifact_root) / "evidence" / "run-report.json", report)
    return report


def verify_english_corrected_continuation(
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
    if rows[:18] != carry_rows or len(rows) != 80 or len(telemetry) != 80:
        raise ValueError("English-corrected carry-forward or final counts changed")
    if not 62 <= len(ledger) <= CONTINUATION_CONTRACT["additional_inference_attempts"]:
        raise ValueError("English-corrected attempt count is invalid")
    if sum(item.get("status") == "accepted" for item in ledger) != 62:
        raise ValueError("English-corrected accepted-attempt count changed")
    if base.sha256_value(ledger) != report.get("schema_completion", {}).get(
        "attempt_ledger_internal_sha256"
    ):
        raise ValueError("English-corrected attempt ledger changed")
    if file_sha256(ledger_path) != report.get("attempt_ledger_file_sha256"):
        raise ValueError("English-corrected attempt ledger file changed")

    scenarios = {scenario.scenario_id: scenario for scenario in base.SCENARIOS}
    terminal_rows = 0
    replacement_count = 0
    english_rows = 0
    for row, item_telemetry in zip(rows[18:], telemetry[18:], strict=True):
        provenance = row.get("provenance", {})
        if provenance.get("prompt_contract") != PROMPT_CONTRACT:
            raise ValueError("English-corrected prompt contract changed")
        language_mode = row.get("language_mode")
        if language_mode == "banglish":
            suffix_v8._verify_banglish_row(row)  # noqa: SLF001
        elif language_mode == "natural_code_switching":
            scenario = scenarios[str(row["skill_id"])]
            replacement_count += terminal_v9._verify_terminal_composition(  # noqa: SLF001
                row,
                item_telemetry,
                scenario=scenario,
            )
            terminal_rows += 1
        elif language_mode == "english":
            english_rows += 1
            if provenance.get("english_prompt_method") != ENGLISH_PROMPT_METHOD:
                raise ValueError("English-corrected prompt provenance changed")
            natural = [
                str(message["content"])
                for message in row.get("messages", [])
                if message.get("role") in {"user", "assistant"}
            ]
            if any(base._BENGALI.search(text) is not None for text in natural):  # noqa: SLF001
                raise ValueError("English-corrected row contains Bengali script")
    if report.get("carry_forward", {}).get("records") != 18:
        raise ValueError("English-corrected report carry count changed")
    if replacement_count != report.get("schema_completion", {}).get(
        "terminal_punctuation_replacements"
    ):
        raise ValueError("English-corrected replacement count changed")
    verification["checks"]["carry_forward_exact_eighteen_rows"] = True
    verification["checks"]["corrected_banglish_marker_gate"] = True
    verification["checks"]["English_mode_zero_Bengali"] = True
    verification["checks"]["terminal_normalization_exact_and_only_edit"] = True
    verification["attempt_ledger_file_sha256"] = file_sha256(ledger_path)
    verification["english_rows_verified"] = english_rows
    verification["terminal_rows_verified"] = terminal_rows
    verification["terminal_punctuation_replacements"] = replacement_count
    base.write_json(artifact / "evidence" / "verification-report.json", verification)
    return verification
