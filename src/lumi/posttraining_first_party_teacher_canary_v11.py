"""Banglish-terminal-normalized continuation of the local Qwen teacher canary."""

from __future__ import annotations

import json
import time
import unicodedata
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import posttraining_first_party_teacher_canary as base
from . import posttraining_first_party_teacher_canary_v4 as schema_v4
from . import posttraining_first_party_teacher_canary_v5 as corrected_v5
from . import posttraining_first_party_teacher_canary_v8 as suffix_v8
from . import posttraining_first_party_teacher_canary_v9 as terminal_v9
from . import posttraining_first_party_teacher_canary_v10 as english_v10
from .posttraining_first_party import file_sha256

CONTINUATION_CONTRACT = {
    "additional_inference_attempts": 180,
    "maximum_attempts_per_record": 3,
    "maximum_network_bytes": 0,
    "maximum_retained_new_records": 60,
    "maximum_retained_supervised_tokens_total": 24_000,
    "maximum_runtime_minutes": 120,
    "total_retained_records": 80,
}
PROMPT_CONTRACT = "lumi.first-party-teacher-prompt.v0.11"
BANGLISH_TERMINAL_METHOD = "append_missing_terminal_punctuation_v1"
ALLOWED_TERMINAL_PUNCTUATION = frozenset({"?", "!", ".", "।"})
FIELD_SUFFIXES = {"assistant_response": ".", "user_prompt": "?"}


def continuation_prompt(scenario: base.Scenario, language_mode: str) -> str:
    """Preserve the exact v0.10 prompt for every language mode."""

    return english_v10.continuation_prompt(scenario, language_mode)


def continuation_payload(
    record_id: str,
    scenario: base.Scenario,
    language_mode: str,
    *,
    attempt_number: int,
) -> dict[str, Any]:
    """Preserve the exact v0.10 request and deterministic seed contract."""

    return english_v10.continuation_payload(
        record_id,
        scenario,
        language_mode,
        attempt_number=attempt_number,
    )


def normalize_banglish_terminal(text: str, *, field: str) -> tuple[str, str | None]:
    """Append one bound terminal only when a Banglish field has none."""

    if field not in FIELD_SUFFIXES:
        raise ValueError("unsupported Banglish terminal field")
    if text != text.strip():
        raise ValueError(f"Banglish terminal {field} has surrounding whitespace")
    if len(text) < 2:
        raise ValueError(f"Banglish terminal {field} is too short")
    if any(marker in text for marker in ("\n", "\r", "\t")):
        raise ValueError(f"Banglish terminal {field} contains control whitespace")
    if text[-1] in ALLOWED_TERMINAL_PUNCTUATION:
        return text, None
    if unicodedata.category(text[-1]).startswith("P"):
        raise ValueError(f"Banglish terminal {field} ends with unsupported punctuation")
    suffix = FIELD_SUFFIXES[field]
    return f"{text}{suffix}", suffix


def compose_banglish_terminal_output(
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
        raise ValueError("Banglish terminal teacher JSON envelope fields changed")

    normalized = dict(output)
    composition: dict[str, Any] = {
        "edits_per_field_maximum": 1,
        "method": BANGLISH_TERMINAL_METHOD,
    }
    for field in ("user_prompt", "assistant_response"):
        original = output.get(field)
        if not isinstance(original, str):
            raise ValueError(f"Banglish terminal {field} is not a string")
        after, appended = normalize_banglish_terminal(original, field=field)
        normalized[field] = after
        composition[f"{field}_action"] = (
            "append_missing_terminal" if appended is not None else "preserve_existing_terminal"
        )
        composition[f"{field}_appended_terminal"] = appended
        composition[f"{field}_before_sha256"] = base.sha256_value(original)
        composition[f"{field}_after_sha256"] = base.sha256_value(after)

    messages = corrected_v5.validate_corrected_output(
        normalized,
        scenario=scenario,
        language_mode="banglish",
    )
    return normalized, messages, composition


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
    row = english_v10._make_row(  # noqa: SLF001
        record_id,
        scenario,
        language_mode,
        messages,
        attempt_number=attempt_number,
        composition=terminal_composition,
    )
    row["source_id"] = "lumi_first_party_multilingual_safety_tool_v0_11"
    provenance = row["provenance"]
    provenance["prompt_contract"] = PROMPT_CONTRACT
    provenance["banglish_terminal_composition_method"] = (
        BANGLISH_TERMINAL_METHOD if language_mode == "banglish" else None
    )
    provenance["banglish_terminal_source_internal_sha256"] = (
        base.sha256_value(banglish_composition) if banglish_composition is not None else None
    )
    provenance["deterministic_banglish_terminal_normalization"] = language_mode == "banglish"
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
    terminal_composition: dict[str, Any] | None = None
    banglish_composition: dict[str, Any] | None = None
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
            _, messages, terminal_composition = terminal_v9.compose_terminal_output(
                output,
                scenario=scenario,
            )
        elif language_mode == "banglish":
            _, messages, banglish_composition = compose_banglish_terminal_output(
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
                "schema_version": "lumi.sft-first-party-teacher-rejected-output.v0.11",
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
        terminal_composition=terminal_composition,
        banglish_composition=banglish_composition,
    )
    telemetry = {
        "attempt_number": attempt_number,
        "banglish_marker_contract": corrected_v5.BANGLISH_MARKER_CONTRACT
        if language_mode == "banglish"
        else None,
        "banglish_terminal_composition": banglish_composition,
        "elapsed_seconds": round(elapsed, 6),
        "english_prompt_method": english_v10.ENGLISH_PROMPT_METHOD
        if language_mode == "english"
        else None,
        "eval_count": response.get("eval_count"),
        "eval_duration_ns": response.get("eval_duration"),
        "load_duration_ns": response.get("load_duration"),
        "prompt_eval_count": response.get("prompt_eval_count"),
        "prompt_eval_duration_ns": response.get("prompt_eval_duration"),
        "record_id": record_id,
        "request_sha256": base.sha256_value(payload),
        "response_envelope_sha256": base.sha256_value(response),
        "terminal_composition": terminal_composition,
    }
    return row, telemetry


def _validate_continuation_plan(plan: dict[str, Any], *, artifact: Path) -> None:
    if plan.get("schema_completion") != CONTINUATION_CONTRACT:
        raise ValueError("Banglish-terminal continuation contract changed")
    carry = plan.get("carry_forward")
    if not isinstance(carry, dict):
        raise ValueError("Banglish-terminal carry-forward binding is missing")
    family_root = artifact.parent
    expected = {
        "candidate_rows_path": str(
            family_root / "attempt-010" / "private" / "candidate-rows.partial.jsonl"
        ),
        "candidate_rows_sha256": (
            "3b725b7b334c3220960256246b82f0cad0f44b9f495ff964991a06c7c50dbeba"
        ),
        "failure_report_path": str(
            family_root / "attempt-010" / "evidence" / "failure-report.json"
        ),
        "failure_report_sha256": (
            "9ebf59f8f9e8b94b03f0ea28f78ab81e7ad4b153d2eb34a60795d1bd07e99912"
        ),
        "failure_verification_path": str(
            family_root / "attempt-010" / "evidence" / "failure-verification.json"
        ),
        "failure_verification_sha256": (
            "2d5baa0c9f7c4acef6b7b63f9627c204450d00c3c6f97d95e1afdcb9109f85b1"
        ),
        "records": 20,
        "telemetry_path": str(
            family_root / "attempt-010" / "private" / "generation-telemetry.partial.jsonl"
        ),
        "telemetry_sha256": ("5fc1c7b2ecc62c72ba057393ff8d79817debf1683789e52109b7aea73194b72e"),
    }
    if carry != expected:
        raise ValueError("Banglish-terminal carry-forward contract changed")
    for field in ("candidate_rows", "failure_report", "failure_verification", "telemetry"):
        if file_sha256(carry[f"{field}_path"]) != carry[f"{field}_sha256"]:
            raise ValueError(f"Banglish-terminal carry-forward changed: {field}")


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
        raise ValueError("Banglish-terminal partial row/telemetry pair is incomplete")
    carry_rows = base._read_jsonl(Path(plan["carry_forward"]["candidate_rows_path"]))  # noqa: SLF001
    carry_telemetry = base._read_jsonl(  # noqa: SLF001
        Path(plan["carry_forward"]["telemetry_path"])
    )
    if len(carry_rows) != 20 or len(carry_telemetry) != 20:
        raise ValueError("Banglish-terminal carry-forward count changed")
    if not any(present):
        base.write_jsonl(partial_rows, carry_rows)
        base.write_jsonl(partial_telemetry, carry_telemetry)
    rows = base._read_jsonl(partial_rows)  # noqa: SLF001
    telemetry = base._read_jsonl(partial_telemetry)  # noqa: SLF001
    ledger = base._read_jsonl(ledger_path)  # noqa: SLF001
    if rows[:20] != carry_rows or telemetry[:20] != carry_telemetry:
        raise ValueError("Banglish-terminal exact carry-forward changed")
    expected_ids = base.expected_record_ids()
    if [str(row.get("record_id")) for row in rows] != list(expected_ids[: len(rows)]):
        raise ValueError("Banglish-terminal rows are not an exact expected prefix")
    if len(rows) != len(telemetry) or not 20 <= len(rows) <= 80:
        raise ValueError("Banglish-terminal row and telemetry counts differ")
    if len(ledger) > CONTINUATION_CONTRACT["additional_inference_attempts"]:
        raise ValueError("Banglish-terminal attempt ledger exceeds authorization")
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


def run_banglish_terminal_continuation(
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
    new_telemetry = telemetry[20:]
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
    report["carry_forward"]["records"] = 20
    report["schema_completion"]["additional_records_retained"] = 60
    report["schema_completion"]["banglish_terminal_appends"] = banglish_appends
    report["schema_completion"]["banglish_terminal_method"] = BANGLISH_TERMINAL_METHOD
    report["schema_completion"]["english_prompt_method"] = english_v10.ENGLISH_PROMPT_METHOD
    report["schema_completion"]["terminal_punctuation_replacements"] = terminal_replacements
    report["generation"]["natural_code_switching_temperature"] = (
        terminal_v9.NATURAL_CODE_SWITCH_TEMPERATURE
    )
    base.write_json(Path(artifact_root) / "evidence" / "run-report.json", report)
    return report


def _verify_banglish_composition(
    row: Mapping[str, Any],
    telemetry: Mapping[str, Any],
) -> int:
    suffix_v8._verify_banglish_row(row)  # noqa: SLF001
    provenance = row.get("provenance")
    composition = telemetry.get("banglish_terminal_composition")
    if not isinstance(provenance, Mapping) or not isinstance(composition, Mapping):
        raise ValueError("Banglish terminal composition evidence is missing")
    if provenance.get("banglish_terminal_composition_method") != BANGLISH_TERMINAL_METHOD:
        raise ValueError("Banglish terminal provenance method changed")
    if provenance.get("deterministic_banglish_terminal_normalization") is not True:
        raise ValueError("Banglish terminal deterministic provenance changed")
    if provenance.get("banglish_terminal_source_internal_sha256") != base.sha256_value(composition):
        raise ValueError("Banglish terminal evidence binding changed")
    if (
        composition.get("method") != BANGLISH_TERMINAL_METHOD
        or composition.get("edits_per_field_maximum") != 1
    ):
        raise ValueError("Banglish terminal composition contract changed")

    natural = {
        message["role"]: str(message["content"])
        for message in row.get("messages", [])
        if message.get("role") in {"user", "assistant"}
    }
    appends = 0
    for field, role in (("user_prompt", "user"), ("assistant_response", "assistant")):
        after = natural.get(role)
        if after is None or base.sha256_value(after) != composition.get(f"{field}_after_sha256"):
            raise ValueError("Banglish terminal final field hash changed")
        appended = composition.get(f"{field}_appended_terminal")
        if appended is None:
            if (
                after[-1] not in ALLOWED_TERMINAL_PUNCTUATION
                or composition.get(f"{field}_action") != "preserve_existing_terminal"
                or composition.get(f"{field}_before_sha256") != base.sha256_value(after)
            ):
                raise ValueError("Banglish terminal preserved field changed")
            continue
        if appended != FIELD_SUFFIXES[field] or not after.endswith(appended):
            raise ValueError("Banglish terminal appended suffix changed")
        before = after[: -len(appended)]
        if (
            composition.get(f"{field}_action") != "append_missing_terminal"
            or composition.get(f"{field}_before_sha256") != base.sha256_value(before)
            or before[-1] in ALLOWED_TERMINAL_PUNCTUATION
        ):
            raise ValueError("Banglish terminal changed text beyond one exact append")
        appends += 1
    return appends


def _verify_terminal_composition(
    row: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    *,
    scenario: base.Scenario,
) -> int:
    provenance = row.get("provenance")
    composition = telemetry.get("terminal_composition")
    if not isinstance(provenance, Mapping) or not isinstance(composition, Mapping):
        raise ValueError("code-switch terminal composition evidence is missing")
    term = schema_v4.CODE_SWITCH_TERMS[scenario.scenario_id]
    expected = {
        "deterministic_terminal_normalization": True,
        "english_schema_constant": term,
        "freeform_manual_rewriting": False,
        "human_authored": False,
        "prompt_contract": PROMPT_CONTRACT,
        "terminal_composition_method": terminal_v9.TERMINAL_COMPOSITION_METHOD,
    }
    for field, value in expected.items():
        if provenance.get(field) != value:
            raise ValueError(f"code-switch terminal provenance changed: {field}")
    if provenance.get("terminal_source_internal_sha256") != base.sha256_value(composition):
        raise ValueError("code-switch terminal evidence binding changed")

    natural = {
        message["role"]: str(message["content"])
        for message in row.get("messages", [])
        if message.get("role") in {"user", "assistant"}
    }
    expected_suffixes = {"assistant": f" ({term})।", "user": f" ({term})?"}
    replacements = 0
    for field, role in (("user", "user"), ("assistant", "assistant")):
        before = str(composition[f"{field}_before"])
        normalized, removed = terminal_v9.normalize_terminal_bengali_body(before, field=field)
        suffix = expected_suffixes[field]
        after = str(composition[f"{field}_after"])
        expected_action = "replace_terminal_punctuation" if removed else "append_suffix"
        if (
            composition.get(f"{field}_normalized_body") != normalized
            or composition.get(f"{field}_removed_terminal") != removed
            or composition.get(f"{field}_action") != expected_action
            or composition.get(f"{field}_suffix") != suffix
            or after != f"{normalized}{suffix}"
            or natural.get(role) != after
            or after.count(suffix) != 1
        ):
            raise ValueError("code-switch text changed beyond terminal normalization")
        replacements += removed is not None
    return replacements


def verify_banglish_terminal_continuation(
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
    if rows[:20] != carry_rows or len(rows) != 80 or len(telemetry) != 80:
        raise ValueError("Banglish-terminal carry-forward or final counts changed")
    if not 60 <= len(ledger) <= CONTINUATION_CONTRACT["additional_inference_attempts"]:
        raise ValueError("Banglish-terminal attempt count is invalid")
    if sum(item.get("status") == "accepted" for item in ledger) != 60:
        raise ValueError("Banglish-terminal accepted-attempt count changed")
    if base.sha256_value(ledger) != report.get("schema_completion", {}).get(
        "attempt_ledger_internal_sha256"
    ):
        raise ValueError("Banglish-terminal attempt ledger changed")
    if file_sha256(ledger_path) != report.get("attempt_ledger_file_sha256"):
        raise ValueError("Banglish-terminal attempt ledger file changed")

    scenarios = {scenario.scenario_id: scenario for scenario in base.SCENARIOS}
    banglish_appends = 0
    banglish_rows = 0
    english_rows = 0
    terminal_replacements = 0
    terminal_rows = 0
    for row, item_telemetry in zip(rows[20:], telemetry[20:], strict=True):
        provenance = row.get("provenance", {})
        if provenance.get("prompt_contract") != PROMPT_CONTRACT:
            raise ValueError("Banglish-terminal prompt contract changed")
        language_mode = row.get("language_mode")
        if language_mode == "banglish":
            banglish_appends += _verify_banglish_composition(row, item_telemetry)
            banglish_rows += 1
        elif language_mode == "natural_code_switching":
            scenario = scenarios[str(row["skill_id"])]
            terminal_replacements += _verify_terminal_composition(
                row,
                item_telemetry,
                scenario=scenario,
            )
            terminal_rows += 1
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
    if report.get("carry_forward", {}).get("records") != 20:
        raise ValueError("Banglish-terminal report carry count changed")
    if banglish_appends != report.get("schema_completion", {}).get("banglish_terminal_appends"):
        raise ValueError("Banglish-terminal append count changed")
    if terminal_replacements != report.get("schema_completion", {}).get(
        "terminal_punctuation_replacements"
    ):
        raise ValueError("code-switch terminal replacement count changed")

    verification["checks"]["carry_forward_exact_twenty_rows"] = True
    verification["checks"]["Banglish_terminal_exact_append_only"] = True
    verification["checks"]["corrected_Banglish_marker_gate"] = True
    verification["checks"]["English_mode_zero_Bengali"] = True
    verification["checks"]["code_switch_terminal_normalization_exact_and_only_edit"] = True
    verification["attempt_ledger_file_sha256"] = file_sha256(ledger_path)
    verification["banglish_rows_verified"] = banglish_rows
    verification["banglish_terminal_appends"] = banglish_appends
    verification["english_rows_verified"] = english_rows
    verification["terminal_rows_verified"] = terminal_rows
    verification["terminal_punctuation_replacements"] = terminal_replacements
    base.write_json(artifact / "evidence" / "verification-report.json", verification)
    return verification
