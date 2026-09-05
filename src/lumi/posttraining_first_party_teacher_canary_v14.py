"""Internal-punctuation-safe continuation of the local Qwen teacher canary."""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import posttraining_first_party_teacher_canary as base
from . import posttraining_first_party_teacher_canary_v4 as schema_v4
from . import posttraining_first_party_teacher_canary_v9 as terminal_v9
from . import posttraining_first_party_teacher_canary_v13 as safety_v13
from .posttraining_first_party import file_sha256

CONTINUATION_CONTRACT = {
    "additional_inference_attempts": 123,
    "maximum_attempts_per_record": 3,
    "maximum_network_bytes": 0,
    "maximum_retained_new_records": 41,
    "maximum_retained_supervised_tokens_total": 24_000,
    "maximum_runtime_minutes": 120,
    "total_retained_records": 80,
}
CARRY_RECORDS = 39
NEW_RECORDS = 41
NEW_NATURAL_CODE_SWITCHING_ROWS = 11
NEW_LANGUAGE_BOUND_SAFETY_ROWS = 9
PROMPT_CONTRACT = "lumi.first-party-teacher-prompt.v0.14"
TERMINAL_COMPOSITION_METHOD = "preserve_internal_punctuation_replace_one_terminal_v2"
TERMINAL_PROMPT_METHOD = "allow_internal_punctuation_one_terminal_suffix_v2"

_V13_CONTINUATION_PAYLOAD = safety_v13.continuation_payload
_V13_CONTINUATION_PROMPT = safety_v13.continuation_prompt
_V13_GENERATE = safety_v13.generate_continuation_attempt
_V13_MAKE_ROW = safety_v13._make_row  # noqa: SLF001

_OLD_TERMINAL_INSTRUCTION = (
    "Body-তে internal punctuation লিখবে না; চাইলে শেষে শুধু একটি ?, !, . অথবা । লিখতে পারো।"
)
_NEW_TERMINAL_INSTRUCTION = (
    "Body-তে স্বাভাবিক internal punctuation লিখতে পারো। শেষে চাইলে শুধু একটি ?, !, . "
    "অথবা । লিখতে পারো; runtime internal punctuation byte-exact রেখে কেবল ওই optional "
    "শেষের mark replace করবে।"
)


def continuation_prompt(scenario: base.Scenario, language_mode: str) -> str:
    """Keep v0.13 prompts except for the contradictory code-switch punctuation rule."""

    prompt = _V13_CONTINUATION_PROMPT(scenario, language_mode)
    if language_mode != "natural_code_switching":
        return prompt
    if prompt.count(_OLD_TERMINAL_INSTRUCTION) != 1:
        raise AssertionError("natural-code-switch terminal instruction changed")
    corrected = prompt.replace(
        _OLD_TERMINAL_INSTRUCTION,
        _NEW_TERMINAL_INSTRUCTION,
        1,
    )
    if _NEW_TERMINAL_INSTRUCTION not in corrected:
        raise AssertionError("internal-punctuation correction was not installed")
    return corrected


def continuation_payload(
    record_id: str,
    scenario: base.Scenario,
    language_mode: str,
    *,
    attempt_number: int,
) -> dict[str, Any]:
    payload = _V13_CONTINUATION_PAYLOAD(
        record_id,
        scenario,
        language_mode,
        attempt_number=attempt_number,
    )
    payload["prompt"] = continuation_prompt(scenario, language_mode)
    return payload


def normalize_terminal_bengali_body(
    text: str,
    *,
    field: str,
) -> tuple[str, str | None]:
    """Preserve internal punctuation and remove at most one supported final mark."""

    if text != text.strip():
        raise ValueError(f"terminal teacher {field} has surrounding whitespace")
    if len(text) < 2:
        raise ValueError(f"terminal teacher {field} is too short")
    if any(marker in text for marker in ("\n", "\r", "\t")):
        raise ValueError(f"terminal teacher {field} contains control whitespace")
    if base._BENGALI.search(text) is None:  # noqa: SLF001
        raise ValueError(f"terminal teacher {field} lacks Bengali script")
    if base._LATIN.search(text) is not None:  # noqa: SLF001
        raise ValueError(f"terminal teacher {field} contains pre-suffix Latin text")

    final = text[-1]
    if not unicodedata.category(final).startswith("P"):
        return text, None
    if final not in terminal_v9.ALLOWED_TERMINAL_PUNCTUATION:
        raise ValueError(f"terminal teacher {field} has unsupported final punctuation")
    body = text[:-1]
    if (
        len(body) < 2
        or base._BENGALI.search(body) is None  # noqa: SLF001
        or body[-1].isspace()
        or unicodedata.category(body[-1]).startswith("P")
    ):
        raise ValueError(f"terminal teacher {field} has multiple trailing punctuation marks")
    return body, final


def compose_terminal_output(
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
        raise ValueError("terminal teacher JSON envelope fields changed")

    term = schema_v4.CODE_SWITCH_TERMS[scenario.scenario_id]
    suffixes = {
        "assistant_response": f" ({term})।",
        "user_prompt": f" ({term})?",
    }
    originals: dict[str, str] = {}
    normalized: dict[str, str] = {}
    removed: dict[str, str | None] = {}
    replacements: dict[str, str] = {}
    for field in ("user_prompt", "assistant_response"):
        text = output.get(field)
        if not isinstance(text, str):
            raise ValueError(f"terminal teacher {field} is not a string")
        body, terminal = normalize_terminal_bengali_body(text, field=field)
        originals[field] = text
        normalized[field] = body
        removed[field] = terminal
        replacements[field] = f"{body}{suffixes[field]}"

    base._validate_safety(normalized["assistant_response"], scenario.safety_label)  # noqa: SLF001
    flattened = dict(output)
    flattened.update(replacements)
    messages = base._validate_output(  # noqa: SLF001
        flattened,
        scenario=scenario,
        language_mode="natural_code_switching",
    )
    composition = {
        "assistant_action": "replace_terminal_punctuation"
        if removed["assistant_response"] is not None
        else "append_suffix",
        "assistant_after": replacements["assistant_response"],
        "assistant_before": originals["assistant_response"],
        "assistant_normalized_body": normalized["assistant_response"],
        "assistant_removed_terminal": removed["assistant_response"],
        "assistant_suffix": suffixes["assistant_response"],
        "english_schema_constant": term,
        "internal_punctuation_policy": "preserve_byte_exact",
        "method": TERMINAL_COMPOSITION_METHOD,
        "suffixes_per_field": 1,
        "user_action": "replace_terminal_punctuation"
        if removed["user_prompt"] is not None
        else "append_suffix",
        "user_after": replacements["user_prompt"],
        "user_before": originals["user_prompt"],
        "user_normalized_body": normalized["user_prompt"],
        "user_removed_terminal": removed["user_prompt"],
        "user_suffix": suffixes["user_prompt"],
    }
    return flattened, messages, composition


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
    row = _V13_MAKE_ROW(
        record_id,
        scenario,
        language_mode,
        messages,
        attempt_number=attempt_number,
        terminal_composition=terminal_composition,
        banglish_composition=banglish_composition,
    )
    row["source_id"] = "lumi_first_party_multilingual_safety_tool_v0_14"
    provenance = row["provenance"]
    provenance["prompt_contract"] = PROMPT_CONTRACT
    provenance["code_switch_terminal_prompt_method"] = (
        TERMINAL_PROMPT_METHOD if language_mode == "natural_code_switching" else None
    )
    provenance["internal_punctuation_policy"] = (
        "preserve_byte_exact" if language_mode == "natural_code_switching" else None
    )
    if language_mode == "natural_code_switching":
        if terminal_composition is None:
            raise ValueError("code-switch terminal composition is missing")
        provenance["terminal_composition_method"] = TERMINAL_COMPOSITION_METHOD
        provenance["terminal_source_internal_sha256"] = base.sha256_value(terminal_composition)
    return row


@contextmanager
def _patched_v13_generation() -> Iterator[None]:
    original_payload = safety_v13.continuation_payload
    original_make_row = safety_v13._make_row  # noqa: SLF001
    original_compose = terminal_v9.compose_terminal_output
    original_method = terminal_v9.TERMINAL_COMPOSITION_METHOD
    safety_v13.continuation_payload = continuation_payload
    safety_v13._make_row = _make_row  # noqa: SLF001
    terminal_v9.compose_terminal_output = compose_terminal_output
    terminal_v9.TERMINAL_COMPOSITION_METHOD = TERMINAL_COMPOSITION_METHOD
    try:
        yield
    finally:
        safety_v13.continuation_payload = original_payload
        safety_v13._make_row = original_make_row  # noqa: SLF001
        terminal_v9.compose_terminal_output = original_compose
        terminal_v9.TERMINAL_COMPOSITION_METHOD = original_method


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
        with _patched_v13_generation():
            return _V13_GENERATE(
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
            rejection["schema_version"] = "lumi.sft-first-party-teacher-rejected-output.v0.14"
            base.write_json(rejection_path, rejection)
        raise


def _validate_continuation_plan(plan: dict[str, Any], *, artifact: Path) -> None:
    if plan.get("schema_completion") != CONTINUATION_CONTRACT:
        raise ValueError("internal-punctuation continuation contract changed")
    carry = plan.get("carry_forward")
    if not isinstance(carry, dict):
        raise ValueError("internal-punctuation carry-forward binding is missing")
    family_root = artifact.parent
    expected = {
        "candidate_rows_path": str(
            family_root / "attempt-013" / "private" / "candidate-rows.partial.jsonl"
        ),
        "candidate_rows_sha256": (
            "7e10afe87db6e9c15fc3b266e80a4f2b72cd937c94c8a0b9cfb87723770a1bc0"
        ),
        "failure_report_path": str(
            family_root / "attempt-013" / "evidence" / "failure-report.json"
        ),
        "failure_report_sha256": (
            "688a72791b994629f03b8060f0edd8028698f927d41b6cb9b177089f01f2c5b3"
        ),
        "failure_verification_path": str(
            family_root / "attempt-013" / "evidence" / "failure-verification.json"
        ),
        "failure_verification_sha256": (
            "244eededaafa79c084d3aa9c44efea7819d59419fe1430c2ef3dccaa454ab471"
        ),
        "records": CARRY_RECORDS,
        "telemetry_path": str(
            family_root / "attempt-013" / "private" / "generation-telemetry.partial.jsonl"
        ),
        "telemetry_sha256": ("b817e643dca601a0115838f416debdfea9bc8e548774d0fc3c0c128d3600c0fe"),
    }
    if carry != expected:
        raise ValueError("internal-punctuation carry-forward contract changed")
    for field in ("candidate_rows", "failure_report", "failure_verification", "telemetry"):
        if file_sha256(carry[f"{field}_path"]) != carry[f"{field}_sha256"]:
            raise ValueError(f"internal-punctuation carry-forward changed: {field}")


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
        raise ValueError("internal-punctuation partial row/telemetry pair is incomplete")
    carry_rows = base._read_jsonl(Path(plan["carry_forward"]["candidate_rows_path"]))  # noqa: SLF001
    carry_telemetry = base._read_jsonl(  # noqa: SLF001
        Path(plan["carry_forward"]["telemetry_path"])
    )
    if len(carry_rows) != CARRY_RECORDS or len(carry_telemetry) != CARRY_RECORDS:
        raise ValueError("internal-punctuation carry-forward count changed")
    if not any(present):
        base.write_jsonl(partial_rows, carry_rows)
        base.write_jsonl(partial_telemetry, carry_telemetry)
    rows = base._read_jsonl(partial_rows)  # noqa: SLF001
    telemetry = base._read_jsonl(partial_telemetry)  # noqa: SLF001
    ledger = base._read_jsonl(ledger_path)  # noqa: SLF001
    if rows[:CARRY_RECORDS] != carry_rows or telemetry[:CARRY_RECORDS] != carry_telemetry:
        raise ValueError("internal-punctuation exact carry-forward changed")
    expected_ids = base.expected_record_ids()
    if [str(row.get("record_id")) for row in rows] != list(expected_ids[: len(rows)]):
        raise ValueError("internal-punctuation rows are not an exact expected prefix")
    if len(rows) != len(telemetry) or not CARRY_RECORDS <= len(rows) <= 80:
        raise ValueError("internal-punctuation row and telemetry counts differ")
    if len(ledger) > CONTINUATION_CONTRACT["additional_inference_attempts"]:
        raise ValueError("internal-punctuation attempt ledger exceeds authorization")
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


def run_internal_punctuation_continuation(
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
    report["schema_completion"]["internal_punctuation_policy"] = "preserve_byte_exact"
    report["schema_completion"]["language_bound_safety_method"] = safety_v13.SAFETY_PROMPT_METHOD
    report["schema_completion"]["language_bound_safety_rows"] = sum(
        row.get("provenance", {}).get("language_bound_safety_method")
        == safety_v13.SAFETY_PROMPT_METHOD
        for row in new_rows
    )
    report["schema_completion"]["natural_code_switching_rows"] = sum(
        row.get("language_mode") == "natural_code_switching" for row in new_rows
    )
    report["schema_completion"]["terminal_composition_method"] = TERMINAL_COMPOSITION_METHOD
    base.write_json(Path(artifact_root) / "evidence" / "run-report.json", report)
    return report


def _verify_terminal_composition(
    row: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    *,
    scenario: base.Scenario,
) -> int:
    provenance = row.get("provenance", {})
    expected_provenance = {
        "code_switch_terminal_prompt_method": TERMINAL_PROMPT_METHOD,
        "deterministic_terminal_normalization": True,
        "english_schema_constant": schema_v4.CODE_SWITCH_TERMS[scenario.scenario_id],
        "internal_punctuation_policy": "preserve_byte_exact",
        "prompt_contract": PROMPT_CONTRACT,
        "terminal_composition_method": TERMINAL_COMPOSITION_METHOD,
    }
    for field, expected in expected_provenance.items():
        if provenance.get(field) != expected:
            raise ValueError(f"internal-punctuation provenance changed: {field}")
    composition = telemetry.get("terminal_composition")
    if not isinstance(composition, dict):
        raise ValueError("internal-punctuation composition evidence is missing")
    if provenance.get("terminal_source_internal_sha256") != base.sha256_value(composition):
        raise ValueError("internal-punctuation evidence binding changed")
    term = schema_v4.CODE_SWITCH_TERMS[scenario.scenario_id]
    suffixes = {
        "assistant_response": f" ({term})।",
        "user_prompt": f" ({term})?",
    }
    replacements = 0
    for field in ("user_prompt", "assistant_response"):
        before = composition.get(f"{field}_before")
        after = composition.get(f"{field}_after")
        if not isinstance(before, str) or not isinstance(after, str):
            raise ValueError("internal-punctuation before/after evidence changed")
        normalized, removed = normalize_terminal_bengali_body(before, field=field)
        expected_action = "replace_terminal_punctuation" if removed else "append_suffix"
        if (
            composition.get(f"{field}_normalized_body") != normalized
            or composition.get(f"{field}_removed_terminal") != removed
            or composition.get(f"{field}_suffix") != suffixes[field]
            or composition.get(f"{field}_action") != expected_action
            or after != f"{normalized}{suffixes[field]}"
        ):
            raise ValueError("internal-punctuation deterministic composition changed")
        message_role = "assistant" if field == "assistant_response" else "user"
        final = next(
            str(message["content"])
            for message in row.get("messages", [])
            if message.get("role") == message_role
        )
        if final != after:
            raise ValueError("internal-punctuation final field hash changed")
        replacements += removed is not None
    if composition.get("method") != TERMINAL_COMPOSITION_METHOD:
        raise ValueError("internal-punctuation method changed")
    if composition.get("internal_punctuation_policy") != "preserve_byte_exact":
        raise ValueError("internal-punctuation policy changed")
    return replacements


def verify_internal_punctuation_continuation(
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
        raise ValueError("internal-punctuation carry-forward or final counts changed")
    if not NEW_RECORDS <= len(ledger) <= CONTINUATION_CONTRACT["additional_inference_attempts"]:
        raise ValueError("internal-punctuation attempt count is invalid")
    if sum(item.get("status") == "accepted" for item in ledger) != NEW_RECORDS:
        raise ValueError("internal-punctuation accepted-attempt count changed")
    if base.sha256_value(ledger) != report.get("schema_completion", {}).get(
        "attempt_ledger_internal_sha256"
    ):
        raise ValueError("internal-punctuation attempt ledger changed")
    if file_sha256(ledger_path) != report.get("attempt_ledger_file_sha256"):
        raise ValueError("internal-punctuation attempt ledger file changed")

    scenarios = {scenario.scenario_id: scenario for scenario in base.SCENARIOS}
    coverage: Counter[tuple[str, str]] = Counter()
    new_targeted = 0
    new_code_switch = 0
    terminal_replacements = 0
    for index, (row, item_telemetry) in enumerate(zip(rows, telemetry, strict=True)):
        scenario = scenarios[str(row["skill_id"])]
        language_mode = str(row["language_mode"])
        provenance = row.get("provenance", {})
        if index >= CARRY_RECORDS:
            if provenance.get("prompt_contract") != PROMPT_CONTRACT:
                raise ValueError("internal-punctuation prompt contract changed")
            targeted = scenario.safety_label != "safe"
            expected_method = safety_v13.SAFETY_PROMPT_METHOD if targeted else None
            if provenance.get("language_bound_safety_method") != expected_method:
                raise ValueError("language-bound safety prompt provenance changed")
            new_targeted += targeted
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
            new_code_switch += 1
            terminal_replacements += _verify_terminal_composition(
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
    if coverage != expected_coverage or new_targeted != NEW_LANGUAGE_BOUND_SAFETY_ROWS:
        raise ValueError("language-bound safety coverage changed")
    if new_code_switch != NEW_NATURAL_CODE_SWITCHING_ROWS:
        raise ValueError("internal-punctuation code-switch row count changed")
    schema_report = report.get("schema_completion", {})
    if report.get("carry_forward", {}).get("records") != CARRY_RECORDS:
        raise ValueError("internal-punctuation report carry count changed")
    if schema_report.get("language_bound_safety_rows") != NEW_LANGUAGE_BOUND_SAFETY_ROWS:
        raise ValueError("language-bound safety report target count changed")
    if schema_report.get("natural_code_switching_rows") != NEW_NATURAL_CODE_SWITCHING_ROWS:
        raise ValueError("internal-punctuation report code-switch count changed")
    if schema_report.get("terminal_composition_method") != TERMINAL_COMPOSITION_METHOD:
        raise ValueError("internal-punctuation report method changed")

    verification["checks"]["carry_forward_exact_thirty_nine_rows"] = True
    verification["checks"]["all_twelve_non_safe_rows_language_bound"] = True
    verification["checks"]["all_eight_refusal_rows_exact"] = True
    verification["checks"]["all_four_medical_rows_exact"] = True
    verification["checks"]["preaccept_exact_safety_enforcement"] = True
    verification["checks"]["internal_punctuation_preserved_byte_exact"] = True
    verification["checks"]["only_one_optional_terminal_mark_replaced"] = True
    verification["attempt_ledger_file_sha256"] = file_sha256(ledger_path)
    verification["language_bound_safety_rows_verified"] = sum(coverage.values())
    verification["natural_code_switching_rows_verified"] = new_code_switch
    verification["terminal_punctuation_replacements"] = terminal_replacements
    base.write_json(artifact / "evidence" / "verification-report.json", verification)
    return verification


def validate_recorded_attempt_013_rejections(
    rejection_paths: list[str | Path],
) -> dict[str, Any]:
    """Prove the corrected composer accepts the three otherwise-valid recorded outputs."""

    scenario = next(item for item in base.SCENARIOS if item.scenario_id == "credential_privacy")
    accepted = 0
    internal_punctuation_cases = 0
    for path in rejection_paths:
        rejection = base._json_object(path)  # noqa: SLF001
        response = rejection.get("response")
        if not isinstance(response, dict):
            raise ValueError("attempt-013 rejection response envelope is missing")
        raw = response.get("response")
        if not isinstance(raw, str):
            raise ValueError("attempt-013 rejection raw response is missing")
        output = json.loads(raw)
        if not isinstance(output, dict):
            raise ValueError("attempt-013 rejection output is not an object")
        with safety_v13._patched_v11_generation(  # noqa: SLF001
            language_mode="natural_code_switching"
        ):
            flattened, messages, composition = compose_terminal_output(
                output,
                scenario=scenario,
            )
        if not flattened or not messages or not composition:
            raise ValueError("attempt-013 rejection correction produced empty output")
        safety_v13.validate_language_bound_safety(
            composition["assistant_normalized_body"],
            "refuse",
            language_mode="natural_code_switching",
        )
        internal_punctuation_cases += any(
            unicodedata.category(character).startswith("P")
            for character in composition["assistant_normalized_body"]
        )
        accepted += 1
    return {
        "accepted": accepted,
        "internal_punctuation_cases": internal_punctuation_cases,
        "rejections_checked": len(rejection_paths),
    }
