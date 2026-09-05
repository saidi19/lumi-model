"""Terminal-punctuation-normalized completion of the local Qwen teacher canary."""

from __future__ import annotations

import copy
import hashlib
import json
import time
import unicodedata
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import posttraining_first_party_teacher_canary as base
from . import posttraining_first_party_teacher_canary_v2 as retry_v2
from . import posttraining_first_party_teacher_canary_v4 as schema_v4
from . import posttraining_first_party_teacher_canary_v5 as corrected_v5
from . import posttraining_first_party_teacher_canary_v8 as suffix_v8
from .posttraining_first_party import file_sha256

TERMINAL_COMPLETION_CONTRACT = dict(schema_v4.SCHEMA_COMPLETION_CONTRACT)
PROMPT_CONTRACT = "lumi.first-party-teacher-prompt.v0.9"
TERMINAL_COMPOSITION_METHOD = "terminal_punctuation_replace_or_suffix_append_v1"
NATURAL_CODE_SWITCH_TEMPERATURE = 0.2
ALLOWED_TERMINAL_PUNCTUATION = frozenset({"?", "!", ".", "।"})
FORMAT_EXAMPLE = {
    "assistant_response": "এক বছরে 12 মাস থাকে।",
    "user_prompt": "এক বছরে কত মাস থাকে?",
}


def terminal_seed(record_id: str, attempt_number: int) -> int:
    if not 1 <= attempt_number <= TERMINAL_COMPLETION_CONTRACT["maximum_attempts_per_record"]:
        raise ValueError("terminal-completion attempt number is outside authorization")
    material = f"{record_id}:terminal-completion-attempt:{attempt_number}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def terminal_output_schema(scenario: base.Scenario) -> dict[str, Any]:
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


def terminal_prompt(scenario: base.Scenario, language_mode: str) -> str:
    if language_mode != "natural_code_switching":
        return corrected_v5.corrected_prompt(scenario, language_mode)
    prompt = retry_v2.build_retry_prompt(scenario, language_mode)
    example = json.dumps(FORMAT_EXAMPLE, ensure_ascii=False, sort_keys=True)
    instruction = (
        "user_prompt ও assistant_response-এর value দুটি পূর্ণ অর্থবোধক, সংক্ষিপ্ত বাংলা-"
        "লিপির body হিসেবে লেখো। Value-তে কোনো Latin/English অক্ষর লিখবে না এবং শুরু বা "
        "শেষে whitespace রাখবে না। Body-তে internal punctuation লিখবে না; চাইলে শেষে শুধু "
        "একটি ?, !, . অথবা । লিখতে পারো। সংখ্যা ও গাণিতিক symbol লেখা যাবে। Scenario-তে "
        "required ASCII literal থাকলেও value-তে লিখবে না; runtime exact bound suffix-এর "
        "মধ্যে সেগুলো যোগ করবে। Runtime একমাত্র terminal punctuation থাকলে সেটি replace করবে, "
        "না থাকলে append করবে: user body-র শেষে একটি English parenthetical ও question mark, "
        "assistant body-র শেষে একই parenthetical ও Bengali danda। কোনো placeholder, extra "
        f"field বা meta-commentary লিখবে না। Format example: {example}. Example-এর বিষয় copy "
        "করবে না; বর্তমান scenario-র সঠিক উত্তর দেবে।"
    )
    prompt = prompt.replace(base.LANGUAGE_INSTRUCTIONS[language_mode], instruction)
    if instruction not in prompt:
        raise AssertionError("terminal-normalization instruction was not installed")
    return prompt


def terminal_payload(
    record_id: str,
    scenario: base.Scenario,
    language_mode: str,
    *,
    attempt_number: int,
) -> dict[str, Any]:
    payload = base.ollama_payload(record_id, scenario, language_mode)
    payload["prompt"] = terminal_prompt(scenario, language_mode)
    payload["options"]["seed"] = terminal_seed(record_id, attempt_number)
    if language_mode == "natural_code_switching":
        payload["format"] = terminal_output_schema(scenario)
        payload["options"]["temperature"] = NATURAL_CODE_SWITCH_TEMPERATURE
    return payload


def normalize_terminal_bengali_body(text: str, *, field: str) -> tuple[str, str | None]:
    """Remove at most one allowed terminal punctuation mark and nothing else."""

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

    punctuation = [
        (index, character)
        for index, character in enumerate(text)
        if unicodedata.category(character).startswith("P")
    ]
    if not punctuation:
        return text, None
    if (
        len(punctuation) != 1
        or punctuation[0][0] != len(text) - 1
        or punctuation[0][1] not in ALLOWED_TERMINAL_PUNCTUATION
    ):
        raise ValueError(f"terminal teacher {field} contains unsupported pre-suffix punctuation")
    body = text[:-1]
    if len(body) < 2 or base._BENGALI.search(body) is None:  # noqa: SLF001
        raise ValueError(f"terminal teacher {field} body is invalid after normalization")
    return body, punctuation[0][1]


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
    composition: Mapping[str, Any] | None,
) -> dict[str, Any]:
    composed = language_mode == "natural_code_switching"
    term = schema_v4.CODE_SWITCH_TERMS[scenario.scenario_id]
    return {
        "schema_version": base.ROW_SCHEMA_VERSION,
        "record_id": record_id,
        "source_id": "lumi_first_party_multilingual_safety_tool_v0_9",
        "task": scenario.task,
        "skill_id": scenario.scenario_id,
        "language_mode": language_mode,
        "split": "validation" if scenario.variant == 3 else "train",
        "messages": messages,
        "provenance": {
            "ai_assisted": True,
            "banglish_marker_contract": corrected_v5.BANGLISH_MARKER_CONTRACT
            if language_mode == "banglish"
            else None,
            "deterministic_terminal_normalization": composed,
            "english_schema_constant": term if composed else None,
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
            "seed": terminal_seed(record_id, attempt_number),
            "self_hosted": True,
            "terminal_composition_method": TERMINAL_COMPOSITION_METHOD if composed else None,
            "terminal_source_internal_sha256": base.sha256_value(composition)
            if composition is not None
            else None,
        },
        "rights_status": "private_research_candidate_pending_owner_review",
    }


def generate_terminal_attempt(
    record_id: str,
    scenario: base.Scenario,
    language_mode: str,
    *,
    attempt_number: int,
    endpoint: str,
    rejection_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = terminal_payload(
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
            _, messages, composition = compose_terminal_output(output, scenario=scenario)
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
                "schema_version": "lumi.sft-first-party-teacher-rejected-output.v0.9",
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


def _validate_terminal_plan(plan: dict[str, Any], *, artifact: Path) -> None:
    if plan.get("schema_completion") != TERMINAL_COMPLETION_CONTRACT:
        raise ValueError("terminal-completion contract changed")
    carry = plan.get("carry_forward")
    if not isinstance(carry, dict):
        raise ValueError("terminal-completion carry-forward binding is missing")
    family_root = artifact.parent
    expected = {
        "candidate_rows_path": str(
            family_root / "attempt-003" / "private" / "candidate-rows.partial.jsonl"
        ),
        "candidate_rows_sha256": "108b126c1a32fe281637b9e7ccaa2b12189a688d6d8056d612737ec44cc21095",
        "failure_report_path": str(
            family_root / "attempt-008" / "evidence" / "failure-report.json"
        ),
        "failure_report_sha256": "081f4de405a85f2182d9ecdd56fc6b55983e3ebbb43ed3aabbf0d76923727681",
        "failure_verification_path": str(
            family_root / "attempt-008" / "evidence" / "failure-verification.json"
        ),
        "failure_verification_sha256": (
            "bf6a691880252a0d00c7229151d1f6d620ea0dd2d463e5e8d9d759ea4647a9a4"
        ),
        "records": 3,
        "telemetry_path": str(
            family_root / "attempt-003" / "private" / "generation-telemetry.partial.jsonl"
        ),
        "telemetry_sha256": "cf3deb7325d64b756082830d9135a8633ee0c79d64c0436189fff57c0b35f757",
    }
    if carry != expected:
        raise ValueError("terminal-completion carry-forward contract changed")
    for field in ("candidate_rows", "failure_report", "failure_verification", "telemetry"):
        if file_sha256(carry[f"{field}_path"]) != carry[f"{field}_sha256"]:
            raise ValueError(f"terminal-completion carry-forward changed: {field}")


@contextmanager
def _patched_v4_runtime() -> Iterator[None]:
    original_validate = schema_v4._validate_schema_completion_plan
    original_generate = schema_v4.generate_schema_completion_attempt
    schema_v4._validate_schema_completion_plan = _validate_terminal_plan
    schema_v4.generate_schema_completion_attempt = generate_terminal_attempt
    try:
        yield
    finally:
        schema_v4._validate_schema_completion_plan = original_validate
        schema_v4.generate_schema_completion_attempt = original_generate


def run_terminal_completion(
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
    compositions = [
        item["terminal_composition"]
        for item in telemetry[3:]
        if item.get("terminal_composition") is not None
    ]
    replacement_count = sum(
        composition.get(f"{field}_removed_terminal") is not None
        for composition in compositions
        for field in ("user", "assistant")
    )
    report["schema_completion"]["composition_method"] = TERMINAL_COMPOSITION_METHOD
    report["schema_completion"]["deterministic_terminal_compositions"] = len(compositions)
    report["schema_completion"]["terminal_punctuation_replacements"] = replacement_count
    report["schema_completion"]["pre_normalization_text_retained_privately"] = True
    report["generation"]["natural_code_switching_temperature"] = NATURAL_CODE_SWITCH_TEMPERATURE
    base.write_json(Path(artifact_root) / "evidence" / "run-report.json", report)
    return report


def _verify_terminal_composition(
    row: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    *,
    scenario: base.Scenario,
) -> int:
    term = schema_v4.CODE_SWITCH_TERMS[scenario.scenario_id]
    provenance = row.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("terminal-completion provenance is missing")
    expected = {
        "deterministic_terminal_normalization": True,
        "english_schema_constant": term,
        "freeform_manual_rewriting": False,
        "human_authored": False,
        "prompt_contract": PROMPT_CONTRACT,
        "terminal_composition_method": TERMINAL_COMPOSITION_METHOD,
    }
    for field, value in expected.items():
        if provenance.get(field) != value:
            raise ValueError(f"terminal-completion provenance changed: {field}")
    composition = telemetry.get("terminal_composition")
    if not isinstance(composition, Mapping):
        raise ValueError("terminal-completion private composition evidence is missing")
    if provenance.get("terminal_source_internal_sha256") != base.sha256_value(composition):
        raise ValueError("terminal-completion evidence binding changed")
    if (
        composition.get("english_schema_constant") != term
        or composition.get("method") != TERMINAL_COMPOSITION_METHOD
        or composition.get("suffixes_per_field") != 1
    ):
        raise ValueError("terminal-completion constants changed")

    natural = {
        message["role"]: str(message["content"])
        for message in row.get("messages", [])
        if message.get("role") in {"user", "assistant"}
    }
    expected_suffixes = {"assistant": f" ({term})।", "user": f" ({term})?"}
    replacements = 0
    for field, role in (("user", "user"), ("assistant", "assistant")):
        before = str(composition[f"{field}_before"])
        normalized, removed = normalize_terminal_bengali_body(before, field=field)
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
            raise ValueError("terminal-completion changed text beyond terminal normalization")
        replacements += removed is not None
    return replacements


def verify_terminal_completion(
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
    _validate_terminal_plan(plan, artifact=artifact)
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
    if rows[:3] != carry_rows or len(rows) != 80 or len(telemetry) != 80:
        raise ValueError("terminal-completion carry-forward or final counts changed")
    if not 77 <= len(ledger) <= TERMINAL_COMPLETION_CONTRACT["additional_inference_attempts"]:
        raise ValueError("terminal-completion attempt count is invalid")
    if sum(item.get("status") == "accepted" for item in ledger) != 77:
        raise ValueError("terminal-completion accepted-attempt count changed")
    if base.sha256_value(ledger) != report.get("schema_completion", {}).get(
        "attempt_ledger_internal_sha256"
    ):
        raise ValueError("terminal-completion attempt ledger changed")
    if file_sha256(ledger_path) != report.get("attempt_ledger_file_sha256"):
        raise ValueError("terminal-completion attempt ledger file changed")

    scenarios = {scenario.scenario_id: scenario for scenario in base.SCENARIOS}
    terminal_rows = 0
    replacement_count = 0
    for row, item_telemetry in zip(rows[3:], telemetry[3:], strict=True):
        provenance = row.get("provenance", {})
        if provenance.get("prompt_contract") != PROMPT_CONTRACT:
            raise ValueError("terminal-completion prompt contract changed")
        if row.get("language_mode") == "banglish":
            suffix_v8._verify_banglish_row(row)  # noqa: SLF001
        elif row.get("language_mode") == "natural_code_switching":
            scenario = scenarios[str(row["skill_id"])]
            replacement_count += _verify_terminal_composition(
                row,
                item_telemetry,
                scenario=scenario,
            )
            terminal_rows += 1
    if terminal_rows != 20:
        raise ValueError("terminal-completion code-switch row count changed")
    if replacement_count != report.get("schema_completion", {}).get(
        "terminal_punctuation_replacements"
    ):
        raise ValueError("terminal-completion replacement count changed")
    verification["checks"]["carry_forward_original_three_rows"] = True
    verification["checks"]["corrected_banglish_marker_gate"] = True
    verification["checks"]["private_pre_normalization_evidence_bound"] = True
    verification["checks"]["terminal_normalization_exact_and_only_edit"] = True
    verification["attempt_ledger_file_sha256"] = file_sha256(ledger_path)
    verification["terminal_punctuation_replacements"] = replacement_count
    base.write_json(artifact / "evidence" / "verification-report.json", verification)
    return verification
