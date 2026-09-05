"""Corrected result-preserving completion of the local Qwen teacher canary."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import posttraining_first_party_teacher_canary as base
from . import posttraining_first_party_teacher_canary_v2 as retry_v2
from . import posttraining_first_party_teacher_canary_v4 as schema_v4
from .posttraining_first_party import file_sha256

CORRECTED_COMPLETION_CONTRACT = dict(schema_v4.SCHEMA_COMPLETION_CONTRACT)
PROMPT_CONTRACT = "lumi.first-party-teacher-prompt.v0.5"
BANGLISH_MARKER_CONTRACT = "curated-romanized-bengali-markers.v0.2"

# Every added marker is a common romanized Bengali word or inflection. English task
# words such as day, week, summary, or calculate deliberately remain ineligible.
ADDITIONAL_BANGLISH_MARKERS = frozenset(
    {
        "ache",
        "aikta",
        "ase",
        "dite",
        "kemon",
        "keno",
        "korbo",
        "korte",
        "koto",
        "koyta",
        "parbe",
        "parbo",
    }
)
CORRECTED_BANGLISH_MARKERS = frozenset(base._BANGLISH_MARKERS) | (  # noqa: SLF001
    ADDITIONAL_BANGLISH_MARKERS
)

STRICT_BANGLISH_INSTRUCTION = (
    "Write both user_prompt and assistant_response in natural romanized Bangla "
    "(Banglish) using Latin letters only. English-only text is forbidden. Each field "
    "must independently contain at least one natural romanized Bengali word, for "
    "example ami, tumi, eta, ki, koro, hobe, koyta, aikta, koto, ache, korbo, or "
    "parbe. Do not use Bengali script. Preserve required literals exactly and use "
    "clear punctuation."
)


def corrected_seed(record_id: str, attempt_number: int) -> int:
    if not 1 <= attempt_number <= CORRECTED_COMPLETION_CONTRACT["maximum_attempts_per_record"]:
        raise ValueError("corrected-completion attempt number is outside authorization")
    material = f"{record_id}:corrected-completion-attempt:{attempt_number}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def corrected_prompt(scenario: base.Scenario, language_mode: str) -> str:
    prompt = retry_v2.build_retry_prompt(scenario, language_mode)
    if language_mode == "banglish":
        prompt = prompt.replace(retry_v2.STRICT_BANGLISH_INSTRUCTION, STRICT_BANGLISH_INSTRUCTION)
        if STRICT_BANGLISH_INSTRUCTION not in prompt:
            raise AssertionError("corrected Banglish instruction was not installed")
        return prompt
    if language_mode != "natural_code_switching":
        return prompt
    term = schema_v4.CODE_SWITCH_TERMS[scenario.scenario_id]
    instruction = (
        "Return user_prompt and assistant_response as objects with exactly two fields: "
        "prefix and suffix. Each prefix must be an unfinished Bengali-script fragment "
        "that leads directly into a reserved English phrase; never end a prefix with a "
        "period, question mark, exclamation mark, Bengali danda, colon, or semicolon. "
        "Each suffix must continue that same sentence naturally and finish it with "
        "correct punctuation. Prefixes and suffixes may contain Bengali script, digits, "
        "and punctuation only; Latin letters are forbidden. The runtime will place this "
        f"reserved phrase exactly once between them: {json.dumps(term, ensure_ascii=False)}. "
        "Do not output, translate, paraphrase, spell, or repeat the reserved phrase. Do "
        "not write a complete standalone sentence in a prefix. Keep both composed fields "
        "concise, grammatical, context-sensitive, and non-repetitive."
    )
    prompt = prompt.replace(base.LANGUAGE_INSTRUCTIONS[language_mode], instruction)
    if instruction not in prompt:
        raise AssertionError("refined code-switch instruction was not installed")
    return prompt


def corrected_payload(
    record_id: str,
    scenario: base.Scenario,
    language_mode: str,
    *,
    attempt_number: int,
) -> dict[str, Any]:
    payload = base.ollama_payload(record_id, scenario, language_mode)
    payload["prompt"] = corrected_prompt(scenario, language_mode)
    payload["options"]["seed"] = corrected_seed(record_id, attempt_number)
    if language_mode == "natural_code_switching":
        payload["format"] = schema_v4.structured_code_switch_schema(scenario)
    return payload


@contextmanager
def _corrected_banglish_marker_scope() -> Iterator[None]:
    original = base._BANGLISH_MARKERS  # noqa: SLF001
    base._BANGLISH_MARKERS = set(CORRECTED_BANGLISH_MARKERS)  # noqa: SLF001
    try:
        yield
    finally:
        base._BANGLISH_MARKERS = original  # noqa: SLF001


def validate_corrected_output(
    output: Mapping[str, Any],
    *,
    scenario: base.Scenario,
    language_mode: str,
) -> list[dict[str, str]]:
    if language_mode != "banglish":
        return base._validate_output(  # noqa: SLF001
            output,
            scenario=scenario,
            language_mode=language_mode,
        )
    with _corrected_banglish_marker_scope():
        return base._validate_output(  # noqa: SLF001
            output,
            scenario=scenario,
            language_mode=language_mode,
        )


def compose_refined_code_switch_output(
    output: Mapping[str, Any],
    *,
    scenario: base.Scenario,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    for field in ("user_prompt", "assistant_response"):
        value = output.get(field)
        if not isinstance(value, Mapping):
            raise ValueError(f"{field} structured fragments are missing")
        prefix = value.get("prefix")
        suffix = value.get("suffix")
        if not isinstance(prefix, str) or not isinstance(suffix, str):
            raise ValueError(f"{field} structured fragments must be strings")
        stripped_prefix = prefix.rstrip()
        if any(marker in stripped_prefix for marker in (".", "!", "?", "।", ":", ";")):
            raise ValueError(f"{field} prefix is a complete sentence, not a fragment")
        if len(stripped_prefix) > 160 or len(suffix.strip()) > 240:
            raise ValueError(f"{field} structured fragments exceed the quality bound")
    return schema_v4.compose_schema_constrained_output(output, scenario=scenario)


def _make_row(
    record_id: str,
    scenario: base.Scenario,
    language_mode: str,
    messages: list[dict[str, str]],
    *,
    attempt_number: int,
) -> dict[str, Any]:
    composed = language_mode == "natural_code_switching"
    return {
        "schema_version": base.ROW_SCHEMA_VERSION,
        "record_id": record_id,
        "source_id": "lumi_first_party_multilingual_safety_tool_v0_5",
        "task": scenario.task,
        "skill_id": scenario.scenario_id,
        "language_mode": language_mode,
        "split": "validation" if scenario.variant == 3 else "train",
        "messages": messages,
        "provenance": {
            "ai_assisted": True,
            "banglish_marker_contract": BANGLISH_MARKER_CONTRACT
            if language_mode == "banglish"
            else None,
            "code_switch_prefix_fragment_gate": composed,
            "deterministic_schema_composition": composed,
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
            "schema_composition_method": schema_v4.SCHEMA_COMPOSITION_METHOD if composed else None,
            "schema_constant_term": schema_v4.CODE_SWITCH_TERMS[scenario.scenario_id]
            if composed
            else None,
            "seed": corrected_seed(record_id, attempt_number),
            "self_hosted": True,
        },
        "rights_status": "private_research_candidate_pending_owner_review",
    }


def generate_corrected_attempt(
    record_id: str,
    scenario: base.Scenario,
    language_mode: str,
    *,
    attempt_number: int,
    endpoint: str,
    rejection_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = corrected_payload(
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
        if language_mode == "natural_code_switching":
            _, messages = compose_refined_code_switch_output(output, scenario=scenario)
        else:
            messages = validate_corrected_output(
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
                "schema_version": "lumi.sft-first-party-teacher-rejected-output.v0.4",
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
    )
    telemetry = {
        "attempt_number": attempt_number,
        "banglish_marker_contract": BANGLISH_MARKER_CONTRACT
        if language_mode == "banglish"
        else None,
        "code_switch_prefix_fragment_gate": language_mode == "natural_code_switching",
        "deterministic_schema_composition": language_mode == "natural_code_switching",
        "elapsed_seconds": round(elapsed, 6),
        "eval_count": response.get("eval_count"),
        "eval_duration_ns": response.get("eval_duration"),
        "load_duration_ns": response.get("load_duration"),
        "prompt_eval_count": response.get("prompt_eval_count"),
        "prompt_eval_duration_ns": response.get("prompt_eval_duration"),
        "record_id": record_id,
        "request_sha256": base.sha256_value(payload),
        "response_envelope_sha256": base.sha256_value(response),
        "schema_composition_method": schema_v4.SCHEMA_COMPOSITION_METHOD
        if language_mode == "natural_code_switching"
        else None,
    }
    return row, telemetry


def _validate_corrected_plan(plan: dict[str, Any], *, artifact: Path) -> None:
    if plan.get("schema_completion") != CORRECTED_COMPLETION_CONTRACT:
        raise ValueError("corrected-completion contract changed")
    carry = plan.get("carry_forward")
    if not isinstance(carry, dict):
        raise ValueError("corrected-completion carry-forward binding is missing")
    family_root = artifact.parent
    expected = {
        "candidate_rows_path": str(
            family_root / "attempt-003" / "private" / "candidate-rows.partial.jsonl"
        ),
        "candidate_rows_sha256": "108b126c1a32fe281637b9e7ccaa2b12189a688d6d8056d612737ec44cc21095",
        "failure_report_path": str(
            family_root / "attempt-004" / "evidence" / "failure-report.json"
        ),
        "failure_report_sha256": "b9fb68b6c59941774faae71870aaab0d5f0b1342a5d35bad37334ab997770ad3",
        "records": 3,
        "telemetry_path": str(
            family_root / "attempt-003" / "private" / "generation-telemetry.partial.jsonl"
        ),
        "telemetry_sha256": "cf3deb7325d64b756082830d9135a8633ee0c79d64c0436189fff57c0b35f757",
    }
    if carry != expected:
        raise ValueError("corrected-completion carry-forward contract changed")
    for field in ("candidate_rows", "failure_report", "telemetry"):
        if file_sha256(carry[f"{field}_path"]) != carry[f"{field}_sha256"]:
            raise ValueError(f"corrected-completion carry-forward changed: {field}")
    diagnostic = plan.get("diagnostic_exclusion")
    expected_diagnostic_path = (
        family_root / "attempt-004" / "private" / "candidate-rows.partial.jsonl"
    )
    if diagnostic != {
        "candidate_rows_path": str(expected_diagnostic_path),
        "candidate_rows_sha256": "573b3c2a3af4bce477c46e627e7cc93da21162c73175c948a896b200532f79b1",
        "excluded_new_records": 1,
        "status": "private_diagnostic_only_not_carried_forward",
    }:
        raise ValueError("attempt-004 diagnostic exclusion binding changed")
    if file_sha256(expected_diagnostic_path) != diagnostic["candidate_rows_sha256"]:
        raise ValueError("attempt-004 diagnostic candidate changed")


@contextmanager
def _patched_v4_runtime() -> Iterator[None]:
    original_validate = schema_v4._validate_schema_completion_plan
    original_generate = schema_v4.generate_schema_completion_attempt
    original_prompt_contract = schema_v4.PROMPT_CONTRACT
    schema_v4._validate_schema_completion_plan = _validate_corrected_plan
    schema_v4.generate_schema_completion_attempt = generate_corrected_attempt
    schema_v4.PROMPT_CONTRACT = PROMPT_CONTRACT
    try:
        yield
    finally:
        schema_v4._validate_schema_completion_plan = original_validate
        schema_v4.generate_schema_completion_attempt = original_generate
        schema_v4.PROMPT_CONTRACT = original_prompt_contract


def run_corrected_completion(
    *,
    plan_path: str | Path,
    repository_root: str | Path,
    artifact_root: str | Path,
    endpoint: str = "http://127.0.0.1:11434/api/generate",
) -> dict[str, Any]:
    with _patched_v4_runtime():
        return schema_v4.run_schema_completion(
            plan_path=plan_path,
            repository_root=repository_root,
            artifact_root=artifact_root,
            endpoint=endpoint,
        )


def _verify_corrected_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != 80:
        raise ValueError("corrected-completion final row count changed")
    for row in rows[3:]:
        provenance = row.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("corrected-completion provenance is missing")
        if provenance.get("prompt_contract") != PROMPT_CONTRACT:
            raise ValueError("corrected-completion prompt contract changed")
        if provenance.get("freeform_manual_rewriting") is not False:
            raise ValueError("corrected-completion manual-rewriting flag changed")
        mode = row.get("language_mode")
        if mode == "banglish":
            if provenance.get("banglish_marker_contract") != BANGLISH_MARKER_CONTRACT:
                raise ValueError("corrected Banglish marker contract changed")
            natural = [
                str(message["content"])
                for message in row.get("messages", [])
                if message.get("role") in {"user", "assistant"}
            ]
            if len(natural) != 2:
                raise ValueError("corrected Banglish natural-message count changed")
            for text in natural:
                markers = set(re.findall(r"[a-z]+", text.casefold())) & set(
                    CORRECTED_BANGLISH_MARKERS
                )
                if not markers:
                    raise ValueError("corrected Banglish row lacks a curated marker")
        elif mode == "natural_code_switching":
            if provenance.get("code_switch_prefix_fragment_gate") is not True:
                raise ValueError("code-switch prefix-fragment provenance changed")


def verify_corrected_completion(
    *,
    plan_path: str | Path,
    repository_root: str | Path,
    artifact_root: str | Path,
) -> dict[str, Any]:
    with _patched_v4_runtime():
        verification = schema_v4.verify_schema_completion(
            plan_path=plan_path,
            repository_root=repository_root,
            artifact_root=artifact_root,
        )
    artifact = Path(artifact_root).resolve()
    plan = base.load_canary_plan(plan_path, repository_root=repository_root)
    _validate_corrected_plan(plan, artifact=artifact)
    rows = base._read_jsonl(artifact / "private" / "candidate-rows.jsonl")
    carry_rows = base._read_jsonl(Path(plan["carry_forward"]["candidate_rows_path"]))
    diagnostic_rows = base._read_jsonl(Path(plan["diagnostic_exclusion"]["candidate_rows_path"]))
    if rows[:3] != carry_rows:
        raise ValueError("corrected-completion carry-forward rows changed")
    if (
        len(diagnostic_rows) != 4
        or rows[3].get("provenance", {}).get("prompt_contract") != PROMPT_CONTRACT
    ):
        raise ValueError("attempt-004 diagnostic row was not excluded and regenerated")
    _verify_corrected_rows(rows)
    verification["checks"]["attempt_004_new_row_excluded"] = True
    verification["checks"]["corrected_banglish_marker_gate"] = True
    verification["checks"]["refined_code_switch_fragment_gate"] = True
    base.write_json(artifact / "evidence" / "verification-report.json", verification)
    return verification
