"""Schema-constrained completion of the local Qwen teacher canary."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import posttraining_first_party_teacher_canary as base
from . import posttraining_first_party_teacher_canary_v2 as retry_v2
from .posttraining_first_party import file_sha256
from .posttraining_owner_review import build_owner_review_html

SCHEMA_COMPLETION_CONTRACT = {
    "additional_inference_attempts": 120,
    "maximum_attempts_per_record": 3,
    "maximum_network_bytes": 0,
    "maximum_retained_new_records": 77,
    "maximum_retained_supervised_tokens_total": 24_000,
    "maximum_runtime_minutes": 120,
    "total_retained_records": 80,
}

SCHEMA_COMPOSITION_METHOD = "trim_join_single_spaces_v1"
PROMPT_CONTRACT = "lumi.first-party-teacher-prompt.v0.4"

# These phrases are deterministic runtime constants, never free-form model output. The
# four writing phrases also carry every grounded ASCII literal required by the base gate.
CODE_SWITCH_TERMS = {
    "kilometer_to_meters": "meter",
    "days_in_week": "week",
    "triangle_sides": "triangle",
    "water_freezing_celsius": "Celsius",
    "friendly_greeting": "hello",
    "punctuation_repair": "punctuation",
    "three_day_study_plan": "study plan",
    "polite_rewrite": "rewrite",
    "benign_study_help": "quiz",
    "credential_privacy": "password",
    "urgent_medical_caution": "emergency",
    "phishing_request": "phishing",
    "calculator_addition": "calculate",
    "calculator_multiplication": "calculate",
    "calculator_division": "calculate",
    "list_sort": "sort 2 5 9",
    "meeting_summary": "10:00 Room B ID summary",
    "release_summary": "2.4 dark mode offline search summary",
    "plant_care_summary": "3 times/week indirect light Friday summary",
    "project_summary": "Phase Alpha 12 May Nadia summary",
}


def schema_completion_seed(record_id: str, attempt_number: int) -> int:
    if not 1 <= attempt_number <= SCHEMA_COMPLETION_CONTRACT["maximum_attempts_per_record"]:
        raise ValueError("schema-completion attempt number is outside the authorization")
    material = f"{record_id}:schema-completion-attempt:{attempt_number}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def structured_code_switch_schema(scenario: base.Scenario) -> dict[str, Any]:
    component = {
        "type": "object",
        "properties": {
            "prefix": {"type": "string"},
            "suffix": {"type": "string"},
        },
        "required": ["prefix", "suffix"],
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
    schema["properties"]["user_prompt"] = copy.deepcopy(component)
    schema["properties"]["assistant_response"] = copy.deepcopy(component)
    return schema


def schema_completion_prompt(scenario: base.Scenario, language_mode: str) -> str:
    prompt = retry_v2.build_retry_prompt(scenario, language_mode)
    if language_mode != "natural_code_switching":
        return prompt
    term = CODE_SWITCH_TERMS[scenario.scenario_id]
    instruction = (
        "Return user_prompt and assistant_response as objects with exactly two fields: "
        "prefix and suffix. Write every prefix and suffix only in natural Bengali script; "
        "digits and punctuation are allowed, but Latin letters are forbidden. Design each "
        "prefix/suffix pair so the runtime can place one reserved English phrase between "
        f"them: {json.dumps(term, ensure_ascii=False)}. Do not output, translate, spell, or "
        "repeat that reserved phrase inside prefix or suffix. Both composed fields must be "
        "grammatical, context-sensitive, concise, and correctly punctuated."
    )
    prompt = prompt.replace(base.LANGUAGE_INSTRUCTIONS[language_mode], instruction)
    if instruction not in prompt:
        raise AssertionError("schema-constrained code-switch instruction was not installed")
    return prompt


def schema_completion_payload(
    record_id: str,
    scenario: base.Scenario,
    language_mode: str,
    *,
    attempt_number: int,
) -> dict[str, Any]:
    payload = base.ollama_payload(record_id, scenario, language_mode)
    payload["prompt"] = schema_completion_prompt(scenario, language_mode)
    payload["options"]["seed"] = schema_completion_seed(record_id, attempt_number)
    if language_mode == "natural_code_switching":
        payload["format"] = structured_code_switch_schema(scenario)
    return payload


def _validate_bengali_component(value: Any, *, field: str) -> tuple[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"prefix", "suffix"}:
        raise ValueError(f"{field} must contain only structured prefix and suffix fields")
    prefix = value.get("prefix")
    suffix = value.get("suffix")
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError(f"{field} Bengali prefix is empty")
    if not isinstance(suffix, str) or not suffix.strip():
        raise ValueError(f"{field} Bengali suffix is empty")
    for name, text in (("prefix", prefix), ("suffix", suffix)):
        if base._BENGALI.search(text) is None:  # noqa: SLF001
            raise ValueError(f"{field} {name} lacks Bengali script")
        if base._LATIN.search(text) is not None:  # noqa: SLF001
            raise ValueError(f"{field} {name} contains free-form Latin text")
    return prefix.strip(), suffix.strip()


def compose_schema_constrained_output(
    output: Mapping[str, Any],
    *,
    scenario: base.Scenario,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    expected_fields = {
        "assistant_response",
        "safety_label",
        "scenario_id",
        "tool_call",
        "tool_result",
        "user_prompt",
    }
    if set(output) != expected_fields:
        raise ValueError("schema-constrained teacher JSON envelope fields changed")
    user_prefix, user_suffix = _validate_bengali_component(
        output.get("user_prompt"), field="user_prompt"
    )
    assistant_prefix, assistant_suffix = _validate_bengali_component(
        output.get("assistant_response"), field="assistant_response"
    )
    term = CODE_SWITCH_TERMS[scenario.scenario_id]

    def compose(prefix: str, suffix: str) -> str:
        return f"{prefix.rstrip()} {term} {suffix.lstrip()}"

    flattened = dict(output)
    flattened["user_prompt"] = compose(user_prefix, user_suffix)
    flattened["assistant_response"] = compose(assistant_prefix, assistant_suffix)
    messages = base._validate_output(  # noqa: SLF001
        flattened,
        scenario=scenario,
        language_mode="natural_code_switching",
    )
    natural = [
        message["content"] for message in messages if message["role"] in {"user", "assistant"}
    ]
    if len(natural) != 2:
        raise ValueError("schema composition did not create two natural-language fields")
    for text in natural:
        if text.count(term) != 1:
            raise ValueError("schema-constant English phrase was not inserted exactly once")
        outside_term = text.replace(term, "", 1)
        if base._LATIN.search(outside_term) is not None:  # noqa: SLF001
            raise ValueError("free-form Latin text survived outside the schema constant")
    return flattened, messages


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
        "source_id": "lumi_first_party_multilingual_safety_tool_v0_4",
        "task": scenario.task,
        "skill_id": scenario.scenario_id,
        "language_mode": language_mode,
        "split": "validation" if scenario.variant == 3 else "train",
        "messages": messages,
        "provenance": {
            "ai_assisted": True,
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
            "schema_composition_method": SCHEMA_COMPOSITION_METHOD if composed else None,
            "schema_constant_term": CODE_SWITCH_TERMS[scenario.scenario_id] if composed else None,
            "seed": schema_completion_seed(record_id, attempt_number),
            "self_hosted": True,
        },
        "rights_status": "private_research_candidate_pending_owner_review",
    }


def generate_schema_completion_attempt(
    record_id: str,
    scenario: base.Scenario,
    language_mode: str,
    *,
    attempt_number: int,
    endpoint: str,
    rejection_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = schema_completion_payload(
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
            _, messages = compose_schema_constrained_output(output, scenario=scenario)
        else:
            messages = base._validate_output(  # noqa: SLF001
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
                "schema_version": "lumi.sft-first-party-teacher-rejected-output.v0.3",
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
        "schema_composition_method": SCHEMA_COMPOSITION_METHOD
        if language_mode == "natural_code_switching"
        else None,
    }
    return row, telemetry


def _validate_schema_completion_plan(plan: dict[str, Any], *, artifact: Path) -> None:
    if plan.get("schema_completion") != SCHEMA_COMPLETION_CONTRACT:
        raise ValueError("schema-completion contract changed")
    carry = plan.get("carry_forward")
    if not isinstance(carry, dict):
        raise ValueError("schema-completion carry-forward binding is missing")
    expected_root = artifact.parent / "attempt-003"
    expected = {
        "candidate_rows_path": str(expected_root / "private" / "candidate-rows.partial.jsonl"),
        "candidate_rows_sha256": "108b126c1a32fe281637b9e7ccaa2b12189a688d6d8056d612737ec44cc21095",
        "failure_report_path": str(expected_root / "evidence" / "failure-report.json"),
        "failure_report_sha256": "8823d46e8345dd4ec8d65dade2ebdea749573a9dc0a086edb3c339650f152457",
        "records": 3,
        "telemetry_path": str(expected_root / "private" / "generation-telemetry.partial.jsonl"),
        "telemetry_sha256": "cf3deb7325d64b756082830d9135a8633ee0c79d64c0436189fff57c0b35f757",
    }
    if carry != expected:
        raise ValueError("schema-completion carry-forward contract changed")
    for field in ("candidate_rows", "failure_report", "telemetry"):
        if file_sha256(carry[f"{field}_path"]) != carry[f"{field}_sha256"]:
            raise ValueError(f"schema-completion carry-forward changed: {field}")


def _initialize_or_load(
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
        raise ValueError("schema-completion partial row/telemetry pair is incomplete")
    carry_rows = base._read_jsonl(Path(plan["carry_forward"]["candidate_rows_path"]))
    carry_telemetry = base._read_jsonl(Path(plan["carry_forward"]["telemetry_path"]))
    if not any(present):
        if len(carry_rows) != 3 or len(carry_telemetry) != 3:
            raise ValueError("schema-completion carry-forward count changed")
        base.write_jsonl(partial_rows, carry_rows)
        base.write_jsonl(partial_telemetry, carry_telemetry)
    rows = base._read_jsonl(partial_rows)
    telemetry = base._read_jsonl(partial_telemetry)
    ledger = base._read_jsonl(ledger_path)
    if rows[:3] != carry_rows or telemetry[:3] != carry_telemetry:
        raise ValueError("schema-completion exact carry-forward rows or telemetry changed")
    expected_ids = base.expected_record_ids()
    if [str(row.get("record_id")) for row in rows] != list(expected_ids[: len(rows)]):
        raise ValueError("schema-completion rows are not an exact expected prefix")
    if len(rows) != len(telemetry) or not 3 <= len(rows) <= 80:
        raise ValueError("schema-completion row and telemetry counts differ")
    if len(ledger) > SCHEMA_COMPLETION_CONTRACT["additional_inference_attempts"]:
        raise ValueError("schema-completion attempt ledger exceeds authorization")
    return rows, telemetry, ledger


def _assert_runtime(started_monotonic: float) -> None:
    elapsed_minutes = (time.monotonic() - started_monotonic) / 60
    if elapsed_minutes > SCHEMA_COMPLETION_CONTRACT["maximum_runtime_minutes"]:
        raise RuntimeError("schema-completion exceeded runtime authorization")


def run_schema_completion(
    *,
    plan_path: str | Path,
    repository_root: str | Path,
    artifact_root: str | Path,
    endpoint: str = "http://127.0.0.1:11434/api/generate",
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    artifact = Path(artifact_root).resolve()
    plan = base.load_canary_plan(plan_path, repository_root=root)
    _validate_schema_completion_plan(plan, artifact=artifact)
    started_wall = time.time()
    started_monotonic = time.monotonic()
    rows, telemetry, ledger = _initialize_or_load(plan, artifact=artifact)
    private = artifact / "private"
    ledger_path = private / "additional-attempt-ledger.jsonl"
    expected_ids = base.expected_record_ids()
    scenarios = {(item.task, item.variant): item for item in base.SCENARIOS}
    per_record_attempts = Counter(str(item["record_id"]) for item in ledger)

    initial_memory = base.memory_free_percent()
    if initial_memory < 20:
        raise RuntimeError("memory pressure is unsafe before schema-completion")

    while len(rows) < SCHEMA_COMPLETION_CONTRACT["total_retained_records"]:
        _assert_runtime(started_monotonic)
        record_id = expected_ids[len(rows)]
        task, language_mode, variant_text = record_id.removeprefix("lumi-teacher-").rsplit("-", 2)
        scenario = scenarios[(task, int(variant_text))]
        succeeded = False
        while (
            per_record_attempts[record_id]
            < SCHEMA_COMPLETION_CONTRACT["maximum_attempts_per_record"]
        ):
            _assert_runtime(started_monotonic)
            if len(ledger) >= SCHEMA_COMPLETION_CONTRACT["additional_inference_attempts"]:
                raise RuntimeError("schema-completion exhausted its inference-attempt cap")
            attempt_number = per_record_attempts[record_id] + 1
            rejection_path = private / (
                f"rejected-{len(ledger) + 1:03d}-{record_id}-attempt-{attempt_number}.json"
            )
            try:
                row, item_telemetry = generate_schema_completion_attempt(
                    record_id,
                    scenario,
                    language_mode,
                    attempt_number=attempt_number,
                    endpoint=endpoint,
                    rejection_path=rejection_path,
                )
            except Exception as error:
                ledger.append(
                    {
                        "attempt_number": attempt_number,
                        "error_code": type(error).__name__,
                        "record_id": record_id,
                        "rejection_file_sha256": file_sha256(rejection_path),
                        "status": "rejected",
                    }
                )
                per_record_attempts[record_id] += 1
                base.write_jsonl(ledger_path, ledger)
                continue
            rows.append(row)
            telemetry.append(item_telemetry)
            ledger.append(
                {
                    "attempt_number": attempt_number,
                    "record_id": record_id,
                    "request_sha256": item_telemetry["request_sha256"],
                    "response_envelope_sha256": item_telemetry["response_envelope_sha256"],
                    "status": "accepted",
                }
            )
            per_record_attempts[record_id] += 1
            base.write_jsonl(private / "candidate-rows.partial.jsonl", rows)
            base.write_jsonl(private / "generation-telemetry.partial.jsonl", telemetry)
            base.write_jsonl(ledger_path, ledger)
            succeeded = True
            if len(rows) % 10 == 0:
                free = base.memory_free_percent()
                if free < 15:
                    raise RuntimeError("memory pressure became unsafe during schema-completion")
                print(
                    f"retained {len(rows)}/80; additional_calls={len(ledger)}; memory_free={free}%",
                    flush=True,
                )
            break
        if not succeeded:
            raise RuntimeError(f"schema-completion exhausted retries for {record_id}")

    _assert_runtime(started_monotonic)
    evidence = artifact / "evidence"
    review = artifact / "review"
    base.write_jsonl(private / "candidate-rows.jsonl", rows)
    base.write_jsonl(private / "generation-telemetry.jsonl", telemetry)
    bindings = plan["bindings"]
    audit = base.audit_rows(
        rows,
        repository_root=root,
        quality_config_path=root / str(bindings["data_quality"]["path"]),
        chat_template_path=root / str(bindings["chat_template"]["path"]),
        tokenizer_path=str(plan["tokenizer"]["path"]),
        maximum_supervised_tokens=SCHEMA_COMPLETION_CONTRACT[
            "maximum_retained_supervised_tokens_total"
        ],
    )
    owner_sample = base.select_owner_sample(rows)
    owner_sample_sha256 = base.sha256_value(owner_sample)
    base.write_jsonl(private / "owner-review-sample-32.jsonl", owner_sample)
    review.mkdir(parents=True, exist_ok=True)
    (review / "owner-review-32.html").write_text(
        build_owner_review_html(owner_sample, sample_sha256=owner_sample_sha256),
        encoding="utf-8",
    )

    durations = [float(item["elapsed_seconds"]) for item in telemetry[3:]]
    report = {
        "schema_version": base.REPORT_SCHEMA_VERSION,
        "status": "verified_private_candidate_pending_owner_review",
        "production_eligible": False,
        "candidate_registry_changed": False,
        "dataset_sealed": False,
        "external_api_used": False,
        "human_authorship_claimed": False,
        "sft_started": False,
        "authorization_file_sha256": file_sha256(artifact / "authorization.json"),
        "plan_file_sha256": file_sha256(plan_path),
        "carry_forward": {
            "records": 3,
            "source_candidate_rows_file_sha256": plan["carry_forward"]["candidate_rows_sha256"],
            "source_failure_report_sha256": plan["carry_forward"]["failure_report_sha256"],
        },
        "schema_completion": {
            "additional_calls_used": len(ledger),
            "additional_records_retained": 77,
            "attempt_ledger_internal_sha256": base.sha256_value(ledger),
            "code_switch_rows_composed": sum(
                row["language_mode"] == "natural_code_switching" for row in rows
            ),
            "composition_method": SCHEMA_COMPOSITION_METHOD,
            "freeform_manual_rewriting": False,
            "rejected_outputs_retained": sum(item["status"] == "rejected" for item in ledger),
        },
        "model": {
            "file_sha256": base.MODEL_SHA256,
            "id": base.MODEL_ID,
            "revision": base.MODEL_REVISION,
            "runtime": "ollama-0.32.5-local-metal-offline",
        },
        "audit": audit,
        "generation": {
            "calls": len(telemetry),
            "maximum_call_seconds": round(max(durations), 6),
            "median_call_seconds": round(sorted(durations)[len(durations) // 2], 6),
            "telemetry_internal_sha256": base.sha256_value(telemetry),
        },
        "owner_review": {
            "records": len(owner_sample),
            "sample_internal_sha256": owner_sample_sha256,
            "status": "pending",
        },
        "runtime": {
            "elapsed_seconds": round(time.monotonic() - started_monotonic, 6),
            "finished_unix": int(time.time()),
            "initial_memory_free_percent": initial_memory,
            "started_unix": int(started_wall),
        },
        "private_artifacts": {
            "candidate_rows_file_sha256": file_sha256(private / "candidate-rows.jsonl"),
            "generation_telemetry_file_sha256": file_sha256(private / "generation-telemetry.jsonl"),
            "owner_review_html_file_sha256": file_sha256(review / "owner-review-32.html"),
            "owner_review_sample_file_sha256": file_sha256(
                private / "owner-review-sample-32.jsonl"
            ),
        },
        "attempt_ledger_file_sha256": file_sha256(ledger_path),
    }
    base.write_json(evidence / "run-report.json", report)
    return report


def _verify_schema_composition(rows: Sequence[Mapping[str, Any]]) -> None:
    code_switch_rows = [row for row in rows if row.get("language_mode") == "natural_code_switching"]
    if len(code_switch_rows) != 20:
        raise ValueError("schema-completion code-switch row count changed")
    for row in code_switch_rows:
        scenario_id = str(row.get("skill_id"))
        term = CODE_SWITCH_TERMS.get(scenario_id)
        if term is None:
            raise ValueError("schema-completion scenario term binding is missing")
        provenance = row.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("schema-completion provenance is missing")
        expected = {
            "deterministic_schema_composition": True,
            "freeform_manual_rewriting": False,
            "human_authored": False,
            "prompt_contract": PROMPT_CONTRACT,
            "schema_composition_method": SCHEMA_COMPOSITION_METHOD,
            "schema_constant_term": term,
        }
        for field, value in expected.items():
            if provenance.get(field) != value:
                raise ValueError(f"schema-completion provenance changed: {field}")
        natural = [
            message["content"]
            for message in row.get("messages", [])
            if message.get("role") in {"user", "assistant"}
        ]
        if len(natural) != 2:
            raise ValueError("schema-completion natural message count changed")
        for text in natural:
            if text.count(term) != 1:
                raise ValueError("schema constant does not occur exactly once")
            if base._LATIN.search(text.replace(term, "", 1)) is not None:  # noqa: SLF001
                raise ValueError("free-form Latin text exists outside schema constant")


def verify_schema_completion(
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
    _validate_schema_completion_plan(plan, artifact=artifact)
    report = base._json_object(artifact / "evidence" / "run-report.json")
    ledger_path = artifact / "private" / "additional-attempt-ledger.jsonl"
    ledger = base._read_jsonl(ledger_path)
    rows = base._read_jsonl(artifact / "private" / "candidate-rows.jsonl")
    carry_rows = base._read_jsonl(Path(plan["carry_forward"]["candidate_rows_path"]))
    if rows[:3] != carry_rows:
        raise ValueError("schema-completion final carry-forward rows changed")
    if not 77 <= len(ledger) <= SCHEMA_COMPLETION_CONTRACT["additional_inference_attempts"]:
        raise ValueError("schema-completion attempt count is invalid")
    if base.sha256_value(ledger) != report.get("schema_completion", {}).get(
        "attempt_ledger_internal_sha256"
    ):
        raise ValueError("schema-completion attempt ledger changed")
    if file_sha256(ledger_path) != report.get("attempt_ledger_file_sha256"):
        raise ValueError("schema-completion attempt ledger file changed")
    if sum(item.get("status") == "accepted" for item in ledger) != 77:
        raise ValueError("schema-completion accepted-attempt count changed")
    _verify_schema_composition(rows)
    verification["checks"]["carry_forward_three_rows"] = True
    verification["checks"]["schema_completion_attempt_cap"] = True
    verification["checks"]["schema_constant_code_switch_composition"] = True
    verification["checks"]["zero_freeform_manual_rewriting"] = True
    verification["attempt_ledger_file_sha256"] = file_sha256(ledger_path)
    base.write_json(artifact / "evidence" / "verification-report.json", verification)
    return verification
