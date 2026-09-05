"""Result-preserving completion of the local Qwen teacher canary."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from . import posttraining_first_party_teacher_canary as base
from . import posttraining_first_party_teacher_canary_v2 as retry_v2
from .posttraining_first_party import file_sha256
from .posttraining_owner_review import build_owner_review_html

COMPLETION_CONTRACT = {
    "additional_inference_attempts": 120,
    "maximum_attempts_per_record": 3,
    "maximum_network_bytes": 0,
    "maximum_retained_new_records": 77,
    "maximum_retained_supervised_tokens_total": 24_000,
    "maximum_runtime_minutes": 120,
    "total_retained_records": 80,
}

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
    "list_sort": "sort",
    "meeting_summary": "summary",
    "release_summary": "summary",
    "plant_care_summary": "summary",
    "project_summary": "summary",
}


def alternate_seed(record_id: str, attempt_number: int) -> int:
    if not 1 <= attempt_number <= COMPLETION_CONTRACT["maximum_attempts_per_record"]:
        raise ValueError("completion attempt number is outside the authorization")
    material = f"{record_id}:completion-attempt:{attempt_number}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def completion_prompt(
    scenario: base.Scenario,
    language_mode: str,
) -> str:
    prompt = retry_v2.build_retry_prompt(scenario, language_mode)
    if language_mode != "natural_code_switching":
        return prompt
    required_term = CODE_SWITCH_TERMS[scenario.scenario_id]
    strict = (
        "Write both user_prompt and assistant_response using natural Bengali-script and "
        f"English code-switching. Each field must independently contain Bengali script "
        f"and the exact English term {json.dumps(required_term)}. Bengali-only and "
        "English-only fields are forbidden. Preserve all other required literals and use "
        "correct punctuation."
    )
    prompt = prompt.replace(
        base.LANGUAGE_INSTRUCTIONS["natural_code_switching"],
        strict,
    )
    if strict not in prompt:
        raise AssertionError("strict code-switch instruction was not installed")
    return prompt


def completion_payload(
    record_id: str,
    scenario: base.Scenario,
    language_mode: str,
    *,
    attempt_number: int,
) -> dict[str, Any]:
    payload = base.ollama_payload(record_id, scenario, language_mode)
    payload["prompt"] = completion_prompt(scenario, language_mode)
    payload["options"]["seed"] = alternate_seed(record_id, attempt_number)
    return payload


def _validate_code_switch_term(
    messages: list[dict[str, str]],
    *,
    scenario: base.Scenario,
    language_mode: str,
) -> None:
    if language_mode != "natural_code_switching":
        return
    required = CODE_SWITCH_TERMS[scenario.scenario_id].casefold()
    natural = [
        message["content"] for message in messages if message["role"] in {"user", "assistant"}
    ]
    if len(natural) != 2 or any(required not in text.casefold() for text in natural):
        raise ValueError("scenario-specific code-switch term is missing")


def _make_row(
    record_id: str,
    scenario: base.Scenario,
    language_mode: str,
    messages: list[dict[str, str]],
    *,
    attempt_number: int,
) -> dict[str, Any]:
    return {
        "schema_version": base.ROW_SCHEMA_VERSION,
        "record_id": record_id,
        "source_id": "lumi_first_party_multilingual_safety_tool_v0_3",
        "task": scenario.task,
        "skill_id": scenario.scenario_id,
        "language_mode": language_mode,
        "split": "validation" if scenario.variant == 3 else "train",
        "messages": messages,
        "provenance": {
            "ai_assisted": True,
            "generation_attempt": attempt_number,
            "human_authored": False,
            "inference_location": "mac_local_ollama_metal",
            "model_file_sha256": base.MODEL_SHA256,
            "model_id": base.MODEL_ID,
            "model_revision": base.MODEL_REVISION,
            "ollama_model": base.OLLAMA_MODEL,
            "prompt_contract": "lumi.first-party-teacher-prompt.v0.3",
            "quantization": "Q4_K_M",
            "seed": alternate_seed(record_id, attempt_number),
            "self_hosted": True,
        },
        "rights_status": "private_research_candidate_pending_owner_review",
    }


def generate_completion_attempt(
    record_id: str,
    scenario: base.Scenario,
    language_mode: str,
    *,
    attempt_number: int,
    endpoint: str,
    rejection_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = completion_payload(
        record_id,
        scenario,
        language_mode,
        attempt_number=attempt_number,
    )
    response: dict[str, Any] | None = None
    started = time.monotonic()
    try:
        response = base._post_local_json(endpoint, payload)
        if response.get("done") is not True or response.get("done_reason") != "stop":
            raise ValueError("local teacher did not finish with stop")
        if response.get("thinking") not in (None, ""):
            raise ValueError("local teacher returned hidden thinking")
        raw = response.get("response")
        if not isinstance(raw, str) or base._HIDDEN_REASONING.search(raw):
            raise ValueError("local teacher raw response contains hidden reasoning")
        try:
            output = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError("local teacher response is not JSON") from error
        if not isinstance(output, dict):
            raise ValueError("local teacher response root is not an object")
        messages = base._validate_output(
            output,
            scenario=scenario,
            language_mode=language_mode,
        )
        _validate_code_switch_term(
            messages,
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
                "schema_version": "lumi.sft-first-party-teacher-rejected-output.v0.2",
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
        "elapsed_seconds": round(elapsed, 6),
        "eval_count": response.get("eval_count"),
        "eval_duration_ns": response.get("eval_duration"),
        "load_duration_ns": response.get("load_duration"),
        "prompt_eval_count": response.get("prompt_eval_count"),
        "prompt_eval_duration_ns": response.get("prompt_eval_duration"),
        "record_id": record_id,
        "request_sha256": base.sha256_value(payload),
        "response_envelope_sha256": base.sha256_value(response),
    }
    return row, telemetry


def _validate_completion_plan(plan: dict[str, Any], *, artifact: Path) -> None:
    if plan.get("completion") != COMPLETION_CONTRACT:
        raise ValueError("teacher completion contract changed")
    carry = plan.get("carry_forward")
    if not isinstance(carry, dict):
        raise ValueError("teacher completion carry-forward binding is missing")
    expected_root = artifact.parent / "attempt-002"
    expected = {
        "candidate_rows_path": str(expected_root / "private" / "candidate-rows.partial.jsonl"),
        "candidate_rows_sha256": "108b126c1a32fe281637b9e7ccaa2b12189a688d6d8056d612737ec44cc21095",
        "failure_report_path": str(expected_root / "evidence" / "failure-report.json"),
        "failure_report_sha256": "a356a96fff6051049a5998780468822651730db8bf3a839629bc650d20ab00cc",
        "records": 3,
        "telemetry_path": str(expected_root / "private" / "generation-telemetry.partial.jsonl"),
        "telemetry_sha256": "cf3deb7325d64b756082830d9135a8633ee0c79d64c0436189fff57c0b35f757",
    }
    if carry != expected:
        raise ValueError("teacher completion carry-forward contract changed")
    for field in ("candidate_rows", "failure_report", "telemetry"):
        if file_sha256(carry[f"{field}_path"]) != carry[f"{field}_sha256"]:
            raise ValueError(f"teacher completion carry-forward changed: {field}")


def _initialize_or_load(
    plan: dict[str, Any],
    *,
    artifact: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    private = artifact / "private"
    partial_rows = private / "candidate-rows.partial.jsonl"
    partial_telemetry = private / "generation-telemetry.partial.jsonl"
    ledger_path = private / "additional-attempt-ledger.jsonl"
    if not partial_rows.exists() and not partial_telemetry.exists():
        rows = base._read_jsonl(Path(plan["carry_forward"]["candidate_rows_path"]))
        telemetry = base._read_jsonl(Path(plan["carry_forward"]["telemetry_path"]))
        if len(rows) != 3 or len(telemetry) != 3:
            raise ValueError("teacher completion carry-forward count changed")
        base.write_jsonl(partial_rows, rows)
        base.write_jsonl(partial_telemetry, telemetry)
    else:
        rows = base._read_jsonl(partial_rows)
        telemetry = base._read_jsonl(partial_telemetry)
    ledger = base._read_jsonl(ledger_path)
    expected_ids = base.expected_record_ids()
    if [str(row.get("record_id")) for row in rows] != list(expected_ids[: len(rows)]):
        raise ValueError("teacher completion rows are not an exact expected prefix")
    if len(rows) != len(telemetry) or not 3 <= len(rows) <= 80:
        raise ValueError("teacher completion row and telemetry counts differ")
    if len(ledger) > COMPLETION_CONTRACT["additional_inference_attempts"]:
        raise ValueError("teacher completion attempt ledger exceeds authorization")
    return rows, telemetry, ledger


def run_canary_completion(
    *,
    plan_path: str | Path,
    repository_root: str | Path,
    artifact_root: str | Path,
    endpoint: str = "http://127.0.0.1:11434/api/generate",
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    artifact = Path(artifact_root).resolve()
    plan = base.load_canary_plan(plan_path, repository_root=root)
    _validate_completion_plan(plan, artifact=artifact)
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
        raise RuntimeError("memory pressure is unsafe before teacher completion")

    while len(rows) < 80:
        record_id = expected_ids[len(rows)]
        task, language_mode, variant_text = record_id.removeprefix("lumi-teacher-").rsplit("-", 2)
        scenario = scenarios[(task, int(variant_text))]
        succeeded = False
        while per_record_attempts[record_id] < COMPLETION_CONTRACT["maximum_attempts_per_record"]:
            if len(ledger) >= COMPLETION_CONTRACT["additional_inference_attempts"]:
                raise RuntimeError("teacher completion exhausted its inference-attempt cap")
            attempt_number = per_record_attempts[record_id] + 1
            rejection_path = private / (
                f"rejected-{len(ledger) + 1:03d}-{record_id}-attempt-{attempt_number}.json"
            )
            try:
                row, item_telemetry = generate_completion_attempt(
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
                    raise RuntimeError("memory pressure became unsafe during teacher completion")
                elapsed_minutes = (time.monotonic() - started_monotonic) / 60
                if elapsed_minutes > COMPLETION_CONTRACT["maximum_runtime_minutes"]:
                    raise RuntimeError("teacher completion exceeded runtime authorization")
                print(
                    f"retained {len(rows)}/80; additional_calls={len(ledger)}; memory_free={free}%",
                    flush=True,
                )
            break
        if not succeeded:
            raise RuntimeError(f"teacher completion exhausted retries for {record_id}")

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
        maximum_supervised_tokens=COMPLETION_CONTRACT["maximum_retained_supervised_tokens_total"],
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
        "completion": {
            "additional_calls_used": len(ledger),
            "additional_records_retained": 77,
            "attempt_ledger_internal_sha256": base.sha256_value(ledger),
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


def verify_canary_completion(
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
    _validate_completion_plan(plan, artifact=artifact)
    report = base._json_object(artifact / "evidence" / "run-report.json")
    ledger_path = artifact / "private" / "additional-attempt-ledger.jsonl"
    ledger = base._read_jsonl(ledger_path)
    if not 77 <= len(ledger) <= COMPLETION_CONTRACT["additional_inference_attempts"]:
        raise ValueError("teacher completion attempt count is invalid")
    if base.sha256_value(ledger) != report.get("completion", {}).get(
        "attempt_ledger_internal_sha256"
    ):
        raise ValueError("teacher completion attempt ledger changed")
    if file_sha256(ledger_path) != report.get("attempt_ledger_file_sha256"):
        raise ValueError("teacher completion attempt ledger file changed")
    if sum(item.get("status") == "accepted" for item in ledger) != 77:
        raise ValueError("teacher completion accepted-attempt count changed")
    verification["checks"]["carry_forward_three_rows"] = True
    verification["checks"]["completion_attempt_cap"] = True
    verification["checks"]["scenario_specific_code_switching"] = True
    verification["attempt_ledger_file_sha256"] = file_sha256(ledger_path)
    base.write_json(artifact / "evidence" / "verification-report.json", verification)
    return verification
