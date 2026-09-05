"""Corrected local Qwen teacher canary retry with strict Banglish prompting."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from . import posttraining_first_party_teacher_canary as base
from .posttraining_first_party import file_sha256
from .posttraining_owner_review import build_owner_review_html

RETRY_CONTRACT = {
    "additional_generation_attempts": 80,
    "maximum_network_bytes": 0,
    "maximum_retained_candidate_records": 80,
    "maximum_retained_supervised_tokens": 24_000,
    "maximum_runtime_minutes": 120,
}

STRICT_BANGLISH_INSTRUCTION = (
    "Write both user_prompt and assistant_response in natural romanized Bangla "
    "(Banglish) using Latin letters only. English-only text is forbidden. Each field "
    "must independently include at least one exact natural Banglish word from this list: "
    "ami, tumi, eta, ki, hoy, hobe, koro, lagbe. Do not use Bengali script. Preserve "
    "required literals exactly and use clear punctuation."
)


def build_retry_prompt(
    scenario: base.Scenario,
    language_mode: str,
) -> str:
    prompt = base.build_generation_prompt(scenario, language_mode)
    if language_mode == "banglish":
        prompt = prompt.replace(
            base.LANGUAGE_INSTRUCTIONS["banglish"],
            STRICT_BANGLISH_INSTRUCTION,
        )
        if STRICT_BANGLISH_INSTRUCTION not in prompt:
            raise AssertionError("strict Banglish instruction was not installed")
    return prompt


def retry_payload(
    record_id: str,
    scenario: base.Scenario,
    language_mode: str,
) -> dict[str, Any]:
    payload = base.ollama_payload(record_id, scenario, language_mode)
    payload["prompt"] = build_retry_prompt(scenario, language_mode)
    return payload


def generate_one_retry(
    record_id: str,
    scenario: base.Scenario,
    language_mode: str,
    *,
    endpoint: str,
    failure_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request_payload = retry_payload(record_id, scenario, language_mode)
    response: dict[str, Any] | None = None
    started = time.monotonic()
    try:
        response = base._post_local_json(endpoint, request_payload)
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
    except Exception as error:
        base.write_json(
            failure_path,
            {
                "error_message": str(error),
                "error_type": type(error).__name__,
                "record_id": record_id,
                "request": request_payload,
                "response": response,
                "schema_version": "lumi.sft-first-party-teacher-rejected-output.v0.1",
                "status": "rejected_private_diagnostic_only",
            },
        )
        raise

    split = "validation" if scenario.variant == 3 else "train"
    row = {
        "schema_version": base.ROW_SCHEMA_VERSION,
        "record_id": record_id,
        "source_id": "lumi_first_party_multilingual_safety_tool_v0_3",
        "task": scenario.task,
        "skill_id": scenario.scenario_id,
        "language_mode": language_mode,
        "split": split,
        "messages": messages,
        "provenance": {
            "ai_assisted": True,
            "human_authored": False,
            "inference_location": "mac_local_ollama_metal",
            "model_file_sha256": base.MODEL_SHA256,
            "model_id": base.MODEL_ID,
            "model_revision": base.MODEL_REVISION,
            "ollama_model": base.OLLAMA_MODEL,
            "prompt_contract": "lumi.first-party-teacher-prompt.v0.2",
            "quantization": "Q4_K_M",
            "seed": base.seed_for_record(record_id),
            "self_hosted": True,
        },
        "rights_status": "private_research_candidate_pending_owner_review",
    }
    elapsed = time.monotonic() - started
    telemetry = {
        "record_id": record_id,
        "request_sha256": base.sha256_value(request_payload),
        "response_envelope_sha256": base.sha256_value(response),
        "elapsed_seconds": round(elapsed, 6),
        "load_duration_ns": response.get("load_duration"),
        "prompt_eval_count": response.get("prompt_eval_count"),
        "prompt_eval_duration_ns": response.get("prompt_eval_duration"),
        "eval_count": response.get("eval_count"),
        "eval_duration_ns": response.get("eval_duration"),
    }
    return row, telemetry


def _validate_retry_plan(plan: dict[str, Any], *, artifact_root: Path) -> None:
    if plan.get("retry") != RETRY_CONTRACT:
        raise ValueError("corrected teacher retry contract changed")
    predecessor = plan.get("predecessor_failure")
    expected_path = artifact_root.parent / "attempt-001" / "evidence" / "failure-report.json"
    if not isinstance(predecessor, dict):
        raise ValueError("corrected teacher predecessor failure binding is missing")
    if Path(str(predecessor.get("path"))).resolve() != expected_path.resolve():
        raise ValueError("corrected teacher predecessor failure path changed")
    if file_sha256(expected_path) != predecessor.get("sha256"):
        raise ValueError("corrected teacher predecessor failure evidence changed")


def run_canary_retry(
    *,
    plan_path: str | Path,
    repository_root: str | Path,
    artifact_root: str | Path,
    endpoint: str = "http://127.0.0.1:11434/api/generate",
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    artifact = Path(artifact_root).resolve()
    plan = base.load_canary_plan(plan_path, repository_root=root)
    _validate_retry_plan(plan, artifact_root=artifact)
    started_wall = time.time()
    started_monotonic = time.monotonic()
    private = artifact / "private"
    partial_rows = private / "candidate-rows.partial.jsonl"
    partial_telemetry = private / "generation-telemetry.partial.jsonl"
    rows = base._read_jsonl(partial_rows)
    telemetry = base._read_jsonl(partial_telemetry)
    expected_ids = base.expected_record_ids()
    if [str(row.get("record_id")) for row in rows] != list(expected_ids[: len(rows)]):
        raise ValueError("partial corrected teacher rows are not an expected prefix")
    if len(rows) != len(telemetry) or len(rows) > RETRY_CONTRACT["additional_generation_attempts"]:
        raise ValueError("partial corrected teacher rows and telemetry differ")

    initial_memory = base.memory_free_percent()
    if initial_memory < 20:
        raise RuntimeError("memory pressure is unsafe before corrected teacher generation")
    scenarios = {(item.task, item.variant): item for item in base.SCENARIOS}
    for index, record_id in enumerate(expected_ids[len(rows) :], start=len(rows) + 1):
        task, language_mode, variant_text = record_id.removeprefix("lumi-teacher-").rsplit("-", 2)
        scenario = scenarios[(task, int(variant_text))]
        row, item_telemetry = generate_one_retry(
            record_id,
            scenario,
            language_mode,
            endpoint=endpoint,
            failure_path=private / f"rejected-{index:03d}-{record_id}.json",
        )
        rows.append(row)
        telemetry.append(item_telemetry)
        base.write_jsonl(partial_rows, rows)
        base.write_jsonl(partial_telemetry, telemetry)
        if index % 10 == 0:
            free = base.memory_free_percent()
            if free < 15:
                raise RuntimeError(
                    "memory pressure became unsafe during corrected teacher generation"
                )
            elapsed_minutes = (time.monotonic() - started_monotonic) / 60
            if elapsed_minutes > RETRY_CONTRACT["maximum_runtime_minutes"]:
                raise RuntimeError("corrected teacher retry exceeded runtime authorization")
            print(f"generated {index}/80; memory_free={free}%", flush=True)

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
        maximum_supervised_tokens=RETRY_CONTRACT["maximum_retained_supervised_tokens"],
    )
    owner_sample = base.select_owner_sample(rows)
    owner_sample_sha256 = base.sha256_value(owner_sample)
    base.write_jsonl(private / "owner-review-sample-32.jsonl", owner_sample)
    review.mkdir(parents=True, exist_ok=True)
    (review / "owner-review-32.html").write_text(
        build_owner_review_html(owner_sample, sample_sha256=owner_sample_sha256),
        encoding="utf-8",
    )

    durations = [float(item["elapsed_seconds"]) for item in telemetry]
    report = {
        "schema_version": base.REPORT_SCHEMA_VERSION,
        "status": "verified_private_candidate_pending_owner_review",
        "production_eligible": False,
        "candidate_registry_changed": False,
        "dataset_sealed": False,
        "sft_started": False,
        "external_api_used": False,
        "human_authorship_claimed": False,
        "authorization_file_sha256": file_sha256(artifact / "authorization.json"),
        "plan_file_sha256": file_sha256(plan_path),
        "predecessor_failure_report_sha256": plan["predecessor_failure"]["sha256"],
        "retry": {
            "additional_calls_used": len(telemetry),
            "failed_outputs_retained": len(list(private.glob("rejected-*.json"))),
            "prompt_contract": "lumi.first-party-teacher-prompt.v0.2",
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
    }
    base.write_json(evidence / "run-report.json", report)
    return report
