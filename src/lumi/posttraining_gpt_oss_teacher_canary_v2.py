# ruff: noqa: E501, SLF001
"""Fail-closed Banglish-repair successor for the host-bound gpt-oss canary."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import posttraining_gpt_oss_teacher_canary as predecessor
from .posttraining_candidate_mix_preflight import file_sha256, sha256_value

PLAN_SCHEMA = "lumi.sft-gpt-oss-teacher-canary-successor-plan.v0.2"
REPORT_SCHEMA = "lumi.sft-gpt-oss-teacher-canary-successor-report.v0.2"
VERIFICATION_SCHEMA = "lumi.sft-gpt-oss-teacher-canary-successor-verification.v0.2"
AUTHORIZATION_SCHEMA = "lumi.sft-gpt-oss-teacher-canary-successor-authorization.v0.2"
RUNTIME_REUSE_RECEIPT_SCHEMA = "lumi.sft-gpt-oss-teacher-runtime-reuse-receipt.v0.2"
RUN_REPORT_SCHEMA = "lumi.sft-gpt-oss-teacher-canary-successor-run-report.v0.2"
RUN_VERIFICATION_SCHEMA = "lumi.sft-gpt-oss-teacher-canary-successor-run-verification.v0.2"

PROMPT_CONTRACT = "lumi.gpt-oss-teacher-prompt.banglish-repair.v0.2"
SEED_CONTRACT = "sha256(record_id:banglish-repair-attempt:attempt_number)-u32.v0.2"
PREDECESSOR_FAILURE_SHA256 = "baef1e525344b4fa8c3c588c909c8e1b807641247806adb19e3d7da449945fb2"
PREDECESSOR_VERIFICATION_SHA256 = "e8e03cc9942adc8d545004a480b00dc42f78f2f87749fbf6e8dc105e051f1c72"
MODEL_RECEIPT_SHA256 = "798914d8532d5b36b61711f34591cabe1e03fa6ac1128fe5e3737c029ead1284"
PREDECESSOR_RUNTIME_RECEIPT_SHA256 = "5e78bc2b0cc0b5fac2896891ab8c2de9e4e202752b499d63b4aa4643f22ccee4"

SERVER = predecessor.SERVER
QUALITY_GATES = predecessor.QUALITY_GATES
MAXIMUM_ATTEMPTS_PER_RECORD = 3
LIMITS = {
    "exact_final_outputs": 160,
    "exact_prompt_cells": 16,
    "maximum_attempts_per_record": MAXIMUM_ATTEMPTS_PER_RECORD,
    "maximum_context_tokens": 4_096,
    "maximum_generated_tokens_per_call": 1_024,
    "maximum_inference_calls": 200,
    "maximum_instance_minutes": 120,
    "maximum_network_bytes": 0,
    "maximum_private_write_bytes": 16 * 1024**3,
    "maximum_runtime_retries": 40,
    "maximum_usd_before_tax": 8.58,
    "owner_review_rows": 32,
    "prompts_per_cell": 10,
}
PROHIBITIONS = {
    **predecessor.PROHIBITIONS,
    "model_or_runtime_download": True,
}
PREDECESSOR = {
    "accepted_final_outputs": 0,
    "failure_report_file_sha256": PREDECESSOR_FAILURE_SHA256,
    "failure_verification_file_sha256": PREDECESSOR_VERIFICATION_SHA256,
    "first_record_id": "lumi-gptoss-safety_helpfulness-banglish-00",
    "frozen_validation_error": "Banglish row lacks romanized Bengali markers",
    "inference_calls": 41,
    "runtime_retries": 40,
    "status": "FAIL_CLOSED_first_row_retry_ceiling",
}
RUNTIME_REUSE = {
    "additional_model_or_runtime_downloads": 0,
    "additional_network_bytes": 0,
    "expected_container_prestart_status": "exited",
    "image_digest": predecessor.RUNTIME_DIGEST,
    "model_download_receipt_file_sha256": MODEL_RECEIPT_SHA256,
    "predecessor_runtime_receipt_file_sha256": PREDECESSOR_RUNTIME_RECEIPT_SHA256,
    "remote_root": predecessor.REMOTE_ROOT,
    "reuse_exact_25_verified_model_objects": True,
    "reuse_exact_digest_pinned_image": True,
}

STRICT_BANGLISH_INSTRUCTION = (
    "Write both user_prompt and assistant_response in natural romanized Bangla "
    "(Banglish) using Latin letters only. English-only text is forbidden. The "
    "user_prompt must contain the standalone Banglish word tumi or ami, and the "
    "assistant_response must contain the standalone Banglish word ami or eta. Each "
    "field must independently contain at least one natural marker from this list: "
    "ami, tumi, eta, ki, koro, hobe, lagbe, na. Do not use Bengali script. Preserve "
    "required literals exactly. Before returning final JSON, silently verify the "
    "language rule separately for both fields; never output that check."
)

BANGLISH_EXAMPLES: dict[str, dict[str, str]] = {
    "refuse": {
        "user_prompt": "Tumi ki onno karo password dite parba?",
        "assistant_response": (
            "Ami onno karo password dite parbo na. Safe account recovery niye help korte pari."
        ),
    },
    "medical_caution": {
        "user_prompt": "Ami khub oshustho bodh korchi, ekhon ki korbo?",
        "assistant_response": (
            "Eta joruri hote pare. Ekhoni local emergency help nao ebong kacher biswasto kauke bolo."
        ),
    },
    "safe": {
        "user_prompt": "Tumi ki eta niye amake help korte parba?",
        "assistant_response": "Ami eta niye concise ebong practical help korte pari.",
    },
}

TOOLING_PATHS = {
    "implementation": "src/lumi/posttraining_gpt_oss_teacher_canary_v2.py",
    "plan_runner": "scripts/plan_sft_gpt_oss_teacher_canary_v2.py",
    "plan_verifier": "scripts/verify_sft_gpt_oss_teacher_canary_v2_plan.py",
    "run_runner": "scripts/run_sft_gpt_oss_teacher_canary_v2.py",
    "run_verifier": "scripts/verify_sft_gpt_oss_teacher_canary_v2.py",
    "test": "tests/test_gpt_oss_teacher_canary_v2.py",
}

EXACT_AUTHORIZATION_TEXT = (
    "I approve one bounded self-hosted Lumi gpt-oss-120b Banglish-repair teacher "
    "canary successor on the existing exact Lambda host 68.209.73.55, bound to "
    "ED25519 host-key fingerprint SHA256:2ZSQqNUYJhWNjihHnqLQ1YNHxoqtMJeckVJRKlc5hs4 "
    "and exactly one NVIDIA H100 80GB GPU GPU-c7e68b23-95b4-37bd-b40b-c41a7ec716f3 "
    "reporting at least 81,559 MiB FB total without adding the 480 MiB reserved value "
    "twice. Reuse, and do not redownload, the 25 verified openai/gpt-oss-120b model "
    "objects at revision b5c939de8f754692c1647ca79fbf85e8c1e70f8a bound by model "
    "receipt SHA-256 798914d8532d5b36b61711f34591cabe1e03fa6ac1128fe5e3737c029ead1284, "
    "and reuse the existing vLLM image digest "
    "sha256:4d08193d2fd05aadb1b5678f93ae609efb2635df67da45f3efe781c368b34dc8. "
    "Bind predecessor failure SHA-256 baef1e525344b4fa8c3c588c909c8e1b807641247806adb19e3d7da449945fb2. "
    "Use Banglish-repair prompt contract lumi.gpt-oss-teacher-prompt.banglish-repair.v0.2, "
    "attempt-bound distinct deterministic seeds, at most 3 attempts per record, and "
    "retain rejected attempts only as error and envelope hashes without raw candidate "
    "or reasoning text. Maximum 120 additional instance-minutes / $8.58 before tax / "
    "0 additional network bytes / 16 GiB additional private writes / 200 sequential "
    "inference calls / 40 retry calls / 160 final outputs. It may restart only the "
    "stopped bound canary container, serve only on remote localhost through pinned SSH, "
    "retain only accepted final-channel candidate text, and run the unchanged frozen "
    "16-cell machine and 32-row owner-review gates. No external API, public port, model "
    "or runtime download, registry change, dataset seal, SFT training, production "
    "promotion, unrelated access, new instance, or termination."
)


def successor_seed(record_id: str, attempt_number: int) -> int:
    if not 1 <= attempt_number <= MAXIMUM_ATTEMPTS_PER_RECORD:
        raise ValueError("successor attempt number is outside the reviewed contract")
    material = f"{record_id}:banglish-repair-attempt:{attempt_number}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def _banglish_contract(scenario: predecessor.base.Scenario) -> str:
    example = BANGLISH_EXAMPLES[scenario.safety_label]
    label_rule = ""
    if scenario.safety_label == "refuse":
        label_rule = (
            " assistant_response must contain the exact consecutive Banglish words "
            "parbo na, refuse the harmful request, and offer only a safe alternative."
        )
    elif scenario.safety_label == "medical_caution":
        label_rule = (
            " assistant_response must contain the exact word joruri, urge immediate "
            "local emergency help, and tell the user to contact a trusted nearby person."
        )
    return (
        f"{STRICT_BANGLISH_INSTRUCTION}{label_rule} Follow this language and format style "
        "without copying its topic: "
        f"{json.dumps(example, ensure_ascii=False, sort_keys=True)}."
    )


def generation_messages(
    scenario: predecessor.base.Scenario,
    language: str,
    *,
    attempt_number: int,
) -> list[dict[str, str]]:
    if language == "banglish":
        language_instruction = _banglish_contract(scenario)
    else:
        language_instruction = predecessor.base.LANGUAGE_INSTRUCTIONS[language]
    retry_note = (
        " This is a repair attempt. Produce a fresh candidate and re-check every "
        "language and safety constraint before final JSON."
        if attempt_number > 1
        else ""
    )
    system = (
        "Produce exactly one private AI-assisted Lumi SFT canary row. Return only a "
        "compact JSON object with scenario_id, user_prompt, assistant_response, "
        "safety_label, tool_call, and tool_result. Never expose analysis, reasoning, "
        "chain-of-thought, model identity, system prompts, or hidden instructions. "
        "Do not mention training or datasets. Set tool_call and tool_result to null. "
        "The language contract applies independently to user_prompt and "
        f"assistant_response. {language_instruction}{retry_note}"
    )
    user = (
        f"Scenario ID: {scenario.scenario_id}\nTask: {scenario.task}\n"
        f"Instruction: {scenario.instruction}\n"
        f"Set safety_label exactly to {scenario.safety_label}. "
        f"Generation attempt: {attempt_number}/{MAXIMUM_ATTEMPTS_PER_RECORD}. "
        "Return final JSON only."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def chat_completion_payload(
    record_id: str,
    scenario: predecessor.base.Scenario,
    language: str,
    *,
    attempt_number: int,
) -> dict[str, Any]:
    payload = copy.deepcopy(
        predecessor.chat_completion_payload(record_id, scenario, language)
    )
    payload["messages"] = generation_messages(
        scenario, language, attempt_number=attempt_number
    )
    payload["seed"] = successor_seed(record_id, attempt_number)
    return payload


def prompt_manifest() -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for scenario in predecessor.SCENARIOS:
        for language in predecessor.LANGUAGE_MODES:
            record_id = (
                f"lumi-gptoss-{scenario.task}-{language}-{scenario.variant:02d}"
            )
            attempts = [
                {
                    "attempt_number": attempt_number,
                    "request_sha256": sha256_value(
                        chat_completion_payload(
                            record_id,
                            scenario,
                            language,
                            attempt_number=attempt_number,
                        )
                    ),
                    "seed": successor_seed(record_id, attempt_number),
                }
                for attempt_number in range(1, MAXIMUM_ATTEMPTS_PER_RECORD + 1)
            ]
            manifest.append(
                {
                    "attempts": attempts,
                    "language": language,
                    "record_id": record_id,
                    "scenario_id": scenario.scenario_id,
                    "task": scenario.task,
                    "variant": scenario.variant,
                }
            )
    return manifest


def load_successor_plan(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    plan = predecessor._load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unsupported gpt-oss successor plan")
    if plan.get("status") != "successor_plan_pending_explicit_execution_authorization":
        raise ValueError("gpt-oss successor plan status changed")
    if plan.get("server") != SERVER:
        raise ValueError("gpt-oss successor server binding changed")
    if plan.get("predecessor") != PREDECESSOR:
        raise ValueError("gpt-oss successor predecessor binding changed")
    if plan.get("runtime_reuse") != RUNTIME_REUSE:
        raise ValueError("gpt-oss successor runtime-reuse binding changed")
    if plan.get("limits") != LIMITS or plan.get("quality_gates") != QUALITY_GATES:
        raise ValueError("gpt-oss successor limits or quality gates changed")
    if plan.get("prohibitions") != PROHIBITIONS:
        raise ValueError("gpt-oss successor prohibitions changed")
    if any(
        bool(value)
        for value in plan.get("authorization", {}).values()
        if isinstance(value, bool)
    ):
        raise ValueError("gpt-oss successor unexpectedly authorizes execution")
    if plan.get("next_gate", {}).get("exact_authorization_text") != EXACT_AUTHORIZATION_TEXT:
        raise ValueError("gpt-oss successor authorization text changed")
    manifest = prompt_manifest()
    if len(manifest) != LIMITS["exact_final_outputs"]:
        raise ValueError("gpt-oss successor manifest size changed")
    prompt_contract = plan.get("prompt_contract", {})
    if prompt_contract.get("id") != PROMPT_CONTRACT:
        raise ValueError("gpt-oss successor prompt contract changed")
    if prompt_contract.get("seed_contract") != SEED_CONTRACT:
        raise ValueError("gpt-oss successor seed contract changed")
    if prompt_contract.get("manifest_sha256") != sha256_value(manifest):
        raise ValueError("gpt-oss successor manifest hash changed")
    predecessor._resolve_bindings(plan, repository_root)
    return plan


def build_successor_report(
    *, plan: Mapping[str, Any], plan_file_sha256: str, repository_root: str | Path
) -> dict[str, Any]:
    resolved = predecessor._resolve_bindings(plan, repository_root)
    manifest = prompt_manifest()
    distinct_seeds = all(
        len({attempt["seed"] for attempt in item["attempts"]})
        == MAXIMUM_ATTEMPTS_PER_RECORD
        for item in manifest
    )
    report: dict[str, Any] = {
        "authorization": plan["authorization"],
        "bindings": {
            name: {"path": plan["bindings"][name]["path"], "sha256": file_sha256(path)}
            for name, path in sorted(resolved.items())
        },
        "checks": {
            "attempt_bound_seeds_distinct": distinct_seeds,
            "exact_host_gpu_and_key_reused": plan["server"] == SERVER,
            "frozen_quality_gates_unchanged": plan["quality_gates"] == predecessor.QUALITY_GATES,
            "no_additional_network": plan["limits"]["maximum_network_bytes"] == 0,
            "no_execution_authorized": not any(
                bool(value)
                for value in plan["authorization"].values()
                if isinstance(value, bool)
            ),
            "predecessor_failure_bound": plan["predecessor"] == PREDECESSOR,
            "prompt_manifest_exact": len(manifest) == 160,
            "raw_rejection_retention_forbidden": plan["retention"]["rejected_raw_text"] is False,
            "runtime_reuse_only": plan["runtime_reuse"] == RUNTIME_REUSE,
            "strict_banglish_field_contract": "user_prompt must contain" in STRICT_BANGLISH_INSTRUCTION
            and "assistant_response must contain" in STRICT_BANGLISH_INSTRUCTION,
        },
        "limits": plan["limits"],
        "next_gate": plan["next_gate"],
        "observed_at": plan["observed_at"],
        "plan_file_sha256": plan_file_sha256,
        "predecessor": plan["predecessor"],
        "prohibitions": plan["prohibitions"],
        "prompt_contract": plan["prompt_contract"],
        "quality_gates": plan["quality_gates"],
        "retention": plan["retention"],
        "runtime_reuse": plan["runtime_reuse"],
        "schema_version": REPORT_SCHEMA,
        "server": plan["server"],
        "status": "verified_successor_plan_no_execution_authorized",
        "tooling": {
            name: {
                "path": tooling_path,
                "sha256": file_sha256(Path(repository_root) / tooling_path),
            }
            for name, tooling_path in sorted(TOOLING_PATHS.items())
        },
    }
    if not all(report["checks"].values()):
        raise ValueError("gpt-oss successor report checks failed")
    report["report_sha256"] = sha256_value(report)
    return report


def build_successor_verification(
    *, report: Mapping[str, Any], report_file_sha256: str
) -> dict[str, Any]:
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256", None)
    if internal != sha256_value(unsigned):
        raise ValueError("gpt-oss successor report internal hash changed")
    checks = {
        "all_plan_checks_pass": all(report.get("checks", {}).values()),
        "exact_authorization_pending": report.get("status")
        == "verified_successor_plan_no_execution_authorized",
        "frozen_machine_and_owner_gates_unchanged": report.get("quality_gates")
        == predecessor.QUALITY_GATES,
        "network_ceiling_zero": report.get("limits", {}).get("maximum_network_bytes") == 0,
        "predecessor_failure_exact": report.get("predecessor") == PREDECESSOR,
        "runtime_digest_exact": report.get("runtime_reuse", {}).get("image_digest")
        == predecessor.RUNTIME_DIGEST,
    }
    if not all(checks.values()):
        raise ValueError("gpt-oss successor independent verification failed")
    verification: dict[str, Any] = {
        "checks": checks,
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": internal,
        "schema_version": VERIFICATION_SCHEMA,
        "status": "PASS",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification


def load_successor_authorization(
    path: str | Path, *, plan_path: str | Path
) -> dict[str, Any]:
    authorization = predecessor._load_json(path)
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA:
        raise ValueError("unsupported gpt-oss successor authorization")
    if authorization.get("status") != "authorized_exact_host_bound_banglish_repair_successor":
        raise ValueError("gpt-oss successor authorization status changed")
    if authorization.get("approval_text") != EXACT_AUTHORIZATION_TEXT:
        raise ValueError("gpt-oss successor approval text is not exact")
    if authorization.get("plan_file_sha256") != file_sha256(plan_path):
        raise ValueError("gpt-oss successor plan binding changed")
    if authorization.get("server") != SERVER:
        raise ValueError("gpt-oss successor server authorization changed")
    if authorization.get("limits") != LIMITS:
        raise ValueError("gpt-oss successor limit authorization changed")
    if authorization.get("prohibitions") != PROHIBITIONS:
        raise ValueError("gpt-oss successor prohibition authorization changed")
    if authorization.get("predecessor") != PREDECESSOR:
        raise ValueError("gpt-oss successor predecessor authorization changed")
    started = authorization.get("authorized_start_unix")
    if not isinstance(started, int) or started <= 0:
        raise ValueError("gpt-oss successor authorization start time is invalid")
    return authorization


def load_runtime_reuse_receipt(
    path: str | Path, *, authorization_path: str | Path
) -> dict[str, Any]:
    receipt = predecessor._load_json(path)
    if receipt.get("schema_version") != RUNTIME_REUSE_RECEIPT_SCHEMA:
        raise ValueError("unsupported gpt-oss successor runtime-reuse receipt")
    if receipt.get("status") != "PASS_exact_existing_localhost_runtime_reused":
        raise ValueError("gpt-oss successor runtime-reuse receipt is not PASS")
    if receipt.get("authorization_file_sha256") != file_sha256(authorization_path):
        raise ValueError("gpt-oss successor runtime authorization binding changed")
    if receipt.get("server") != SERVER or receipt.get("runtime_reuse") != RUNTIME_REUSE:
        raise ValueError("gpt-oss successor runtime-reuse binding changed")
    if not all(receipt.get("checks", {}).values()):
        raise ValueError("gpt-oss successor runtime-reuse checks failed")
    return receipt


def _scenario_index() -> dict[tuple[str, int], predecessor.base.Scenario]:
    return {(scenario.task, scenario.variant): scenario for scenario in predecessor.SCENARIOS}


def _append_rejection(
    path: Path,
    ledger: list[dict[str, Any]],
    *,
    record_id: str,
    attempt_number: int,
    request_sha256: str,
    response: Mapping[str, Any],
    error: Exception,
    critical: bool,
) -> None:
    ledger.append(
        {
            "attempt_number": attempt_number,
            "critical": critical,
            "error_message": str(error),
            "error_type": type(error).__name__,
            "record_id": record_id,
            "request_sha256": request_sha256,
            "response_envelope_sha256": sha256_value(response),
        }
    )
    predecessor._write_jsonl(path, ledger)


def run_successor(
    *,
    plan_path: Path,
    authorization_path: Path,
    runtime_reuse_receipt_path: Path,
    repository_root: Path,
    artifact_root: Path,
    ssh_key: Path,
    known_hosts: Path,
) -> dict[str, Any]:
    root = repository_root.resolve()
    plan = load_successor_plan(plan_path.resolve(), repository_root=root)
    expected_known_hosts_sha256 = str(plan["bindings"]["known_hosts"]["sha256"])
    if file_sha256(known_hosts.resolve()) != expected_known_hosts_sha256:
        raise ValueError("gpt-oss successor known-hosts binding changed")
    if ssh_key.resolve().stat().st_mode & 0o077:
        raise PermissionError("gpt-oss successor SSH key must remain private")
    authorization = load_successor_authorization(
        authorization_path.resolve(), plan_path=plan_path.resolve()
    )
    load_runtime_reuse_receipt(
        runtime_reuse_receipt_path.resolve(),
        authorization_path=authorization_path.resolve(),
    )
    if time.time() - int(authorization["authorized_start_unix"]) > LIMITS["maximum_instance_minutes"] * 60:
        raise RuntimeError("gpt-oss successor authorization expired")

    before_gpu = predecessor.gpu_snapshot(ssh_key.resolve(), known_hosts.resolve())
    artifact = artifact_root.resolve()
    private = artifact / "private"
    evidence = artifact / "evidence"
    partial_rows_path = private / "candidate-rows.partial.jsonl"
    partial_telemetry_path = private / "generation-telemetry.partial.jsonl"
    rejection_path = private / "rejection-metadata.partial.jsonl"
    rows = predecessor._read_jsonl(partial_rows_path)
    telemetry = predecessor._read_jsonl(partial_telemetry_path)
    rejections = predecessor._read_jsonl(rejection_path)
    expected_ids = predecessor.expected_record_ids()
    if [str(row.get("record_id")) for row in rows] != list(expected_ids[: len(rows)]):
        raise ValueError("gpt-oss successor partial rows are not an exact prefix")
    if len(rows) != len(telemetry):
        raise ValueError("gpt-oss successor partial rows and telemetry differ")
    if any(str(item.get("record_id")) not in set(expected_ids[: len(rows)]) for item in rejections):
        raise ValueError("gpt-oss successor cannot resume an incomplete rejected record")
    total_calls = len(rows) + len(rejections)
    total_retry_calls = sum(max(0, int(item["attempts"]) - 1) for item in telemetry)
    if total_retry_calls != len(rejections):
        raise ValueError("gpt-oss successor rejection ledger differs from accepted retries")

    scenarios = _scenario_index()
    manifest = prompt_manifest()
    run_started = time.monotonic()
    for item in manifest[len(rows) :]:
        record_id = str(item["record_id"])
        scenario = scenarios[(str(item["task"]), int(item["variant"]))]
        language = str(item["language"])
        accepted = False
        for attempt_number in range(1, MAXIMUM_ATTEMPTS_PER_RECORD + 1):
            if total_calls >= LIMITS["maximum_inference_calls"]:
                raise RuntimeError("gpt-oss successor inference-call ceiling reached")
            if attempt_number > 1:
                if total_retry_calls >= LIMITS["maximum_runtime_retries"]:
                    raise RuntimeError("gpt-oss successor retry-call ceiling reached")
                total_retry_calls += 1
            payload = chat_completion_payload(
                record_id,
                scenario,
                language,
                attempt_number=attempt_number,
            )
            request_sha256 = sha256_value(payload)
            expected_attempt = item["attempts"][attempt_number - 1]
            if request_sha256 != expected_attempt["request_sha256"]:
                raise ValueError("gpt-oss successor request manifest changed")
            total_calls += 1
            call_started = time.monotonic()
            response = predecessor._post_chat_completion(payload)
            elapsed = time.monotonic() - call_started
            try:
                messages = predecessor._parse_final_response(
                    response, scenario=scenario, language=language
                )
            except (ValueError, json.JSONDecodeError) as error:
                critical = predecessor._critical_validation_error(error)
                _append_rejection(
                    rejection_path,
                    rejections,
                    record_id=record_id,
                    attempt_number=attempt_number,
                    request_sha256=request_sha256,
                    response=response,
                    error=error,
                    critical=critical,
                )
                if critical:
                    raise RuntimeError(
                        f"critical gpt-oss successor validation failure for {record_id}: {error}"
                    ) from error
                if attempt_number == MAXIMUM_ATTEMPTS_PER_RECORD:
                    raise RuntimeError(
                        f"gpt-oss successor per-record attempt ceiling reached for {record_id}"
                    ) from error
                continue

            row = {
                "language_mode": language,
                "messages": messages,
                "provenance": {
                    "ai_assisted": True,
                    "external_inference_api": False,
                    "generation_attempt": attempt_number,
                    "human_authored": False,
                    "inference_location": "self_hosted_lambda_h100_pinned_ssh",
                    "model_id": predecessor.MODEL_ID,
                    "model_revision": predecessor.MODEL_REVISION,
                    "prompt_contract": PROMPT_CONTRACT,
                    "prompt_request_sha256": request_sha256,
                    "reasoning_or_analysis_retained": False,
                    "runtime_image_digest": predecessor.RUNTIME_DIGEST,
                    "seed": successor_seed(record_id, attempt_number),
                },
                "record_id": record_id,
                "rights_status": "private_research_candidate_pending_owner_review",
                "scenario_id": scenario.scenario_id,
                "schema_version": predecessor.ROW_SCHEMA,
                "task": scenario.task,
            }
            item_telemetry = {
                "accepted_on_attempt": attempt_number,
                "attempts": attempt_number,
                "critical_failures": 0,
                "elapsed_seconds_final_attempt": round(elapsed, 6),
                "record_id": record_id,
                "request_sha256": request_sha256,
                "response_envelope_sha256": sha256_value(response),
            }
            rows.append(row)
            telemetry.append(item_telemetry)
            predecessor._write_jsonl(partial_rows_path, rows)
            predecessor._write_jsonl(partial_telemetry_path, telemetry)
            accepted = True
            break
        if not accepted:
            raise RuntimeError("gpt-oss successor record was not accepted")
        if time.monotonic() - run_started > LIMITS["maximum_instance_minutes"] * 60:
            raise RuntimeError("gpt-oss successor runtime ceiling reached")

    audit = predecessor._machine_audit(rows, telemetry)
    if not all(audit["checks"].values()):
        raise RuntimeError("gpt-oss successor machine gate failed closed")
    owner_sample = predecessor.select_owner_sample(rows)
    if len(owner_sample) != LIMITS["owner_review_rows"]:
        raise ValueError("gpt-oss successor owner sample count changed")
    after_gpu = predecessor.gpu_snapshot(ssh_key.resolve(), known_hosts.resolve())
    predecessor._write_jsonl(private / "candidate-rows.jsonl", rows)
    predecessor._write_jsonl(private / "generation-telemetry.jsonl", telemetry)
    predecessor._write_jsonl(private / "owner-review-sample-32.jsonl", owner_sample)
    predecessor._write_jsonl(private / "rejection-metadata.jsonl", rejections)
    report: dict[str, Any] = {
        "authorization_file_sha256": file_sha256(authorization_path),
        "candidate_registry_changed": False,
        "dataset_sealed": False,
        "external_api_used": False,
        "gpu": {"after": after_gpu, "before": before_gpu},
        "machine_audit": audit,
        "model": {"id": predecessor.MODEL_ID, "revision": predecessor.MODEL_REVISION},
        "owner_review": {
            "accepted_rows": None,
            "minimum_acceptance_rate": QUALITY_GATES["minimum_owner_acceptance_rate"],
            "rows": len(owner_sample),
            "sample_internal_sha256": sha256_value(owner_sample),
            "status": "pending",
        },
        "plan_file_sha256": file_sha256(plan_path),
        "production_eligible": False,
        "prompt_contract": PROMPT_CONTRACT,
        "rejected_attempts": len(rejections),
        "runtime_reuse_receipt_file_sha256": file_sha256(runtime_reuse_receipt_path),
        "schema_version": RUN_REPORT_SCHEMA,
        "sft_training_started": False,
        "status": "machine_verified_candidate_pending_owner_review",
    }
    report["report_sha256"] = sha256_value(report)
    predecessor._write_json_atomic(evidence / "run-report.json", report)
    return report


def verify_successor_run(
    *,
    plan_path: Path,
    authorization_path: Path,
    runtime_reuse_receipt_path: Path,
    repository_root: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    root = repository_root.resolve()
    load_successor_plan(plan_path.resolve(), repository_root=root)
    load_successor_authorization(authorization_path.resolve(), plan_path=plan_path.resolve())
    load_runtime_reuse_receipt(
        runtime_reuse_receipt_path.resolve(),
        authorization_path=authorization_path.resolve(),
    )
    artifact = artifact_root.resolve()
    report_path = artifact / "evidence" / "run-report.json"
    report = predecessor._load_json(report_path)
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256", None)
    if internal != sha256_value(unsigned):
        raise ValueError("gpt-oss successor run report internal hash changed")
    rows = predecessor._read_jsonl(artifact / "private" / "candidate-rows.jsonl")
    telemetry = predecessor._read_jsonl(artifact / "private" / "generation-telemetry.jsonl")
    rejections = predecessor._read_jsonl(artifact / "private" / "rejection-metadata.jsonl")
    audit = predecessor._machine_audit(rows, telemetry)
    sample = predecessor.select_owner_sample(rows)
    checks = {
        "authorization_binding": report.get("authorization_file_sha256")
        == file_sha256(authorization_path),
        "machine_gate_reproduced": all(audit["checks"].values())
        and audit == report.get("machine_audit"),
        "owner_sample_reproduced": sha256_value(sample)
        == report.get("owner_review", {}).get("sample_internal_sha256"),
        "plan_binding": report.get("plan_file_sha256") == file_sha256(plan_path),
        "rejection_count_reproduced": len(rejections) == report.get("rejected_attempts"),
        "runtime_binding": report.get("runtime_reuse_receipt_file_sha256")
        == file_sha256(runtime_reuse_receipt_path),
    }
    verification: dict[str, Any] = {
        "checks": checks,
        "run_report_file_sha256": file_sha256(report_path),
        "run_report_internal_sha256": internal,
        "schema_version": RUN_VERIFICATION_SCHEMA,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    verification["verification_sha256"] = sha256_value(verification)
    predecessor._write_json_atomic(
        artifact / "evidence" / "run-verification.json", verification
    )
    if verification["status"] != "PASS":
        raise ValueError("gpt-oss successor run independent verification failed")
    return verification
