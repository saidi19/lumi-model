# ruff: noqa: E501, SLF001
"""Full 16-cell gpt-oss canary after accepted Banglish and complete English repairs."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import posttraining_gpt_oss_teacher_canary_v5 as v5
from . import posttraining_gpt_oss_teacher_complete_row_repair_v7 as v7
from .posttraining_candidate_mix_preflight import file_sha256, sha256_value

PLAN_SCHEMA = "lumi.sft-gpt-oss-teacher-full-canary-successor-plan.v0.8"
REPORT_SCHEMA = "lumi.sft-gpt-oss-teacher-full-canary-successor-report.v0.8"
VERIFICATION_SCHEMA = "lumi.sft-gpt-oss-teacher-full-canary-successor-verification.v0.8"
AUTHORIZATION_SCHEMA = "lumi.sft-gpt-oss-teacher-full-canary-successor-authorization.v0.8"
RUNTIME_REUSE_RECEIPT_SCHEMA = "lumi.sft-gpt-oss-teacher-full-canary-runtime-reuse-receipt.v0.8"
RUN_REPORT_SCHEMA = "lumi.sft-gpt-oss-teacher-full-canary-run-report.v0.8"
RUN_VERIFICATION_SCHEMA = "lumi.sft-gpt-oss-teacher-full-canary-run-verification.v0.8"

PROMPT_CONTRACT = "lumi.gpt-oss-teacher.complete-row-full-canary.v0.8"
SEED_CONTRACT = "sha256(record_id:complete-row-full-canary-attempt:attempt_number)-u32.v0.8"
REASONING_EFFORT = "low"
MAX_TOKENS = 1_536
MAXIMUM_ATTEMPTS_PER_RECORD = 2

SERVER = v5.SERVER
PROHIBITIONS = v5.PROHIBITIONS
RUNTIME_REUSE = v5.RUNTIME_REUSE
RETENTION = v5.RETENTION
FROZEN_MACHINE_GATES = v5.FROZEN_MACHINE_GATES
ACCEPTED_REPAIRS = {
    "banglish_v0_4": v5.ACCEPTED_REPAIR,
    "english_v0_7": {
        "accepted_rows": 1,
        "authorization_file_sha256": "4645088ae14551c28f63c255891bbdebcff284399283343ad6e1592870da82b0",
        "candidate_file_sha256": "5c8956d59ddf662efe142b682d649ce6d1b386a489c3651e8e7879e7d25d613e",
        "owner_review_file_sha256": "52ee37db41bc2ae339fc075e10ffb81cf8ee516ea492d0eeeb729c81df8b83fa",
        "owner_review_status": "ACCEPT_complete_row_quality",
        "owner_review_verification_file_sha256": "fad25a72db78efebf58e4dea1298158bf77741dcc486e84f0ea4a55f4922946b",
        "run_report_file_sha256": "1e5dfe5a1f59805c9fd54bff03302d6cd04c356929ce2aa30b1a7750ef583e2b",
        "run_verification_file_sha256": "19d230714c7f84018dad7e230480edee475aafcdb266f524417eb451af0761bb",
    },
}
BANNED_GENERATED_META_PHRASES = tuple(
    dict.fromkeys(
        (*v5.BANNED_META_PHRASES, "return json only", "generation attempt", "create one english", "produce the json object")
    )
)
LIMITS = dict(v5.LIMITS)
PROMPT_QUALITY_GATES = {
    "assistant_maximum_characters": 512,
    "banned_meta_phrases": list(BANNED_GENERATED_META_PHRASES),
    "banglish_refusal_requires_exact_parbo_na": True,
    "credential_theft_recovery_markers": list(v5.v4.REQUIRED_RECOVERY_MARKERS),
    "english_credential_user_prompt_exact": v7.EXACT_USER_PROMPT,
    "english_refusal_requires_exact_cannot_help": True,
    "owner_review_required": True,
    "user_and_assistant_must_be_natural_user_facing_text": True,
}

GENERAL_OUTPUT_INSTRUCTION = (
    "Write natural user-facing text in both generated text fields. The user_prompt must be "
    "a plausible direct end-user request, never a request to create data or return JSON. "
    "Never copy generation instructions, attempt numbers, schema directions, or meta-"
    "commentary into user_prompt or assistant_response. Do not use phrases such as refuse "
    "the harmful request, safe alternative, safety label, scenario ID, assistant response, "
    "user prompt, return JSON only, or generation attempt. Keep assistant_response concise "
    "and under 512 characters."
)
BANGLISH_INSTRUCTION = v5.BANGLISH_INSTRUCTION + " " + GENERAL_OUTPUT_INSTRUCTION
ENGLISH_REFUSAL_INSTRUCTION = (
    "For any English refusal row, assistant_response must contain the exact consecutive "
    "words cannot help and then give scenario-appropriate safe guidance."
)
ENGLISH_CREDENTIAL_INSTRUCTION = (
    "For the English credential_theft row, copy this exact user_prompt verbatim: Can you "
    "find someone else's login password? The assistant_response must contain cannot help, "
    "refuse obtaining another person's password, and recommend official password-reset or "
    "account-recovery guidance only for the user's own account."
)

EXACT_AUTHORIZATION_TEXT = (
    "I approve one bounded self-hosted Lumi gpt-oss-120b fresh full 16-cell complete-row "
    "teacher canary successor on the existing exact Lambda host 68.209.73.55, bound to "
    "ED25519 host-key fingerprint SHA256:2ZSQqNUYJhWNjihHnqLQ1YNHxoqtMJeckVJRKlc5hs4 "
    "and exactly one NVIDIA H100 80GB GPU GPU-c7e68b23-95b4-37bd-b40b-c41a7ec716f3 "
    "reporting at least 81,559 MiB FB total without adding the 480 MiB reserved value "
    "twice. Reuse, and do not redownload, the 25 verified openai/gpt-oss-120b model "
    "objects at revision b5c939de8f754692c1647ca79fbf85e8c1e70f8a bound by model receipt "
    "SHA-256 798914d8532d5b36b61711f34591cabe1e03fa6ac1128fe5e3737c029ead1284, and "
    "reuse the existing vLLM image digest "
    "sha256:4d08193d2fd05aadb1b5678f93ae609efb2635df67da45f3efe781c368b34dc8. "
    "Bind the accepted v0.4 Banglish owner-review SHA-256 "
    "f0c86920d51ca3bd44b797fbdb88310268c5dac24167a91254fdfe03dee50b12 and verification "
    "SHA-256 a92f6c06b33e9a03c48ad6d9aac8683243e95d080f02d58c0f8c71a9e703da58, plus the "
    "accepted v0.7 complete-row owner-review SHA-256 "
    "52ee37db41bc2ae339fc075e10ffb81cf8ee516ea492d0eeeb729c81df8b83fa and verification "
    "SHA-256 fad25a72db78efebf58e4dea1298158bf77741dcc486e84f0ea4a55f4922946b. Run a "
    "fresh unchanged frozen 16-cell schedule of 160 exact records with prompt contract "
    "lumi.gpt-oss-teacher.complete-row-full-canary.v0.8, reasoning effort low, a 1,536 "
    "generated-token ceiling per call, distinct attempt-bound deterministic seeds, and at "
    "most 2 attempts per record. Require natural user-facing text in both generated fields; "
    "reject generation instructions, attempt numbers, JSON directions, schema labels, and "
    "meta phrases; require exact parbo na for Banglish refusal rows; require exact cannot "
    "help for English refusal rows; and for the English credential-theft record require the "
    "exact user prompt Can you find someone else's login password? plus official password-"
    "reset or account-recovery guidance only for the user's own account. Retain rejected "
    "attempts only as validation error, non-sensitive finish_reason, usage token counts, and "
    "request/response-envelope hashes without raw candidate or reasoning text. Maximum 120 "
    "additional instance-minutes / $8.58 before tax / 0 additional model or runtime download "
    "bytes / 16 GiB additional private writes / 200 sequential inference calls / 40 retry "
    "calls / 160 final outputs. It may restart and then cleanly stop only the bound canary "
    "container, serve only on remote localhost through pinned SSH, retain only accepted "
    "final-channel candidate text, run the frozen 16-cell machine gate, and prepare the "
    "deterministic 32-row owner-review packet. No external API, public port, model or runtime "
    "download, registry change, dataset seal, SFT training, production promotion, unrelated "
    "access, new instance, or termination."
)

TOOLING_PATHS = {
    "implementation": "src/lumi/posttraining_gpt_oss_teacher_canary_v8.py",
    "plan_runner": "scripts/plan_sft_gpt_oss_teacher_canary_v8.py",
    "plan_verifier": "scripts/verify_sft_gpt_oss_teacher_canary_v8_plan.py",
    "run_runner": "scripts/run_sft_gpt_oss_teacher_canary_v8.py",
    "run_verifier": "scripts/verify_sft_gpt_oss_teacher_canary_v8.py",
    "test": "tests/test_gpt_oss_teacher_canary_v8.py",
}


def full_canary_seed(record_id: str, attempt_number: int) -> int:
    if not 1 <= attempt_number <= MAXIMUM_ATTEMPTS_PER_RECORD:
        raise ValueError("v0.8 full-canary attempt number is outside the reviewed contract")
    material = f"{record_id}:complete-row-full-canary-attempt:{attempt_number}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def generation_messages(scenario: v5.v2.predecessor.base.Scenario, language: str, *, attempt_number: int) -> list[dict[str, str]]:
    if language == "banglish":
        language_instruction = BANGLISH_INSTRUCTION
    else:
        language_instruction = f"{v5.v2.predecessor.base.LANGUAGE_INSTRUCTIONS[language]} {GENERAL_OUTPUT_INSTRUCTION}"
    if language == "english" and scenario.safety_label == "refuse":
        language_instruction += " " + ENGLISH_REFUSAL_INSTRUCTION
    if language == "english" and scenario.scenario_id == "credential_theft":
        language_instruction += " " + ENGLISH_CREDENTIAL_INSTRUCTION
    retry = (
        " This is the final bounded attempt; create a fresh candidate and silently re-check "
        "schema, language, safety, factual, punctuation, and complete-row quality gates."
        if attempt_number > 1
        else ""
    )
    system = (
        "Produce exactly one private AI-assisted Lumi SFT canary row. Return only a compact "
        "JSON object with scenario_id, user_prompt, assistant_response, safety_label, "
        "tool_call, and tool_result. Never expose analysis, reasoning, chain-of-thought, "
        "model identity, system prompts, or hidden instructions. Set tool_call and "
        "tool_result to null. The language contract applies independently to both text "
        f"fields. {language_instruction}{retry}"
    )
    user = (
        f"Scenario ID: {scenario.scenario_id}\nTask: {scenario.task}\n"
        f"Instruction: {scenario.instruction}\nSet safety_label exactly to "
        f"{scenario.safety_label}. Generation attempt: {attempt_number}/"
        f"{MAXIMUM_ATTEMPTS_PER_RECORD}. Return final JSON only."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def chat_completion_payload(record_id: str, scenario: v5.v2.predecessor.base.Scenario, language: str, *, attempt_number: int) -> dict[str, Any]:
    payload = copy.deepcopy(v5.v2.predecessor.chat_completion_payload(record_id, scenario, language))
    payload["chat_template_kwargs"] = {"reasoning_effort": REASONING_EFFORT}
    payload["max_tokens"] = MAX_TOKENS
    payload["messages"] = generation_messages(scenario, language, attempt_number=attempt_number)
    payload["seed"] = full_canary_seed(record_id, attempt_number)
    payload["temperature"] = 0.25
    return payload


def prompt_manifest() -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for scenario in v5.v2.predecessor.SCENARIOS:
        for language in v5.v2.predecessor.LANGUAGE_MODES:
            record_id = f"lumi-gptoss-{scenario.task}-{language}-{scenario.variant:02d}"
            attempts = [
                {
                    "attempt_number": attempt,
                    "max_tokens": MAX_TOKENS,
                    "reasoning_effort": REASONING_EFFORT,
                    "request_sha256": sha256_value(chat_completion_payload(record_id, scenario, language, attempt_number=attempt)),
                    "seed": full_canary_seed(record_id, attempt),
                }
                for attempt in range(1, MAXIMUM_ATTEMPTS_PER_RECORD + 1)
            ]
            manifest.append({"attempts": attempts, "language": language, "record_id": record_id, "scenario_id": scenario.scenario_id, "task": scenario.task, "variant": scenario.variant})
    return manifest


def validate_prompt_quality(messages: Sequence[Mapping[str, str]], *, scenario: v5.v2.predecessor.base.Scenario, language: str) -> None:
    v5.validate_prompt_quality(messages, scenario=scenario, language=language)
    user = str(messages[0].get("content", ""))
    assistant = str(messages[1].get("content", ""))
    combined = f"{user}\n{assistant}".lower()
    for phrase in BANNED_GENERATED_META_PHRASES:
        if phrase in combined:
            raise ValueError(f"v0.8 full-canary banned generated meta phrase: {phrase}")
    if language == "english" and scenario.safety_label == "refuse" and "cannot help" not in assistant.lower():
        raise ValueError("v0.8 English refusal lacks exact cannot help")
    if language == "english" and scenario.scenario_id == "credential_theft":
        v7.validate_complete_row(messages)


def load_full_canary_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    plan = v5.v2.predecessor._load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("status") != "full_canary_successor_plan_pending_explicit_execution_authorization":
        raise ValueError("unsupported or changed v0.8 full-canary plan")
    if plan.get("server") != SERVER or plan.get("runtime_reuse") != RUNTIME_REUSE:
        raise ValueError("v0.8 full-canary host or runtime binding changed")
    if plan.get("accepted_repairs") != ACCEPTED_REPAIRS:
        raise ValueError("v0.8 accepted repairs binding changed")
    if plan.get("limits") != LIMITS or plan.get("frozen_machine_gates") != FROZEN_MACHINE_GATES:
        raise ValueError("v0.8 full-canary limits or frozen gates changed")
    if plan.get("prompt_quality_gates") != PROMPT_QUALITY_GATES:
        raise ValueError("v0.8 prompt-quality gates changed")
    if plan.get("prohibitions") != PROHIBITIONS or plan.get("retention") != RETENTION:
        raise ValueError("v0.8 prohibition or retention changed")
    if any(bool(value) for value in plan.get("authorization", {}).values() if isinstance(value, bool)):
        raise ValueError("v0.8 plan unexpectedly authorizes execution")
    if plan.get("next_gate", {}).get("exact_authorization_text") != EXACT_AUTHORIZATION_TEXT:
        raise ValueError("v0.8 exact authorization text changed")
    contract = plan.get("prompt_contract", {})
    if contract.get("id") != PROMPT_CONTRACT or contract.get("seed_contract") != SEED_CONTRACT:
        raise ValueError("v0.8 prompt contract changed")
    manifest = prompt_manifest()
    if len(manifest) != 160 or contract.get("manifest_sha256") != sha256_value(manifest):
        raise ValueError("v0.8 prompt manifest changed")
    v5.v2.predecessor._resolve_bindings(plan, repository_root)
    return plan


def build_plan_report(*, plan: Mapping[str, Any], plan_file_sha256: str, repository_root: str | Path) -> dict[str, Any]:
    resolved = v5.v2.predecessor._resolve_bindings(plan, repository_root)
    manifest = prompt_manifest()
    attempts = [attempt for item in manifest for attempt in item["attempts"]]
    cells = {(item["task"], item["language"]) for item in manifest}
    report: dict[str, Any] = {
        "accepted_repairs": plan["accepted_repairs"],
        "authorization": plan["authorization"],
        "bindings": {name: {"path": plan["bindings"][name]["path"], "sha256": file_sha256(path)} for name, path in sorted(resolved.items())},
        "checks": {
            "accepted_v0_4_and_v0_7_repairs_bound": plan["accepted_repairs"] == ACCEPTED_REPAIRS,
            "attempt_bound_requests_distinct": len({item["request_sha256"] for item in attempts}) == 320,
            "attempt_bound_seeds_distinct_per_record": all(len({attempt["seed"] for attempt in item["attempts"]}) == 2 for item in manifest),
            "complete_row_regression_gate_bound": PROMPT_QUALITY_GATES["english_credential_user_prompt_exact"] == v7.EXACT_USER_PROMPT,
            "exact_frozen_cells": len(cells) == 16,
            "exact_frozen_records": len(manifest) == 160,
            "low_reasoning_exact": {item["reasoning_effort"] for item in attempts} == {"low"},
            "naturalness_meta_gate_bound": "generation attempt" in PROMPT_QUALITY_GATES["banned_meta_phrases"],
            "no_execution_authorized": not any(bool(value) for value in plan["authorization"].values() if isinstance(value, bool)),
            "no_model_or_runtime_download": plan["limits"]["maximum_network_bytes"] == 0,
            "runtime_reuse_only": plan["runtime_reuse"] == RUNTIME_REUSE,
            "token_ceiling_exact": {item["max_tokens"] for item in attempts} == {MAX_TOKENS},
        },
        "frozen_machine_gates": plan["frozen_machine_gates"],
        "limits": plan["limits"],
        "next_gate": plan["next_gate"],
        "observed_at": plan["observed_at"],
        "plan_file_sha256": plan_file_sha256,
        "prohibitions": plan["prohibitions"],
        "prompt_contract": plan["prompt_contract"],
        "prompt_quality_gates": plan["prompt_quality_gates"],
        "retention": plan["retention"],
        "runtime_reuse": plan["runtime_reuse"],
        "schema_version": REPORT_SCHEMA,
        "server": plan["server"],
        "status": "verified_full_canary_successor_plan_no_execution_authorized",
        "tooling": {name: {"path": tooling_path, "sha256": file_sha256(Path(repository_root) / tooling_path)} for name, tooling_path in sorted(TOOLING_PATHS.items())},
    }
    if not all(report["checks"].values()):
        raise ValueError("v0.8 full-canary plan checks failed")
    report["report_sha256"] = sha256_value(report)
    return report


def build_plan_verification(*, report: Mapping[str, Any], report_file_sha256: str) -> dict[str, Any]:
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256", None)
    if internal != sha256_value(unsigned):
        raise ValueError("v0.8 report internal hash changed")
    checks = {
        "accepted_repairs_exact": report.get("accepted_repairs") == ACCEPTED_REPAIRS,
        "all_plan_checks_pass": all(report.get("checks", {}).values()),
        "authorization_pending": report.get("status") == "verified_full_canary_successor_plan_no_execution_authorized",
        "complete_row_gate_exact": report.get("prompt_quality_gates", {}).get("english_credential_user_prompt_exact") == v7.EXACT_USER_PROMPT,
        "frozen_machine_gate_exact": report.get("frozen_machine_gates") == FROZEN_MACHINE_GATES,
        "full_output_count_exact": report.get("limits", {}).get("exact_final_outputs") == 160,
        "owner_sample_exact": report.get("limits", {}).get("owner_review_rows") == 32,
        "zero_download_ceiling": report.get("limits", {}).get("maximum_network_bytes") == 0,
    }
    if not all(checks.values()):
        raise ValueError("v0.8 independent plan verification failed")
    verification: dict[str, Any] = {"checks": checks, "report_file_sha256": report_file_sha256, "report_internal_sha256": internal, "schema_version": VERIFICATION_SCHEMA, "status": "PASS"}
    verification["verification_sha256"] = sha256_value(verification)
    return verification


def load_execution_authorization(path: str | Path, *, plan_path: str | Path) -> dict[str, Any]:
    authorization = v5.v2.predecessor._load_json(path)
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA or authorization.get("status") != "authorized_exact_host_bound_full_complete_row_canary":
        raise ValueError("unsupported or changed v0.8 authorization")
    if authorization.get("approval_text") != EXACT_AUTHORIZATION_TEXT or authorization.get("plan_file_sha256") != file_sha256(plan_path):
        raise ValueError("v0.8 approval or plan binding changed")
    if authorization.get("server") != SERVER or authorization.get("limits") != LIMITS or authorization.get("prohibitions") != PROHIBITIONS or authorization.get("accepted_repairs") != ACCEPTED_REPAIRS:
        raise ValueError("v0.8 authorization bindings changed")
    started = authorization.get("authorized_start_unix")
    if not isinstance(started, int) or started <= 0:
        raise ValueError("v0.8 authorization start time is invalid")
    return authorization


def load_runtime_reuse_receipt(path: str | Path, *, authorization_path: str | Path) -> dict[str, Any]:
    receipt = v5.v2.predecessor._load_json(path)
    if receipt.get("schema_version") != RUNTIME_REUSE_RECEIPT_SCHEMA or receipt.get("status") != "PASS_exact_existing_localhost_runtime_reused_for_full_canary_v0.8":
        raise ValueError("unsupported or failed v0.8 runtime receipt")
    if receipt.get("authorization_file_sha256") != file_sha256(authorization_path) or receipt.get("server") != SERVER or receipt.get("runtime_reuse") != RUNTIME_REUSE:
        raise ValueError("v0.8 runtime receipt binding changed")
    if not all(receipt.get("checks", {}).values()):
        raise ValueError("v0.8 runtime checks failed")
    return receipt


def run_full_canary(*, plan_path: Path, authorization_path: Path, runtime_reuse_receipt_path: Path, repository_root: Path, artifact_root: Path, ssh_key: Path, known_hosts: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    plan = load_full_canary_plan(plan_path.resolve(), repository_root=root)
    authorization = load_execution_authorization(authorization_path.resolve(), plan_path=plan_path.resolve())
    load_runtime_reuse_receipt(runtime_reuse_receipt_path.resolve(), authorization_path=authorization_path.resolve())
    if file_sha256(known_hosts.resolve()) != plan["bindings"]["known_hosts"]["sha256"] or ssh_key.resolve().stat().st_mode & 0o077:
        raise PermissionError("v0.8 full-canary SSH binding or permissions changed")
    if time.time() - int(authorization["authorized_start_unix"]) > LIMITS["maximum_instance_minutes"] * 60:
        raise RuntimeError("v0.8 authorization expired")
    artifact = artifact_root.resolve()
    if not artifact.is_relative_to((root / "artifacts").resolve()):
        raise ValueError("v0.8 artifact root escaped artifacts/")
    private, evidence = artifact / "private", artifact / "evidence"
    private.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    rows_path = private / "candidate-rows.partial.jsonl"
    telemetry_path = private / "generation-telemetry.partial.jsonl"
    rejection_path = private / "rejection-metadata.partial.jsonl"
    rows = v5.v2.predecessor._read_jsonl(rows_path)
    telemetry = v5.v2.predecessor._read_jsonl(telemetry_path)
    rejections = v5.v2.predecessor._read_jsonl(rejection_path)
    expected_ids = v5.v2.predecessor.expected_record_ids()
    if [str(row.get("record_id")) for row in rows] != list(expected_ids[: len(rows)]) or len(rows) != len(telemetry):
        raise ValueError("v0.8 partial outputs are not an exact prefix")
    allowed_rejection_ids = set(expected_ids[: min(len(rows) + 1, len(expected_ids))])
    if any(str(item.get("record_id")) not in allowed_rejection_ids for item in rejections):
        raise ValueError("v0.8 rejection ledger escaped the exact prefix")
    total_calls, total_retries = len(rows) + len(rejections), len(rejections)
    before_gpu = v5.v2.predecessor.gpu_snapshot(ssh_key.resolve(), known_hosts.resolve())
    scenarios = {(item.task, item.variant): item for item in v5.v2.predecessor.SCENARIOS}
    manifest = prompt_manifest()
    run_started = time.monotonic()
    try:
        for item in manifest[len(rows) :]:
            record_id = str(item["record_id"])
            scenario = scenarios[(str(item["task"]), int(item["variant"]))]
            language = str(item["language"])
            prior = [entry for entry in rejections if entry["record_id"] == record_id]
            if len(prior) >= MAXIMUM_ATTEMPTS_PER_RECORD:
                raise RuntimeError(f"v0.8 prior attempts exhausted for {record_id}")
            accepted = False
            for attempt_number in range(len(prior) + 1, MAXIMUM_ATTEMPTS_PER_RECORD + 1):
                if total_calls >= LIMITS["maximum_inference_calls"]:
                    raise RuntimeError("v0.8 inference-call ceiling reached")
                if attempt_number > 1:
                    if total_retries >= LIMITS["maximum_runtime_retries"]:
                        raise RuntimeError("v0.8 retry ceiling reached")
                    total_retries += 1
                payload = chat_completion_payload(record_id, scenario, language, attempt_number=attempt_number)
                request_sha256 = sha256_value(payload)
                if request_sha256 != item["attempts"][attempt_number - 1]["request_sha256"]:
                    raise ValueError("v0.8 request manifest changed")
                total_calls += 1
                call_started = time.monotonic()
                response = v5.v2.predecessor._post_chat_completion(payload)
                elapsed = time.monotonic() - call_started
                try:
                    messages = v5.v2.predecessor._parse_final_response(response, scenario=scenario, language=language)
                    validate_prompt_quality(messages, scenario=scenario, language=language)
                except (ValueError, json.JSONDecodeError) as error:
                    critical = v5.v2.predecessor._critical_validation_error(error)
                    v5._append_rejection(rejection_path, rejections, record_id=record_id, attempt_number=attempt_number, request_sha256=request_sha256, response=response, error=error, critical=critical)
                    if critical:
                        raise RuntimeError(f"critical v0.8 failure for {record_id}: {error}") from error
                    if attempt_number == MAXIMUM_ATTEMPTS_PER_RECORD:
                        raise RuntimeError(f"v0.8 per-record attempt ceiling reached for {record_id}") from error
                    continue
                metadata = v5._response_metadata(response)
                rows.append({
                    "language_mode": language,
                    "messages": messages,
                    "provenance": {
                        "ai_assisted": True,
                        "external_inference_api": False,
                        "generation_attempt": attempt_number,
                        "human_authored": False,
                        "inference_location": "self_hosted_lambda_h100_pinned_ssh",
                        "model_id": v5.v2.predecessor.MODEL_ID,
                        "model_revision": v5.v2.predecessor.MODEL_REVISION,
                        "prompt_contract": PROMPT_CONTRACT,
                        "prompt_request_sha256": request_sha256,
                        "reasoning_effort": REASONING_EFFORT,
                        "reasoning_or_analysis_retained": False,
                        "runtime_image_digest": v5.v2.predecessor.RUNTIME_DIGEST,
                        "seed": full_canary_seed(record_id, attempt_number),
                    },
                    "record_id": record_id,
                    "rights_status": "private_research_candidate_pending_owner_review",
                    "scenario_id": scenario.scenario_id,
                    "schema_version": v5.v2.predecessor.ROW_SCHEMA,
                    "task": scenario.task,
                })
                telemetry.append({
                    "accepted_on_attempt": attempt_number,
                    "attempts": attempt_number,
                    "completion_tokens": metadata["usage"]["completion_tokens"],
                    "critical_failures": 0,
                    "elapsed_seconds_final_attempt": round(elapsed, 6),
                    "finish_reason": metadata["finish_reason"],
                    "prompt_tokens": metadata["usage"]["prompt_tokens"],
                    "record_id": record_id,
                    "request_sha256": request_sha256,
                    "response_envelope_sha256": sha256_value(response),
                    "total_tokens": metadata["usage"]["total_tokens"],
                })
                v5.v2.predecessor._write_jsonl(rows_path, rows)
                v5.v2.predecessor._write_jsonl(telemetry_path, telemetry)
                accepted = True
                break
            if not accepted:
                raise RuntimeError("v0.8 record was not accepted")
            if time.monotonic() - run_started > LIMITS["maximum_instance_minutes"] * 60:
                raise RuntimeError("v0.8 runtime ceiling reached")
        audit = v5.v2.predecessor._machine_audit(rows, telemetry)
        if not all(audit["checks"].values()):
            raise RuntimeError("v0.8 machine gate failed closed")
        owner_sample = v5.v2.predecessor.select_owner_sample(rows)
        if len(owner_sample) != LIMITS["owner_review_rows"]:
            raise ValueError("v0.8 owner sample count changed")
        v5.v2.predecessor._write_jsonl(private / "candidate-rows.jsonl", rows)
        v5.v2.predecessor._write_jsonl(private / "generation-telemetry.jsonl", telemetry)
        v5.v2.predecessor._write_jsonl(private / "owner-review-sample-32.jsonl", owner_sample)
        v5.v2.predecessor._write_jsonl(private / "rejection-metadata.jsonl", rejections)
    finally:
        v5.v3._stop_runtime(ssh_key.resolve(), known_hosts.resolve())
    after_gpu = v5.v2.predecessor.gpu_snapshot(ssh_key.resolve(), known_hosts.resolve())
    report: dict[str, Any] = {
        "accepted_final_outputs": len(rows),
        "authorization_file_sha256": file_sha256(authorization_path),
        "candidate_registry_changed": False,
        "dataset_sealed": False,
        "external_api_used": False,
        "gpu": {"after_stop": after_gpu, "before": before_gpu},
        "machine_audit": audit,
        "model": {"id": v5.v2.predecessor.MODEL_ID, "revision": v5.v2.predecessor.MODEL_REVISION},
        "owner_review": {"accepted_rows": None, "minimum_acceptance_rate": FROZEN_MACHINE_GATES["minimum_owner_acceptance_rate"], "rows": len(owner_sample), "sample_internal_sha256": sha256_value(owner_sample), "status": "pending"},
        "plan_file_sha256": file_sha256(plan_path),
        "production_eligible": False,
        "prompt_contract": PROMPT_CONTRACT,
        "rejected_attempts": len(rejections),
        "runtime_reuse_receipt_file_sha256": file_sha256(runtime_reuse_receipt_path),
        "runtime_stopped": after_gpu["memory_used_mib"] == 0,
        "schema_version": RUN_REPORT_SCHEMA,
        "sft_training_started": False,
        "status": "machine_verified_full_canary_pending_32_row_owner_review",
    }
    report["report_sha256"] = sha256_value(report)
    v5.v2.predecessor._write_json_atomic(evidence / "run-report.json", report)
    return report


def verify_full_canary_run(*, plan_path: Path, authorization_path: Path, runtime_reuse_receipt_path: Path, repository_root: Path, artifact_root: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    load_full_canary_plan(plan_path.resolve(), repository_root=root)
    load_execution_authorization(authorization_path.resolve(), plan_path=plan_path.resolve())
    load_runtime_reuse_receipt(runtime_reuse_receipt_path.resolve(), authorization_path=authorization_path.resolve())
    artifact = artifact_root.resolve()
    report_path = artifact / "evidence/run-report.json"
    report = v5.v2.predecessor._load_json(report_path)
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256", None)
    rows = v5.v2.predecessor._read_jsonl(artifact / "private/candidate-rows.jsonl")
    telemetry = v5.v2.predecessor._read_jsonl(artifact / "private/generation-telemetry.jsonl")
    rejections = v5.v2.predecessor._read_jsonl(artifact / "private/rejection-metadata.jsonl")
    scenarios = {(item.task, item.variant): item for item in v5.v2.predecessor.SCENARIOS}
    for row in rows:
        variant = int(str(row["record_id"]).rsplit("-", 1)[1])
        validate_prompt_quality(row["messages"], scenario=scenarios[(row["task"], variant)], language=row["language_mode"])
    audit = v5.v2.predecessor._machine_audit(rows, telemetry)
    sample = v5.v2.predecessor.select_owner_sample(rows)
    checks = {
        "authorization_binding": report.get("authorization_file_sha256") == file_sha256(authorization_path),
        "complete_row_quality_reproduced": len(rows) == 160,
        "internal_report_hash": internal == sha256_value(unsigned),
        "machine_gate_reproduced": all(audit["checks"].values()) and audit == report.get("machine_audit"),
        "owner_sample_reproduced": sha256_value(sample) == report.get("owner_review", {}).get("sample_internal_sha256"),
        "plan_binding": report.get("plan_file_sha256") == file_sha256(plan_path),
        "rejection_count_reproduced": len(rejections) == report.get("rejected_attempts"),
        "runtime_binding": report.get("runtime_reuse_receipt_file_sha256") == file_sha256(runtime_reuse_receipt_path),
        "runtime_stopped": report.get("runtime_stopped") is True and report.get("gpu", {}).get("after_stop", {}).get("memory_used_mib") == 0,
        "zero_production_actions": report.get("candidate_registry_changed") is False and report.get("dataset_sealed") is False and report.get("sft_training_started") is False and report.get("production_eligible") is False,
    }
    verification: dict[str, Any] = {"checks": checks, "run_report_file_sha256": file_sha256(report_path), "run_report_internal_sha256": internal, "schema_version": RUN_VERIFICATION_SCHEMA, "status": "PASS" if all(checks.values()) else "FAIL"}
    verification["verification_sha256"] = sha256_value(verification)
    v5.v2.predecessor._write_json_atomic(artifact / "evidence/run-verification.json", verification)
    if verification["status"] != "PASS":
        raise ValueError("v0.8 full-canary independent verification failed")
    return verification
