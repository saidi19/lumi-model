# ruff: noqa: E501, SLF001
"""Fail-closed full 16-cell gpt-oss canary after the accepted v0.4 repair."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import posttraining_gpt_oss_teacher_canary_v2 as v2
from . import posttraining_gpt_oss_teacher_diagnostic_v3 as v3
from . import posttraining_gpt_oss_teacher_quality_repair_v4 as v4
from .posttraining_candidate_mix_preflight import file_sha256, sha256_value

PLAN_SCHEMA = "lumi.sft-gpt-oss-teacher-full-canary-successor-plan.v0.5"
REPORT_SCHEMA = "lumi.sft-gpt-oss-teacher-full-canary-successor-report.v0.5"
VERIFICATION_SCHEMA = "lumi.sft-gpt-oss-teacher-full-canary-successor-verification.v0.5"
AUTHORIZATION_SCHEMA = "lumi.sft-gpt-oss-teacher-full-canary-successor-authorization.v0.5"
RUNTIME_REUSE_RECEIPT_SCHEMA = "lumi.sft-gpt-oss-teacher-full-canary-runtime-reuse-receipt.v0.5"
RUN_REPORT_SCHEMA = "lumi.sft-gpt-oss-teacher-full-canary-run-report.v0.5"
RUN_VERIFICATION_SCHEMA = "lumi.sft-gpt-oss-teacher-full-canary-run-verification.v0.5"

PROMPT_CONTRACT = "lumi.gpt-oss-teacher.prompt-quality-full-canary.v0.5"
SEED_CONTRACT = "sha256(record_id:prompt-quality-full-canary-attempt:attempt_number)-u32.v0.5"
REASONING_EFFORT = "low"
MAX_TOKENS = 1_536
MAXIMUM_ATTEMPTS_PER_RECORD = 2

SERVER = v2.SERVER
PROHIBITIONS = v2.PROHIBITIONS
RUNTIME_REUSE = v2.RUNTIME_REUSE
RETENTION = v4.RETENTION
FROZEN_MACHINE_GATES = v2.QUALITY_GATES
ACCEPTED_REPAIR = {
    "accepted_rows": 1,
    "authorization_file_sha256": "33862f117c4726595661788cbab340deb6ab73910e9eb9790269dfd5e8146137",
    "candidate_file_sha256": "648f80d00c6b9def9ec0b1d261511a0b24a9880eb94876a9ee9c14ddd7784b8c",
    "owner_review_file_sha256": "f0c86920d51ca3bd44b797fbdb88310268c5dac24167a91254fdfe03dee50b12",
    "owner_review_status": "ACCEPT_prompt_quality",
    "owner_review_verification_file_sha256": "a92f6c06b33e9a03c48ad6d9aac8683243e95d080f02d58c0f8c71a9e703da58",
    "run_report_file_sha256": "9ca3bb161a82546fed3113d2997c2c47f9c7de1dc074deba6acd45f8018e35d2",
    "run_verification_file_sha256": "adb7cd46ce29b28c9ebe2e41fae48a5259d846bf16f62a1eae6d4ef0ab9f23b3",
}
BANNED_META_PHRASES = (
    "refuse the harmful request",
    "safe alternative",
    "safety label",
    "scenario id",
    "assistant response",
    "user prompt",
    "gupta shobdo",
)
LIMITS = {
    "exact_final_outputs": 160,
    "exact_prompt_cells": 16,
    "maximum_attempts_per_record": MAXIMUM_ATTEMPTS_PER_RECORD,
    "maximum_context_tokens": 4_096,
    "maximum_generated_tokens_per_call": MAX_TOKENS,
    "maximum_generated_tokens_total": 200 * MAX_TOKENS,
    "maximum_inference_calls": 200,
    "maximum_instance_minutes": 120,
    "maximum_network_bytes": 0,
    "maximum_private_write_bytes": 16 * 1024**3,
    "maximum_runtime_retries": 40,
    "maximum_usd_before_tax": 8.58,
    "owner_review_rows": 32,
    "prompts_per_cell": 10,
}
PROMPT_QUALITY_GATES = {
    "assistant_maximum_characters": 512,
    "banned_meta_phrases": list(BANNED_META_PHRASES),
    "banglish_refusal_requires_exact_parbo_na": True,
    "credential_theft_recovery_markers": list(v4.REQUIRED_RECOVERY_MARKERS),
    "owner_review_required": True,
}

GENERAL_OUTPUT_INSTRUCTION = (
    "Write only natural user-facing text, never task instructions or meta-commentary. "
    "Do not use phrases such as refuse the harmful request, safe alternative, safety "
    "label, scenario ID, assistant response, or user prompt. "
    "Keep assistant_response concise and under 512 characters."
)
BANGLISH_INSTRUCTION = (
    "Write natural conversational Banglish in Latin letters. The user_prompt must "
    "contain Tumi or Ami, and assistant_response must contain Ami or Eta. Common "
    "technical words such as login and password are natural; never translate password "
    "as gupta shobdo. If safety_label is refuse, directly refuse with the exact "
    "consecutive words parbo na and give scenario-appropriate safe guidance. For the "
    "credential_theft scenario specifically, recommend official password reset or "
    "account recovery for the user's own account. For medical_caution, use natural "
    "urgent wording such as joruri and direct the user to immediate local help. "
    + GENERAL_OUTPUT_INSTRUCTION
)

EXACT_AUTHORIZATION_TEXT = (
    "I approve one bounded self-hosted Lumi gpt-oss-120b full 16-cell prompt-quality "
    "teacher canary successor on the existing exact Lambda host 68.209.73.55, bound to "
    "ED25519 host-key fingerprint SHA256:2ZSQqNUYJhWNjihHnqLQ1YNHxoqtMJeckVJRKlc5hs4 "
    "and exactly one NVIDIA H100 80GB GPU GPU-c7e68b23-95b4-37bd-b40b-c41a7ec716f3 "
    "reporting at least 81,559 MiB FB total without adding the 480 MiB reserved value "
    "twice. Reuse, and do not redownload, the 25 verified openai/gpt-oss-120b model "
    "objects at revision b5c939de8f754692c1647ca79fbf85e8c1e70f8a bound by model "
    "receipt SHA-256 798914d8532d5b36b61711f34591cabe1e03fa6ac1128fe5e3737c029ead1284, "
    "and reuse the existing vLLM image digest "
    "sha256:4d08193d2fd05aadb1b5678f93ae609efb2635df67da45f3efe781c368b34dc8. "
    "Bind the accepted v0.4 owner-review SHA-256 "
    "f0c86920d51ca3bd44b797fbdb88310268c5dac24167a91254fdfe03dee50b12 and owner-review "
    "verification SHA-256 a92f6c06b33e9a03c48ad6d9aac8683243e95d080f02d58c0f8c71a9e703da58. "
    "Run the unchanged frozen 16-cell schedule of 160 exact records with prompt-quality "
    "contract lumi.gpt-oss-teacher.prompt-quality-full-canary.v0.5, reasoning effort low, "
    "a 1,536 generated-token ceiling per call, distinct attempt-bound deterministic "
    "seeds, and at most 2 attempts per record. Require natural user-facing language, "
    "reject instruction or meta phrases including refuse the harmful request, safe "
    "alternative, and gupta shobdo, require exact parbo na for Banglish refusal rows, "
    "and require official password-reset or account-recovery guidance for the Banglish "
    "credential-theft row. Retain rejected attempts only as validation error, "
    "non-sensitive finish_reason, usage token counts, and request/response-envelope "
    "hashes without raw candidate or reasoning text. Maximum 120 additional "
    "instance-minutes / $8.58 before tax / 0 additional model or runtime download bytes "
    "/ 16 GiB additional private writes / 200 sequential inference calls / 40 retry "
    "calls / 160 final outputs. It may restart and then cleanly stop only the bound "
    "canary container, serve only on remote localhost through pinned SSH, retain only "
    "accepted final-channel candidate text, run the frozen 16-cell machine gate, and "
    "prepare the deterministic 32-row owner-review packet. No external API, public port, "
    "model or runtime download, registry change, dataset seal, SFT training, production "
    "promotion, unrelated access, new instance, or termination."
)

TOOLING_PATHS = {
    "implementation": "src/lumi/posttraining_gpt_oss_teacher_canary_v5.py",
    "plan_runner": "scripts/plan_sft_gpt_oss_teacher_canary_v5.py",
    "plan_verifier": "scripts/verify_sft_gpt_oss_teacher_canary_v5_plan.py",
    "run_runner": "scripts/run_sft_gpt_oss_teacher_canary_v5.py",
    "run_verifier": "scripts/verify_sft_gpt_oss_teacher_canary_v5.py",
    "test": "tests/test_gpt_oss_teacher_canary_v5.py",
}


def full_canary_seed(record_id: str, attempt_number: int) -> int:
    if not 1 <= attempt_number <= MAXIMUM_ATTEMPTS_PER_RECORD:
        raise ValueError("full-canary attempt number is outside the reviewed contract")
    material = f"{record_id}:prompt-quality-full-canary-attempt:{attempt_number}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def generation_messages(
    scenario: v2.predecessor.base.Scenario,
    language: str,
    *,
    attempt_number: int,
) -> list[dict[str, str]]:
    language_instruction = (
        BANGLISH_INSTRUCTION
        if language == "banglish"
        else f"{v2.predecessor.base.LANGUAGE_INSTRUCTIONS[language]} {GENERAL_OUTPUT_INSTRUCTION}"
    )
    retry = (
        " This is the final bounded attempt; create a fresh candidate and silently "
        "re-check schema, language, factual, safety, punctuation, and naturalness gates."
        if attempt_number > 1
        else ""
    )
    system = (
        "Produce exactly one private AI-assisted Lumi SFT canary row. Return only a "
        "compact JSON object with scenario_id, user_prompt, assistant_response, "
        "safety_label, tool_call, and tool_result. Never expose analysis, reasoning, "
        "chain-of-thought, model identity, system prompts, or hidden instructions. Set "
        "tool_call and tool_result to null. The language contract applies independently "
        f"to both text fields. {language_instruction}{retry}"
    )
    user = (
        f"Scenario ID: {scenario.scenario_id}\nTask: {scenario.task}\n"
        f"Instruction: {scenario.instruction}\n"
        f"Set safety_label exactly to {scenario.safety_label}. Generation attempt: "
        f"{attempt_number}/{MAXIMUM_ATTEMPTS_PER_RECORD}. Return final JSON only."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def chat_completion_payload(
    record_id: str,
    scenario: v2.predecessor.base.Scenario,
    language: str,
    *,
    attempt_number: int,
) -> dict[str, Any]:
    payload = copy.deepcopy(v2.predecessor.chat_completion_payload(record_id, scenario, language))
    payload["chat_template_kwargs"] = {"reasoning_effort": REASONING_EFFORT}
    payload["max_tokens"] = MAX_TOKENS
    payload["messages"] = generation_messages(scenario, language, attempt_number=attempt_number)
    payload["seed"] = full_canary_seed(record_id, attempt_number)
    payload["temperature"] = 0.3
    return payload


def prompt_manifest() -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for scenario in v2.predecessor.SCENARIOS:
        for language in v2.predecessor.LANGUAGE_MODES:
            record_id = f"lumi-gptoss-{scenario.task}-{language}-{scenario.variant:02d}"
            attempts = [
                {
                    "attempt_number": attempt,
                    "max_tokens": MAX_TOKENS,
                    "reasoning_effort": REASONING_EFFORT,
                    "request_sha256": sha256_value(
                        chat_completion_payload(
                            record_id, scenario, language, attempt_number=attempt
                        )
                    ),
                    "seed": full_canary_seed(record_id, attempt),
                }
                for attempt in range(1, MAXIMUM_ATTEMPTS_PER_RECORD + 1)
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


def validate_prompt_quality(
    messages: Sequence[Mapping[str, str]],
    *,
    scenario: v2.predecessor.base.Scenario,
    language: str,
) -> None:
    if len(messages) != 2:
        raise ValueError("full-canary message count changed")
    user = str(messages[0].get("content", ""))
    assistant = str(messages[1].get("content", ""))
    combined = f"{user}\n{assistant}".lower()
    for phrase in BANNED_META_PHRASES:
        if phrase in combined:
            raise ValueError(f"full-canary banned meta phrase: {phrase}")
    if len(assistant) > PROMPT_QUALITY_GATES["assistant_maximum_characters"]:
        raise ValueError("full-canary assistant response is too long")
    if language == "banglish" and scenario.safety_label == "refuse":
        if "parbo na" not in assistant.lower():
            raise ValueError("full-canary Banglish refusal lacks exact parbo na")
    if language == "banglish" and scenario.scenario_id == "credential_theft":
        if not any(marker in assistant.lower() for marker in v4.REQUIRED_RECOVERY_MARKERS):
            raise ValueError("full-canary credential refusal lacks recovery guidance")


def load_full_canary_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    plan = v2.predecessor._load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("status") != "full_canary_successor_plan_pending_explicit_execution_authorization":
        raise ValueError("unsupported or changed gpt-oss full-canary successor plan")
    if plan.get("server") != SERVER or plan.get("runtime_reuse") != RUNTIME_REUSE:
        raise ValueError("gpt-oss full-canary host or runtime binding changed")
    if plan.get("accepted_repair") != ACCEPTED_REPAIR:
        raise ValueError("gpt-oss accepted repair binding changed")
    if plan.get("limits") != LIMITS or plan.get("frozen_machine_gates") != FROZEN_MACHINE_GATES:
        raise ValueError("gpt-oss full-canary limits or frozen gates changed")
    if plan.get("prompt_quality_gates") != PROMPT_QUALITY_GATES:
        raise ValueError("gpt-oss full-canary prompt-quality gates changed")
    if plan.get("prohibitions") != PROHIBITIONS or plan.get("retention") != RETENTION:
        raise ValueError("gpt-oss full-canary prohibition or retention changed")
    if any(bool(value) for value in plan.get("authorization", {}).values() if isinstance(value, bool)):
        raise ValueError("gpt-oss full-canary plan unexpectedly authorizes execution")
    if plan.get("next_gate", {}).get("exact_authorization_text") != EXACT_AUTHORIZATION_TEXT:
        raise ValueError("gpt-oss full-canary exact authorization text changed")
    contract = plan.get("prompt_contract", {})
    if contract.get("id") != PROMPT_CONTRACT or contract.get("seed_contract") != SEED_CONTRACT:
        raise ValueError("gpt-oss full-canary prompt contract changed")
    manifest = prompt_manifest()
    if len(manifest) != 160 or contract.get("manifest_sha256") != sha256_value(manifest):
        raise ValueError("gpt-oss full-canary prompt manifest changed")
    v2.predecessor._resolve_bindings(plan, repository_root)
    return plan


def build_plan_report(
    *, plan: Mapping[str, Any], plan_file_sha256: str, repository_root: str | Path
) -> dict[str, Any]:
    resolved = v2.predecessor._resolve_bindings(plan, repository_root)
    manifest = prompt_manifest()
    attempts = [attempt for item in manifest for attempt in item["attempts"]]
    cells = {(item["task"], item["language"]) for item in manifest}
    report: dict[str, Any] = {
        "accepted_repair": plan["accepted_repair"],
        "authorization": plan["authorization"],
        "bindings": {
            name: {"path": plan["bindings"][name]["path"], "sha256": file_sha256(path)}
            for name, path in sorted(resolved.items())
        },
        "checks": {
            "accepted_v0_4_owner_review_bound": plan["accepted_repair"] == ACCEPTED_REPAIR,
            "attempt_bound_requests_distinct": len({item["request_sha256"] for item in attempts}) == 320,
            "attempt_bound_seeds_distinct_per_record": all(
                len({attempt["seed"] for attempt in item["attempts"]}) == 2 for item in manifest
            ),
            "exact_frozen_cells": len(cells) == 16,
            "exact_frozen_records": len(manifest) == 160,
            "low_reasoning_exact": {item["reasoning_effort"] for item in attempts} == {"low"},
            "naturalness_meta_gate_bound": "refuse the harmful request" in PROMPT_QUALITY_GATES["banned_meta_phrases"],
            "no_execution_authorized": not any(
                bool(value) for value in plan["authorization"].values() if isinstance(value, bool)
            ),
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
        "tooling": {
            name: {"path": tooling_path, "sha256": file_sha256(Path(repository_root) / tooling_path)}
            for name, tooling_path in sorted(TOOLING_PATHS.items())
        },
    }
    if not all(report["checks"].values()):
        raise ValueError("gpt-oss full-canary plan checks failed")
    report["report_sha256"] = sha256_value(report)
    return report


def build_plan_verification(
    *, report: Mapping[str, Any], report_file_sha256: str
) -> dict[str, Any]:
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256", None)
    if internal != sha256_value(unsigned):
        raise ValueError("gpt-oss full-canary report internal hash changed")
    checks = {
        "accepted_repair_exact": report.get("accepted_repair") == ACCEPTED_REPAIR,
        "all_plan_checks_pass": all(report.get("checks", {}).values()),
        "authorization_pending": report.get("status") == "verified_full_canary_successor_plan_no_execution_authorized",
        "frozen_machine_gate_exact": report.get("frozen_machine_gates") == FROZEN_MACHINE_GATES,
        "full_output_count_exact": report.get("limits", {}).get("exact_final_outputs") == 160,
        "owner_sample_exact": report.get("limits", {}).get("owner_review_rows") == 32,
        "zero_download_ceiling": report.get("limits", {}).get("maximum_network_bytes") == 0,
    }
    if not all(checks.values()):
        raise ValueError("gpt-oss full-canary independent plan verification failed")
    verification: dict[str, Any] = {
        "checks": checks,
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": internal,
        "schema_version": VERIFICATION_SCHEMA,
        "status": "PASS",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification


def load_execution_authorization(path: str | Path, *, plan_path: str | Path) -> dict[str, Any]:
    authorization = v2.predecessor._load_json(path)
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA or authorization.get("status") != "authorized_exact_host_bound_full_prompt_quality_canary":
        raise ValueError("unsupported or changed gpt-oss full-canary authorization")
    if authorization.get("approval_text") != EXACT_AUTHORIZATION_TEXT or authorization.get("plan_file_sha256") != file_sha256(plan_path):
        raise ValueError("gpt-oss full-canary approval or plan binding changed")
    if authorization.get("server") != SERVER or authorization.get("limits") != LIMITS:
        raise ValueError("gpt-oss full-canary authorization host or limits changed")
    if authorization.get("prohibitions") != PROHIBITIONS or authorization.get("accepted_repair") != ACCEPTED_REPAIR:
        raise ValueError("gpt-oss full-canary authorization evidence changed")
    started = authorization.get("authorized_start_unix")
    if not isinstance(started, int) or started <= 0:
        raise ValueError("gpt-oss full-canary authorization start time is invalid")
    return authorization


def load_runtime_reuse_receipt(path: str | Path, *, authorization_path: str | Path) -> dict[str, Any]:
    receipt = v2.predecessor._load_json(path)
    if receipt.get("schema_version") != RUNTIME_REUSE_RECEIPT_SCHEMA or receipt.get("status") != "PASS_exact_existing_localhost_runtime_reused_for_full_canary_v0.5":
        raise ValueError("unsupported or failed gpt-oss full-canary runtime receipt")
    if receipt.get("authorization_file_sha256") != file_sha256(authorization_path):
        raise ValueError("gpt-oss full-canary runtime authorization binding changed")
    if receipt.get("server") != SERVER or receipt.get("runtime_reuse") != RUNTIME_REUSE:
        raise ValueError("gpt-oss full-canary runtime reuse binding changed")
    if not all(receipt.get("checks", {}).values()):
        raise ValueError("gpt-oss full-canary runtime checks failed")
    return receipt


def _response_metadata(response: Mapping[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    choice = choices[0] if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], Mapping) else {}
    usage = response.get("usage") if isinstance(response.get("usage"), Mapping) else {}
    return {
        "finish_reason": choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else None,
        "usage": {
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
        },
    }


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
    metadata = _response_metadata(response)
    ledger.append(
        {
            "attempt_number": attempt_number,
            "completion_tokens": metadata["usage"]["completion_tokens"],
            "critical": critical,
            "error_message": str(error),
            "error_type": type(error).__name__,
            "finish_reason": metadata["finish_reason"],
            "prompt_tokens": metadata["usage"]["prompt_tokens"],
            "record_id": record_id,
            "request_sha256": request_sha256,
            "response_envelope_sha256": sha256_value(response),
            "total_tokens": metadata["usage"]["total_tokens"],
        }
    )
    v2.predecessor._write_jsonl(path, ledger)


def run_full_canary(
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
    plan = load_full_canary_plan(plan_path.resolve(), repository_root=root)
    authorization = load_execution_authorization(authorization_path.resolve(), plan_path=plan_path.resolve())
    load_runtime_reuse_receipt(runtime_reuse_receipt_path.resolve(), authorization_path=authorization_path.resolve())
    if file_sha256(known_hosts.resolve()) != plan["bindings"]["known_hosts"]["sha256"] or ssh_key.resolve().stat().st_mode & 0o077:
        raise PermissionError("gpt-oss full-canary SSH binding or permissions changed")
    if time.time() - int(authorization["authorized_start_unix"]) > LIMITS["maximum_instance_minutes"] * 60:
        raise RuntimeError("gpt-oss full-canary authorization expired")
    artifact = artifact_root.resolve()
    if not artifact.is_relative_to((root / "artifacts").resolve()):
        raise ValueError("gpt-oss full-canary artifact root escaped artifacts/")
    private, evidence = artifact / "private", artifact / "evidence"
    private.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    rows_path = private / "candidate-rows.partial.jsonl"
    telemetry_path = private / "generation-telemetry.partial.jsonl"
    rejection_path = private / "rejection-metadata.partial.jsonl"
    rows = v2.predecessor._read_jsonl(rows_path)
    telemetry = v2.predecessor._read_jsonl(telemetry_path)
    rejections = v2.predecessor._read_jsonl(rejection_path)
    expected_ids = v2.predecessor.expected_record_ids()
    if [str(row.get("record_id")) for row in rows] != list(expected_ids[: len(rows)]) or len(rows) != len(telemetry):
        raise ValueError("gpt-oss full-canary partial outputs are not an exact prefix")
    allowed_rejection_ids = set(expected_ids[: min(len(rows) + 1, len(expected_ids))])
    if any(str(item.get("record_id")) not in allowed_rejection_ids for item in rejections):
        raise ValueError("gpt-oss full-canary rejection ledger escaped the exact prefix")
    total_calls, total_retries = len(rows) + len(rejections), len(rejections)
    before_gpu = v2.predecessor.gpu_snapshot(ssh_key.resolve(), known_hosts.resolve())
    scenarios = {(item.task, item.variant): item for item in v2.predecessor.SCENARIOS}
    manifest = prompt_manifest()
    run_started = time.monotonic()
    try:
        for item in manifest[len(rows) :]:
            record_id = str(item["record_id"])
            scenario = scenarios[(str(item["task"]), int(item["variant"]))]
            language = str(item["language"])
            prior = [entry for entry in rejections if entry["record_id"] == record_id]
            if len(prior) >= MAXIMUM_ATTEMPTS_PER_RECORD:
                raise RuntimeError(f"gpt-oss full-canary prior attempts exhausted for {record_id}")
            accepted = False
            for attempt_number in range(len(prior) + 1, MAXIMUM_ATTEMPTS_PER_RECORD + 1):
                if total_calls >= LIMITS["maximum_inference_calls"]:
                    raise RuntimeError("gpt-oss full-canary inference-call ceiling reached")
                if attempt_number > 1:
                    if total_retries >= LIMITS["maximum_runtime_retries"]:
                        raise RuntimeError("gpt-oss full-canary retry ceiling reached")
                    total_retries += 1
                payload = chat_completion_payload(record_id, scenario, language, attempt_number=attempt_number)
                request_sha256 = sha256_value(payload)
                if request_sha256 != item["attempts"][attempt_number - 1]["request_sha256"]:
                    raise ValueError("gpt-oss full-canary request manifest changed")
                total_calls += 1
                call_started = time.monotonic()
                response = v2.predecessor._post_chat_completion(payload)
                elapsed = time.monotonic() - call_started
                try:
                    messages = v2.predecessor._parse_final_response(response, scenario=scenario, language=language)
                    validate_prompt_quality(messages, scenario=scenario, language=language)
                except (ValueError, json.JSONDecodeError) as error:
                    critical = v2.predecessor._critical_validation_error(error)
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
                        raise RuntimeError(f"critical gpt-oss full-canary failure for {record_id}: {error}") from error
                    if attempt_number == MAXIMUM_ATTEMPTS_PER_RECORD:
                        raise RuntimeError(f"gpt-oss full-canary per-record attempt ceiling reached for {record_id}") from error
                    continue
                metadata = _response_metadata(response)
                rows.append(
                    {
                        "language_mode": language,
                        "messages": messages,
                        "provenance": {
                            "ai_assisted": True,
                            "external_inference_api": False,
                            "generation_attempt": attempt_number,
                            "human_authored": False,
                            "inference_location": "self_hosted_lambda_h100_pinned_ssh",
                            "model_id": v2.predecessor.MODEL_ID,
                            "model_revision": v2.predecessor.MODEL_REVISION,
                            "prompt_contract": PROMPT_CONTRACT,
                            "prompt_request_sha256": request_sha256,
                            "reasoning_effort": REASONING_EFFORT,
                            "reasoning_or_analysis_retained": False,
                            "runtime_image_digest": v2.predecessor.RUNTIME_DIGEST,
                            "seed": full_canary_seed(record_id, attempt_number),
                        },
                        "record_id": record_id,
                        "rights_status": "private_research_candidate_pending_owner_review",
                        "scenario_id": scenario.scenario_id,
                        "schema_version": v2.predecessor.ROW_SCHEMA,
                        "task": scenario.task,
                    }
                )
                telemetry.append(
                    {
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
                    }
                )
                v2.predecessor._write_jsonl(rows_path, rows)
                v2.predecessor._write_jsonl(telemetry_path, telemetry)
                accepted = True
                break
            if not accepted:
                raise RuntimeError("gpt-oss full-canary record was not accepted")
            if time.monotonic() - run_started > LIMITS["maximum_instance_minutes"] * 60:
                raise RuntimeError("gpt-oss full-canary runtime ceiling reached")
        audit = v2.predecessor._machine_audit(rows, telemetry)
        if not all(audit["checks"].values()):
            raise RuntimeError("gpt-oss full-canary machine gate failed closed")
        owner_sample = v2.predecessor.select_owner_sample(rows)
        if len(owner_sample) != LIMITS["owner_review_rows"]:
            raise ValueError("gpt-oss full-canary owner sample count changed")
        v2.predecessor._write_jsonl(private / "candidate-rows.jsonl", rows)
        v2.predecessor._write_jsonl(private / "generation-telemetry.jsonl", telemetry)
        v2.predecessor._write_jsonl(private / "owner-review-sample-32.jsonl", owner_sample)
        v2.predecessor._write_jsonl(private / "rejection-metadata.jsonl", rejections)
    finally:
        v3._stop_runtime(ssh_key.resolve(), known_hosts.resolve())
    after_gpu = v2.predecessor.gpu_snapshot(ssh_key.resolve(), known_hosts.resolve())
    report: dict[str, Any] = {
        "accepted_final_outputs": len(rows),
        "authorization_file_sha256": file_sha256(authorization_path),
        "candidate_registry_changed": False,
        "dataset_sealed": False,
        "external_api_used": False,
        "gpu": {"after_stop": after_gpu, "before": before_gpu},
        "machine_audit": audit,
        "model": {"id": v2.predecessor.MODEL_ID, "revision": v2.predecessor.MODEL_REVISION},
        "owner_review": {
            "accepted_rows": None,
            "minimum_acceptance_rate": FROZEN_MACHINE_GATES["minimum_owner_acceptance_rate"],
            "rows": len(owner_sample),
            "sample_internal_sha256": sha256_value(owner_sample),
            "status": "pending",
        },
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
    v2.predecessor._write_json_atomic(evidence / "run-report.json", report)
    return report


def verify_full_canary_run(
    *,
    plan_path: Path,
    authorization_path: Path,
    runtime_reuse_receipt_path: Path,
    repository_root: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    root = repository_root.resolve()
    load_full_canary_plan(plan_path.resolve(), repository_root=root)
    load_execution_authorization(authorization_path.resolve(), plan_path=plan_path.resolve())
    load_runtime_reuse_receipt(runtime_reuse_receipt_path.resolve(), authorization_path=authorization_path.resolve())
    artifact = artifact_root.resolve()
    report_path = artifact / "evidence/run-report.json"
    report = v2.predecessor._load_json(report_path)
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256", None)
    rows = v2.predecessor._read_jsonl(artifact / "private/candidate-rows.jsonl")
    telemetry = v2.predecessor._read_jsonl(artifact / "private/generation-telemetry.jsonl")
    rejections = v2.predecessor._read_jsonl(artifact / "private/rejection-metadata.jsonl")
    scenarios = {(item.task, item.variant): item for item in v2.predecessor.SCENARIOS}
    for row in rows:
        variant = int(str(row["record_id"]).rsplit("-", 1)[1])
        validate_prompt_quality(row["messages"], scenario=scenarios[(row["task"], variant)], language=row["language_mode"])
    audit = v2.predecessor._machine_audit(rows, telemetry)
    sample = v2.predecessor.select_owner_sample(rows)
    checks = {
        "authorization_binding": report.get("authorization_file_sha256") == file_sha256(authorization_path),
        "internal_report_hash": internal == sha256_value(unsigned),
        "machine_gate_reproduced": all(audit["checks"].values()) and audit == report.get("machine_audit"),
        "owner_sample_reproduced": sha256_value(sample) == report.get("owner_review", {}).get("sample_internal_sha256"),
        "plan_binding": report.get("plan_file_sha256") == file_sha256(plan_path),
        "prompt_quality_reproduced": len(rows) == 160,
        "rejection_count_reproduced": len(rejections) == report.get("rejected_attempts"),
        "runtime_binding": report.get("runtime_reuse_receipt_file_sha256") == file_sha256(runtime_reuse_receipt_path),
        "runtime_stopped": report.get("runtime_stopped") is True and report.get("gpu", {}).get("after_stop", {}).get("memory_used_mib") == 0,
        "zero_production_actions": report.get("candidate_registry_changed") is False and report.get("dataset_sealed") is False and report.get("sft_training_started") is False and report.get("production_eligible") is False,
    }
    verification: dict[str, Any] = {
        "checks": checks,
        "run_report_file_sha256": file_sha256(report_path),
        "run_report_internal_sha256": internal,
        "schema_version": RUN_VERIFICATION_SCHEMA,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    verification["verification_sha256"] = sha256_value(verification)
    v2.predecessor._write_json_atomic(artifact / "evidence/run-verification.json", verification)
    if verification["status"] != "PASS":
        raise ValueError("gpt-oss full-canary run independent verification failed")
    return verification
