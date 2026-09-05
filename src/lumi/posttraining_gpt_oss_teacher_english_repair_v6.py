# ruff: noqa: E501, SLF001
"""Fail-closed one-row English refusal repair after the v0.5 full-canary stop."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import posttraining_gpt_oss_teacher_canary_v5 as v5
from .posttraining_candidate_mix_preflight import file_sha256, sha256_value

PLAN_SCHEMA = "lumi.sft-gpt-oss-teacher-english-refusal-repair-plan.v0.6"
REPORT_SCHEMA = "lumi.sft-gpt-oss-teacher-english-refusal-repair-plan-report.v0.6"
VERIFICATION_SCHEMA = "lumi.sft-gpt-oss-teacher-english-refusal-repair-plan-verification.v0.6"
AUTHORIZATION_SCHEMA = "lumi.sft-gpt-oss-teacher-english-refusal-repair-authorization.v0.6"
RUNTIME_REUSE_RECEIPT_SCHEMA = "lumi.sft-gpt-oss-teacher-english-refusal-repair-runtime-reuse-receipt.v0.6"
RUN_REPORT_SCHEMA = "lumi.sft-gpt-oss-teacher-english-refusal-repair-run-report.v0.6"
RUN_VERIFICATION_SCHEMA = "lumi.sft-gpt-oss-teacher-english-refusal-repair-run-verification.v0.6"

PROMPT_CONTRACT = "lumi.gpt-oss-teacher.english-direct-refusal-repair.v0.6"
SEED_CONTRACT = "sha256(record_id:english-direct-refusal-repair-attempt:attempt_number)-u32.v0.6"
RECORD_ID = "lumi-gptoss-safety_helpfulness-english-00"
REASONING_EFFORT = "low"
MAX_TOKENS = 1_536
MAXIMUM_ATTEMPTS = 2

SERVER = v5.SERVER
PROHIBITIONS = v5.PROHIBITIONS
RUNTIME_REUSE = v5.RUNTIME_REUSE
RETENTION = v5.RETENTION
PREDECESSOR_FAILURE = {
    "accepted_prefix_outputs": 3,
    "authorization_file_sha256": "12f96be85a6c19334c82f804d8c44baba2ff47930c99ce6cf7a84e941599b633",
    "failed_record_id": RECORD_ID,
    "failure_report_file_sha256": "79c835991f841fb54588821f67d7507da94d9200d09ee30bf4eb5563f9e339ea",
    "failure_verification_file_sha256": "ee890c4197e150eb684aae8e19921c2520765285cae55dd5d674b93975ea7882",
    "frozen_validation_error": "refusal row lacks a clear boundary",
    "inference_calls": 5,
    "rejection_metadata_file_sha256": "0474e477a80e6712804bb0776adfbdcaca27b8da9dfe7347ca7926ae12bcebd7",
    "status": "FAIL_CLOSED_fourth_record_attempt_ceiling",
}
LIMITS = {
    "exact_repair_records": 1,
    "maximum_accepted_final_outputs": 1,
    "maximum_attempts_per_record": MAXIMUM_ATTEMPTS,
    "maximum_context_tokens": 4_096,
    "maximum_generated_tokens_per_call": MAX_TOKENS,
    "maximum_generated_tokens_total": MAX_TOKENS * MAXIMUM_ATTEMPTS,
    "maximum_inference_calls": MAXIMUM_ATTEMPTS,
    "maximum_instance_minutes": 20,
    "maximum_network_bytes": 0,
    "maximum_private_write_bytes": 1 * 1024**3,
    "maximum_runtime_retries": 1,
    "maximum_usd_before_tax": 1.43,
    "owner_review_rows_if_accepted": 1,
}
QUALITY_GATES = {
    "accepted_output_must_pass_frozen_parser": True,
    "assistant_maximum_characters": 320,
    "banned_meta_phrases": list(v5.BANNED_META_PHRASES),
    "exact_direct_refusal_words": "cannot help",
    "owner_review_required_if_accepted": True,
    "required_recovery_markers": ["password reset", "account recovery", "your own account", "official"],
}

ENGLISH_REFUSAL_INSTRUCTION = (
    "Write natural, concise English. The user asks for another person's login password. "
    "The assistant_response must contain the exact consecutive words cannot help, "
    "directly refuse credential theft, and recommend an official password-reset or "
    "account-recovery path only for the user's own account. Do not use indirect refusal "
    "wording in place of cannot help. Never echo task instructions or use meta-commentary "
    "such as refuse the harmful request, safe alternative, safety label, scenario ID, "
    "assistant response, user prompt, dataset, training, or policy. Keep the "
    "assistant_response under 320 characters. Style target: Can you find someone else's "
    "login password? / I cannot help obtain another person's password. For your own "
    "account, use the official password-reset or account-recovery process."
)

EXACT_AUTHORIZATION_TEXT = (
    "I approve one bounded self-hosted Lumi gpt-oss-120b one-row English direct-refusal "
    "repair on the existing exact Lambda host 68.209.73.55, bound to ED25519 host-key "
    "fingerprint SHA256:2ZSQqNUYJhWNjihHnqLQ1YNHxoqtMJeckVJRKlc5hs4 and exactly one "
    "NVIDIA H100 80GB GPU GPU-c7e68b23-95b4-37bd-b40b-c41a7ec716f3 reporting at "
    "least 81,559 MiB FB total without adding the 480 MiB reserved value twice. Reuse, "
    "and do not redownload, the 25 verified openai/gpt-oss-120b model objects at revision "
    "b5c939de8f754692c1647ca79fbf85e8c1e70f8a bound by model receipt SHA-256 "
    "798914d8532d5b36b61711f34591cabe1e03fa6ac1128fe5e3737c029ead1284, and reuse "
    "the existing vLLM image digest "
    "sha256:4d08193d2fd05aadb1b5678f93ae609efb2635df67da45f3efe781c368b34dc8. "
    "Bind the v0.5 failure report SHA-256 "
    "79c835991f841fb54588821f67d7507da94d9200d09ee30bf4eb5563f9e339ea and failure "
    "verification SHA-256 ee890c4197e150eb684aae8e19921c2520765285cae55dd5d674b93975ea7882. "
    "Run only record lumi-gptoss-safety_helpfulness-english-00 with the frozen safety and "
    "schema parser, prompt contract lumi.gpt-oss-teacher.english-direct-refusal-repair.v0.6, "
    "reasoning effort low, a 1,536 generated-token ceiling per call, distinct attempt-bound "
    "deterministic seeds, and at most 2 attempts. Require natural English, the exact "
    "consecutive refusal words cannot help, and official password-reset or account-recovery "
    "guidance only for the user's own account; reject instruction or meta phrases including "
    "refuse the harmful request and safe alternative. Retain rejected attempts only as "
    "validation error, non-sensitive finish_reason, usage token counts, and request/response-"
    "envelope hashes without raw candidate or reasoning text. Maximum 20 additional instance-"
    "minutes / $1.43 before tax / 0 additional model or runtime download bytes / 1 GiB "
    "additional private writes / 2 sequential inference calls / 1 retry call / 1 accepted "
    "repair output. It may restart and then cleanly stop only the bound canary container, "
    "serve only on remote localhost through pinned SSH, run the one-row frozen machine and "
    "English-refusal gates, and prepare the accepted row for one-row owner review. This "
    "repair cannot satisfy the 16-cell gate or authorize resuming the larger canary. No "
    "external API, public port, model or runtime download, registry change, dataset seal, "
    "SFT training, production promotion, unrelated access, new instance, or termination."
)

TOOLING_PATHS = {
    "implementation": "src/lumi/posttraining_gpt_oss_teacher_english_repair_v6.py",
    "plan_runner": "scripts/plan_sft_gpt_oss_teacher_english_repair_v6.py",
    "plan_verifier": "scripts/verify_sft_gpt_oss_teacher_english_repair_v6_plan.py",
    "run_runner": "scripts/run_sft_gpt_oss_teacher_english_repair_v6.py",
    "run_verifier": "scripts/verify_sft_gpt_oss_teacher_english_repair_v6.py",
    "test": "tests/test_gpt_oss_teacher_english_repair_v6.py",
}


def repair_seed(attempt_number: int) -> int:
    if not 1 <= attempt_number <= MAXIMUM_ATTEMPTS:
        raise ValueError("English repair attempt number is outside the reviewed contract")
    material = f"{RECORD_ID}:english-direct-refusal-repair-attempt:{attempt_number}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def repair_messages(attempt_number: int) -> list[dict[str, str]]:
    retry = " This is the final repair attempt; silently re-check every frozen and English-refusal gate." if attempt_number > 1 else ""
    system = (
        "Return only one compact JSON object with scenario_id, user_prompt, assistant_response, "
        "safety_label, tool_call, and tool_result. Set scenario_id to credential_theft, "
        "safety_label to refuse, and both tool fields to null. Never expose reasoning. "
        + ENGLISH_REFUSAL_INSTRUCTION
        + retry
    )
    user = f"Create one English credential-theft refusal row. Attempt {attempt_number}/{MAXIMUM_ATTEMPTS}. Return JSON only."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def repair_payload(attempt_number: int) -> dict[str, Any]:
    scenario = v5.v2.predecessor.SAFETY_SCENARIOS[0]
    payload = copy.deepcopy(v5.v2.predecessor.chat_completion_payload(RECORD_ID, scenario, "english"))
    payload["chat_template_kwargs"] = {"reasoning_effort": REASONING_EFFORT}
    payload["max_tokens"] = MAX_TOKENS
    payload["messages"] = repair_messages(attempt_number)
    payload["seed"] = repair_seed(attempt_number)
    payload["temperature"] = 0.3
    return payload


def repair_manifest() -> dict[str, Any]:
    return {
        "attempts": [
            {
                "attempt_number": attempt,
                "max_tokens": MAX_TOKENS,
                "reasoning_effort": REASONING_EFFORT,
                "request_sha256": sha256_value(repair_payload(attempt)),
                "seed": repair_seed(attempt),
            }
            for attempt in range(1, MAXIMUM_ATTEMPTS + 1)
        ],
        "language": "english",
        "record_id": RECORD_ID,
        "scenario_id": "credential_theft",
        "task": "safety_helpfulness",
        "variant": 0,
    }


def validate_english_refusal(messages: Sequence[Mapping[str, str]]) -> None:
    if len(messages) != 2:
        raise ValueError("English repair message count changed")
    user = str(messages[0].get("content", ""))
    assistant = str(messages[1].get("content", ""))
    combined = f"{user}\n{assistant}".lower()
    for phrase in v5.BANNED_META_PHRASES:
        if phrase in combined:
            raise ValueError(f"English repair banned meta phrase: {phrase}")
    if len(assistant) > QUALITY_GATES["assistant_maximum_characters"]:
        raise ValueError("English repair assistant response is too long")
    if "cannot help" not in assistant.lower():
        raise ValueError("English repair lacks exact direct refusal words")
    if not any(marker in assistant.lower() for marker in QUALITY_GATES["required_recovery_markers"]):
        raise ValueError("English repair lacks own-account recovery guidance")


def load_repair_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    plan = v5.v2.predecessor._load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("status") != "english_refusal_repair_plan_pending_explicit_execution_authorization":
        raise ValueError("unsupported or changed English-refusal repair plan")
    if plan.get("server") != SERVER or plan.get("runtime_reuse") != RUNTIME_REUSE:
        raise ValueError("English repair host or runtime binding changed")
    if plan.get("predecessor_failure") != PREDECESSOR_FAILURE:
        raise ValueError("English repair predecessor failure changed")
    if plan.get("limits") != LIMITS or plan.get("quality_gates") != QUALITY_GATES:
        raise ValueError("English repair limits or gates changed")
    if plan.get("prohibitions") != PROHIBITIONS or plan.get("retention") != RETENTION:
        raise ValueError("English repair prohibitions or retention changed")
    if any(bool(value) for value in plan.get("authorization", {}).values() if isinstance(value, bool)):
        raise ValueError("English repair plan unexpectedly authorizes execution")
    if plan.get("next_gate", {}).get("exact_authorization_text") != EXACT_AUTHORIZATION_TEXT:
        raise ValueError("English repair authorization text changed")
    contract = plan.get("prompt_contract", {})
    if contract.get("id") != PROMPT_CONTRACT or contract.get("seed_contract") != SEED_CONTRACT:
        raise ValueError("English repair prompt contract changed")
    if contract.get("manifest_sha256") != sha256_value(repair_manifest()):
        raise ValueError("English repair manifest changed")
    v5.v2.predecessor._resolve_bindings(plan, repository_root)
    return plan


def build_plan_report(*, plan: Mapping[str, Any], plan_file_sha256: str, repository_root: str | Path) -> dict[str, Any]:
    resolved = v5.v2.predecessor._resolve_bindings(plan, repository_root)
    manifest = repair_manifest()
    report: dict[str, Any] = {
        "authorization": plan["authorization"],
        "bindings": {name: {"path": plan["bindings"][name]["path"], "sha256": file_sha256(path)} for name, path in sorted(resolved.items())},
        "checks": {
            "distinct_attempt_requests": len({item["request_sha256"] for item in manifest["attempts"]}) == 2,
            "distinct_attempt_seeds": len({item["seed"] for item in manifest["attempts"]}) == 2,
            "exact_direct_refusal_bound": "cannot help" in ENGLISH_REFUSAL_INSTRUCTION,
            "exact_one_failed_record": manifest["record_id"] == PREDECESSOR_FAILURE["failed_record_id"],
            "low_reasoning_exact": {item["reasoning_effort"] for item in manifest["attempts"]} == {"low"},
            "no_execution_authorized": not any(bool(value) for value in plan["authorization"].values() if isinstance(value, bool)),
            "no_model_or_runtime_download": plan["limits"]["maximum_network_bytes"] == 0,
            "predecessor_failure_bound": plan["predecessor_failure"] == PREDECESSOR_FAILURE,
            "recovery_guidance_required": "official password-reset or account-recovery" in ENGLISH_REFUSAL_INSTRUCTION,
            "token_ceiling_exact": {item["max_tokens"] for item in manifest["attempts"]} == {MAX_TOKENS},
        },
        "limits": plan["limits"],
        "manifest": manifest,
        "next_gate": plan["next_gate"],
        "observed_at": plan["observed_at"],
        "plan_file_sha256": plan_file_sha256,
        "predecessor_failure": plan["predecessor_failure"],
        "prohibitions": plan["prohibitions"],
        "prompt_contract": plan["prompt_contract"],
        "quality_gates": plan["quality_gates"],
        "retention": plan["retention"],
        "runtime_reuse": plan["runtime_reuse"],
        "schema_version": REPORT_SCHEMA,
        "server": plan["server"],
        "status": "verified_english_refusal_repair_plan_no_execution_authorized",
        "tooling": {name: {"path": tooling_path, "sha256": file_sha256(Path(repository_root) / tooling_path)} for name, tooling_path in sorted(TOOLING_PATHS.items())},
    }
    if not all(report["checks"].values()):
        raise ValueError("English repair plan checks failed")
    report["report_sha256"] = sha256_value(report)
    return report


def build_plan_verification(*, report: Mapping[str, Any], report_file_sha256: str) -> dict[str, Any]:
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256", None)
    if internal != sha256_value(unsigned):
        raise ValueError("English repair report internal hash changed")
    checks = {
        "all_plan_checks_pass": all(report.get("checks", {}).values()),
        "authorization_pending": report.get("status") == "verified_english_refusal_repair_plan_no_execution_authorized",
        "call_ceiling_two": report.get("limits", {}).get("maximum_inference_calls") == 2,
        "direct_refusal_gate_exact": report.get("quality_gates", {}).get("exact_direct_refusal_words") == "cannot help",
        "predecessor_failure_exact": report.get("predecessor_failure") == PREDECESSOR_FAILURE,
        "zero_download_ceiling": report.get("limits", {}).get("maximum_network_bytes") == 0,
    }
    if not all(checks.values()):
        raise ValueError("English repair independent plan verification failed")
    verification: dict[str, Any] = {"checks": checks, "report_file_sha256": report_file_sha256, "report_internal_sha256": internal, "schema_version": VERIFICATION_SCHEMA, "status": "PASS"}
    verification["verification_sha256"] = sha256_value(verification)
    return verification


def load_execution_authorization(path: str | Path, *, plan_path: str | Path) -> dict[str, Any]:
    authorization = v5.v2.predecessor._load_json(path)
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA or authorization.get("status") != "authorized_exact_host_bound_one_row_english_refusal_repair":
        raise ValueError("unsupported or changed English repair authorization")
    if authorization.get("approval_text") != EXACT_AUTHORIZATION_TEXT or authorization.get("plan_file_sha256") != file_sha256(plan_path):
        raise ValueError("English repair approval or plan binding changed")
    if authorization.get("server") != SERVER or authorization.get("limits") != LIMITS or authorization.get("prohibitions") != PROHIBITIONS or authorization.get("predecessor_failure") != PREDECESSOR_FAILURE:
        raise ValueError("English repair authorization bindings changed")
    started = authorization.get("authorized_start_unix")
    if not isinstance(started, int) or started <= 0:
        raise ValueError("English repair authorization start time is invalid")
    return authorization


def load_runtime_reuse_receipt(path: str | Path, *, authorization_path: str | Path) -> dict[str, Any]:
    receipt = v5.v2.predecessor._load_json(path)
    if receipt.get("schema_version") != RUNTIME_REUSE_RECEIPT_SCHEMA or receipt.get("status") != "PASS_exact_existing_localhost_runtime_reused_for_english_refusal_repair":
        raise ValueError("unsupported or failed English repair runtime receipt")
    if receipt.get("authorization_file_sha256") != file_sha256(authorization_path) or receipt.get("server") != SERVER or receipt.get("runtime_reuse") != RUNTIME_REUSE:
        raise ValueError("English repair runtime receipt binding changed")
    if not all(receipt.get("checks", {}).values()):
        raise ValueError("English repair runtime checks failed")
    return receipt


def run_english_repair(*, plan_path: Path, authorization_path: Path, runtime_reuse_receipt_path: Path, repository_root: Path, artifact_root: Path, ssh_key: Path, known_hosts: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    plan = load_repair_plan(plan_path.resolve(), repository_root=root)
    authorization = load_execution_authorization(authorization_path.resolve(), plan_path=plan_path.resolve())
    load_runtime_reuse_receipt(runtime_reuse_receipt_path.resolve(), authorization_path=authorization_path.resolve())
    if file_sha256(known_hosts.resolve()) != plan["bindings"]["known_hosts"]["sha256"] or ssh_key.resolve().stat().st_mode & 0o077:
        raise PermissionError("English repair SSH binding or permissions changed")
    if time.time() - int(authorization["authorized_start_unix"]) > LIMITS["maximum_instance_minutes"] * 60:
        raise RuntimeError("English repair authorization expired")
    artifact = artifact_root.resolve()
    if not artifact.is_relative_to((root / "artifacts").resolve()):
        raise ValueError("English repair artifact root escaped artifacts/")
    private, evidence = artifact / "private", artifact / "evidence"
    private.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    metadata_path, candidate_path, report_path = private / "repair-attempt-metadata.jsonl", private / "repair-candidate.jsonl", evidence / "run-report.json"
    if any(path.exists() for path in (metadata_path, candidate_path, report_path)):
        raise FileExistsError("refusing to replay a consumed English repair artifact root")
    scenario = v5.v2.predecessor.SAFETY_SCENARIOS[0]
    manifest = repair_manifest()
    attempts: list[dict[str, Any]] = []
    candidate: dict[str, Any] | None = None
    before_gpu = v5.v2.predecessor.gpu_snapshot(ssh_key.resolve(), known_hosts.resolve())
    try:
        for expected in manifest["attempts"]:
            number = int(expected["attempt_number"])
            payload = repair_payload(number)
            request_sha256 = sha256_value(payload)
            if request_sha256 != expected["request_sha256"]:
                raise ValueError("English repair request manifest changed")
            response = v5.v2.predecessor._post_chat_completion(payload)
            diagnostic = v5._response_metadata(response)
            error: Exception | None = None
            messages: list[dict[str, str]] | None = None
            try:
                messages = v5.v2.predecessor._parse_final_response(response, scenario=scenario, language="english")
                validate_english_refusal(messages)
            except (ValueError, json.JSONDecodeError) as observed:
                error = observed
            attempts.append({
                "accepted": error is None, "attempt_number": number,
                "completion_tokens": diagnostic["usage"]["completion_tokens"],
                "error_message": str(error) if error is not None else None,
                "error_type": type(error).__name__ if error is not None else None,
                "finish_reason": diagnostic["finish_reason"], "max_tokens": MAX_TOKENS,
                "prompt_tokens": diagnostic["usage"]["prompt_tokens"], "reasoning_effort": REASONING_EFFORT,
                "record_id": RECORD_ID, "request_sha256": request_sha256,
                "response_envelope_sha256": sha256_value(response), "total_tokens": diagnostic["usage"]["total_tokens"],
            })
            v5.v2.predecessor._write_jsonl(metadata_path, attempts)
            if error is not None:
                if v5.v2.predecessor._critical_validation_error(error):
                    break
                continue
            assert messages is not None
            candidate = {
                "language_mode": "english", "messages": messages,
                "provenance": {"ai_assisted": True, "diagnostic_only": True, "external_inference_api": False, "generation_attempt": number, "human_authored": False, "model_id": v5.v2.predecessor.MODEL_ID, "model_revision": v5.v2.predecessor.MODEL_REVISION, "prompt_contract": PROMPT_CONTRACT, "reasoning_effort": REASONING_EFFORT, "reasoning_or_analysis_retained": False, "request_sha256": request_sha256, "runtime_image_digest": v5.v2.predecessor.RUNTIME_DIGEST, "seed": expected["seed"]},
                "record_id": RECORD_ID, "rights_status": "private_research_english_repair_pending_owner_review",
                "scenario_id": scenario.scenario_id, "schema_version": v5.v2.predecessor.ROW_SCHEMA, "task": scenario.task,
            }
            v5.v2.predecessor._write_jsonl(candidate_path, [candidate])
            break
    finally:
        v5.v3._stop_runtime(ssh_key.resolve(), known_hosts.resolve())
    after_gpu = v5.v2.predecessor.gpu_snapshot(ssh_key.resolve(), known_hosts.resolve())
    report: dict[str, Any] = {
        "accepted_final_outputs": 1 if candidate is not None else 0,
        "attempt_metadata_file_sha256": file_sha256(metadata_path), "authorization_file_sha256": file_sha256(authorization_path),
        "candidate_file_sha256": file_sha256(candidate_path) if candidate_path.exists() else None,
        "candidate_registry_changed": False, "dataset_sealed": False, "external_api_used": False,
        "gpu": {"after_stop": after_gpu, "before": before_gpu}, "inference_calls": len(attempts),
        "machine_and_english_refusal_validation_passed": candidate is not None,
        "owner_review": {"rows": 1 if candidate is not None else 0, "status": "pending" if candidate is not None else "not_run"},
        "plan_file_sha256": file_sha256(plan_path), "production_eligible": False, "prompt_contract": PROMPT_CONTRACT,
        "runtime_reuse_receipt_file_sha256": file_sha256(runtime_reuse_receipt_path), "runtime_stopped": after_gpu["memory_used_mib"] == 0,
        "schema_version": RUN_REPORT_SCHEMA, "sft_training_started": False,
        "status": "english_refusal_repair_candidate_machine_valid_pending_owner_review" if candidate is not None else "FAIL_CLOSED_english_refusal_repair_exhausted",
    }
    report["report_sha256"] = sha256_value(report)
    v5.v2.predecessor._write_json_atomic(report_path, report)
    return report


def verify_english_repair_run(*, plan_path: Path, authorization_path: Path, runtime_reuse_receipt_path: Path, repository_root: Path, artifact_root: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    load_repair_plan(plan_path.resolve(), repository_root=root)
    load_execution_authorization(authorization_path.resolve(), plan_path=plan_path.resolve())
    load_runtime_reuse_receipt(runtime_reuse_receipt_path.resolve(), authorization_path=authorization_path.resolve())
    artifact = artifact_root.resolve()
    report_path = artifact / "evidence/run-report.json"
    report = v5.v2.predecessor._load_json(report_path)
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256", None)
    metadata = v5.v2.predecessor._read_jsonl(artifact / "private/repair-attempt-metadata.jsonl")
    candidate = v5.v2.predecessor._read_jsonl(artifact / "private/repair-candidate.jsonl")
    if candidate:
        validate_english_refusal(candidate[0]["messages"])
    checks = {
        "authorization_binding": report.get("authorization_file_sha256") == file_sha256(authorization_path),
        "candidate_count_matches": len(candidate) == report.get("accepted_final_outputs"),
        "candidate_english_refusal_reproduced": len(candidate) <= 1,
        "internal_report_hash": internal == sha256_value(unsigned),
        "metadata_is_bounded": 1 <= len(metadata) <= 2,
        "request_manifest_prefix": [item["request_sha256"] for item in metadata] == [item["request_sha256"] for item in repair_manifest()["attempts"][: len(metadata)]],
        "runtime_binding": report.get("runtime_reuse_receipt_file_sha256") == file_sha256(runtime_reuse_receipt_path),
        "runtime_stopped": report.get("runtime_stopped") is True and report.get("gpu", {}).get("after_stop", {}).get("memory_used_mib") == 0,
        "zero_production_actions": report.get("candidate_registry_changed") is False and report.get("dataset_sealed") is False and report.get("sft_training_started") is False and report.get("production_eligible") is False,
    }
    verification: dict[str, Any] = {"checks": checks, "run_report_file_sha256": file_sha256(report_path), "run_report_internal_sha256": internal, "schema_version": RUN_VERIFICATION_SCHEMA, "status": "PASS" if all(checks.values()) else "FAIL"}
    verification["verification_sha256"] = sha256_value(verification)
    v5.v2.predecessor._write_json_atomic(artifact / "evidence/run-verification.json", verification)
    if verification["status"] != "PASS":
        raise ValueError("English repair run verification failed")
    return verification
