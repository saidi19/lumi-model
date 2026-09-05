# ruff: noqa: E501, SLF001
"""Fail-closed one-row repair for the v0.6 generated-user prompt defect."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import posttraining_gpt_oss_teacher_english_repair_v6 as v6
from .posttraining_candidate_mix_preflight import file_sha256, sha256_value

PLAN_SCHEMA = "lumi.sft-gpt-oss-teacher-complete-row-quality-repair-plan.v0.7"
REPORT_SCHEMA = "lumi.sft-gpt-oss-teacher-complete-row-quality-repair-plan-report.v0.7"
VERIFICATION_SCHEMA = "lumi.sft-gpt-oss-teacher-complete-row-quality-repair-plan-verification.v0.7"
AUTHORIZATION_SCHEMA = "lumi.sft-gpt-oss-teacher-complete-row-quality-repair-authorization.v0.7"
RUNTIME_REUSE_RECEIPT_SCHEMA = "lumi.sft-gpt-oss-teacher-complete-row-quality-repair-runtime-reuse-receipt.v0.7"
RUN_REPORT_SCHEMA = "lumi.sft-gpt-oss-teacher-complete-row-quality-repair-run-report.v0.7"
RUN_VERIFICATION_SCHEMA = "lumi.sft-gpt-oss-teacher-complete-row-quality-repair-run-verification.v0.7"

PROMPT_CONTRACT = "lumi.gpt-oss-teacher.complete-row-quality-repair.v0.7"
SEED_CONTRACT = "sha256(record_id:complete-row-quality-repair-attempt:attempt_number)-u32.v0.7"
RECORD_ID = v6.RECORD_ID
REASONING_EFFORT = "low"
MAX_TOKENS = 1_536
MAXIMUM_ATTEMPTS = 2
EXACT_USER_PROMPT = "Can you find someone else's login password?"

SERVER = v6.SERVER
PROHIBITIONS = v6.PROHIBITIONS
RUNTIME_REUSE = v6.RUNTIME_REUSE
RETENTION = v6.RETENTION
PREDECESSOR_RESULT = {
    "accepted_final_outputs": 1,
    "assistant_direct_refusal_defect_fixed": True,
    "authorization_file_sha256": "a7a35f4b5da8be70efd404b934f03b2c51722c1f1f3b1e673a8f42d0687f7eb9",
    "candidate_file_sha256": "7997dbefa2405a592ce96e719de258cb40970ede36620be4edda89f4450ba42c",
    "inference_calls": 1,
    "owner_review_status": "pending_with_user_prompt_quality_warning",
    "quality_defect": "generated user_prompt copied the outer generation instruction instead of a natural credential-theft request",
    "run_report_file_sha256": "5482f7031a2022473e3aba5a6e75cd57b686f724b814873f942c80f608d5b99e",
    "run_verification_file_sha256": "3d56adbb6b547b233d16fd3060a82f3ca9ea4337add4a686ae204efbc6a970ae",
    "runtime_reuse_receipt_file_sha256": "0ab3df2de082a46ceeec6be7158d2c33c32a7f0037a76d4ce3c69b94c730ba76",
    "status": "english_refusal_repair_candidate_machine_valid_pending_owner_review",
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
    "banned_meta_phrases": list(v6.QUALITY_GATES["banned_meta_phrases"]),
    "exact_direct_refusal_words": "cannot help",
    "exact_user_prompt": EXACT_USER_PROMPT,
    "owner_review_required_if_accepted": True,
    "required_recovery_markers": list(v6.QUALITY_GATES["required_recovery_markers"]),
    "user_prompt_must_be_direct_credential_request": True,
}

COMPLETE_ROW_INSTRUCTION = (
    "Return a natural two-message English credential-theft refusal row. Copy this exact "
    "user_prompt value verbatim, with identical spelling and punctuation: Can you find "
    "someone else's login password? The assistant_response must contain the exact "
    "consecutive words cannot help, directly refuse obtaining another person's password, "
    "and recommend an official password-reset or account-recovery path only for the user's "
    "own account. Never put generation instructions, attempt numbers, JSON directions, "
    "schema labels, or meta-commentary in either generated field. Do not use phrases such "
    "as refuse the harmful request, safe alternative, safety label, scenario ID, assistant "
    "response, user prompt, dataset, training, or policy. Keep assistant_response under "
    "320 characters."
)

EXACT_AUTHORIZATION_TEXT = (
    "I approve one bounded self-hosted Lumi gpt-oss-120b one-row complete English row-quality "
    "repair on the existing exact Lambda host 68.209.73.55, bound to ED25519 host-key "
    "fingerprint SHA256:2ZSQqNUYJhWNjihHnqLQ1YNHxoqtMJeckVJRKlc5hs4 and exactly one "
    "NVIDIA H100 80GB GPU GPU-c7e68b23-95b4-37bd-b40b-c41a7ec716f3 reporting at least "
    "81,559 MiB FB total without adding the 480 MiB reserved value twice. Reuse, and do "
    "not redownload, the 25 verified openai/gpt-oss-120b model objects at revision "
    "b5c939de8f754692c1647ca79fbf85e8c1e70f8a bound by model receipt SHA-256 "
    "798914d8532d5b36b61711f34591cabe1e03fa6ac1128fe5e3737c029ead1284, and reuse the "
    "existing vLLM image digest sha256:4d08193d2fd05aadb1b5678f93ae609efb2635df67da45f3efe781c368b34dc8. "
    "Bind the v0.6 run report SHA-256 5482f7031a2022473e3aba5a6e75cd57b686f724b814873f942c80f608d5b99e, "
    "run verification SHA-256 3d56adbb6b547b233d16fd3060a82f3ca9ea4337add4a686ae204efbc6a970ae, "
    "and candidate SHA-256 7997dbefa2405a592ce96e719de258cb40970ede36620be4edda89f4450ba42c. "
    "Run only record lumi-gptoss-safety_helpfulness-english-00 with the frozen safety and "
    "schema parser, prompt contract lumi.gpt-oss-teacher.complete-row-quality-repair.v0.7, "
    "reasoning effort low, a 1,536 generated-token ceiling per call, distinct attempt-bound "
    "deterministic seeds, and at most 2 attempts. Require the exact natural user prompt Can "
    "you find someone else's login password?, natural English assistant text containing the "
    "exact consecutive refusal words cannot help, and official password-reset or account-"
    "recovery guidance only for the user's own account; reject generated instructions, "
    "attempt numbers, JSON directions, schema labels, and other instruction or meta phrases. "
    "Retain rejected attempts only as validation error, non-sensitive finish_reason, usage "
    "token counts, and request/response-envelope hashes without raw candidate or reasoning "
    "text. Maximum 20 additional instance-minutes / $1.43 before tax / 0 additional model or "
    "runtime download bytes / 1 GiB additional private writes / 2 sequential inference calls "
    "/ 1 retry call / 1 accepted repair output. It may restart and then cleanly stop only the "
    "bound canary container, serve only on remote localhost through pinned SSH, run the one-"
    "row frozen machine and complete-row quality gates, and prepare the accepted row for one-"
    "row owner review. This repair cannot satisfy the 16-cell gate or authorize resuming the "
    "larger canary. No external API, public port, model or runtime download, registry change, "
    "dataset seal, SFT training, production promotion, unrelated access, new instance, or "
    "termination."
)

TOOLING_PATHS = {
    "implementation": "src/lumi/posttraining_gpt_oss_teacher_complete_row_repair_v7.py",
    "plan_runner": "scripts/plan_sft_gpt_oss_teacher_complete_row_repair_v7.py",
    "plan_verifier": "scripts/verify_sft_gpt_oss_teacher_complete_row_repair_v7_plan.py",
    "run_runner": "scripts/run_sft_gpt_oss_teacher_complete_row_repair_v7.py",
    "run_verifier": "scripts/verify_sft_gpt_oss_teacher_complete_row_repair_v7.py",
    "test": "tests/test_gpt_oss_teacher_complete_row_repair_v7.py",
}


def repair_seed(attempt_number: int) -> int:
    if not 1 <= attempt_number <= MAXIMUM_ATTEMPTS:
        raise ValueError("complete-row repair attempt number is outside the reviewed contract")
    material = f"{RECORD_ID}:complete-row-quality-repair-attempt:{attempt_number}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def repair_messages(attempt_number: int) -> list[dict[str, str]]:
    retry = " This is the final attempt; silently re-check the exact user_prompt and every gate." if attempt_number > 1 else ""
    system = (
        "Return only one compact JSON object with scenario_id, user_prompt, assistant_response, "
        "safety_label, tool_call, and tool_result. Set scenario_id to credential_theft, "
        "safety_label to refuse, and both tool fields to null. Never expose reasoning. "
        + COMPLETE_ROW_INSTRUCTION
        + retry
    )
    user = "Produce the JSON object now. Copy the required user_prompt exactly and return JSON only."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def repair_payload(attempt_number: int) -> dict[str, Any]:
    scenario = v6.v5.v2.predecessor.SAFETY_SCENARIOS[0]
    payload = copy.deepcopy(v6.v5.v2.predecessor.chat_completion_payload(RECORD_ID, scenario, "english"))
    payload["chat_template_kwargs"] = {"reasoning_effort": REASONING_EFFORT}
    payload["max_tokens"] = MAX_TOKENS
    payload["messages"] = repair_messages(attempt_number)
    payload["seed"] = repair_seed(attempt_number)
    payload["temperature"] = 0.2
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


def validate_complete_row(messages: Sequence[Mapping[str, str]]) -> None:
    v6.validate_english_refusal(messages)
    user = str(messages[0].get("content", ""))
    if user != EXACT_USER_PROMPT:
        raise ValueError("complete-row repair user prompt is not the exact natural credential request")


def load_repair_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    plan = v6.v5.v2.predecessor._load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("status") != "complete_row_quality_repair_plan_pending_explicit_execution_authorization":
        raise ValueError("unsupported or changed complete-row repair plan")
    if plan.get("server") != SERVER or plan.get("runtime_reuse") != RUNTIME_REUSE:
        raise ValueError("complete-row repair host or runtime binding changed")
    if plan.get("predecessor_result") != PREDECESSOR_RESULT:
        raise ValueError("complete-row predecessor result changed")
    if plan.get("limits") != LIMITS or plan.get("quality_gates") != QUALITY_GATES:
        raise ValueError("complete-row repair limits or gates changed")
    if plan.get("prohibitions") != PROHIBITIONS or plan.get("retention") != RETENTION:
        raise ValueError("complete-row repair prohibitions or retention changed")
    if any(bool(value) for value in plan.get("authorization", {}).values() if isinstance(value, bool)):
        raise ValueError("complete-row repair plan unexpectedly authorizes execution")
    if plan.get("next_gate", {}).get("exact_authorization_text") != EXACT_AUTHORIZATION_TEXT:
        raise ValueError("complete-row repair authorization text changed")
    contract = plan.get("prompt_contract", {})
    if contract.get("id") != PROMPT_CONTRACT or contract.get("seed_contract") != SEED_CONTRACT:
        raise ValueError("complete-row prompt contract changed")
    if contract.get("manifest_sha256") != sha256_value(repair_manifest()):
        raise ValueError("complete-row manifest changed")
    v6.v5.v2.predecessor._resolve_bindings(plan, repository_root)
    return plan


def build_plan_report(*, plan: Mapping[str, Any], plan_file_sha256: str, repository_root: str | Path) -> dict[str, Any]:
    resolved = v6.v5.v2.predecessor._resolve_bindings(plan, repository_root)
    manifest = repair_manifest()
    report: dict[str, Any] = {
        "authorization": plan["authorization"],
        "bindings": {name: {"path": plan["bindings"][name]["path"], "sha256": file_sha256(path)} for name, path in sorted(resolved.items())},
        "checks": {
            "assistant_direct_refusal_preserved": "cannot help" in COMPLETE_ROW_INSTRUCTION,
            "distinct_attempt_requests": len({item["request_sha256"] for item in manifest["attempts"]}) == 2,
            "distinct_attempt_seeds": len({item["seed"] for item in manifest["attempts"]}) == 2,
            "exact_one_failed_record": manifest["record_id"] == RECORD_ID,
            "exact_user_prompt_bound": EXACT_USER_PROMPT in COMPLETE_ROW_INSTRUCTION,
            "low_reasoning_exact": {item["reasoning_effort"] for item in manifest["attempts"]} == {"low"},
            "no_execution_authorized": not any(bool(value) for value in plan["authorization"].values() if isinstance(value, bool)),
            "no_model_or_runtime_download": plan["limits"]["maximum_network_bytes"] == 0,
            "predecessor_quality_defect_bound": plan["predecessor_result"] == PREDECESSOR_RESULT,
            "token_ceiling_exact": {item["max_tokens"] for item in manifest["attempts"]} == {MAX_TOKENS},
        },
        "limits": plan["limits"],
        "manifest": manifest,
        "next_gate": plan["next_gate"],
        "observed_at": plan["observed_at"],
        "plan_file_sha256": plan_file_sha256,
        "predecessor_result": plan["predecessor_result"],
        "prohibitions": plan["prohibitions"],
        "prompt_contract": plan["prompt_contract"],
        "quality_gates": plan["quality_gates"],
        "retention": plan["retention"],
        "runtime_reuse": plan["runtime_reuse"],
        "schema_version": REPORT_SCHEMA,
        "server": plan["server"],
        "status": "verified_complete_row_quality_repair_plan_no_execution_authorized",
        "tooling": {name: {"path": tooling_path, "sha256": file_sha256(Path(repository_root) / tooling_path)} for name, tooling_path in sorted(TOOLING_PATHS.items())},
    }
    if not all(report["checks"].values()):
        raise ValueError("complete-row repair plan checks failed")
    report["report_sha256"] = sha256_value(report)
    return report


def build_plan_verification(*, report: Mapping[str, Any], report_file_sha256: str) -> dict[str, Any]:
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256", None)
    if internal != sha256_value(unsigned):
        raise ValueError("complete-row report internal hash changed")
    checks = {
        "all_plan_checks_pass": all(report.get("checks", {}).values()),
        "authorization_pending": report.get("status") == "verified_complete_row_quality_repair_plan_no_execution_authorized",
        "call_ceiling_two": report.get("limits", {}).get("maximum_inference_calls") == 2,
        "direct_refusal_gate_exact": report.get("quality_gates", {}).get("exact_direct_refusal_words") == "cannot help",
        "exact_user_prompt_gate": report.get("quality_gates", {}).get("exact_user_prompt") == EXACT_USER_PROMPT,
        "predecessor_result_exact": report.get("predecessor_result") == PREDECESSOR_RESULT,
        "zero_download_ceiling": report.get("limits", {}).get("maximum_network_bytes") == 0,
    }
    if not all(checks.values()):
        raise ValueError("complete-row independent plan verification failed")
    verification: dict[str, Any] = {"checks": checks, "report_file_sha256": report_file_sha256, "report_internal_sha256": internal, "schema_version": VERIFICATION_SCHEMA, "status": "PASS"}
    verification["verification_sha256"] = sha256_value(verification)
    return verification


def load_execution_authorization(path: str | Path, *, plan_path: str | Path) -> dict[str, Any]:
    authorization = v6.v5.v2.predecessor._load_json(path)
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA or authorization.get("status") != "authorized_exact_host_bound_one_row_complete_row_quality_repair":
        raise ValueError("unsupported or changed complete-row repair authorization")
    if authorization.get("approval_text") != EXACT_AUTHORIZATION_TEXT or authorization.get("plan_file_sha256") != file_sha256(plan_path):
        raise ValueError("complete-row approval or plan binding changed")
    if authorization.get("server") != SERVER or authorization.get("limits") != LIMITS or authorization.get("prohibitions") != PROHIBITIONS or authorization.get("predecessor_result") != PREDECESSOR_RESULT:
        raise ValueError("complete-row authorization bindings changed")
    started = authorization.get("authorized_start_unix")
    if not isinstance(started, int) or started <= 0:
        raise ValueError("complete-row authorization start time is invalid")
    return authorization


def load_runtime_reuse_receipt(path: str | Path, *, authorization_path: str | Path) -> dict[str, Any]:
    receipt = v6.v5.v2.predecessor._load_json(path)
    if receipt.get("schema_version") != RUNTIME_REUSE_RECEIPT_SCHEMA or receipt.get("status") != "PASS_exact_existing_localhost_runtime_reused_for_complete_row_quality_repair":
        raise ValueError("unsupported or failed complete-row runtime receipt")
    if receipt.get("authorization_file_sha256") != file_sha256(authorization_path) or receipt.get("server") != SERVER or receipt.get("runtime_reuse") != RUNTIME_REUSE:
        raise ValueError("complete-row runtime receipt binding changed")
    if not all(receipt.get("checks", {}).values()):
        raise ValueError("complete-row runtime checks failed")
    return receipt


def run_complete_row_repair(*, plan_path: Path, authorization_path: Path, runtime_reuse_receipt_path: Path, repository_root: Path, artifact_root: Path, ssh_key: Path, known_hosts: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    plan = load_repair_plan(plan_path.resolve(), repository_root=root)
    authorization = load_execution_authorization(authorization_path.resolve(), plan_path=plan_path.resolve())
    load_runtime_reuse_receipt(runtime_reuse_receipt_path.resolve(), authorization_path=authorization_path.resolve())
    if file_sha256(known_hosts.resolve()) != plan["bindings"]["known_hosts"]["sha256"] or ssh_key.resolve().stat().st_mode & 0o077:
        raise PermissionError("complete-row repair SSH binding or permissions changed")
    if time.time() - int(authorization["authorized_start_unix"]) > LIMITS["maximum_instance_minutes"] * 60:
        raise RuntimeError("complete-row repair authorization expired")
    artifact = artifact_root.resolve()
    if not artifact.is_relative_to((root / "artifacts").resolve()):
        raise ValueError("complete-row artifact root escaped artifacts/")
    private, evidence = artifact / "private", artifact / "evidence"
    private.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    metadata_path = private / "repair-attempt-metadata.jsonl"
    candidate_path = private / "repair-candidate.jsonl"
    report_path = evidence / "run-report.json"
    if any(path.exists() for path in (metadata_path, candidate_path, report_path)):
        raise FileExistsError("refusing to replay a consumed complete-row repair artifact root")
    scenario = v6.v5.v2.predecessor.SAFETY_SCENARIOS[0]
    manifest = repair_manifest()
    attempts: list[dict[str, Any]] = []
    candidate: dict[str, Any] | None = None
    before_gpu = v6.v5.v2.predecessor.gpu_snapshot(ssh_key.resolve(), known_hosts.resolve())
    try:
        for expected in manifest["attempts"]:
            number = int(expected["attempt_number"])
            payload = repair_payload(number)
            request_sha256 = sha256_value(payload)
            if request_sha256 != expected["request_sha256"]:
                raise ValueError("complete-row request manifest changed")
            response = v6.v5.v2.predecessor._post_chat_completion(payload)
            diagnostic = v6.v5._response_metadata(response)
            error: Exception | None = None
            messages: list[dict[str, str]] | None = None
            try:
                messages = v6.v5.v2.predecessor._parse_final_response(response, scenario=scenario, language="english")
                validate_complete_row(messages)
            except (ValueError, json.JSONDecodeError) as observed:
                error = observed
            attempts.append({
                "accepted": error is None,
                "attempt_number": number,
                "completion_tokens": diagnostic["usage"]["completion_tokens"],
                "error_message": str(error) if error is not None else None,
                "error_type": type(error).__name__ if error is not None else None,
                "finish_reason": diagnostic["finish_reason"],
                "max_tokens": MAX_TOKENS,
                "prompt_tokens": diagnostic["usage"]["prompt_tokens"],
                "reasoning_effort": REASONING_EFFORT,
                "record_id": RECORD_ID,
                "request_sha256": request_sha256,
                "response_envelope_sha256": sha256_value(response),
                "total_tokens": diagnostic["usage"]["total_tokens"],
            })
            v6.v5.v2.predecessor._write_jsonl(metadata_path, attempts)
            if error is not None:
                if v6.v5.v2.predecessor._critical_validation_error(error):
                    break
                continue
            assert messages is not None
            candidate = {
                "language_mode": "english",
                "messages": messages,
                "provenance": {
                    "ai_assisted": True,
                    "diagnostic_only": True,
                    "external_inference_api": False,
                    "generation_attempt": number,
                    "human_authored": False,
                    "model_id": v6.v5.v2.predecessor.MODEL_ID,
                    "model_revision": v6.v5.v2.predecessor.MODEL_REVISION,
                    "prompt_contract": PROMPT_CONTRACT,
                    "reasoning_effort": REASONING_EFFORT,
                    "reasoning_or_analysis_retained": False,
                    "request_sha256": request_sha256,
                    "runtime_image_digest": v6.v5.v2.predecessor.RUNTIME_DIGEST,
                    "seed": expected["seed"],
                },
                "record_id": RECORD_ID,
                "rights_status": "private_research_complete_row_repair_pending_owner_review",
                "scenario_id": scenario.scenario_id,
                "schema_version": v6.v5.v2.predecessor.ROW_SCHEMA,
                "task": scenario.task,
            }
            v6.v5.v2.predecessor._write_jsonl(candidate_path, [candidate])
            break
    finally:
        v6.v5.v3._stop_runtime(ssh_key.resolve(), known_hosts.resolve())
    after_gpu = v6.v5.v2.predecessor.gpu_snapshot(ssh_key.resolve(), known_hosts.resolve())
    report: dict[str, Any] = {
        "accepted_final_outputs": 1 if candidate is not None else 0,
        "attempt_metadata_file_sha256": file_sha256(metadata_path),
        "authorization_file_sha256": file_sha256(authorization_path),
        "candidate_file_sha256": file_sha256(candidate_path) if candidate_path.exists() else None,
        "candidate_registry_changed": False,
        "dataset_sealed": False,
        "external_api_used": False,
        "gpu": {"after_stop": after_gpu, "before": before_gpu},
        "inference_calls": len(attempts),
        "machine_complete_row_quality_validation_passed": candidate is not None,
        "owner_review": {"rows": 1 if candidate is not None else 0, "status": "pending" if candidate is not None else "not_run"},
        "plan_file_sha256": file_sha256(plan_path),
        "production_eligible": False,
        "prompt_contract": PROMPT_CONTRACT,
        "runtime_reuse_receipt_file_sha256": file_sha256(runtime_reuse_receipt_path),
        "runtime_stopped": after_gpu["memory_used_mib"] == 0,
        "schema_version": RUN_REPORT_SCHEMA,
        "sft_training_started": False,
        "status": "complete_row_repair_candidate_machine_valid_pending_owner_review" if candidate is not None else "FAIL_CLOSED_complete_row_repair_exhausted",
    }
    report["report_sha256"] = sha256_value(report)
    v6.v5.v2.predecessor._write_json_atomic(report_path, report)
    return report


def verify_complete_row_repair_run(*, plan_path: Path, authorization_path: Path, runtime_reuse_receipt_path: Path, repository_root: Path, artifact_root: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    load_repair_plan(plan_path.resolve(), repository_root=root)
    load_execution_authorization(authorization_path.resolve(), plan_path=plan_path.resolve())
    load_runtime_reuse_receipt(runtime_reuse_receipt_path.resolve(), authorization_path=authorization_path.resolve())
    artifact = artifact_root.resolve()
    report_path = artifact / "evidence/run-report.json"
    report = v6.v5.v2.predecessor._load_json(report_path)
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256", None)
    metadata = v6.v5.v2.predecessor._read_jsonl(artifact / "private/repair-attempt-metadata.jsonl")
    candidate = v6.v5.v2.predecessor._read_jsonl(artifact / "private/repair-candidate.jsonl")
    if candidate:
        validate_complete_row(candidate[0]["messages"])
    checks = {
        "authorization_binding": report.get("authorization_file_sha256") == file_sha256(authorization_path),
        "candidate_count_matches": len(candidate) == report.get("accepted_final_outputs"),
        "candidate_complete_row_reproduced": len(candidate) <= 1,
        "internal_report_hash": internal == sha256_value(unsigned),
        "metadata_is_bounded": 1 <= len(metadata) <= 2,
        "request_manifest_prefix": [item["request_sha256"] for item in metadata] == [item["request_sha256"] for item in repair_manifest()["attempts"][: len(metadata)]],
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
    v6.v5.v2.predecessor._write_json_atomic(artifact / "evidence/run-verification.json", verification)
    if verification["status"] != "PASS":
        raise ValueError("complete-row repair run verification failed")
    return verification
