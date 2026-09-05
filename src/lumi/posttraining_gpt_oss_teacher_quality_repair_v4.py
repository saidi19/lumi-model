# ruff: noqa: E501, SLF001
"""Fail-closed one-row natural-Banglish quality repair for gpt-oss."""

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
from .posttraining_candidate_mix_preflight import file_sha256, sha256_value

PLAN_SCHEMA = "lumi.sft-gpt-oss-teacher-quality-repair-plan.v0.4"
REPORT_SCHEMA = "lumi.sft-gpt-oss-teacher-quality-repair-plan-report.v0.4"
VERIFICATION_SCHEMA = "lumi.sft-gpt-oss-teacher-quality-repair-plan-verification.v0.4"
AUTHORIZATION_SCHEMA = "lumi.sft-gpt-oss-teacher-quality-repair-authorization.v0.4"
RUNTIME_REUSE_RECEIPT_SCHEMA = "lumi.sft-gpt-oss-teacher-quality-repair-runtime-reuse-receipt.v0.4"
RUN_REPORT_SCHEMA = "lumi.sft-gpt-oss-teacher-quality-repair-run-report.v0.4"
RUN_VERIFICATION_SCHEMA = "lumi.sft-gpt-oss-teacher-quality-repair-run-verification.v0.4"

QUALITY_CONTRACT = "lumi.gpt-oss-teacher.prompt-quality-repair.v0.4"
SEED_CONTRACT = "sha256(record_id:prompt-quality-repair-attempt:attempt_number)-u32.v0.4"
RECORD_ID = v3.RECORD_ID
REASONING_EFFORT = "low"
MAX_TOKENS = 1_536
MAXIMUM_ATTEMPTS = 2

SERVER = v2.SERVER
PROHIBITIONS = v2.PROHIBITIONS
RUNTIME_REUSE = v2.RUNTIME_REUSE
RETENTION = v3.RETENTION
OWNER_REJECTION = {
    "accepted_rows": 0,
    "candidate_file_sha256": "f1a6a373dc4d299565c57efb5e962cf9e31fd20d51d21d92eb34b996824ce37b",
    "diagnostic_attempt_metadata_file_sha256": "f84ced5e15d0c47e3bd36f5a1e67d35cb18f998b07b99b981b2598b033c273b8",
    "diagnostic_run_report_file_sha256": "82e92ddf049e111b1458ebbcec6e7dd57813874173f9f47a4c68ce8ada3e8df5",
    "owner_review_file_sha256": "860da2c5944a0ad8515ff699ecc330a16fe1580387c77b6d754f944014ac2d90",
    "owner_review_verification_file_sha256": "e78221fb83bd6bf79d87b2c894073923198d407ff479e177b9da317a0740f1ee",
    "reason": "The embedded English meta-phrase 'refuse the harmful request' is unnatural in the otherwise Banglish assistant response.",
    "rejected_rows": 1,
    "status": "REJECT_prompt_naturalness",
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
BANNED_META_PHRASES = (
    "refuse the harmful request",
    "safe alternative",
    "safety label",
    "scenario id",
    "assistant response",
    "user prompt",
    "dataset",
    "training",
    "policy",
    "gupta shobdo",
)
REQUIRED_RECOVERY_MARKERS = ("password reset", "account recovery", "nijer account", "official")
QUALITY_GATES = {
    "accepted_output_must_pass_frozen_parser": True,
    "assistant_maximum_characters": 320,
    "banned_meta_phrases": list(BANNED_META_PHRASES),
    "critical_factual_safety_or_tool_failures": 0,
    "diagnostic_may_not_satisfy_16_cell_gate": True,
    "owner_review_required_if_accepted": True,
    "required_recovery_markers": list(REQUIRED_RECOVERY_MARKERS),
}

NATURAL_BANGLISH_INSTRUCTION = (
    "Write natural, conversational Banglish in Latin letters. The user_prompt must "
    "contain Tumi or Ami. The assistant_response must contain Ami or Eta and the exact "
    "consecutive words parbo na. Refuse credential theft directly, then suggest an "
    "official password reset or account recovery path for the user's own account. "
    "Common technical words such as login and password are natural; do not translate "
    "password as gupta shobdo. Never echo task instructions or use meta-commentary such "
    "as refuse the harmful request, safe alternative, safety label, scenario ID, policy, "
    "prompt, dataset, or training. Keep the assistant_response under 320 characters. "
    "Style target: Tumi ki onno karo login password ber kore dite parba? / Ami onno "
    "karo login password ber kore dite parbo na. Nijer account hole official password "
    "reset ba account recovery poddhoti use koro."
)

EXACT_AUTHORIZATION_TEXT = (
    "I approve one bounded self-hosted Lumi gpt-oss-120b one-row prompt-quality repair "
    "on the existing exact Lambda host 68.209.73.55, bound to ED25519 host-key "
    "fingerprint SHA256:2ZSQqNUYJhWNjihHnqLQ1YNHxoqtMJeckVJRKlc5hs4 and exactly "
    "one NVIDIA H100 80GB GPU GPU-c7e68b23-95b4-37bd-b40b-c41a7ec716f3 reporting "
    "at least 81,559 MiB FB total without adding the 480 MiB reserved value twice. "
    "Reuse, and do not redownload, the 25 verified openai/gpt-oss-120b model objects "
    "at revision b5c939de8f754692c1647ca79fbf85e8c1e70f8a bound by model receipt "
    "SHA-256 798914d8532d5b36b61711f34591cabe1e03fa6ac1128fe5e3737c029ead1284, "
    "and reuse the existing vLLM image digest "
    "sha256:4d08193d2fd05aadb1b5678f93ae609efb2635df67da45f3efe781c368b34dc8. "
    "Bind the rejected v0.3 owner-review SHA-256 "
    "860da2c5944a0ad8515ff699ecc330a16fe1580387c77b6d754f944014ac2d90. "
    "Run only record lumi-gptoss-safety_helpfulness-banglish-00 with the frozen safety "
    "and schema parser, prompt-quality contract lumi.gpt-oss-teacher.prompt-quality-repair.v0.4, "
    "reasoning effort low, a 1,536 generated-token ceiling per call, distinct "
    "attempt-bound deterministic seeds, and at most 2 attempts. Require natural "
    "Banglish, exact refusal words parbo na, and official password-reset or account-recovery "
    "guidance; reject gupta shobdo and instruction/meta phrases including refuse the "
    "harmful request and safe alternative. Retain rejected attempts only as validation "
    "error, non-sensitive finish_reason, usage token counts, and request/response-envelope "
    "hashes without raw candidate or reasoning text. Maximum 20 additional instance-minutes "
    "/ $1.43 before tax / 0 additional model or runtime download bytes / 1 GiB additional "
    "private writes / 2 sequential inference calls / 1 retry call / 1 accepted repair "
    "output. It may restart and then cleanly stop only the bound canary container, serve "
    "only on remote localhost through pinned SSH, run the one-row frozen machine and "
    "naturalness gates, and prepare the accepted row for one-row owner review. This repair "
    "cannot satisfy the 16-cell gate or authorize any larger canary. No external API, "
    "public port, model or runtime download, registry change, dataset seal, SFT training, "
    "production promotion, unrelated access, new instance, or termination."
)

TOOLING_PATHS = {
    "implementation": "src/lumi/posttraining_gpt_oss_teacher_quality_repair_v4.py",
    "plan_runner": "scripts/plan_sft_gpt_oss_teacher_quality_repair_v4.py",
    "plan_verifier": "scripts/verify_sft_gpt_oss_teacher_quality_repair_v4_plan.py",
    "run_runner": "scripts/run_sft_gpt_oss_teacher_quality_repair_v4.py",
    "run_verifier": "scripts/verify_sft_gpt_oss_teacher_quality_repair_v4.py",
    "test": "tests/test_gpt_oss_teacher_quality_repair_v4.py",
}


def quality_seed(attempt_number: int) -> int:
    if not 1 <= attempt_number <= MAXIMUM_ATTEMPTS:
        raise ValueError("quality-repair attempt number is outside the reviewed contract")
    material = f"{RECORD_ID}:prompt-quality-repair-attempt:{attempt_number}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def quality_messages(attempt_number: int) -> list[dict[str, str]]:
    retry = " This is the final repair attempt; silently re-check every naturalness gate." if attempt_number > 1 else ""
    system = (
        "Return only one compact JSON object with scenario_id, user_prompt, assistant_response, "
        "safety_label, tool_call, and tool_result. Set scenario_id to credential_theft, "
        "safety_label to refuse, and both tool fields to null. Never expose reasoning. "
        + NATURAL_BANGLISH_INSTRUCTION
        + retry
    )
    user = (
        "Create one natural Banglish credential-theft refusal row. Do not copy or mention "
        f"these instructions. Attempt {attempt_number}/{MAXIMUM_ATTEMPTS}. Return JSON only."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def quality_payload(attempt_number: int) -> dict[str, Any]:
    scenario = v2.predecessor.SAFETY_SCENARIOS[0]
    payload = copy.deepcopy(v2.predecessor.chat_completion_payload(RECORD_ID, scenario, "banglish"))
    payload["chat_template_kwargs"] = {"reasoning_effort": REASONING_EFFORT}
    payload["max_tokens"] = MAX_TOKENS
    payload["messages"] = quality_messages(attempt_number)
    payload["seed"] = quality_seed(attempt_number)
    payload["temperature"] = 0.3
    return payload


def quality_manifest() -> dict[str, Any]:
    return {
        "attempts": [
            {
                "attempt_number": attempt,
                "max_tokens": MAX_TOKENS,
                "reasoning_effort": REASONING_EFFORT,
                "request_sha256": sha256_value(quality_payload(attempt)),
                "seed": quality_seed(attempt),
            }
            for attempt in range(1, MAXIMUM_ATTEMPTS + 1)
        ],
        "language": "banglish",
        "record_id": RECORD_ID,
        "scenario_id": "credential_theft",
        "task": "safety_helpfulness",
        "variant": 0,
    }


def validate_naturalness(messages: Sequence[Mapping[str, str]]) -> None:
    if len(messages) != 2:
        raise ValueError("quality-repair message count changed")
    user = str(messages[0].get("content", ""))
    assistant = str(messages[1].get("content", ""))
    combined = f"{user}\n{assistant}".lower()
    for phrase in BANNED_META_PHRASES:
        if phrase in combined:
            raise ValueError(f"quality-repair banned meta phrase: {phrase}")
    if len(assistant) > QUALITY_GATES["assistant_maximum_characters"]:
        raise ValueError("quality-repair assistant response is too long")
    if not any(marker in assistant.lower() for marker in REQUIRED_RECOVERY_MARKERS):
        raise ValueError("quality-repair response lacks account-recovery guidance")
    if "parbo na" not in assistant.lower():
        raise ValueError("quality-repair response lacks exact refusal words")


def load_quality_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    plan = v2.predecessor._load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("status") != "quality_repair_plan_pending_explicit_execution_authorization":
        raise ValueError("unsupported or changed gpt-oss quality-repair plan")
    if plan.get("server") != SERVER or plan.get("runtime_reuse") != RUNTIME_REUSE:
        raise ValueError("gpt-oss quality-repair host or runtime binding changed")
    if plan.get("owner_rejection") != OWNER_REJECTION:
        raise ValueError("gpt-oss quality-repair owner rejection changed")
    if plan.get("limits") != LIMITS or plan.get("quality_gates") != QUALITY_GATES:
        raise ValueError("gpt-oss quality-repair limits or gates changed")
    if plan.get("prohibitions") != PROHIBITIONS or plan.get("retention") != RETENTION:
        raise ValueError("gpt-oss quality-repair prohibitions or retention changed")
    if any(bool(value) for value in plan.get("authorization", {}).values() if isinstance(value, bool)):
        raise ValueError("gpt-oss quality-repair unexpectedly authorizes execution")
    if plan.get("next_gate", {}).get("exact_authorization_text") != EXACT_AUTHORIZATION_TEXT:
        raise ValueError("gpt-oss quality-repair authorization text changed")
    contract = plan.get("quality_contract", {})
    if contract.get("id") != QUALITY_CONTRACT or contract.get("seed_contract") != SEED_CONTRACT:
        raise ValueError("gpt-oss quality-repair contract changed")
    if contract.get("manifest_sha256") != sha256_value(quality_manifest()):
        raise ValueError("gpt-oss quality-repair manifest changed")
    v2.predecessor._resolve_bindings(plan, repository_root)
    return plan


def build_plan_report(*, plan: Mapping[str, Any], plan_file_sha256: str, repository_root: str | Path) -> dict[str, Any]:
    resolved = v2.predecessor._resolve_bindings(plan, repository_root)
    manifest = quality_manifest()
    report: dict[str, Any] = {
        "authorization": plan["authorization"],
        "bindings": {name: {"path": plan["bindings"][name]["path"], "sha256": file_sha256(path)} for name, path in sorted(resolved.items())},
        "checks": {
            "banned_owner_phrase_is_machine_gated": "refuse the harmful request" in BANNED_META_PHRASES,
            "distinct_attempt_requests": len({item["request_sha256"] for item in manifest["attempts"]}) == 2,
            "distinct_attempt_seeds": len({item["seed"] for item in manifest["attempts"]}) == 2,
            "exact_one_record": manifest["record_id"] == RECORD_ID,
            "low_reasoning_exact": {item["reasoning_effort"] for item in manifest["attempts"]} == {"low"},
            "no_execution_authorized": not any(bool(value) for value in plan["authorization"].values() if isinstance(value, bool)),
            "no_model_or_runtime_download": plan["limits"]["maximum_network_bytes"] == 0,
            "owner_rejection_bound": plan["owner_rejection"] == OWNER_REJECTION,
            "recovery_guidance_required": set(REQUIRED_RECOVERY_MARKERS) == set(plan["quality_gates"]["required_recovery_markers"]),
            "token_ceiling_exact": {item["max_tokens"] for item in manifest["attempts"]} == {MAX_TOKENS},
        },
        "limits": plan["limits"],
        "manifest": manifest,
        "next_gate": plan["next_gate"],
        "observed_at": plan["observed_at"],
        "owner_rejection": plan["owner_rejection"],
        "plan_file_sha256": plan_file_sha256,
        "prohibitions": plan["prohibitions"],
        "quality_contract": plan["quality_contract"],
        "quality_gates": plan["quality_gates"],
        "retention": plan["retention"],
        "runtime_reuse": plan["runtime_reuse"],
        "schema_version": REPORT_SCHEMA,
        "server": plan["server"],
        "status": "verified_prompt_quality_repair_plan_no_execution_authorized",
        "tooling": {name: {"path": tooling_path, "sha256": file_sha256(Path(repository_root) / tooling_path)} for name, tooling_path in sorted(TOOLING_PATHS.items())},
    }
    if not all(report["checks"].values()):
        raise ValueError("gpt-oss quality-repair plan checks failed")
    report["report_sha256"] = sha256_value(report)
    return report


def build_plan_verification(*, report: Mapping[str, Any], report_file_sha256: str) -> dict[str, Any]:
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256", None)
    if internal != sha256_value(unsigned):
        raise ValueError("gpt-oss quality-repair report internal hash changed")
    checks = {
        "all_plan_checks_pass": all(report.get("checks", {}).values()),
        "authorization_pending": report.get("status") == "verified_prompt_quality_repair_plan_no_execution_authorized",
        "call_ceiling_two": report.get("limits", {}).get("maximum_inference_calls") == 2,
        "naturalness_gate_bound": "refuse the harmful request" in report.get("quality_gates", {}).get("banned_meta_phrases", []),
        "owner_rejection_exact": report.get("owner_rejection") == OWNER_REJECTION,
        "zero_download_ceiling": report.get("limits", {}).get("maximum_network_bytes") == 0,
    }
    if not all(checks.values()):
        raise ValueError("gpt-oss quality-repair plan verification failed")
    verification: dict[str, Any] = {"checks": checks, "report_file_sha256": report_file_sha256, "report_internal_sha256": internal, "schema_version": VERIFICATION_SCHEMA, "status": "PASS"}
    verification["verification_sha256"] = sha256_value(verification)
    return verification


def load_quality_authorization(path: str | Path, *, plan_path: str | Path) -> dict[str, Any]:
    authorization = v2.predecessor._load_json(path)
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA or authorization.get("status") != "authorized_exact_host_bound_one_row_prompt_quality_repair":
        raise ValueError("unsupported or changed gpt-oss quality-repair authorization")
    if authorization.get("approval_text") != EXACT_AUTHORIZATION_TEXT or authorization.get("plan_file_sha256") != file_sha256(plan_path):
        raise ValueError("gpt-oss quality-repair approval or plan binding changed")
    if authorization.get("server") != SERVER or authorization.get("limits") != LIMITS or authorization.get("prohibitions") != PROHIBITIONS or authorization.get("owner_rejection") != OWNER_REJECTION:
        raise ValueError("gpt-oss quality-repair authorization bindings changed")
    started = authorization.get("authorized_start_unix")
    if not isinstance(started, int) or started <= 0:
        raise ValueError("gpt-oss quality-repair authorization start time is invalid")
    return authorization


def load_runtime_reuse_receipt(path: str | Path, *, authorization_path: str | Path) -> dict[str, Any]:
    receipt = v2.predecessor._load_json(path)
    if receipt.get("schema_version") != RUNTIME_REUSE_RECEIPT_SCHEMA or receipt.get("status") != "PASS_exact_existing_localhost_runtime_reused_for_quality_repair":
        raise ValueError("unsupported or failed gpt-oss quality-repair runtime receipt")
    if receipt.get("authorization_file_sha256") != file_sha256(authorization_path) or receipt.get("server") != SERVER or receipt.get("runtime_reuse") != RUNTIME_REUSE:
        raise ValueError("gpt-oss quality-repair runtime receipt binding changed")
    if not all(receipt.get("checks", {}).values()):
        raise ValueError("gpt-oss quality-repair runtime checks failed")
    return receipt


def run_quality_repair(*, plan_path: Path, authorization_path: Path, runtime_reuse_receipt_path: Path, repository_root: Path, artifact_root: Path, ssh_key: Path, known_hosts: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    plan = load_quality_plan(plan_path.resolve(), repository_root=root)
    authorization = load_quality_authorization(authorization_path.resolve(), plan_path=plan_path.resolve())
    load_runtime_reuse_receipt(runtime_reuse_receipt_path.resolve(), authorization_path=authorization_path.resolve())
    if file_sha256(known_hosts.resolve()) != plan["bindings"]["known_hosts"]["sha256"] or ssh_key.resolve().stat().st_mode & 0o077:
        raise PermissionError("gpt-oss quality-repair SSH binding or permissions changed")
    if time.time() - int(authorization["authorized_start_unix"]) > LIMITS["maximum_instance_minutes"] * 60:
        raise RuntimeError("gpt-oss quality-repair authorization expired")
    artifact = artifact_root.resolve()
    if not artifact.is_relative_to((root / "artifacts").resolve()):
        raise ValueError("gpt-oss quality-repair artifact root escaped artifacts/")
    private, evidence = artifact / "private", artifact / "evidence"
    private.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    metadata_path, candidate_path, report_path = private / "quality-attempt-metadata.jsonl", private / "quality-candidate.jsonl", evidence / "run-report.json"
    if any(path.exists() for path in (metadata_path, candidate_path, report_path)):
        raise FileExistsError("refusing to replay a consumed gpt-oss quality-repair artifact root")
    scenario = v2.predecessor.SAFETY_SCENARIOS[0]
    manifest = quality_manifest()
    attempts: list[dict[str, Any]] = []
    candidate: dict[str, Any] | None = None
    before_gpu = v2.predecessor.gpu_snapshot(ssh_key.resolve(), known_hosts.resolve())
    try:
        for expected in manifest["attempts"]:
            number = int(expected["attempt_number"])
            payload = quality_payload(number)
            request_sha256 = sha256_value(payload)
            if request_sha256 != expected["request_sha256"]:
                raise ValueError("gpt-oss quality-repair request manifest changed")
            response = v2.predecessor._post_chat_completion(payload)
            diagnostic = v3._response_metadata(response)
            error: Exception | None = None
            messages: list[dict[str, str]] | None = None
            try:
                messages = v2.predecessor._parse_final_response(response, scenario=scenario, language="banglish")
                validate_naturalness(messages)
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
            v2.predecessor._write_jsonl(metadata_path, attempts)
            if error is not None:
                if v2.predecessor._critical_validation_error(error):
                    break
                continue
            assert messages is not None
            candidate = {
                "language_mode": "banglish", "messages": messages,
                "provenance": {"ai_assisted": True, "diagnostic_only": True, "external_inference_api": False, "generation_attempt": number, "human_authored": False, "model_id": v2.predecessor.MODEL_ID, "model_revision": v2.predecessor.MODEL_REVISION, "quality_contract": QUALITY_CONTRACT, "reasoning_effort": REASONING_EFFORT, "reasoning_or_analysis_retained": False, "request_sha256": request_sha256, "runtime_image_digest": v2.predecessor.RUNTIME_DIGEST, "seed": expected["seed"]},
                "record_id": RECORD_ID, "rights_status": "private_research_quality_repair_pending_owner_review",
                "scenario_id": scenario.scenario_id, "schema_version": v2.predecessor.ROW_SCHEMA, "task": scenario.task,
            }
            v2.predecessor._write_jsonl(candidate_path, [candidate])
            break
    finally:
        v3._stop_runtime(ssh_key.resolve(), known_hosts.resolve())
    after_gpu = v2.predecessor.gpu_snapshot(ssh_key.resolve(), known_hosts.resolve())
    report: dict[str, Any] = {
        "accepted_final_outputs": 1 if candidate is not None else 0,
        "attempt_metadata_file_sha256": file_sha256(metadata_path), "authorization_file_sha256": file_sha256(authorization_path),
        "candidate_file_sha256": file_sha256(candidate_path) if candidate_path.exists() else None,
        "candidate_registry_changed": False, "dataset_sealed": False, "external_api_used": False,
        "gpu": {"after_stop": after_gpu, "before": before_gpu}, "inference_calls": len(attempts),
        "machine_and_naturalness_validation_passed": candidate is not None,
        "owner_review": {"rows": 1 if candidate is not None else 0, "status": "pending" if candidate is not None else "not_run"},
        "plan_file_sha256": file_sha256(plan_path), "production_eligible": False, "quality_contract": QUALITY_CONTRACT,
        "runtime_reuse_receipt_file_sha256": file_sha256(runtime_reuse_receipt_path), "runtime_stopped": after_gpu["memory_used_mib"] == 0,
        "schema_version": RUN_REPORT_SCHEMA, "sft_training_started": False,
        "status": "quality_repair_candidate_machine_valid_pending_owner_review" if candidate is not None else "FAIL_CLOSED_prompt_quality_repair_exhausted",
    }
    report["report_sha256"] = sha256_value(report)
    v2.predecessor._write_json_atomic(report_path, report)
    return report


def verify_quality_run(*, plan_path: Path, authorization_path: Path, runtime_reuse_receipt_path: Path, repository_root: Path, artifact_root: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    load_quality_plan(plan_path.resolve(), repository_root=root)
    load_quality_authorization(authorization_path.resolve(), plan_path=plan_path.resolve())
    load_runtime_reuse_receipt(runtime_reuse_receipt_path.resolve(), authorization_path=authorization_path.resolve())
    artifact = artifact_root.resolve()
    report_path = artifact / "evidence/run-report.json"
    report = v2.predecessor._load_json(report_path)
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256", None)
    metadata = v2.predecessor._read_jsonl(artifact / "private/quality-attempt-metadata.jsonl")
    candidate = v2.predecessor._read_jsonl(artifact / "private/quality-candidate.jsonl")
    if candidate:
        validate_naturalness(candidate[0]["messages"])
    checks = {
        "authorization_binding": report.get("authorization_file_sha256") == file_sha256(authorization_path),
        "candidate_count_matches": len(candidate) == report.get("accepted_final_outputs"),
        "candidate_naturalness_reproduced": len(candidate) <= 1,
        "internal_report_hash": internal == sha256_value(unsigned),
        "metadata_is_bounded": 1 <= len(metadata) <= 2,
        "request_manifest_prefix": [item["request_sha256"] for item in metadata] == [item["request_sha256"] for item in quality_manifest()["attempts"][: len(metadata)]],
        "runtime_binding": report.get("runtime_reuse_receipt_file_sha256") == file_sha256(runtime_reuse_receipt_path),
        "runtime_stopped": report.get("runtime_stopped") is True and report.get("gpu", {}).get("after_stop", {}).get("memory_used_mib") == 0,
        "zero_production_actions": report.get("candidate_registry_changed") is False and report.get("dataset_sealed") is False and report.get("sft_training_started") is False and report.get("production_eligible") is False,
    }
    verification: dict[str, Any] = {"checks": checks, "run_report_file_sha256": file_sha256(report_path), "run_report_internal_sha256": internal, "schema_version": RUN_VERIFICATION_SCHEMA, "status": "PASS" if all(checks.values()) else "FAIL"}
    verification["verification_sha256"] = sha256_value(verification)
    v2.predecessor._write_json_atomic(artifact / "evidence/run-verification.json", verification)
    if verification["status"] != "PASS":
        raise ValueError("gpt-oss quality-repair run verification failed")
    return verification
