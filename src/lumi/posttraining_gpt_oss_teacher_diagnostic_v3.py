# ruff: noqa: E501, SLF001
"""Fail-closed one-row stopping diagnostic for the host-bound gpt-oss teacher."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import posttraining_gpt_oss_teacher_canary_v2 as v2
from .posttraining_candidate_mix_preflight import file_sha256, sha256_value

PLAN_SCHEMA = "lumi.sft-gpt-oss-teacher-diagnostic-plan.v0.3"
REPORT_SCHEMA = "lumi.sft-gpt-oss-teacher-diagnostic-plan-report.v0.3"
VERIFICATION_SCHEMA = "lumi.sft-gpt-oss-teacher-diagnostic-plan-verification.v0.3"
AUTHORIZATION_SCHEMA = "lumi.sft-gpt-oss-teacher-diagnostic-authorization.v0.3"
RUNTIME_REUSE_RECEIPT_SCHEMA = "lumi.sft-gpt-oss-teacher-diagnostic-runtime-reuse-receipt.v0.3"
RUN_REPORT_SCHEMA = "lumi.sft-gpt-oss-teacher-diagnostic-run-report.v0.3"
RUN_VERIFICATION_SCHEMA = "lumi.sft-gpt-oss-teacher-diagnostic-run-verification.v0.3"

PROMPT_CONTRACT = "lumi.gpt-oss-teacher-prompt.banglish-repair.v0.2"
DIAGNOSTIC_CONTRACT = "lumi.gpt-oss-teacher.finish-diagnostic.v0.3"
SEED_CONTRACT = "sha256(record_id:finish-diagnostic-attempt:attempt_number)-u32.v0.3"
RECORD_ID = "lumi-gptoss-safety_helpfulness-banglish-00"
ATTEMPT_TOKEN_BUDGETS = (1_024, 1_536, 2_048)
REASONING_EFFORT = "low"

SERVER = v2.SERVER
PROHIBITIONS = v2.PROHIBITIONS
RUNTIME_REUSE = v2.RUNTIME_REUSE
PRIOR_FAILURE = {
    "accepted_final_outputs": 0,
    "authorization_file_sha256": "0b283c943bc49c05acc8cbe3338de009acda631359c752a19b4d0c01c31de6eb",
    "failure_report_file_sha256": "563a17f404de10a0b167a4417e98b9ada02b112d53ecbb639d40c95bd6473bc9",
    "failure_verification_file_sha256": "f7aa5ef270c37c7b6b8f65e7ab5c08e2a9d3697e27c4612a5fcf08e1b96b0036",
    "first_record_id": RECORD_ID,
    "frozen_validation_error": "vLLM final response did not finish with stop",
    "inference_calls": 3,
    "rejection_metadata_file_sha256": "e4ee25400eb57343961203404202ce8458c8191fbe9c0d21c811d7f178e1a2de",
    "runtime_retries": 2,
    "status": "FAIL_CLOSED_first_row_attempt_ceiling",
}
LIMITS = {
    "exact_diagnostic_records": 1,
    "maximum_accepted_final_outputs": 1,
    "maximum_attempts_per_record": 3,
    "maximum_context_tokens": 4_096,
    "maximum_generated_tokens_by_attempt": list(ATTEMPT_TOKEN_BUDGETS),
    "maximum_generated_tokens_total": sum(ATTEMPT_TOKEN_BUDGETS),
    "maximum_inference_calls": 3,
    "maximum_instance_minutes": 30,
    "maximum_network_bytes": 0,
    "maximum_private_write_bytes": 1 * 1024**3,
    "maximum_runtime_retries": 2,
    "maximum_usd_before_tax": 2.15,
    "owner_review_rows_if_accepted": 1,
}
QUALITY_GATES = {
    "accepted_output_must_pass_frozen_parser": True,
    "critical_factual_safety_or_tool_failures": 0,
    "diagnostic_may_not_satisfy_16_cell_gate": True,
    "owner_review_required_if_accepted": True,
}
RETENTION = {
    "accepted_final_channel_candidate_text": True,
    "finish_reason": True,
    "reasoning_or_analysis": False,
    "rejected_raw_text": False,
    "request_and_response_envelope_hashes": True,
    "usage_token_counts": True,
}

EXACT_AUTHORIZATION_TEXT = (
    "I approve one bounded self-hosted Lumi gpt-oss-120b one-row stopping diagnostic "
    "on the existing exact Lambda host 68.209.73.55, bound to ED25519 host-key "
    "fingerprint SHA256:2ZSQqNUYJhWNjihHnqLQ1YNHxoqtMJeckVJRKlc5hs4 and exactly "
    "one NVIDIA H100 80GB GPU GPU-c7e68b23-95b4-37bd-b40b-c41a7ec716f3 reporting "
    "at least 81,559 MiB FB total without adding the 480 MiB reserved value twice. "
    "Reuse, and do not redownload, the 25 verified openai/gpt-oss-120b model objects "
    "at revision b5c939de8f754692c1647ca79fbf85e8c1e70f8a bound by model receipt "
    "SHA-256 798914d8532d5b36b61711f34591cabe1e03fa6ac1128fe5e3737c029ead1284, "
    "and reuse the existing vLLM image digest "
    "sha256:4d08193d2fd05aadb1b5678f93ae609efb2635df67da45f3efe781c368b34dc8. "
    "Bind the v0.2 failure report SHA-256 "
    "563a17f404de10a0b167a4417e98b9ada02b112d53ecbb639d40c95bd6473bc9. "
    "Run only record lumi-gptoss-safety_helpfulness-banglish-00 with the unchanged "
    "Banglish-repair prompt and frozen safety/parser gates, reasoning effort low, "
    "distinct attempt-bound deterministic seeds, and sequential generated-token "
    "ceilings of 1,024, 1,536, and 2,048 across at most 3 attempts. Retain rejected "
    "attempts only as validation error, non-sensitive finish_reason, usage token counts, "
    "and request/response-envelope hashes without raw candidate or reasoning text. "
    "Maximum 30 additional instance-minutes / $2.15 before tax / 0 additional model "
    "or runtime download bytes / 1 GiB additional private writes / 3 sequential "
    "inference calls / 2 retry calls / 1 accepted diagnostic output. It may restart "
    "and then cleanly stop only the bound canary container, serve only on remote "
    "localhost through pinned SSH, run the one-row frozen machine validation, and "
    "prepare the accepted row for one-row owner review. This diagnostic cannot satisfy "
    "the 16-cell gate or authorize any larger canary. No external API, public port, "
    "model or runtime download, registry change, dataset seal, SFT training, production "
    "promotion, unrelated access, new instance, or termination."
)

TOOLING_PATHS = {
    "implementation": "src/lumi/posttraining_gpt_oss_teacher_diagnostic_v3.py",
    "plan_runner": "scripts/plan_sft_gpt_oss_teacher_diagnostic_v3.py",
    "plan_verifier": "scripts/verify_sft_gpt_oss_teacher_diagnostic_v3_plan.py",
    "run_runner": "scripts/run_sft_gpt_oss_teacher_diagnostic_v3.py",
    "run_verifier": "scripts/verify_sft_gpt_oss_teacher_diagnostic_v3.py",
    "test": "tests/test_gpt_oss_teacher_diagnostic_v3.py",
}


def diagnostic_seed(record_id: str, attempt_number: int) -> int:
    if not 1 <= attempt_number <= len(ATTEMPT_TOKEN_BUDGETS):
        raise ValueError("diagnostic attempt number is outside the reviewed contract")
    material = f"{record_id}:finish-diagnostic-attempt:{attempt_number}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def diagnostic_payload(attempt_number: int) -> dict[str, Any]:
    scenario = v2.predecessor.SAFETY_SCENARIOS[0]
    payload = copy.deepcopy(
        v2.chat_completion_payload(
            RECORD_ID,
            scenario,
            "banglish",
            attempt_number=attempt_number,
        )
    )
    payload["chat_template_kwargs"] = {"reasoning_effort": REASONING_EFFORT}
    payload["max_tokens"] = ATTEMPT_TOKEN_BUDGETS[attempt_number - 1]
    payload["seed"] = diagnostic_seed(RECORD_ID, attempt_number)
    return payload


def diagnostic_manifest() -> dict[str, Any]:
    scenario = v2.predecessor.SAFETY_SCENARIOS[0]
    return {
        "attempts": [
            {
                "attempt_number": attempt,
                "max_tokens": ATTEMPT_TOKEN_BUDGETS[attempt - 1],
                "reasoning_effort": REASONING_EFFORT,
                "request_sha256": sha256_value(diagnostic_payload(attempt)),
                "seed": diagnostic_seed(RECORD_ID, attempt),
            }
            for attempt in range(1, len(ATTEMPT_TOKEN_BUDGETS) + 1)
        ],
        "language": "banglish",
        "record_id": RECORD_ID,
        "scenario_id": scenario.scenario_id,
        "task": scenario.task,
        "variant": scenario.variant,
    }


def load_diagnostic_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    plan = v2.predecessor._load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unsupported gpt-oss diagnostic plan")
    if plan.get("status") != "diagnostic_plan_pending_explicit_execution_authorization":
        raise ValueError("gpt-oss diagnostic plan status changed")
    if plan.get("server") != SERVER or plan.get("runtime_reuse") != RUNTIME_REUSE:
        raise ValueError("gpt-oss diagnostic host or runtime binding changed")
    if plan.get("prior_failure") != PRIOR_FAILURE:
        raise ValueError("gpt-oss diagnostic prior-failure binding changed")
    if plan.get("limits") != LIMITS or plan.get("quality_gates") != QUALITY_GATES:
        raise ValueError("gpt-oss diagnostic limits or gates changed")
    if plan.get("prohibitions") != PROHIBITIONS or plan.get("retention") != RETENTION:
        raise ValueError("gpt-oss diagnostic prohibitions or retention changed")
    if any(bool(value) for value in plan.get("authorization", {}).values() if isinstance(value, bool)):
        raise ValueError("gpt-oss diagnostic unexpectedly authorizes execution")
    if plan.get("next_gate", {}).get("exact_authorization_text") != EXACT_AUTHORIZATION_TEXT:
        raise ValueError("gpt-oss diagnostic authorization text changed")
    contract = plan.get("diagnostic_contract", {})
    if contract.get("id") != DIAGNOSTIC_CONTRACT:
        raise ValueError("gpt-oss diagnostic contract changed")
    if contract.get("prompt_contract") != PROMPT_CONTRACT:
        raise ValueError("gpt-oss diagnostic prompt contract changed")
    if contract.get("seed_contract") != SEED_CONTRACT:
        raise ValueError("gpt-oss diagnostic seed contract changed")
    if contract.get("manifest_sha256") != sha256_value(diagnostic_manifest()):
        raise ValueError("gpt-oss diagnostic manifest hash changed")
    v2.predecessor._resolve_bindings(plan, repository_root)
    return plan


def build_plan_report(*, plan: Mapping[str, Any], plan_file_sha256: str, repository_root: str | Path) -> dict[str, Any]:
    resolved = v2.predecessor._resolve_bindings(plan, repository_root)
    manifest = diagnostic_manifest()
    attempts = manifest["attempts"]
    report: dict[str, Any] = {
        "authorization": plan["authorization"],
        "bindings": {
            name: {"path": plan["bindings"][name]["path"], "sha256": file_sha256(path)}
            for name, path in sorted(resolved.items())
        },
        "checks": {
            "attempt_request_hashes_distinct": len({item["request_sha256"] for item in attempts}) == 3,
            "attempt_seeds_distinct": len({item["seed"] for item in attempts}) == 3,
            "exact_one_record": manifest["record_id"] == RECORD_ID,
            "frozen_banglish_prompt_reused": plan["diagnostic_contract"]["prompt_contract"] == PROMPT_CONTRACT,
            "no_execution_authorized": not any(bool(value) for value in plan["authorization"].values() if isinstance(value, bool)),
            "no_model_or_runtime_download": plan["limits"]["maximum_network_bytes"] == 0,
            "non_sensitive_finish_metadata_only": plan["retention"] == RETENTION,
            "prior_failure_bound": plan["prior_failure"] == PRIOR_FAILURE,
            "reasoning_effort_low": {item["reasoning_effort"] for item in attempts} == {"low"},
            "token_staircase_exact": [item["max_tokens"] for item in attempts] == list(ATTEMPT_TOKEN_BUDGETS),
        },
        "diagnostic_contract": plan["diagnostic_contract"],
        "limits": plan["limits"],
        "manifest": manifest,
        "next_gate": plan["next_gate"],
        "observed_at": plan["observed_at"],
        "plan_file_sha256": plan_file_sha256,
        "prior_failure": plan["prior_failure"],
        "prohibitions": plan["prohibitions"],
        "quality_gates": plan["quality_gates"],
        "retention": plan["retention"],
        "runtime_reuse": plan["runtime_reuse"],
        "schema_version": REPORT_SCHEMA,
        "server": plan["server"],
        "status": "verified_one_row_diagnostic_plan_no_execution_authorized",
        "tooling": {
            name: {"path": tooling_path, "sha256": file_sha256(Path(repository_root) / tooling_path)}
            for name, tooling_path in sorted(TOOLING_PATHS.items())
        },
    }
    if not all(report["checks"].values()):
        raise ValueError("gpt-oss diagnostic plan checks failed")
    report["report_sha256"] = sha256_value(report)
    return report


def build_plan_verification(*, report: Mapping[str, Any], report_file_sha256: str) -> dict[str, Any]:
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256", None)
    if internal != sha256_value(unsigned):
        raise ValueError("gpt-oss diagnostic report internal hash changed")
    checks = {
        "all_plan_checks_pass": all(report.get("checks", {}).values()),
        "authorization_pending": report.get("status") == "verified_one_row_diagnostic_plan_no_execution_authorized",
        "call_ceiling_three": report.get("limits", {}).get("maximum_inference_calls") == 3,
        "one_record_only": report.get("manifest", {}).get("record_id") == RECORD_ID,
        "prior_failure_exact": report.get("prior_failure") == PRIOR_FAILURE,
        "token_staircase_exact": [item["max_tokens"] for item in report.get("manifest", {}).get("attempts", [])] == list(ATTEMPT_TOKEN_BUDGETS),
        "zero_download_ceiling": report.get("limits", {}).get("maximum_network_bytes") == 0,
    }
    if not all(checks.values()):
        raise ValueError("gpt-oss diagnostic plan verification failed")
    verification: dict[str, Any] = {
        "checks": checks,
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": internal,
        "schema_version": VERIFICATION_SCHEMA,
        "status": "PASS",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification


def load_diagnostic_authorization(path: str | Path, *, plan_path: str | Path) -> dict[str, Any]:
    authorization = v2.predecessor._load_json(path)
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA:
        raise ValueError("unsupported gpt-oss diagnostic authorization")
    if authorization.get("status") != "authorized_exact_host_bound_one_row_stopping_diagnostic":
        raise ValueError("gpt-oss diagnostic authorization status changed")
    if authorization.get("approval_text") != EXACT_AUTHORIZATION_TEXT:
        raise ValueError("gpt-oss diagnostic approval text is not exact")
    if authorization.get("plan_file_sha256") != file_sha256(plan_path):
        raise ValueError("gpt-oss diagnostic plan binding changed")
    if authorization.get("server") != SERVER or authorization.get("limits") != LIMITS:
        raise ValueError("gpt-oss diagnostic server or limit authorization changed")
    if authorization.get("prohibitions") != PROHIBITIONS or authorization.get("prior_failure") != PRIOR_FAILURE:
        raise ValueError("gpt-oss diagnostic authorization bindings changed")
    started = authorization.get("authorized_start_unix")
    if not isinstance(started, int) or started <= 0:
        raise ValueError("gpt-oss diagnostic authorization start time is invalid")
    return authorization


def load_runtime_reuse_receipt(path: str | Path, *, authorization_path: str | Path) -> dict[str, Any]:
    receipt = v2.predecessor._load_json(path)
    if receipt.get("schema_version") != RUNTIME_REUSE_RECEIPT_SCHEMA:
        raise ValueError("unsupported gpt-oss diagnostic runtime receipt")
    if receipt.get("status") != "PASS_exact_existing_localhost_runtime_reused_for_diagnostic":
        raise ValueError("gpt-oss diagnostic runtime receipt is not PASS")
    if receipt.get("authorization_file_sha256") != file_sha256(authorization_path):
        raise ValueError("gpt-oss diagnostic runtime authorization binding changed")
    if receipt.get("server") != SERVER or receipt.get("runtime_reuse") != RUNTIME_REUSE:
        raise ValueError("gpt-oss diagnostic runtime binding changed")
    if not all(receipt.get("checks", {}).values()):
        raise ValueError("gpt-oss diagnostic runtime checks failed")
    return receipt


def _response_metadata(response: Mapping[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    finish_reason: str | None = None
    if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], Mapping):
        observed = choices[0].get("finish_reason")
        if observed is None or isinstance(observed, str):
            finish_reason = observed
    usage = response.get("usage")
    counts: dict[str, int | None] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key) if isinstance(usage, Mapping) else None
        counts[key] = value if isinstance(value, int) and value >= 0 else None
    return {"finish_reason": finish_reason, "usage": counts}


def _stop_runtime(ssh_key: Path, known_hosts: Path) -> None:
    subprocess.run(
        v2.predecessor._ssh_prefix(ssh_key, known_hosts)
        + ["sudo docker stop --timeout 30 lumi-gpt-oss-teacher-canary-v0-1"],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


def run_diagnostic(
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
    plan = load_diagnostic_plan(plan_path.resolve(), repository_root=root)
    authorization = load_diagnostic_authorization(authorization_path.resolve(), plan_path=plan_path.resolve())
    load_runtime_reuse_receipt(runtime_reuse_receipt_path.resolve(), authorization_path=authorization_path.resolve())
    if file_sha256(known_hosts.resolve()) != plan["bindings"]["known_hosts"]["sha256"]:
        raise ValueError("gpt-oss diagnostic known-hosts binding changed")
    if ssh_key.resolve().stat().st_mode & 0o077:
        raise PermissionError("gpt-oss diagnostic SSH private key must remain private")
    if time.time() - int(authorization["authorized_start_unix"]) > LIMITS["maximum_instance_minutes"] * 60:
        raise RuntimeError("gpt-oss diagnostic authorization expired")
    artifact = artifact_root.resolve()
    if not artifact.is_relative_to((root / "artifacts").resolve()):
        raise ValueError("gpt-oss diagnostic artifact root escaped artifacts/")
    private = artifact / "private"
    evidence = artifact / "evidence"
    private.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    metadata_path = private / "diagnostic-attempt-metadata.jsonl"
    candidate_path = private / "diagnostic-candidate.jsonl"
    report_path = evidence / "run-report.json"
    if metadata_path.exists() or candidate_path.exists() or report_path.exists():
        raise FileExistsError("refusing to replay a consumed gpt-oss diagnostic artifact root")

    scenario = v2.predecessor.SAFETY_SCENARIOS[0]
    manifest = diagnostic_manifest()
    attempts: list[dict[str, Any]] = []
    candidate: dict[str, Any] | None = None
    before_gpu = v2.predecessor.gpu_snapshot(ssh_key.resolve(), known_hosts.resolve())
    try:
        for expected in manifest["attempts"]:
            attempt_number = int(expected["attempt_number"])
            payload = diagnostic_payload(attempt_number)
            request_sha256 = sha256_value(payload)
            if request_sha256 != expected["request_sha256"]:
                raise ValueError("gpt-oss diagnostic request manifest changed")
            response = v2.predecessor._post_chat_completion(payload)
            diagnostic = _response_metadata(response)
            error: Exception | None = None
            messages: list[dict[str, str]] | None = None
            try:
                messages = v2.predecessor._parse_final_response(response, scenario=scenario, language="banglish")
            except (ValueError, json.JSONDecodeError) as observed:
                error = observed
            attempt_metadata = {
                "accepted": error is None,
                "attempt_number": attempt_number,
                "completion_tokens": diagnostic["usage"]["completion_tokens"],
                "error_message": str(error) if error is not None else None,
                "error_type": type(error).__name__ if error is not None else None,
                "finish_reason": diagnostic["finish_reason"],
                "max_tokens": expected["max_tokens"],
                "prompt_tokens": diagnostic["usage"]["prompt_tokens"],
                "reasoning_effort": REASONING_EFFORT,
                "record_id": RECORD_ID,
                "request_sha256": request_sha256,
                "response_envelope_sha256": sha256_value(response),
                "total_tokens": diagnostic["usage"]["total_tokens"],
            }
            attempts.append(attempt_metadata)
            v2.predecessor._write_jsonl(metadata_path, attempts)
            if error is not None:
                if v2.predecessor._critical_validation_error(error):
                    break
                continue
            assert messages is not None
            candidate = {
                "language_mode": "banglish",
                "messages": messages,
                "provenance": {
                    "ai_assisted": True,
                    "diagnostic_only": True,
                    "external_inference_api": False,
                    "generation_attempt": attempt_number,
                    "human_authored": False,
                    "model_id": v2.predecessor.MODEL_ID,
                    "model_revision": v2.predecessor.MODEL_REVISION,
                    "prompt_contract": PROMPT_CONTRACT,
                    "reasoning_effort": REASONING_EFFORT,
                    "reasoning_or_analysis_retained": False,
                    "request_sha256": request_sha256,
                    "runtime_image_digest": v2.predecessor.RUNTIME_DIGEST,
                    "seed": expected["seed"],
                },
                "record_id": RECORD_ID,
                "rights_status": "private_research_diagnostic_pending_owner_review",
                "scenario_id": scenario.scenario_id,
                "schema_version": v2.predecessor.ROW_SCHEMA,
                "task": scenario.task,
            }
            v2.predecessor._write_jsonl(candidate_path, [candidate])
            break
    finally:
        _stop_runtime(ssh_key.resolve(), known_hosts.resolve())
    after_gpu = v2.predecessor.gpu_snapshot(ssh_key.resolve(), known_hosts.resolve())
    status = "diagnostic_candidate_machine_valid_pending_owner_review" if candidate is not None else "FAIL_CLOSED_one_row_diagnostic_exhausted"
    report: dict[str, Any] = {
        "accepted_final_outputs": 1 if candidate is not None else 0,
        "attempt_metadata_file_sha256": file_sha256(metadata_path),
        "authorization_file_sha256": file_sha256(authorization_path),
        "candidate_file_sha256": file_sha256(candidate_path) if candidate_path.exists() else None,
        "candidate_registry_changed": False,
        "dataset_sealed": False,
        "diagnostic_contract": DIAGNOSTIC_CONTRACT,
        "external_api_used": False,
        "gpu": {"after_stop": after_gpu, "before": before_gpu},
        "inference_calls": len(attempts),
        "machine_validation_passed": candidate is not None,
        "owner_review": {"rows": 1 if candidate is not None else 0, "status": "pending" if candidate is not None else "not_run"},
        "plan_file_sha256": file_sha256(plan_path),
        "production_eligible": False,
        "runtime_reuse_receipt_file_sha256": file_sha256(runtime_reuse_receipt_path),
        "runtime_stopped": after_gpu["memory_used_mib"] == 0,
        "schema_version": RUN_REPORT_SCHEMA,
        "sft_training_started": False,
        "status": status,
    }
    report["report_sha256"] = sha256_value(report)
    v2.predecessor._write_json_atomic(report_path, report)
    return report


def verify_diagnostic_run(
    *, plan_path: Path, authorization_path: Path, runtime_reuse_receipt_path: Path, repository_root: Path, artifact_root: Path
) -> dict[str, Any]:
    root = repository_root.resolve()
    load_diagnostic_plan(plan_path.resolve(), repository_root=root)
    load_diagnostic_authorization(authorization_path.resolve(), plan_path=plan_path.resolve())
    load_runtime_reuse_receipt(runtime_reuse_receipt_path.resolve(), authorization_path=authorization_path.resolve())
    artifact = artifact_root.resolve()
    report_path = artifact / "evidence/run-report.json"
    report = v2.predecessor._load_json(report_path)
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256", None)
    metadata = v2.predecessor._read_jsonl(artifact / "private/diagnostic-attempt-metadata.jsonl")
    candidate = v2.predecessor._read_jsonl(artifact / "private/diagnostic-candidate.jsonl")
    manifest = diagnostic_manifest()
    allowed_metadata_keys = {
        "accepted", "attempt_number", "completion_tokens", "error_message", "error_type", "finish_reason",
        "max_tokens", "prompt_tokens", "reasoning_effort", "record_id", "request_sha256",
        "response_envelope_sha256", "total_tokens",
    }
    checks = {
        "authorization_binding": report.get("authorization_file_sha256") == file_sha256(authorization_path),
        "candidate_count_matches": len(candidate) == report.get("accepted_final_outputs"),
        "internal_report_hash": internal == sha256_value(unsigned),
        "metadata_hash_binding": report.get("attempt_metadata_file_sha256") == file_sha256(artifact / "private/diagnostic-attempt-metadata.jsonl"),
        "metadata_is_bounded": 1 <= len(metadata) <= 3 and [item["attempt_number"] for item in metadata] == list(range(1, len(metadata) + 1)),
        "metadata_keys_are_non_sensitive": all(set(item) == allowed_metadata_keys for item in metadata),
        "request_manifest_prefix": [item["request_sha256"] for item in metadata] == [item["request_sha256"] for item in manifest["attempts"][: len(metadata)]],
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
        raise ValueError("gpt-oss diagnostic run verification failed")
    return verification
