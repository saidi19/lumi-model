# ruff: noqa: E501, SLF001
"""One-row Bengali-script repair for the dual-field grounded-literal mismatch."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import posttraining_gpt_oss_teacher_canary_v10 as v10
from .posttraining_candidate_mix_preflight import file_sha256, sha256_value

base = v10.v8.v5.v2.predecessor

PLAN_SCHEMA = "lumi.sft-gpt-oss-teacher-dual-field-literal-repair-plan.v0.11"
REPORT_SCHEMA = "lumi.sft-gpt-oss-teacher-dual-field-literal-repair-plan-report.v0.11"
VERIFICATION_SCHEMA = "lumi.sft-gpt-oss-teacher-dual-field-literal-repair-plan-verification.v0.11"
AUTHORIZATION_SCHEMA = "lumi.sft-gpt-oss-teacher-dual-field-literal-repair-authorization.v0.11"
RUNTIME_REUSE_RECEIPT_SCHEMA = "lumi.sft-gpt-oss-teacher-dual-field-literal-repair-runtime-reuse-receipt.v0.11"
RUN_REPORT_SCHEMA = "lumi.sft-gpt-oss-teacher-dual-field-literal-repair-run-report.v0.11"
RUN_VERIFICATION_SCHEMA = "lumi.sft-gpt-oss-teacher-dual-field-literal-repair-run-verification.v0.11"

PROMPT_CONTRACT = "lumi.gpt-oss-teacher.dual-field-literal-repair.v0.11"
SEED_CONTRACT = "sha256(record_id:dual-field-literal-repair-attempt:attempt_number)-u32.v0.11"
RECORD_ID = "lumi-gptoss-writing_summarization-bengali_script-03"
LANGUAGE = "bengali_script"
REASONING_EFFORT = "low"
MAX_TOKENS = 1_536
MAXIMUM_ATTEMPTS = 2
REQUIRED_LITERALS = ("14:20", "API gateway", "18 minutes")

SERVER = v10.SERVER
PROHIBITIONS = v10.PROHIBITIONS
RUNTIME_REUSE = v10.RUNTIME_REUSE
RETENTION = v10.RETENTION
PREDECESSOR_FAILURE = {
    "accepted_prefix_outputs": 54,
    "additional_final_outputs": 1,
    "additional_inference_calls": 3,
    "additional_retry_calls": 1,
    "candidate_prefix_file_sha256": "1c02b41eac74461779118545eb128629e8f8363dc87b5b0db0bd8e085bb37888",
    "failed_record_id": RECORD_ID,
    "failure_report_file_sha256": "a05ffbb0f6e49c04e683891ea5a9ed849fa5fbe788c92bd75510631e875b5724",
    "failure_verification_file_sha256": "790164f892231313370c870097fe95fadaba629ec2d0e4085226a8c239c6b553",
    "generation_telemetry_file_sha256": "7d2573bcff0a6efebdf5c276fa420e998347b34cf45bad481c8fedf3b74af708",
    "rejection_metadata_file_sha256": "31144d365519ab7e62aecdcf37053e52e4cc5a3575680be917185721ff1c9bca",
    "status": "FAIL_CLOSED_55th_record_dual_field_literal_mismatch",
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
    "assistant_maximum_characters": 512,
    "exact_language_mode": LANGUAGE,
    "exact_record_id": RECORD_ID,
    "literals_required_separately_in_assistant_response": list(REQUIRED_LITERALS),
    "literals_required_separately_in_user_prompt": list(REQUIRED_LITERALS),
    "owner_review_required_if_accepted": True,
    "user_and_assistant_must_be_natural_user_facing_text": True,
}

DUAL_FIELD_INSTRUCTION = (
    "Create a natural Bengali-script user request about an outage and a concise Bengali-script "
    "assistant summary. The JSON string value user_prompt must separately contain each of these "
    "exact verbatim substrings: 14:20, API gateway, 18 minutes. The JSON string value "
    "assistant_response must also separately contain each of the same exact verbatim substrings: "
    "14:20, API gateway, 18 minutes. Copy their case, digits, colon, and spacing exactly in both "
    "fields; do not translate, transliterate, inflect, reformat, or omit them. Outside those three "
    "immutable substrings, use natural Bengali script. Never put generation instructions, attempt "
    "numbers, JSON directions, schema labels, or meta-commentary inside either generated field."
)

EXACT_AUTHORIZATION_TEXT = (
    "I approve one bounded self-hosted Lumi gpt-oss-120b one-row Bengali-script dual-field literal "
    "repair on the existing exact Lambda host 68.209.73.55, bound to ED25519 host-key fingerprint "
    "SHA256:2ZSQqNUYJhWNjihHnqLQ1YNHxoqtMJeckVJRKlc5hs4 and exactly one NVIDIA H100 80GB GPU "
    "GPU-c7e68b23-95b4-37bd-b40b-c41a7ec716f3 reporting at least 81,559 MiB FB total without "
    "adding the 480 MiB reserved value twice. Reuse, and do not redownload, the 25 verified "
    "openai/gpt-oss-120b model objects at revision b5c939de8f754692c1647ca79fbf85e8c1e70f8a "
    "bound by model receipt SHA-256 798914d8532d5b36b61711f34591cabe1e03fa6ac1128fe5e3737c029ead1284, "
    "and reuse the existing vLLM image digest "
    "sha256:4d08193d2fd05aadb1b5678f93ae609efb2635df67da45f3efe781c368b34dc8. Bind the v0.10 "
    "failure report SHA-256 a05ffbb0f6e49c04e683891ea5a9ed849fa5fbe788c92bd75510631e875b5724 and "
    "failure verification SHA-256 790164f892231313370c870097fe95fadaba629ec2d0e4085226a8c239c6b553. "
    "Run only record lumi-gptoss-writing_summarization-bengali_script-03 with the frozen safety, "
    "schema, language, punctuation, and literal parser gates under prompt contract "
    "lumi.gpt-oss-teacher.dual-field-literal-repair.v0.11, reasoning effort low, a 1,536 "
    "generated-token ceiling per call, distinct attempt-bound deterministic seeds, and at most 2 "
    "attempts. Require natural Bengali-script user_prompt and assistant_response values, and require "
    "each field separately to contain exact 14:20, API gateway, and 18 minutes verbatim without "
    "translation, transliteration, inflection, reformatting, or omission. Reject generated "
    "instructions, attempt numbers, JSON directions, schema labels, and meta-commentary in either "
    "generated field. Retain rejected attempts only as validation error, non-sensitive finish_reason, "
    "usage token counts, and request/response-envelope hashes without raw candidate or reasoning text. "
    "Maximum 20 additional instance-minutes / $1.43 before tax / 0 additional model or runtime "
    "download bytes / 1 GiB additional private writes / 2 sequential inference calls / 1 retry call / "
    "1 accepted repair output. It may restart and then cleanly stop only the bound canary container, "
    "serve only on remote localhost through pinned SSH, run the one-row frozen machine and dual-field "
    "literal gates, and prepare the accepted row for one-row owner review. This repair cannot satisfy "
    "the 16-cell gate or authorize resuming the larger canary. No external API, public port, model or "
    "runtime download, registry change, dataset seal, SFT training, production promotion, unrelated "
    "access, new instance, or termination."
)

TOOLING_PATHS = {
    "implementation": "src/lumi/posttraining_gpt_oss_teacher_dual_field_literal_repair_v11.py",
    "plan_runner": "scripts/plan_sft_gpt_oss_teacher_dual_field_literal_repair_v11.py",
    "plan_verifier": "scripts/verify_sft_gpt_oss_teacher_dual_field_literal_repair_v11_plan.py",
    "run_runner": "scripts/run_sft_gpt_oss_teacher_dual_field_literal_repair_v11.py",
    "run_verifier": "scripts/verify_sft_gpt_oss_teacher_dual_field_literal_repair_v11.py",
    "test": "tests/test_gpt_oss_teacher_dual_field_literal_repair_v11.py",
}


def _scenario() -> Any:
    return next(item for item in base.SCENARIOS if item.task == "writing_summarization" and item.variant == 3)


def repair_seed(attempt_number: int) -> int:
    if not 1 <= attempt_number <= MAXIMUM_ATTEMPTS:
        raise ValueError("dual-field repair attempt number is outside the reviewed contract")
    material = f"{RECORD_ID}:dual-field-literal-repair-attempt:{attempt_number}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def repair_messages(attempt_number: int) -> list[dict[str, str]]:
    retry = " This is the final bounded attempt; silently re-check all six literal occurrences before returning." if attempt_number > 1 else ""
    system = (
        "Return only one compact JSON object with scenario_id, user_prompt, assistant_response, "
        "safety_label, tool_call, and tool_result. Set scenario_id to outage_summary, safety_label "
        "to safe, and both tool fields to null. Never expose reasoning. "
        + DUAL_FIELD_INSTRUCTION
        + retry
    )
    user = "Produce the final JSON object now. Ensure all three immutable literals occur separately in both text fields."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def repair_payload(attempt_number: int) -> dict[str, Any]:
    payload = copy.deepcopy(base.chat_completion_payload(RECORD_ID, _scenario(), LANGUAGE))
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
        "language": LANGUAGE,
        "record_id": RECORD_ID,
        "scenario_id": "outage_summary",
        "task": "writing_summarization",
        "variant": 3,
    }


def validate_dual_field_row(messages: Sequence[Mapping[str, str]]) -> None:
    if len(messages) != 2 or messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
        raise ValueError("dual-field repair message shape changed")
    user = str(messages[0].get("content", ""))
    assistant = str(messages[1].get("content", ""))
    for literal in REQUIRED_LITERALS:
        if literal not in user:
            raise ValueError(f"dual-field repair user_prompt missing literal: {literal}")
        if literal not in assistant:
            raise ValueError(f"dual-field repair assistant_response missing literal: {literal}")
    v10.validate_prompt_quality(messages, scenario=_scenario(), language=LANGUAGE)


def load_repair_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    plan = base._load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("status") != "dual_field_literal_repair_plan_pending_explicit_execution_authorization":
        raise ValueError("unsupported or changed v0.11 repair plan")
    if plan.get("server") != SERVER or plan.get("runtime_reuse") != RUNTIME_REUSE:
        raise ValueError("v0.11 host or runtime binding changed")
    if plan.get("predecessor_failure") != PREDECESSOR_FAILURE:
        raise ValueError("v0.11 predecessor failure changed")
    if plan.get("limits") != LIMITS or plan.get("quality_gates") != QUALITY_GATES:
        raise ValueError("v0.11 limits or gates changed")
    if plan.get("prohibitions") != PROHIBITIONS or plan.get("retention") != RETENTION:
        raise ValueError("v0.11 prohibition or retention changed")
    if any(bool(value) for value in plan.get("authorization", {}).values() if isinstance(value, bool)):
        raise ValueError("v0.11 plan unexpectedly authorizes execution")
    if plan.get("next_gate", {}).get("exact_authorization_text") != EXACT_AUTHORIZATION_TEXT:
        raise ValueError("v0.11 authorization text changed")
    contract = plan.get("prompt_contract", {})
    if contract.get("id") != PROMPT_CONTRACT or contract.get("seed_contract") != SEED_CONTRACT or contract.get("manifest_sha256") != sha256_value(repair_manifest()):
        raise ValueError("v0.11 prompt contract changed")
    base._resolve_bindings(plan, repository_root)
    return plan


def build_plan_report(*, plan: Mapping[str, Any], plan_file_sha256: str, repository_root: str | Path) -> dict[str, Any]:
    resolved = base._resolve_bindings(plan, repository_root)
    manifest = repair_manifest()
    prompt_text = json.dumps(repair_messages(1), ensure_ascii=False)
    report: dict[str, Any] = {
        "authorization": plan["authorization"],
        "bindings": {name: {"path": plan["bindings"][name]["path"], "sha256": file_sha256(path)} for name, path in sorted(resolved.items())},
        "checks": {
            "distinct_attempt_requests": len({item["request_sha256"] for item in manifest["attempts"]}) == 2,
            "distinct_attempt_seeds": len({item["seed"] for item in manifest["attempts"]}) == 2,
            "dual_field_instruction_exact": all(prompt_text.count(literal) >= 2 for literal in REQUIRED_LITERALS),
            "exact_failed_record": manifest["record_id"] == RECORD_ID,
            "frozen_parser_accepts_reference": _reference_row_passes(),
            "no_execution_authorized": not any(bool(value) for value in plan["authorization"].values() if isinstance(value, bool)),
            "no_model_or_runtime_download": LIMITS["maximum_network_bytes"] == 0,
            "predecessor_failure_bound": plan["predecessor_failure"] == PREDECESSOR_FAILURE,
        },
        "limits": LIMITS,
        "manifest": manifest,
        "next_gate": plan["next_gate"],
        "observed_at": plan["observed_at"],
        "plan_file_sha256": plan_file_sha256,
        "predecessor_failure": PREDECESSOR_FAILURE,
        "prohibitions": PROHIBITIONS,
        "prompt_contract": plan["prompt_contract"],
        "quality_gates": QUALITY_GATES,
        "retention": RETENTION,
        "runtime_reuse": RUNTIME_REUSE,
        "schema_version": REPORT_SCHEMA,
        "server": SERVER,
        "status": "verified_dual_field_literal_repair_plan_no_execution_authorized",
        "tooling": {name: {"path": tooling_path, "sha256": file_sha256(Path(repository_root) / tooling_path)} for name, tooling_path in sorted(TOOLING_PATHS.items())},
    }
    if not all(report["checks"].values()):
        raise ValueError("v0.11 plan checks failed")
    report["report_sha256"] = sha256_value(report)
    return report


def _reference_row_passes() -> bool:
    messages = [
        {"role": "user", "content": "আজ 14:20-এ API gateway বন্ধ ছিল এবং সমস্যা 18 minutes স্থায়ী হয়েছে। সংক্ষেপ চাই।"},
        {"role": "assistant", "content": "14:20-এ API gateway বিভ্রাট শুরু হয় এবং 18 minutes পরে সেবা স্বাভাবিক হয়।"},
    ]
    base.base._validate_output(
        {"assistant_response": messages[1]["content"], "safety_label": "safe", "scenario_id": "outage_summary", "tool_call": None, "tool_result": None, "user_prompt": messages[0]["content"]},
        scenario=_scenario(),
        language_mode=LANGUAGE,
    )
    validate_dual_field_row(messages)
    return True


def build_plan_verification(*, report: Mapping[str, Any], report_file_sha256: str) -> dict[str, Any]:
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256", None)
    checks = {
        "all_plan_checks_pass": all(report.get("checks", {}).values()),
        "authorization_pending": report.get("status") == "verified_dual_field_literal_repair_plan_no_execution_authorized",
        "call_ceiling_two": report.get("limits", {}).get("maximum_inference_calls") == 2,
        "internal_report_hash": internal == sha256_value(unsigned),
        "predecessor_failure_exact": report.get("predecessor_failure") == PREDECESSOR_FAILURE,
        "zero_download_ceiling": report.get("limits", {}).get("maximum_network_bytes") == 0,
    }
    verification: dict[str, Any] = {"checks": checks, "report_file_sha256": report_file_sha256, "report_internal_sha256": internal, "schema_version": VERIFICATION_SCHEMA, "status": "PASS" if all(checks.values()) else "FAIL"}
    verification["verification_sha256"] = sha256_value(verification)
    if verification["status"] != "PASS":
        raise ValueError("v0.11 independent plan verification failed")
    return verification


def load_execution_authorization(path: str | Path, *, plan_path: str | Path) -> dict[str, Any]:
    authorization = base._load_json(path)
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA or authorization.get("status") != "authorized_exact_host_bound_one_row_dual_field_literal_repair":
        raise ValueError("unsupported or changed v0.11 authorization")
    if authorization.get("approval_text") != EXACT_AUTHORIZATION_TEXT or authorization.get("plan_file_sha256") != file_sha256(plan_path):
        raise ValueError("v0.11 approval or plan binding changed")
    if authorization.get("server") != SERVER or authorization.get("limits") != LIMITS or authorization.get("prohibitions") != PROHIBITIONS or authorization.get("predecessor_failure") != PREDECESSOR_FAILURE:
        raise ValueError("v0.11 authorization bindings changed")
    if not isinstance(authorization.get("authorized_start_unix"), int) or authorization["authorized_start_unix"] <= 0:
        raise ValueError("v0.11 authorization start time is invalid")
    return authorization


def load_runtime_reuse_receipt(path: str | Path, *, authorization_path: str | Path) -> dict[str, Any]:
    receipt = base._load_json(path)
    if receipt.get("schema_version") != RUNTIME_REUSE_RECEIPT_SCHEMA or receipt.get("status") != "PASS_exact_existing_localhost_runtime_reused_for_dual_field_literal_repair_v0.11":
        raise ValueError("unsupported or failed v0.11 runtime receipt")
    if receipt.get("authorization_file_sha256") != file_sha256(authorization_path) or receipt.get("server") != SERVER or receipt.get("runtime_reuse") != RUNTIME_REUSE:
        raise ValueError("v0.11 runtime receipt binding changed")
    if not all(receipt.get("checks", {}).values()):
        raise ValueError("v0.11 runtime checks failed")
    return receipt


def run_dual_field_repair(*, plan_path: Path, authorization_path: Path, runtime_reuse_receipt_path: Path, repository_root: Path, artifact_root: Path, ssh_key: Path, known_hosts: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    plan = load_repair_plan(plan_path.resolve(), repository_root=root)
    authorization = load_execution_authorization(authorization_path.resolve(), plan_path=plan_path.resolve())
    load_runtime_reuse_receipt(runtime_reuse_receipt_path.resolve(), authorization_path=authorization_path.resolve())
    if file_sha256(known_hosts.resolve()) != plan["bindings"]["known_hosts"]["sha256"] or ssh_key.resolve().stat().st_mode & 0o077:
        raise PermissionError("v0.11 SSH binding or permissions changed")
    if time.time() - int(authorization["authorized_start_unix"]) > LIMITS["maximum_instance_minutes"] * 60:
        raise RuntimeError("v0.11 authorization expired")
    artifact = artifact_root.resolve()
    if not artifact.is_relative_to((root / "artifacts").resolve()):
        raise ValueError("v0.11 artifact root escaped artifacts/")
    private, evidence = artifact / "private", artifact / "evidence"
    private.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    metadata_path = private / "repair-attempt-metadata.jsonl"
    candidate_path = private / "repair-candidate.jsonl"
    report_path = evidence / "run-report.json"
    if any(path.exists() for path in (metadata_path, candidate_path, report_path)):
        raise FileExistsError("refusing to replay a consumed v0.11 artifact root")
    scenario = _scenario()
    manifest = repair_manifest()
    attempts: list[dict[str, Any]] = []
    candidate: dict[str, Any] | None = None
    before_gpu = base.gpu_snapshot(ssh_key.resolve(), known_hosts.resolve())
    try:
        for expected in manifest["attempts"]:
            number = int(expected["attempt_number"])
            payload = repair_payload(number)
            request_sha256 = sha256_value(payload)
            if request_sha256 != expected["request_sha256"]:
                raise ValueError("v0.11 request manifest changed")
            response = base._post_chat_completion(payload)
            diagnostic = v10.v8.v5._response_metadata(response)
            error: Exception | None = None
            messages: list[dict[str, str]] | None = None
            try:
                messages = base._parse_final_response(response, scenario=scenario, language=LANGUAGE)
                validate_dual_field_row(messages)
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
            base._write_jsonl(metadata_path, attempts)
            if error is not None:
                if base._critical_validation_error(error):
                    break
                continue
            assert messages is not None
            candidate = {
                "language_mode": LANGUAGE,
                "messages": messages,
                "provenance": {
                    "ai_assisted": True,
                    "diagnostic_only": True,
                    "external_inference_api": False,
                    "generation_attempt": number,
                    "human_authored": False,
                    "model_id": base.MODEL_ID,
                    "model_revision": base.MODEL_REVISION,
                    "prompt_contract": PROMPT_CONTRACT,
                    "reasoning_effort": REASONING_EFFORT,
                    "reasoning_or_analysis_retained": False,
                    "request_sha256": request_sha256,
                    "runtime_image_digest": base.RUNTIME_DIGEST,
                    "seed": expected["seed"],
                },
                "record_id": RECORD_ID,
                "rights_status": "private_research_dual_field_literal_repair_pending_owner_review",
                "scenario_id": scenario.scenario_id,
                "schema_version": base.ROW_SCHEMA,
                "task": scenario.task,
            }
            base._write_jsonl(candidate_path, [candidate])
            break
    finally:
        v10.v8.v5.v3._stop_runtime(ssh_key.resolve(), known_hosts.resolve())
    after_gpu = base.gpu_snapshot(ssh_key.resolve(), known_hosts.resolve())
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
        "machine_dual_field_literal_validation_passed": candidate is not None,
        "owner_review": {"rows": 1 if candidate is not None else 0, "status": "pending" if candidate is not None else "not_run"},
        "plan_file_sha256": file_sha256(plan_path),
        "production_eligible": False,
        "prompt_contract": PROMPT_CONTRACT,
        "runtime_reuse_receipt_file_sha256": file_sha256(runtime_reuse_receipt_path),
        "runtime_stopped": after_gpu["memory_used_mib"] == 0,
        "schema_version": RUN_REPORT_SCHEMA,
        "sft_training_started": False,
        "status": "dual_field_literal_repair_candidate_machine_valid_pending_owner_review" if candidate is not None else "FAIL_CLOSED_dual_field_literal_repair_exhausted",
    }
    report["report_sha256"] = sha256_value(report)
    base._write_json_atomic(report_path, report)
    return report


def verify_dual_field_repair_run(*, plan_path: Path, authorization_path: Path, runtime_reuse_receipt_path: Path, repository_root: Path, artifact_root: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    load_repair_plan(plan_path.resolve(), repository_root=root)
    load_execution_authorization(authorization_path.resolve(), plan_path=plan_path.resolve())
    load_runtime_reuse_receipt(runtime_reuse_receipt_path.resolve(), authorization_path=authorization_path.resolve())
    artifact = artifact_root.resolve()
    report_path = artifact / "evidence/run-report.json"
    report = base._load_json(report_path)
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256", None)
    metadata = base._read_jsonl(artifact / "private/repair-attempt-metadata.jsonl")
    candidate = base._read_jsonl(artifact / "private/repair-candidate.jsonl")
    if candidate:
        validate_dual_field_row(candidate[0]["messages"])
    checks = {
        "authorization_binding": report.get("authorization_file_sha256") == file_sha256(authorization_path),
        "candidate_count_matches": len(candidate) == report.get("accepted_final_outputs"),
        "candidate_dual_field_gate_reproduced": len(candidate) <= 1,
        "internal_report_hash": internal == sha256_value(unsigned),
        "metadata_is_bounded": 1 <= len(metadata) <= 2,
        "request_manifest_prefix": [item["request_sha256"] for item in metadata] == [item["request_sha256"] for item in repair_manifest()["attempts"][: len(metadata)]],
        "runtime_binding": report.get("runtime_reuse_receipt_file_sha256") == file_sha256(runtime_reuse_receipt_path),
        "runtime_stopped": report.get("runtime_stopped") is True and report.get("gpu", {}).get("after_stop", {}).get("memory_used_mib") == 0,
        "zero_production_actions": report.get("candidate_registry_changed") is False and report.get("dataset_sealed") is False and report.get("sft_training_started") is False and report.get("production_eligible") is False,
    }
    verification: dict[str, Any] = {"checks": checks, "run_report_file_sha256": file_sha256(report_path), "run_report_internal_sha256": internal, "schema_version": RUN_VERIFICATION_SCHEMA, "status": "PASS" if all(checks.values()) else "FAIL"}
    verification["verification_sha256"] = sha256_value(verification)
    base._write_json_atomic(artifact / "evidence/run-verification.json", verification)
    if verification["status"] != "PASS":
        raise ValueError("v0.11 run verification failed")
    return verification
