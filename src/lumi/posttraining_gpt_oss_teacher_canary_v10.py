# ruff: noqa: E501, SLF001
"""Repair the v0.9 patch-context recursion without changing generation semantics."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import posttraining_gpt_oss_teacher_canary_v9 as v9
from .posttraining_candidate_mix_preflight import file_sha256, sha256_value

v8 = v9.v8

PLAN_SCHEMA = "lumi.sft-gpt-oss-teacher-runner-repair-resume-plan.v0.10"
REPORT_SCHEMA = "lumi.sft-gpt-oss-teacher-runner-repair-resume-report.v0.10"
VERIFICATION_SCHEMA = "lumi.sft-gpt-oss-teacher-runner-repair-resume-verification.v0.10"
AUTHORIZATION_SCHEMA = "lumi.sft-gpt-oss-teacher-runner-repair-resume-authorization.v0.10"
RUNTIME_REUSE_RECEIPT_SCHEMA = "lumi.sft-gpt-oss-teacher-runner-repair-resume-runtime-reuse-receipt.v0.10"
RUN_REPORT_SCHEMA = "lumi.sft-gpt-oss-teacher-runner-repair-resume-run-report.v0.10"
RUN_VERIFICATION_SCHEMA = "lumi.sft-gpt-oss-teacher-runner-repair-resume-run-verification.v0.10"

SERVER = v9.SERVER
LIMITS = v9.LIMITS
PROHIBITIONS = v9.PROHIBITIONS
RUNTIME_REUSE = v9.RUNTIME_REUSE
RETENTION = v9.RETENTION
FROZEN_MACHINE_GATES = v9.FROZEN_MACHINE_GATES
ACCEPTED_REPAIRS = v9.ACCEPTED_REPAIRS
PROMPT_CONTRACT = v9.PROMPT_CONTRACT

PREDECESSOR_FAILURE = {
    "accepted_prefix_outputs": 53,
    "additional_inference_calls": 0,
    "additional_runtime_retries": 0,
    "candidate_prefix_file_sha256": "866a6ec5afea7fbfd23925c1fdad1dd877bb208dc024bdbbdae2722c7be349c9",
    "failure_report_file_sha256": "1e168271fb9bc4d366dfab8abb822d6488d1663022a58e86dff3f49f1fcd5bd2",
    "failure_verification_file_sha256": "5538daaf52bc6765133a7f1edbd23f6f5bde5f609e850424619f4a5d8e1aee34",
    "generation_telemetry_file_sha256": "76c87fda02c9ccb693cce736632d24c51d6b5b0bfc86cb1fbeea55781925550b",
    "recovered_rejection_metadata_file_sha256": "8d12b1adccc445e090195002ced514fe3b1ee33f15999cfc27baf15156d9bf8c",
    "recovered_rejection_metadata_rows": 4,
    "status": "FAIL_CLOSED_pre_inference_runner_recursion",
}

EXACT_AUTHORIZATION_TEXT = (
    "I approve one bounded self-hosted Lumi gpt-oss-120b full-canary runner-recursion repair "
    "resume successor on the existing exact Lambda host 68.209.73.55, bound to ED25519 "
    "host-key fingerprint SHA256:2ZSQqNUYJhWNjihHnqLQ1YNHxoqtMJeckVJRKlc5hs4 and exactly "
    "one NVIDIA H100 80GB GPU GPU-c7e68b23-95b4-37bd-b40b-c41a7ec716f3 reporting at "
    "least 81,559 MiB FB total without adding the 480 MiB reserved value twice. Reuse, and "
    "do not redownload, the 25 verified openai/gpt-oss-120b model objects at revision "
    "b5c939de8f754692c1647ca79fbf85e8c1e70f8a bound by model receipt SHA-256 "
    "798914d8532d5b36b61711f34591cabe1e03fa6ac1128fe5e3737c029ead1284, and reuse the "
    "existing vLLM image digest sha256:4d08193d2fd05aadb1b5678f93ae609efb2635df67da45f3efe781c368b34dc8. "
    "Bind the v0.9 pre-inference failure report SHA-256 "
    "1e168271fb9bc4d366dfab8abb822d6488d1663022a58e86dff3f49f1fcd5bd2 and failure "
    "verification SHA-256 5538daaf52bc6765133a7f1edbd23f6f5bde5f609e850424619f4a5d8e1aee34; "
    "the failed v0.9 execution consumed zero inference calls, zero retry calls, and generated "
    "no model candidate text. Reuse without regeneration the exact 53 accepted prefix rows "
    "bound by SHA-256 866a6ec5afea7fbfd23925c1fdad1dd877bb208dc024bdbbdae2722c7be349c9, "
    "their exact 53-row telemetry bound by SHA-256 "
    "76c87fda02c9ccb693cce736632d24c51d6b5b0bfc86cb1fbeea55781925550b, and only the four "
    "recovered-record rejection metadata entries bound by SHA-256 "
    "8d12b1adccc445e090195002ced514fe3b1ee33f15999cfc27baf15156d9bf8c. Generate only the "
    "remaining 107 records, beginning with lumi-gptoss-writing_summarization-natural_code_switching-03, "
    "under unchanged prompt contract lumi.gpt-oss-teacher.grounded-literal-resume.v0.9, "
    "reasoning effort low, a 1,536 generated-token ceiling per call, distinct attempt-bound "
    "deterministic seeds, and at most 2 new attempts per remaining record. Use immutable "
    "pre-patch v8 base-function captures and require the patch-context execution regression "
    "gate to pass before remote restart. Preserve all v0.8 complete-row gates and require "
    "every frozen required literal verbatim; for the first resumed outage-summary record this "
    "includes exact 14:20, API gateway, and 18 minutes. Retain rejected new attempts only as "
    "validation error, non-sensitive finish_reason, usage token counts, and request/response-envelope "
    "hashes without raw candidate or reasoning text. Maximum 90 additional instance-minutes / "
    "$6.44 before tax / 0 additional model or runtime download bytes / 16 GiB additional private "
    "writes / 143 additional sequential inference calls / 36 additional retry calls / 107 "
    "additional final outputs, while the combined retained-prefix plus resume audit remains capped "
    "at 200 calls and 40 retries. It may restart and then cleanly stop only the bound canary "
    "container, serve only on remote localhost through pinned SSH, retain only accepted final-channel "
    "candidate text, run the full frozen 16-cell/160-row machine gate, and prepare the deterministic "
    "32-row owner-review packet. No external API, public port, model or runtime download, registry "
    "change, dataset seal, SFT training, production promotion, unrelated access, new instance, or termination."
)

_BASE_V8_CHAT_COMPLETION_PAYLOAD = v8.chat_completion_payload
_BASE_V8_VALIDATE_PROMPT_QUALITY = v8.validate_prompt_quality


def chat_completion_payload(record_id: str, scenario: Any, language: str, *, attempt_number: int) -> dict[str, Any]:
    payload = copy.deepcopy(_BASE_V8_CHAT_COMPLETION_PAYLOAD(record_id, scenario, language, attempt_number=attempt_number))
    payload["messages"] = v9.generation_messages(scenario, language, attempt_number=attempt_number)
    payload["seed"] = v9.full_canary_seed(record_id, attempt_number)
    payload["temperature"] = 0.2
    return payload


def validate_prompt_quality(messages: Sequence[Mapping[str, str]], *, scenario: Any, language: str) -> None:
    _BASE_V8_VALIDATE_PROMPT_QUALITY(messages, scenario=scenario, language=language)
    combined = "\n".join(str(message.get("content", "")) for message in messages)
    for literal in scenario.required_literals:
        if literal not in combined:
            raise ValueError(f"v0.10 required grounded literal missing: {literal}")


def prompt_manifest() -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for scenario in v8.v5.v2.predecessor.SCENARIOS:
        for language in v8.v5.v2.predecessor.LANGUAGE_MODES:
            record_id = f"lumi-gptoss-{scenario.task}-{language}-{scenario.variant:02d}"
            attempts = [
                {
                    "attempt_number": attempt,
                    "max_tokens": v9.MAX_TOKENS,
                    "reasoning_effort": v9.REASONING_EFFORT,
                    "request_sha256": sha256_value(chat_completion_payload(record_id, scenario, language, attempt_number=attempt)),
                    "seed": v9.full_canary_seed(record_id, attempt),
                }
                for attempt in range(1, v9.MAXIMUM_ATTEMPTS_PER_RECORD + 1)
            ]
            manifest.append({"attempts": attempts, "language": language, "record_id": record_id, "scenario_id": scenario.scenario_id, "task": scenario.task, "variant": scenario.variant})
    return manifest


def _resolve_bindings(plan: Mapping[str, Any], repository_root: Path) -> dict[str, Path]:
    return v8.v5.v2.predecessor._resolve_bindings(plan, repository_root)


def load_resume_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    plan = v8.v5.v2.predecessor._load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("status") != "runner_repair_resume_plan_pending_explicit_execution_authorization":
        raise ValueError("unsupported or changed v0.10 runner-repair plan")
    if plan.get("server") != SERVER or plan.get("runtime_reuse") != RUNTIME_REUSE:
        raise ValueError("v0.10 host or runtime binding changed")
    if plan.get("limits") != LIMITS or plan.get("prohibitions") != PROHIBITIONS or plan.get("retention") != RETENTION:
        raise ValueError("v0.10 execution boundary changed")
    if plan.get("predecessor_failure") != PREDECESSOR_FAILURE or plan.get("accepted_repairs") != ACCEPTED_REPAIRS:
        raise ValueError("v0.10 predecessor binding changed")
    if plan.get("next_gate", {}).get("exact_authorization_text") != EXACT_AUTHORIZATION_TEXT:
        raise ValueError("v0.10 authorization text changed")
    if plan.get("prompt_contract") != {
        "id": PROMPT_CONTRACT,
        "manifest_sha256": sha256_value(prompt_manifest()),
        "maximum_attempts_per_record": v9.MAXIMUM_ATTEMPTS_PER_RECORD,
        "maximum_generated_tokens_per_call": v9.MAX_TOKENS,
        "reasoning_effort": v9.REASONING_EFFORT,
        "records": 160,
        "remaining_records": 107,
        "seed_contract": v9.SEED_CONTRACT,
        "temperature": 0.2,
    }:
        raise ValueError("v0.10 prompt contract changed")
    _resolve_bindings(plan, Path(repository_root).resolve())
    v9._load_and_validate_prefix(Path(repository_root).resolve())
    return plan


def load_execution_authorization(path: str | Path, *, plan_path: str | Path) -> dict[str, Any]:
    authorization = v8.v5.v2.predecessor._load_json(path)
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA or authorization.get("status") != "authorized_exact_host_bound_runner_repair_resume":
        raise ValueError("unsupported or changed v0.10 authorization")
    if authorization.get("approval_text") != EXACT_AUTHORIZATION_TEXT or authorization.get("plan_file_sha256") != file_sha256(plan_path):
        raise ValueError("v0.10 approval or plan binding changed")
    for key, expected in (("server", SERVER), ("limits", LIMITS), ("prohibitions", PROHIBITIONS), ("predecessor_failure", PREDECESSOR_FAILURE)):
        if authorization.get(key) != expected:
            raise ValueError(f"v0.10 authorization {key} binding changed")
    if not isinstance(authorization.get("authorized_start_unix"), int) or authorization["authorized_start_unix"] <= 0:
        raise ValueError("v0.10 authorization start time is invalid")
    return authorization


def load_runtime_reuse_receipt(path: str | Path, *, authorization_path: str | Path) -> dict[str, Any]:
    receipt = v8.v5.v2.predecessor._load_json(path)
    if receipt.get("schema_version") != RUNTIME_REUSE_RECEIPT_SCHEMA or receipt.get("status") != "PASS_exact_existing_localhost_runtime_reused_for_runner_repair_resume_v0.10":
        raise ValueError("unsupported or failed v0.10 runtime receipt")
    if receipt.get("authorization_file_sha256") != file_sha256(authorization_path) or receipt.get("server") != SERVER or receipt.get("runtime_reuse") != RUNTIME_REUSE:
        raise ValueError("v0.10 runtime receipt binding changed")
    if not all(receipt.get("checks", {}).values()):
        raise ValueError("v0.10 runtime checks failed")
    return receipt


@contextmanager
def _patched_v8() -> Iterator[None]:
    replacements = {
        "LIMITS": LIMITS,
        "PROMPT_CONTRACT": PROMPT_CONTRACT,
        "RUN_REPORT_SCHEMA": RUN_REPORT_SCHEMA,
        "RUN_VERIFICATION_SCHEMA": RUN_VERIFICATION_SCHEMA,
        "chat_completion_payload": chat_completion_payload,
        "full_canary_seed": v9.full_canary_seed,
        "load_execution_authorization": load_execution_authorization,
        "load_full_canary_plan": load_resume_plan,
        "load_runtime_reuse_receipt": load_runtime_reuse_receipt,
        "prompt_manifest": prompt_manifest,
        "validate_prompt_quality": validate_prompt_quality,
    }
    originals = {name: getattr(v8, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(v8, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(v8, name, value)


def patch_context_regression_gate(plan_path: Path, repository_root: Path) -> dict[str, Any]:
    scenario = next(item for item in v8.v5.v2.predecessor.SCENARIOS if item.task == "writing_summarization" and item.variant == 3)
    record_id = PREDECESSOR_FAILURE.get("failed_record_id", "lumi-gptoss-writing_summarization-natural_code_switching-03")
    with _patched_v8():
        plan = v8.load_full_canary_plan(plan_path.resolve(), repository_root=repository_root.resolve())
        payload = v8.chat_completion_payload(record_id, scenario, "natural_code_switching", attempt_number=1)
        manifest = v8.prompt_manifest()
    system_text = payload["messages"][0]["content"]
    checks = {
        "base_chat_capture_is_immutable": _BASE_V8_CHAT_COMPLETION_PAYLOAD is not chat_completion_payload,
        "base_validator_capture_is_immutable": _BASE_V8_VALIDATE_PROMPT_QUALITY is not validate_prompt_quality,
        "first_resume_record_exact": manifest[53]["record_id"] == "lumi-gptoss-writing_summarization-natural_code_switching-03",
        "manifest_exact": plan["prompt_contract"]["manifest_sha256"] == sha256_value(manifest),
        "patch_context_payload_completed": payload["seed"] == v9.full_canary_seed(record_id, 1),
        "required_literals_hard_bound": all(literal in system_text for literal in ("14:20", "API gateway", "18 minutes")),
    }
    if not all(checks.values()):
        raise ValueError("v0.10 patch-context regression gate failed")
    return checks


def build_plan_report(*, plan: Mapping[str, Any], plan_file_sha256: str, repository_root: Path) -> dict[str, Any]:
    resolved = _resolve_bindings(plan, repository_root.resolve())
    rows, telemetry, rejections = v9._load_and_validate_prefix(repository_root.resolve())
    checks = patch_context_regression_gate(repository_root.resolve() / str(plan["self_path"]), repository_root.resolve())
    checks.update({
        "authorization_pending": plan["authorization_pending"] is True,
        "prefix_exact": len(rows) == 53 and len(telemetry) == 53 and len(rejections) == 4,
        "zero_predecessor_inference": PREDECESSOR_FAILURE["additional_inference_calls"] == 0,
        "zero_download_ceiling": LIMITS["maximum_network_bytes"] == 0,
    })
    if not all(checks.values()):
        raise ValueError("v0.10 plan checks failed")
    report: dict[str, Any] = {
        "bindings": {name: {"path": plan["bindings"][name]["path"], "sha256": file_sha256(path)} for name, path in sorted(resolved.items())},
        "checks": checks,
        "limits": LIMITS,
        "next_gate": plan["next_gate"],
        "plan_file_sha256": plan_file_sha256,
        "predecessor_failure": PREDECESSOR_FAILURE,
        "prompt_contract": plan["prompt_contract"],
        "schema_version": REPORT_SCHEMA,
        "status": "verified_runner_repair_resume_plan_no_execution_authorized",
    }
    report["report_sha256"] = sha256_value(report)
    return report


def build_plan_verification(*, report: Mapping[str, Any], report_file_sha256: str) -> dict[str, Any]:
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256", None)
    checks = {
        "all_plan_checks_pass": all(report.get("checks", {}).values()),
        "internal_report_hash": internal == sha256_value(unsigned),
        "predecessor_failure_exact": report.get("predecessor_failure") == PREDECESSOR_FAILURE,
        "resume_output_count_exact": report.get("limits", {}).get("additional_final_outputs") == 107,
        "status_exact": report.get("status") == "verified_runner_repair_resume_plan_no_execution_authorized",
    }
    verification: dict[str, Any] = {
        "checks": checks,
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": internal,
        "schema_version": VERIFICATION_SCHEMA,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    verification["verification_sha256"] = sha256_value(verification)
    if verification["status"] != "PASS":
        raise ValueError("v0.10 independent plan verification failed")
    return verification


def run_full_canary_resume(*, plan_path: Path, authorization_path: Path, runtime_reuse_receipt_path: Path, repository_root: Path, artifact_root: Path, ssh_key: Path, known_hosts: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    load_resume_plan(plan_path.resolve(), repository_root=root)
    patch_context_regression_gate(plan_path.resolve(), root)
    load_execution_authorization(authorization_path.resolve(), plan_path=plan_path.resolve())
    load_runtime_reuse_receipt(runtime_reuse_receipt_path.resolve(), authorization_path=authorization_path.resolve())
    v9._seed_resume_artifact(root, artifact_root.resolve())
    with _patched_v8():
        return v8.run_full_canary(plan_path=plan_path, authorization_path=authorization_path, runtime_reuse_receipt_path=runtime_reuse_receipt_path, repository_root=repository_root, artifact_root=artifact_root, ssh_key=ssh_key, known_hosts=known_hosts)


def verify_full_canary_resume(*, plan_path: Path, authorization_path: Path, runtime_reuse_receipt_path: Path, repository_root: Path, artifact_root: Path) -> dict[str, Any]:
    with _patched_v8():
        return v8.verify_full_canary_run(plan_path=plan_path, authorization_path=authorization_path, runtime_reuse_receipt_path=runtime_reuse_receipt_path, repository_root=repository_root, artifact_root=artifact_root)
