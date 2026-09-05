# ruff: noqa: E501, SLF001
"""Recover one historical first-attempt miss and replay the full frozen canary gate."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import posttraining_gpt_oss_teacher_language_anchor_resume_v14 as v14
from .posttraining_candidate_mix_preflight import file_sha256, sha256_value

base = v14.base

PLAN_SCHEMA = "lumi.sft-gpt-oss-teacher-per-cell-recovery-plan.v0.15"
REPORT_SCHEMA = "lumi.sft-gpt-oss-teacher-per-cell-recovery-plan-report.v0.15"
VERIFICATION_SCHEMA = "lumi.sft-gpt-oss-teacher-per-cell-recovery-plan-verification.v0.15"
AUTHORIZATION_SCHEMA = "lumi.sft-gpt-oss-teacher-per-cell-recovery-authorization.v0.15"
RUNTIME_REUSE_RECEIPT_SCHEMA = "lumi.sft-gpt-oss-teacher-per-cell-recovery-runtime-reuse-receipt.v0.15"
RUN_REPORT_SCHEMA = "lumi.sft-gpt-oss-teacher-per-cell-recovery-run-report.v0.15"
RUN_VERIFICATION_SCHEMA = "lumi.sft-gpt-oss-teacher-per-cell-recovery-run-verification.v0.15"

PROMPT_CONTRACT = "lumi.gpt-oss-teacher.multilingual-english-first-attempt-recovery.v0.15"
SEED_CONTRACT = "sha256(record_id:multilingual-english-first-attempt-recovery:1)-u32.v0.15"
RECORD_ID = "lumi-gptoss-multilingual_bengali-english-00"
LANGUAGE = "english"
REASONING_EFFORT = "low"
MAX_TOKENS = 1_536
EXACT_USER_PROMPT = "Hi, could you give me a friendly greeting?"
EXACT_ASSISTANT_RESPONSE = "Hello! I hope you have a wonderful day."

SERVER = v14.SERVER
PROHIBITIONS = v14.PROHIBITIONS
RUNTIME_REUSE = v14.RUNTIME_REUSE
RETENTION = v14.RETENTION
FROZEN_MACHINE_GATES = v14.FROZEN_MACHINE_GATES
PREDECESSOR_FAILURE = {
    "accepted_outputs": 160,
    "candidate_file_sha256": "84258bf80ae4a2934ff7009d27b7f50bb69e64693b9cb252fed6a3828ab3e97e",
    "failing_cell": "multilingual_bengali:english",
    "failing_cell_first_attempt_acceptance": 0.7,
    "failure_report_file_sha256": "120e6965507b3651682d243f185928ae7aa810c740f07bc5d233e187324f9ac0",
    "failure_verification_file_sha256": "3e15724f8aeb62226e5ac7f33a76f5f756ba013eac9387c20c08f45d8a3ff161",
    "generation_telemetry_file_sha256": "f767af185db34b2dc3ddaa164273800f1ef5f09c9192b8502b1ed7c830fc53fe",
    "inference_calls": 169,
    "rejection_metadata_file_sha256": "428b5a288631eda0e584e43c054c873e712bfd25670c0c802bd4675b40cc7840",
    "runtime_retries": 9,
    "selected_rejection_sha256": "d0296683eef2cde049072dbbe60565cb063b41b0fd85f1aaed9a3d2b13d48e2a",
    "selected_row_sha256": "d622f6f4d2fb773a6ec01184920bdab51dce97221c4c6c4c9f4516fb06fc8959",
    "selected_telemetry_sha256": "1d42116f073b5265f19357640a8020fae08bee936cbada113bc6245184286b62",
    "status": "FAIL_CLOSED_inherited_multilingual_english_per_cell_acceptance_0_7",
}
LIMITS = {
    "exact_final_outputs_after_replacement": 160,
    "exact_repair_records": 1,
    "maximum_accepted_repair_outputs": 1,
    "maximum_attempts_per_record": 1,
    "maximum_context_tokens": 4_096,
    "maximum_generated_tokens_per_call": MAX_TOKENS,
    "maximum_generated_tokens_total": MAX_TOKENS,
    "maximum_inference_calls": 1,
    "maximum_instance_minutes": 20,
    "maximum_network_bytes": 0,
    "maximum_private_write_bytes": 1 * 1024**3,
    "maximum_runtime_retries": 0,
    "maximum_usd_before_tax": 1.43,
    "owner_review_rows_if_machine_gate_passes": 32,
    "projected_combined_inference_calls": 168,
    "projected_combined_runtime_retries": 8,
}
QUALITY_GATES = {
    "accepted_output_must_pass_frozen_parser": True,
    "exact_assistant_response": EXACT_ASSISTANT_RESPONSE,
    "exact_language_mode": LANGUAGE,
    "exact_record_id": RECORD_ID,
    "exact_user_prompt": EXACT_USER_PROMPT,
    "failing_cell_first_attempt_acceptance_after_replacement": 0.8,
    "full_16_cell_machine_gate_required_after_replacement": True,
    "no_second_attempt_allowed": True,
    "owner_review_sample_contains_replacement": True,
    "user_and_assistant_must_be_natural_user_facing_text": True,
}

EXACT_AUTHORIZATION_TEXT = (
    "I approve one bounded self-hosted Lumi gpt-oss-120b one-row multilingual-English first-attempt "
    "cell recovery on the existing exact Lambda host 68.209.73.55, bound to ED25519 host-key "
    "fingerprint SHA256:2ZSQqNUYJhWNjihHnqLQ1YNHxoqtMJeckVJRKlc5hs4 and exactly one NVIDIA "
    "H100 80GB GPU GPU-c7e68b23-95b4-37bd-b40b-c41a7ec716f3 reporting at least 81,559 MiB FB "
    "total without adding the 480 MiB reserved value twice. Reuse, and do not redownload, the 25 "
    "verified openai/gpt-oss-120b model objects at revision "
    "b5c939de8f754692c1647ca79fbf85e8c1e70f8a bound by model receipt SHA-256 "
    "798914d8532d5b36b61711f34591cabe1e03fa6ac1128fe5e3737c029ead1284, and reuse the existing "
    "vLLM image digest sha256:4d08193d2fd05aadb1b5678f93ae609efb2635df67da45f3efe781c368b34dc8. "
    "Bind the v0.14 failure report SHA-256 "
    "120e6965507b3651682d243f185928ae7aa810c740f07bc5d233e187324f9ac0 and failure verification "
    "SHA-256 3e15724f8aeb62226e5ac7f33a76f5f756ba013eac9387c20c08f45d8a3ff161. Reuse without "
    "regeneration the exact 160 individually accepted v0.14 rows bound by SHA-256 "
    "84258bf80ae4a2934ff7009d27b7f50bb69e64693b9cb252fed6a3828ab3e97e, their exact telemetry "
    "bound by SHA-256 f767af185db34b2dc3ddaa164273800f1ef5f09c9192b8502b1ed7c830fc53fe, and their nine-entry "
    "rejection metadata bound by SHA-256 428b5a288631eda0e584e43c054c873e712bfd25670c0c802bd4675b40cc7840. "
    "Run only record lumi-gptoss-multilingual_bengali-english-00 under prompt contract "
    "lumi.gpt-oss-teacher.multilingual-english-first-attempt-recovery.v0.15, reasoning effort low, "
    "a 1,536 generated-token ceiling, one deterministic seed, exactly one inference attempt, and "
    "zero retries. Require user_prompt to equal exactly Hi, could you give me a friendly greeting? "
    "and assistant_response to equal exactly Hello! I hope you have a wonderful day. If that exact "
    "row does not pass on the first call, fail closed and do not make a second call. On first-call "
    "success only, replace the predecessor copy of that record and its telemetry, exclude exactly "
    "its one bound historical rejection entry, preserve the other 159 rows and eight rejection "
    "metadata entries byte-for-byte, replay the full frozen 16-cell/160-row machine gate, and "
    "prepare the deterministic 32-row owner-review packet containing the replacement. Retain any "
    "failed response only as validation error, non-sensitive finish_reason, usage token counts, and "
    "request/response-envelope hashes without raw candidate or reasoning text. Maximum 20 additional "
    "instance-minutes / $1.43 before tax / 0 additional model or runtime download bytes / 1 GiB "
    "additional private writes / 1 sequential inference call / 0 retry calls / 1 accepted repair "
    "output. It may restart and then cleanly stop only the bound canary container and serve only on "
    "remote localhost through pinned SSH. No external API, public port, model or runtime download, "
    "registry change, dataset seal, SFT training, production promotion, unrelated access, new "
    "instance, or termination."
)

TOOLING_PATHS = {
    "implementation": "src/lumi/posttraining_gpt_oss_teacher_per_cell_recovery_v15.py",
    "plan_runner": "scripts/plan_sft_gpt_oss_teacher_per_cell_recovery_v15.py",
    "plan_verifier": "scripts/verify_sft_gpt_oss_teacher_per_cell_recovery_v15_plan.py",
    "run_runner": "scripts/run_sft_gpt_oss_teacher_per_cell_recovery_v15.py",
    "run_verifier": "scripts/verify_sft_gpt_oss_teacher_per_cell_recovery_v15.py",
    "test": "tests/test_gpt_oss_teacher_per_cell_recovery_v15.py",
}


def _scenario() -> Any:
    return next(item for item in base.SCENARIOS if item.task == "multilingual_bengali" and item.variant == 0)


def repair_seed() -> int:
    material = f"{RECORD_ID}:multilingual-english-first-attempt-recovery:1".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def repair_messages() -> list[dict[str, str]]:
    system = (
        "Return only one compact JSON object with scenario_id, user_prompt, assistant_response, "
        "safety_label, tool_call, and tool_result. Set scenario_id to friendly_greeting, "
        "safety_label to safe, and both tool fields to null. Never expose reasoning. Set "
        f"user_prompt to exactly this English string: {EXACT_USER_PROMPT} Set assistant_response "
        f"to exactly this English string: {EXACT_ASSISTANT_RESPONSE} Do not add, remove, translate, "
        "reformat, quote, label, or explain either field value. Use English only and no Bengali "
        "characters. Silently compare both values character-for-character before returning."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": "Return the exact final JSON object now."}]


def repair_payload() -> dict[str, Any]:
    payload = copy.deepcopy(base.chat_completion_payload(RECORD_ID, _scenario(), LANGUAGE))
    payload["chat_template_kwargs"] = {"reasoning_effort": REASONING_EFFORT}
    payload["max_tokens"] = MAX_TOKENS
    payload["messages"] = repair_messages()
    payload["seed"] = repair_seed()
    payload["temperature"] = 0.0
    return payload


def repair_manifest() -> dict[str, Any]:
    return {
        "attempt": {"attempt_number": 1, "max_tokens": MAX_TOKENS, "reasoning_effort": REASONING_EFFORT, "request_sha256": sha256_value(repair_payload()), "seed": repair_seed()},
        "language": LANGUAGE,
        "record_id": RECORD_ID,
        "scenario_id": "friendly_greeting",
        "task": "multilingual_bengali",
        "variant": 0,
    }


def validate_exact_fields(messages: Sequence[Mapping[str, str]]) -> None:
    if len(messages) != 2 or messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
        raise ValueError("v0.15 exact recovery message shape changed")
    if str(messages[0].get("content", "")) != EXACT_USER_PROMPT:
        raise ValueError("v0.15 user_prompt is not exact")
    if str(messages[1].get("content", "")) != EXACT_ASSISTANT_RESPONSE:
        raise ValueError("v0.15 assistant_response is not exact")
    v14.validate_prompt_quality(messages, scenario=_scenario(), language=LANGUAGE)


def _paths(repository_root: Path) -> dict[str, Path]:
    root = repository_root / "artifacts/lumi-gpt-oss-teacher-language-anchor-resume-v0.14/ce98cc04946e/attempt-001"
    return {
        "candidate": root / "private/candidate-rows.partial.jsonl",
        "failure_report": root / "evidence/failure-report.json",
        "failure_verification": root / "evidence/failure-verification.json",
        "rejections": root / "private/rejection-metadata.partial.jsonl",
        "telemetry": root / "private/generation-telemetry.partial.jsonl",
    }


def load_predecessor(repository_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    paths = _paths(repository_root)
    expected = {
        "candidate": PREDECESSOR_FAILURE["candidate_file_sha256"],
        "failure_report": PREDECESSOR_FAILURE["failure_report_file_sha256"],
        "failure_verification": PREDECESSOR_FAILURE["failure_verification_file_sha256"],
        "rejections": PREDECESSOR_FAILURE["rejection_metadata_file_sha256"],
        "telemetry": PREDECESSOR_FAILURE["generation_telemetry_file_sha256"],
    }
    for name, path in paths.items():
        if file_sha256(path) != expected[name]:
            raise ValueError(f"v0.15 predecessor binding changed: {name}")
    failure = base._load_json(paths["failure_report"])
    verification = base._load_json(paths["failure_verification"])
    if failure.get("status") != PREDECESSOR_FAILURE["status"] or verification.get("status") != "PASS":
        raise ValueError("v0.15 predecessor failure status changed")
    rows = base._read_jsonl(paths["candidate"])
    telemetry = base._read_jsonl(paths["telemetry"])
    rejections = base._read_jsonl(paths["rejections"])
    if len(rows) != 160 or len(telemetry) != 160 or len(rejections) != 9:
        raise ValueError("v0.15 predecessor counts changed")
    if [row["record_id"] for row in rows] != list(base.expected_record_ids()):
        raise ValueError("v0.15 predecessor order changed")
    selected_rows = [row for row in rows if row["record_id"] == RECORD_ID]
    selected_telemetry = [row for row in telemetry if row["record_id"] == RECORD_ID]
    selected_rejections = [row for row in rejections if row["record_id"] == RECORD_ID]
    if len(selected_rows) != 1 or len(selected_telemetry) != 1 or len(selected_rejections) != 1:
        raise ValueError("v0.15 selected predecessor membership changed")
    if sha256_value(selected_rows[0]) != PREDECESSOR_FAILURE["selected_row_sha256"] or sha256_value(selected_telemetry[0]) != PREDECESSOR_FAILURE["selected_telemetry_sha256"] or sha256_value(selected_rejections[0]) != PREDECESSOR_FAILURE["selected_rejection_sha256"]:
        raise ValueError("v0.15 selected predecessor object changed")
    if int(selected_telemetry[0]["attempts"]) != 2:
        raise ValueError("v0.15 selected predecessor was not a second-attempt acceptance")
    return rows, telemetry, rejections


def _replacement_row(*, request_sha256: str, seed: int, messages: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    return {
        "language_mode": LANGUAGE,
        "messages": [dict(item) for item in messages],
        "provenance": {
            "ai_assisted": True,
            "external_inference_api": False,
            "generation_attempt": 1,
            "human_authored": False,
            "inference_location": "self_hosted_lambda_h100_pinned_ssh",
            "model_id": base.MODEL_ID,
            "model_revision": base.MODEL_REVISION,
            "prompt_contract": PROMPT_CONTRACT,
            "prompt_request_sha256": request_sha256,
            "reasoning_effort": REASONING_EFFORT,
            "reasoning_or_analysis_retained": False,
            "runtime_image_digest": base.RUNTIME_DIGEST,
            "seed": seed,
        },
        "record_id": RECORD_ID,
        "rights_status": "private_research_candidate_pending_owner_review",
        "scenario_id": "friendly_greeting",
        "schema_version": base.ROW_SCHEMA,
        "task": "multilingual_bengali",
    }


def assemble_replacement(repository_root: Path, replacement_row: Mapping[str, Any], replacement_telemetry: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[Mapping[str, Any]]]:
    rows, telemetry, rejections = load_predecessor(repository_root)
    index = next(i for i, row in enumerate(rows) if row["record_id"] == RECORD_ID)
    rows[index] = dict(replacement_row)
    telemetry[index] = dict(replacement_telemetry)
    rejections = [item for item in rejections if item["record_id"] != RECORD_ID]
    if len(rejections) != 8 or sum(int(item["attempts"]) for item in telemetry) != 168:
        raise ValueError("v0.15 replacement accounting changed")
    scenarios = {(item.task, item.variant): item for item in base.SCENARIOS}
    for row in rows:
        variant = int(str(row["record_id"]).rsplit("-", 1)[1])
        v14.validate_prompt_quality(row["messages"], scenario=scenarios[(row["task"], variant)], language=row["language_mode"])
    audit = base._machine_audit(rows, telemetry)
    owner_sample = base.select_owner_sample(rows)
    if not all(audit["checks"].values()) or audit["per_cell_first_attempt_acceptance_rate"]["multilingual_bengali:english"] != 0.8:
        raise ValueError("v0.15 full machine gate did not recover")
    if RECORD_ID not in {str(item["record_id"]) for item in owner_sample} or len(owner_sample) != 32:
        raise ValueError("v0.15 owner sample binding changed")
    return rows, telemetry, rejections, audit, owner_sample


def _reference_replacement() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = repair_manifest()
    messages = [{"role": "user", "content": EXACT_USER_PROMPT}, {"role": "assistant", "content": EXACT_ASSISTANT_RESPONSE}]
    validate_exact_fields(messages)
    row = _replacement_row(request_sha256=manifest["attempt"]["request_sha256"], seed=manifest["attempt"]["seed"], messages=messages)
    telemetry = {
        "accepted_on_attempt": 1,
        "attempts": 1,
        "completion_tokens": 1,
        "critical_failures": 0,
        "elapsed_seconds_final_attempt": 0.0,
        "finish_reason": "stop",
        "prompt_tokens": 1,
        "record_id": RECORD_ID,
        "request_sha256": manifest["attempt"]["request_sha256"],
        "response_envelope_sha256": "0" * 64,
        "total_tokens": 2,
    }
    return row, telemetry


def load_recovery_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    plan = base._load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("status") != "per_cell_recovery_plan_pending_explicit_execution_authorization":
        raise ValueError("unsupported or changed v0.15 recovery plan")
    for key, expected in (("server", SERVER), ("runtime_reuse", RUNTIME_REUSE), ("limits", LIMITS), ("prohibitions", PROHIBITIONS), ("retention", RETENTION), ("predecessor_failure", PREDECESSOR_FAILURE), ("quality_gates", QUALITY_GATES), ("frozen_machine_gates", FROZEN_MACHINE_GATES)):
        if plan.get(key) != expected:
            raise ValueError(f"v0.15 {key} binding changed")
    if any(bool(value) for value in plan.get("authorization", {}).values() if isinstance(value, bool)):
        raise ValueError("v0.15 plan unexpectedly authorizes execution")
    if plan.get("next_gate", {}).get("exact_authorization_text") != EXACT_AUTHORIZATION_TEXT:
        raise ValueError("v0.15 authorization text changed")
    contract = plan.get("prompt_contract", {})
    if contract.get("id") != PROMPT_CONTRACT or contract.get("seed_contract") != SEED_CONTRACT or contract.get("manifest_sha256") != sha256_value(repair_manifest()):
        raise ValueError("v0.15 prompt contract changed")
    base._resolve_bindings(plan, repository_root)
    load_predecessor(Path(repository_root).resolve())
    return plan


def build_plan_report(*, plan: Mapping[str, Any], plan_file_sha256: str, repository_root: Path) -> dict[str, Any]:
    resolved = base._resolve_bindings(plan, repository_root.resolve())
    row, telemetry = _reference_replacement()
    _, _, rejections, audit, owner_sample = assemble_replacement(repository_root.resolve(), row, telemetry)
    prompt_text = json.dumps(repair_messages(), ensure_ascii=False)
    checks = {
        "authorization_pending": not any(bool(value) for value in plan["authorization"].values() if isinstance(value, bool)),
        "exact_fields_bound": EXACT_USER_PROMPT in prompt_text and EXACT_ASSISTANT_RESPONSE in prompt_text,
        "exact_one_call_zero_retry": LIMITS["maximum_inference_calls"] == 1 and LIMITS["maximum_runtime_retries"] == 0,
        "failing_cell_recovers_to_exact_minimum": audit["per_cell_first_attempt_acceptance_rate"]["multilingual_bengali:english"] == 0.8,
        "full_machine_gate_reference_passes": all(audit["checks"].values()),
        "owner_sample_exact_and_contains_replacement": len(owner_sample) == 32 and RECORD_ID in {str(item["record_id"]) for item in owner_sample},
        "predecessor_rejections_reduce_by_exactly_one": len(rejections) == 8,
        "zero_download_ceiling": LIMITS["maximum_network_bytes"] == 0,
    }
    if not all(checks.values()):
        raise ValueError("v0.15 plan checks failed")
    report: dict[str, Any] = {
        "authorization": plan["authorization"],
        "bindings": {name: {"path": plan["bindings"][name]["path"], "sha256": file_sha256(path)} for name, path in sorted(resolved.items())},
        "checks": checks,
        "frozen_machine_gates": FROZEN_MACHINE_GATES,
        "limits": LIMITS,
        "manifest": repair_manifest(),
        "next_gate": plan["next_gate"],
        "plan_file_sha256": plan_file_sha256,
        "predecessor_failure": PREDECESSOR_FAILURE,
        "prompt_contract": plan["prompt_contract"],
        "quality_gates": QUALITY_GATES,
        "schema_version": REPORT_SCHEMA,
        "status": "verified_per_cell_recovery_plan_no_execution_authorized",
        "tooling": {name: {"path": tooling_path, "sha256": file_sha256(repository_root / tooling_path)} for name, tooling_path in sorted(TOOLING_PATHS.items())},
    }
    report["report_sha256"] = sha256_value(report)
    return report


def build_plan_verification(*, report: Mapping[str, Any], report_file_sha256: str) -> dict[str, Any]:
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256", None)
    checks = {
        "all_plan_checks_pass": all(report.get("checks", {}).values()),
        "call_ceiling_one": report.get("limits", {}).get("maximum_inference_calls") == 1,
        "internal_report_hash": internal == sha256_value(unsigned),
        "predecessor_failure_exact": report.get("predecessor_failure") == PREDECESSOR_FAILURE,
        "status_exact": report.get("status") == "verified_per_cell_recovery_plan_no_execution_authorized",
        "zero_retry_exact": report.get("limits", {}).get("maximum_runtime_retries") == 0,
    }
    verification: dict[str, Any] = {"checks": checks, "report_file_sha256": report_file_sha256, "report_internal_sha256": internal, "schema_version": VERIFICATION_SCHEMA, "status": "PASS" if all(checks.values()) else "FAIL"}
    verification["verification_sha256"] = sha256_value(verification)
    if verification["status"] != "PASS":
        raise ValueError("v0.15 independent plan verification failed")
    return verification


def load_execution_authorization(path: str | Path, *, plan_path: str | Path) -> dict[str, Any]:
    authorization = base._load_json(path)
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA or authorization.get("status") != "authorized_exact_host_bound_one_row_per_cell_recovery":
        raise ValueError("unsupported or changed v0.15 authorization")
    if authorization.get("approval_text") != EXACT_AUTHORIZATION_TEXT or authorization.get("plan_file_sha256") != file_sha256(plan_path):
        raise ValueError("v0.15 approval or plan binding changed")
    for key, expected in (("server", SERVER), ("limits", LIMITS), ("prohibitions", PROHIBITIONS), ("predecessor_failure", PREDECESSOR_FAILURE)):
        if authorization.get(key) != expected:
            raise ValueError(f"v0.15 authorization {key} binding changed")
    if not isinstance(authorization.get("authorized_start_unix"), int) or authorization["authorized_start_unix"] <= 0:
        raise ValueError("v0.15 authorization start time is invalid")
    return authorization


def load_runtime_reuse_receipt(path: str | Path, *, authorization_path: str | Path) -> dict[str, Any]:
    receipt = base._load_json(path)
    if receipt.get("schema_version") != RUNTIME_REUSE_RECEIPT_SCHEMA or receipt.get("status") != "PASS_exact_existing_localhost_runtime_reused_for_per_cell_recovery_v0.15":
        raise ValueError("unsupported or failed v0.15 runtime receipt")
    if receipt.get("authorization_file_sha256") != file_sha256(authorization_path) or receipt.get("server") != SERVER or receipt.get("runtime_reuse") != RUNTIME_REUSE:
        raise ValueError("v0.15 runtime receipt binding changed")
    if not all(receipt.get("checks", {}).values()):
        raise ValueError("v0.15 runtime checks failed")
    return receipt


def run_per_cell_recovery(*, plan_path: Path, authorization_path: Path, runtime_reuse_receipt_path: Path, repository_root: Path, artifact_root: Path, ssh_key: Path, known_hosts: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    plan = load_recovery_plan(plan_path.resolve(), repository_root=root)
    authorization = load_execution_authorization(authorization_path.resolve(), plan_path=plan_path.resolve())
    load_runtime_reuse_receipt(runtime_reuse_receipt_path.resolve(), authorization_path=authorization_path.resolve())
    if file_sha256(known_hosts.resolve()) != plan["bindings"]["known_hosts"]["sha256"] or ssh_key.resolve().stat().st_mode & 0o077:
        raise PermissionError("v0.15 SSH binding or permissions changed")
    if time.time() - int(authorization["authorized_start_unix"]) > LIMITS["maximum_instance_minutes"] * 60:
        raise RuntimeError("v0.15 authorization expired")
    artifact = artifact_root.resolve()
    if not artifact.is_relative_to((root / "artifacts").resolve()):
        raise ValueError("v0.15 artifact root escaped artifacts/")
    private, evidence = artifact / "private", artifact / "evidence"
    private.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    attempt_path = private / "repair-attempt-metadata.jsonl"
    report_path = evidence / "run-report.json"
    if any(path.exists() for path in (attempt_path, report_path, private / "candidate-rows.jsonl")):
        raise FileExistsError("refusing to replay a consumed v0.15 artifact root")
    manifest = repair_manifest()
    payload = repair_payload()
    request_sha256 = sha256_value(payload)
    if request_sha256 != manifest["attempt"]["request_sha256"]:
        raise ValueError("v0.15 request manifest changed")
    candidate: dict[str, Any] | None = None
    replacement_telemetry: dict[str, Any] | None = None
    error: Exception | None = None
    before_gpu = base.gpu_snapshot(ssh_key.resolve(), known_hosts.resolve())
    call_started = time.monotonic()
    try:
        response = base._post_chat_completion(payload)
        elapsed = time.monotonic() - call_started
        diagnostic = v14.v12.v10.v8.v5._response_metadata(response)
        try:
            messages = base._parse_final_response(response, scenario=_scenario(), language=LANGUAGE)
            validate_exact_fields(messages)
        except (ValueError, json.JSONDecodeError) as observed:
            error = observed
            messages = None
        attempt = {
            "accepted": error is None,
            "attempt_number": 1,
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
        }
        base._write_jsonl(attempt_path, [attempt])
        if error is None:
            assert messages is not None
            candidate = _replacement_row(request_sha256=request_sha256, seed=repair_seed(), messages=messages)
            replacement_telemetry = {
                "accepted_on_attempt": 1,
                "attempts": 1,
                "completion_tokens": diagnostic["usage"]["completion_tokens"],
                "critical_failures": 0,
                "elapsed_seconds_final_attempt": round(elapsed, 6),
                "finish_reason": diagnostic["finish_reason"],
                "prompt_tokens": diagnostic["usage"]["prompt_tokens"],
                "record_id": RECORD_ID,
                "request_sha256": request_sha256,
                "response_envelope_sha256": sha256_value(response),
                "total_tokens": diagnostic["usage"]["total_tokens"],
            }
            rows, telemetry, rejections, audit, owner_sample = assemble_replacement(root, candidate, replacement_telemetry)
            base._write_jsonl(private / "repair-candidate.jsonl", [candidate])
            base._write_jsonl(private / "candidate-rows.jsonl", rows)
            base._write_jsonl(private / "generation-telemetry.jsonl", telemetry)
            base._write_jsonl(private / "rejection-metadata.jsonl", rejections)
            base._write_jsonl(private / "owner-review-sample-32.jsonl", owner_sample)
    finally:
        v14.v12.v10.v8.v5.v3._stop_runtime(ssh_key.resolve(), known_hosts.resolve())
    after_gpu = base.gpu_snapshot(ssh_key.resolve(), known_hosts.resolve())
    success = candidate is not None and replacement_telemetry is not None
    report: dict[str, Any] = {
        "accepted_repair_outputs": 1 if success else 0,
        "attempt_metadata_file_sha256": file_sha256(attempt_path),
        "authorization_file_sha256": file_sha256(authorization_path),
        "candidate_registry_changed": False,
        "dataset_sealed": False,
        "external_api_used": False,
        "gpu": {"after_stop": after_gpu, "before": before_gpu},
        "inference_calls": 1,
        "machine_audit": audit if success else None,
        "owner_review": {"rows": 32 if success else 0, "status": "pending" if success else "not_run"},
        "plan_file_sha256": file_sha256(plan_path),
        "production_eligible": False,
        "prompt_contract": PROMPT_CONTRACT,
        "runtime_retries": 0,
        "runtime_reuse_receipt_file_sha256": file_sha256(runtime_reuse_receipt_path),
        "runtime_stopped": after_gpu["memory_used_mib"] == 0,
        "schema_version": RUN_REPORT_SCHEMA,
        "sft_training_started": False,
        "status": "machine_verified_160_row_canary_pending_32_row_owner_review" if success else "FAIL_CLOSED_first_attempt_recovery_not_exact",
    }
    if success:
        report["candidate_file_sha256"] = file_sha256(private / "candidate-rows.jsonl")
        report["generation_telemetry_file_sha256"] = file_sha256(private / "generation-telemetry.jsonl")
        report["owner_review_sample_file_sha256"] = file_sha256(private / "owner-review-sample-32.jsonl")
        report["rejection_metadata_file_sha256"] = file_sha256(private / "rejection-metadata.jsonl")
        report["repair_candidate_file_sha256"] = file_sha256(private / "repair-candidate.jsonl")
    report["report_sha256"] = sha256_value(report)
    base._write_json_atomic(report_path, report)
    return report


def verify_per_cell_recovery(*, plan_path: Path, authorization_path: Path, runtime_reuse_receipt_path: Path, repository_root: Path, artifact_root: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    load_recovery_plan(plan_path.resolve(), repository_root=root)
    load_execution_authorization(authorization_path.resolve(), plan_path=plan_path.resolve())
    load_runtime_reuse_receipt(runtime_reuse_receipt_path.resolve(), authorization_path=authorization_path.resolve())
    artifact = artifact_root.resolve()
    report_path = artifact / "evidence/run-report.json"
    report = base._load_json(report_path)
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256", None)
    attempts = base._read_jsonl(artifact / "private/repair-attempt-metadata.jsonl")
    rows = base._read_jsonl(artifact / "private/candidate-rows.jsonl")
    telemetry = base._read_jsonl(artifact / "private/generation-telemetry.jsonl")
    rejections = base._read_jsonl(artifact / "private/rejection-metadata.jsonl")
    owner_sample = base._read_jsonl(artifact / "private/owner-review-sample-32.jsonl")
    scenarios = {(item.task, item.variant): item for item in base.SCENARIOS}
    for row in rows:
        variant = int(str(row["record_id"]).rsplit("-", 1)[1])
        v14.validate_prompt_quality(row["messages"], scenario=scenarios[(row["task"], variant)], language=row["language_mode"])
    audit = base._machine_audit(rows, telemetry)
    checks = {
        "authorization_binding": report.get("authorization_file_sha256") == file_sha256(authorization_path),
        "exact_one_attempt": len(attempts) == 1 and attempts[0].get("accepted") is True,
        "full_machine_gate_reproduced": all(audit["checks"].values()) and audit == report.get("machine_audit"),
        "internal_report_hash": internal == sha256_value(unsigned),
        "owner_sample_reproduced": owner_sample == base.select_owner_sample(rows) and RECORD_ID in {str(item["record_id"]) for item in owner_sample},
        "replacement_accounting_exact": len(rows) == 160 and sum(int(item["attempts"]) for item in telemetry) == 168 and len(rejections) == 8,
        "runtime_binding": report.get("runtime_reuse_receipt_file_sha256") == file_sha256(runtime_reuse_receipt_path),
        "runtime_stopped": report.get("runtime_stopped") is True and report.get("gpu", {}).get("after_stop", {}).get("memory_used_mib") == 0,
        "zero_production_actions": report.get("candidate_registry_changed") is False and report.get("dataset_sealed") is False and report.get("sft_training_started") is False and report.get("production_eligible") is False,
    }
    verification: dict[str, Any] = {"checks": checks, "run_report_file_sha256": file_sha256(report_path), "run_report_internal_sha256": internal, "schema_version": RUN_VERIFICATION_SCHEMA, "status": "PASS" if all(checks.values()) else "FAIL"}
    verification["verification_sha256"] = sha256_value(verification)
    base._write_json_atomic(artifact / "evidence/run-verification.json", verification)
    if verification["status"] != "PASS":
        raise ValueError("v0.15 run verification failed")
    return verification
