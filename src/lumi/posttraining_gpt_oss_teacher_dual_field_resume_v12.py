# ruff: noqa: E501, SLF001
"""Resume the full canary from 55 accepted rows with exact dual-field literals."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import posttraining_gpt_oss_teacher_canary_v10 as v10
from . import posttraining_gpt_oss_teacher_dual_field_literal_repair_v11 as v11
from .posttraining_candidate_mix_preflight import file_sha256, sha256_value

v8 = v10.v8
base = v11.base

PLAN_SCHEMA = "lumi.sft-gpt-oss-teacher-dual-field-resume-plan.v0.12"
REPORT_SCHEMA = "lumi.sft-gpt-oss-teacher-dual-field-resume-report.v0.12"
VERIFICATION_SCHEMA = "lumi.sft-gpt-oss-teacher-dual-field-resume-verification.v0.12"
AUTHORIZATION_SCHEMA = "lumi.sft-gpt-oss-teacher-dual-field-resume-authorization.v0.12"
RUNTIME_REUSE_RECEIPT_SCHEMA = "lumi.sft-gpt-oss-teacher-dual-field-resume-runtime-reuse-receipt.v0.12"
RUN_REPORT_SCHEMA = "lumi.sft-gpt-oss-teacher-dual-field-resume-run-report.v0.12"
RUN_VERIFICATION_SCHEMA = "lumi.sft-gpt-oss-teacher-dual-field-resume-run-verification.v0.12"

PROMPT_CONTRACT = "lumi.gpt-oss-teacher.dual-field-literal-resume.v0.12"
SEED_CONTRACT = "sha256(record_id:dual-field-literal-resume-attempt:attempt_number)-u32.v0.12"
REASONING_EFFORT = "low"
MAX_TOKENS = 1_536
MAXIMUM_ATTEMPTS_PER_RECORD = 2

SERVER = v10.SERVER
PROHIBITIONS = v10.PROHIBITIONS
RUNTIME_REUSE = v10.RUNTIME_REUSE
RETENTION = v10.RETENTION
FROZEN_MACHINE_GATES = v10.FROZEN_MACHINE_GATES
ACCEPTED_REPAIRS = dict(v10.ACCEPTED_REPAIRS)
ACCEPTED_REPAIRS["bengali_dual_field_v0_11"] = {
    "accepted_rows": 1,
    "candidate_file_sha256": "77daa9279fdfeb806e79ce9c81bb1711ecff8a8ba3315cc853bec5b8cc4556a9",
    "owner_review_file_sha256": "6c6c526a6b1528dc31bb5f994cc67b013193a8c4f7aaf91cbc63e8cdc3a82cd2",
    "owner_review_status": "ACCEPT_dual_field_literal_quality",
    "owner_review_verification_file_sha256": "01c23d0bf9b699d3408e13dcf669f6900dce54bede2ae3785e806937649579dd",
    "run_report_file_sha256": "fe7fa5ff46e46a50d7ec8daa4355e9a1bc0f0518a3f26a2be3fd8e0e5f793245",
    "run_verification_file_sha256": "e4e93951a3f8a84f30f67f7a264c971dc89e87399faa2d6d66eb3cdbbed5cc0a",
}
PREDECESSOR = {
    "accepted_prefix_outputs": 54,
    "candidate_prefix_file_sha256": "1c02b41eac74461779118545eb128629e8f8363dc87b5b0db0bd8e085bb37888",
    "generation_telemetry_file_sha256": "7d2573bcff0a6efebdf5c276fa420e998347b34cf45bad481c8fedf3b74af708",
    "recovered_rejection_metadata_rows": 4,
    "rejection_metadata_file_sha256": "31144d365519ab7e62aecdcf37053e52e4cc5a3575680be917185721ff1c9bca",
    "replacement_candidate_file_sha256": "77daa9279fdfeb806e79ce9c81bb1711ecff8a8ba3315cc853bec5b8cc4556a9",
    "replacement_owner_review_file_sha256": "6c6c526a6b1528dc31bb5f994cc67b013193a8c4f7aaf91cbc63e8cdc3a82cd2",
    "replacement_owner_verification_file_sha256": "01c23d0bf9b699d3408e13dcf669f6900dce54bede2ae3785e806937649579dd",
    "replacement_telemetry_file_sha256": "4a02a2f1361ef509e04ef6ce90227baa71de4453275323a8d41b1060444cfe12",
    "status": "accepted_55_row_prefix_with_v0_11_replacement",
}
LIMITS = {
    "additional_final_outputs": 105,
    "exact_final_outputs": 160,
    "exact_prompt_cells": 16,
    "maximum_additional_generated_tokens_total": 216_576,
    "maximum_additional_inference_calls": 141,
    "maximum_additional_runtime_retries": 36,
    "maximum_attempts_per_record": 2,
    "maximum_context_tokens": 4_096,
    "maximum_generated_tokens_per_call": MAX_TOKENS,
    "maximum_inference_calls": 200,
    "maximum_instance_minutes": 90,
    "maximum_network_bytes": 0,
    "maximum_private_write_bytes": 16 * 1024**3,
    "maximum_runtime_retries": 40,
    "maximum_usd_before_tax": 6.44,
    "owner_review_rows": 32,
    "predecessor_accepted_prefix_outputs": 55,
    "predecessor_retained_prefix_attempts": 59,
    "prompts_per_cell": 10,
}
PROMPT_QUALITY_GATES = dict(v10.v9.PROMPT_QUALITY_GATES)
PROMPT_QUALITY_GATES.update({
    "all_frozen_required_literals_verbatim_in_assistant_response": True,
    "all_frozen_required_literals_verbatim_in_user_prompt": True,
    "accepted_v0_11_replacement_revalidated": True,
    "predecessor_prefix_revalidated": True,
})

EXACT_AUTHORIZATION_TEXT = (
    "I approve one bounded self-hosted Lumi gpt-oss-120b remaining-105 dual-field literal full-canary "
    "resume successor on the existing exact Lambda host 68.209.73.55, bound to ED25519 host-key "
    "fingerprint SHA256:2ZSQqNUYJhWNjihHnqLQ1YNHxoqtMJeckVJRKlc5hs4 and exactly one NVIDIA "
    "H100 80GB GPU GPU-c7e68b23-95b4-37bd-b40b-c41a7ec716f3 reporting at least 81,559 MiB FB "
    "total without adding the 480 MiB reserved value twice. Reuse, and do not redownload, the 25 "
    "verified openai/gpt-oss-120b model objects at revision "
    "b5c939de8f754692c1647ca79fbf85e8c1e70f8a bound by model receipt SHA-256 "
    "798914d8532d5b36b61711f34591cabe1e03fa6ac1128fe5e3737c029ead1284, and reuse the existing "
    "vLLM image digest sha256:4d08193d2fd05aadb1b5678f93ae609efb2635df67da45f3efe781c368b34dc8. "
    "Bind the accepted v0.11 owner-review SHA-256 "
    "6c6c526a6b1528dc31bb5f994cc67b013193a8c4f7aaf91cbc63e8cdc3a82cd2 and owner-review "
    "verification SHA-256 01c23d0bf9b699d3408e13dcf669f6900dce54bede2ae3785e806937649579dd. "
    "Reuse without regeneration the exact 54 accepted v0.10 prefix rows bound by SHA-256 "
    "1c02b41eac74461779118545eb128629e8f8363dc87b5b0db0bd8e085bb37888 and their exact telemetry "
    "bound by SHA-256 7d2573bcff0a6efebdf5c276fa420e998347b34cf45bad481c8fedf3b74af708, append the exact "
    "accepted v0.11 replacement row bound by SHA-256 "
    "77daa9279fdfeb806e79ce9c81bb1711ecff8a8ba3315cc853bec5b8cc4556a9 and its accepted-attempt "
    "metadata bound by SHA-256 4a02a2f1361ef509e04ef6ce90227baa71de4453275323a8d41b1060444cfe12, "
    "carry forward only the four recovered-record rejection metadata entries, and exclude both "
    "failed v0.10 attempts for the repaired record. Generate only the remaining 105 records, "
    "beginning with lumi-gptoss-writing_summarization-english-03, under prompt contract "
    "lumi.gpt-oss-teacher.dual-field-literal-resume.v0.12, reasoning effort low, a 1,536 "
    "generated-token ceiling per call, distinct attempt-bound deterministic seeds, and at most 2 "
    "new attempts per remaining record. Preserve all complete-row gates and, for every scenario "
    "with frozen required literals, require each literal separately and verbatim in both user_prompt "
    "and assistant_response without translation, transliteration, inflection, reformatting, or "
    "omission. Retain rejected new attempts only as validation error, non-sensitive finish_reason, "
    "usage token counts, and request/response-envelope hashes without raw candidate or reasoning "
    "text. Maximum 90 additional instance-minutes / $6.44 before tax / 0 additional model or "
    "runtime download bytes / 16 GiB additional private writes / 141 additional sequential inference "
    "calls / 36 additional retry calls / 105 additional final outputs, while the combined retained-"
    "prefix plus resume audit remains capped at 200 calls and 40 retries. It may restart and then "
    "cleanly stop only the bound canary container, serve only on remote localhost through pinned SSH, "
    "retain only accepted final-channel candidate text, run the full frozen 16-cell/160-row machine "
    "gate, and prepare the deterministic 32-row owner-review packet. No external API, public port, "
    "model or runtime download, registry change, dataset seal, SFT training, production promotion, "
    "unrelated access, new instance, or termination."
)

_BASE_CHAT = v10._BASE_V8_CHAT_COMPLETION_PAYLOAD
_BASE_VALIDATE = v10._BASE_V8_VALIDATE_PROMPT_QUALITY


def full_canary_seed(record_id: str, attempt_number: int) -> int:
    if not 1 <= attempt_number <= MAXIMUM_ATTEMPTS_PER_RECORD:
        raise ValueError("v0.12 attempt number is outside the reviewed contract")
    material = f"{record_id}:dual-field-literal-resume-attempt:{attempt_number}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def generation_messages(scenario: Any, language: str, *, attempt_number: int) -> list[dict[str, str]]:
    messages = copy.deepcopy(v10.v9.generation_messages(scenario, language, attempt_number=attempt_number))
    if scenario.required_literals:
        literals = " | ".join(scenario.required_literals)
        messages[0]["content"] += (
            " The JSON string value user_prompt must separately contain every following required "
            f"literal verbatim: {literals}. The JSON string value assistant_response must also "
            f"separately contain every following required literal verbatim: {literals}. Copy each "
            "literal with identical case, digits, punctuation, and spacing in both fields; do not "
            "translate, transliterate, inflect, reformat, or omit any literal. Silently verify all "
            "required occurrences in both fields before returning the JSON object."
        )
    return messages


def chat_completion_payload(record_id: str, scenario: Any, language: str, *, attempt_number: int) -> dict[str, Any]:
    payload = copy.deepcopy(_BASE_CHAT(record_id, scenario, language, attempt_number=attempt_number))
    payload["messages"] = generation_messages(scenario, language, attempt_number=attempt_number)
    payload["seed"] = full_canary_seed(record_id, attempt_number)
    payload["temperature"] = 0.2
    return payload


def validate_prompt_quality(messages: Sequence[Mapping[str, str]], *, scenario: Any, language: str) -> None:
    _BASE_VALIDATE(messages, scenario=scenario, language=language)
    user = str(messages[0].get("content", ""))
    assistant = str(messages[-1].get("content", ""))
    for literal in scenario.required_literals:
        if literal not in user:
            raise ValueError(f"v0.12 user_prompt missing required literal: {literal}")
        if literal not in assistant:
            raise ValueError(f"v0.12 assistant_response missing required literal: {literal}")


def prompt_manifest() -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for scenario in base.SCENARIOS:
        for language in base.LANGUAGE_MODES:
            record_id = f"lumi-gptoss-{scenario.task}-{language}-{scenario.variant:02d}"
            attempts = [
                {"attempt_number": attempt, "max_tokens": MAX_TOKENS, "reasoning_effort": REASONING_EFFORT, "request_sha256": sha256_value(chat_completion_payload(record_id, scenario, language, attempt_number=attempt)), "seed": full_canary_seed(record_id, attempt)}
                for attempt in range(1, MAXIMUM_ATTEMPTS_PER_RECORD + 1)
            ]
            manifest.append({"attempts": attempts, "language": language, "record_id": record_id, "scenario_id": scenario.scenario_id, "task": scenario.task, "variant": scenario.variant})
    return manifest


def _paths(repository_root: Path) -> dict[str, Path]:
    v10_root = repository_root / "artifacts/lumi-gpt-oss-teacher-runner-repair-resume-v0.10/184f349b4aaa/attempt-001"
    v11_root = repository_root / "artifacts/lumi-gpt-oss-teacher-dual-field-literal-repair-v0.11/9dc3357f2caf/attempt-001"
    return {
        "v10_rows": v10_root / "private/candidate-rows.partial.jsonl",
        "v10_telemetry": v10_root / "private/generation-telemetry.partial.jsonl",
        "v10_rejections": v10_root / "private/rejection-metadata.partial.jsonl",
        "v11_candidate": v11_root / "private/repair-candidate.jsonl",
        "v11_telemetry": v11_root / "private/repair-attempt-metadata.jsonl",
        "v11_owner": v11_root / "evidence/owner-review.json",
        "v11_owner_verification": v11_root / "evidence/owner-review-verification.json",
    }


def _load_and_validate_prefix(repository_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    paths = _paths(repository_root)
    expected_hashes = {
        "v10_rows": PREDECESSOR["candidate_prefix_file_sha256"],
        "v10_telemetry": PREDECESSOR["generation_telemetry_file_sha256"],
        "v10_rejections": PREDECESSOR["rejection_metadata_file_sha256"],
        "v11_candidate": PREDECESSOR["replacement_candidate_file_sha256"],
        "v11_telemetry": PREDECESSOR["replacement_telemetry_file_sha256"],
        "v11_owner": PREDECESSOR["replacement_owner_review_file_sha256"],
        "v11_owner_verification": PREDECESSOR["replacement_owner_verification_file_sha256"],
    }
    for name, path in paths.items():
        if file_sha256(path) != expected_hashes[name]:
            raise ValueError(f"v0.12 predecessor binding changed: {name}")
    rows = base._read_jsonl(paths["v10_rows"])
    telemetry = base._read_jsonl(paths["v10_telemetry"])
    all_rejections = base._read_jsonl(paths["v10_rejections"])
    replacement = base._read_jsonl(paths["v11_candidate"])
    replacement_metadata = base._read_jsonl(paths["v11_telemetry"])
    owner = base._load_json(paths["v11_owner"])
    owner_verification = base._load_json(paths["v11_owner_verification"])
    if len(rows) != 54 or len(telemetry) != 54 or len(replacement) != 1 or len(replacement_metadata) != 1:
        raise ValueError("v0.12 predecessor count changed")
    if owner.get("status") != "ACCEPT_dual_field_literal_quality" or owner_verification.get("status") != "PASS_owner_acceptance_verified":
        raise ValueError("v0.12 replacement owner acceptance changed")
    expected_ids = base.expected_record_ids()
    if [row["record_id"] for row in rows] != list(expected_ids[:54]) or replacement[0]["record_id"] != expected_ids[54]:
        raise ValueError("v0.12 accepted prefix ordering changed")
    rows.append(replacement[0])
    item = replacement_metadata[0]
    telemetry.append({
        "accepted_on_attempt": int(item["attempt_number"]),
        "attempts": int(item["attempt_number"]),
        "completion_tokens": int(item["completion_tokens"]),
        "critical_failures": 0,
        "elapsed_seconds_final_attempt": None,
        "finish_reason": item["finish_reason"],
        "prompt_tokens": int(item["prompt_tokens"]),
        "record_id": item["record_id"],
        "request_sha256": item["request_sha256"],
        "response_envelope_sha256": item["response_envelope_sha256"],
        "total_tokens": int(item["total_tokens"]),
    })
    accepted_ids = set(expected_ids[:54])
    rejections = [entry for entry in all_rejections if entry["record_id"] in accepted_ids]
    if len(rejections) != 4:
        raise ValueError("v0.12 recovered rejection count changed")
    scenarios = {(item.task, item.variant): item for item in base.SCENARIOS}
    for row in rows:
        variant = int(str(row["record_id"]).rsplit("-", 1)[1])
        validate_prompt_quality(row["messages"], scenario=scenarios[(row["task"], variant)], language=row["language_mode"])
    if sum(int(item["attempts"]) for item in telemetry) != LIMITS["predecessor_retained_prefix_attempts"]:
        raise ValueError("v0.12 retained attempt accounting changed")
    return rows, telemetry, rejections


def load_resume_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    plan = base._load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("status") != "dual_field_resume_plan_pending_explicit_execution_authorization":
        raise ValueError("unsupported or changed v0.12 resume plan")
    for key, expected in (("server", SERVER), ("runtime_reuse", RUNTIME_REUSE), ("limits", LIMITS), ("prohibitions", PROHIBITIONS), ("retention", RETENTION), ("predecessor", PREDECESSOR), ("accepted_repairs", ACCEPTED_REPAIRS), ("prompt_quality_gates", PROMPT_QUALITY_GATES), ("frozen_machine_gates", FROZEN_MACHINE_GATES)):
        if plan.get(key) != expected:
            raise ValueError(f"v0.12 {key} binding changed")
    if any(bool(value) for value in plan.get("authorization", {}).values() if isinstance(value, bool)):
        raise ValueError("v0.12 plan unexpectedly authorizes execution")
    if plan.get("next_gate", {}).get("exact_authorization_text") != EXACT_AUTHORIZATION_TEXT:
        raise ValueError("v0.12 authorization text changed")
    contract = plan.get("prompt_contract", {})
    if contract.get("id") != PROMPT_CONTRACT or contract.get("seed_contract") != SEED_CONTRACT or contract.get("manifest_sha256") != sha256_value(prompt_manifest()):
        raise ValueError("v0.12 prompt contract changed")
    base._resolve_bindings(plan, repository_root)
    _load_and_validate_prefix(Path(repository_root).resolve())
    return plan


def load_execution_authorization(path: str | Path, *, plan_path: str | Path) -> dict[str, Any]:
    authorization = base._load_json(path)
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA or authorization.get("status") != "authorized_exact_host_bound_dual_field_resume":
        raise ValueError("unsupported or changed v0.12 authorization")
    if authorization.get("approval_text") != EXACT_AUTHORIZATION_TEXT or authorization.get("plan_file_sha256") != file_sha256(plan_path):
        raise ValueError("v0.12 approval or plan binding changed")
    for key, expected in (("server", SERVER), ("limits", LIMITS), ("prohibitions", PROHIBITIONS), ("predecessor", PREDECESSOR), ("accepted_repairs", ACCEPTED_REPAIRS)):
        if authorization.get(key) != expected:
            raise ValueError(f"v0.12 authorization {key} binding changed")
    if not isinstance(authorization.get("authorized_start_unix"), int) or authorization["authorized_start_unix"] <= 0:
        raise ValueError("v0.12 authorization start time is invalid")
    return authorization


def load_runtime_reuse_receipt(path: str | Path, *, authorization_path: str | Path) -> dict[str, Any]:
    receipt = base._load_json(path)
    if receipt.get("schema_version") != RUNTIME_REUSE_RECEIPT_SCHEMA or receipt.get("status") != "PASS_exact_existing_localhost_runtime_reused_for_dual_field_resume_v0.12":
        raise ValueError("unsupported or failed v0.12 runtime receipt")
    if receipt.get("authorization_file_sha256") != file_sha256(authorization_path) or receipt.get("server") != SERVER or receipt.get("runtime_reuse") != RUNTIME_REUSE:
        raise ValueError("v0.12 runtime receipt binding changed")
    if not all(receipt.get("checks", {}).values()):
        raise ValueError("v0.12 runtime checks failed")
    return receipt


@contextmanager
def _patched_v8() -> Iterator[None]:
    replacements = {
        "LIMITS": LIMITS,
        "PROMPT_CONTRACT": PROMPT_CONTRACT,
        "RUN_REPORT_SCHEMA": RUN_REPORT_SCHEMA,
        "RUN_VERIFICATION_SCHEMA": RUN_VERIFICATION_SCHEMA,
        "chat_completion_payload": chat_completion_payload,
        "full_canary_seed": full_canary_seed,
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


def _seed_resume_artifact(repository_root: Path, artifact_root: Path) -> None:
    private = artifact_root / "private"
    private.mkdir(parents=True, exist_ok=True)
    targets = [private / "candidate-rows.partial.jsonl", private / "generation-telemetry.partial.jsonl", private / "rejection-metadata.partial.jsonl"]
    if any(path.exists() for path in targets):
        return
    rows, telemetry, rejections = _load_and_validate_prefix(repository_root)
    base._write_jsonl(targets[0], rows)
    base._write_jsonl(targets[1], telemetry)
    base._write_jsonl(targets[2], rejections)


def patch_context_regression_gate(plan_path: Path, repository_root: Path) -> dict[str, Any]:
    first = prompt_manifest()[55]
    scenario = next(item for item in base.SCENARIOS if item.task == first["task"] and item.variant == first["variant"])
    literal_scenario = next(item for item in base.SCENARIOS if item.task == "writing_summarization" and item.variant == 4)
    with _patched_v8():
        plan = v8.load_full_canary_plan(plan_path.resolve(), repository_root=repository_root.resolve())
        payload = v8.chat_completion_payload(first["record_id"], scenario, first["language"], attempt_number=1)
        literal_payload = v8.chat_completion_payload("lumi-gptoss-writing_summarization-banglish-04", literal_scenario, "banglish", attempt_number=1)
        manifest = v8.prompt_manifest()
    text = literal_payload["messages"][0]["content"]
    checks = {
        "first_remaining_record_exact": first["record_id"] == "lumi-gptoss-writing_summarization-english-03",
        "manifest_exact": plan["prompt_contract"]["manifest_sha256"] == sha256_value(manifest),
        "patch_context_payload_completed": payload["seed"] == full_canary_seed(first["record_id"], 1),
        "user_prompt_field_instruction_exact": "user_prompt must separately contain" in text,
        "assistant_response_field_instruction_exact": "assistant_response must also separately contain" in text,
        "representative_literals_repeated_for_both_fields": all(text.count(literal) >= 2 for literal in literal_scenario.required_literals),
    }
    if not all(checks.values()):
        raise ValueError("v0.12 patch-context regression gate failed")
    return checks


def build_plan_report(*, plan: Mapping[str, Any], plan_file_sha256: str, repository_root: Path) -> dict[str, Any]:
    resolved = base._resolve_bindings(plan, repository_root.resolve())
    rows, telemetry, rejections = _load_and_validate_prefix(repository_root.resolve())
    checks = patch_context_regression_gate(repository_root.resolve() / str(plan["self_path"]), repository_root.resolve())
    checks.update({
        "accepted_prefix_exact": len(rows) == 55 and len(telemetry) == 55,
        "additional_call_arithmetic_exact": LIMITS["predecessor_retained_prefix_attempts"] + LIMITS["maximum_additional_inference_calls"] == LIMITS["maximum_inference_calls"],
        "additional_output_exact": len(prompt_manifest()[55:]) == 105,
        "additional_retry_arithmetic_exact": len(rejections) + LIMITS["maximum_additional_runtime_retries"] == LIMITS["maximum_runtime_retries"],
        "authorization_pending": not any(bool(value) for value in plan["authorization"].values() if isinstance(value, bool)),
        "owner_acceptance_bound": plan["accepted_repairs"]["bengali_dual_field_v0_11"]["owner_review_status"] == "ACCEPT_dual_field_literal_quality",
        "zero_download_ceiling": LIMITS["maximum_network_bytes"] == 0,
    })
    if not all(checks.values()):
        raise ValueError("v0.12 plan checks failed")
    report: dict[str, Any] = {
        "accepted_repairs": ACCEPTED_REPAIRS,
        "bindings": {name: {"path": plan["bindings"][name]["path"], "sha256": file_sha256(path)} for name, path in sorted(resolved.items())},
        "checks": checks,
        "limits": LIMITS,
        "next_gate": plan["next_gate"],
        "plan_file_sha256": plan_file_sha256,
        "predecessor": PREDECESSOR,
        "prompt_contract": plan["prompt_contract"],
        "schema_version": REPORT_SCHEMA,
        "status": "verified_dual_field_resume_plan_no_execution_authorized",
    }
    report["report_sha256"] = sha256_value(report)
    return report


def build_plan_verification(*, report: Mapping[str, Any], report_file_sha256: str) -> dict[str, Any]:
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256", None)
    checks = {
        "all_plan_checks_pass": all(report.get("checks", {}).values()),
        "full_output_count_exact": report.get("limits", {}).get("exact_final_outputs") == 160,
        "internal_report_hash": internal == sha256_value(unsigned),
        "owner_sample_exact": report.get("limits", {}).get("owner_review_rows") == 32,
        "predecessor_exact": report.get("predecessor") == PREDECESSOR,
        "remaining_output_count_exact": report.get("limits", {}).get("additional_final_outputs") == 105,
        "status_exact": report.get("status") == "verified_dual_field_resume_plan_no_execution_authorized",
    }
    verification: dict[str, Any] = {"checks": checks, "report_file_sha256": report_file_sha256, "report_internal_sha256": internal, "schema_version": VERIFICATION_SCHEMA, "status": "PASS" if all(checks.values()) else "FAIL"}
    verification["verification_sha256"] = sha256_value(verification)
    if verification["status"] != "PASS":
        raise ValueError("v0.12 independent plan verification failed")
    return verification


def run_full_canary_resume(*, plan_path: Path, authorization_path: Path, runtime_reuse_receipt_path: Path, repository_root: Path, artifact_root: Path, ssh_key: Path, known_hosts: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    load_resume_plan(plan_path.resolve(), repository_root=root)
    patch_context_regression_gate(plan_path.resolve(), root)
    load_execution_authorization(authorization_path.resolve(), plan_path=plan_path.resolve())
    load_runtime_reuse_receipt(runtime_reuse_receipt_path.resolve(), authorization_path=authorization_path.resolve())
    _seed_resume_artifact(root, artifact_root.resolve())
    with _patched_v8():
        return v8.run_full_canary(plan_path=plan_path, authorization_path=authorization_path, runtime_reuse_receipt_path=runtime_reuse_receipt_path, repository_root=repository_root, artifact_root=artifact_root, ssh_key=ssh_key, known_hosts=known_hosts)


def verify_full_canary_resume(*, plan_path: Path, authorization_path: Path, runtime_reuse_receipt_path: Path, repository_root: Path, artifact_root: Path) -> dict[str, Any]:
    with _patched_v8():
        return v8.verify_full_canary_run(plan_path=plan_path, authorization_path=authorization_path, runtime_reuse_receipt_path=runtime_reuse_receipt_path, repository_root=repository_root, artifact_root=artifact_root)
