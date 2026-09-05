# ruff: noqa: E501, SLF001
"""Resume the v0.8 full canary from its exact accepted prefix with literal hard-binding."""

from __future__ import annotations

import copy
import hashlib
import json
from contextlib import contextmanager
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import posttraining_gpt_oss_teacher_canary_v8 as v8
from .posttraining_candidate_mix_preflight import file_sha256, sha256_value

PLAN_SCHEMA = "lumi.sft-gpt-oss-teacher-full-canary-resume-plan.v0.9"
REPORT_SCHEMA = "lumi.sft-gpt-oss-teacher-full-canary-resume-report.v0.9"
VERIFICATION_SCHEMA = "lumi.sft-gpt-oss-teacher-full-canary-resume-verification.v0.9"
AUTHORIZATION_SCHEMA = "lumi.sft-gpt-oss-teacher-full-canary-resume-authorization.v0.9"
RUNTIME_REUSE_RECEIPT_SCHEMA = "lumi.sft-gpt-oss-teacher-full-canary-resume-runtime-reuse-receipt.v0.9"
RUN_REPORT_SCHEMA = "lumi.sft-gpt-oss-teacher-full-canary-resume-run-report.v0.9"
RUN_VERIFICATION_SCHEMA = "lumi.sft-gpt-oss-teacher-full-canary-resume-run-verification.v0.9"

PROMPT_CONTRACT = "lumi.gpt-oss-teacher.grounded-literal-resume.v0.9"
SEED_CONTRACT = "sha256(record_id:grounded-literal-resume-attempt:attempt_number)-u32.v0.9"
REASONING_EFFORT = "low"
MAX_TOKENS = 1_536
MAXIMUM_ATTEMPTS_PER_RECORD = 2

SERVER = v8.SERVER
PROHIBITIONS = v8.PROHIBITIONS
RUNTIME_REUSE = v8.RUNTIME_REUSE
RETENTION = v8.RETENTION
FROZEN_MACHINE_GATES = v8.FROZEN_MACHINE_GATES
ACCEPTED_REPAIRS = v8.ACCEPTED_REPAIRS
PREDECESSOR_FAILURE = {
    "accepted_prefix_outputs": 53,
    "authorization_file_sha256": "89ceafa8d4edb830e693ee7925a207b5d192caa4161cc977b0fb166f06906a5c",
    "candidate_prefix_file_sha256": "866a6ec5afea7fbfd23925c1fdad1dd877bb208dc024bdbbdae2722c7be349c9",
    "failed_record_id": "lumi-gptoss-writing_summarization-natural_code_switching-03",
    "failure_report_file_sha256": "ae0352c67eab42de9cc2755c9fda4e6aa078e247933433eabc1669bb0062a0f8",
    "failure_verification_file_sha256": "709a956cdf0a9e8c0add436112d7013eaeddfd73015d383ef74466c5a64880cf",
    "generation_telemetry_file_sha256": "76c87fda02c9ccb693cce736632d24c51d6b5b0bfc86cb1fbeea55781925550b",
    "recovered_rejection_metadata_rows": 4,
    "rejection_metadata_file_sha256": "565149fbaa7e27170c5071cf0e718dd125b778118864a47ca293d9793ce960ef",
    "status": "FAIL_CLOSED_54th_record_attempt_ceiling",
}
LIMITS = {
    "additional_final_outputs": 107,
    "exact_final_outputs": 160,
    "exact_prompt_cells": 16,
    "maximum_additional_generated_tokens_total": 219_648,
    "maximum_additional_inference_calls": 143,
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
    "predecessor_accepted_prefix_outputs": 53,
    "predecessor_retained_prefix_attempts": 57,
    "prompts_per_cell": 10,
}
PROMPT_QUALITY_GATES = dict(v8.PROMPT_QUALITY_GATES)
PROMPT_QUALITY_GATES["all_frozen_required_literals_verbatim"] = True
PROMPT_QUALITY_GATES["predecessor_prefix_revalidated"] = True

EXACT_AUTHORIZATION_TEXT = (
    "I approve one bounded self-hosted Lumi gpt-oss-120b full-canary grounded-literal resume "
    "successor on the existing exact Lambda host 68.209.73.55, bound to ED25519 host-key "
    "fingerprint SHA256:2ZSQqNUYJhWNjihHnqLQ1YNHxoqtMJeckVJRKlc5hs4 and exactly one "
    "NVIDIA H100 80GB GPU GPU-c7e68b23-95b4-37bd-b40b-c41a7ec716f3 reporting at least "
    "81,559 MiB FB total without adding the 480 MiB reserved value twice. Reuse, and do "
    "not redownload, the 25 verified openai/gpt-oss-120b model objects at revision "
    "b5c939de8f754692c1647ca79fbf85e8c1e70f8a bound by model receipt SHA-256 "
    "798914d8532d5b36b61711f34591cabe1e03fa6ac1128fe5e3737c029ead1284, and reuse the "
    "existing vLLM image digest sha256:4d08193d2fd05aadb1b5678f93ae609efb2635df67da45f3efe781c368b34dc8. "
    "Bind the v0.8 failure report SHA-256 ae0352c67eab42de9cc2755c9fda4e6aa078e247933433eabc1669bb0062a0f8 "
    "and failure verification SHA-256 709a956cdf0a9e8c0add436112d7013eaeddfd73015d383ef74466c5a64880cf. "
    "Reuse without regeneration the exact 53 accepted prefix rows bound by SHA-256 "
    "866a6ec5afea7fbfd23925c1fdad1dd877bb208dc024bdbbdae2722c7be349c9 and their exact "
    "53-row telemetry bound by SHA-256 76c87fda02c9ccb693cce736632d24c51d6b5b0bfc86cb1fbeea55781925550b; "
    "carry forward only the four recovered-record rejection metadata entries and exclude "
    "both failed-record attempts. Generate only the remaining 107 records, beginning with "
    "lumi-gptoss-writing_summarization-natural_code_switching-03, under prompt contract "
    "lumi.gpt-oss-teacher.grounded-literal-resume.v0.9, reasoning effort low, a 1,536 "
    "generated-token ceiling per call, distinct attempt-bound deterministic seeds, and at "
    "most 2 new attempts per remaining record. Preserve all v0.8 complete-row gates and "
    "require every frozen required literal to appear verbatim in the generated row; for the "
    "failed outage-summary record this includes exact 14:20, API gateway, and 18 minutes. "
    "Retain rejected new attempts only as validation error, non-sensitive finish_reason, "
    "usage token counts, and request/response-envelope hashes without raw candidate or "
    "reasoning text. Maximum 90 additional instance-minutes / $6.44 before tax / 0 "
    "additional model or runtime download bytes / 16 GiB additional private writes / 143 "
    "additional sequential inference calls / 36 additional retry calls / 107 additional "
    "final outputs, while the combined retained-prefix plus resume audit remains capped at "
    "200 calls and 40 retries. It may restart and then cleanly stop only the bound canary "
    "container, serve only on remote localhost through pinned SSH, retain only accepted "
    "final-channel candidate text, run the full frozen 16-cell/160-row machine gate, and "
    "prepare the deterministic 32-row owner-review packet. No external API, public port, "
    "model or runtime download, registry change, dataset seal, SFT training, production "
    "promotion, unrelated access, new instance, or termination."
)

TOOLING_PATHS = {
    "implementation": "src/lumi/posttraining_gpt_oss_teacher_canary_v9.py",
    "plan_runner": "scripts/plan_sft_gpt_oss_teacher_canary_v9.py",
    "plan_verifier": "scripts/verify_sft_gpt_oss_teacher_canary_v9_plan.py",
    "run_runner": "scripts/run_sft_gpt_oss_teacher_canary_v9.py",
    "run_verifier": "scripts/verify_sft_gpt_oss_teacher_canary_v9.py",
    "test": "tests/test_gpt_oss_teacher_canary_v9.py",
}


def full_canary_seed(record_id: str, attempt_number: int) -> int:
    if not 1 <= attempt_number <= MAXIMUM_ATTEMPTS_PER_RECORD:
        raise ValueError("v0.9 attempt number is outside the reviewed contract")
    material = f"{record_id}:grounded-literal-resume-attempt:{attempt_number}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def generation_messages(scenario: v8.v5.v2.predecessor.base.Scenario, language: str, *, attempt_number: int) -> list[dict[str, str]]:
    messages = copy.deepcopy(v8.generation_messages(scenario, language, attempt_number=attempt_number))
    if scenario.required_literals:
        literals = " | ".join(scenario.required_literals)
        messages[0]["content"] += (
            " The generated row must contain every following required literal verbatim, "
            f"without translation, paraphrase, deletion, or reformatting: {literals}. "
            "Silently check each literal before returning the JSON object."
        )
    return messages


def chat_completion_payload(record_id: str, scenario: v8.v5.v2.predecessor.base.Scenario, language: str, *, attempt_number: int) -> dict[str, Any]:
    payload = copy.deepcopy(v8.chat_completion_payload(record_id, scenario, language, attempt_number=attempt_number))
    payload["messages"] = generation_messages(scenario, language, attempt_number=attempt_number)
    payload["seed"] = full_canary_seed(record_id, attempt_number)
    payload["temperature"] = 0.2
    return payload


def prompt_manifest() -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for scenario in v8.v5.v2.predecessor.SCENARIOS:
        for language in v8.v5.v2.predecessor.LANGUAGE_MODES:
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


def validate_prompt_quality(messages: Sequence[Mapping[str, str]], *, scenario: v8.v5.v2.predecessor.base.Scenario, language: str) -> None:
    v8.validate_prompt_quality(messages, scenario=scenario, language=language)
    combined = "\n".join(str(message.get("content", "")) for message in messages)
    for literal in scenario.required_literals:
        if literal not in combined:
            raise ValueError(f"v0.9 required grounded literal missing: {literal}")


def _predecessor_paths(repository_root: Path) -> tuple[Path, Path, Path]:
    base = repository_root / "artifacts/lumi-gpt-oss-teacher-canary-v0.8/b56e27a0024a/attempt-001/private"
    return base / "candidate-rows.partial.jsonl", base / "generation-telemetry.partial.jsonl", base / "rejection-metadata.partial.jsonl"


def _load_and_validate_prefix(repository_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows_path, telemetry_path, rejection_path = _predecessor_paths(repository_root)
    rows = v8.v5.v2.predecessor._read_jsonl(rows_path)
    telemetry = v8.v5.v2.predecessor._read_jsonl(telemetry_path)
    all_rejections = v8.v5.v2.predecessor._read_jsonl(rejection_path)
    if len(rows) != 53 or len(telemetry) != 53:
        raise ValueError("v0.9 predecessor prefix count changed")
    if file_sha256(rows_path) != PREDECESSOR_FAILURE["candidate_prefix_file_sha256"] or file_sha256(telemetry_path) != PREDECESSOR_FAILURE["generation_telemetry_file_sha256"] or file_sha256(rejection_path) != PREDECESSOR_FAILURE["rejection_metadata_file_sha256"]:
        raise ValueError("v0.9 predecessor prefix hash changed")
    expected = v8.v5.v2.predecessor.expected_record_ids()
    if [row["record_id"] for row in rows] != list(expected[:53]):
        raise ValueError("v0.9 predecessor row prefix changed")
    recovered_rejections = [item for item in all_rejections if item["record_id"] in set(expected[:53])]
    if len(recovered_rejections) != PREDECESSOR_FAILURE["recovered_rejection_metadata_rows"]:
        raise ValueError("v0.9 recovered rejection metadata count changed")
    scenarios = {(item.task, item.variant): item for item in v8.v5.v2.predecessor.SCENARIOS}
    for row in rows:
        variant = int(str(row["record_id"]).rsplit("-", 1)[1])
        validate_prompt_quality(row["messages"], scenario=scenarios[(row["task"], variant)], language=row["language_mode"])
    return rows, telemetry, recovered_rejections


def load_resume_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    plan = v8.v5.v2.predecessor._load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("status") != "full_canary_resume_plan_pending_explicit_execution_authorization":
        raise ValueError("unsupported or changed v0.9 resume plan")
    if plan.get("server") != SERVER or plan.get("runtime_reuse") != RUNTIME_REUSE or plan.get("accepted_repairs") != ACCEPTED_REPAIRS:
        raise ValueError("v0.9 host, runtime, or accepted repair binding changed")
    if plan.get("predecessor_failure") != PREDECESSOR_FAILURE or plan.get("limits") != LIMITS:
        raise ValueError("v0.9 predecessor failure or limits changed")
    if plan.get("frozen_machine_gates") != FROZEN_MACHINE_GATES or plan.get("prompt_quality_gates") != PROMPT_QUALITY_GATES:
        raise ValueError("v0.9 validation gates changed")
    if plan.get("prohibitions") != PROHIBITIONS or plan.get("retention") != RETENTION:
        raise ValueError("v0.9 prohibition or retention changed")
    if any(bool(value) for value in plan.get("authorization", {}).values() if isinstance(value, bool)):
        raise ValueError("v0.9 plan unexpectedly authorizes execution")
    if plan.get("next_gate", {}).get("exact_authorization_text") != EXACT_AUTHORIZATION_TEXT:
        raise ValueError("v0.9 authorization text changed")
    contract = plan.get("prompt_contract", {})
    manifest = prompt_manifest()
    if contract.get("id") != PROMPT_CONTRACT or contract.get("seed_contract") != SEED_CONTRACT or contract.get("manifest_sha256") != sha256_value(manifest):
        raise ValueError("v0.9 prompt contract changed")
    v8.v5.v2.predecessor._resolve_bindings(plan, repository_root)
    _load_and_validate_prefix(Path(repository_root).resolve())
    return plan


def build_plan_report(*, plan: Mapping[str, Any], plan_file_sha256: str, repository_root: str | Path) -> dict[str, Any]:
    resolved = v8.v5.v2.predecessor._resolve_bindings(plan, repository_root)
    rows, telemetry, recovered_rejections = _load_and_validate_prefix(Path(repository_root).resolve())
    manifest = prompt_manifest()
    suffix = manifest[len(rows):]
    attempts = [attempt for item in suffix for attempt in item["attempts"]]
    report: dict[str, Any] = {
        "accepted_repairs": plan["accepted_repairs"],
        "authorization": plan["authorization"],
        "bindings": {name: {"path": plan["bindings"][name]["path"], "sha256": file_sha256(path)} for name, path in sorted(resolved.items())},
        "checks": {
            "accepted_repairs_bound": plan["accepted_repairs"] == ACCEPTED_REPAIRS,
            "additional_call_ceiling_arithmetic": LIMITS["predecessor_retained_prefix_attempts"] + LIMITS["maximum_additional_inference_calls"] == LIMITS["maximum_inference_calls"],
            "additional_output_count_exact": len(suffix) == LIMITS["additional_final_outputs"],
            "additional_retry_ceiling_arithmetic": len(recovered_rejections) + LIMITS["maximum_additional_runtime_retries"] == LIMITS["maximum_runtime_retries"],
            "attempt_bound_requests_distinct": len({item["request_sha256"] for item in attempts}) == 214,
            "exact_prefix_revalidated": len(rows) == 53 and len(telemetry) == 53,
            "failed_record_is_first_resume_record": suffix[0]["record_id"] == PREDECESSOR_FAILURE["failed_record_id"],
            "grounded_literals_hard_bound": "14:20 | API gateway | 18 minutes" in generation_messages(v8.v5.v2.predecessor.SCENARIOS[13], "natural_code_switching", attempt_number=1)[0]["content"],
            "no_execution_authorized": not any(bool(value) for value in plan["authorization"].values() if isinstance(value, bool)),
            "no_model_or_runtime_download": plan["limits"]["maximum_network_bytes"] == 0,
            "predecessor_failure_bound": plan["predecessor_failure"] == PREDECESSOR_FAILURE,
        },
        "frozen_machine_gates": plan["frozen_machine_gates"],
        "limits": plan["limits"],
        "next_gate": plan["next_gate"],
        "observed_at": plan["observed_at"],
        "plan_file_sha256": plan_file_sha256,
        "predecessor_failure": plan["predecessor_failure"],
        "prohibitions": plan["prohibitions"],
        "prompt_contract": plan["prompt_contract"],
        "prompt_quality_gates": plan["prompt_quality_gates"],
        "retention": plan["retention"],
        "runtime_reuse": plan["runtime_reuse"],
        "schema_version": REPORT_SCHEMA,
        "server": plan["server"],
        "status": "verified_full_canary_resume_plan_no_execution_authorized",
        "tooling": {name: {"path": tooling_path, "sha256": file_sha256(Path(repository_root) / tooling_path)} for name, tooling_path in sorted(TOOLING_PATHS.items())},
    }
    if not all(report["checks"].values()):
        raise ValueError("v0.9 resume plan checks failed")
    report["report_sha256"] = sha256_value(report)
    return report


def build_plan_verification(*, report: Mapping[str, Any], report_file_sha256: str) -> dict[str, Any]:
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256", None)
    if internal != sha256_value(unsigned):
        raise ValueError("v0.9 report internal hash changed")
    checks = {
        "all_plan_checks_pass": all(report.get("checks", {}).values()),
        "authorization_pending": report.get("status") == "verified_full_canary_resume_plan_no_execution_authorized",
        "combined_call_ceiling_exact": report.get("limits", {}).get("maximum_inference_calls") == 200,
        "full_output_count_exact": report.get("limits", {}).get("exact_final_outputs") == 160,
        "grounded_literal_gate_exact": report.get("prompt_quality_gates", {}).get("all_frozen_required_literals_verbatim") is True,
        "owner_sample_exact": report.get("limits", {}).get("owner_review_rows") == 32,
        "predecessor_failure_exact": report.get("predecessor_failure") == PREDECESSOR_FAILURE,
        "resume_output_count_exact": report.get("limits", {}).get("additional_final_outputs") == 107,
        "zero_download_ceiling": report.get("limits", {}).get("maximum_network_bytes") == 0,
    }
    if not all(checks.values()):
        raise ValueError("v0.9 independent plan verification failed")
    verification: dict[str, Any] = {"checks": checks, "report_file_sha256": report_file_sha256, "report_internal_sha256": internal, "schema_version": VERIFICATION_SCHEMA, "status": "PASS"}
    verification["verification_sha256"] = sha256_value(verification)
    return verification


def load_execution_authorization(path: str | Path, *, plan_path: str | Path) -> dict[str, Any]:
    authorization = v8.v5.v2.predecessor._load_json(path)
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA or authorization.get("status") != "authorized_exact_host_bound_grounded_literal_resume":
        raise ValueError("unsupported or changed v0.9 authorization")
    if authorization.get("approval_text") != EXACT_AUTHORIZATION_TEXT or authorization.get("plan_file_sha256") != file_sha256(plan_path):
        raise ValueError("v0.9 approval or plan binding changed")
    if authorization.get("server") != SERVER or authorization.get("limits") != LIMITS or authorization.get("prohibitions") != PROHIBITIONS or authorization.get("predecessor_failure") != PREDECESSOR_FAILURE:
        raise ValueError("v0.9 authorization bindings changed")
    started = authorization.get("authorized_start_unix")
    if not isinstance(started, int) or started <= 0:
        raise ValueError("v0.9 authorization start time is invalid")
    return authorization


def load_runtime_reuse_receipt(path: str | Path, *, authorization_path: str | Path) -> dict[str, Any]:
    receipt = v8.v5.v2.predecessor._load_json(path)
    if receipt.get("schema_version") != RUNTIME_REUSE_RECEIPT_SCHEMA or receipt.get("status") != "PASS_exact_existing_localhost_runtime_reused_for_full_canary_resume_v0.9":
        raise ValueError("unsupported or failed v0.9 runtime receipt")
    if receipt.get("authorization_file_sha256") != file_sha256(authorization_path) or receipt.get("server") != SERVER or receipt.get("runtime_reuse") != RUNTIME_REUSE:
        raise ValueError("v0.9 runtime receipt binding changed")
    if not all(receipt.get("checks", {}).values()):
        raise ValueError("v0.9 runtime checks failed")
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
    rows, telemetry, recovered_rejections = _load_and_validate_prefix(repository_root)
    v8.v5.v2.predecessor._write_jsonl(targets[0], rows)
    v8.v5.v2.predecessor._write_jsonl(targets[1], telemetry)
    v8.v5.v2.predecessor._write_jsonl(targets[2], recovered_rejections)


def run_full_canary_resume(*, plan_path: Path, authorization_path: Path, runtime_reuse_receipt_path: Path, repository_root: Path, artifact_root: Path, ssh_key: Path, known_hosts: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    load_resume_plan(plan_path.resolve(), repository_root=root)
    load_execution_authorization(authorization_path.resolve(), plan_path=plan_path.resolve())
    load_runtime_reuse_receipt(runtime_reuse_receipt_path.resolve(), authorization_path=authorization_path.resolve())
    _seed_resume_artifact(root, artifact_root.resolve())
    with _patched_v8():
        return v8.run_full_canary(plan_path=plan_path, authorization_path=authorization_path, runtime_reuse_receipt_path=runtime_reuse_receipt_path, repository_root=repository_root, artifact_root=artifact_root, ssh_key=ssh_key, known_hosts=known_hosts)


def verify_full_canary_resume(*, plan_path: Path, authorization_path: Path, runtime_reuse_receipt_path: Path, repository_root: Path, artifact_root: Path) -> dict[str, Any]:
    with _patched_v8():
        return v8.verify_full_canary_run(plan_path=plan_path, authorization_path=authorization_path, runtime_reuse_receipt_path=runtime_reuse_receipt_path, repository_root=repository_root, artifact_root=artifact_root)
