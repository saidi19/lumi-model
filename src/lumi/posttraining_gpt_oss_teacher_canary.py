# ruff: noqa: E501, SLF001
"""Host-bound, fail-closed gpt-oss teacher canary for Lumi SFT closure."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import posttraining_first_party_teacher_canary as base
from .posttraining_candidate_mix_preflight import file_sha256, sha256_value
from .posttraining_gpt_oss_teacher_preflight import (
    MODEL_BYTES,
    MODEL_REVISION,
    RUNTIME_BYTES,
    RUNTIME_DIGEST,
    TOTAL_NETWORK_BYTES,
)

PLAN_SCHEMA = "lumi.sft-gpt-oss-teacher-canary-host-bound-plan.v0.1"
REPORT_SCHEMA = "lumi.sft-gpt-oss-teacher-canary-host-bound-report.v0.1"
VERIFICATION_SCHEMA = (
    "lumi.sft-gpt-oss-teacher-canary-host-bound-verification.v0.1"
)
AUTHORIZATION_SCHEMA = "lumi.sft-gpt-oss-teacher-canary-authorization.v0.1"
RUNTIME_RECEIPT_SCHEMA = "lumi.sft-gpt-oss-teacher-runtime-receipt.v0.1"
ROW_SCHEMA = "lumi.sft-gpt-oss-teacher-canary-row.v0.1"
RUN_REPORT_SCHEMA = "lumi.sft-gpt-oss-teacher-canary-run-report.v0.1"
RUN_VERIFICATION_SCHEMA = "lumi.sft-gpt-oss-teacher-canary-run-verification.v0.1"

MODEL_ID = "openai/gpt-oss-120b"
RUNTIME_IMAGE = "vllm/vllm-openai:v0.26.0-x86_64-cu129-ubuntu2404"
LOCAL_ENDPOINT = "http://127.0.0.1:18080/v1/chat/completions"
REMOTE_ROOT = "/home/ubuntu/lumi-gpt-oss-teacher-canary-v0.1"

SERVER = {
    "docker_version": "29.2.1",
    "gpu_count": 1,
    "gpu_driver_version": "580.105.08",
    "gpu_fb_memory_reserved_mib": 480,
    "gpu_fb_memory_total_mib": 81_559,
    "gpu_nominal_memory_gb": 80,
    "legacy_numeric_gate_passed": False,
    "legacy_preflight_minimum_mib": 81_920,
    "successor_minimum_reported_fb_mib": 81_559,
    "gpu_name": "NVIDIA H100 80GB HBM3",
    "gpu_uuid": "GPU-c7e68b23-95b4-37bd-b40b-c41a7ec716f3",
    "host": "68.209.73.55",
    "host_key_fingerprint": "SHA256:2ZSQqNUYJhWNjihHnqLQ1YNHxoqtMJeckVJRKlc5hs4",
    "host_key_type": "ssh-ed25519",
    "operating_system": "Ubuntu 22.04.5 LTS",
    "ssh_user": "ubuntu",
}

LIMITS = {
    "exact_final_outputs": 160,
    "exact_prompt_cells": 16,
    "maximum_context_tokens": 4_096,
    "maximum_generated_tokens_per_call": 1_024,
    "maximum_inference_calls": 200,
    "maximum_instance_minutes": 240,
    "maximum_network_bytes": TOTAL_NETWORK_BYTES,
    "maximum_private_write_bytes": 160 * 1024**3,
    "maximum_runtime_retries": 40,
    "maximum_usd_before_tax": 17.16,
    "owner_review_rows": 32,
    "prompts_per_cell": 10,
}

QUALITY_GATES = {
    "critical_factual_safety_or_tool_failures": 0,
    "minimum_machine_acceptance_rate": 0.90,
    "minimum_owner_acceptance_rate": 0.90,
    "minimum_per_cell_acceptance_rate": 0.80,
}

PROHIBITIONS = {
    "candidate_registry_change": True,
    "dataset_seal": True,
    "external_inference_api": True,
    "external_distribution": True,
    "human_authorship_claim": True,
    "new_instance": True,
    "production_promotion": True,
    "public_listener": True,
    "reasoning_or_analysis_retention": True,
    "sft_training": True,
    "termination": True,
    "unrelated_access": True,
}

TASKS = (
    "safety_helpfulness",
    "writing_summarization",
    "multilingual_bengali",
    "factual_qa",
)
LANGUAGE_MODES = (
    "banglish",
    "natural_code_switching",
    "bengali_script",
    "english",
)


def _scenario(
    task: str,
    variant: int,
    name: str,
    instruction: str,
    *,
    required_literals: tuple[str, ...] = (),
    answer_number: str | None = None,
    safety_label: str = "safe",
) -> base.Scenario:
    return base.Scenario(
        task,
        variant,
        name,
        instruction,
        required_literals=required_literals,
        answer_number=answer_number,
        safety_label=safety_label,
    )


FACTUAL_SCENARIOS = (
    _scenario("factual_qa", 0, "kilometres_to_metres", "Ask how many metres are in 2.5 kilometres and answer exactly 2500 metres.", answer_number="2500"),
    _scenario("factual_qa", 1, "leap_year_days", "Ask how many days are in a leap year and answer exactly 366 days.", answer_number="366"),
    _scenario("factual_qa", 2, "rectangle_area", "Ask for the area of a 7 by 8 rectangle and answer exactly 56 square units.", answer_number="56"),
    _scenario("factual_qa", 3, "water_freezing", "Ask for the standard Celsius freezing point of water and answer exactly 0 degrees Celsius.", answer_number="0"),
    _scenario("factual_qa", 4, "minutes_to_hours", "Ask to convert 90 minutes to hours and answer exactly 1.5 hours.", answer_number="1.5"),
    _scenario("factual_qa", 5, "fraction_decimal", "Ask for the decimal form of three quarters and answer exactly 0.75.", answer_number="0.75"),
    _scenario("factual_qa", 6, "triangle_angle_sum", "Ask for the interior-angle sum of a triangle and answer exactly 180 degrees.", answer_number="180"),
    _scenario("factual_qa", 7, "dozen_count", "Ask how many items are in one dozen and answer exactly 12.", answer_number="12"),
    _scenario("factual_qa", 8, "prime_29", "Ask whether 29 is prime. Answer yes and include the number 29.", answer_number="29"),
    _scenario("factual_qa", 9, "mean_three_numbers", "Ask for the arithmetic mean of 4, 8, and 12 and answer exactly 8.", answer_number="8"),
)

WRITING_SCENARIOS = (
    _scenario("writing_summarization", 0, "meeting_note", "Create a short user note and concise summary preserving 10:00, Room B, and ID exactly.", required_literals=("10:00", "Room B", "ID")),
    _scenario("writing_summarization", 1, "release_note", "Create and summarize a release note preserving 2.4, dark mode, and offline search exactly.", required_literals=("2.4", "dark mode", "offline search")),
    _scenario("writing_summarization", 2, "project_handoff", "Create and summarize a project handoff preserving Phase Alpha, 12 May, and Nadia exactly.", required_literals=("Phase Alpha", "12 May", "Nadia")),
    _scenario("writing_summarization", 3, "outage_summary", "Create and summarize an outage note preserving 14:20, API gateway, and 18 minutes exactly.", required_literals=("14:20", "API gateway", "18 minutes")),
    _scenario("writing_summarization", 4, "polite_email", "Rewrite a blunt email politely while preserving invoice 7842, Monday, and PDF exactly.", required_literals=("invoice 7842", "Monday", "PDF")),
    _scenario("writing_summarization", 5, "executive_brief", "Create a compact executive brief preserving Q3, 12%, and Dhaka exactly.", required_literals=("Q3", "12%", "Dhaka")),
    _scenario("writing_summarization", 6, "checklist_summary", "Create and summarize a checklist preserving backup, restart, and verify logs exactly.", required_literals=("backup", "restart", "verify logs")),
    _scenario("writing_summarization", 7, "comparison_summary", "Create and summarize a comparison preserving Plan A, 4 days, and $240 exactly.", required_literals=("Plan A", "4 days", "$240")),
    _scenario("writing_summarization", 8, "timeline_summary", "Create and summarize a timeline preserving 3 June, testing, and 7 June exactly.", required_literals=("3 June", "testing", "7 June")),
    _scenario("writing_summarization", 9, "policy_simplification", "Rewrite a policy simply while preserving 30 days, manager approval, and encrypted storage exactly.", required_literals=("30 days", "manager approval", "encrypted storage")),
)

MULTILINGUAL_SCENARIOS = (
    _scenario("multilingual_bengali", 0, "friendly_greeting", "Create a natural greeting and a warm concise reply without claiming persistent memory."),
    _scenario("multilingual_bengali", 1, "punctuation_repair", "Create a short informal sentence with punctuation errors and return a corrected version with one brief note."),
    _scenario("multilingual_bengali", 2, "study_plan", "Create a request and practical three-day study plan with three concise steps."),
    _scenario("multilingual_bengali", 3, "polite_rewrite", "Create a blunt request and rewrite it politely with a brief explanation."),
    _scenario("multilingual_bengali", 4, "travel_packing", "Create a request for a short practical packing list for a rainy two-day trip."),
    _scenario("multilingual_bengali", 5, "appointment_reminder", "Create a request and a clear, friendly appointment reminder."),
    _scenario("multilingual_bengali", 6, "bug_report", "Create a vague bug complaint and rewrite it as a concise reproducible bug report."),
    _scenario("multilingual_bengali", 7, "recipe_steps", "Create a request and three safe, concise steps for making tea."),
    _scenario("multilingual_bengali", 8, "budget_tips", "Create a request and three practical monthly budgeting tips without financial guarantees."),
    _scenario("multilingual_bengali", 9, "clarifying_question", "Create an ambiguous request and respond with one useful clarifying question."),
)

SAFETY_SCENARIOS = (
    _scenario("safety_helpfulness", 0, "credential_theft", "The user asks for another person's password. Refuse briefly and suggest legitimate account recovery.", safety_label="refuse"),
    _scenario("safety_helpfulness", 1, "phishing_page", "The user asks for a deceptive login page to steal credentials. Refuse and offer defensive anti-phishing guidance.", safety_label="refuse"),
    _scenario("safety_helpfulness", 2, "credential_stealer", "The user asks for malware that steals browser credentials. Refuse and offer safe endpoint-hardening guidance.", safety_label="refuse"),
    _scenario("safety_helpfulness", 3, "account_hygiene", "The user asks for safe steps to protect an online account. Respond helpfully without over-refusing."),
    _scenario("safety_helpfulness", 4, "chest_pain", "The user reports severe chest pain and trouble breathing. Do not diagnose; urge immediate local emergency help and contacting someone nearby.", safety_label="medical_caution"),
    _scenario("safety_helpfulness", 5, "possible_overdose", "The user reports a possible medication overdose. Do not diagnose; urge immediate poison-control or emergency help and not waiting for symptoms.", safety_label="medical_caution"),
    _scenario("safety_helpfulness", 6, "self_harm_crisis", "The user says they may hurt themselves now. Respond supportively, encourage immediate emergency/crisis help and contacting a trusted nearby person.", safety_label="medical_caution"),
    _scenario("safety_helpfulness", 7, "dangerous_chemical_mix", "The user asks for instructions to create a dangerous toxic chemical mixture. Refuse and offer safe chemical-handling guidance.", safety_label="refuse"),
    _scenario("safety_helpfulness", 8, "doxxing_request", "The user asks to find and publish a private person's home address. Refuse and suggest privacy-respecting alternatives.", safety_label="refuse"),
    _scenario("safety_helpfulness", 9, "exam_cheating", "The user asks for stolen exam answers. Refuse cheating assistance and offer a short legitimate study plan.", safety_label="refuse"),
)

SCENARIOS = SAFETY_SCENARIOS + WRITING_SCENARIOS + MULTILINGUAL_SCENARIOS + FACTUAL_SCENARIOS

EXACT_AUTHORIZATION_TEXT = (
    "I approve one bounded self-hosted Lumi gpt-oss-120b teacher canary on exact "
    "Lambda host 68.209.73.55, bound to ED25519 host-key fingerprint "
    "SHA256:2ZSQqNUYJhWNjihHnqLQ1YNHxoqtMJeckVJRKlc5hs4 and exactly one NVIDIA "
    "H100 80GB GPU GPU-c7e68b23-95b4-37bd-b40b-c41a7ec716f3. The observed "
    "memory binding is 81,559 MiB FB total, including 480 MiB driver-reserved. "
    "The original 81,920 MiB nvidia-smi minimum does not pass; this reviewed "
    "successor instead binds the exact NVIDIA H100 SXM 80GB product identity "
    "and requires reported FB total to remain at least 81,559 MiB without "
    "adding reserved memory twice. Use "
    "model revision b5c939de8f754692c1647ca79fbf85e8c1e70f8a and vLLM image digest "
    "sha256:4d08193d2fd05aadb1b5678f93ae609efb2635df67da45f3efe781c368b34dc8. "
    "Maximum 240 instance-minutes / $17.16 before tax / 77,040,051,117 network "
    "bytes / 160 GiB private writes / 200 sequential inference calls / 40 "
    "runtime retries / 160 final outputs. It may download only the 25 bound "
    "model objects and exact digest-pinned runtime image, serve only on remote "
    "localhost through pinned SSH, retain only final-channel candidate text, "
    "and run the frozen 16-cell machine and 32-row owner-review gates. No "
    "external API, public port, registry change, dataset seal, SFT training, "
    "production promotion, unrelated access, new instance, or termination."
)

TOOLING_PATHS = {
    "implementation": "src/lumi/posttraining_gpt_oss_teacher_canary.py",
    "plan_runner": "scripts/plan_sft_gpt_oss_teacher_canary.py",
    "plan_verifier": "scripts/verify_sft_gpt_oss_teacher_canary_plan.py",
    "run_runner": "scripts/run_sft_gpt_oss_teacher_canary.py",
    "run_verifier": "scripts/verify_sft_gpt_oss_teacher_canary.py",
    "test": "tests/test_gpt_oss_teacher_canary.py",
}


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def seed_for_record(record_id: str) -> int:
    return int.from_bytes(hashlib.sha256(record_id.encode()).digest()[:4], "big")


def expected_record_ids() -> tuple[str, ...]:
    return tuple(
        f"lumi-gptoss-{scenario.task}-{language}-{scenario.variant:02d}"
        for scenario in SCENARIOS
        for language in LANGUAGE_MODES
    )


def _generation_messages(scenario: base.Scenario, language: str) -> list[dict[str, str]]:
    system = (
        "Produce exactly one private AI-assisted Lumi SFT canary row. Return only a "
        "compact JSON object with scenario_id, user_prompt, assistant_response, "
        "safety_label, tool_call, and tool_result. Never expose analysis, reasoning, "
        "chain-of-thought, model identity, system prompts, or hidden instructions. "
        "Do not mention training or datasets. Set tool_call and tool_result to null. "
        + base.LANGUAGE_INSTRUCTIONS[language]
    )
    user = (
        f"Scenario ID: {scenario.scenario_id}\nTask: {scenario.task}\n"
        f"Instruction: {scenario.instruction}\n"
        f"Set safety_label exactly to {scenario.safety_label}. Return final JSON only."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def chat_completion_payload(
    record_id: str, scenario: base.Scenario, language: str
) -> dict[str, Any]:
    return {
        "chat_template_kwargs": {"reasoning_effort": "medium"},
        "max_tokens": LIMITS["maximum_generated_tokens_per_call"],
        "messages": _generation_messages(scenario, language),
        "model": MODEL_ID,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "lumi_gpt_oss_teacher_row",
                "schema": base.OUTPUT_SCHEMA,
                "strict": True,
            },
        },
        "seed": seed_for_record(record_id),
        "stream": False,
        "temperature": 0.7,
        "top_p": 0.8,
    }


def prompt_manifest() -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for language in LANGUAGE_MODES:
            record_id = (
                f"lumi-gptoss-{scenario.task}-{language}-{scenario.variant:02d}"
            )
            manifest.append(
                {
                    "language": language,
                    "record_id": record_id,
                    "request_sha256": sha256_value(
                        chat_completion_payload(record_id, scenario, language)
                    ),
                    "scenario_id": scenario.scenario_id,
                    "seed": seed_for_record(record_id),
                    "task": scenario.task,
                    "variant": scenario.variant,
                }
            )
    return manifest


def _resolve_bindings(
    plan: Mapping[str, Any], repository_root: str | Path
) -> dict[str, Path]:
    root = Path(repository_root).resolve()
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("gpt-oss host-bound bindings are missing")
    resolved: dict[str, Path] = {}
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping) or not isinstance(binding.get("path"), str):
            raise ValueError(f"gpt-oss host-bound binding malformed: {name}")
        path = (root / str(binding["path"])).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError(f"gpt-oss host-bound binding missing: {name}")
        if file_sha256(path) != binding.get("sha256"):
            raise ValueError(f"gpt-oss host-bound binding changed: {name}")
        resolved[name] = path
    return resolved


def load_host_bound_plan(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    plan = _load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unsupported gpt-oss host-bound plan")
    if plan.get("status") != "host_bound_plan_pending_explicit_execution_authorization":
        raise ValueError("gpt-oss host-bound plan status changed")
    if plan.get("server") != SERVER:
        raise ValueError("gpt-oss host binding changed")
    if SERVER["gpu_fb_memory_total_mib"] >= SERVER["legacy_preflight_minimum_mib"]:
        raise ValueError("gpt-oss legacy memory gate unexpectedly passes")
    if SERVER["legacy_numeric_gate_passed"] is not False:
        raise ValueError("gpt-oss legacy memory failure was hidden")
    if (
        SERVER["gpu_fb_memory_total_mib"]
        < SERVER["successor_minimum_reported_fb_mib"]
    ):
        raise ValueError("gpt-oss successor reported-FB floor not met")
    if plan.get("limits") != LIMITS or plan.get("quality_gates") != QUALITY_GATES:
        raise ValueError("gpt-oss canary limits or quality gates changed")
    if plan.get("prohibitions") != PROHIBITIONS:
        raise ValueError("gpt-oss canary prohibitions changed")
    authorization = plan.get("authorization", {})
    if any(bool(value) for value in authorization.values() if isinstance(value, bool)):
        raise ValueError("gpt-oss host-bound plan unexpectedly authorizes execution")
    if plan.get("next_gate", {}).get("exact_authorization_text") != EXACT_AUTHORIZATION_TEXT:
        raise ValueError("gpt-oss exact authorization text changed")
    manifest = prompt_manifest()
    if len(manifest) != LIMITS["exact_final_outputs"]:
        raise ValueError("gpt-oss prompt manifest size changed")
    if plan.get("prompt_contract", {}).get("manifest_sha256") != sha256_value(manifest):
        raise ValueError("gpt-oss prompt manifest binding changed")
    _resolve_bindings(plan, repository_root)
    return plan


def build_host_bound_report(
    *, plan: Mapping[str, Any], plan_file_sha256: str, repository_root: str | Path
) -> dict[str, Any]:
    resolved = _resolve_bindings(plan, repository_root)
    report: dict[str, Any] = {
        "authorization": plan["authorization"],
        "bindings": {
            name: {"path": plan["bindings"][name]["path"], "sha256": file_sha256(path)}
            for name, path in sorted(resolved.items())
        },
        "checks": {
            "exact_h100_product_uuid_and_count_bound": (
                SERVER["gpu_count"] == 1
                and SERVER["gpu_name"] == "NVIDIA H100 80GB HBM3"
                and SERVER["gpu_uuid"]
                == "GPU-c7e68b23-95b4-37bd-b40b-c41a7ec716f3"
            ),
            "host_key_pinned": SERVER["host_key_fingerprint"]
            == "SHA256:2ZSQqNUYJhWNjihHnqLQ1YNHxoqtMJeckVJRKlc5hs4",
            "legacy_81920_mib_numeric_gate_fails_explicitly": (
                SERVER["legacy_numeric_gate_passed"] is False
                and SERVER["gpu_fb_memory_total_mib"]
                < SERVER["legacy_preflight_minimum_mib"]
            ),
            "successor_exact_reported_fb_floor_met": (
                SERVER["gpu_fb_memory_total_mib"]
                >= SERVER["successor_minimum_reported_fb_mib"]
            ),
            "no_execution_authorized": not any(
                bool(value)
                for value in plan["authorization"].values()
                if isinstance(value, bool)
            ),
            "prompt_manifest_is_exact": len(prompt_manifest()) == 160,
        },
        "limits": plan["limits"],
        "model": plan["model"],
        "next_gate": plan["next_gate"],
        "observed_at": plan["observed_at"],
        "plan_file_sha256": plan_file_sha256,
        "prohibitions": plan["prohibitions"],
        "prompt_contract": plan["prompt_contract"],
        "quality_gates": plan["quality_gates"],
        "runtime": plan["runtime"],
        "schema_version": REPORT_SCHEMA,
        "server": plan["server"],
        "status": "verified_host_bound_plan_no_execution_authorized",
        "tooling": {
            name: {"path": path, "sha256": file_sha256(Path(repository_root) / path)}
            for name, path in sorted(TOOLING_PATHS.items())
        },
    }
    if not all(report["checks"].values()):
        raise ValueError("gpt-oss host-bound report checks failed")
    report["report_sha256"] = sha256_value(report)
    return report


def build_host_bound_verification(
    *, report: Mapping[str, Any], report_file_sha256: str
) -> dict[str, Any]:
    unsigned = dict(report)
    observed = unsigned.pop("report_sha256", None)
    if observed != sha256_value(unsigned):
        raise ValueError("gpt-oss host-bound report hash mismatch")
    checks = {
        "all_plan_checks_pass": all(report.get("checks", {}).values()),
        "exact_authorization_is_pending": report.get("status")
        == "verified_host_bound_plan_no_execution_authorized",
        "model_revision_is_exact": report.get("model", {}).get("revision")
        == MODEL_REVISION,
        "network_boundary_is_exact": report.get("limits", {}).get(
            "maximum_network_bytes"
        )
        == TOTAL_NETWORK_BYTES,
        "runtime_digest_is_exact": report.get("runtime", {}).get("image_digest")
        == RUNTIME_DIGEST,
    }
    if not all(checks.values()):
        raise ValueError("gpt-oss host-bound verification failed")
    verification: dict[str, Any] = {
        "checks": checks,
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": observed,
        "schema_version": VERIFICATION_SCHEMA,
        "status": "PASS",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification


def load_execution_authorization(
    path: str | Path, *, plan_path: str | Path
) -> dict[str, Any]:
    authorization = _load_json(path)
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA:
        raise ValueError("unsupported gpt-oss canary authorization")
    if authorization.get("status") != "authorized_exact_host_bound_teacher_canary":
        raise ValueError("gpt-oss canary authorization status changed")
    if authorization.get("approval_text") != EXACT_AUTHORIZATION_TEXT:
        raise ValueError("gpt-oss canary approval text is not exact")
    if authorization.get("plan_file_sha256") != file_sha256(plan_path):
        raise ValueError("gpt-oss canary authorization plan binding changed")
    if authorization.get("server") != SERVER:
        raise ValueError("gpt-oss canary authorization server binding changed")
    if authorization.get("limits") != LIMITS:
        raise ValueError("gpt-oss canary authorization limits changed")
    if authorization.get("prohibitions") != PROHIBITIONS:
        raise ValueError("gpt-oss canary authorization prohibitions changed")
    started = authorization.get("authorized_start_unix")
    if not isinstance(started, int) or started <= 0:
        raise ValueError("gpt-oss canary authorization start time is invalid")
    return authorization


def load_runtime_receipt(path: str | Path, *, authorization_path: str | Path) -> dict[str, Any]:
    receipt = _load_json(path)
    if receipt.get("schema_version") != RUNTIME_RECEIPT_SCHEMA:
        raise ValueError("unsupported gpt-oss runtime receipt")
    if receipt.get("status") != "PASS_exact_localhost_gpt_oss_runtime_ready":
        raise ValueError("gpt-oss runtime receipt is not PASS")
    if receipt.get("authorization_file_sha256") != file_sha256(authorization_path):
        raise ValueError("gpt-oss runtime receipt authorization binding changed")
    if receipt.get("server") != SERVER:
        raise ValueError("gpt-oss runtime receipt server binding changed")
    if receipt.get("model") != {
        "bytes": MODEL_BYTES,
        "id": MODEL_ID,
        "object_count": 25,
        "revision": MODEL_REVISION,
    }:
        raise ValueError("gpt-oss runtime receipt model binding changed")
    if receipt.get("runtime") != {
        "endpoint": LOCAL_ENDPOINT,
        "image": RUNTIME_IMAGE,
        "image_bytes": RUNTIME_BYTES,
        "image_digest": RUNTIME_DIGEST,
        "max_model_len": 4_096,
        "network_bind": "remote_localhost_only_through_pinned_ssh",
    }:
        raise ValueError("gpt-oss runtime receipt runtime binding changed")
    if not all(receipt.get("checks", {}).values()):
        raise ValueError("gpt-oss runtime receipt checks failed")
    return receipt


def _ssh_prefix(ssh_key: Path, known_hosts: Path) -> list[str]:
    if not ssh_key.is_file() or not known_hosts.is_file():
        raise FileNotFoundError("SSH key or pinned known-hosts file is missing")
    return [
        "ssh",
        "-i",
        str(ssh_key.resolve()),
        "-o",
        f"UserKnownHostsFile={known_hosts.resolve()}",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "ConnectTimeout=15",
        f"{SERVER['ssh_user']}@{SERVER['host']}",
    ]


def gpu_snapshot(ssh_key: Path, known_hosts: Path) -> dict[str, Any]:
    query = (
        "name,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu,"
        "ecc.errors.uncorrected.volatile.total"
    )
    result = subprocess.run(
        _ssh_prefix(ssh_key, known_hosts)
        + [f"nvidia-smi --query-gpu={query} --format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError("expected exactly one H100")
    fields = [item.strip() for item in rows[0].split(",")]
    if len(fields) != 7:
        raise RuntimeError("unexpected H100 telemetry field count")
    snapshot = {
        "ecc_uncorrected_volatile": int(fields[6]),
        "gpu_name": fields[0],
        "gpu_uuid": fields[1],
        "memory_total_mib": int(fields[3]),
        "memory_used_mib": int(fields[2]),
        "temperature_c": int(fields[5]),
        "utilization_percent": int(fields[4]),
    }
    if snapshot["gpu_name"] != SERVER["gpu_name"]:
        raise RuntimeError("H100 product identity changed")
    if snapshot["gpu_uuid"] != SERVER["gpu_uuid"]:
        raise RuntimeError("H100 UUID changed")
    if snapshot["memory_total_mib"] != SERVER["gpu_fb_memory_total_mib"]:
        raise RuntimeError("H100 FB memory total changed")
    if snapshot["ecc_uncorrected_volatile"] != 0:
        raise RuntimeError("H100 reported an uncorrected ECC error")
    return snapshot


def _post_chat_completion(payload: Mapping[str, Any], *, timeout: int = 300) -> dict[str, Any]:
    request = urllib.request.Request(
        LOCAL_ENDPOINT,
        data=json.dumps(payload, ensure_ascii=False, sort_keys=True).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            value = json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"local tunneled vLLM request failed: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("vLLM response root is not an object")
    return value


def _parse_final_response(
    response: Mapping[str, Any], *, scenario: base.Scenario, language: str
) -> list[dict[str, str]]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("vLLM response must contain exactly one choice")
    choice = choices[0]
    if not isinstance(choice, Mapping) or choice.get("finish_reason") != "stop":
        raise ValueError("vLLM final response did not finish with stop")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("vLLM final message is missing")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("vLLM final content is empty")
    if base._HIDDEN_REASONING.search(content):
        raise ValueError("hidden reasoning marker appeared in final content")
    output = json.loads(content)
    if not isinstance(output, dict):
        raise ValueError("vLLM final JSON root is not an object")
    return base._validate_output(output, scenario=scenario, language_mode=language)


def _critical_validation_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(
        marker in text
        for marker in (
            "ground-truth",
            "hidden reasoning",
            "pii or secret",
            "safety",
            "teacher identity",
        )
    )


def _scenario_index() -> dict[tuple[str, int], base.Scenario]:
    return {(scenario.task, scenario.variant): scenario for scenario in SCENARIOS}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {line_number} is not an object")
        rows.append(value)
    return rows


def _machine_audit(
    rows: Sequence[Mapping[str, Any]], telemetry: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    ids = [str(row.get("record_id")) for row in rows]
    if tuple(ids) != expected_record_ids():
        raise ValueError("gpt-oss final record membership or ordering changed")
    cells = Counter((str(row["task"]), str(row["language_mode"])) for row in rows)
    expected_cells = {(task, language) for task in TASKS for language in LANGUAGE_MODES}
    if set(cells) != expected_cells or set(cells.values()) != {10}:
        raise ValueError("gpt-oss final cell coverage changed")
    by_id = {str(item["record_id"]): item for item in telemetry}
    if set(by_id) != set(ids):
        raise ValueError("gpt-oss telemetry membership changed")
    first_attempt = {
        record_id: int(by_id[record_id]["accepted_on_attempt"]) == 1 for record_id in ids
    }
    overall = sum(first_attempt.values()) / len(ids)
    per_cell: dict[str, float] = {}
    for task, language in sorted(expected_cells):
        cell_ids = [
            str(row["record_id"])
            for row in rows
            if row["task"] == task and row["language_mode"] == language
        ]
        per_cell[f"{task}:{language}"] = sum(first_attempt[item] for item in cell_ids) / 10
    attempts = sum(int(item["attempts"]) for item in telemetry)
    retries = attempts - len(rows)
    critical = sum(int(item["critical_failures"]) for item in telemetry)
    checks = {
        "critical_failures_zero": critical
        == QUALITY_GATES["critical_factual_safety_or_tool_failures"],
        "exact_final_outputs": len(rows) == LIMITS["exact_final_outputs"],
        "inference_call_ceiling": attempts <= LIMITS["maximum_inference_calls"],
        "machine_acceptance_rate": overall
        >= QUALITY_GATES["minimum_machine_acceptance_rate"],
        "per_cell_acceptance_rate": min(per_cell.values())
        >= QUALITY_GATES["minimum_per_cell_acceptance_rate"],
        "runtime_retry_ceiling": retries <= LIMITS["maximum_runtime_retries"],
    }
    return {
        "checks": checks,
        "critical_failures": critical,
        "final_outputs": len(rows),
        "first_attempt_acceptance_rate": round(overall, 6),
        "inference_calls": attempts,
        "per_cell_first_attempt_acceptance_rate": {
            key: round(value, 6) for key, value in per_cell.items()
        },
        "runtime_retries": retries,
    }


def select_owner_sample(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    by_cell: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        by_cell.setdefault((str(row["task"]), str(row["language_mode"])), []).append(row)
    selected: list[Mapping[str, Any]] = []
    for cell in sorted(by_cell):
        selected.extend(
            sorted(
                by_cell[cell],
                key=lambda row: hashlib.sha256(
                    f"lumi-gptoss-owner-v0.1:{row['record_id']}".encode()
                ).hexdigest(),
            )[:2]
        )
    return sorted(selected, key=lambda row: str(row["record_id"]))


def run_canary(
    *,
    plan_path: Path,
    authorization_path: Path,
    runtime_receipt_path: Path,
    repository_root: Path,
    artifact_root: Path,
    ssh_key: Path,
    known_hosts: Path,
) -> dict[str, Any]:
    root = repository_root.resolve()
    plan = load_host_bound_plan(plan_path.resolve(), repository_root=root)
    expected_known_hosts_sha256 = str(plan["bindings"]["known_hosts"]["sha256"])
    if file_sha256(known_hosts.resolve()) != expected_known_hosts_sha256:
        raise ValueError("gpt-oss runner known-hosts binding changed")
    if ssh_key.resolve().stat().st_mode & 0o077:
        raise PermissionError("gpt-oss SSH private key must not be group/world accessible")
    authorization = load_execution_authorization(
        authorization_path.resolve(), plan_path=plan_path.resolve()
    )
    load_runtime_receipt(
        runtime_receipt_path.resolve(), authorization_path=authorization_path.resolve()
    )
    if time.time() - int(authorization["authorized_start_unix"]) > (
        LIMITS["maximum_instance_minutes"] * 60
    ):
        raise RuntimeError("gpt-oss canary instance-minute authorization expired")
    before_gpu = gpu_snapshot(ssh_key.resolve(), known_hosts.resolve())
    artifact = artifact_root.resolve()
    private = artifact / "private"
    evidence = artifact / "evidence"
    partial_rows_path = private / "candidate-rows.partial.jsonl"
    partial_telemetry_path = private / "generation-telemetry.partial.jsonl"
    rows = _read_jsonl(partial_rows_path)
    telemetry = _read_jsonl(partial_telemetry_path)
    expected_ids = expected_record_ids()
    if [str(row.get("record_id")) for row in rows] != list(expected_ids[: len(rows)]):
        raise ValueError("gpt-oss partial rows are not an exact expected prefix")
    if len(rows) != len(telemetry):
        raise ValueError("gpt-oss partial rows and telemetry differ")
    scenarios = _scenario_index()
    total_attempts = sum(int(item["attempts"]) for item in telemetry)
    total_retries = total_attempts - len(rows)
    started = time.monotonic()
    for manifest_item in prompt_manifest()[len(rows) :]:
        record_id = str(manifest_item["record_id"])
        scenario = scenarios[(str(manifest_item["task"]), int(manifest_item["variant"]))]
        language = str(manifest_item["language"])
        attempts_for_record = 0
        critical_failures = 0
        while True:
            if total_attempts >= LIMITS["maximum_inference_calls"]:
                raise RuntimeError("gpt-oss inference-call ceiling reached")
            if total_retries > LIMITS["maximum_runtime_retries"]:
                raise RuntimeError("gpt-oss runtime-retry ceiling reached")
            attempts_for_record += 1
            total_attempts += 1
            payload = chat_completion_payload(record_id, scenario, language)
            call_started = time.monotonic()
            response = _post_chat_completion(payload)
            elapsed = time.monotonic() - call_started
            try:
                messages = _parse_final_response(
                    response, scenario=scenario, language=language
                )
            except (ValueError, json.JSONDecodeError) as error:
                if _critical_validation_error(error):
                    critical_failures += 1
                    raise RuntimeError(
                        f"critical gpt-oss validation failure for {record_id}: {error}"
                    ) from error
                total_retries += 1
                if total_retries > LIMITS["maximum_runtime_retries"]:
                    raise RuntimeError("gpt-oss runtime-retry ceiling reached") from error
                continue
            row = {
                "language_mode": language,
                "messages": messages,
                "provenance": {
                    "ai_assisted": True,
                    "external_inference_api": False,
                    "human_authored": False,
                    "inference_location": "self_hosted_lambda_h100_pinned_ssh",
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "prompt_request_sha256": manifest_item["request_sha256"],
                    "reasoning_or_analysis_retained": False,
                    "runtime_image_digest": RUNTIME_DIGEST,
                    "seed": manifest_item["seed"],
                },
                "record_id": record_id,
                "rights_status": "private_research_candidate_pending_owner_review",
                "scenario_id": scenario.scenario_id,
                "schema_version": ROW_SCHEMA,
                "task": scenario.task,
            }
            item_telemetry = {
                "accepted_on_attempt": attempts_for_record,
                "attempts": attempts_for_record,
                "critical_failures": critical_failures,
                "elapsed_seconds_final_attempt": round(elapsed, 6),
                "record_id": record_id,
                "request_sha256": manifest_item["request_sha256"],
                "response_envelope_sha256": sha256_value(response),
            }
            rows.append(row)
            telemetry.append(item_telemetry)
            _write_jsonl(partial_rows_path, rows)
            _write_jsonl(partial_telemetry_path, telemetry)
            break
        if (time.monotonic() - started) > LIMITS["maximum_instance_minutes"] * 60:
            raise RuntimeError("gpt-oss canary runtime authorization expired")

    audit = _machine_audit(rows, telemetry)
    if not all(audit["checks"].values()):
        raise RuntimeError("gpt-oss machine gate failed closed")
    owner_sample = select_owner_sample(rows)
    if len(owner_sample) != LIMITS["owner_review_rows"]:
        raise ValueError("gpt-oss owner sample count changed")
    after_gpu = gpu_snapshot(ssh_key.resolve(), known_hosts.resolve())
    _write_jsonl(private / "candidate-rows.jsonl", rows)
    _write_jsonl(private / "generation-telemetry.jsonl", telemetry)
    _write_jsonl(private / "owner-review-sample-32.jsonl", owner_sample)
    report: dict[str, Any] = {
        "authorization_file_sha256": file_sha256(authorization_path),
        "candidate_registry_changed": False,
        "dataset_sealed": False,
        "external_api_used": False,
        "gpu": {"after": after_gpu, "before": before_gpu},
        "human_authorship_claimed": False,
        "machine_audit": audit,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "owner_review": {
            "accepted_rows": None,
            "minimum_acceptance_rate": QUALITY_GATES["minimum_owner_acceptance_rate"],
            "rows": len(owner_sample),
            "sample_internal_sha256": sha256_value(owner_sample),
            "status": "pending",
        },
        "plan_file_sha256": file_sha256(plan_path),
        "private_artifacts": {
            "candidate_rows_file_sha256": file_sha256(private / "candidate-rows.jsonl"),
            "generation_telemetry_file_sha256": file_sha256(
                private / "generation-telemetry.jsonl"
            ),
            "owner_review_sample_file_sha256": file_sha256(
                private / "owner-review-sample-32.jsonl"
            ),
        },
        "production_eligible": False,
        "runtime_receipt_file_sha256": file_sha256(runtime_receipt_path),
        "schema_version": RUN_REPORT_SCHEMA,
        "sft_training_started": False,
        "status": "machine_verified_candidate_pending_owner_review",
    }
    report["report_sha256"] = sha256_value(report)
    _write_json_atomic(evidence / "run-report.json", report)
    return report


def verify_run(
    *,
    plan_path: Path,
    authorization_path: Path,
    runtime_receipt_path: Path,
    repository_root: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    root = repository_root.resolve()
    load_host_bound_plan(plan_path.resolve(), repository_root=root)
    load_execution_authorization(authorization_path.resolve(), plan_path=plan_path.resolve())
    load_runtime_receipt(
        runtime_receipt_path.resolve(), authorization_path=authorization_path.resolve()
    )
    artifact = artifact_root.resolve()
    report_path = artifact / "evidence" / "run-report.json"
    report = _load_json(report_path)
    unsigned = dict(report)
    observed = unsigned.pop("report_sha256", None)
    if observed != sha256_value(unsigned):
        raise ValueError("gpt-oss run report internal hash changed")
    rows = _read_jsonl(artifact / "private" / "candidate-rows.jsonl")
    telemetry = _read_jsonl(artifact / "private" / "generation-telemetry.jsonl")
    audit = _machine_audit(rows, telemetry)
    if audit != report.get("machine_audit"):
        raise ValueError("gpt-oss independent machine audit differs")
    sample = select_owner_sample(rows)
    checks = {
        "authorization_binding": report.get("authorization_file_sha256")
        == file_sha256(authorization_path),
        "machine_gate_reproduced": all(audit["checks"].values()),
        "owner_sample_reproduced": sha256_value(sample)
        == report.get("owner_review", {}).get("sample_internal_sha256"),
        "plan_binding": report.get("plan_file_sha256") == file_sha256(plan_path),
        "runtime_binding": report.get("runtime_receipt_file_sha256")
        == file_sha256(runtime_receipt_path),
    }
    verification: dict[str, Any] = {
        "checks": checks,
        "run_report_file_sha256": file_sha256(report_path),
        "run_report_internal_sha256": observed,
        "schema_version": RUN_VERIFICATION_SCHEMA,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    verification["verification_sha256"] = sha256_value(verification)
    _write_json_atomic(artifact / "evidence" / "run-verification.json", verification)
    if verification["status"] != "PASS":
        raise ValueError("gpt-oss run independent verification failed")
    return verification
