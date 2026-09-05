# ruff: noqa: E501, SLF001
"""Resume the final 17 canary rows with field-specific language anchors."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import posttraining_gpt_oss_teacher_dual_field_resume_v12 as v12
from . import posttraining_gpt_oss_teacher_bengali_script_field_repair_v13 as v13
from .posttraining_candidate_mix_preflight import file_sha256, sha256_value

v8 = v12.v8
base = v12.base

PLAN_SCHEMA = "lumi.sft-gpt-oss-teacher-language-anchor-resume-plan.v0.14"
REPORT_SCHEMA = "lumi.sft-gpt-oss-teacher-language-anchor-resume-report.v0.14"
VERIFICATION_SCHEMA = "lumi.sft-gpt-oss-teacher-language-anchor-resume-verification.v0.14"
AUTHORIZATION_SCHEMA = "lumi.sft-gpt-oss-teacher-language-anchor-resume-authorization.v0.14"
RUNTIME_REUSE_RECEIPT_SCHEMA = "lumi.sft-gpt-oss-teacher-language-anchor-resume-runtime-reuse-receipt.v0.14"
RUN_REPORT_SCHEMA = "lumi.sft-gpt-oss-teacher-language-anchor-resume-run-report.v0.14"
RUN_VERIFICATION_SCHEMA = "lumi.sft-gpt-oss-teacher-language-anchor-resume-run-verification.v0.14"

PROMPT_CONTRACT = "lumi.gpt-oss-teacher.language-anchor-resume.v0.14"
SEED_CONTRACT = "sha256(record_id:language-anchor-resume-attempt:attempt_number)-u32.v0.14"
REASONING_EFFORT = "low"
MAX_TOKENS = 1_536
MAXIMUM_ATTEMPTS_PER_RECORD = 2

SERVER = v12.SERVER
PROHIBITIONS = v12.PROHIBITIONS
RUNTIME_REUSE = v12.RUNTIME_REUSE
RETENTION = v12.RETENTION
FROZEN_MACHINE_GATES = v12.FROZEN_MACHINE_GATES
ACCEPTED_REPAIRS = dict(v12.ACCEPTED_REPAIRS)
ACCEPTED_REPAIRS["bengali_exact_field_v0_13"] = {
    "accepted_rows": 1,
    "candidate_file_sha256": "3906151cefa70fdc7cea7d83ff48c661fb80fe4de2dba413d7f51bbfea690e10",
    "owner_review_file_sha256": "2570ad51a2e29858e1204810f3199c2c5b5666f21a85a81ba4c3242ed1a30576",
    "owner_review_status": "ACCEPT_exact_bengali_field_quality",
    "owner_review_verification_file_sha256": "9407999c8b5e6491b68bdec77466dc0ec3e36725070f5c85adcbcc38c9e5fac9",
    "run_report_file_sha256": "f2fe17264b25dad0035ed041e078cc45f32a2f4ccd332620a87b4cb4c5900d61",
    "run_verification_file_sha256": "0b7e4c6adb24e1eb78e344907f4866d9032d1ef018c77d341f1f7bfdb4c63b37",
}
PREDECESSOR = {
    "accepted_prefix_outputs": 142,
    "candidate_prefix_file_sha256": "f87a17034916dfcc272afb8e35fe8d5618fdcc585da5e6df93efd60873b237e3",
    "failure_report_file_sha256": "542dbdd71a23fafa799403eb1a6b8fd03f6ef85c4e1c9292b21fc0a275a28be0",
    "failure_verification_file_sha256": "7747f0f2df487d330cfe888ae3ac81eb10673d33ca31e9d3ddda12b14e8fa4db",
    "generation_telemetry_file_sha256": "67a0832d4fd0c7a700672889febaf84bb872850fcdd292fa53ecfe581da7b55d",
    "recovered_rejection_metadata_rows": 9,
    "rejection_metadata_file_sha256": "c3f4b22dc8c6af5a96455429f2e8c37d2c25786cd29b6e12520e29ae6fb7cfab",
    "replacement_candidate_file_sha256": "3906151cefa70fdc7cea7d83ff48c661fb80fe4de2dba413d7f51bbfea690e10",
    "replacement_owner_review_file_sha256": "2570ad51a2e29858e1204810f3199c2c5b5666f21a85a81ba4c3242ed1a30576",
    "replacement_owner_verification_file_sha256": "9407999c8b5e6491b68bdec77466dc0ec3e36725070f5c85adcbcc38c9e5fac9",
    "replacement_telemetry_file_sha256": "e69a647267bc22f483807ccf7e6dca5ce916967a491414a7284c219b5623ff26",
    "status": "accepted_143_row_prefix_with_v0_13_replacement",
}
LIMITS = {
    "additional_final_outputs": 17,
    "exact_final_outputs": 160,
    "exact_prompt_cells": 16,
    "maximum_additional_generated_tokens_total": 52_224,
    "maximum_additional_inference_calls": 34,
    "maximum_additional_runtime_retries": 17,
    "maximum_attempts_per_record": 2,
    "maximum_context_tokens": 4_096,
    "maximum_generated_tokens_per_call": MAX_TOKENS,
    "maximum_inference_calls": 186,
    "maximum_instance_minutes": 30,
    "maximum_network_bytes": 0,
    "maximum_private_write_bytes": 4 * 1024**3,
    "maximum_runtime_retries": 26,
    "maximum_usd_before_tax": 2.15,
    "owner_review_rows": 32,
    "predecessor_accepted_prefix_outputs": 143,
    "predecessor_retained_prefix_attempts": 152,
    "prompts_per_cell": 10,
}
PROMPT_QUALITY_GATES = dict(v12.PROMPT_QUALITY_GATES)
PROMPT_QUALITY_GATES.update({
    "answer_number_required_in_assistant_response": True,
    "accepted_v0_13_replacement_revalidated": True,
    "field_specific_language_anchors_required_for_remaining_records": True,
    "predecessor_prefix_revalidated": True,
})

EXACT_AUTHORIZATION_TEXT = (
    "I approve one bounded self-hosted Lumi gpt-oss-120b remaining-17 language-anchor full-canary "
    "resume successor on the existing exact Lambda host 68.209.73.55, bound to ED25519 host-key "
    "fingerprint SHA256:2ZSQqNUYJhWNjihHnqLQ1YNHxoqtMJeckVJRKlc5hs4 and exactly one NVIDIA "
    "H100 80GB GPU GPU-c7e68b23-95b4-37bd-b40b-c41a7ec716f3 reporting at least 81,559 MiB FB "
    "total without adding the 480 MiB reserved value twice. Reuse, and do not redownload, the 25 "
    "verified openai/gpt-oss-120b model objects at revision "
    "b5c939de8f754692c1647ca79fbf85e8c1e70f8a bound by model receipt SHA-256 "
    "798914d8532d5b36b61711f34591cabe1e03fa6ac1128fe5e3737c029ead1284, and reuse the existing "
    "vLLM image digest sha256:4d08193d2fd05aadb1b5678f93ae609efb2635df67da45f3efe781c368b34dc8. "
    "Bind the v0.12 failure report SHA-256 "
    "542dbdd71a23fafa799403eb1a6b8fd03f6ef85c4e1c9292b21fc0a275a28be0 and failure verification "
    "SHA-256 7747f0f2df487d330cfe888ae3ac81eb10673d33ca31e9d3ddda12b14e8fa4db, plus the accepted v0.13 "
    "owner-review SHA-256 2570ad51a2e29858e1204810f3199c2c5b5666f21a85a81ba4c3242ed1a30576 and owner-review "
    "verification SHA-256 9407999c8b5e6491b68bdec77466dc0ec3e36725070f5c85adcbcc38c9e5fac9. "
    "Reuse without regeneration the exact 142 accepted v0.12 prefix rows bound by SHA-256 "
    "f87a17034916dfcc272afb8e35fe8d5618fdcc585da5e6df93efd60873b237e3 and their exact telemetry "
    "bound by SHA-256 67a0832d4fd0c7a700672889febaf84bb872850fcdd292fa53ecfe581da7b55d, append the exact "
    "accepted v0.13 replacement row bound by SHA-256 "
    "3906151cefa70fdc7cea7d83ff48c661fb80fe4de2dba413d7f51bbfea690e10 and its accepted-attempt "
    "metadata bound by SHA-256 e69a647267bc22f483807ccf7e6dca5ce916967a491414a7284c219b5623ff26, "
    "carry forward only the nine recovered-record rejection metadata entries, and exclude both "
    "failed v0.12 attempts for the repaired record. Generate only the remaining 17 records, "
    "beginning with lumi-gptoss-factual_qa-english-05, under prompt contract "
    "lumi.gpt-oss-teacher.language-anchor-resume.v0.14, reasoning effort low, a 1,536 "
    "generated-token ceiling per call, distinct attempt-bound deterministic seeds, and at most 2 "
    "new attempts per remaining record. Preserve all v0.12 complete-row gates; require exact field "
    "anchors Please tell me and The answer is for English, eta ki and uttor hobe for Banglish, "
    "এই question and সঠিক answer for natural code-switching, and দয়া করে বলুন and উত্তর হলো for "
    "Bengali script, in user_prompt and assistant_response respectively. Require the exact frozen "
    "answer number in every assistant_response and terminal punctuation in every accepted assistant "
    "field. Retain rejected new attempts only as validation error, non-sensitive finish_reason, "
    "usage token counts, and request/response-envelope hashes without raw candidate or reasoning "
    "text. Maximum 30 additional instance-minutes / $2.15 before tax / 0 additional model or "
    "runtime download bytes / 4 GiB additional private writes / 34 additional sequential inference "
    "calls / 17 additional retry calls / 17 additional final outputs, while the combined retained-"
    "prefix plus resume audit remains capped at 186 calls and 26 retries. It may restart and then "
    "cleanly stop only the bound canary container, serve only on remote localhost through pinned SSH, "
    "retain only accepted final-channel candidate text, run the full frozen 16-cell/160-row machine "
    "gate, and prepare the deterministic 32-row owner-review packet. No external API, public port, "
    "model or runtime download, registry change, dataset seal, SFT training, production promotion, "
    "unrelated access, new instance, or termination."
)

_BASE_CHAT = v12.chat_completion_payload
_BASE_VALIDATE = v12.validate_prompt_quality

LANGUAGE_FIELD_ANCHORS = {
    "banglish": ("eta ki", "uttor hobe"),
    "natural_code_switching": ("এই question", "সঠিক answer"),
    "bengali_script": ("দয়া করে বলুন", "উত্তর হলো"),
    "english": ("Please tell me", "The answer is"),
}


def _language_anchor_applies(scenario: Any, language: str) -> bool:
    return scenario.task == "factual_qa" and (
        int(scenario.variant) >= 6 or (int(scenario.variant) == 5 and language == "english")
    )


def full_canary_seed(record_id: str, attempt_number: int) -> int:
    if not 1 <= attempt_number <= MAXIMUM_ATTEMPTS_PER_RECORD:
        raise ValueError("v0.14 attempt number is outside the reviewed contract")
    material = f"{record_id}:language-anchor-resume-attempt:{attempt_number}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def generation_messages(scenario: Any, language: str, *, attempt_number: int) -> list[dict[str, str]]:
    messages = copy.deepcopy(v12.generation_messages(scenario, language, attempt_number=attempt_number))
    if _language_anchor_applies(scenario, language):
        user_anchor, assistant_anchor = LANGUAGE_FIELD_ANCHORS[language]
        answer_number = str(scenario.answer_number)
        messages[0]["content"] += (
            f" In the JSON string value user_prompt, include this exact natural phrase verbatim: "
            f"{user_anchor}. In the JSON string value assistant_response, include this different "
            f"exact natural phrase verbatim: {assistant_anchor}. The assistant_response must contain "
            f"the exact ASCII answer number {answer_number} and end in terminal punctuation. Keep "
            "each anchor only in its assigned field. Do not translate, transliterate, inflect, "
            "reformat, quote, label, or explain these requirements. Silently verify both field "
            "anchors, the answer number, the requested language, and punctuation before returning."
        )
        if language == "english":
            messages[0]["content"] += " Both generated fields must contain English only and no Bengali characters."
        elif language == "banglish":
            messages[0]["content"] += " Both generated fields must be natural Latin-script Banglish and contain no Bengali characters."
        elif language == "natural_code_switching":
            messages[0]["content"] += " Both generated fields must naturally mix Bengali script and English words."
        elif language == "bengali_script":
            messages[0]["content"] += " Both generated fields must contain natural Bengali Unicode script."
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
    if scenario.task == "factual_qa" and int(scenario.variant) == 5 and language == "bengali_script":
        v13.validate_exact_bengali_fields(messages)
    if _language_anchor_applies(scenario, language):
        user_anchor, assistant_anchor = LANGUAGE_FIELD_ANCHORS[language]
        if user_anchor not in user:
            raise ValueError(f"v0.14 user_prompt missing exact {language} anchor: {user_anchor}")
        if assistant_anchor not in assistant:
            raise ValueError(f"v0.14 assistant_response missing exact {language} anchor: {assistant_anchor}")
        if str(scenario.answer_number) not in assistant:
            raise ValueError(f"v0.14 assistant_response missing exact answer number: {scenario.answer_number}")
        if not assistant.rstrip().endswith((".", "?", "!", "।")):
            raise ValueError("v0.14 assistant_response lacks terminal punctuation")


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
    v12_root = repository_root / "artifacts/lumi-gpt-oss-teacher-dual-field-resume-v0.12/bf83317b28bb/attempt-001"
    v13_root = repository_root / "artifacts/lumi-gpt-oss-teacher-bengali-script-field-repair-v0.13/a8577ace9fa3/attempt-001"
    return {
        "v12_failure_report": v12_root / "evidence/failure-report.json",
        "v12_failure_verification": v12_root / "evidence/failure-verification.json",
        "v12_rejections": v12_root / "private/rejection-metadata.partial.jsonl",
        "v12_rows": v12_root / "private/candidate-rows.partial.jsonl",
        "v12_telemetry": v12_root / "private/generation-telemetry.partial.jsonl",
        "v13_candidate": v13_root / "private/repair-candidate.jsonl",
        "v13_owner": v13_root / "evidence/owner-review.json",
        "v13_owner_verification": v13_root / "evidence/owner-review-verification.json",
        "v13_telemetry": v13_root / "private/repair-attempt-metadata.jsonl",
    }


def _load_and_validate_prefix(repository_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    paths = _paths(repository_root)
    expected_hashes = {
        "v12_failure_report": PREDECESSOR["failure_report_file_sha256"],
        "v12_failure_verification": PREDECESSOR["failure_verification_file_sha256"],
        "v12_rejections": PREDECESSOR["rejection_metadata_file_sha256"],
        "v12_rows": PREDECESSOR["candidate_prefix_file_sha256"],
        "v12_telemetry": PREDECESSOR["generation_telemetry_file_sha256"],
        "v13_candidate": PREDECESSOR["replacement_candidate_file_sha256"],
        "v13_owner": PREDECESSOR["replacement_owner_review_file_sha256"],
        "v13_owner_verification": PREDECESSOR["replacement_owner_verification_file_sha256"],
        "v13_telemetry": PREDECESSOR["replacement_telemetry_file_sha256"],
    }
    for name, path in paths.items():
        if file_sha256(path) != expected_hashes[name]:
            raise ValueError(f"v0.14 predecessor binding changed: {name}")
    rows = base._read_jsonl(paths["v12_rows"])
    telemetry = base._read_jsonl(paths["v12_telemetry"])
    all_rejections = base._read_jsonl(paths["v12_rejections"])
    replacement = base._read_jsonl(paths["v13_candidate"])
    replacement_metadata = base._read_jsonl(paths["v13_telemetry"])
    owner = base._load_json(paths["v13_owner"])
    owner_verification = base._load_json(paths["v13_owner_verification"])
    if len(rows) != 142 or len(telemetry) != 142 or len(replacement) != 1 or len(replacement_metadata) != 1:
        raise ValueError("v0.14 predecessor count changed")
    if owner.get("status") != "ACCEPT_exact_bengali_field_quality" or owner_verification.get("status") != "PASS_owner_acceptance_verified":
        raise ValueError("v0.14 replacement owner acceptance changed")
    expected_ids = base.expected_record_ids()
    if [row["record_id"] for row in rows] != list(expected_ids[:142]) or replacement[0]["record_id"] != expected_ids[142]:
        raise ValueError("v0.14 accepted prefix ordering changed")
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
    accepted_ids = set(expected_ids[:142])
    rejections = [entry for entry in all_rejections if entry["record_id"] in accepted_ids]
    if len(rejections) != 9:
        raise ValueError("v0.14 recovered rejection count changed")
    scenarios = {(item.task, item.variant): item for item in base.SCENARIOS}
    for row in rows:
        variant = int(str(row["record_id"]).rsplit("-", 1)[1])
        validate_prompt_quality(row["messages"], scenario=scenarios[(row["task"], variant)], language=row["language_mode"])
    if sum(int(item["attempts"]) for item in telemetry) != LIMITS["predecessor_retained_prefix_attempts"]:
        raise ValueError("v0.14 retained attempt accounting changed")
    return rows, telemetry, rejections


def load_resume_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    plan = base._load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("status") != "language_anchor_resume_plan_pending_explicit_execution_authorization":
        raise ValueError("unsupported or changed v0.14 resume plan")
    for key, expected in (("server", SERVER), ("runtime_reuse", RUNTIME_REUSE), ("limits", LIMITS), ("prohibitions", PROHIBITIONS), ("retention", RETENTION), ("predecessor", PREDECESSOR), ("accepted_repairs", ACCEPTED_REPAIRS), ("prompt_quality_gates", PROMPT_QUALITY_GATES), ("frozen_machine_gates", FROZEN_MACHINE_GATES)):
        if plan.get(key) != expected:
            raise ValueError(f"v0.14 {key} binding changed")
    if any(bool(value) for value in plan.get("authorization", {}).values() if isinstance(value, bool)):
        raise ValueError("v0.14 plan unexpectedly authorizes execution")
    if plan.get("next_gate", {}).get("exact_authorization_text") != EXACT_AUTHORIZATION_TEXT:
        raise ValueError("v0.14 authorization text changed")
    contract = plan.get("prompt_contract", {})
    if contract.get("id") != PROMPT_CONTRACT or contract.get("seed_contract") != SEED_CONTRACT or contract.get("manifest_sha256") != sha256_value(prompt_manifest()):
        raise ValueError("v0.14 prompt contract changed")
    base._resolve_bindings(plan, repository_root)
    _load_and_validate_prefix(Path(repository_root).resolve())
    return plan


def load_execution_authorization(path: str | Path, *, plan_path: str | Path) -> dict[str, Any]:
    authorization = base._load_json(path)
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA or authorization.get("status") != "authorized_exact_host_bound_language_anchor_resume":
        raise ValueError("unsupported or changed v0.14 authorization")
    if authorization.get("approval_text") != EXACT_AUTHORIZATION_TEXT or authorization.get("plan_file_sha256") != file_sha256(plan_path):
        raise ValueError("v0.14 approval or plan binding changed")
    for key, expected in (("server", SERVER), ("limits", LIMITS), ("prohibitions", PROHIBITIONS), ("predecessor", PREDECESSOR), ("accepted_repairs", ACCEPTED_REPAIRS)):
        if authorization.get(key) != expected:
            raise ValueError(f"v0.14 authorization {key} binding changed")
    if not isinstance(authorization.get("authorized_start_unix"), int) or authorization["authorized_start_unix"] <= 0:
        raise ValueError("v0.14 authorization start time is invalid")
    return authorization


def load_runtime_reuse_receipt(path: str | Path, *, authorization_path: str | Path) -> dict[str, Any]:
    receipt = base._load_json(path)
    if receipt.get("schema_version") != RUNTIME_REUSE_RECEIPT_SCHEMA or receipt.get("status") != "PASS_exact_existing_localhost_runtime_reused_for_language_anchor_resume_v0.14":
        raise ValueError("unsupported or failed v0.14 runtime receipt")
    if receipt.get("authorization_file_sha256") != file_sha256(authorization_path) or receipt.get("server") != SERVER or receipt.get("runtime_reuse") != RUNTIME_REUSE:
        raise ValueError("v0.14 runtime receipt binding changed")
    if not all(receipt.get("checks", {}).values()):
        raise ValueError("v0.14 runtime checks failed")
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
    first = prompt_manifest()[143]
    scenario = next(item for item in base.SCENARIOS if item.task == first["task"] and item.variant == first["variant"])
    anchor_scenario = next(item for item in base.SCENARIOS if item.task == "factual_qa" and item.variant == 6)
    with _patched_v8():
        plan = v8.load_full_canary_plan(plan_path.resolve(), repository_root=repository_root.resolve())
        payload = v8.chat_completion_payload(first["record_id"], scenario, first["language"], attempt_number=1)
        anchor_payloads = {
            language: v8.chat_completion_payload(
                f"lumi-gptoss-factual_qa-{language}-06",
                anchor_scenario,
                language,
                attempt_number=1,
            )
            for language in base.LANGUAGE_MODES
        }
        manifest = v8.prompt_manifest()
    anchor_texts = {
        language: anchor_payloads[language]["messages"][0]["content"]
        for language in base.LANGUAGE_MODES
    }
    checks = {
        "all_language_field_anchors_bound": all(
            user_anchor in anchor_texts[language] and assistant_anchor in anchor_texts[language]
            for language, (user_anchor, assistant_anchor) in LANGUAGE_FIELD_ANCHORS.items()
        ),
        "answer_number_bound_for_all_languages": all("180" in text for text in anchor_texts.values()),
        "first_remaining_record_exact": first["record_id"] == "lumi-gptoss-factual_qa-english-05",
        "manifest_exact": plan["prompt_contract"]["manifest_sha256"] == sha256_value(manifest),
        "patch_context_payload_completed": payload["seed"] == full_canary_seed(first["record_id"], 1),
        "v0_13_exact_reference_passes": _reference_v13_row_passes(),
    }
    if not all(checks.values()):
        raise ValueError("v0.14 patch-context regression gate failed")
    return checks


def build_plan_report(*, plan: Mapping[str, Any], plan_file_sha256: str, repository_root: Path) -> dict[str, Any]:
    resolved = base._resolve_bindings(plan, repository_root.resolve())
    rows, telemetry, rejections = _load_and_validate_prefix(repository_root.resolve())
    checks = patch_context_regression_gate(repository_root.resolve() / str(plan["self_path"]), repository_root.resolve())
    checks.update({
        "accepted_prefix_exact": len(rows) == 143 and len(telemetry) == 143,
        "additional_call_arithmetic_exact": LIMITS["predecessor_retained_prefix_attempts"] + LIMITS["maximum_additional_inference_calls"] == LIMITS["maximum_inference_calls"],
        "additional_output_exact": len(prompt_manifest()[143:]) == 17,
        "additional_retry_arithmetic_exact": len(rejections) + LIMITS["maximum_additional_runtime_retries"] == LIMITS["maximum_runtime_retries"],
        "authorization_pending": not any(bool(value) for value in plan["authorization"].values() if isinstance(value, bool)),
        "owner_acceptance_bound": plan["accepted_repairs"]["bengali_exact_field_v0_13"]["owner_review_status"] == "ACCEPT_exact_bengali_field_quality",
        "zero_download_ceiling": LIMITS["maximum_network_bytes"] == 0,
    })
    if not all(checks.values()):
        raise ValueError("v0.14 plan checks failed")
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
        "status": "verified_language_anchor_resume_plan_no_execution_authorized",
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
        "remaining_output_count_exact": report.get("limits", {}).get("additional_final_outputs") == 17,
        "status_exact": report.get("status") == "verified_language_anchor_resume_plan_no_execution_authorized",
    }
    verification: dict[str, Any] = {"checks": checks, "report_file_sha256": report_file_sha256, "report_internal_sha256": internal, "schema_version": VERIFICATION_SCHEMA, "status": "PASS" if all(checks.values()) else "FAIL"}
    verification["verification_sha256"] = sha256_value(verification)
    if verification["status"] != "PASS":
        raise ValueError("v0.14 independent plan verification failed")
    return verification


def _reference_v13_row_passes() -> bool:
    try:
        v13.validate_exact_bengali_fields(
            [
                {"role": "user", "content": v13.EXACT_USER_PROMPT},
                {"role": "assistant", "content": v13.EXACT_ASSISTANT_RESPONSE},
            ]
        )
    except ValueError:
        return False
    return True


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
