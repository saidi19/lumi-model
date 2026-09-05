"""Immutable candidate-only registry successor for the AI gap-closure canary."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .posttraining_data import load_sft_policy
from .posttraining_sources import SFTSource, load_sft_source_registry


PLAN_SCHEMA = "lumi.sft-ai-gap-source-registration-plan.v0.1"
RECEIPT_SCHEMA = "lumi.sft-ai-gap-source-registration-receipt.v0.1"
VERIFICATION_SCHEMA = "lumi.sft-ai-gap-source-registration-verification.v0.1"
PLAN_STATUS = "authorized_machine_only_candidate_registry_v0_23_successor"
RECEIPT_STATUS = "PASS_ai_gap_candidate_registered_in_v0_23_nonproduction"
VERIFICATION_STATUS = "PASS_verified_ai_gap_candidate_registry_v0_23_nonproduction"

AUTHORIZATION_TEXT = "ok shuru kor"
AUTHORIZATION_BYTES = 12
AUTHORIZATION_SHA256 = "b0caef0209b73d948972ed43dc099417ffe08ecdd00d0d7fa26510df1327e5f8"

SOURCE_ID = "lumi_ai_assisted_gap_closure_canary_v0_1"
SOURCE_REVISION = "e1c04ed8b2e6ee75cf462eb85db29daa9433e075c856ebc97ae6da0a21b25671"
DATASET_ID = "Afnan-Technologies-Inc/lumi-ai-assisted-gap-closure-canary-v0.1"
PREDECESSOR_SOURCES = 25
SUCCESSOR_SOURCES = 26

EXPECTED_CODE_PATHS = {
    "implementation": "src/lumi/posttraining_ai_gap_source_registration_v1.py",
    "runner": "scripts/register_sft_ai_gap_source_v1.py",
    "verifier": "scripts/verify_sft_ai_gap_source_registration_v1.py",
    "test": "tests/test_ai_gap_source_registration_v1.py",
}

EXPECTED_AUTHORIZATION = {
    "candidate_source_registration_authorized": True,
    "immutable_registry_successor_authorized": True,
    "machine_only_nonproduction_policy_authorized": True,
    "candidate_row_reads_authorized": False,
    "candidate_row_writes_authorized": False,
    "dataset_seal_authorized": False,
    "external_model_api_authorized": False,
    "gpu_authorized": False,
    "network_authorized": False,
    "predecessor_registry_mutation_authorized": False,
    "production_data_contract_change_authorized": False,
    "production_promotion_authorized": False,
    "remote_access_authorized": False,
    "sft_training_authorized": False,
    "unrelated_action_authorized": False,
}

EXPECTED_LIMITS = {
    "added_sources": 1,
    "changed_predecessor_sources": 0,
    "candidate_row_reads": 0,
    "candidate_row_writes": 0,
    "factual_verification_passes": 2,
    "maximum_evidence_write_bytes": 2_097_152,
    "maximum_external_model_api_calls": 0,
    "maximum_gpu_actions": 0,
    "maximum_network_bytes": 0,
    "maximum_runtime_minutes": 10,
    "maximum_training_actions": 0,
    "predecessor_sources": PREDECESSOR_SOURCES,
    "registry_successor_writes": 1,
    "removed_sources": 0,
    "successor_sources": SUCCESSOR_SOURCES,
}

EXPECTED_SOURCE = {
    "access": "first_party",
    "adapter": "lumi_ai_assisted_gap_closure_deterministic_renderer_v0.1",
    "blockers": [],
    "dataset_id": DATASET_ID,
    "evidence_urls": [],
    "languages": ["multilingual"],
    "license_id": "lumi-owned",
    "origin": "synthetic",
    "production_blockers": [
        "candidate_is_machine_only_ai_assisted_canary_not_the_complete_sft_mix",
        "full_registry_v0_23_cross_source_hygiene_reaudit_not_complete",
        "production_dataset_seal_and_sft_authorization_not_complete",
    ],
    "production_eligible": False,
    "repository_url": None,
    "revision": SOURCE_REVISION,
    "source_id": SOURCE_ID,
    "splits": ["train", "validation"],
    "status": "candidate_eligible",
    "task_categories": [
        "conversation",
        "factual_qa",
        "multilingual_bengali",
        "safety_helpfulness",
        "tool_api",
        "writing_summarization",
    ],
}

EXPECTED_OUTPUT = {
    "active_production_data_contract_update_permitted": False,
    "candidate_rows_read": False,
    "predecessor_registry_mutation_permitted": False,
    "production_dataset_promotions": 0,
    "raw_candidate_data_in_evidence": False,
    "receipt_path": "evidence/lumi-4b-sft-ai-gap-source-registration-v0.1.json",
    "successor_registry_path": "configs/lumi-4b-sft-source-registry-v0.23.json",
    "verification_path": "evidence/lumi-4b-sft-ai-gap-source-registration-verification-v0.1.json",
}

_FORBIDDEN_EVIDENCE_KEYS = {
    "assistant_response", "content", "messages", "prompt", "response",
    "source_text", "text", "tool_result", "user_prompt",
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def read_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"binding escapes repository: {value}")
    return resolved


def resolve_binding(root: Path, binding: Mapping[str, Any], label: str) -> Path:
    relative = binding.get("path")
    expected = binding.get("sha256")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} binding path is invalid")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{label} binding SHA-256 is invalid")
    path = _resolve(root, relative)
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"{label} binding changed")
    return path


def _verify_internal(value: Mapping[str, Any], field: str, label: str) -> None:
    unsigned = {key: item for key, item in value.items() if key != field}
    if value.get(field) != canonical_sha256(unsigned):
        raise ValueError(f"{label} internal seal changed")


def _assert_raw_text_free(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in _FORBIDDEN_EVIDENCE_KEYS:
                raise ValueError(f"raw candidate key in evidence: {'.'.join(path + (str(key),))}")
            _assert_raw_text_free(item, path + (str(key),))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_raw_text_free(item, path + (str(index),))


def load_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = read_object(path)
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("status") != PLAN_STATUS:
        raise ValueError("AI gap source-registration plan schema or status changed")
    if plan.get("authorization_text") != AUTHORIZATION_TEXT:
        raise ValueError("source-registration authorization text changed")
    if plan.get("authorization_bytes") != AUTHORIZATION_BYTES:
        raise ValueError("source-registration authorization byte count changed")
    if plan.get("authorization_sha256") != AUTHORIZATION_SHA256:
        raise ValueError("source-registration authorization SHA-256 changed")
    if hashlib.sha256(AUTHORIZATION_TEXT.encode()).hexdigest() != AUTHORIZATION_SHA256:
        raise ValueError("source-registration internal authorization binding changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("source-registration authorization boundary changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("source-registration limits changed")
    if plan.get("source_after") != EXPECTED_SOURCE:
        raise ValueError("source-registration exact source entry changed")
    if plan.get("output") != EXPECTED_OUTPUT:
        raise ValueError("source-registration output boundary changed")
    if plan.get("observed_at") != "2026-09-02":
        raise ValueError("source-registration observed date changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("source-registration bindings are missing")
    for label, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"{label} binding must be an object")
        resolve_binding(root, binding, str(label))
    for label, expected_path in EXPECTED_CODE_PATHS.items():
        if bindings.get(label, {}).get("path") != expected_path:
            raise ValueError(f"{label} code path changed")
    return plan


def validate_canary_evidence(
    report: Mapping[str, Any], verification: Mapping[str, Any]
) -> dict[str, Any]:
    if report.get("schema_version") != "lumi.sft-ai-assisted-gap-closure-canary-report.v0.1":
        raise ValueError("AI gap canary report schema changed")
    if report.get("status") != "PASS_local_ai_assisted_machine_only_canary_nonproduction":
        raise ValueError("AI gap canary report is not PASS")
    _verify_internal(report, "report_sha256", "AI gap canary report")
    if verification.get("schema_version") != (
        "lumi.sft-ai-assisted-gap-closure-canary-verification.v0.1"
    ):
        raise ValueError("AI gap canary verification schema changed")
    if verification.get("status") != (
        "PASS_verified_local_ai_assisted_machine_only_canary_nonproduction"
    ):
        raise ValueError("AI gap canary verification is not PASS")
    _verify_internal(verification, "verification_sha256", "AI gap canary verification")
    if report.get("production", {}).get("production_eligible") is not False:
        raise ValueError("AI gap canary unexpectedly became production eligible")
    if verification.get("production_eligible") is not False:
        raise ValueError("AI gap verification unexpectedly became production eligible")
    candidate = report.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("AI gap candidate aggregate is missing")
    if candidate.get("records") != 1_000 or candidate.get("source_revision") != SOURCE_REVISION:
        raise ValueError("AI gap candidate identity changed")
    if candidate.get("raw_candidate_data_in_evidence") is not False:
        raise ValueError("AI gap canary evidence retained raw candidate data")
    audit = report.get("audit")
    if not isinstance(audit, Mapping):
        raise ValueError("AI gap audit summary is missing")
    first, second = audit.get("pass_1"), audit.get("pass_2")
    if not isinstance(first, Mapping) or not isinstance(second, Mapping):
        raise ValueError("AI gap two-pass audit is missing")
    if canonical_sha256(first) != canonical_sha256(second):
        raise ValueError("AI gap audit passes differ")
    if first.get("accepted_records") != 1_000 or first.get("acceptance_rate") != 1.0:
        raise ValueError("AI gap audit acceptance changed")
    checks = first.get("checks")
    if not isinstance(checks, Mapping) or not checks or not all(value is True for value in checks.values()):
        raise ValueError("AI gap audit contains a failed check")
    tokens = first.get("supervised_tokens", {}).get("by_split", {})
    if tokens != {"train": 86_712, "validation": 21_592}:
        raise ValueError("AI gap frozen token totals changed")
    verification_checks = verification.get("checks")
    if not isinstance(verification_checks, Mapping) or not all(
        value is True for value in verification_checks.values()
    ):
        raise ValueError("AI gap structural verification contains a failed check")
    if verification.get("candidate_file_sha256_from_bound_two_pass_report") != candidate.get(
        "candidate_file_sha256"
    ):
        raise ValueError("AI gap candidate file binding changed")
    _assert_raw_text_free(report)
    _assert_raw_text_free(verification)
    return {
        "accepted_records": 1_000,
        "candidate_file_sha256": candidate["candidate_file_sha256"],
        "source_revision": SOURCE_REVISION,
        "train_supervised_tokens": 86_712,
        "validation_supervised_tokens": 21_592,
    }


def build_successor(
    predecessor: Mapping[str, Any], *, source_after: Mapping[str, Any], observed_at: str
) -> dict[str, Any]:
    sources = predecessor.get("sources")
    if not isinstance(sources, list) or len(sources) != PREDECESSOR_SOURCES:
        raise ValueError("predecessor registry source count changed")
    if any(
        isinstance(source, Mapping) and source.get("source_id") == SOURCE_ID
        for source in sources
    ):
        raise ValueError("AI gap source already exists in predecessor registry")
    successor = copy.deepcopy(dict(predecessor))
    successor["observed_at"] = observed_at
    successor["sources"] = copy.deepcopy(sources) + [copy.deepcopy(dict(source_after))]
    return successor


def validate_successor(
    predecessor: Mapping[str, Any],
    successor: Mapping[str, Any],
    *,
    data_contract_path: str | Path,
) -> None:
    before = predecessor.get("sources")
    after = successor.get("sources")
    if not isinstance(before, list) or not isinstance(after, list):
        raise ValueError("registry sources must be arrays")
    if len(before) != PREDECESSOR_SOURCES or len(after) != SUCCESSOR_SOURCES:
        raise ValueError("registry delta source counts changed")
    if after[:PREDECESSOR_SOURCES] != before:
        raise ValueError("predecessor registry sources changed")
    if after[-1] != EXPECTED_SOURCE:
        raise ValueError("successor appended source changed")
    if successor.get("schema_version") != predecessor.get("schema_version"):
        raise ValueError("source registry schema version changed")
    if successor.get("authorization") != predecessor.get("authorization"):
        raise ValueError("source registry authorization boundary changed")
    policy = load_sft_policy(data_contract_path)
    parsed = [
        SFTSource.from_mapping(
            source,
            allowed_licenses=policy.allowed_licenses,
            allowed_languages=policy.allowed_languages,
            allowed_splits=policy.allowed_splits,
            allowed_tasks=frozenset(policy.task_mix),
        )
        for source in after
    ]
    if parsed[-1].source_id != SOURCE_ID:
        raise ValueError("parsed successor source identity changed")
    if parsed[-1].status != "candidate_eligible" or parsed[-1].production_eligible:
        raise ValueError("parsed successor source admissibility changed")


def build_receipt(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    predecessor_file_sha256: str,
    successor_file_sha256: str,
    canary: Mapping[str, Any],
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": RECEIPT_STATUS,
        "observed_at": plan["observed_at"],
        "authorization": copy.deepcopy(plan["authorization"]),
        "bindings": {
            "plan_file_sha256": plan_file_sha256,
            "predecessor_registry_sha256": predecessor_file_sha256,
            "successor_registry_sha256": successor_file_sha256,
            "canary_report_sha256": plan["bindings"]["canary_report"]["sha256"],
            "canary_verification_sha256": plan["bindings"]["canary_verification"]["sha256"],
        },
        "registry_delta": {
            "added_source_id": SOURCE_ID,
            "added_sources": 1,
            "changed_predecessor_sources": 0,
            "predecessor_sources": PREDECESSOR_SOURCES,
            "removed_sources": 0,
            "status_after": "candidate_eligible",
            "successor_sources": SUCCESSOR_SOURCES,
        },
        "candidate": {
            "records": canary["accepted_records"],
            "source_revision": canary["source_revision"],
            "candidate_file_sha256": canary["candidate_file_sha256"],
            "train_supervised_tokens": canary["train_supervised_tokens"],
            "validation_supervised_tokens": canary["validation_supervised_tokens"],
            "candidate_rows_read": 0,
            "candidate_rows_modified": 0,
            "raw_candidate_data_retained": False,
        },
        "policy": {
            "machine_only_candidate_registration": True,
            "human_authorship_claimed": False,
            "human_review_claimed": False,
            "active_production_data_contract_updated": False,
            "production_dataset_promotions": 0,
            "production_eligible": False,
        },
        "execution": {
            "factual_verification_pass": 1,
            "network_bytes": 0,
            "external_model_api_calls": 0,
            "gpu_actions": 0,
            "training_actions": 0,
        },
        "remaining_blockers": list(EXPECTED_SOURCE["production_blockers"]),
        "next_gate": "registry_v0_23_full_candidate_mix_cross_source_reaudit",
        "production_eligible": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _assert_raw_text_free(receipt)
    return receipt


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pretty_json_bytes(value))


def run_registration(
    *,
    plan_path: str | Path,
    repository_root: str | Path,
    successor_path: str | Path,
    receipt_path: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan_file = Path(plan_path).resolve()
    successor_file = Path(successor_path).resolve()
    receipt_file = Path(receipt_path).resolve()
    plan = load_plan(plan_file, repository_root=root)
    bindings = plan["bindings"]
    predecessor_path = resolve_binding(root, bindings["predecessor_registry"], "predecessor_registry")
    data_contract_path = resolve_binding(root, bindings["data_contract"], "data_contract")
    canary_report = read_object(resolve_binding(root, bindings["canary_report"], "canary_report"))
    canary_verification = read_object(
        resolve_binding(root, bindings["canary_verification"], "canary_verification")
    )
    canary = validate_canary_evidence(canary_report, canary_verification)
    predecessor = read_object(predecessor_path)
    successor = build_successor(
        predecessor, source_after=plan["source_after"], observed_at=plan["observed_at"]
    )
    validate_successor(predecessor, successor, data_contract_path=data_contract_path)
    if successor_file.exists() or receipt_file.exists():
        raise FileExistsError("v0.23 registration outputs already exist; retry is forbidden")
    _write_once(successor_file, successor)
    successor_sha256 = file_sha256(successor_file)
    parsed = load_sft_source_registry(
        successor_file,
        allowed_licenses=load_sft_policy(data_contract_path).allowed_licenses,
        allowed_languages=load_sft_policy(data_contract_path).allowed_languages,
        allowed_splits=load_sft_policy(data_contract_path).allowed_splits,
        allowed_tasks=frozenset(load_sft_policy(data_contract_path).task_mix),
        expected_sha256=successor_sha256,
    )
    if len(parsed.sources) != SUCCESSOR_SOURCES or parsed.sources[-1].source_id != SOURCE_ID:
        raise ValueError("materialized v0.23 registry parsing changed")
    receipt = build_receipt(
        plan=plan,
        plan_file_sha256=file_sha256(plan_file),
        predecessor_file_sha256=file_sha256(predecessor_path),
        successor_file_sha256=successor_sha256,
        canary=canary,
    )
    _write_once(receipt_file, receipt)
    if receipt_file.stat().st_size > EXPECTED_LIMITS["maximum_evidence_write_bytes"]:
        raise ValueError("source-registration evidence write cap exceeded")
    return receipt


def verify_registration(
    *,
    plan_path: str | Path,
    repository_root: str | Path,
    successor_path: str | Path,
    receipt_path: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan_file = Path(plan_path).resolve()
    successor_file = Path(successor_path).resolve()
    receipt_file = Path(receipt_path).resolve()
    plan = load_plan(plan_file, repository_root=root)
    bindings = plan["bindings"]
    predecessor_path = resolve_binding(root, bindings["predecessor_registry"], "predecessor_registry")
    data_contract_path = resolve_binding(root, bindings["data_contract"], "data_contract")
    predecessor = read_object(predecessor_path)
    expected_successor = build_successor(
        predecessor, source_after=plan["source_after"], observed_at=plan["observed_at"]
    )
    observed_successor = read_object(successor_file)
    if observed_successor != expected_successor:
        raise ValueError("materialized v0.23 registry differs from reconstruction")
    validate_successor(predecessor, observed_successor, data_contract_path=data_contract_path)
    canary = validate_canary_evidence(
        read_object(resolve_binding(root, bindings["canary_report"], "canary_report")),
        read_object(resolve_binding(root, bindings["canary_verification"], "canary_verification")),
    )
    expected_receipt = build_receipt(
        plan=plan,
        plan_file_sha256=file_sha256(plan_file),
        predecessor_file_sha256=file_sha256(predecessor_path),
        successor_file_sha256=file_sha256(successor_file),
        canary=canary,
    )
    receipt = read_object(receipt_file)
    if receipt != expected_receipt:
        raise ValueError("source-registration receipt differs from reconstruction")
    _verify_internal(receipt, "receipt_sha256", "source-registration receipt")
    if file_sha256(predecessor_path) != plan["bindings"]["predecessor_registry"]["sha256"]:
        raise ValueError("predecessor registry changed")
    verification: dict[str, Any] = {
        "schema_version": VERIFICATION_SCHEMA,
        "status": VERIFICATION_STATUS,
        "observed_at": plan["observed_at"],
        "receipt_file_sha256": file_sha256(receipt_file),
        "receipt_internal_sha256": receipt["receipt_sha256"],
        "successor_registry_sha256": file_sha256(successor_file),
        "execution": {
            "factual_verification_pass": 2,
            "candidate_row_reads": 0,
            "network_bytes": 0,
            "external_model_api_calls": 0,
            "gpu_actions": 0,
            "training_actions": 0,
        },
        "checks": {
            "canary_evidence_reconstructed": True,
            "candidate_rows_not_read_or_modified": True,
            "exactly_one_candidate_source_added": True,
            "human_authorship_or_review_not_claimed": True,
            "predecessor_registry_unchanged": True,
            "production_contract_unchanged": True,
            "production_promotion_count_zero": True,
            "successor_registry_reconstructed": True,
            "two_zero_network_factual_passes_complete": True,
        },
        "production_eligible": False,
        "next_gate": "registry_v0_23_full_candidate_mix_cross_source_reaudit",
    }
    verification["verification_sha256"] = canonical_sha256(verification)
    _assert_raw_text_free(verification)
    return verification
