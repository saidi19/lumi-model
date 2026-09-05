"""Immutable promotion of the exact two-source hygiene registry successor."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .posttraining_candidate_mix_preflight import file_sha256, sha256_value
from .posttraining_candidate_mix_reaudit import resolve_binding
from .posttraining_data import load_sft_policy
from .posttraining_hygiene_registry_preflight import (
    EXPECTED_SUCCESSOR,
    SOURCE_UPDATES,
    build_projected_registry,
    json_file_bytes,
    load_hygiene_registry_preflight_plan,
    rebuild_hygiene_registry_preflight,
    validate_hygiene_registry_preflight_report,
)
from .posttraining_sources import SFTSource

PLAN_SCHEMA_VERSION = "lumi.sft-hygiene-registry-promotion-plan.v0.1"
RECEIPT_SCHEMA_VERSION = "lumi.sft-hygiene-registry-promotion-receipt.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-hygiene-registry-promotion-verification.v0.1"
)

PREDECESSOR_REGISTRY_SHA256 = (
    "5353c57189d82ee22306ccd11d116d017c5e683811cec2d6704cf45bcf7f02e7"
)
PREFLIGHT_PLAN_FILE_SHA256 = (
    "d6a7cd340a68e07f2383557c709433559f06cec3747139e56cf2909505107f48"
)
PREFLIGHT_REPORT_FILE_SHA256 = (
    "d84afad8ed5eadf3ff66b6dcfe232b601386617d0afbb276e58a37e8d30d85f5"
)
PREFLIGHT_REPORT_INTERNAL_SHA256 = (
    "ab7096e78718081c6d0953afeece1999cdfb8dcf7b45efcfe24e2319227aad7f"
)
PREFLIGHT_VERIFICATION_FILE_SHA256 = (
    "ecd37e08b297076f389cbde35d781a254cd40299a7a8076fafcf92593f8b57c4"
)
PREFLIGHT_VERIFICATION_INTERNAL_SHA256 = (
    "a6d8069759a53d51cc88801ff0b34944f000e44fe50277a7bdbd14c466c840c7"
)
SUCCESSOR_REGISTRY_SHA256 = EXPECTED_SUCCESSOR["sha256"]
SUCCESSOR_REGISTRY_BYTES = EXPECTED_SUCCESSOR["bytes"]

EXPECTED_AUTHORIZATION = {
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "predecessor_registry_mutation_authorized": False,
    "production_promotion_authorized": False,
    "production_sft_authorized": False,
    "purpose": "materialize_exact_two_source_registry_v0_12_successor_only",
    "source_registry_promotion_authorized": True,
    "unrelated_access_authorized": False,
}
EXPECTED_LIMITS = {
    "changed_sources": 2,
    "maximum_network_bytes": 0,
    "maximum_runtime_minutes": 30,
    "predecessor_sources": 14,
    "successor_bytes": 16_157,
    "successor_sources": 14,
}
EXPECTED_OUTPUT = {
    "predecessor_registry_mutation_permitted": False,
    "production_dataset_promotions": 0,
    "raw_training_text_in_receipts": False,
    "receipt_path": "evidence/lumi-4b-sft-hygiene-registry-v0.12-promotion-v0.1.json",
    "successor_registry_path": "configs/lumi-4b-sft-source-registry-v0.12.json",
    "verification_path": (
        "evidence/lumi-4b-sft-hygiene-registry-v0.12-promotion-"
        "verification-v0.1.json"
    ),
}
EXPECTED_BINDINGS = frozenset(
    {
        "data_contract",
        "implementation",
        "predecessor_registry",
        "preflight_implementation",
        "preflight_plan",
        "preflight_report",
        "preflight_verification",
        "runner",
        "source_registry_implementation",
        "test",
        "verifier",
    }
)


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _verify_internal_sha(value: Mapping[str, Any], field: str) -> str:
    observed = value.get(field)
    if not isinstance(observed, str) or len(observed) != 64:
        raise ValueError(f"missing internal SHA-256: {field}")
    unsigned = {key: item for key, item in value.items() if key != field}
    if sha256_value(unsigned) != observed:
        raise ValueError(f"internal SHA-256 changed: {field}")
    return observed


def load_hygiene_registry_promotion_plan(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    plan = _load_json(path)
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "changed_source_ids": sorted(SOURCE_UPDATES),
        "limits": EXPECTED_LIMITS,
        "observed_at": "2026-08-08",
        "output": EXPECTED_OUTPUT,
        "predecessor_registry_sha256": PREDECESSOR_REGISTRY_SHA256,
        "proposed_successor_registry_sha256": SUCCESSOR_REGISTRY_SHA256,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "authorized_exact_immutable_registry_v0_12_promotion",
    }
    for field, expected_value in expected.items():
        if plan.get(field) != expected_value:
            raise ValueError(f"hygiene registry promotion plan changed: {field}")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("hygiene registry promotion binding set changed")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"invalid hygiene registry promotion binding: {name}")
        resolve_binding(repository_root, binding, name=name)
    exact_bindings = {
        "predecessor_registry": PREDECESSOR_REGISTRY_SHA256,
        "preflight_plan": PREFLIGHT_PLAN_FILE_SHA256,
        "preflight_report": PREFLIGHT_REPORT_FILE_SHA256,
        "preflight_verification": PREFLIGHT_VERIFICATION_FILE_SHA256,
    }
    for name, expected_sha256 in exact_bindings.items():
        if bindings[name]["sha256"] != expected_sha256:
            raise ValueError(f"hygiene registry promotion authorization changed: {name}")
    return plan


def verify_promotion_prerequisites(
    *, plan: Mapping[str, Any], repository_root: str | Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = Path(repository_root).resolve()
    preflight_plan_path = resolve_binding(
        root, plan["bindings"]["preflight_plan"], name="preflight_plan"
    )
    preflight_plan = load_hygiene_registry_preflight_plan(
        preflight_plan_path,
        repository_root=root,
    )
    preflight_report_path = resolve_binding(
        root, plan["bindings"]["preflight_report"], name="preflight_report"
    )
    preflight_report = _load_json(preflight_report_path)
    validate_hygiene_registry_preflight_report(preflight_report)
    if (
        file_sha256(preflight_report_path) != PREFLIGHT_REPORT_FILE_SHA256
        or preflight_report.get("report_sha256")
        != PREFLIGHT_REPORT_INTERNAL_SHA256
        or preflight_report.get("status")
        != "verified_two_source_registry_v0_12_projection_not_materialized"
        or preflight_report.get("projected_successor")
        != {
            **EXPECTED_SUCCESSOR,
            "materialized": False,
            "path": EXPECTED_OUTPUT["successor_registry_path"],
        }
    ):
        raise ValueError("authorized hygiene registry preflight report changed")
    replayed_report = rebuild_hygiene_registry_preflight(
        plan=preflight_plan,
        plan_file_sha256=file_sha256(preflight_plan_path),
        repository_root=root,
    )
    if preflight_report != replayed_report:
        raise ValueError("hygiene registry preflight report no longer replays")

    preflight_verification_path = resolve_binding(
        root,
        plan["bindings"]["preflight_verification"],
        name="preflight_verification",
    )
    preflight_verification = _load_json(preflight_verification_path)
    if (
        file_sha256(preflight_verification_path)
        != PREFLIGHT_VERIFICATION_FILE_SHA256
        or _verify_internal_sha(preflight_verification, "verification_sha256")
        != PREFLIGHT_VERIFICATION_INTERNAL_SHA256
        or preflight_verification.get("status")
        != "verified_non_mutating_two_source_registry_v0_12_preflight"
        or preflight_verification.get("projected_successor_sha256")
        != SUCCESSOR_REGISTRY_SHA256
        or preflight_verification.get("report_file_sha256")
        != PREFLIGHT_REPORT_FILE_SHA256
    ):
        raise ValueError("authorized hygiene registry preflight verification changed")

    predecessor_path = resolve_binding(
        root, plan["bindings"]["predecessor_registry"], name="predecessor_registry"
    )
    predecessor = _load_json(predecessor_path)
    return preflight_report, preflight_verification, predecessor


def build_and_validate_successor(
    *,
    predecessor: Mapping[str, Any],
    data_contract_path: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    successor, changed = build_projected_registry(predecessor)
    encoded = json_file_bytes(successor)
    if len(encoded) != SUCCESSOR_REGISTRY_BYTES:
        raise ValueError("hygiene registry successor byte count changed")
    if hashlib.sha256(encoded).hexdigest() != SUCCESSOR_REGISTRY_SHA256:
        raise ValueError("hygiene registry successor SHA-256 changed")
    policy = load_sft_policy(data_contract_path)
    for source in successor["sources"]:
        SFTSource.from_mapping(
            source,
            allowed_licenses=policy.allowed_licenses,
            allowed_languages=policy.allowed_languages,
            allowed_splits=policy.allowed_splits,
            allowed_tasks=frozenset(policy.task_mix),
        )
    predecessor_without_changed = copy.deepcopy(dict(predecessor))
    successor_without_changed = copy.deepcopy(successor)
    predecessor_without_changed["sources"] = [
        item
        for item in predecessor_without_changed["sources"]
        if item["source_id"] not in SOURCE_UPDATES
    ]
    successor_without_changed["sources"] = [
        item
        for item in successor_without_changed["sources"]
        if item["source_id"] not in SOURCE_UPDATES
    ]
    if predecessor_without_changed != successor_without_changed:
        raise ValueError("hygiene registry successor changed unrelated metadata")
    return successor, changed


def build_promotion_receipt(
    *,
    plan_file_sha256: str,
    preflight_report: Mapping[str, Any],
    changed_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "authorization": dict(EXPECTED_AUTHORIZATION),
        "bindings": {
            "plan_file_sha256": plan_file_sha256,
            "predecessor_registry_sha256": PREDECESSOR_REGISTRY_SHA256,
            "preflight_report_file_sha256": PREFLIGHT_REPORT_FILE_SHA256,
            "preflight_report_internal_sha256": preflight_report["report_sha256"],
            "preflight_verification_file_sha256": (
                PREFLIGHT_VERIFICATION_FILE_SHA256
            ),
            "preflight_verification_internal_sha256": (
                PREFLIGHT_VERIFICATION_INTERNAL_SHA256
            ),
        },
        "changed_sources": changed_sources,
        "checks": {
            "changed_source_ids": sorted(SOURCE_UPDATES),
            "predecessor_registry_mutated": False,
            "production_dataset_promotions": 0,
            "raw_training_text_retained": False,
            "registry_promotions": 1,
            "unrelated_sources_changed": 0,
        },
        "observed_at": "2026-08-08",
        "production_eligible": False,
        "registry": {
            "predecessor_path": "configs/lumi-4b-sft-source-registry-v0.11.json",
            "predecessor_sha256": PREDECESSOR_REGISTRY_SHA256,
            "successor_bytes": SUCCESSOR_REGISTRY_BYTES,
            "successor_path": EXPECTED_OUTPUT["successor_registry_path"],
            "successor_sha256": SUCCESSOR_REGISTRY_SHA256,
            "successor_sources": 14,
        },
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "verified_immutable_two_source_registry_v0_12_nonproduction",
    }
    receipt["receipt_sha256"] = sha256_value(receipt)
    validate_promotion_receipt(receipt)
    return receipt


def validate_promotion_receipt(receipt: Mapping[str, Any]) -> None:
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported hygiene registry promotion receipt")
    _verify_internal_sha(receipt, "receipt_sha256")
    if receipt.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("hygiene registry promotion receipt authorization changed")
    if receipt.get("status") != (
        "verified_immutable_two_source_registry_v0_12_nonproduction"
    ):
        raise ValueError("hygiene registry promotion receipt status changed")
    if receipt.get("production_eligible") is not False:
        raise ValueError("hygiene registry promotion cannot be production eligible")
    if receipt.get("registry") != {
        "predecessor_path": "configs/lumi-4b-sft-source-registry-v0.11.json",
        "predecessor_sha256": PREDECESSOR_REGISTRY_SHA256,
        "successor_bytes": SUCCESSOR_REGISTRY_BYTES,
        "successor_path": EXPECTED_OUTPUT["successor_registry_path"],
        "successor_sha256": SUCCESSOR_REGISTRY_SHA256,
        "successor_sources": 14,
    }:
        raise ValueError("hygiene registry promotion receipt registry identity changed")
    if [item.get("source_id") for item in receipt.get("changed_sources", [])] != (
        sorted(SOURCE_UPDATES)
    ):
        raise ValueError("hygiene registry promotion receipt source set changed")
    serialized = json.dumps(receipt, ensure_ascii=False, sort_keys=True)
    for forbidden in ('"messages"', '"content"', '"assistant_response"'):
        if forbidden in serialized:
            raise ValueError("hygiene registry promotion receipt contains training text")


def build_promotion_verification(
    *,
    predecessor_file_sha256: str,
    receipt_file_sha256: str,
    receipt_internal_sha256: str,
    successor_file_sha256: str,
    successor_file_bytes: int,
) -> dict[str, Any]:
    verification: dict[str, Any] = {
        "checks": {
            "exact_two_source_metadata_updates_reproduced": True,
            "predecessor_registry_unchanged": True,
            "promotion_receipt_reproduced": True,
            "raw_training_text_retained": False,
            "successor_registry_reproduced": True,
            "unrelated_sources_and_top_level_metadata_unchanged": True,
        },
        "observed_at": "2026-08-08",
        "predecessor_registry_file_sha256": predecessor_file_sha256,
        "production_eligible": False,
        "receipt_file_sha256": receipt_file_sha256,
        "receipt_internal_sha256": receipt_internal_sha256,
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_hygiene_registry_v0_12_nonproduction",
        "successor_registry_bytes": successor_file_bytes,
        "successor_registry_file_sha256": successor_file_sha256,
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification
