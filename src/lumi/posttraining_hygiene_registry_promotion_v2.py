"""Immutable promotion of the exact two-source registry-v0.14 successor."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .posttraining_ai_v3_provenance_materialization import _verify_internal_sha
from .posttraining_candidate_mix_preflight import (
    _load_json,
    file_sha256,
    sha256_value,
)
from .posttraining_candidate_mix_reaudit import resolve_binding
from .posttraining_data import load_sft_policy
from .posttraining_hygiene_registry_preflight_v2 import (
    EXPECTED_SUCCESSOR,
    SOURCE_UPDATES,
    build_projected_registry_v14,
    json_file_bytes,
    load_hygiene_registry_v14_preflight_plan,
    rebuild_hygiene_registry_v14_preflight,
    validate_hygiene_registry_v14_preflight_report,
)
from .posttraining_sources import SFTSource

PLAN_SCHEMA_VERSION = "lumi.sft-hygiene-registry-promotion-plan.v0.2"
RECEIPT_SCHEMA_VERSION = "lumi.sft-hygiene-registry-promotion-receipt.v0.2"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-hygiene-registry-promotion-verification.v0.2"

PREDECESSOR_REGISTRY_SHA256 = "1e8f7b0d2f1bfc10118fd76a51813e942cc7d0ce558f73ebfdc0382e6ee2b077"
PREFLIGHT_PLAN_FILE_SHA256 = "b0e00f2350074b59126d6e7c3e221c870ca6216a2e8f67a8e1dce7d7e476a7bf"
PREFLIGHT_REPORT_FILE_SHA256 = "923859be3b55b6a26d82fe3c84ed03d5feb81571c0b2620ef2564cb7a7bf7d2c"
PREFLIGHT_REPORT_INTERNAL_SHA256 = (
    "1d1b4827edc2d1f24cb1a76c26896dbeebf0e923b9f3172d9e77e3cc2ab3b5cd"
)
PREFLIGHT_VERIFICATION_FILE_SHA256 = (
    "d74985c9c811cd97ca4571a7308a466fc8beaebab49d9ae89d24641759fee801"
)
PREFLIGHT_VERIFICATION_INTERNAL_SHA256 = (
    "e094931b647f69a162741f4e13b58a68656f79ff457a0992c1c612a9747619ab"
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
    "purpose": "materialize_exact_two_source_registry_v0_14_successor_only",
    "source_registry_promotion_authorized": True,
    "unrelated_access_authorized": False,
}
EXPECTED_LIMITS = {
    "changed_sources": 2,
    "maximum_network_bytes": 0,
    "maximum_runtime_minutes": 30,
    "predecessor_sources": 15,
    "successor_bytes": 17_657,
    "successor_sources": 15,
}
EXPECTED_OUTPUT = {
    "predecessor_registry_mutation_permitted": False,
    "production_dataset_promotions": 0,
    "raw_training_text_in_receipts": False,
    "receipt_path": "evidence/lumi-4b-sft-hygiene-registry-v0.14-promotion-v0.1.json",
    "successor_registry_path": "configs/lumi-4b-sft-source-registry-v0.14.json",
    "verification_path": (
        "evidence/lumi-4b-sft-hygiene-registry-v0.14-promotion-verification-v0.1.json"
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


def load_hygiene_registry_v14_promotion_plan(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    plan = _load_json(Path(path))
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "changed_source_ids": sorted(SOURCE_UPDATES),
        "limits": EXPECTED_LIMITS,
        "observed_at": "2026-08-08",
        "output": EXPECTED_OUTPUT,
        "predecessor_registry_sha256": PREDECESSOR_REGISTRY_SHA256,
        "proposed_successor_registry_sha256": SUCCESSOR_REGISTRY_SHA256,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "authorized_exact_immutable_registry_v0_14_promotion",
    }
    for field, expected_value in expected.items():
        if plan.get(field) != expected_value:
            raise ValueError(f"registry-v0.14 promotion plan changed: {field}")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("registry-v0.14 promotion binding set changed")
    for name, binding in bindings.items():
        resolve_binding(repository_root, binding, name=name)
    exact_bindings = {
        "predecessor_registry": PREDECESSOR_REGISTRY_SHA256,
        "preflight_plan": PREFLIGHT_PLAN_FILE_SHA256,
        "preflight_report": PREFLIGHT_REPORT_FILE_SHA256,
        "preflight_verification": PREFLIGHT_VERIFICATION_FILE_SHA256,
    }
    for name, expected_sha256 in exact_bindings.items():
        if bindings[name]["sha256"] != expected_sha256:
            raise ValueError(f"registry-v0.14 promotion binding changed: {name}")
    return plan


def verify_v14_promotion_prerequisites(
    *, plan: Mapping[str, Any], repository_root: str | Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = Path(repository_root).resolve()
    preflight_plan_path = resolve_binding(
        root, plan["bindings"]["preflight_plan"], name="preflight_plan"
    )
    preflight_plan = load_hygiene_registry_v14_preflight_plan(
        preflight_plan_path,
        repository_root=root,
    )
    preflight_report_path = resolve_binding(
        root, plan["bindings"]["preflight_report"], name="preflight_report"
    )
    preflight_report = _load_json(preflight_report_path)
    validate_hygiene_registry_v14_preflight_report(preflight_report)
    if (
        file_sha256(preflight_report_path) != PREFLIGHT_REPORT_FILE_SHA256
        or preflight_report.get("report_sha256") != PREFLIGHT_REPORT_INTERNAL_SHA256
        or preflight_report.get("status")
        != "verified_two_source_registry_v0_14_projection_not_materialized"
        or preflight_report.get("projected_successor")
        != {
            **EXPECTED_SUCCESSOR,
            "materialized": False,
            "path": EXPECTED_OUTPUT["successor_registry_path"],
        }
    ):
        raise ValueError("authorized registry-v0.14 preflight report changed")
    replayed_report = rebuild_hygiene_registry_v14_preflight(
        plan=preflight_plan,
        plan_file_sha256=file_sha256(preflight_plan_path),
        repository_root=root,
    )
    if preflight_report != replayed_report:
        raise ValueError("registry-v0.14 preflight report no longer replays")

    preflight_verification_path = resolve_binding(
        root,
        plan["bindings"]["preflight_verification"],
        name="preflight_verification",
    )
    preflight_verification = _load_json(preflight_verification_path)
    if (
        file_sha256(preflight_verification_path) != PREFLIGHT_VERIFICATION_FILE_SHA256
        or _verify_internal_sha(preflight_verification, "verification_sha256")
        != PREFLIGHT_VERIFICATION_INTERNAL_SHA256
        or preflight_verification.get("status")
        != "verified_non_mutating_two_source_registry_v0_14_preflight"
        or preflight_verification.get("projected_successor_sha256") != SUCCESSOR_REGISTRY_SHA256
        or preflight_verification.get("report_file_sha256") != PREFLIGHT_REPORT_FILE_SHA256
    ):
        raise ValueError("authorized registry-v0.14 preflight verification changed")

    predecessor_path = resolve_binding(
        root, plan["bindings"]["predecessor_registry"], name="predecessor_registry"
    )
    predecessor = _load_json(predecessor_path)
    return preflight_report, preflight_verification, predecessor


def build_and_validate_registry_v14_successor(
    *, predecessor: Mapping[str, Any], data_contract_path: str | Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    successor, changed = build_projected_registry_v14(predecessor)
    encoded = json_file_bytes(successor)
    if len(encoded) != SUCCESSOR_REGISTRY_BYTES:
        raise ValueError("registry-v0.14 successor byte count changed")
    if hashlib.sha256(encoded).hexdigest() != SUCCESSOR_REGISTRY_SHA256:
        raise ValueError("registry-v0.14 successor SHA-256 changed")
    policy = load_sft_policy(data_contract_path)
    for source in successor["sources"]:
        SFTSource.from_mapping(
            source,
            allowed_licenses=policy.allowed_licenses,
            allowed_languages=policy.allowed_languages,
            allowed_splits=policy.allowed_splits,
            allowed_tasks=frozenset(policy.task_mix),
        )
    predecessor_unchanged = copy.deepcopy(dict(predecessor))
    successor_unchanged = copy.deepcopy(successor)
    predecessor_unchanged["sources"] = [
        item for item in predecessor_unchanged["sources"] if item["source_id"] not in SOURCE_UPDATES
    ]
    successor_unchanged["sources"] = [
        item for item in successor_unchanged["sources"] if item["source_id"] not in SOURCE_UPDATES
    ]
    if predecessor_unchanged != successor_unchanged:
        raise ValueError("registry-v0.14 successor changed unrelated metadata")
    return successor, changed


def build_registry_v14_promotion_receipt(
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
            "preflight_verification_file_sha256": (PREFLIGHT_VERIFICATION_FILE_SHA256),
            "preflight_verification_internal_sha256": (PREFLIGHT_VERIFICATION_INTERNAL_SHA256),
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
            "predecessor_path": "configs/lumi-4b-sft-source-registry-v0.13.json",
            "predecessor_sha256": PREDECESSOR_REGISTRY_SHA256,
            "successor_bytes": SUCCESSOR_REGISTRY_BYTES,
            "successor_path": EXPECTED_OUTPUT["successor_registry_path"],
            "successor_sha256": SUCCESSOR_REGISTRY_SHA256,
            "successor_sources": 15,
        },
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "verified_immutable_two_source_registry_v0_14_nonproduction",
    }
    receipt["receipt_sha256"] = sha256_value(receipt)
    validate_registry_v14_promotion_receipt(receipt)
    return receipt


def validate_registry_v14_promotion_receipt(receipt: Mapping[str, Any]) -> None:
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported registry-v0.14 promotion receipt")
    _verify_internal_sha(receipt, "receipt_sha256")
    if receipt.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("registry-v0.14 receipt authorization changed")
    if receipt.get("status") != ("verified_immutable_two_source_registry_v0_14_nonproduction"):
        raise ValueError("registry-v0.14 receipt status changed")
    if receipt.get("production_eligible") is not False:
        raise ValueError("registry-v0.14 promotion cannot be production eligible")
    if receipt.get("registry") != {
        "predecessor_path": "configs/lumi-4b-sft-source-registry-v0.13.json",
        "predecessor_sha256": PREDECESSOR_REGISTRY_SHA256,
        "successor_bytes": SUCCESSOR_REGISTRY_BYTES,
        "successor_path": EXPECTED_OUTPUT["successor_registry_path"],
        "successor_sha256": SUCCESSOR_REGISTRY_SHA256,
        "successor_sources": 15,
    }:
        raise ValueError("registry-v0.14 receipt registry identity changed")
    if [item.get("source_id") for item in receipt.get("changed_sources", [])] != (
        sorted(SOURCE_UPDATES)
    ):
        raise ValueError("registry-v0.14 receipt source set changed")
    serialized = json.dumps(receipt, ensure_ascii=False, sort_keys=True)
    for forbidden in ('"messages"', '"content"', '"assistant_response"'):
        if forbidden in serialized:
            raise ValueError("registry-v0.14 receipt contains raw training text")


def build_registry_v14_promotion_verification(
    *,
    predecessor_file_sha256: str,
    receipt_file_sha256: str,
    receipt_internal_sha256: str,
    successor_file_sha256: str,
    successor_file_bytes: int,
) -> dict[str, Any]:
    verification: dict[str, Any] = {
        "checks": {
            "exact_oasst1_oasst2_metadata_updates_reproduced": True,
            "predecessor_registry_v0_13_unchanged": True,
            "promotion_receipt_reproduced": True,
            "raw_training_text_retained": False,
            "successor_registry_v0_14_reproduced": True,
            "unrelated_thirteen_sources_and_top_level_metadata_unchanged": True,
        },
        "observed_at": "2026-08-08",
        "predecessor_registry_file_sha256": predecessor_file_sha256,
        "production_eligible": False,
        "receipt_file_sha256": receipt_file_sha256,
        "receipt_internal_sha256": receipt_internal_sha256,
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_hygiene_registry_v0_14_nonproduction",
        "successor_registry_bytes": successor_file_bytes,
        "successor_registry_file_sha256": successor_file_sha256,
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification


__all__ = [
    "EXPECTED_LIMITS",
    "SUCCESSOR_REGISTRY_BYTES",
    "SUCCESSOR_REGISTRY_SHA256",
    "build_and_validate_registry_v14_successor",
    "build_registry_v14_promotion_receipt",
    "build_registry_v14_promotion_verification",
    "load_hygiene_registry_v14_promotion_plan",
    "validate_registry_v14_promotion_receipt",
    "verify_v14_promotion_prerequisites",
]
