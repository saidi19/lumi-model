"""Immutable registry promotion for the reviewed OpenCodeInstruct candidate."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .data_quality import sha256_file
from .posttraining_bigcodebench_signatures import sha256_value
from .posttraining_opencodeinstruct_source_promotion_preflight import (
    build_projected_successor,
    load_promotion_preflight_plan,
    validate_report,
)
from .posttraining_sources import SFTSource

PLAN_SCHEMA_VERSION = "lumi.sft-opencodeinstruct-source-promotion-plan.v0.1"
RECEIPT_SCHEMA_VERSION = "lumi.sft-opencodeinstruct-source-promotion-receipt.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-opencodeinstruct-source-promotion-verification.v0.1"
)

PREDECESSOR_REGISTRY_SHA256 = (
    "272b88f69ae19da25d3e979cf9d74ab0d4c8a1e6d276b9adcc148fd10738490f"
)
PREFLIGHT_PLAN_FILE_SHA256 = (
    "382b114f37c07126237727d724313dee7a48d2d4095bec29d851dd5d99e62455"
)
PREFLIGHT_REPORT_FILE_SHA256 = (
    "bcd61babf421c04805ecd33dcc4e0435ede2d2f50a485ec74b30aa7ac205731c"
)
PREFLIGHT_REPORT_INTERNAL_SHA256 = (
    "0b63aec845ccf277889783377d14b2568ed446b742a2f81a44eb9ab942d58005"
)
PREFLIGHT_VERIFICATION_FILE_SHA256 = (
    "6c073065e00e4096ce36c6f040fad13a8326f5b844b0520cc136c129e501aab3"
)
PREFLIGHT_VERIFICATION_INTERNAL_SHA256 = (
    "34815256abeb327199d09cef3f9cfd2e0c114f0529d7ab8821e03c924c8aacc8"
)
SUCCESSOR_REGISTRY_SHA256 = (
    "5353c57189d82ee22306ccd11d116d017c5e683811cec2d6704cf45bcf7f02e7"
)
SUCCESSOR_REGISTRY_BYTES = 16_129
SOURCE_ID = "nvidia_opencodeinstruct"

EXPECTED_AUTHORIZATION = {
    "candidate_registry_promotion_authorized": True,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "gpu_authorized": False,
    "network_authorized": False,
    "predecessor_registry_mutation_authorized": False,
    "production_promotion_authorized": False,
    "production_sft_authorized": False,
    "purpose": "exact_reviewed_opencodeinstruct_registry_v0_11_successor_only",
    "unrelated_access_authorized": False,
}
EXPECTED_LIMITS = {
    "added_sources": 1,
    "candidate_records": 11_100,
    "maximum_runtime_minutes": 30,
    "predecessor_sources": 13,
    "successor_sources": 14,
    "train_supervised_tokens": 2_100_077,
    "validation_supervised_tokens": 110_619,
}
EXPECTED_OUTPUT = {
    "active_data_contract_update_permitted": False,
    "predecessor_registry_mutation_permitted": False,
    "production_dataset_promotions": 0,
    "raw_training_text_in_receipts": False,
    "receipt_path": (
        "evidence/lumi-4b-sft-opencodeinstruct-source-promotion-v0.1.json"
    ),
    "successor_registry_path": "configs/lumi-4b-sft-source-registry-v0.11.json",
    "verification_path": (
        "evidence/"
        "lumi-4b-sft-opencodeinstruct-source-promotion-verification-v0.1.json"
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
        "verifier",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _verify_internal_seal(value: Mapping[str, Any], field: str) -> str:
    seal = _require_sha256(value.get(field), field)
    unsigned = {key: item for key, item in value.items() if key != field}
    if sha256_value(unsigned) != seal:
        raise ValueError(f"OpenCodeInstruct {field} seal changed")
    return seal


def resolve_binding(
    repository_root: str | Path,
    specification: Mapping[str, Any],
    *,
    name: str,
) -> Path:
    root = Path(repository_root).resolve()
    relative = specification.get("path")
    expected = _require_sha256(specification.get("sha256"), f"bindings.{name}.sha256")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"invalid OpenCodeInstruct promotion binding: {name}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.parent.resolve()) or not resolved.is_file():
        raise FileNotFoundError(f"OpenCodeInstruct promotion binding missing: {name}")
    if sha256_file(resolved) != expected:
        raise ValueError(f"OpenCodeInstruct promotion binding changed: {name}")
    return resolved


def load_promotion_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = _load_json(path)
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "limits": EXPECTED_LIMITS,
        "observed_at": "2026-08-07",
        "output": EXPECTED_OUTPUT,
        "predecessor_registry_sha256": PREDECESSOR_REGISTRY_SHA256,
        "proposed_successor_registry_sha256": SUCCESSOR_REGISTRY_SHA256,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "authorized_exact_candidate_registry_successor_promotion",
    }
    for field, expected_value in expected.items():
        if plan.get(field) != expected_value:
            raise ValueError(f"OpenCodeInstruct promotion plan changed: {field}")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("OpenCodeInstruct promotion binding inventory changed")
    for name, specification in bindings.items():
        if not isinstance(specification, Mapping):
            raise ValueError(f"OpenCodeInstruct promotion binding invalid: {name}")
        resolve_binding(root, specification, name=name)
    if bindings["predecessor_registry"]["sha256"] != PREDECESSOR_REGISTRY_SHA256:
        raise ValueError("OpenCodeInstruct predecessor authorization changed")
    if bindings["preflight_plan"]["sha256"] != PREFLIGHT_PLAN_FILE_SHA256:
        raise ValueError("OpenCodeInstruct preflight-plan authorization changed")
    if bindings["preflight_report"]["sha256"] != PREFLIGHT_REPORT_FILE_SHA256:
        raise ValueError("OpenCodeInstruct preflight-report authorization changed")
    if (
        bindings["preflight_verification"]["sha256"]
        != PREFLIGHT_VERIFICATION_FILE_SHA256
    ):
        raise ValueError("OpenCodeInstruct preflight-verification authorization changed")
    return plan


def verify_promotion_prerequisites(
    *,
    plan: Mapping[str, Any],
    repository_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = Path(repository_root).resolve()
    bindings = plan["bindings"]
    preflight_plan_path = resolve_binding(
        root, bindings["preflight_plan"], name="preflight_plan"
    )
    preflight_plan = load_promotion_preflight_plan(
        preflight_plan_path,
        repository_root=root,
    )
    if plan.get("source_after") != preflight_plan.get("source_after"):
        raise ValueError("OpenCodeInstruct source proposal changed after preflight")

    preflight_report = _load_json(
        resolve_binding(root, bindings["preflight_report"], name="preflight_report")
    )
    validate_report(preflight_report)
    projected = preflight_report.get("projected_successor", {})
    if (
        preflight_report.get("report_sha256") != PREFLIGHT_REPORT_INTERNAL_SHA256
        or preflight_report.get("status")
        != "verified_local_preflight_registry_successor_not_materialized"
        or projected.get("sha256") != SUCCESSOR_REGISTRY_SHA256
        or projected.get("bytes") != SUCCESSOR_REGISTRY_BYTES
        or projected.get("source_count") != 14
    ):
        raise ValueError("OpenCodeInstruct authorized preflight report changed")

    verification = _load_json(
        resolve_binding(
            root,
            bindings["preflight_verification"],
            name="preflight_verification",
        )
    )
    if (
        _verify_internal_seal(verification, "verification_sha256")
        != PREFLIGHT_VERIFICATION_INTERNAL_SHA256
        or verification.get("status")
        != "verified_preflight_source_registry_successor_not_materialized"
        or verification.get("projected_successor_sha256")
        != SUCCESSOR_REGISTRY_SHA256
        or verification.get("report_file_sha256") != PREFLIGHT_REPORT_FILE_SHA256
        or verification.get("report_internal_sha256")
        != PREFLIGHT_REPORT_INTERNAL_SHA256
    ):
        raise ValueError("OpenCodeInstruct authorized preflight verification changed")

    predecessor = _load_json(
        resolve_binding(
            root,
            bindings["predecessor_registry"],
            name="predecessor_registry",
        )
    )
    return preflight_plan, preflight_report, predecessor


def build_and_validate_successor(
    *,
    plan: Mapping[str, Any],
    predecessor: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    successor = build_projected_successor(
        predecessor,
        source_after=plan["source_after"],
    )
    encoded = json_file_bytes(successor)
    if len(encoded) != SUCCESSOR_REGISTRY_BYTES:
        raise ValueError("OpenCodeInstruct successor byte count changed")
    if hashlib.sha256(encoded).hexdigest() != SUCCESSOR_REGISTRY_SHA256:
        raise ValueError("OpenCodeInstruct successor SHA-256 changed")
    before_sources = predecessor.get("sources")
    after_sources = successor.get("sources")
    if not isinstance(before_sources, list) or not isinstance(after_sources, list):
        raise ValueError("OpenCodeInstruct registry source arrays changed")
    if len(before_sources) != 13 or len(after_sources) != 14:
        raise ValueError("OpenCodeInstruct registry source count changed")
    if after_sources[:-1] != before_sources or after_sources[-1] != plan["source_after"]:
        raise ValueError("OpenCodeInstruct successor changed non-authorized sources")

    sources = [
        SFTSource.from_mapping(
            source,
            allowed_licenses=frozenset(policy["allowed_licenses"]),
            allowed_languages=frozenset(policy["allowed_languages"]),
            allowed_splits=frozenset(policy["allowed_splits"]),
            allowed_tasks=frozenset(policy["task_mix"]),
        )
        for source in after_sources
    ]
    source_ids = [source.source_id for source in sources]
    dataset_ids = [source.dataset_id for source in sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("OpenCodeInstruct successor source IDs are not unique")
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("OpenCodeInstruct successor dataset IDs are not unique")
    if source_ids[-1] != SOURCE_ID or sources[-1].production_eligible:
        raise ValueError("OpenCodeInstruct successor production boundary changed")
    return successor


def build_promotion_receipt(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    preflight_report: Mapping[str, Any],
) -> dict[str, Any]:
    _require_sha256(plan_file_sha256, "plan_file_sha256")
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
        "candidate": {
            "records": 11_100,
            "train_records": 10_553,
            "train_supervised_tokens": 2_100_077,
            "validation_records": 547,
            "validation_supervised_tokens": 110_619,
        },
        "checks": {
            "active_data_contract_updated": False,
            "added_source_ids": [SOURCE_ID],
            "candidate_registry_promotions": 1,
            "predecessor_registry_mutated": False,
            "production_dataset_promotions": 0,
            "raw_training_text_retained": False,
        },
        "observed_at": "2026-08-07",
        "production_eligible": False,
        "registry": {
            "predecessor_path": "configs/lumi-4b-sft-source-registry-v0.10.json",
            "predecessor_sha256": PREDECESSOR_REGISTRY_SHA256,
            "successor_bytes": SUCCESSOR_REGISTRY_BYTES,
            "successor_path": EXPECTED_OUTPUT["successor_registry_path"],
            "successor_sha256": SUCCESSOR_REGISTRY_SHA256,
        },
        "remaining_blockers": list(plan["source_after"]["production_blockers"]),
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "source_after": dict(plan["source_after"]),
        "status": "verified_candidate_registry_v0_11_nonproduction",
    }
    receipt["receipt_sha256"] = sha256_value(receipt)
    return receipt


def validate_promotion_receipt(receipt: Mapping[str, Any]) -> None:
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported OpenCodeInstruct promotion receipt")
    _verify_internal_seal(receipt, "receipt_sha256")
    if receipt.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("OpenCodeInstruct promotion receipt authorization changed")
    if receipt.get("status") != "verified_candidate_registry_v0_11_nonproduction":
        raise ValueError("OpenCodeInstruct promotion receipt status changed")
    if receipt.get("production_eligible") is not False:
        raise ValueError("OpenCodeInstruct promotion crossed production boundary")
    checks = receipt.get("checks")
    if not isinstance(checks, Mapping) or checks != {
        "active_data_contract_updated": False,
        "added_source_ids": [SOURCE_ID],
        "candidate_registry_promotions": 1,
        "predecessor_registry_mutated": False,
        "production_dataset_promotions": 0,
        "raw_training_text_retained": False,
    }:
        raise ValueError("OpenCodeInstruct promotion receipt checks changed")


__all__ = [
    "EXPECTED_AUTHORIZATION",
    "EXPECTED_LIMITS",
    "EXPECTED_OUTPUT",
    "PLAN_SCHEMA_VERSION",
    "PREDECESSOR_REGISTRY_SHA256",
    "RECEIPT_SCHEMA_VERSION",
    "SUCCESSOR_REGISTRY_BYTES",
    "SUCCESSOR_REGISTRY_SHA256",
    "VERIFICATION_SCHEMA_VERSION",
    "build_and_validate_successor",
    "build_promotion_receipt",
    "json_file_bytes",
    "load_promotion_plan",
    "resolve_binding",
    "validate_promotion_receipt",
    "verify_promotion_prerequisites",
]
