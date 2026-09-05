"""Immutable registry promotion for the reviewed OASST2 candidate."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .data_quality import sha256_file
from .posttraining_oasst2_curation import sha256_value
from .posttraining_oasst2_source_promotion_preflight import (
    build_projected_successor,
    load_promotion_preflight_plan,
    validate_report,
)
from .posttraining_sources import SFTSource

PLAN_SCHEMA_VERSION = "lumi.sft-oasst2-source-promotion-plan.v0.1"
RECEIPT_SCHEMA_VERSION = "lumi.sft-oasst2-source-promotion-receipt.v0.1"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-oasst2-source-promotion-verification.v0.1"

PREDECESSOR_REGISTRY_SHA256 = "8f286b096ee2e406fe4ee0a6d69ea37a8e7017e489fda6c765df46ed2d1b86b7"
PREFLIGHT_PLAN_FILE_SHA256 = "b699826a834a932e6a5efb0f3533ac53543df7ae18e22a1dd5f2a11c3c01d32d"
PREFLIGHT_REPORT_FILE_SHA256 = "24047e848657f84bb37a54825270b2eb2b69abef686da4d652d6d67988f7f59a"
PREFLIGHT_REPORT_INTERNAL_SHA256 = (
    "d9eeda5081b48bdc3140daaaef90e321f10de9102a1191bc3c24e057d93f04ee"
)
PREFLIGHT_VERIFICATION_FILE_SHA256 = (
    "5a91e010644be2d3e8c99177161b51b7d0818b2fd434791a8f92e197b7084ccc"
)
PREFLIGHT_VERIFICATION_INTERNAL_SHA256 = (
    "b95f7924f64c8cbcd599745fb40e544e10c24a0c841b6cc24d4e005d355ae054"
)
SUCCESSOR_REGISTRY_SHA256 = "1e8f7b0d2f1bfc10118fd76a51813e942cc7d0ce558f73ebfdc0382e6ee2b077"
SUCCESSOR_REGISTRY_BYTES = 17_725
SOURCE_ID = "openassistant_oasst2"

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
    "purpose": "exact_reviewed_oasst2_registry_v0_13_successor_only",
    "unrelated_access_authorized": False,
}
EXPECTED_LIMITS = {
    "added_sources": 1,
    "candidate_records": 6_426,
    "maximum_runtime_minutes": 30,
    "predecessor_sources": 14,
    "successor_sources": 15,
    "train_supervised_tokens": 2_221_091,
    "validation_supervised_tokens": 96_436,
}
EXPECTED_OUTPUT = {
    "active_data_contract_update_permitted": False,
    "predecessor_registry_mutation_permitted": False,
    "production_dataset_promotions": 0,
    "raw_training_text_in_receipts": False,
    "receipt_path": "evidence/lumi-4b-sft-oasst2-source-promotion-v0.1.json",
    "successor_registry_path": "configs/lumi-4b-sft-source-registry-v0.13.json",
    "verification_path": ("evidence/lumi-4b-sft-oasst2-source-promotion-verification-v0.1.json"),
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
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _verify_internal_seal(value: Mapping[str, Any], field: str) -> str:
    seal = _require_sha256(value.get(field), field)
    unsigned = {key: item for key, item in value.items() if key != field}
    if sha256_value(unsigned) != seal:
        raise ValueError(f"OASST2 {field} seal changed")
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
        raise ValueError(f"invalid OASST2 promotion binding: {name}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.parent.resolve()) or not resolved.is_file():
        raise FileNotFoundError(f"OASST2 promotion binding missing: {name}")
    if sha256_file(resolved) != expected:
        raise ValueError(f"OASST2 promotion binding changed: {name}")
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
        "observed_at": "2026-08-08",
        "output": EXPECTED_OUTPUT,
        "predecessor_registry_sha256": PREDECESSOR_REGISTRY_SHA256,
        "proposed_successor_registry_sha256": SUCCESSOR_REGISTRY_SHA256,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "authorized_exact_candidate_registry_v0_13_successor_promotion",
    }
    for field, expected_value in expected.items():
        if plan.get(field) != expected_value:
            raise ValueError(f"OASST2 promotion plan changed: {field}")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("OASST2 promotion binding inventory changed")
    for name, specification in bindings.items():
        if not isinstance(specification, Mapping):
            raise ValueError(f"OASST2 promotion binding invalid: {name}")
        resolve_binding(root, specification, name=name)
    if bindings["predecessor_registry"]["sha256"] != PREDECESSOR_REGISTRY_SHA256:
        raise ValueError("OASST2 predecessor authorization changed")
    if bindings["preflight_plan"]["sha256"] != PREFLIGHT_PLAN_FILE_SHA256:
        raise ValueError("OASST2 preflight-plan authorization changed")
    if bindings["preflight_report"]["sha256"] != PREFLIGHT_REPORT_FILE_SHA256:
        raise ValueError("OASST2 preflight-report authorization changed")
    if bindings["preflight_verification"]["sha256"] != PREFLIGHT_VERIFICATION_FILE_SHA256:
        raise ValueError("OASST2 preflight-verification authorization changed")
    return plan


def verify_promotion_prerequisites(
    *,
    plan: Mapping[str, Any],
    repository_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = Path(repository_root).resolve()
    bindings = plan["bindings"]
    preflight_plan_path = resolve_binding(root, bindings["preflight_plan"], name="preflight_plan")
    preflight_plan = load_promotion_preflight_plan(preflight_plan_path, repository_root=root)
    if plan.get("source_after") != preflight_plan.get("source_after"):
        raise ValueError("OASST2 source proposal changed after preflight")

    preflight_report = _load_json(
        resolve_binding(root, bindings["preflight_report"], name="preflight_report")
    )
    validate_report(preflight_report)
    projected = preflight_report.get("projected_successor", {})
    if (
        preflight_report.get("report_sha256") != PREFLIGHT_REPORT_INTERNAL_SHA256
        or preflight_report.get("status")
        != "verified_local_preflight_registry_v0_13_successor_not_materialized"
        or projected.get("sha256") != SUCCESSOR_REGISTRY_SHA256
        or projected.get("bytes") != SUCCESSOR_REGISTRY_BYTES
        or projected.get("source_count") != 15
    ):
        raise ValueError("OASST2 authorized preflight report changed")

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
        != "verified_preflight_registry_v0_13_successor_not_materialized"
        or verification.get("projected_successor_sha256") != SUCCESSOR_REGISTRY_SHA256
        or verification.get("report_file_sha256") != PREFLIGHT_REPORT_FILE_SHA256
        or verification.get("report_internal_sha256") != PREFLIGHT_REPORT_INTERNAL_SHA256
    ):
        raise ValueError("OASST2 authorized preflight verification changed")

    predecessor = _load_json(
        resolve_binding(root, bindings["predecessor_registry"], name="predecessor_registry")
    )
    return preflight_plan, preflight_report, predecessor


def build_and_validate_successor(
    *,
    plan: Mapping[str, Any],
    predecessor: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    successor = build_projected_successor(predecessor, source_after=plan["source_after"])
    encoded = json_file_bytes(successor)
    if len(encoded) != SUCCESSOR_REGISTRY_BYTES:
        raise ValueError("OASST2 successor byte count changed")
    if hashlib.sha256(encoded).hexdigest() != SUCCESSOR_REGISTRY_SHA256:
        raise ValueError("OASST2 successor SHA-256 changed")
    before_sources = predecessor.get("sources")
    after_sources = successor.get("sources")
    if not isinstance(before_sources, list) or not isinstance(after_sources, list):
        raise ValueError("OASST2 registry source arrays changed")
    if len(before_sources) != 14 or len(after_sources) != 15:
        raise ValueError("OASST2 registry source count changed")
    if after_sources[:-1] != before_sources or after_sources[-1] != plan["source_after"]:
        raise ValueError("OASST2 successor changed non-authorized sources")

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
        raise ValueError("OASST2 successor source IDs are not unique")
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("OASST2 successor dataset IDs are not unique")
    if source_ids[-1] != SOURCE_ID or sources[-1].production_eligible:
        raise ValueError("OASST2 successor production boundary changed")
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
            "preflight_verification_file_sha256": (PREFLIGHT_VERIFICATION_FILE_SHA256),
            "preflight_verification_internal_sha256": (PREFLIGHT_VERIFICATION_INTERNAL_SHA256),
        },
        "candidate": {
            "records": 6_426,
            "train_records": 6_136,
            "train_supervised_tokens": 2_221_091,
            "validation_records": 290,
            "validation_supervised_tokens": 96_436,
        },
        "checks": {
            "active_data_contract_updated": False,
            "added_source_ids": [SOURCE_ID],
            "candidate_registry_promotions": 1,
            "predecessor_registry_mutated": False,
            "production_dataset_promotions": 0,
            "raw_training_text_retained": False,
        },
        "observed_at": "2026-08-08",
        "production_eligible": False,
        "registry": {
            "predecessor_path": "configs/lumi-4b-sft-source-registry-v0.12.json",
            "predecessor_sha256": PREDECESSOR_REGISTRY_SHA256,
            "successor_bytes": SUCCESSOR_REGISTRY_BYTES,
            "successor_path": EXPECTED_OUTPUT["successor_registry_path"],
            "successor_sha256": SUCCESSOR_REGISTRY_SHA256,
        },
        "remaining_blockers": list(plan["source_after"]["production_blockers"]),
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "source_after": dict(plan["source_after"]),
        "status": "verified_candidate_registry_v0_13_nonproduction",
    }
    receipt["receipt_sha256"] = sha256_value(receipt)
    return receipt


def validate_promotion_receipt(receipt: Mapping[str, Any]) -> None:
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported OASST2 promotion receipt")
    _verify_internal_seal(receipt, "receipt_sha256")
    if receipt.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("OASST2 promotion receipt authorization changed")
    if receipt.get("status") != "verified_candidate_registry_v0_13_nonproduction":
        raise ValueError("OASST2 promotion receipt status changed")
    if receipt.get("production_eligible") is not False:
        raise ValueError("OASST2 promotion crossed production boundary")
    checks = receipt.get("checks")
    if not isinstance(checks, Mapping) or checks != {
        "active_data_contract_updated": False,
        "added_source_ids": [SOURCE_ID],
        "candidate_registry_promotions": 1,
        "predecessor_registry_mutated": False,
        "production_dataset_promotions": 0,
        "raw_training_text_retained": False,
    }:
        raise ValueError("OASST2 promotion receipt checks changed")


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
