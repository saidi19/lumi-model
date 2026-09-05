"""Independent replay for the materialized AI-assisted v0.3 registry successor."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .posttraining_ai_v3_source_promotion import (
    SUCCESSOR_SHA256,
    build_source_promotion_receipt,
    build_source_promotion_verification,
    canonical_sha256,
    file_sha256,
    load_ai_v3_source_promotion_plan,
    read_object,
    reconstruct_authorized_successor,
    verify_source_promotion_receipt,
)

EXPECTED_CHECKS = {
    "active_production_data_contract_updated": False,
    "candidate_rows_modified": False,
    "predecessor_registry_unchanged": True,
    "production_dataset_promotions": 0,
    "receipt_reconstructed": True,
    "successor_adds_exactly_one_source": True,
    "successor_registry_reconstructed": True,
}


def verify_independent_materialization(
    plan_path: Path,
    *,
    repository_root: Path,
) -> dict[str, Any]:
    """Reconstruct every output without mutating the registry or receipt."""

    root = repository_root.resolve()
    plan = load_ai_v3_source_promotion_plan(plan_path, repository_root=root)
    predecessor, expected_successor, preflight = reconstruct_authorized_successor(
        plan,
        repository_root=root,
    )
    successor_path = (root / plan["output"]["successor_registry_path"]).resolve()
    actual_successor = read_object(successor_path)
    if actual_successor != expected_successor:
        raise ValueError("AI v0.3 successor registry reconstruction changed")
    if file_sha256(successor_path) != SUCCESSOR_SHA256:
        raise ValueError("AI v0.3 successor registry file hash changed")
    if actual_successor["sources"][:12] != predecessor["sources"]:
        raise ValueError("AI v0.3 predecessor registry entries changed")
    if len(actual_successor["sources"]) != 13:
        raise ValueError("AI v0.3 successor source count changed")

    receipt_path = (root / plan["output"]["receipt_path"]).resolve()
    actual_receipt = read_object(receipt_path)
    verify_source_promotion_receipt(actual_receipt)
    expected_receipt = build_source_promotion_receipt(
        plan,
        plan_file_sha256=file_sha256(plan_path),
        predecessor=predecessor,
        successor=expected_successor,
        preflight_report=preflight,
    )
    if actual_receipt != expected_receipt:
        raise ValueError("AI v0.3 source-promotion receipt reconstruction changed")

    verification = build_source_promotion_verification(
        plan=plan,
        receipt_file_sha256=file_sha256(receipt_path),
        receipt_internal_sha256=actual_receipt["receipt_sha256"],
    )
    if verification.get("checks") != EXPECTED_CHECKS:
        raise ValueError("AI v0.3 independent verification checks changed")
    unsigned = dict(verification)
    claimed = unsigned.pop("verification_sha256", None)
    if claimed != canonical_sha256(unsigned):
        raise ValueError("AI v0.3 independent verification seal mismatch")
    if verification.get("successor_registry_sha256") != SUCCESSOR_SHA256:
        raise ValueError("AI v0.3 independent successor hash changed")
    if verification.get("production_eligible") is not False:
        raise ValueError("AI v0.3 independent verification became production eligible")
    return verification


def verify_independent_verification_report(report: Mapping[str, Any]) -> None:
    """Validate a persisted independent verification report."""

    if report.get("schema_version") != (
        "lumi.sft-ai-v3-source-promotion-verification.v0.1"
    ):
        raise ValueError("unsupported AI v0.3 independent verification report")
    if report.get("checks") != EXPECTED_CHECKS:
        raise ValueError("AI v0.3 persisted verification checks changed")
    unsigned = dict(report)
    claimed = unsigned.pop("verification_sha256", None)
    if claimed != canonical_sha256(unsigned):
        raise ValueError("AI v0.3 persisted verification seal mismatch")
