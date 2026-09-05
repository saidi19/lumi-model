"""Owner-review finalization for the verified DeepMind Mathematics scale candidate."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .posttraining_deepmind_scale_generation import file_sha256, sha256_value
from .posttraining_owner_review import validate_owner_decisions as validate_generic_decisions

PLAN_SCHEMA_VERSION = "lumi.sft-deepmind-scale-owner-review-plan.v0.1"
RECEIPT_SCHEMA_VERSION = "lumi.sft-deepmind-scale-owner-review-receipt.v0.1"
EXPECTED_SOURCE_REVISION = "427f45075f84b8b9774950196ad63867ca20ffb3"
EXPECTED_AUTHORIZATION = {
    "candidate_registry_promotion_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "gpu_authorized": False,
    "network_authorized": False,
    "owner_review_finalization_authorized": True,
    "sft_training_authorized": False,
    "unrelated_access_authorized": False,
}


def _require_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be lowercase SHA-256")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            rows.append(value)
    return rows


def resolve_binding(root: Path, specification: Mapping[str, Any]) -> Path:
    path = (root / str(specification.get("path", ""))).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"DeepMind owner-review binding is missing: {path}")
    expected = _require_sha256(specification.get("sha256"), "binding.sha256")
    if file_sha256(path) != expected:
        raise ValueError(f"DeepMind owner-review binding changed: {path}")
    return path


def load_owner_review_plan(path: Path, *, repository_root: Path) -> dict[str, Any]:
    plan = _read_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported DeepMind scale owner-review plan")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("DeepMind scale owner-review authorization changed")
    if plan.get("source_revision") != EXPECTED_SOURCE_REVISION:
        raise ValueError("DeepMind scale owner-review source revision changed")
    candidate = plan.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("DeepMind scale owner-review candidate binding is missing")
    expected_candidate = {
        "records": 236_698,
        "train_records": 224_688,
        "train_supervised_tokens": 2_221_113,
        "validation_records": 12_010,
        "validation_supervised_tokens": 118_053,
    }
    if candidate != expected_candidate:
        raise ValueError("DeepMind scale owner-review candidate counts changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "accepted_records",
        "finalizer",
        "generic_owner_review_implementation",
        "owner_review_implementation",
        "owner_sample",
        "scale_report",
        "scale_verification",
    }:
        raise ValueError("DeepMind scale owner-review binding inventory changed")
    for specification in bindings.values():
        if not isinstance(specification, Mapping):
            raise ValueError("DeepMind scale owner-review binding must be an object")
        resolve_binding(repository_root, specification)
    return plan


def validate_bound_candidate(
    plan: Mapping[str, Any], *, repository_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    bindings = plan["bindings"]
    sample_path = resolve_binding(repository_root, bindings["owner_sample"])
    sample = read_jsonl(sample_path)
    if len(sample) != 32:
        raise ValueError("DeepMind scale owner sample must contain exactly 32 rows")
    sample_sha256 = sha256_value(sample)
    expected_sample = _require_sha256(
        bindings["owner_sample"].get("internal_sha256"),
        "owner_sample.internal_sha256",
    )
    if sample_sha256 != expected_sample:
        raise ValueError("DeepMind scale owner sample content changed")

    report = _read_json(resolve_binding(repository_root, bindings["scale_report"]))
    verification = _read_json(
        resolve_binding(repository_root, bindings["scale_verification"])
    )
    if report.get("report_sha256") != bindings["scale_report"].get("internal_sha256"):
        raise ValueError("DeepMind scale report internal hash changed")
    if verification.get("verification_sha256") != bindings["scale_verification"].get(
        "internal_sha256"
    ):
        raise ValueError("DeepMind scale verification internal hash changed")
    if verification.get("status") != "verified_cpu_only_nonproduction_deepmind_scale_candidate":
        raise ValueError("DeepMind scale candidate is not independently verified")
    if verification.get("production_eligible") is not False:
        raise ValueError("DeepMind scale verification crossed production eligibility")
    if verification.get("retained_records") != plan["candidate"]["records"]:
        raise ValueError("DeepMind scale retained-record count changed")
    if verification.get("retained_train_supervised_tokens") != plan["candidate"][
        "train_supervised_tokens"
    ]:
        raise ValueError("DeepMind scale train-token count changed")
    if report["owner_review"].get("sample_sha256") != sample_sha256:
        raise ValueError("DeepMind scale report owner sample changed")
    return sample, report, verification


def validate_owner_decisions(
    value: Mapping[str, Any],
    *,
    sample_rows: list[Mapping[str, Any]],
    plan: Mapping[str, Any],
    decision_file_sha256: str,
    plan_file_sha256: str,
) -> dict[str, Any]:
    """Return a sealed, raw-text-free, non-promoting owner-review receipt."""

    decision_file_sha256 = _require_sha256(
        decision_file_sha256, "decision_file_sha256"
    )
    plan_file_sha256 = _require_sha256(plan_file_sha256, "plan_file_sha256")
    sample_sha256 = sha256_value(sample_rows)
    generic = validate_generic_decisions(
        value,
        sample_rows=sample_rows,
        sample_sha256=sample_sha256,
    )
    decision_ids = sorted(str(decision["record_id"]) for decision in value["decisions"])
    bindings = plan["bindings"]
    receipt: dict[str, Any] = {
        "accepted_records": generic["accepted_records"],
        "all_sampled_rows_accepted": generic["all_sampled_rows_accepted"],
        "authorization": EXPECTED_AUTHORIZATION,
        "bindings": {
            "accepted_records_file_sha256": bindings["accepted_records"]["sha256"],
            "decision_file_sha256": decision_file_sha256,
            "decision_internal_sha256": sha256_value(value),
            "owner_sample_file_sha256": bindings["owner_sample"]["sha256"],
            "owner_sample_internal_sha256": sample_sha256,
            "plan_file_sha256": plan_file_sha256,
            "report_file_sha256": bindings["scale_report"]["sha256"],
            "report_internal_sha256": bindings["scale_report"]["internal_sha256"],
            "reviewed_record_ids_sha256": sha256_value(decision_ids),
            "verification_file_sha256": bindings["scale_verification"]["sha256"],
            "verification_internal_sha256": bindings["scale_verification"][
                "internal_sha256"
            ],
        },
        "candidate": dict(plan["candidate"]),
        "owner_review_claim": (
            "upstream_deepmind_generated_math_plus_human_owner_quality_reviewed_"
            "not_human_written"
        ),
        "production_eligible": False,
        "production_promotions": 0,
        "raw_training_text_retained": False,
        "rejected_record_ids": generic["rejected_record_ids"],
        "rejected_records": generic["rejected_records"],
        "remaining_blockers": [
            "deepmind_scale_candidate_registry_promotion_not_authorized",
            "complete_sft_learning_mix_not_ready",
            "production_dataset_seal_and_sft_not_authorized",
        ],
        "reviewed_at": generic["reviewed_at"],
        "reviewer_id": generic["reviewer_id"],
        "sample_records": generic["sample_records"],
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "source_revision": EXPECTED_SOURCE_REVISION,
        "status": (
            "owner_quality_sample_passed_source_promotion_still_separate"
            if generic["all_sampled_rows_accepted"]
            else "owner_quality_sample_rejected_rows_require_repair"
        ),
    }
    receipt["receipt_sha256"] = sha256_value(receipt)
    return receipt


def validate_owner_review_receipt(receipt: Mapping[str, Any]) -> None:
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported DeepMind scale owner-review receipt")
    seal = _require_sha256(receipt.get("receipt_sha256"), "receipt_sha256")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if sha256_value(unsigned) != seal:
        raise ValueError("DeepMind scale owner-review receipt seal is invalid")
    if receipt.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("DeepMind scale owner-review receipt authorization changed")
    if receipt.get("raw_training_text_retained") is not False:
        raise ValueError("DeepMind scale owner-review receipt retained raw text")
    if receipt.get("production_eligible") is not False:
        raise ValueError("DeepMind scale owner review crossed production eligibility")
    if receipt.get("production_promotions") != 0:
        raise ValueError("DeepMind scale owner review crossed promotion boundary")
    accepted = receipt.get("accepted_records")
    rejected = receipt.get("rejected_records")
    if not isinstance(accepted, int) or not isinstance(rejected, int):
        raise ValueError("DeepMind scale owner-review counts are invalid")
    if accepted + rejected != 32:
        raise ValueError("DeepMind scale owner-review sample count changed")


__all__ = [
    "EXPECTED_AUTHORIZATION",
    "PLAN_SCHEMA_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "load_owner_review_plan",
    "read_jsonl",
    "resolve_binding",
    "validate_bound_candidate",
    "validate_owner_decisions",
    "validate_owner_review_receipt",
]
