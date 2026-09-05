"""Fail-closed human-owner review finalization for the curated DMath candidate."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .posttraining_dmath_curation import sha256_value
from .posttraining_first_party import file_sha256

PLAN_SCHEMA_VERSION = "lumi.sft-dmath-owner-review-plan.v0.1"
DECISIONS_SCHEMA_VERSION = "lumi.dmath-owner-decisions.v0.1"
RECEIPT_SCHEMA_VERSION = "lumi.sft-dmath-owner-review-receipt.v0.1"

EXPECTED_AUTHORIZATION = {
    "ai_or_automated_owner_approval_authorized": False,
    "candidate_registry_promotion_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_spend_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "purpose": "private_human_owner_quality_spot_check_only",
}
EXPECTED_CANDIDATE = {
    "accepted_records": 6943,
    "accepted_records_file_sha256": (
        "8c523314c7051df20e2334faf4f4c95d8e9f342058f445dbc28db6f83e7b494d"
    ),
    "curation_report_file_sha256": (
        "eba3135a6072ea55cee2b1e16630d54d1c9d95fd30845eda1896509fb4620f45"
    ),
    "curation_report_internal_sha256": (
        "c00b88c977a95f6bd8f49350a210dce0954c9255ca5b35730ab84001df3415c4"
    ),
    "curation_run_plan_file_sha256": (
        "7a58eb9e6df2097219a91d19fddc315009b06f7a2ae72de2f8b17efd61b5bb75"
    ),
    "curation_verification_file_sha256": (
        "cacd900762e4414809105fb1358cb5fd588703aaab6f65d13c60339da87fceb8"
    ),
    "curation_verification_internal_sha256": (
        "05a22df0cb0d84483a8bb9f0e6f395228ef3bc706c1ec57da253321a5eef2132"
    ),
    "owner_review_html_file_sha256": (
        "22462211c09569d58bf74c322e5a78a60c2608fb73bc1c45a91020f564ebd3a3"
    ),
    "owner_sample_file_sha256": (
        "a90b3e46703d775252e9d3c7b376f32894260c15c5590b9769764c8d9ccac9d9"
    ),
    "owner_sample_internal_sha256": (
        "f4e2cbf6d068b4a69eb2bae7829de8acd7bffcd8d29e095aa9ba9ffbe32b806b"
    ),
    "source_revision": "149a7403619ac1bd6ed10f50694ad82699d5cc03",
    "supervised_tokens": 51026,
    "train_records": 6596,
    "train_supervised_tokens": 48480,
    "validation_records": 347,
    "validation_supervised_tokens": 2546,
}
EXPECTED_LIMITS = {"records": 32}
EXPECTED_OUTPUT = {
    "decision_filename": "lumi-dmath-owner-decisions-v0.1.json",
    "production_promotion_permitted": False,
    "receipt_contains_raw_training_text": False,
    "receipt_path": "evidence/lumi-4b-sft-dmath-owner-review-v0.1.json",
    "verification_path": (
        "evidence/lumi-4b-sft-dmath-owner-review-verification-v0.1.json"
    ),
}
EXPECTED_BINDINGS = frozenset(
    {
        "accepted_records",
        "curation_report",
        "curation_run_plan",
        "curation_verification",
        "owner_review_html",
        "owner_sample",
        "receipt_finalizer",
        "receipt_verifier",
        "review_implementation",
    }
)
EXPECTED_CATEGORIES = {
    "Arithmetic calculation",
    "Comparison",
    "Correspondence",
    "Geometry",
    "Possibility",
}
EXPECTED_COVERAGE = {
    ("train", "Arithmetic calculation"): 4,
    ("train", "Comparison"): 4,
    ("train", "Correspondence"): 3,
    ("train", "Geometry"): 3,
    ("train", "Possibility"): 3,
    ("validation", "Arithmetic calculation"): 3,
    ("validation", "Comparison"): 3,
    ("validation", "Correspondence"): 3,
    ("validation", "Geometry"): 3,
    ("validation", "Possibility"): 3,
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UTC_TIMESTAMP = re.compile(
    r"^20[0-9]{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])T"
    r"(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z$"
)
_BANNED_REVIEWER_MARKERS = {"chatgpt", "claude", "codex", "gemini"}


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a SHA-256")
    return value


def resolve_plan_binding(
    repository_root: str | Path,
    binding: Mapping[str, Any],
) -> Path:
    root = Path(repository_root).resolve()
    workspace = root.parent.resolve()
    relative = binding.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("DMath owner-review binding path is invalid")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError("DMath owner-review binding escapes workspace")
    return resolved


def load_owner_review_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Load the exact DMath review plan and verify every bound object."""

    root = Path(repository_root).resolve()
    plan = _load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported DMath owner-review plan")
    if plan.get("status") != "human_owner_review_ready_no_promotion":
        raise ValueError("DMath owner-review status changed")
    if plan.get("observed_at") != "2026-08-06":
        raise ValueError("DMath owner-review date changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("DMath owner-review authorization changed")
    if plan.get("candidate") != EXPECTED_CANDIDATE:
        raise ValueError("DMath owner-review candidate changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("DMath owner-review limits changed")
    if plan.get("output") != EXPECTED_OUTPUT:
        raise ValueError("DMath owner-review output boundary changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("DMath owner-review binding set changed")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"DMath owner-review binding must be an object: {name}")
        expected = _require_sha256(binding.get("sha256"), f"bindings.{name}.sha256")
        resolved = resolve_plan_binding(root, binding)
        if not resolved.is_file() or file_sha256(resolved) != expected:
            raise ValueError(f"DMath owner-review binding changed: {name}")
    return plan


def load_exact_owner_sample(
    path: str | Path,
    *,
    curation_report: Mapping[str, Any],
    verification: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Load and verify the exact deterministic 32-row DMath sample."""

    report_unsigned = {
        key: value for key, value in curation_report.items() if key != "report_sha256"
    }
    if curation_report.get("report_sha256") != sha256_value(report_unsigned):
        raise ValueError("DMath curation report seal changed")
    if curation_report.get("report_sha256") != EXPECTED_CANDIDATE[
        "curation_report_internal_sha256"
    ]:
        raise ValueError("DMath curation report binding changed")
    if curation_report.get("status") != "curated_nonproduction_owner_review_pending":
        raise ValueError("DMath curation report status changed")
    if curation_report.get("owner_review") != {
        "complete": False,
        "record_count": 32,
        "sample_sha256": EXPECTED_CANDIDATE["owner_sample_internal_sha256"],
        "status": "pending_owner_review",
    }:
        raise ValueError("DMath owner-review boundary changed")
    if curation_report.get("production") != {
        "candidate_registry_promotions": 0,
        "dataset_sealed": False,
        "production_eligible": False,
        "production_sft_authorized": False,
    }:
        raise ValueError("DMath curation crossed a production boundary")

    verification_unsigned = {
        key: value for key, value in verification.items() if key != "verification_sha256"
    }
    if verification.get("verification_sha256") != sha256_value(verification_unsigned):
        raise ValueError("DMath curation verification seal changed")
    if verification.get("verification_sha256") != EXPECTED_CANDIDATE[
        "curation_verification_internal_sha256"
    ]:
        raise ValueError("DMath curation verification binding changed")
    if verification.get("status") != "verified_nonproduction_owner_review_pending":
        raise ValueError("DMath curation verification status changed")
    if verification.get("production_eligible") is not False:
        raise ValueError("DMath curation unexpectedly claims production eligibility")

    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != EXPECTED_LIMITS["records"]:
        raise ValueError("DMath owner-review sample count changed")
    if sha256_value(rows) != EXPECTED_CANDIDATE["owner_sample_internal_sha256"]:
        raise ValueError("DMath owner-review sample content changed")

    record_ids: set[str] = set()
    coverage: Counter[tuple[str, str]] = Counter()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("DMath owner-review sample row changed")
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id or record_id in record_ids:
            raise ValueError("DMath owner-review record identity changed")
        split = row.get("split")
        category = row.get("source_category")
        if split not in {"train", "validation"} or category not in EXPECTED_CATEGORIES:
            raise ValueError("DMath owner-review coverage changed")
        if row.get("schema_version") != "lumi.sft.v0.1" or row.get("language") != "en":
            raise ValueError("DMath owner-review record schema changed")
        source = row.get("source")
        if (
            not isinstance(source, Mapping)
            or source.get("source_id") != "dmath"
            or source.get("revision") != EXPECTED_CANDIDATE["source_revision"]
        ):
            raise ValueError("DMath owner-review source binding changed")
        messages = row.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) != 2
            or [message.get("role") for message in messages] != ["user", "assistant"]
        ):
            raise ValueError("DMath owner-review message structure changed")
        record_ids.add(record_id)
        coverage[(str(split), str(category))] += 1
    if dict(coverage) != EXPECTED_COVERAGE:
        raise ValueError("DMath owner-review split/category balance changed")
    return rows


def validate_owner_decisions(
    value: Mapping[str, Any],
    *,
    sample_rows: Sequence[Mapping[str, Any]],
    decision_file_sha256: str,
    plan_file_sha256: str,
) -> dict[str, Any]:
    """Validate owner decisions and return a raw-text-free non-promoting receipt."""

    _require_sha256(decision_file_sha256, "decision_file_sha256")
    _require_sha256(plan_file_sha256, "plan_file_sha256")
    sample_sha256 = sha256_value(list(sample_rows))
    if value.get("schema_version") != DECISIONS_SCHEMA_VERSION:
        raise ValueError("unsupported DMath owner decision schema")
    if value.get("sample_sha256") != sample_sha256:
        raise ValueError("DMath owner decision sample changed")
    reviewer = value.get("reviewer_id")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("DMath owner reviewer alias is required")
    normalized_reviewer = reviewer.strip().casefold()
    if any(marker in normalized_reviewer for marker in _BANNED_REVIEWER_MARKERS):
        raise ValueError("an AI system cannot claim DMath owner review")
    reviewed_at = value.get("reviewed_at")
    if not isinstance(reviewed_at, str) or _UTC_TIMESTAMP.fullmatch(reviewed_at) is None:
        raise ValueError("DMath owner reviewed_at must be exact UTC ISO-8601")
    decisions = value.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != len(sample_rows):
        raise ValueError("DMath owner decisions must cover every sampled row")

    expected_ids = {str(row["record_id"]) for row in sample_rows}
    observed_ids: set[str] = set()
    accepted_ids: list[str] = []
    rejected_ids: list[str] = []
    for decision in decisions:
        if not isinstance(decision, Mapping) or set(decision) != {
            "accepted",
            "note",
            "record_id",
        }:
            raise ValueError("DMath owner decision fields changed")
        record_id = decision.get("record_id")
        if not isinstance(record_id, str) or record_id not in expected_ids:
            raise ValueError("DMath owner decision contains an unknown record")
        if record_id in observed_ids:
            raise ValueError("DMath owner decision contains a duplicate record")
        accepted = decision.get("accepted")
        note = decision.get("note")
        if not isinstance(accepted, bool):
            raise ValueError("DMath owner decision must be boolean")
        if not isinstance(note, str) or len(note) > 2000:
            raise ValueError("DMath owner decision note is invalid")
        observed_ids.add(record_id)
        (accepted_ids if accepted else rejected_ids).append(record_id)
    if observed_ids != expected_ids:
        raise ValueError("DMath owner decisions do not match the sample")

    status = (
        "owner_quality_sample_passed_source_promotion_still_separate"
        if not rejected_ids
        else "owner_quality_sample_rejected_rows_require_repair"
    )
    receipt: dict[str, Any] = {
        "accepted_records": len(accepted_ids),
        "all_sampled_rows_accepted": not rejected_ids,
        "authorization": EXPECTED_AUTHORIZATION,
        "bindings": {
            "accepted_candidate_file_sha256": EXPECTED_CANDIDATE[
                "accepted_records_file_sha256"
            ],
            "curation_report_internal_sha256": EXPECTED_CANDIDATE[
                "curation_report_internal_sha256"
            ],
            "decision_file_sha256": decision_file_sha256,
            "owner_sample_internal_sha256": sample_sha256,
            "plan_file_sha256": plan_file_sha256,
            "reviewed_record_ids_sha256": sha256_value(sorted(observed_ids)),
        },
        "candidate": {
            "records": EXPECTED_CANDIDATE["accepted_records"],
            "supervised_tokens": EXPECTED_CANDIDATE["supervised_tokens"],
            "train_records": EXPECTED_CANDIDATE["train_records"],
            "train_supervised_tokens": EXPECTED_CANDIDATE[
                "train_supervised_tokens"
            ],
            "validation_records": EXPECTED_CANDIDATE["validation_records"],
            "validation_supervised_tokens": EXPECTED_CANDIDATE[
                "validation_supervised_tokens"
            ],
        },
        "owner_review_claim": (
            "upstream_dmath_source_plus_human_owner_quality_reviewed_"
            "not_first_party_authored"
        ),
        "production_eligible": False,
        "production_promotions": 0,
        "raw_training_text_retained": False,
        "rejected_record_ids": sorted(rejected_ids),
        "rejected_records": len(rejected_ids),
        "remaining_blockers": [
            "dmath_source_registry_promotion_not_authorized",
            "complete_10m_to_30m_learning_mix_not_ready",
            "production_dataset_seal_and_sft_not_authorized",
        ],
        "reviewed_at": reviewed_at,
        "reviewer_alias": reviewer.strip(),
        "sample_records": len(sample_rows),
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": status,
    }
    receipt["receipt_sha256"] = sha256_value(receipt)
    return receipt


def validate_owner_review_receipt(receipt: Mapping[str, Any]) -> None:
    """Verify the receipt seal and all closed production gates."""

    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported DMath owner-review receipt")
    seal = _require_sha256(receipt.get("receipt_sha256"), "receipt_sha256")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if sha256_value(unsigned) != seal:
        raise ValueError("DMath owner-review receipt seal is invalid")
    if receipt.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("DMath owner-review receipt authorization changed")
    if receipt.get("raw_training_text_retained") is not False:
        raise ValueError("DMath owner-review receipt retained raw training text")
    if receipt.get("production_eligible") is not False:
        raise ValueError("DMath owner review crossed production eligibility")
    if receipt.get("production_promotions") != 0:
        raise ValueError("DMath owner review crossed a promotion boundary")
    accepted = receipt.get("accepted_records")
    rejected = receipt.get("rejected_records")
    if not isinstance(accepted, int) or not isinstance(rejected, int):
        raise ValueError("DMath owner-review counts are invalid")
    if accepted + rejected != EXPECTED_LIMITS["records"]:
        raise ValueError("DMath owner-review count total changed")
