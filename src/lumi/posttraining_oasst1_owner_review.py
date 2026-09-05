"""Private human-owner quality review for the curated OASST1 sample."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .posttraining_first_party import canonical_json_bytes, file_sha256
from .posttraining_owner_review import build_owner_review_html

PLAN_SCHEMA_VERSION = "lumi.sft-oasst1-owner-review-plan.v0.1"
DECISIONS_SCHEMA_VERSION = "lumi.sft-oasst1-owner-decisions.v0.1"
RECEIPT_SCHEMA_VERSION = "lumi.sft-oasst1-owner-review-receipt.v0.1"
READY_SCHEMA_VERSION = "lumi.sft-oasst1-owner-review-ready.v0.1"

EXPECTED_AUTHORIZATION = {
    "ai_or_automated_owner_approval_authorized": False,
    "external_api_spend_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_dataset_promotion_authorized": False,
    "production_dataset_seal_authorized": False,
    "production_sft_authorized": False,
    "purpose": "private_human_owner_quality_spot_check_only",
}
EXPECTED_LIMITS = {
    "records": 32,
    "records_per_split_task_pair": 4,
    "splits": 2,
    "task_categories": 4,
}
EXPECTED_OUTPUT = {
    "decision_filename": "lumi-oasst1-owner-decisions-v0.1.json",
    "production_promotion_permitted": False,
    "receipt_contains_raw_training_text": False,
}
EXPECTED_CANDIDATE = {
    "accepted_records": 1745,
    "curation_report_file_sha256": (
        "f4e8199249c8e96f333bfa3d1ba1dc73545be7b6dcc9adc8cd70b67964edf293"
    ),
    "curation_report_internal_sha256": (
        "0df672f2b34a6e580fc9a577749fb03359c9628ca92a4cfe0a6ba801e25ff580"
    ),
    "owner_sample_file_sha256": (
        "6bfd1a4421b28c5bbc64eb5c2540f983abbf17ada3b8fd18e17b35d25d54ed53"
    ),
    "owner_sample_internal_sha256": (
        "2fe1796731e7ae29ab2fa126f81d726bd43d5fc058b4ffe1360904503f8c01a1"
    ),
    "run_plan_file_sha256": (
        "04878ad9c39ffc039144e768f87360ed1533a2518f1eab12f0363b318f02b6dd"
    ),
    "source_revision": "fdf72ae0827c1cda404aff25b6603abec9e3399b",
    "supervised_tokens": 477456,
    "verification_file_sha256": (
        "d18bf2bb4bd2366afcff04c668a33d73b9cf0972af86ba18180f36444a5dae7f"
    ),
    "verification_internal_sha256": (
        "bdd099dae98812117db028f98a4d4aa11b23ab1e08b59ae67667b1a0484b2cc3"
    ),
}
EXPECTED_TASKS = {
    "coding",
    "conversation",
    "factual_qa",
    "writing_summarization",
}
EXPECTED_SPLITS = {"train", "validation"}
EXPECTED_BINDINGS = frozenset(
    {
        "curation_report",
        "curation_run_plan",
        "curation_verification",
        "owner_sample",
        "receipt_finalizer",
        "review_implementation",
        "review_renderer",
        "ui_builder",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UTC_TIMESTAMP = re.compile(
    r"^20[0-9]{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])T"
    r"(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z$"
)
_BANNED_REVIEWER_MARKERS = {
    "chatgpt",
    "claude",
    "codex",
    "gemini",
}
_BANNED_REVIEWERS = {"openai-codex-current-session"}


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


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
        raise ValueError("OASST1 owner-review binding path is invalid")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError("OASST1 owner-review binding escapes workspace")
    return resolved


def load_owner_review_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Load the exact private owner-review plan and verify every binding."""

    root = Path(repository_root).resolve()
    plan = _load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported OASST1 owner-review plan")
    if plan.get("status") != "human_owner_review_ready_no_promotion":
        raise ValueError("OASST1 owner-review status changed")
    if plan.get("observed_at") != "2026-08-06":
        raise ValueError("OASST1 owner-review date changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("OASST1 owner-review authorization changed")
    if plan.get("candidate") != EXPECTED_CANDIDATE:
        raise ValueError("OASST1 owner-review candidate changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("OASST1 owner-review limits changed")
    if plan.get("output") != EXPECTED_OUTPUT:
        raise ValueError("OASST1 owner-review output boundary changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("OASST1 owner-review binding set changed")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"OASST1 owner-review binding must be an object: {name}")
        expected = _require_sha256(binding.get("sha256"), f"bindings.{name}.sha256")
        resolved = resolve_plan_binding(root, binding)
        if not resolved.is_file() or file_sha256(resolved) != expected:
            raise ValueError(f"OASST1 owner-review binding changed: {name}")
    return plan


def review_payload(sample_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Add display-only aliases without modifying the private source sample."""

    return [
        {
            **dict(row),
            "language_mode": row["language"],
            "skill_id": row["task_category"],
        }
        for row in sample_rows
    ]


def validate_owner_sample(
    sample_rows: Sequence[Mapping[str, Any]],
    *,
    curation_report: Mapping[str, Any],
    verification: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the exact balanced 32-row OASST1 owner sample."""

    report_unsigned = {
        key: value for key, value in curation_report.items() if key != "report_sha256"
    }
    if curation_report.get("report_sha256") != sha256_value(report_unsigned):
        raise ValueError("OASST1 curation report internal SHA-256 changed")
    if curation_report.get("report_sha256") != plan["candidate"][
        "curation_report_internal_sha256"
    ]:
        raise ValueError("OASST1 curation report binding changed")
    if curation_report.get("status") != "curated_nonproduction_owner_review_pending":
        raise ValueError("OASST1 curation report status changed")
    if curation_report.get("owner_review") != {
        "complete": False,
        "record_count": 32,
        "sample_record_ids_sha256": (
            "05c993c61f52217b60ce5cd8261e632edf6234ea43328b1903450a3356eaeb19"
        ),
        "status": "pending_owner_review",
    }:
        raise ValueError("OASST1 owner-review report boundary changed")
    verification_unsigned = {
        key: value for key, value in verification.items() if key != "verification_sha256"
    }
    if verification.get("verification_sha256") != sha256_value(verification_unsigned):
        raise ValueError("OASST1 curation verification SHA-256 changed")
    if verification.get("verification_sha256") != plan["candidate"][
        "verification_internal_sha256"
    ]:
        raise ValueError("OASST1 curation verification binding changed")
    if verification.get("status") != "verified_nonproduction_owner_review_pending":
        raise ValueError("OASST1 curation verification status changed")
    if verification.get("production_eligible") is not False:
        raise ValueError("OASST1 unexpectedly claims production eligibility")

    if len(sample_rows) != EXPECTED_LIMITS["records"]:
        raise ValueError("OASST1 owner-review sample count changed")
    if sha256_value(list(sample_rows)) != plan["candidate"][
        "owner_sample_internal_sha256"
    ]:
        raise ValueError("OASST1 owner-review sample content changed")
    record_ids: set[str] = set()
    coverage: Counter[tuple[str, str]] = Counter()
    for row in sample_rows:
        record_id = row.get("record_id")
        split = row.get("split")
        task = row.get("task_category")
        if not isinstance(record_id, str) or not record_id or record_id in record_ids:
            raise ValueError("OASST1 owner-review record identity changed")
        if split not in EXPECTED_SPLITS or task not in EXPECTED_TASKS:
            raise ValueError("OASST1 owner-review coverage changed")
        if row.get("schema_version") != "lumi.sft.v0.1" or row.get("language") != "en":
            raise ValueError("OASST1 owner-review record schema changed")
        source = row.get("source")
        if not isinstance(source, Mapping) or source.get("revision") != EXPECTED_CANDIDATE[
            "source_revision"
        ]:
            raise ValueError("OASST1 owner-review source binding changed")
        if source.get("source_id") != "openassistant_oasst1":
            raise ValueError("OASST1 owner-review source identity changed")
        messages = row.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) != 2
            or [message.get("role") for message in messages] != ["user", "assistant"]
        ):
            raise ValueError("OASST1 owner-review message structure changed")
        record_ids.add(record_id)
        coverage[(str(split), str(task))] += 1
    expected_pairs = {
        (split, task) for split in EXPECTED_SPLITS for task in EXPECTED_TASKS
    }
    if set(coverage) != expected_pairs or set(coverage.values()) != {4}:
        raise ValueError("OASST1 owner-review balance changed")
    payload = review_payload(sample_rows)
    return {
        "payload_sha256": sha256_value(payload),
        "records": len(sample_rows),
        "records_per_split_task_pair": 4,
        "splits": 2,
        "task_categories": 4,
    }


def build_oasst1_owner_review_html(
    sample_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    """Build a self-contained private OASST1 review page."""

    payload = review_payload(sample_rows)
    payload_sha256 = sha256_value(payload)
    html = build_owner_review_html(payload, sample_sha256=payload_sha256)
    replacements = {
        "<title>Lumi owner review</title>": "<title>Lumi OASST1 owner review</title>",
        "<h1>Lumi owner spot-check</h1>": "<h1>Lumi OASST1 quality spot-check</h1>",
        (
            f"<p>Review all {len(payload)} rows. Check correctness, naturalness, "
            "language, context, and safety.</p>"
        ): (
            f"<p>Review all {len(payload)} rows. Accept only when the response is "
            "correct, helpful, natural, context-sensitive, safe, and suitable for "
            "Lumi instruction tuning.</p>"
        ),
        (
            "These rows are AI-generated and owner-reviewed. Approving them does not "
            "make them human-written."
        ): (
            "These are upstream human-authored OASST1 conversations selected by "
            "machine gates. Your decision is a private owner quality review, not a "
            "first-party authorship claim or permission to start SFT."
        ),
        "lumi-owner-review-${sampleSha256}": (
            "lumi-oasst1-owner-review-${sampleSha256}"
        ),
        "lumi.ai-assisted-owner-decisions.v0.3": DECISIONS_SCHEMA_VERSION,
        "lumi-owner-review-decisions-v0.3.json": EXPECTED_OUTPUT["decision_filename"],
    }
    for old, new in replacements.items():
        if old not in html:
            raise ValueError("OASST1 owner-review renderer contract changed")
        html = html.replace(old, new)
    return html, payload_sha256


def validate_owner_decisions(
    value: Mapping[str, Any],
    *,
    sample_rows: Sequence[Mapping[str, Any]],
    decision_file_sha256: str,
    plan_file_sha256: str,
) -> dict[str, Any]:
    """Validate human decisions and return a raw-text-free non-promoting receipt."""

    _require_sha256(decision_file_sha256, "decision_file_sha256")
    _require_sha256(plan_file_sha256, "plan_file_sha256")
    payload = review_payload(sample_rows)
    payload_sha256 = sha256_value(payload)
    if value.get("schema_version") != DECISIONS_SCHEMA_VERSION:
        raise ValueError("unsupported OASST1 owner decision schema")
    if value.get("sample_sha256") != payload_sha256:
        raise ValueError("OASST1 owner decision sample changed")
    reviewer = value.get("reviewer_id")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("OASST1 owner reviewer alias is required")
    normalized_reviewer = reviewer.strip().casefold()
    if normalized_reviewer in _BANNED_REVIEWERS or any(
        marker in normalized_reviewer for marker in _BANNED_REVIEWER_MARKERS
    ):
        raise ValueError("an AI system cannot claim OASST1 owner review")
    reviewed_at = value.get("reviewed_at")
    if not isinstance(reviewed_at, str) or _UTC_TIMESTAMP.fullmatch(reviewed_at) is None:
        raise ValueError("OASST1 owner reviewed_at must be exact UTC ISO-8601")
    decisions = value.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != len(payload):
        raise ValueError("OASST1 owner decisions must cover every sampled row")
    expected_ids = {str(row["record_id"]) for row in payload}
    observed_ids: set[str] = set()
    accepted_ids: list[str] = []
    rejected_ids: list[str] = []
    for decision in decisions:
        if not isinstance(decision, Mapping) or set(decision) != {
            "accepted",
            "note",
            "record_id",
        }:
            raise ValueError("OASST1 owner decision fields changed")
        record_id = decision.get("record_id")
        if not isinstance(record_id, str) or record_id not in expected_ids:
            raise ValueError("OASST1 owner decision contains an unknown record")
        if record_id in observed_ids:
            raise ValueError("OASST1 owner decision contains a duplicate record")
        accepted = decision.get("accepted")
        note = decision.get("note")
        if not isinstance(accepted, bool):
            raise ValueError("OASST1 owner decision must be boolean")
        if not isinstance(note, str) or len(note) > 2000:
            raise ValueError("OASST1 owner decision note is invalid")
        observed_ids.add(record_id)
        (accepted_ids if accepted else rejected_ids).append(record_id)
    if observed_ids != expected_ids:
        raise ValueError("OASST1 owner decisions do not match the sample")
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
            "decision_file_sha256": decision_file_sha256,
            "owner_sample_internal_sha256": EXPECTED_CANDIDATE[
                "owner_sample_internal_sha256"
            ],
            "plan_file_sha256": plan_file_sha256,
            "review_payload_sha256": payload_sha256,
        },
        "owner_review_claim": (
            "upstream_human_authored_plus_human_owner_quality_reviewed_"
            "not_first_party_authored"
        ),
        "production_eligible": False,
        "production_promotions": 0,
        "raw_training_text_retained": False,
        "rejected_record_ids": sorted(rejected_ids),
        "rejected_records": len(rejected_ids),
        "remaining_blockers": [
            "source_registry_promotion_not_authorized",
            "complete_10m_to_30m_learning_mix_not_ready",
            "production_dataset_seal_and_sft_not_authorized",
        ],
        "reviewed_at": reviewed_at,
        "reviewer_id": reviewer.strip(),
        "sample_records": len(payload),
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": status,
    }
    receipt["receipt_sha256"] = sha256_value(receipt)
    return receipt
