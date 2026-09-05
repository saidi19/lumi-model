"""Private owner-review gate for the Lumi safety/tool SFT candidate."""

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

PLAN_SCHEMA_VERSION = "lumi.safety-tool-owner-review-plan.v0.1"
DECISIONS_SCHEMA_VERSION = "lumi.safety-tool-owner-decisions.v0.1"
RECEIPT_SCHEMA_VERSION = "lumi.safety-tool-owner-review-receipt.v0.1"

EXPECTED_AUTHORIZATION = {
    "ai_or_automated_owner_approval_authorized": False,
    "external_api_spend_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_dataset_promotion_authorized": False,
    "production_sft_authorized": False,
    "purpose": "private_human_owner_spot_check_only",
}
EXPECTED_LIMITS = {
    "language_modes": 4,
    "records": 40,
    "records_per_skill_language_pair": 1,
    "skills": 10,
}
EXPECTED_OUTPUT = {
    "decision_filename": "lumi-safety-tool-owner-decisions-v0.1.json",
    "receipt_contains_raw_training_text": False,
    "source_registry_promotion_permitted": False,
}
EXPECTED_CANDIDATE = {
    "candidate_evidence_internal_sha256": (
        "62605cc07466ffc9c583655e3a85a3ad64c2bd4ef95a0c95deeb68d72823b57b"
    ),
    "candidate_source_revision": (
        "88765d93ed9df2effaf977ea535e901a5bf987abcfb511286f783679b7788915"
    ),
    "manifest_internal_sha256": (
        "1c3652b8c1c1804e35867555a6d1d9386c4f0ad4f94144e9506c7d4806458cf7"
    ),
    "sample_internal_sha256": (
        "182176e8c72c4dbc0d265f49e61997f11fea2e55321bd2cd21a3310642dd580d"
    ),
}
EXPECTED_BINDINGS = frozenset(
    {
        "candidate_evidence",
        "owner_sample",
        "pilot_manifest",
        "receipt_finalizer",
        "review_implementation",
        "review_renderer",
        "ui_builder",
    }
)
EXPECTED_LANGUAGE_MODES = {
    "banglish",
    "bengali_script",
    "english",
    "natural_code_switching",
}
EXPECTED_SKILLS = {
    "benign_request_without_overrefusal",
    "calculator_tool_error",
    "calculator_tool_success",
    "current_information_retrieval_boundary",
    "harmful_request_refusal_and_safe_alternative",
    "high_stakes_uncertainty_and_escalation",
    "list_sorter_tool_grounding",
    "privacy_and_secret_protection",
    "prompt_injection_resistance",
    "tool_capability_honesty",
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UTC_TIMESTAMP = re.compile(
    r"^20[0-9]{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])T"
    r"(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z$"
)
_BANNED_REVIEWERS = {
    "openai-codex-current-session",
    "lumi-safety-tool-deterministic-guard-v0.1",
    "chatgpt",
    "claude",
    "gemini",
}


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
    workspace = root.parent
    relative = binding.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("safety/tool owner-review binding path is invalid")
    resolved = (root / relative).resolve()
    if resolved != workspace and workspace not in resolved.parents:
        raise ValueError("safety/tool owner-review binding escapes workspace")
    return resolved


def load_owner_review_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Load the exact owner-review plan and verify all local bindings."""

    root = Path(repository_root).resolve()
    plan = _load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported safety/tool owner-review plan")
    if plan.get("status") != "human_owner_review_ready_no_promotion":
        raise ValueError("safety/tool owner-review status changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("safety/tool owner-review authorization changed")
    if plan.get("candidate") != EXPECTED_CANDIDATE:
        raise ValueError("safety/tool owner-review candidate changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("safety/tool owner-review limits changed")
    if plan.get("output") != EXPECTED_OUTPUT:
        raise ValueError("safety/tool owner-review output boundary changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("safety/tool owner-review binding set changed")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"owner-review binding must be an object: {name}")
        expected_sha256 = _require_sha256(
            binding.get("sha256"), f"bindings.{name}.sha256"
        )
        resolved = resolve_plan_binding(root, binding)
        if not resolved.is_file() or file_sha256(resolved) != expected_sha256:
            raise ValueError(f"safety/tool owner-review binding changed: {name}")
    return plan


def validate_owner_sample(
    sample_rows: Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
    candidate_evidence: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify exact 10-skill by 4-language owner-sample coverage."""

    manifest_without_seal = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if manifest.get("manifest_sha256") != sha256_value(manifest_without_seal):
        raise ValueError("safety/tool pilot manifest seal changed")
    if manifest.get("manifest_sha256") != plan["candidate"]["manifest_internal_sha256"]:
        raise ValueError("safety/tool pilot manifest binding changed")
    if manifest.get("candidate_source_revision") != plan["candidate"][
        "candidate_source_revision"
    ]:
        raise ValueError("safety/tool owner-review source revision changed")
    if manifest.get("owner_review") != {
        "complete": False,
        "sample_records": EXPECTED_LIMITS["records"],
        "sample_sha256": plan["candidate"]["sample_internal_sha256"],
        "status": "pending",
    }:
        raise ValueError("safety/tool owner-review manifest boundary changed")

    evidence_without_seal = {
        key: value
        for key, value in candidate_evidence.items()
        if key != "evidence_sha256"
    }
    if candidate_evidence.get("evidence_sha256") != sha256_value(evidence_without_seal):
        raise ValueError("safety/tool candidate evidence seal changed")
    if candidate_evidence.get("evidence_sha256") != plan["candidate"][
        "candidate_evidence_internal_sha256"
    ]:
        raise ValueError("safety/tool candidate evidence binding changed")
    if candidate_evidence.get("owner_review_complete") is not False:
        raise ValueError("safety/tool candidate unexpectedly claims owner review")

    if len(sample_rows) != EXPECTED_LIMITS["records"]:
        raise ValueError("safety/tool owner-review sample count changed")
    if sha256_value(sample_rows) != plan["candidate"]["sample_internal_sha256"]:
        raise ValueError("safety/tool owner-review sample content changed")
    record_ids: set[str] = set()
    coverage: Counter[tuple[str, str]] = Counter()
    for row in sample_rows:
        record_id = row.get("record_id")
        skill = row.get("skill_id")
        language_mode = row.get("language_mode")
        if not isinstance(record_id, str) or not record_id or record_id in record_ids:
            raise ValueError("safety/tool owner-review record identity changed")
        if skill not in EXPECTED_SKILLS or language_mode not in EXPECTED_LANGUAGE_MODES:
            raise ValueError("safety/tool owner-review coverage changed")
        if row.get("split") != "validation":
            raise ValueError("safety/tool owner-review sample must remain validation-only")
        if row.get("owner_review") != {
            "accepted": None,
            "reviewed_at": "",
            "reviewer_id": "",
            "status": "pending",
        }:
            raise ValueError("safety/tool owner-review row already claims approval")
        generation = row.get("generation")
        if not isinstance(generation, Mapping) or generation.get(
            "human_written_from_scratch"
        ) is not False:
            raise ValueError("safety/tool owner-review provenance changed")
        if not isinstance(row.get("messages"), list) or not isinstance(
            row.get("task_evidence"), Mapping
        ):
            raise ValueError("safety/tool owner-review row schema changed")
        record_ids.add(record_id)
        coverage[(str(skill), str(language_mode))] += 1
    expected_pairs = {
        (skill, language_mode)
        for skill in EXPECTED_SKILLS
        for language_mode in EXPECTED_LANGUAGE_MODES
    }
    if set(coverage) != expected_pairs or set(coverage.values()) != {1}:
        raise ValueError("safety/tool owner-review pair coverage changed")
    return {
        "language_modes": len(EXPECTED_LANGUAGE_MODES),
        "records": len(sample_rows),
        "records_per_skill_language_pair": 1,
        "skills": len(EXPECTED_SKILLS),
    }


def build_safety_tool_owner_review_html(
    sample_rows: list[Mapping[str, Any]],
    *,
    sample_sha256: str,
) -> str:
    """Build a private review page with a safety/tool-specific decision schema."""

    html = build_owner_review_html(sample_rows, sample_sha256=sample_sha256)
    replacements = {
        "<title>Lumi owner review</title>": "<title>Lumi safety/tool owner review</title>",
        "<h1>Lumi owner spot-check</h1>": "<h1>Lumi safety/tool owner spot-check</h1>",
        (
            f"<p>Review all {len(sample_rows)} rows. Check correctness, naturalness, "
            "language, context, and safety.</p>"
        ): (
            f"<p>Review all {len(sample_rows)} rows. Accept only when the answer is "
            "correct, natural, language-matched, safe, and grounded in any shown "
            "tool result.</p>"
        ),
        (
            "These rows are AI-generated and owner-reviewed. Approving them does not "
            "make them human-written."
        ): (
            "These rows are AI-generated. Your decision is an owner spot-check, not a "
            "human-authorship claim or permission to start SFT."
        ),
        "lumi-owner-review-${sampleSha256}": (
            "lumi-safety-tool-owner-review-${sampleSha256}"
        ),
        "lumi.ai-assisted-owner-decisions.v0.3": DECISIONS_SCHEMA_VERSION,
        "lumi-owner-review-decisions-v0.3.json": EXPECTED_OUTPUT["decision_filename"],
    }
    for old, new in replacements.items():
        if old not in html:
            raise ValueError("owner-review renderer contract changed")
        html = html.replace(old, new)
    return html


def validate_owner_decisions(
    value: Mapping[str, Any],
    *,
    sample_rows: Sequence[Mapping[str, Any]],
    sample_sha256: str,
    decision_file_sha256: str,
    plan_file_sha256: str,
) -> dict[str, Any]:
    """Validate human decisions and seal a raw-text-free non-promoting receipt."""

    _require_sha256(decision_file_sha256, "decision_file_sha256")
    _require_sha256(plan_file_sha256, "plan_file_sha256")
    if value.get("schema_version") != DECISIONS_SCHEMA_VERSION:
        raise ValueError("unsupported safety/tool owner decision schema")
    if value.get("sample_sha256") != sample_sha256:
        raise ValueError("safety/tool owner decision sample changed")
    if sha256_value(sample_rows) != sample_sha256:
        raise ValueError("safety/tool owner-review sample content changed")
    reviewer = value.get("reviewer_id")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("safety/tool owner reviewer alias is required")
    if reviewer.strip().casefold() in _BANNED_REVIEWERS:
        raise ValueError("an AI generator or automated guard cannot claim owner review")
    reviewed_at = value.get("reviewed_at")
    if not isinstance(reviewed_at, str) or _UTC_TIMESTAMP.fullmatch(reviewed_at) is None:
        raise ValueError("safety/tool owner reviewed_at must be exact UTC ISO-8601")

    decisions = value.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != len(sample_rows):
        raise ValueError("safety/tool owner decisions must cover every sampled row")
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
            raise ValueError("safety/tool owner decision fields changed")
        record_id = decision.get("record_id")
        if not isinstance(record_id, str) or record_id not in expected_ids:
            raise ValueError("safety/tool owner decision contains an unknown record")
        if record_id in observed_ids:
            raise ValueError("safety/tool owner decision contains a duplicate record")
        accepted = decision.get("accepted")
        note = decision.get("note")
        if not isinstance(accepted, bool):
            raise ValueError("safety/tool owner decision must be boolean")
        if not isinstance(note, str) or len(note) > 2000:
            raise ValueError("safety/tool owner decision note is invalid")
        observed_ids.add(record_id)
        (accepted_ids if accepted else rejected_ids).append(record_id)
    if observed_ids != expected_ids:
        raise ValueError("safety/tool owner decisions do not match the sample")

    status = (
        "owner_spot_check_passed_source_registry_promotion_still_separate"
        if not rejected_ids
        else "owner_spot_check_rejected_rows_require_repair"
    )
    receipt: dict[str, Any] = {
        "accepted_records": len(accepted_ids),
        "all_sampled_rows_accepted": not rejected_ids,
        "authorization": EXPECTED_AUTHORIZATION,
        "bindings": {
            "decision_file_sha256": decision_file_sha256,
            "plan_file_sha256": plan_file_sha256,
            "sample_internal_sha256": sample_sha256,
        },
        "owner_review_claim": "ai_generated_plus_human_owner_reviewed_not_human_written",
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
        "sample_records": len(sample_rows),
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": status,
    }
    receipt["receipt_sha256"] = sha256_value(receipt)
    return receipt

