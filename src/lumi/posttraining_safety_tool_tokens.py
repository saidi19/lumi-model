"""Exact Lumi-tokenizer measurements for the private safety/tool SFT pilot."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .posttraining_first_party import canonical_json_bytes, file_sha256
from .posttraining_format import ChatTemplate, ChatTokenizer, encode_sft_messages

PLAN_SCHEMA_VERSION = "lumi.safety-tool-sft-token-audit-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.safety-tool-sft-token-audit-report.v0.1"

EXPECTED_AUTHORIZATION = {
    "external_api_spend_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_dataset_promotion_authorized": False,
    "production_sft_authorized": False,
    "purpose": "existing_private_candidate_local_token_measurement_only",
}
EXPECTED_LIMITS = {
    "context_window": 4096,
    "expected_records": 320,
    "expected_train_records": 240,
    "expected_validation_records": 80,
    "maximum_records": 320,
}
EXPECTED_OUTPUT = {
    "contains_raw_training_text": False,
    "owner_review_complete": False,
    "production_eligible": False,
    "production_promotions": 0,
}
EXPECTED_CANDIDATE = {
    "candidate_evidence_internal_sha256": (
        "62605cc07466ffc9c583655e3a85a3ad64c2bd4ef95a0c95deeb68d72823b57b"
    ),
    "candidate_source_revision": (
        "88765d93ed9df2effaf977ea535e901a5bf987abcfb511286f783679b7788915"
    ),
}
EXPECTED_BINDINGS = frozenset(
    {
        "audit_implementation",
        "audit_runner",
        "candidate_evidence",
        "candidate_records",
        "chat_template",
        "encoding_implementation",
        "tokenizer_config",
        "tokenizer_json",
    }
)
EXPECTED_LANGUAGE_COUNTS = {
    "banglish": 80,
    "bengali_script": 80,
    "english": 80,
    "natural_code_switching": 80,
}
EXPECTED_LANGUAGE_VALUES = {
    "banglish": "multilingual",
    "bengali_script": "bn",
    "english": "en",
    "natural_code_switching": "multilingual",
}
EXPECTED_TASK_COUNTS = {
    "safety_helpfulness": 160,
    "tool_api": 160,
}
EXPECTED_SPLIT_COUNTS = {"train": 240, "validation": 80}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


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
        raise ValueError("token-audit binding path is invalid")
    resolved = (root / relative).resolve()
    if resolved != workspace and workspace not in resolved.parents:
        raise ValueError("token-audit binding escapes workspace")
    return resolved


def load_token_audit_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Load the exact CPU-only audit plan and verify every bound file."""

    root = Path(repository_root).resolve()
    plan = _load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported safety/tool token-audit plan")
    if plan.get("status") != "local_measurement_authorized_no_promotion":
        raise ValueError("safety/tool token-audit status changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("safety/tool token-audit authorization changed")
    if plan.get("candidate") != EXPECTED_CANDIDATE:
        raise ValueError("safety/tool token-audit candidate changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("safety/tool token-audit limits changed")
    if plan.get("output") != EXPECTED_OUTPUT:
        raise ValueError("safety/tool token-audit output boundary changed")

    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("safety/tool token-audit binding set changed")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"token-audit binding must be an object: {name}")
        expected_sha256 = _require_sha256(
            binding.get("sha256"), f"bindings.{name}.sha256"
        )
        resolved = resolve_plan_binding(root, binding)
        if not resolved.is_file() or file_sha256(resolved) != expected_sha256:
            raise ValueError(f"safety/tool token-audit binding changed: {name}")
    return plan


def verify_candidate_evidence(
    evidence: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> None:
    """Verify the immutable machine-validated candidate receipt."""

    if evidence.get("schema_version") != "lumi.safety-tool-sft-evidence.v0.1":
        raise ValueError("unsupported safety/tool candidate evidence")
    seal = evidence.get("evidence_sha256")
    without_seal = {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    if seal != sha256_value(without_seal):
        raise ValueError("safety/tool candidate evidence seal changed")
    if seal != plan["candidate"]["candidate_evidence_internal_sha256"]:
        raise ValueError("safety/tool candidate evidence binding changed")
    if evidence.get("candidate_source_revision") != plan["candidate"][
        "candidate_source_revision"
    ]:
        raise ValueError("safety/tool candidate source revision changed")
    if evidence.get("counts") != {
        "languages": EXPECTED_LANGUAGE_COUNTS,
        "records": EXPECTED_LIMITS["expected_records"],
        "skills": {
            "benign_request_without_overrefusal": 32,
            "calculator_tool_error": 32,
            "calculator_tool_success": 32,
            "current_information_retrieval_boundary": 32,
            "harmful_request_refusal_and_safe_alternative": 32,
            "high_stakes_uncertainty_and_escalation": 32,
            "list_sorter_tool_grounding": 32,
            "privacy_and_secret_protection": 32,
            "prompt_injection_resistance": 32,
            "tool_capability_honesty": 32,
        },
        "splits": EXPECTED_SPLIT_COUNTS,
        "tasks": EXPECTED_TASK_COUNTS,
    }:
        raise ValueError("safety/tool candidate evidence counts changed")
    if (
        evidence.get("owner_review_complete") is not False
        or evidence.get("production_eligible") is not False
        or evidence.get("production_promotions") != 0
        or evidence.get("raw_training_text_retained") is not False
    ):
        raise ValueError("safety/tool candidate evidence boundary changed")
    if evidence.get("bindings", {}).get("candidate_file_sha256") != plan["bindings"][
        "candidate_records"
    ]["sha256"]:
        raise ValueError("safety/tool candidate file binding changed")


def _language_mode(record_id: str) -> str:
    matches = [
        mode for mode in EXPECTED_LANGUAGE_VALUES if record_id.endswith(f"-{mode}")
    ]
    if len(matches) != 1:
        raise ValueError("safety/tool candidate language mode is ambiguous")
    return matches[0]


def _length_summary(lengths: Sequence[int]) -> dict[str, int | float]:
    if not lengths:
        raise ValueError("token length summary cannot be empty")
    ordered = sorted(lengths)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "maximum": ordered[-1],
        "mean": round(statistics.fmean(ordered), 4),
        "median": statistics.median(ordered),
        "minimum": ordered[0],
        "p95_nearest_rank": ordered[p95_index],
    }


def _empty_group() -> dict[str, int]:
    return {
        "masked_tokens": 0,
        "records": 0,
        "serialized_tokens": 0,
        "supervised_tokens": 0,
    }


def audit_candidate_token_counts(
    records: Sequence[Mapping[str, Any]],
    *,
    tokenizer: ChatTokenizer,
    template: ChatTemplate,
) -> dict[str, Any]:
    """Encode every exact candidate row and return aggregate, raw-free metrics."""

    if template.context_window != EXPECTED_LIMITS["context_window"]:
        raise ValueError("safety/tool token-audit context changed")
    if len(records) != EXPECTED_LIMITS["expected_records"]:
        raise ValueError("safety/tool token-audit record count changed")

    seen: set[str] = set()
    split_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    by_split: dict[str, dict[str, int]] = {}
    by_task: dict[str, dict[str, int]] = {}
    by_language: dict[str, dict[str, int]] = {}
    serialized_lengths: list[int] = []
    supervised_lengths: list[int] = []

    for record in records:
        record_id = record.get("record_id")
        messages = record.get("messages")
        source = record.get("source")
        task = record.get("task_category")
        language = record.get("language")
        if not isinstance(record_id, str) or not record_id or record_id in seen:
            raise ValueError("safety/tool token-audit record identity changed")
        if not isinstance(messages, list) or not isinstance(source, Mapping):
            raise ValueError("safety/tool token-audit candidate schema changed")
        split = source.get("split")
        mode = _language_mode(record_id)
        if split not in EXPECTED_SPLIT_COUNTS or task not in EXPECTED_TASK_COUNTS:
            raise ValueError("safety/tool token-audit category changed")
        if language != EXPECTED_LANGUAGE_VALUES[mode]:
            raise ValueError("safety/tool token-audit language label changed")
        if source.get("revision") != EXPECTED_CANDIDATE["candidate_source_revision"]:
            raise ValueError("safety/tool token-audit candidate revision changed")
        seen.add(record_id)

        encoded = encode_sft_messages(messages, tokenizer=tokenizer, template=template)
        serialized = len(encoded.input_ids)
        supervised = encoded.supervised_tokens
        masked = encoded.masked_tokens
        serialized_lengths.append(serialized)
        supervised_lengths.append(supervised)
        split_counts[str(split)] += 1
        task_counts[str(task)] += 1
        language_counts[mode] += 1
        for groups, key in (
            (by_split, str(split)),
            (by_task, str(task)),
            (by_language, mode),
        ):
            group = groups.setdefault(key, _empty_group())
            group["masked_tokens"] += masked
            group["records"] += 1
            group["serialized_tokens"] += serialized
            group["supervised_tokens"] += supervised

    if dict(sorted(split_counts.items())) != EXPECTED_SPLIT_COUNTS:
        raise ValueError("safety/tool token-audit split counts changed")
    if dict(sorted(task_counts.items())) != EXPECTED_TASK_COUNTS:
        raise ValueError("safety/tool token-audit task counts changed")
    if dict(sorted(language_counts.items())) != EXPECTED_LANGUAGE_COUNTS:
        raise ValueError("safety/tool token-audit language counts changed")

    serialized_total = sum(serialized_lengths)
    supervised_total = sum(supervised_lengths)
    metrics: dict[str, Any] = {
        "accepted_records": len(records),
        "by_language_mode": dict(sorted(by_language.items())),
        "by_split": dict(sorted(by_split.items())),
        "by_task": dict(sorted(by_task.items())),
        "context_window": template.context_window,
        "masked_tokens": serialized_total - supervised_total,
        "overlength_records": 0,
        "raw_text_retained": False,
        "rejected_records": 0,
        "serialized_length": _length_summary(serialized_lengths),
        "serialized_tokens": serialized_total,
        "supervised_fraction": round(supervised_total / serialized_total, 8),
        "supervised_length": _length_summary(supervised_lengths),
        "supervised_tokens": supervised_total,
        "validation_tokens_excluded_from_training_target": True,
    }
    return metrics


def build_token_audit_report(
    *,
    metrics: Mapping[str, Any],
    observed_at: str,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
) -> dict[str, Any]:
    """Seal a raw-text-free, non-promoting tokenizer measurement receipt."""

    _require_sha256(plan_file_sha256, "plan_file_sha256")
    if (
        metrics.get("accepted_records") != EXPECTED_LIMITS["expected_records"]
        or metrics.get("rejected_records") != 0
        or metrics.get("overlength_records") != 0
        or metrics.get("raw_text_retained") is not False
    ):
        raise ValueError("safety/tool token-audit metrics do not pass")
    report: dict[str, Any] = {
        "authorization": EXPECTED_AUTHORIZATION,
        "bindings": {
            "candidate_evidence_internal_sha256": plan["candidate"][
                "candidate_evidence_internal_sha256"
            ],
            "candidate_file_sha256": plan["bindings"]["candidate_records"]["sha256"],
            "chat_template_file_sha256": plan["bindings"]["chat_template"]["sha256"],
            "encoding_implementation_sha256": plan["bindings"][
                "encoding_implementation"
            ]["sha256"],
            "plan_file_sha256": plan_file_sha256,
            "tokenizer_config_sha256": plan["bindings"]["tokenizer_config"]["sha256"],
            "tokenizer_json_sha256": plan["bindings"]["tokenizer_json"]["sha256"],
        },
        "candidate_source_revision": plan["candidate"]["candidate_source_revision"],
        "decision": "all_candidate_rows_fit_frozen_4096_context_owner_review_pending",
        "metrics": dict(metrics),
        "observed_at": observed_at,
        "owner_review_complete": False,
        "production_eligible": False,
        "production_promotions": 0,
        "raw_training_text_retained": False,
        "schema_version": REPORT_SCHEMA_VERSION,
    }
    report["report_sha256"] = sha256_value(report)
    return report
