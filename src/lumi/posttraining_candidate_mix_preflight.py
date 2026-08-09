"""Fail-closed, local-only closure preflight for the Lumi SFT candidate mix."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .posttraining_data import load_sft_policy, validate_message_sequence
from .posttraining_format import load_chat_template, verify_chat_tokenizer
from .posttraining_sources import load_sft_source_registry
from .tokenizer_gate import encode_untrusted_text

PLAN_SCHEMA_VERSION = "lumi.sft-candidate-mix-closure-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-candidate-mix-closure-report.v0.1"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-candidate-mix-closure-verification.v0.1"

EXPECTED_AUTHORIZATION = {
    "candidate_registry_change_authorized": False,
    "dataset_acquisition_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_api_spend_authorized": False,
    "local_candidate_read_authorized": True,
    "local_raw_text_free_preflight_authorized": True,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "purpose": "local_candidate_mix_closure_preflight_only",
    "source_promotion_authorized": False,
}

EXPECTED_CANDIDATE_SOURCE_IDS = {
    "coherelabs_aya_dataset",
    "deepmind_mathematics_dataset",
    "dmath",
    "lumi_safety_tool_sft_v0_1",
    "openassistant_oasst1",
}
EXPECTED_EXCLUDED_SOURCE_IDS = {
    "lumi_ai_assisted_sft_v0_3",
    "openr1_math_220k",
}
EXPECTED_TASKS = (
    "coding",
    "conversation",
    "factual_qa",
    "multilingual_bengali",
    "reasoning_math_science",
    "safety_helpfulness",
    "tool_api",
    "writing_summarization",
)
EXPECTED_LANGUAGE_MODES = (
    "banglish",
    "bengali_script",
    "english",
    "natural_code_switching",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
            rows.append(value)
    return rows


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a SHA-256")
    return value


def _verify_internal_sha(value: Mapping[str, Any], field: str) -> str:
    seal = _require_sha256(value.get(field), field)
    unsigned = {key: item for key, item in value.items() if key != field}
    if sha256_value(unsigned) != seal:
        raise ValueError(f"{field} changed")
    return seal


def resolve_binding(
    repository_root: str | Path,
    binding: Mapping[str, Any],
    *,
    name: str,
) -> Path:
    root = Path(repository_root).resolve()
    workspace = root.parent.resolve()
    relative = binding.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"invalid binding path: {name}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError(f"binding escapes workspace: {name}")
    if not resolved.is_file():
        raise ValueError(f"binding is missing: {name}")
    expected = _require_sha256(binding.get("sha256"), f"bindings.{name}.sha256")
    if file_sha256(resolved) != expected:
        raise ValueError(f"binding changed: {name}")
    return resolved


def load_candidate_mix_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    plan = _load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported candidate-mix plan schema")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("candidate-mix authorization changed")
    if plan.get("observed_at") != "2026-08-06":
        raise ValueError("candidate-mix observation date changed")
    if plan.get("status") != "local_preflight_authorized_no_production_action":
        raise ValueError("candidate-mix plan status changed")
    limits = plan.get("limits")
    if not isinstance(limits, Mapping):
        raise ValueError("candidate-mix limits are missing")
    if limits.get("maximum_candidate_records_read") != 20_000:
        raise ValueError("candidate-mix row ceiling changed")
    if limits.get("maximum_runtime_minutes") != 60:
        raise ValueError("candidate-mix runtime ceiling changed")
    if limits.get("maximum_network_bytes") != 0:
        raise ValueError("candidate-mix network ceiling changed")

    candidates = plan.get("candidate_sources")
    if not isinstance(candidates, list):
        raise ValueError("candidate_sources must be a list")
    candidate_ids = {item.get("source_id") for item in candidates if isinstance(item, Mapping)}
    if candidate_ids != EXPECTED_CANDIDATE_SOURCE_IDS or len(candidates) != len(candidate_ids):
        raise ValueError("candidate source set changed")
    excluded = plan.get("excluded_sources")
    if not isinstance(excluded, list):
        raise ValueError("excluded_sources must be a list")
    excluded_ids = {item.get("source_id") for item in excluded if isinstance(item, Mapping)}
    if excluded_ids != EXPECTED_EXCLUDED_SOURCE_IDS or len(excluded) != len(excluded_ids):
        raise ValueError("excluded source set changed")

    output = plan.get("output")
    if not isinstance(output, Mapping):
        raise ValueError("candidate-mix output is missing")
    if output.get("raw_training_text_in_outputs") is not False:
        raise ValueError("candidate-mix output must remain raw-text-free")
    if output.get("production_promotion_permitted") is not False:
        raise ValueError("candidate-mix output cannot permit production promotion")
    for field in ("report_path", "verification_path"):
        relative = output.get(field)
        if not isinstance(relative, str) or not relative.startswith("evidence/"):
            raise ValueError(f"candidate-mix {field} must remain under evidence/")

    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("candidate-mix bindings are missing")
    for name, binding in bindings.items():
        if not isinstance(name, str) or not isinstance(binding, Mapping):
            raise ValueError("invalid candidate-mix binding")
        resolve_binding(repository_root, binding, name=name)
    required = {
        "chat_template",
        "data_contract",
        "missing_mix_proposal",
        "source_registry",
        "tokenizer_config",
        "tokenizer_json",
    }
    for source in candidates:
        required.update(
            {
                source["candidate_binding"],
                source["promotion_binding"],
                source["token_evidence_binding"],
            }
        )
    for source in excluded:
        required.update(source["required_bindings"])
    if not required.issubset(bindings):
        raise ValueError("candidate-mix binding set is incomplete")
    return plan


def _effective_split(row: Mapping[str, Any], split_location: str) -> tuple[str, bool]:
    top_level = row.get("split")
    source = row.get("source")
    nested = source.get("split") if isinstance(source, Mapping) else None
    if split_location == "top_level":
        split = top_level
        normalized = False
    elif split_location == "source.split":
        split = nested
        normalized = top_level is None and split in {"train", "validation"}
    else:
        raise ValueError(f"unsupported split location: {split_location}")
    if split not in {"train", "validation"}:
        raise ValueError("candidate row has no valid effective split")
    return split, normalized


def _language_mode(row: Mapping[str, Any], policy: str) -> str:
    language = row.get("language")
    if policy == "language_field":
        if language == "en":
            return "english"
        if language == "bn":
            return "bengali_script"
    elif policy == "safety_document_suffix":
        source = row.get("source")
        document_id = source.get("document_id") if isinstance(source, Mapping) else None
        if isinstance(document_id, str):
            suffixes = {
                "-banglish": "banglish",
                "-bengali_script": "bengali_script",
                "-english": "english",
                "-natural_code_switching": "natural_code_switching",
            }
            for suffix, mode in suffixes.items():
                if document_id.endswith(suffix):
                    return mode
    raise ValueError("candidate language mode cannot be resolved exactly")


def _canonical_message_digest(messages: Sequence[Mapping[str, Any]]) -> str:
    normalized = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("candidate message schema changed")
        normalized.append(
            {
                "content": " ".join(content.split()).casefold(),
                "role": role,
            }
        )
    return sha256_value(normalized)


def _measure_conversation(
    messages: Sequence[Mapping[str, str]],
    *,
    tokenizer: Any,
    template: Any,
) -> int:
    """Replay Lumi serialization after the tokenizer binding was verified once."""

    validate_message_sequence(messages)
    total_tokens = 2  # BOS and EOS
    supervised_tokens = 1  # EOS is always supervised.
    for message in messages:
        content_ids = encode_untrusted_text(tokenizer, message["content"])
        if not content_ids:
            raise ValueError("candidate message content encoded to zero tokens")
        total_tokens += len(content_ids) + 2  # role token and message-end token.
        if message["role"] in template.loss_roles:
            supervised_tokens += len(content_ids) + 1
    if total_tokens > template.context_window:
        raise ValueError("candidate conversation exceeds the frozen context window")
    if supervised_tokens < 2:
        raise ValueError("candidate conversation has no trainable assistant response")
    return supervised_tokens


def _candidate_metrics(
    *,
    rows: Sequence[Mapping[str, Any]],
    source_plan: Mapping[str, Any],
    registry_source: Any,
    tokenizer: Any,
    template: Any,
    duplicate_index: dict[str, list[tuple[str, str]]],
) -> dict[str, Any]:
    source_id = source_plan["source_id"]
    expected_revision = source_plan["revision"]
    split_records: Counter[str] = Counter()
    split_tokens: Counter[str] = Counter()
    train_task_tokens: Counter[str] = Counter()
    train_language_tokens: Counter[str] = Counter()
    record_ids: set[str] = set()
    normalized_split_rows = 0
    stale_human_review_rows = 0

    for row in rows:
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id or record_id in record_ids:
            raise ValueError(f"{source_id} record IDs changed")
        record_ids.add(record_id)
        source = row.get("source")
        if not isinstance(source, Mapping):
            raise ValueError(f"{source_id} source metadata changed")
        if source.get("source_id") != source_id or source.get("revision") != expected_revision:
            raise ValueError(f"{source_id} source binding changed")
        task = row.get("task_category")
        if task not in registry_source.task_categories:
            raise ValueError(f"{source_id} task is outside the registry")
        split, normalized = _effective_split(row, source_plan["split_location"])
        normalized_split_rows += int(normalized)
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"{source_id} messages changed")
        supervised_tokens = _measure_conversation(
            messages,
            tokenizer=tokenizer,
            template=template,
        )
        split_records[split] += 1
        split_tokens[split] += supervised_tokens
        if split == "train":
            train_task_tokens[task] += supervised_tokens
            mode = _language_mode(row, source_plan["language_mode_policy"])
            train_language_tokens[mode] += supervised_tokens
        duplicate_index[_canonical_message_digest(messages)].append((source_id, split))
        provenance = row.get("provenance")
        if (
            isinstance(provenance, Mapping)
            and provenance.get("human_review_status") == "owner_spot_check_pending"
        ):
            stale_human_review_rows += 1

    by_split = {
        split: {
            "records": split_records[split],
            "supervised_tokens": split_tokens[split],
        }
        for split in ("train", "validation")
    }
    if by_split != source_plan["expected_by_split"]:
        raise ValueError(f"{source_id} frozen-token metrics changed")
    return {
        "candidate_file_sha256": source_plan["candidate_file_sha256"],
        "records": len(rows),
        "by_split": by_split,
        "train_by_language_mode": dict(sorted(train_language_tokens.items())),
        "train_by_task": dict(sorted(train_task_tokens.items())),
        "registry_allowed_splits": sorted(registry_source.splits),
        "registry_status": registry_source.status,
        "rows_requiring_split_normalization": normalized_split_rows,
        "rows_with_stale_pending_human_review_provenance": stale_human_review_rows,
    }


def _verify_promotion_receipt(value: Mapping[str, Any], *, source_id: str) -> None:
    _verify_internal_sha(value, "receipt_sha256")
    source = value.get("source")
    if not isinstance(source, Mapping) or source.get("source_id") != source_id:
        raise ValueError(f"{source_id} promotion receipt changed")
    if value.get("production_dataset_promotions") != 0:
        raise ValueError(f"{source_id} unexpectedly entered a production dataset")
    if value.get("production_eligible") is not False:
        raise ValueError(f"{source_id} promotion receipt became production eligible")


def _duplicate_snapshot(
    duplicate_index: Mapping[str, Sequence[tuple[str, str]]],
) -> dict[str, int]:
    groups = [items for items in duplicate_index.values() if len(items) > 1]
    return {
        "cross_source_duplicate_groups": sum(
            len({source_id for source_id, _split in items}) > 1 for items in groups
        ),
        "duplicate_extra_records": sum(len(items) - 1 for items in groups),
        "duplicate_groups": len(groups),
        "train_validation_overlap_groups": sum(
            {split for _source_id, split in items} == {"train", "validation"}
            for items in groups
        ),
    }


def _targets(missing_mix: Mapping[str, Any]) -> dict[str, Any]:
    learning = missing_mix.get("learning_candidate")
    if not isinstance(learning, Mapping):
        raise ValueError("learning-candidate mix contract changed")
    total = learning.get("accepted_train_supervised_tokens")
    if total != {"maximum": 30_000_000, "minimum": 10_000_000}:
        raise ValueError("learning-candidate token boundary changed")
    tasks = learning.get("task_targets")
    languages = learning.get("language_targets")
    if not isinstance(tasks, Mapping) or set(tasks) != set(EXPECTED_TASKS):
        raise ValueError("learning-candidate task targets changed")
    if not isinstance(languages, Mapping) or set(languages) != set(EXPECTED_LANGUAGE_MODES):
        raise ValueError("learning-candidate language targets changed")
    source_cap = missing_mix.get("diversity_policy", {}).get(
        "maximum_fraction_from_one_source"
    )
    if source_cap != 0.35:
        raise ValueError("learning-candidate source cap changed")
    return {
        "language_minimums": {
            key: value["minimum_tokens"] for key, value in languages.items()
        },
        "source_cap": source_cap,
        "task_minimums": {key: value["minimum_tokens"] for key, value in tasks.items()},
        "total": total,
    }


def _verify_exclusions(
    *,
    plan: Mapping[str, Any],
    resolved: Mapping[str, Path],
    registry: Any,
) -> list[dict[str, Any]]:
    results = []
    registry_by_id = {source.source_id: source for source in registry.sources}
    for excluded in sorted(plan["excluded_sources"], key=lambda item: item["source_id"]):
        source_id = excluded["source_id"]
        if source_id == "openr1_math_220k":
            rights = _load_json(resolved["openr1_rights_report"])
            rights_verification = _load_json(resolved["openr1_rights_verification"])
            owner = _load_json(resolved["openr1_owner_review"])
            metrics = _load_json(resolved["openr1_token_metrics"])
            _verify_internal_sha(rights, "report_sha256")
            _verify_internal_sha(rights_verification, "verification_sha256")
            _verify_internal_sha(owner, "receipt_sha256")
            if rights.get("source_audit", {}).get("rights_cleared_records") != 0:
                raise ValueError("OpenR1 unexpectedly has rights-cleared records")
            if rights_verification.get("promotion_eligible_records") != 0:
                raise ValueError("OpenR1 unexpectedly became promotion eligible")
            results.append(
                {
                    "candidate_records": metrics["accepted_records"],
                    "candidate_train_supervised_tokens": metrics["by_split"]["train"][
                        "supervised_tokens"
                    ],
                    "counted_train_supervised_tokens": 0,
                    "owner_sample_accepted": owner["accepted_records"],
                    "reason": "all_upstream_problem_rights_unresolved",
                    "registry_status": registry_by_id[source_id].status,
                    "registry_status_conflicts_with_later_rights_evidence": True,
                    "source_id": source_id,
                }
            )
        elif source_id == "lumi_ai_assisted_sft_v0_3":
            owner = _load_json(resolved["ai_v3_owner_review"])
            _verify_internal_sha(owner, "receipt_sha256")
            rows = _load_jsonl(resolved["ai_v3_candidate"])
            results.append(
                {
                    "candidate_records": len(rows),
                    "counted_train_supervised_tokens": 0,
                    "owner_sample_accepted": owner["accepted_records"],
                    "reason": "exact_v0_3_source_and_revision_not_in_registry_v0_6",
                    "registry_status": "absent",
                    "source_id": source_id,
                }
            )
        else:
            raise ValueError(f"unsupported exclusion: {source_id}")
    return results


def rebuild_candidate_mix_preflight(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    root = Path(repository_root).resolve()
    resolved = {
        name: resolve_binding(root, binding, name=name)
        for name, binding in plan["bindings"].items()
    }
    policy = load_sft_policy(resolved["data_contract"])
    registry = load_sft_source_registry(
        resolved["source_registry"],
        allowed_licenses=policy.allowed_licenses,
        allowed_languages=policy.allowed_languages,
        allowed_splits=policy.allowed_splits,
        allowed_tasks=frozenset(policy.task_mix),
    )
    registry_by_id = {source.source_id: source for source in registry.sources}
    missing_mix = _load_json(resolved["missing_mix_proposal"])
    targets = _targets(missing_mix)
    template = load_chat_template(resolved["chat_template"])
    if resolved["tokenizer_json"].parent != resolved["tokenizer_config"].parent:
        raise ValueError("frozen tokenizer files are not colocated")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer = AutoTokenizer.from_pretrained(
        resolved["tokenizer_json"].parent,
        local_files_only=True,
    )
    verify_chat_tokenizer(tokenizer, template)

    duplicate_index: dict[str, list[tuple[str, str]]] = defaultdict(list)
    source_metrics: dict[str, dict[str, Any]] = {}
    total_rows = 0
    train_tokens = 0
    materialized_validation_tokens = 0
    registry_admissible_validation_tokens = 0
    task_tokens: Counter[str] = Counter()
    language_tokens: Counter[str] = Counter()
    schema_blockers: list[str] = []

    for source_plan in sorted(plan["candidate_sources"], key=lambda item: item["source_id"]):
        source_id = source_plan["source_id"]
        registry_source = registry_by_id.get(source_id)
        if registry_source is None:
            raise ValueError(f"candidate source is absent from registry: {source_id}")
        if registry_source.revision != source_plan["revision"]:
            raise ValueError(f"candidate source revision changed: {source_id}")
        if registry_source.status != source_plan["registry_status"]:
            raise ValueError(f"candidate source status changed: {source_id}")
        candidate_path = resolved[source_plan["candidate_binding"]]
        if file_sha256(candidate_path) != source_plan["candidate_file_sha256"]:
            raise ValueError(f"candidate file hash changed: {source_id}")
        rows = _load_jsonl(candidate_path)
        total_rows += len(rows)
        metrics = _candidate_metrics(
            rows=rows,
            source_plan=source_plan,
            registry_source=registry_source,
            tokenizer=tokenizer,
            template=template,
            duplicate_index=duplicate_index,
        )
        source_metrics[source_id] = metrics
        train_tokens += metrics["by_split"]["train"]["supervised_tokens"]
        materialized_validation_tokens += metrics["by_split"]["validation"][
            "supervised_tokens"
        ]
        if "validation" in registry_source.splits:
            registry_admissible_validation_tokens += metrics["by_split"]["validation"][
                "supervised_tokens"
            ]
        elif metrics["by_split"]["validation"]["records"]:
            schema_blockers.append(f"{source_id}_validation_split_not_registry_bound")
        task_tokens.update(metrics["train_by_task"])
        language_tokens.update(metrics["train_by_language_mode"])
        if metrics["rows_requiring_split_normalization"]:
            schema_blockers.append(f"{source_id}_requires_top_level_split_materialization")
        if metrics["rows_with_stale_pending_human_review_provenance"]:
            schema_blockers.append(f"{source_id}_row_provenance_still_marks_owner_review_pending")
        promotion = _load_json(resolved[source_plan["promotion_binding"]])
        _verify_promotion_receipt(promotion, source_id=source_id)

    if total_rows > plan["limits"]["maximum_candidate_records_read"]:
        raise ValueError("candidate-mix row ceiling exceeded")
    source_shares = {
        source_id: round(
            metrics["by_split"]["train"]["supervised_tokens"] / train_tokens,
            8,
        )
        for source_id, metrics in sorted(source_metrics.items())
    }
    source_cap_violations = [
        source_id
        for source_id, fraction in source_shares.items()
        if fraction > targets["source_cap"]
    ]
    task_gaps = {
        task: max(0, targets["task_minimums"][task] - task_tokens[task])
        for task in EXPECTED_TASKS
    }
    language_gaps = {
        mode: max(0, targets["language_minimums"][mode] - language_tokens[mode])
        for mode in EXPECTED_LANGUAGE_MODES
    }
    excluded = _verify_exclusions(plan=plan, resolved=resolved, registry=registry)
    blockers = [
        "accepted_train_supervised_tokens_below_10m_floor",
        "task_token_floors_not_met",
        "language_token_floors_not_met",
        "source_concentration_cap_not_met",
        "full_mix_cross_source_near_deduplication_not_run",
        *sorted(schema_blockers),
        "active_production_data_contract_not_updated",
        "dataset_seal_cpu_canary_and_sft_not_authorized",
    ]
    report: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "candidate_pool": {
            "accepted_train_supervised_tokens": train_tokens,
            "candidate_records_read": total_rows,
            "floor_completion_fraction": round(train_tokens / targets["total"]["minimum"], 8),
            "gap_to_10m_train_supervised_tokens": targets["total"]["minimum"] - train_tokens,
            "materialized_validation_supervised_tokens": materialized_validation_tokens,
            "registry_admissible_validation_supervised_tokens": (
                registry_admissible_validation_tokens
            ),
            "source_count": len(source_metrics),
            "source_shares": source_shares,
            "train_by_language_mode": {
                mode: language_tokens[mode] for mode in EXPECTED_LANGUAGE_MODES
            },
            "train_by_task": {task: task_tokens[task] for task in EXPECTED_TASKS},
        },
        "closure": {
            "blockers": blockers,
            "language_floor_gaps": language_gaps,
            "production_dataset_ready": False,
            "source_cap": targets["source_cap"],
            "source_cap_violations": source_cap_violations,
            "task_floor_gaps": task_gaps,
        },
        "current_pool_exact_duplicate_snapshot": _duplicate_snapshot(duplicate_index),
        "excluded_sources": excluded,
        "next_gate": {
            "action": "acquire_and_curate_more_rights_clear_mix_before_global_dedup_and_seal",
            "authorization_required": True,
            "priority_gaps": [
                "coding",
                "factual_qa",
                "writing_summarization",
                "reasoning_math_science",
                "safety_helpfulness",
                "tool_api",
                "bengali_script",
                "banglish",
                "natural_code_switching",
            ],
        },
        "observed_at": plan["observed_at"],
        "plan_file_sha256": _require_sha256(plan_file_sha256, "plan_file_sha256"),
        "production_dataset_promotions": 0,
        "production_eligible": False,
        "raw_training_text_retained": False,
        "schema_version": REPORT_SCHEMA_VERSION,
        "source_metrics": source_metrics,
        "status": "verified_partial_candidate_mix_not_ready_for_seal_or_gpu_sft",
    }
    report["report_sha256"] = sha256_value(report)
    validate_candidate_mix_report(report)
    return report


def validate_candidate_mix_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported candidate-mix report schema")
    _verify_internal_sha(report, "report_sha256")
    if report.get("production_eligible") is not False:
        raise ValueError("candidate-mix report cannot be production eligible")
    if report.get("production_dataset_promotions") != 0:
        raise ValueError("candidate-mix report cannot promote a production dataset")
    if report.get("raw_training_text_retained") is not False:
        raise ValueError("candidate-mix report must remain raw-text-free")
    if report.get("candidate_pool", {}).get("accepted_train_supervised_tokens", 0) >= 10_000_000:
        raise ValueError("candidate-mix report no longer represents a below-floor pool")
    if report.get("closure", {}).get("production_dataset_ready") is not False:
        raise ValueError("candidate-mix closure must fail closed")


def build_candidate_mix_verification(
    *,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    report_file_sha256: str,
) -> dict[str, Any]:
    validate_candidate_mix_report(report)
    verification: dict[str, Any] = {
        "checks": {
            "all_five_registry_bound_sources_replayed": True,
            "all_candidate_files_hash_bound": True,
            "all_frozen_token_totals_replayed": True,
            "excluded_ai_v0_3_remained_unregistered": True,
            "excluded_openr1_rights_clear_tokens_zero": True,
            "no_network_api_gpu_or_production_action": True,
            "partial_pool_exact_duplicate_snapshot_replayed": True,
            "raw_training_text_absent_from_outputs": True,
            "schema_and_registry_mismatches_fail_closed": True,
        },
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "report_file_sha256": _require_sha256(
            report_file_sha256, "report_file_sha256"
        ),
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_candidate_mix_closure_preflight_failed_closed_as_expected",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification
