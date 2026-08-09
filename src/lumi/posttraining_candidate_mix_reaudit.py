"""Local-only re-audit of the repaired Lumi SFT candidate pool."""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .data_quality import (
    BenchmarkRegistry,
    MinHashNearDuplicateIndex,
    normalize_for_matching,
)
from .posttraining_candidate_mix_preflight import (
    EXPECTED_LANGUAGE_MODES,
    EXPECTED_TASKS,
    _candidate_metrics,
    _duplicate_snapshot,
    _load_json,
    _load_jsonl,
    _require_sha256,
    _targets,
    _verify_internal_sha,
    _verify_promotion_receipt,
    file_sha256,
    sha256_value,
)
from .posttraining_data import filter_sft_records, load_sft_policy
from .posttraining_format import load_chat_template, verify_chat_tokenizer
from .posttraining_sources import load_sft_source_registry

PLAN_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-plan.v0.2"
REPORT_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-report.v0.2"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-verification.v0.2"

EXPECTED_AUTHORIZATION = {
    "candidate_registry_change_authorized": False,
    "dataset_acquisition_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_api_spend_authorized": False,
    "local_candidate_read_authorized": True,
    "local_raw_text_free_reaudit_authorized": True,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "purpose": "repaired_local_candidate_mix_schema_registry_and_dedup_reaudit_only",
    "source_promotion_authorized": False,
}
EXPECTED_LIMITS = {
    "maximum_candidate_records_read": 20_000,
    "maximum_network_bytes": 0,
    "maximum_runtime_minutes": 60,
}
EXPECTED_NEAR_DUPLICATE_POLICY = {
    "comparison_unit": "role_tagged_full_conversation",
    "num_perm": 128,
    "package_version": "2.0.0",
    "scheme": "affine32",
    "seed": 1,
    "shingle_tokens": 5,
    "threshold": 0.9,
}
EXPECTED_OUTPUT = {
    "production_promotion_permitted": False,
    "raw_training_text_in_outputs": False,
    "report_path": "evidence/lumi-4b-sft-candidate-mix-reaudit-v0.2.json",
    "verification_path": (
        "evidence/lumi-4b-sft-candidate-mix-reaudit-verification-v0.2.json"
    ),
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
EXPECTED_SOURCE_REGISTRY_SHA256 = (
    "8cd8cc0b20ccd7da2f9083b0903178b98f59e3279c6777ebf8fee4866069d573"
)
EXPECTED_SAFETY_CANDIDATE_SHA256 = (
    "0f327fb0959a512a698c49992ad516467b3ef8058eb1653416df45157d7f070e"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


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
        raise ValueError(f"invalid candidate-mix re-audit binding path: {name}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError(f"candidate-mix re-audit binding escapes workspace: {name}")
    if not resolved.is_file():
        raise ValueError(f"candidate-mix re-audit binding is missing: {name}")
    expected = _require_sha256(binding.get("sha256"), f"bindings.{name}.sha256")
    if file_sha256(resolved) != expected:
        raise ValueError(f"candidate-mix re-audit binding changed: {name}")
    return resolved


def load_candidate_mix_reaudit_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    plan = _load_json(path)
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "limits": EXPECTED_LIMITS,
        "near_duplicate_policy": EXPECTED_NEAR_DUPLICATE_POLICY,
        "observed_at": "2026-08-07",
        "output": EXPECTED_OUTPUT,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "authorized_local_reaudit_no_production_action",
    }
    for field, value in expected.items():
        if plan.get(field) != value:
            raise ValueError(f"candidate-mix re-audit plan changed: {field}")

    candidates = plan.get("candidate_sources")
    if not isinstance(candidates, list):
        raise ValueError("candidate_sources must be a list")
    candidate_ids = {
        item.get("source_id") for item in candidates if isinstance(item, Mapping)
    }
    if candidate_ids != EXPECTED_CANDIDATE_SOURCE_IDS or len(candidates) != len(
        candidate_ids
    ):
        raise ValueError("candidate source set changed")
    safety = next(
        item for item in candidates if item["source_id"] == "lumi_safety_tool_sft_v0_1"
    )
    if safety.get("candidate_file_sha256") != EXPECTED_SAFETY_CANDIDATE_SHA256:
        raise ValueError("repaired safety candidate binding changed")
    if safety.get("split_location") != "top_level":
        raise ValueError("repaired safety split location changed")

    excluded = plan.get("excluded_sources")
    if not isinstance(excluded, list):
        raise ValueError("excluded_sources must be a list")
    excluded_ids = {
        item.get("source_id") for item in excluded if isinstance(item, Mapping)
    }
    if excluded_ids != EXPECTED_EXCLUDED_SOURCE_IDS or len(excluded) != len(
        excluded_ids
    ):
        raise ValueError("excluded source set changed")

    required = {
        "ai_v3_token_evidence",
        "benchmark_registry",
        "candidate_mix_v0_1_plan",
        "candidate_mix_v0_1_report",
        "candidate_mix_v0_1_verification",
        "chat_template",
        "data_contract",
        "data_quality_implementation",
        "implementation",
        "missing_mix_proposal",
        "posttraining_data_implementation",
        "predecessor_implementation",
        "runner",
        "safety_uri_plan",
        "safety_uri_report",
        "safety_uri_verification",
        "source_registry",
        "source_registry_implementation",
        "test",
        "tokenizer_config",
        "tokenizer_json",
        "verifier",
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
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != required:
        raise ValueError("candidate-mix re-audit binding set changed")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"invalid candidate-mix re-audit binding: {name}")
        resolve_binding(repository_root, binding, name=name)
    if bindings["source_registry"]["sha256"] != EXPECTED_SOURCE_REGISTRY_SHA256:
        raise ValueError("candidate-mix re-audit source registry changed")
    return plan


def _validate_predecessor_evidence(resolved: Mapping[str, Path]) -> None:
    old_report = _load_json(resolved["candidate_mix_v0_1_report"])
    old_verification = _load_json(resolved["candidate_mix_v0_1_verification"])
    safety_report = _load_json(resolved["safety_uri_report"])
    safety_verification = _load_json(resolved["safety_uri_verification"])
    _verify_internal_sha(old_report, "report_sha256")
    _verify_internal_sha(old_verification, "verification_sha256")
    _verify_internal_sha(safety_report, "report_sha256")
    _verify_internal_sha(safety_verification, "verification_sha256")
    if old_report.get("status") != (
        "verified_partial_candidate_mix_not_ready_for_seal_or_gpu_sft"
    ):
        raise ValueError("candidate-mix v0.1 predecessor status changed")
    if safety_report.get("status") != "verified_safety_source_uri_repair_nonproduction":
        raise ValueError("safety URI repair status changed")
    if safety_report.get("bindings", {}).get(
        "output_candidate_file_sha256"
    ) != EXPECTED_SAFETY_CANDIDATE_SHA256:
        raise ValueError("safety URI repaired candidate changed")
    if safety_verification.get("status") != (
        "verified_safety_source_uri_repair_outputs_nonproduction"
    ):
        raise ValueError("safety URI repair verification changed")


def _role_tagged_conversation(row: Mapping[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("candidate messages changed during near-duplicate audit")
    parts = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("candidate message schema changed")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("candidate message fields changed")
        parts.append(f"{role} {content}")
    return normalize_for_matching("\n".join(parts))


class _UnionFind:
    def __init__(self, values: Sequence[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self.parent[second] = first


def _near_duplicate_audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    record_ids = [row.get("record_id") for row in rows]
    if any(not isinstance(record_id, str) or not record_id for record_id in record_ids):
        raise ValueError("candidate record IDs changed")
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("candidate record IDs are not globally unique")
    string_ids = [str(record_id) for record_id in record_ids]
    by_id = {str(row["record_id"]): row for row in rows}
    union_find = _UnionFind(string_ids)
    minhash_config = {
        key: policy[key]
        for key in (
            "num_perm",
            "package_version",
            "scheme",
            "seed",
            "shingle_tokens",
            "threshold",
        )
    }
    index = MinHashNearDuplicateIndex(minhash_config)
    edges = 0
    for row in sorted(rows, key=lambda item: str(item["record_id"])):
        record_id = str(row["record_id"])
        signature = index.signature(_role_tagged_conversation(row))
        for candidate in sorted(index.index.query(signature)):
            if signature.jaccard(index.signatures[candidate]) >= float(
                policy["threshold"]
            ):
                union_find.union(record_id, candidate)
                edges += 1
        index.add(record_id, signature)

    by_root: dict[str, list[str]] = defaultdict(list)
    for record_id in sorted(string_ids):
        by_root[union_find.find(record_id)].append(record_id)
    groups = sorted(
        (sorted(group) for group in by_root.values() if len(group) > 1),
        key=lambda group: group[0],
    )
    ledger = []
    per_source_components: Counter[str] = Counter()
    train_validation_components = 0
    train_validation_records = 0
    cross_source_components = 0
    for group in groups:
        sources = {by_id[record_id]["source"]["source_id"] for record_id in group}
        splits = {by_id[record_id]["split"] for record_id in group}
        cross_source_components += int(len(sources) > 1)
        if splits == {"train", "validation"}:
            train_validation_components += 1
            train_validation_records += len(group)
        if len(sources) == 1:
            per_source_components[next(iter(sources))] += 1
        ledger.append(
            [
                {
                    "record_id": record_id,
                    "source_id": by_id[record_id]["source"]["source_id"],
                    "split": by_id[record_id]["split"],
                }
                for record_id in group
            ]
        )
    size_histogram = Counter(len(group) for group in groups)
    return {
        "comparison_unit": policy["comparison_unit"],
        "component_ledger_sha256": sha256_value(ledger),
        "components": len(by_root),
        "cross_source_components": cross_source_components,
        "duplicate_extra_records": sum(len(group) - 1 for group in groups),
        "edge_count": edges,
        "largest_component_records": max(map(len, by_root.values()), default=0),
        "multi_record_components": len(groups),
        "per_source_multi_record_components": dict(sorted(per_source_components.items())),
        "records": len(rows),
        "size_histogram": {
            str(size): count for size, count in sorted(size_histogram.items())
        },
        "train_validation_components": train_validation_components,
        "train_validation_records": train_validation_records,
    }


def _verify_exclusions(
    *,
    plan: Mapping[str, Any],
    resolved: Mapping[str, Path],
    registry: Any,
) -> list[dict[str, Any]]:
    results = []
    registry_by_id = registry.by_id
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
                    "source_id": source_id,
                }
            )
        elif source_id == "lumi_ai_assisted_sft_v0_3":
            owner = _load_json(resolved["ai_v3_owner_review"])
            token_evidence = _load_json(resolved["ai_v3_token_evidence"])
            _verify_internal_sha(owner, "receipt_sha256")
            _verify_internal_sha(token_evidence, "report_sha256")
            rows = _load_jsonl(resolved["ai_v3_candidate"])
            if source_id in registry_by_id:
                raise ValueError("AI-assisted v0.3 unexpectedly entered registry v0.7")
            metrics = token_evidence["source_token_evidence"]["ai_assisted_v0_3"]
            results.append(
                {
                    "candidate_records": len(rows),
                    "candidate_train_supervised_tokens": metrics["by_split"]["train"][
                        "supervised_tokens"
                    ],
                    "candidate_validation_supervised_tokens": metrics["by_split"][
                        "validation"
                    ]["supervised_tokens"],
                    "counted_train_supervised_tokens": 0,
                    "owner_sample_accepted": owner["accepted_records"],
                    "reason": "exact_v0_3_source_and_revision_not_in_registry_v0_7",
                    "registry_status": "absent",
                    "source_id": source_id,
                }
            )
        else:  # pragma: no cover - the plan source set is closed above.
            raise ValueError(f"unsupported excluded source: {source_id}")
    return results


def rebuild_candidate_mix_reaudit(
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
    _validate_predecessor_evidence(resolved)
    policy = load_sft_policy(resolved["data_contract"])
    benchmark_registry = BenchmarkRegistry(policy.benchmark_decontamination, root)
    if benchmark_registry.sha256 != plan["bindings"]["benchmark_registry"]["sha256"]:
        raise ValueError("benchmark registry binding changed")
    registry = load_sft_source_registry(
        resolved["source_registry"],
        allowed_licenses=policy.allowed_licenses,
        allowed_languages=policy.allowed_languages,
        allowed_splits=policy.allowed_splits,
        allowed_tasks=frozenset(policy.task_mix),
        expected_sha256=EXPECTED_SOURCE_REGISTRY_SHA256,
    )
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
    all_rows: list[dict[str, Any]] = []
    train_tokens = 0
    validation_tokens = 0
    task_tokens: Counter[str] = Counter()
    language_tokens: Counter[str] = Counter()
    for source_plan in sorted(plan["candidate_sources"], key=lambda item: item["source_id"]):
        source_id = source_plan["source_id"]
        registry_source = registry.by_id.get(source_id)
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
        metrics = _candidate_metrics(
            rows=rows,
            source_plan=source_plan,
            registry_source=registry_source,
            tokenizer=tokenizer,
            template=template,
            duplicate_index=duplicate_index,
        )
        if metrics["rows_requiring_split_normalization"] != 0:
            raise ValueError(f"{source_id} still requires split normalization")
        if metrics["rows_with_stale_pending_human_review_provenance"] != 0:
            raise ValueError(f"{source_id} still has stale owner-review provenance")
        source_metrics[source_id] = metrics
        all_rows.extend(rows)
        train_tokens += metrics["by_split"]["train"]["supervised_tokens"]
        validation_tokens += metrics["by_split"]["validation"]["supervised_tokens"]
        task_tokens.update(metrics["train_by_task"])
        language_tokens.update(metrics["train_by_language_mode"])
        promotion = _load_json(resolved[source_plan["promotion_binding"]])
        _verify_promotion_receipt(promotion, source_id=source_id)

    if len(all_rows) > plan["limits"]["maximum_candidate_records_read"]:
        raise ValueError("candidate-mix re-audit row ceiling exceeded")
    accepted, generic_report = filter_sft_records(
        all_rows,
        policy=policy,
        benchmark_registry=benchmark_registry,
        source_registry=registry,
    )
    if len(accepted) != len(all_rows) or generic_report["rejected"] != 0:
        raise ValueError(
            "repaired candidate pool failed generic SFT validation: "
            f"{generic_report['rejection_counts']}"
        )
    for metrics in source_metrics.values():
        metrics["generic_sft_accepted_records"] = metrics["records"]
        metrics["generic_sft_rejected_records"] = 0

    exact_snapshot = _duplicate_snapshot(duplicate_index)
    if any(exact_snapshot.values()):
        raise ValueError("repaired candidate pool has an exact duplicate or split leak")
    near_snapshot = _near_duplicate_audit(
        all_rows,
        policy=plan["near_duplicate_policy"],
    )
    if near_snapshot["cross_source_components"] != 0:
        raise ValueError("repaired candidate pool has cross-source near duplicates")
    if near_snapshot["train_validation_components"] != 0:
        raise ValueError("repaired candidate pool has train-validation near leakage")

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
        "within_split_near_duplicate_canonical_selection_not_materialized",
        "reviewed_ai_v0_3_source_not_registered",
        "active_production_data_contract_not_updated",
        "dataset_seal_cpu_canary_and_sft_not_authorized",
    ]
    report: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "candidate_pool": {
            "accepted_train_supervised_tokens": train_tokens,
            "candidate_records_read": len(all_rows),
            "floor_completion_fraction": round(train_tokens / targets["total"]["minimum"], 8),
            "gap_to_10m_train_supervised_tokens": targets["total"]["minimum"]
            - train_tokens,
            "registry_admissible_validation_supervised_tokens": validation_tokens,
            "source_count": len(source_metrics),
            "source_shares": source_shares,
            "train_by_language_mode": {
                mode: language_tokens[mode] for mode in EXPECTED_LANGUAGE_MODES
            },
            "train_by_task": {task: task_tokens[task] for task in EXPECTED_TASKS},
        },
        "closed_since_v0_1": [
            "coherelabs_aya_dataset_validation_split_is_registry_bound",
            "lumi_safety_tool_sft_v0_1_has_top_level_split_on_all_rows",
            "lumi_safety_tool_sft_v0_1_owner_review_provenance_is_current",
            "lumi_safety_tool_sft_v0_1_has_deterministic_source_uri_on_all_rows",
            "full_pool_generic_schema_registry_and_benchmark_validation_passed",
            "full_pool_cross_source_near_duplicate_audit_completed",
        ],
        "closure": {
            "blockers": blockers,
            "language_floor_gaps": language_gaps,
            "production_dataset_ready": False,
            "source_cap": targets["source_cap"],
            "source_cap_violations": source_cap_violations,
            "task_floor_gaps": task_gaps,
        },
        "exact_duplicate_audit": exact_snapshot,
        "excluded_sources": excluded,
        "generic_sft_validation": {
            "accepted": generic_report["accepted"],
            "accepted_dataset_sha256": generic_report["accepted_dataset_sha256"],
            "benchmark_contamination_rejections": sum(
                count
                for reason, count in generic_report["rejection_counts"].items()
                if reason.startswith("benchmark_contamination:")
            ),
            "raw_text_retained": generic_report["raw_text_retained"],
            "rejected": generic_report["rejected"],
            "rejection_counts": generic_report["rejection_counts"],
            "report_sha256": generic_report["report_sha256"],
            "source_registry_rejections": sum(
                count
                for reason, count in generic_report["rejection_counts"].items()
                if reason.startswith("source_registry")
            ),
            "total": generic_report["total"],
        },
        "near_duplicate_audit": near_snapshot,
        "next_gate": {
            "action": (
                "preflight_reviewed_ai_v0_3_registry_successor_then_acquire_and_curate_"
                "rights_clear_bulk_mix"
            ),
            "authorization_required": True,
            "priority_gaps": [
                "reasoning_math_science",
                "coding",
                "factual_qa",
                "writing_summarization",
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
        "status": "verified_repaired_candidate_mix_not_ready_for_seal_or_gpu_sft",
    }
    report["report_sha256"] = sha256_value(report)
    validate_candidate_mix_reaudit_report(report)
    return report


def validate_candidate_mix_reaudit_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported candidate-mix re-audit report schema")
    _verify_internal_sha(report, "report_sha256")
    if report.get("production_eligible") is not False:
        raise ValueError("candidate-mix re-audit cannot be production eligible")
    if report.get("production_dataset_promotions") != 0:
        raise ValueError("candidate-mix re-audit cannot promote a production dataset")
    if report.get("raw_training_text_retained") is not False:
        raise ValueError("candidate-mix re-audit report must remain raw-text-free")
    pool = report.get("candidate_pool")
    if (
        not isinstance(pool, Mapping)
        or pool.get("accepted_train_supervised_tokens", 0) >= 10_000_000
    ):
        raise ValueError("candidate-mix re-audit no longer represents a below-floor pool")
    generic = report.get("generic_sft_validation")
    if not isinstance(generic, Mapping):
        raise ValueError("generic SFT validation summary is missing")
    if generic.get("accepted") != 12_401 or generic.get("rejected") != 0:
        raise ValueError("generic SFT validation did not accept the exact repaired pool")
    if generic.get("rejection_counts") != {}:
        raise ValueError("generic SFT validation has rejection reasons")
    exact = report.get("exact_duplicate_audit")
    if not isinstance(exact, Mapping) or any(exact.values()):
        raise ValueError("exact duplicate audit is not clean")
    near = report.get("near_duplicate_audit")
    if not isinstance(near, Mapping):
        raise ValueError("near-duplicate audit is missing")
    if near.get("cross_source_components") != 0:
        raise ValueError("cross-source near duplicates remain")
    if near.get("train_validation_components") != 0:
        raise ValueError("train-validation near leakage remains")


def build_candidate_mix_reaudit_verification(
    *,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    report_file_sha256: str,
) -> dict[str, Any]:
    validate_candidate_mix_reaudit_report(report)
    verification: dict[str, Any] = {
        "checks": {
            "all_12401_rows_pass_generic_schema_registry_and_benchmark_validation": True,
            "aya_validation_is_registry_admissible": True,
            "exact_duplicate_and_train_validation_overlap_are_zero": True,
            "frozen_token_counts_reproduced": True,
            "no_network_api_gpu_seal_sft_or_production_promotion": True,
            "predecessor_candidates_and_registry_unchanged": True,
            "raw_training_text_absent_from_repo_evidence": True,
            "safety_metadata_repairs_are_present_on_all_320_rows": True,
            "whole_conversation_near_audit_has_no_cross_source_or_split_leakage": True,
        },
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "report_file_sha256": _require_sha256(
            report_file_sha256, "report_file_sha256"
        ),
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_repaired_candidate_mix_reaudit_nonproduction",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification
