"""Non-mutating registry-v0.13 cross-source hygiene projection."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from .data_quality import MinHashNearDuplicateIndex
from .posttraining_ai_v3_provenance_materialization import _verify_internal_sha
from .posttraining_candidate_mix_preflight import (
    EXPECTED_LANGUAGE_MODES,
    EXPECTED_TASKS,
    _canonical_message_digest,
    _duplicate_snapshot,
    _load_json,
    _load_jsonl,
    _measure_conversation,
    _targets,
    file_sha256,
    sha256_value,
)
from .posttraining_candidate_mix_reaudit import (
    BenchmarkRegistry,
    _near_duplicate_audit,
    _role_tagged_conversation,
    _UnionFind,
    resolve_binding,
)
from .posttraining_candidate_mix_reaudit_v8 import (
    EXPECTED_BENCHMARK_REGISTRY_SHA256,
    EXPECTED_INPUT_RECORDS,
    EXPECTED_REGISTRY_SHA256,
    _effective_plan,
    _row_language_mode,
    load_candidate_mix_reaudit_v8_plan,
    validate_candidate_mix_reaudit_v8_report,
)
from .posttraining_data import filter_sft_records, load_sft_policy
from .posttraining_format import load_chat_template, verify_chat_tokenizer
from .posttraining_sources import load_sft_source_registry

PLAN_SCHEMA_VERSION = "lumi.sft-candidate-mix-hygiene-preflight-plan.v0.2"
REPORT_SCHEMA_VERSION = "lumi.sft-candidate-mix-hygiene-preflight-report.v0.2"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-candidate-mix-hygiene-preflight-verification.v0.2"

EXPECTED_AUTHORIZATION = {
    "candidate_materialization_authorized": False,
    "candidate_registry_change_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "local_candidate_read_authorized": True,
    "local_projection_authorized": True,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_promotion_authorized": False,
    "production_sft_authorized": False,
    "purpose": "registry_v0_13_cross_source_hygiene_projection_only",
    "unrelated_access_authorized": False,
}
EXPECTED_LIMITS = {
    "maximum_candidate_records_read": 300_000,
    "maximum_network_bytes": 0,
    "maximum_projected_removals": 2_000,
    "maximum_runtime_minutes": 180,
}
EXPECTED_OUTPUT = {
    "candidate_files_materialized": False,
    "production_promotion_permitted": False,
    "raw_training_text_in_outputs": False,
    "report_path": "evidence/lumi-4b-sft-candidate-mix-hygiene-preflight-v0.2.json",
    "verification_path": (
        "evidence/lumi-4b-sft-candidate-mix-hygiene-preflight-verification-v0.2.json"
    ),
}
EXPECTED_BINDINGS = frozenset(
    {
        "implementation",
        "predecessor_plan",
        "predecessor_report",
        "predecessor_verification",
        "runner",
        "source_registry",
        "test",
        "verifier",
    }
)
EXPECTED_SELECTION_POLICY = {
    "primary": "validation_before_train",
    "secondary": "quality_score_descending",
    "tertiary": "supervised_tokens_descending",
    "quaternary": "source_id_ascending",
    "tie_breaker": "record_id_ascending",
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

EXPECTED_PREDECESSOR_PLAN_SHA256 = (
    "83eb5b2186069d172aa679c26fc8a7f3161acb0df32bf4a0f502e03e263b4a48"
)
EXPECTED_PREDECESSOR_REPORT_SHA256 = (
    "cf6246e42b5d4a78a1e0bff461b05eaf71def13e3a3938b1ccdc95c0a965fb90"
)
EXPECTED_PREDECESSOR_REPORT_INTERNAL_SHA256 = (
    "c59536268128799d08e83d2a022566b76bcd4c11ffa8795bf225d464d68683b6"
)
EXPECTED_PREDECESSOR_VERIFICATION_SHA256 = (
    "bcc6e91c94ee8a99eaaf55e1cf2ea9cf9417188ade428a08b6a3088fae926aa8"
)
EXPECTED_PREDECESSOR_VERIFICATION_INTERNAL_SHA256 = (
    "89de48a1be6d6d6c49a4031c5fa53b4fcec574930079c52ed4faff4cf17b8f0e"
)
EXPECTED_COMPONENT_LEDGER_SHA256 = (
    "05897225145e7a94f2f442d4fe1c5d294f3649e87ceddc8a2d81fadbc4f2cd41"
)
EXPECTED_GENERIC_REJECTIONS = 766
EXPECTED_COMPONENTS = 484
EXPECTED_PROJECTED_REMOVALS = 1_250
EXPECTED_PROJECTED_RECORDS = 263_757
OASST1_SOURCE_ID = "openassistant_oasst1"
OASST2_SOURCE_ID = "openassistant_oasst2"
AFFECTED_SOURCE_IDS = frozenset({OASST1_SOURCE_ID, OASST2_SOURCE_ID})


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_hygiene_preflight_v2_plan(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    plan = _load_json(Path(path))
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "canonical_selection_policy": EXPECTED_SELECTION_POLICY,
        "limits": EXPECTED_LIMITS,
        "near_duplicate_policy": EXPECTED_NEAR_DUPLICATE_POLICY,
        "observed_at": "2026-08-08",
        "output": EXPECTED_OUTPUT,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "authorized_bounded_cpu_only_non_mutating_registry_v0_13_hygiene_preflight",
    }
    for field, expected_value in expected.items():
        if plan.get(field) != expected_value:
            raise ValueError(f"candidate hygiene v0.2 plan changed: {field}")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("candidate hygiene v0.2 binding set changed")
    resolved = {
        name: resolve_binding(repository_root, binding, name=name)
        for name, binding in bindings.items()
    }
    exact = {
        "predecessor_plan": EXPECTED_PREDECESSOR_PLAN_SHA256,
        "predecessor_report": EXPECTED_PREDECESSOR_REPORT_SHA256,
        "predecessor_verification": EXPECTED_PREDECESSOR_VERIFICATION_SHA256,
        "source_registry": EXPECTED_REGISTRY_SHA256,
    }
    for name, expected_sha256 in exact.items():
        if bindings[name]["sha256"] != expected_sha256:
            raise ValueError(f"candidate hygiene v0.2 binding changed: {name}")
    predecessor_plan = load_candidate_mix_reaudit_v8_plan(
        resolved["predecessor_plan"], repository_root=repository_root
    )
    if predecessor_plan["bindings"]["source_registry"]["sha256"] != (EXPECTED_REGISTRY_SHA256):
        raise ValueError("candidate hygiene v0.2 predecessor registry changed")
    predecessor_report = _load_json(resolved["predecessor_report"])
    predecessor_verification = _load_json(resolved["predecessor_verification"])
    validate_candidate_mix_reaudit_v8_report(predecessor_report)
    _verify_internal_sha(predecessor_verification, "verification_sha256")
    if predecessor_report.get("report_sha256") != (EXPECTED_PREDECESSOR_REPORT_INTERNAL_SHA256):
        raise ValueError("candidate hygiene v0.2 predecessor report changed")
    if predecessor_verification.get("verification_sha256") != (
        EXPECTED_PREDECESSOR_VERIFICATION_INTERNAL_SHA256
    ):
        raise ValueError("candidate hygiene v0.2 predecessor verification changed")
    if predecessor_verification.get("report_file_sha256") != (EXPECTED_PREDECESSOR_REPORT_SHA256):
        raise ValueError("candidate hygiene v0.2 report link changed")
    return plan


def _load_tokenizer(*, resolved: Mapping[str, Path], template: Any) -> Any:
    from transformers import AutoTokenizer

    if resolved["tokenizer_json"].parent != resolved["tokenizer_config"].parent:
        raise ValueError("frozen tokenizer files are not colocated")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer = AutoTokenizer.from_pretrained(
        resolved["tokenizer_json"].parent,
        local_files_only=True,
    )
    verify_chat_tokenizer(tokenizer, template)
    return tokenizer


def _candidate_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _measure_rows(
    rows: Sequence[Mapping[str, Any]], *, tokenizer: Any, template: Any
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    by_id: dict[str, int] = {}
    for row in rows:
        record_id = row.get("record_id")
        split = row.get("split")
        if not isinstance(record_id, str) or record_id in by_id:
            raise ValueError("candidate hygiene v0.2 record IDs changed")
        if split not in {"train", "validation"}:
            raise ValueError("candidate hygiene v0.2 split changed")
        tokens = _measure_conversation(row["messages"], tokenizer=tokenizer, template=template)
        by_id[record_id] = tokens
        by_split[split]["records"] += 1
        by_split[split]["supervised_tokens"] += tokens
    return (
        {
            split: {
                "records": by_split[split]["records"],
                "supervised_tokens": by_split[split]["supervised_tokens"],
            }
            for split in ("train", "validation")
        },
        by_id,
    )


def _rank_member(member: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        0 if member["split"] == "validation" else 1,
        -float(member["quality_score"]),
        -int(member["supervised_tokens"]),
        str(member["source_id"]),
        str(member["record_id"]),
    )


def _cross_source_components(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any],
    tokenizer: Any,
    template: Any,
) -> tuple[list[dict[str, Any]], str]:
    record_ids = [row.get("record_id") for row in rows]
    if any(not isinstance(record_id, str) or not record_id for record_id in record_ids):
        raise ValueError("candidate hygiene v0.2 component IDs changed")
    string_ids = [str(record_id) for record_id in record_ids]
    if len(set(string_ids)) != len(string_ids):
        raise ValueError("candidate hygiene v0.2 component IDs are not unique")
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
    edges: dict[frozenset[str], float] = {}
    for row in sorted(rows, key=lambda item: str(item["record_id"])):
        record_id = str(row["record_id"])
        signature = index.signature(_role_tagged_conversation(row))
        for candidate in sorted(index.index.query(signature)):
            similarity = signature.jaccard(index.signatures[candidate])
            if similarity >= float(policy["threshold"]):
                union_find.union(record_id, candidate)
                edges[frozenset((record_id, candidate))] = similarity
        index.add(record_id, signature)

    by_root: dict[str, list[str]] = defaultdict(list)
    for record_id in sorted(string_ids):
        by_root[union_find.find(record_id)].append(record_id)
    groups = sorted(
        (sorted(group) for group in by_root.values() if len(group) > 1),
        key=lambda group: group[0],
    )
    ledger = [
        [
            {
                "record_id": record_id,
                "source_id": by_id[record_id]["source"]["source_id"],
                "split": by_id[record_id]["split"],
            }
            for record_id in group
        ]
        for group in groups
    ]
    ledger_sha256 = sha256_value(ledger)
    if ledger_sha256 != EXPECTED_COMPONENT_LEDGER_SHA256:
        raise ValueError("candidate hygiene v0.2 component ledger changed")
    if len(groups) != EXPECTED_COMPONENTS:
        raise ValueError("candidate hygiene v0.2 component count changed")

    components: list[dict[str, Any]] = []
    for group in groups:
        if len(group) != 2:
            raise ValueError("candidate hygiene v0.2 component size changed")
        group_rows = [by_id[record_id] for record_id in group]
        sources = {row["source"]["source_id"] for row in group_rows}
        if sources != AFFECTED_SOURCE_IDS:
            raise ValueError("candidate hygiene v0.2 component sources changed")
        members = []
        for row in group_rows:
            score = row.get("quality", {}).get("score")
            if not isinstance(score, (int, float)):
                raise ValueError("candidate hygiene v0.2 quality score changed")
            tokens = _measure_conversation(row["messages"], tokenizer=tokenizer, template=template)
            members.append(
                {
                    "messages_sha256": _canonical_message_digest(row["messages"]),
                    "quality_score": float(score),
                    "record_id": row["record_id"],
                    "source_id": row["source"]["source_id"],
                    "split": row["split"],
                    "supervised_tokens": tokens,
                    "task_category": row["task_category"],
                }
            )
        ranked = sorted(members, key=_rank_member)
        component_binding = {
            "members": sorted(members, key=lambda item: item["record_id"]),
            "selection_policy": EXPECTED_SELECTION_POLICY,
        }
        components.append(
            {
                "component_sha256": sha256_value(component_binding),
                "drop": ranked[1],
                "estimated_jaccard": round(edges[frozenset(group)], 8),
                "keep": ranked[0],
                "members": component_binding["members"],
                "selection_policy": EXPECTED_SELECTION_POLICY,
            }
        )
    return components, ledger_sha256


def _projection_summary(
    rows: Sequence[Mapping[str, Any]], *, by_split: Mapping[str, Any]
) -> dict[str, Any]:
    candidate = _candidate_bytes(rows)
    return {
        "by_split": dict(by_split),
        "candidate_file_sha256": _bytes_sha256(candidate),
        "candidate_jsonl_bytes": len(candidate),
        "message_binding_sha256": sha256_value(
            [
                {
                    "messages_sha256": _canonical_message_digest(row["messages"]),
                    "record_id": row["record_id"],
                }
                for row in rows
            ]
        ),
        "record_ids_sha256": sha256_value([row["record_id"] for row in rows]),
        "records": len(rows),
    }


def rebuild_hygiene_preflight_v2(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan_resolved = {
        name: resolve_binding(root, binding, name=name)
        for name, binding in plan["bindings"].items()
    }
    predecessor_plan = load_candidate_mix_reaudit_v8_plan(
        plan_resolved["predecessor_plan"], repository_root=root
    )
    predecessor_report = _load_json(plan_resolved["predecessor_report"])
    predecessor_verification = _load_json(plan_resolved["predecessor_verification"])
    validate_candidate_mix_reaudit_v8_report(predecessor_report)
    _verify_internal_sha(predecessor_verification, "verification_sha256")
    if predecessor_report["report_sha256"] != (EXPECTED_PREDECESSOR_REPORT_INTERNAL_SHA256):
        raise ValueError("candidate hygiene v0.2 predecessor report changed")
    if predecessor_verification["verification_sha256"] != (
        EXPECTED_PREDECESSOR_VERIFICATION_INTERNAL_SHA256
    ):
        raise ValueError("candidate hygiene v0.2 predecessor verification changed")

    effective = _effective_plan(predecessor_plan, repository_root=root)
    resolved = {
        name: resolve_binding(root, binding, name=name)
        for name, binding in effective["bindings"].items()
    }
    registry = load_sft_source_registry(
        plan_resolved["source_registry"],
        allowed_licenses=load_sft_policy(resolved["data_contract"]).allowed_licenses,
        allowed_languages=load_sft_policy(resolved["data_contract"]).allowed_languages,
        allowed_splits=load_sft_policy(resolved["data_contract"]).allowed_splits,
        allowed_tasks=frozenset(load_sft_policy(resolved["data_contract"]).task_mix),
        expected_sha256=EXPECTED_REGISTRY_SHA256,
    )
    base_policy = load_sft_policy(resolved["data_contract"])
    audit_policy = replace(
        base_policy,
        benchmark_decontamination=dict(predecessor_plan["benchmark_decontamination"]),
    )
    benchmark_registry = BenchmarkRegistry(audit_policy.benchmark_decontamination, root)
    if benchmark_registry.sha256 != EXPECTED_BENCHMARK_REGISTRY_SHA256:
        raise ValueError("candidate hygiene v0.2 benchmark registry changed")
    template = load_chat_template(resolved["chat_template"])
    tokenizer = _load_tokenizer(resolved=resolved, template=template)

    source_plans = {source["source_id"]: source for source in effective["candidate_sources"]}
    rows_by_source: dict[str, list[dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []
    for source_id, source_plan in sorted(source_plans.items()):
        candidate_path = resolved[source_plan["candidate_binding"]]
        if file_sha256(candidate_path) != source_plan["candidate_file_sha256"]:
            raise ValueError(f"candidate hygiene v0.2 input changed: {source_id}")
        rows = _load_jsonl(candidate_path)
        if any(row.get("source", {}).get("source_id") != source_id for row in rows):
            raise ValueError(f"candidate hygiene v0.2 source ID changed: {source_id}")
        rows_by_source[source_id] = rows
        all_rows.extend(rows)
    if len(all_rows) != EXPECTED_INPUT_RECORDS:
        raise ValueError("candidate hygiene v0.2 input record count changed")
    if len(all_rows) > plan["limits"]["maximum_candidate_records_read"]:
        raise ValueError("candidate hygiene v0.2 row ceiling exceeded")

    generic_accepted, generic = filter_sft_records(
        all_rows,
        policy=audit_policy,
        benchmark_registry=benchmark_registry,
        source_registry=registry,
    )
    if generic["accepted"] != 264_241 or generic["rejected"] != (EXPECTED_GENERIC_REJECTIONS):
        raise ValueError("candidate hygiene v0.2 generic audit changed")
    if generic["rejection_counts"] != {
        "duplicate_normalized_conversation": EXPECTED_GENERIC_REJECTIONS
    }:
        raise ValueError("candidate hygiene v0.2 generic reasons changed")
    rows_by_id = {row["record_id"]: row for row in all_rows}
    generic_drop_ledger = []
    generic_drop_ids: set[str] = set()
    for decision in generic["decisions"]:
        if decision["accepted"]:
            continue
        record_id = decision["record_id"]
        row = rows_by_id[record_id]
        tokens = _measure_conversation(row["messages"], tokenizer=tokenizer, template=template)
        generic_drop_ids.add(record_id)
        generic_drop_ledger.append(
            {
                "normalized_sha256": decision["normalized_sha256"],
                "reasons": decision["reasons"],
                "record_id": record_id,
                "source_id": row["source"]["source_id"],
                "split": row["split"],
                "supervised_tokens": tokens,
            }
        )
    if {item["source_id"] for item in generic_drop_ledger} != {OASST2_SOURCE_ID}:
        raise ValueError("candidate hygiene v0.2 generic loser source changed")

    components, component_ledger_sha256 = _cross_source_components(
        generic_accepted,
        policy=plan["near_duplicate_policy"],
        tokenizer=tokenizer,
        template=template,
    )
    component_drop_ids = {item["drop"]["record_id"] for item in components}
    if generic_drop_ids & component_drop_ids:
        raise ValueError("candidate hygiene v0.2 loser ledgers overlap")
    all_drop_ids = generic_drop_ids | component_drop_ids
    if len(all_drop_ids) != EXPECTED_PROJECTED_REMOVALS:
        raise ValueError("candidate hygiene v0.2 projected removal count changed")
    if len(all_drop_ids) > plan["limits"]["maximum_projected_removals"]:
        raise ValueError("candidate hygiene v0.2 projected removal ceiling exceeded")
    affected_sources = {rows_by_id[record_id]["source"]["source_id"] for record_id in all_drop_ids}
    if affected_sources != AFFECTED_SOURCE_IDS:
        raise ValueError("candidate hygiene v0.2 affected sources changed")

    projected_rows = [row for row in all_rows if row["record_id"] not in all_drop_ids]
    if len(projected_rows) != EXPECTED_PROJECTED_RECORDS:
        raise ValueError("candidate hygiene v0.2 projected row count changed")
    projected_accepted, projected_filter = filter_sft_records(
        projected_rows,
        policy=audit_policy,
        benchmark_registry=benchmark_registry,
        source_registry=registry,
    )
    if len(projected_accepted) != len(projected_rows) or projected_filter["rejected"]:
        raise ValueError("candidate hygiene v0.2 projection remains inadmissible")

    exact_index: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in projected_rows:
        exact_index[_canonical_message_digest(row["messages"])].append(
            (row["source"]["source_id"], row["split"])
        )
    exact_snapshot = _duplicate_snapshot(exact_index)
    near_snapshot = _near_duplicate_audit(projected_rows, policy=plan["near_duplicate_policy"])
    if exact_snapshot["duplicate_groups"] != 0:
        raise ValueError("candidate hygiene v0.2 projection retains exact duplicates")
    if near_snapshot["multi_record_components"] != 0:
        raise ValueError("candidate hygiene v0.2 projection retains near duplicates")

    projected_candidates: dict[str, dict[str, Any]] = {}
    projected_source_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in projected_rows:
        projected_source_rows[row["source"]["source_id"]].append(row)
    for source_id in sorted(AFFECTED_SOURCE_IDS):
        by_split, _ = _measure_rows(
            projected_source_rows[source_id], tokenizer=tokenizer, template=template
        )
        projected_candidates[source_id] = _projection_summary(
            projected_source_rows[source_id], by_split=by_split
        )

    predecessor_pool = predecessor_report["candidate_pool"]
    train_tokens = predecessor_pool["accepted_train_supervised_tokens"]
    validation_tokens = predecessor_pool["registry_admissible_validation_supervised_tokens"]
    task_tokens = Counter(predecessor_pool["train_by_task"])
    language_tokens = Counter(predecessor_pool["train_by_language_mode"])
    source_train_tokens = Counter()
    for source_id, metrics in predecessor_report["source_metrics"].items():
        source_train_tokens[source_id] = (
            metrics["by_split"]["train"]["supervised_tokens"]
            - metrics["quarantined_by_split"]["train"]["supervised_tokens"]
        )
    component_loser_ledger = []
    for component in components:
        loser = component["drop"]
        row = rows_by_id[loser["record_id"]]
        tokens = loser["supervised_tokens"]
        if loser["split"] == "train":
            train_tokens -= tokens
            task_tokens[row["task_category"]] -= tokens
            language_tokens[_row_language_mode(row, source_plans[loser["source_id"]])] -= tokens
            source_train_tokens[loser["source_id"]] -= tokens
        else:
            validation_tokens -= tokens
        component_loser_ledger.append(
            {
                "component_sha256": component["component_sha256"],
                **loser,
            }
        )
    if (
        min(
            [
                train_tokens,
                validation_tokens,
                *task_tokens.values(),
                *language_tokens.values(),
                *source_train_tokens.values(),
            ]
        )
        < 0
    ):
        raise ValueError("candidate hygiene v0.2 token projection underflowed")
    targets = _targets(_load_json(resolved["missing_mix_proposal"]))
    source_shares = {
        source_id: round(tokens / train_tokens, 8)
        for source_id, tokens in sorted(source_train_tokens.items())
    }
    source_cap_violations = [
        source_id for source_id, share in source_shares.items() if share > targets["source_cap"]
    ]
    task_gaps = {
        task: max(0, targets["task_minimums"][task] - task_tokens[task]) for task in EXPECTED_TASKS
    }
    language_gaps = {
        mode: max(0, targets["language_minimums"][mode] - language_tokens[mode])
        for mode in EXPECTED_LANGUAGE_MODES
    }

    report: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "candidate_files_materialized": False,
        "component_projection": {
            "component_ledger_sha256": component_ledger_sha256,
            "components": components,
            "components_count": len(components),
            "drop_record_ids_sha256": sha256_value(sorted(component_drop_ids)),
            "loser_ledger": component_loser_ledger,
            "selection_ledger_sha256": sha256_value(components),
            "selection_policy": dict(plan["canonical_selection_policy"]),
        },
        "generic_quarantine": {
            "drop_ledger_sha256": sha256_value(generic_drop_ledger),
            "reason_counts": generic["rejection_counts"],
            "records": generic_drop_ledger,
            "total_records": len(generic_drop_ledger),
        },
        "input": {
            "accepted_after_generic_quarantine": generic["accepted"],
            "candidate_records": len(all_rows),
            "predecessor_report_file_sha256": EXPECTED_PREDECESSOR_REPORT_SHA256,
            "predecessor_report_internal_sha256": predecessor_report["report_sha256"],
            "registry_sha256": EXPECTED_REGISTRY_SHA256,
            "source_count": len(rows_by_source),
        },
        "observed_at": plan["observed_at"],
        "plan_file_sha256": plan_file_sha256,
        "post_projection_audit": {
            "exact": exact_snapshot,
            "generic_filter_accepted": projected_filter["accepted"],
            "generic_filter_rejected": projected_filter["rejected"],
            "near": near_snapshot,
        },
        "production_dataset_promotions": 0,
        "production_eligible": False,
        "projected_candidates": projected_candidates,
        "projected_pool": {
            "affected_source_ids": sorted(affected_sources),
            "language_floor_gaps": language_gaps,
            "records": len(projected_rows),
            "removed_records": len(all_drop_ids),
            "source_cap": targets["source_cap"],
            "source_cap_violations": source_cap_violations,
            "source_shares": source_shares,
            "task_floor_gaps": task_gaps,
            "train_by_language_mode": {
                mode: language_tokens[mode] for mode in EXPECTED_LANGUAGE_MODES
            },
            "train_by_task": {task: task_tokens[task] for task in EXPECTED_TASKS},
            "train_supervised_tokens": train_tokens,
            "validation_supervised_tokens": validation_tokens,
        },
        "raw_training_text_retained": False,
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": (
            "verified_non_mutating_registry_v0_13_hygiene_projection_ready_for_materialization_gate"
        ),
    }
    report["report_sha256"] = sha256_value(report)
    validate_hygiene_preflight_v2_report(report)
    return report


def validate_hygiene_preflight_v2_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported candidate hygiene v0.2 report schema")
    _verify_internal_sha(report, "report_sha256")
    if report.get("candidate_files_materialized") is not False:
        raise ValueError("candidate hygiene v0.2 cannot materialize candidates")
    if report.get("production_eligible") is not False:
        raise ValueError("candidate hygiene v0.2 cannot be production eligible")
    if report.get("production_dataset_promotions") != 0:
        raise ValueError("candidate hygiene v0.2 cannot promote a dataset")
    if report.get("raw_training_text_retained") is not False:
        raise ValueError("candidate hygiene v0.2 report must remain raw-text-free")
    if report.get("generic_quarantine", {}).get("total_records") != (EXPECTED_GENERIC_REJECTIONS):
        raise ValueError("candidate hygiene v0.2 generic quarantine changed")
    components = report.get("component_projection", {})
    if components.get("components_count") != EXPECTED_COMPONENTS:
        raise ValueError("candidate hygiene v0.2 component count changed")
    projected = report.get("projected_pool", {})
    if (
        projected.get("records") != EXPECTED_PROJECTED_RECORDS
        or projected.get("removed_records") != EXPECTED_PROJECTED_REMOVALS
        or set(projected.get("affected_source_ids", [])) != AFFECTED_SOURCE_IDS
    ):
        raise ValueError("candidate hygiene v0.2 projected pool changed")
    audit = report.get("post_projection_audit", {})
    if audit.get("generic_filter_rejected") != 0:
        raise ValueError("candidate hygiene v0.2 projection remains inadmissible")
    if audit.get("exact", {}).get("duplicate_groups") != 0:
        raise ValueError("candidate hygiene v0.2 projection retains exact duplicates")
    if audit.get("near", {}).get("multi_record_components") != 0:
        raise ValueError("candidate hygiene v0.2 projection retains near duplicates")
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    for forbidden in ('"messages"', '"content"', '"assistant_response"'):
        if forbidden in serialized:
            raise ValueError("candidate hygiene v0.2 report contains raw training text")


def build_hygiene_preflight_v2_verification(
    *,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    report_file_sha256: str,
    replayed_report: Mapping[str, Any],
) -> dict[str, Any]:
    validate_hygiene_preflight_v2_report(report)
    validate_hygiene_preflight_v2_report(replayed_report)
    if report != replayed_report:
        raise ValueError("candidate hygiene v0.2 independent replay changed")
    verification: dict[str, Any] = {
        "checks": {
            "all_265007_registry_v0_13_rows_replayed": True,
            "candidate_files_not_materialized": True,
            "generic_766_row_quarantine_reproduced": True,
            "no_network_api_gpu_seal_sft_or_production_promotion": True,
            "post_projection_exact_and_near_duplicate_counts_zero": True,
            "projected_oasst1_and_oasst2_hashes_reproduced": True,
            "raw_training_text_absent_from_repo_evidence": True,
            "remaining_484_component_keep_drop_ledger_replayed": True,
            "validation_before_train_selection_reproduced": True,
        },
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "projected_records": report["projected_pool"]["records"],
        "projected_removals": report["projected_pool"]["removed_records"],
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_registry_v0_13_hygiene_projection_nonproduction",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification


__all__ = [
    "EXPECTED_LIMITS",
    "EXPECTED_PROJECTED_RECORDS",
    "EXPECTED_PROJECTED_REMOVALS",
    "build_hygiene_preflight_v2_verification",
    "json_file_bytes",
    "load_hygiene_preflight_v2_plan",
    "rebuild_hygiene_preflight_v2",
    "validate_hygiene_preflight_v2_report",
]
