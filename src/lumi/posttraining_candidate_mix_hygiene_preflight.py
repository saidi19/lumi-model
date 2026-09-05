"""Non-mutating hygiene projection for the Lumi SFT OASST1 and Aya candidates."""

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
from .posttraining_candidate_mix_preflight import (
    _canonical_message_digest,
    _duplicate_snapshot,
    _load_json,
    _load_jsonl,
    _measure_conversation,
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
from .posttraining_candidate_mix_reaudit_v6 import (
    validate_candidate_mix_reaudit_v6_report,
)
from .posttraining_data import filter_sft_records, load_sft_policy
from .posttraining_format import load_chat_template, verify_chat_tokenizer
from .posttraining_sources import load_sft_source_registry

PLAN_SCHEMA_VERSION = "lumi.sft-candidate-mix-hygiene-preflight-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-candidate-mix-hygiene-preflight-report.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-candidate-mix-hygiene-preflight-verification.v0.1"
)

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
    "purpose": "oasst1_benchmark_quarantine_and_aya_near_duplicate_projection_only",
    "unrelated_access_authorized": False,
}
EXPECTED_LIMITS = {
    "maximum_candidate_records_read": 3_223,
    "maximum_network_bytes": 0,
    "maximum_projected_removals": 23,
    "maximum_runtime_minutes": 60,
}
EXPECTED_OUTPUT = {
    "candidate_files_materialized": False,
    "production_promotion_permitted": False,
    "raw_training_text_in_outputs": False,
    "report_path": (
        "evidence/lumi-4b-sft-candidate-mix-hygiene-preflight-v0.1.json"
    ),
    "verification_path": (
        "evidence/lumi-4b-sft-candidate-mix-hygiene-preflight-verification-v0.1.json"
    ),
}
EXPECTED_BINDINGS = frozenset(
    {
        "aya_candidate",
        "benchmark_registry",
        "candidate_mix_preflight_implementation",
        "data_contract",
        "data_quality_implementation",
        "implementation",
        "near_duplicate_implementation",
        "oasst1_candidate",
        "posttraining_data_implementation",
        "predecessor_report",
        "predecessor_verification",
        "runner",
        "source_registry",
        "test",
        "tokenizer_config",
        "tokenizer_json",
        "verifier",
        "chat_template",
    }
)
EXPECTED_NEAR_DUPLICATE_POLICY = {
    "comparison_unit": "role_tagged_full_conversation",
    "num_perm": 128,
    "package_version": "2.0.0",
    "scheme": "affine32",
    "seed": 1,
    "shingle_tokens": 5,
    "threshold": 0.9,
}
EXPECTED_SELECTION_POLICY = {
    "primary": "quality_score_descending",
    "secondary": "supervised_tokens_descending",
    "tie_breaker": "record_id_ascending",
}

OASST1_SOURCE_ID = "openassistant_oasst1"
AYA_SOURCE_ID = "coherelabs_aya_dataset"
EXPECTED_INPUTS = {
    OASST1_SOURCE_ID: {
        "candidate_file_sha256": (
            "15bdb511c4011fd4680cdf87c84de1cbcc979ce5e3792709e16ed3bef25e7b98"
        ),
        "by_split": {
            "train": {"records": 1_652, "supervised_tokens": 453_981},
            "validation": {"records": 93, "supervised_tokens": 23_475},
        },
    },
    AYA_SOURCE_ID: {
        "candidate_file_sha256": (
            "b9a9154565d769b001fc6a595699bff6bd63587aa66c5c53070257716dad143b"
        ),
        "by_split": {
            "train": {"records": 1_330, "supervised_tokens": 520_269},
            "validation": {"records": 148, "supervised_tokens": 57_184},
        },
    },
}
EXPECTED_PROJECTED = {
    OASST1_SOURCE_ID: {
        "by_split": {
            "train": {"records": 1_649, "supervised_tokens": 452_690},
            "validation": {"records": 93, "supervised_tokens": 23_475},
        },
        "removed_records": 3,
    },
    AYA_SOURCE_ID: {
        "by_split": {
            "train": {"records": 1_313, "supervised_tokens": 513_379},
            "validation": {"records": 145, "supervised_tokens": 55_581},
        },
        "removed_records": 20,
    },
}
EXPECTED_OASST1_DROPS = {
    "openassistant_oasst1-7df6fce018173558068fd806": {
        "normalized_sha256": (
            "86ddbf25724e0748b4792a967756ce9521d50452eda51dabb4348d493686c96c"
        ),
        "supervised_tokens": 306,
    },
    "openassistant_oasst1-8765b63c3f8fa940fa4f0d74": {
        "normalized_sha256": (
            "a9327c219b0c3e50fcda1fcdfe437c4e89d34e0c7a5f9d38415fadffda933574"
        ),
        "supervised_tokens": 326,
    },
    "openassistant_oasst1-b5bef8c16cda087ee5ba944a": {
        "normalized_sha256": (
            "f7035dc4f51367129732a845c726d513828633f95ec84568b92501305aec99ed"
        ),
        "supervised_tokens": 659,
    },
}
EXPECTED_PREDECESSOR_REPORT_SHA256 = (
    "58d5cb212566ca9292a611b053b70378863bf00c391bd94a700f7322a2a67057"
)
EXPECTED_PREDECESSOR_VERIFICATION_SHA256 = (
    "7ee9929ab6ec54e2e665645f1e8827607d3dfd69853fb84b0fa71b8b32fc158d"
)
EXPECTED_PREDECESSOR_COMPONENT_LEDGER_SHA256 = (
    "b2a5e1b894f2d6c2a4413475a00ede0177323b517dedf6b02e6601a328dd3740"
)
EXPECTED_SOURCE_REGISTRY_SHA256 = (
    "5353c57189d82ee22306ccd11d116d017c5e683811cec2d6704cf45bcf7f02e7"
)
EXPECTED_BENCHMARK_REGISTRY_SHA256 = (
    "1ffb5e938ab28c2fc876b9efd6c6ed59059f1e85aed2c396c024ae5925a4745e"
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _verify_internal_sha(value: Mapping[str, Any], field: str) -> str:
    observed = value.get(field)
    if not isinstance(observed, str) or len(observed) != 64:
        raise ValueError(f"missing internal SHA-256: {field}")
    unsigned = {key: item for key, item in value.items() if key != field}
    if sha256_value(unsigned) != observed:
        raise ValueError(f"internal SHA-256 changed: {field}")
    return observed


def load_hygiene_preflight_plan(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    plan = _load_json(Path(path))
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "canonical_selection_policy": EXPECTED_SELECTION_POLICY,
        "limits": EXPECTED_LIMITS,
        "near_duplicate_policy": EXPECTED_NEAR_DUPLICATE_POLICY,
        "observed_at": "2026-08-07",
        "output": EXPECTED_OUTPUT,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "authorized_bounded_cpu_only_non_mutating_hygiene_preflight",
    }
    for field, expected_value in expected.items():
        if plan.get(field) != expected_value:
            raise ValueError(f"candidate hygiene preflight plan changed: {field}")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("candidate hygiene preflight binding set changed")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"invalid candidate hygiene binding: {name}")
        resolve_binding(repository_root, binding, name=name)
    if bindings["source_registry"]["sha256"] != EXPECTED_SOURCE_REGISTRY_SHA256:
        raise ValueError("candidate hygiene source-registry binding changed")
    if (
        bindings["benchmark_registry"]["sha256"]
        != EXPECTED_BENCHMARK_REGISTRY_SHA256
    ):
        raise ValueError("candidate hygiene benchmark-registry binding changed")
    if bindings["oasst1_candidate"]["sha256"] != EXPECTED_INPUTS[
        OASST1_SOURCE_ID
    ]["candidate_file_sha256"]:
        raise ValueError("candidate hygiene OASST1 binding changed")
    if bindings["aya_candidate"]["sha256"] != EXPECTED_INPUTS[AYA_SOURCE_ID][
        "candidate_file_sha256"
    ]:
        raise ValueError("candidate hygiene Aya binding changed")
    return plan


def _load_tokenizer(
    *, resolved: Mapping[str, Path], template: Any
) -> Any:
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


def _measure_rows(
    rows: Sequence[Mapping[str, Any]], *, tokenizer: Any, template: Any
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    by_id: dict[str, int] = {}
    seen: set[str] = set()
    for row in rows:
        record_id = row.get("record_id")
        split = row.get("split")
        messages = row.get("messages")
        if not isinstance(record_id, str) or not record_id or record_id in seen:
            raise ValueError("candidate record IDs changed")
        if split not in {"train", "validation"} or not isinstance(messages, list):
            raise ValueError("candidate split or messages changed")
        seen.add(record_id)
        supervised_tokens = _measure_conversation(
            messages,
            tokenizer=tokenizer,
            template=template,
        )
        by_id[record_id] = supervised_tokens
        by_split[split]["records"] += 1
        by_split[split]["supervised_tokens"] += supervised_tokens
    metrics = {
        split: {
            "records": by_split[split]["records"],
            "supervised_tokens": by_split[split]["supervised_tokens"],
        }
        for split in ("train", "validation")
    }
    return metrics, by_id


def _near_duplicate_components(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any],
    supervised_tokens_by_id: Mapping[str, int],
) -> tuple[list[dict[str, Any]], str]:
    record_ids = [row.get("record_id") for row in rows]
    if any(not isinstance(item, str) or not item for item in record_ids):
        raise ValueError("Aya record IDs changed")
    string_ids = [str(item) for item in record_ids]
    if len(set(string_ids)) != len(string_ids):
        raise ValueError("Aya record IDs are not unique")
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
    predecessor_ledger: list[list[dict[str, str]]] = []
    components: list[dict[str, Any]] = []
    for group in groups:
        if len(group) != 2:
            raise ValueError("Aya near-duplicate component size changed")
        rows_in_group = [by_id[record_id] for record_id in group]
        sources = {row["source"]["source_id"] for row in rows_in_group}
        splits = {row["split"] for row in rows_in_group}
        if sources != {AYA_SOURCE_ID} or len(splits) != 1:
            raise ValueError("Aya component crossed source or split boundary")
        predecessor_ledger.append(
            [
                {
                    "record_id": record_id,
                    "source_id": AYA_SOURCE_ID,
                    "split": by_id[record_id]["split"],
                }
                for record_id in group
            ]
        )
        members = []
        for record_id in group:
            row = by_id[record_id]
            score = row.get("quality", {}).get("score")
            if not isinstance(score, (int, float)):
                raise ValueError("Aya quality score changed")
            members.append(
                {
                    "content_sha256": row["provenance"]["content_sha256"],
                    "quality_score": float(score),
                    "record_id": record_id,
                    "split": row["split"],
                    "supervised_tokens": supervised_tokens_by_id[record_id],
                }
            )
        ranked = sorted(
            members,
            key=lambda item: (
                -item["quality_score"],
                -item["supervised_tokens"],
                item["record_id"],
            ),
        )
        winner = ranked[0]
        loser = ranked[1]
        similarity = edges.get(frozenset(group))
        if similarity is None:
            raise ValueError("Aya component edge changed")
        component_binding = {
            "members": members,
            "selection_policy": EXPECTED_SELECTION_POLICY,
        }
        components.append(
            {
                "component_sha256": sha256_value(component_binding),
                "drop": loser,
                "estimated_jaccard": round(similarity, 8),
                "keep": winner,
                "members": members,
                "selection_policy": EXPECTED_SELECTION_POLICY,
                "source_id": AYA_SOURCE_ID,
                "split": winner["split"],
            }
        )
    ledger_sha256 = sha256_value(predecessor_ledger)
    if ledger_sha256 != EXPECTED_PREDECESSOR_COMPONENT_LEDGER_SHA256:
        raise ValueError("Aya component ledger no longer matches the v0.6 audit")
    if len(components) != 20:
        raise ValueError("Aya near-duplicate component count changed")
    return components, ledger_sha256


def _candidate_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _projection_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    by_split: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    candidate = _candidate_bytes(rows)
    message_bindings = [
        {
            "messages_sha256": sha256_value(row["messages"]),
            "record_id": row["record_id"],
        }
        for row in rows
    ]
    return {
        "by_split": dict(by_split),
        "candidate_file_sha256": _bytes_sha256(candidate),
        "candidate_jsonl_bytes": len(candidate),
        "message_binding_sha256": sha256_value(message_bindings),
        "record_ids_sha256": sha256_value([row["record_id"] for row in rows]),
        "records": len(rows),
    }


def rebuild_hygiene_preflight(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    resolved = {
        name: resolve_binding(root, binding, name=name)
        for name, binding in plan["bindings"].items()
    }
    predecessor = _load_json(resolved["predecessor_report"])
    predecessor_verification = _load_json(resolved["predecessor_verification"])
    validate_candidate_mix_reaudit_v6_report(predecessor)
    _verify_internal_sha(predecessor_verification, "verification_sha256")
    if predecessor.get("report_sha256") != EXPECTED_PREDECESSOR_REPORT_SHA256:
        raise ValueError("candidate hygiene predecessor report changed")
    if (
        predecessor_verification.get("verification_sha256")
        != EXPECTED_PREDECESSOR_VERIFICATION_SHA256
    ):
        raise ValueError("candidate hygiene predecessor verification changed")

    base_policy = load_sft_policy(resolved["data_contract"])
    audit_policy = replace(
        base_policy,
        benchmark_decontamination={
            "anchor_ngram_tokens": 8,
            "minimum_anchor_matches": 2,
            "registry": plan["bindings"]["benchmark_registry"]["path"],
            "registry_sha256": EXPECTED_BENCHMARK_REGISTRY_SHA256,
        },
    )
    benchmark_registry = BenchmarkRegistry(
        audit_policy.benchmark_decontamination,
        root,
    )
    registry = load_sft_source_registry(
        resolved["source_registry"],
        allowed_licenses=audit_policy.allowed_licenses,
        allowed_languages=audit_policy.allowed_languages,
        allowed_splits=audit_policy.allowed_splits,
        allowed_tasks=frozenset(audit_policy.task_mix),
        expected_sha256=EXPECTED_SOURCE_REGISTRY_SHA256,
    )
    template = load_chat_template(resolved["chat_template"])
    tokenizer = _load_tokenizer(resolved=resolved, template=template)

    oasst1_rows = _load_jsonl(resolved["oasst1_candidate"])
    aya_rows = _load_jsonl(resolved["aya_candidate"])
    if len(oasst1_rows) + len(aya_rows) > EXPECTED_LIMITS[
        "maximum_candidate_records_read"
    ]:
        raise ValueError("candidate hygiene input row ceiling exceeded")
    input_metrics: dict[str, dict[str, Any]] = {}
    token_maps: dict[str, dict[str, int]] = {}
    for source_id, rows in (
        (OASST1_SOURCE_ID, oasst1_rows),
        (AYA_SOURCE_ID, aya_rows),
    ):
        if any(row.get("source", {}).get("source_id") != source_id for row in rows):
            raise ValueError(f"candidate source ID changed: {source_id}")
        by_split, tokens_by_id = _measure_rows(
            rows,
            tokenizer=tokenizer,
            template=template,
        )
        if by_split != EXPECTED_INPUTS[source_id]["by_split"]:
            raise ValueError(f"candidate frozen-token totals changed: {source_id}")
        input_metrics[source_id] = {
            "by_split": by_split,
            "candidate_file_sha256": file_sha256(
                resolved[
                    "oasst1_candidate"
                    if source_id == OASST1_SOURCE_ID
                    else "aya_candidate"
                ]
            ),
            "records": len(rows),
        }
        token_maps[source_id] = tokens_by_id

    input_rows = [*oasst1_rows, *aya_rows]
    input_accepted, input_filter = filter_sft_records(
        input_rows,
        policy=audit_policy,
        benchmark_registry=benchmark_registry,
        source_registry=registry,
    )
    rejected = [item for item in input_filter["decisions"] if not item["accepted"]]
    if input_filter["rejection_counts"] != {
        "benchmark_contamination:bigcodebench": 3
    }:
        raise ValueError("candidate hygiene benchmark quarantine changed")
    rejected_by_id = {item["record_id"]: item for item in rejected}
    if set(rejected_by_id) != set(EXPECTED_OASST1_DROPS):
        raise ValueError("candidate hygiene OASST1 quarantine IDs changed")

    predecessor_drops = {
        item["record_id"]: item for item in predecessor["quarantine"]["records"]
    }
    oasst1_drop_ledger = []
    for record_id in sorted(EXPECTED_OASST1_DROPS):
        expected = EXPECTED_OASST1_DROPS[record_id]
        decision = rejected_by_id[record_id]
        predecessor_item = predecessor_drops.get(record_id)
        if predecessor_item is None:
            raise ValueError("candidate hygiene predecessor quarantine ID changed")
        if (
            decision.get("normalized_sha256") != expected["normalized_sha256"]
            or token_maps[OASST1_SOURCE_ID][record_id]
            != expected["supervised_tokens"]
            or predecessor_item["supervised_tokens"]
            != expected["supervised_tokens"]
        ):
            raise ValueError("candidate hygiene OASST1 quarantine binding changed")
        oasst1_drop_ledger.append(
            {
                "normalized_sha256": expected["normalized_sha256"],
                "reason": "benchmark_contamination:bigcodebench",
                "record_id": record_id,
                "source_id": OASST1_SOURCE_ID,
                "split": "train",
                "supervised_tokens": expected["supervised_tokens"],
            }
        )

    components, component_ledger_sha256 = _near_duplicate_components(
        aya_rows,
        policy=plan["near_duplicate_policy"],
        supervised_tokens_by_id=token_maps[AYA_SOURCE_ID],
    )
    aya_drop_ids = {component["drop"]["record_id"] for component in components}
    oasst1_drop_ids = set(EXPECTED_OASST1_DROPS)
    if len(aya_drop_ids) != 20:
        raise ValueError("candidate hygiene Aya drop count changed")
    oasst1_projected = [
        row for row in oasst1_rows if row["record_id"] not in oasst1_drop_ids
    ]
    aya_projected = [row for row in aya_rows if row["record_id"] not in aya_drop_ids]
    if len(input_rows) - len(oasst1_projected) - len(aya_projected) != 23:
        raise ValueError("candidate hygiene projected removal count changed")

    projected_rows = [*oasst1_projected, *aya_projected]
    projected_accepted, projected_filter = filter_sft_records(
        projected_rows,
        policy=audit_policy,
        benchmark_registry=benchmark_registry,
        source_registry=registry,
    )
    if len(projected_accepted) != len(projected_rows) or projected_filter["rejected"]:
        raise ValueError("candidate hygiene projection remains inadmissible")

    projected_summaries: dict[str, dict[str, Any]] = {}
    for source_id, rows in (
        (OASST1_SOURCE_ID, oasst1_projected),
        (AYA_SOURCE_ID, aya_projected),
    ):
        by_split, _ = _measure_rows(rows, tokenizer=tokenizer, template=template)
        if by_split != EXPECTED_PROJECTED[source_id]["by_split"]:
            raise ValueError(f"candidate hygiene projected totals changed: {source_id}")
        projected_summaries[source_id] = _projection_summary(
            rows,
            by_split=by_split,
        )

    duplicate_index: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in projected_rows:
        duplicate_index[_canonical_message_digest(row["messages"])].append(
            (row["source"]["source_id"], row["split"])
        )
    exact_snapshot = _duplicate_snapshot(duplicate_index)
    near_snapshot = _near_duplicate_audit(
        projected_rows,
        policy=plan["near_duplicate_policy"],
    )
    if exact_snapshot["duplicate_groups"] != 0:
        raise ValueError("candidate hygiene projection retains exact duplicates")
    if near_snapshot["multi_record_components"] != 0:
        raise ValueError("candidate hygiene projection retains near duplicates")

    projected_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    for source in projected_summaries.values():
        for split, values in source["by_split"].items():
            projected_by_split[split].update(values)
    report: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "benchmark_replay": {
            "input_accepted_records": len(input_accepted),
            "input_rejected_records": input_filter["rejected"],
            "input_rejection_counts": input_filter["rejection_counts"],
            "projected_accepted_dataset_sha256": projected_filter[
                "accepted_dataset_sha256"
            ],
            "projected_accepted_records": len(projected_accepted),
            "projected_rejected_records": projected_filter["rejected"],
            "registry_sha256": EXPECTED_BENCHMARK_REGISTRY_SHA256,
        },
        "candidate_files_materialized": False,
        "input_candidates": input_metrics,
        "near_duplicate_projection": {
            "component_ledger_sha256": component_ledger_sha256,
            "components": components,
            "components_count": len(components),
            "drop_record_ids_sha256": sha256_value(sorted(aya_drop_ids)),
            "selection_ledger_sha256": sha256_value(components),
            "selection_policy": dict(plan["canonical_selection_policy"]),
        },
        "oasst1_benchmark_quarantine": {
            "drop_ledger_sha256": sha256_value(oasst1_drop_ledger),
            "records": oasst1_drop_ledger,
            "total_records": len(oasst1_drop_ledger),
        },
        "observed_at": plan["observed_at"],
        "plan_file_sha256": plan_file_sha256,
        "post_projection_duplicate_audit": {
            "exact": exact_snapshot,
            "near": near_snapshot,
        },
        "production_dataset_promotions": 0,
        "production_eligible": False,
        "projected_candidates": projected_summaries,
        "projected_pool": {
            "by_split": {
                split: {
                    "records": projected_by_split[split]["records"],
                    "supervised_tokens": projected_by_split[split][
                        "supervised_tokens"
                    ],
                }
                for split in ("train", "validation")
            },
            "records": len(projected_rows),
            "removed_records": len(input_rows) - len(projected_rows),
        },
        "raw_training_text_retained": False,
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "verified_non_mutating_hygiene_projection_ready_for_materialization_gate",
    }
    report["report_sha256"] = sha256_value(report)
    validate_hygiene_preflight_report(report)
    return report


def validate_hygiene_preflight_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported candidate hygiene preflight report schema")
    _verify_internal_sha(report, "report_sha256")
    if report.get("candidate_files_materialized") is not False:
        raise ValueError("candidate hygiene preflight cannot materialize candidates")
    if report.get("production_eligible") is not False:
        raise ValueError("candidate hygiene preflight cannot be production eligible")
    if report.get("production_dataset_promotions") != 0:
        raise ValueError("candidate hygiene preflight cannot promote a dataset")
    if report.get("raw_training_text_retained") is not False:
        raise ValueError("candidate hygiene preflight must remain raw-text-free")
    if report.get("oasst1_benchmark_quarantine", {}).get("total_records") != 3:
        raise ValueError("candidate hygiene OASST1 drop count changed")
    near = report.get("near_duplicate_projection", {})
    if near.get("components_count") != 20:
        raise ValueError("candidate hygiene Aya component count changed")
    projected = report.get("projected_pool", {})
    if projected.get("records") != 3_200 or projected.get("removed_records") != 23:
        raise ValueError("candidate hygiene projected pool count changed")
    if projected.get("by_split") != {
        "train": {"records": 2_962, "supervised_tokens": 966_069},
        "validation": {"records": 238, "supervised_tokens": 79_056},
    }:
        raise ValueError("candidate hygiene projected pool tokens changed")
    benchmark = report.get("benchmark_replay", {})
    if benchmark.get("input_rejected_records") != 3:
        raise ValueError("candidate hygiene input quarantine count changed")
    if benchmark.get("projected_rejected_records") != 0:
        raise ValueError("candidate hygiene projection retains benchmark matches")
    duplicate = report.get("post_projection_duplicate_audit", {})
    if duplicate.get("exact", {}).get("duplicate_groups") != 0:
        raise ValueError("candidate hygiene projection retains exact duplicates")
    if duplicate.get("near", {}).get("multi_record_components") != 0:
        raise ValueError("candidate hygiene projection retains near duplicates")
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    for forbidden in ('"messages"', '"content"', '"assistant_response"'):
        if forbidden in serialized:
            raise ValueError("candidate hygiene report contains raw training text")


def build_hygiene_preflight_verification(
    *,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    report_file_sha256: str,
    replayed_report: Mapping[str, Any],
) -> dict[str, Any]:
    validate_hygiene_preflight_report(report)
    validate_hygiene_preflight_report(replayed_report)
    if report != replayed_report:
        raise ValueError("candidate hygiene independent replay changed")
    verification: dict[str, Any] = {
        "checks": {
            "aya_20_component_keep_drop_ledger_replayed": True,
            "benchmark_registry_v0_3_replayed_before_and_after_projection": True,
            "candidate_files_not_materialized": True,
            "frozen_token_counts_recomputed": True,
            "no_network_api_gpu_seal_sft_or_production_promotion": True,
            "oasst1_3_bigcodebench_rows_bound_for_removal": True,
            "post_projection_exact_and_near_duplicate_counts_zero": True,
            "projected_candidate_hashes_reproduced": True,
            "raw_training_text_absent_from_repo_evidence": True,
        },
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "projected_candidate_hashes": {
            source_id: values["candidate_file_sha256"]
            for source_id, values in sorted(report["projected_candidates"].items())
        },
        "replayed_report_sha256": replayed_report["report_sha256"],
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_non_mutating_candidate_hygiene_preflight",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification
