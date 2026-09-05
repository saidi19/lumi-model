"""Registry-v0.11-bound full local re-audit of the Lumi SFT candidate pool."""

from __future__ import annotations

import copy
import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import posttraining_candidate_mix_reaudit as v2
from . import posttraining_candidate_mix_reaudit_v3 as v3
from . import posttraining_candidate_mix_reaudit_v4 as v4
from . import posttraining_candidate_mix_reaudit_v5 as v5
from .posttraining_ai_v3_provenance_materialization import _verify_internal_sha
from .posttraining_candidate_mix_preflight import (
    EXPECTED_LANGUAGE_MODES,
    EXPECTED_TASKS,
    _canonical_message_digest,
    _duplicate_snapshot,
    _language_mode,
    _load_json,
    _load_jsonl,
    _measure_conversation,
    _require_sha256,
    _targets,
    file_sha256,
    sha256_value,
)
from .posttraining_data import filter_sft_records, load_sft_policy
from .posttraining_format import load_chat_template, verify_chat_tokenizer
from .posttraining_opencodeinstruct_bigcodebench_reaudit import (
    validate_report as validate_bigcodebench_reaudit,
)
from .posttraining_opencodeinstruct_livecodebench_reaudit import (
    validate_report as validate_livecodebench_reaudit,
)
from .posttraining_opencodeinstruct_owner_review import (
    validate_owner_review_receipt,
)
from .posttraining_opencodeinstruct_source_promotion import (
    validate_promotion_receipt as validate_opencodeinstruct_promotion,
)
from .posttraining_sources import load_sft_source_registry

PLAN_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-plan.v0.6"
REPORT_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-report.v0.6"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-verification.v0.6"

EXPECTED_AUTHORIZATION = {
    "candidate_registry_change_authorized": False,
    "dataset_acquisition_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_api_spend_authorized": False,
    "external_distribution_authorized": False,
    "local_candidate_read_authorized": True,
    "local_raw_text_free_reaudit_authorized": True,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "purpose": "registry_v0_11_full_local_candidate_mix_reaudit_only",
    "source_promotion_authorized": False,
    "unrelated_access_authorized": False,
}
EXPECTED_LIMITS = {
    "maximum_candidate_records_read": 300_000,
    "maximum_network_bytes": 0,
    "maximum_runtime_minutes": 180,
}
EXPECTED_OUTPUT = {
    "production_promotion_permitted": False,
    "raw_training_text_in_outputs": False,
    "report_path": "evidence/lumi-4b-sft-candidate-mix-reaudit-v0.6.json",
    "verification_path": (
        "evidence/lumi-4b-sft-candidate-mix-reaudit-verification-v0.6.json"
    ),
}
EXPECTED_BINDINGS = frozenset(
    {
        "benchmark_registry_v0_3",
        "implementation",
        "opencode_bigcodebench_reaudit",
        "opencode_bigcodebench_reaudit_verification",
        "opencode_candidate",
        "opencode_livecodebench_reaudit",
        "opencode_livecodebench_reaudit_verification",
        "opencode_owner_review",
        "opencode_promotion",
        "opencode_promotion_verification",
        "opencode_token_metrics",
        "predecessor_plan",
        "predecessor_report",
        "predecessor_verification",
        "runner",
        "source_registry",
        "test",
        "verifier",
    }
)
EXPECTED_REGISTRY_SHA256 = (
    "5353c57189d82ee22306ccd11d116d017c5e683811cec2d6704cf45bcf7f02e7"
)
EXPECTED_BENCHMARK_REGISTRY_SHA256 = (
    "1ffb5e938ab28c2fc876b9efd6c6ed59059f1e85aed2c396c024ae5925a4745e"
)
EXPECTED_OPENCODE_CANDIDATE_SHA256 = (
    "9be50e3b6078fdc50d3b887e7850ffc4e57f9a634f72fb928b62908da854446e"
)
EXPECTED_OPENCODE_PROMOTION_INTERNAL_SHA256 = (
    "62e0d9127012a28cf522812cf90dc5662a63e3a854832bd8e593b394636d9b1b"
)
EXPECTED_OPENCODE_PROMOTION_VERIFICATION_INTERNAL_SHA256 = (
    "8f29216b98f61d46f944ed404cded0da7221e4e81574172493fcfb01b7a7af9a"
)
EXPECTED_OPENCODE_OWNER_REVIEW_INTERNAL_SHA256 = (
    "46093a7728812f30524c29ff5005b32c3234f3c755858312b12afb726f25c065"
)
EXPECTED_PREDECESSOR_REPORT_INTERNAL_SHA256 = (
    "b1ac1c25f58480ee03fb00ea6a50d12d8971302a90029ed2a93124597d0a2e1f"
)
EXPECTED_PREDECESSOR_VERIFICATION_INTERNAL_SHA256 = (
    "144db4a28128d33f5e1449afdbc4c5a9a22e53406fe9c7dd63b6956e4eef5f9d"
)
EXPECTED_RECORDS = 258_604
EXPECTED_ACCEPTED_RECORDS = 258_601
EXPECTED_BENCHMARK_QUARANTINE_RECORDS = 3
EXPECTED_TRAIN_TOKENS = 5_371_870
EXPECTED_VALIDATION_TOKENS = 320_403
EXPECTED_CODING_TOKENS = 2_154_516
EXPECTED_ENGLISH_TOKENS = 4_828_676
OPENCODE_SOURCE_ID = "nvidia_opencodeinstruct"


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _resolve_explicit_bindings(
    plan: Mapping[str, Any], repository_root: str | Path
) -> dict[str, Path]:
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("candidate-mix v0.6 binding set changed")
    return {
        name: v2.resolve_binding(repository_root, binding, name=name)
        for name, binding in bindings.items()
    }


def _load_predecessor_plan(
    plan: Mapping[str, Any], *, repository_root: str | Path
) -> dict[str, Any]:
    path = v2.resolve_binding(
        repository_root,
        plan["bindings"]["predecessor_plan"],
        name="predecessor_plan",
    )
    return v5.load_candidate_mix_reaudit_v5_plan(
        path,
        repository_root=repository_root,
    )


def load_candidate_mix_reaudit_v6_plan(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    plan = _load_json(Path(path))
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "benchmark_decontamination": {
            "anchor_ngram_tokens": 8,
            "minimum_anchor_matches": 2,
            "registry": "data/benchmark-signatures-v0.3.jsonl.gz",
            "registry_sha256": EXPECTED_BENCHMARK_REGISTRY_SHA256,
        },
        "limits": EXPECTED_LIMITS,
        "near_duplicate_policy": v2.EXPECTED_NEAR_DUPLICATE_POLICY,
        "observed_at": "2026-08-07",
        "output": EXPECTED_OUTPUT,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "authorized_registry_v0_11_full_local_reaudit_no_production_action",
    }
    for field, expected_value in expected.items():
        if plan.get(field) != expected_value:
            raise ValueError(f"candidate-mix v0.6 plan changed: {field}")
    resolved = _resolve_explicit_bindings(plan, repository_root)
    if plan["bindings"]["source_registry"]["sha256"] != EXPECTED_REGISTRY_SHA256:
        raise ValueError("candidate-mix v0.6 registry binding changed")
    if (
        plan["bindings"]["benchmark_registry_v0_3"]["sha256"]
        != EXPECTED_BENCHMARK_REGISTRY_SHA256
    ):
        raise ValueError("candidate-mix v0.6 benchmark registry binding changed")
    if (
        plan["bindings"]["opencode_candidate"]["sha256"]
        != EXPECTED_OPENCODE_CANDIDATE_SHA256
    ):
        raise ValueError("candidate-mix v0.6 OpenCode candidate changed")

    _load_predecessor_plan(plan, repository_root=repository_root)
    predecessor_report = _load_json(resolved["predecessor_report"])
    predecessor_verification = _load_json(resolved["predecessor_verification"])
    _verify_internal_sha(predecessor_report, "report_sha256")
    _verify_internal_sha(predecessor_verification, "verification_sha256")
    if (
        predecessor_report.get("report_sha256")
        != EXPECTED_PREDECESSOR_REPORT_INTERNAL_SHA256
        or predecessor_report.get("status")
        != "verified_registry_v0_10_candidate_mix_not_ready_for_seal_or_gpu_sft"
    ):
        raise ValueError("candidate-mix v0.6 predecessor report changed")
    if (
        predecessor_verification.get("verification_sha256")
        != EXPECTED_PREDECESSOR_VERIFICATION_INTERNAL_SHA256
        or predecessor_verification.get("status")
        != "verified_registry_v0_10_candidate_mix_reaudit_nonproduction"
    ):
        raise ValueError("candidate-mix v0.6 predecessor verification changed")
    _validate_opencode_evidence(resolved)
    return plan


def _effective_plan(
    plan: Mapping[str, Any], *, repository_root: str | Path
) -> dict[str, Any]:
    predecessor = _load_predecessor_plan(plan, repository_root=repository_root)
    effective = copy.deepcopy(v5._effective_plan(predecessor, repository_root=repository_root))
    effective["authorization"] = dict(plan["authorization"])
    effective["limits"] = dict(plan["limits"])
    effective["near_duplicate_policy"] = dict(plan["near_duplicate_policy"])
    effective["observed_at"] = plan["observed_at"]
    effective["output"] = dict(plan["output"])
    effective["bindings"].update(plan["bindings"])
    effective["bindings"]["source_registry"] = dict(
        plan["bindings"]["source_registry"]
    )
    effective["bindings"]["benchmark_registry"] = dict(
        plan["bindings"]["benchmark_registry_v0_3"]
    )
    if any(
        source["source_id"] == OPENCODE_SOURCE_ID
        for source in effective["candidate_sources"]
    ):
        raise ValueError("candidate-mix v0.6 OpenCode source already exists")
    effective["candidate_sources"].append(copy.deepcopy(plan["source_addition"]))
    return effective


def _validate_opencode_evidence(resolved: Mapping[str, Path]) -> None:
    promotion = _load_json(resolved["opencode_promotion"])
    promotion_verification = _load_json(resolved["opencode_promotion_verification"])
    owner_review = _load_json(resolved["opencode_owner_review"])
    token_metrics = _load_json(resolved["opencode_token_metrics"])
    validate_opencodeinstruct_promotion(promotion)
    validate_owner_review_receipt(owner_review)
    _verify_internal_sha(promotion_verification, "verification_sha256")
    if promotion.get("receipt_sha256") != EXPECTED_OPENCODE_PROMOTION_INTERNAL_SHA256:
        raise ValueError("OpenCodeInstruct promotion receipt changed")
    if promotion_verification.get("verification_sha256") != (
        EXPECTED_OPENCODE_PROMOTION_VERIFICATION_INTERNAL_SHA256
    ):
        raise ValueError("OpenCodeInstruct promotion verification changed")
    if owner_review.get("receipt_sha256") != (
        EXPECTED_OPENCODE_OWNER_REVIEW_INTERNAL_SHA256
    ):
        raise ValueError("OpenCodeInstruct owner-review receipt changed")
    if token_metrics.get("accepted_records") != 11_100:
        raise ValueError("OpenCodeInstruct token-metrics record count changed")
    if token_metrics.get("by_split") != {
        "train": {
            "masked_tokens": 1_752_360,
            "records": 10_553,
            "serialized_tokens": 3_852_437,
            "supervised_tokens": 2_100_077,
        },
        "validation": {
            "masked_tokens": 90_624,
            "records": 547,
            "serialized_tokens": 201_243,
            "supervised_tokens": 110_619,
        },
    }:
        raise ValueError("OpenCodeInstruct token-metrics split totals changed")

    bigcode_report = _load_json(resolved["opencode_bigcodebench_reaudit"])
    bigcode_verification = _load_json(
        resolved["opencode_bigcodebench_reaudit_verification"]
    )
    livecode_report = _load_json(resolved["opencode_livecodebench_reaudit"])
    livecode_verification = _load_json(
        resolved["opencode_livecodebench_reaudit_verification"]
    )
    validate_bigcodebench_reaudit(bigcode_report)
    validate_livecodebench_reaudit(livecode_report)
    _verify_internal_sha(bigcode_verification, "verification_sha256")
    _verify_internal_sha(livecode_verification, "verification_sha256")
    for report in (bigcode_report, livecode_report):
        if report.get("results", {}).get("matched_records") != 0:
            raise ValueError("OpenCodeInstruct benchmark re-audit gained matches")


def _verify_candidate_promotion(
    *, source_id: str, value: Mapping[str, Any]
) -> None:
    if source_id == OPENCODE_SOURCE_ID:
        validate_opencodeinstruct_promotion(value)
        if value.get("receipt_sha256") != EXPECTED_OPENCODE_PROMOTION_INTERNAL_SHA256:
            raise ValueError("OpenCodeInstruct promotion receipt changed")
        return
    v5._verify_candidate_promotion(source_id=source_id, value=value)


def rebuild_candidate_mix_reaudit_v6(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    root = Path(repository_root).resolve()
    effective = _effective_plan(plan, repository_root=root)
    resolved = {
        name: v2.resolve_binding(root, binding, name=name)
        for name, binding in effective["bindings"].items()
    }
    resolved.update(_resolve_explicit_bindings(plan, root))
    predecessor_v5 = _load_predecessor_plan(plan, repository_root=root)
    predecessor_v4 = v5._load_predecessor_plan(
        predecessor_v5,
        repository_root=root,
    )
    resolved.update(v4._resolve_explicit_bindings(predecessor_v4, root))
    resolved["source_registry"] = v2.resolve_binding(
        root,
        plan["bindings"]["source_registry"],
        name="source_registry",
    )
    v3._validate_bound_evidence(resolved)
    v4._validate_new_evidence(resolved)
    v5._validate_deepmind_evidence(resolved)
    _validate_opencode_evidence(resolved)

    base_policy = load_sft_policy(resolved["data_contract"])
    audit_policy = replace(
        base_policy,
        benchmark_decontamination=dict(plan["benchmark_decontamination"]),
    )
    benchmark_registry = v2.BenchmarkRegistry(
        audit_policy.benchmark_decontamination,
        root,
    )
    if benchmark_registry.sha256 != EXPECTED_BENCHMARK_REGISTRY_SHA256:
        raise ValueError("candidate-mix v0.6 benchmark registry changed")
    registry = load_sft_source_registry(
        resolved["source_registry"],
        allowed_licenses=audit_policy.allowed_licenses,
        allowed_languages=audit_policy.allowed_languages,
        allowed_splits=audit_policy.allowed_splits,
        allowed_tasks=frozenset(audit_policy.task_mix),
        expected_sha256=EXPECTED_REGISTRY_SHA256,
    )
    targets = _targets(_load_json(resolved["missing_mix_proposal"]))
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
    for source_plan in sorted(
        effective["candidate_sources"], key=lambda item: item["source_id"]
    ):
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
        metrics = v3._candidate_metrics_v3(
            rows=rows,
            source_plan=source_plan,
            registry_source=registry_source,
            tokenizer=tokenizer,
            template=template,
            duplicate_index=duplicate_index,
        )
        if metrics["rows_requiring_split_normalization"] != 0:
            raise ValueError(f"{source_id} still requires split normalization")
        if metrics["rows_with_stale_pending_human_review_provenance"] != (
            source_plan["expected_pending_owner_review_rows"]
        ):
            raise ValueError(f"{source_id} owner-review provenance count changed")
        source_metrics[source_id] = metrics
        all_rows.extend(rows)
        train_tokens += metrics["by_split"]["train"]["supervised_tokens"]
        validation_tokens += metrics["by_split"]["validation"]["supervised_tokens"]
        task_tokens.update(metrics["train_by_task"])
        language_tokens.update(metrics["train_by_language_mode"])
        promotion = _load_json(resolved[source_plan["promotion_binding"]])
        _verify_candidate_promotion(source_id=source_id, value=promotion)

    if len(all_rows) > effective["limits"]["maximum_candidate_records_read"]:
        raise ValueError("candidate-mix v0.6 row ceiling exceeded")
    accepted, generic = filter_sft_records(
        all_rows,
        policy=audit_policy,
        benchmark_registry=benchmark_registry,
        source_registry=registry,
    )
    if (
        len(accepted) != EXPECTED_ACCEPTED_RECORDS
        or generic["rejected"] != EXPECTED_BENCHMARK_QUARANTINE_RECORDS
        or generic["rejection_counts"]
        != {"benchmark_contamination:bigcodebench": 3}
    ):
        raise ValueError(
            "registry-v0.11 candidate quarantine set changed: "
            f"{generic['rejection_counts']}"
        )

    if len(all_rows) != EXPECTED_RECORDS:
        raise ValueError("registry-v0.11 exact candidate record count changed")
    if train_tokens != EXPECTED_TRAIN_TOKENS:
        raise ValueError("registry-v0.11 train token count changed")
    if validation_tokens != EXPECTED_VALIDATION_TOKENS:
        raise ValueError("registry-v0.11 validation token count changed")

    source_plans = {
        item["source_id"]: item for item in effective["candidate_sources"]
    }
    rows_by_id = {row["record_id"]: row for row in all_rows}
    rejected_decisions = [
        decision for decision in generic["decisions"] if not decision["accepted"]
    ]
    accepted_train_tokens = train_tokens
    accepted_validation_tokens = validation_tokens
    accepted_task_tokens = Counter(task_tokens)
    accepted_language_tokens = Counter(language_tokens)
    accepted_source_train_tokens = {
        source_id: metrics["by_split"]["train"]["supervised_tokens"]
        for source_id, metrics in source_metrics.items()
    }
    quarantined_by_split: dict[str, dict[str, int]] = {
        "train": {"records": 0, "supervised_tokens": 0},
        "validation": {"records": 0, "supervised_tokens": 0},
    }
    quarantined_rows: list[dict[str, Any]] = []
    per_source_quarantine: dict[str, Counter[str]] = defaultdict(Counter)
    for decision in rejected_decisions:
        record_id = decision["record_id"]
        if not isinstance(record_id, str) or record_id not in rows_by_id:
            raise ValueError("candidate-mix v0.6 quarantine identity changed")
        row = rows_by_id[record_id]
        source_id = row["source"]["source_id"]
        split = row["split"]
        supervised_tokens = _measure_conversation(
            row["messages"],
            tokenizer=tokenizer,
            template=template,
        )
        quarantined_by_split[split]["records"] += 1
        quarantined_by_split[split]["supervised_tokens"] += supervised_tokens
        per_source_quarantine[source_id]["records"] += 1
        per_source_quarantine[source_id][f"{split}_records"] += 1
        per_source_quarantine[source_id][f"{split}_tokens"] += supervised_tokens
        task_category = row["task_category"]
        language_mode: str | None = None
        if split == "train":
            accepted_train_tokens -= supervised_tokens
            accepted_source_train_tokens[source_id] -= supervised_tokens
            accepted_task_tokens[task_category] -= supervised_tokens
            language_policy = source_plans[source_id]["language_mode_policy"]
            if (
                language_policy == "language_field_multilingual_as_banglish"
                and row.get("language") == "multilingual"
            ):
                language_mode = "banglish"
            else:
                language_mode = _language_mode(row, "language_field")
            accepted_language_tokens[language_mode] -= supervised_tokens
        else:
            accepted_validation_tokens -= supervised_tokens
        quarantined_rows.append(
            {
                "normalized_sha256": decision["normalized_sha256"],
                "reasons": decision["reasons"],
                "record_id": record_id,
                "source_id": source_id,
                "split": split,
                "supervised_tokens": supervised_tokens,
                "task_category": task_category,
                "train_language_mode": language_mode,
            }
        )
    if len(quarantined_rows) != EXPECTED_BENCHMARK_QUARANTINE_RECORDS:
        raise ValueError("candidate-mix v0.6 quarantine count changed")
    for source_id, metrics in source_metrics.items():
        quarantined = per_source_quarantine[source_id]
        metrics["generic_sft_accepted_records"] = (
            metrics["records"] - quarantined["records"]
        )
        metrics["generic_sft_rejected_records"] = quarantined["records"]
        metrics["quarantined_by_split"] = {
            split: {
                "records": quarantined[f"{split}_records"],
                "supervised_tokens": quarantined[f"{split}_tokens"],
            }
            for split in ("train", "validation")
        }

    accepted_duplicate_index: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in accepted:
        accepted_duplicate_index[_canonical_message_digest(row["messages"])].append(
            (row["source"]["source_id"], row["split"])
        )
    exact_snapshot = _duplicate_snapshot(accepted_duplicate_index)
    near_snapshot = v2._near_duplicate_audit(
        accepted,
        policy=effective["near_duplicate_policy"],
    )
    source_shares = {
        source_id: round(
            accepted_source_train_tokens[source_id] / accepted_train_tokens,
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
        task: max(0, targets["task_minimums"][task] - accepted_task_tokens[task])
        for task in EXPECTED_TASKS
    }
    language_gaps = {
        mode: max(
            0,
            targets["language_minimums"][mode] - accepted_language_tokens[mode],
        )
        for mode in EXPECTED_LANGUAGE_MODES
    }
    blockers: list[str] = []
    if accepted_train_tokens < targets["total"]["minimum"]:
        blockers.append("accepted_train_supervised_tokens_below_10m_floor")
    if any(task_gaps.values()):
        blockers.append("task_token_floors_not_met")
    if any(language_gaps.values()):
        blockers.append("language_token_floors_not_met")
    if source_cap_violations:
        blockers.append("source_concentration_cap_not_met")
    blockers.extend(
        [
            "benchmark_contaminated_rows_require_quarantine_materialization",
            "active_production_data_contract_not_updated",
            "dataset_seal_cpu_canary_and_sft_not_authorized",
        ]
    )
    if exact_snapshot["duplicate_groups"]:
        blockers.append("exact_duplicate_canonical_selection_not_materialized")
    if near_snapshot["multi_record_components"]:
        blockers.append("near_duplicate_canonical_selection_not_materialized")
    if near_snapshot["cross_source_components"]:
        blockers.append("cross_source_near_duplicates_not_resolved")
    if near_snapshot["train_validation_components"]:
        blockers.append("train_validation_near_leakage_not_resolved")
    excluded = v2._verify_exclusions(plan=effective, resolved=resolved, registry=registry)
    priority_gaps = [
        task for task in EXPECTED_TASKS if task_gaps[task] > 0
    ] + [mode for mode in EXPECTED_LANGUAGE_MODES if language_gaps[mode] > 0]

    report: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "benchmark_decontamination": {
            "benchmarks": sorted(benchmark_registry.benchmarks),
            "registry_entries": 50_944,
            "registry_sha256": benchmark_registry.sha256,
            "raw_benchmark_text_retained": False,
            "unique_exact_signatures": len(benchmark_registry.exact),
        },
        "candidate_pool": {
            "accepted_records_after_benchmark_quarantine": len(accepted),
            "accepted_train_supervised_tokens": accepted_train_tokens,
            "candidate_records_read": len(all_rows),
            "floor_completion_fraction": round(
                accepted_train_tokens / targets["total"]["minimum"], 8
            ),
            "gap_to_10m_train_supervised_tokens": (
                targets["total"]["minimum"] - accepted_train_tokens
            ),
            "registry_admissible_validation_supervised_tokens": (
                accepted_validation_tokens
            ),
            "source_count": len(source_metrics),
            "source_shares": source_shares,
            "train_by_language_mode": {
                mode: accepted_language_tokens[mode]
                for mode in EXPECTED_LANGUAGE_MODES
            },
            "train_by_task": {
                task: accepted_task_tokens[task] for task in EXPECTED_TASKS
            },
        },
        "closed_since_v0_5": [
            "opencodeinstruct_candidate_owner_sample_accepted_32_of_32",
            "opencodeinstruct_candidate_registered_in_immutable_registry_v0_11",
            "bigcodebench_and_livecodebench_zero_match_reaudits_bound",
            "coding_train_token_floor_met",
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
            "accepted": generic["accepted"],
            "accepted_dataset_sha256": generic["accepted_dataset_sha256"],
            "benchmark_contamination_rejections": sum(
                count
                for reason, count in generic["rejection_counts"].items()
                if reason.startswith("benchmark_contamination:")
            ),
            "raw_text_retained": generic["raw_text_retained"],
            "rejected": generic["rejected"],
            "rejection_counts": generic["rejection_counts"],
            "report_sha256": generic["report_sha256"],
            "source_registry_rejections": sum(
                count
                for reason, count in generic["rejection_counts"].items()
                if reason.startswith("source_registry")
            ),
            "total": generic["total"],
        },
        "near_duplicate_audit": near_snapshot,
        "next_gate": {
            "action": (
                "close_remaining_diversified_conversation_factual_writing_"
                "safety_tool_and_bengali_mix_gaps_before_dataset_seal"
            ),
            "authorization_required": True,
            "priority_gaps": priority_gaps,
        },
        "observed_at": plan["observed_at"],
        "plan_file_sha256": _require_sha256(plan_file_sha256, "plan_file_sha256"),
        "production_dataset_promotions": 0,
        "production_eligible": False,
        "quarantine": {
            "by_split": quarantined_by_split,
            "materialized_in_candidate_files": False,
            "reason_counts": generic["rejection_counts"],
            "records": sorted(quarantined_rows, key=lambda item: item["record_id"]),
            "total_records": len(quarantined_rows),
        },
        "raw_training_text_retained": False,
        "registry": {
            "path": plan["bindings"]["source_registry"]["path"],
            "sha256": EXPECTED_REGISTRY_SHA256,
            "source_count": len(registry.by_id),
        },
        "schema_version": REPORT_SCHEMA_VERSION,
        "source_metrics": source_metrics,
        "status": "verified_registry_v0_11_candidate_mix_not_ready_for_seal_or_gpu_sft",
    }
    report["report_sha256"] = sha256_value(report)
    validate_candidate_mix_reaudit_v6_report(report)
    return report


def validate_candidate_mix_reaudit_v6_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported candidate-mix v0.6 report schema")
    _verify_internal_sha(report, "report_sha256")
    if report.get("production_eligible") is not False:
        raise ValueError("candidate-mix v0.6 cannot be production eligible")
    if report.get("production_dataset_promotions") != 0:
        raise ValueError("candidate-mix v0.6 cannot promote a production dataset")
    if report.get("raw_training_text_retained") is not False:
        raise ValueError("candidate-mix v0.6 report must remain raw-text-free")
    pool = report.get("candidate_pool")
    if not isinstance(pool, Mapping):
        raise ValueError("candidate-mix v0.6 pool summary is missing")
    quarantine = report.get("quarantine")
    if not isinstance(quarantine, Mapping):
        raise ValueError("candidate-mix v0.6 quarantine summary is missing")
    if (
        quarantine.get("total_records") != EXPECTED_BENCHMARK_QUARANTINE_RECORDS
        or quarantine.get("reason_counts")
        != {"benchmark_contamination:bigcodebench": 3}
        or quarantine.get("materialized_in_candidate_files") is not False
    ):
        raise ValueError("candidate-mix v0.6 quarantine changed")
    quarantined_by_split = quarantine.get("by_split")
    if not isinstance(quarantined_by_split, Mapping):
        raise ValueError("candidate-mix v0.6 quarantine splits are missing")
    expected_accepted_train = EXPECTED_TRAIN_TOKENS - quarantined_by_split[
        "train"
    ]["supervised_tokens"]
    expected_accepted_validation = EXPECTED_VALIDATION_TOKENS - quarantined_by_split[
        "validation"
    ]["supervised_tokens"]
    expected_pool = {
        "accepted_records_after_benchmark_quarantine": EXPECTED_ACCEPTED_RECORDS,
        "accepted_train_supervised_tokens": expected_accepted_train,
        "candidate_records_read": EXPECTED_RECORDS,
        "gap_to_10m_train_supervised_tokens": 10_000_000 - expected_accepted_train,
        "registry_admissible_validation_supervised_tokens": (
            expected_accepted_validation
        ),
        "source_count": 7,
    }
    for field, expected_value in expected_pool.items():
        if pool.get(field) != expected_value:
            raise ValueError(f"candidate-mix v0.6 pool changed: {field}")
    quarantined_train_coding = sum(
        item["supervised_tokens"]
        for item in quarantine["records"]
        if item["split"] == "train"
        and item["task_category"] == "coding"
    )
    if pool.get("train_by_task", {}).get("coding") != (
        EXPECTED_CODING_TOKENS - quarantined_train_coding
    ):
        raise ValueError("candidate-mix v0.6 coding total changed")
    quarantined_train_english = sum(
        item["supervised_tokens"]
        for item in quarantine["records"]
        if item["split"] == "train" and item["train_language_mode"] == "english"
    )
    if pool.get("train_by_language_mode", {}).get("english") != (
        EXPECTED_ENGLISH_TOKENS - quarantined_train_english
    ):
        raise ValueError("candidate-mix v0.6 English total changed")
    generic = report.get("generic_sft_validation")
    if not isinstance(generic, Mapping):
        raise ValueError("candidate-mix v0.6 generic validation is missing")
    if (
        generic.get("accepted") != EXPECTED_ACCEPTED_RECORDS
        or generic.get("rejected") != EXPECTED_BENCHMARK_QUARANTINE_RECORDS
    ):
        raise ValueError("candidate-mix v0.6 generic validation changed")
    if generic.get("rejection_counts") != {
        "benchmark_contamination:bigcodebench": 3
    }:
        raise ValueError("candidate-mix v0.6 generic validation has rejection reasons")
    opencode = report.get("source_metrics", {}).get(OPENCODE_SOURCE_ID)
    if not isinstance(opencode, Mapping):
        raise ValueError("candidate-mix v0.6 OpenCode metrics are missing")
    if opencode.get("by_split") != {
        "train": {"records": 10_553, "supervised_tokens": 2_100_077},
        "validation": {"records": 547, "supervised_tokens": 110_619},
    }:
        raise ValueError("candidate-mix v0.6 OpenCode metrics changed")
    if report.get("registry", {}).get("sha256") != EXPECTED_REGISTRY_SHA256:
        raise ValueError("candidate-mix v0.6 registry evidence changed")
    benchmark = report.get("benchmark_decontamination")
    if not isinstance(benchmark, Mapping) or benchmark.get("registry_sha256") != (
        EXPECTED_BENCHMARK_REGISTRY_SHA256
    ):
        raise ValueError("candidate-mix v0.6 benchmark evidence changed")
    if benchmark.get("registry_entries") != 50_944:
        raise ValueError("candidate-mix v0.6 benchmark entry count changed")
    near = report.get("near_duplicate_audit")
    if not isinstance(near, Mapping) or near.get("records") != (
        EXPECTED_ACCEPTED_RECORDS
    ):
        raise ValueError("candidate-mix v0.6 near-audit row count changed")


def build_candidate_mix_reaudit_v6_verification(
    *, plan: Mapping[str, Any], report: Mapping[str, Any], report_file_sha256: str
) -> dict[str, Any]:
    validate_candidate_mix_reaudit_v6_report(report)
    verification: dict[str, Any] = {
        "checks": {
            "all_258604_rows_audited_and_3_bigcodebench_matches_quarantined": True,
            "benchmark_registry_v0_3_bigcodebench_and_livecodebench_verified": True,
            "exact_and_near_duplicate_audits_completed": True,
            "frozen_token_task_language_and_source_totals_reproduced": True,
            "no_network_api_gpu_seal_sft_or_production_promotion": True,
            "opencodeinstruct_11100_rows_are_registry_admissible": True,
            "raw_training_text_absent_from_repo_evidence": True,
            "registry_v0_11_hash_verified": True,
        },
        "duplicate_summary": {
            "cross_source_exact_groups": report["exact_duplicate_audit"][
                "cross_source_duplicate_groups"
            ],
            "cross_source_near_components": report["near_duplicate_audit"][
                "cross_source_components"
            ],
            "train_validation_exact_groups": report["exact_duplicate_audit"][
                "train_validation_overlap_groups"
            ],
            "train_validation_near_components": report["near_duplicate_audit"][
                "train_validation_components"
            ],
        },
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "report_file_sha256": _require_sha256(
            report_file_sha256, "report_file_sha256"
        ),
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_registry_v0_11_candidate_mix_reaudit_nonproduction",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification


__all__ = [
    "EXPECTED_BENCHMARK_REGISTRY_SHA256",
    "EXPECTED_RECORDS",
    "EXPECTED_REGISTRY_SHA256",
    "EXPECTED_TRAIN_TOKENS",
    "EXPECTED_VALIDATION_TOKENS",
    "build_candidate_mix_reaudit_v6_verification",
    "json_file_bytes",
    "load_candidate_mix_reaudit_v6_plan",
    "rebuild_candidate_mix_reaudit_v6",
    "validate_candidate_mix_reaudit_v6_report",
]
