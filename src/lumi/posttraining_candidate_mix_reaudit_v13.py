"""Registry-v0.19-bound full local candidate-mix re-audit."""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import posttraining_candidate_mix_reaudit as v2
from . import posttraining_candidate_mix_reaudit_v12 as v12
from . import posttraining_existing_candidate_cross_source_reaudit_v18 as v18
from .posttraining_candidate_mix_preflight import (
    EXPECTED_LANGUAGE_MODES,
    EXPECTED_TASKS,
    _canonical_message_digest,
    _duplicate_snapshot,
    _load_json,
    _load_jsonl,
    _targets,
    file_sha256,
    sha256_value,
)
from .posttraining_data import filter_sft_records, load_sft_policy
from .posttraining_format import encode_sft_messages, load_chat_template, verify_chat_tokenizer
from .posttraining_gpt_oss_teacher_source_promotion_preflight_v16 import (
    PureByteLevelBPETokenizer,
)
from .posttraining_sources import load_sft_source_registry

PLAN_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-plan.v0.13"
REPORT_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-report.v0.13"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-verification.v0.13"

EXPECTED_AUTHORIZATION = {
    "candidate_mutation_authorized": False,
    "dataset_seal_authorized": False,
    "download_authorized": False,
    "external_api_authorized": False,
    "full_local_reaudit_authorized": True,
    "gpu_authorized": False,
    "network_authorized": False,
    "production_promotion_authorized": False,
    "registry_mutation_authorized": False,
    "remote_access_authorized": False,
    "s3_action_authorized": False,
    "sft_training_authorized": False,
    "source_promotion_authorized": False,
    "unrelated_access_authorized": False,
}
EXPECTED_LIMITS = {
    "maximum_candidate_records_per_pass": 309_838,
    "maximum_full_audit_passes": 2,
    "maximum_network_bytes": 0,
    "maximum_private_write_bytes": 2_097_152,
    "maximum_runtime_minutes": 180,
    "maximum_total_candidate_row_reads": 619_676,
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
EXPECTED_BINDINGS = frozenset(
    {
        "benchmark_registry",
        "chat_template",
        "data_contract",
        "implementation",
        "predecessor_plan",
        "predecessor_report",
        "predecessor_verification",
        "prosocial_candidate",
        "prosocial_promotion_receipt",
        "prosocial_promotion_verification",
        "runner",
        "source_registry",
        "test",
        "tokenizer_json",
        "verifier",
    }
)

EXPECTED_INPUT_RECORDS = 309_838
EXPECTED_INPUT_TRAIN_TOKENS = 9_553_473
EXPECTED_INPUT_VALIDATION_TOKENS = 578_618
EXPECTED_INCLUDED_SOURCE_COUNT = 16
EXPECTED_REGISTRY_SOURCE_COUNT = 23
EXPECTED_REGISTRY_SHA256 = (
    "d88ef71863c38c277260172dd6b3b5ce5e2330726370329bc438b7750f43ec9c"
)
EXPECTED_PROSOCIAL_CANDIDATE_SHA256 = (
    "9f9f8dfdbd3bc667c0329936a9ef312f3a05b9547cfe061f5d3e66d751d02829"
)
EXPECTED_PROSOCIAL_PROMOTION_RECEIPT_SHA256 = (
    "1f8776cdd1a24fb69b354a8184acf2a6fa5305e282064146241ece4fbd4f6332"
)
EXPECTED_PROSOCIAL_PROMOTION_VERIFICATION_SHA256 = (
    "958ce9fa54016bf8eb54c44766d1be45a27ee6835970e6c2abaaef5d6871e5bc"
)
EXPECTED_PREDECESSOR_REPORT_SHA256 = (
    "0b170105caf5713e58363f39270a6efe2bbca8b0c4641badf4d3c79b5e767429"
)
EXPECTED_PREDECESSOR_VERIFICATION_SHA256 = (
    "35f4aceb519b4681e611ed104b30ee53dfed6de4f23a02eaa4b730d340b3f43b"
)
EXPECTED_BENCHMARK_REGISTRY_SHA256 = (
    "1ffb5e938ab28c2fc876b9efd6c6ed59059f1e85aed2c396c024ae5925a4745e"
)
EXPECTED_TOKENIZER_SHA256 = (
    "1022c9538d9ec795b07c2e2bda3c5c6f21c321de933cfe5ad197628a94caa2ee"
)
EXPECTED_CHAT_TEMPLATE_SHA256 = (
    "bc7853a9277feddae2fd3db786c6a12f4fa3602a135a0408c38d4390f33bec12"
)
PROSOCIAL_SOURCE = {
    "binding": "prosocial_candidate",
    "candidate_sha256": EXPECTED_PROSOCIAL_CANDIDATE_SHA256,
    "family": "allenai_prosocial_family",
    "language_policy": "language_field",
    "revision": "d77d7ad3c624c51030f2f32c83e892b3d620b3d4",
    "source_id": "allenai_prosocial_dialog_safety_candidate",
}


def json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _resolve_binding(root: Path, binding: Mapping[str, Any], name: str) -> Path:
    path = (root / str(binding.get("path"))).resolve()
    if not path.is_relative_to(root.parent.resolve()) or not path.is_file():
        raise ValueError(f"invalid or missing v0.13 binding: {name}")
    if file_sha256(path) != binding.get("sha256"):
        raise ValueError(f"v0.13 binding hash mismatch: {name}")
    return path


def load_plan(path: str | Path, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = _load_json(Path(path).resolve())
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "expected": {
            "included_candidate_sources": EXPECTED_INCLUDED_SOURCE_COUNT,
            "input_records": EXPECTED_INPUT_RECORDS,
            "input_train_supervised_tokens": EXPECTED_INPUT_TRAIN_TOKENS,
            "input_validation_supervised_tokens": EXPECTED_INPUT_VALIDATION_TOKENS,
            "registry_sources": EXPECTED_REGISTRY_SOURCE_COUNT,
        },
        "limits": EXPECTED_LIMITS,
        "near_duplicate_policy": EXPECTED_NEAR_DUPLICATE_POLICY,
        "observed_at": "2026-08-13",
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "authorized_registry_v0_19_full_local_reaudit_no_production_action",
    }
    for field, value in expected.items():
        if plan.get(field) != value:
            raise ValueError(f"candidate-mix v0.13 plan changed: {field}")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("candidate-mix v0.13 binding inventory changed")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"invalid v0.13 binding: {name}")
        _resolve_binding(root, binding, name)
    exact = {
        "benchmark_registry": EXPECTED_BENCHMARK_REGISTRY_SHA256,
        "predecessor_report": EXPECTED_PREDECESSOR_REPORT_SHA256,
        "predecessor_verification": EXPECTED_PREDECESSOR_VERIFICATION_SHA256,
        "prosocial_candidate": EXPECTED_PROSOCIAL_CANDIDATE_SHA256,
        "prosocial_promotion_receipt": EXPECTED_PROSOCIAL_PROMOTION_RECEIPT_SHA256,
        "prosocial_promotion_verification": EXPECTED_PROSOCIAL_PROMOTION_VERIFICATION_SHA256,
        "source_registry": EXPECTED_REGISTRY_SHA256,
    }
    for name, expected_sha256 in exact.items():
        if bindings[name]["sha256"] != expected_sha256:
            raise ValueError(f"candidate-mix v0.13 exact binding changed: {name}")
    return plan


def _validate_predecessor_and_promotion(
    plan: Mapping[str, Any], bound: Mapping[str, Path]
) -> None:
    predecessor = _load_json(bound["predecessor_report"])
    predecessor_verification = _load_json(bound["predecessor_verification"])
    promotion = _load_json(bound["prosocial_promotion_receipt"])
    promotion_verification = _load_json(bound["prosocial_promotion_verification"])
    if (
        predecessor.get("status")
        != "PASS_registry_v0_18_full_cross_source_reaudit_nonproduction"
        or predecessor.get("report_sha256")
        != "1ba45418deb32ef3930f038ba9e49c6a1edd89261aae5f58071febae525397e7"
        or predecessor.get("candidate_pool", {}).get("candidate_records_read") != 280_574
        or predecessor.get("candidate_pool", {}).get("train_supervised_tokens")
        != 8_653_475
        or predecessor_verification.get("status")
        != "PASS_verified_registry_v0_18_full_cross_source_reaudit_nonproduction"
        or predecessor_verification.get("verification_sha256")
        != "aef58d40d608399c6974d5bb06e33f250dd7e6b1edc7e0f9ce025ea0257398b3"
    ):
        raise ValueError("candidate-mix v0.13 predecessor evidence changed")
    if (
        promotion.get("status")
        != "verified_prosocial_dialog_candidate_and_registry_v0_19_nonproduction"
        or promotion.get("candidate", {}).get("file_sha256")
        != EXPECTED_PROSOCIAL_CANDIDATE_SHA256
        or promotion.get("registry", {}).get("successor_sha256")
        != EXPECTED_REGISTRY_SHA256
        or promotion.get("checks", {}).get("full_candidate_mix_reaudit_run") is not False
        or promotion_verification.get("status")
        != "PASS_verified_prosocial_dialog_candidate_and_registry_v0_19_nonproduction"
        or promotion_verification.get("successor_registry", {}).get("source_count") != 23
        or not all(promotion_verification.get("checks", {}).values())
    ):
        raise ValueError("candidate-mix v0.13 Prosocial promotion evidence changed")


def _source_inventory(
    *,
    root: Path,
    predecessor_plan: Mapping[str, Any],
    bound: Mapping[str, Path],
) -> tuple[dict[str, dict[str, Any]], dict[str, Path], Mapping[str, Any], Mapping[str, Path]]:
    v18_plan = v18.load_plan(bound["predecessor_plan"], root)
    v12_plan_path = v18._resolve_binding(
        root, v18_plan["bindings"]["predecessor_audit_plan"], "predecessor_audit_plan"
    )
    v12_plan = v12.load_candidate_mix_reaudit_v12_plan(v12_plan_path, repository_root=root)
    effective = v12._effective_plan(v12_plan, repository_root=root)
    old_resolved = {
        name: v2.resolve_binding(root, binding, name=name)
        for name, binding in effective["bindings"].items()
    }
    source_plans: dict[str, dict[str, Any]] = {}
    candidate_paths: dict[str, Path] = {}
    for source_plan in effective["candidate_sources"]:
        source_id = str(source_plan["source_id"])
        source_plans[source_id] = {
            "family": v18._old_family(source_id),
            "language_policy": source_plan["language_mode_policy"],
            "revision": source_plan["revision"],
            "source_id": source_id,
        }
        candidate_paths[source_id] = old_resolved[source_plan["candidate_binding"]]
        if file_sha256(candidate_paths[source_id]) != source_plan["candidate_file_sha256"]:
            raise ValueError(f"candidate-mix v0.13 old candidate changed: {source_id}")
    for addition in v18.NEW_SOURCES:
        source_id = addition["source_id"]
        source_plans[source_id] = {
            "family": addition["family"],
            "language_policy": addition["language_policy"],
            "revision": addition["revision"],
            "source_id": source_id,
        }
        candidate_paths[source_id] = v18._resolve_binding(
            root, v18_plan["bindings"][addition["binding"]], addition["binding"]
        )
        if file_sha256(candidate_paths[source_id]) != addition["candidate_sha256"]:
            raise ValueError(f"candidate-mix v0.13 v0.18 candidate changed: {source_id}")
    source_id = PROSOCIAL_SOURCE["source_id"]
    source_plans[source_id] = {
        "family": PROSOCIAL_SOURCE["family"],
        "language_policy": PROSOCIAL_SOURCE["language_policy"],
        "revision": PROSOCIAL_SOURCE["revision"],
        "source_id": source_id,
    }
    candidate_paths[source_id] = bound["prosocial_candidate"]
    if file_sha256(candidate_paths[source_id]) != EXPECTED_PROSOCIAL_CANDIDATE_SHA256:
        raise ValueError("candidate-mix v0.13 Prosocial candidate changed")
    if len(source_plans) != EXPECTED_INCLUDED_SOURCE_COUNT:
        raise ValueError("candidate-mix v0.13 source inventory changed")
    return source_plans, candidate_paths, effective, old_resolved


def rebuild_report(
    *, plan: Mapping[str, Any], plan_file_sha256: str, repository_root: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    bound = {
        name: _resolve_binding(root, binding, name)
        for name, binding in plan["bindings"].items()
    }
    _validate_predecessor_and_promotion(plan, bound)
    source_plans, candidate_paths, effective, old_resolved = _source_inventory(
        root=root, predecessor_plan=plan, bound=bound
    )

    base_policy = load_sft_policy(bound["data_contract"])
    audit_policy = replace(
        base_policy,
        benchmark_decontamination={
            "anchor_ngram_tokens": 8,
            "minimum_anchor_matches": 2,
            "registry": "data/benchmark-signatures-v0.3.jsonl.gz",
            "registry_sha256": EXPECTED_BENCHMARK_REGISTRY_SHA256,
        },
    )
    benchmark_registry = v2.BenchmarkRegistry(audit_policy.benchmark_decontamination, root)
    if benchmark_registry.sha256 != EXPECTED_BENCHMARK_REGISTRY_SHA256:
        raise ValueError("candidate-mix v0.13 benchmark registry changed")
    registry = load_sft_source_registry(
        bound["source_registry"],
        allowed_licenses=audit_policy.allowed_licenses,
        allowed_languages=audit_policy.allowed_languages,
        allowed_splits=audit_policy.allowed_splits,
        allowed_tasks=frozenset(audit_policy.task_mix),
        expected_sha256=EXPECTED_REGISTRY_SHA256,
    )
    if len(registry.by_id) != EXPECTED_REGISTRY_SOURCE_COUNT:
        raise ValueError("candidate-mix v0.13 registry inventory changed")
    template = load_chat_template(bound["chat_template"])
    if file_sha256(bound["chat_template"]) != EXPECTED_CHAT_TEMPLATE_SHA256:
        raise ValueError("candidate-mix v0.13 chat template changed")
    if file_sha256(bound["tokenizer_json"]) != EXPECTED_TOKENIZER_SHA256:
        raise ValueError("candidate-mix v0.13 tokenizer changed")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer = PureByteLevelBPETokenizer(bound["tokenizer_json"])
    verify_chat_tokenizer(tokenizer, template)

    all_rows: list[dict[str, Any]] = []
    source_metrics: dict[str, dict[str, Any]] = {}
    for source_id in sorted(source_plans):
        source_plan = source_plans[source_id]
        registry_source = registry.by_id.get(source_id)
        if (
            registry_source is None
            or registry_source.revision != source_plan["revision"]
            or registry_source.status not in {"canary_eligible", "candidate_eligible"}
        ):
            raise ValueError(f"candidate-mix v0.13 registry source mismatch: {source_id}")
        rows = _load_jsonl(candidate_paths[source_id])
        metrics = v18._source_metrics(
            rows=rows,
            source_id=source_id,
            family=str(source_plan["family"]),
            language_policy=str(source_plan["language_policy"]),
            candidate_sha256=file_sha256(candidate_paths[source_id]),
            tokenizer=tokenizer,
            template=template,
        )
        source_metrics[source_id] = metrics
        all_rows.extend(rows)

    input_train, input_validation, *_ = v18._aggregate_metrics(source_metrics)
    if (
        len(all_rows) != EXPECTED_INPUT_RECORDS
        or input_train != EXPECTED_INPUT_TRAIN_TOKENS
        or input_validation != EXPECTED_INPUT_VALIDATION_TOKENS
    ):
        raise ValueError("candidate-mix v0.13 input totals changed")

    accepted, generic = filter_sft_records(
        all_rows,
        policy=audit_policy,
        benchmark_registry=benchmark_registry,
        source_registry=registry,
    )
    accepted_metrics = v18._accepted_metrics(
        rows=accepted,
        source_plans=source_plans,
        tokenizer=tokenizer,
        template=template,
    )
    rows_by_id = {str(row["record_id"]): row for row in all_rows}
    quarantine: list[dict[str, Any]] = []
    for decision in generic["decisions"]:
        if decision.get("accepted") is True:
            continue
        row = rows_by_id[str(decision["record_id"])]
        encoded = encode_sft_messages(row["messages"], tokenizer=tokenizer, template=template)
        quarantine.append(
            {
                "reasons": decision["reasons"],
                "record_id": row["record_id"],
                "source_id": row["source"]["source_id"],
                "split": row["split"],
                "supervised_tokens": encoded.supervised_tokens,
            }
        )

    duplicate_index: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in accepted:
        duplicate_index[_canonical_message_digest(row["messages"])].append(
            (row["source"]["source_id"], row["split"])
        )
    exact = _duplicate_snapshot(duplicate_index)
    near = v2._near_duplicate_audit(accepted, policy=plan["near_duplicate_policy"])
    targets = _targets(_load_json(old_resolved["missing_mix_proposal"]))

    train_tokens = int(accepted_metrics["train_supervised_tokens"])
    task_behavior = Counter(accepted_metrics["task_train_supervised_tokens_behavior_safe"])
    language_behavior = Counter(
        accepted_metrics["language_train_supervised_tokens_behavior_safe"]
    )
    task_gaps = {
        task: max(0, int(targets["task_minimums"][task]) - task_behavior[task])
        for task in EXPECTED_TASKS
    }
    language_gaps = {
        mode: max(0, int(targets["language_minimums"][mode]) - language_behavior[mode])
        for mode in EXPECTED_LANGUAGE_MODES
    }
    family_tokens = accepted_metrics["family_train_supervised_tokens"]
    family_shares = {
        family: round(int(tokens) / train_tokens, 8)
        for family, tokens in sorted(family_tokens.items())
    }
    family_cap_violations = [
        family
        for family, share in family_shares.items()
        if share > float(targets["source_cap"])
    ]
    family_dilution_gap = max(
        (
            max(0, math.ceil(int(tokens) / float(targets["source_cap"]) - train_tokens))
            for tokens in family_tokens.values()
        ),
        default=0,
    )
    numeric_gap = max(0, int(targets["total"]["minimum"]) - train_tokens)
    strict_gap = max(
        numeric_gap,
        sum(task_gaps.values()),
        sum(language_gaps.values()),
        family_dilution_gap,
    )
    hygiene_needed = bool(
        generic["rejected"] or exact["duplicate_groups"] or near["multi_record_components"]
    )
    blockers: list[str] = []
    if generic["rejected"]:
        blockers.append("generic_filter_quarantine_not_materialized")
    if exact["duplicate_groups"]:
        blockers.append("cross_source_exact_duplicates_not_materialized")
    if near["multi_record_components"]:
        blockers.append("cross_source_near_duplicates_not_materialized")
    if numeric_gap:
        blockers.append("accepted_train_supervised_tokens_below_10m_floor")
    if any(task_gaps.values()):
        blockers.append("task_token_floors_not_met")
    if any(language_gaps.values()):
        blockers.append("language_token_floors_not_met")
    if family_cap_violations:
        blockers.append("source_family_concentration_cap_not_met")
    blockers.extend(
        [
            "active_production_data_contract_not_updated",
            "dataset_seal_cpu_canary_and_sft_not_authorized",
        ]
    )

    generic_summary = {key: value for key, value in generic.items() if key != "decisions"}
    report: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "candidate_pool": {
            "accepted_records_after_generic_quarantine": len(accepted),
            "accepted_train_supervised_tokens": train_tokens,
            "accepted_validation_supervised_tokens": accepted_metrics[
                "validation_supervised_tokens"
            ],
            "candidate_records_read": len(all_rows),
            "input_train_supervised_tokens": input_train,
            "input_validation_supervised_tokens": input_validation,
            "numeric_gap_to_10m": numeric_gap,
            "strict_targeted_gap_lower_bound": strict_gap,
            **accepted_metrics,
        },
        "closure": {
            "blockers": blockers,
            "family_cap_dilution_gap": family_dilution_gap,
            "language_floor_gaps_behavior_safe": language_gaps,
            "production_dataset_ready": False,
            "source_family_cap": targets["source_cap"],
            "source_family_cap_violations": family_cap_violations,
            "source_family_shares": family_shares,
            "task_floor_gaps_behavior_safe": task_gaps,
        },
        "exact_duplicate_audit": exact,
        "generic_sft_validation": generic_summary,
        "input_projection": {
            "predecessor_records": 280_574,
            "predecessor_train_supervised_tokens": 8_653_475,
            "predecessor_validation_supervised_tokens": 518_619,
            "prosocial_records": 29_264,
            "prosocial_train_supervised_tokens": 899_998,
            "prosocial_validation_supervised_tokens": 59_999,
        },
        "near_duplicate_audit": near,
        "next_gate": {
            "action": (
                "materialize_registry_v0_19_hygiene_successor"
                if hygiene_needed
                else "close_residual_behavior_safe_task_language_and_10m_gaps"
            ),
            "authorization_required": True,
            "priority": (
                "hygiene_before_gap_closure" if hygiene_needed else "targeted_gap_closure"
            ),
        },
        "observed_at": plan["observed_at"],
        "plan_file_sha256": plan_file_sha256,
        "production_eligible": False,
        "quarantine": {
            "materialized": False,
            "reason_counts": generic["rejection_counts"],
            "records": quarantine,
            "total_records": len(quarantine),
        },
        "raw_training_text_retained": False,
        "registry": {
            "file_sha256": EXPECTED_REGISTRY_SHA256,
            "source_count": len(registry.by_id),
        },
        "schema_version": REPORT_SCHEMA_VERSION,
        "source_metrics": source_metrics,
        "status": "PASS_registry_v0_19_full_candidate_mix_reaudit_nonproduction",
        "token_consistency": {
            "chat_template_sha256": EXPECTED_CHAT_TEMPLATE_SHA256,
            "predecessor_report_file_sha256": EXPECTED_PREDECESSOR_REPORT_SHA256,
            "pure_byte_level_tokenizer_used": True,
            "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        },
    }
    report["report_sha256"] = sha256_value(report)
    validate_report(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported candidate-mix v0.13 report")
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256", None)
    if internal != sha256_value(unsigned):
        raise ValueError("candidate-mix v0.13 report internal hash mismatch")
    pool = report.get("candidate_pool", {})
    if (
        report.get("status") != "PASS_registry_v0_19_full_candidate_mix_reaudit_nonproduction"
        or report.get("production_eligible") is not False
        or report.get("raw_training_text_retained") is not False
        or pool.get("candidate_records_read") != EXPECTED_INPUT_RECORDS
        or pool.get("input_train_supervised_tokens") != EXPECTED_INPUT_TRAIN_TOKENS
        or pool.get("input_validation_supervised_tokens")
        != EXPECTED_INPUT_VALIDATION_TOKENS
        or report.get("registry", {}).get("source_count") != EXPECTED_REGISTRY_SOURCE_COUNT
        or len(report.get("source_metrics", {})) != EXPECTED_INCLUDED_SOURCE_COUNT
    ):
        raise ValueError("candidate-mix v0.13 report boundary or totals changed")
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    for forbidden in ('"messages"', '"content"', '"assistant_response"', '"user_prompt"'):
        if forbidden in serialized:
            raise ValueError("candidate-mix v0.13 report retained raw training text")


def build_verification(
    *, plan: Mapping[str, Any], report: Mapping[str, Any], report_file_sha256: str
) -> dict[str, Any]:
    validate_report(report)
    verification: dict[str, Any] = {
        "checks": {
            "all_309838_rows_replayed_with_pure_byte_level_tokenizer": True,
            "exact_and_near_duplicate_results_reproduced": True,
            "generic_schema_rights_provenance_secret_pii_and_benchmark_gates_replayed": True,
            "no_network_gpu_seal_sft_registry_or_production_action": True,
            "predecessor_v0_18_audit_and_prosocial_promotion_bound": True,
            "raw_training_text_absent_from_outputs": True,
            "source_family_task_and_language_ledgers_reproduced": True,
        },
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "PASS_verified_registry_v0_19_full_candidate_mix_reaudit_nonproduction",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification


__all__ = [
    "build_verification",
    "json_bytes",
    "load_plan",
    "rebuild_report",
    "validate_report",
]
