"""Registry-v0.20-bound full local candidate-mix re-audit."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import posttraining_candidate_mix_reaudit as v2
from . import posttraining_candidate_mix_reaudit_v13 as v13
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

PLAN_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-plan.v0.14"
REPORT_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-report.v0.14"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-verification.v0.14"

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
    "maximum_candidate_records_per_pass": 314_271,
    "maximum_full_audit_passes": 2,
    "maximum_network_bytes": 0,
    "maximum_private_write_bytes": 2_097_152,
    "maximum_runtime_minutes": 180,
    "maximum_total_candidate_row_reads": 628_542,
}
EXPECTED_NEAR_DUPLICATE_POLICY = dict(v13.EXPECTED_NEAR_DUPLICATE_POLICY)
EXPECTED_BINDINGS = frozenset(
    {
        "benchmark_registry",
        "billsum_candidate",
        "billsum_promotion_receipt",
        "billsum_promotion_verification",
        "chat_template",
        "data_contract",
        "draft",
        "implementation",
        "predecessor_plan",
        "predecessor_report",
        "predecessor_verification",
        "runner",
        "source_registry",
        "test",
        "tokenizer_json",
        "verifier",
    }
)

EXPECTED_INPUT_RECORDS = 314_271
EXPECTED_INPUT_TRAIN_TOKENS = 10_403_464
EXPECTED_INPUT_VALIDATION_TOKENS = 628_604
EXPECTED_INCLUDED_SOURCE_COUNT = 17
EXPECTED_REGISTRY_SOURCE_COUNT = 24
EXPECTED_REGISTRY_SHA256 = (
    "ce5de72f3cd308aaf99b35c1316236a316959f87beed6bd9876c2e28be0a2e7b"
)
EXPECTED_PREDECESSOR_PLAN_SHA256 = (
    "56321dc603c58ca7280f97bcc2be87d3593e9124276f7ee26fed1167e2147ca8"
)
EXPECTED_PREDECESSOR_REPORT_SHA256 = (
    "9f77596c80c555c3a44c7a4993db3862f2a5503ed834c57a22d4b8d2f23f8361"
)
EXPECTED_PREDECESSOR_VERIFICATION_SHA256 = (
    "bbacce4844c201c70e3b3b48bc3871c65c7ad4162d7ece973e6e27dc4c7cf0d4"
)
EXPECTED_DRAFT_SHA256 = (
    "92b81f6d25b8c73f4126b84875a66cab9f390d982db6753431df9c30b80a8e0f"
)
EXPECTED_BILLSUM_CANDIDATE_SHA256 = (
    "a2641db45f5ebd8e2cbca7b3f3cf4f0cbfb29c99e6953e16001d4358ecb63327"
)
EXPECTED_BILLSUM_PROMOTION_RECEIPT_SHA256 = (
    "e1a054e219f84a4d9d1e19d756ec6bca4a99a12f8a46e7f3159b23da28610dd7"
)
EXPECTED_BILLSUM_PROMOTION_VERIFICATION_SHA256 = (
    "9e0a65bfa4dcd0cefef249aa489edbfc766b1ed719727b83c01f1fdaf1c96770"
)
EXPECTED_BENCHMARK_REGISTRY_SHA256 = v13.EXPECTED_BENCHMARK_REGISTRY_SHA256
EXPECTED_TOKENIZER_SHA256 = v13.EXPECTED_TOKENIZER_SHA256
EXPECTED_CHAT_TEMPLATE_SHA256 = v13.EXPECTED_CHAT_TEMPLATE_SHA256
EXPECTED_EVALUATION_PROHIBITION = {
    "billsum_ca_test_claimed_evaluation": "permanently_prohibited",
    "billsum_test_claimed_evaluation": "permanently_prohibited",
    "reason": "BillSum train split used for SFT candidate construction",
}
BILLSUM_SOURCE = {
    "binding": "billsum_candidate",
    "candidate_sha256": EXPECTED_BILLSUM_CANDIDATE_SHA256,
    "family": "fiscalnote_billsum_family",
    "language_policy": "language_field",
    "revision": "3d8510441c06a3d9dfb32eb0d7f80151730bcc4f",
    "source_id": "fiscalnote_billsum_writing_candidate",
}
EXPECTED_EXCLUDED_SOURCE_IDS = frozenset(
    {
        "bigcode_self_oss_instruct_50k",
        "huggingfaceh4_ultrachat_200k",
        "lumi_ai_assisted_sft_v0_1",
        "lumi_ai_assisted_sft_v0_2",
        "lumi_owned_sft_v0_1",
        "openr1_math_220k",
        "salesforce_xlam_function_calling_60k",
    }
)


def json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _resolve_binding(root: Path, binding: Mapping[str, Any], name: str) -> Path:
    path = (root / str(binding.get("path"))).resolve()
    if not path.is_relative_to(root.parent.resolve()) or not path.is_file():
        raise ValueError(f"invalid or missing v0.14 binding: {name}")
    if file_sha256(path) != binding.get("sha256"):
        raise ValueError(f"v0.14 binding hash mismatch: {name}")
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
        "output": {
            "production_promotion_permitted": False,
            "raw_training_text_in_outputs": False,
            "report_path": "evidence/lumi-4b-sft-candidate-mix-reaudit-v0.14.json",
            "verification_path": (
                "evidence/lumi-4b-sft-candidate-mix-reaudit-verification-v0.14.json"
            ),
        },
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "authorized_registry_v0_20_full_local_reaudit_no_production_action",
    }
    for field, value in expected.items():
        if plan.get(field) != value:
            raise ValueError(f"candidate-mix v0.14 plan changed: {field}")
    text = plan.get("authorization_text")
    if not isinstance(text, str) or hashlib.sha256(text.encode()).hexdigest() != plan.get(
        "authorization_text_sha256"
    ):
        raise ValueError("candidate-mix v0.14 authorization text binding changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("candidate-mix v0.14 binding inventory changed")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"invalid v0.14 binding: {name}")
        _resolve_binding(root, binding, name)
    exact = {
        "benchmark_registry": EXPECTED_BENCHMARK_REGISTRY_SHA256,
        "billsum_candidate": EXPECTED_BILLSUM_CANDIDATE_SHA256,
        "billsum_promotion_receipt": EXPECTED_BILLSUM_PROMOTION_RECEIPT_SHA256,
        "billsum_promotion_verification": (
            EXPECTED_BILLSUM_PROMOTION_VERIFICATION_SHA256
        ),
        "draft": EXPECTED_DRAFT_SHA256,
        "predecessor_plan": EXPECTED_PREDECESSOR_PLAN_SHA256,
        "predecessor_report": EXPECTED_PREDECESSOR_REPORT_SHA256,
        "predecessor_verification": EXPECTED_PREDECESSOR_VERIFICATION_SHA256,
        "source_registry": EXPECTED_REGISTRY_SHA256,
    }
    for name, expected_sha256 in exact.items():
        if bindings[name]["sha256"] != expected_sha256:
            raise ValueError(f"candidate-mix v0.14 exact binding changed: {name}")
    return plan


def _validate_predecessor_and_promotion(bound: Mapping[str, Path]) -> None:
    predecessor = _load_json(bound["predecessor_report"])
    predecessor_verification = _load_json(bound["predecessor_verification"])
    promotion = _load_json(bound["billsum_promotion_receipt"])
    promotion_verification = _load_json(bound["billsum_promotion_verification"])
    if (
        predecessor.get("status")
        != "PASS_registry_v0_19_full_candidate_mix_reaudit_nonproduction"
        or predecessor.get("report_sha256")
        != "ca25db3c494f0f0e29020bb08c75f831a0c2f4802b6a9061bde70bf62497d79e"
        or predecessor.get("candidate_pool", {}).get("candidate_records_read")
        != 309_838
        or predecessor.get("candidate_pool", {}).get("input_train_supervised_tokens")
        != 9_553_473
        or predecessor.get("candidate_pool", {}).get(
            "input_validation_supervised_tokens"
        )
        != 578_618
        or predecessor_verification.get("status")
        != "PASS_verified_registry_v0_19_full_candidate_mix_reaudit_nonproduction"
        or predecessor_verification.get("verification_sha256")
        != "cd6dbb092f1dc8b0a88cea040ea120e3780fa949c80a8f99e77f8f5962f41799"
    ):
        raise ValueError("candidate-mix v0.14 predecessor evidence changed")
    if (
        promotion.get("status")
        != "verified_billsum_candidate_and_registry_v0_20_nonproduction"
        or promotion.get("candidate", {}).get("file_sha256")
        != EXPECTED_BILLSUM_CANDIDATE_SHA256
        or promotion.get("registry", {}).get("successor_sha256")
        != EXPECTED_REGISTRY_SHA256
        or promotion.get("checks", {}).get("full_candidate_mix_reaudit_run") is not False
        or promotion.get("evaluation_prohibition") != EXPECTED_EVALUATION_PROHIBITION
        or promotion_verification.get("status")
        != "PASS_verified_billsum_candidate_and_registry_v0_20_nonproduction"
        or promotion_verification.get("successor_registry", {}).get("source_count")
        != 24
        or promotion_verification.get("evaluation_prohibition")
        != EXPECTED_EVALUATION_PROHIBITION
        or not all(promotion_verification.get("checks", {}).values())
    ):
        raise ValueError("candidate-mix v0.14 BillSum promotion evidence changed")


def _source_inventory(
    *, root: Path, bound: Mapping[str, Path]
) -> tuple[dict[str, dict[str, Any]], dict[str, Path], Mapping[str, Any], Mapping[str, Path]]:
    predecessor_plan = v13.load_plan(bound["predecessor_plan"], root)
    predecessor_bound = {
        name: v13._resolve_binding(root, binding, name)
        for name, binding in predecessor_plan["bindings"].items()
    }
    v13._validate_predecessor_and_promotion(predecessor_plan, predecessor_bound)
    source_plans, candidate_paths, effective, old_resolved = v13._source_inventory(
        root=root, predecessor_plan=predecessor_plan, bound=predecessor_bound
    )
    source_id = BILLSUM_SOURCE["source_id"]
    if source_id in source_plans:
        raise ValueError("candidate-mix v0.14 BillSum source already exists")
    source_plans[source_id] = {
        "family": BILLSUM_SOURCE["family"],
        "language_policy": BILLSUM_SOURCE["language_policy"],
        "revision": BILLSUM_SOURCE["revision"],
        "source_id": source_id,
    }
    candidate_paths[source_id] = bound["billsum_candidate"]
    if file_sha256(candidate_paths[source_id]) != EXPECTED_BILLSUM_CANDIDATE_SHA256:
        raise ValueError("candidate-mix v0.14 BillSum candidate changed")
    if len(source_plans) != EXPECTED_INCLUDED_SOURCE_COUNT:
        raise ValueError("candidate-mix v0.14 source inventory changed")
    return source_plans, candidate_paths, effective, old_resolved


def _validate_billsum_rows(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 4_433:
        raise ValueError("BillSum candidate row count changed")
    for row in rows:
        source = row.get("source")
        provenance = row.get("provenance")
        if (
            not isinstance(source, Mapping)
            or source.get("source_id") != BILLSUM_SOURCE["source_id"]
            or source.get("dataset_id") != "FiscalNote/billsum"
            or source.get("revision") != BILLSUM_SOURCE["revision"]
            or source.get("license_id") != "cc0-1.0"
            or "data/train-00000-of-00001.parquet#row-" not in str(source.get("uri"))
            or str(source.get("document_id", "")).startswith("billsum-train:") is False
        ):
            raise ValueError("BillSum train-only source binding changed")
        uri = str(source.get("uri", "")).casefold()
        if "/test" in uri or "ca_test" in uri:
            raise ValueError("BillSum test or ca_test entered candidate inventory")
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("assistant_response_origin")
            != "source_provided_legislative_summary_authorship_not_established"
            or provenance.get("source_pair_integrity")
            != "exact_untruncated_source_text_and_summary"
        ):
            raise ValueError("BillSum provenance or source-pair integrity changed")


def rebuild_report(
    *, plan: Mapping[str, Any], plan_file_sha256: str, repository_root: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    bound = {
        name: _resolve_binding(root, binding, name)
        for name, binding in plan["bindings"].items()
    }
    _validate_predecessor_and_promotion(bound)
    source_plans, candidate_paths, effective, old_resolved = _source_inventory(
        root=root, bound=bound
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
        raise ValueError("candidate-mix v0.14 benchmark registry changed")
    registry = load_sft_source_registry(
        bound["source_registry"],
        allowed_licenses=audit_policy.allowed_licenses,
        allowed_languages=audit_policy.allowed_languages,
        allowed_splits=audit_policy.allowed_splits,
        allowed_tasks=frozenset(audit_policy.task_mix),
        expected_sha256=EXPECTED_REGISTRY_SHA256,
    )
    if len(registry.by_id) != EXPECTED_REGISTRY_SOURCE_COUNT:
        raise ValueError("candidate-mix v0.14 registry inventory changed")
    excluded_source_ids = frozenset(registry.by_id) - frozenset(source_plans)
    if excluded_source_ids != EXPECTED_EXCLUDED_SOURCE_IDS:
        raise ValueError("candidate-mix v0.14 excluded-source inventory changed")
    excluded = v2._verify_exclusions(
        plan=effective,
        resolved=old_resolved,
        registry=registry,
    )
    template = load_chat_template(bound["chat_template"])
    if file_sha256(bound["chat_template"]) != EXPECTED_CHAT_TEMPLATE_SHA256:
        raise ValueError("candidate-mix v0.14 chat template changed")
    if file_sha256(bound["tokenizer_json"]) != EXPECTED_TOKENIZER_SHA256:
        raise ValueError("candidate-mix v0.14 tokenizer changed")
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
            raise ValueError(f"candidate-mix v0.14 registry source mismatch: {source_id}")
        rows = _load_jsonl(candidate_paths[source_id])
        if source_id == BILLSUM_SOURCE["source_id"]:
            _validate_billsum_rows(rows)
        metrics = v13.v18._source_metrics(
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

    input_train, input_validation, *_ = v13.v18._aggregate_metrics(source_metrics)
    if (
        len(all_rows) != EXPECTED_INPUT_RECORDS
        or input_train != EXPECTED_INPUT_TRAIN_TOKENS
        or input_validation != EXPECTED_INPUT_VALIDATION_TOKENS
    ):
        raise ValueError("candidate-mix v0.14 input totals changed")

    accepted, generic = filter_sft_records(
        all_rows,
        policy=audit_policy,
        benchmark_registry=benchmark_registry,
        source_registry=registry,
    )
    accepted_metrics = v13.v18._accepted_metrics(
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
        "evaluation_prohibition": EXPECTED_EVALUATION_PROHIBITION,
        "exact_duplicate_audit": exact,
        "excluded_source_gate": {
            "candidate_rows_from_excluded_sources": 0,
            "predecessor_evidence_replayed": excluded,
            "registry_sources_not_in_candidate_inventory": sorted(excluded_source_ids),
        },
        "generic_sft_validation": generic_summary,
        "input_projection": {
            "billsum_records": 4_433,
            "billsum_train_supervised_tokens": 849_991,
            "billsum_validation_supervised_tokens": 49_986,
            "predecessor_records": 309_838,
            "predecessor_train_supervised_tokens": 9_553_473,
            "predecessor_validation_supervised_tokens": 578_618,
        },
        "near_duplicate_audit": near,
        "next_gate": {
            "action": (
                "materialize_registry_v0_20_hygiene_successor"
                if hygiene_needed
                else "close_residual_behavior_safe_task_and_language_gaps"
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
        "status": "PASS_registry_v0_20_full_candidate_mix_reaudit_nonproduction",
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
        raise ValueError("unsupported candidate-mix v0.14 report")
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256", None)
    if internal != sha256_value(unsigned):
        raise ValueError("candidate-mix v0.14 report internal hash mismatch")
    pool = report.get("candidate_pool", {})
    if (
        report.get("status")
        != "PASS_registry_v0_20_full_candidate_mix_reaudit_nonproduction"
        or report.get("production_eligible") is not False
        or report.get("raw_training_text_retained") is not False
        or report.get("evaluation_prohibition") != EXPECTED_EVALUATION_PROHIBITION
        or pool.get("candidate_records_read") != EXPECTED_INPUT_RECORDS
        or pool.get("input_train_supervised_tokens") != EXPECTED_INPUT_TRAIN_TOKENS
        or pool.get("input_validation_supervised_tokens")
        != EXPECTED_INPUT_VALIDATION_TOKENS
        or report.get("registry", {}).get("source_count") != EXPECTED_REGISTRY_SOURCE_COUNT
        or len(report.get("source_metrics", {})) != EXPECTED_INCLUDED_SOURCE_COUNT
    ):
        raise ValueError("candidate-mix v0.14 report boundary or totals changed")
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    for forbidden in ('"messages"', '"content"', '"assistant_response"', '"user_prompt"'):
        if forbidden in serialized:
            raise ValueError("candidate-mix v0.14 report retained raw training text")


def build_verification(
    *, plan: Mapping[str, Any], report: Mapping[str, Any], report_file_sha256: str
) -> dict[str, Any]:
    validate_report(report)
    verification: dict[str, Any] = {
        "checks": {
            "all_314271_rows_replayed_with_pure_byte_level_tokenizer": True,
            "billsum_test_and_ca_test_absent_and_evaluation_prohibition_preserved": True,
            "exact_and_near_duplicate_results_reproduced": True,
            "generic_schema_rights_provenance_secret_pii_and_benchmark_gates_replayed": True,
            "no_network_gpu_seal_sft_registry_or_production_action": True,
            "predecessor_v0_13_audit_and_billsum_promotion_bound": True,
            "raw_training_text_absent_from_outputs": True,
            "source_family_task_and_language_ledgers_reproduced": True,
        },
        "evaluation_prohibition": EXPECTED_EVALUATION_PROHIBITION,
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "PASS_verified_registry_v0_20_full_candidate_mix_reaudit_nonproduction",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification


__all__ = [
    "BILLSUM_SOURCE",
    "build_verification",
    "json_bytes",
    "load_plan",
    "rebuild_report",
    "validate_report",
]
