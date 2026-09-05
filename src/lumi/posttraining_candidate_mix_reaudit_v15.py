"""Registry-v0.21-bound full local candidate-mix re-audit."""

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
from . import posttraining_candidate_mix_reaudit_v14 as v14
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

PLAN_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-plan.v0.15"
REPORT_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-report.v0.15"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-verification.v0.15"
EXPECTED_AUTHORIZATION_TEXT_SHA256 = (
    "3872d6133f4b6d67cb194ece8168ac97db947023528e1de45e1b875e7d639dfa"
)

EXPECTED_AUTHORIZATION = {
    "active_data_contract_change_authorized": False,
    "candidate_mutation_authorized": False,
    "dataset_seal_authorized": False,
    "download_authorized": False,
    "external_api_authorized": False,
    "full_local_reaudit_authorized": True,
    "gpu_authorized": False,
    "instance_action_authorized": False,
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
    "maximum_candidate_records_per_pass": 314_716,
    "maximum_full_audit_passes": 2,
    "maximum_inference_calls": 0,
    "maximum_network_bytes": 0,
    "maximum_private_write_bytes": 2_097_152,
    "maximum_runtime_minutes": 180,
    "maximum_total_candidate_row_reads": 629_432,
}
EXPECTED_NEAR_DUPLICATE_POLICY = dict(v14.EXPECTED_NEAR_DUPLICATE_POLICY)
EXPECTED_BINDINGS = frozenset(
    {
        "benchmark_registry",
        "banglabridge_candidate",
        "banglabridge_promotion_receipt",
        "banglabridge_promotion_verification",
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

EXPECTED_INPUT_RECORDS = 314_716
EXPECTED_INPUT_TRAIN_TOKENS = 10_604_149
EXPECTED_INPUT_VALIDATION_TOKENS = 628_604
EXPECTED_INCLUDED_SOURCE_COUNT = 17
EXPECTED_REGISTRY_SOURCE_COUNT = 24
EXPECTED_REGISTRY_SHA256 = (
    "daa2d81755456728c5c72bca9ee95d5424aba0fc05e9fdd13223018f4055e3a1"
)
EXPECTED_PREDECESSOR_PLAN_SHA256 = (
    "3b4ef584b1ca8196db5451ff101e8993eaeb0d6d708cb76d75d95304f1be02dd"
)
EXPECTED_PREDECESSOR_REPORT_SHA256 = (
    "73a5c39e30fb8e64336211b33ae3bcd0a1a59849302d834ff260877d94e9c120"
)
EXPECTED_PREDECESSOR_VERIFICATION_SHA256 = (
    "2ba639e39e26af3e06f184d0f5ab2775c208b24d8ffa755c3b6e6bb9c4f518dd"
)
EXPECTED_DRAFT_SHA256 = (
    "c800bf8a10680171d97cd0322d98fc9b901c91882361be5e2b07625b4fe71c16"
)
EXPECTED_BANGLABRIDGE_CANDIDATE_SHA256 = (
    "5cf0b5f4ad4029f7a9e25b49ac7785f43607fb3196baf8c9f5dd272da0551058"
)
EXPECTED_BANGLABRIDGE_PROMOTION_RECEIPT_SHA256 = (
    "3bf74725a378540ffe63297d44d3f1f6474dcc3d11b7a1c4c4e57751c5cd620a"
)
EXPECTED_BANGLABRIDGE_PROMOTION_VERIFICATION_SHA256 = (
    "eca0064a8d04c0b9d89ea07f6d041de1235ed78ef8bbaae7de3740ef65ada8dc"
)
EXPECTED_OLD_BANGLABRIDGE_CANDIDATE_SHA256 = (
    "e32a8b74202c3bc40674da326c2998e2600a486a335524eb79e245e99061ccba"
)
EXPECTED_BENCHMARK_REGISTRY_SHA256 = v14.EXPECTED_BENCHMARK_REGISTRY_SHA256
EXPECTED_TOKENIZER_SHA256 = v14.EXPECTED_TOKENIZER_SHA256
EXPECTED_CHAT_TEMPLATE_SHA256 = v14.EXPECTED_CHAT_TEMPLATE_SHA256
EXPECTED_EVALUATION_PROHIBITION = dict(v14.EXPECTED_EVALUATION_PROHIBITION)
BILLSUM_SOURCE = dict(v14.BILLSUM_SOURCE)
BANGLABRIDGE_SOURCE = {
    "candidate_sha256": EXPECTED_BANGLABRIDGE_CANDIDATE_SHA256,
    "family": "banglabridge_family",
    "language_policy": "banglabridge_language_field",
    "revision": "b5ace9b2e56702cfd3aa5fe2e8c3870f67f57300",
    "source_id": "banglabridge_instructions_closure_canary",
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
        raise ValueError(f"invalid or missing v0.15 binding: {name}")
    if file_sha256(path) != binding.get("sha256"):
        raise ValueError(f"v0.15 binding hash mismatch: {name}")
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
            "report_path": "evidence/lumi-4b-sft-candidate-mix-reaudit-v0.15.json",
            "verification_path": (
                "evidence/lumi-4b-sft-candidate-mix-reaudit-verification-v0.15.json"
            ),
        },
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "authorized_registry_v0_21_full_local_reaudit_no_production_action",
    }
    for field, value in expected.items():
        if plan.get(field) != value:
            raise ValueError(f"candidate-mix v0.15 plan changed: {field}")
    text = plan.get("authorization_text")
    if not isinstance(text, str) or hashlib.sha256(text.encode()).hexdigest() != plan.get(
        "authorization_text_sha256"
    ):
        raise ValueError("candidate-mix v0.15 authorization text binding changed")
    if plan.get("authorization_text_sha256") != EXPECTED_AUTHORIZATION_TEXT_SHA256:
        raise ValueError("candidate-mix v0.15 exact authorization changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("candidate-mix v0.15 binding inventory changed")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"invalid v0.15 binding: {name}")
        _resolve_binding(root, binding, name)
    exact = {
        "benchmark_registry": EXPECTED_BENCHMARK_REGISTRY_SHA256,
        "banglabridge_candidate": EXPECTED_BANGLABRIDGE_CANDIDATE_SHA256,
        "banglabridge_promotion_receipt": (
            EXPECTED_BANGLABRIDGE_PROMOTION_RECEIPT_SHA256
        ),
        "banglabridge_promotion_verification": (
            EXPECTED_BANGLABRIDGE_PROMOTION_VERIFICATION_SHA256
        ),
        "draft": EXPECTED_DRAFT_SHA256,
        "predecessor_plan": EXPECTED_PREDECESSOR_PLAN_SHA256,
        "predecessor_report": EXPECTED_PREDECESSOR_REPORT_SHA256,
        "predecessor_verification": EXPECTED_PREDECESSOR_VERIFICATION_SHA256,
        "source_registry": EXPECTED_REGISTRY_SHA256,
    }
    for name, expected_sha256 in exact.items():
        if bindings[name]["sha256"] != expected_sha256:
            raise ValueError(f"candidate-mix v0.15 exact binding changed: {name}")
    return plan


def _validate_predecessor_and_promotion(bound: Mapping[str, Path]) -> None:
    predecessor = _load_json(bound["predecessor_report"])
    predecessor_verification = _load_json(bound["predecessor_verification"])
    promotion = _load_json(bound["banglabridge_promotion_receipt"])
    promotion_verification = _load_json(bound["banglabridge_promotion_verification"])
    v14.validate_report(predecessor)
    predecessor_verification_unsigned = dict(predecessor_verification)
    predecessor_verification_internal = predecessor_verification_unsigned.pop(
        "verification_sha256", None
    )
    promotion_unsigned = dict(promotion)
    promotion_internal = promotion_unsigned.pop("receipt_sha256", None)
    promotion_verification_unsigned = dict(promotion_verification)
    promotion_verification_internal = promotion_verification_unsigned.pop(
        "verification_sha256", None
    )
    if (
        predecessor.get("status")
        != "PASS_registry_v0_20_full_candidate_mix_reaudit_nonproduction"
        or predecessor.get("report_sha256")
        != "d6166562f6ac909dc984f1c0b025122b11c19eb2d9bdd9185b3c5db86030a4bd"
        or predecessor.get("candidate_pool", {}).get("candidate_records_read")
        != 314_271
        or predecessor.get("candidate_pool", {}).get("input_train_supervised_tokens")
        != 10_403_464
        or predecessor.get("candidate_pool", {}).get(
            "input_validation_supervised_tokens"
        )
        != 628_604
        or predecessor_verification.get("status")
        != "PASS_verified_registry_v0_20_full_candidate_mix_reaudit_nonproduction"
        or predecessor_verification_internal
        != "de6a1ee4b2626c4ca486a4bce2ec63a561fd67b13f9bb66832de5eee06084f88"
        or predecessor_verification_internal
        != sha256_value(predecessor_verification_unsigned)
        or predecessor_verification.get("report_file_sha256")
        != EXPECTED_PREDECESSOR_REPORT_SHA256
    ):
        raise ValueError("candidate-mix v0.15 predecessor evidence changed")
    if (
        promotion.get("status")
        != "verified_banglabridge_expanded_candidate_registry_v0_21_nonproduction"
        or promotion_internal
        != "eb182e4e8c2fce4dc0f9633afe440771f5389cd37880389958cbb61c3073ef09"
        or promotion_internal != sha256_value(promotion_unsigned)
        or promotion.get("candidate", {}).get("file_sha256")
        != EXPECTED_BANGLABRIDGE_CANDIDATE_SHA256
        or promotion.get("candidate", {}).get("bytes") != 1_967_432
        or promotion.get("candidate", {}).get("records") != 561
        or promotion.get("candidate", {}).get("train_records") != 544
        or promotion.get("candidate", {}).get("train_supervised_tokens") != 250_441
        or promotion.get("candidate", {}).get("validation_records") != 17
        or promotion.get("candidate", {}).get("validation_supervised_tokens") != 4_960
        or promotion.get("candidate", {}).get(
            "reused_without_regeneration_or_rematerialization"
        )
        is not True
        or promotion.get("registry", {}).get("successor_sha256")
        != EXPECTED_REGISTRY_SHA256
        or promotion.get("registry", {}).get("successor_bytes") != 28_405
        or promotion.get("registry", {}).get("replacement_source_id")
        != BANGLABRIDGE_SOURCE["source_id"]
        or promotion.get("checks", {}).get("full_candidate_mix_reaudit_run") is not False
        or promotion.get("registry", {}).get("changed_sources") != 1
        or promotion.get("registry", {}).get("added_sources") != 0
        or promotion.get("registry", {}).get("removed_sources") != 0
        or promotion.get("source_after", {}).get("source_id")
        != BANGLABRIDGE_SOURCE["source_id"]
        or promotion.get("source_after", {}).get("origin") != "mixed"
        or promotion.get("source_after", {}).get("adapter")
        != "banglabridge_expanded_delta_owner_bound_byte_exact_v0.21"
        or promotion.get("source_after", {}).get("production_eligible") is not False
        or promotion.get("source_after", {}).get("production_blockers")
        != [
            "owner_quality_review_is_32_row_deterministic_sample_not_full_record_review",
            "full_registry_v0_21_cross_source_hygiene_audit_not_complete",
            "production_dataset_seal_and_sft_authorization_not_complete",
        ]
        or promotion.get("next_candidate_mix_projection")
        != {
            "included_candidate_sources": EXPECTED_INCLUDED_SOURCE_COUNT,
            "input_records": EXPECTED_INPUT_RECORDS,
            "train_supervised_tokens": EXPECTED_INPUT_TRAIN_TOKENS,
            "validation_supervised_tokens": EXPECTED_INPUT_VALIDATION_TOKENS,
        }
        or promotion_verification.get("status")
        != "PASS_verified_banglabridge_expanded_candidate_registry_v0_21_nonproduction"
        or promotion_verification_internal
        != "d0b40764072a756a2e8f01c122f848e56dc6f4f56c19f3b1ff4d035ede830654"
        or promotion_verification_internal
        != sha256_value(promotion_verification_unsigned)
        or promotion_verification.get("receipt_file_sha256")
        != EXPECTED_BANGLABRIDGE_PROMOTION_RECEIPT_SHA256
        or promotion_verification.get("successor_registry", {}).get("source_count")
        != 24
        or promotion_verification.get("successor_registry", {}).get("file_sha256")
        != EXPECTED_REGISTRY_SHA256
        or not all(promotion_verification.get("checks", {}).values())
    ):
        raise ValueError("candidate-mix v0.15 BanglaBridge promotion evidence changed")


def _source_inventory(
    *, root: Path, bound: Mapping[str, Path]
) -> tuple[dict[str, dict[str, Any]], dict[str, Path], Mapping[str, Any], Mapping[str, Path]]:
    predecessor_plan = v14.load_plan(bound["predecessor_plan"], root)
    predecessor_bound = {
        name: v14._resolve_binding(root, binding, name)
        for name, binding in predecessor_plan["bindings"].items()
    }
    v14._validate_predecessor_and_promotion(predecessor_bound)
    source_plans, candidate_paths, effective, old_resolved = v14._source_inventory(
        root=root, bound=predecessor_bound
    )
    source_id = BANGLABRIDGE_SOURCE["source_id"]
    if (
        source_plans.get(source_id)
        != {
            "family": BANGLABRIDGE_SOURCE["family"],
            "language_policy": BANGLABRIDGE_SOURCE["language_policy"],
            "revision": BANGLABRIDGE_SOURCE["revision"],
            "source_id": source_id,
        }
        or file_sha256(candidate_paths[source_id])
        != EXPECTED_OLD_BANGLABRIDGE_CANDIDATE_SHA256
    ):
        raise ValueError("candidate-mix v0.15 predecessor BanglaBridge changed")
    candidate_paths[source_id] = bound["banglabridge_candidate"]
    if file_sha256(candidate_paths[source_id]) != EXPECTED_BANGLABRIDGE_CANDIDATE_SHA256:
        raise ValueError("candidate-mix v0.15 BanglaBridge successor changed")
    if len(source_plans) != EXPECTED_INCLUDED_SOURCE_COUNT:
        raise ValueError("candidate-mix v0.15 source inventory changed")
    return source_plans, candidate_paths, effective, old_resolved


def _validate_billsum_rows(rows: list[dict[str, Any]]) -> None:
    v14._validate_billsum_rows(rows)
    for row in rows:
        source = row.get("source", {})
        provenance = row.get("provenance", {})
        upstream_fields = (
            source.get("uri"),
            source.get("document_id"),
            source.get("upstream_split"),
            provenance.get("source_split") if isinstance(provenance, Mapping) else None,
        )
        for value in upstream_fields:
            normalized = str(value or "").casefold()
            if "ca_test" in normalized or "/test" in normalized:
                raise ValueError("BillSum test or ca_test entered candidate inventory")


def _validate_banglabridge_rows(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 561:
        raise ValueError("BanglaBridge successor row count changed")
    for row in rows:
        source = row.get("source")
        if (
            not isinstance(source, Mapping)
            or source.get("source_id") != BANGLABRIDGE_SOURCE["source_id"]
            or source.get("dataset_id") != "subhajitmahata84/banglabridge-instructions"
            or source.get("revision") != BANGLABRIDGE_SOURCE["revision"]
            or source.get("license_id") != "cc-by-4.0"
            or row.get("task_category") != "multilingual_bengali"
        ):
            raise ValueError("BanglaBridge successor source binding changed")


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
        raise ValueError("candidate-mix v0.15 benchmark registry changed")
    registry = load_sft_source_registry(
        bound["source_registry"],
        allowed_licenses=audit_policy.allowed_licenses,
        allowed_languages=audit_policy.allowed_languages,
        allowed_splits=audit_policy.allowed_splits,
        allowed_tasks=frozenset(audit_policy.task_mix),
        expected_sha256=EXPECTED_REGISTRY_SHA256,
    )
    if len(registry.by_id) != EXPECTED_REGISTRY_SOURCE_COUNT:
        raise ValueError("candidate-mix v0.15 registry inventory changed")
    excluded_source_ids = frozenset(registry.by_id) - frozenset(source_plans)
    if excluded_source_ids != EXPECTED_EXCLUDED_SOURCE_IDS:
        raise ValueError("candidate-mix v0.15 excluded-source inventory changed")
    predecessor_report = _load_json(bound["predecessor_report"])
    excluded = predecessor_report.get("excluded_source_gate", {}).get(
        "predecessor_evidence_replayed"
    )
    if not isinstance(excluded, list):
        raise ValueError("candidate-mix v0.15 predecessor exclusion evidence changed")
    template = load_chat_template(bound["chat_template"])
    if file_sha256(bound["chat_template"]) != EXPECTED_CHAT_TEMPLATE_SHA256:
        raise ValueError("candidate-mix v0.15 chat template changed")
    if file_sha256(bound["tokenizer_json"]) != EXPECTED_TOKENIZER_SHA256:
        raise ValueError("candidate-mix v0.15 tokenizer changed")
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
            raise ValueError(f"candidate-mix v0.15 registry source mismatch: {source_id}")
        rows = _load_jsonl(candidate_paths[source_id])
        if source_id == BILLSUM_SOURCE["source_id"]:
            _validate_billsum_rows(rows)
        elif source_id == BANGLABRIDGE_SOURCE["source_id"]:
            _validate_banglabridge_rows(rows)
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
        raise ValueError("candidate-mix v0.15 input totals changed")

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
            "final_records": EXPECTED_INPUT_RECORDS,
            "final_train_supervised_tokens": EXPECTED_INPUT_TRAIN_TOKENS,
            "final_validation_supervised_tokens": EXPECTED_INPUT_VALIDATION_TOKENS,
            "net_added_records": 445,
            "net_added_train_supervised_tokens": 200_685,
            "net_added_validation_supervised_tokens": 0,
            "predecessor_banglabridge_records": 116,
            "predecessor_banglabridge_train_supervised_tokens": 49_756,
            "predecessor_banglabridge_validation_supervised_tokens": 4_960,
            "predecessor_records": 314_271,
            "predecessor_train_supervised_tokens": 10_403_464,
            "predecessor_validation_supervised_tokens": 628_604,
            "successor_banglabridge_records": 561,
            "successor_banglabridge_train_supervised_tokens": 250_441,
            "successor_banglabridge_validation_supervised_tokens": 4_960,
        },
        "near_duplicate_audit": near,
        "next_gate": {
            "action": (
                "materialize_registry_v0_21_hygiene_successor"
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
        "resource_accounting": {
            "candidate_row_reads": EXPECTED_INPUT_RECORDS,
            "full_audit_passes": 1,
        },
        "schema_version": REPORT_SCHEMA_VERSION,
        "source_metrics": source_metrics,
        "status": "PASS_registry_v0_21_full_candidate_mix_reaudit_nonproduction",
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
        raise ValueError("unsupported candidate-mix v0.15 report")
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256", None)
    if internal != sha256_value(unsigned):
        raise ValueError("candidate-mix v0.15 report internal hash mismatch")
    pool = report.get("candidate_pool", {})
    generic = report.get("generic_sft_validation", {})
    exact = report.get("exact_duplicate_audit", {})
    near = report.get("near_duplicate_audit", {})
    closure = report.get("closure", {})
    banglabridge = report.get("source_metrics", {}).get(
        BANGLABRIDGE_SOURCE["source_id"], {}
    )
    excluded_gate = report.get("excluded_source_gate", {})
    predecessor_exclusions = excluded_gate.get("predecessor_evidence_replayed")
    expected_language_gaps = {
        "banglish": 772_114,
        "bengali_script": 0,
        "english": 0,
        "natural_code_switching": 365_420,
    }
    expected_task_gaps = {
        "coding": 0,
        "conversation": 5_934,
        "factual_qa": 114_306,
        "multilingual_bengali": 0,
        "reasoning_math_science": 0,
        "safety_helpfulness": 0,
        "tool_api": 2,
        "writing_summarization": 0,
    }
    expected_projection = {
        "final_records": 314_716,
        "final_train_supervised_tokens": 10_604_149,
        "final_validation_supervised_tokens": 628_604,
        "net_added_records": 445,
        "net_added_train_supervised_tokens": 200_685,
        "net_added_validation_supervised_tokens": 0,
        "predecessor_banglabridge_records": 116,
        "predecessor_banglabridge_train_supervised_tokens": 49_756,
        "predecessor_banglabridge_validation_supervised_tokens": 4_960,
        "predecessor_records": 314_271,
        "predecessor_train_supervised_tokens": 10_403_464,
        "predecessor_validation_supervised_tokens": 628_604,
        "successor_banglabridge_records": 561,
        "successor_banglabridge_train_supervised_tokens": 250_441,
        "successor_banglabridge_validation_supervised_tokens": 4_960,
    }
    if (
        report.get("status")
        != "PASS_registry_v0_21_full_candidate_mix_reaudit_nonproduction"
        or report.get("authorization") != EXPECTED_AUTHORIZATION
        or report.get("production_eligible") is not False
        or report.get("raw_training_text_retained") is not False
        or report.get("evaluation_prohibition") != EXPECTED_EVALUATION_PROHIBITION
        or pool.get("candidate_records_read") != EXPECTED_INPUT_RECORDS
        or pool.get("input_train_supervised_tokens") != EXPECTED_INPUT_TRAIN_TOKENS
        or pool.get("input_validation_supervised_tokens")
        != EXPECTED_INPUT_VALIDATION_TOKENS
        or report.get("registry")
        != {
            "file_sha256": EXPECTED_REGISTRY_SHA256,
            "source_count": EXPECTED_REGISTRY_SOURCE_COUNT,
        }
        or len(report.get("source_metrics", {})) != EXPECTED_INCLUDED_SOURCE_COUNT
        or report.get("resource_accounting")
        != {
            "candidate_row_reads": EXPECTED_INPUT_RECORDS,
            "full_audit_passes": 1,
        }
        or excluded_gate.get("candidate_rows_from_excluded_sources") != 0
        or excluded_gate.get("registry_sources_not_in_candidate_inventory")
        != sorted(EXPECTED_EXCLUDED_SOURCE_IDS)
        or not isinstance(predecessor_exclusions, list)
        or len(predecessor_exclusions) != 1
        or predecessor_exclusions[0].get("source_id") != "openr1_math_220k"
        or predecessor_exclusions[0].get("counted_train_supervised_tokens") != 0
        or generic.get("accepted") != EXPECTED_INPUT_RECORDS
        or generic.get("rejected") != 0
        or generic.get("total") != EXPECTED_INPUT_RECORDS
        or generic.get("rejection_counts") != {}
        or exact.get("duplicate_groups") != 0
        or exact.get("cross_source_duplicate_groups") != 0
        or exact.get("train_validation_overlap_groups") != 0
        or exact.get("duplicate_extra_records") != 0
        or near.get("records") != EXPECTED_INPUT_RECORDS
        or near.get("multi_record_components") != 0
        or near.get("cross_source_components") != 0
        or near.get("train_validation_components") != 0
        or near.get("train_validation_records") != 0
        or near.get("duplicate_extra_records") != 0
        or near.get("edge_count") != 0
        or pool.get("accepted_records_after_generic_quarantine")
        != EXPECTED_INPUT_RECORDS
        or pool.get("accepted_train_supervised_tokens")
        != EXPECTED_INPUT_TRAIN_TOKENS
        or pool.get("accepted_validation_supervised_tokens")
        != EXPECTED_INPUT_VALIDATION_TOKENS
        or pool.get("strict_targeted_gap_lower_bound") != 1_137_534
        or closure.get("language_floor_gaps_behavior_safe") != expected_language_gaps
        or closure.get("task_floor_gaps_behavior_safe") != expected_task_gaps
        or closure.get("family_cap_dilution_gap") != 0
        or closure.get("source_family_cap_violations") != []
        or closure.get("source_family_shares", {}).get("banglabridge_family")
        != 0.02361727
        or banglabridge.get("candidate_file_sha256")
        != EXPECTED_BANGLABRIDGE_CANDIDATE_SHA256
        or banglabridge.get("records") != 561
        or banglabridge.get("rows_with_pending_owner_review_provenance") != 561
        or banglabridge.get("by_split", {}).get("train", {}).get("records") != 544
        or banglabridge.get("by_split", {}).get("train", {}).get(
            "supervised_tokens"
        )
        != 250_441
        or banglabridge.get("by_split", {}).get("validation", {}).get("records")
        != 17
        or banglabridge.get("by_split", {}).get("validation", {}).get(
            "supervised_tokens"
        )
        != 4_960
        or banglabridge.get("train_by_language_mode")
        != {"bengali_script": 239_144, "natural_code_switching": 11_297}
        or banglabridge.get("train_by_task")
        != {"multilingual_bengali": 250_441}
        or report.get("input_projection") != expected_projection
        or report.get("quarantine", {}).get("total_records") != 0
        or report.get("quarantine", {}).get("records") != []
    ):
        raise ValueError("candidate-mix v0.15 report boundary or totals changed")
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    for forbidden in ('"messages"', '"content"', '"assistant_response"', '"user_prompt"'):
        if forbidden in serialized:
            raise ValueError("candidate-mix v0.15 report retained raw training text")


def build_verification(
    *, plan: Mapping[str, Any], report: Mapping[str, Any], report_file_sha256: str
) -> dict[str, Any]:
    validate_report(report)
    verification: dict[str, Any] = {
        "checks": {
            "all_314716_rows_replayed_with_pure_byte_level_tokenizer": True,
            "banglabridge_561_row_replacement_and_owner_bound_promotion_reproduced": True,
            "billsum_test_and_ca_test_absent_and_evaluation_prohibition_preserved": True,
            "exact_and_near_duplicate_results_reproduced": True,
            "generic_schema_rights_provenance_secret_pii_and_benchmark_gates_replayed": True,
            "no_network_gpu_seal_sft_registry_or_production_action": True,
            "predecessor_v0_14_audit_and_banglabridge_replacement_promotion_bound": True,
            "raw_training_text_absent_from_outputs": True,
            "source_family_task_and_language_ledgers_reproduced": True,
        },
        "evaluation_prohibition": EXPECTED_EVALUATION_PROHIBITION,
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": report["report_sha256"],
        "resource_accounting": {
            "candidate_row_reads": EXPECTED_LIMITS["maximum_total_candidate_row_reads"],
            "full_audit_passes": EXPECTED_LIMITS["maximum_full_audit_passes"],
        },
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "PASS_verified_registry_v0_21_full_candidate_mix_reaudit_nonproduction",
    }
    verification["verification_sha256"] = sha256_value(verification)
    validate_verification(verification)
    return verification


def validate_verification(verification: Mapping[str, Any]) -> None:
    if verification.get("schema_version") != VERIFICATION_SCHEMA_VERSION:
        raise ValueError("unsupported candidate-mix v0.15 verification")
    unsigned = dict(verification)
    internal = unsigned.pop("verification_sha256", None)
    if internal != sha256_value(unsigned):
        raise ValueError("candidate-mix v0.15 verification internal hash mismatch")
    checks = verification.get("checks")
    if (
        verification.get("status")
        != "PASS_verified_registry_v0_21_full_candidate_mix_reaudit_nonproduction"
        or verification.get("production_eligible") is not False
        or verification.get("evaluation_prohibition") != EXPECTED_EVALUATION_PROHIBITION
        or not isinstance(checks, Mapping)
        or len(checks) != 9
        or not all(value is True for value in checks.values())
        or verification.get("resource_accounting")
        != {
            "candidate_row_reads": EXPECTED_LIMITS["maximum_total_candidate_row_reads"],
            "full_audit_passes": EXPECTED_LIMITS["maximum_full_audit_passes"],
        }
    ):
        raise ValueError("candidate-mix v0.15 verification boundary changed")
    serialized = json.dumps(verification, ensure_ascii=False, sort_keys=True)
    for forbidden in ('"messages"', '"content"', '"assistant_response"', '"user_prompt"'):
        if forbidden in serialized:
            raise ValueError("candidate-mix v0.15 verification retained raw training text")


__all__ = [
    "BANGLABRIDGE_SOURCE",
    "BILLSUM_SOURCE",
    "build_verification",
    "json_bytes",
    "load_plan",
    "rebuild_report",
    "validate_report",
    "validate_verification",
]
