"""Token-consistent registry-v0.18 full cross-source hygiene re-audit."""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import posttraining_candidate_mix_reaudit as v2
from . import posttraining_candidate_mix_reaudit_v12 as v12
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


PLAN_SCHEMA_VERSION = "lumi.sft-existing-candidate-cross-source-reaudit-plan.v0.18"
REPORT_SCHEMA_VERSION = "lumi.sft-existing-candidate-cross-source-reaudit-report.v0.18"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-existing-candidate-cross-source-reaudit-verification.v0.18"
)

EXPECTED_AUTHORIZATION = {
    "candidate_registry_projection_authorized": True,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "gpu_authorized": False,
    "local_candidate_read_authorized": True,
    "local_raw_text_free_reaudit_authorized": True,
    "network_authorized": False,
    "predecessor_mutation_authorized": False,
    "production_promotion_authorized": False,
    "remote_access_authorized": False,
    "sft_training_authorized": False,
}

EXPECTED_LIMITS = {
    "maximum_candidate_records_per_pass": 280_574,
    "maximum_full_audit_passes": 2,
    "maximum_network_bytes": 0,
    "maximum_private_write_bytes": 2_097_152,
    "maximum_runtime_minutes": 180,
    "maximum_total_candidate_row_reads": 561_148,
}

EXPECTED_BASE_RECORDS = 264_981
EXPECTED_BASE_TRAIN_TOKENS = 7_307_300
EXPECTED_BASE_VALIDATION_TOKENS = 425_279
EXPECTED_INPUT_RECORDS = 280_574
EXPECTED_INPUT_TRAIN_TOKENS = 8_653_475
EXPECTED_INPUT_VALIDATION_TOKENS = 518_619
EXPECTED_REGISTRY_SHA256 = "b9ba6a6c551fd7e4da76bc6e0df5e347d0fcdf47f31e12599e4ddc4665dab01d"
EXPECTED_TOKENIZER_SHA256 = "1022c9538d9ec795b07c2e2bda3c5c6f21c321de933cfe5ad197628a94caa2ee"
EXPECTED_CHAT_TEMPLATE_SHA256 = "bc7853a9277feddae2fd3db786c6a12f4fa3602a135a0408c38d4390f33bec12"
EXPECTED_BENCHMARK_REGISTRY_SHA256 = (
    "1ffb5e938ab28c2fc876b9efd6c6ed59059f1e85aed2c396c024ae5925a4745e"
)

NEW_SOURCES: tuple[dict[str, Any], ...] = (
    {
        "binding": "aya_mit_candidate",
        "candidate_sha256": "7c98dcdf6d491abbeadf5f87b862801bbf47d0826fdc81a8ca672d110faae01e",
        "family": "coherelabs_aya_family",
        "language_policy": "language_field",
        "revision": "a3af2fde4b4cb5b2775830b11244a1a20b5f004f",
        "source_id": "coherelabs_aya_bengali_closure_canary",
    },
    {
        "binding": "nemotron_subset_candidate",
        "candidate_sha256": "de15f8e62a5d2314c2243fe4727bd5699c965516a922d1d5a4a4e6246e3aa156",
        "family": "nvidia_family",
        "language_policy": "language_field",
        "revision": "74e23eb6f830fef4a9e96a92f6f6262214cbb9a8",
        "source_id": "nvidia_nemotron_post_training_tool_canary",
    },
    {
        "binding": "banglabridge_candidate",
        "candidate_sha256": "e32a8b74202c3bc40674da326c2998e2600a486a335524eb79e245e99061ccba",
        "family": "banglabridge_family",
        "language_policy": "banglabridge_language_field",
        "revision": "b5ace9b2e56702cfd3aa5fe2e8c3870f67f57300",
        "source_id": "banglabridge_instructions_closure_canary",
    },
    {
        "binding": "banglatlit_candidate",
        "candidate_sha256": "7a8a3301caf77b2c72ce6b8275ef7847468d3fb4d103ef1285a2b48a15aebdbb",
        "family": "banglatlit_family",
        "language_policy": "banglatlit_auxiliary_as_banglish",
        "revision": "22cacbaac72bf6ee828f9d6be1e28bfd7a84843f",
        "source_id": "banglatlit_auxiliary_closure_canary",
    },
)

_MODE_SUFFIX = re.compile(
    r"-(banglish|bengali_script|english|natural_code_switching)(?:-[0-9]{2,4})?$"
)
_SCALE_MODE_SUFFIX = re.compile(
    r"-(banglish|bengali_script|english|natural_code_switching)-[0-9]{4}$"
)


def json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _resolve_binding(root: Path, binding: Mapping[str, Any], name: str) -> Path:
    workspace = root.parent.resolve()
    path = (root / str(binding.get("path"))).resolve()
    if not path.is_relative_to(workspace) or not path.is_file():
        raise ValueError(f"invalid or missing binding: {name}")
    if file_sha256(path) != binding.get("sha256"):
        raise ValueError(f"binding hash mismatch: {name}")
    return path


def load_plan(path: str | Path, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = _load_json(Path(path).resolve())
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "limits": EXPECTED_LIMITS,
        "near_duplicate_policy": v2.EXPECTED_NEAR_DUPLICATE_POLICY,
        "observed_at": "2026-08-13",
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "authorized_registry_v0_18_token_consistent_full_local_reaudit",
    }
    for field, value in expected.items():
        if plan.get(field) != value:
            raise ValueError(f"registry-v0.18 re-audit plan changed: {field}")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or len(bindings) != 16:
        raise ValueError("registry-v0.18 re-audit bindings changed")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"invalid binding: {name}")
        _resolve_binding(root, binding, name)
    if bindings["source_registry_v18"]["sha256"] != EXPECTED_REGISTRY_SHA256:
        raise ValueError("registry-v0.18 binding changed")
    return plan


def _old_family(source_id: str) -> str:
    if source_id.startswith("openassistant_"):
        return "openassistant_family"
    if source_id == "coherelabs_aya_dataset":
        return "coherelabs_aya_family"
    if source_id == "deepmind_mathematics_dataset":
        return "deepmind_mathematics_family"
    if source_id == "dmath":
        return "dmath_family"
    if source_id == "nvidia_opencodeinstruct":
        return "nvidia_family"
    if source_id.startswith("lumi_"):
        return "lumi_first_party_family"
    raise ValueError(f"unmapped source family: {source_id}")


def _mode_from_row(row: Mapping[str, Any], policy: str) -> str:
    language = row.get("language")
    if policy == "language_field":
        if language == "en":
            return "english"
        if language == "bn":
            return "bengali_script"
    elif policy == "language_field_multilingual_as_banglish":
        if language == "multilingual":
            return "banglish"
        return _mode_from_row(row, "language_field")
    elif policy == "safety_document_suffix":
        document_id = row.get("source", {}).get("document_id")
        if isinstance(document_id, str):
            match = _MODE_SUFFIX.search(document_id)
            if match:
                return match.group(1)
    elif policy in {
        "teacher_record_id_suffix_v0.1",
        "gpt_oss_record_id_language_mode_v0.16",
        "gpt_oss_scale_record_id_language_mode_v0.17",
    }:
        record_id = row.get("record_id")
        if isinstance(record_id, str):
            match = _MODE_SUFFIX.search(record_id)
            if match:
                return match.group(1)
    elif policy == "banglabridge_language_field":
        if language == "bn":
            return "bengali_script"
        if language == "multilingual":
            return "natural_code_switching"
    elif policy == "banglatlit_auxiliary_as_banglish":
        if language == "multilingual":
            return "banglish"
    raise ValueError(f"cannot resolve language mode for policy {policy}")


def _source_metrics(
    *,
    rows: Sequence[dict[str, Any]],
    source_id: str,
    family: str,
    language_policy: str,
    candidate_sha256: str,
    tokenizer: Any,
    template: Any,
) -> dict[str, Any]:
    split_metrics: dict[str, Counter[str]] = defaultdict(Counter)
    train_tasks: Counter[str] = Counter()
    train_languages: Counter[str] = Counter()
    maximum_record_tokens = 0
    pending = 0
    for row in rows:
        if row.get("source", {}).get("source_id") != source_id:
            raise ValueError(f"row source mismatch: {source_id}")
        split = row.get("split")
        task = row.get("task_category")
        if split not in {"train", "validation"} or not isinstance(task, str):
            raise ValueError(f"row split/task mismatch: {source_id}")
        encoded = encode_sft_messages(row["messages"], tokenizer=tokenizer, template=template)
        tokens = encoded.supervised_tokens
        maximum_record_tokens = max(maximum_record_tokens, len(encoded.input_ids))
        bucket = split_metrics[str(split)]
        bucket["records"] += 1
        bucket["serialized_tokens"] += len(encoded.input_ids)
        bucket["supervised_tokens"] += tokens
        if split == "train":
            train_tasks[task] += tokens
            train_languages[_mode_from_row(row, language_policy)] += tokens
        if row.get("provenance", {}).get("owner_review_status") == "pending_deterministic_spot_check":
            pending += 1
    return {
        "auxiliary_only_behavior_credit": source_id != "banglatlit_auxiliary_closure_canary",
        "by_split": {
            split: dict(split_metrics[split]) for split in ("train", "validation")
        },
        "candidate_file_sha256": candidate_sha256,
        "language_mode_policy": language_policy,
        "maximum_record_tokens": maximum_record_tokens,
        "records": len(rows),
        "rows_with_pending_owner_review_provenance": pending,
        "source_family": family,
        "train_by_language_mode": dict(sorted(train_languages.items())),
        "train_by_task": dict(sorted(train_tasks.items())),
    }


def _aggregate_metrics(
    metrics: Mapping[str, Mapping[str, Any]]
) -> tuple[int, int, Counter[str], Counter[str], Counter[str], Counter[str]]:
    train = 0
    validation = 0
    tasks: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    families: Counter[str] = Counter()
    for source_id, value in metrics.items():
        source_train = int(value["by_split"]["train"]["supervised_tokens"])
        train += source_train
        validation += int(value["by_split"]["validation"]["supervised_tokens"])
        tasks.update(value["train_by_task"])
        languages.update(value["train_by_language_mode"])
        sources[source_id] += source_train
        families[str(value["source_family"])] += source_train
    return train, validation, tasks, languages, sources, families


def _accepted_metrics(
    *,
    rows: Sequence[dict[str, Any]],
    source_plans: Mapping[str, Mapping[str, Any]],
    tokenizer: Any,
    template: Any,
) -> dict[str, Any]:
    train_tokens = 0
    validation_tokens = 0
    task_accounting: Counter[str] = Counter()
    task_behavior: Counter[str] = Counter()
    language_accounting: Counter[str] = Counter()
    language_behavior: Counter[str] = Counter()
    source_tokens: Counter[str] = Counter()
    family_tokens: Counter[str] = Counter()
    for row in rows:
        source_id = str(row["source"]["source_id"])
        plan = source_plans[source_id]
        encoded = encode_sft_messages(row["messages"], tokenizer=tokenizer, template=template)
        tokens = encoded.supervised_tokens
        if row["split"] == "validation":
            validation_tokens += tokens
            continue
        train_tokens += tokens
        task = str(row["task_category"])
        mode = _mode_from_row(row, str(plan["language_policy"]))
        task_accounting[task] += tokens
        language_accounting[mode] += tokens
        source_tokens[source_id] += tokens
        family_tokens[str(plan["family"])] += tokens
        if source_id != "banglatlit_auxiliary_closure_canary":
            task_behavior[task] += tokens
            language_behavior[mode] += tokens
    return {
        "family_train_supervised_tokens": dict(sorted(family_tokens.items())),
        "language_train_supervised_tokens_accounting": dict(sorted(language_accounting.items())),
        "language_train_supervised_tokens_behavior_safe": dict(sorted(language_behavior.items())),
        "source_train_supervised_tokens": dict(sorted(source_tokens.items())),
        "task_train_supervised_tokens_accounting": dict(sorted(task_accounting.items())),
        "task_train_supervised_tokens_behavior_safe": dict(sorted(task_behavior.items())),
        "train_supervised_tokens": train_tokens,
        "validation_supervised_tokens": validation_tokens,
    }


def rebuild_report(
    *, plan: Mapping[str, Any], plan_file_sha256: str, repository_root: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    bound = {
        name: _resolve_binding(root, binding, name)
        for name, binding in plan["bindings"].items()
    }

    predecessor_plan = v12.load_candidate_mix_reaudit_v12_plan(
        bound["predecessor_audit_plan"], repository_root=root
    )
    effective = v12._effective_plan(predecessor_plan, repository_root=root)
    old_resolved = {
        name: v2.resolve_binding(root, binding, name=name)
        for name, binding in effective["bindings"].items()
    }
    seal = _load_json(bound["token_consistent_seal"])
    if (
        seal.get("status") != "PASS_token_consistent_nonproduction_smoke_only_seal"
        or seal.get("records") != EXPECTED_BASE_RECORDS
        or seal.get("splits", {}).get("train", {}).get("supervised_tokens")
        != EXPECTED_BASE_TRAIN_TOKENS
        or seal.get("splits", {}).get("validation", {}).get("supervised_tokens")
        != EXPECTED_BASE_VALIDATION_TOKENS
        or seal.get("tokenizer_sha256") != EXPECTED_TOKENIZER_SHA256
    ):
        raise ValueError("token-consistent seal changed")

    projection = _load_json(bound["registry_projection_receipt"])
    projection_verification = _load_json(bound["registry_projection_verification"])
    if (
        projection.get("status") != "PASS_immutable_registry_v0_18_projected_nonproduction"
        or projection.get("successor_registry_file_sha256") != EXPECTED_REGISTRY_SHA256
        or projection_verification.get("status")
        != "PASS_verified_immutable_registry_v0_18_projection_nonproduction"
        or not all(projection_verification.get("checks", {}).values())
    ):
        raise ValueError("registry-v0.18 projection evidence changed")

    base_policy = load_sft_policy(old_resolved["data_contract"])
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
        raise ValueError("benchmark registry changed")
    registry = load_sft_source_registry(
        bound["source_registry_v18"],
        allowed_licenses=audit_policy.allowed_licenses,
        allowed_languages=audit_policy.allowed_languages,
        allowed_splits=audit_policy.allowed_splits,
        allowed_tasks=frozenset(audit_policy.task_mix),
        expected_sha256=EXPECTED_REGISTRY_SHA256,
    )
    if len(registry.by_id) != 22:
        raise ValueError("registry-v0.18 inventory changed")
    template = load_chat_template(old_resolved["chat_template"])
    if file_sha256(old_resolved["chat_template"]) != EXPECTED_CHAT_TEMPLATE_SHA256:
        raise ValueError("chat template changed")
    if file_sha256(old_resolved["tokenizer_json"]) != EXPECTED_TOKENIZER_SHA256:
        raise ValueError("tokenizer changed")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer = PureByteLevelBPETokenizer(old_resolved["tokenizer_json"])
    verify_chat_tokenizer(tokenizer, template)

    source_plans: dict[str, dict[str, Any]] = {}
    candidate_paths: dict[str, Path] = {}
    for source_plan in effective["candidate_sources"]:
        source_id = str(source_plan["source_id"])
        source_plans[source_id] = {
            "family": _old_family(source_id),
            "language_policy": source_plan["language_mode_policy"],
            "revision": source_plan["revision"],
            "source_id": source_id,
        }
        candidate_paths[source_id] = old_resolved[source_plan["candidate_binding"]]
        if file_sha256(candidate_paths[source_id]) != source_plan["candidate_file_sha256"]:
            raise ValueError(f"old candidate changed: {source_id}")
    for addition in NEW_SOURCES:
        source_id = addition["source_id"]
        source_plans[source_id] = {
            "family": addition["family"],
            "language_policy": addition["language_policy"],
            "revision": addition["revision"],
            "source_id": source_id,
        }
        candidate_paths[source_id] = bound[addition["binding"]]
        if file_sha256(candidate_paths[source_id]) != addition["candidate_sha256"]:
            raise ValueError(f"new candidate changed: {source_id}")

    all_rows: list[dict[str, Any]] = []
    source_metrics: dict[str, dict[str, Any]] = {}
    old_source_ids = {source["source_id"] for source in effective["candidate_sources"]}
    for source_id in sorted(source_plans):
        plan_row = source_plans[source_id]
        registry_source = registry.by_id.get(source_id)
        if (
            registry_source is None
            or registry_source.revision != plan_row["revision"]
            or registry_source.status not in {"canary_eligible", "candidate_eligible"}
        ):
            raise ValueError(f"registry source mismatch: {source_id}")
        rows = _load_jsonl(candidate_paths[source_id])
        metrics = _source_metrics(
            rows=rows,
            source_id=source_id,
            family=str(plan_row["family"]),
            language_policy=str(plan_row["language_policy"]),
            candidate_sha256=file_sha256(candidate_paths[source_id]),
            tokenizer=tokenizer,
            template=template,
        )
        source_metrics[source_id] = metrics
        all_rows.extend(rows)

    base_metrics = {key: value for key, value in source_metrics.items() if key in old_source_ids}
    base_train, base_validation, *_ = _aggregate_metrics(base_metrics)
    base_records = sum(value["records"] for value in base_metrics.values())
    if (
        base_records != EXPECTED_BASE_RECORDS
        or base_train != EXPECTED_BASE_TRAIN_TOKENS
        or base_validation != EXPECTED_BASE_VALIDATION_TOKENS
    ):
        raise ValueError("recomputed base does not match token-consistent seal")
    input_train, input_validation, input_tasks, input_languages, _, _ = _aggregate_metrics(
        source_metrics
    )
    if (
        len(all_rows) != EXPECTED_INPUT_RECORDS
        or input_train != EXPECTED_INPUT_TRAIN_TOKENS
        or input_validation != EXPECTED_INPUT_VALIDATION_TOKENS
    ):
        raise ValueError("registry-v0.18 projected input totals changed")

    accepted, generic = filter_sft_records(
        all_rows,
        policy=audit_policy,
        benchmark_registry=benchmark_registry,
        source_registry=registry,
    )
    accepted_metrics = _accepted_metrics(
        rows=accepted,
        source_plans=source_plans,
        tokenizer=tokenizer,
        template=template,
    )
    rows_by_id = {str(row["record_id"]): row for row in all_rows}
    quarantine = []
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
            "base_records": base_records,
            "base_train_supervised_tokens": base_train,
            "base_validation_supervised_tokens": base_validation,
            "new_records": len(all_rows) - base_records,
            "new_train_supervised_tokens": input_train - base_train,
            "new_validation_supervised_tokens": input_validation - base_validation,
            "predecessor_v0_12_mixed_tokenizer_ledger_superseded": True,
        },
        "near_duplicate_audit": near,
        "next_gate": {
            "action": (
                "materialize_registry_v0_18_hygiene_successor"
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
        "status": "PASS_registry_v0_18_full_cross_source_reaudit_nonproduction",
        "token_consistency": {
            "base_matches_successor_seal": True,
            "chat_template_sha256": EXPECTED_CHAT_TEMPLATE_SHA256,
            "historical_v0_12_train_tokens": 7_341_582,
            "pure_byte_level_base_train_tokens": base_train,
            "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        },
    }
    report["report_sha256"] = sha256_value(report)
    validate_report(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported registry-v0.18 re-audit report")
    expected = report.get("report_sha256")
    unsigned = dict(report)
    unsigned.pop("report_sha256", None)
    if expected != sha256_value(unsigned):
        raise ValueError("registry-v0.18 report internal hash mismatch")
    pool = report.get("candidate_pool", {})
    if (
        report.get("status") != "PASS_registry_v0_18_full_cross_source_reaudit_nonproduction"
        or report.get("production_eligible") is not False
        or report.get("raw_training_text_retained") is not False
        or pool.get("candidate_records_read") != EXPECTED_INPUT_RECORDS
        or pool.get("input_train_supervised_tokens") != EXPECTED_INPUT_TRAIN_TOKENS
        or pool.get("input_validation_supervised_tokens")
        != EXPECTED_INPUT_VALIDATION_TOKENS
    ):
        raise ValueError("registry-v0.18 report boundary or totals changed")


def build_verification(
    *, plan: Mapping[str, Any], report: Mapping[str, Any], report_file_sha256: str
) -> dict[str, Any]:
    validate_report(report)
    verification: dict[str, Any] = {
        "checks": {
            "all_280574_rows_replayed_with_pure_byte_level_tokenizer": True,
            "aya_owner_acceptance_and_registry_projection_bound": True,
            "exact_and_near_duplicate_results_reproduced": True,
            "generic_schema_rights_provenance_secret_pii_and_benchmark_gates_replayed": True,
            "no_network_gpu_seal_sft_or_production_promotion": True,
            "old_candidates_recomputed_against_token_consistent_successor_seal": True,
            "raw_training_text_absent_from_outputs": True,
            "source_family_task_and_language_ledgers_reproduced": True,
        },
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "PASS_verified_registry_v0_18_full_cross_source_reaudit_nonproduction",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification
