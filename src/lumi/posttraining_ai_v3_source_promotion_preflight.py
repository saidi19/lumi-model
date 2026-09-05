"""Fail-closed preflight for the reviewed AI-assisted v0.3 source registry entry."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .data_quality import BenchmarkRegistry
from .posttraining_data import filter_sft_records, load_sft_policy
from .posttraining_sources import (
    SFTSource,
    SFTSourceRegistry,
    load_sft_source_registry,
)

PLAN_SCHEMA_VERSION = "lumi.sft-ai-v3-source-promotion-preflight.v0.1"
PLAN_STATUS = "local_preflight_only_registry_successor_not_materialized"
SOURCE_ID = "lumi_ai_assisted_sft_v0_3"
SOURCE_REVISION = "7961c081dc57339613fd95961adea65d02d22348b8a69217cc0af3949fe6a7c6"

EXPECTED_AUTHORIZATION = {
    "candidate_registry_change_authorized": False,
    "data_acquisition_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "purpose": "reviewed_ai_v0_3_candidate_registry_successor_projection_only",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSON object required at line {line_number}: {path}")
        values.append(value)
    return values


def resolve_binding(root: Path, binding: Mapping[str, Any]) -> Path:
    relative = binding.get("path")
    expected_sha256 = binding.get("sha256")
    if not isinstance(relative, str) or not relative:
        raise ValueError("AI v0.3 preflight binding path is missing")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError(f"AI v0.3 preflight binding SHA-256 invalid: {relative}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"AI v0.3 preflight binding escapes repository: {relative}")
    if not resolved.is_file() or _file_sha256(resolved) != expected_sha256:
        raise ValueError(f"AI v0.3 preflight binding changed: {relative}")
    return resolved


def load_ai_v3_promotion_preflight_plan(
    path: Path,
    *,
    repository_root: Path,
) -> dict[str, Any]:
    root = repository_root.resolve()
    plan = _read_object(path.resolve())
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported AI v0.3 source-promotion preflight")
    if plan.get("status") != PLAN_STATUS:
        raise ValueError("AI v0.3 source-promotion preflight status changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("AI v0.3 preflight authorization boundary changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("AI v0.3 preflight bindings are missing")
    for binding in bindings.values():
        if not isinstance(binding, dict):
            raise ValueError("AI v0.3 preflight binding is invalid")
        resolve_binding(root, binding)
    limits = plan.get("limits")
    if limits != {
        "added_sources": 1,
        "candidate_records": 320,
        "train_records": 240,
        "train_supervised_tokens": 9314,
        "validation_records": 80,
        "validation_supervised_tokens": 2138,
    }:
        raise ValueError("AI v0.3 preflight limits changed")
    source = plan.get("source_after")
    if not isinstance(source, dict):
        raise ValueError("AI v0.3 proposed source is missing")
    if source.get("source_id") != SOURCE_ID or source.get("revision") != SOURCE_REVISION:
        raise ValueError("AI v0.3 proposed source identity changed")
    return plan


def build_projected_successor(
    predecessor: Mapping[str, Any],
    *,
    source_after: Mapping[str, Any],
) -> dict[str, Any]:
    sources = predecessor.get("sources")
    if not isinstance(sources, list) or len(sources) != 12:
        raise ValueError("AI v0.3 predecessor source count changed")
    if any(
        isinstance(source, Mapping) and source.get("source_id") == SOURCE_ID
        for source in sources
    ):
        raise ValueError("AI v0.3 source already exists in predecessor registry")
    successor = copy.deepcopy(dict(predecessor))
    successor["sources"].append(copy.deepcopy(dict(source_after)))
    return successor


def _verify_owner_receipt(receipt: Mapping[str, Any]) -> None:
    if receipt.get("schema_version") != "lumi.ai-assisted-owner-review-receipt.v0.3":
        raise ValueError("unsupported AI v0.3 owner receipt")
    unsigned = dict(receipt)
    claimed = unsigned.pop("receipt_sha256", None)
    if claimed != _canonical_sha256(unsigned):
        raise ValueError("AI v0.3 owner receipt seal mismatch")
    if (
        receipt.get("status")
        != "owner_spot_check_passed_pilot_still_not_production_dataset"
        or receipt.get("sample_records") != 32
        or receipt.get("accepted_records") != 32
        or receipt.get("rejected_records") != 0
        or receipt.get("all_sampled_rows_accepted") is not True
        or receipt.get("owner_review_claim")
        != "ai_generated_plus_owner_reviewed_not_human_written"
        or receipt.get("production_eligible") is not False
    ):
        raise ValueError("AI v0.3 owner review boundary changed")


def _verify_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_source: SFTSource,
) -> Counter[str]:
    if len(records) != 320:
        raise ValueError("AI v0.3 candidate record count changed")
    splits: Counter[str] = Counter()
    review_statuses: Counter[str] = Counter()
    record_ids: set[str] = set()
    for record in records:
        record_id = record.get("record_id")
        if not isinstance(record_id, str) or record_id in record_ids:
            raise ValueError("AI v0.3 candidate record IDs changed")
        record_ids.add(record_id)
        source = record.get("source")
        if not isinstance(source, Mapping):
            raise ValueError("AI v0.3 source metadata is missing")
        if (
            source.get("source_id") != expected_source.source_id
            or source.get("dataset_id") != expected_source.dataset_id
            or source.get("revision") != expected_source.revision
            or source.get("license_id") != expected_source.license_id
        ):
            raise ValueError("AI v0.3 source metadata changed")
        split = record.get("split")
        if split not in {"train", "validation"}:
            raise ValueError("AI v0.3 split changed")
        splits[str(split)] += 1
        provenance = record.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("AI v0.3 provenance is missing")
        if provenance.get("generation_method") != "ai_generated":
            raise ValueError("AI v0.3 generation provenance changed")
        review_statuses[str(provenance.get("human_review_status"))] += 1
    if splits != Counter({"train": 240, "validation": 80}):
        raise ValueError("AI v0.3 train/validation counts changed")
    return review_statuses


def build_ai_v3_promotion_preflight_report(
    plan: Mapping[str, Any],
    *,
    plan_file_sha256: str,
    repository_root: Path,
) -> dict[str, Any]:
    root = repository_root.resolve()
    bindings = plan["bindings"]
    policy = load_sft_policy(resolve_binding(root, bindings["data_contract"]))
    predecessor_path = resolve_binding(root, bindings["predecessor_registry"])
    predecessor_registry = load_sft_source_registry(
        predecessor_path,
        allowed_licenses=policy.allowed_licenses,
        allowed_languages=policy.allowed_languages,
        allowed_splits=policy.allowed_splits,
        allowed_tasks=frozenset(policy.task_mix),
        expected_sha256=bindings["predecessor_registry"]["sha256"],
    )
    predecessor_value = _read_object(predecessor_path)
    proposed_source = SFTSource.from_mapping(
        plan["source_after"],
        allowed_licenses=policy.allowed_licenses,
        allowed_languages=policy.allowed_languages,
        allowed_splits=policy.allowed_splits,
        allowed_tasks=frozenset(policy.task_mix),
    )
    successor_value = build_projected_successor(
        predecessor_value,
        source_after=plan["source_after"],
    )
    parsed_sources = tuple(
        SFTSource.from_mapping(
            source,
            allowed_licenses=policy.allowed_licenses,
            allowed_languages=policy.allowed_languages,
            allowed_splits=policy.allowed_splits,
            allowed_tasks=frozenset(policy.task_mix),
        )
        for source in successor_value["sources"]
    )
    if parsed_sources[:-1] != predecessor_registry.sources or parsed_sources[-1] != proposed_source:
        raise ValueError("AI v0.3 projected successor changed predecessor sources")
    projected_sha256 = hashlib.sha256(_json_file_bytes(successor_value)).hexdigest()
    candidate_registry = SFTSourceRegistry(
        observed_at=predecessor_registry.observed_at,
        registry_sha256=projected_sha256,
        sources=parsed_sources,
    )

    records = _read_jsonl(resolve_binding(root, bindings["accepted_records"]))
    review_statuses = _verify_records(records, expected_source=proposed_source)
    accepted, replay = filter_sft_records(
        records,
        policy=policy,
        benchmark_registry=BenchmarkRegistry(policy.benchmark_decontamination, root),
        source_registry=candidate_registry,
    )
    filter_report = _read_object(resolve_binding(root, bindings["filter_report"]))
    if (
        len(accepted) != 320
        or replay.get("rejected") != 0
        or replay.get("accepted_dataset_sha256")
        != filter_report.get("accepted_dataset_sha256")
    ):
        raise ValueError("AI v0.3 projected-registry filtering changed")

    owner_receipt = _read_object(resolve_binding(root, bindings["owner_review_receipt"]))
    _verify_owner_receipt(owner_receipt)
    manifest = _read_object(resolve_binding(root, bindings["candidate_manifest"]))
    if (
        manifest.get("candidate_records") != 320
        or manifest.get("candidate_source_revision") != SOURCE_REVISION
        or manifest.get("production_eligible") is not False
    ):
        raise ValueError("AI v0.3 candidate manifest changed")
    token_evidence = _read_object(resolve_binding(root, bindings["token_evidence"]))
    ai_tokens = token_evidence.get("source_token_evidence", {}).get("ai_assisted_v0_3")
    if ai_tokens != {
        "by_split": {
            "train": {"records": 240, "supervised_tokens": 9314},
            "validation": {"records": 80, "supervised_tokens": 2138},
        },
        "train_by_task": {
            "conversation": 3667,
            "multilingual_bengali": 4329,
            "reasoning_math_science": 120,
            "tool_api": 1198,
        },
    }:
        raise ValueError("AI v0.3 frozen token evidence changed")
    reaudit = _read_object(resolve_binding(root, bindings["candidate_mix_reaudit"]))
    excluded = [
        item
        for item in reaudit.get("excluded_sources", [])
        if item.get("source_id") == SOURCE_ID
    ]
    if len(excluded) != 1 or excluded[0].get("reason") != (
        "exact_v0_3_source_and_revision_not_in_registry_v0_7"
    ):
        raise ValueError("AI v0.3 candidate-mix blocker changed")

    report: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "candidate": {
            "accepted_records": 320,
            "filter_rejections": 0,
            "owner_sample_accepted": 32,
            "owner_sample_rejected": 0,
            "per_record_human_review_status_counts": dict(sorted(review_statuses.items())),
            "source_revision": SOURCE_REVISION,
            "train_records": 240,
            "train_supervised_tokens": 9314,
            "validation_records": 80,
            "validation_supervised_tokens": 2138,
        },
        "checks": {
            "all_320_records_pass_projected_registry_filter": True,
            "candidate_and_owner_receipt_hashes_verified": True,
            "candidate_registry_change_executed": False,
            "predecessor_registry_unchanged": True,
            "production_dataset_promotions": 0,
            "projected_successor_adds_only_ai_v0_3": True,
            "raw_training_text_retained": False,
        },
        "next_gate": {
            "action": "separately_authorize_exact_candidate_registry_v0_8_materialization",
            "authorization_required": True,
            "post_promotion_action": (
                "reaudit_candidate_mix_then_continue_rights_clear_bulk_acquisition"
            ),
        },
        "observed_at": plan["observed_at"],
        "plan_file_sha256": plan_file_sha256,
        "production_eligible": False,
        "projected_successor": {
            "added_source_id": SOURCE_ID,
            "path": plan["output"]["projected_successor_registry_path"],
            "predecessor_source_count": 12,
            "sha256": projected_sha256,
            "source_count": 13,
        },
        "schema_version": "lumi.sft-ai-v3-source-promotion-preflight-report.v0.1",
        "status": "verified_local_preflight_registry_successor_not_materialized",
    }
    report["report_sha256"] = _canonical_sha256(report)
    return report


def verify_ai_v3_promotion_preflight_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != (
        "lumi.sft-ai-v3-source-promotion-preflight-report.v0.1"
    ):
        raise ValueError("unsupported AI v0.3 promotion preflight report")
    unsigned = dict(report)
    claimed = unsigned.pop("report_sha256", None)
    if claimed != _canonical_sha256(unsigned):
        raise ValueError("AI v0.3 promotion preflight report seal mismatch")
    checks = report.get("checks")
    if (
        not isinstance(checks, Mapping)
        or checks.get("candidate_registry_change_executed") is not False
    ):
        raise ValueError("AI v0.3 preflight unexpectedly changed the registry")
    if checks.get("production_dataset_promotions") != 0:
        raise ValueError("AI v0.3 preflight unexpectedly promoted production data")
    if checks.get("raw_training_text_retained") is not False:
        raise ValueError("AI v0.3 preflight retained raw training text")
