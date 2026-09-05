"""Fail-closed source-registry preflight for the reviewed OASST2 candidate."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .data_quality import sha256_file
from .posttraining_oasst2_curation import sha256_value
from .posttraining_oasst2_owner_review import validate_owner_review_receipt
from .posttraining_sources import SFTSource, SFTSourceRegistry

PLAN_SCHEMA_VERSION = "lumi.sft-oasst2-source-promotion-preflight.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-oasst2-source-promotion-preflight-report.v0.1"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-oasst2-source-promotion-preflight-verification.v0.1"
PLAN_STATUS = "local_preflight_only_registry_v0_13_successor_not_materialized"
OWNER_REQUEST_TEXT = "okay next step shuru kor"

SOURCE_ID = "openassistant_oasst2"
DATASET_ID = "OpenAssistant/oasst2"
SOURCE_REVISION = "179dd21fc55192153d94adb0e0ce8f69e222bf75"
SOURCE_OBJECT_SHA256 = "7a886a16ccfc1173c4f00a6897523e3c95b2785a86ee44a18a98f4f2807ee29b"
EXPECTED_CANDIDATE_SHA256 = "1642c444f75f29c70f5182d0ff0ff94133f52b942e36fb3655be62caa4a4e6df"
EXPECTED_ATTRIBUTION_SHA256 = "05d1c52ed95c8981ddaf9f7647c60983302f7be9f461e851790a1979298d052b"
EXPECTED_TOKEN_METRICS_SHA256 = "54a5d9581b23fe121a0bc2e5d099a34e442a25af92a9215c973df78cf61e4624"
EXPECTED_RECORDS = 6_426
EXPECTED_TRAIN_RECORDS = 6_136
EXPECTED_TRAIN_TOKENS = 2_221_091
EXPECTED_VALIDATION_RECORDS = 290
EXPECTED_VALIDATION_TOKENS = 96_436
EXPECTED_TOTAL_TOKENS = 2_317_527
EXPECTED_PREDECESSOR_SOURCES = 14

EXPECTED_AUTHORIZATION = {
    "candidate_registry_change_authorized": False,
    "data_acquisition_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "purpose": "reviewed_oasst2_registry_v0_13_successor_projection_only",
    "source_registry_promotion_authorized": False,
    "unrelated_access_authorized": False,
}
EXPECTED_LIMITS = {
    "added_sources": 1,
    "candidate_records": EXPECTED_RECORDS,
    "maximum_report_bytes": 1_048_576,
    "maximum_runtime_minutes": 30,
    "train_records": EXPECTED_TRAIN_RECORDS,
    "train_supervised_tokens": EXPECTED_TRAIN_TOKENS,
    "validation_records": EXPECTED_VALIDATION_RECORDS,
    "validation_supervised_tokens": EXPECTED_VALIDATION_TOKENS,
}
EXPECTED_BINDINGS = {
    "accepted_records",
    "attribution_manifest",
    "curation_plan",
    "curation_report",
    "curation_verification",
    "data_contract",
    "implementation",
    "mix_closure_report",
    "mix_closure_verification",
    "owner_review_receipt",
    "owner_review_verification",
    "predecessor_registry",
    "runner",
    "test",
    "token_metrics",
    "verifier",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _read_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def resolve_binding(
    repository_root: str | Path,
    specification: Mapping[str, Any],
    *,
    name: str,
) -> Path:
    root = Path(repository_root).resolve()
    relative = specification.get("path")
    expected = _require_sha256(specification.get("sha256"), f"bindings.{name}.sha256")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"invalid OASST2 promotion-preflight binding: {name}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.parent.resolve()) or not resolved.is_file():
        raise ValueError(f"OASST2 promotion-preflight binding missing: {name}")
    if sha256_file(resolved) != expected:
        raise ValueError(f"OASST2 promotion-preflight binding changed: {name}")
    return resolved


def _policy_sets(contract: Mapping[str, Any]) -> dict[str, frozenset[str]]:
    task_mix = contract.get("task_mix")
    if not isinstance(task_mix, Mapping):
        raise ValueError("OASST2 data-contract task mix changed")
    values = {
        "licenses": contract.get("allowed_licenses"),
        "languages": contract.get("allowed_languages"),
        "splits": contract.get("allowed_splits"),
        "tasks": list(task_mix),
    }
    if any(
        not isinstance(items, list) or not items or not all(isinstance(item, str) for item in items)
        for items in values.values()
    ):
        raise ValueError("OASST2 data-contract allowlists changed")
    return {name: frozenset(items) for name, items in values.items()}


def load_promotion_preflight_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = _read_object(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported OASST2 source-promotion preflight")
    if plan.get("status") != PLAN_STATUS:
        raise ValueError("OASST2 source-promotion preflight status changed")
    if plan.get("observed_at") != "2026-08-08":
        raise ValueError("OASST2 source-promotion preflight date changed")
    if plan.get("owner_request_text") != OWNER_REQUEST_TEXT:
        raise ValueError("OASST2 source-promotion owner request changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("OASST2 source-promotion authorization boundary changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("OASST2 source-promotion preflight limits changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("OASST2 source-promotion preflight bindings changed")
    for name, specification in bindings.items():
        if not isinstance(specification, Mapping):
            raise ValueError(f"invalid OASST2 promotion binding: {name}")
        resolve_binding(root, specification, name=name)
    if bindings["accepted_records"]["sha256"] != EXPECTED_CANDIDATE_SHA256:
        raise ValueError("OASST2 candidate identity changed")
    if bindings["attribution_manifest"]["sha256"] != EXPECTED_ATTRIBUTION_SHA256:
        raise ValueError("OASST2 attribution identity changed")
    if bindings["token_metrics"]["sha256"] != EXPECTED_TOKEN_METRICS_SHA256:
        raise ValueError("OASST2 token metrics identity changed")
    source = plan.get("source_after")
    if not isinstance(source, Mapping):
        raise ValueError("OASST2 proposed source is missing")
    if source.get("source_id") != SOURCE_ID or source.get("revision") != SOURCE_REVISION:
        raise ValueError("OASST2 proposed source identity changed")
    if plan.get("output") != {
        "preflight_report_path": (
            "evidence/lumi-4b-sft-oasst2-source-promotion-preflight-v0.1.json"
        ),
        "projected_successor_registry_path": "configs/lumi-4b-sft-source-registry-v0.13.json",
        "verification_path": (
            "evidence/lumi-4b-sft-oasst2-source-promotion-preflight-verification-v0.1.json"
        ),
    }:
        raise ValueError("OASST2 source-promotion preflight outputs changed")
    return plan


def build_projected_successor(
    predecessor: Mapping[str, Any],
    *,
    source_after: Mapping[str, Any],
) -> dict[str, Any]:
    sources = predecessor.get("sources")
    if not isinstance(sources, list) or len(sources) != EXPECTED_PREDECESSOR_SOURCES:
        raise ValueError("OASST2 predecessor source count changed")
    if any(
        isinstance(source, Mapping) and source.get("source_id") == SOURCE_ID for source in sources
    ):
        raise ValueError("OASST2 source already exists in predecessor registry")
    successor = copy.deepcopy(dict(predecessor))
    successor["sources"].append(copy.deepcopy(dict(source_after)))
    return successor


def _verify_bound_evidence(root: Path, bindings: Mapping[str, Any]) -> None:
    mix = _read_object(
        resolve_binding(root, bindings["mix_closure_report"], name="mix_closure_report")
    )
    if mix.get("report_sha256") != (
        "ee5adbc4b9ad0c41a00977c61f4c0b4ba04df0f8719eef0bbd5920c4cb7331fc"
    ):
        raise ValueError("OASST2 mix-closure source-selection report changed")
    next_gate = mix.get("next_gate", {})
    if (
        next_gate.get("dataset_id") != DATASET_ID
        or next_gate.get("revision") != SOURCE_REVISION
        or next_gate.get("target_retained_train_supervised_tokens") != 2_600_000
    ):
        raise ValueError("OASST2 mix-closure selection changed")
    mix_verification = _read_object(
        resolve_binding(
            root,
            bindings["mix_closure_verification"],
            name="mix_closure_verification",
        )
    )
    if mix_verification.get("verification_sha256") != (
        "c341b9714a00c21bd6cb6503adcfefafe7d8259db30a79d06f3ad4ebf3521419"
    ):
        raise ValueError("OASST2 mix-closure verification changed")

    curation = _read_object(
        resolve_binding(root, bindings["curation_report"], name="curation_report")
    )
    if curation.get("report_sha256") != (
        "47fa0458020b9328c1b355de73164b2387c1a86f2eaaa6262838229cf8aeda37"
    ):
        raise ValueError("OASST2 curation report seal changed")
    source = curation.get("source", {})
    if (
        source.get("source_id") != SOURCE_ID
        or source.get("dataset_id") != DATASET_ID
        or source.get("revision") != SOURCE_REVISION
        or source.get("license_id") != "apache-2.0"
        or source.get("object_sha256") != SOURCE_OBJECT_SHA256
    ):
        raise ValueError("OASST2 curation source evidence changed")
    if curation.get("production", {}).get("production_eligible") is not False:
        raise ValueError("OASST2 curation crossed production boundary")
    curation_verification = _read_object(
        resolve_binding(root, bindings["curation_verification"], name="curation_verification")
    )
    if curation_verification.get("verification_sha256") != (
        "63557338ac4d27ae043acd2fb671a67b8633fdca8c1448084972ad7566ce9a68"
    ):
        raise ValueError("OASST2 curation verification changed")
    if curation_verification.get("curated") != {
        "accepted_records": EXPECTED_RECORDS,
        "owner_sample_records": 32,
        "planning_target_reached": False,
        "train_supervised_tokens": EXPECTED_TRAIN_TOKENS,
        "validation_supervised_tokens": EXPECTED_VALIDATION_TOKENS,
    }:
        raise ValueError("OASST2 verified candidate counts changed")

    owner = _read_object(
        resolve_binding(root, bindings["owner_review_receipt"], name="owner_review_receipt")
    )
    validate_owner_review_receipt(owner)
    if (
        owner.get("all_sampled_rows_accepted") is not True
        or owner.get("accepted_records") != 32
        or owner.get("rejected_records") != 0
        or owner.get("planning_target_reached") is not False
    ):
        raise ValueError("OASST2 owner-review outcome changed")
    owner_verification = _read_object(
        resolve_binding(
            root,
            bindings["owner_review_verification"],
            name="owner_review_verification",
        )
    )
    if owner_verification.get("verification_sha256") != (
        "ad72545b1214b1df92695f40bb9cde8013fefcaf1e40902a5b16b0e051bc837f"
    ):
        raise ValueError("OASST2 owner-review verification changed")


def _verify_token_metrics(metrics: Mapping[str, Any]) -> None:
    if (
        metrics.get("accepted_records") != EXPECTED_RECORDS
        or metrics.get("assistant_only_loss_verified") is not True
        or metrics.get("supervised_tokens") != EXPECTED_TOTAL_TOKENS
        or metrics.get("context_window") != 4096
        or metrics.get("overlength_rejected") != 2
        or metrics.get("planning_target_reached") is not False
    ):
        raise ValueError("OASST2 token metrics changed")
    if metrics.get("by_split") != {
        "train": {
            "masked_tokens": 263_945,
            "records": EXPECTED_TRAIN_RECORDS,
            "serialized_tokens": 2_485_036,
            "supervised_tokens": EXPECTED_TRAIN_TOKENS,
        },
        "validation": {
            "masked_tokens": 11_017,
            "records": EXPECTED_VALIDATION_RECORDS,
            "serialized_tokens": 107_453,
            "supervised_tokens": EXPECTED_VALIDATION_TOKENS,
        },
    }:
        raise ValueError("OASST2 split token metrics changed")
    if metrics.get("by_task") != {
        "conversation": {
            "masked_tokens": 55_415,
            "records": 1_226,
            "serialized_tokens": 440_970,
            "supervised_tokens": 385_555,
        },
        "factual_qa": {
            "masked_tokens": 165_579,
            "records": 4_190,
            "serialized_tokens": 1_680_741,
            "supervised_tokens": 1_515_162,
        },
        "writing_summarization": {
            "masked_tokens": 53_968,
            "records": 1_010,
            "serialized_tokens": 470_778,
            "supervised_tokens": 416_810,
        },
    }:
        raise ValueError("OASST2 task token metrics changed")
    if metrics.get("task_target_shortfalls") != {
        "conversation": 0,
        "factual_qa": 0,
        "writing_summarization": 716_755,
    }:
        raise ValueError("OASST2 task shortfall evidence changed")


def _verify_records_and_attribution(
    records_path: Path,
    attribution_path: Path,
    *,
    registry: SFTSourceRegistry,
) -> tuple[Counter[str], Counter[str], str]:
    split_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    records: dict[str, tuple[str, str, str, str]] = {}
    with records_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"OASST2 record is not an object: {line_number}")
            record_id = value.get("record_id")
            source = value.get("source")
            split = value.get("split")
            task = value.get("task_category")
            provenance = value.get("provenance")
            if not isinstance(record_id, str) or not record_id or record_id in records:
                raise ValueError("OASST2 record IDs changed")
            if not isinstance(source, Mapping) or not isinstance(provenance, Mapping):
                raise ValueError("OASST2 source/provenance metadata is missing")
            if split not in {"train", "validation"} or not isinstance(task, str):
                raise ValueError("OASST2 split/task changed")
            if value.get("language") != "en":
                raise ValueError("OASST2 language changed")
            if not isinstance(value.get("messages"), list) or len(value["messages"]) < 2:
                raise ValueError("OASST2 messages changed")
            reasons = registry.record_reasons(
                dataset_id=source.get("dataset_id"),
                language=value.get("language"),
                license_id=source.get("license_id"),
                revision=source.get("revision"),
                source_id=source.get("source_id"),
                split=split,
                task_category=task,
            )
            if reasons:
                raise ValueError(f"OASST2 projected registry rejected row: {reasons}")
            content_sha256 = _require_sha256(
                provenance.get("content_sha256"), f"records[{line_number}].content_sha256"
            )
            transform_sha256 = _require_sha256(
                provenance.get("transform_sha256"), f"records[{line_number}].transform_sha256"
            )
            records[record_id] = (str(split), task, content_sha256, transform_sha256)
            split_counts[str(split)] += 1
            task_counts[task] += 1
    if len(records) != EXPECTED_RECORDS or split_counts != Counter(
        {"train": EXPECTED_TRAIN_RECORDS, "validation": EXPECTED_VALIDATION_RECORDS}
    ):
        raise ValueError("OASST2 candidate record inventory changed")
    if task_counts != Counter(
        {"conversation": 1_226, "factual_qa": 4_190, "writing_summarization": 1_010}
    ):
        raise ValueError("OASST2 candidate task inventory changed")

    attribution_ids: set[str] = set()
    with attribution_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"OASST2 attribution is invalid: {line_number}")
            record_id = value.get("record_id")
            if not isinstance(record_id, str) or record_id in attribution_ids:
                raise ValueError("OASST2 attribution IDs changed")
            if record_id not in records:
                raise ValueError("OASST2 attribution contains an unknown record")
            split, task, content_sha256, transform_sha256 = records[record_id]
            if (
                value.get("schema_version") != "lumi.sft-oasst2-attribution.v0.1"
                or value.get("source_id") != SOURCE_ID
                or value.get("dataset_id") != DATASET_ID
                or value.get("revision") != SOURCE_REVISION
                or value.get("license_id") != "apache-2.0"
                or value.get("source_object_sha256") != SOURCE_OBJECT_SHA256
                or value.get("split") != split
                or value.get("task_category") != task
                or value.get("content_sha256") != content_sha256
                or value.get("transform_sha256") != transform_sha256
            ):
                raise ValueError("OASST2 attribution source binding changed")
            message_ids = value.get("message_ids")
            if not isinstance(message_ids, list) or not message_ids:
                raise ValueError("OASST2 attribution message IDs changed")
            attribution_ids.add(record_id)
    if attribution_ids != set(records):
        raise ValueError("OASST2 attribution coverage changed")
    record_ids_sha256 = hashlib.sha256("\n".join(sorted(records)).encode("utf-8")).hexdigest()
    return split_counts, task_counts, record_ids_sha256


def build_promotion_preflight_report(
    plan: Mapping[str, Any],
    *,
    plan_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    bindings = plan["bindings"]
    _verify_bound_evidence(root, bindings)
    contract = _read_object(resolve_binding(root, bindings["data_contract"], name="data_contract"))
    allowed = _policy_sets(contract)
    predecessor_path = resolve_binding(
        root, bindings["predecessor_registry"], name="predecessor_registry"
    )
    predecessor = _read_object(predecessor_path)
    proposed_source = SFTSource.from_mapping(
        plan["source_after"],
        allowed_licenses=allowed["licenses"],
        allowed_languages=allowed["languages"],
        allowed_splits=allowed["splits"],
        allowed_tasks=allowed["tasks"],
    )
    successor = build_projected_successor(predecessor, source_after=plan["source_after"])
    parsed_sources = tuple(
        SFTSource.from_mapping(
            source,
            allowed_licenses=allowed["licenses"],
            allowed_languages=allowed["languages"],
            allowed_splits=allowed["splits"],
            allowed_tasks=allowed["tasks"],
        )
        for source in successor["sources"]
    )
    if parsed_sources[-1] != proposed_source or len(parsed_sources) != 15:
        raise ValueError("OASST2 projected successor changed")
    projected_bytes = _json_file_bytes(successor)
    projected_sha256 = hashlib.sha256(projected_bytes).hexdigest()
    candidate_registry = SFTSourceRegistry(
        observed_at=str(predecessor.get("observed_at")),
        registry_sha256=projected_sha256,
        sources=parsed_sources,
    )
    records_path = resolve_binding(root, bindings["accepted_records"], name="accepted_records")
    attribution_path = resolve_binding(
        root, bindings["attribution_manifest"], name="attribution_manifest"
    )
    split_counts, task_counts, record_ids_sha256 = _verify_records_and_attribution(
        records_path, attribution_path, registry=candidate_registry
    )
    metrics = _read_object(resolve_binding(root, bindings["token_metrics"], name="token_metrics"))
    _verify_token_metrics(metrics)

    report: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "candidate": {
            "accepted_records": EXPECTED_RECORDS,
            "accepted_records_file_sha256": sha256_file(records_path),
            "attribution_manifest_file_sha256": sha256_file(attribution_path),
            "benchmark_matches_rejected": 4,
            "owner_sample_accepted": 32,
            "owner_sample_rejected": 0,
            "planning_target_reached": False,
            "record_ids_sha256": record_ids_sha256,
            "source_revision": SOURCE_REVISION,
            "split_counts": dict(sorted(split_counts.items())),
            "task_counts": dict(sorted(task_counts.items())),
            "train_supervised_tokens": EXPECTED_TRAIN_TOKENS,
            "validation_supervised_tokens": EXPECTED_VALIDATION_TOKENS,
            "writing_token_shortfall": 716_755,
        },
        "checks": {
            "all_6426_records_pass_projected_registry": True,
            "apache_2_0_attribution_bound": True,
            "attribution_manifest_covers_all_records": True,
            "candidate_and_evidence_hashes_verified": True,
            "candidate_registry_change_executed": False,
            "curation_and_owner_review_verified": True,
            "planning_shortfall_preserved": True,
            "predecessor_registry_unchanged": True,
            "production_dataset_promotions": 0,
            "projected_successor_adds_only_oasst2": True,
            "raw_training_text_retained_in_report": False,
        },
        "next_gate": {
            "action": "separately_authorize_exact_candidate_registry_v0_13_materialization",
            "authorization_required": True,
            "post_promotion_action": (
                "full_local_candidate_mix_reaudit_then_remaining_mix_closure_gate"
            ),
        },
        "observed_at": plan["observed_at"],
        "plan_file_sha256": plan_file_sha256,
        "production_eligible": False,
        "projected_successor": {
            "added_source_id": SOURCE_ID,
            "bytes": len(projected_bytes),
            "path": plan["output"]["projected_successor_registry_path"],
            "predecessor_file_sha256": sha256_file(predecessor_path),
            "predecessor_source_count": EXPECTED_PREDECESSOR_SOURCES,
            "sha256": projected_sha256,
            "source_count": 15,
        },
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "verified_local_preflight_registry_v0_13_successor_not_materialized",
    }
    report["report_sha256"] = sha256_value(report)
    validate_report(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported OASST2 source-promotion preflight report")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if report.get("report_sha256") != sha256_value(unsigned):
        raise ValueError("OASST2 source-promotion preflight report seal mismatch")
    if report.get("production_eligible") is not False:
        raise ValueError("OASST2 preflight crossed production boundary")
    checks = report.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("OASST2 source-promotion preflight checks are missing")
    if checks.get("candidate_registry_change_executed") is not False:
        raise ValueError("OASST2 preflight changed the source registry")
    if checks.get("production_dataset_promotions") != 0:
        raise ValueError("OASST2 preflight promoted production data")
    if checks.get("raw_training_text_retained_in_report") is not False:
        raise ValueError("OASST2 preflight retained raw training text")
    projected = report.get("projected_successor")
    if not isinstance(projected, Mapping) or projected.get("source_count") != 15:
        raise ValueError("OASST2 projected registry count changed")


__all__ = [
    "PLAN_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "VERIFICATION_SCHEMA_VERSION",
    "build_projected_successor",
    "build_promotion_preflight_report",
    "load_promotion_preflight_plan",
    "validate_report",
]
