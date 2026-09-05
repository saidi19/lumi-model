"""Fail-closed source-registry preflight for reviewed OpenCodeInstruct data."""

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
from .posttraining_bigcodebench_signatures import sha256_value
from .posttraining_sources import SFTSource, SFTSourceRegistry

PLAN_SCHEMA_VERSION = "lumi.sft-opencodeinstruct-source-promotion-preflight.v0.1"
REPORT_SCHEMA_VERSION = (
    "lumi.sft-opencodeinstruct-source-promotion-preflight-report.v0.1"
)
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-opencodeinstruct-source-promotion-preflight-verification.v0.1"
)
PLAN_STATUS = "local_preflight_only_registry_successor_not_materialized"
OWNER_REQUEST_TEXT = "next step shuru kor"

SOURCE_ID = "nvidia_opencodeinstruct"
SOURCE_REVISION = "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d"
SOURCE_OBJECT_SHA256 = (
    "2839de0d6c4d1691710dbc63f43cd816e7c02a943f8e2c07087bbf5fa26172ce"
)
EXPECTED_CANDIDATE_SHA256 = (
    "9be50e3b6078fdc50d3b887e7850ffc4e57f9a634f72fb928b62908da854446e"
)
EXPECTED_ATTRIBUTION_SHA256 = (
    "cbd79a40015cb38597fb16712cca7d251cd97ea392622a15000da631472a23d0"
)
EXPECTED_RECORDS = 11_100
EXPECTED_TRAIN_RECORDS = 10_553
EXPECTED_TRAIN_TOKENS = 2_100_077
EXPECTED_VALIDATION_RECORDS = 547
EXPECTED_VALIDATION_TOKENS = 110_619
EXPECTED_TOTAL_TOKENS = 2_210_696
EXPECTED_PREDECESSOR_SOURCES = 13

EXPECTED_AUTHORIZATION = {
    "candidate_registry_change_authorized": False,
    "data_acquisition_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "purpose": "reviewed_opencodeinstruct_registry_v0_11_successor_projection_only",
    "source_registry_promotion_authorized": False,
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
    "bigcodebench_reaudit_report",
    "bigcodebench_reaudit_verification",
    "curation_report",
    "curation_verification",
    "data_contract",
    "implementation",
    "livecodebench_reaudit_report",
    "livecodebench_reaudit_verification",
    "owner_review_receipt",
    "owner_review_verification",
    "predecessor_registry",
    "runner",
    "token_metrics",
    "verifier",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ATTRIBUTION_HASH_FIELDS = (
    "code_sha256",
    "content_sha256",
    "output_sha256",
    "prompt_sha256",
    "published_status_sha256",
    "tests_sha256",
    "transform_sha256",
)


def _read_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


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
        raise ValueError(f"invalid OpenCodeInstruct promotion-preflight binding: {name}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.parent.resolve()) or not resolved.is_file():
        raise ValueError(f"OpenCodeInstruct promotion-preflight binding missing: {name}")
    if sha256_file(resolved) != expected:
        raise ValueError(f"OpenCodeInstruct promotion-preflight binding changed: {name}")
    return resolved


def _policy_sets(contract: Mapping[str, Any]) -> dict[str, frozenset[str]]:
    task_mix = contract.get("task_mix")
    if not isinstance(task_mix, Mapping):
        raise ValueError("OpenCodeInstruct data-contract task mix changed")
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
        raise ValueError("OpenCodeInstruct data-contract allowlists changed")
    return {name: frozenset(items) for name, items in values.items()}


def load_promotion_preflight_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = _read_object(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported OpenCodeInstruct source-promotion preflight")
    if plan.get("status") != PLAN_STATUS:
        raise ValueError("OpenCodeInstruct source-promotion preflight status changed")
    if plan.get("observed_at") != "2026-08-07":
        raise ValueError("OpenCodeInstruct source-promotion preflight date changed")
    if plan.get("owner_request_text") != OWNER_REQUEST_TEXT:
        raise ValueError("OpenCodeInstruct source-promotion owner request changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("OpenCodeInstruct preflight authorization boundary changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("OpenCodeInstruct promotion-preflight limits changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("OpenCodeInstruct promotion-preflight bindings changed")
    for name, specification in bindings.items():
        if not isinstance(specification, Mapping):
            raise ValueError(f"invalid OpenCodeInstruct promotion binding: {name}")
        resolve_binding(root, specification, name=name)
    if bindings["accepted_records"]["sha256"] != EXPECTED_CANDIDATE_SHA256:
        raise ValueError("OpenCodeInstruct candidate identity changed")
    if bindings["attribution_manifest"]["sha256"] != EXPECTED_ATTRIBUTION_SHA256:
        raise ValueError("OpenCodeInstruct attribution identity changed")
    source = plan.get("source_after")
    if not isinstance(source, Mapping):
        raise ValueError("OpenCodeInstruct proposed source is missing")
    if source.get("source_id") != SOURCE_ID or source.get("revision") != SOURCE_REVISION:
        raise ValueError("OpenCodeInstruct proposed source identity changed")
    if plan.get("output") != {
        "preflight_report_path": (
            "evidence/lumi-4b-sft-opencodeinstruct-source-promotion-preflight-v0.1.json"
        ),
        "projected_successor_registry_path": (
            "configs/lumi-4b-sft-source-registry-v0.11.json"
        ),
        "verification_path": (
            "evidence/"
            "lumi-4b-sft-opencodeinstruct-source-promotion-preflight-verification-v0.1.json"
        ),
    }:
        raise ValueError("OpenCodeInstruct promotion-preflight outputs changed")
    return plan


def build_projected_successor(
    predecessor: Mapping[str, Any],
    *,
    source_after: Mapping[str, Any],
) -> dict[str, Any]:
    sources = predecessor.get("sources")
    if not isinstance(sources, list) or len(sources) != EXPECTED_PREDECESSOR_SOURCES:
        raise ValueError("OpenCodeInstruct predecessor source count changed")
    if any(
        isinstance(source, Mapping) and source.get("source_id") == SOURCE_ID
        for source in sources
    ):
        raise ValueError("OpenCodeInstruct source already exists in predecessor registry")
    successor = copy.deepcopy(dict(predecessor))
    successor["sources"].append(copy.deepcopy(dict(source_after)))
    return successor


def _verify_bound_evidence(root: Path, bindings: Mapping[str, Any]) -> None:
    curation = _read_object(
        resolve_binding(root, bindings["curation_report"], name="curation_report")
    )
    if curation.get("report_sha256") != (
        "b2dcd796b3c2110394f2544ff452711b3aacbe16e00cc9f14bacb03826084b16"
    ):
        raise ValueError("OpenCodeInstruct curation report seal changed")
    source = curation.get("source", {})
    if (
        source.get("source_id") != SOURCE_ID
        or source.get("revision") != SOURCE_REVISION
        or source.get("license_id") != "cc-by-4.0"
        or source.get("object_sha256") != SOURCE_OBJECT_SHA256
    ):
        raise ValueError("OpenCodeInstruct curation source evidence changed")
    curation_verification = _read_object(
        resolve_binding(
            root,
            bindings["curation_verification"],
            name="curation_verification",
        )
    )
    if curation_verification.get("verification_sha256") != (
        "aa9281dcbac5502647d04b33e20b2e76014048b41dd6da5f0aa33ee3dab2be58"
    ):
        raise ValueError("OpenCodeInstruct curation verification changed")
    owner = _read_object(
        resolve_binding(root, bindings["owner_review_receipt"], name="owner_review_receipt")
    )
    unsigned_owner = {key: value for key, value in owner.items() if key != "receipt_sha256"}
    if owner.get("receipt_sha256") != sha256_value(unsigned_owner):
        raise ValueError("OpenCodeInstruct owner-review receipt seal changed")
    if (
        owner.get("all_sampled_rows_accepted") is not True
        or owner.get("accepted_records") != 32
        or owner.get("rejected_records") != 0
        or owner.get("candidate")
        != {
            "records": EXPECTED_RECORDS,
            "train_records": EXPECTED_TRAIN_RECORDS,
            "train_supervised_tokens": EXPECTED_TRAIN_TOKENS,
            "validation_records": EXPECTED_VALIDATION_RECORDS,
            "validation_supervised_tokens": EXPECTED_VALIDATION_TOKENS,
        }
    ):
        raise ValueError("OpenCodeInstruct owner-review outcome changed")
    owner_verification = _read_object(
        resolve_binding(
            root,
            bindings["owner_review_verification"],
            name="owner_review_verification",
        )
    )
    if owner_verification.get("verification_sha256") != (
        "579a86a729b38a72fbd94148007c021802bc81ae4914884152d0e3da98ab1ada"
    ):
        raise ValueError("OpenCodeInstruct owner-review verification changed")
    for report_name, verification_name, report_seal, verification_seal in (
        (
            "bigcodebench_reaudit_report",
            "bigcodebench_reaudit_verification",
            "71b98dd2fb7127a4bf3a781f8fc8e12c31e5aa514eae0dd92a4ec42694b6ef13",
            "29bf1a08d6ee74f9719c6a0d396742a189a8df90ae7d80ddacbfe7c7491f3a12",
        ),
        (
            "livecodebench_reaudit_report",
            "livecodebench_reaudit_verification",
            "b0ca7612b6737e4db5c4dad7cc585b77bb3a5951aa0a4608c8810d1f23af8768",
            "9528ea47bd843b598c1463a3f3febd87558dd040521eac80354d3b63b9ccb180",
        ),
    ):
        report = _read_object(resolve_binding(root, bindings[report_name], name=report_name))
        verification = _read_object(
            resolve_binding(root, bindings[verification_name], name=verification_name)
        )
        if report.get("report_sha256") != report_seal:
            raise ValueError(f"OpenCodeInstruct benchmark report changed: {report_name}")
        if report.get("results", {}).get("matched_records") != 0:
            raise ValueError(f"OpenCodeInstruct benchmark matches changed: {report_name}")
        if verification.get("verification_sha256") != verification_seal:
            raise ValueError(
                f"OpenCodeInstruct benchmark verification changed: {verification_name}"
            )


def _verify_token_metrics(metrics: Mapping[str, Any]) -> None:
    if (
        metrics.get("accepted_records") != EXPECTED_RECORDS
        or metrics.get("assistant_only_loss_verified") is not True
        or metrics.get("supervised_tokens") != EXPECTED_TOTAL_TOKENS
        or metrics.get("overlength_rejected") != 0
        or metrics.get("context_window") != 4096
    ):
        raise ValueError("OpenCodeInstruct token metrics changed")
    if metrics.get("by_split") != {
        "train": {
            "masked_tokens": 1_752_360,
            "records": EXPECTED_TRAIN_RECORDS,
            "serialized_tokens": 3_852_437,
            "supervised_tokens": EXPECTED_TRAIN_TOKENS,
        },
        "validation": {
            "masked_tokens": 90_624,
            "records": EXPECTED_VALIDATION_RECORDS,
            "serialized_tokens": 201_243,
            "supervised_tokens": EXPECTED_VALIDATION_TOKENS,
        },
    }:
        raise ValueError("OpenCodeInstruct split token metrics changed")


def _verify_records_and_attribution(
    records_path: Path,
    attribution_path: Path,
    *,
    registry: SFTSourceRegistry,
) -> tuple[Counter[str], str]:
    split_counts: Counter[str] = Counter()
    record_ids: set[str] = set()
    with records_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"OpenCodeInstruct record is not an object: {line_number}")
            record_id = value.get("record_id")
            source = value.get("source")
            split = value.get("split")
            if not isinstance(record_id, str) or not record_id or record_id in record_ids:
                raise ValueError("OpenCodeInstruct record IDs changed")
            if not isinstance(source, Mapping):
                raise ValueError("OpenCodeInstruct source metadata is missing")
            if value.get("language") != "en" or value.get("task_category") != "coding":
                raise ValueError("OpenCodeInstruct language/task declaration changed")
            if split not in {"train", "validation"}:
                raise ValueError("OpenCodeInstruct split changed")
            if not isinstance(value.get("messages"), list) or not value["messages"]:
                raise ValueError("OpenCodeInstruct messages changed")
            reasons = registry.record_reasons(
                dataset_id=source.get("dataset_id"),
                language=value.get("language"),
                license_id=source.get("license_id"),
                revision=source.get("revision"),
                source_id=source.get("source_id"),
                split=split,
                task_category=value.get("task_category"),
            )
            if reasons:
                raise ValueError(f"OpenCodeInstruct projected registry rejected row: {reasons}")
            record_ids.add(record_id)
            split_counts[str(split)] += 1
    if len(record_ids) != EXPECTED_RECORDS or split_counts != Counter(
        {"train": EXPECTED_TRAIN_RECORDS, "validation": EXPECTED_VALIDATION_RECORDS}
    ):
        raise ValueError("OpenCodeInstruct candidate record inventory changed")

    attribution_ids: set[str] = set()
    with attribution_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"OpenCodeInstruct attribution is invalid: {line_number}")
            record_id = value.get("record_id")
            if not isinstance(record_id, str) or record_id in attribution_ids:
                raise ValueError("OpenCodeInstruct attribution IDs changed")
            if (
                value.get("source_revision") != SOURCE_REVISION
                or value.get("source_object_sha256") != SOURCE_OBJECT_SHA256
                or value.get("split") not in {"train", "validation"}
            ):
                raise ValueError("OpenCodeInstruct attribution source binding changed")
            for field in _ATTRIBUTION_HASH_FIELDS:
                _require_sha256(value.get(field), f"attribution[{line_number}].{field}")
            attribution_ids.add(record_id)
    if attribution_ids != record_ids:
        raise ValueError("OpenCodeInstruct attribution coverage changed")
    record_ids_sha256 = hashlib.sha256(
        "\n".join(sorted(record_ids)).encode("utf-8")
    ).hexdigest()
    return split_counts, record_ids_sha256


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
        root,
        bindings["predecessor_registry"],
        name="predecessor_registry",
    )
    predecessor = _read_object(predecessor_path)
    proposed_source = SFTSource.from_mapping(
        plan["source_after"],
        allowed_licenses=allowed["licenses"],
        allowed_languages=allowed["languages"],
        allowed_splits=allowed["splits"],
        allowed_tasks=allowed["tasks"],
    )
    successor = build_projected_successor(
        predecessor,
        source_after=plan["source_after"],
    )
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
    if parsed_sources[-1] != proposed_source or len(parsed_sources) != 14:
        raise ValueError("OpenCodeInstruct projected successor changed")
    projected_bytes = _json_file_bytes(successor)
    projected_sha256 = hashlib.sha256(projected_bytes).hexdigest()
    candidate_registry = SFTSourceRegistry(
        observed_at=str(predecessor.get("observed_at")),
        registry_sha256=projected_sha256,
        sources=parsed_sources,
    )
    records_path = resolve_binding(root, bindings["accepted_records"], name="accepted_records")
    attribution_path = resolve_binding(
        root,
        bindings["attribution_manifest"],
        name="attribution_manifest",
    )
    split_counts, record_ids_sha256 = _verify_records_and_attribution(
        records_path,
        attribution_path,
        registry=candidate_registry,
    )
    token_metrics = _read_object(
        resolve_binding(root, bindings["token_metrics"], name="token_metrics")
    )
    _verify_token_metrics(token_metrics)

    report: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "candidate": {
            "accepted_records": EXPECTED_RECORDS,
            "accepted_records_file_sha256": sha256_file(records_path),
            "attribution_manifest_file_sha256": sha256_file(attribution_path),
            "benchmark_matches": 0,
            "owner_sample_accepted": 32,
            "owner_sample_rejected": 0,
            "record_ids_sha256": record_ids_sha256,
            "source_revision": SOURCE_REVISION,
            "split_counts": dict(sorted(split_counts.items())),
            "train_supervised_tokens": EXPECTED_TRAIN_TOKENS,
            "validation_supervised_tokens": EXPECTED_VALIDATION_TOKENS,
        },
        "checks": {
            "all_11100_records_pass_projected_registry": True,
            "attribution_manifest_covers_all_records": True,
            "benchmark_reaudits_passed_zero_matches": True,
            "candidate_and_owner_receipt_hashes_verified": True,
            "candidate_registry_change_executed": False,
            "cc_by_4_0_attribution_bound": True,
            "predecessor_registry_unchanged": True,
            "production_dataset_promotions": 0,
            "projected_successor_adds_only_opencodeinstruct": True,
            "raw_training_text_retained_in_report": False,
        },
        "next_gate": {
            "action": "separately_authorize_exact_candidate_registry_v0_11_materialization",
            "authorization_required": True,
            "post_promotion_action": "full_local_candidate_mix_reaudit_then_dataset_seal_gate",
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
            "source_count": 14,
        },
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "verified_local_preflight_registry_successor_not_materialized",
    }
    report["report_sha256"] = sha256_value(report)
    validate_report(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported OpenCodeInstruct promotion-preflight report")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if report.get("report_sha256") != sha256_value(unsigned):
        raise ValueError("OpenCodeInstruct promotion-preflight report seal mismatch")
    if report.get("production_eligible") is not False:
        raise ValueError("OpenCodeInstruct preflight crossed production boundary")
    checks = report.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("OpenCodeInstruct promotion-preflight checks are missing")
    if checks.get("candidate_registry_change_executed") is not False:
        raise ValueError("OpenCodeInstruct preflight changed the source registry")
    if checks.get("production_dataset_promotions") != 0:
        raise ValueError("OpenCodeInstruct preflight promoted production data")
    if checks.get("raw_training_text_retained_in_report") is not False:
        raise ValueError("OpenCodeInstruct preflight retained raw training text")
    projected = report.get("projected_successor")
    if not isinstance(projected, Mapping) or projected.get("source_count") != 14:
        raise ValueError("OpenCodeInstruct projected registry count changed")


__all__ = [
    "PLAN_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "VERIFICATION_SCHEMA_VERSION",
    "build_projected_successor",
    "build_promotion_preflight_report",
    "load_promotion_preflight_plan",
    "validate_report",
]
