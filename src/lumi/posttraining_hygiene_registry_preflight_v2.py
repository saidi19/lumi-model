"""Project the two-source immutable registry-v0.14 successor without writing it."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .posttraining_ai_v3_provenance_materialization import _verify_internal_sha
from .posttraining_candidate_mix_hygiene_materialization_v2 import (
    validate_hygiene_materialization_v2_manifest,
    validate_hygiene_materialization_v2_receipt,
)
from .posttraining_candidate_mix_preflight import (
    _load_json,
    file_sha256,
    sha256_value,
)
from .posttraining_candidate_mix_reaudit import resolve_binding
from .posttraining_data import load_sft_policy
from .posttraining_sources import SFTSource

PLAN_SCHEMA_VERSION = "lumi.sft-hygiene-registry-successor-preflight-plan.v0.2"
REPORT_SCHEMA_VERSION = "lumi.sft-hygiene-registry-successor-preflight-report.v0.2"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-hygiene-registry-successor-preflight-verification.v0.2"

EXPECTED_AUTHORIZATION = {
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "purpose": "project_exact_two_source_registry_v0_14_successor_only",
    "source_registry_mutation_authorized": False,
    "source_registry_promotion_authorized": False,
    "unrelated_access_authorized": False,
}
EXPECTED_LIMITS = {
    "changed_sources": 2,
    "maximum_local_candidate_records": 6_918,
    "maximum_network_bytes": 0,
    "maximum_report_bytes": 1_048_576,
    "maximum_runtime_minutes": 30,
    "predecessor_sources": 15,
}
EXPECTED_OUTPUT = {
    "preflight_report_path": ("evidence/lumi-4b-sft-hygiene-registry-v0.14-preflight-v0.1.json"),
    "projected_successor_materialized": False,
    "projected_successor_registry_path": ("configs/lumi-4b-sft-source-registry-v0.14.json"),
    "raw_training_text_in_outputs": False,
    "verification_path": (
        "evidence/lumi-4b-sft-hygiene-registry-v0.14-preflight-verification-v0.1.json"
    ),
}
EXPECTED_BINDINGS = frozenset(
    {
        "data_contract",
        "hygiene_materialization_implementation",
        "hygiene_materialization_manifest",
        "hygiene_materialization_receipt",
        "hygiene_materialization_verification",
        "implementation",
        "oasst1_clean_candidate",
        "oasst2_clean_candidate",
        "predecessor_registry",
        "runner",
        "source_registry_implementation",
        "test",
        "verifier",
    }
)
EXPECTED_PREDECESSOR_SHA256 = "1e8f7b0d2f1bfc10118fd76a51813e942cc7d0ce558f73ebfdc0382e6ee2b077"
EXPECTED_MATERIALIZATION = {
    "manifest_file_sha256": ("ba4dcd6302678d5deb2029514db667ed7563a0ebfb0921e475e3d363cf9d948d"),
    "manifest_internal_sha256": (
        "fec0e85da3b187de02fea3675d0859ec4028b15ebeda1b1981f9e5017c7d1f55"
    ),
    "receipt_file_sha256": ("600c0878f8400ce7ce166e7bbbfb4cdcb7a17252b195377470625236fae95062"),
    "receipt_internal_sha256": ("5082f5d04a9eeea861ea9350cbe8dc1bb6612926fdce517252893fd915884ce4"),
    "verification_file_sha256": (
        "3cd8e9642894f7657164df82a200119115960745d2d6786c9276fbb32de709af"
    ),
    "verification_internal_sha256": (
        "7c899c4c0157dc46dc3f677440331bdc233feae9f626432e2bbe1be1ccc080bb"
    ),
}
EXPECTED_SUCCESSOR = {
    "bytes": 17_657,
    "sha256": "a7310e469abb06cf4fbd12a72b2dcf2fbbebeac064e7b9182b994a9a77254e16",
    "source_count": 15,
}
SOURCE_UPDATES = {
    "openassistant_oasst1": {
        "adapter_after": (
            "oasst1_root_direct_rank0_quality_decontaminated_cross_source_deduplicated_v0.5"
        ),
        "adapter_before": "oasst1_root_direct_rank0_quality_decontaminated_v0.4",
        "candidate": {
            "records": 1_720,
            "sha256": ("6a0828f05bcf737921e9388d00705e7dd0a07cf447ba6b35fec57124a29bc369"),
            "train_supervised_tokens": 447_746,
            "validation_supervised_tokens": 23_475,
        },
        "production_blockers_after": [
            "candidate_mix_is_only_1720_records_471221_supervised_tokens",
            "production_dataset_seal_and_sft_authorization_not_complete",
        ],
        "production_blockers_before": [
            "candidate_mix_is_only_1742_records_476165_supervised_tokens",
            "full_mix_cross_source_deduplication_not_complete",
            "production_dataset_seal_and_sft_authorization_not_complete",
        ],
    },
    "openassistant_oasst2": {
        "adapter_after": (
            "oasst2_ready_tree_ranked_paths_quality_decontaminated_cross_source_deduplicated_v0.2"
        ),
        "adapter_before": "oasst2_ready_tree_ranked_paths_quality_decontaminated_v0.1",
        "candidate": {
            "records": 5_198,
            "sha256": ("57ba5ea18e2f621b9bcc79d18340df92ba7bf3a0e378d3273498d5e497870459"),
            "train_supervised_tokens": 1_895_923,
            "validation_supervised_tokens": 86_526,
        },
        "production_blockers_after": [
            (
                "candidate_is_5198_records_1982449_supervised_tokens_"
                "and_short_of_2600000_train_token_target"
            ),
            "owner_quality_review_is_32_row_deterministic_sample_not_full_record_review",
            "writing_summarization_target_shortfall_is_805793_supervised_tokens",
            "production_dataset_seal_and_sft_authorization_not_complete",
        ],
        "production_blockers_before": [
            (
                "candidate_is_6426_records_2317527_supervised_tokens_"
                "and_short_of_2600000_train_token_target"
            ),
            "owner_quality_review_is_32_row_deterministic_sample_not_full_record_review",
            "writing_summarization_target_shortfall_is_716755_supervised_tokens",
            "full_mix_cross_source_deduplication_not_complete",
            "production_dataset_seal_and_sft_authorization_not_complete",
        ],
    },
}


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_hygiene_registry_v14_preflight_plan(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    plan = _load_json(Path(path))
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "changed_source_ids": sorted(SOURCE_UPDATES),
        "limits": EXPECTED_LIMITS,
        "observed_at": "2026-08-08",
        "output": EXPECTED_OUTPUT,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "authorized_cpu_only_non_mutating_registry_v0_14_preflight",
    }
    for field, expected_value in expected.items():
        if plan.get(field) != expected_value:
            raise ValueError(f"hygiene registry v0.14 preflight changed: {field}")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("hygiene registry v0.14 binding set changed")
    for name, binding in bindings.items():
        resolve_binding(repository_root, binding, name=name)
    exact = {
        "hygiene_materialization_manifest": EXPECTED_MATERIALIZATION["manifest_file_sha256"],
        "hygiene_materialization_receipt": EXPECTED_MATERIALIZATION["receipt_file_sha256"],
        "hygiene_materialization_verification": EXPECTED_MATERIALIZATION[
            "verification_file_sha256"
        ],
        "oasst1_clean_candidate": SOURCE_UPDATES["openassistant_oasst1"]["candidate"]["sha256"],
        "oasst2_clean_candidate": SOURCE_UPDATES["openassistant_oasst2"]["candidate"]["sha256"],
        "predecessor_registry": EXPECTED_PREDECESSOR_SHA256,
    }
    for name, expected_sha256 in exact.items():
        if bindings[name]["sha256"] != expected_sha256:
            raise ValueError(f"hygiene registry v0.14 binding changed: {name}")
    return plan


def _verify_materialization_evidence(*, root: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    resolved = {
        name: resolve_binding(root, binding, name=name)
        for name, binding in plan["bindings"].items()
    }
    manifest = _load_json(resolved["hygiene_materialization_manifest"])
    receipt = _load_json(resolved["hygiene_materialization_receipt"])
    verification = _load_json(resolved["hygiene_materialization_verification"])
    validate_hygiene_materialization_v2_manifest(manifest)
    validate_hygiene_materialization_v2_receipt(receipt)
    _verify_internal_sha(verification, "verification_sha256")
    if manifest.get("manifest_sha256") != EXPECTED_MATERIALIZATION["manifest_internal_sha256"]:
        raise ValueError("registry v0.14 materialization manifest changed")
    if receipt.get("receipt_sha256") != EXPECTED_MATERIALIZATION["receipt_internal_sha256"]:
        raise ValueError("registry v0.14 materialization receipt changed")
    if (
        verification.get("verification_sha256")
        != EXPECTED_MATERIALIZATION["verification_internal_sha256"]
    ):
        raise ValueError("registry v0.14 materialization verification changed")
    if receipt.get("manifest_file_sha256") != EXPECTED_MATERIALIZATION["manifest_file_sha256"]:
        raise ValueError("registry v0.14 manifest-receipt link changed")
    if verification.get("receipt_file_sha256") != EXPECTED_MATERIALIZATION["receipt_file_sha256"]:
        raise ValueError("registry v0.14 receipt-verification link changed")
    total_records = 0
    for source_id, update in SOURCE_UPDATES.items():
        output = receipt.get("outputs", {}).get(source_id, {})
        binding_name = (
            "oasst1_clean_candidate"
            if source_id == "openassistant_oasst1"
            else "oasst2_clean_candidate"
        )
        candidate_path = resolved[binding_name]
        candidate = update["candidate"]
        if (
            output.get("sha256") != candidate["sha256"]
            or output.get("records") != candidate["records"]
            or file_sha256(candidate_path) != candidate["sha256"]
        ):
            raise ValueError(f"registry v0.14 clean candidate changed: {source_id}")
        if output.get("by_split") != {
            "train": {
                "records": 1_627 if source_id == "openassistant_oasst1" else 4_944,
                "supervised_tokens": candidate["train_supervised_tokens"],
            },
            "validation": {
                "records": 93 if source_id == "openassistant_oasst1" else 254,
                "supervised_tokens": candidate["validation_supervised_tokens"],
            },
        }:
            raise ValueError(f"registry v0.14 split metrics changed: {source_id}")
        total_records += candidate["records"]
    if total_records != EXPECTED_LIMITS["maximum_local_candidate_records"]:
        raise ValueError("registry v0.14 candidate record total changed")
    return receipt


def build_projected_registry_v14(
    predecessor: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_sources = predecessor.get("sources")
    if not isinstance(raw_sources, list) or len(raw_sources) != 15:
        raise ValueError("registry v0.14 predecessor source count changed")
    successor = copy.deepcopy(dict(predecessor))
    changed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in successor["sources"]:
        source_id = source.get("source_id")
        if source_id not in SOURCE_UPDATES:
            continue
        update = SOURCE_UPDATES[source_id]
        if source.get("adapter") != update["adapter_before"]:
            raise ValueError(f"registry v0.14 predecessor adapter changed: {source_id}")
        if source.get("production_blockers") != update["production_blockers_before"]:
            raise ValueError(f"registry v0.14 predecessor blockers changed: {source_id}")
        before_sha256 = sha256_value(source)
        source["adapter"] = update["adapter_after"]
        source["production_blockers"] = list(update["production_blockers_after"])
        changed.append(
            {
                "adapter_after": update["adapter_after"],
                "adapter_before": update["adapter_before"],
                "candidate": dict(update["candidate"]),
                "production_blockers_after": list(update["production_blockers_after"]),
                "production_blockers_before": list(update["production_blockers_before"]),
                "source_after_sha256": sha256_value(source),
                "source_before_sha256": before_sha256,
                "source_id": source_id,
            }
        )
        seen.add(source_id)
    if seen != set(SOURCE_UPDATES) or len(changed) != 2:
        raise ValueError("registry v0.14 did not change exactly two sources")
    changed.sort(key=lambda item: item["source_id"])
    return successor, changed


def _validate_successor_sources(successor: Mapping[str, Any], *, data_contract_path: Path) -> None:
    policy = load_sft_policy(data_contract_path)
    sources = successor.get("sources")
    if not isinstance(sources, list) or len(sources) != 15:
        raise ValueError("registry v0.14 successor source count changed")
    for source in sources:
        SFTSource.from_mapping(
            source,
            allowed_licenses=policy.allowed_licenses,
            allowed_languages=policy.allowed_languages,
            allowed_splits=policy.allowed_splits,
            allowed_tasks=frozenset(policy.task_mix),
        )


def rebuild_hygiene_registry_v14_preflight(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    receipt = _verify_materialization_evidence(root=root, plan=plan)
    predecessor_path = resolve_binding(
        root, plan["bindings"]["predecessor_registry"], name="predecessor_registry"
    )
    data_contract_path = resolve_binding(
        root, plan["bindings"]["data_contract"], name="data_contract"
    )
    predecessor = _load_json(predecessor_path)
    successor, changed = build_projected_registry_v14(predecessor)
    _validate_successor_sources(successor, data_contract_path=data_contract_path)
    successor_bytes = json_file_bytes(successor)
    successor_summary = {
        "bytes": len(successor_bytes),
        "materialized": False,
        "path": plan["output"]["projected_successor_registry_path"],
        "sha256": _bytes_sha256(successor_bytes),
        "source_count": len(successor["sources"]),
    }
    if {
        key: successor_summary[key] for key in ("bytes", "sha256", "source_count")
    } != EXPECTED_SUCCESSOR:
        raise ValueError("registry v0.14 projected successor changed")
    predecessor_unchanged = copy.deepcopy(predecessor)
    successor_unchanged = copy.deepcopy(successor)
    predecessor_unchanged["sources"] = [
        item for item in predecessor_unchanged["sources"] if item["source_id"] not in SOURCE_UPDATES
    ]
    successor_unchanged["sources"] = [
        item for item in successor_unchanged["sources"] if item["source_id"] not in SOURCE_UPDATES
    ]
    if predecessor_unchanged != successor_unchanged:
        raise ValueError("registry v0.14 projection changed unrelated metadata")
    report: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "changed_sources": changed,
        "changed_sources_count": len(changed),
        "materialization_binding": {
            "manifest_file_sha256": EXPECTED_MATERIALIZATION["manifest_file_sha256"],
            "manifest_internal_sha256": EXPECTED_MATERIALIZATION["manifest_internal_sha256"],
            "receipt_file_sha256": EXPECTED_MATERIALIZATION["receipt_file_sha256"],
            "receipt_internal_sha256": receipt["receipt_sha256"],
            "verification_file_sha256": EXPECTED_MATERIALIZATION["verification_file_sha256"],
            "verification_internal_sha256": EXPECTED_MATERIALIZATION[
                "verification_internal_sha256"
            ],
        },
        "observed_at": plan["observed_at"],
        "plan_file_sha256": plan_file_sha256,
        "predecessor_registry": {
            "path": plan["bindings"]["predecessor_registry"]["path"],
            "sha256": EXPECTED_PREDECESSOR_SHA256,
            "source_count": len(predecessor["sources"]),
        },
        "production_eligible": False,
        "projected_successor": successor_summary,
        "raw_training_text_retained": False,
        "registry_files_materialized": 0,
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "verified_two_source_registry_v0_14_projection_not_materialized",
        "unchanged_registry_metadata_sha256": sha256_value(predecessor_unchanged),
    }
    report["report_sha256"] = sha256_value(report)
    validate_hygiene_registry_v14_preflight_report(report)
    if len(json_file_bytes(report)) > plan["limits"]["maximum_report_bytes"]:
        raise ValueError("registry v0.14 preflight report ceiling exceeded")
    return report


def validate_hygiene_registry_v14_preflight_report(
    report: Mapping[str, Any],
) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported registry v0.14 preflight report schema")
    _verify_internal_sha(report, "report_sha256")
    if report.get("changed_sources_count") != 2:
        raise ValueError("registry v0.14 changed-source count changed")
    if report.get("registry_files_materialized") != 0:
        raise ValueError("registry v0.14 preflight materialized a registry")
    if report.get("production_eligible") is not False:
        raise ValueError("registry v0.14 preflight cannot be production eligible")
    if report.get("raw_training_text_retained") is not False:
        raise ValueError("registry v0.14 preflight must remain raw-text-free")
    projected = report.get("projected_successor", {})
    if projected.get("materialized") is not False:
        raise ValueError("registry v0.14 projected successor was materialized")
    if {
        key: projected.get(key) for key in ("bytes", "sha256", "source_count")
    } != EXPECTED_SUCCESSOR:
        raise ValueError("registry v0.14 successor summary changed")
    if [item.get("source_id") for item in report.get("changed_sources", [])] != (
        sorted(SOURCE_UPDATES)
    ):
        raise ValueError("registry v0.14 changed-source identities changed")
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    for forbidden in ('"messages"', '"content"', '"assistant_response"'):
        if forbidden in serialized:
            raise ValueError("registry v0.14 report contains raw training text")


def build_hygiene_registry_v14_preflight_verification(
    *,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    report_file_sha256: str,
    replayed_report: Mapping[str, Any],
) -> dict[str, Any]:
    validate_hygiene_registry_v14_preflight_report(report)
    validate_hygiene_registry_v14_preflight_report(replayed_report)
    if report != replayed_report:
        raise ValueError("registry v0.14 independent replay changed")
    verification: dict[str, Any] = {
        "checks": {
            "exact_two_source_metadata_updates_replayed": True,
            "materialized_clean_candidate_hashes_counts_and_splits_verified": True,
            "no_network_api_gpu_registry_mutation_seal_or_sft": True,
            "predecessor_registry_v0_13_remains_immutable": True,
            "projected_registry_v0_14_not_materialized": True,
            "projected_successor_hash_reproduced": True,
            "raw_training_text_absent_from_repo_evidence": True,
            "unrelated_thirteen_sources_and_top_level_metadata_unchanged": True,
        },
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "projected_successor_sha256": report["projected_successor"]["sha256"],
        "replayed_report_sha256": replayed_report["report_sha256"],
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_non_mutating_two_source_registry_v0_14_preflight",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification


__all__ = [
    "EXPECTED_LIMITS",
    "EXPECTED_SUCCESSOR",
    "build_hygiene_registry_v14_preflight_verification",
    "build_projected_registry_v14",
    "json_file_bytes",
    "load_hygiene_registry_v14_preflight_plan",
    "rebuild_hygiene_registry_v14_preflight",
    "validate_hygiene_registry_v14_preflight_report",
]
