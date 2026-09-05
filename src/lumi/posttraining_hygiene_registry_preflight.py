"""Project a two-source immutable registry successor without materializing it."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .posttraining_candidate_mix_hygiene_materialization import (
    validate_hygiene_materialization_receipt,
)
from .posttraining_candidate_mix_preflight import (
    _load_json,
    file_sha256,
    sha256_value,
)
from .posttraining_candidate_mix_reaudit import resolve_binding
from .posttraining_data import load_sft_policy
from .posttraining_sources import SFTSource

PLAN_SCHEMA_VERSION = "lumi.sft-hygiene-registry-successor-preflight-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-hygiene-registry-successor-preflight-report.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-hygiene-registry-successor-preflight-verification.v0.1"
)

EXPECTED_AUTHORIZATION = {
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "purpose": "project_exact_two_source_registry_v0_12_successor_only",
    "source_registry_mutation_authorized": False,
    "source_registry_promotion_authorized": False,
    "unrelated_access_authorized": False,
}
EXPECTED_LIMITS = {
    "changed_sources": 2,
    "maximum_network_bytes": 0,
    "maximum_report_bytes": 1_048_576,
    "maximum_runtime_minutes": 30,
    "predecessor_sources": 14,
}
EXPECTED_OUTPUT = {
    "preflight_report_path": (
        "evidence/lumi-4b-sft-hygiene-registry-v0.12-preflight-v0.1.json"
    ),
    "projected_successor_materialized": False,
    "projected_successor_registry_path": (
        "configs/lumi-4b-sft-source-registry-v0.12.json"
    ),
    "raw_training_text_in_outputs": False,
    "verification_path": (
        "evidence/lumi-4b-sft-hygiene-registry-v0.12-preflight-"
        "verification-v0.1.json"
    ),
}
EXPECTED_BINDINGS = frozenset(
    {
        "aya_clean_candidate",
        "data_contract",
        "hygiene_materialization_implementation",
        "hygiene_materialization_receipt",
        "hygiene_materialization_verification",
        "implementation",
        "oasst1_clean_candidate",
        "predecessor_registry",
        "runner",
        "source_registry_implementation",
        "test",
        "verifier",
    }
)
EXPECTED_PREDECESSOR_SHA256 = (
    "5353c57189d82ee22306ccd11d116d017c5e683811cec2d6704cf45bcf7f02e7"
)
EXPECTED_MATERIALIZATION = {
    "receipt_file_sha256": (
        "08d9430be9b0fe8ffbf85d11bd0caa4cb81a78c9faa8ee57df45798f38371a78"
    ),
    "receipt_internal_sha256": (
        "f06ba2b2c572c8e17c8c7c146bc99f6f8da7e8ae1e219f41d86a2a66ac0435ed"
    ),
    "verification_file_sha256": (
        "24ec151e1e96bb159a46bbac32ab686476d8460fd15a3b6171b27fdfc6694a74"
    ),
    "verification_internal_sha256": (
        "4c15a03a2ab50e0fc91086a34dfbe78b89c524bb584b692aa1a127cc27d438b3"
    ),
}
EXPECTED_SUCCESSOR = {
    "bytes": 16_157,
    "sha256": "8f286b096ee2e406fe4ee0a6d69ea37a8e7017e489fda6c765df46ed2d1b86b7",
    "source_count": 14,
}
SOURCE_UPDATES = {
    "coherelabs_aya_dataset": {
        "adapter_after": "aya_bengali_repaired_4k_reviewed_split_deduplicated_v0.4",
        "adapter_before": "aya_bengali_repaired_4k_reviewed_split_v0.3",
        "candidate": {
            "records": 1_458,
            "sha256": (
                "6e601c9eee1cc3f90647cd4701a8bf24667dd36e25a42f49f7ef4059b9683558"
            ),
            "train_supervised_tokens": 513_379,
            "validation_supervised_tokens": 55_581,
        },
        "count_blocker_after": (
            "candidate_mix_is_only_1458_records_568960_supervised_tokens"
        ),
        "count_blocker_before": (
            "candidate_mix_is_only_1478_records_577453_supervised_tokens"
        ),
    },
    "openassistant_oasst1": {
        "adapter_after": "oasst1_root_direct_rank0_quality_decontaminated_v0.4",
        "adapter_before": "oasst1_root_direct_rank0_quality_v0.3",
        "candidate": {
            "records": 1_742,
            "sha256": (
                "1c970f94fe009acc4d379e9c53eee5ecefe29f42db33f01c6d9c09c87e7950c2"
            ),
            "train_supervised_tokens": 452_690,
            "validation_supervised_tokens": 23_475,
        },
        "count_blocker_after": (
            "candidate_mix_is_only_1742_records_476165_supervised_tokens"
        ),
        "count_blocker_before": (
            "candidate_mix_is_only_1745_records_477456_supervised_tokens"
        ),
    },
}


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


def load_hygiene_registry_preflight_plan(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    plan = _load_json(Path(path))
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "limits": EXPECTED_LIMITS,
        "observed_at": "2026-08-08",
        "output": EXPECTED_OUTPUT,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "authorized_cpu_only_non_mutating_registry_v0_12_preflight",
    }
    for field, expected_value in expected.items():
        if plan.get(field) != expected_value:
            raise ValueError(f"hygiene registry preflight plan changed: {field}")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("hygiene registry preflight binding set changed")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"invalid hygiene registry preflight binding: {name}")
        resolve_binding(repository_root, binding, name=name)
    if bindings["predecessor_registry"]["sha256"] != EXPECTED_PREDECESSOR_SHA256:
        raise ValueError("hygiene registry predecessor binding changed")
    if bindings["hygiene_materialization_receipt"]["sha256"] != (
        EXPECTED_MATERIALIZATION["receipt_file_sha256"]
    ):
        raise ValueError("hygiene registry materialization receipt changed")
    if bindings["hygiene_materialization_verification"]["sha256"] != (
        EXPECTED_MATERIALIZATION["verification_file_sha256"]
    ):
        raise ValueError("hygiene registry materialization verification changed")
    expected_source_ids = plan.get("changed_source_ids")
    if expected_source_ids != sorted(SOURCE_UPDATES):
        raise ValueError("hygiene registry changed-source set changed")
    return plan


def _verify_materialization_evidence(
    *, root: Path, plan: Mapping[str, Any]
) -> dict[str, Any]:
    receipt_path = resolve_binding(
        root,
        plan["bindings"]["hygiene_materialization_receipt"],
        name="hygiene_materialization_receipt",
    )
    verification_path = resolve_binding(
        root,
        plan["bindings"]["hygiene_materialization_verification"],
        name="hygiene_materialization_verification",
    )
    receipt = _load_json(receipt_path)
    verification = _load_json(verification_path)
    validate_hygiene_materialization_receipt(receipt)
    _verify_internal_sha(verification, "verification_sha256")
    if receipt.get("receipt_sha256") != EXPECTED_MATERIALIZATION[
        "receipt_internal_sha256"
    ]:
        raise ValueError("hygiene registry receipt internal seal changed")
    if verification.get("verification_sha256") != EXPECTED_MATERIALIZATION[
        "verification_internal_sha256"
    ]:
        raise ValueError("hygiene registry verification internal seal changed")
    if verification.get("receipt_file_sha256") != EXPECTED_MATERIALIZATION[
        "receipt_file_sha256"
    ]:
        raise ValueError("hygiene registry materialization evidence link changed")
    for source_id, update in SOURCE_UPDATES.items():
        output = receipt.get("candidate_outputs", {}).get(source_id, {})
        candidate_binding_name = (
            "aya_clean_candidate"
            if source_id == "coherelabs_aya_dataset"
            else "oasst1_clean_candidate"
        )
        candidate_path = resolve_binding(
            root,
            plan["bindings"][candidate_binding_name],
            name=candidate_binding_name,
        )
        if (
            output.get("sha256") != update["candidate"]["sha256"]
            or output.get("records") != update["candidate"]["records"]
            or file_sha256(candidate_path) != update["candidate"]["sha256"]
        ):
            raise ValueError(f"hygiene registry clean candidate changed: {source_id}")
        if output.get("by_split") != {
            "train": {
                "records": (
                    1_313 if source_id == "coherelabs_aya_dataset" else 1_649
                ),
                "supervised_tokens": update["candidate"][
                    "train_supervised_tokens"
                ],
            },
            "validation": {
                "records": 145 if source_id == "coherelabs_aya_dataset" else 93,
                "supervised_tokens": update["candidate"][
                    "validation_supervised_tokens"
                ],
            },
        }:
            raise ValueError(f"hygiene registry clean split metrics changed: {source_id}")
    return receipt


def build_projected_registry(
    predecessor: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_sources = predecessor.get("sources")
    if not isinstance(raw_sources, list) or len(raw_sources) != 14:
        raise ValueError("hygiene registry predecessor source count changed")
    successor = copy.deepcopy(dict(predecessor))
    changed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in successor["sources"]:
        source_id = source.get("source_id")
        if source_id not in SOURCE_UPDATES:
            continue
        update = SOURCE_UPDATES[source_id]
        if source.get("adapter") != update["adapter_before"]:
            raise ValueError(f"hygiene registry predecessor adapter changed: {source_id}")
        blockers = source.get("production_blockers")
        if (
            not isinstance(blockers, list)
            or blockers
            != [
                update["count_blocker_before"],
                "full_mix_cross_source_deduplication_not_complete",
                "production_dataset_seal_and_sft_authorization_not_complete",
            ]
        ):
            raise ValueError(f"hygiene registry predecessor blockers changed: {source_id}")
        before_sha256 = sha256_value(source)
        source["adapter"] = update["adapter_after"]
        source["production_blockers"][0] = update["count_blocker_after"]
        changed.append(
            {
                "adapter_after": update["adapter_after"],
                "adapter_before": update["adapter_before"],
                "candidate": dict(update["candidate"]),
                "count_blocker_after": update["count_blocker_after"],
                "count_blocker_before": update["count_blocker_before"],
                "source_after_sha256": sha256_value(source),
                "source_before_sha256": before_sha256,
                "source_id": source_id,
            }
        )
        seen.add(source_id)
    if seen != set(SOURCE_UPDATES) or len(changed) != 2:
        raise ValueError("hygiene registry did not change exactly two sources")
    changed.sort(key=lambda item: item["source_id"])
    return successor, changed


def _validate_successor_sources(
    successor: Mapping[str, Any], *, data_contract_path: Path
) -> None:
    policy = load_sft_policy(data_contract_path)
    sources = successor.get("sources")
    if not isinstance(sources, list) or len(sources) != 14:
        raise ValueError("hygiene registry successor source count changed")
    for source in sources:
        SFTSource.from_mapping(
            source,
            allowed_licenses=policy.allowed_licenses,
            allowed_languages=policy.allowed_languages,
            allowed_splits=policy.allowed_splits,
            allowed_tasks=frozenset(policy.task_mix),
        )


def rebuild_hygiene_registry_preflight(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    receipt = _verify_materialization_evidence(root=root, plan=plan)
    predecessor_path = resolve_binding(
        root,
        plan["bindings"]["predecessor_registry"],
        name="predecessor_registry",
    )
    data_contract_path = resolve_binding(
        root,
        plan["bindings"]["data_contract"],
        name="data_contract",
    )
    predecessor = _load_json(predecessor_path)
    successor, changed = build_projected_registry(predecessor)
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
        raise ValueError("hygiene registry projected successor changed")
    predecessor_unchanged = copy.deepcopy(predecessor)
    successor_unchanged = copy.deepcopy(successor)
    predecessor_unchanged["sources"] = [
        item
        for item in predecessor_unchanged["sources"]
        if item["source_id"] not in SOURCE_UPDATES
    ]
    successor_unchanged["sources"] = [
        item
        for item in successor_unchanged["sources"]
        if item["source_id"] not in SOURCE_UPDATES
    ]
    if predecessor_unchanged != successor_unchanged:
        raise ValueError("hygiene registry projection changed unrelated metadata")
    report: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "changed_sources": changed,
        "changed_sources_count": len(changed),
        "materialization_binding": {
            "receipt_file_sha256": EXPECTED_MATERIALIZATION[
                "receipt_file_sha256"
            ],
            "receipt_internal_sha256": receipt["receipt_sha256"],
            "verification_file_sha256": EXPECTED_MATERIALIZATION[
                "verification_file_sha256"
            ],
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
        "status": "verified_two_source_registry_v0_12_projection_not_materialized",
        "unchanged_registry_metadata_sha256": sha256_value(predecessor_unchanged),
    }
    report["report_sha256"] = sha256_value(report)
    validate_hygiene_registry_preflight_report(report)
    if len(json_file_bytes(report)) > plan["limits"]["maximum_report_bytes"]:
        raise ValueError("hygiene registry preflight report ceiling exceeded")
    return report


def validate_hygiene_registry_preflight_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported hygiene registry preflight report schema")
    _verify_internal_sha(report, "report_sha256")
    if report.get("changed_sources_count") != 2:
        raise ValueError("hygiene registry preflight changed-source count changed")
    if report.get("registry_files_materialized") != 0:
        raise ValueError("hygiene registry preflight materialized a registry")
    if report.get("production_eligible") is not False:
        raise ValueError("hygiene registry preflight cannot be production eligible")
    if report.get("raw_training_text_retained") is not False:
        raise ValueError("hygiene registry preflight must remain raw-text-free")
    projected = report.get("projected_successor", {})
    if projected.get("materialized") is not False:
        raise ValueError("hygiene registry projected successor was materialized")
    if {
        key: projected.get(key) for key in ("bytes", "sha256", "source_count")
    } != EXPECTED_SUCCESSOR:
        raise ValueError("hygiene registry successor summary changed")
    if [item.get("source_id") for item in report.get("changed_sources", [])] != sorted(
        SOURCE_UPDATES
    ):
        raise ValueError("hygiene registry changed-source identities changed")
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    for forbidden in ('"messages"', '"content"', '"assistant_response"'):
        if forbidden in serialized:
            raise ValueError("hygiene registry report contains raw training text")


def build_hygiene_registry_preflight_verification(
    *,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    report_file_sha256: str,
    replayed_report: Mapping[str, Any],
) -> dict[str, Any]:
    validate_hygiene_registry_preflight_report(report)
    validate_hygiene_registry_preflight_report(replayed_report)
    if report != replayed_report:
        raise ValueError("hygiene registry independent replay changed")
    verification: dict[str, Any] = {
        "checks": {
            "exact_two_source_metadata_updates_replayed": True,
            "materialized_clean_candidate_hashes_and_counts_verified": True,
            "no_network_api_gpu_registry_mutation_seal_or_sft": True,
            "predecessor_registry_v0_11_remains_immutable": True,
            "projected_registry_v0_12_not_materialized": True,
            "projected_successor_hash_reproduced": True,
            "raw_training_text_absent_from_repo_evidence": True,
            "unrelated_sources_and_top_level_metadata_unchanged": True,
        },
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "projected_successor_sha256": report["projected_successor"]["sha256"],
        "replayed_report_sha256": replayed_report["report_sha256"],
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_non_mutating_two_source_registry_v0_12_preflight",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification

