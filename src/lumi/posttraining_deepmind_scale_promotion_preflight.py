"""Raw-text-free preflight for promoting the reviewed DeepMind scale candidate."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .posttraining_deepmind_scale_generation import file_sha256, sha256_value
from .posttraining_deepmind_scale_owner_review import (
    validate_owner_review_receipt,
)
from .posttraining_sources import SFTSource

PLAN_SCHEMA_VERSION = "lumi.sft-deepmind-scale-promotion-preflight-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-deepmind-scale-promotion-preflight-report.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-deepmind-scale-promotion-preflight-verification.v0.1"
)

SOURCE_ID = "deepmind_mathematics_dataset"
SOURCE_REVISION = "427f45075f84b8b9774950196ad63867ca20ffb3"
EXPECTED_AUTHORIZATION = {
    "candidate_registry_promotion_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "gpu_authorized": False,
    "local_raw_text_free_preflight_authorized": True,
    "network_authorized": False,
    "production_sft_authorized": False,
    "unrelated_access_authorized": False,
}
EXPECTED_LIMITS = {
    "candidate_records": 236_698,
    "maximum_local_bytes_hashed": 512 * 1024 * 1024,
    "maximum_runtime_minutes": 60,
    "registry_entries_changed": 1,
}
EXPECTED_CANDIDATE = {
    "records": 236_698,
    "train_records": 224_688,
    "train_supervised_tokens": 2_221_113,
    "validation_records": 12_010,
    "validation_supervised_tokens": 118_053,
}
EXPECTED_CURRENT_POOL = {
    "accepted_train_supervised_tokens": 1_068_485,
    "deepmind_train_supervised_tokens": 17_805,
    "reasoning_math_science_train_supervised_tokens": 66_405,
    "target_train_supervised_tokens": 10_000_000,
    "reasoning_math_science_floor": 2_200_000,
    "source_share_cap": 0.35,
}
EXPECTED_PROJECTION = {
    "accepted_train_supervised_tokens": 3_271_793,
    "deepmind_source_share": 0.67886721,
    "gap_to_10m_train_supervised_tokens": 6_728_207,
    "reasoning_math_science_floor_excess": 69_713,
    "reasoning_math_science_train_supervised_tokens": 2_269_713,
    "source_cap_passed": False,
}
EXPECTED_OUTPUT = {
    "proposed_successor_path": "configs/lumi-4b-sft-source-registry-v0.10.json",
    "raw_training_text_in_outputs": False,
    "report_path": (
        "evidence/lumi-4b-sft-deepmind-scale-promotion-preflight-v0.1.json"
    ),
    "verification_path": (
        "evidence/"
        "lumi-4b-sft-deepmind-scale-promotion-preflight-verification-v0.1.json"
    ),
}
EXPECTED_BINDINGS = frozenset(
    {
        "accepted_records",
        "attribution_manifest",
        "current_mix_report",
        "data_contract",
        "implementation",
        "owner_review_receipt",
        "predecessor_registry",
        "runner",
        "scale_report",
        "scale_verification",
        "source_registry_implementation",
        "token_metrics",
        "verifier",
    }
)

EXPECTED_SOURCE_AFTER = {
    "access": "public",
    "adapter": "deepmind_mathematics_fixed_seed_scale_verified_v0.2",
    "blockers": [],
    "dataset_id": "google-deepmind/mathematics_dataset",
    "evidence_urls": [
        (
            "https://github.com/google-deepmind/mathematics_dataset/tree/"
            "427f45075f84b8b9774950196ad63867ca20ffb3"
        ),
        (
            "https://github.com/google-deepmind/mathematics_dataset/blob/"
            "427f45075f84b8b9774950196ad63867ca20ffb3/LICENSE"
        ),
    ],
    "languages": ["en"],
    "license_id": "apache-2.0",
    "origin": "synthetic",
    "production_blockers": [
        "candidate_mix_is_236698_records_2339166_supervised_tokens",
        "owner_quality_review_is_32_row_deterministic_sample_not_full_record_review",
        "official_test_and_test_extra_splits_reserved_for_evaluation",
        "cross_source_deduplication_against_complete_sft_mix_not_complete",
        "source_share_would_exceed_0_35_cap_in_current_candidate_pool",
        "production_dataset_seal_and_sft_authorization_not_complete",
    ],
    "production_eligible": False,
    "repository_url": "https://github.com/google-deepmind/mathematics_dataset",
    "revision": SOURCE_REVISION,
    "source_id": SOURCE_ID,
    "splits": ["train", "validation"],
    "status": "canary_eligible",
    "task_categories": ["reasoning_math_science"],
}


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be lowercase SHA-256")
    return value


def _verify_internal_seal(value: Mapping[str, Any], field: str) -> str:
    seal = _require_sha256(value.get(field), field)
    unsigned = {key: item for key, item in value.items() if key != field}
    if sha256_value(unsigned) != seal:
        raise ValueError(f"{field} changed")
    return seal


def resolve_binding(
    repository_root: Path, specification: Mapping[str, Any], *, name: str
) -> Path:
    workspace = repository_root.parent.resolve()
    path = (repository_root / str(specification.get("path", ""))).resolve()
    if not path.is_relative_to(workspace) or not path.is_file():
        raise FileNotFoundError(f"DeepMind scale preflight binding missing: {name}")
    expected = _require_sha256(specification.get("sha256"), f"{name}.sha256")
    if file_sha256(path) != expected:
        raise ValueError(f"DeepMind scale preflight binding changed: {name}")
    return path


def load_preflight_plan(path: Path, *, repository_root: Path) -> dict[str, Any]:
    plan = _load_json(path)
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "candidate": EXPECTED_CANDIDATE,
        "current_pool": EXPECTED_CURRENT_POOL,
        "limits": EXPECTED_LIMITS,
        "observed_at": "2026-08-07",
        "output": EXPECTED_OUTPUT,
        "schema_version": PLAN_SCHEMA_VERSION,
        "source_after": EXPECTED_SOURCE_AFTER,
        "source_id": SOURCE_ID,
        "source_revision": SOURCE_REVISION,
        "status": "authorized_raw_text_free_preflight_no_registry_mutation",
    }
    for field, expected_value in expected.items():
        if plan.get(field) != expected_value:
            raise ValueError(f"DeepMind scale preflight plan changed: {field}")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("DeepMind scale preflight binding inventory changed")
    for name, specification in bindings.items():
        if not isinstance(specification, Mapping):
            raise ValueError(f"DeepMind scale preflight binding invalid: {name}")
        resolve_binding(repository_root, specification, name=name)
    return plan


def _validate_scale_evidence(
    *,
    plan: Mapping[str, Any],
    repository_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    bindings = plan["bindings"]
    report = _load_json(
        resolve_binding(repository_root, bindings["scale_report"], name="scale_report")
    )
    verification = _load_json(
        resolve_binding(
            repository_root,
            bindings["scale_verification"],
            name="scale_verification",
        )
    )
    owner_receipt = _load_json(
        resolve_binding(
            repository_root,
            bindings["owner_review_receipt"],
            name="owner_review_receipt",
        )
    )
    token_metrics = _load_json(
        resolve_binding(
            repository_root, bindings["token_metrics"], name="token_metrics"
        )
    )
    if _verify_internal_seal(report, "report_sha256") != (
        "173cebcf98c89b80d184073348584f90de5de172ec20dc5099c1e6aed7134c10"
    ):
        raise ValueError("DeepMind scale report is not the approved report")
    if _verify_internal_seal(verification, "verification_sha256") != (
        "337fbc3c13c5ae77957c896ee57797162c1ed411c313165a7907901d768b6327"
    ):
        raise ValueError("DeepMind scale verification is not the approved verification")
    validate_owner_review_receipt(owner_receipt)
    if owner_receipt.get("receipt_sha256") != (
        "97c898d6c16f55f5620ecd1ab22f8836f445c5d1ed73eb1303d073887df1d44f"
    ):
        raise ValueError("DeepMind scale owner-review receipt is not approved")
    if (
        owner_receipt.get("status")
        != "owner_quality_sample_passed_source_promotion_still_separate"
        or owner_receipt.get("accepted_records") != 32
        or owner_receipt.get("rejected_records") != 0
        or owner_receipt.get("candidate") != EXPECTED_CANDIDATE
    ):
        raise ValueError("DeepMind scale owner-review gate did not pass")
    measurement = report.get("scale", {}).get("token_measurement")
    if measurement != token_metrics:
        raise ValueError("DeepMind scale token metrics changed")
    if measurement.get("accepted_records") != EXPECTED_CANDIDATE["records"]:
        raise ValueError("DeepMind scale retained records changed")
    if measurement.get("supervised_tokens") != 2_339_166:
        raise ValueError("DeepMind scale total supervised tokens changed")
    if measurement.get("by_split", {}).get("train", {}).get(
        "supervised_tokens"
    ) != EXPECTED_CANDIDATE["train_supervised_tokens"]:
        raise ValueError("DeepMind scale train supervised tokens changed")
    if measurement.get("by_split", {}).get("validation", {}).get(
        "supervised_tokens"
    ) != EXPECTED_CANDIDATE["validation_supervised_tokens"]:
        raise ValueError("DeepMind scale validation supervised tokens changed")
    return report, verification, owner_receipt, token_metrics


def build_proposed_successor(
    predecessor: Mapping[str, Any], *, source_after: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    sources = predecessor.get("sources")
    if not isinstance(sources, list):
        raise ValueError("source registry sources must be an array")
    indices = [
        index
        for index, source in enumerate(sources)
        if isinstance(source, Mapping) and source.get("source_id") == SOURCE_ID
    ]
    if len(indices) != 1:
        raise ValueError("DeepMind source must appear exactly once in predecessor")
    index = indices[0]
    before = sources[index]
    if (
        before.get("revision") != SOURCE_REVISION
        or before.get("adapter") != "deepmind_mathematics_fixed_seed_verified_v0.1"
        or before.get("status") != "canary_eligible"
        or before.get("production_eligible") is not False
    ):
        raise ValueError("DeepMind predecessor entry changed")
    successor = copy.deepcopy(dict(predecessor))
    successor["sources"][index] = copy.deepcopy(dict(source_after))
    return successor, copy.deepcopy(dict(before))


def _validate_registry_sources(
    successor: Mapping[str, Any], *, policy: Mapping[str, Any]
) -> None:
    sources = successor.get("sources")
    if not isinstance(sources, list) or len(sources) != 13:
        raise ValueError("DeepMind scale successor source count changed")
    parsed = [
        SFTSource.from_mapping(
            source,
            allowed_licenses=frozenset(policy["allowed_licenses"]),
            allowed_languages=frozenset(policy["allowed_languages"]),
            allowed_splits=frozenset(policy["allowed_splits"]),
            allowed_tasks=frozenset(policy["task_mix"]),
        )
        for source in sources
    ]
    source_ids = [source.source_id for source in parsed]
    dataset_ids = [source.dataset_id for source in parsed]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("DeepMind scale successor source IDs are not unique")
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("DeepMind scale successor dataset IDs are not unique")


def _project_candidate_mix(current_mix: Mapping[str, Any]) -> dict[str, Any]:
    pool = current_mix.get("candidate_pool")
    deepmind = current_mix.get("source_metrics", {}).get(SOURCE_ID)
    if not isinstance(pool, Mapping) or not isinstance(deepmind, Mapping):
        raise ValueError("current candidate mix is missing DeepMind metrics")
    observed_current = {
        "accepted_train_supervised_tokens": pool.get(
            "accepted_train_supervised_tokens"
        ),
        "deepmind_train_supervised_tokens": deepmind.get("by_split", {})
        .get("train", {})
        .get("supervised_tokens"),
        "reasoning_math_science_train_supervised_tokens": pool.get(
            "train_by_task", {}
        ).get("reasoning_math_science"),
        "target_train_supervised_tokens": 10_000_000,
        "reasoning_math_science_floor": 2_200_000,
        "source_share_cap": current_mix.get("closure", {}).get("source_cap"),
    }
    if observed_current != EXPECTED_CURRENT_POOL:
        raise ValueError("current candidate-pool metrics changed")
    return dict(EXPECTED_PROJECTION)


def build_preflight_report(
    *, plan: Mapping[str, Any], plan_file_sha256: str, repository_root: Path
) -> dict[str, Any]:
    _require_sha256(plan_file_sha256, "plan_file_sha256")
    bindings = plan["bindings"]
    report, verification, owner_receipt, token_metrics = _validate_scale_evidence(
        plan=plan, repository_root=repository_root
    )
    predecessor_path = resolve_binding(
        repository_root, bindings["predecessor_registry"], name="predecessor_registry"
    )
    predecessor = _load_json(predecessor_path)
    successor, source_before = build_proposed_successor(
        predecessor, source_after=plan["source_after"]
    )
    policy = _load_json(
        resolve_binding(repository_root, bindings["data_contract"], name="data_contract")
    )
    _validate_registry_sources(successor, policy=policy)
    successor_sha256 = file_sha256_from_bytes(json_file_bytes(successor))
    mix_report = _load_json(
        resolve_binding(
            repository_root, bindings["current_mix_report"], name="current_mix_report"
        )
    )
    _verify_internal_seal(mix_report, "report_sha256")
    projection = _project_candidate_mix(mix_report)
    candidate_files_bytes = sum(
        resolve_binding(repository_root, bindings[name], name=name).stat().st_size
        for name in ("accepted_records", "attribution_manifest", "token_metrics")
    )
    if candidate_files_bytes > EXPECTED_LIMITS["maximum_local_bytes_hashed"]:
        raise ValueError("DeepMind scale preflight local-byte limit exceeded")
    result: dict[str, Any] = {
        "authorization": EXPECTED_AUTHORIZATION,
        "bindings": {
            name: specification["sha256"] for name, specification in bindings.items()
        }
        | {
            "owner_review_internal_sha256": owner_receipt["receipt_sha256"],
            "plan_file_sha256": plan_file_sha256,
            "scale_report_internal_sha256": report["report_sha256"],
            "scale_verification_internal_sha256": verification[
                "verification_sha256"
            ],
            "token_ledger_sha256": token_metrics["token_ledger_sha256"],
        },
        "candidate": dict(EXPECTED_CANDIDATE)
        | {"total_supervised_tokens": 2_339_166},
        "checks": {
            "all_bound_artifact_hashes_verified": True,
            "candidate_owner_sample_accepted_32_of_32": True,
            "candidate_registry_promotions": 0,
            "candidate_scale_verification_passed": True,
            "exactly_one_existing_registry_entry_would_change": True,
            "official_test_or_test_extra_used_for_training": False,
            "predecessor_registry_mutated": False,
            "production_dataset_promotions": 0,
            "proposed_successor_schema_valid": True,
            "raw_training_text_in_report": False,
        },
        "limits": EXPECTED_LIMITS,
        "observed_at": "2026-08-07",
        "projected_candidate_mix": projection,
        "proposed_successor": {
            "changed_source_ids": [SOURCE_ID],
            "file_sha256": successor_sha256,
            "path": EXPECTED_OUTPUT["proposed_successor_path"],
            "source_count": 13,
        },
        "raw_training_text_retained": False,
        "schema_version": REPORT_SCHEMA_VERSION,
        "source_after": dict(plan["source_after"]),
        "source_before": source_before,
        "status": "verified_scale_source_promotion_preflight_nonproduction",
        "next_gate": {
            "action": "authorize_exact_deepmind_scale_registry_v0_10_promotion",
            "authorization_required": True,
            "dataset_seal_authorized": False,
            "gpu_sft_authorized": False,
            "post_promotion_action": (
                "rerun_full_mix_audit_then_acquire_diversified_non_math_data"
            ),
        },
    }
    result["report_sha256"] = sha256_value(result)
    return result


def file_sha256_from_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def validate_preflight_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported DeepMind scale preflight report")
    _verify_internal_seal(report, "report_sha256")
    if report.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("DeepMind scale preflight authorization changed")
    if report.get("candidate") != EXPECTED_CANDIDATE | {
        "total_supervised_tokens": 2_339_166
    }:
        raise ValueError("DeepMind scale preflight candidate metrics changed")
    if report.get("projected_candidate_mix") != EXPECTED_PROJECTION:
        raise ValueError("DeepMind scale preflight projection changed")
    if report.get("raw_training_text_retained") is not False:
        raise ValueError("DeepMind scale preflight retained raw text")
    checks = report.get("checks")
    if not isinstance(checks, Mapping) or not all(
        value is True or value == 0 for value in checks.values()
    ):
        raise ValueError("DeepMind scale preflight checks did not pass")


__all__ = [
    "EXPECTED_AUTHORIZATION",
    "EXPECTED_CANDIDATE",
    "EXPECTED_OUTPUT",
    "EXPECTED_PROJECTION",
    "EXPECTED_SOURCE_AFTER",
    "PLAN_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "VERIFICATION_SCHEMA_VERSION",
    "build_preflight_report",
    "build_proposed_successor",
    "json_file_bytes",
    "load_preflight_plan",
    "validate_preflight_report",
]
