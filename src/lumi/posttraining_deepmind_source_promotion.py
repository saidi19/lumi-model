"""Fail-closed candidate-registry promotion for DeepMind Mathematics."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .posttraining_data import SFTPolicy, load_sft_policy
from .posttraining_deepmind_canary_run import (
    file_sha256,
    json_file_bytes,
    sha256_value,
)
from .posttraining_deepmind_source_promotion_preflight import (
    EXPECTED_CANDIDATE,
    EXPECTED_SOURCE_AFTER,
    build_proposed_successor,
    load_jsonl,
    validate_preflight_report,
    verify_canary_prerequisites,
)
from .posttraining_sources import (
    SFTSource,
    SFTSourceRegistry,
    build_sft_source_readiness_report,
    load_sft_source_registry,
)

PLAN_SCHEMA_VERSION = "lumi.sft-deepmind-source-promotion-plan.v0.1"
RECEIPT_SCHEMA_VERSION = "lumi.sft-deepmind-source-promotion-receipt.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-deepmind-source-promotion-verification.v0.1"
)

PREDECESSOR_REGISTRY_SHA256 = (
    "5fc48b8475ff74d7c4d6d9a7b21efb7ce9f9527cf27c60a6ae1a28bf41a67b58"
)
ACCEPTED_RECORDS_SHA256 = (
    "d9495290bf08c56f3de9733cf6b05d89398b16153d8bec39ef4ec915661e8255"
)
SUCCESSOR_REGISTRY_SHA256 = (
    "4892b05c49dc082cb15157b48f72e9f90f89b23fe6e7ac7daa3db85e5dff1ebc"
)
PREFLIGHT_PLAN_FILE_SHA256 = (
    "1c4a82b3f0604f3b3dba7161ab692d703853baefaef48ea5e29b5b43f757f9ce"
)
PREFLIGHT_REPORT_FILE_SHA256 = (
    "a0722faf3815789d7a79d80584f915a3ec54a1875137cb8c86f26669ee06c098"
)
PREFLIGHT_REPORT_INTERNAL_SHA256 = (
    "b3d889ad8445db5c94095fd112dfcc9f8acceb9e61d0f6b363bb82f24ad24646"
)
PREFLIGHT_VERIFICATION_FILE_SHA256 = (
    "88ec28ce1f3d1e551ac13b8f621860984f6de98054b714770575e0c551603d4f"
)
PREFLIGHT_VERIFICATION_INTERNAL_SHA256 = (
    "db5d03616d811d1f62c55b50d39a90b883532881290a066625d54612f1a80dd5"
)

EXPECTED_AUTHORIZATION = {
    "candidate_source_registry_promotion_authorized": True,
    "dataset_seal_authorized": False,
    "existing_registry_mutation_authorized": False,
    "external_api_spend_authorized": False,
    "external_distribution_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "purpose": "local_verified_deepmind_successor_registry_only",
    "repository_url_schema_extension_authorized": True,
    "unrelated_data_access_authorized": False,
}
EXPECTED_LIMITS = {
    "added_sources": 1,
    "candidate_records": 1_915,
    "candidate_supervised_tokens": 18_731,
    "changed_predecessor_sources": 0,
    "predecessor_sources": 11,
    "removed_sources": 0,
    "successor_sources": 12,
}
EXPECTED_OUTPUT = {
    "active_production_data_contract_update_permitted": False,
    "predecessor_registry_mutation_permitted": False,
    "production_dataset_promotions": 0,
    "receipt_contains_raw_training_text": False,
    "receipt_path": (
        "evidence/lumi-4b-sft-deepmind-source-promotion-v0.1.json"
    ),
    "successor_registry_path": "configs/lumi-4b-sft-source-registry-v0.6.json",
    "verification_path": (
        "evidence/lumi-4b-sft-deepmind-source-promotion-verification-v0.1.json"
    ),
}
EXPECTED_REQUIRED_TASKS = [
    "coding",
    "conversation",
    "factual_qa",
    "multilingual_bengali",
    "reasoning_math_science",
    "safety_helpfulness",
    "tool_api",
    "writing_summarization",
]
EXPECTED_SCHEMA_DELTA = {
    "added_repository_prefix": "https://github.com/",
    "existing_huggingface_prefix_preserved": True,
    "field": "repository_url",
    "source_registry_schema_version_unchanged": True,
}
EXPECTED_BINDING_PATHS = {
    "accepted_records": (
        "../lumi-artifacts/4b-sft-deepmind-canary-v0.1/427f45075f84/"
        "attempt-001/canary/curated/accepted-records.jsonl"
    ),
    "attribution_manifest": (
        "../lumi-artifacts/4b-sft-deepmind-canary-v0.1/427f45075f84/"
        "attempt-001/canary/curated/attribution-manifest.jsonl"
    ),
    "canary_report": "evidence/lumi-4b-sft-deepmind-canary-run-v0.1.json",
    "canary_verification": (
        "evidence/lumi-4b-sft-deepmind-canary-run-verification-v0.1.json"
    ),
    "data_contract": "configs/lumi-4b-sft-data-contract-v0.1.json",
    "predecessor_registry": "configs/lumi-4b-sft-source-registry-v0.5.json",
    "preflight_plan": (
        "configs/lumi-4b-sft-deepmind-source-promotion-preflight-v0.1.json"
    ),
    "preflight_report": (
        "evidence/lumi-4b-sft-deepmind-source-promotion-preflight-v0.1.json"
    ),
    "preflight_verification": (
        "evidence/"
        "lumi-4b-sft-deepmind-source-promotion-preflight-verification-v0.1.json"
    ),
    "promotion_implementation": (
        "src/lumi/posttraining_deepmind_source_promotion.py"
    ),
    "promotion_runner": "scripts/promote_deepmind_source.py",
    "promotion_verifier": "scripts/verify_deepmind_source_promotion.py",
    "source_registry_implementation": "src/lumi/posttraining_sources.py",
    "token_metrics": (
        "../lumi-artifacts/4b-sft-deepmind-canary-v0.1/427f45075f84/"
        "attempt-001/canary/curated/token-metrics.json"
    ),
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a SHA-256")
    return value


def _verify_internal_seal(value: Mapping[str, Any], field: str) -> str:
    seal = _require_sha256(value.get(field), field)
    unsigned = {key: item for key, item in value.items() if key != field}
    if sha256_value(unsigned) != seal:
        raise ValueError(f"{field} changed")
    return seal


def resolve_promotion_binding(
    repository_root: str | Path,
    binding: Mapping[str, Any],
    *,
    name: str,
) -> Path:
    root = Path(repository_root).resolve()
    workspace = root.parent.resolve()
    relative = binding.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"DeepMind source-promotion binding path invalid: {name}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError(f"DeepMind source-promotion binding escapes workspace: {name}")
    if not resolved.is_file():
        raise FileNotFoundError(f"DeepMind source-promotion binding missing: {name}")
    expected = _require_sha256(binding.get("sha256"), f"bindings.{name}.sha256")
    if file_sha256(resolved) != expected:
        raise ValueError(f"DeepMind source-promotion binding changed: {name}")
    return resolved


def load_source_promotion_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    plan = _load_json(path)
    expected = {
        "approved_successor_registry_sha256": SUCCESSOR_REGISTRY_SHA256,
        "authorization": EXPECTED_AUTHORIZATION,
        "candidate": EXPECTED_CANDIDATE,
        "limits": EXPECTED_LIMITS,
        "observed_at": "2026-08-06",
        "output": EXPECTED_OUTPUT,
        "required_tasks": EXPECTED_REQUIRED_TASKS,
        "schema_delta": EXPECTED_SCHEMA_DELTA,
        "schema_version": PLAN_SCHEMA_VERSION,
        "source_after": EXPECTED_SOURCE_AFTER,
        "status": "authorized_candidate_registry_successor_promotion",
    }
    for field, expected_value in expected.items():
        if plan.get(field) != expected_value:
            raise ValueError(f"DeepMind source-promotion plan changed: {field}")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(
        EXPECTED_BINDING_PATHS
    ):
        raise ValueError("DeepMind source-promotion binding set changed")
    for name, expected_path in EXPECTED_BINDING_PATHS.items():
        binding = bindings[name]
        if not isinstance(binding, Mapping) or binding.get("path") != expected_path:
            raise ValueError(f"DeepMind source-promotion path changed: {name}")
        resolve_promotion_binding(repository_root, binding, name=name)
    if bindings["accepted_records"]["sha256"] != ACCEPTED_RECORDS_SHA256:
        raise ValueError("DeepMind accepted-records authorization changed")
    if bindings["predecessor_registry"]["sha256"] != (
        PREDECESSOR_REGISTRY_SHA256
    ):
        raise ValueError("DeepMind predecessor authorization changed")
    if bindings["preflight_plan"]["sha256"] != PREFLIGHT_PLAN_FILE_SHA256:
        raise ValueError("DeepMind preflight-plan authorization changed")
    if bindings["preflight_report"]["sha256"] != PREFLIGHT_REPORT_FILE_SHA256:
        raise ValueError("DeepMind preflight-report authorization changed")
    if bindings["preflight_verification"]["sha256"] != (
        PREFLIGHT_VERIFICATION_FILE_SHA256
    ):
        raise ValueError("DeepMind preflight-verification authorization changed")
    return plan


def verify_promotion_prerequisites(
    *,
    plan: Mapping[str, Any],
    repository_root: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(repository_root).resolve()
    bindings = plan["bindings"]
    records = load_jsonl(
        resolve_promotion_binding(root, bindings["accepted_records"], name="accepted_records")
    )
    attributions = load_jsonl(
        resolve_promotion_binding(
            root,
            bindings["attribution_manifest"],
            name="attribution_manifest",
        )
    )
    candidate_audit = verify_canary_prerequisites(
        accepted_records=records,
        attributions=attributions,
        canary_report=_load_json(
            resolve_promotion_binding(
                root,
                bindings["canary_report"],
                name="canary_report",
            )
        ),
        canary_verification=_load_json(
            resolve_promotion_binding(
                root,
                bindings["canary_verification"],
                name="canary_verification",
            )
        ),
        token_metrics=_load_json(
            resolve_promotion_binding(
                root,
                bindings["token_metrics"],
                name="token_metrics",
            )
        ),
    )
    preflight_plan = _load_json(
        resolve_promotion_binding(
            root,
            bindings["preflight_plan"],
            name="preflight_plan",
        )
    )
    if (
        preflight_plan.get("status")
        != "authorized_cpu_only_preflight_no_registry_mutation"
        or preflight_plan.get("authorization", {}).get(
            "candidate_source_registry_promotion_authorized"
        )
        is not False
    ):
        raise ValueError("DeepMind historical preflight boundary changed")
    preflight_report = _load_json(
        resolve_promotion_binding(
            root,
            bindings["preflight_report"],
            name="preflight_report",
        )
    )
    validate_preflight_report(preflight_report)
    if preflight_report.get("report_sha256") != PREFLIGHT_REPORT_INTERNAL_SHA256:
        raise ValueError("DeepMind preflight report seal is not approved")
    if preflight_report.get("candidate") != {**EXPECTED_CANDIDATE, **candidate_audit}:
        raise ValueError("DeepMind preflight candidate audit changed")
    next_gate = preflight_report.get("next_gate")
    if not isinstance(next_gate, Mapping) or (
        next_gate.get("proposed_successor_file_sha256")
        != SUCCESSOR_REGISTRY_SHA256
        or next_gate.get("proposed_source") != EXPECTED_SOURCE_AFTER
    ):
        raise ValueError("DeepMind preflight successor binding changed")
    preflight_verification = _load_json(
        resolve_promotion_binding(
            root,
            bindings["preflight_verification"],
            name="preflight_verification",
        )
    )
    if (
        preflight_verification.get("schema_version")
        != "lumi.sft-deepmind-source-promotion-preflight-verification.v0.1"
        or preflight_verification.get("status")
        != "verified_deepmind_source_promotion_preflight_nonproduction"
        or _verify_internal_seal(preflight_verification, "verification_sha256")
        != PREFLIGHT_VERIFICATION_INTERNAL_SHA256
        or preflight_verification.get("report_internal_sha256")
        != PREFLIGHT_REPORT_INTERNAL_SHA256
        or preflight_verification.get("candidate_records") != 1_915
        or preflight_verification.get("candidate_supervised_tokens") != 18_731
        or preflight_verification.get("production_eligible") is not False
    ):
        raise ValueError("DeepMind preflight verification changed")
    return records, candidate_audit


def load_policy_and_predecessor(
    *,
    plan: Mapping[str, Any],
    repository_root: str | Path,
) -> tuple[SFTPolicy, SFTSourceRegistry, dict[str, Any]]:
    root = Path(repository_root).resolve()
    bindings = plan["bindings"]
    policy = load_sft_policy(
        resolve_promotion_binding(root, bindings["data_contract"], name="data_contract")
    )
    if (
        policy.source_registry_path
        != "configs/lumi-4b-sft-source-registry-v0.1.json"
        or policy.source_registry_sha256
        != "f73676bbfab84f9c421ff3eb611d8e395c0bdfb50a9fa87f5559d98ac74107c9"
    ):
        raise ValueError("active production data-contract registry changed")
    registry_path = resolve_promotion_binding(
        root,
        bindings["predecessor_registry"],
        name="predecessor_registry",
    )
    predecessor = load_sft_source_registry(
        registry_path,
        allowed_licenses=policy.allowed_licenses,
        allowed_languages=policy.allowed_languages,
        allowed_splits=policy.allowed_splits,
        allowed_tasks=frozenset(policy.task_mix),
        expected_sha256=PREDECESSOR_REGISTRY_SHA256,
    )
    if len(predecessor.sources) != 11:
        raise ValueError("DeepMind predecessor source count changed")
    if "deepmind_mathematics_dataset" in predecessor.by_id:
        raise ValueError("DeepMind already exists in predecessor registry")
    return policy, predecessor, _load_json(registry_path)


def build_successor_registry(
    predecessor_value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    if plan.get("source_after") != EXPECTED_SOURCE_AFTER:
        raise ValueError("DeepMind source append changed")
    sources = predecessor_value.get("sources")
    if not isinstance(sources, list) or len(sources) != 11:
        raise ValueError("DeepMind predecessor source array changed")
    if any(
        isinstance(source, Mapping)
        and source.get("source_id") == "deepmind_mathematics_dataset"
        for source in sources
    ):
        raise ValueError("DeepMind already exists in predecessor registry")
    return build_proposed_successor(predecessor_value)


def validate_successor_registry(
    *,
    policy: SFTPolicy,
    predecessor_value: Mapping[str, Any],
    successor_value: Mapping[str, Any],
    successor_file_sha256: str,
) -> SFTSourceRegistry:
    if successor_file_sha256 != SUCCESSOR_REGISTRY_SHA256:
        raise ValueError("DeepMind successor registry SHA-256 is not authorized")
    before = predecessor_value.get("sources")
    after = successor_value.get("sources")
    if not isinstance(before, list) or not isinstance(after, list):
        raise ValueError("DeepMind registry sources must be arrays")
    if len(before) != 11 or len(after) != 12:
        raise ValueError("DeepMind successor source count changed")
    predecessor_metadata = {
        key: value for key, value in predecessor_value.items() if key != "sources"
    }
    successor_metadata = {
        key: value for key, value in successor_value.items() if key != "sources"
    }
    if predecessor_metadata != successor_metadata:
        raise ValueError("DeepMind successor changed registry metadata")
    if after[:11] != before or after[11] != EXPECTED_SOURCE_AFTER:
        raise ValueError("DeepMind successor changed an unauthorized source")
    parsed = tuple(
        SFTSource.from_mapping(
            source,
            allowed_licenses=policy.allowed_licenses,
            allowed_languages=policy.allowed_languages,
            allowed_splits=policy.allowed_splits,
            allowed_tasks=frozenset(policy.task_mix),
        )
        for source in after
    )
    source_ids = [source.source_id for source in parsed]
    dataset_ids = [source.dataset_id for source in parsed]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("DeepMind successor source IDs are not unique")
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("DeepMind successor dataset IDs are not unique")
    return SFTSourceRegistry(
        observed_at=str(successor_value["observed_at"]),
        registry_sha256=successor_file_sha256,
        sources=parsed,
    )


def validate_records_against_successor(
    records: Sequence[Mapping[str, Any]],
    *,
    successor: SFTSourceRegistry,
) -> None:
    if len(records) != 1_915:
        raise ValueError("DeepMind candidate record count changed")
    for row in records:
        source = row["source"]
        reasons = successor.record_reasons(
            dataset_id=source["dataset_id"],
            language=row["language"],
            license_id=source["license_id"],
            revision=source["revision"],
            source_id=source["source_id"],
            split=row["split"],
            task_category=row["task_category"],
        )
        if reasons:
            raise ValueError(
                f"DeepMind successor rejected an accepted record: {reasons}"
            )


def build_source_promotion_receipt(
    *,
    candidate_audit: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    predecessor: SFTSourceRegistry,
    successor: SFTSourceRegistry,
) -> dict[str, Any]:
    _require_sha256(plan_file_sha256, "plan_file_sha256")
    required_tasks = frozenset(plan["required_tasks"])
    before = build_sft_source_readiness_report(
        predecessor,
        required_tasks=required_tasks,
    )
    after = build_sft_source_readiness_report(
        successor,
        required_tasks=required_tasks,
    )
    if before["missing_canary_tasks"] or after["missing_canary_tasks"]:
        raise ValueError("DeepMind registry lost candidate/canary task coverage")
    if before["missing_production_tasks"] != after["missing_production_tasks"]:
        raise ValueError("DeepMind promotion changed production coverage")
    if set(after["missing_production_tasks"]) != required_tasks:
        raise ValueError("DeepMind promotion unexpectedly enabled production")
    receipt: dict[str, Any] = {
        "active_production_data_contract_updated": False,
        "authorization": EXPECTED_AUTHORIZATION,
        "bindings": {
            "accepted_records_file_sha256": ACCEPTED_RECORDS_SHA256,
            "attribution_manifest_file_sha256": plan["bindings"][
                "attribution_manifest"
            ]["sha256"],
            "canary_report_file_sha256": plan["bindings"]["canary_report"][
                "sha256"
            ],
            "canary_verification_file_sha256": plan["bindings"][
                "canary_verification"
            ]["sha256"],
            "plan_file_sha256": plan_file_sha256,
            "predecessor_registry_sha256": predecessor.registry_sha256,
            "preflight_report_internal_sha256": (
                PREFLIGHT_REPORT_INTERNAL_SHA256
            ),
            "preflight_verification_internal_sha256": (
                PREFLIGHT_VERIFICATION_INTERNAL_SHA256
            ),
            "source_registry_implementation_file_sha256": plan["bindings"][
                "source_registry_implementation"
            ]["sha256"],
            "successor_registry_sha256": successor.registry_sha256,
        },
        "candidate": {**EXPECTED_CANDIDATE, **dict(candidate_audit)},
        "candidate_source_registry_promotions": 1,
        "counts": EXPECTED_LIMITS,
        "observed_at": plan["observed_at"],
        "production_dataset_promotions": 0,
        "production_eligible": False,
        "raw_training_text_retained": False,
        "registry_delta": {
            "added_source_id": "deepmind_mathematics_dataset",
            "changed_predecessor_sources": 0,
            "removed_sources": 0,
            "status_after": "canary_eligible",
        },
        "remaining_blockers": [
            "complete_10m_to_30m_learning_mix_not_ready",
            "full_mix_cross_source_deduplication_not_complete",
            "active_production_data_contract_not_updated",
            "production_dataset_seal_cpu_canary_and_sft_not_authorized",
        ],
        "schema_delta": EXPECTED_SCHEMA_DELTA,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "source": copy.deepcopy(EXPECTED_SOURCE_AFTER),
        "status": (
            "verified_deepmind_registered_in_immutable_candidate_successor_"
            "production_dataset_still_blocked"
        ),
    }
    receipt["receipt_sha256"] = sha256_value(receipt)
    return receipt


def verify_source_promotion_receipt(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported DeepMind source-promotion receipt")
    _verify_internal_seal(value, "receipt_sha256")
    if (
        value.get("status")
        != (
            "verified_deepmind_registered_in_immutable_candidate_successor_"
            "production_dataset_still_blocked"
        )
        or value.get("candidate_source_registry_promotions") != 1
        or value.get("production_dataset_promotions") != 0
        or value.get("production_eligible") is not False
        or value.get("active_production_data_contract_updated") is not False
        or value.get("raw_training_text_retained") is not False
        or value.get("counts") != EXPECTED_LIMITS
        or value.get("source") != EXPECTED_SOURCE_AFTER
        or value.get("schema_delta") != EXPECTED_SCHEMA_DELTA
    ):
        raise ValueError("DeepMind source-promotion receipt boundary changed")


def successor_payload_sha256(value: Mapping[str, Any]) -> str:
    """Return the approved on-disk JSON serialization SHA-256."""

    return hashlib.sha256(json_file_bytes(value)).hexdigest()
