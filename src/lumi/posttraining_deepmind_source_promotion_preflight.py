"""Fail-closed preflight for a DeepMind Mathematics candidate-registry promotion."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .posttraining_data import SFTPolicy, load_sft_policy
from .posttraining_deepmind_canary_run import (
    file_sha256,
    json_file_bytes,
    sha256_value,
)
from .posttraining_sources import SFTSource, SFTSourceRegistry, load_sft_source_registry

PLAN_SCHEMA_VERSION = "lumi.sft-deepmind-source-promotion-preflight-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-deepmind-source-promotion-preflight-report.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-deepmind-source-promotion-preflight-verification.v0.1"
)

CANARY_REPORT_INTERNAL_SHA256 = (
    "7d10bf58686f0ededde3e44108abec7c962be8880f8550c80498511385352a93"
)
CANARY_VERIFICATION_INTERNAL_SHA256 = (
    "aafdecb358b41c7385cb884ce0380281ec638399519f22d3ff04cc396c32774a"
)
PREDECESSOR_REGISTRY_SHA256 = (
    "5fc48b8475ff74d7c4d6d9a7b21efb7ce9f9527cf27c60a6ae1a28bf41a67b58"
)

EXPECTED_AUTHORIZATION = {
    "candidate_source_registry_promotion_authorized": False,
    "dataset_seal_authorized": False,
    "existing_registry_mutation_authorized": False,
    "external_api_spend_authorized": False,
    "external_distribution_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "purpose": "deepmind_candidate_registry_promotion_preflight_only",
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
    "proposed_successor_registry_path": "configs/lumi-4b-sft-source-registry-v0.6.json",
    "registry_successor_write_permitted": False,
    "report_contains_raw_training_text": False,
    "report_path": (
        "evidence/lumi-4b-sft-deepmind-source-promotion-preflight-v0.1.json"
    ),
    "verification_path": (
        "evidence/"
        "lumi-4b-sft-deepmind-source-promotion-preflight-verification-v0.1.json"
    ),
}
EXPECTED_CANDIDATE = {
    "accepted_dataset_sha256": (
        "47a63e136400a4879f92c3bee8e056c91fbc6e9a71da0dbcf73912d17c6e7bbb"
    ),
    "accepted_records": 1_915,
    "assistant_only_loss_verified": True,
    "context_window": 4_096,
    "global_exact_and_near_duplicates": 0,
    "official_test_or_evaluation_records": 0,
    "split_counts": {"train": 1_820, "validation": 95},
    "supervised_tokens": 18_731,
}
EXPECTED_CANARY_VERIFICATION_CHECKS = {
    "assistant_only_loss_and_4k_fit_replayed": True,
    "benchmark_decontamination_replayed": True,
    "cross_process_same_seed_replay_verified": True,
    "decoded_byte_ceiling_verified": 5_782_100,
    "exact_30_object_source_reverified": True,
    "generated_record_identities": 2_000,
    "global_exact_and_near_duplicates_in_final_candidate": 0,
    "official_test_or_evaluation_records": 0,
    "package_installations": 0,
    "per_record_attribution_verified": True,
    "pii_and_secret_filter_replayed": True,
    "runtime_network_guard_verified": True,
    "source_registry_check_deferred_until_separate_promotion": True,
}
EXPECTED_SOURCE_AFTER = {
    "access": "public",
    "adapter": "deepmind_mathematics_fixed_seed_verified_v0.1",
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
        "candidate_mix_is_only_1915_records_18731_supervised_tokens",
        "official_test_and_test_extra_splits_reserved_for_evaluation",
        "full_mix_cross_source_deduplication_not_complete",
        "production_dataset_seal_and_sft_authorization_not_complete",
    ],
    "production_eligible": False,
    "repository_url": "https://github.com/google-deepmind/mathematics_dataset",
    "revision": "427f45075f84b8b9774950196ad63867ca20ffb3",
    "source_id": "deepmind_mathematics_dataset",
    "splits": ["train", "validation"],
    "status": "canary_eligible",
    "task_categories": ["reasoning_math_science"],
}
EXPECTED_SCHEMA_DELTA = {
    "current_accepted_repository_prefix": "https://huggingface.co/datasets/",
    "field": "repository_url",
    "proposed_additional_repository_prefix": "https://github.com/",
    "required_for_exact_source": EXPECTED_SOURCE_AFTER["repository_url"],
    "source_registry_schema_version_unchanged": True,
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
EXPECTED_BINDING_PATHS = {
    "accepted_records": (
        "../lumi-artifacts/4b-sft-deepmind-canary-v0.1/427f45075f84/"
        "attempt-001/canary/curated/accepted-records.jsonl"
    ),
    "attribution_manifest": (
        "../lumi-artifacts/4b-sft-deepmind-canary-v0.1/427f45075f84/"
        "attempt-001/canary/curated/attribution-manifest.jsonl"
    ),
    "canary_plan": "configs/lumi-4b-sft-deepmind-canary-run-v0.1.json",
    "canary_report": "evidence/lumi-4b-sft-deepmind-canary-run-v0.1.json",
    "canary_verification": (
        "evidence/lumi-4b-sft-deepmind-canary-run-verification-v0.1.json"
    ),
    "data_contract": "configs/lumi-4b-sft-data-contract-v0.1.json",
    "predecessor_registry": "configs/lumi-4b-sft-source-registry-v0.5.json",
    "preflight_implementation": (
        "src/lumi/posttraining_deepmind_source_promotion_preflight.py"
    ),
    "preflight_runner": "scripts/preflight_deepmind_source_promotion.py",
    "preflight_verifier": "scripts/verify_deepmind_source_promotion_preflight.py",
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


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
            rows.append(value)
    return rows


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


def resolve_binding(
    repository_root: str | Path,
    binding: Mapping[str, Any],
    *,
    name: str,
) -> Path:
    root = Path(repository_root).resolve()
    workspace = root.parent.resolve()
    relative = binding.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"DeepMind promotion-preflight binding path invalid: {name}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError(f"DeepMind promotion-preflight binding escapes workspace: {name}")
    if not resolved.is_file():
        raise FileNotFoundError(f"DeepMind promotion-preflight binding missing: {name}")
    expected = _require_sha256(binding.get("sha256"), f"bindings.{name}.sha256")
    if file_sha256(resolved) != expected:
        raise ValueError(f"DeepMind promotion-preflight binding changed: {name}")
    return resolved


def load_preflight_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    plan = _load_json(path)
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "candidate": EXPECTED_CANDIDATE,
        "limits": EXPECTED_LIMITS,
        "observed_at": "2026-08-06",
        "output": EXPECTED_OUTPUT,
        "proposed_source": EXPECTED_SOURCE_AFTER,
        "required_tasks": EXPECTED_REQUIRED_TASKS,
        "schema_delta": EXPECTED_SCHEMA_DELTA,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "authorized_cpu_only_preflight_no_registry_mutation",
    }
    for field, expected_value in expected.items():
        if plan.get(field) != expected_value:
            raise ValueError(f"DeepMind promotion-preflight plan changed: {field}")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(
        EXPECTED_BINDING_PATHS
    ):
        raise ValueError("DeepMind promotion-preflight binding set changed")
    for name, expected_path in EXPECTED_BINDING_PATHS.items():
        binding = bindings[name]
        if not isinstance(binding, Mapping) or binding.get("path") != expected_path:
            raise ValueError(f"DeepMind promotion-preflight path changed: {name}")
        resolve_binding(repository_root, binding, name=name)
    return plan


def verify_canary_prerequisites(
    *,
    accepted_records: Sequence[Mapping[str, Any]],
    attributions: Sequence[Mapping[str, Any]],
    canary_report: Mapping[str, Any],
    canary_verification: Mapping[str, Any],
    token_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    if canary_report.get("schema_version") != (
        "lumi.sft-deepmind-canary-run-report.v0.1"
    ):
        raise ValueError("unsupported DeepMind canary report")
    if _verify_internal_seal(canary_report, "report_sha256") != (
        CANARY_REPORT_INTERNAL_SHA256
    ):
        raise ValueError("DeepMind canary report seal is not approved")
    if canary_report.get("status") != "verified_cpu_only_nonproduction_adapter_canary":
        raise ValueError("DeepMind canary report status changed")
    canary = canary_report.get("canary")
    if not isinstance(canary, Mapping):
        raise ValueError("DeepMind canary report body changed")
    filter_report = canary.get("filter")
    if (
        canary.get("retained_records") != 1_915
        or canary.get("split_counts") != {"train": 1_820, "validation": 95}
        or not isinstance(filter_report, Mapping)
        or filter_report.get("accepted_dataset_sha256")
        != EXPECTED_CANDIDATE["accepted_dataset_sha256"]
        or canary_report.get("raw_training_text_present") is not False
    ):
        raise ValueError("DeepMind canary candidate metrics changed")
    if canary_report.get("production") != {
        "candidate_registry_promotions": 0,
        "dataset_sealed": False,
        "gpu_used": False,
        "production_eligible": False,
        "production_sft_authorized": False,
        "sft_training_started": False,
    }:
        raise ValueError("DeepMind canary production boundary changed")
    if canary_verification.get("schema_version") != (
        "lumi.sft-deepmind-canary-run-verification.v0.1"
    ):
        raise ValueError("unsupported DeepMind canary verification")
    if _verify_internal_seal(canary_verification, "verification_sha256") != (
        CANARY_VERIFICATION_INTERNAL_SHA256
    ):
        raise ValueError("DeepMind canary verification seal is not approved")
    if (
        canary_verification.get("status")
        != "verified_cpu_only_nonproduction_deepmind_adapter_canary"
        or canary_verification.get("retained_records") != 1_915
        or canary_verification.get("supervised_tokens") != 18_731
        or canary_verification.get("production_eligible") is not False
    ):
        raise ValueError("DeepMind canary verification boundary changed")
    if canary_verification.get("checks") != EXPECTED_CANARY_VERIFICATION_CHECKS:
        raise ValueError("DeepMind canary verification checks changed")
    if token_metrics != canary.get("token_measurement"):
        raise ValueError("DeepMind token metrics changed from canary report")
    if (
        token_metrics.get("accepted_records") != 1_915
        or token_metrics.get("supervised_tokens") != 18_731
        or token_metrics.get("context_window") != 4_096
        or token_metrics.get("assistant_only_loss_verified") is not True
        or token_metrics.get("overlength_rejected") != 0
    ):
        raise ValueError("DeepMind token metrics boundary changed")
    if len(accepted_records) != 1_915 or len(attributions) != 1_915:
        raise ValueError("DeepMind candidate or attribution count changed")

    record_ids: set[str] = set()
    split_counts: Counter[str] = Counter()
    for row in accepted_records:
        record_id = row.get("record_id")
        source = row.get("source")
        messages = row.get("messages")
        split = row.get("split")
        if not isinstance(record_id, str) or not record_id or record_id in record_ids:
            raise ValueError("DeepMind accepted record identity changed")
        if (
            row.get("schema_version") != "lumi.sft.v0.1"
            or row.get("language") != "en"
            or row.get("task_category") != "reasoning_math_science"
            or split not in {"train", "validation"}
            or not isinstance(source, Mapping)
            or source.get("source_id") != "deepmind_mathematics_dataset"
            or source.get("dataset_id") != "google-deepmind/mathematics_dataset"
            or source.get("license_id") != "apache-2.0"
            or source.get("revision") != EXPECTED_SOURCE_AFTER["revision"]
            or not isinstance(messages, list)
            or [message.get("role") for message in messages]
            != ["user", "assistant"]
        ):
            raise ValueError("DeepMind accepted record schema or provenance changed")
        record_ids.add(record_id)
        split_counts[str(split)] += 1
    if dict(sorted(split_counts.items())) != EXPECTED_CANDIDATE["split_counts"]:
        raise ValueError("DeepMind accepted record split counts changed")
    attribution_ids = {str(row.get("record_id")) for row in attributions}
    if len(attribution_ids) != len(attributions) or attribution_ids != record_ids:
        raise ValueError("DeepMind per-record attribution coverage changed")
    return {
        "accepted_record_ids_sha256": sha256_value(sorted(record_ids)),
        "attribution_record_ids_sha256": sha256_value(sorted(attribution_ids)),
        "split_counts": dict(sorted(split_counts.items())),
    }


def load_policy_and_predecessor(
    *,
    plan: Mapping[str, Any],
    repository_root: str | Path,
) -> tuple[SFTPolicy, SFTSourceRegistry, dict[str, Any]]:
    root = Path(repository_root).resolve()
    bindings = plan["bindings"]
    policy = load_sft_policy(
        resolve_binding(root, bindings["data_contract"], name="data_contract")
    )
    if policy.source_registry_path != "configs/lumi-4b-sft-source-registry-v0.1.json":
        raise ValueError("active production data-contract registry changed")
    registry_path = resolve_binding(
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


def build_proposed_successor(
    predecessor_value: Mapping[str, Any],
) -> dict[str, Any]:
    sources = predecessor_value.get("sources")
    if not isinstance(sources, list) or len(sources) != 11:
        raise ValueError("DeepMind predecessor source array changed")
    successor = copy.deepcopy(dict(predecessor_value))
    successor["sources"].append(copy.deepcopy(EXPECTED_SOURCE_AFTER))
    if successor["sources"][:11] != sources or len(successor["sources"]) != 12:
        raise AssertionError("DeepMind proposed successor append changed")
    return successor


def verify_current_schema_blocker(*, policy: SFTPolicy) -> str:
    try:
        SFTSource.from_mapping(
            EXPECTED_SOURCE_AFTER,
            allowed_licenses=policy.allowed_licenses,
            allowed_languages=policy.allowed_languages,
            allowed_splits=policy.allowed_splits,
            allowed_tasks=frozenset(policy.task_mix),
        )
    except ValueError as error:
        expected = "source repository URL is not accepted: deepmind_mathematics_dataset"
        if str(error) != expected:
            raise ValueError("DeepMind source hit an unexpected registry blocker") from error
        return expected
    raise ValueError("DeepMind source unexpectedly passed the current repository URL schema")


def build_preflight_report(
    *,
    candidate_audit: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    predecessor: SFTSourceRegistry,
    predecessor_value: Mapping[str, Any],
    schema_blocker: str,
) -> dict[str, Any]:
    _require_sha256(plan_file_sha256, "plan_file_sha256")
    successor_value = build_proposed_successor(predecessor_value)
    successor_payload = json_file_bytes(successor_value)
    report: dict[str, Any] = {
        "authorization": EXPECTED_AUTHORIZATION,
        "candidate": {
            **EXPECTED_CANDIDATE,
            **dict(candidate_audit),
        },
        "decision": (
            "ready_for_exact_promotion_authorization_with_narrow_"
            "repository_url_schema_extension"
        ),
        "execution": {
            "candidate_registry_promotions": 0,
            "external_api_calls": 0,
            "gpu_used": False,
            "network_bytes": 0,
            "predecessor_registry_mutated": False,
            "successor_registry_written": False,
        },
        "limits": EXPECTED_LIMITS,
        "next_gate": {
            "new_explicit_authorization_required": True,
            "proposed_source": EXPECTED_SOURCE_AFTER,
            "proposed_successor_file_sha256": hashlib.sha256(
                successor_payload
            ).hexdigest(),
            "proposed_successor_registry_path": EXPECTED_OUTPUT[
                "proposed_successor_registry_path"
            ],
            "required_schema_delta": EXPECTED_SCHEMA_DELTA,
            "verified_current_schema_blocker": schema_blocker,
        },
        "observed_at": plan["observed_at"],
        "plan_file_sha256": plan_file_sha256,
        "predecessor": {
            "registry_sha256": predecessor.registry_sha256,
            "source_count": len(predecessor.sources),
            "unchanged": True,
        },
        "production": {
            "active_data_contract_updated": False,
            "dataset_sealed": False,
            "production_eligible": False,
            "production_sft_authorized": False,
            "sft_training_started": False,
        },
        "raw_training_text_present": False,
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "verified_cpu_only_nonproduction_promotion_preflight",
    }
    report["report_sha256"] = sha256_value(report)
    return report


def validate_preflight_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported DeepMind promotion-preflight report")
    if report.get("status") != "verified_cpu_only_nonproduction_promotion_preflight":
        raise ValueError("DeepMind promotion-preflight status changed")
    if report.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("DeepMind promotion-preflight authorization changed")
    if report.get("execution") != {
        "candidate_registry_promotions": 0,
        "external_api_calls": 0,
        "gpu_used": False,
        "network_bytes": 0,
        "predecessor_registry_mutated": False,
        "successor_registry_written": False,
    }:
        raise ValueError("DeepMind promotion-preflight execution boundary changed")
    if report.get("production") != {
        "active_data_contract_updated": False,
        "dataset_sealed": False,
        "production_eligible": False,
        "production_sft_authorized": False,
        "sft_training_started": False,
    }:
        raise ValueError("DeepMind promotion-preflight production boundary changed")
    if report.get("raw_training_text_present") is not False:
        raise ValueError("DeepMind promotion-preflight report contains raw training text")
    seal = _require_sha256(report.get("report_sha256"), "report_sha256")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if sha256_value(unsigned) != seal:
        raise ValueError("DeepMind promotion-preflight report seal changed")
