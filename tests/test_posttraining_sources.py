from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lumi.posttraining_data import load_sft_policy
from lumi.posttraining_sources import (
    SFTSource,
    build_sft_source_readiness_report,
    load_sft_source_registry,
)

ROOT = Path(__file__).parents[1]
POLICY_PATH = ROOT / "configs" / "lumi-4b-sft-data-contract-v0.1.json"


def _load_registry():
    policy = load_sft_policy(POLICY_PATH)
    return policy, load_sft_source_registry(
        ROOT / policy.source_registry_path,
        allowed_licenses=policy.allowed_licenses,
        allowed_languages=policy.allowed_languages,
        allowed_splits=policy.allowed_splits,
        allowed_tasks=frozenset(policy.task_mix),
        expected_sha256=policy.source_registry_sha256,
    )


def test_source_registry_is_pinned_and_fail_closed() -> None:
    policy, registry = _load_registry()
    assert len(registry.sources) == 9
    assert len(registry.registry_sha256) == 64
    assert registry.by_id["openassistant_oasst1"].revision == (
        "fdf72ae0827c1cda404aff25b6603abec9e3399b"
    )
    assert registry.by_id["salesforce_xlam_function_calling_60k"].status == "blocked"
    assert registry.by_id["lumi_ai_assisted_sft_v0_1"].status == "blocked"
    assert registry.by_id["lumi_ai_assisted_sft_v0_2"].status == "candidate_eligible"
    assert registry.by_id["lumi_ai_assisted_sft_v0_2"].origin == "synthetic"
    assert all(not source.production_eligible for source in registry.sources)
    assert policy.source_registry_required is True


def test_source_binding_rejects_drift_and_blocked_sources() -> None:
    _, registry = _load_registry()
    assert registry.record_reasons(
        dataset_id="CohereLabs/aya_dataset",
        language="bn",
        license_id="apache-2.0",
        revision="f9ea04583f02a8f86404ff6c58bf75fe637df8a2",
        source_id="coherelabs_aya_dataset",
        split="train",
        task_category="multilingual_bengali",
    ) == ()
    assert "source_registry_revision" in registry.record_reasons(
        dataset_id="CohereLabs/aya_dataset",
        language="bn",
        license_id="apache-2.0",
        revision="0" * 40,
        source_id="coherelabs_aya_dataset",
        split="train",
        task_category="multilingual_bengali",
    )
    assert "source_registry_status:blocked" in registry.record_reasons(
        dataset_id="Salesforce/xlam-function-calling-60k",
        language="en",
        license_id="cc-by-4.0",
        revision=None,
        source_id="salesforce_xlam_function_calling_60k",
        split="train",
        task_category="tool_api",
    )


def test_readiness_report_exposes_missing_canary_and_production_coverage() -> None:
    policy, registry = _load_registry()
    report = build_sft_source_readiness_report(
        registry,
        required_tasks=frozenset(policy.task_mix),
    )
    assert report["metadata_audit_complete"] is True
    assert report["production_seal_ready"] is False
    assert report["source_status_counts"] == {
        "blocked": 4,
        "candidate_eligible": 1,
        "canary_eligible": 3,
        "planned": 1,
    }
    assert report["missing_canary_tasks"] == ["safety_helpfulness"]
    assert set(report["missing_production_tasks"]) == set(policy.task_mix)
    assert report["authorization"]["paid_gpu_authorized"] is False


def test_canary_status_cannot_hide_a_blocker() -> None:
    policy = load_sft_policy(POLICY_PATH)
    value = json.loads(
        (ROOT / "configs" / "lumi-4b-sft-source-registry-v0.1.json").read_text()
    )["sources"][0]
    changed = copy.deepcopy(value)
    changed["blockers"] = ["hidden_problem"]
    with pytest.raises(ValueError, match="has blockers"):
        SFTSource.from_mapping(
            changed,
            allowed_licenses=policy.allowed_licenses,
            allowed_languages=policy.allowed_languages,
            allowed_splits=policy.allowed_splits,
            allowed_tasks=frozenset(policy.task_mix),
        )


def test_reviewed_first_party_source_can_be_hash_pinned_and_production_eligible() -> None:
    policy = load_sft_policy(POLICY_PATH)
    value = json.loads(
        (ROOT / "configs" / "lumi-4b-sft-source-registry-v0.1.json").read_text()
    )["sources"][-1]
    reviewed = {
        **value,
        "blockers": [],
        "production_blockers": [],
        "production_eligible": True,
        "revision": "a" * 64,
        "status": "reviewed",
    }
    source = SFTSource.from_mapping(
        reviewed,
        allowed_licenses=policy.allowed_licenses,
        allowed_languages=policy.allowed_languages,
        allowed_splits=policy.allowed_splits,
        allowed_tasks=frozenset(policy.task_mix),
    )
    assert source.production_eligible is True
    assert source.revision == "a" * 64


def test_repaired_ai_assisted_candidate_is_hash_pinned_but_not_production() -> None:
    _, registry = _load_registry()
    rejected = registry.by_id["lumi_ai_assisted_sft_v0_1"]
    assert rejected.status == "blocked"
    assert "source_registry_status:blocked" in registry.record_reasons(
        dataset_id=rejected.dataset_id,
        language="bn",
        license_id=rejected.license_id,
        revision=rejected.revision,
        source_id=rejected.source_id,
        split="train",
        task_category="multilingual_bengali",
    )
    source = registry.by_id["lumi_ai_assisted_sft_v0_2"]
    assert source.revision == (
        "d492d2fa0817d08b55941b0393cb3d95ae5572a0bcc3c6de39f902bd8491a9b0"
    )
    assert source.production_eligible is False
    assert "dataset_level_owner_spot_check_not_complete" in source.production_blockers
    assert registry.record_reasons(
        dataset_id=source.dataset_id,
        language="bn",
        license_id=source.license_id,
        revision=source.revision,
        source_id=source.source_id,
        split="train",
        task_category="multilingual_bengali",
    ) == ()


def test_public_source_accepts_https_github_repository() -> None:
    policy = load_sft_policy(POLICY_PATH)
    source = SFTSource.from_mapping(
        {
            "access": "public",
            "adapter": "deepmind_mathematics_fixed_seed_verified_v0.1",
            "blockers": [],
            "dataset_id": "google-deepmind/mathematics_dataset",
            "evidence_urls": [
                "https://github.com/google-deepmind/mathematics_dataset"
            ],
            "languages": ["en"],
            "license_id": "apache-2.0",
            "origin": "synthetic",
            "production_blockers": ["production_dataset_not_sealed"],
            "production_eligible": False,
            "repository_url": (
                "https://github.com/google-deepmind/mathematics_dataset"
            ),
            "revision": "427f45075f84b8b9774950196ad63867ca20ffb3",
            "source_id": "deepmind_mathematics_dataset",
            "splits": ["train", "validation"],
            "status": "canary_eligible",
            "task_categories": ["reasoning_math_science"],
        },
        allowed_licenses=policy.allowed_licenses,
        allowed_languages=policy.allowed_languages,
        allowed_splits=policy.allowed_splits,
        allowed_tasks=frozenset(policy.task_mix),
    )
    assert source.repository_url == (
        "https://github.com/google-deepmind/mathematics_dataset"
    )


@pytest.mark.parametrize(
    "repository_url",
    [
        "http://github.com/google-deepmind/mathematics_dataset",
        "https://gitlab.com/google-deepmind/mathematics_dataset",
        "https://github.com.evil.example/google-deepmind/mathematics_dataset",
    ],
)
def test_external_source_rejects_unapproved_repository_url(
    repository_url: str,
) -> None:
    policy = load_sft_policy(POLICY_PATH)
    value = json.loads(
        (ROOT / "configs" / "lumi-4b-sft-source-registry-v0.1.json").read_text()
    )["sources"][0]
    changed = {**value, "repository_url": repository_url}
    with pytest.raises(ValueError, match="repository URL is not accepted"):
        SFTSource.from_mapping(
            changed,
            allowed_licenses=policy.allowed_licenses,
            allowed_languages=policy.allowed_languages,
            allowed_splits=policy.allowed_splits,
            allowed_tasks=frozenset(policy.task_mix),
        )
