"""Pure metadata-contract successor to the frozen safety URI v1 verifier.

This module performs no file access, imports no model/data implementation, and
does not replay candidate rows or certify past execution actions. The caller
must authenticate both input blobs against externally trusted file hashes
before calling ``audit_report``; an internally valid seal is not authenticity.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

SCHEMA_VERSION = "lumi.sft-safety-uri-metadata-assurance-report.v0.2"
EXPECTED_PREDECESSOR_SHA256 = "ffc4a91613dd228282d2bc8ef2b93a6d508e31fa90e2925d91de837c1b4bd0e1"
EXPECTED_REGISTRY_SHA256 = "8cd8cc0b20ccd7da2f9083b0903178b98f59e3279c6777ebf8fee4866069d573"
EXPECTED_AUTHORIZATION = {
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_promotion_authorized": False,
    "production_sft_authorized": False,
    "purpose": "safety_tool_deterministic_first_party_uri_and_validation_only",
    "source_uri_repair_authorized": True,
    "unrelated_data_access_authorized": False,
}
EXPECTED_LIMITS = {
    "maximum_network_bytes": 0,
    "maximum_runtime_minutes": 60,
    "safety_candidate_records": 320,
    "source_registry_entries_changed": 0,
}
EXPECTED_OUTPUT = {
    "artifact_manifest_path": (
        "../lumi-artifacts/4b-sft-safety-uri-repair-v0.1/"
        "8cd8cc0b20cc/attempt-001/safety-tool-uri-repair-manifest.json"
    ),
    "private_safety_candidate_path": (
        "../lumi-artifacts/4b-sft-safety-uri-repair-v0.1/"
        "8cd8cc0b20cc/attempt-001/safety-tool-candidate.jsonl"
    ),
    "production_dataset_promotions": 0,
    "raw_training_text_in_repo_evidence": False,
    "report_path": "evidence/lumi-4b-sft-safety-uri-repair-v0.1.json",
    "verification_path": "evidence/lumi-4b-sft-safety-uri-repair-verification-v0.1.json",
}
EXPECTED_SAFETY_TOKENS = {
    "train": {"records": 240, "supervised_tokens": 18_636},
    "validation": {"records": 80, "supervised_tokens": 6_388},
}
EXPECTED_BINDING_NAMES = frozenset(
    {
        "benchmark_registry",
        "chat_template",
        "data_contract",
        "implementation",
        "posttraining_data_implementation",
        "predecessor_candidate",
        "predecessor_hygiene_manifest",
        "predecessor_hygiene_report",
        "predecessor_hygiene_verification",
        "runner",
        "safety_owner_review_receipt",
        "source_registry",
        "source_registry_implementation",
        "test",
        "tokenizer_config",
        "tokenizer_json",
        "verifier",
    }
)
EXPECTED_REMAINING_BLOCKERS = [
    "accepted_train_supervised_tokens_below_10m_floor",
    "task_language_and_source_balance_not_closed",
    "full_mix_cross_source_near_deduplication_not_complete",
    "active_production_data_contract_not_updated",
    "dataset_seal_cpu_canary_and_sft_not_authorized",
]
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ERROR = "invalid safety URI metadata contract"


def _require(condition: bool) -> None:
    if not condition:
        raise ValueError(_ERROR)


def _fields(value: Any, names: set[str] | frozenset[str]) -> None:
    _require(type(value) is dict and set(value) == names)


def _exact(value: Any, expected: Any) -> None:
    """Compare JSON values without Python's bool/int/float equality aliases."""
    _require(type(value) is type(expected))
    if type(expected) is dict:
        _fields(value, set(expected))
        for key, item in expected.items():
            _exact(value[key], item)
    elif type(expected) is list:
        _require(len(value) == len(expected))
        for actual, item in zip(value, expected, strict=True):
            _exact(actual, item)
    else:
        _require(value == expected)


def _sha256(value: Any) -> None:
    _require(type(value) is str and _SHA256.fullmatch(value) is not None)


def _canonical_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_plan(plan: dict[str, Any]) -> None:
    _fields(
        plan,
        {
            "authorization",
            "bindings",
            "limits",
            "observed_at",
            "output",
            "schema_version",
            "status",
        },
    )
    for key, expected in {
        "authorization": EXPECTED_AUTHORIZATION,
        "limits": EXPECTED_LIMITS,
        "observed_at": "2026-08-06",
        "output": EXPECTED_OUTPUT,
        "schema_version": "lumi.sft-safety-uri-repair-plan.v0.1",
        "status": "authorized_bounded_cpu_only_safety_uri_repair",
    }.items():
        _exact(plan[key], expected)
    _fields(plan["bindings"], EXPECTED_BINDING_NAMES)
    for binding in plan["bindings"].values():
        _fields(binding, {"path", "sha256"})
        path = binding["path"]
        _require(type(path) is str and bool(path.strip()))
        _require(not any(character in path for character in ("\x00", "\n", "\r")))
        _sha256(binding["sha256"])
    _exact(plan["bindings"]["predecessor_candidate"]["sha256"], EXPECTED_PREDECESSOR_SHA256)
    _exact(plan["bindings"]["source_registry"]["sha256"], EXPECTED_REGISTRY_SHA256)


def audit_report(
    plan: dict[str, Any],
    report: dict[str, Any],
    *,
    plan_file_sha256: str,
    report_file_sha256: str,
) -> dict[str, Any]:
    """Validate exact v1 metadata and return only limited v2 assurance.

    ``plan_file_sha256`` must match the report's declared plan binding. Both
    supplied file hashes must be externally authenticated by the caller against
    the original bytes; this pure function cannot authenticate those bytes.
    Binding paths remain opaque metadata and are never opened or resolved.
    Unknown fields are rejected, including raw-text-bearing extensions.
    """
    _sha256(plan_file_sha256)
    _sha256(report_file_sha256)
    _validate_plan(plan)
    _fields(
        report,
        {
            "authorization",
            "bindings",
            "frozen_token_verification",
            "generic_sft_validation",
            "metadata_repair",
            "observed_at",
            "production_dataset_promotions",
            "production_eligible",
            "raw_training_text_retained",
            "remaining_blockers",
            "report_sha256",
            "schema_version",
            "status",
        },
    )
    for key, expected in {
        "authorization": EXPECTED_AUTHORIZATION,
        "observed_at": "2026-08-06",
        "production_dataset_promotions": 0,
        "production_eligible": False,
        "raw_training_text_retained": False,
        "remaining_blockers": EXPECTED_REMAINING_BLOCKERS,
        "schema_version": "lumi.sft-safety-uri-repair-report.v0.1",
        "status": "verified_safety_source_uri_repair_nonproduction",
    }.items():
        _exact(report[key], expected)

    bindings = report["bindings"]
    _fields(
        bindings,
        {
            "artifact_manifest_file_sha256",
            "benchmark_registry_sha256",
            "generic_validation_report_sha256",
            "output_candidate_file_sha256",
            "plan_file_sha256",
            "predecessor_candidate_file_sha256",
            "source_registry_sha256",
        },
    )
    for digest in bindings.values():
        _sha256(digest)
    _exact(bindings["plan_file_sha256"], plan_file_sha256)
    for report_key, plan_key in {
        "benchmark_registry_sha256": "benchmark_registry",
        "predecessor_candidate_file_sha256": "predecessor_candidate",
        "source_registry_sha256": "source_registry",
    }.items():
        _exact(bindings[report_key], plan["bindings"][plan_key]["sha256"])

    tokens = report["frozen_token_verification"]
    _fields(tokens, {"by_split", "chat_template_sha256", "context_window", "tokenizer_json_sha256"})
    _exact(tokens["by_split"], EXPECTED_SAFETY_TOKENS)
    _exact(tokens["context_window"], 4096)
    for report_key, plan_key in {
        "chat_template_sha256": "chat_template",
        "tokenizer_json_sha256": "tokenizer_json",
    }.items():
        _sha256(tokens[report_key])
        _exact(tokens[report_key], plan["bindings"][plan_key]["sha256"])

    metadata = report["metadata_repair"]
    counts = {
        "records",
        "records_with_deterministic_source_uri",
        "records_with_only_authorized_metadata_delta",
        "unique_source_uris",
    }
    hashes = {
        "candidate_file_sha256",
        "content_and_messages_binding_sha256",
        "transform_chain_binding_sha256",
        "uri_set_sha256",
    }
    _fields(metadata, counts | hashes)
    for key in counts:
        _exact(metadata[key], 320)
    for key in hashes:
        _sha256(metadata[key])
    _exact(metadata["candidate_file_sha256"], bindings["output_candidate_file_sha256"])

    generic = report["generic_sft_validation"]
    expected_generic = {
        "accepted": 320,
        "benchmark_contamination_rejections": 0,
        "exact_normalized_duplicates": 0,
        "raw_text_retained": False,
        "rejected": 0,
        "rejection_counts": {},
        "source_registry_rejections": 0,
        "total": 320,
    }
    _fields(generic, set(expected_generic) | {"report_sha256"})
    for key, expected in expected_generic.items():
        _exact(generic[key], expected)
    _sha256(generic["report_sha256"])
    _exact(generic["report_sha256"], bindings["generic_validation_report_sha256"])
    _sha256(report["report_sha256"])
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    _exact(report["report_sha256"], _canonical_sha256(unsigned))

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_METADATA_CONTRACT_ONLY",
        "metadata_contract_passed": True,
        "candidate_rows_replayed": False,
        "prior_execution_actions_verified": False,
        "full_private_integration_passed": False,
        "execution_authorized": False,
        "production_eligible": False,
    }
