"""Exact materialization of the reviewed Lumi teacher-canary source."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .data_quality import BenchmarkRegistry
from .posttraining_data import filter_sft_records, load_sft_policy
from .posttraining_sources import SFTSource, SFTSourceRegistry
from .posttraining_teacher_canary_source_promotion_preflight import (
    EXPECTED_OWNER_RECEIPT_FILE_SHA256,
    SOURCE_ID,
    build_projected_successor,
    build_promotion_preflight_report,
    file_sha256,
    json_file_bytes,
    jsonl_file_bytes,
    load_promotion_preflight_plan,
    project_canonical_candidate,
    resolve_binding as resolve_preflight_binding,
    sha256_value,
    validate_report,
    validate_verification as validate_preflight_verification,
)

PLAN_SCHEMA_VERSION = "lumi.sft-first-party-teacher-source-promotion-plan.v0.1"
RECEIPT_SCHEMA_VERSION = "lumi.sft-first-party-teacher-source-promotion-receipt.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-first-party-teacher-source-promotion-verification.v0.1"
)
PLAN_STATUS = "authorized_exact_teacher_canary_candidate_and_registry_v0_15"

PREFLIGHT_PLAN_FILE_SHA256 = (
    "d54ad33c3cd4129844ba7a5e6b3e41f2d7056dbb8523559f2850149225a3dc1d"
)
PREFLIGHT_REPORT_FILE_SHA256 = (
    "004bebe9bf1d45273a1af470f7ab0633b4f0727e98e25656a9ca15d36c7b4539"
)
PREFLIGHT_REPORT_INTERNAL_SHA256 = (
    "8f3f3840356efab126a43e5da7c548e07a8d377d5c9007b5c36d286a284560d0"
)
PREFLIGHT_VERIFICATION_FILE_SHA256 = (
    "9d24a8e5d8318853262c46ebe582c381e552fda2484fb13de8504591ff794ec9"
)
PREFLIGHT_VERIFICATION_INTERNAL_SHA256 = (
    "dab2bc045cd75c30dcfa941b2242569af719805841655ce3b0613a73b77159e9"
)
PREDECESSOR_REGISTRY_SHA256 = (
    "a7310e469abb06cf4fbd12a72b2dcf2fbbebeac064e7b9182b994a9a77254e16"
)
CANONICAL_CANDIDATE_SHA256 = (
    "fcb0720ea178157ffbd0a2ece6eee647aea8d46fcd9a896251806f14996076e7"
)
CANONICAL_CANDIDATE_INTERNAL_SHA256 = (
    "016df8cf7f23b4d37c629a09e7f955d362e03862712d65f2a01a8ae619b03cdf"
)
CANONICAL_CANDIDATE_BYTES = 142_009
SUCCESSOR_REGISTRY_SHA256 = (
    "de81a51f0288b4851ff5ea4ca0598ddcf4936b2bf2c4127ab2b1c0d6deb1a4de"
)
SUCCESSOR_REGISTRY_BYTES = 18_859
EXPECTED_RECORDS = 80
EXPECTED_TRAIN_RECORDS = 60
EXPECTED_VALIDATION_RECORDS = 20
EXPECTED_TRAIN_TOKENS = 2_002
EXPECTED_VALIDATION_TOKENS = 799

CANONICAL_CANDIDATE_PATH = (
    "../lumi-artifacts/4b-sft-first-party-teacher-canary-v0.1/"
    "e80b7aa7936b/attempt-018/private/canonical-candidate-v0.1.jsonl"
)
SUCCESSOR_REGISTRY_PATH = "configs/lumi-4b-sft-source-registry-v0.15.json"
RECEIPT_PATH = (
    "evidence/lumi-4b-sft-first-party-teacher-source-promotion-v0.1.json"
)
VERIFICATION_PATH = (
    "evidence/"
    "lumi-4b-sft-first-party-teacher-source-promotion-verification-v0.1.json"
)

EXPECTED_AUTHORIZATION = {
    "candidate_registry_change_authorized": True,
    "canonical_candidate_materialization_authorized": True,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "gpu_authorized": False,
    "network_authorized": False,
    "predecessor_registry_mutation_authorized": False,
    "production_promotion_authorized": False,
    "production_sft_authorized": False,
    "purpose": "exact_teacher_canary_candidate_and_registry_v0_15_only",
    "unrelated_access_authorized": False,
}
EXPECTED_LIMITS = {
    "added_sources": 1,
    "candidate_bytes": CANONICAL_CANDIDATE_BYTES,
    "candidate_records": EXPECTED_RECORDS,
    "maximum_runtime_minutes": 30,
    "predecessor_sources": 15,
    "registry_bytes": SUCCESSOR_REGISTRY_BYTES,
    "successor_sources": 16,
    "train_records": EXPECTED_TRAIN_RECORDS,
    "train_supervised_tokens": EXPECTED_TRAIN_TOKENS,
    "validation_records": EXPECTED_VALIDATION_RECORDS,
    "validation_supervised_tokens": EXPECTED_VALIDATION_TOKENS,
}
EXPECTED_OUTPUT = {
    "canonical_candidate_path": CANONICAL_CANDIDATE_PATH,
    "dataset_seal_permitted": False,
    "predecessor_registry_mutation_permitted": False,
    "production_dataset_promotions": 0,
    "raw_training_text_in_receipts": False,
    "receipt_path": RECEIPT_PATH,
    "successor_registry_path": SUCCESSOR_REGISTRY_PATH,
    "verification_path": VERIFICATION_PATH,
}
EXPECTED_BINDINGS = frozenset(
    {
        "candidate_rows",
        "data_contract",
        "data_quality",
        "implementation",
        "owner_review_receipt",
        "predecessor_registry",
        "preflight_implementation",
        "preflight_plan",
        "preflight_report",
        "preflight_verification",
        "runner",
        "source_registry_implementation",
        "test",
        "verifier",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _read_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSON object required at line {line_number}: {path}")
            values.append(value)
    return values


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _verify_internal_seal(value: Mapping[str, Any], field: str) -> str:
    observed = _require_sha256(value.get(field), field)
    unsigned = {key: item for key, item in value.items() if key != field}
    if observed != sha256_value(unsigned):
        raise ValueError(f"teacher-canary {field} seal mismatch")
    return observed


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
        raise ValueError(f"invalid teacher-canary materialization binding: {name}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.parent.resolve()) or not resolved.is_file():
        raise ValueError(f"teacher-canary materialization binding missing: {name}")
    if file_sha256(resolved) != expected:
        raise ValueError(f"teacher-canary materialization binding changed: {name}")
    return resolved


def load_materialization_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    plan = _read_object(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported teacher-canary materialization plan")
    if plan.get("status") != PLAN_STATUS:
        raise ValueError("teacher-canary materialization plan status changed")
    if plan.get("observed_at") != "2026-08-09":
        raise ValueError("teacher-canary materialization date changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("teacher-canary materialization authorization changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("teacher-canary materialization limits changed")
    if plan.get("output") != EXPECTED_OUTPUT:
        raise ValueError("teacher-canary materialization outputs changed")
    if plan.get("predecessor_registry_sha256") != PREDECESSOR_REGISTRY_SHA256:
        raise ValueError("teacher-canary predecessor identity changed")
    if plan.get("canonical_candidate_sha256") != CANONICAL_CANDIDATE_SHA256:
        raise ValueError("teacher-canary canonical candidate identity changed")
    if plan.get("successor_registry_sha256") != SUCCESSOR_REGISTRY_SHA256:
        raise ValueError("teacher-canary successor registry identity changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("teacher-canary materialization bindings changed")
    for name, specification in bindings.items():
        if not isinstance(specification, Mapping):
            raise ValueError(f"invalid teacher-canary materialization binding: {name}")
        resolve_binding(repository_root, specification, name=name)
    return plan


def verify_materialization_prerequisites(
    *,
    plan: Mapping[str, Any],
    repository_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = Path(repository_root).resolve()
    bindings = plan["bindings"]
    preflight_plan_path = resolve_binding(
        root, bindings["preflight_plan"], name="preflight_plan"
    )
    if file_sha256(preflight_plan_path) != PREFLIGHT_PLAN_FILE_SHA256:
        raise ValueError("teacher-canary preflight plan identity changed")
    preflight_plan = load_promotion_preflight_plan(
        preflight_plan_path,
        repository_root=root,
    )
    if plan.get("source_after") != preflight_plan.get("source_after"):
        raise ValueError("teacher-canary authorized source metadata changed")
    report_path = resolve_binding(
        root, bindings["preflight_report"], name="preflight_report"
    )
    if file_sha256(report_path) != PREFLIGHT_REPORT_FILE_SHA256:
        raise ValueError("teacher-canary approved preflight report changed")
    report = _read_object(report_path)
    validate_report(report)
    if report.get("report_sha256") != PREFLIGHT_REPORT_INTERNAL_SHA256:
        raise ValueError("teacher-canary preflight report seal changed")
    replay = build_promotion_preflight_report(
        preflight_plan,
        plan_file_sha256=file_sha256(preflight_plan_path),
        repository_root=root,
    )
    if replay != report:
        raise ValueError("teacher-canary preflight report no longer replays")

    verification_path = resolve_binding(
        root,
        bindings["preflight_verification"],
        name="preflight_verification",
    )
    if file_sha256(verification_path) != PREFLIGHT_VERIFICATION_FILE_SHA256:
        raise ValueError("teacher-canary approved preflight verification changed")
    verification = _read_object(verification_path)
    validate_preflight_verification(verification)
    if (
        verification.get("verification_sha256")
        != PREFLIGHT_VERIFICATION_INTERNAL_SHA256
        or verification.get("preflight_report_file_sha256")
        != PREFLIGHT_REPORT_FILE_SHA256
        or verification.get("preflight_report_internal_sha256")
        != PREFLIGHT_REPORT_INTERNAL_SHA256
        or verification.get("projected_canonical_candidate_file_sha256")
        != CANONICAL_CANDIDATE_SHA256
        or verification.get("projected_successor_registry_sha256")
        != SUCCESSOR_REGISTRY_SHA256
    ):
        raise ValueError("teacher-canary preflight verification identity changed")

    predecessor_path = resolve_binding(
        root,
        bindings["predecessor_registry"],
        name="predecessor_registry",
    )
    predecessor = _read_object(predecessor_path)
    return preflight_plan, report, predecessor


def reconstruct_materialized_values(
    *,
    plan: Mapping[str, Any],
    repository_root: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    root = Path(repository_root).resolve()
    preflight_plan, report, predecessor = verify_materialization_prerequisites(
        plan=plan,
        repository_root=root,
    )
    candidate_path = resolve_preflight_binding(
        root,
        preflight_plan["bindings"]["candidate_rows"],
        name="candidate_rows",
    )
    canonical = project_canonical_candidate(
        _read_jsonl(candidate_path),
        owner_receipt_file_sha256=EXPECTED_OWNER_RECEIPT_FILE_SHA256,
    )
    canonical_bytes = jsonl_file_bytes(canonical)
    if (
        len(canonical) != EXPECTED_RECORDS
        or len(canonical_bytes) != CANONICAL_CANDIDATE_BYTES
        or hashlib.sha256(canonical_bytes).hexdigest() != CANONICAL_CANDIDATE_SHA256
        or sha256_value(canonical) != CANONICAL_CANDIDATE_INTERNAL_SHA256
    ):
        raise ValueError("teacher-canary canonical candidate no longer reproduces")
    if report["canonical_candidate_projection"] != {
        "bytes": CANONICAL_CANDIDATE_BYTES,
        "file_sha256": CANONICAL_CANDIDATE_SHA256,
        "filter_accepted": EXPECTED_RECORDS,
        "filter_rejected": 0,
        "internal_sha256": CANONICAL_CANDIDATE_INTERNAL_SHA256,
        "materialized": False,
        "path": CANONICAL_CANDIDATE_PATH,
        "records": EXPECTED_RECORDS,
    }:
        raise ValueError("teacher-canary approved candidate projection changed")

    successor = build_projected_successor(
        predecessor,
        source_after=preflight_plan["source_after"],
    )
    successor_bytes = json_file_bytes(successor)
    if (
        len(successor_bytes) != SUCCESSOR_REGISTRY_BYTES
        or hashlib.sha256(successor_bytes).hexdigest() != SUCCESSOR_REGISTRY_SHA256
        or successor.get("sources", [])[:-1] != predecessor.get("sources")
        or successor.get("sources", [])[-1] != preflight_plan["source_after"]
    ):
        raise ValueError("teacher-canary successor registry no longer reproduces")
    if report["projected_successor"]["sha256"] != SUCCESSOR_REGISTRY_SHA256:
        raise ValueError("teacher-canary approved successor projection changed")

    contract_path = resolve_binding(
        root, plan["bindings"]["data_contract"], name="data_contract"
    )
    contract = _read_object(contract_path)
    parsed_sources = tuple(
        SFTSource.from_mapping(
            source,
            allowed_licenses=frozenset(contract["allowed_licenses"]),
            allowed_languages=frozenset(contract["allowed_languages"]),
            allowed_splits=frozenset(contract["allowed_splits"]),
            allowed_tasks=frozenset(contract["task_mix"]),
        )
        for source in successor["sources"]
    )
    if len(parsed_sources) != 16 or parsed_sources[-1].source_id != SOURCE_ID:
        raise ValueError("teacher-canary successor source inventory changed")
    source_registry = SFTSourceRegistry(
        observed_at=str(successor["observed_at"]),
        registry_sha256=SUCCESSOR_REGISTRY_SHA256,
        sources=parsed_sources,
    )
    quality_path = resolve_binding(
        root, plan["bindings"]["data_quality"], name="data_quality"
    )
    quality = _read_object(quality_path)
    benchmark_specification = quality.get("benchmark_decontamination")
    if not isinstance(benchmark_specification, Mapping):
        raise ValueError("teacher-canary benchmark contract changed")
    accepted, filter_report = filter_sft_records(
        canonical,
        policy=load_sft_policy(contract_path),
        benchmark_registry=BenchmarkRegistry(benchmark_specification, root),
        source_registry=source_registry,
    )
    if len(accepted) != EXPECTED_RECORDS or filter_report.get("rejected") != 0:
        raise ValueError("teacher-canary canonical candidate failed final filtering")
    return canonical, successor, report


def _split_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "train": sum(row.get("split") == "train" for row in rows),
        "validation": sum(row.get("split") == "validation" for row in rows),
    }


def validate_canonical_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != EXPECTED_RECORDS or _split_counts(rows) != {
        "train": EXPECTED_TRAIN_RECORDS,
        "validation": EXPECTED_VALIDATION_RECORDS,
    }:
        raise ValueError("teacher-canary canonical row inventory changed")
    for row in rows:
        provenance = row.get("provenance")
        source = row.get("source")
        if (
            row.get("schema_version") != "lumi.sft.v0.1"
            or not isinstance(provenance, Mapping)
            or provenance.get("ai_assisted") is not True
            or provenance.get("human_authored") is not False
            or provenance.get("human_authorship_claimed") is not False
            or provenance.get("owner_review_claim")
            != "ai_generated_plus_owner_reviewed_not_human_written"
            or not isinstance(source, Mapping)
            or source.get("source_id") != SOURCE_ID
        ):
            raise ValueError("teacher-canary canonical provenance changed")


def build_promotion_receipt(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    _require_sha256(plan_file_sha256, "plan_file_sha256")
    receipt: dict[str, Any] = {
        "authorization": dict(EXPECTED_AUTHORIZATION),
        "bindings": {
            "owner_review_receipt_file_sha256": EXPECTED_OWNER_RECEIPT_FILE_SHA256,
            "plan_file_sha256": plan_file_sha256,
            "predecessor_registry_file_sha256": PREDECESSOR_REGISTRY_SHA256,
            "preflight_report_file_sha256": PREFLIGHT_REPORT_FILE_SHA256,
            "preflight_report_internal_sha256": report["report_sha256"],
            "preflight_verification_file_sha256": (
                PREFLIGHT_VERIFICATION_FILE_SHA256
            ),
            "preflight_verification_internal_sha256": (
                PREFLIGHT_VERIFICATION_INTERNAL_SHA256
            ),
        },
        "candidate": {
            "bytes": CANONICAL_CANDIDATE_BYTES,
            "file_sha256": CANONICAL_CANDIDATE_SHA256,
            "internal_sha256": CANONICAL_CANDIDATE_INTERNAL_SHA256,
            "path": CANONICAL_CANDIDATE_PATH,
            "records": EXPECTED_RECORDS,
            "train_records": EXPECTED_TRAIN_RECORDS,
            "train_supervised_tokens": EXPECTED_TRAIN_TOKENS,
            "validation_records": EXPECTED_VALIDATION_RECORDS,
            "validation_supervised_tokens": EXPECTED_VALIDATION_TOKENS,
        },
        "checks": {
            "canonical_candidate_materialized": True,
            "candidate_registry_promotions": 1,
            "data_contract_updated": False,
            "dataset_sealed": False,
            "external_api_used": False,
            "gpu_used": False,
            "human_authorship_claimed": False,
            "network_used": False,
            "predecessor_registry_mutated": False,
            "production_dataset_promotions": 0,
            "raw_training_text_retained": False,
            "sft_started": False,
        },
        "observed_at": "2026-08-09",
        "production_eligible": False,
        "registry": {
            "added_source_id": SOURCE_ID,
            "predecessor_path": "configs/lumi-4b-sft-source-registry-v0.14.json",
            "predecessor_sha256": PREDECESSOR_REGISTRY_SHA256,
            "successor_bytes": SUCCESSOR_REGISTRY_BYTES,
            "successor_path": SUCCESSOR_REGISTRY_PATH,
            "successor_sha256": SUCCESSOR_REGISTRY_SHA256,
            "source_count": 16,
        },
        "remaining_blockers": list(plan["source_after"]["production_blockers"]),
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "verified_teacher_canary_candidate_and_registry_v0_15_nonproduction",
    }
    receipt["receipt_sha256"] = sha256_value(receipt)
    return receipt


def validate_promotion_receipt(receipt: Mapping[str, Any]) -> None:
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported teacher-canary promotion receipt")
    _verify_internal_seal(receipt, "receipt_sha256")
    if receipt.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("teacher-canary receipt authorization changed")
    if (
        receipt.get("status")
        != "verified_teacher_canary_candidate_and_registry_v0_15_nonproduction"
        or receipt.get("production_eligible") is not False
    ):
        raise ValueError("teacher-canary receipt production boundary changed")
    checks = receipt.get("checks")
    if not isinstance(checks, Mapping) or checks != {
        "canonical_candidate_materialized": True,
        "candidate_registry_promotions": 1,
        "data_contract_updated": False,
        "dataset_sealed": False,
        "external_api_used": False,
        "gpu_used": False,
        "human_authorship_claimed": False,
        "network_used": False,
        "predecessor_registry_mutated": False,
        "production_dataset_promotions": 0,
        "raw_training_text_retained": False,
        "sft_started": False,
    }:
        raise ValueError("teacher-canary promotion receipt checks changed")


__all__ = [
    "CANONICAL_CANDIDATE_BYTES",
    "CANONICAL_CANDIDATE_PATH",
    "CANONICAL_CANDIDATE_SHA256",
    "EXPECTED_AUTHORIZATION",
    "EXPECTED_LIMITS",
    "EXPECTED_OUTPUT",
    "PLAN_SCHEMA_VERSION",
    "PREDECESSOR_REGISTRY_SHA256",
    "RECEIPT_SCHEMA_VERSION",
    "SUCCESSOR_REGISTRY_BYTES",
    "SUCCESSOR_REGISTRY_PATH",
    "SUCCESSOR_REGISTRY_SHA256",
    "VERIFICATION_SCHEMA_VERSION",
    "build_promotion_receipt",
    "file_sha256",
    "json_file_bytes",
    "jsonl_file_bytes",
    "load_materialization_plan",
    "reconstruct_materialized_values",
    "resolve_binding",
    "sha256_value",
    "validate_canonical_rows",
    "validate_promotion_receipt",
]
