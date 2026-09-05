"""Exact immutable promotion of verified LiveCodeBench research signatures."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .data_quality import sha256_file
from .posttraining_bigcodebench_signatures import load_registry, sha256_value
from .posttraining_livecodebench_promotion_preflight import (
    deterministic_registry_bytes,
    validate_promotion_plan,
)
from .posttraining_livecodebench_promotion_preflight import (
    validate_report as validate_preflight_report,
)

PLAN_SCHEMA_VERSION = "lumi.livecodebench-signature-registry-promotion-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.livecodebench-signature-registry-promotion.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.livecodebench-signature-registry-promotion-verification.v0.1"
)

OWNER_AUTHORIZATION_STATEMENT = (
    "I approve one bounded CPU-only LiveCodeBench private "
    "research/evaluation/decontamination signature-registry promotion, bound to "
    "promotion plan SHA-256 "
    "3eaf346757a35a8371163194339dc4cb8866d499177c00a13e2c46b7fe341c8e, "
    "preflight report SHA-256 "
    "9765efaa8484f827742e9b4cb29958fade4da0133090f08628977e7d28b9f0a6, "
    "verification SHA-256 "
    "82c8a65107cabc3270197777e5ad026f2165cca058b3f7e870ec8a230e09eec0, "
    "and candidate SHA-256 "
    "ee280477605762c9e9c19e848a0a6139007eba89e76495b15636a984def3ba7f, "
    "maximum 30 minutes. It may materialize exactly "
    "data/benchmark-signatures-v0.3.jsonl.gz (50,944 entries / 6,065,385 bytes / "
    "SHA-256 "
    "1ffb5e938ab28c2fc876b9efd6c6ed59059f1e85aed2c396c024ae5925a4745e) "
    "and configs/data-quality-v0.3.json (806 bytes / SHA-256 "
    "23fba21dbc822c84a5d16e63147fc1a7716576b8f1258cda8c035391f04f52c0), "
    "then independently verify both. This promotes only raw-text-free benchmark "
    "hashes for private decontamination; it does not claim rights clearance or "
    "authorize LiveCodeBench training ingestion. No network/API/GPU/SFT, raw "
    "benchmark text, dataset seal, unrelated access, external distribution, or "
    "mutation of v0.2 files."
)

PROMOTION_PLAN_INTERNAL_SHA256 = (
    "3eaf346757a35a8371163194339dc4cb8866d499177c00a13e2c46b7fe341c8e"
)
PREFLIGHT_REPORT_INTERNAL_SHA256 = (
    "9765efaa8484f827742e9b4cb29958fade4da0133090f08628977e7d28b9f0a6"
)
PREFLIGHT_VERIFICATION_INTERNAL_SHA256 = (
    "82c8a65107cabc3270197777e5ad026f2165cca058b3f7e870ec8a230e09eec0"
)
CANDIDATE_SHA256 = "ee280477605762c9e9c19e848a0a6139007eba89e76495b15636a984def3ba7f"
PREDECESSOR_REGISTRY_SHA256 = (
    "4a9e17b75c0447b19b01c845d5f22cb117cb6c68bcb70b94a7393620ac5a4873"
)
PREDECESSOR_QUALITY_SHA256 = (
    "b2dfabd7e7ba6d6b2bd6600a3f65e43a72a551b02857b6eae42ddbd90187f816"
)
TARGET_REGISTRY_SHA256 = (
    "1ffb5e938ab28c2fc876b9efd6c6ed59059f1e85aed2c396c024ae5925a4745e"
)
TARGET_REGISTRY_BYTES = 6_065_385
TARGET_QUALITY_SHA256 = (
    "23fba21dbc822c84a5d16e63147fc1a7716576b8f1258cda8c035391f04f52c0"
)
TARGET_QUALITY_BYTES = 806
TARGET_ENTRIES = 50_944

EXPECTED_AUTHORIZATION = {
    "active_contract_activation_authorized": False,
    "cpu_only_registry_promotion_authorized": True,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "gpu_authorized": False,
    "network_access_authorized": False,
    "raw_benchmark_text_access_authorized": False,
    "rights_clearance_claimed": False,
    "sft_training_authorized": False,
    "training_ingestion_authorized": False,
    "unrelated_access_authorized": False,
    "v0_2_mutation_authorized": False,
}
EXPECTED_LIMITS = {"maximum_runtime_minutes": 30}
EXPECTED_BINDINGS = {
    "candidate_registry",
    "implementation",
    "predecessor_data_quality_config",
    "predecessor_registry",
    "preflight_report",
    "preflight_verification",
    "promotion_plan",
    "promotion_preflight_implementation",
    "runner",
    "verifier",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


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
        raise ValueError(f"invalid LiveCodeBench promotion binding: {name}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"LiveCodeBench promotion binding missing: {name}")
    if sha256_file(resolved) != expected:
        raise ValueError(f"LiveCodeBench promotion binding changed: {name}")
    return resolved


def load_promotion_authorization(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = _read_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported LiveCodeBench registry-promotion plan")
    if plan.get("status") != "authorized_exact_immutable_research_promotion_no_activation":
        raise ValueError("LiveCodeBench registry-promotion status changed")
    if plan.get("observed_at") != "2026-08-07":
        raise ValueError("LiveCodeBench registry-promotion date changed")
    if plan.get("owner_authorization_statement") != OWNER_AUTHORIZATION_STATEMENT:
        raise ValueError("LiveCodeBench registry-promotion authorization text changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("LiveCodeBench registry-promotion authorization changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("LiveCodeBench registry-promotion limits changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("LiveCodeBench registry-promotion bindings changed")
    for name, specification in bindings.items():
        if not isinstance(specification, Mapping):
            raise ValueError(f"invalid LiveCodeBench registry-promotion binding: {name}")
        resolve_binding(root, specification, name=name)
    if bindings["candidate_registry"]["sha256"] != CANDIDATE_SHA256:
        raise ValueError("LiveCodeBench registry-promotion candidate changed")
    if bindings["predecessor_registry"]["sha256"] != PREDECESSOR_REGISTRY_SHA256:
        raise ValueError("LiveCodeBench predecessor registry changed")
    if bindings["predecessor_data_quality_config"]["sha256"] != (
        PREDECESSOR_QUALITY_SHA256
    ):
        raise ValueError("LiveCodeBench predecessor quality config changed")

    promotion = _read_json(
        resolve_binding(root, bindings["promotion_plan"], name="promotion_plan")
    )
    validate_promotion_plan(promotion)
    if promotion.get("plan_sha256") != PROMOTION_PLAN_INTERNAL_SHA256:
        raise ValueError("LiveCodeBench promotion-plan seal changed")
    if promotion.get("proposed_registry_successor", {}).get("file_sha256") != (
        TARGET_REGISTRY_SHA256
    ):
        raise ValueError("LiveCodeBench target registry binding changed")
    if promotion.get("proposed_data_quality_successor", {}).get("file_sha256") != (
        TARGET_QUALITY_SHA256
    ):
        raise ValueError("LiveCodeBench target quality binding changed")
    report = _read_json(
        resolve_binding(root, bindings["preflight_report"], name="preflight_report")
    )
    validate_preflight_report(report)
    if report.get("report_sha256") != PREFLIGHT_REPORT_INTERNAL_SHA256:
        raise ValueError("LiveCodeBench promotion-preflight report seal changed")
    verification = _read_json(
        resolve_binding(
            root,
            bindings["preflight_verification"],
            name="preflight_verification",
        )
    )
    if verification.get("verification_sha256") != (
        PREFLIGHT_VERIFICATION_INTERNAL_SHA256
    ):
        raise ValueError("LiveCodeBench promotion-preflight verification seal changed")
    if plan.get("output") != {
        "data_quality_successor_path": "configs/data-quality-v0.3.json",
        "promotion_report_path": (
            "evidence/lumi-4b-livecodebench-signature-registry-promotion-v0.1.json"
        ),
        "registry_successor_path": "data/benchmark-signatures-v0.3.jsonl.gz",
        "verification_path": (
            "evidence/lumi-4b-livecodebench-signature-registry-promotion-verification-v0.1.json"
        ),
    }:
        raise ValueError("LiveCodeBench registry-promotion output changed")
    return plan


def build_successor_payloads(
    plan: Mapping[str, Any],
    *,
    repository_root: str | Path,
) -> tuple[bytes, bytes, dict[str, Any]]:
    root = Path(repository_root).resolve()
    promotion = _read_json(
        resolve_binding(
            root,
            plan["bindings"]["promotion_plan"],
            name="promotion_plan",
        )
    )
    candidate_path = resolve_binding(
        root,
        plan["bindings"]["candidate_registry"],
        name="candidate_registry",
    )
    _candidate_metadata, candidate_entries = load_registry(candidate_path)
    registry_specification = promotion["proposed_registry_successor"]
    if len(candidate_entries) != TARGET_ENTRIES:
        raise ValueError("LiveCodeBench promotion candidate count changed")
    registry_bytes = deterministic_registry_bytes(
        registry_specification["metadata"],
        candidate_entries,
    )
    if len(registry_bytes) != TARGET_REGISTRY_BYTES:
        raise ValueError("LiveCodeBench promoted registry size changed")
    if hashlib.sha256(registry_bytes).hexdigest() != TARGET_REGISTRY_SHA256:
        raise ValueError("LiveCodeBench promoted registry SHA-256 changed")
    quality_specification = promotion["proposed_data_quality_successor"]
    quality_content = quality_specification["content"]
    quality_bytes = (
        json.dumps(quality_content, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if len(quality_bytes) != TARGET_QUALITY_BYTES:
        raise ValueError("LiveCodeBench promoted quality-config size changed")
    if hashlib.sha256(quality_bytes).hexdigest() != TARGET_QUALITY_SHA256:
        raise ValueError("LiveCodeBench promoted quality-config SHA-256 changed")
    return registry_bytes, quality_bytes, promotion


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite LiveCodeBench successor: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported LiveCodeBench registry-promotion report")
    seal = _require_sha256(report.get("report_sha256"), "report_sha256")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if sha256_value(unsigned) != seal:
        raise ValueError("LiveCodeBench registry-promotion report seal is invalid")
    if report.get("production_eligible") is not False:
        raise ValueError("LiveCodeBench registry promotion activated production")
    if report.get("boundaries") != {
        "active_contract_activated": False,
        "dataset_sealed": False,
        "external_distribution_performed": False,
        "gpu_or_sft_used": False,
        "network_used": False,
        "raw_benchmark_text_accessed": False,
        "rights_clearance_claimed": False,
        "training_ingestion_performed": False,
        "v0_2_config_mutated": False,
        "v0_2_registry_mutated": False,
    }:
        raise ValueError("LiveCodeBench registry-promotion boundary changed")


__all__ = [
    "OWNER_AUTHORIZATION_STATEMENT",
    "PLAN_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "TARGET_ENTRIES",
    "TARGET_QUALITY_BYTES",
    "TARGET_QUALITY_SHA256",
    "TARGET_REGISTRY_BYTES",
    "TARGET_REGISTRY_SHA256",
    "VERIFICATION_SCHEMA_VERSION",
    "build_successor_payloads",
    "load_promotion_authorization",
    "validate_report",
    "write_bytes_atomic",
]
