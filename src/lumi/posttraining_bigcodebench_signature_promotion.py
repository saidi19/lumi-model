"""Exact immutable promotion of the verified BigCodeBench signature registry."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .data_quality import sha256_file
from .posttraining_bigcodebench_promotion_preflight import (
    deterministic_registry_bytes,
    validate_promotion_plan,
)
from .posttraining_bigcodebench_signatures import load_registry, sha256_value

PLAN_SCHEMA_VERSION = "lumi.bigcodebench-signature-registry-promotion-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.bigcodebench-signature-registry-promotion.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.bigcodebench-signature-registry-promotion-verification.v0.1"
)

OWNER_AUTHORIZATION_STATEMENT = (
    "I approve one bounded CPU-only BigCodeBench benchmark-signature registry "
    "promotion, bound to promotion plan SHA-256 "
    "c4b605cded3cb834636a47b51d807dce983b0f8d0574a6f71a49e3f1be8e08cd, "
    "preflight report SHA-256 "
    "5382d84d75308cd83b2d4eebf1163dcb30d42864218d21c7fad45ca673b5f492, "
    "verification SHA-256 "
    "6561069a59ffe66f5d5751f0479113de2bf9c8d23c3a0a00c3b738f7cdcff3c0, "
    "and candidate SHA-256 "
    "3e00fe6f8960bc3dd328b17814ad87b7e69750447cc7fdbd4c14b7f4adfb2b26, "
    "maximum 30 minutes. It may materialize exactly "
    "data/benchmark-signatures-v0.2.jsonl.gz (50,419 entries / 6,026,608 bytes / "
    "SHA-256 "
    "4a9e17b75c0447b19b01c845d5f22cb117cb6c68bcb70b94a7393620ac5a4873) "
    "and configs/data-quality-v0.2.json (806 bytes / SHA-256 "
    "b2dfabd7e7ba6d6b2bd6600a3f65e43a72a551b02857b6eae42ddbd90187f816), "
    "then independently verify both. It must not modify v0.1 files or existing SFT "
    "configs, activate production, access network/API/GPU, run SFT, access "
    "LiveCodeBench, seal a dataset, access unrelated data, or distribute externally."
)

PROMOTION_PLAN_INTERNAL_SHA256 = (
    "c4b605cded3cb834636a47b51d807dce983b0f8d0574a6f71a49e3f1be8e08cd"
)
PREFLIGHT_REPORT_INTERNAL_SHA256 = (
    "5382d84d75308cd83b2d4eebf1163dcb30d42864218d21c7fad45ca673b5f492"
)
PREFLIGHT_VERIFICATION_INTERNAL_SHA256 = (
    "6561069a59ffe66f5d5751f0479113de2bf9c8d23c3a0a00c3b738f7cdcff3c0"
)
CANDIDATE_SHA256 = "3e00fe6f8960bc3dd328b17814ad87b7e69750447cc7fdbd4c14b7f4adfb2b26"
PREDECESSOR_REGISTRY_SHA256 = (
    "be02bdbd7f2c0e807c86cf9678eff86d711aaa34ffdc8e6ebb687a26220ffcbb"
)
PREDECESSOR_QUALITY_SHA256 = (
    "b271bddd603949ac7577b7bb81ab4d1f0ed4b7cb6e8a2c7ba91956b1237430cc"
)
TARGET_REGISTRY_SHA256 = (
    "4a9e17b75c0447b19b01c845d5f22cb117cb6c68bcb70b94a7393620ac5a4873"
)
TARGET_REGISTRY_BYTES = 6_026_608
TARGET_QUALITY_SHA256 = (
    "b2dfabd7e7ba6d6b2bd6600a3f65e43a72a551b02857b6eae42ddbd90187f816"
)
TARGET_QUALITY_BYTES = 806
TARGET_ENTRIES = 50_419

EXPECTED_AUTHORIZATION = {
    "active_contract_activation_authorized": False,
    "cpu_only_registry_promotion_authorized": True,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "gpu_authorized": False,
    "livecodebench_access_authorized": False,
    "network_access_authorized": False,
    "sft_training_authorized": False,
    "unrelated_access_authorized": False,
    "v0_1_mutation_authorized": False,
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
        raise ValueError(f"invalid BigCodeBench promotion binding: {name}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.parent.resolve()) or not resolved.is_file():
        raise ValueError(f"BigCodeBench promotion binding missing: {name}")
    if sha256_file(resolved) != expected:
        raise ValueError(f"BigCodeBench promotion binding changed: {name}")
    return resolved


def load_promotion_authorization(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = _read_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported BigCodeBench registry-promotion plan")
    if plan.get("status") != "authorized_exact_immutable_promotion_no_activation":
        raise ValueError("BigCodeBench registry-promotion status changed")
    if plan.get("observed_at") != "2026-08-07":
        raise ValueError("BigCodeBench registry-promotion date changed")
    if plan.get("owner_authorization_statement") != OWNER_AUTHORIZATION_STATEMENT:
        raise ValueError("BigCodeBench registry-promotion authorization text changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("BigCodeBench registry-promotion authorization changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("BigCodeBench registry-promotion limits changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("BigCodeBench registry-promotion bindings changed")
    for name, specification in bindings.items():
        if not isinstance(specification, Mapping):
            raise ValueError(f"invalid BigCodeBench registry-promotion binding: {name}")
        resolve_binding(root, specification, name=name)
    if bindings["candidate_registry"]["sha256"] != CANDIDATE_SHA256:
        raise ValueError("BigCodeBench registry-promotion candidate changed")
    if bindings["predecessor_registry"]["sha256"] != PREDECESSOR_REGISTRY_SHA256:
        raise ValueError("BigCodeBench predecessor registry changed")
    if bindings["predecessor_data_quality_config"]["sha256"] != (
        PREDECESSOR_QUALITY_SHA256
    ):
        raise ValueError("BigCodeBench predecessor quality config changed")
    promotion = _read_json(
        resolve_binding(root, bindings["promotion_plan"], name="promotion_plan")
    )
    validate_promotion_plan(promotion)
    if promotion.get("plan_sha256") != PROMOTION_PLAN_INTERNAL_SHA256:
        raise ValueError("BigCodeBench promotion-plan seal changed")
    report = _read_json(
        resolve_binding(root, bindings["preflight_report"], name="preflight_report")
    )
    if report.get("report_sha256") != PREFLIGHT_REPORT_INTERNAL_SHA256:
        raise ValueError("BigCodeBench promotion-preflight report seal changed")
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
        raise ValueError("BigCodeBench promotion-preflight verification seal changed")
    if plan.get("output") != {
        "data_quality_successor_path": "configs/data-quality-v0.2.json",
        "promotion_report_path": (
            "evidence/lumi-4b-bigcodebench-signature-registry-promotion-v0.1.json"
        ),
        "registry_successor_path": "data/benchmark-signatures-v0.2.jsonl.gz",
        "verification_path": (
            "evidence/lumi-4b-bigcodebench-signature-registry-promotion-verification-v0.1.json"
        ),
    }:
        raise ValueError("BigCodeBench registry-promotion output changed")
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
        raise ValueError("BigCodeBench promotion candidate count changed")
    registry_bytes = deterministic_registry_bytes(
        registry_specification["metadata"],
        candidate_entries,
    )
    if len(registry_bytes) != TARGET_REGISTRY_BYTES:
        raise ValueError("BigCodeBench promoted registry size changed")
    if hashlib.sha256(registry_bytes).hexdigest() != TARGET_REGISTRY_SHA256:
        raise ValueError("BigCodeBench promoted registry SHA-256 changed")
    quality_specification = promotion["proposed_data_quality_successor"]
    quality_content = quality_specification["content"]
    quality_bytes = (
        json.dumps(quality_content, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if len(quality_bytes) != TARGET_QUALITY_BYTES:
        raise ValueError("BigCodeBench promoted quality-config size changed")
    if hashlib.sha256(quality_bytes).hexdigest() != TARGET_QUALITY_SHA256:
        raise ValueError("BigCodeBench promoted quality-config SHA-256 changed")
    return registry_bytes, quality_bytes, promotion


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite BigCodeBench successor: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported BigCodeBench registry-promotion report")
    seal = _require_sha256(report.get("report_sha256"), "report_sha256")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if sha256_value(unsigned) != seal:
        raise ValueError("BigCodeBench registry-promotion report seal is invalid")
    if report.get("production_eligible") is not False:
        raise ValueError("BigCodeBench registry promotion activated production")
    if report.get("boundaries") != {
        "active_contract_activated": False,
        "dataset_sealed": False,
        "gpu_or_sft_used": False,
        "livecodebench_accessed": False,
        "network_used": False,
        "v0_1_config_mutated": False,
        "v0_1_registry_mutated": False,
    }:
        raise ValueError("BigCodeBench registry-promotion boundary changed")


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
