"""Fail-closed pre-network gate for the pinned DeepMind Mathematics canary."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

PLAN_SCHEMA_VERSION = "lumi.sft-deepmind-canary-gate-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-deepmind-canary-gate-report.v0.1"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-deepmind-canary-gate-verification.v0.1"

SOURCE_REVISION = "427f45075f84b8b9774950196ad63867ca20ffb3"
SOURCE_OBJECTS = 30
SOURCE_BYTES = 234_672
SOURCE_INVENTORY_SHA256 = (
    "47e472f97f7092cf264edb727bd0821db59869133f56b12cddf3f8a4ef1a6ce9"
)

EXPECTED_AUTHORIZATION = {
    "candidate_registry_promotion_authorized": False,
    "dataset_acquisition_authorized": True,
    "dataset_seal_authorized": False,
    "external_api_spend_authorized": False,
    "external_distribution_authorized": False,
    "generator_execution_authorized": True,
    "network_access_authorized": True,
    "official_test_or_evaluation_generation_authorized": False,
    "package_installation_authorized": False,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "purpose": "bounded_deepmind_source_acquisition_and_adapter_canary",
    "unrelated_data_access_authorized": False,
}
EXPECTED_LIMITS = {
    "exact_source_bytes": SOURCE_BYTES,
    "exact_source_objects": SOURCE_OBJECTS,
    "maximum_canary_generated_records": 2_000,
    "maximum_decoded_bytes": 8_388_608,
    "maximum_network_bytes": 1_048_576,
    "maximum_runtime_minutes": 180,
}
EXPECTED_OUTPUT = {
    "private_canary_data_permitted_only_after_dependency_gate": True,
    "raw_training_text_in_report": False,
    "report_path": "evidence/lumi-4b-sft-deepmind-canary-gate-v0.1.json",
    "verification_path": (
        "evidence/lumi-4b-sft-deepmind-canary-gate-verification-v0.1.json"
    ),
}
REQUIRED_MODULES = [
    {"distribution": "absl-py", "module": "absl"},
    {"distribution": "numpy", "module": "numpy"},
    {"distribution": "six", "module": "six"},
    {"distribution": "sympy", "module": "sympy"},
]
EXPECTED_BINDING_PATHS = {
    "adapter_preflight_plan": (
        "configs/lumi-4b-sft-deepmind-adapter-preflight-v0.1.json"
    ),
    "adapter_preflight_report": (
        "evidence/lumi-4b-sft-deepmind-adapter-preflight-v0.1.json"
    ),
    "adapter_preflight_verification": (
        "evidence/lumi-4b-sft-deepmind-adapter-preflight-verification-v0.1.json"
    ),
    "dmath_promotion_receipt": (
        "evidence/lumi-4b-sft-dmath-source-promotion-v0.1.json"
    ),
    "dmath_promotion_verification": (
        "evidence/lumi-4b-sft-dmath-source-promotion-verification-v0.1.json"
    ),
    "dmath_successor_registry": "configs/lumi-4b-sft-source-registry-v0.5.json",
    "gate_implementation": "src/lumi/posttraining_deepmind_canary_gate.py",
    "gate_runner": "scripts/run_deepmind_canary_gate.py",
    "gate_verifier": "scripts/verify_deepmind_canary_gate.py",
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_file_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a SHA-256")
    return value


def resolve_binding(
    repository_root: str | Path,
    binding: Mapping[str, Any],
    *,
    name: str,
) -> Path:
    root = Path(repository_root).resolve()
    relative = binding.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"DeepMind canary-gate binding path is invalid: {name}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.parent):
        raise ValueError(f"DeepMind canary-gate binding escapes workspace: {name}")
    if not resolved.is_file():
        raise FileNotFoundError(f"DeepMind canary-gate binding is missing: {name}")
    expected = _require_sha256(binding.get("sha256"), f"bindings.{name}.sha256")
    if file_sha256(resolved) != expected:
        raise ValueError(f"DeepMind canary-gate binding changed: {name}")
    return resolved


def load_canary_gate_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    plan = _load_json(path)
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "dependency_gate": {
            "fail_closed_before_network_if_any_required_module_missing": True,
            "required_modules": REQUIRED_MODULES,
        },
        "limits": EXPECTED_LIMITS,
        "observed_at": "2026-08-06",
        "output": EXPECTED_OUTPUT,
        "schema_version": PLAN_SCHEMA_VERSION,
        "source": {
            "declared_license_id": "apache-2.0",
            "exact_object_bytes": SOURCE_BYTES,
            "exact_object_count": SOURCE_OBJECTS,
            "exact_object_inventory_sha256": SOURCE_INVENTORY_SHA256,
            "repository_url": "https://github.com/google-deepmind/mathematics_dataset",
            "revision": SOURCE_REVISION,
            "source_id": "deepmind_mathematics_dataset",
        },
        "status": "authorized_dependency_gated_source_acquisition_and_canary",
    }
    for field, value in expected.items():
        if plan.get(field) != value:
            raise ValueError(f"DeepMind canary-gate plan changed: {field}")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(
        EXPECTED_BINDING_PATHS
    ):
        raise ValueError("DeepMind canary-gate binding set changed")
    for name, expected_path in EXPECTED_BINDING_PATHS.items():
        binding = bindings[name]
        if not isinstance(binding, Mapping) or binding.get("path") != expected_path:
            raise ValueError(f"DeepMind canary-gate binding path changed: {name}")
        resolve_binding(repository_root, binding, name=name)
    return plan


def audit_runtime_dependencies(
    *,
    requirements: Sequence[Mapping[str, str]] = REQUIRED_MODULES,
    find_spec: Callable[[str], Any] = importlib.util.find_spec,
    distribution_version: Callable[[str], str] = importlib.metadata.version,
    executable: str | None = None,
    python_version: str | None = None,
) -> dict[str, Any]:
    modules: list[dict[str, Any]] = []
    for requirement in requirements:
        module = requirement["module"]
        distribution = requirement["distribution"]
        try:
            present = find_spec(module) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            present = False
        version: str | None = None
        if present:
            try:
                version = distribution_version(distribution)
            except importlib.metadata.PackageNotFoundError:
                present = False
        modules.append(
            {
                "distribution": distribution,
                "module": module,
                "present": present,
                "version": version,
            }
        )
    missing = [item["module"] for item in modules if not item["present"]]
    return {
        "all_required_modules_available": not missing,
        "executable": executable or sys.executable,
        "missing_modules": missing,
        "modules": modules,
        "platform": platform.platform(),
        "python_version": python_version or platform.python_version(),
    }


def build_blocked_gate_report(
    *,
    dependency_audit: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_file_sha256: str,
) -> dict[str, Any]:
    missing = dependency_audit.get("missing_modules")
    if not isinstance(missing, list) or not missing:
        raise ValueError("blocked DeepMind gate requires a missing dependency")
    report: dict[str, Any] = {
        "authorization": dict(EXPECTED_AUTHORIZATION),
        "decision": "stop_fail_closed_before_source_acquisition",
        "dependency_audit": dict(dependency_audit),
        "execution": {
            "canary_generation_attempted": False,
            "decoded_bytes": 0,
            "generated_records": 0,
            "network_bytes": 0,
            "package_installation_attempted": False,
            "source_acquisition_attempted": False,
            "source_objects_fetched": 0,
        },
        "limits": dict(EXPECTED_LIMITS),
        "next_gate": {
            "dependency_resolution_or_preprovisioned_runtime_required": True,
            "missing_distributions": [
                item["distribution"]
                for item in dependency_audit["modules"]
                if not item["present"]
            ],
            "new_explicit_authorization_required": True,
        },
        "observed_at": plan["observed_at"],
        "plan_file_sha256": _require_sha256(plan_file_sha256, "plan_file_sha256"),
        "production": {
            "candidate_registry_promotions": 0,
            "dataset_sealed": False,
            "production_eligible": False,
            "production_sft_authorized": False,
        },
        "raw_training_text_present": False,
        "schema_version": REPORT_SCHEMA_VERSION,
        "source": dict(plan["source"]),
        "status": "blocked_pre_network_missing_runtime_dependencies",
    }
    report["report_sha256"] = sha256_value(report)
    return report


def validate_blocked_gate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported DeepMind canary-gate report")
    if report.get("status") != "blocked_pre_network_missing_runtime_dependencies":
        raise ValueError("DeepMind canary-gate status changed")
    audit = report.get("dependency_audit")
    if not isinstance(audit, Mapping) or audit.get(
        "all_required_modules_available"
    ) is not False:
        raise ValueError("DeepMind dependency blocker is not proven")
    if not audit.get("missing_modules"):
        raise ValueError("DeepMind missing dependency set is empty")
    if report.get("execution") != {
        "canary_generation_attempted": False,
        "decoded_bytes": 0,
        "generated_records": 0,
        "network_bytes": 0,
        "package_installation_attempted": False,
        "source_acquisition_attempted": False,
        "source_objects_fetched": 0,
    }:
        raise ValueError("DeepMind fail-closed execution boundary changed")
    if report.get("production") != {
        "candidate_registry_promotions": 0,
        "dataset_sealed": False,
        "production_eligible": False,
        "production_sft_authorized": False,
    }:
        raise ValueError("DeepMind production boundary changed")
    if report.get("raw_training_text_present") is not False:
        raise ValueError("DeepMind gate report contains raw training text")
    seal = _require_sha256(report.get("report_sha256"), "report_sha256")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if sha256_value(unsigned) != seal:
        raise ValueError("DeepMind canary-gate report seal changed")
