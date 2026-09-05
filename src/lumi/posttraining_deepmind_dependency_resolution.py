"""No-install dependency resolution for the pinned DeepMind math adapter."""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import re
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

PLAN_SCHEMA_VERSION = "lumi.sft-deepmind-dependency-resolution-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-deepmind-dependency-resolution-report.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-deepmind-dependency-resolution-verification.v0.1"
)

SOURCE_REVISION = "427f45075f84b8b9774950196ad63867ca20ffb3"
SETUP_OBJECT = {
    "bytes": 2458,
    "content_sha256": "cb2e8a0d3284f1637adae9ab897c2fda3066af56544697d61eef9e0e3bd0e58d",
    "git_blob_sha1": "150dd3f3599ea68a56767f1c669585a765e5d914",
    "path": "setup.py",
    "url": (
        "https://raw.githubusercontent.com/google-deepmind/mathematics_dataset/"
        f"{SOURCE_REVISION}/setup.py"
    ),
}
DECLARED_REQUIREMENTS = ["absl-py>=0.1.0", "numpy>=1.10", "six", "sympy>=1.2"]

PYPI_METADATA = {
    "absl-py": {
        "bytes": 5461,
        "content_sha256": (
            "603b0e5f6c57db1ab049baaa4d37b2b2961b73ae582d20b09db71becff0c8789"
        ),
        "url": "https://pypi.org/pypi/absl-py/2.5.0/json",
        "version": "2.5.0",
        "wheel": {
            "filename": "absl_py-2.5.0-py3-none-any.whl",
            "requires_python": ">=3.10",
            "sha256": (
                "0f17b89f2a4eaaedc4f28c622998aa690564b3012a396a4ffad0821007fe03ba"
            ),
            "size": 137410,
            "url": (
                "https://files.pythonhosted.org/packages/58/0a/"
                "a10b45aab35b175aded078a462dc8d0c698f5b13946e7cb0869097b78bb6/"
                "absl_py-2.5.0-py3-none-any.whl"
            ),
        },
    },
    "six": {
        "bytes": 3904,
        "content_sha256": (
            "d8de39dd52b73048d81a8fd78022ee2bc72f8b568bcf0d88e5d5069c302b1208"
        ),
        "url": "https://pypi.org/pypi/six/1.17.0/json",
        "version": "1.17.0",
        "wheel": {
            "filename": "six-1.17.0-py2.py3-none-any.whl",
            "requires_python": "!=3.0.*,!=3.1.*,!=3.2.*,>=2.7",
            "sha256": (
                "4721f391ed90541fddacab5acf947aa0d3dc7d27b2e1e8eda2be8970586c3274"
            ),
            "size": 11050,
            "url": (
                "https://files.pythonhosted.org/packages/b7/ce/"
                "149a00dd41f10bc29e5921b496af8b574d8413afcd5e30dfa0ed46c2cc5e/"
                "six-1.17.0-py2.py3-none-any.whl"
            ),
        },
    },
}

EXPECTED_AUTHORIZATION = {
    "canary_generation_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_spend_authorized": False,
    "gpu_authorized": False,
    "network_access_authorized": True,
    "package_installation_authorized": False,
    "production_sft_authorized": False,
    "purpose": "pinned_setup_and_pypi_metadata_dependency_resolution_only",
    "source_acquisition_authorized": False,
    "unrelated_data_access_authorized": False,
}
EXPECTED_LIMITS = {
    "exact_network_objects": 3,
    "maximum_network_bytes": 1_048_576,
    "maximum_runtime_minutes": 60,
    "pypi_metadata_objects": 2,
    "setup_objects": 1,
}
EXPECTED_OUTPUT = {
    "raw_source_text_in_report": False,
    "report_path": (
        "evidence/lumi-4b-sft-deepmind-dependency-resolution-v0.1.json"
    ),
    "verification_path": (
        "evidence/lumi-4b-sft-deepmind-dependency-resolution-verification-v0.1.json"
    ),
}
EXPECTED_BINDING_PATHS = {
    "previous_gate_plan": "configs/lumi-4b-sft-deepmind-canary-gate-v0.1.json",
    "previous_gate_report": "evidence/lumi-4b-sft-deepmind-canary-gate-v0.1.json",
    "previous_gate_verification": (
        "evidence/lumi-4b-sft-deepmind-canary-gate-verification-v0.1.json"
    ),
    "resolution_implementation": (
        "src/lumi/posttraining_deepmind_dependency_resolution.py"
    ),
    "resolution_runner": "scripts/resolve_deepmind_dependencies.py",
    "resolution_verifier": "scripts/verify_deepmind_dependency_resolution.py",
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


def git_blob_sha1(payload: bytes) -> str:
    prefix = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(prefix + payload, usedforsecurity=False).hexdigest()


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
        raise ValueError(f"DeepMind dependency binding path is invalid: {name}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.parent):
        raise ValueError(f"DeepMind dependency binding escapes workspace: {name}")
    if not resolved.is_file():
        raise FileNotFoundError(f"DeepMind dependency binding is missing: {name}")
    expected = _require_sha256(binding.get("sha256"), f"bindings.{name}.sha256")
    if file_sha256(resolved) != expected:
        raise ValueError(f"DeepMind dependency binding changed: {name}")
    return resolved


def load_resolution_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    plan = _load_json(path)
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "declared_requirements": DECLARED_REQUIREMENTS,
        "limits": EXPECTED_LIMITS,
        "observed_at": "2026-08-06",
        "output": EXPECTED_OUTPUT,
        "pypi_metadata": PYPI_METADATA,
        "schema_version": PLAN_SCHEMA_VERSION,
        "setup_object": SETUP_OBJECT,
        "source_revision": SOURCE_REVISION,
        "status": "authorized_no_install_dependency_resolution",
    }
    for field, expected_value in expected.items():
        if plan.get(field) != expected_value:
            raise ValueError(f"DeepMind dependency-resolution plan changed: {field}")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(
        EXPECTED_BINDING_PATHS
    ):
        raise ValueError("DeepMind dependency-resolution binding set changed")
    for name, expected_path in EXPECTED_BINDING_PATHS.items():
        binding = bindings[name]
        if not isinstance(binding, Mapping) or binding.get("path") != expected_path:
            raise ValueError(f"DeepMind dependency binding path changed: {name}")
        resolve_binding(repository_root, binding, name=name)
    return plan


def parse_literal_install_requires(payload: bytes) -> list[str]:
    if len(payload) != SETUP_OBJECT["bytes"]:
        raise ValueError("DeepMind setup.py byte count changed")
    if hashlib.sha256(payload).hexdigest() != SETUP_OBJECT["content_sha256"]:
        raise ValueError("DeepMind setup.py SHA-256 changed")
    if git_blob_sha1(payload) != SETUP_OBJECT["git_blob_sha1"]:
        raise ValueError("DeepMind setup.py Git blob changed")
    tree = ast.parse(payload.decode("utf-8"), filename="setup.py")
    setup_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setup"
    ]
    if len(setup_calls) != 1:
        raise ValueError("DeepMind setup.py must contain one literal setup call")
    keyword = next(
        (item for item in setup_calls[0].keywords if item.arg == "install_requires"),
        None,
    )
    if keyword is None:
        raise ValueError("DeepMind setup.py install_requires is missing")
    value = ast.literal_eval(keyword.value)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("DeepMind install_requires is not a literal string list")
    if value != DECLARED_REQUIREMENTS:
        raise ValueError("DeepMind declared requirements changed")
    return value


def validate_pypi_metadata(
    package: str,
    payload: bytes,
) -> dict[str, Any]:
    expected = PYPI_METADATA[package]
    if len(payload) != expected["bytes"]:
        raise ValueError(f"{package} PyPI metadata byte count changed")
    if hashlib.sha256(payload).hexdigest() != expected["content_sha256"]:
        raise ValueError(f"{package} PyPI metadata SHA-256 changed")
    metadata = json.loads(payload)
    info = metadata.get("info", {})
    if info.get("name").lower() != package or info.get("version") != expected["version"]:
        raise ValueError(f"{package} PyPI identity changed")
    wheels = [item for item in metadata.get("urls", []) if item.get("packagetype") == "bdist_wheel"]
    if len(wheels) != 1:
        raise ValueError(f"{package} expected exactly one universal wheel")
    wheel = wheels[0]
    expected_wheel = expected["wheel"]
    observed = {
        "filename": wheel.get("filename"),
        "requires_python": wheel.get("requires_python"),
        "sha256": wheel.get("digests", {}).get("sha256"),
        "size": wheel.get("size"),
        "url": wheel.get("url"),
    }
    if observed != expected_wheel or wheel.get("yanked") is not False:
        raise ValueError(f"{package} wheel metadata changed")
    if info.get("requires_dist") not in (None, []):
        raise ValueError(f"{package} unexpected transitive dependencies")
    return {
        "metadata_content_sha256": expected["content_sha256"],
        "metadata_url": expected["url"],
        "name": package,
        "requires_dist": [],
        "version": expected["version"],
        "wheel": dict(expected_wheel),
        "yanked": False,
    }


def audit_runtime(
    *,
    find_spec: Callable[[str], Any] = importlib.util.find_spec,
    distribution_version: Callable[[str], str] = importlib.metadata.version,
    executable: str | None = None,
    python_version: str | None = None,
) -> dict[str, Any]:
    packages = [
        ("absl-py", "absl"),
        ("numpy", "numpy"),
        ("six", "six"),
        ("sympy", "sympy"),
    ]
    observed = []
    for distribution, module in packages:
        try:
            present = find_spec(module) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            present = False
        version = None
        if present:
            try:
                version = distribution_version(distribution)
            except importlib.metadata.PackageNotFoundError:
                present = False
        observed.append(
            {
                "distribution": distribution,
                "module": module,
                "present": present,
                "version": version,
            }
        )
    return {
        "executable": executable or sys.executable,
        "packages": observed,
        "platform": platform.platform(),
        "python_version": python_version or platform.python_version(),
    }


def build_resolution_report(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    setup_payload: bytes,
    pypi_payloads: Mapping[str, bytes],
    runtime_audit: Mapping[str, Any],
) -> dict[str, Any]:
    requirements = parse_literal_install_requires(setup_payload)
    metadata = {
        package: validate_pypi_metadata(package, pypi_payloads[package])
        for package in sorted(PYPI_METADATA)
    }
    runtime_by_distribution = {
        item["distribution"]: item for item in runtime_audit["packages"]
    }
    python_version = Version(str(runtime_audit["python_version"]))
    resolved = []
    for requirement_text in requirements:
        requirement = Requirement(requirement_text)
        canonical_name = requirement.name.lower()
        runtime = runtime_by_distribution[canonical_name]
        if runtime["present"]:
            version = Version(runtime["version"])
            if requirement.specifier and version not in requirement.specifier:
                raise ValueError(f"installed {canonical_name} violates setup.py")
            resolved.append(
                {
                    "declared_requirement": requirement_text,
                    "install_required": False,
                    "name": canonical_name,
                    "selected_version": runtime["version"],
                    "source": "preinstalled_isolated_runtime",
                }
            )
            continue
        selected = metadata[canonical_name]
        selected_version = Version(selected["version"])
        if requirement.specifier and selected_version not in requirement.specifier:
            raise ValueError(f"selected {canonical_name} violates setup.py")
        if python_version not in SpecifierSet(selected["wheel"]["requires_python"]):
            raise ValueError(f"selected {canonical_name} wheel rejects runtime Python")
        resolved.append(
            {
                "declared_requirement": requirement_text,
                "install_required": True,
                "name": canonical_name,
                "requires_dist": [],
                "selected_version": selected["version"],
                "source": "pypi_universal_wheel_lock_proposal",
                "wheel": selected["wheel"],
            }
        )
    report: dict[str, Any] = {
        "authorization": dict(EXPECTED_AUTHORIZATION),
        "decision": "exact_two_wheel_no_deps_install_proposal_ready_not_authorized",
        "declared_requirements": requirements,
        "inputs": {
            "network_bytes": SETUP_OBJECT["bytes"]
            + sum(item["bytes"] for item in PYPI_METADATA.values()),
            "network_objects": 3,
            "pypi_metadata": metadata,
            "setup_object": dict(SETUP_OBJECT),
        },
        "next_gate": {
            "canary_retry_requires_separate_authorization": True,
            "install_command_policy": ["--no-deps", "--require-hashes"],
            "installation_authorized": False,
            "maximum_wheel_bytes": sum(
                item["wheel"]["size"] for item in metadata.values()
            ),
            "packages_to_install": [
                f"{item['name']}=={item['selected_version']}"
                for item in resolved
                if item["install_required"]
            ],
        },
        "observed_at": plan["observed_at"],
        "plan_file_sha256": _require_sha256(plan_file_sha256, "plan_file_sha256"),
        "production": {
            "canary_records_generated": 0,
            "dataset_sealed": False,
            "packages_installed": 0,
            "production_sft_authorized": False,
        },
        "raw_source_text_present": False,
        "resolved_dependencies": resolved,
        "runtime_audit": dict(runtime_audit),
        "schema_version": REPORT_SCHEMA_VERSION,
        "source_revision": SOURCE_REVISION,
        "status": "verified_no_install_dependency_lock_proposal",
    }
    report["report_sha256"] = sha256_value(report)
    return report


def validate_resolution_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported DeepMind dependency-resolution report")
    if report.get("status") != "verified_no_install_dependency_lock_proposal":
        raise ValueError("DeepMind dependency-resolution status changed")
    if report.get("production") != {
        "canary_records_generated": 0,
        "dataset_sealed": False,
        "packages_installed": 0,
        "production_sft_authorized": False,
    }:
        raise ValueError("DeepMind no-install boundary changed")
    if report.get("raw_source_text_present") is not False:
        raise ValueError("DeepMind dependency report contains raw source text")
    install = [
        item for item in report.get("resolved_dependencies", []) if item.get("install_required")
    ]
    if [f"{item['name']}=={item['selected_version']}" for item in install] != [
        "absl-py==2.5.0",
        "six==1.17.0",
    ]:
        raise ValueError("DeepMind exact install proposal changed")
    seal = _require_sha256(report.get("report_sha256"), "report_sha256")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if sha256_value(unsigned) != seal:
        raise ValueError("DeepMind dependency-resolution report seal changed")
