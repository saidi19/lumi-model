"""Exact, isolated dependency-overlay installation for the DeepMind canary."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PLAN_SCHEMA_VERSION = "lumi.sft-deepmind-dependency-overlay-plan.v0.1"
RECEIPT_SCHEMA_VERSION = "lumi.sft-deepmind-dependency-overlay-receipt.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-deepmind-dependency-overlay-verification.v0.1"
)

OVERLAY_PATH = "/private/tmp/lumi-deepmind-runtime-v0.1"
WHEELHOUSE_PATH = "/private/tmp/lumi-deepmind-wheelhouse-v0.1"
WHEELS = [
    {
        "distribution": "absl-py",
        "filename": "absl_py-2.5.0-py3-none-any.whl",
        "module": "absl",
        "sha256": "0f17b89f2a4eaaedc4f28c622998aa690564b3012a396a4ffad0821007fe03ba",
        "size": 137410,
        "url": (
            "https://files.pythonhosted.org/packages/58/0a/"
            "a10b45aab35b175aded078a462dc8d0c698f5b13946e7cb0869097b78bb6/"
            "absl_py-2.5.0-py3-none-any.whl"
        ),
        "version": "2.5.0",
    },
    {
        "distribution": "six",
        "filename": "six-1.17.0-py2.py3-none-any.whl",
        "module": "six",
        "sha256": "4721f391ed90541fddacab5acf947aa0d3dc7d27b2e1e8eda2be8970586c3274",
        "size": 11050,
        "url": (
            "https://files.pythonhosted.org/packages/b7/ce/"
            "149a00dd41f10bc29e5921b496af8b574d8413afcd5e30dfa0ed46c2cc5e/"
            "six-1.17.0-py2.py3-none-any.whl"
        ),
        "version": "1.17.0",
    },
]
WHEEL_BYTES = 148460

EXPECTED_AUTHORIZATION = {
    "canary_generation_authorized": False,
    "dataset_seal_authorized": False,
    "dependency_overlay_installation_authorized": True,
    "external_api_spend_authorized": False,
    "external_distribution_authorized": False,
    "gpu_authorized": False,
    "import_version_smoke_authorized": True,
    "network_access_authorized": True,
    "production_sft_authorized": False,
    "purpose": "exact_two_wheel_isolated_dependency_overlay_and_smoke_only",
    "source_acquisition_authorized": False,
    "unrelated_data_or_package_access_authorized": False,
}
EXPECTED_LIMITS = {
    "exact_wheel_bytes": WHEEL_BYTES,
    "exact_wheels": 2,
    "maximum_installed_bytes": 16_777_216,
    "maximum_network_bytes": 1_048_576,
    "maximum_runtime_minutes": 30,
}
EXPECTED_INSTALL_POLICY = [
    "--no-cache-dir",
    "--no-compile",
    "--no-deps",
    "--no-index",
    "--require-hashes",
    "--target",
]
EXPECTED_OUTPUT = {
    "overlay_path": OVERLAY_PATH,
    "receipt_path": "evidence/lumi-4b-sft-deepmind-dependency-overlay-v0.1.json",
    "verification_path": (
        "evidence/lumi-4b-sft-deepmind-dependency-overlay-verification-v0.1.json"
    ),
    "wheelhouse_path": WHEELHOUSE_PATH,
}
EXPECTED_BINDING_PATHS = {
    "dependency_lock": "configs/lumi-4b-sft-deepmind-dependency-lock-v0.1.txt",
    "dependency_resolution_plan": (
        "configs/lumi-4b-sft-deepmind-dependency-resolution-v0.1.json"
    ),
    "dependency_resolution_report": (
        "evidence/lumi-4b-sft-deepmind-dependency-resolution-v0.1.json"
    ),
    "dependency_resolution_verification": (
        "evidence/lumi-4b-sft-deepmind-dependency-resolution-verification-v0.1.json"
    ),
    "overlay_implementation": (
        "src/lumi/posttraining_deepmind_dependency_overlay.py"
    ),
    "overlay_runner": "scripts/install_deepmind_dependency_overlay.py",
    "overlay_verifier": "scripts/verify_deepmind_dependency_overlay.py",
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
        raise ValueError(f"DeepMind overlay binding path is invalid: {name}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.parent):
        raise ValueError(f"DeepMind overlay binding escapes workspace: {name}")
    if not resolved.is_file():
        raise FileNotFoundError(f"DeepMind overlay binding is missing: {name}")
    expected = _require_sha256(binding.get("sha256"), f"bindings.{name}.sha256")
    if file_sha256(resolved) != expected:
        raise ValueError(f"DeepMind overlay binding changed: {name}")
    return resolved


def load_overlay_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    plan = _load_json(path)
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "install_policy": EXPECTED_INSTALL_POLICY,
        "limits": EXPECTED_LIMITS,
        "observed_at": "2026-08-06",
        "output": EXPECTED_OUTPUT,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "authorized_exact_dependency_overlay_installation",
        "wheels": WHEELS,
    }
    for field, expected_value in expected.items():
        if plan.get(field) != expected_value:
            raise ValueError(f"DeepMind dependency-overlay plan changed: {field}")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(
        EXPECTED_BINDING_PATHS
    ):
        raise ValueError("DeepMind dependency-overlay binding set changed")
    for name, expected_path in EXPECTED_BINDING_PATHS.items():
        binding = bindings[name]
        if not isinstance(binding, Mapping) or binding.get("path") != expected_path:
            raise ValueError(f"DeepMind overlay binding path changed: {name}")
        resolve_binding(repository_root, binding, name=name)
    return plan


def validate_wheelhouse(path: str | Path) -> list[dict[str, Any]]:
    wheelhouse = Path(path).resolve()
    files = sorted(item for item in wheelhouse.iterdir() if item.is_file())
    expected_names = sorted(item["filename"] for item in WHEELS)
    if [item.name for item in files] != expected_names:
        raise ValueError("DeepMind wheelhouse file set changed")
    observed = []
    expected_by_name = {item["filename"]: item for item in WHEELS}
    for wheel_path in files:
        expected = expected_by_name[wheel_path.name]
        size = wheel_path.stat().st_size
        sha256 = file_sha256(wheel_path)
        if size != expected["size"] or sha256 != expected["sha256"]:
            raise ValueError(f"DeepMind wheel changed: {wheel_path.name}")
        observed.append({"filename": wheel_path.name, "sha256": sha256, "size": size})
    if sum(item["size"] for item in observed) != WHEEL_BYTES:
        raise ValueError("DeepMind wheel byte total changed")
    return observed


def install_overlay(
    *,
    lock_path: Path,
    overlay_path: Path,
    python_executable: str,
    wheelhouse_path: Path,
) -> list[str]:
    if overlay_path.exists():
        raise FileExistsError(f"refusing to overwrite DeepMind overlay: {overlay_path}")
    validate_wheelhouse(wheelhouse_path)
    command = [
        python_executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-cache-dir",
        "--no-compile",
        "--no-deps",
        "--no-index",
        "--require-hashes",
        "--target",
        str(overlay_path),
        "--find-links",
        str(wheelhouse_path),
        "-r",
        str(lock_path),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"DeepMind overlay installation failed: {completed.stderr}")
    return command


def overlay_inventory(path: str | Path) -> dict[str, Any]:
    overlay = Path(path).resolve()
    if not overlay.is_dir():
        raise FileNotFoundError("DeepMind overlay is missing")
    files = []
    for item in sorted(candidate for candidate in overlay.rglob("*") if candidate.is_file()):
        relative = item.relative_to(overlay).as_posix()
        files.append(
            {
                "path": relative,
                "sha256": file_sha256(item),
                "size": item.stat().st_size,
            }
        )
    total_bytes = sum(item["size"] for item in files)
    if total_bytes > EXPECTED_LIMITS["maximum_installed_bytes"]:
        raise ValueError("DeepMind overlay exceeds installed-byte ceiling")
    distributions = sorted(
        {
            (
                distribution.metadata["Name"].lower(),
                distribution.version,
            )
            for distribution in importlib.metadata.distributions(path=[str(overlay)])
        }
    )
    if distributions != [("absl-py", "2.5.0"), ("six", "1.17.0")]:
        raise ValueError("DeepMind overlay distribution set changed")
    return {
        "distribution_versions": [
            {"name": name, "version": version} for name, version in distributions
        ],
        "file_count": len(files),
        "file_inventory_sha256": sha256_value(files),
        "installed_bytes": total_bytes,
    }


def run_import_smoke(
    *,
    overlay_path: Path,
    python_executable: str,
) -> dict[str, Any]:
    code = """
import importlib.metadata as metadata
import json
from pathlib import Path
import absl
import six

overlay = Path(__import__('os').environ['LUMI_DEEPMIND_OVERLAY']).resolve()
paths = {'absl': Path(absl.__file__).resolve(), 'six': Path(six.__file__).resolve()}
if not all(path.is_relative_to(overlay) for path in paths.values()):
    raise SystemExit('overlay import escaped target')
print(json.dumps({
    'imports': sorted(paths),
    'paths_within_overlay': True,
    'versions': {
        'absl-py': metadata.version('absl-py'),
        'six': metadata.version('six'),
    },
}, sort_keys=True))
"""
    environment = os.environ.copy()
    environment["LUMI_DEEPMIND_OVERLAY"] = str(overlay_path)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(overlay_path)
    completed = subprocess.run(
        [python_executable, "-c", code],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"DeepMind overlay import smoke failed: {completed.stderr}")
    result = json.loads(completed.stdout)
    expected = {
        "imports": ["absl", "six"],
        "paths_within_overlay": True,
        "versions": {"absl-py": "2.5.0", "six": "1.17.0"},
    }
    if result != expected:
        raise ValueError("DeepMind overlay import smoke changed")
    return result


def build_overlay_receipt(
    *,
    command: Sequence[str],
    inventory: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    smoke: Mapping[str, Any],
    wheel_inventory: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    policy_flags = sorted(flag for flag in EXPECTED_INSTALL_POLICY if flag in command)
    if policy_flags != sorted(EXPECTED_INSTALL_POLICY):
        raise ValueError("DeepMind pip install policy changed")
    receipt: dict[str, Any] = {
        "authorization": dict(EXPECTED_AUTHORIZATION),
        "decision": "exact_dependency_overlay_installed_import_smoke_passed",
        "install": {
            "network_payload_bytes": WHEEL_BYTES,
            "overlay_inventory": dict(inventory),
            "overlay_path": OVERLAY_PATH,
            "policy_flags": policy_flags,
            "wheel_inventory": list(wheel_inventory),
        },
        "limits": dict(EXPECTED_LIMITS),
        "observed_at": plan["observed_at"],
        "plan_file_sha256": _require_sha256(plan_file_sha256, "plan_file_sha256"),
        "production": {
            "canary_records_generated": 0,
            "dataset_sealed": False,
            "packages_installed": 2,
            "production_sft_authorized": False,
        },
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "smoke": dict(smoke),
        "status": "verified_isolated_dependency_overlay_ready_for_canary_gate",
    }
    receipt["receipt_sha256"] = sha256_value(receipt)
    return receipt


def validate_overlay_receipt(receipt: Mapping[str, Any]) -> None:
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported DeepMind overlay receipt")
    if receipt.get("status") != (
        "verified_isolated_dependency_overlay_ready_for_canary_gate"
    ):
        raise ValueError("DeepMind overlay status changed")
    if receipt.get("production") != {
        "canary_records_generated": 0,
        "dataset_sealed": False,
        "packages_installed": 2,
        "production_sft_authorized": False,
    }:
        raise ValueError("DeepMind overlay production boundary changed")
    seal = _require_sha256(receipt.get("receipt_sha256"), "receipt_sha256")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if sha256_value(unsigned) != seal:
        raise ValueError("DeepMind overlay receipt seal changed")
