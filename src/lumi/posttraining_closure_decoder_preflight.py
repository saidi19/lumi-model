"""Metadata-only dependency preflight for the Lumi closure-canary decoder."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "lumi.sft-closure-parquet-decoder-overlay-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-closure-parquet-decoder-preflight-report.v0.1"
ACQUISITION_PLAN_SHA256 = (
    "31568114288e46c1c8613e20d9167af952050e5ed2a2556a64722bcec9da488a"
)
ACQUISITION_REPORT_FILE_SHA256 = (
    "0c9ab1aaaf763fb067b1da656e145addd364942dcc902bf7a08a1877dcf624e1"
)
ACQUISITION_VERIFICATION_FILE_SHA256 = (
    "1b6e3c93e2bbac55e0cadd81723d1069433c9c6606997cfb2b178d700180a6e0"
)
PYARROW_VERSION = "25.0.0"
PYARROW_METADATA_BYTES = 40_590
PYARROW_METADATA_SHA256 = (
    "952cdc36d7d21120c401fed07931c74e1b85732b43ff7f56861c687a4c531867"
)
WHEEL_FILENAME = "pyarrow-25.0.0-cp313-cp313-macosx_12_0_arm64.whl"
WHEEL_BYTES = 35_844_549
WHEEL_SHA256 = "8831a3ba52fa7cdb78d368d968b1dcd06171e6dff5461e16d90de91d371e47bc"
WHEEL_URL = (
    "https://files.pythonhosted.org/packages/6c/c8/"
    "098ce17d778fd9d29e40bb8c5f19a40cc90c3f0b46c9057b0d7993f42f54/"
    + WHEEL_FILENAME
)
OVERLAY_ROOT = "/private/tmp/lumi-sft-closure-pyarrow-v0.1"
MAXIMUM_METADATA_OBJECTS = 3
MAXIMUM_METADATA_NETWORK_BYTES = 4 * 1024 * 1024
MAXIMUM_PRIVATE_WRITES = 16 * 1024 * 1024


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


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _compatible_tags(python_executable: Path) -> set[str]:
    completed = subprocess.run(
        [str(python_executable), "-m", "pip", "debug", "--verbose"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        line.strip()
        for line in completed.stdout.splitlines()
        if line.startswith("  ") and "-" in line
    }


def _selected_wheel(metadata: dict[str, Any]) -> dict[str, Any]:
    info = metadata.get("info")
    urls = metadata.get("urls")
    if not isinstance(info, dict) or not isinstance(urls, list):
        raise ValueError("PyPI pyarrow metadata schema changed")
    if info.get("name") != "pyarrow" or info.get("version") != PYARROW_VERSION:
        raise ValueError("PyPI pyarrow package identity changed")
    if info.get("requires_python") != ">=3.10":
        raise ValueError("PyPI pyarrow Python requirement changed")
    if info.get("requires_dist") not in (None, []):
        raise ValueError("PyPI pyarrow declared dependencies changed")
    matches = [item for item in urls if item.get("filename") == WHEEL_FILENAME]
    if len(matches) != 1:
        raise ValueError("PyPI pyarrow exact macOS arm64 wheel selection changed")
    wheel = matches[0]
    if (
        wheel.get("packagetype") != "bdist_wheel"
        or wheel.get("size") != WHEEL_BYTES
        or wheel.get("digests", {}).get("sha256") != WHEEL_SHA256
        or wheel.get("url") != WHEEL_URL
    ):
        raise ValueError("PyPI pyarrow exact wheel metadata changed")
    return wheel


def build_overlay_plan(
    *,
    metadata_path: str | Path,
    acquisition_plan_path: str | Path,
    acquisition_report_path: str | Path,
    acquisition_verification_path: str | Path,
    python_executable: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata_file = Path(metadata_path).resolve()
    acquisition_plan = Path(acquisition_plan_path).resolve()
    acquisition_report = Path(acquisition_report_path).resolve()
    acquisition_verification = Path(acquisition_verification_path).resolve()
    executable = Path(python_executable).resolve()
    if metadata_file.stat().st_size != PYARROW_METADATA_BYTES:
        raise ValueError("PyPI pyarrow metadata size changed")
    if file_sha256(metadata_file) != PYARROW_METADATA_SHA256:
        raise ValueError("PyPI pyarrow metadata SHA-256 changed")
    if file_sha256(acquisition_plan) != ACQUISITION_PLAN_SHA256:
        raise ValueError("closure-canary acquisition plan changed")
    if file_sha256(acquisition_report) != ACQUISITION_REPORT_FILE_SHA256:
        raise ValueError("closure-canary acquisition report changed")
    if (
        file_sha256(acquisition_verification)
        != ACQUISITION_VERIFICATION_FILE_SHA256
    ):
        raise ValueError("closure-canary acquisition verification changed")
    metadata = _load_json(metadata_file)
    _selected_wheel(metadata)
    required_tag = "cp313-cp313-macosx_12_0_arm64"
    if required_tag not in _compatible_tags(executable):
        raise ValueError("local CPython does not accept the selected pyarrow wheel")
    runtime_smoke = (
        "import platform,sys;"
        "print(sys.version.split()[0]);"
        "print(platform.machine())"
    )
    version_output = subprocess.run(
        [str(executable), "-c", runtime_smoke],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if version_output != ["3.13.7", "arm64"] or platform.system() != "Darwin":
        raise ValueError("local decoder runtime identity changed")

    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "proposed_requires_explicit_owner_authorization",
        "purpose": "private_cpu_only_pyarrow_overlay_for_acquired_sft_closure_canary",
        "bindings": {
            "acquisition_plan_sha256": ACQUISITION_PLAN_SHA256,
            "acquisition_report_file_sha256": ACQUISITION_REPORT_FILE_SHA256,
            "acquisition_verification_file_sha256": (
                ACQUISITION_VERIFICATION_FILE_SHA256
            ),
            "pypi_metadata": {
                "bytes": PYARROW_METADATA_BYTES,
                "sha256": PYARROW_METADATA_SHA256,
                "url": "https://pypi.org/pypi/pyarrow/25.0.0/json",
            },
        },
        "runtime": {
            "abi": "cp313",
            "architecture": "arm64",
            "compatible_tag": required_tag,
            "operating_system": "Darwin",
            "overlay_root": OVERLAY_ROOT,
            "python_executable": str(executable),
            "python_version": "3.13.7",
        },
        "wheel": {
            "bytes": WHEEL_BYTES,
            "filename": WHEEL_FILENAME,
            "requires_dist": [],
            "sha256": WHEEL_SHA256,
            "url": WHEEL_URL,
            "version": PYARROW_VERSION,
        },
        "installation_contract": {
            "download_exact_wheel_only": True,
            "install_target": OVERLAY_ROOT,
            "pip_flags": ["--no-deps", "--require-hashes", "--target"],
            "post_install_smoke": [
                "import pyarrow",
                "import pyarrow.parquet",
                "verify pyarrow.__version__ == 25.0.0",
                "read only the three already acquired pinned parquet objects",
            ],
        },
        "proposed_limits": {
            "exact_wheel_bytes": WHEEL_BYTES,
            "exact_wheels": 1,
            "maximum_network_bytes": 40 * 1024 * 1024,
            "maximum_private_write_bytes": 256 * 1024 * 1024,
            "maximum_runtime_minutes": 45,
        },
        "prohibitions": {
            "additional_dataset_network_access": True,
            "candidate_registry_change": True,
            "dataset_seal": True,
            "external_api": True,
            "external_distribution": True,
            "gpu": True,
            "production_promotion": True,
            "sft_training": True,
            "unrelated_access": True,
        },
    }
    plan["plan_sha256"] = sha256_value(plan)
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "verified_exact_overlay_plan_ready_authorization_required",
        "metadata": {
            "objects_fetched": 1,
            "total_bytes": PYARROW_METADATA_BYTES,
            "within_authorized_object_limit": 1 <= MAXIMUM_METADATA_OBJECTS,
            "within_authorized_network_limit": (
                PYARROW_METADATA_BYTES <= MAXIMUM_METADATA_NETWORK_BYTES
            ),
        },
        "resolution": {
            "declared_runtime_dependencies": [],
            "exact_compatible_wheels": 1,
            "selected_wheel_bytes": WHEEL_BYTES,
            "selected_wheel_sha256": WHEEL_SHA256,
            "target_tag_verified": required_tag,
        },
        "wheel_downloaded": False,
        "wheel_installed": False,
        "overlay_plan_internal_sha256": plan["plan_sha256"],
    }
    report["report_sha256"] = sha256_value(report)
    return plan, report


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "ACQUISITION_PLAN_SHA256",
    "PYARROW_METADATA_BYTES",
    "PYARROW_METADATA_SHA256",
    "WHEEL_BYTES",
    "WHEEL_FILENAME",
    "WHEEL_SHA256",
    "WHEEL_URL",
    "build_overlay_plan",
    "canonical_json_bytes",
    "file_sha256",
    "sha256_value",
    "write_json",
]
