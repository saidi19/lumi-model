"""Reproducible Lingua 2.2.0 compatibility build for AWS Glue Python 3.11."""

from __future__ import annotations

import hashlib
import json
import os
import re
import zipfile
from email.parser import Parser
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_lingua_glue_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported Lingua Glue compatibility schema")
    if config.get("gate") != "lumi_lingua_glue_cp311_compatibility_v0.1":
        raise ValueError("unexpected Lingua Glue compatibility gate")
    upstream = config.get("upstream", {})
    if upstream.get("repository") != "https://github.com/pemistahl/lingua-rs.git":
        raise ValueError("Lingua upstream repository changed")
    if upstream.get("tag") != "v1.8.0":
        raise ValueError("Lingua upstream tag changed")
    if re.fullmatch(r"[0-9a-f]{40}", str(upstream.get("commit"))) is None:
        raise ValueError("Lingua upstream commit is not pinned")
    if upstream.get("package_name") != "lingua-language-detector":
        raise ValueError("Lingua package name changed")
    if upstream.get("package_version") != "2.2.0":
        raise ValueError("Lingua package version changed")
    for key in ("cargo_lock_sha256", "cargo_toml_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(upstream.get(key))) is None:
            raise ValueError(f"Lingua {key} is not pinned")
    patch = config.get("patch", {})
    if patch.get("target") != 'requires-python = ">=3.12"':
        raise ValueError("Lingua compatibility patch target changed")
    if patch.get("replacement") != 'requires-python = ">=3.11"':
        raise ValueError("Lingua compatibility patch replacement changed")
    for key in (
        "expected_original_pyproject_sha256",
        "expected_patched_pyproject_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(patch.get(key))) is None:
            raise ValueError(f"Lingua {key} is not pinned")
    wheel = config.get("wheel", {})
    if wheel != {
        "compatibility": "manylinux_2_34",
        "expected_tag": "cp311-cp311-manylinux_2_34_x86_64",
        "python_version": "3.11",
    }:
        raise ValueError("Lingua Glue wheel contract changed")
    parity = config.get("parity", {})
    if parity.get("expected_file_count") != 225:
        raise ValueError("Lingua parity file count changed")
    if parity.get("expected_record_count") != 222_790:
        raise ValueError("Lingua parity record count changed")
    if parity.get("minimum_confidence") != 0.7:
        raise ValueError("Lingua parity threshold changed")
    for key in (
        "expected_lumi_decision_sha256",
        "expected_threshold_0_7_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(parity.get(key))) is None:
            raise ValueError(f"Lingua parity {key} is not pinned")
    build = config.get("build", {})
    if build.get("maximum_build_minutes") != 30:
        raise ValueError("Lingua CodeBuild timeout changed")
    if build.get("maximum_compute_usd") != "0.60":
        raise ValueError("Lingua CodeBuild cost bound changed")
    return config


def patch_lingua_pyproject(path: Path, config: dict[str, Any]) -> dict[str, str]:
    patch = config["patch"]
    original = path.read_bytes()
    original_sha256 = hashlib.sha256(original).hexdigest()
    if original_sha256 != patch["expected_original_pyproject_sha256"]:
        raise ValueError("Lingua upstream pyproject SHA-256 changed")
    target = str(patch["target"]).encode()
    replacement = str(patch["replacement"]).encode()
    if original.count(target) != 1:
        raise ValueError("Lingua Python requirement patch is not singular")
    modified = original.replace(target, replacement)
    modified_sha256 = hashlib.sha256(modified).hexdigest()
    if modified_sha256 != patch["expected_patched_pyproject_sha256"]:
        raise ValueError("Lingua patched pyproject SHA-256 changed")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(modified)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return {
        "original_sha256": original_sha256,
        "patched_sha256": modified_sha256,
    }


def validate_lingua_wheel(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    expected_tag = config["wheel"]["expected_tag"]
    expected_name = f"lingua_language_detector-2.2.0-{expected_tag}.whl"
    if path.name != expected_name:
        raise ValueError(f"unexpected Lingua wheel filename: {path.name}")
    with zipfile.ZipFile(path) as archive:
        metadata_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        wheel_names = [name for name in archive.namelist() if name.endswith(".dist-info/WHEEL")]
        if len(metadata_names) != 1 or len(wheel_names) != 1:
            raise ValueError("Lingua wheel metadata layout changed")
        metadata = Parser().parsestr(archive.read(metadata_names[0]).decode())
        wheel_metadata = Parser().parsestr(archive.read(wheel_names[0]).decode())
    if metadata["Name"] != config["upstream"]["package_name"]:
        raise ValueError("Lingua wheel package name changed")
    if metadata["Version"] != config["upstream"]["package_version"]:
        raise ValueError("Lingua wheel package version changed")
    if metadata["Requires-Python"] != ">=3.11":
        raise ValueError("Lingua wheel Python requirement changed")
    tags = wheel_metadata.get_all("Tag", [])
    if tags != [expected_tag]:
        raise ValueError("Lingua wheel compatibility tag changed")
    return {
        "bytes": path.stat().st_size,
        "filename": path.name,
        "requires_python": metadata["Requires-Python"],
        "sha256": sha256_file(path),
        "tags": tags,
        "version": metadata["Version"],
    }
