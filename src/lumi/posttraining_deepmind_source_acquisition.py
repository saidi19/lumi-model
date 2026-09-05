"""Exact source acquisition for the pinned DeepMind Mathematics canary."""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .posttraining_math_replacement_preflight import (
    DEEPMIND_OBJECTS,
    DEEPMIND_REVISION,
)

PLAN_SCHEMA_VERSION = "lumi.sft-deepmind-source-acquisition-plan.v0.1"
RECEIPT_SCHEMA_VERSION = "lumi.sft-deepmind-source-acquisition-receipt.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-deepmind-source-acquisition-verification.v0.1"
)

SOURCE_BYTES = 234_672
SOURCE_OBJECTS = 30
SOURCE_INVENTORY_SHA256 = (
    "47e472f97f7092cf264edb727bd0821db59869133f56b12cddf3f8a4ef1a6ce9"
)
SOURCE_ROOT = (
    "/Users/afnan/Documents/Flowaistudio/lumi-artifacts/"
    "4b-sft-deepmind-canary-v0.1/427f45075f84/attempt-001/source"
)
OVERLAY_RECEIPT_INTERNAL_SHA256 = (
    "192fbac2bf5cf1049ff5bcd33e84507bf55d686e296b9dfb2d67cd7b5c0bce72"
)
OVERLAY_VERIFICATION_INTERNAL_SHA256 = (
    "10ce6707e13472fcdf8641a4f9c5b5a8cb120865e0ad5a1e0b4ff51ceffabd59"
)

EXPECTED_AUTHORIZATION = {
    "canary_generation_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_spend_authorized": False,
    "external_distribution_authorized": False,
    "gpu_authorized": False,
    "network_access_authorized": True,
    "package_installation_authorized": False,
    "production_sft_authorized": False,
    "purpose": "exact_pinned_deepmind_source_acquisition_substage_only",
    "source_acquisition_authorized": True,
    "test_or_evaluation_generation_authorized": False,
    "unrelated_data_access_authorized": False,
}
EXPECTED_LIMITS = {
    "exact_source_bytes": SOURCE_BYTES,
    "exact_source_objects": SOURCE_OBJECTS,
    "maximum_decoded_bytes": 8_388_608,
    "maximum_network_bytes": 1_048_576,
    "maximum_runtime_minutes": 180,
}
EXPECTED_OUTPUT = {
    "raw_source_text_in_receipt": False,
    "receipt_path": "evidence/lumi-4b-sft-deepmind-source-acquisition-v0.1.json",
    "source_root": SOURCE_ROOT,
    "verification_path": (
        "evidence/lumi-4b-sft-deepmind-source-acquisition-verification-v0.1.json"
    ),
}
EXPECTED_BINDING_PATHS = {
    "acquisition_implementation": (
        "src/lumi/posttraining_deepmind_source_acquisition.py"
    ),
    "acquisition_runner": "scripts/acquire_deepmind_source.py",
    "acquisition_verifier": "scripts/verify_deepmind_source_acquisition.py",
    "adapter_preflight_plan": (
        "configs/lumi-4b-sft-deepmind-adapter-preflight-v0.1.json"
    ),
    "adapter_preflight_report": (
        "evidence/lumi-4b-sft-deepmind-adapter-preflight-v0.1.json"
    ),
    "adapter_preflight_verification": (
        "evidence/lumi-4b-sft-deepmind-adapter-preflight-verification-v0.1.json"
    ),
    "dependency_overlay_receipt": (
        "evidence/lumi-4b-sft-deepmind-dependency-overlay-v0.1.json"
    ),
    "dependency_overlay_verification": (
        "evidence/lumi-4b-sft-deepmind-dependency-overlay-verification-v0.1.json"
    ),
    "source_inventory_implementation": (
        "src/lumi/posttraining_math_replacement_preflight.py"
    ),
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")


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
        raise ValueError(f"DeepMind acquisition binding path is invalid: {name}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.parent):
        raise ValueError(f"DeepMind acquisition binding escapes workspace: {name}")
    if not resolved.is_file():
        raise FileNotFoundError(f"DeepMind acquisition binding is missing: {name}")
    expected = _require_sha256(binding.get("sha256"), f"bindings.{name}.sha256")
    if file_sha256(resolved) != expected:
        raise ValueError(f"DeepMind acquisition binding changed: {name}")
    return resolved


def load_acquisition_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    plan = _load_json(path)
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
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
            "revision": DEEPMIND_REVISION,
            "source_id": "deepmind_mathematics_dataset",
        },
        "status": "authorized_exact_pinned_source_acquisition",
    }
    for field, expected_value in expected.items():
        if plan.get(field) != expected_value:
            raise ValueError(f"DeepMind source-acquisition plan changed: {field}")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(
        EXPECTED_BINDING_PATHS
    ):
        raise ValueError("DeepMind source-acquisition binding set changed")
    for name, expected_path in EXPECTED_BINDING_PATHS.items():
        binding = bindings[name]
        if not isinstance(binding, Mapping) or binding.get("path") != expected_path:
            raise ValueError(f"DeepMind acquisition binding path changed: {name}")
        resolve_binding(repository_root, binding, name=name)
    _verify_overlay_bindings(plan, repository_root=repository_root)
    validate_expected_inventory(DEEPMIND_OBJECTS)
    return plan


def _verify_internal_seal(value: Mapping[str, Any], field: str) -> str:
    seal = _require_sha256(value.get(field), field)
    unsigned = {key: item for key, item in value.items() if key != field}
    if sha256_value(unsigned) != seal:
        raise ValueError(f"DeepMind bound seal changed: {field}")
    return seal


def _verify_overlay_bindings(
    plan: Mapping[str, Any],
    *,
    repository_root: str | Path,
) -> None:
    bindings = plan["bindings"]
    receipt = _load_json(
        resolve_binding(
            repository_root,
            bindings["dependency_overlay_receipt"],
            name="dependency_overlay_receipt",
        )
    )
    verification = _load_json(
        resolve_binding(
            repository_root,
            bindings["dependency_overlay_verification"],
            name="dependency_overlay_verification",
        )
    )
    if _verify_internal_seal(receipt, "receipt_sha256") != (
        OVERLAY_RECEIPT_INTERNAL_SHA256
    ):
        raise ValueError("DeepMind overlay receipt is not authorized")
    if _verify_internal_seal(verification, "verification_sha256") != (
        OVERLAY_VERIFICATION_INTERNAL_SHA256
    ):
        raise ValueError("DeepMind overlay verification is not authorized")
    if receipt.get("status") != (
        "verified_isolated_dependency_overlay_ready_for_canary_gate"
    ) or verification.get("status") != "verified_exact_deepmind_dependency_overlay":
        raise ValueError("DeepMind overlay gate is not ready")


def validate_expected_inventory(objects: Sequence[Mapping[str, Any]]) -> None:
    if list(objects) != DEEPMIND_OBJECTS:
        raise ValueError("DeepMind source inventory changed")
    if len(objects) != SOURCE_OBJECTS:
        raise ValueError("DeepMind source object count changed")
    if sum(int(item["size_bytes"]) for item in objects) != SOURCE_BYTES:
        raise ValueError("DeepMind source byte count changed")
    if sha256_value(list(objects)) != SOURCE_INVENTORY_SHA256:
        raise ValueError("DeepMind source inventory SHA-256 changed")
    for item in objects:
        relative = Path(str(item["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("DeepMind source inventory path is unsafe")
        if _SHA1.fullmatch(str(item["git_blob_sha1"])) is None:
            raise ValueError("DeepMind source inventory Git blob is invalid")


def raw_url(relative_path: str) -> str:
    quoted = "/".join(urllib.parse.quote(part, safe="") for part in Path(relative_path).parts)
    return (
        "https://raw.githubusercontent.com/google-deepmind/mathematics_dataset/"
        f"{DEEPMIND_REVISION}/{quoted}"
    )


def acquire_exact_source(
    *,
    target_root: str | Path,
    urlopen: Any = urllib.request.urlopen,
) -> dict[str, Any]:
    target = Path(target_root).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite DeepMind source: {target}")
    temporary = target.with_name(f".{target.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"DeepMind temporary source already exists: {temporary}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    inventory = []
    payload_bytes = 0
    for expected in DEEPMIND_OBJECTS:
        request = urllib.request.Request(
            raw_url(str(expected["path"])),
            headers={"User-Agent": "Lumi-DeepMind-Canary-v0.1"},
        )
        with urlopen(request, timeout=30) as response:
            payload = response.read(int(expected["size_bytes"]) + 1)
        if len(payload) != expected["size_bytes"]:
            raise ValueError(f"DeepMind source size changed: {expected['path']}")
        if git_blob_sha1(payload) != expected["git_blob_sha1"]:
            raise ValueError(f"DeepMind source blob changed: {expected['path']}")
        payload_bytes += len(payload)
        if payload_bytes > EXPECTED_LIMITS["maximum_network_bytes"]:
            raise ValueError("DeepMind source exceeded network byte ceiling")
        output = temporary / str(expected["path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        inventory.append(
            {
                "git_blob_sha1": expected["git_blob_sha1"],
                "path": expected["path"],
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    if payload_bytes != SOURCE_BYTES or len(inventory) != SOURCE_OBJECTS:
        raise ValueError("DeepMind acquired source boundary changed")
    temporary.replace(target)
    return {
        "inventory": inventory,
        "inventory_sha256": sha256_value(inventory),
        "network_payload_bytes": payload_bytes,
        "objects": len(inventory),
    }


def inspect_source_root(target_root: str | Path) -> dict[str, Any]:
    target = Path(target_root).resolve()
    if not target.is_dir():
        raise FileNotFoundError("DeepMind source root is missing")
    inventory = []
    expected_by_path = {str(item["path"]): item for item in DEEPMIND_OBJECTS}
    files = sorted(item for item in target.rglob("*") if item.is_file())
    if [item.relative_to(target).as_posix() for item in files] != sorted(expected_by_path):
        raise ValueError("DeepMind source-root file set changed")
    for path in files:
        relative = path.relative_to(target).as_posix()
        payload = path.read_bytes()
        expected = expected_by_path[relative]
        if len(payload) != expected["size_bytes"]:
            raise ValueError(f"DeepMind local source size changed: {relative}")
        blob = git_blob_sha1(payload)
        if blob != expected["git_blob_sha1"]:
            raise ValueError(f"DeepMind local source blob changed: {relative}")
        inventory.append(
            {
                "git_blob_sha1": blob,
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    return {
        "inventory": inventory,
        "inventory_sha256": sha256_value(inventory),
        "network_payload_bytes": sum(item["size_bytes"] for item in inventory),
        "objects": len(inventory),
    }


def build_acquisition_receipt(
    *,
    acquisition: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_file_sha256: str,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "authorization": dict(EXPECTED_AUTHORIZATION),
        "decision": "exact_source_acquired_canary_generation_still_separate_substage",
        "limits": dict(EXPECTED_LIMITS),
        "observed_at": plan["observed_at"],
        "plan_file_sha256": _require_sha256(plan_file_sha256, "plan_file_sha256"),
        "production": {
            "canary_records_generated": 0,
            "dataset_sealed": False,
            "production_sft_authorized": False,
            "source_registry_promotions": 0,
        },
        "raw_source_text_present": False,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "source": {
            "inventory_sha256": acquisition["inventory_sha256"],
            "network_payload_bytes": acquisition["network_payload_bytes"],
            "objects": acquisition["objects"],
            "revision": DEEPMIND_REVISION,
            "root": SOURCE_ROOT,
        },
        "status": "verified_exact_deepmind_source_acquired_nonproduction",
    }
    receipt["receipt_sha256"] = sha256_value(receipt)
    return receipt


def validate_acquisition_receipt(receipt: Mapping[str, Any]) -> None:
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported DeepMind source-acquisition receipt")
    if receipt.get("status") != "verified_exact_deepmind_source_acquired_nonproduction":
        raise ValueError("DeepMind source-acquisition status changed")
    source = receipt.get("source")
    if not isinstance(source, Mapping) or source.get("objects") != SOURCE_OBJECTS or source.get(
        "network_payload_bytes"
    ) != SOURCE_BYTES:
        raise ValueError("DeepMind acquired source metrics changed")
    if receipt.get("production") != {
        "canary_records_generated": 0,
        "dataset_sealed": False,
        "production_sft_authorized": False,
        "source_registry_promotions": 0,
    }:
        raise ValueError("DeepMind source-acquisition production boundary changed")
    if receipt.get("raw_source_text_present") is not False:
        raise ValueError("DeepMind source-acquisition receipt contains source text")
    seal = _require_sha256(receipt.get("receipt_sha256"), "receipt_sha256")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if sha256_value(unsigned) != seal:
        raise ValueError("DeepMind source-acquisition receipt seal changed")
