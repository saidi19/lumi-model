"""Hash-bound acquisition receipt for the Lumi SFT closure canary."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PLAN_SHA256 = "31568114288e46c1c8613e20d9167af952050e5ed2a2556a64722bcec9da488a"
PLAN_SCHEMA_VERSION = "lumi.sft-closure-canary-acquisition-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-closure-canary-acquisition-report.v0.1"
EXPECTED_OBJECTS = 9
EXPECTED_OBJECT_BYTES = 840_904_222
MAXIMUM_NETWORK_BYTES = 943_718_400
MAXIMUM_PRIVATE_WRITE_BYTES = 4_294_967_296
TOKENIZER_SHA256 = "1022c9538d9ec795b07c2e2bda3c5c6f21c321de933cfe5ad197628a94caa2ee"
BENCHMARK_REGISTRY_SHA256 = (
    "1ffb5e938ab28c2fc876b9efd6c6ed59059f1e85aed2c396c024ae5925a4745e"
)
DATA_QUALITY_SHA256 = (
    "23fba21dbc822c84a5d16e63147fc1a7716576b8f1258cda8c035391f04f52c0"
)

RAW_DIRECTORY_BY_REPOSITORY = {
    "nvidia/Nemotron-Post-Training-Dataset-v1": (
        "nvidia_Nemotron-Post-Training-Dataset-v1"
    ),
    "CohereLabs/aya_collection_language_split": (
        "CohereLabs_aya_collection_language_split"
    ),
    "aplycaebous/BanglaTLit": "aplycaebous_BanglaTLit",
    "subhajitmahata84/banglabridge-instructions": (
        "subhajitmahata84_banglabridge-instructions"
    ),
}


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


def git_blob_sha1(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {source.stat().st_size}\0".encode("ascii"))
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _parquet_envelope(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        leading = handle.read(4)
        handle.seek(-8, 2)
        footer_length = int.from_bytes(handle.read(4), "little")
        trailing = handle.read(4)
    if leading != b"PAR1" or trailing != b"PAR1":
        raise ValueError(f"invalid Parquet envelope: {path}")
    if footer_length <= 0 or footer_length + 12 > path.stat().st_size:
        raise ValueError(f"invalid Parquet footer length: {path}")
    return {
        "container": "parquet",
        "footer_length_bytes": footer_length,
        "magic_verified": True,
    }


def _csv_schema_preflight(path: Path) -> dict[str, Any]:
    expected = ["id", "text_transliterated", "text_bengali"]
    rows = 0
    invalid_rows = 0
    identities: set[str] = set()
    duplicate_id_rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected:
            raise ValueError("BanglaTLit CSV schema changed")
        for row in reader:
            rows += 1
            identity = row.get("id", "").strip()
            transliterated = row.get("text_transliterated", "").strip()
            bengali = row.get("text_bengali", "").strip()
            if not identity or not transliterated or not bengali:
                invalid_rows += 1
            if identity in identities:
                duplicate_id_rows += 1
            identities.add(identity)
    return {
        "columns": expected,
        "duplicate_id_rows": duplicate_id_rows,
        "invalid_or_empty_rows": invalid_rows,
        "rows": rows,
    }


def _jsonl_schema_preflight(path: Path) -> dict[str, Any]:
    expected = {"instruction", "orig_instruction", "orig_output", "output"}
    rows = 0
    invalid_rows = 0
    observed_keys: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows += 1
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                invalid_rows += 1
                continue
            if not isinstance(value, dict):
                invalid_rows += 1
                continue
            observed_keys.update(value)
            if set(value) != expected or not all(
                isinstance(value.get(key), str) and value[key].strip()
                for key in expected
            ):
                invalid_rows += 1
    if observed_keys != expected:
        raise ValueError("BanglaBridge JSONL schema changed")
    return {
        "columns": sorted(expected),
        "invalid_or_empty_rows": invalid_rows,
        "rows": rows,
    }


def _dependency_smoke(tokenizers_overlay: Path | None) -> dict[str, Any]:
    if tokenizers_overlay is not None and tokenizers_overlay.is_dir():
        sys.path.insert(0, str(tokenizers_overlay))
    result: dict[str, Any] = {}
    try:
        import pyarrow  # type: ignore[import-not-found]

        result["pyarrow"] = {
            "available": True,
            "version": getattr(pyarrow, "__version__", "unknown"),
        }
    except ModuleNotFoundError:
        result["pyarrow"] = {"available": False, "version": None}
    try:
        import tokenizers  # type: ignore[import-not-found]

        result["tokenizers"] = {
            "available": True,
            "version": getattr(tokenizers, "__version__", "unknown"),
        }
    except ModuleNotFoundError:
        result["tokenizers"] = {"available": False, "version": None}
    return result


def _verify_plan(plan_path: Path) -> dict[str, Any]:
    if file_sha256(plan_path) != PLAN_SHA256:
        raise ValueError("closure-canary plan SHA-256 changed")
    plan = _load_json(plan_path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("closure-canary plan schema changed")
    objects = plan.get("objects")
    if not isinstance(objects, list) or len(objects) != EXPECTED_OBJECTS:
        raise ValueError("closure-canary exact object count changed")
    if sum(int(item.get("bytes", -1)) for item in objects) != EXPECTED_OBJECT_BYTES:
        raise ValueError("closure-canary exact object bytes changed")
    limits = plan.get("limits")
    if not isinstance(limits, dict) or limits != {
        "exact_objects": EXPECTED_OBJECTS,
        "exact_object_bytes": EXPECTED_OBJECT_BYTES,
        "maximum_network_bytes": MAXIMUM_NETWORK_BYTES,
        "maximum_decoded_and_private_write_bytes": MAXIMUM_PRIVATE_WRITE_BYTES,
        "maximum_runtime_minutes": 240,
    }:
        raise ValueError("closure-canary limits changed")
    prohibitions = plan.get("prohibitions")
    if not isinstance(prohibitions, dict) or not prohibitions or not all(
        value is True for value in prohibitions.values()
    ):
        raise ValueError("closure-canary prohibitions changed")
    return plan


def build_acquisition_report(
    *,
    repository_root: str | Path,
    plan_path: str | Path,
    private_root: str | Path,
    tokenizer_path: str | Path,
    benchmark_registry_path: str | Path,
    data_quality_path: str | Path,
    tokenizers_overlay: str | Path | None = None,
) -> dict[str, Any]:
    repository = Path(repository_root).resolve()
    plan_file = Path(plan_path).resolve()
    private = Path(private_root).resolve()
    raw_root = private / "raw"
    plan = _verify_plan(plan_file)

    object_receipts: list[dict[str, Any]] = []
    for item in plan["objects"]:
        repository_id = str(item["repository"])
        local_path = raw_root / RAW_DIRECTORY_BY_REPOSITORY[repository_id] / str(
            item["path"]
        )
        if not local_path.is_file():
            raise FileNotFoundError(f"closure-canary object missing: {item['path']}")
        actual_bytes = local_path.stat().st_size
        if actual_bytes != int(item["bytes"]):
            raise ValueError(f"closure-canary object size changed: {item['path']}")
        actual_sha256 = file_sha256(local_path)
        expected_sha256 = item.get("sha256")
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise ValueError(f"closure-canary object SHA-256 changed: {item['path']}")
        actual_git_blob_sha1 = None
        if item.get("git_blob_sha1") is not None:
            actual_git_blob_sha1 = git_blob_sha1(local_path)
            if actual_git_blob_sha1 != item["git_blob_sha1"]:
                raise ValueError(f"closure-canary Git blob changed: {item['path']}")
        object_receipts.append(
            {
                "bytes": actual_bytes,
                "git_blob_sha1": actual_git_blob_sha1,
                "path": item["path"],
                "repository": repository_id,
                "revision": item["revision"],
                "sha256": actual_sha256,
            }
        )

    tokenizer = Path(tokenizer_path).resolve()
    benchmark = Path(benchmark_registry_path).resolve()
    data_quality = Path(data_quality_path).resolve()
    for name, path, expected in (
        ("tokenizer", tokenizer, TOKENIZER_SHA256),
        ("benchmark_registry", benchmark, BENCHMARK_REGISTRY_SHA256),
        ("data_quality", data_quality, DATA_QUALITY_SHA256),
    ):
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"closure-canary frozen binding changed: {name}")

    schema_preflight = {
        "banglabridge": _jsonl_schema_preflight(
            raw_root
            / RAW_DIRECTORY_BY_REPOSITORY[
                "subhajitmahata84/banglabridge-instructions"
            ]
            / "banglish_finetune.adapted.jsonl"
        ),
        "banglatlit": _csv_schema_preflight(
            raw_root
            / RAW_DIRECTORY_BY_REPOSITORY["aplycaebous/BanglaTLit"]
            / "BanglaTLit_train.csv"
        ),
        "parquet_envelopes": {
            "aya_bengali": _parquet_envelope(
                raw_root
                / RAW_DIRECTORY_BY_REPOSITORY[
                    "CohereLabs/aya_collection_language_split"
                ]
                / "bengali/train-00000-of-00003.parquet"
            ),
            "nemotron_chat": _parquet_envelope(
                raw_root
                / RAW_DIRECTORY_BY_REPOSITORY[
                    "nvidia/Nemotron-Post-Training-Dataset-v1"
                ]
                / "data/chat-00000-of-00008.parquet"
            ),
            "nemotron_tool": _parquet_envelope(
                raw_root
                / RAW_DIRECTORY_BY_REPOSITORY[
                    "nvidia/Nemotron-Post-Training-Dataset-v1"
                ]
                / "data/tool-00000-of-00013.parquet"
            ),
        },
    }
    dependencies = _dependency_smoke(
        Path(tokenizers_overlay).resolve() if tokenizers_overlay else None
    )
    pyarrow_available = dependencies["pyarrow"]["available"] is True
    tokenizer_available = dependencies["tokenizers"]["available"] is True
    blockers = []
    if not pyarrow_available:
        blockers.append("missing_local_parquet_decoder_no_dependency_download_authorized")
    if not tokenizer_available:
        blockers.append("missing_local_frozen_tokenizer_runtime")

    private_bytes = sum(
        path.stat().st_size for path in private.rglob("*") if path.is_file()
    )
    if private_bytes > MAXIMUM_PRIVATE_WRITE_BYTES:
        raise ValueError("closure-canary private write ceiling exceeded")

    status = (
        "acquisition_verified_ready_for_curation"
        if not blockers
        else "acquisition_verified_curation_blocked"
    )
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": status,
        "authorization": {
            "explicit_owner_authorization_received": True,
            "plan_sha256": PLAN_SHA256,
        },
        "bindings": {
            "benchmark_registry_sha256": BENCHMARK_REGISTRY_SHA256,
            "data_quality_sha256": DATA_QUALITY_SHA256,
            "tokenizer_json_sha256": TOKENIZER_SHA256,
        },
        "blockers": blockers,
        "bounds": {
            "authorized_object_bytes": EXPECTED_OBJECT_BYTES,
            "authorized_objects": EXPECTED_OBJECTS,
            "maximum_network_bytes": MAXIMUM_NETWORK_BYTES,
            "maximum_private_write_bytes": MAXIMUM_PRIVATE_WRITE_BYTES,
            "private_bytes_observed_before_report": private_bytes,
            "source_payload_bytes": sum(item["bytes"] for item in object_receipts),
        },
        "curation": {
            "candidate_records_written": 0,
            "dataset_sealed": False,
            "owner_review_packets_written": 0,
            "sft_started": False,
        },
        "dependencies": dependencies,
        "objects": object_receipts,
        "plan_file_sha256": file_sha256(plan_file),
        "prohibitions_preserved": dict(plan["prohibitions"]),
        "repository_root": str(repository),
        "schema_preflight": schema_preflight,
    }
    report["report_sha256"] = sha256_value(report)
    return report


def write_report(path: str | Path, report: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "BENCHMARK_REGISTRY_SHA256",
    "DATA_QUALITY_SHA256",
    "PLAN_SHA256",
    "REPORT_SCHEMA_VERSION",
    "TOKENIZER_SHA256",
    "build_acquisition_report",
    "canonical_json_bytes",
    "file_sha256",
    "git_blob_sha1",
    "sha256_value",
    "write_report",
]
