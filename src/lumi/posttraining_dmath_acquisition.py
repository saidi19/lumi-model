"""Exact, authorization-bound DMath source acquisition."""

from __future__ import annotations

import hashlib
import json
import os
import re
import ssl
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

import certifi

RUN_SCHEMA_VERSION = "lumi.sft-dmath-acquisition-run.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-dmath-acquisition-report.v0.1"

SOURCE_ID = "dmath"
SOURCE_REVISION = "149a7403619ac1bd6ed10f50694ad82699d5cc03"
SOURCE_REPOSITORY = "https://github.com/JiwooKimAR/dmath"
SOURCE_LICENSE = "mit"

OWNER_AUTHORIZATION_STATEMENT = (
    "approve one bounded CPU-only DMath train-source acquisition and curation at "
    f"revision {SOURCE_REVISION}, maximum 3 exact objects / 7,943 train rows / "
    "16 MiB network / 64 MiB decoded / 120 minutes. It may fetch only LICENSE, "
    "README.md, and dataset/train.json, write private raw data, a non-production "
    "candidate, attribution evidence, raw-text-free reports, and a deterministic "
    "32-row owner-review packet outside Git. Reserve the official test split; use "
    "question_en and answer_en, treat expression/Python solutions only as verifier "
    "inputs under a fail-closed sandbox, and run schema, PII/secret, deduplication, "
    "benchmark decontamination, 4K-fit, and frozen-tokenizer measurement. No external "
    "API, GPU, source-registry promotion, dataset seal, SFT training, unrelated data, "
    "or external distribution."
)
RETRY_AUTHORIZATION_STATEMENT = (
    "I approve exactly one bounded DMath acquisition retry at revision "
    f"{SOURCE_REVISION}, with an additional maximum 16 MiB network / 64 MiB decoded / "
    "120 minutes, limited to the same exact LICENSE, README.md, and dataset/train.json "
    "objects and the already-authorized CPU-only curation. This retry is solely to "
    "recover from the verified object-by-ID JSON-root mismatch. No test split, external "
    "API, GPU, registry promotion, dataset seal, SFT training, unrelated data, or external "
    "distribution."
)

EXACT_OBJECTS = [
    {
        "git_blob_sha1": "cde816142c26a3c795a3dcb570a7132de77784bc",
        "kind": "license",
        "path": "LICENSE",
        "size_bytes": 1_066,
        "url": f"https://raw.githubusercontent.com/JiwooKimAR/dmath/{SOURCE_REVISION}/LICENSE",
    },
    {
        "git_blob_sha1": "4e8b363e51a54a926cf4d9264e020a70e89b4ac2",
        "kind": "documentation",
        "path": "README.md",
        "size_bytes": 6_992,
        "url": f"https://raw.githubusercontent.com/JiwooKimAR/dmath/{SOURCE_REVISION}/README.md",
    },
    {
        "git_blob_sha1": "be400482da3c53d61c1601ed701efba7d8e2efda",
        "kind": "data",
        "path": "dataset/train.json",
        "size_bytes": 12_198_829,
        "split": "train",
        "url": (
            "https://raw.githubusercontent.com/JiwooKimAR/dmath/"
            f"{SOURCE_REVISION}/dataset/train.json"
        ),
    },
]

EXPECTED_AUTHORIZATION = {
    "candidate_registry_promotion_authorized": False,
    "dataset_acquisition_authorized": True,
    "dataset_seal_authorized": False,
    "external_api_spend_authorized": False,
    "external_distribution_authorized": False,
    "network_access_authorized": True,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "purpose": "exact_dmath_cpu_only_acquisition_and_curation",
    "test_split_access_authorized": False,
    "unrelated_dataset_access_authorized": False,
}
EXPECTED_LIMITS = {
    "expected_network_bytes": 12_206_887,
    "expected_objects": 3,
    "expected_train_rows": 7_943,
    "maximum_decoded_bytes": 67_108_864,
    "maximum_network_bytes": 16_777_216,
    "maximum_objects": 3,
    "maximum_runtime_minutes": 120,
    "maximum_train_rows": 7_943,
}
EXPECTED_OUTPUT = {
    "private_root": ("../lumi-artifacts/4b-sft-dmath-acquisition-v0.1/aaba91c90ea8/attempt-001"),
    "report_path": "evidence/lumi-4b-sft-dmath-acquisition-v0.1.json",
    "raw_data_in_git_permitted": False,
    "raw_text_in_report": False,
}
EXPECTED_BINDING_PATHS = {
    "acquisition_implementation": "src/lumi/posttraining_dmath_acquisition.py",
    "acquisition_runner": "scripts/run_dmath_acquisition.py",
    "data_contract": "configs/lumi-4b-sft-data-contract-v0.1.json",
    "replacement_plan": "configs/lumi-4b-sft-math-replacement-preflight-v0.1.json",
    "replacement_report": "evidence/lumi-4b-sft-math-replacement-preflight-v0.1.json",
    "replacement_verification": (
        "evidence/lumi-4b-sft-math-replacement-preflight-verification-v0.1.json"
    ),
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DownloadResponse(Protocol):
    headers: Mapping[str, str]

    def __enter__(self) -> DownloadResponse: ...

    def __exit__(self, *args: object) -> None: ...

    def geturl(self) -> str: ...

    def read(self, size: int = -1) -> bytes: ...


OpenURL = Callable[[urllib.request.Request], DownloadResponse]


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
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


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
    workspace = root.parent.resolve()
    relative = binding.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"DMath acquisition binding path is invalid: {name}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError(f"DMath acquisition binding escapes workspace: {name}")
    if not resolved.is_file():
        raise ValueError(f"DMath acquisition binding is missing: {name}")
    expected = _require_sha256(binding.get("sha256"), f"bindings.{name}.sha256")
    if file_sha256(resolved) != expected:
        raise ValueError(f"DMath acquisition binding changed: {name}")
    return resolved


def load_dmath_acquisition_plan(
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
        "owner_authorization_statement": OWNER_AUTHORIZATION_STATEMENT,
        "retry_authorization_statement": RETRY_AUTHORIZATION_STATEMENT,
        "schema_version": RUN_SCHEMA_VERSION,
        "source": {
            "declared_license_id": SOURCE_LICENSE,
            "exact_objects": EXACT_OBJECTS,
            "repository_url": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "source_id": SOURCE_ID,
            "split": "train",
        },
        "status": "authorized_exact_dmath_cpu_acquisition_and_curation",
    }
    for field, value in expected.items():
        if plan.get(field) != value:
            raise ValueError(f"DMath acquisition plan changed: {field}")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(EXPECTED_BINDING_PATHS):
        raise ValueError("DMath acquisition binding set changed")
    for name, expected_path in EXPECTED_BINDING_PATHS.items():
        binding = bindings[name]
        if not isinstance(binding, Mapping) or binding.get("path") != expected_path:
            raise ValueError(f"DMath acquisition binding path changed: {name}")
        resolve_binding(repository_root, binding, name=name)
    replacement = _load_json(
        resolve_binding(repository_root, bindings["replacement_report"], name="replacement_report")
    )
    if replacement.get("report_sha256") != (
        "aaba91c90ea8d99a42cdeeb677df5826041d4e9de39ffa0012ae26b764a24ba4"
    ):
        raise ValueError("DMath replacement preflight seal changed")
    verification = _load_json(
        resolve_binding(
            repository_root,
            bindings["replacement_verification"],
            name="replacement_verification",
        )
    )
    if verification.get("verification_sha256") != (
        "3a21d15ae37097e76c48c5522a914ced9360369d104594f2a15e088f1a92d69e"
    ):
        raise ValueError("DMath replacement preflight verification changed")
    return plan


def git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload, usedforsecurity=False).hexdigest()


def default_open_url(request: urllib.request.Request) -> DownloadResponse:
    context = ssl.create_default_context(cafile=certifi.where())
    return urllib.request.urlopen(request, context=context, timeout=60)


def download_exact_object(
    specification: Mapping[str, Any],
    *,
    maximum_network_bytes: int,
    open_url: OpenURL = default_open_url,
) -> tuple[bytes, dict[str, Any]]:
    request = urllib.request.Request(
        str(specification["url"]),
        headers={"User-Agent": "lumi-dmath-acquisition-v0.1"},
    )
    chunks: list[bytes] = []
    observed_bytes = 0
    with open_url(request) as response:
        final_url = response.geturl()
        if final_url != specification["url"]:
            raise ValueError(f"DMath object redirected outside exact URL: {specification['path']}")
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            observed_bytes += len(chunk)
            if observed_bytes > int(specification["size_bytes"]):
                raise ValueError(f"DMath object exceeded exact size: {specification['path']}")
            if observed_bytes > maximum_network_bytes:
                raise ValueError("DMath acquisition exceeded network byte limit")
            chunks.append(chunk)
    payload = b"".join(chunks)
    if len(payload) != specification["size_bytes"]:
        raise ValueError(f"DMath object exact size changed: {specification['path']}")
    if git_blob_sha1(payload) != specification["git_blob_sha1"]:
        raise ValueError(f"DMath object Git blob SHA-1 changed: {specification['path']}")
    return payload, {
        "bytes": len(payload),
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "final_url": final_url,
        "git_blob_sha1": specification["git_blob_sha1"],
        "kind": specification["kind"],
        "path": specification["path"],
    }


def summarize_train_payload(payload: bytes) -> dict[str, Any]:
    if len(payload) > EXPECTED_LIMITS["maximum_decoded_bytes"]:
        raise ValueError("DMath train payload exceeded decoded byte limit")
    value = json.loads(payload)
    if isinstance(value, list):
        rows = value
        record_keys: list[str] | None = None
        root_type = "array"
    elif isinstance(value, Mapping):
        record_keys = list(value)
        expected_keys = [
            str(index) for index in range(1, EXPECTED_LIMITS["expected_train_rows"] + 1)
        ]
        if record_keys != expected_keys:
            raise ValueError("DMath train record-key sequence changed")
        rows = list(value.values())
        root_type = "object_by_one_based_record_id"
    else:
        raise ValueError("DMath train JSON root changed")
    if len(rows) != EXPECTED_LIMITS["expected_train_rows"]:
        raise ValueError("DMath train row count changed")
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("DMath train row shape changed")
    field_sets = {tuple(sorted(row)) for row in rows}
    summary = {
        "field_set_count": len(field_sets),
        "field_sets_sha256": sha256_value(sorted(field_sets)),
        "rows": len(rows),
        "root_type": root_type,
    }
    if record_keys is not None:
        summary["record_keys_sha256"] = sha256_value(record_keys)
    return summary


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite DMath artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def validate_dmath_acquisition_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported DMath acquisition report")
    if report.get("status") != "exact_dmath_source_acquired_private_nonproduction":
        raise ValueError("DMath acquisition status changed")
    if report.get("production", {}).get("production_eligible") is not False:
        raise ValueError("DMath acquisition cannot establish production eligibility")
    if report.get("production", {}).get("production_dataset_promotions") != 0:
        raise ValueError("DMath acquisition cannot promote records")
    if report.get("test_split_accessed") is not False:
        raise ValueError("DMath official test split must remain untouched")
    if report.get("raw_training_text_retained_in_report") is not False:
        raise ValueError("DMath acquisition report must remain raw-text-free")
    seal = _require_sha256(report.get("report_sha256"), "report_sha256")
    payload = {key: value for key, value in report.items() if key != "report_sha256"}
    if sha256_value(payload) != seal:
        raise ValueError("DMath acquisition report seal mismatch")
