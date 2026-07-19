"""Plan and package Lumi's bounded AWS Glue corpus canary.

This module never calls AWS.  It creates a content-addressed deployment
package and validates the separate owner authorization that a paid Glue job
must receive at runtime.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .distributed_corpus import build_distributed_corpus_plan
from .production_corpus import (
    _canonical_json_bytes,
    _component_provenance,
    _read_json,
    _sha256_file,
    build_round_robin_source_schedule,
)

CANARY_APPROVAL_TEXT = (
    "I approve the bounded Lumi AWS Glue distributed corpus canary only."
)
CANARY_RELEASE_SCHEMA_VERSION = 1
CANARY_AUTHORIZATION_SCHEMA_VERSION = 1
PACKAGE_MANIFEST = "manifest.sha256"
RELEASE_FILE = "glue-canary-release-v0.1.json"
CODE_ARCHIVE = "lumi-glue-code-v0.1.zip"
JOB_SCRIPT = "run_glue_distributed_canary.py"


def _valid_commit(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("Glue canary source commit must be a full Git SHA-1")
    return value


def _write_bytes_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _zip_code(root: Path, destination: Path) -> dict[str, str]:
    source_files = sorted((root / "src" / "lumi").glob("*.py"))
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    implementation = {}
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for source in source_files:
            relative = source.relative_to(root)
            implementation[str(relative)] = _sha256_file(source)
            info = zipfile.ZipInfo(f"lumi/{source.name}", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)
    temporary.replace(destination)
    return implementation


def _artifact(
    *,
    bucket: str,
    key: str,
    sha256: str,
    size: int,
    state: str,
    package_file: str | None = None,
) -> dict[str, Any]:
    result = {
        "bytes": size,
        "s3_uri": f"s3://{bucket}/{key}",
        "sha256": sha256,
        "state": state,
    }
    if package_file is not None:
        result["package_file"] = package_file
    return result


def _select_canary_sources(
    schedule: list[dict[str, Any]],
    canary: dict[str, Any],
    baseline_partial: dict[str, Any],
) -> list[dict[str, Any]]:
    """Select a bounded all-category slice while retaining the exact resume source."""

    requested = int(canary["source_files"])
    resume_ordinal = int(baseline_partial["source_ordinal"])
    resume = [row for row in schedule if int(row["ordinal"]) == resume_ordinal]
    if len(resume) != 1:
        raise ValueError("Glue canary resume source changed")
    chosen = {resume_ordinal: resume[0]}
    categories = sorted({str(row["category"]) for row in schedule})
    if requested < len(categories):
        raise ValueError("Glue canary cannot cover every corpus category")
    for category in categories:
        if category == resume[0]["category"]:
            continue
        candidates = [row for row in schedule if row["category"] == category]
        smallest = min(
            candidates,
            key=lambda row: (
                int(row["source_bytes"]),
                int(row["ordinal"]),
                str(row["source_file"]),
            ),
        )
        chosen[int(smallest["ordinal"])] = smallest
    remaining = [row for row in schedule if int(row["ordinal"]) not in chosen]
    remaining.sort(
        key=lambda row: (
            int(row["source_bytes"]),
            int(row["ordinal"]),
            str(row["source_file"]),
        )
    )
    for row in remaining:
        if len(chosen) >= requested:
            break
        chosen[int(row["ordinal"])] = row
    selected = sorted(chosen.values(), key=lambda row: int(row["ordinal"]))
    if len(selected) != requested:
        raise ValueError("Glue canary source count changed")
    maximum_source = int(canary["maximum_source_compressed_bytes"])
    maximum_total = int(canary["maximum_total_source_compressed_bytes"])
    if any(int(row["source_bytes"]) > maximum_source for row in selected):
        raise ValueError("Glue canary source byte cap exceeded")
    if sum(int(row["source_bytes"]) for row in selected) > maximum_total:
        raise ValueError("Glue canary total source byte cap exceeded")
    if {str(row["category"]) for row in selected} != set(categories):
        raise ValueError("Glue canary category coverage changed")
    return selected


def build_glue_canary_package(
    *,
    contract_path: Path,
    repository_root: Path,
    source_commit: str,
    anchor_export_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Create the deterministic, non-authorizing Glue canary package."""

    root = repository_root.resolve()
    commit = _valid_commit(source_commit)
    plan = build_distributed_corpus_plan(
        contract_path,
        repository_root=root,
        source_commit=commit,
    )
    if plan["decision"]["paid_glue_canary"] != "not_authorized":
        raise ValueError("Glue canary package unexpectedly authorizes paid compute")
    contract = _read_json(contract_path)
    paths = {
        name: root / pin["path"] for name, pin in plan["inputs"].items()
    }
    baseline = _read_json(paths["baseline_evidence"])
    anchor_pin = baseline["derived_resume_anchor"]
    anchor_path = anchor_export_directory / "resume-anchors-v0.1.jsonl.gz"
    if not anchor_path.is_file():
        raise FileNotFoundError(f"missing verified resume anchors: {anchor_path}")
    if anchor_path.stat().st_size != anchor_pin["bytes"]:
        raise ValueError("Glue canary resume anchor byte size changed")
    if _sha256_file(anchor_path) != anchor_pin["sha256"]:
        raise ValueError("Glue canary resume anchor SHA-256 changed")

    source_manifest = _read_json(paths["source_file_manifest"])
    mixture = _read_json(paths["data_mix"])
    schedule, _ = build_round_robin_source_schedule(source_manifest, mixture)
    provenance = _component_provenance(_read_json(paths["component_inventory"]))
    baseline_partial = baseline["partial_prefix"]
    canary = contract["canary"]
    selected = _select_canary_sources(schedule, canary, baseline_partial)
    sources = []
    for source in selected:
        start_line = 1
        if source["ordinal"] == baseline_partial["source_ordinal"]:
            start_line = int(baseline_partial["source_next_line"]) + 1
        sources.append(
            {
                **source,
                "maximum_records": int(canary["maximum_records_per_source"]),
                "source_provenance": provenance[source["component"]],
                "start_line": start_line,
            }
        )
    if len(sources) != 8 or sources[0]["start_line"] != 38_849:
        raise ValueError("Glue canary source slice changed")

    output_directory.mkdir(parents=True, exist_ok=True)
    code_path = output_directory / CODE_ARCHIVE
    implementation = _zip_code(root, code_path)
    job_script_path = root / "scripts" / JOB_SCRIPT
    packaged_job_script = output_directory / JOB_SCRIPT
    _write_bytes_atomic(packaged_job_script, job_script_path.read_bytes())
    implementation[str(job_script_path.relative_to(root))] = _sha256_file(job_script_path)
    quality = _read_json(paths["data_quality"])
    benchmark_path = root / quality["benchmark_decontamination"]["registry"]
    if _sha256_file(benchmark_path) != quality["benchmark_decontamination"][
        "registry_sha256"
    ]:
        raise ValueError("Glue canary benchmark registry changed")
    bucket = baseline["bucket"]
    namespace = baseline["namespace"]
    bootstrap_prefix = f"{namespace}/control/distributed/bootstrap"
    code_sha256 = _sha256_file(code_path)
    job_sha256 = _sha256_file(job_script_path)
    benchmark_sha256 = _sha256_file(benchmark_path)
    packaged_benchmark = output_directory / benchmark_path.name
    _write_bytes_atomic(packaged_benchmark, benchmark_path.read_bytes())
    packaged_anchors = output_directory / anchor_path.name
    _write_bytes_atomic(packaged_anchors, anchor_path.read_bytes())
    external = contract["external_artifacts"]
    artifacts = {
        "benchmark_registry": _artifact(
            bucket=bucket,
            key=f"{bootstrap_prefix}/benchmark/{benchmark_sha256}.jsonl.gz",
            sha256=benchmark_sha256,
            size=benchmark_path.stat().st_size,
            state="upload_required_before_authorized_launch",
            package_file=packaged_benchmark.name,
        ),
        "code_archive": _artifact(
            bucket=bucket,
            key=f"{bootstrap_prefix}/code/{code_sha256}.zip",
            sha256=code_sha256,
            size=code_path.stat().st_size,
            state="upload_required_before_authorized_launch",
            package_file=code_path.name,
        ),
        "frozen_tokenizer": _artifact(
            bucket=bucket,
            key=external["frozen_tokenizer"]["s3_key"],
            sha256=external["frozen_tokenizer"]["sha256"],
            size=external["frozen_tokenizer"]["bytes"],
            state="existing_verified_production_object",
        ),
        "job_script": _artifact(
            bucket=bucket,
            key=f"{bootstrap_prefix}/jobs/{job_sha256}.py",
            sha256=job_sha256,
            size=job_script_path.stat().st_size,
            state="upload_required_before_authorized_launch",
            package_file=packaged_job_script.name,
        ),
        "resume_anchors": _artifact(
            bucket=bucket,
            key=f"{bootstrap_prefix}/anchors/{anchor_pin['sha256']}.jsonl.gz",
            sha256=anchor_pin["sha256"],
            size=anchor_pin["bytes"],
            state="upload_required_before_authorized_launch",
            package_file=packaged_anchors.name,
        ),
        "training_exclusion_ledger": _artifact(
            bucket=bucket,
            key=external["training_exclusion_ledger"]["s3_key"],
            sha256=external["training_exclusion_ledger"]["sha256"],
            size=external["training_exclusion_ledger"]["bytes"],
            state="existing_verified_production_object",
        ),
    }
    release_material = {
        "artifacts": artifacts,
        "authorization": {
            "full_4b_training": False,
            "model_training": False,
            "paid_glue_canary": False,
            "paid_glue_production": False,
            "paid_gpu_compute": False,
        },
        "canary": dict(canary),
        "cost": dict(plan["cost"]),
        "data_quality": quality,
        "deduplication": {
            "maximum_candidate_pairs": plan["backend"]["maximum_candidate_pairs"],
            "maximum_lsh_bucket_documents": plan["backend"][
                "maximum_lsh_bucket_documents"
            ],
            "maximum_rounds": plan["backend"]["maximum_near_dedup_rounds"],
            "strategy": plan["backend"]["deduplication"],
        },
        "filters": _read_json(paths["filter_policy"])["filters"],
        "gate": "lumi_distributed_glue_canary_release_v0.1",
        "implementation": implementation,
        "job": {
            "additional_python_modules": (
                "tokenizers==0.22.2,datasketch==2.0.0,"
                "lingua-language-detector==2.2.0"
            ),
            "auto_scaling": False,
            "execution_class": "STANDARD",
            "glue_version": "5.1",
            "maximum_concurrent_runs": 1,
            "number_of_workers": 8,
            "python_version": "3.11",
            "timeout_minutes": 30,
            "worker_type": "G.1X",
        },
        "output": {
            "checkpoint_prefix": f"{namespace}/control/distributed/canary/checkpoints",
            "result_prefix": f"{namespace}/control/distributed/canary/results",
            "write_production_shards": False,
        },
        "plan_fingerprint": plan["plan_fingerprint"],
        "resume_baseline": dict(plan["resume_baseline"]),
        "schema_version": CANARY_RELEASE_SCHEMA_VERSION,
        "source_commit": commit,
        "sources": sources,
    }
    fingerprint = hashlib.sha256(_canonical_json_bytes(release_material)).hexdigest()
    release = {
        **release_material,
        "decision": {
            "paid_glue_canary": "not_authorized",
            "paid_glue_production": "not_authorized",
            "status": "ready_for_local_verification_and_owner_approval_only",
        },
        "release_fingerprint": fingerprint,
    }
    release_path = output_directory / RELEASE_FILE
    _write_bytes_atomic(
        release_path,
        (json.dumps(release, indent=2, sort_keys=True) + "\n").encode(),
    )
    manifest_paths = [
        packaged_benchmark,
        release_path,
        code_path,
        packaged_anchors,
        packaged_job_script,
    ]
    manifest = "".join(
        f"{_sha256_file(path)}  {path.name}\n" for path in manifest_paths
    )
    _write_bytes_atomic(output_directory / PACKAGE_MANIFEST, manifest.encode())
    return release


def validate_glue_canary_release(release: dict[str, Any]) -> dict[str, Any]:
    """Validate a rendered release independently of its package manifest."""

    if release.get("schema_version") != CANARY_RELEASE_SCHEMA_VERSION:
        raise ValueError("unsupported Glue canary release schema")
    if release.get("gate") != "lumi_distributed_glue_canary_release_v0.1":
        raise ValueError("unexpected Glue canary release gate")
    material = {
        key: value
        for key, value in release.items()
        if key not in {"decision", "release_fingerprint"}
    }
    fingerprint = hashlib.sha256(_canonical_json_bytes(material)).hexdigest()
    if fingerprint != release.get("release_fingerprint"):
        raise ValueError("Glue canary release fingerprint changed")
    authorization = release.get("authorization", {})
    if any(authorization.values()):
        raise ValueError("Glue canary release must not authorize paid compute or training")
    if release.get("output", {}).get("write_production_shards") is not False:
        raise ValueError("Glue canary release may not write production shards")
    if release.get("job") != {
        "additional_python_modules": (
            "tokenizers==0.22.2,datasketch==2.0.0,"
            "lingua-language-detector==2.2.0"
        ),
        "auto_scaling": False,
        "execution_class": "STANDARD",
        "glue_version": "5.1",
        "maximum_concurrent_runs": 1,
        "number_of_workers": 8,
        "python_version": "3.11",
        "timeout_minutes": 30,
        "worker_type": "G.1X",
    }:
        raise ValueError("Glue canary job envelope changed")
    if len(release.get("sources", [])) != 8:
        raise ValueError("Glue canary source count changed")
    canary = release.get("canary", {})
    if canary != {
        "maximum_dpu": 8,
        "maximum_records_per_source": 2_000,
        "maximum_runtime_minutes": 30,
        "maximum_source_compressed_bytes": 700_000_000,
        "maximum_total_source_compressed_bytes": 1_500_000_000,
        "requires_new_owner_approval": True,
        "source_files": 8,
    }:
        raise ValueError("Glue canary contract changed")
    sources = release["sources"]
    if any(int(source["source_bytes"]) > 700_000_000 for source in sources):
        raise ValueError("Glue canary source byte cap exceeded")
    if sum(int(source["source_bytes"]) for source in sources) > 1_500_000_000:
        raise ValueError("Glue canary total source byte cap exceeded")
    if any(int(source["maximum_records"]) != 2_000 for source in sources):
        raise ValueError("Glue canary record cap changed")
    expected_categories = {
        "books",
        "code",
        "government_legal_and_patents",
        "open_web",
        "questions_and_discussion",
        "reference",
        "science_and_education",
    }
    if {str(source["category"]) for source in sources} != expected_categories:
        raise ValueError("Glue canary category coverage changed")
    return {"release_fingerprint": fingerprint, "status": "verified_not_authorized"}


def verify_glue_canary_package(output_directory: Path) -> dict[str, Any]:
    """Verify package hashes, release fingerprint, and non-authorization."""

    manifest_path = output_directory / PACKAGE_MANIFEST
    rows = manifest_path.read_text(encoding="utf-8").splitlines()
    expected = {
        CODE_ARCHIVE,
        RELEASE_FILE,
        "benchmark-signatures-v0.1.jsonl.gz",
        "resume-anchors-v0.1.jsonl.gz",
        JOB_SCRIPT,
    }
    seen = set()
    for line in rows:
        digest, name = line.split("  ", 1)
        if name not in expected or name in seen:
            raise ValueError("Glue canary package manifest changed")
        if _sha256_file(output_directory / name) != digest:
            raise ValueError(f"Glue canary package hash changed: {name}")
        seen.add(name)
    if seen != expected:
        raise ValueError("Glue canary package manifest is incomplete")
    release = _read_json(output_directory / RELEASE_FILE)
    validated = validate_glue_canary_release(release)
    for artifact in release["artifacts"].values():
        package_file = artifact.get("package_file")
        if package_file is None:
            continue
        path = output_directory / package_file
        if path.stat().st_size != artifact["bytes"] or _sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"Glue canary packaged artifact changed: {package_file}")
    with zipfile.ZipFile(output_directory / CODE_ARCHIVE) as archive:
        names = set(archive.namelist())
        if "lumi/glue_corpus.py" not in names or "lumi/distributed_corpus.py" not in names:
            raise ValueError("Glue canary code archive is incomplete")
        if any(name.startswith("/") or ".." in Path(name).parts for name in names):
            raise ValueError("Glue canary code archive contains an unsafe path")
    return {
        "code_archive_sha256": _sha256_file(output_directory / CODE_ARCHIVE),
        "release_fingerprint": validated["release_fingerprint"],
        "release_sha256": _sha256_file(output_directory / RELEASE_FILE),
        "status": "verified_not_authorized",
    }


def validate_glue_canary_authorization(
    authorization: dict[str, Any],
    release: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Fail unless a separate authorization matches the bounded release."""

    if authorization.get("schema_version") != CANARY_AUTHORIZATION_SCHEMA_VERSION:
        raise ValueError("unsupported Glue canary authorization schema")
    if authorization.get("gate") != "lumi_distributed_glue_canary_authorization_v0.1":
        raise ValueError("unexpected Glue canary authorization gate")
    if authorization.get("approval_text") != CANARY_APPROVAL_TEXT:
        raise ValueError("Glue canary authorization text changed")
    if authorization.get("owner") != "Afnan Technologies Inc.":
        raise ValueError("Glue canary authorization owner changed")
    if authorization.get("release_fingerprint") != release.get("release_fingerprint"):
        raise ValueError("Glue canary authorization targets another release")
    if authorization.get("source_commit") != release.get("source_commit"):
        raise ValueError("Glue canary authorization targets another source commit")
    expected = {
        "full_4b_training": False,
        "model_training": False,
        "paid_glue_canary": True,
        "paid_glue_production": False,
        "paid_gpu_compute": False,
    }
    if authorization.get("authorization") != expected:
        raise ValueError("Glue canary authorization boundary changed")
    bounds = authorization.get("bounds", {})
    if bounds != {
        "maximum_dpu": 8,
        "maximum_records_per_source": 2_000,
        "maximum_source_compressed_bytes": 700_000_000,
        "maximum_runtime_minutes": 30,
        "maximum_source_files": 8,
        "maximum_total_source_compressed_bytes": 1_500_000_000,
        "maximum_usd": "1.76",
    }:
        raise ValueError("Glue canary authorization bounds changed")
    try:
        expires = datetime.fromisoformat(str(authorization["expires_at"]).replace("Z", "+00:00"))
    except (KeyError, ValueError) as error:
        raise ValueError("Glue canary authorization expiry is invalid") from error
    current = now or datetime.now(UTC)
    if expires.tzinfo is None or expires <= current.astimezone(UTC):
        raise ValueError("Glue canary authorization has expired")
    return {
        "expires_at": expires.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "release_fingerprint": release["release_fingerprint"],
        "status": "authorized_bounded_canary_only",
    }
