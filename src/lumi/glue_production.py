"""Fail-closed packaging and authorization for Lumi's Glue production staging.

This module is deliberately local-only: it never calls AWS.  The first paid
phase only mirrors the 758 pinned source objects into deterministic,
content-addressed zstd chunks in private S3.  Mapping, deduplication, shard
publication, GPU work, and model training require separate runtime gates.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from .emr_serverless_production import build_emr_production_plan
from .glue_canary import (
    _artifact,
    _valid_commit,
    _validate_lingua_wheel_pin,
    _write_bytes_atomic,
    _zip_code,
)
from .production_corpus import _canonical_json_bytes, _read_json, _sha256_file

SCHEMA_VERSION = 1
PACKAGE_MANIFEST = "manifest.sha256"
RELEASE_FILE = "glue-production-stage-release-v0.1.json"
AUTHORIZATION_FILE = "glue-production-stage-authorization-v0.1.json"
CODE_ARCHIVE = "lumi-glue-production-code-v0.1.zip"
JOB_SCRIPT = "run_glue_production_stage.py"
OWNER_APPROVAL_TEXT = (
    "para nei mama 2k+ hole o amar fast and safe lagbe .150B koro budget to "
    "ektu kom beshi hoy e re mama real life e budget er moddhe kichui hoy na "
    "bhalo kichu and time maintain korle para nai chill mama credit to free "
    "amar to shes kora lagbe 25k abar nite"
)


def _fingerprint(value: dict[str, Any], *excluded: str) -> str:
    material = {key: row for key, row in value.items() if key not in set(excluded)}
    return hashlib.sha256(_canonical_json_bytes(material)).hexdigest()


def _read_contract(path: Path, root: Path) -> dict[str, Any]:
    contract = _read_json(path)
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Glue production contract schema")
    if contract.get("gate") != "lumi_glue_production_150b_plan_v0.1":
        raise ValueError("unexpected Glue production contract gate")
    if contract.get("authorization") != {
        "full_4b_training": False,
        "model_training": False,
        "paid_glue_dedup": False,
        "paid_glue_mapping": False,
        "paid_glue_sharding": False,
        "paid_glue_staging": False,
        "paid_gpu_compute": False,
        "production_candidate_acquisition": True,
    }:
        raise ValueError("Glue production contract authorization changed")
    expected_budget = {
        "hard_total_usd": "4000.00",
        "maximum_glue_compute_usd": "3600.00",
        "maximum_s3_logs_requests_usd": "200.00",
        "reserve_usd": "200.00",
    }
    if contract.get("budget") != expected_budget:
        raise ValueError("Glue production budget changed")
    if sum(
        Decimal(expected_budget[key])
        for key in ("maximum_glue_compute_usd", "maximum_s3_logs_requests_usd", "reserve_usd")
    ) != Decimal(expected_budget["hard_total_usd"]):
        raise ValueError("Glue production budget does not conserve its hard total")
    for pin in contract.get("inputs", {}).values():
        pinned = root / str(pin["path"])
        if not pinned.is_file() or _sha256_file(pinned) != pin["sha256"]:
            raise ValueError(f"Glue production input changed: {pin['path']}")
    limits = contract.get("limits", {})
    if limits != {
        "candidate_target_tokens": 150_000_000_000,
        "clean_category_floor_tokens": 120_000_000_000,
        "clean_shard_target_tokens": 100_000_000,
        "final_training_tokens": 80_000_000_000,
        "hard_logical_storage_bytes": 2_000_000_000_000,
        "maximum_category_overshoot_tokens": 100_000_000,
    }:
        raise ValueError("Glue production token or storage limits changed")
    staging = contract.get("staging", {})
    if staging != {
        "hard_maximum_dpu": 256,
        "maximum_runtime_minutes": 360,
        "maximum_source_compressed_bytes": 4_000_000_000,
        "source_files": 758,
        "target_chunk_uncompressed_bytes": 134_217_728,
    }:
        raise ValueError("Glue production staging envelope changed")
    return contract


def _manifest(directory: Path, paths: list[Path]) -> None:
    body = "".join(f"{_sha256_file(path)}  {path.name}\n" for path in paths)
    _write_bytes_atomic(directory / PACKAGE_MANIFEST, body.encode())


def build_glue_production_stage_package(
    *,
    contract_path: Path,
    repository_root: Path,
    source_commit: str,
    lingua_wheel_pin_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Build one deterministic, non-authorizing staging release."""

    root = repository_root.resolve()
    commit = _valid_commit(source_commit)
    contract = _read_contract(contract_path, root)
    emr_pin = contract["inputs"]["emr_resume_contract"]
    emr_plan = build_emr_production_plan(
        root / emr_pin["path"], repository_root=root, source_commit=commit
    )
    if emr_plan["decision"]["status"] != (
        "local_plan_ready_waiting_for_quota_and_exact_paid_authorization"
    ):
        raise ValueError("Glue production source plan is not locally ready")
    schedule = [dict(row) for row in emr_plan["source_schedule"]]
    staging = contract["staging"]
    if len(schedule) != int(staging["source_files"]):
        raise ValueError("Glue production source count changed")
    if any(
        int(row["source_bytes"]) > int(staging["maximum_source_compressed_bytes"])
        for row in schedule
    ):
        raise ValueError("Glue production source exceeds its per-object cap")
    resume = emr_plan["resume"]
    resume_ordinal = int(resume["source_ordinal"])
    resume_line = int(resume["source_next_line"])
    for row in schedule:
        row["start_after_line"] = resume_line if int(row["ordinal"]) == resume_ordinal else 0

    lingua_pin = _read_json(lingua_wheel_pin_path)
    _validate_lingua_wheel_pin(lingua_pin)
    output_directory.mkdir(parents=True, exist_ok=True)
    code_path = output_directory / CODE_ARCHIVE
    implementation = _zip_code(root, code_path)
    source_script = root / "scripts" / JOB_SCRIPT
    packaged_script = output_directory / JOB_SCRIPT
    _write_bytes_atomic(packaged_script, source_script.read_bytes())
    implementation[str(source_script.relative_to(root))] = _sha256_file(source_script)
    bucket = str(emr_plan["storage"]["bucket"])
    namespace = str(emr_plan["storage"]["namespace"])
    bootstrap = f"{namespace}/control/distributed/glue-production/bootstrap"
    code_sha = _sha256_file(code_path)
    script_sha = _sha256_file(source_script)
    artifacts = {
        "code_archive": _artifact(
            bucket=bucket,
            key=f"{bootstrap}/code/{code_sha}.zip",
            sha256=code_sha,
            size=code_path.stat().st_size,
            state="upload_required_before_authorized_launch",
            package_file=code_path.name,
        ),
        "job_script": _artifact(
            bucket=bucket,
            key=f"{bootstrap}/jobs/{script_sha}.py",
            sha256=script_sha,
            size=source_script.stat().st_size,
            state="upload_required_before_authorized_launch",
            package_file=packaged_script.name,
        ),
        "lingua_cp311_wheel": {
            **lingua_pin["wheel"],
            "state": "existing_verified_codebuild_object",
        },
    }
    pricing = _read_json(root / contract["inputs"]["glue_pricing"]["path"])
    job = {
        "additional_python_modules": "zstandard==0.25.0",
        "auto_scaling": False,
        "execution_class": "STANDARD",
        "glue_version": "5.1",
        "maximum_concurrent_runs": 1,
        "number_of_workers": int(staging["hard_maximum_dpu"]),
        "python_version": "3.11",
        "timeout_minutes": int(staging["maximum_runtime_minutes"]),
        "worker_type": "G.1X",
    }
    maximum_compute = (
        Decimal(job["number_of_workers"])
        * Decimal(job["timeout_minutes"])
        / Decimal(60)
        * Decimal(str(pricing["dpu_hour_usd"]))
    ).quantize(Decimal("0.000001"))
    release = {
        "artifacts": artifacts,
        "authorization": dict(contract["authorization"]),
        "budget": dict(contract["budget"]),
        "candidate_target_tokens": int(contract["limits"]["candidate_target_tokens"]),
        "gate": "lumi_glue_production_stage_release_v0.1",
        "implementation": implementation,
        "job": job,
        "limits": {
            "hard_logical_storage_bytes": int(
                contract["limits"]["hard_logical_storage_bytes"]
            ),
            "maximum_compute_usd": str(maximum_compute),
            "maximum_source_compressed_bytes": int(
                staging["maximum_source_compressed_bytes"]
            ),
            "source_files": len(schedule),
            "target_chunk_uncompressed_bytes": int(
                staging["target_chunk_uncompressed_bytes"]
            ),
        },
        "lingua_compatibility": lingua_pin,
        "output": {
            "result_prefix": f"{namespace}/control/distributed/glue-production/stage-runs",
            "staged_prefix": f"{namespace}/transient/glue-production/raw",
        },
        "region": str(contract["region"]),
        "resume": {
            "accepted_documents": int(resume["accepted_documents"]),
            "accepted_tokens": int(resume["accepted_tokens"]),
            "source_next_line": resume_line,
            "source_ordinal": resume_ordinal,
        },
        "schema_version": SCHEMA_VERSION,
        "source_commit": commit,
        "source_inventory": dict(emr_plan["source_inventory"]),
        "storage": {
            "bucket": bucket,
            "namespace": namespace,
        },
        "sources": schedule,
    }
    release["release_fingerprint"] = _fingerprint(release, "release_fingerprint")
    validate_glue_production_stage_release(release)
    release_path = output_directory / RELEASE_FILE
    _write_bytes_atomic(
        release_path,
        (json.dumps(release, indent=2, sort_keys=True) + "\n").encode(),
    )
    _manifest(output_directory, [code_path, release_path, packaged_script])
    return release


def validate_glue_production_stage_release(release: dict[str, Any]) -> dict[str, Any]:
    if release.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Glue production stage release schema")
    if release.get("gate") != "lumi_glue_production_stage_release_v0.1":
        raise ValueError("unexpected Glue production stage release gate")
    if release.get("release_fingerprint") != _fingerprint(
        release, "release_fingerprint"
    ):
        raise ValueError("Glue production stage release fingerprint changed")
    authorization = release.get("authorization", {})
    if authorization.get("production_candidate_acquisition") is not True or any(
        value
        for key, value in authorization.items()
        if key != "production_candidate_acquisition"
    ):
        raise ValueError("Glue production stage release authorizes paid work")
    sources = release.get("sources")
    limits = release.get("limits", {})
    if not isinstance(sources, list) or len(sources) != limits.get("source_files"):
        raise ValueError("Glue production stage source inventory changed")
    if len(sources) != 758:
        raise ValueError("Glue production stage must bind all 758 source objects")
    ordinals = [int(row["ordinal"]) for row in sources]
    if ordinals != list(range(len(sources))):
        raise ValueError("Glue production stage source order changed")
    if len({str(row["source_sha256"]) for row in sources}) != len(sources):
        raise ValueError("Glue production stage repeats a source digest")
    storage = release.get("storage", {})
    if not isinstance(storage.get("bucket"), str) or not storage["bucket"]:
        raise ValueError("Glue production stage bucket is missing")
    if storage.get("namespace") != "lumi-corpus/v0.1":
        raise ValueError("Glue production stage namespace changed")
    if any(
        int(row["source_bytes"]) > int(limits["maximum_source_compressed_bytes"])
        for row in sources
    ):
        raise ValueError("Glue production stage source byte cap changed")
    resume = release.get("resume", {})
    for row in sources:
        expected = (
            int(resume["source_next_line"])
            if int(row["ordinal"]) == int(resume["source_ordinal"])
            else 0
        )
        if int(row.get("start_after_line", -1)) != expected:
            raise ValueError("Glue production stage resume line changed")
    job = release.get("job", {})
    if job.get("number_of_workers") != 256 or job.get("timeout_minutes") != 360:
        raise ValueError("Glue production stage capacity changed")
    if job.get("worker_type") != "G.1X" or job.get("glue_version") != "5.1":
        raise ValueError("Glue production stage runtime changed")
    maximum = (
        Decimal(job["number_of_workers"])
        * Decimal(job["timeout_minutes"])
        / Decimal(60)
        * Decimal("0.44")
    ).quantize(Decimal("0.000001"))
    if Decimal(str(limits.get("maximum_compute_usd"))) != maximum:
        raise ValueError("Glue production stage cost envelope changed")
    _validate_lingua_wheel_pin(release.get("lingua_compatibility"))
    return {
        "maximum_compute_usd": str(maximum),
        "release_fingerprint": release["release_fingerprint"],
        "source_bytes": sum(int(row["source_bytes"]) for row in sources),
        "source_files": len(sources),
        "status": "verified_not_authorized",
    }


def build_glue_production_stage_authorization(
    release: dict[str, Any], *, now: datetime | None = None
) -> dict[str, Any]:
    """Bind the owner's recorded 150B approval to exactly one staging run."""

    validate_glue_production_stage_release(release)
    issued = (now or datetime.now(UTC)).astimezone(UTC)
    authorization = {
        "approval_text": OWNER_APPROVAL_TEXT,
        "bounds": {
            "authorized_runs": 1,
            "maximum_dpu": 256,
            "maximum_runtime_minutes": 360,
            "maximum_usd": release["limits"]["maximum_compute_usd"],
            "source_files": 758,
        },
        "expires_at": (issued + timedelta(hours=24)).isoformat(),
        "gate": "lumi_glue_production_stage_authorization_v0.1",
        "issued_at": issued.isoformat(),
        "permissions": {
            "full_4b_training": False,
            "model_training": False,
            "paid_glue_dedup": False,
            "paid_glue_mapping": False,
            "paid_glue_sharding": False,
            "paid_glue_staging": True,
            "paid_gpu_compute": False,
        },
        "release_fingerprint": release["release_fingerprint"],
        "schema_version": SCHEMA_VERSION,
        "source_commit": release["source_commit"],
    }
    authorization["authorization_sha256"] = _fingerprint(
        authorization, "authorization_sha256"
    )
    validate_glue_production_stage_authorization(authorization, release, now=issued)
    return authorization


def validate_glue_production_stage_authorization(
    authorization: dict[str, Any],
    release: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    validate_glue_production_stage_release(release)
    if authorization.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Glue production stage authorization schema")
    if authorization.get("gate") != "lumi_glue_production_stage_authorization_v0.1":
        raise ValueError("unexpected Glue production stage authorization gate")
    if authorization.get("authorization_sha256") != _fingerprint(
        authorization, "authorization_sha256"
    ):
        raise ValueError("Glue production stage authorization fingerprint changed")
    if authorization.get("approval_text") != OWNER_APPROVAL_TEXT:
        raise ValueError("Glue production stage owner approval text changed")
    if authorization.get("release_fingerprint") != release["release_fingerprint"]:
        raise ValueError("Glue production stage authorization targets another release")
    if authorization.get("source_commit") != release["source_commit"]:
        raise ValueError("Glue production stage authorization source commit changed")
    if authorization.get("permissions") != {
        "full_4b_training": False,
        "model_training": False,
        "paid_glue_dedup": False,
        "paid_glue_mapping": False,
        "paid_glue_sharding": False,
        "paid_glue_staging": True,
        "paid_gpu_compute": False,
    }:
        raise ValueError("Glue production stage authorization permissions changed")
    if authorization.get("bounds") != {
        "authorized_runs": 1,
        "maximum_dpu": 256,
        "maximum_runtime_minutes": 360,
        "maximum_usd": release["limits"]["maximum_compute_usd"],
        "source_files": 758,
    }:
        raise ValueError("Glue production stage authorization bounds changed")
    current = (now or datetime.now(UTC)).astimezone(UTC)
    issued = datetime.fromisoformat(str(authorization["issued_at"])).astimezone(UTC)
    expires = datetime.fromisoformat(str(authorization["expires_at"])).astimezone(UTC)
    if expires <= issued or expires - issued > timedelta(hours=24):
        raise ValueError("Glue production stage authorization lifetime changed")
    if current < issued - timedelta(minutes=5) or current >= expires:
        raise ValueError("Glue production stage authorization is not currently valid")
    return {
        "authorization_sha256": authorization["authorization_sha256"],
        "maximum_compute_usd": authorization["bounds"]["maximum_usd"],
        "status": "authorized_one_stage_run",
    }


def write_glue_production_stage_authorization(
    release_path: Path,
    output_path: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    release = _read_json(release_path)
    authorization = build_glue_production_stage_authorization(release, now=now)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(authorization, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return authorization
