"""Package Lumi's bounded EMR Serverless corpus canary without launching AWS.

The Glue 5.1 canary proved that its fixed Python 3.11 runtime cannot install
Lumi's pinned Lingua 2.2.0 dependency. This module preserves the exact data and
Spark semantics while preparing an EMR 7.13.0 custom-image release with Python
3.12. Building the image and running the canary remain separately authorized.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import zipfile
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .distributed_corpus import build_distributed_corpus_plan
from .glue_canary import (
    _artifact,
    _select_canary_sources,
    _valid_commit,
    _write_bytes_atomic,
    _zip_code,
)
from .production_corpus import (
    _canonical_json_bytes,
    _component_provenance,
    _read_json,
    _sha256_file,
    build_round_robin_source_schedule,
)

EMR_CANARY_APPROVAL_TEXT = (
    "I approve one bounded Lumi AWS CodeBuild image build and one bounded AWS EMR "
    "Serverless distributed corpus canary only."
)
EMR_RELEASE_SCHEMA_VERSION = 1
EMR_AUTHORIZATION_SCHEMA_VERSION = 1
MAX_AUTHORIZATION_LIFETIME = timedelta(hours=2)
PACKAGE_MANIFEST = "manifest.sha256"
RELEASE_FILE = "emr-serverless-canary-release-v0.1.json"
CODE_ARCHIVE = "lumi-emr-serverless-code-v0.1.zip"
IMAGE_CONTEXT = "lumi-emr-serverless-image-context-v0.1.zip"
JOB_SCRIPT = "run_emr_serverless_distributed_canary.py"
BASE_IMAGE_DIGEST = "sha256:3e465888c2b00add9d298a2b4c14c7951262d9c2b35a3f8c1b109dc21cd9714d"
BASE_IMAGE_URI = (
    "public.ecr.aws/emr-serverless/spark/emr-7.13.0@" + BASE_IMAGE_DIGEST
)
PRIVATE_IMAGE_PATTERN = re.compile(
    r"^(?P<account>[0-9]{12})\.dkr\.ecr\.us-east-1\.amazonaws\.com/"
    r"lumi-emr-serverless@(?P<digest>sha256:[0-9a-f]{64})$"
)


def _zip_named_files(destination: Path, files: list[tuple[Path, str]]) -> None:
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for source, name in sorted(files, key=lambda row: row[1]):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)
    temporary.replace(destination)


def _verify_pin(root: Path, pin: dict[str, Any], label: str) -> Path:
    path = root / str(pin.get("path", ""))
    if not path.is_file():
        raise FileNotFoundError(f"missing EMR canary {label}: {path}")
    if _sha256_file(path) != pin.get("sha256"):
        raise ValueError(f"EMR canary pinned input changed: {label}")
    return path


def _validate_contract(contract: dict[str, Any], root: Path) -> dict[str, Path]:
    if contract.get("schema_version") != 1:
        raise ValueError("unsupported EMR Serverless canary contract schema")
    if contract.get("gate") != "distributed_production_corpus_emr_serverless_canary_v0.1":
        raise ValueError("unexpected EMR Serverless canary gate")
    expected_authorization = {
        "full_4b_training": False,
        "model_training": False,
        "paid_codebuild_image": False,
        "paid_emr_serverless_canary": False,
        "paid_emr_serverless_production": False,
        "paid_gpu_compute": False,
    }
    if contract.get("authorization") != expected_authorization:
        raise ValueError("EMR canary contract must not authorize paid work or training")
    pins = contract.get("inputs", {})
    if set(pins) != {"codebuild_pricing", "emr_pricing", "source_contract"}:
        raise ValueError("EMR canary input set changed")
    paths = {name: _verify_pin(root, pin, name) for name, pin in pins.items()}
    image = contract.get("image", {})
    if image.get("base_digest") != BASE_IMAGE_DIGEST:
        raise ValueError("EMR Serverless base image digest changed")
    if image.get("base_uri") != BASE_IMAGE_URI:
        raise ValueError("EMR Serverless base image URI changed")
    for label in ("dockerfile", "requirements_lock", "verifier"):
        paths[label] = _verify_pin(root, image.get(label, {}), label)
    if image.get("python_minor") != "3.12" or image.get("python_executable") != (
        "/opt/lumi/python312/bin/python"
    ):
        raise ValueError("EMR canary Python runtime changed")
    if image.get("maximum_image_gib") != 10:
        raise ValueError("EMR canary image-size cap changed")

    runtime = contract.get("runtime", {})
    if runtime != {
        "release_label": "emr-7.13.0",
        "service": "aws_emr_serverless",
        "spark_version": "3.5.6",
    }:
        raise ValueError("EMR canary runtime changed")
    application = contract.get("application", {})
    if application != {
        "architecture": "X86_64",
        "auto_start": True,
        "auto_stop_idle_timeout_minutes": 5,
        "initial_capacity_workers": 0,
        "maximum_disk_gib": 80,
        "maximum_memory_gib": 64,
        "maximum_vcpu": 16,
    }:
        raise ValueError("EMR canary application capacity changed")
    job = contract.get("job", {})
    if job != {
        "driver": {"cores": 4, "disk_gib": 20, "memory_gib": 14},
        "dynamic_allocation": False,
        "executor": {"cores": 4, "disk_gib": 20, "instances": 3, "memory_gib": 14},
        "spark_shuffle_partitions": 12,
    }:
        raise ValueError("EMR canary Spark worker shape changed")
    canary = contract.get("canary", {})
    if canary != {
        "maximum_records_per_source": 2_000,
        "maximum_runtime_minutes": 30,
        "maximum_source_compressed_bytes": 700_000_000,
        "maximum_total_source_compressed_bytes": 1_500_000_000,
        "requires_new_owner_approval": True,
        "source_files": 8,
    }:
        raise ValueError("EMR canary source/runtime envelope changed")
    image_build = contract.get("image_build", {})
    if image_build != {
        "compute_type": "BUILD_GENERAL1_SMALL",
        "maximum_build_minutes": 30,
        "maximum_compute_usd": "0.15",
        "requires_new_owner_approval": True,
    }:
        raise ValueError("EMR canary image-build envelope changed")
    if contract.get("region") != "us-east-1":
        raise ValueError("EMR canary region changed")
    return paths


def _cost_envelope(contract: dict[str, Any], paths: dict[str, Path]) -> dict[str, str]:
    emr = _read_json(paths["emr_pricing"])
    codebuild = _read_json(paths["codebuild_pricing"])
    if (
        emr.get("region") != "us-east-1"
        or emr.get("architecture") != "X86_64"
        or emr.get("vcpu_hour_usd") != "0.052624"
        or emr.get("memory_gb_hour_usd") != "0.0057785"
        or emr.get("standard_storage_gb_hour_usd") != "0.000111"
    ):
        raise ValueError("EMR Serverless pricing snapshot changed")
    if (
        codebuild.get("compute_type") != "BUILD_GENERAL1_SMALL"
        or codebuild.get("linux_x86_build_minute_usd") != "0.005"
        or codebuild.get("maximum_build_minutes") != 30
        or codebuild.get("maximum_compute_usd") != "0.15"
    ):
        raise ValueError("CodeBuild pricing snapshot changed")
    application = contract["application"]
    minutes = Decimal(contract["canary"]["maximum_runtime_minutes"])
    hourly = (
        Decimal(application["maximum_vcpu"]) * Decimal(emr["vcpu_hour_usd"])
        + Decimal(application["maximum_memory_gib"])
        * Decimal(emr["memory_gb_hour_usd"])
    )
    maximum_emr = (hourly * minutes / Decimal(60)).quantize(Decimal("0.000001"))
    if maximum_emr != Decimal("0.605904"):
        raise ValueError("EMR canary compute cost envelope changed")
    maximum_codebuild = Decimal(codebuild["maximum_compute_usd"])
    return {
        "codebuild_compute_hard_maximum_usd": str(maximum_codebuild),
        "combined_compute_hard_maximum_usd": str(maximum_emr + maximum_codebuild),
        "emr_compute_hard_maximum_usd": str(maximum_emr),
        "memory_gb_hour_usd": emr["memory_gb_hour_usd"],
        "standard_storage_extra_gb_hour_usd": emr["standard_storage_gb_hour_usd"],
        "vcpu_hour_usd": emr["vcpu_hour_usd"],
    }


def _release_fingerprint(material: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(material)).hexdigest()


def build_emr_serverless_canary_package(
    *,
    contract_path: Path,
    repository_root: Path,
    source_commit: str,
    anchor_export_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Create a deterministic, image-unbound, non-authorizing EMR package."""

    root = repository_root.resolve()
    commit = _valid_commit(source_commit)
    contract = _read_json(contract_path)
    paths = _validate_contract(contract, root)
    source_plan = build_distributed_corpus_plan(
        paths["source_contract"],
        repository_root=root,
        source_commit=commit,
    )
    if source_plan["decision"]["paid_glue_canary"] != "not_authorized":
        raise ValueError("source corpus plan unexpectedly authorizes paid work")
    source_contract = _read_json(paths["source_contract"])
    source_paths = {
        name: root / pin["path"] for name, pin in source_plan["inputs"].items()
    }
    baseline = _read_json(source_paths["baseline_evidence"])
    anchor_pin = baseline["derived_resume_anchor"]
    anchor_path = anchor_export_directory / "resume-anchors-v0.1.jsonl.gz"
    if not anchor_path.is_file():
        raise FileNotFoundError(f"missing verified resume anchors: {anchor_path}")
    if anchor_path.stat().st_size != anchor_pin["bytes"]:
        raise ValueError("EMR canary resume anchor byte size changed")
    if _sha256_file(anchor_path) != anchor_pin["sha256"]:
        raise ValueError("EMR canary resume anchor SHA-256 changed")

    source_manifest = _read_json(source_paths["source_file_manifest"])
    mixture = _read_json(source_paths["data_mix"])
    schedule, _ = build_round_robin_source_schedule(source_manifest, mixture)
    provenance = _component_provenance(_read_json(source_paths["component_inventory"]))
    baseline_partial = baseline["partial_prefix"]
    selected = _select_canary_sources(schedule, contract["canary"], baseline_partial)
    sources = []
    for source in selected:
        start_line = 1
        if source["ordinal"] == baseline_partial["source_ordinal"]:
            start_line = int(baseline_partial["source_next_line"]) + 1
        sources.append(
            {
                **source,
                "maximum_records": int(contract["canary"]["maximum_records_per_source"]),
                "source_provenance": provenance[source["component"]],
                "start_line": start_line,
            }
        )
    if len(sources) != 8 or sources[0]["start_line"] != 38_849:
        raise ValueError("EMR canary source slice changed")

    output_directory.mkdir(parents=True, exist_ok=True)
    code_path = output_directory / CODE_ARCHIVE
    implementation = _zip_code(root, code_path)
    job_source = root / "scripts" / JOB_SCRIPT
    job_copy = output_directory / JOB_SCRIPT
    _write_bytes_atomic(job_copy, job_source.read_bytes())
    implementation[str(job_source.relative_to(root))] = _sha256_file(job_source)

    container_files = [
        (paths["dockerfile"], "Dockerfile"),
        (paths["requirements_lock"], "requirements-linux-x86_64.lock"),
        (paths["verifier"], "verify_runtime.py"),
    ]
    context_path = output_directory / IMAGE_CONTEXT
    _zip_named_files(context_path, container_files)
    packaged_container = []
    for source, name in container_files:
        destination = output_directory / name
        _write_bytes_atomic(destination, source.read_bytes())
        packaged_container.append(destination)
        implementation[str(source.relative_to(root))] = _sha256_file(source)

    quality = _read_json(source_paths["data_quality"])
    benchmark_path = root / quality["benchmark_decontamination"]["registry"]
    if _sha256_file(benchmark_path) != quality["benchmark_decontamination"][
        "registry_sha256"
    ]:
        raise ValueError("EMR canary benchmark registry changed")
    packaged_benchmark = output_directory / benchmark_path.name
    _write_bytes_atomic(packaged_benchmark, benchmark_path.read_bytes())
    packaged_anchors = output_directory / anchor_path.name
    _write_bytes_atomic(packaged_anchors, anchor_path.read_bytes())

    bucket = baseline["bucket"]
    namespace = baseline["namespace"]
    bootstrap_prefix = f"{namespace}/control/distributed/emr-serverless/bootstrap"
    external = source_contract["external_artifacts"]
    artifacts = {
        "benchmark_registry": _artifact(
            bucket=bucket,
            key=(
                f"{bootstrap_prefix}/benchmark/{_sha256_file(benchmark_path)}.jsonl.gz"
            ),
            sha256=_sha256_file(benchmark_path),
            size=benchmark_path.stat().st_size,
            state="upload_required_after_new_owner_approval",
            package_file=packaged_benchmark.name,
        ),
        "code_archive": _artifact(
            bucket=bucket,
            key=f"{bootstrap_prefix}/code/{_sha256_file(code_path)}.zip",
            sha256=_sha256_file(code_path),
            size=code_path.stat().st_size,
            state="upload_required_after_new_owner_approval",
            package_file=code_path.name,
        ),
        "frozen_tokenizer": _artifact(
            bucket=bucket,
            key=external["frozen_tokenizer"]["s3_key"],
            sha256=external["frozen_tokenizer"]["sha256"],
            size=external["frozen_tokenizer"]["bytes"],
            state="existing_verified_production_object",
        ),
        "image_context": _artifact(
            bucket=bucket,
            key=f"{bootstrap_prefix}/image/{_sha256_file(context_path)}.zip",
            sha256=_sha256_file(context_path),
            size=context_path.stat().st_size,
            state="codebuild_not_authorized",
            package_file=context_path.name,
        ),
        "job_script": _artifact(
            bucket=bucket,
            key=f"{bootstrap_prefix}/jobs/{_sha256_file(job_source)}.py",
            sha256=_sha256_file(job_source),
            size=job_source.stat().st_size,
            state="upload_required_after_new_owner_approval",
            package_file=job_copy.name,
        ),
        "resume_anchors": _artifact(
            bucket=bucket,
            key=f"{bootstrap_prefix}/anchors/{anchor_pin['sha256']}.jsonl.gz",
            sha256=anchor_pin["sha256"],
            size=anchor_pin["bytes"],
            state="upload_required_after_new_owner_approval",
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
        "application": deepcopy(contract["application"]),
        "artifacts": artifacts,
        "authorization": deepcopy(contract["authorization"]),
        "canary": deepcopy(contract["canary"]),
        "cost": _cost_envelope(contract, paths),
        "data_quality": quality,
        "deduplication": {
            "maximum_candidate_pairs": source_plan["backend"]["maximum_candidate_pairs"],
            "maximum_lsh_bucket_documents": source_plan["backend"][
                "maximum_lsh_bucket_documents"
            ],
            "maximum_rounds": source_plan["backend"]["maximum_near_dedup_rounds"],
            "strategy": source_plan["backend"]["deduplication"],
        },
        "filters": _read_json(source_paths["filter_policy"])["filters"],
        "gate": "lumi_distributed_emr_serverless_canary_release_v0.1",
        "image": {
            "base_digest": contract["image"]["base_digest"],
            "base_uri": contract["image"]["base_uri"],
            "context_sha256": _sha256_file(context_path),
            "custom_image_bytes": None,
            "custom_image_digest": None,
            "custom_image_uri": None,
            "maximum_image_gib": contract["image"]["maximum_image_gib"],
            "source_release_fingerprint": None,
            "state": "build_not_authorized",
        },
        "image_build": deepcopy(contract["image_build"]),
        "implementation": implementation,
        "job": {
            **deepcopy(contract["job"]),
            "python_executable": contract["image"]["python_executable"],
        },
        "output": {
            "checkpoint_prefix": (
                f"{namespace}/control/distributed/emr-serverless/canary/checkpoints"
            ),
            "result_prefix": f"{namespace}/control/distributed/emr-serverless/canary/results",
            "write_production_shards": False,
        },
        "predecessor": {
            "charged_compute_usd": "0.0374",
            "failure_stage": "additional_python_modules_bootstrap",
            "required_lingua_version": "2.2.0",
            "runtime_python": "3.11",
            "service": "aws_glue",
            "state": "failed_safe_and_cleaned_up",
        },
        "region": contract["region"],
        "resume_baseline": deepcopy(source_plan["resume_baseline"]),
        "runtime": deepcopy(contract["runtime"]),
        "schema_version": EMR_RELEASE_SCHEMA_VERSION,
        "source_commit": commit,
        "source_plan_fingerprint": source_plan["plan_fingerprint"],
        "sources": sources,
    }
    fingerprint = _release_fingerprint(release_material)
    release = {
        **release_material,
        "decision": {
            "paid_codebuild_image": "not_authorized",
            "paid_emr_serverless_canary": "not_authorized",
            "paid_emr_serverless_production": "not_authorized",
            "status": "verified_local_package_waiting_for_new_owner_approval",
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
        context_path,
        packaged_anchors,
        job_copy,
        *packaged_container,
    ]
    manifest = "".join(
        f"{_sha256_file(path)}  {path.name}\n" for path in manifest_paths
    )
    _write_bytes_atomic(output_directory / PACKAGE_MANIFEST, manifest.encode())
    return release


def validate_emr_serverless_canary_release(
    release: dict[str, Any],
    *,
    require_bound_image: bool = False,
) -> dict[str, Any]:
    """Validate an image-unbound package release or a digest-bound launch release."""

    if release.get("schema_version") != EMR_RELEASE_SCHEMA_VERSION:
        raise ValueError("unsupported EMR canary release schema")
    if release.get("gate") != "lumi_distributed_emr_serverless_canary_release_v0.1":
        raise ValueError("unexpected EMR canary release gate")
    material = {
        key: value
        for key, value in release.items()
        if key not in {"decision", "release_fingerprint"}
    }
    fingerprint = _release_fingerprint(material)
    if fingerprint != release.get("release_fingerprint"):
        raise ValueError("EMR canary release fingerprint changed")
    if release.get("authorization") != {
        "full_4b_training": False,
        "model_training": False,
        "paid_codebuild_image": False,
        "paid_emr_serverless_canary": False,
        "paid_emr_serverless_production": False,
        "paid_gpu_compute": False,
    }:
        raise ValueError("EMR canary release must not authorize paid work or training")
    if release.get("output", {}).get("write_production_shards") is not False:
        raise ValueError("EMR canary release may not write production shards")
    if release.get("runtime") != {
        "release_label": "emr-7.13.0",
        "service": "aws_emr_serverless",
        "spark_version": "3.5.6",
    }:
        raise ValueError("EMR canary release runtime changed")
    if release.get("application") != {
        "architecture": "X86_64",
        "auto_start": True,
        "auto_stop_idle_timeout_minutes": 5,
        "initial_capacity_workers": 0,
        "maximum_disk_gib": 80,
        "maximum_memory_gib": 64,
        "maximum_vcpu": 16,
    }:
        raise ValueError("EMR canary release capacity changed")
    if release.get("job") != {
        "driver": {"cores": 4, "disk_gib": 20, "memory_gib": 14},
        "dynamic_allocation": False,
        "executor": {"cores": 4, "disk_gib": 20, "instances": 3, "memory_gib": 14},
        "python_executable": "/opt/lumi/python312/bin/python",
        "spark_shuffle_partitions": 12,
    }:
        raise ValueError("EMR canary release job shape changed")
    if release.get("canary") != {
        "maximum_records_per_source": 2_000,
        "maximum_runtime_minutes": 30,
        "maximum_source_compressed_bytes": 700_000_000,
        "maximum_total_source_compressed_bytes": 1_500_000_000,
        "requires_new_owner_approval": True,
        "source_files": 8,
    }:
        raise ValueError("EMR canary release envelope changed")
    if release.get("cost") != {
        "codebuild_compute_hard_maximum_usd": "0.15",
        "combined_compute_hard_maximum_usd": "0.755904",
        "emr_compute_hard_maximum_usd": "0.605904",
        "memory_gb_hour_usd": "0.0057785",
        "standard_storage_extra_gb_hour_usd": "0.000111",
        "vcpu_hour_usd": "0.052624",
    }:
        raise ValueError("EMR canary release cost changed")
    sources = release.get("sources", [])
    if len(sources) != 8:
        raise ValueError("EMR canary source count changed")
    if any(int(source["maximum_records"]) != 2_000 for source in sources):
        raise ValueError("EMR canary record cap changed")
    if any(int(source["source_bytes"]) > 700_000_000 for source in sources):
        raise ValueError("EMR canary source byte cap exceeded")
    if sum(int(source["source_bytes"]) for source in sources) > 1_500_000_000:
        raise ValueError("EMR canary total source byte cap exceeded")
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
        raise ValueError("EMR canary category coverage changed")
    image = release.get("image", {})
    if (
        image.get("base_digest") != BASE_IMAGE_DIGEST
        or image.get("base_uri") != BASE_IMAGE_URI
        or image.get("maximum_image_gib") != 10
    ):
        raise ValueError("EMR canary image boundary changed")
    state = image.get("state")
    if state == "build_not_authorized":
        if any(
            image.get(name) is not None
            for name in (
                "custom_image_bytes",
                "custom_image_digest",
                "custom_image_uri",
                "source_release_fingerprint",
            )
        ):
            raise ValueError("unbound EMR canary release contains a custom image")
        if require_bound_image:
            raise ValueError("EMR canary release is not bound to a custom image digest")
    elif state == "digest_bound_not_authorized":
        match = PRIVATE_IMAGE_PATTERN.fullmatch(str(image.get("custom_image_uri", "")))
        if match is None or match.group("digest") != image.get("custom_image_digest"):
            raise ValueError("EMR canary custom image URI is not digest-bound")
        parent = image.get("source_release_fingerprint")
        if not isinstance(parent, str) or not re.fullmatch(r"[0-9a-f]{64}", parent):
            raise ValueError("EMR canary custom image lacks source-release lineage")
        size = image.get("custom_image_bytes")
        if not isinstance(size, int) or size < 1 or size > 10 * 1024**3:
            raise ValueError("EMR canary custom image size exceeds its hard cap")
    else:
        raise ValueError("EMR canary image state changed")
    expected_status = (
        "verified_local_package_waiting_for_new_owner_approval"
        if state == "build_not_authorized"
        else "image_bound_requires_matching_unexpired_owner_authorization"
    )
    if release.get("decision") != {
        "paid_codebuild_image": "not_authorized",
        "paid_emr_serverless_canary": "not_authorized",
        "paid_emr_serverless_production": "not_authorized",
        "status": expected_status,
    }:
        raise ValueError("EMR canary release decision boundary changed")
    return {
        "image_state": state,
        "release_fingerprint": fingerprint,
        "status": "verified_not_authorized",
    }


def bind_emr_serverless_image(
    release: dict[str, Any],
    *,
    image_uri: str,
    image_bytes: int,
) -> dict[str, Any]:
    """Bind a verified private-ECR digest without authorizing a job run."""

    validate_emr_serverless_canary_release(release)
    if release["image"]["state"] != "build_not_authorized":
        raise ValueError("EMR canary release is already image-bound")
    match = PRIVATE_IMAGE_PATTERN.fullmatch(image_uri)
    if match is None:
        raise ValueError("EMR canary image must be a digest-bound private us-east-1 ECR URI")
    if image_bytes < 1 or image_bytes > 10 * 1024**3:
        raise ValueError("EMR canary image size exceeds its hard cap")
    bound = deepcopy(release)
    source_release_fingerprint = release["release_fingerprint"]
    bound["image"].update(
        {
            "custom_image_bytes": image_bytes,
            "custom_image_digest": match.group("digest"),
            "custom_image_uri": image_uri,
            "source_release_fingerprint": source_release_fingerprint,
            "state": "digest_bound_not_authorized",
        }
    )
    material = {
        key: value
        for key, value in bound.items()
        if key not in {"decision", "release_fingerprint"}
    }
    bound["release_fingerprint"] = _release_fingerprint(material)
    bound["decision"]["status"] = (
        "image_bound_requires_matching_unexpired_owner_authorization"
    )
    validate_emr_serverless_canary_release(bound, require_bound_image=True)
    return bound


def build_emr_serverless_application_request(release: dict[str, Any]) -> dict[str, Any]:
    """Render the exact bounded application request without calling AWS."""

    validate_emr_serverless_canary_release(release, require_bound_image=True)
    fingerprint = release["release_fingerprint"]
    return {
        "architecture": "X86_64",
        "autoStartConfiguration": {"enabled": True},
        "autoStopConfiguration": {"enabled": True, "idleTimeoutMinutes": 5},
        "clientToken": f"lumi-app-{fingerprint[:32]}",
        "imageConfiguration": {"imageUri": release["image"]["custom_image_uri"]},
        "maximumCapacity": {"cpu": "16 vCPU", "disk": "80 GB", "memory": "64 GB"},
        "name": f"lumi-corpus-canary-{fingerprint[:12]}",
        "releaseLabel": "emr-7.13.0",
        "tags": {
            "LumiReleaseFingerprint": fingerprint,
            "LumiScope": "bounded-corpus-canary-only",
        },
        "type": "SPARK",
    }


def _bounded_s3_file(uri: str, *, bucket: str | None = None) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError("EMR canary launch artifact must be a private s3:// object URI")
    if parsed.params or parsed.query or parsed.fragment:
        raise ValueError("EMR canary launch artifact URI contains unsupported components")
    if bucket is not None and parsed.netloc != bucket:
        raise ValueError("EMR canary launch artifacts must use the sealed corpus bucket")
    return parsed.netloc, Path(parsed.path).name


def build_emr_serverless_job_request(
    release: dict[str, Any],
    authorization: dict[str, Any],
    *,
    application_id: str,
    execution_role_arn: str,
    release_s3_uri: str,
    authorization_s3_uri: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Render one fail-closed StartJobRun request without calling AWS."""

    validate_emr_serverless_canary_release(release, require_bound_image=True)
    validate_emr_serverless_canary_authorization(authorization, release, now=now)
    if re.fullmatch(r"[0-9a-z]+", application_id) is None:
        raise ValueError("EMR canary application ID is invalid")
    role_pattern = r"arn:aws:iam::[0-9]{12}:role/[A-Za-z0-9+=,.@_/-]+"
    if re.fullmatch(role_pattern, execution_role_arn) is None:
        raise ValueError("EMR canary execution role ARN is invalid")
    anchor_uri = release["artifacts"]["resume_anchors"]["s3_uri"]
    bucket, _ = _bounded_s3_file(anchor_uri)
    files = [
        release_s3_uri,
        authorization_s3_uri,
        release["artifacts"]["benchmark_registry"]["s3_uri"],
        release["artifacts"]["frozen_tokenizer"]["s3_uri"],
        anchor_uri,
        release["artifacts"]["training_exclusion_ledger"]["s3_uri"],
    ]
    file_names = []
    for uri in files:
        _, name = _bounded_s3_file(uri, bucket=bucket)
        file_names.append(name)
    if len(file_names) != len(set(file_names)):
        raise ValueError("EMR canary SparkFiles basenames collide")
    code_uri = release["artifacts"]["code_archive"]["s3_uri"]
    job_uri = release["artifacts"]["job_script"]["s3_uri"]
    _bounded_s3_file(code_uri, bucket=bucket)
    _bounded_s3_file(job_uri, bucket=bucket)
    python = release["job"]["python_executable"]
    spark_parameters = [
        "--conf", "spark.driver.cores=4",
        "--conf", "spark.driver.memory=14g",
        "--conf", "spark.executor.cores=4",
        "--conf", "spark.executor.memory=14g",
        "--conf", "spark.executor.instances=3",
        "--conf", "spark.dynamicAllocation.enabled=false",
        "--conf", "spark.emr-serverless.driver.disk=20G",
        "--conf", "spark.emr-serverless.executor.disk=20G",
        "--conf", "spark.sql.shuffle.partitions=12",
        "--conf", "spark.default.parallelism=12",
        "--conf", "spark.network.crypto.enabled=true",
        "--conf", "spark.io.encryption.enabled=true",
        "--conf", f"spark.emr-serverless.driverEnv.PYSPARK_DRIVER_PYTHON={python}",
        "--conf", f"spark.emr-serverless.driverEnv.PYSPARK_PYTHON={python}",
        "--conf", f"spark.executorEnv.PYSPARK_PYTHON={python}",
        "--py-files", code_uri,
        "--files", ",".join(files),
    ]
    run_token = authorization["run_token"]
    log_uri = (
        f"s3://{bucket}/{release['output']['result_prefix']}/"
        f"{release['release_fingerprint']}/{run_token}/logs"
    )
    return {
        "applicationId": application_id,
        "clientToken": f"lumi-job-{run_token}",
        "configurationOverrides": {
            "monitoringConfiguration": {
                "s3MonitoringConfiguration": {"logUri": log_uri}
            }
        },
        "executionRoleArn": execution_role_arn,
        "executionTimeoutMinutes": 30,
        "jobDriver": {
            "sparkSubmit": {
                "entryPoint": job_uri,
                "entryPointArguments": [
                    "--lumi-authorization-file",
                    Path(urlparse(authorization_s3_uri).path).name,
                    "--lumi-release-file",
                    Path(urlparse(release_s3_uri).path).name,
                    "--lumi-run-token",
                    run_token,
                ],
                "sparkSubmitParameters": " ".join(spark_parameters),
            }
        },
        "mode": "BATCH",
        "name": f"lumi-corpus-canary-{run_token[:12]}",
        "retryPolicy": {"maxAttempts": 1},
        "tags": {
            "LumiReleaseFingerprint": release["release_fingerprint"],
            "LumiScope": "bounded-corpus-canary-only",
        },
    }


def verify_emr_serverless_canary_package(output_directory: Path) -> dict[str, Any]:
    """Verify package hashes, container contract, and non-authorization."""

    expected = {
        CODE_ARCHIVE,
        IMAGE_CONTEXT,
        RELEASE_FILE,
        "Dockerfile",
        "benchmark-signatures-v0.1.jsonl.gz",
        "requirements-linux-x86_64.lock",
        "resume-anchors-v0.1.jsonl.gz",
        "verify_runtime.py",
        JOB_SCRIPT,
    }
    seen = set()
    for line in (output_directory / PACKAGE_MANIFEST).read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if name not in expected or name in seen:
            raise ValueError("EMR canary package manifest changed")
        if _sha256_file(output_directory / name) != digest:
            raise ValueError(f"EMR canary package hash changed: {name}")
        seen.add(name)
    if seen != expected:
        raise ValueError("EMR canary package manifest is incomplete")
    release = _read_json(output_directory / RELEASE_FILE)
    validated = validate_emr_serverless_canary_release(release)
    if validated["image_state"] != "build_not_authorized":
        raise ValueError("packaged EMR release must remain image-unbound")
    for artifact in release["artifacts"].values():
        package_file = artifact.get("package_file")
        if package_file is None:
            continue
        path = output_directory / package_file
        if path.stat().st_size != artifact["bytes"] or _sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"EMR canary packaged artifact changed: {package_file}")
    with zipfile.ZipFile(output_directory / CODE_ARCHIVE) as archive:
        names = set(archive.namelist())
        required = {
            "lumi/distributed_corpus.py",
            "lumi/emr_serverless_canary.py",
            "lumi/glue_corpus.py",
        }
        if not required.issubset(names):
            raise ValueError("EMR canary code archive is incomplete")
        if any(name.startswith("/") or ".." in Path(name).parts for name in names):
            raise ValueError("EMR canary code archive contains an unsafe path")
    with zipfile.ZipFile(output_directory / IMAGE_CONTEXT) as archive:
        if set(archive.namelist()) != {
            "Dockerfile",
            "requirements-linux-x86_64.lock",
            "verify_runtime.py",
        }:
            raise ValueError("EMR canary image context changed")
    dockerfile = (output_directory / "Dockerfile").read_text(encoding="utf-8")
    expected_from = (
        "FROM " + BASE_IMAGE_URI
    )
    if not dockerfile.startswith(expected_from):
        raise ValueError("EMR canary Dockerfile base image changed")
    if dockerfile.rstrip().splitlines()[-1] != "USER hadoop:hadoop":
        raise ValueError("EMR canary Dockerfile final user changed")
    protected = ("JAVA_HOME", "SPARK_HOME", "HIVE_HOME", "TEZ_HOME")
    if any(f"ENV {name}" in dockerfile for name in protected):
        raise ValueError("EMR canary Dockerfile modifies a protected EMR environment variable")
    return {
        "code_archive_sha256": _sha256_file(output_directory / CODE_ARCHIVE),
        "image_context_sha256": _sha256_file(output_directory / IMAGE_CONTEXT),
        "release_fingerprint": validated["release_fingerprint"],
        "release_sha256": _sha256_file(output_directory / RELEASE_FILE),
        "status": "verified_not_authorized",
    }


def validate_emr_serverless_canary_authorization(
    authorization: dict[str, Any],
    release: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Fail unless a fresh owner approval matches one bounded build and canary."""

    release_validation = validate_emr_serverless_canary_release(release)
    if authorization.get("schema_version") != EMR_AUTHORIZATION_SCHEMA_VERSION:
        raise ValueError("unsupported EMR canary authorization schema")
    if authorization.get("gate") != "lumi_emr_serverless_canary_authorization_v0.1":
        raise ValueError("unexpected EMR canary authorization gate")
    if authorization.get("approval_text") != EMR_CANARY_APPROVAL_TEXT:
        raise ValueError("EMR canary authorization text changed")
    if authorization.get("owner") != "Afnan Technologies Inc.":
        raise ValueError("EMR canary authorization owner changed")
    expected_release = release["release_fingerprint"]
    if release_validation["image_state"] == "digest_bound_not_authorized":
        expected_release = release["image"]["source_release_fingerprint"]
    if authorization.get("release_fingerprint") != expected_release:
        raise ValueError("EMR canary authorization targets another release")
    if authorization.get("source_commit") != release.get("source_commit"):
        raise ValueError("EMR canary authorization targets another source commit")
    if authorization.get("image_context_sha256") != release["image"]["context_sha256"]:
        raise ValueError("EMR canary authorization targets another image context")
    if authorization.get("authorization") != {
        "full_4b_training": False,
        "model_training": False,
        "paid_codebuild_image": True,
        "paid_emr_serverless_canary": True,
        "paid_emr_serverless_production": False,
        "paid_gpu_compute": False,
    }:
        raise ValueError("EMR canary authorization boundary changed")
    expected_bounds = {
        "maximum_codebuild_compute_usd": "0.15",
        "maximum_codebuild_minutes": 30,
        "maximum_combined_compute_usd": "0.755904",
        "maximum_emr_compute_usd": "0.605904",
        "maximum_emr_disk_gib": 80,
        "maximum_emr_memory_gib": 64,
        "maximum_emr_runtime_minutes": 30,
        "maximum_emr_vcpu": 16,
        "maximum_records_per_source": 2_000,
        "maximum_source_compressed_bytes": 700_000_000,
        "maximum_source_files": 8,
        "maximum_total_source_compressed_bytes": 1_500_000_000,
    }
    if authorization.get("bounds") != expected_bounds:
        raise ValueError("EMR canary authorization bounds changed")
    run_token = authorization.get("run_token")
    if not isinstance(run_token, str) or re.fullmatch(r"[0-9a-f]{32}", run_token) is None:
        raise ValueError("EMR canary authorization run token is invalid")
    try:
        approved = datetime.fromisoformat(
            str(authorization["approved_at"]).replace("Z", "+00:00")
        )
        expires = datetime.fromisoformat(str(authorization["expires_at"]).replace("Z", "+00:00"))
    except (KeyError, ValueError) as error:
        raise ValueError("EMR canary authorization timestamps are invalid") from error
    if approved.tzinfo is None or expires.tzinfo is None:
        raise ValueError("EMR canary authorization timestamps must include time zones")
    approved_utc = approved.astimezone(UTC)
    expires_utc = expires.astimezone(UTC)
    current = (now or datetime.now(UTC)).astimezone(UTC)
    if approved_utc > current:
        raise ValueError("EMR canary authorization approval time is in the future")
    if expires_utc <= approved_utc or expires_utc > approved_utc + MAX_AUTHORIZATION_LIFETIME:
        raise ValueError("EMR canary authorization lifetime exceeds two hours")
    if expires_utc <= current:
        raise ValueError("EMR canary authorization has expired")
    return {
        "approved_at": approved_utc.isoformat().replace("+00:00", "Z"),
        "expires_at": expires_utc.isoformat().replace("+00:00", "Z"),
        "release_fingerprint": expected_release,
        "status": (
            "authorized_one_bounded_image_build_and_canary_only"
            if release_validation["image_state"] == "build_not_authorized"
            else "authorized_bounded_canary_from_approved_image_context"
        ),
    }
