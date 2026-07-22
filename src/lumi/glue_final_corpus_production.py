"""Fail-closed AWS Glue package for Lumi's production final-80B corpus."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any

from .final_corpus import (
    build_final_corpus_plan,
    final_corpus_plan_fingerprint,
    validate_final_selection_result,
    validate_final_shard_manifest,
)
from .glue_canary import _artifact, _valid_commit, _write_bytes_atomic, _zip_code
from .glue_final_corpus import _baseline_release
from .production_corpus import _canonical_json_bytes, _read_json, _sha256_file

SCHEMA_VERSION = 1
PACKAGE_MANIFEST = "manifest.sha256"
RELEASE_FILE = "glue-final-corpus-production-release-v0.1.json"
AUTHORIZATION_FILE = "glue-final-corpus-production-authorization-v0.1.json"
PLAN_FILE = "final-corpus-80b-plan-v0.1.json"
CANARY_REPORT_FILE = "verified-final-corpus-canary-result-v0.1.json"
CODE_ARCHIVE = "lumi-glue-final-corpus-production-code-v0.1.zip"
JOB_SCRIPT = "run_glue_final_corpus_production.py"
RELEASE_GATE = "lumi_glue_final_corpus_production_release_v0.1"
AUTHORIZATION_GATE = "lumi_glue_final_corpus_production_authorization_v0.1"
CATEGORY_CHECKPOINT_GATE = (
    "lumi_glue_final_corpus_production_category_checkpoint_v0.1"
)
RESULT_GATE = "lumi_glue_final_corpus_production_result_v0.1"


def _fingerprint(value: Mapping[str, Any], *excluded: str) -> str:
    skipped = set(excluded)
    material = {key: row for key, row in value.items() if key not in skipped}
    return hashlib.sha256(_canonical_json_bytes(material)).hexdigest()


def _valid_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{label} is outside its allowed range")
    return value


def _bounded_uri(value: Any, *, bucket: str, prefix: str, label: str) -> str:
    expected = f"s3://{bucket}/{prefix}/"
    if not isinstance(value, str) or not value.startswith(expected):
        raise ValueError(f"{label} left its bounded S3 prefix")
    if "/../" in f"/{value}/" or value.endswith("/"):
        raise ValueError(f"{label} is not a canonical S3 URI")
    return value


def _exact_prefix_uri(value: Any, *, bucket: str, prefix: str, label: str) -> str:
    expected = f"s3://{bucket}/{prefix}"
    if value != expected or "/../" in f"/{value}/" or str(value).endswith("/"):
        raise ValueError(f"{label} changed its exact S3 prefix")
    return str(value)


def _manifest(directory: Path, paths: Sequence[Path]) -> None:
    rows = []
    for path in sorted(paths, key=lambda item: str(item.relative_to(directory))):
        rows.append(f"{_sha256_file(path)}  {path.relative_to(directory)}\n")
    _write_bytes_atomic(directory / PACKAGE_MANIFEST, "".join(rows).encode())


def _read_contract(path: Path, root: Path) -> dict[str, Any]:
    contract = _read_json(path)
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Glue final-corpus production plan schema")
    if contract.get("gate") != "lumi_glue_final_corpus_production_plan_v0.1":
        raise ValueError("unexpected Glue final-corpus production plan gate")
    if contract.get("authorization") != {
        "full_4b_training": False,
        "model_training": False,
        "paid_glue_production": False,
        "paid_gpu_compute": False,
    }:
        raise ValueError("Glue final-corpus production plan authorizes paid work")
    if contract.get("job") != {
        "hard_maximum_dpu": 100,
        "maximum_compute_usd": "528.000000",
        "maximum_runtime_minutes": 720,
        "worker_type": "G.1X",
    }:
        raise ValueError("Glue final-corpus production job envelope changed")
    if contract.get("limits") != {
        "expected_new_packed_shards": 160,
        "expected_total_packed_shards": 163,
        "maximum_category_checkpoints": 7,
        "maximum_input_bytes": 507_737_434_761,
        "maximum_input_objects": 2_772,
        "maximum_new_output_bytes": 750_000_000_000,
        "packed_shard_target_tokens": 500_000_000,
        "selection_partitions": 1_600,
    }:
        raise ValueError("Glue final-corpus production limits changed")
    if contract.get("runtime") != {
        "additional_python_modules": "zstandard==0.25.0",
        "glue_version": "5.1",
        "python_version": "3.11",
        "spark_version": "3.5.6",
    }:
        raise ValueError("Glue final-corpus production runtime changed")
    if contract.get("cost_model") != {
        "dpu_second_safety_multiplier": "3.000000"
    }:
        raise ValueError("Glue final-corpus production cost model changed")
    if contract.get("region") != "us-east-1":
        raise ValueError("Glue final-corpus production region changed")
    inputs = contract.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "final_corpus_contract",
        "glue_pricing",
        "verified_canary_report",
    }:
        raise ValueError("Glue final-corpus production input pins changed")
    for name, pin in inputs.items():
        if not isinstance(pin, Mapping):
            raise ValueError(f"Glue final-corpus production input is invalid: {name}")
        relative = pin.get("path")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError(f"Glue final-corpus production path is invalid: {name}")
        resolved = (root / relative).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Glue final-corpus production input escaped: {name}") from error
        if not resolved.is_file() or _sha256_file(resolved) != pin.get("sha256"):
            raise ValueError(f"Glue final-corpus production input changed: {relative}")
    pricing = _read_json(root / str(inputs["glue_pricing"]["path"]))
    if pricing.get("dpu_hour_usd") != "0.44" or pricing.get("region") != "us-east-1":
        raise ValueError("Glue final-corpus production pricing changed")
    maximum = (
        Decimal(contract["job"]["hard_maximum_dpu"])
        * Decimal(contract["job"]["maximum_runtime_minutes"])
        / Decimal(60)
        * Decimal(str(pricing["dpu_hour_usd"]))
    ).quantize(Decimal("0.000001"))
    if maximum != Decimal(contract["job"]["maximum_compute_usd"]):
        raise ValueError("Glue final-corpus production cost envelope changed")
    return contract


def _validate_canary_report(report: Mapping[str, Any]) -> dict[str, Any]:
    if report.get("schema_version") != SCHEMA_VERSION or report.get("gate") != (
        "lumi_glue_final_corpus_canary_verified_result_v0.1"
    ):
        raise ValueError("unexpected final-corpus canary report schema")
    if report.get("status") != "independently_verified_pass":
        raise ValueError("final-corpus canary has not passed independent verification")
    _valid_commit(report.get("source_commit"))
    for name in (
        "authorization_sha256",
        "dedup_result_sha256",
        "plan_fingerprint",
        "release_fingerprint",
        "result_sha256",
    ):
        _valid_sha256(report.get(name), f"canary report {name}")
    if report.get("authorization_scope") != {
        "model_training": False,
        "paid_glue_production": False,
        "paid_gpu_compute": False,
    }:
        raise ValueError("final-corpus canary report broadens authorization")
    if report.get("cleanup") != {
        "job": "deleted",
        "role": "deleted",
        "status": "consumed_job_and_role_removed",
    }:
        raise ValueError("final-corpus canary cleanup is incomplete")
    calibration = report.get("calibration")
    if calibration != {
        "input_documents": 173_380,
        "input_tokens": 301_453_386,
        "selected_tokens": 7_026_092,
    }:
        raise ValueError("final-corpus canary calibration changed")
    job = report.get("job")
    if not isinstance(job, Mapping) or (
        job.get("attempt"),
        job.get("dpu_seconds"),
        job.get("execution_time_seconds"),
        job.get("glue_version"),
        job.get("number_of_workers"),
        job.get("state"),
        job.get("timeout_minutes"),
        job.get("worker_type"),
    ) != (0, 1887.0, 235, "5.1", 8, "SUCCEEDED", 30, "G.1X"):
        raise ValueError("final-corpus canary job evidence changed")
    input_inventory = report.get("input_inventory")
    if input_inventory != {
        "documents": 68_671_669,
        "object_bytes": 507_737_434_761,
        "object_count": 2_772,
        "objects_sha256": (
            "edfc524e9d83628c3a135eb2490345d3a7176e26b8be31e449cc9079cc6348d8"
        ),
        "tokens": 119_762_267_075,
    }:
        raise ValueError("final-corpus canary input inventory changed")
    output = report.get("output")
    if output != {
        "output_bytes": 45_093_486,
        "packed_bytes": 14_052_184,
        "packed_objects": 28,
        "selected_bytes": 31_041_302,
        "selected_documents": 3_803,
        "selected_objects": 7,
        "selected_tokens": 7_026_092,
    }:
        raise ValueError("final-corpus canary output evidence changed")
    categories = report.get("categories")
    if not isinstance(categories, list) or len(categories) != 7:
        raise ValueError("final-corpus canary category evidence changed")
    if any(row.get("partition_parity") != "pass" for row in categories):
        raise ValueError("final-corpus canary partition parity failed")
    if sum(int(row["selected_tokens"]) for row in categories) != 7_026_092:
        raise ValueError("final-corpus canary category accounting changed")
    return dict(report)


def _ceil_decimal(value: Decimal) -> int:
    return int(value.to_integral_value(rounding=ROUND_CEILING))


def _derive_cost_model(
    *,
    report: Mapping[str, Any],
    input_tokens: int,
    maximum_candidate_selection_tokens: int,
    maximum_final_tokens: int,
    dpu: int,
    price: Decimal,
    safety_multiplier: Decimal,
) -> dict[str, Any]:
    canary_dpu_seconds = Decimal(str(report["job"]["dpu_seconds"]))
    canary_input_tokens = Decimal(int(report["calibration"]["input_tokens"]))
    ideal_dpu_seconds_decimal = (
        canary_dpu_seconds * Decimal(input_tokens) / canary_input_tokens
    )
    conservative_dpu_seconds_decimal = ideal_dpu_seconds_decimal * safety_multiplier
    ideal_dpu_seconds = _ceil_decimal(ideal_dpu_seconds_decimal)
    conservative_dpu_seconds = _ceil_decimal(conservative_dpu_seconds_decimal)
    canary_selected_tokens = Decimal(int(report["output"]["selected_tokens"]))
    selected_parquet_bytes = _ceil_decimal(
        Decimal(int(report["output"]["selected_bytes"]))
        * Decimal(maximum_candidate_selection_tokens)
        / canary_selected_tokens
    )
    packed_bytes = maximum_final_tokens * 2
    return {
        "calibration_canary_dpu_seconds": int(canary_dpu_seconds),
        "calibration_canary_input_tokens": int(canary_input_tokens),
        "conservative_dpu_seconds": conservative_dpu_seconds,
        "conservative_runtime_minutes_at_100_dpu": math.ceil(
            conservative_dpu_seconds / dpu / 60
        ),
        "conservative_usd": str(
            (Decimal(conservative_dpu_seconds) / Decimal(3600) * price).quantize(
                Decimal("0.000001")
            )
        ),
        "dpu_second_safety_multiplier": str(safety_multiplier.quantize(Decimal("0.000001"))),
        "estimated_maximum_output_bytes": selected_parquet_bytes + packed_bytes,
        "estimated_selected_parquet_bytes": selected_parquet_bytes,
        "ideal_linear_dpu_seconds": ideal_dpu_seconds,
        "ideal_linear_runtime_minutes_at_100_dpu": math.ceil(
            ideal_dpu_seconds / dpu / 60
        ),
        "ideal_linear_usd": str(
            (Decimal(ideal_dpu_seconds) / Decimal(3600) * price).quantize(
                Decimal("0.000001")
            )
        ),
        "packed_uint16_bytes_at_maximum_selection": packed_bytes,
        "projection": "linear_input_token_calibration_with_3x_safety_not_a_guarantee",
    }


def build_glue_final_corpus_production_package(
    *,
    contract_path: Path,
    final_plan_path: Path,
    repository_root: Path,
    source_commit: str,
    output_directory: Path,
) -> dict[str, Any]:
    """Build a deterministic production package without authorizing AWS."""

    root = repository_root.resolve()
    commit = _valid_commit(source_commit)
    contract = _read_contract(contract_path, root)
    canary_source = root / str(
        contract["inputs"]["verified_canary_report"]["path"]
    )
    canary_report = _validate_canary_report(_read_json(canary_source))
    supplied_plan = _read_json(final_plan_path)
    expected_plan = build_final_corpus_plan(
        root / str(contract["inputs"]["final_corpus_contract"]["path"]),
        repository_root=root,
        source_commit=str(canary_report["source_commit"]),
    )
    if supplied_plan != expected_plan:
        raise ValueError("Glue production received a different canary-proven final plan")
    if supplied_plan.get("plan_fingerprint") != final_corpus_plan_fingerprint(
        supplied_plan
    ):
        raise ValueError("Glue production final plan fingerprint changed")
    if supplied_plan["plan_fingerprint"] != canary_report["plan_fingerprint"]:
        raise ValueError("Glue production plan was not proven by the canary")
    if supplied_plan["inputs"]["dedup_result"]["canonical_sha256"] != (
        canary_report["dedup_result_sha256"]
    ):
        raise ValueError("Glue production canary targets another dedup result")

    output_directory.mkdir(parents=True, exist_ok=True)
    if any(output_directory.iterdir()):
        raise FileExistsError("Glue final-corpus production package is not empty")
    code_path = output_directory / CODE_ARCHIVE
    implementation = _zip_code(root, code_path)
    source_script = root / "scripts" / JOB_SCRIPT
    packaged_script = output_directory / JOB_SCRIPT
    _write_bytes_atomic(packaged_script, source_script.read_bytes())
    implementation[str(source_script.relative_to(root))] = _sha256_file(source_script)
    packaged_plan = output_directory / PLAN_FILE
    _write_bytes_atomic(
        packaged_plan,
        (json.dumps(supplied_plan, indent=2, sort_keys=True) + "\n").encode(),
    )
    packaged_canary = output_directory / CANARY_REPORT_FILE
    _write_bytes_atomic(packaged_canary, canary_source.read_bytes())

    bucket = str(supplied_plan["output"]["bucket"])
    namespace = str(supplied_plan["output"]["namespace"])
    bootstrap = f"{namespace}/control/glue-production/bootstrap"
    artifacts = {}
    for name, path, suffix in (
        ("code_archive", code_path, "zip"),
        ("final_plan", packaged_plan, "json"),
        ("job_script", packaged_script, "py"),
        ("verified_canary_report", packaged_canary, "json"),
    ):
        digest = _sha256_file(path)
        artifacts[name] = _artifact(
            bucket=bucket,
            key=f"{bootstrap}/{name}/{digest}.{suffix}",
            sha256=digest,
            size=path.stat().st_size,
            state="upload_required_before_authorized_launch",
            package_file=path.name,
        )

    sequence = int(supplied_plan["output"]["preserved_baseline_shards"])
    categories = []
    for ordinal, row in enumerate(supplied_plan["categories"]):
        count = int(row["new_packed_shards"])
        categories.append(
            {
                **dict(row),
                "first_new_shard_sequence": sequence,
                "last_new_shard_sequence": sequence + count - 1,
                "ordinal": ordinal,
            }
        )
        sequence += count
    if sequence != int(supplied_plan["output"]["expected_total_packed_shards"]):
        raise ValueError("Glue production shard sequence planning changed")

    accepted = dict(supplied_plan["inputs"]["dedup_result"]["accepted_candidates"])
    maximum_candidate_selection = sum(
        int(row["maximum_selected_candidate_tokens"]) for row in categories
    )
    pricing = _read_json(
        root / str(contract["inputs"]["glue_pricing"]["path"])
    )
    cost_model = _derive_cost_model(
        report=canary_report,
        input_tokens=int(accepted["tokens"]),
        maximum_candidate_selection_tokens=maximum_candidate_selection,
        maximum_final_tokens=int(
            supplied_plan["limits"]["maximum_final_selected_tokens"]
        ),
        dpu=int(contract["job"]["hard_maximum_dpu"]),
        price=Decimal(str(pricing["dpu_hour_usd"])),
        safety_multiplier=Decimal(
            contract["cost_model"]["dpu_second_safety_multiplier"]
        ),
    )
    if cost_model["estimated_maximum_output_bytes"] > int(
        contract["limits"]["maximum_new_output_bytes"]
    ):
        raise ValueError("Glue production projected output exceeds its hard cap")

    release: dict[str, Any] = {
        "artifacts": artifacts,
        "authorization": dict(contract["authorization"]),
        "baseline": _baseline_release(supplied_plan, root),
        "canary": {
            "authorization_sha256": canary_report["authorization_sha256"],
            "release_fingerprint": canary_report["release_fingerprint"],
            "report_file_sha256": _sha256_file(canary_source),
            "result_sha256": canary_report["result_sha256"],
            "source_commit": canary_report["source_commit"],
            "status": canary_report["status"],
        },
        "categories": categories,
        "cost_model": cost_model,
        "decision": {
            "full_4b_training": "not_authorized",
            "paid_glue_production": "not_authorized",
            "status": "production_package_ready_waiting_for_exact_authorization",
        },
        "gate": RELEASE_GATE,
        "implementation": implementation,
        "input": {
            "accepted_candidates": accepted,
            "dedup_result_sha256": supplied_plan["inputs"]["dedup_result"][
                "canonical_sha256"
            ],
            "plan_fingerprint": supplied_plan["plan_fingerprint"],
            "selection_seed_sha256": supplied_plan["selection"]["seed_sha256"],
        },
        "job": {
            **dict(contract["job"]),
            "execution_class": "STANDARD",
            "glue_version": contract["runtime"]["glue_version"],
            "maximum_concurrent_runs": 1,
            "number_of_workers": int(contract["job"]["hard_maximum_dpu"]),
        },
        "limits": {
            **dict(contract["limits"]),
            "maximum_document_boundary_overshoot_tokens": supplied_plan["limits"][
                "maximum_document_boundary_overshoot_tokens_per_category"
            ],
            "maximum_final_selected_tokens": supplied_plan["limits"][
                "maximum_final_selected_tokens"
            ],
            "maximum_selected_candidate_tokens": maximum_candidate_selection,
            "training_token_limit": supplied_plan["limits"]["training_token_limit"],
        },
        "output": {
            "result_prefix": f"{namespace}/production/results",
            "run_prefix": f"{namespace}/production/runs",
        },
        "region": contract["region"],
        "runtime": dict(contract["runtime"]),
        "schema_version": SCHEMA_VERSION,
        "source_commit": commit,
        "storage": {"bucket": bucket, "namespace": namespace},
    }
    release["release_fingerprint"] = _fingerprint(release, "release_fingerprint")
    validate_glue_final_corpus_production_release(release)
    release_path = output_directory / RELEASE_FILE
    _write_bytes_atomic(
        release_path,
        (json.dumps(release, indent=2, sort_keys=True) + "\n").encode(),
    )
    _manifest(
        output_directory,
        [code_path, packaged_canary, packaged_plan, packaged_script, release_path],
    )
    return release


def validate_glue_final_corpus_production_release(
    release: Mapping[str, Any],
) -> dict[str, Any]:
    if release.get("schema_version") != SCHEMA_VERSION or release.get("gate") != (
        RELEASE_GATE
    ):
        raise ValueError("unexpected Glue final-corpus production release schema")
    if release.get("release_fingerprint") != _fingerprint(
        release, "release_fingerprint"
    ):
        raise ValueError("Glue final-corpus production release fingerprint changed")
    _valid_commit(release.get("source_commit"))
    if release.get("authorization") != {
        "full_4b_training": False,
        "model_training": False,
        "paid_glue_production": False,
        "paid_gpu_compute": False,
    }:
        raise ValueError("Glue final-corpus production release authorizes paid work")
    if release.get("decision") != {
        "full_4b_training": "not_authorized",
        "paid_glue_production": "not_authorized",
        "status": "production_package_ready_waiting_for_exact_authorization",
    }:
        raise ValueError("Glue final-corpus production decision changed")
    job = release.get("job")
    if not isinstance(job, Mapping) or (
        job.get("hard_maximum_dpu"),
        job.get("maximum_compute_usd"),
        job.get("maximum_runtime_minutes"),
        job.get("worker_type"),
        job.get("number_of_workers"),
        job.get("maximum_concurrent_runs"),
    ) != (100, "528.000000", 720, "G.1X", 100, 1):
        raise ValueError("Glue final-corpus production job envelope changed")
    if release.get("runtime") != {
        "additional_python_modules": "zstandard==0.25.0",
        "glue_version": "5.1",
        "python_version": "3.11",
        "spark_version": "3.5.6",
    }:
        raise ValueError("Glue final-corpus production runtime changed")
    storage = release.get("storage")
    if not isinstance(storage, Mapping) or storage.get("namespace") != (
        "lumi-corpus/v0.1/final-80b"
    ):
        raise ValueError("Glue final-corpus production storage changed")
    bucket = str(storage.get("bucket"))
    if not bucket:
        raise ValueError("Glue final-corpus production bucket is missing")
    input_value = release.get("input")
    if not isinstance(input_value, Mapping):
        raise ValueError("Glue final-corpus production input is missing")
    for name in ("dedup_result_sha256", "plan_fingerprint", "selection_seed_sha256"):
        _valid_sha256(input_value.get(name), f"production input {name}")
    accepted = input_value.get("accepted_candidates")
    if not isinstance(accepted, Mapping) or accepted != {
        "documents": 68_671_669,
        "object_bytes": 507_737_434_761,
        "object_count": 2_772,
        "objects_sha256": (
            "edfc524e9d83628c3a135eb2490345d3a7176e26b8be31e449cc9079cc6348d8"
        ),
        "s3_uri": (
            "s3://lumi-corpus-prod-27777c02b29b76efcac3/lumi-corpus/v0.1/"
            "deduplicated/glue-production/9cf19bdd8c90d157755c706113f99959/"
            "accepted-candidates"
        ),
        "tokens": 119_762_267_075,
    }:
        raise ValueError("Glue final-corpus production accepted inventory changed")
    canary = release.get("canary")
    if not isinstance(canary, Mapping) or canary.get("status") != (
        "independently_verified_pass"
    ):
        raise ValueError("Glue final-corpus production canary binding changed")
    for name in (
        "authorization_sha256",
        "release_fingerprint",
        "report_file_sha256",
        "result_sha256",
    ):
        _valid_sha256(canary.get(name), f"production canary {name}")
    _valid_commit(canary.get("source_commit"))
    artifacts = release.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "code_archive",
        "final_plan",
        "job_script",
        "verified_canary_report",
    }:
        raise ValueError("Glue final-corpus production artifacts changed")
    for name, artifact in artifacts.items():
        if not isinstance(artifact, Mapping):
            raise ValueError(f"Glue production artifact is invalid: {name}")
        _positive_int(artifact.get("bytes"), f"production artifact bytes: {name}")
        _valid_sha256(artifact.get("sha256"), f"production artifact SHA-256: {name}")
        _bounded_uri(
            artifact.get("s3_uri"),
            bucket=bucket,
            prefix=str(storage["namespace"]),
            label=f"production artifact: {name}",
        )
    categories = release.get("categories")
    if not isinstance(categories, list) or len(categories) != 7:
        raise ValueError("Glue final-corpus production categories changed")
    if [int(row.get("ordinal", -1)) for row in categories] != list(range(7)):
        raise ValueError("Glue final-corpus production category order changed")
    expected_sequence = 3
    candidate_maximum = 0
    for row in categories:
        count = int(row.get("new_packed_shards", -1))
        if count <= 0 or int(row.get("first_new_shard_sequence", -1)) != expected_sequence:
            raise ValueError("Glue final-corpus production shard range changed")
        if int(row.get("last_new_shard_sequence", -1)) != expected_sequence + count - 1:
            raise ValueError("Glue final-corpus production shard end changed")
        expected_sequence += count
        candidate_maximum += int(row["maximum_selected_candidate_tokens"])
    if expected_sequence != 163 or sum(
        int(row["target_final_tokens"]) for row in categories
    ) != 80_000_000_000:
        raise ValueError("Glue final-corpus production shard/category totals changed")
    limits = release.get("limits")
    if not isinstance(limits, Mapping) or limits != {
        "expected_new_packed_shards": 160,
        "expected_total_packed_shards": 163,
        "maximum_category_checkpoints": 7,
        "maximum_document_boundary_overshoot_tokens": 131_072,
        "maximum_final_selected_tokens": 80_000_917_504,
        "maximum_input_bytes": 507_737_434_761,
        "maximum_input_objects": 2_772,
        "maximum_new_output_bytes": 750_000_000_000,
        "maximum_selected_candidate_tokens": candidate_maximum,
        "packed_shard_target_tokens": 500_000_000,
        "selection_partitions": 1_600,
        "training_token_limit": 80_000_000_000,
    }:
        raise ValueError("Glue final-corpus production release limits changed")
    baseline = release.get("baseline")
    if not isinstance(baseline, Mapping) or len(baseline.get("sealed_shards", [])) != 2:
        raise ValueError("Glue final-corpus production baseline changed")
    if int(baseline.get("partial", {}).get("tokens", -1)) != 37_154_932:
        raise ValueError("Glue final-corpus production partial baseline changed")
    output = release.get("output")
    if output != {
        "result_prefix": "lumi-corpus/v0.1/final-80b/production/results",
        "run_prefix": "lumi-corpus/v0.1/final-80b/production/runs",
    }:
        raise ValueError("Glue final-corpus production output boundary changed")
    cost = _derive_cost_model(
        report={
            "calibration": {
                "input_tokens": 301_453_386,
            },
            "job": {"dpu_seconds": 1887.0},
            "output": {
                "selected_bytes": 31_041_302,
                "selected_tokens": 7_026_092,
            },
        },
        input_tokens=int(accepted["tokens"]),
        maximum_candidate_selection_tokens=candidate_maximum,
        maximum_final_tokens=int(limits["maximum_final_selected_tokens"]),
        dpu=100,
        price=Decimal("0.44"),
        safety_multiplier=Decimal("3.000000"),
    )
    if release.get("cost_model") != cost:
        raise ValueError("Glue final-corpus production cost projection changed")
    if int(cost["estimated_maximum_output_bytes"]) > int(
        limits["maximum_new_output_bytes"]
    ):
        raise ValueError("Glue final-corpus production output projection exceeds cap")
    return {
        "conservative_runtime_minutes": cost[
            "conservative_runtime_minutes_at_100_dpu"
        ],
        "conservative_usd": cost["conservative_usd"],
        "maximum_usd": job["maximum_compute_usd"],
        "release_fingerprint": release["release_fingerprint"],
        "status": "verified_not_authorized",
    }


def expected_glue_final_corpus_production_approval(
    release: Mapping[str, Any],
) -> str:
    validate_glue_final_corpus_production_release(release)
    return (
        "I approve exactly one Lumi 100-DPU AWS Glue production final-80B "
        "selection-and-sharding run against dedup result "
        f"{release['input']['dedup_result_sha256']} and plan "
        f"{release['input']['plan_fingerprint']}, maximum 12 hours/$528. "
        "No GPU or model training."
    )


def build_glue_final_corpus_production_authorization(
    release: Mapping[str, Any],
    *,
    approval_text: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    validate_glue_final_corpus_production_release(release)
    if approval_text != expected_glue_final_corpus_production_approval(release):
        raise ValueError("Glue final-corpus production owner approval does not match")
    issued = (now or datetime.now(UTC)).astimezone(UTC)
    authorization: dict[str, Any] = {
        "approval_text": approval_text,
        "bounds": {
            "authorized_runs": 1,
            "canary_result_sha256": release["canary"]["result_sha256"],
            "dedup_result_sha256": release["input"]["dedup_result_sha256"],
            "hard_maximum_dpu": 100,
            "maximum_runtime_minutes": 720,
            "maximum_usd": "528.000000",
            "plan_fingerprint": release["input"]["plan_fingerprint"],
        },
        "expires_at": (issued + timedelta(hours=24)).isoformat(),
        "gate": AUTHORIZATION_GATE,
        "issued_at": issued.isoformat(),
        "permissions": {
            "full_4b_training": False,
            "model_training": False,
            "paid_glue_selection": True,
            "paid_glue_sharding": True,
            "paid_gpu_compute": False,
        },
        "release_fingerprint": release["release_fingerprint"],
        "schema_version": SCHEMA_VERSION,
        "source_commit": release["source_commit"],
    }
    authorization["authorization_sha256"] = _fingerprint(
        authorization, "authorization_sha256"
    )
    validate_glue_final_corpus_production_authorization(
        authorization, release, now=issued
    )
    return authorization


def validate_glue_final_corpus_production_authorization(
    authorization: Mapping[str, Any],
    release: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    validate_glue_final_corpus_production_release(release)
    if authorization.get("schema_version") != SCHEMA_VERSION or authorization.get(
        "gate"
    ) != AUTHORIZATION_GATE:
        raise ValueError("unexpected Glue final-corpus production authorization")
    if authorization.get("authorization_sha256") != _fingerprint(
        authorization, "authorization_sha256"
    ):
        raise ValueError("Glue final-corpus production authorization changed")
    if authorization.get("approval_text") != (
        expected_glue_final_corpus_production_approval(release)
    ):
        raise ValueError("Glue final-corpus production approval text changed")
    if authorization.get("release_fingerprint") != release["release_fingerprint"]:
        raise ValueError("Glue final-corpus production authorization targets another release")
    if authorization.get("source_commit") != release["source_commit"]:
        raise ValueError("Glue final-corpus production authorization source changed")
    if authorization.get("permissions") != {
        "full_4b_training": False,
        "model_training": False,
        "paid_glue_selection": True,
        "paid_glue_sharding": True,
        "paid_gpu_compute": False,
    }:
        raise ValueError("Glue final-corpus production permissions changed")
    if authorization.get("bounds") != {
        "authorized_runs": 1,
        "canary_result_sha256": release["canary"]["result_sha256"],
        "dedup_result_sha256": release["input"]["dedup_result_sha256"],
        "hard_maximum_dpu": 100,
        "maximum_runtime_minutes": 720,
        "maximum_usd": "528.000000",
        "plan_fingerprint": release["input"]["plan_fingerprint"],
    }:
        raise ValueError("Glue final-corpus production authorization bounds changed")
    current = (now or datetime.now(UTC)).astimezone(UTC)
    issued = datetime.fromisoformat(str(authorization["issued_at"])).astimezone(UTC)
    expires = datetime.fromisoformat(str(authorization["expires_at"])).astimezone(UTC)
    if expires <= issued or expires - issued > timedelta(hours=24):
        raise ValueError("Glue final-corpus production authorization lifetime changed")
    if current < issued - timedelta(minutes=5) or current >= expires:
        raise ValueError("Glue final-corpus production authorization is not current")
    return {
        "authorization_sha256": authorization["authorization_sha256"],
        "maximum_usd": "528.000000",
        "status": "authorized_one_production_final_corpus_run",
    }


def write_glue_final_corpus_production_authorization(
    release_path: Path,
    output_path: Path,
    *,
    approval_text: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    release = _read_json(release_path)
    authorization = build_glue_final_corpus_production_authorization(
        release, approval_text=approval_text, now=now
    )
    _write_bytes_atomic(
        output_path,
        (json.dumps(authorization, indent=2, sort_keys=True) + "\n").encode(),
    )
    return authorization


def glue_final_corpus_production_job_token(
    authorization: Mapping[str, Any],
) -> str:
    digest = _valid_sha256(
        authorization.get("authorization_sha256"),
        "Glue final-corpus production authorization",
    )
    return hashlib.sha256(f"glue-final-corpus-production:{digest}".encode()).hexdigest()[
        :32
    ]


def category_checkpoint_fingerprint(checkpoint: Mapping[str, Any]) -> str:
    return _fingerprint(checkpoint, "checkpoint_sha256")


def validate_glue_final_corpus_category_checkpoint(
    checkpoint: Mapping[str, Any],
    release: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    issued = datetime.fromisoformat(str(authorization["issued_at"]))
    validate_glue_final_corpus_production_authorization(
        authorization, release, now=issued
    )
    if checkpoint.get("schema_version") != SCHEMA_VERSION or checkpoint.get(
        "gate"
    ) != CATEGORY_CHECKPOINT_GATE:
        raise ValueError("unexpected Glue final-corpus category checkpoint")
    if checkpoint.get("status") != "complete":
        raise ValueError("Glue final-corpus category checkpoint is incomplete")
    if checkpoint.get("checkpoint_sha256") != category_checkpoint_fingerprint(
        checkpoint
    ):
        raise ValueError("Glue final-corpus category checkpoint fingerprint changed")
    expected = {
        str(row["category"]): row for row in release["categories"]
    }
    category = str(checkpoint.get("category"))
    if category not in expected:
        raise ValueError("Glue final-corpus category checkpoint category changed")
    planned = expected[category]
    for name, value in {
        "authorization_sha256": authorization["authorization_sha256"],
        "job_token": glue_final_corpus_production_job_token(authorization),
        "plan_fingerprint": release["input"]["plan_fingerprint"],
        "release_fingerprint": release["release_fingerprint"],
    }.items():
        if checkpoint.get(name) != value:
            raise ValueError(f"Glue final-corpus category checkpoint changed {name}")
    selected_documents = _positive_int(
        checkpoint.get("selected_candidate_documents"), "checkpoint documents"
    )
    selected_tokens = _positive_int(
        checkpoint.get("selected_candidate_tokens"), "checkpoint tokens"
    )
    if not int(planned["requested_candidate_tokens"]) <= selected_tokens <= int(
        planned["maximum_selected_candidate_tokens"]
    ):
        raise ValueError("Glue final-corpus category checkpoint left target envelope")
    if selected_documents > int(planned["available_candidate_documents"]):
        raise ValueError("Glue final-corpus category checkpoint selected too many rows")
    if int(checkpoint.get("available_candidate_documents", -1)) != int(
        planned["available_candidate_documents"]
    ) or int(checkpoint.get("available_candidate_tokens", -1)) != int(
        planned["available_candidate_tokens"]
    ):
        raise ValueError("Glue final-corpus category checkpoint input changed")
    cutoff = checkpoint.get("cutoff")
    if not isinstance(cutoff, Mapping):
        raise ValueError("Glue final-corpus category checkpoint cutoff is missing")
    _valid_sha256(cutoff.get("_lumi_selection_rank"), "checkpoint cutoff rank")
    _valid_sha256(checkpoint.get("selection_sha256"), "checkpoint selection SHA-256")
    token = glue_final_corpus_production_job_token(authorization)
    prefix = f"{release['output']['run_prefix']}/{token}"
    inventory = checkpoint.get("selected_inventory")
    if not isinstance(inventory, Mapping):
        raise ValueError("Glue final-corpus category selected inventory is missing")
    _exact_prefix_uri(
        inventory.get("s3_uri"),
        bucket=str(release["storage"]["bucket"]),
        prefix=f"{prefix}/selected/{category}",
        label="production selected inventory",
    )
    _positive_int(inventory.get("object_count"), "selected object count")
    selected_bytes = _positive_int(inventory.get("object_bytes"), "selected bytes")
    _valid_sha256(inventory.get("objects_sha256"), "selected inventory SHA-256")
    shards = checkpoint.get("packed_shards")
    if not isinstance(shards, list) or len(shards) != int(
        planned["new_packed_shards"]
    ):
        raise ValueError("Glue final-corpus category shard count changed")
    shard_tokens = 0
    shard_documents = 0
    packed_bytes = 0
    first = int(planned["first_new_shard_sequence"])
    for offset, shard in enumerate(shards):
        if not isinstance(shard, Mapping) or int(shard.get("sequence", -1)) != (
            first + offset
        ):
            raise ValueError("Glue final-corpus category shard sequence changed")
        tokens = _positive_int(shard.get("tokens"), "category shard tokens")
        documents = _positive_int(shard.get("documents"), "category shard documents")
        object_bytes = _positive_int(shard.get("object_bytes"), "category shard bytes")
        if object_bytes != tokens * 2 or tokens > 500_131_072:
            raise ValueError("Glue final-corpus category shard bounds changed")
        digest = _valid_sha256(shard.get("object_sha256"), "category shard SHA-256")
        _bounded_uri(
            shard.get("s3_uri"),
            bucket=str(release["storage"]["bucket"]),
            prefix=f"{prefix}/packed/{category}",
            label="production packed shard",
        )
        if not str(shard["s3_uri"]).endswith(f"/{digest}.uint16.bin"):
            raise ValueError("Glue final-corpus category shard is not content-addressed")
        shard_tokens += tokens
        shard_documents += documents
        packed_bytes += object_bytes
    if shard_tokens != selected_tokens or shard_documents != selected_documents:
        raise ValueError("Glue final-corpus category shard conservation failed")
    if int(checkpoint.get("output_bytes", -1)) != selected_bytes + packed_bytes:
        raise ValueError("Glue final-corpus category output accounting changed")
    return dict(checkpoint)


def production_result_fingerprint(result: Mapping[str, Any]) -> str:
    return _fingerprint(result, "result_sha256")


def validate_glue_final_corpus_production_result(
    result: Mapping[str, Any],
    release: Mapping[str, Any],
    authorization: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    issued = datetime.fromisoformat(str(authorization["issued_at"]))
    validate_glue_final_corpus_production_authorization(
        authorization, release, now=issued
    )
    if result.get("schema_version") != SCHEMA_VERSION or result.get("gate") != (
        RESULT_GATE
    ):
        raise ValueError("unexpected Glue final-corpus production result")
    if result.get("status") != "sealed":
        raise ValueError("Glue final-corpus production result is incomplete")
    if result.get("result_sha256") != production_result_fingerprint(result):
        raise ValueError("Glue final-corpus production result fingerprint changed")
    for name, value in {
        "authorization_sha256": authorization["authorization_sha256"],
        "dedup_result_sha256": release["input"]["dedup_result_sha256"],
        "job_token": glue_final_corpus_production_job_token(authorization),
        "plan_fingerprint": release["input"]["plan_fingerprint"],
        "release_fingerprint": release["release_fingerprint"],
    }.items():
        if result.get(name) != value:
            raise ValueError(f"Glue final-corpus production result changed {name}")
    if plan.get("plan_fingerprint") != release["input"]["plan_fingerprint"]:
        raise ValueError("Glue final-corpus production result plan changed")
    checkpoints = result.get("category_checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 7:
        raise ValueError("Glue final-corpus production category checkpoints changed")
    observed = set()
    checkpoint_output_bytes = 0
    for checkpoint in checkpoints:
        validated = validate_glue_final_corpus_category_checkpoint(
            checkpoint, release, authorization
        )
        category = str(validated["category"])
        if category in observed:
            raise ValueError("Glue final-corpus production repeats a checkpoint")
        observed.add(category)
        checkpoint_output_bytes += int(validated["output_bytes"])
    if observed != {str(row["category"]) for row in release["categories"]}:
        raise ValueError("Glue final-corpus production checkpoint set changed")
    selection = result.get("selection_result")
    shards = result.get("shard_manifest")
    if not isinstance(selection, Mapping) or not isinstance(shards, Mapping):
        raise ValueError("Glue final-corpus production sealed outputs are missing")
    validate_final_selection_result(selection, plan)
    validate_final_shard_manifest(shards, plan, selection)
    baseline = [*release["baseline"]["sealed_shards"], release["baseline"]["partial"]]
    for sequence, expected in enumerate(baseline):
        observed_shard = shards["shards"][sequence]
        expected_sha = expected["sha256"]
        if (
            observed_shard.get("category") != "science_and_education"
            or int(observed_shard.get("documents", -1)) != int(expected["documents"])
            or int(observed_shard.get("tokens", -1)) != int(expected["tokens"])
            or observed_shard.get("object_sha256") != expected_sha
        ):
            raise ValueError("Glue final-corpus production baseline materialization changed")
    baseline_bytes = sum(int(row["bytes"]) for row in baseline)
    if int(result.get("data_output_bytes", -1)) != (
        checkpoint_output_bytes + baseline_bytes
    ):
        raise ValueError("Glue final-corpus production output-byte accounting changed")
    if int(result["data_output_bytes"]) > int(
        release["limits"]["maximum_new_output_bytes"]
    ):
        raise ValueError("Glue final-corpus production exceeded its output cap")
    runtime = result.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("Glue final-corpus production runtime is missing")
    if str(runtime.get("python", "")).split(".")[:2] != ["3", "11"]:
        raise ValueError("Glue final-corpus production Python runtime changed")
    if not str(runtime.get("spark", "")).startswith("3.5.6"):
        raise ValueError("Glue final-corpus production Spark runtime changed")
    return dict(result)


def build_glue_final_corpus_production_job_definition(
    release: Mapping[str, Any],
    authorization: Mapping[str, Any],
    *,
    role_arn: str,
    release_s3_uri: str,
    authorization_s3_uri: str,
) -> dict[str, Any]:
    """Render the single production Glue job without creating or starting it."""

    validate_glue_final_corpus_production_authorization(authorization, release)
    if re.fullmatch(
        r"arn:aws:iam::[0-9]{12}:role/[A-Za-z0-9+=,.@_/-]+", role_arn
    ) is None:
        raise ValueError("Glue final-corpus production role ARN is invalid")
    bucket = str(release["storage"]["bucket"])
    namespace = str(release["storage"]["namespace"])
    files = [
        release_s3_uri,
        release["artifacts"]["final_plan"]["s3_uri"],
        authorization_s3_uri,
        release["artifacts"]["verified_canary_report"]["s3_uri"],
    ]
    for uri in files:
        _bounded_uri(
            uri,
            bucket=bucket,
            prefix=namespace,
            label="Glue final-corpus production bootstrap file",
        )
    basenames = [Path(str(uri)).name for uri in files]
    if len(basenames) != len(set(basenames)):
        raise ValueError("Glue final-corpus production bootstrap basenames collide")
    for name in ("code_archive", "job_script"):
        _bounded_uri(
            release["artifacts"][name]["s3_uri"],
            bucket=bucket,
            prefix=namespace,
            label=f"Glue final-corpus production {name}",
        )
    token = glue_final_corpus_production_job_token(authorization)
    return {
        "Command": {
            "Name": "glueetl",
            "PythonVersion": "3",
            "ScriptLocation": release["artifacts"]["job_script"]["s3_uri"],
        },
        "DefaultArguments": {
            "--enable-continuous-cloudwatch-log": "true",
            "--enable-job-insights": "true",
            "--enable-metrics": "true",
            "--enable-observability-metrics": "true",
        },
        "NonOverridableArguments": {
            "--LUMI_AUTHORIZATION_FILE": basenames[2],
            "--LUMI_CANARY_REPORT_FILE": basenames[3],
            "--LUMI_FINAL_PLAN_FILE": basenames[1],
            "--LUMI_RELEASE_FILE": basenames[0],
            "--TempDir": (
                f"s3://{bucket}/{namespace}/control/glue-production/temp/{token}/"
            ),
            "--additional-python-modules": release["runtime"][
                "additional_python_modules"
            ],
            "--enable-auto-scaling": "false",
            "--extra-files": ",".join(files),
            "--extra-py-files": release["artifacts"]["code_archive"]["s3_uri"],
            "--job-language": "python",
        },
        "Description": (
            "Lumi production final-80B selection and sharding only; no GPU or "
            "model training"
        ),
        "ExecutionClass": "STANDARD",
        "ExecutionProperty": {"MaxConcurrentRuns": 1},
        "GlueVersion": release["runtime"]["glue_version"],
        "MaxRetries": 0,
        "Name": f"lumi-glue-final-corpus-production-{token[:12]}",
        "NumberOfWorkers": 100,
        "Role": role_arn,
        "Tags": {
            "LumiPlanFingerprint": str(release["input"]["plan_fingerprint"]),
            "LumiScope": "production-final-80b-selection-sharding-only",
        },
        "Timeout": 720,
        "WorkerType": "G.1X",
    }
