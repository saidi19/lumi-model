"""Non-launching Glue canary package for Lumi's final 80B corpus gate."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from .final_corpus import build_final_corpus_plan, final_corpus_plan_fingerprint
from .glue_canary import _artifact, _valid_commit, _write_bytes_atomic, _zip_code
from .glue_runtime import FINAL_CORPUS_SPARK_CONFIG, glue_conf_argument
from .production_corpus import _canonical_json_bytes, _read_json, _sha256_file

SCHEMA_VERSION = 1
PACKAGE_MANIFEST = "manifest.sha256"
RELEASE_FILE = "glue-final-corpus-canary-release-v0.1.json"
AUTHORIZATION_FILE = "glue-final-corpus-canary-authorization-v0.1.json"
PLAN_FILE = "final-corpus-80b-plan-v0.1.json"
CODE_ARCHIVE = "lumi-glue-final-corpus-canary-code-v0.1.zip"
JOB_SCRIPT = "run_glue_final_corpus_canary.py"
RELEASE_GATE = "lumi_glue_final_corpus_canary_release_v0.1"
AUTHORIZATION_GATE = "lumi_glue_final_corpus_canary_authorization_v0.1"
RESULT_GATE = "lumi_glue_final_corpus_canary_result_v0.1"


def _fingerprint(value: Mapping[str, Any], *excluded: str) -> str:
    skipped = set(excluded)
    material = {key: row for key, row in value.items() if key not in skipped}
    return hashlib.sha256(_canonical_json_bytes(material)).hexdigest()


def _valid_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _non_negative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _positive_int(value: Any, label: str) -> int:
    result = _non_negative_int(value, label)
    if result == 0:
        raise ValueError(f"{label} must be positive")
    return result


def _bounded_uri(value: Any, *, bucket: str, prefix: str, label: str) -> str:
    expected = f"s3://{bucket}/{prefix}/"
    if not isinstance(value, str) or not value.startswith(expected):
        raise ValueError(f"{label} left its bounded S3 prefix")
    if "/../" in f"/{value}/" or value.endswith("/"):
        raise ValueError(f"{label} is not a canonical S3 URI")
    return value


def _manifest(directory: Path, paths: Sequence[Path]) -> None:
    rows = []
    for path in sorted(paths, key=lambda item: str(item.relative_to(directory))):
        rows.append(f"{_sha256_file(path)}  {path.relative_to(directory)}\n")
    _write_bytes_atomic(directory / PACKAGE_MANIFEST, "".join(rows).encode())


def _read_contract(path: Path, root: Path) -> dict[str, Any]:
    contract = _read_json(path)
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Glue final-corpus canary plan schema")
    if contract.get("gate") != "lumi_glue_final_corpus_canary_plan_v0.1":
        raise ValueError("unexpected Glue final-corpus canary plan gate")
    if contract.get("authorization") != {
        "full_4b_training": False,
        "model_training": False,
        "paid_glue_canary": False,
        "paid_glue_production": False,
        "paid_gpu_compute": False,
    }:
        raise ValueError("Glue final-corpus canary plan authorizes paid work")
    if contract.get("job") != {
        "hard_maximum_dpu": 8,
        "maximum_compute_usd": "1.760000",
        "maximum_runtime_minutes": 30,
        "worker_type": "G.1X",
    }:
        raise ValueError("Glue final-corpus canary job envelope changed")
    if contract.get("limits") != {
        "maximum_candidate_documents_per_category": 100_000,
        "maximum_input_objects_per_category": 1,
        "maximum_new_output_bytes": 1_000_000_000,
        "packed_shard_target_tokens": 250_000,
        "selected_target_tokens_per_category": 1_000_000,
    }:
        raise ValueError("Glue final-corpus canary limits changed")
    if contract.get("runtime") != {
        "glue_version": "5.1",
        "python_version": "3.11",
        "spark_version": "3.5.6",
    }:
        raise ValueError("Glue final-corpus canary runtime changed")
    if contract.get("region") != "us-east-1":
        raise ValueError("Glue final-corpus canary region changed")
    inputs = contract.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "final_corpus_contract",
        "glue_pricing",
    }:
        raise ValueError("Glue final-corpus canary input pins changed")
    for name, pin in inputs.items():
        if not isinstance(pin, Mapping):
            raise ValueError(f"Glue final-corpus canary input is invalid: {name}")
        relative = pin.get("path")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError(f"Glue final-corpus canary input path is invalid: {name}")
        resolved = (root / relative).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Glue final-corpus canary input escaped: {name}") from error
        if not resolved.is_file() or _sha256_file(resolved) != pin.get("sha256"):
            raise ValueError(f"Glue final-corpus canary input changed: {relative}")
    pricing = _read_json(root / str(inputs["glue_pricing"]["path"]))
    maximum = (
        Decimal(contract["job"]["hard_maximum_dpu"])
        * Decimal(contract["job"]["maximum_runtime_minutes"])
        / Decimal(60)
        * Decimal(str(pricing["dpu_hour_usd"]))
    ).quantize(Decimal("0.000001"))
    if maximum != Decimal(contract["job"]["maximum_compute_usd"]):
        raise ValueError("Glue final-corpus canary cost envelope changed")
    return contract


def _baseline_release(plan: Mapping[str, Any], root: Path) -> dict[str, Any]:
    baseline = _read_json(root / str(plan["inputs"]["baseline"]["path"]))
    bucket = str(plan["output"]["bucket"])
    sealed = []
    for row in baseline["sealed_shards"]:
        sealed.append(
            {
                "category": str(row["category"]),
                "documents": int(row["documents"]),
                "s3_uri": f"s3://{bucket}/{row['token_object_key']}",
                "sha256": _valid_sha256(row["token_sha256"], "baseline token shard"),
                "tokens": int(row["tokens"]),
                "bytes": int(row["tokens"]) * 2,
            }
        )
    partial = baseline["partial_prefix"]
    archive = baseline["archive"]
    return {
        "archive": {
            "bytes": int(archive["bytes"]),
            "s3_uri": f"s3://{bucket}/{archive['key']}",
            "sha256": _valid_sha256(archive["sha256"], "baseline archive"),
        },
        "partial": {
            "documents": int(partial["accepted_documents"]),
            "member": str(partial["evidence_member_tokens"]),
            "sha256": _valid_sha256(partial["token_sha256"], "baseline partial"),
            "tokens": int(partial["accepted_tokens"]),
            "bytes": int(partial["token_bytes"]),
        },
        "sealed_shards": sealed,
    }


def build_glue_final_corpus_canary_package(
    *,
    contract_path: Path,
    final_plan_path: Path,
    repository_root: Path,
    source_commit: str,
    output_directory: Path,
) -> dict[str, Any]:
    """Build a deterministic canary package without authorizing or calling AWS."""

    root = repository_root.resolve()
    commit = _valid_commit(source_commit)
    contract = _read_contract(contract_path, root)
    supplied_plan = _read_json(final_plan_path)
    expected_plan = build_final_corpus_plan(
        root / str(contract["inputs"]["final_corpus_contract"]["path"]),
        repository_root=root,
        source_commit=commit,
    )
    if supplied_plan != expected_plan:
        raise ValueError("Glue final-corpus canary received a different final plan")
    if supplied_plan.get("plan_fingerprint") != final_corpus_plan_fingerprint(
        supplied_plan
    ):
        raise ValueError("Glue final-corpus plan fingerprint changed")
    if supplied_plan.get("status") != "ready_for_bounded_glue_canary_approval":
        raise ValueError("final-corpus plan is not ready for a bounded canary")
    if any(bool(value) for value in supplied_plan["authorization"].values()):
        raise ValueError("final-corpus plan unexpectedly authorizes paid work")

    output_directory.mkdir(parents=True, exist_ok=True)
    if any(output_directory.iterdir()):
        raise FileExistsError("Glue final-corpus package directory is not empty")
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

    bucket = str(supplied_plan["output"]["bucket"])
    namespace = str(supplied_plan["output"]["namespace"])
    bootstrap = f"{namespace}/control/glue-canary/bootstrap"
    code_sha = _sha256_file(code_path)
    script_sha = _sha256_file(packaged_script)
    plan_sha = _sha256_file(packaged_plan)
    artifacts = {
        "code_archive": _artifact(
            bucket=bucket,
            key=f"{bootstrap}/code/{code_sha}.zip",
            sha256=code_sha,
            size=code_path.stat().st_size,
            state="upload_required_before_authorized_launch",
            package_file=code_path.name,
        ),
        "final_plan": _artifact(
            bucket=bucket,
            key=f"{bootstrap}/plans/{plan_sha}.json",
            sha256=plan_sha,
            size=packaged_plan.stat().st_size,
            state="upload_required_before_authorized_launch",
            package_file=packaged_plan.name,
        ),
        "job_script": _artifact(
            bucket=bucket,
            key=f"{bootstrap}/jobs/{script_sha}.py",
            sha256=script_sha,
            size=packaged_script.stat().st_size,
            state="upload_required_before_authorized_launch",
            package_file=packaged_script.name,
        ),
    }
    release: dict[str, Any] = {
        "artifacts": artifacts,
        "authorization": dict(contract["authorization"]),
        "baseline": _baseline_release(supplied_plan, root),
        "categories": [
            {
                "category": str(row["category"]),
                "canary_target_tokens": int(
                    contract["limits"]["selected_target_tokens_per_category"]
                ),
                "final_target_tokens": int(row["target_final_tokens"]),
                "ordinal": index,
            }
            for index, row in enumerate(supplied_plan["categories"])
        ],
        "decision": {
            "full_4b_training": "not_authorized",
            "paid_glue_canary": "not_authorized",
            "paid_glue_production": "not_authorized",
            "status": "canary_package_ready_waiting_for_exact_authorization",
        },
        "gate": RELEASE_GATE,
        "implementation": implementation,
        "input": {
            "accepted_candidates": dict(
                supplied_plan["inputs"]["dedup_result"]["accepted_candidates"]
            ),
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
        },
        "output": {
            "canary_prefix": f"{namespace}/control/glue-canary/runs",
            "result_prefix": f"{namespace}/control/glue-canary/results",
        },
        "region": contract["region"],
        "runtime": dict(contract["runtime"]),
        "schema_version": SCHEMA_VERSION,
        "source_commit": commit,
        "storage": {"bucket": bucket, "namespace": namespace},
    }
    release["release_fingerprint"] = _fingerprint(release, "release_fingerprint")
    validate_glue_final_corpus_canary_release(release)
    release_path = output_directory / RELEASE_FILE
    _write_bytes_atomic(
        release_path,
        (json.dumps(release, indent=2, sort_keys=True) + "\n").encode(),
    )
    _manifest(
        output_directory,
        [code_path, packaged_plan, packaged_script, release_path],
    )
    return release


def validate_glue_final_corpus_canary_release(
    release: Mapping[str, Any],
) -> dict[str, Any]:
    if release.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Glue final-corpus canary release schema")
    if release.get("gate") != RELEASE_GATE:
        raise ValueError("unexpected Glue final-corpus canary release gate")
    if release.get("release_fingerprint") != _fingerprint(
        release, "release_fingerprint"
    ):
        raise ValueError("Glue final-corpus canary release fingerprint changed")
    _valid_commit(release.get("source_commit"))
    if release.get("authorization") != {
        "full_4b_training": False,
        "model_training": False,
        "paid_glue_canary": False,
        "paid_glue_production": False,
        "paid_gpu_compute": False,
    }:
        raise ValueError("Glue final-corpus canary release authorizes paid work")
    job = release.get("job")
    if not isinstance(job, Mapping) or (
        job.get("hard_maximum_dpu"),
        job.get("number_of_workers"),
        job.get("maximum_runtime_minutes"),
        job.get("maximum_compute_usd"),
        job.get("worker_type"),
        job.get("maximum_concurrent_runs"),
    ) != (8, 8, 30, "1.760000", "G.1X", 1):
        raise ValueError("Glue final-corpus canary job envelope changed")
    limits = release.get("limits")
    if limits != {
        "maximum_candidate_documents_per_category": 100_000,
        "maximum_document_boundary_overshoot_tokens": 131_072,
        "maximum_input_objects_per_category": 1,
        "maximum_new_output_bytes": 1_000_000_000,
        "packed_shard_target_tokens": 250_000,
        "selected_target_tokens_per_category": 1_000_000,
    }:
        raise ValueError("Glue final-corpus canary release limits changed")
    if release.get("runtime") != {
        "glue_version": "5.1",
        "python_version": "3.11",
        "spark_version": "3.5.6",
    }:
        raise ValueError("Glue final-corpus canary release runtime changed")
    storage = release.get("storage")
    if not isinstance(storage, Mapping) or storage.get("namespace") != (
        "lumi-corpus/v0.1/final-80b"
    ):
        raise ValueError("Glue final-corpus canary storage changed")
    bucket = str(storage.get("bucket"))
    if not bucket:
        raise ValueError("Glue final-corpus canary bucket is missing")
    input_value = release.get("input")
    if not isinstance(input_value, Mapping):
        raise ValueError("Glue final-corpus canary input is missing")
    for name in ("dedup_result_sha256", "plan_fingerprint", "selection_seed_sha256"):
        _valid_sha256(input_value.get(name), f"canary input {name}")
    accepted = input_value.get("accepted_candidates")
    if not isinstance(accepted, Mapping):
        raise ValueError("Glue final-corpus accepted input is missing")
    if accepted != {
        "documents": 68_671_669,
        "object_bytes": 507_737_434_761,
        "object_count": 2_772,
        "objects_sha256": "edfc524e9d83628c3a135eb2490345d3a7176e26b8be31e449cc9079cc6348d8",
        "s3_uri": (
            "s3://lumi-corpus-prod-27777c02b29b76efcac3/lumi-corpus/v0.1/"
            "deduplicated/glue-production/9cf19bdd8c90d157755c706113f99959/"
            "accepted-candidates"
        ),
        "tokens": 119_762_267_075,
    }:
        raise ValueError("Glue final-corpus accepted inventory changed")
    categories = release.get("categories")
    if not isinstance(categories, list) or len(categories) != 7:
        raise ValueError("Glue final-corpus canary categories changed")
    if [int(row.get("ordinal", -1)) for row in categories] != list(range(7)):
        raise ValueError("Glue final-corpus canary category order changed")
    if sum(int(row["final_target_tokens"]) for row in categories) != 80_000_000_000:
        raise ValueError("Glue final-corpus canary final targets changed")
    if any(int(row["canary_target_tokens"]) != 1_000_000 for row in categories):
        raise ValueError("Glue final-corpus canary category target changed")
    artifacts = release.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "code_archive",
        "final_plan",
        "job_script",
    }:
        raise ValueError("Glue final-corpus canary artifacts changed")
    for name, artifact in artifacts.items():
        if not isinstance(artifact, Mapping):
            raise ValueError(f"Glue final-corpus artifact is invalid: {name}")
        _positive_int(artifact.get("bytes"), f"canary artifact bytes: {name}")
        _valid_sha256(artifact.get("sha256"), f"canary artifact SHA-256: {name}")
        _bounded_uri(
            artifact.get("s3_uri"),
            bucket=bucket,
            prefix=str(storage["namespace"]),
            label=f"canary artifact: {name}",
        )
    baseline = release.get("baseline")
    if not isinstance(baseline, Mapping) or len(baseline.get("sealed_shards", [])) != 2:
        raise ValueError("Glue final-corpus baseline binding changed")
    archive = baseline.get("archive")
    partial = baseline.get("partial")
    if not isinstance(archive, Mapping) or not isinstance(partial, Mapping):
        raise ValueError("Glue final-corpus baseline partial binding is missing")
    _bounded_uri(
        archive.get("s3_uri"),
        bucket=bucket,
        prefix="lumi-corpus/v0.1",
        label="baseline archive",
    )
    _positive_int(archive.get("bytes"), "baseline archive bytes")
    _valid_sha256(archive.get("sha256"), "baseline archive SHA-256")
    if int(partial.get("bytes", -1)) != int(partial.get("tokens", -1)) * 2:
        raise ValueError("Glue final-corpus baseline partial bytes changed")
    _valid_sha256(partial.get("sha256"), "baseline partial SHA-256")
    for row in baseline["sealed_shards"]:
        if int(row.get("bytes", -1)) != int(row.get("tokens", -1)) * 2:
            raise ValueError("Glue final-corpus baseline shard bytes changed")
        _valid_sha256(row.get("sha256"), "baseline shard SHA-256")
        _bounded_uri(
            row.get("s3_uri"),
            bucket=bucket,
            prefix="lumi-corpus/v0.1",
            label="baseline token shard",
        )
    return {
        "maximum_usd": job["maximum_compute_usd"],
        "release_fingerprint": release["release_fingerprint"],
        "status": "verified_not_authorized",
    }


def expected_glue_final_corpus_canary_approval(release: Mapping[str, Any]) -> str:
    validate_glue_final_corpus_canary_release(release)
    dedup = str(release["input"]["dedup_result_sha256"])
    return (
        "I approve one bounded Lumi 8-DPU, 30-minute/$1.76 Glue final-80B "
        f"selection-and-sharding canary against dedup result {dedup} only. "
        "No production selection, GPU, or model training."
    )


def build_glue_final_corpus_canary_authorization(
    release: Mapping[str, Any],
    *,
    approval_text: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    validate_glue_final_corpus_canary_release(release)
    expected = expected_glue_final_corpus_canary_approval(release)
    if approval_text != expected:
        raise ValueError("Glue final-corpus owner approval text does not match")
    issued = (now or datetime.now(UTC)).astimezone(UTC)
    authorization: dict[str, Any] = {
        "approval_text": approval_text,
        "bounds": {
            "authorized_runs": 1,
            "dedup_result_sha256": release["input"]["dedup_result_sha256"],
            "hard_maximum_dpu": 8,
            "maximum_runtime_minutes": 30,
            "maximum_usd": "1.760000",
            "plan_fingerprint": release["input"]["plan_fingerprint"],
        },
        "expires_at": (issued + timedelta(hours=24)).isoformat(),
        "gate": AUTHORIZATION_GATE,
        "issued_at": issued.isoformat(),
        "permissions": {
            "full_4b_training": False,
            "model_training": False,
            "paid_glue_canary": True,
            "paid_glue_production": False,
            "paid_gpu_compute": False,
        },
        "release_fingerprint": release["release_fingerprint"],
        "schema_version": SCHEMA_VERSION,
        "source_commit": release["source_commit"],
    }
    authorization["authorization_sha256"] = _fingerprint(
        authorization, "authorization_sha256"
    )
    validate_glue_final_corpus_canary_authorization(
        authorization, release, now=issued
    )
    return authorization


def validate_glue_final_corpus_canary_authorization(
    authorization: Mapping[str, Any],
    release: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    validate_glue_final_corpus_canary_release(release)
    if authorization.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Glue final-corpus authorization schema")
    if authorization.get("gate") != AUTHORIZATION_GATE:
        raise ValueError("unexpected Glue final-corpus authorization gate")
    if authorization.get("authorization_sha256") != _fingerprint(
        authorization, "authorization_sha256"
    ):
        raise ValueError("Glue final-corpus authorization fingerprint changed")
    if authorization.get("approval_text") != expected_glue_final_corpus_canary_approval(
        release
    ):
        raise ValueError("Glue final-corpus approval text changed")
    if authorization.get("release_fingerprint") != release["release_fingerprint"]:
        raise ValueError("Glue final-corpus authorization targets another release")
    if authorization.get("source_commit") != release["source_commit"]:
        raise ValueError("Glue final-corpus authorization source commit changed")
    if authorization.get("permissions") != {
        "full_4b_training": False,
        "model_training": False,
        "paid_glue_canary": True,
        "paid_glue_production": False,
        "paid_gpu_compute": False,
    }:
        raise ValueError("Glue final-corpus authorization permissions changed")
    if authorization.get("bounds") != {
        "authorized_runs": 1,
        "dedup_result_sha256": release["input"]["dedup_result_sha256"],
        "hard_maximum_dpu": 8,
        "maximum_runtime_minutes": 30,
        "maximum_usd": "1.760000",
        "plan_fingerprint": release["input"]["plan_fingerprint"],
    }:
        raise ValueError("Glue final-corpus authorization bounds changed")
    current = (now or datetime.now(UTC)).astimezone(UTC)
    issued = datetime.fromisoformat(str(authorization["issued_at"])).astimezone(UTC)
    expires = datetime.fromisoformat(str(authorization["expires_at"])).astimezone(UTC)
    if expires <= issued or expires - issued > timedelta(hours=24):
        raise ValueError("Glue final-corpus authorization lifetime changed")
    if current < issued - timedelta(minutes=5) or current >= expires:
        raise ValueError("Glue final-corpus authorization is not currently valid")
    return {
        "authorization_sha256": authorization["authorization_sha256"],
        "maximum_usd": "1.760000",
        "status": "authorized_one_bounded_final_corpus_canary",
    }


def write_glue_final_corpus_canary_authorization(
    release_path: Path,
    output_path: Path,
    *,
    approval_text: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    release = _read_json(release_path)
    authorization = build_glue_final_corpus_canary_authorization(
        release,
        approval_text=approval_text,
        now=now,
    )
    _write_bytes_atomic(
        output_path,
        (json.dumps(authorization, indent=2, sort_keys=True) + "\n").encode(),
    )
    return authorization


def glue_final_corpus_canary_job_token(authorization: Mapping[str, Any]) -> str:
    digest = _valid_sha256(
        authorization.get("authorization_sha256"),
        "Glue final-corpus authorization",
    )
    return hashlib.sha256(f"glue-final-corpus-canary:{digest}".encode()).hexdigest()[:32]


def _validate_inventory(
    inventory: Any,
    *,
    release: Mapping[str, Any],
    expected_prefix: str,
    label: str,
) -> dict[str, Any]:
    if not isinstance(inventory, Mapping):
        raise ValueError(f"Glue final-corpus {label} inventory is missing")
    expected_uri = f"s3://{release['storage']['bucket']}/{expected_prefix}"
    if inventory.get("s3_uri") != expected_uri:
        raise ValueError(f"Glue final-corpus {label} left its bounded prefix")
    _positive_int(inventory.get("object_bytes"), f"{label} bytes")
    _positive_int(inventory.get("object_count"), f"{label} objects")
    _valid_sha256(inventory.get("objects_sha256"), f"{label} inventory SHA-256")
    return dict(inventory)


def validate_glue_final_corpus_canary_result(
    result: Mapping[str, Any],
    release: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    validate_glue_final_corpus_canary_authorization(authorization, release)
    if result.get("schema_version") != SCHEMA_VERSION or result.get("gate") != RESULT_GATE:
        raise ValueError("unexpected Glue final-corpus canary result schema")
    if result.get("status") != "pass_selection_sharding_canary":
        raise ValueError("Glue final-corpus canary is incomplete")
    expected = {
        "authorization_sha256": authorization["authorization_sha256"],
        "dedup_result_sha256": release["input"]["dedup_result_sha256"],
        "job_token": glue_final_corpus_canary_job_token(authorization),
        "plan_fingerprint": release["input"]["plan_fingerprint"],
        "release_fingerprint": release["release_fingerprint"],
        "selection_seed_sha256": release["input"]["selection_seed_sha256"],
    }
    for name, value in expected.items():
        if result.get(name) != value:
            raise ValueError(f"Glue final-corpus canary result changed {name}")
    observed_input = result.get("accepted_input_inventory")
    if observed_input != release["input"]["accepted_candidates"]:
        raise ValueError("Glue final-corpus canary input inventory changed")
    baseline = result.get("baseline_verification")
    if baseline != {
        "archive_head": "pass",
        "sealed_shard_heads": 2,
        "status": "pass",
    }:
        raise ValueError("Glue final-corpus baseline verification changed")
    rows = result.get("categories")
    if not isinstance(rows, list):
        raise ValueError("Glue final-corpus canary categories are missing")
    expected_categories = {str(row["category"]): row for row in release["categories"]}
    observed: set[str] = set()
    total_documents = 0
    total_tokens = 0
    total_output_bytes = 0
    token_target = int(release["limits"]["selected_target_tokens_per_category"])
    boundary = int(
        release["limits"]["maximum_document_boundary_overshoot_tokens"]
    )
    shard_target = int(release["limits"]["packed_shard_target_tokens"])
    token_uris: set[str] = set()
    token_hashes: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or str(row.get("category")) not in expected_categories:
            raise ValueError("Glue final-corpus canary category is invalid")
        category = str(row["category"])
        if category in observed:
            raise ValueError("Glue final-corpus canary repeats a category")
        observed.add(category)
        available_documents = _positive_int(
            row.get("available_documents"), "canary available documents"
        )
        available_tokens = _positive_int(
            row.get("available_tokens"), "canary available tokens"
        )
        selected_documents = _positive_int(
            row.get("selected_documents"), "canary selected documents"
        )
        selected_tokens = _positive_int(
            row.get("selected_tokens"), "canary selected tokens"
        )
        if selected_documents > available_documents or selected_tokens > available_tokens:
            raise ValueError("Glue final-corpus canary selected unavailable input")
        if not token_target <= selected_tokens <= token_target + boundary:
            raise ValueError("Glue final-corpus canary selection left its token envelope")
        if row.get("partition_parity") != "pass":
            raise ValueError("Glue final-corpus partition parity failed")
        _valid_sha256(row.get("selection_sha256"), "canary selection SHA-256")
        cutoff = row.get("cutoff")
        if not isinstance(cutoff, Mapping):
            raise ValueError("Glue final-corpus canary cutoff is missing")
        _valid_sha256(cutoff.get("_lumi_selection_rank"), "canary cutoff rank")
        selected_prefix = (
            f"{release['output']['canary_prefix']}/"
            f"{glue_final_corpus_canary_job_token(authorization)}/selected/{category}"
        )
        selected_inventory = _validate_inventory(
            row.get("selected_inventory"),
            release=release,
            expected_prefix=selected_prefix,
            label=f"{category} selected",
        )
        total_output_bytes += int(selected_inventory["object_bytes"])
        shards = row.get("packed_shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError("Glue final-corpus canary packed shards are missing")
        shard_tokens = 0
        shard_documents = 0
        for sequence, shard in enumerate(shards):
            if not isinstance(shard, Mapping) or int(shard.get("sequence", -1)) != sequence:
                raise ValueError("Glue final-corpus canary shard sequence changed")
            tokens = _positive_int(shard.get("tokens"), "canary packed tokens")
            documents = _positive_int(shard.get("documents"), "canary packed documents")
            object_bytes = _positive_int(shard.get("object_bytes"), "canary packed bytes")
            if object_bytes != tokens * 2 or tokens > shard_target + boundary:
                raise ValueError("Glue final-corpus packed-shard bounds changed")
            digest = _valid_sha256(shard.get("object_sha256"), "canary packed SHA-256")
            uri = _bounded_uri(
                shard.get("s3_uri"),
                bucket=str(release["storage"]["bucket"]),
                prefix=(
                    f"{release['output']['canary_prefix']}/"
                    f"{glue_final_corpus_canary_job_token(authorization)}/packed/"
                    f"{category}"
                ),
                label="canary packed shard",
            )
            if digest in token_hashes or uri in token_uris:
                raise ValueError("Glue final-corpus canary repeats a packed object")
            token_hashes.add(digest)
            token_uris.add(uri)
            shard_tokens += tokens
            shard_documents += documents
            total_output_bytes += object_bytes
        if shard_tokens != selected_tokens or shard_documents != selected_documents:
            raise ValueError("Glue final-corpus canary shard conservation failed")
        total_documents += selected_documents
        total_tokens += selected_tokens
    if observed != set(expected_categories):
        raise ValueError("Glue final-corpus canary category set changed")
    if int(result.get("selected_documents", -1)) != total_documents or int(
        result.get("selected_tokens", -1)
    ) != total_tokens:
        raise ValueError("Glue final-corpus canary aggregate conservation failed")
    if int(result.get("output_bytes", -1)) != total_output_bytes:
        raise ValueError("Glue final-corpus canary output-byte accounting changed")
    if total_output_bytes > int(release["limits"]["maximum_new_output_bytes"]):
        raise ValueError("Glue final-corpus canary exceeded its output cap")
    runtime = result.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("Glue final-corpus canary runtime is missing")
    if runtime.get("python", "").split(".")[:2] != ["3", "11"]:
        raise ValueError("Glue final-corpus canary Python runtime changed")
    if not str(runtime.get("spark", "")).startswith("3.5.6"):
        raise ValueError("Glue final-corpus canary Spark runtime changed")
    return dict(result)


def build_glue_final_corpus_canary_job_definition(
    release: Mapping[str, Any],
    authorization: Mapping[str, Any],
    *,
    role_arn: str,
    release_s3_uri: str,
    authorization_s3_uri: str,
) -> dict[str, Any]:
    """Render the one bounded Glue canary definition without calling AWS."""

    validate_glue_final_corpus_canary_authorization(authorization, release)
    if re.fullmatch(
        r"arn:aws:iam::[0-9]{12}:role/[A-Za-z0-9+=,.@_/-]+", role_arn
    ) is None:
        raise ValueError("Glue final-corpus canary role ARN is invalid")
    bucket = str(release["storage"]["bucket"])
    namespace = str(release["storage"]["namespace"])
    files = [
        release_s3_uri,
        release["artifacts"]["final_plan"]["s3_uri"],
        authorization_s3_uri,
    ]
    for uri in files:
        _bounded_uri(
            uri,
            bucket=bucket,
            prefix=namespace,
            label="Glue final-corpus bootstrap file",
        )
    basenames = [Path(str(uri)).name for uri in files]
    if len(basenames) != len(set(basenames)):
        raise ValueError("Glue final-corpus bootstrap basenames collide")
    for name in ("code_archive", "job_script"):
        _bounded_uri(
            release["artifacts"][name]["s3_uri"],
            bucket=bucket,
            prefix=namespace,
            label=f"Glue final-corpus {name}",
        )
    token = glue_final_corpus_canary_job_token(authorization)
    job = release["job"]
    temp_uri = (
        f"s3://{bucket}/{namespace}/control/glue-canary/temp/{token}/"
    )
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
            "--LUMI_FINAL_PLAN_FILE": basenames[1],
            "--LUMI_RELEASE_FILE": basenames[0],
            "--TempDir": temp_uri,
            "--conf": glue_conf_argument(FINAL_CORPUS_SPARK_CONFIG),
            "--enable-auto-scaling": "false",
            "--extra-files": ",".join(files),
            "--extra-py-files": release["artifacts"]["code_archive"]["s3_uri"],
            "--job-language": "python",
        },
        "Description": (
            "Lumi bounded final-80B selection/sharding canary; no production, "
            "GPU, or model training"
        ),
        "ExecutionClass": "STANDARD",
        "ExecutionProperty": {"MaxConcurrentRuns": 1},
        "GlueVersion": release["runtime"]["glue_version"],
        "MaxRetries": 0,
        "Name": f"lumi-glue-final-corpus-canary-{token[:12]}",
        "NumberOfWorkers": int(job["number_of_workers"]),
        "Role": role_arn,
        "Tags": {
            "LumiPlanFingerprint": str(release["input"]["plan_fingerprint"]),
            "LumiScope": "bounded-final-corpus-canary-only",
        },
        "Timeout": int(job["maximum_runtime_minutes"]),
        "WorkerType": job["worker_type"],
    }
