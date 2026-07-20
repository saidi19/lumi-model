"""Fail-closed control plane for Lumi's resumable production EMR corpus run.

This module never calls AWS.  It binds the already verified production prefix,
the successful S3/EMR parity result, resource ceilings, source cursor, category
floors, shard accounting, and immutable checkpoint chain.  AWS provisioning is
kept behind a separate exact authorization.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .production_corpus import _component_provenance, build_round_robin_source_schedule

SCHEMA_VERSION = 1
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
PRODUCTION_APPROVAL_TEXT = (
    "I approve one bounded Lumi EMR Serverless production corpus ramp only."
)
PRODUCTION_RUN_APPROVAL_TEXT = (
    "I approve one bounded Lumi EMR Serverless 128-vCPU production corpus "
    "acquisition under the recorded budget."
)
IMPLEMENTATION_FILES = (
    "scripts/orchestrate_emr_two_pass.py",
    "scripts/plan_emr_production_resume.py",
    "scripts/run_emr_serverless_reserved_map.py",
    "scripts/run_emr_serverless_global_dedup.py",
    "scripts/run_emr_serverless_shard_writer.py",
    "scripts/seal_emr_production_corpus.py",
    "src/lumi/data_quality.py",
    "src/lumi/distributed_corpus.py",
    "src/lumi/emr_candidate_mapping.py",
    "src/lumi/emr_production_adaptive.py",
    "src/lumi/emr_production_shards.py",
    "src/lumi/emr_production_staging.py",
    "src/lumi/emr_production_dedup.py",
    "src/lumi/emr_production_map.py",
    "src/lumi/emr_production_seal.py",
    "src/lumi/emr_production_shard_writer.py",
    "src/lumi/emr_serverless_production.py",
    "src/lumi/emr_two_pass_controller.py",
    "src/lumi/glue_corpus.py",
    "src/lumi/production_corpus.py",
)

EXPECTED_LIMITS = {
    "candidate_hard_maximum_tokens": 150_000_000_000,
    "candidate_total_floor_tokens": 120_000_000_000,
    "clean_shard_target_tokens": 100_000_000,
    "final_training_tokens": 80_000_000_000,
    "hard_logical_storage_bytes": 2_000_000_000_000,
    "maximum_category_overshoot_tokens": 100_000_000,
    "maximum_transient_source_bytes": 100_000_000_000,
}

EXPECTED_BUDGET = {
    "contingency_usd": "250.00",
    "hard_total_usd": "2000.00",
    "maximum_archive_migration_usd": "250.00",
    "maximum_emr_compute_usd": "1200.00",
    "maximum_s3_logs_requests_usd": "100.00",
    "maximum_stager_compute_usd": "200.00",
}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _resolve(root: Path, raw_path: Any, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} path is invalid")
    path = (root / raw_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{label} escapes the repository")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _commit(value: Any) -> str:
    if not isinstance(value, str) or COMMIT_PATTERN.fullmatch(value) is None:
        raise ValueError("source commit must be a lowercase 40-character Git SHA-1")
    return value


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{label} must be {'non-negative' if allow_zero else 'positive'}")
    return value


def _category_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for row in rows:
        category = str(row["category"])
        if category in result:
            raise ValueError("production plan contains duplicate categories")
        result[category] = {
            "accepted_documents": int(row["accepted_documents"]),
            "accepted_tokens": int(row["accepted_tokens"]),
            "floor_tokens": int(row["floor_tokens"]),
        }
    return result


def _hourly_cost(
    *,
    vcpu: int,
    memory_gib: int,
    disk_gib: int,
    pricing: Mapping[str, Any],
) -> Decimal:
    return (
        Decimal(vcpu) * Decimal(str(pricing["vcpu_hour_usd"]))
        + Decimal(memory_gib) * Decimal(str(pricing["memory_gb_hour_usd"]))
        + Decimal(disk_gib) * Decimal(str(pricing["standard_storage_gb_hour_usd"]))
    )


def _resource_shape(application: Mapping[str, Any], maximum_vcpu: int) -> dict[str, int]:
    driver = application["driver"]
    executor = application["executor"]
    driver_cores = int(driver["cores"])
    executor_cores = int(executor["cores"])
    if maximum_vcpu <= driver_cores or (maximum_vcpu - driver_cores) % executor_cores:
        raise ValueError("EMR maximum vCPU does not fit the pinned driver/executor shape")
    executors = (maximum_vcpu - driver_cores) // executor_cores
    if executors < 1:
        raise ValueError("EMR production needs at least one executor")
    return {
        "disk_gib": int(driver["disk_gib"]) + executors * int(executor["disk_gib"]),
        "executor_instances": executors,
        "memory_gib": int(driver["memory_gib"]) + executors * int(executor["memory_gib"]),
        "vcpu": driver_cores + executors * executor_cores,
    }


def build_emr_production_plan(
    contract_path: Path,
    *,
    repository_root: Path,
    source_commit: str,
) -> dict[str, Any]:
    """Bind the complete production corpus control plane without launching it."""

    root = repository_root.resolve()
    contract = _read_json(contract_path)
    if contract.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValueError("unsupported EMR production schema")
    if contract.get("gate") != "lumi_emr_serverless_production_resume_plan_v0.1":
        raise ValueError("unexpected EMR production gate")
    commit = _commit(source_commit)
    authorization = contract.get("authorization", {})
    if authorization.get("production_adaptive_acquisition") is not True:
        raise ValueError("production adaptive acquisition is not authorized")
    for name in (
        "full_4b_training",
        "model_training",
        "paid_cpu_stager",
        "paid_emr_production",
        "paid_gpu_compute",
    ):
        if authorization.get(name) is not False:
            raise ValueError(f"local production plan must not authorize {name}")

    configured_inputs = contract.get("inputs")
    expected_inputs = {
        "baseline_evidence",
        "benchmark_registry",
        "canary_parity_report",
        "component_inventory",
        "corpus_storage",
        "data_mix",
        "data_quality",
        "emr_pricing",
        "filter_policy",
        "production_authorization",
        "source_file_manifest",
        "stager_pricing",
    }
    if not isinstance(configured_inputs, Mapping) or set(configured_inputs) != expected_inputs:
        raise ValueError("EMR production input set changed")
    paths: dict[str, Path] = {}
    inputs: dict[str, dict[str, str]] = {}
    for name, pin in sorted(configured_inputs.items()):
        if not isinstance(pin, Mapping):
            raise ValueError(f"invalid EMR production input pin: {name}")
        path = _resolve(root, pin.get("path"), name)
        digest = _sha256_file(path)
        if digest != _sha256(pin.get("sha256"), f"{name} SHA-256"):
            raise ValueError(f"EMR production input changed: {name}")
        paths[name] = path
        inputs[name] = {"path": str(path.relative_to(root)), "sha256": digest}

    limits = contract.get("limits")
    if limits != EXPECTED_LIMITS:
        raise ValueError("EMR production token/storage limits differ from the roadmap")
    production = _read_json(paths["production_authorization"])
    if production.get("limits") != {k: v for k, v in EXPECTED_LIMITS.items() if k not in {
        "clean_shard_target_tokens", "maximum_transient_source_bytes"
    }}:
        raise ValueError("EMR production limits differ from production S3 authorization")
    if production.get("acquisition", {}).get("clean_shard_target_tokens") != 100_000_000:
        raise ValueError("production clean-shard target changed")
    budget = contract.get("budget")
    if budget != EXPECTED_BUDGET:
        raise ValueError("EMR production budget differs from the owner-approved $2,000 cap")
    if sum(Decimal(value) for key, value in budget.items() if key != "hard_total_usd") != (
        Decimal(budget["hard_total_usd"])
    ):
        raise ValueError("EMR production budget allocations do not conserve the hard cap")
    pipeline = contract.get("pipeline")
    if pipeline != {
        "global_dedup_job_timeout_hours": "24",
        "global_dedup_job_vcpu": 128,
        "map_chunks_per_job": 8,
        "map_job_timeout_hours": "2",
        "map_job_vcpu": 16,
        "mapped_category_multiplier": "1.20",
        "maximum_concurrent_map_jobs": 7,
        "maximum_concurrent_shard_writers": 7,
        "shard_writer_job_vcpu": 16,
        "shard_writer_job_timeout_hours": "12",
        "strategy": "two_pass_map_global_dedup_shard_v0.1",
    }:
        raise ValueError("EMR production two-pass pipeline contract changed")
    if pipeline["map_job_vcpu"] * pipeline["maximum_concurrent_map_jobs"] > (
        pipeline["global_dedup_job_vcpu"]
    ):
        raise ValueError("EMR production map concurrency exceeds 128 vCPU")
    if pipeline["shard_writer_job_vcpu"] * pipeline[
        "maximum_concurrent_shard_writers"
    ] > pipeline["global_dedup_job_vcpu"]:
        raise ValueError("EMR production shard-writer concurrency exceeds 128 vCPU")

    runtime = contract.get("runtime", {})
    if runtime != {
        "custom_python": "3.12",
        "release_label": "emr-7.13.0",
        "service": "aws_emr_serverless",
        "spark_version": "3.5.6",
    }:
        raise ValueError("EMR production runtime changed")
    application = contract.get("application", {})
    expected_application = {
        "architecture": "X86_64",
        "auto_stop_idle_timeout_minutes": 5,
        "bootstrap_maximum_vcpu": 16,
        "driver": {"cores": 4, "disk_gib": 20, "memory_gib": 14},
        "executor": {"cores": 4, "disk_gib": 20, "memory_gib": 14},
        "production_maximum_vcpu": 128,
        "quota_code": "L-D05C8A75",
        "ramp_maximum_vcpu": 64,
        "requested_account_vcpu": 128,
    }
    if application != expected_application:
        raise ValueError("EMR production resource envelope changed")
    checkpoint = contract.get("checkpoint", {})
    if checkpoint != {
        "commit_after_each_source": True,
        "commit_after_each_target_shard": True,
        "content_addressed": True,
        "immutable_previous_checkpoint_chain": True,
        "resume_from_verified_prefix": True,
    }:
        raise ValueError("EMR production checkpoint contract changed")
    stager = contract.get("source_stager", {})
    if stager != {
        "bootstrap_maximum_chunks": 8,
        "instance_memory_gib": 128,
        "instance_type": "r7i.4xlarge",
        "instance_vcpus": 16,
        "maximum_local_staged_bytes": 12_000_000_000,
        "maximum_prefetched_sources": 1,
        "maximum_runtime_hours": 96,
        "production_maximum_chunks": None,
        "raw_retention": "delete_after_verified_checkpoint_commit",
        "ramp_maximum_chunks": 32,
        "split_chunk_uncompressed_bytes": 134_217_728,
        "strategy": "internet_facing_cpu_stage_once_to_private_s3_v0.1",
        "volume_gib": 100,
    }:
        raise ValueError("EMR production source-stager contract changed")

    parity = _read_json(paths["canary_parity_report"])
    if parity.get("status") != "pass_exact_cloud_local_parity":
        raise ValueError("S3/EMR cloud-local parity gate did not pass")
    if parity.get("parity", {}).get("local_and_cloud_parity_sha256") != (
        "13edb46411c3b5eeaaf35237f70e379d72eb22b8a223c0d70586ece649c96837"
    ):
        raise ValueError("S3/EMR parity identity changed")
    if parity.get("production_shards_written") is not False:
        raise ValueError("canary unexpectedly wrote production shards")

    baseline = _read_json(paths["baseline_evidence"])
    if baseline.get("verification", {}).get("status") != "independently_verified":
        raise ValueError("production resume baseline is not independently verified")
    if baseline.get("totals") != {
        "accepted_documents": 38_349,
        "accepted_tokens": 237_769_640,
        "sealed_documents": 32_393,
        "sealed_tokens": 200_614_708,
    }:
        raise ValueError("production resume totals changed")
    if baseline.get("partial_prefix", {}).get("source_ordinal") != 0 or baseline.get(
        "partial_prefix", {}
    ).get("source_next_line") != 38_848:
        raise ValueError("production resume source cursor changed")

    source_manifest = _read_json(paths["source_file_manifest"])
    mixture = _read_json(paths["data_mix"])
    data_quality = _read_json(paths["data_quality"])
    filter_contract = _read_json(paths["filter_policy"])
    filters = filter_contract.get("filters")
    if not isinstance(filters, Mapping):
        raise ValueError("EMR production filter policy is missing")
    schedule, raw_categories = build_round_robin_source_schedule(source_manifest, mixture)
    provenance = _component_provenance(_read_json(paths["component_inventory"]))
    schedule = [
        {**row, "source_provenance": provenance[row["component"]]} for row in schedule
    ]
    categories = []
    for row in raw_categories:
        accepted_tokens = 237_769_640 if row["category"] == "science_and_education" else 0
        accepted_documents = 38_349 if row["category"] == "science_and_education" else 0
        categories.append(
            {
                **row,
                "accepted_documents": accepted_documents,
                "accepted_tokens": accepted_tokens,
            }
        )

    pricing = _read_json(paths["emr_pricing"])
    if pricing.get("region") != "us-east-1" or pricing.get("architecture") != "X86_64":
        raise ValueError("EMR pricing region/architecture changed")
    stager_pricing = _read_json(paths["stager_pricing"])
    stager_ec2 = stager_pricing.get("ec2", {})
    if (
        stager_ec2.get("instance_type") != stager["instance_type"]
        or stager_ec2.get("vcpus") != stager["instance_vcpus"]
        or stager_ec2.get("memory_gib") != stager["instance_memory_gib"]
    ):
        raise ValueError("production source-stager pricing shape changed")
    stager_compute_cap = Decimal(str(stager_ec2["on_demand_usd_per_hour"])) * Decimal(
        int(stager["maximum_runtime_hours"])
    )
    stager_storage_cap = (
        Decimal(str(stager_pricing["ebs"]["storage_usd_per_gb_month"]))
        * Decimal(int(stager["volume_gib"]))
        * Decimal(int(stager["maximum_runtime_hours"]))
        / Decimal(730)
    )
    if stager_compute_cap + stager_storage_cap > Decimal(
        budget["maximum_stager_compute_usd"]
    ):
        raise ValueError("production source-stager exceeds its $200 allocation")
    bootstrap_shape = _resource_shape(application, 16)
    ramp_shape = _resource_shape(application, 64)
    production_shape = _resource_shape(application, 128)
    hourly = {
        "bootstrap_usd": str(
            _hourly_cost(
                vcpu=bootstrap_shape["vcpu"],
                memory_gib=bootstrap_shape["memory_gib"],
                disk_gib=bootstrap_shape["disk_gib"],
                pricing=pricing,
            ).quantize(Decimal("0.000001"))
        ),
        "ramp_usd": str(
            _hourly_cost(
                vcpu=ramp_shape["vcpu"],
                memory_gib=ramp_shape["memory_gib"],
                disk_gib=ramp_shape["disk_gib"],
                pricing=pricing,
            ).quantize(Decimal("0.000001"))
        ),
        "production_usd": str(
            _hourly_cost(
                vcpu=production_shape["vcpu"],
                memory_gib=production_shape["memory_gib"],
                disk_gib=production_shape["disk_gib"],
                pricing=pricing,
            ).quantize(Decimal("0.000001"))
        ),
        "stager_maximum_compute_usd": str(
            stager_compute_cap.quantize(Decimal("0.000001"))
        ),
        "stager_maximum_storage_usd": str(
            stager_storage_cap.quantize(Decimal("0.000001"))
        ),
    }
    implementation = {
        path: _sha256_file(_resolve(root, path, "EMR production implementation"))
        for path in IMPLEMENTATION_FILES
    }
    material = {
        "application": application,
        "budget": budget,
        "checkpoint": checkpoint,
        "inputs": inputs,
        "implementation": implementation,
        "limits": limits,
        "pipeline": pipeline,
        "runtime": runtime,
        "source_commit": commit,
        "source_stager": stager,
    }
    return {
        "application": application,
        "authorization": dict(authorization),
        "artifacts": {
            "benchmark_registry": {
                "bytes": paths["benchmark_registry"].stat().st_size,
                "sha256": inputs["benchmark_registry"]["sha256"],
            },
            "frozen_tokenizer": production["external_artifacts"]["frozen_tokenizer"],
            "initial_anchors": {
                "bytes": baseline["derived_resume_anchor"]["bytes"],
                "sha256": baseline["derived_resume_anchor"]["sha256"],
            },
            "training_exclusion_ledger": production["external_artifacts"][
                "training_exclusion_ledger"
            ],
        },
        "budget": budget,
        "categories": categories,
        "checkpoint": checkpoint,
        "cost": {**hourly, "pricing": pricing},
        "decision": {
            "full_4b_training": "not_authorized",
            "paid_cpu_stager": "not_authorized",
            "paid_emr_production": "not_authorized",
            "status": "local_plan_ready_waiting_for_quota_and_exact_paid_authorization",
        },
        "gate": contract["gate"],
        "data_quality": data_quality,
        "filters": dict(filters),
        "inputs": inputs,
        "implementation": implementation,
        "limits": limits,
        "pipeline": pipeline,
        "plan_fingerprint": _sha256_bytes(_canonical_json_bytes(material)),
        "region": contract.get("region"),
        "resume": {
            "accepted_documents": 38_349,
            "accepted_tokens": 237_769_640,
            "archive": baseline["archive"],
            "anchor_bytes": baseline["derived_resume_anchor"]["bytes"],
            "anchor_rows": baseline["derived_resume_anchor"]["rows"],
            "anchor_sha256": baseline["derived_resume_anchor"]["sha256"],
            "partial_prefix": baseline["partial_prefix"],
            "sealed_shards": baseline["sealed_shards"],
            "source_next_line": 38_848,
            "source_ordinal": 0,
        },
        "runtime": runtime,
        "schema_version": SCHEMA_VERSION,
        "shapes": {
            "bootstrap": bootstrap_shape,
            "production": production_shape,
            "ramp": ramp_shape,
        },
        "source_commit": commit,
        "storage": {
            "artifacts": {
                "benchmark_registry": (
                    "lumi-corpus/v0.1/control/distributed/emr-serverless/bootstrap/"
                    "benchmark/be02bdbd7f2c0e807c86cf9678eff86d711aaa34ffdc8e6ebb687a26220ffcbb."
                    "jsonl.gz"
                ),
                "frozen_tokenizer": (
                    "lumi-corpus/v0.1/control/bootstrap/artifacts/frozen_tokenizer/"
                    "1022c9538d9ec795b07c2e2bda3c5c6f21c321de933cfe5ad197628a94caa2ee.json"
                ),
                "initial_anchors": (
                    "lumi-corpus/v0.1/control/distributed/emr-serverless/bootstrap/anchors/"
                    "530ac0e1b6773cc2182aba6bd2dd923e4f0362bae3289b312989ab06cebc1ee1."
                    "jsonl.gz"
                ),
                "training_exclusion_ledger": (
                    "lumi-corpus/v0.1/control/bootstrap/artifacts/"
                    "training_exclusion_ledger/"
                    "709027bb95f3e0f956b22a79d4b29f21490ef0b360ba10f2faca7d5db3d06c62."
                    "jsonl"
                ),
            },
            "bucket": baseline["bucket"],
            "namespace": baseline["namespace"],
        },
        "source_inventory": {
            "compressed_source_bytes": sum(int(row["source_bytes"]) for row in schedule),
            "data_files": len(schedule),
        },
        "source_schedule": schedule,
        "source_stager": stager,
    }


def checkpoint_fingerprint(checkpoint: Mapping[str, Any]) -> str:
    material = dict(checkpoint)
    material.pop("checkpoint_sha256", None)
    return _sha256_bytes(_canonical_json_bytes(material))


def initial_production_checkpoint(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Create checkpoint zero from the independently verified single-writer prefix."""

    checkpoint = {
        "anchor_segments": [],
        "categories": plan["categories"],
        "checkpoint_sequence": 0,
        "gate": "lumi_emr_production_checkpoint_v0.1",
        "logical_storage_bytes": 682_985_279,
        "next_shard_sequence": 2,
        "partial_shard": {
            "archive_bytes": plan["resume"]["archive"]["bytes"],
            "archive_key": plan["resume"]["archive"]["key"],
            "archive_sha256": plan["resume"]["archive"]["sha256"],
            "category": "science_and_education",
            "clean_bytes": plan["resume"]["partial_prefix"]["clean_bytes"],
            "clean_member": plan["resume"]["partial_prefix"]["evidence_member_clean"],
            "clean_sha256": plan["resume"]["partial_prefix"]["clean_sha256"],
            "documents": plan["resume"]["partial_prefix"]["accepted_documents"],
            "sequence": 2,
            "state": "baseline_archive",
            "text_bytes": 178_817_012,
            "token_bytes": plan["resume"]["partial_prefix"]["token_bytes"],
            "token_member": plan["resume"]["partial_prefix"]["evidence_member_tokens"],
            "token_sha256": plan["resume"]["partial_prefix"]["token_sha256"],
            "tokens": plan["resume"]["partial_prefix"]["accepted_tokens"],
        },
        "plan_fingerprint": plan["plan_fingerprint"],
        "previous_checkpoint_sha256": None,
        "run_binding": None,
        "schema_version": SCHEMA_VERSION,
        "sealed_shards": plan["resume"]["sealed_shards"],
        "source_cursor": {
            "next_line": plan["resume"]["source_next_line"],
            "source_ordinal": plan["resume"]["source_ordinal"],
        },
        "status": "ready",
        "total_accepted_documents": plan["resume"]["accepted_documents"],
        "total_accepted_tokens": plan["resume"]["accepted_tokens"],
    }
    checkpoint["checkpoint_sha256"] = checkpoint_fingerprint(checkpoint)
    validate_production_checkpoint(checkpoint, plan)
    return checkpoint


def validate_production_checkpoint(
    checkpoint: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate one immutable checkpoint and every roadmap hard stop."""

    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported production checkpoint schema")
    if checkpoint.get("gate") != "lumi_emr_production_checkpoint_v0.1":
        raise ValueError("unexpected production checkpoint gate")
    if checkpoint.get("plan_fingerprint") != plan.get("plan_fingerprint"):
        raise ValueError("production checkpoint belongs to another plan")
    observed = checkpoint_fingerprint(checkpoint)
    if checkpoint.get("checkpoint_sha256") != observed:
        raise ValueError("production checkpoint SHA-256 changed")
    sequence = _positive_int(
        checkpoint.get("checkpoint_sequence"), "checkpoint sequence", allow_zero=True
    )
    previous = checkpoint.get("previous_checkpoint_sha256")
    if sequence == 0 and previous is not None:
        raise ValueError("initial checkpoint cannot have a parent")
    if sequence > 0:
        _sha256(previous, "previous checkpoint")
    if checkpoint.get("status") not in {"ready", "running", "complete", "paused"}:
        raise ValueError("invalid production checkpoint status")
    binding = checkpoint.get("run_binding")
    if binding is not None:
        if not isinstance(binding, Mapping):
            raise ValueError("production checkpoint run binding is invalid")
        if binding.get("gate") != "lumi_emr_production_run_binding_v0.1":
            raise ValueError("production checkpoint run-binding gate changed")
        if binding.get("plan_fingerprint") != plan["plan_fingerprint"]:
            raise ValueError("production checkpoint run binding targets another plan")
        _sha256(binding.get("root_checkpoint_sha256"), "run-binding root checkpoint")
        run_token = binding.get("run_token")
        if not isinstance(run_token, str) or re.fullmatch(r"[0-9a-f]{32}", run_token) is None:
            raise ValueError("production checkpoint run token is invalid")
    total_tokens = _positive_int(
        checkpoint.get("total_accepted_tokens"), "total accepted tokens", allow_zero=True
    )
    total_documents = _positive_int(
        checkpoint.get("total_accepted_documents"),
        "total accepted documents",
        allow_zero=True,
    )
    limits = plan["limits"]
    if total_tokens > int(limits["candidate_hard_maximum_tokens"]):
        raise ValueError("production checkpoint exceeds the 150B hard maximum")
    logical = _positive_int(
        checkpoint.get("logical_storage_bytes"), "logical storage", allow_zero=True
    )
    if logical > int(limits["hard_logical_storage_bytes"]):
        raise ValueError("production checkpoint exceeds the 2TB hard storage cap")
    next_sequence = _positive_int(
        checkpoint.get("next_shard_sequence"), "next shard sequence", allow_zero=True
    )
    sealed_shards = checkpoint.get("sealed_shards")
    if not isinstance(sealed_shards, list):
        raise ValueError("production checkpoint sealed shards are missing")
    if len(sealed_shards) > next_sequence:
        raise ValueError("production checkpoint shard sequence regressed")
    partial = checkpoint.get("partial_shard")
    if partial is not None:
        if not isinstance(partial, Mapping):
            raise ValueError("production checkpoint partial shard is invalid")
        if _positive_int(partial.get("sequence"), "partial sequence", allow_zero=True) != (
            next_sequence
        ):
            raise ValueError("production partial shard sequence changed")
        partial_tokens = _positive_int(partial.get("tokens"), "partial tokens")
        if partial_tokens >= int(limits["clean_shard_target_tokens"]):
            raise ValueError("production partial shard must stay below the shard target")
        if _positive_int(partial.get("token_bytes"), "partial token bytes") != (
            partial_tokens * 2
        ):
            raise ValueError("production partial token byte count changed")
        _positive_int(partial.get("documents"), "partial documents")
        _positive_int(partial.get("text_bytes"), "partial text bytes")
        _sha256(partial.get("clean_sha256"), "partial clean")
        _sha256(partial.get("token_sha256"), "partial tokens")
        state = partial.get("state")
        if state == "baseline_archive":
            _sha256(partial.get("archive_sha256"), "partial archive")
            if not isinstance(partial.get("archive_key"), str):
                raise ValueError("baseline partial archive key is missing")
        elif state == "s3_snapshot":
            for name in ("clean_s3_key", "clean_version_id", "token_s3_key", "token_version_id"):
                if not isinstance(partial.get(name), str) or not partial[name]:
                    raise ValueError(f"production partial snapshot is missing {name}")
        else:
            raise ValueError("unsupported production partial-shard state")
    anchor_segments = checkpoint.get("anchor_segments")
    if not isinstance(anchor_segments, list):
        raise ValueError("production checkpoint anchor segments are missing")
    anchor_documents = 0
    seen_anchor_manifests: set[str] = set()
    for segment in anchor_segments:
        if not isinstance(segment, Mapping):
            raise ValueError("production anchor segment is invalid")
        digest = _sha256(segment.get("manifest_sha256"), "anchor manifest")
        if digest in seen_anchor_manifests:
            raise ValueError("production checkpoint repeats an anchor segment")
        seen_anchor_manifests.add(digest)
        anchor_documents += _positive_int(segment.get("documents"), "anchor documents")
    if plan["resume"]["anchor_rows"] + anchor_documents != total_documents:
        raise ValueError("production checkpoint anchor-document conservation failed")
    categories = checkpoint.get("categories")
    if not isinstance(categories, list):
        raise ValueError("production checkpoint categories are missing")
    category_values = _category_map(categories)
    plan_values = _category_map(plan["categories"])
    if set(category_values) != set(plan_values):
        raise ValueError("production checkpoint category set changed")
    for name, row in category_values.items():
        if row["floor_tokens"] != plan_values[name]["floor_tokens"]:
            raise ValueError(f"production category floor changed: {name}")
        if row["accepted_tokens"] > row["floor_tokens"] + int(
            limits["maximum_category_overshoot_tokens"]
        ):
            raise ValueError(f"production category overshoot exceeds one shard: {name}")
    if sum(row["accepted_tokens"] for row in category_values.values()) != total_tokens:
        raise ValueError("production checkpoint category-token conservation failed")
    if sum(row["accepted_documents"] for row in category_values.values()) != total_documents:
        raise ValueError("production checkpoint category-document conservation failed")
    cursor = checkpoint.get("source_cursor")
    if not isinstance(cursor, Mapping):
        raise ValueError("production source cursor is missing")
    source_ordinal = _positive_int(
        cursor.get("source_ordinal"), "source ordinal", allow_zero=True
    )
    _positive_int(cursor.get("next_line"), "source next line", allow_zero=True)
    if source_ordinal > len(plan["source_schedule"]):
        raise ValueError("production source cursor exceeds the source schedule")
    if source_ordinal == len(plan["source_schedule"]) and checkpoint.get("status") != (
        "complete"
    ):
        raise ValueError("production source schedule ended before every floor completed")
    if checkpoint.get("status") == "complete" and not production_floors_complete(
        checkpoint, plan
    ):
        raise ValueError("production checkpoint is complete before every category floor")
    return dict(checkpoint)


def production_floors_complete(
    checkpoint: Mapping[str, Any], plan: Mapping[str, Any]
) -> bool:
    values = _category_map(list(checkpoint["categories"]))
    return all(row["accepted_tokens"] >= row["floor_tokens"] for row in values.values())


def next_source_cursor(
    plan: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    staged_manifest: Mapping[str, Any],
    *,
    category_floor_reached: bool,
) -> dict[str, int]:
    """Advance within a staged source slice or move to the next source/category."""

    validate_production_checkpoint(checkpoint, plan)
    cursor = checkpoint["source_cursor"]
    ordinal = int(cursor["source_ordinal"])
    schedule = plan["source_schedule"]
    if ordinal >= len(schedule):
        raise ValueError("production source cursor is already at the schedule end")
    source = schedule[ordinal]
    chunks = staged_manifest.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("staged source slice has no chunks")
    if staged_manifest.get("start_after_line") != int(cursor["next_line"]):
        raise ValueError("staged source slice does not start at the checkpoint cursor")
    if category_floor_reached:
        next_ordinal = ordinal + 1
        while (
            next_ordinal < len(schedule)
            and schedule[next_ordinal]["category"] == source["category"]
        ):
            next_ordinal += 1
        return {"next_line": 0, "source_ordinal": next_ordinal}
    source_complete = staged_manifest.get("source_complete")
    if source_complete is False:
        return {
            "next_line": int(chunks[-1]["end_line"]),
            "source_ordinal": ordinal,
        }
    if source_complete is not True:
        raise ValueError("staged source completion state is invalid")
    next_ordinal = ordinal + 1
    if (
        next_ordinal < len(schedule)
        and schedule[next_ordinal]["category"] != source["category"]
    ):
        raise RuntimeError("production category exhausted before its candidate floor")
    return {"next_line": 0, "source_ordinal": next_ordinal}


def advance_production_checkpoint(
    checkpoint: Mapping[str, Any],
    plan: Mapping[str, Any],
    commit: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply one verified source/shard commit with idempotent retry semantics."""

    current = validate_production_checkpoint(checkpoint, plan)
    if current["status"] == "complete":
        raise ValueError("cannot advance a complete production checkpoint")
    commit_id = _sha256(commit.get("commit_sha256"), "production commit")
    material = dict(commit)
    material.pop("commit_sha256", None)
    if _sha256_bytes(_canonical_json_bytes(material)) != commit_id:
        raise ValueError("production commit SHA-256 changed")
    if commit.get("plan_fingerprint") != plan["plan_fingerprint"]:
        raise ValueError("production commit belongs to another plan")
    if commit.get("parent_checkpoint_sha256") != current["checkpoint_sha256"]:
        raise ValueError("production commit parent checkpoint changed")

    old_cursor = current["source_cursor"]
    new_cursor = commit.get("source_cursor")
    if not isinstance(new_cursor, Mapping):
        raise ValueError("production commit source cursor is missing")
    old_position = (int(old_cursor["source_ordinal"]), int(old_cursor["next_line"]))
    new_position = (int(new_cursor.get("source_ordinal", -1)), int(new_cursor.get("next_line", -1)))
    if new_position <= old_position:
        raise ValueError("production commit source cursor did not advance")
    if new_position[0] > len(plan["source_schedule"]):
        raise ValueError("production commit source cursor exceeds the schedule")

    deltas = commit.get("category_deltas")
    if not isinstance(deltas, Mapping) or not deltas:
        raise ValueError("production commit category deltas are missing")
    category_rows = {row["category"]: dict(row) for row in current["categories"]}
    added_tokens = 0
    added_documents = 0
    for category, delta in deltas.items():
        if category not in category_rows or not isinstance(delta, Mapping):
            raise ValueError("production commit contains an unknown category")
        tokens = _positive_int(
            delta.get("accepted_tokens"), "accepted token delta", allow_zero=True
        )
        documents = _positive_int(
            delta.get("accepted_documents"), "accepted document delta", allow_zero=True
        )
        if tokens == 0 and documents != 0:
            raise ValueError("production commit has documents without tokens")
        category_rows[category]["accepted_tokens"] += tokens
        category_rows[category]["accepted_documents"] += documents
        added_tokens += tokens
        added_documents += documents
    if added_tokens == 0:
        raise ValueError("production commit accepted no tokens")

    new_shards = commit.get("sealed_shards")
    if not isinstance(new_shards, list):
        raise ValueError("production commit sealed-shard list is missing")
    known_shards = {str(row["manifest_sha256"]) for row in current["sealed_shards"]}
    new_shard_tokens = 0
    new_shard_documents = 0
    for shard in new_shards:
        if not isinstance(shard, Mapping):
            raise ValueError("production commit shard is invalid")
        digest = _sha256(shard.get("manifest_sha256"), "shard manifest")
        if digest in known_shards:
            raise ValueError("production commit repeats a sealed shard")
        known_shards.add(digest)
        tokens = _positive_int(shard.get("tokens"), "shard tokens")
        new_shard_tokens += tokens
        new_shard_documents += _positive_int(shard.get("documents"), "shard documents")
        if tokens > int(plan["limits"]["clean_shard_target_tokens"]) + int(
            plan["limits"]["maximum_category_overshoot_tokens"]
        ):
            raise ValueError("production shard exceeds the bounded overshoot")
    old_partial = current.get("partial_shard")
    old_partial_tokens = 0 if old_partial is None else int(old_partial["tokens"])
    old_partial_documents = 0 if old_partial is None else int(old_partial["documents"])
    new_partial = commit.get("partial_shard")
    new_partial_tokens = 0
    new_partial_documents = 0
    if new_partial is not None:
        if not isinstance(new_partial, Mapping):
            raise ValueError("production commit partial shard is invalid")
        new_partial_tokens = _positive_int(new_partial.get("tokens"), "partial tokens")
        new_partial_documents = _positive_int(
            new_partial.get("documents"), "partial documents"
        )
    if new_shard_tokens + new_partial_tokens - old_partial_tokens != added_tokens:
        raise ValueError("production commit shard-token conservation failed")
    if new_shard_documents + new_partial_documents - old_partial_documents != added_documents:
        raise ValueError("production commit shard-document conservation failed")
    next_sequence = _positive_int(
        commit.get("next_shard_sequence"), "next shard sequence", allow_zero=True
    )
    if next_sequence != int(current["next_shard_sequence"]) + len(new_shards):
        raise ValueError("production commit next shard sequence changed")
    anchor_segment = commit.get("anchor_segment")
    if not isinstance(anchor_segment, Mapping):
        raise ValueError("production commit anchor segment is missing")
    if _positive_int(anchor_segment.get("documents"), "anchor documents") != added_documents:
        raise ValueError("production commit anchor-document count changed")
    _sha256(anchor_segment.get("manifest_sha256"), "anchor manifest")
    logical_delta = _positive_int(
        commit.get("logical_storage_bytes"), "logical storage delta", allow_zero=True
    )
    old_binding = current.get("run_binding")
    new_binding = commit.get("run_binding")
    if old_binding is None:
        if new_binding is not None:
            if not isinstance(new_binding, Mapping):
                raise ValueError("production commit run binding is invalid")
            if new_binding.get("gate") != "lumi_emr_production_run_binding_v0.1":
                raise ValueError("production commit run-binding gate changed")
            if new_binding.get("plan_fingerprint") != plan["plan_fingerprint"]:
                raise ValueError("production commit run binding targets another plan")
            if new_binding.get("root_checkpoint_sha256") != current["checkpoint_sha256"]:
                raise ValueError("production run binding must start at the current checkpoint")
            run_token = new_binding.get("run_token")
            if not isinstance(run_token, str) or re.fullmatch(r"[0-9a-f]{32}", run_token) is None:
                raise ValueError("production commit run token is invalid")
            new_binding = dict(new_binding)
    elif new_binding != old_binding:
        raise ValueError("production commit changed the active run binding")
    result = {
        **current,
        "categories": [category_rows[row["category"]] for row in current["categories"]],
        "checkpoint_sequence": int(current["checkpoint_sequence"]) + 1,
        "anchor_segments": [*current["anchor_segments"], dict(anchor_segment)],
        "logical_storage_bytes": int(current["logical_storage_bytes"]) + logical_delta,
        "next_shard_sequence": next_sequence,
        "partial_shard": None if new_partial is None else dict(new_partial),
        "previous_checkpoint_sha256": current["checkpoint_sha256"],
        "run_binding": new_binding,
        "sealed_shards": [*current["sealed_shards"], *new_shards],
        "source_cursor": dict(new_cursor),
        "status": "running",
        "total_accepted_documents": int(current["total_accepted_documents"])
        + added_documents,
        "total_accepted_tokens": int(current["total_accepted_tokens"]) + added_tokens,
    }
    result.pop("checkpoint_sha256", None)
    if production_floors_complete(result, plan):
        result["status"] = "complete"
    result["checkpoint_sha256"] = checkpoint_fingerprint(result)
    return validate_production_checkpoint(result, plan)


def build_application_request(
    plan: Mapping[str, Any],
    *,
    maximum_vcpu: int,
    image_uri: str | None = None,
) -> dict[str, Any]:
    """Render an EMR Serverless application request without creating it."""

    if maximum_vcpu not in {
        int(plan["application"]["bootstrap_maximum_vcpu"]),
        int(plan["application"]["ramp_maximum_vcpu"]),
        int(plan["application"]["production_maximum_vcpu"]),
    }:
        raise ValueError(
            "EMR production vCPU must be the pinned bootstrap, ramp, or production ceiling"
        )
    shape = _resource_shape(plan["application"], maximum_vcpu)
    request = {
        "architecture": plan["application"]["architecture"],
        "autoStartConfiguration": {"enabled": True},
        "autoStopConfiguration": {
            "enabled": True,
            "idleTimeoutMinutes": plan["application"]["auto_stop_idle_timeout_minutes"],
        },
        "maximumCapacity": {
            "cpu": f"{shape['vcpu']} vCPU",
            "disk": f"{shape['disk_gib']} GB",
            "memory": f"{shape['memory_gib']} GB",
        },
        "name": f"lumi-production-corpus-{shape['vcpu']}vcpu",
        "releaseLabel": plan["runtime"]["release_label"],
        "type": "SPARK",
    }
    if image_uri is not None:
        pattern = (
            r"[0-9]{12}\.dkr\.ecr\.us-east-1\.amazonaws\.com/"
            r"[a-z0-9._/-]+@sha256:[0-9a-f]{64}"
        )
        if re.fullmatch(pattern, image_uri) is None:
            raise ValueError("production EMR image must be a digest-bound us-east-1 ECR URI")
        request["imageConfiguration"] = {"imageUri": image_uri}
    return request


def validate_paid_ramp_authorization(
    authorization: Mapping[str, Any],
    plan: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a future exact paid-ramp authorization; never infer it from the plan."""

    validate_production_checkpoint(checkpoint, plan)
    if authorization.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported production-ramp authorization schema")
    if authorization.get("gate") != "lumi_emr_production_ramp_authorization_v0.1":
        raise ValueError("unexpected production-ramp authorization gate")
    if authorization.get("approval_text") != PRODUCTION_APPROVAL_TEXT:
        raise ValueError("production-ramp approval text changed")
    if authorization.get("owner") != "Afnan Technologies Inc.":
        raise ValueError("production-ramp owner changed")
    if authorization.get("plan_fingerprint") != plan["plan_fingerprint"]:
        raise ValueError("production-ramp authorization targets another plan")
    if authorization.get("checkpoint_sha256") != checkpoint["checkpoint_sha256"]:
        raise ValueError("production-ramp authorization targets another checkpoint")
    permissions = authorization.get("authorization", {})
    if permissions != {
        "full_4b_training": False,
        "model_training": False,
        "paid_cpu_stager": True,
        "paid_emr_production": True,
        "paid_gpu_compute": False,
        "production_adaptive_acquisition": True,
    }:
        raise ValueError("production-ramp authorization boundary changed")
    bounds = authorization.get("bounds", {})
    maximum_vcpu = bounds.get("maximum_emr_vcpu")
    if maximum_vcpu not in {
        int(plan["application"]["bootstrap_maximum_vcpu"]),
        int(plan["application"]["ramp_maximum_vcpu"]),
    }:
        raise ValueError("initial production ramp must be capped at 16 or 64 vCPU")
    if bounds.get("maximum_source_files") != 1:
        raise ValueError("initial production ramp must process one retained source")
    maximum_hours = Decimal(str(bounds.get("maximum_runtime_hours")))
    maximum_usd = Decimal(str(bounds.get("maximum_compute_usd")))
    if maximum_hours <= 0 or maximum_hours > Decimal("2"):
        raise ValueError("production ramp runtime must be in (0, 2] hours")
    cost_key = "bootstrap_usd" if maximum_vcpu == 16 else "ramp_usd"
    forecast = Decimal(plan["cost"][cost_key]) * maximum_hours
    if maximum_usd < forecast or maximum_usd > Decimal("50"):
        raise ValueError("production ramp cost bound is unsafe or insufficient")
    run_token = authorization.get("run_token")
    if not isinstance(run_token, str) or re.fullmatch(r"[0-9a-f]{32}", run_token) is None:
        raise ValueError("production ramp run token is invalid")
    return dict(authorization)


def validate_paid_production_authorization(
    authorization: Mapping[str, Any],
    plan: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a future full 128-vCPU run without granting that permission."""

    validate_production_checkpoint(checkpoint, plan)
    if authorization.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported production authorization schema")
    if authorization.get("gate") != "lumi_emr_production_authorization_v0.1":
        raise ValueError("unexpected production authorization gate")
    if authorization.get("approval_text") != PRODUCTION_RUN_APPROVAL_TEXT:
        raise ValueError("production approval text changed")
    if authorization.get("owner") != "Afnan Technologies Inc.":
        raise ValueError("production owner changed")
    if authorization.get("plan_fingerprint") != plan["plan_fingerprint"]:
        raise ValueError("production authorization targets another plan")
    root_checkpoint = authorization.get("checkpoint_sha256")
    _sha256(root_checkpoint, "production authorization root checkpoint")
    expected_binding = {
        "gate": "lumi_emr_production_run_binding_v0.1",
        "plan_fingerprint": plan["plan_fingerprint"],
        "root_checkpoint_sha256": root_checkpoint,
        "run_token": authorization.get("run_token"),
    }
    if checkpoint["checkpoint_sha256"] == root_checkpoint:
        if (
            checkpoint.get("run_binding") is not None
            and checkpoint.get("run_binding") != expected_binding
        ):
            raise ValueError("production root checkpoint has another run binding")
    elif checkpoint.get("run_binding") != expected_binding:
        raise ValueError("production checkpoint is not a descendant of the authorized root")
    permissions = authorization.get("authorization", {})
    if permissions != {
        "full_4b_training": False,
        "model_training": False,
        "paid_cpu_stager": True,
        "paid_emr_production": True,
        "paid_gpu_compute": False,
        "production_adaptive_acquisition": True,
    }:
        raise ValueError("production authorization boundary changed")
    bounds = authorization.get("bounds", {})
    if bounds.get("maximum_emr_vcpu") != int(
        plan["application"]["production_maximum_vcpu"]
    ):
        raise ValueError("production authorization must use the pinned 128-vCPU ceiling")
    maximum_sources = _positive_int(
        bounds.get("maximum_source_files"), "production source-file cap"
    )
    if maximum_sources > len(plan["source_schedule"]):
        raise ValueError("production authorization exceeds the pinned source inventory")
    cumulative_hours = Decimal(str(bounds.get("maximum_cumulative_emr_hours")))
    job_hours = Decimal(str(bounds.get("maximum_job_runtime_hours")))
    if cumulative_hours <= 0 or cumulative_hours > Decimal("96"):
        raise ValueError("production cumulative runtime must be in (0, 96] hours")
    if job_hours <= 0 or job_hours > Decimal("24") or job_hours > cumulative_hours:
        raise ValueError("production per-job runtime must be in (0, 24] hours")
    compute_cap = Decimal(str(bounds.get("maximum_compute_usd")))
    total_cap = Decimal(str(bounds.get("maximum_total_usd")))
    forecast = Decimal(plan["cost"]["production_usd"]) * cumulative_hours
    budget = plan["budget"]
    if compute_cap < forecast or compute_cap > Decimal(
        budget["maximum_emr_compute_usd"]
    ):
        raise ValueError("production EMR compute cap is unsafe or exceeds its allocation")
    if total_cap != Decimal(budget["hard_total_usd"]):
        raise ValueError("production total cost must equal the recorded hard cap")
    run_token = authorization.get("run_token")
    if not isinstance(run_token, str) or re.fullmatch(r"[0-9a-f]{32}", run_token) is None:
        raise ValueError("production run token is invalid")
    return dict(authorization)


def validate_paid_source_authorization(
    authorization: Mapping[str, Any],
    plan: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Dispatch to the exact ramp or production authorization contract."""

    if authorization.get("gate") == "lumi_emr_production_ramp_authorization_v0.1":
        return validate_paid_ramp_authorization(authorization, plan, checkpoint)
    if authorization.get("gate") == "lumi_emr_production_authorization_v0.1":
        return validate_paid_production_authorization(authorization, plan, checkpoint)
    raise ValueError("unsupported paid production authorization gate")


def staging_maximum_chunks(
    authorization: Mapping[str, Any], plan: Mapping[str, Any]
) -> int | None:
    """Return the bounded slice size; production stages the remaining source once."""

    maximum_vcpu = int(authorization["bounds"]["maximum_emr_vcpu"])
    if authorization.get("gate") == "lumi_emr_production_authorization_v0.1":
        return None
    if maximum_vcpu == int(plan["application"]["bootstrap_maximum_vcpu"]):
        return int(plan["source_stager"]["bootstrap_maximum_chunks"])
    return int(plan["source_stager"]["ramp_maximum_chunks"])


def source_job_token(
    authorization: Mapping[str, Any], checkpoint: Mapping[str, Any]
) -> str:
    """Derive one stable AWS idempotency token per authorized checkpoint."""

    run_token = authorization.get("run_token")
    checkpoint_sha256 = checkpoint.get("checkpoint_sha256")
    if not isinstance(run_token, str) or re.fullmatch(r"[0-9a-f]{32}", run_token) is None:
        raise ValueError("production authorization run token is invalid")
    _sha256(checkpoint_sha256, "production source checkpoint")
    return _sha256_bytes(f"{run_token}:{checkpoint_sha256}".encode())[:32]


def _bounded_s3_uri(uri: Any, *, bucket: str) -> str:
    if not isinstance(uri, str):
        raise ValueError("production EMR artifact URI is invalid")
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or parsed.netloc != bucket or not parsed.path.strip("/"):
        raise ValueError("production EMR artifacts must stay in the production S3 bucket")
    return Path(parsed.path).name


def build_ramp_job_request(
    plan: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    authorization: Mapping[str, Any],
    *,
    application_id: str,
    execution_role_arn: str,
    plan_s3_uri: str,
    checkpoint_s3_uri: str,
    authorization_s3_uri: str,
    staged_source_s3_uri: str,
    code_archive_s3_uri: str,
    job_script_s3_uri: str,
) -> dict[str, Any]:
    """Render one exact 64-vCPU production ramp request without calling AWS."""

    validate_paid_source_authorization(authorization, plan, checkpoint)
    if re.fullmatch(r"[0-9a-z]+", application_id) is None:
        raise ValueError("production EMR application ID is invalid")
    if re.fullmatch(
        r"arn:aws:iam::[0-9]{12}:role/[A-Za-z0-9+=,.@_/-]+", execution_role_arn
    ) is None:
        raise ValueError("production EMR execution role ARN is invalid")
    bucket = str(plan["storage"]["bucket"])
    files = [
        plan_s3_uri,
        checkpoint_s3_uri,
        authorization_s3_uri,
        staged_source_s3_uri,
        f"s3://{bucket}/{plan['storage']['artifacts']['benchmark_registry']}",
        f"s3://{bucket}/{plan['storage']['artifacts']['frozen_tokenizer']}",
        f"s3://{bucket}/{plan['storage']['artifacts']['training_exclusion_ledger']}",
    ]
    file_names = [_bounded_s3_uri(uri, bucket=bucket) for uri in files]
    if len(file_names) != len(set(file_names)):
        raise ValueError("production EMR SparkFiles basenames collide")
    _bounded_s3_uri(code_archive_s3_uri, bucket=bucket)
    _bounded_s3_uri(job_script_s3_uri, bucket=bucket)
    maximum_vcpu = int(authorization["bounds"]["maximum_emr_vcpu"])
    shape = _resource_shape(plan["application"], maximum_vcpu)
    python = "/opt/lumi/python312/bin/python"
    partitions = shape["executor_instances"] * 16
    spark_parameters = [
        "--conf", "spark.driver.cores=4",
        "--conf", "spark.driver.memory=14g",
        "--conf", "spark.executor.cores=4",
        "--conf", "spark.executor.memory=14g",
        "--conf", f"spark.executor.instances={shape['executor_instances']}",
        "--conf", "spark.dynamicAllocation.enabled=false",
        "--conf", "spark.emr-serverless.driver.disk=20G",
        "--conf", "spark.emr-serverless.executor.disk=20G",
        "--conf", f"spark.sql.shuffle.partitions={partitions}",
        "--conf", f"spark.default.parallelism={partitions}",
        "--conf", "spark.network.crypto.enabled=true",
        "--conf", "spark.io.encryption.enabled=true",
        "--conf", f"spark.emr-serverless.driverEnv.PYSPARK_DRIVER_PYTHON={python}",
        "--conf", f"spark.emr-serverless.driverEnv.PYSPARK_PYTHON={python}",
        "--conf", f"spark.executorEnv.PYSPARK_PYTHON={python}",
        "--py-files", code_archive_s3_uri,
        "--files", ",".join(files),
    ]
    run_token = str(authorization["run_token"])
    job_token = source_job_token(authorization, checkpoint)
    runtime_key = (
        "maximum_job_runtime_hours"
        if authorization.get("gate") == "lumi_emr_production_authorization_v0.1"
        else "maximum_runtime_hours"
    )
    runtime_minutes = int(Decimal(str(authorization["bounds"][runtime_key])) * Decimal(60))
    return {
        "applicationId": application_id,
        "clientToken": f"lumi-production-source-{job_token}",
        "configurationOverrides": {
            "monitoringConfiguration": {
                "s3MonitoringConfiguration": {
                    "logUri": (
                        f"s3://{bucket}/{plan['storage']['namespace']}/control/distributed/"
                        f"production/runs/{job_token}/logs"
                    )
                }
            }
        },
        "executionRoleArn": execution_role_arn,
        "executionTimeoutMinutes": runtime_minutes,
        "jobDriver": {
            "sparkSubmit": {
                "entryPoint": job_script_s3_uri,
                "entryPointArguments": [
                    "--lumi-authorization-file", file_names[2],
                    "--lumi-checkpoint-file", file_names[1],
                    "--lumi-plan-file", file_names[0],
                    "--lumi-run-token", run_token,
                    "--lumi-job-token", job_token,
                    "--lumi-staged-source-file", file_names[3],
                ],
                "sparkSubmitParameters": " ".join(spark_parameters),
            }
        },
        "mode": "BATCH",
        "name": f"lumi-production-{maximum_vcpu}vcpu-{job_token[:12]}",
        "retryPolicy": {"maxAttempts": 1},
        "tags": {
            "LumiPlanFingerprint": str(plan["plan_fingerprint"]),
            "LumiScope": f"bounded-production-corpus-{maximum_vcpu}vcpu-ramp-only",
        },
    }
