"""Fail-closed planning for Lumi-4B/80B production Segment 2.

The Segment-1 checkpoint uses the original batch-1/gradient-checkpointed run
identity.  The exact-checkpoint gate proved that the same model and optimizer
state can continue with batch 2 and activation checkpointing disabled while
preserving the 256-sequence global batch.  This module binds that one measured
transition without authorizing paid work.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .distributed_training import (
    DISTRIBUTED_AUTHORIZATION_SCHEMA_VERSION,
    PAID_4B_PRODUCTION_MAX_COST_USD,
    PAID_4B_PRODUCTION_MAX_RUNTIME_MINUTES,
    PAID_4B_PRODUCTION_OWNER_ACKNOWLEDGEMENT,
    PAID_4B_PRODUCTION_RUN_KIND,
    PAID_4B_PRODUCTION_TOKEN_CAP,
    build_distributed_run_identity,
)
from .production_pretraining import validate_production_plan

SEGMENT2_CONTINUATION_SCHEMA_VERSION = 1
SEGMENT2_CONTINUATION_GATE = "lumi_4b_80b_segment2_continuation_plan_v0.1"
SEGMENT2_PROFILE = {
    "gradient_accumulation_steps": 16,
    "gradient_checkpointing": False,
    "label": "batch2_no_gc",
    "optimizer_profile": "legacy",
    "per_rank_batch_size": 2,
}
SEGMENT2_START_STEP = 31_500
SEGMENT2_START_TOKENS = 33_030_144_000
SEGMENT2_START_SEQUENCE_CURSOR = 8_064_000
SEGMENT2_STOP_STEP = 63_000
SEGMENT2_STOP_TOKENS = 66_060_288_000
SEGMENT2_INSTANCE_TYPE = "gpu_8x_h100_sxm5"
SEGMENT2_INSTANCE_IP = "192.222.52.81"
SEGMENT2_PROVIDER = "Lambda Cloud"
SEGMENT2_HOURLY_RATE_USD = 31.92
SEGMENT2_S3_BUCKET = "lumi-corpus-prod-27777c02b29b76efcac3"
SEGMENT2_CORPUS_SHARDS = 160
SEGMENT2_CORPUS_BYTES = 160_000_016_728
SEGMENT1_ARCHIVE_OBJECTS = 33
SEGMENT1_ARCHIVE_BYTES = 48_844_401_057
SEGMENT1_CHECKPOINT_MANIFEST_ENTRIES = 18
SEGMENT1_IMPORT_VERIFICATION_GATE = (
    "lumi_4b_80b_segment1_checkpoint_import_independent_verification_v0.1"
)
SEGMENT2_THROUGHPUT_SAFETY_FACTOR = 0.8
SEGMENT2_CHECKPOINT_RELATIVE = PurePosixPath(
    "training/checkpoint-00031500"
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _fingerprint(value: Mapping[str, Any], omitted: str | None = None) -> str:
    material = dict(value)
    if omitted is not None:
        material.pop(omitted, None)
    return hashlib.sha256(_canonical_json_bytes(material)).hexdigest()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} is not a JSON object")
    return value


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite sealed artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _candidate_from_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    if (
        report.get("status") != "pass"
        or report.get("selected_profile") != SEGMENT2_PROFILE["label"]
        or report.get("report_sha256") != _fingerprint(report, "report_sha256")
    ):
        raise ValueError("exact-checkpoint gate did not select batch2_no_gc")
    results = report.get("results")
    if not isinstance(results, list):
        raise ValueError("exact-checkpoint gate results are missing")
    candidates = [
        result
        for result in results
        if isinstance(result, dict)
        and result.get("label") == SEGMENT2_PROFILE["label"]
    ]
    if len(candidates) != 1:
        raise ValueError("exact-checkpoint gate has no unique batch2_no_gc result")
    candidate = candidates[0]
    if candidate.get("profile") != SEGMENT2_PROFILE:
        raise ValueError("exact-checkpoint batch2_no_gc profile changed")
    if any(
        candidate.get(field) is not True
        for field in (
            "checkpoint_restore_pass",
            "resume_canary_pass",
            "source_checkpoint_unchanged",
        )
    ) or candidate.get("status") != "pass":
        raise ValueError("exact-checkpoint batch2_no_gc safety evidence failed")
    for field in (
        "heldout_loss",
        "median_tokens_per_second",
        "p10_tokens_per_second",
        "training_loss",
    ):
        value = candidate.get(field)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError(f"exact-checkpoint candidate has invalid {field}")
    return candidate


def _verified_import_binding(
    report: Mapping[str, Any],
    *,
    import_report_sha256: str,
    archive_root: str,
) -> dict[str, Any]:
    checkpoint = report.get("checkpoint")
    if (
        report.get("schema_version") != 1
        or report.get("gate") != SEGMENT1_IMPORT_VERIFICATION_GATE
        or report.get("status") != "pass"
        or report.get("total_objects") != SEGMENT1_ARCHIVE_OBJECTS
        or report.get("total_bytes") != SEGMENT1_ARCHIVE_BYTES
        or report.get("manifest_entries_verified")
        != SEGMENT1_CHECKPOINT_MANIFEST_ENTRIES
        or report.get("verified_at_utc") is None
        or not isinstance(checkpoint, dict)
    ):
        raise ValueError("Segment-1 archive import is not independently verified")
    expected_checkpoint = {
        "sequence_cursor": SEGMENT2_START_SEQUENCE_CURSOR,
        "step": SEGMENT2_START_STEP,
        "tokens": SEGMENT2_START_TOKENS,
        "world_size": 8,
    }
    if checkpoint != expected_checkpoint:
        raise ValueError("imported Segment-1 checkpoint boundary changed")
    verified = report.get("verified_objects")
    if not isinstance(verified, list):
        raise ValueError("import verification inventory is missing")
    by_path = {
        item.get("relative_path"): item
        for item in verified
        if isinstance(item, dict)
    }
    manifest_entry = by_path.get(
        f"{SEGMENT2_CHECKPOINT_RELATIVE.as_posix()}/manifest.sha256"
    )
    gate_entry = by_path.get(
        "evidence/exact-checkpoint-gate/run-approved-001/"
        "verification-report.json"
    )
    if not isinstance(manifest_entry, dict) or not isinstance(gate_entry, dict):
        raise ValueError("import verification omitted checkpoint-gate evidence")
    checkpoint_manifest_sha256 = manifest_entry.get("sha256")
    if (
        not isinstance(checkpoint_manifest_sha256, str)
        or len(checkpoint_manifest_sha256) != 64
    ):
        raise ValueError("imported checkpoint manifest binding is invalid")
    handoff_sha256 = report.get("handoff_sha256")
    if not isinstance(handoff_sha256, str) or len(handoff_sha256) != 64:
        raise ValueError("imported handoff binding is invalid")
    archive = PurePosixPath(archive_root)
    if not archive.is_absolute() or ".." in archive.parts:
        raise ValueError("imported archive root must be an absolute safe path")
    if report.get("archive") != archive.as_posix():
        raise ValueError("import verification targets another archive")
    for field in ("handoff_sha256", "plan_sha256"):
        digest = report.get(field)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"import verification {field} is invalid")
    return {
        "archive_root": archive.as_posix(),
        "checkpoint_path": (archive / SEGMENT2_CHECKPOINT_RELATIVE).as_posix(),
        "gate_report_file_sha256": gate_entry["sha256"],
        "handoff_sha256": handoff_sha256,
        "import_plan_sha256": report["plan_sha256"],
        "import_verification_sha256": import_report_sha256,
        "manifest_sha256": checkpoint_manifest_sha256,
        "read_only": True,
        "sequence_cursor": SEGMENT2_START_SEQUENCE_CURSOR,
        "step": SEGMENT2_START_STEP,
        "tokens": SEGMENT2_START_TOKENS,
        "world_size": 8,
    }


def build_segment2_continuation_plan(
    *,
    base_plan_path: str | Path,
    checkpoint_gate_report_path: str | Path,
    import_verification_path: str | Path,
    config_path: str | Path,
    corpus_manifest_path: str | Path,
    imported_archive_root: str,
    source_commit: str,
) -> dict[str, Any]:
    """Build a planning-only Segment-2 continuation bound to one checkpoint."""

    base_plan = validate_production_plan(_load_object(base_plan_path))
    source_commit = source_commit.strip().lower()
    if len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise ValueError("Segment-2 source commit must be a full Git SHA-1")
    segment = base_plan["segments"][1]
    if (
        segment.get("index"),
        segment.get("start_step"),
        segment.get("start_tokens"),
        segment.get("stop_step"),
        segment.get("stop_tokens"),
    ) != (
        2,
        SEGMENT2_START_STEP,
        SEGMENT2_START_TOKENS,
        SEGMENT2_STOP_STEP,
        SEGMENT2_STOP_TOKENS,
    ):
        raise ValueError("base production Segment-2 boundary changed")

    gate_report = _load_object(checkpoint_gate_report_path)
    gate_report_file_sha256 = _sha256(checkpoint_gate_report_path)
    candidate = _candidate_from_gate(gate_report)
    import_report = _load_object(import_verification_path)
    import_report_sha256 = _sha256(import_verification_path)
    source_checkpoint = _verified_import_binding(
        import_report,
        import_report_sha256=import_report_sha256,
        archive_root=imported_archive_root,
    )
    gate_source = gate_report.get("source_checkpoint")
    expected_gate_source = {
        "manifest_sha256": source_checkpoint["manifest_sha256"],
        "run_fingerprint": base_plan["run_fingerprint"],
        "sequence_cursor": SEGMENT2_START_SEQUENCE_CURSOR,
        "step": SEGMENT2_START_STEP,
        "tokens": SEGMENT2_START_TOKENS,
    }
    if not isinstance(gate_source, dict) or {
        key: gate_source.get(key) for key in expected_gate_source
    } != expected_gate_source:
        raise ValueError("gate and imported checkpoint bindings differ")
    if source_checkpoint["gate_report_file_sha256"] != gate_report_file_sha256:
        raise ValueError("imported and local checkpoint-gate reports differ")
    source_checkpoint["run_fingerprint"] = base_plan["run_fingerprint"]

    config_path = Path(config_path)
    corpus_manifest_path = Path(corpus_manifest_path)
    config_bytes = config_path.read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != base_plan["model"]["config_sha256"]:
        raise ValueError("Segment-2 model configuration changed")
    corpus_manifest = _load_object(corpus_manifest_path)
    corpus_manifest_sha256 = _sha256(corpus_manifest_path)
    if (
        corpus_manifest_sha256 != base_plan["corpus"]["manifest_sha256"]
        or len(corpus_manifest.get("shards", [])) != SEGMENT2_CORPUS_SHARDS
    ):
        raise ValueError("Segment-2 corpus manifest changed")

    run = {
        **base_plan["run"],
        "gradient_accumulation_steps": SEGMENT2_PROFILE[
            "gradient_accumulation_steps"
        ],
        "gradient_checkpointing": SEGMENT2_PROFILE["gradient_checkpointing"],
        "per_rank_batch_size": SEGMENT2_PROFILE["per_rank_batch_size"],
    }
    run_identity = build_distributed_run_identity(
        config_bytes=config_bytes,
        manifest={"build_fingerprint": base_plan["corpus"]["adapter_build_fingerprint"]},
        manifest_sha256=corpus_manifest_sha256,
        backend="nccl",
        world_size=run["world_size"],
        strategy=run["strategy"],
        per_rank_batch_size=run["per_rank_batch_size"],
        gradient_accumulation_steps=run["gradient_accumulation_steps"],
        learning_rate=run["learning_rate"],
        weight_decay=run["weight_decay"],
        max_grad_norm=run["max_grad_norm"],
        keep_checkpoints=run["keep_checkpoints"],
        checkpoint_steps=run["checkpoint_steps"],
        seed=run["seed"],
        gradient_checkpointing=run["gradient_checkpointing"],
        lr_schedule=run["lr_schedule"],
        warmup_steps=run["warmup_steps"],
        schedule_steps=run["schedule_steps"],
        min_learning_rate_ratio=run["min_learning_rate_ratio"],
        attention_implementation=run["attention_implementation"],
        document_isolation=run["document_isolation"],
        optimizer_profile=run["optimizer_profile"],
        precision=run["precision"],
        gradient_sync_mode=run["gradient_sync_mode"],
        torch_compile=run["torch_compile"],
        source_commit=source_commit,
        torch_version=base_plan["software"]["torch_version"],
    )
    run_fingerprint = _fingerprint(run_identity)
    if run_fingerprint == base_plan["run_fingerprint"]:
        raise ValueError("Segment-2 continuation did not receive a new run identity")

    measured_rate = float(candidate["median_tokens_per_second"])
    conservative_rate = measured_rate * SEGMENT2_THROUGHPUT_SAFETY_FACTOR
    segment_tokens = SEGMENT2_STOP_TOKENS - SEGMENT2_START_TOKENS
    measured_hours = segment_tokens / measured_rate / 3600
    conservative_hours = segment_tokens / conservative_rate / 3600
    template = {
        "approved_from_step": SEGMENT2_START_STEP,
        "approved_from_tokens": SEGMENT2_START_TOKENS,
        "approved_through_step": SEGMENT2_STOP_STEP,
        "approved_token_cap": SEGMENT2_STOP_TOKENS,
        "continuation_source_checkpoint": source_checkpoint,
        "execution_profile": SEGMENT2_PROFILE,
        "full_4b_training": True,
        "max_cost_usd": float(PAID_4B_PRODUCTION_MAX_COST_USD),
        "max_runtime_minutes": PAID_4B_PRODUCTION_MAX_RUNTIME_MINUTES,
        "owner_acknowledgement": PAID_4B_PRODUCTION_OWNER_ACKNOWLEDGEMENT,
        "paid_gpu": False,
        "production_contract_sha256": base_plan["contract"]["sha256"],
        "run_fingerprint": run_fingerprint,
        "run_kind": PAID_4B_PRODUCTION_RUN_KIND,
        "schema_version": DISTRIBUTED_AUTHORIZATION_SCHEMA_VERSION,
        "segment_count": 3,
        "segment_index": 2,
        "status": "planning_only",
        "strategy": "fsdp_full_shard",
        "total_token_cap": PAID_4B_PRODUCTION_TOKEN_CAP,
        "world_size": 8,
    }
    plan: dict[str, Any] = {
        "approval": {
            "full_4b_training": False,
            "paid_gpu": False,
        },
        "authorization_template": template,
        "base_production_plan": {
            "plan_sha256": base_plan["plan_sha256"],
            "run_fingerprint": base_plan["run_fingerprint"],
            "source_commit": base_plan["source_commit"],
        },
        "budget": {
            "conservative_tokens_per_second": conservative_rate,
            "estimated_conservative_compute_cost_usd": (
                conservative_hours * SEGMENT2_HOURLY_RATE_USD
            ),
            "estimated_conservative_training_hours": conservative_hours,
            "estimated_measured_compute_cost_usd": (
                measured_hours * SEGMENT2_HOURLY_RATE_USD
            ),
            "estimated_measured_training_hours": measured_hours,
            "hourly_rate_usd": SEGMENT2_HOURLY_RATE_USD,
            "instance_type": SEGMENT2_INSTANCE_TYPE,
            "maximum_cost_usd": float(PAID_4B_PRODUCTION_MAX_COST_USD),
            "maximum_runtime_minutes": PAID_4B_PRODUCTION_MAX_RUNTIME_MINUTES,
            "provider": SEGMENT2_PROVIDER,
            "throughput_safety_factor": SEGMENT2_THROUGHPUT_SAFETY_FACTOR,
        },
        "checkpoint_gate": {
            "authorization_sha256": gate_report["authorization_sha256"],
            "file_sha256": gate_report_file_sha256,
            "median_throughput_gain_fraction": gate_report["comparisons"][
                "median_throughput_gain_fraction"
            ],
            "plan_sha256": gate_report["plan_sha256"],
            "report_sha256": gate_report["report_sha256"],
            "selected_profile": gate_report["selected_profile"],
        },
        "corpus": {
            "bytes": SEGMENT2_CORPUS_BYTES,
            "manifest_sha256": corpus_manifest_sha256,
            "presigned_get_expiry_seconds": 21_600,
            "s3_objects": SEGMENT2_CORPUS_SHARDS,
            "staging_status": "required_not_authorized",
        },
        "created_at": _utc_now(),
        "execution_profile": SEGMENT2_PROFILE,
        "gate": SEGMENT2_CONTINUATION_GATE,
        "run": run,
        "run_fingerprint": run_fingerprint,
        "run_identity": run_identity,
        "schema_version": SEGMENT2_CONTINUATION_SCHEMA_VERSION,
        "segment": {
            "index": 2,
            "start_step": SEGMENT2_START_STEP,
            "start_tokens": SEGMENT2_START_TOKENS,
            "stop_step": SEGMENT2_STOP_STEP,
            "stop_tokens": SEGMENT2_STOP_TOKENS,
        },
        "source_checkpoint": source_checkpoint,
        "source_commit": source_commit,
        "status": "planning_only_no_paid_gpu",
    }
    plan["plan_sha256"] = _fingerprint(plan)
    return validate_segment2_continuation_plan(plan)


def validate_segment2_continuation_plan(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail if a serialized Segment-2 continuation plan was altered."""

    if (
        plan.get("schema_version") != SEGMENT2_CONTINUATION_SCHEMA_VERSION
        or plan.get("gate") != SEGMENT2_CONTINUATION_GATE
        or plan.get("status") != "planning_only_no_paid_gpu"
    ):
        raise ValueError("unsupported Segment-2 continuation plan")
    if plan.get("approval") != {
        "full_4b_training": False,
        "paid_gpu": False,
    }:
        raise ValueError("Segment-2 continuation plan cannot authorize paid work")
    if plan.get("plan_sha256") != _fingerprint(plan, "plan_sha256"):
        raise ValueError("Segment-2 continuation plan hash changed")
    if plan.get("execution_profile") != SEGMENT2_PROFILE:
        raise ValueError("Segment-2 selected execution profile changed")
    if plan.get("segment") != {
        "index": 2,
        "start_step": SEGMENT2_START_STEP,
        "start_tokens": SEGMENT2_START_TOKENS,
        "stop_step": SEGMENT2_STOP_STEP,
        "stop_tokens": SEGMENT2_STOP_TOKENS,
    }:
        raise ValueError("Segment-2 continuation boundary changed")
    template = plan.get("authorization_template")
    if (
        not isinstance(template, dict)
        or template.get("paid_gpu") is not False
        or template.get("status") != "planning_only"
        or template.get("run_fingerprint") != plan.get("run_fingerprint")
        or template.get("continuation_source_checkpoint")
        != plan.get("source_checkpoint")
    ):
        raise ValueError("Segment-2 authorization template changed")
    corpus = plan.get("corpus")
    if not isinstance(corpus, dict) or (
        corpus.get("s3_objects"),
        corpus.get("bytes"),
        corpus.get("staging_status"),
    ) != (
        SEGMENT2_CORPUS_SHARDS,
        SEGMENT2_CORPUS_BYTES,
        "required_not_authorized",
    ):
        raise ValueError("Segment-2 corpus staging boundary changed")
    return dict(plan)


def owner_approval_text(
    plan: Mapping[str, Any],
    *,
    instance_ip: str,
    s3_bucket: str,
) -> str:
    """Return the one exact owner sentence required to stage and train Segment 2."""

    validated = validate_segment2_continuation_plan(plan)
    if instance_ip != SEGMENT2_INSTANCE_IP or s3_bucket != SEGMENT2_S3_BUCKET:
        raise ValueError("Segment-2 instance or S3 bucket changed")
    source = validated["source_checkpoint"]
    return (
        "I approve exactly one Lumi-4B/80B production Segment 2 on Lambda "
        f"instance {instance_ip} ({SEGMENT2_INSTANCE_TYPE}), using source commit "
        f"{validated['source_commit']}, run fingerprint "
        f"{validated['run_fingerprint']}, and sealed Segment-1 handoff "
        f"{source['handoff_sha256']}; steps {SEGMENT2_START_STEP}→"
        f"{SEGMENT2_STOP_STEP}, tokens {SEGMENT2_START_TOKENS}→"
        f"{SEGMENT2_STOP_TOKENS}, maximum {PAID_4B_PRODUCTION_MAX_RUNTIME_MINUTES} "
        f"minutes/${PAID_4B_PRODUCTION_MAX_COST_USD:,}, "
        "full_4b_training=true. I also approve downloading only the exact "
        f"{SEGMENT2_CORPUS_SHARDS} sealed corpus objects ({SEGMENT2_CORPUS_BYTES} "
        f"bytes) from private S3 bucket {s3_bucket} to that instance using "
        "object-scoped six-hour presigned GET URLs. No AWS credentials on Lambda, "
        "no other S3 objects, no Segment 3, no new instance, and no instance "
        "termination."
    )


def write_segment2_owner_authorization(
    *,
    plan_path: str | Path,
    output: str | Path,
    exact_owner_approval: str,
    instance_ip: str,
    s3_bucket: str,
) -> dict[str, Any]:
    """Write one exact Segment-2 artifact after the owner repeats the request."""

    plan = validate_segment2_continuation_plan(_load_object(plan_path))
    expected = owner_approval_text(
        plan,
        instance_ip=instance_ip,
        s3_bucket=s3_bucket,
    )
    if exact_owner_approval.strip() != expected:
        raise PermissionError("exact Segment-2 owner approval text does not match")
    authorization = {
        **plan["authorization_template"],
        "approved_at": _utc_now(),
        "exact_owner_approval": expected,
        "exact_owner_approval_sha256": hashlib.sha256(
            expected.encode("utf-8")
        ).hexdigest(),
        "hourly_rate_usd": SEGMENT2_HOURLY_RATE_USD,
        "instance_ip": instance_ip,
        "instance_type": SEGMENT2_INSTANCE_TYPE,
        "max_cost_usd": float(PAID_4B_PRODUCTION_MAX_COST_USD),
        "max_runtime_minutes": PAID_4B_PRODUCTION_MAX_RUNTIME_MINUTES,
        "paid_gpu": True,
        "production_plan_sha256": plan["plan_sha256"],
        "provider": SEGMENT2_PROVIDER,
        "s3_corpus_access": {
            "bucket": s3_bucket,
            "bytes": SEGMENT2_CORPUS_BYTES,
            "credentials_on_instance": False,
            "objects": SEGMENT2_CORPUS_SHARDS,
            "presigned_get_expiry_seconds": 21_600,
        },
        "source_commit": plan["source_commit"],
        "status": "owner_approved",
    }
    authorization["authorization_sha256"] = _fingerprint(authorization)
    _write_json_exclusive(Path(output), authorization)
    return authorization
