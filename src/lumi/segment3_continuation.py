"""Fail-closed planning for the final Lumi-4B/80B production segment."""

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
    PAID_4B_PRODUCTION_OWNER_ACKNOWLEDGEMENT,
    PAID_4B_PRODUCTION_RUN_KIND,
    PAID_4B_PRODUCTION_TOKEN_CAP,
    PAID_4B_SEGMENT3_INSTANCE_IP,
    PAID_4B_SEGMENT3_PROFILE,
    PAID_4B_SEGMENT3_START_SEQUENCE_CURSOR,
    PAID_4B_SEGMENT3_START_STEP,
    PAID_4B_SEGMENT3_START_TOKENS,
    PAID_4B_SEGMENT3_STOP_STEP,
    PAID_4B_SEGMENT3_STOP_TOKENS,
    build_distributed_run_identity,
)
from .production_pretraining import validate_production_plan

SEGMENT3_CONTINUATION_SCHEMA_VERSION = 1
SEGMENT3_CONTINUATION_GATE = "lumi_4b_80b_segment3_continuation_plan_v0.1"
SEGMENT3_CHECKPOINT_GATE = "lumi_4b_pre_segment3_checkpoint_gate_v0.1"
SEGMENT2_HANDOFF_GATE = "lumi_4b_checkpoint_handoff_v0.1"
SEGMENT2_RUN_FINGERPRINT = (
    "697be63c99ad3ef17387570a70ecc3e77ee1bdffe3b5f9806b535c55f8c3af7c"
)
SEGMENT2_SOURCE_COMMIT = "3a19884376b7ec07b4d89019c90eab47e5c6217e"
SEGMENT2_CHECKPOINT_MANIFEST_SHA256 = (
    "3f329c660dee5e381eaa75b0d9b9088bcf87054afdc26f9f8c16c3ab1f7b3344"
)
SEGMENT2_HANDOFF_SHA256 = (
    "017805ac78851a84745a98d9f7d76533352da03f44df2c636d51ead1e6f45ae3"
)
SEGMENT3_INSTANCE_TYPE = "gpu_8x_h100_sxm5"
SEGMENT3_PROVIDER = "Lambda Cloud"
SEGMENT3_HOURLY_RATE_USD = 31.92
SEGMENT3_MAX_RUNTIME_MINUTES = 3_600
SEGMENT3_MAX_COST_USD = 1_915.20
SEGMENT3_THROUGHPUT_SAFETY_FACTOR = 0.8
SEGMENT3_CORPUS_OBJECTS = 160
SEGMENT3_CORPUS_BYTES = 160_000_016_728
SEGMENT3_CORPUS_MANIFEST_SHA256 = (
    "6de45464af33d44e1261a023ccb7d50f86346a4126bb2fa060530a91213483e9"
)
SEGMENT3_TOKENS = (
    PAID_4B_SEGMENT3_STOP_TOKENS - PAID_4B_SEGMENT3_START_TOKENS
)
SEGMENT3_LOCAL_REUSE = {
    "aws_credentials_on_instance": False,
    "corpus_bytes": SEGMENT3_CORPUS_BYTES,
    "corpus_manifest_sha256": SEGMENT3_CORPUS_MANIFEST_SHA256,
    "corpus_objects": SEGMENT3_CORPUS_OBJECTS,
    "s3_access": False,
}
SEGMENT3_LOCAL_INPUT_PATHS = {
    "corpus_manifest": (
        "/home/ubuntu/lumi-runs/lumi-4b-80b-segment-2/"
        "data/final-80b/manifest.json"
    ),
    "corpus_root": (
        "/home/ubuntu/lumi-runs/lumi-4b-80b-segment-2/"
        "data/final-80b/shards"
    ),
    "fa3_runtime": (
        "/home/ubuntu/lumi-segment3-checkpoint-gate/"
        "fa3-v2.8.3-060c9188/runtime"
    ),
    "heldout_manifest": (
        "/home/ubuntu/lumi-runs/lumi-4b-80b-segment-2/"
        "data/heldout/manifest.json"
    ),
    "tokenizer": (
        "/home/ubuntu/lumi-runs/lumi-4b-80b-segment-2/data/tokenizer.json"
    ),
}


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


def _valid_digest(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} is not a lowercase SHA-256")
    return value


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


def _validate_gate_report(report: Mapping[str, Any]) -> dict[str, Any]:
    if (
        report.get("schema_version") != 1
        or report.get("gate") != SEGMENT3_CHECKPOINT_GATE
        or report.get("status") != "pass"
        or report.get("production_training_authorized") is not False
        or report.get("selected_profile") != PAID_4B_SEGMENT3_PROFILE["label"]
        or report.get("reasons") != []
        or report.get("report_sha256") != _fingerprint(report, "report_sha256")
    ):
        raise ValueError("exact Segment-2 checkpoint gate is not promotable")
    source = report.get("source_checkpoint")
    expected_source = {
        "manifest_sha256": SEGMENT2_CHECKPOINT_MANIFEST_SHA256,
        "run_fingerprint": SEGMENT2_RUN_FINGERPRINT,
        "sequence_cursor": PAID_4B_SEGMENT3_START_SEQUENCE_CURSOR,
        "step": PAID_4B_SEGMENT3_START_STEP,
        "tokens": PAID_4B_SEGMENT3_START_TOKENS,
    }
    if not isinstance(source, dict) or {
        key: source.get(key) for key in expected_source
    } != expected_source or source.get("read_only") is not True:
        raise ValueError("checkpoint gate source boundary changed")
    comparisons = report.get("comparisons")
    if not isinstance(comparisons, dict) or (
        float(comparisons.get("median_throughput_gain_fraction", -1)) < 0.15
        or float(comparisons.get("p10_throughput_gain_fraction", -1)) < 0.10
        or float(comparisons.get("training_loss_regression_fraction", 1)) > 0.01
        or float(comparisons.get("heldout_loss_regression_fraction", 1)) > 0.005
    ):
        raise ValueError("checkpoint gate promotion thresholds failed")
    if report.get("fa3_parity", {}).get("status") != "pass":
        raise ValueError("checkpoint gate FA3 parity failed")
    results = report.get("results")
    if not isinstance(results, list):
        raise ValueError("checkpoint gate results are missing")
    candidates = [
        result
        for result in results
        if isinstance(result, dict)
        and result.get("label") == PAID_4B_SEGMENT3_PROFILE["label"]
    ]
    if len(candidates) != 1:
        raise ValueError("checkpoint gate has no unique selected profile")
    candidate = candidates[0]
    if candidate.get("profile") != PAID_4B_SEGMENT3_PROFILE:
        raise ValueError("checkpoint gate selected profile changed")
    if candidate.get("status") != "pass" or any(
        candidate.get(field) is not True
        for field in (
            "checkpoint_restore_pass",
            "resume_canary_pass",
            "source_checkpoint_unchanged",
        )
    ):
        raise ValueError("checkpoint gate safety checks failed")
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
            raise ValueError(f"checkpoint gate candidate has invalid {field}")
    peak_memory = candidate.get("peak_memory_bytes")
    if (
        not isinstance(peak_memory, int)
        or isinstance(peak_memory, bool)
        or not 0 < peak_memory <= 76_000_000_000
    ):
        raise ValueError("checkpoint gate HBM safety failed")
    return dict(candidate)


def _validate_handoff(handoff: Mapping[str, Any]) -> dict[str, Any]:
    if (
        handoff.get("schema_version") != 1
        or handoff.get("gate") != SEGMENT2_HANDOFF_GATE
        or handoff.get("status") != "sealed"
        or handoff.get("handoff_sha256") != SEGMENT2_HANDOFF_SHA256
        or handoff.get("handoff_sha256")
        != _fingerprint(handoff, "handoff_sha256")
        or handoff.get("run_fingerprint") != SEGMENT2_RUN_FINGERPRINT
        or handoff.get("source_commit") != SEGMENT2_SOURCE_COMMIT
        or handoff.get("world_size") != 8
    ):
        raise ValueError("Segment-2 checkpoint handoff changed")
    checkpoint = handoff.get("checkpoint")
    if checkpoint != {
        "directory": "checkpoint-00063000",
        "sequence_cursor": PAID_4B_SEGMENT3_START_SEQUENCE_CURSOR,
        "step": PAID_4B_SEGMENT3_START_STEP,
        "tokens_seen": PAID_4B_SEGMENT3_START_TOKENS,
    }:
        raise ValueError("Segment-2 handoff boundary changed")
    files = handoff.get("files")
    if not isinstance(files, list) or len(files) != 19:
        raise ValueError("Segment-2 handoff inventory changed")
    by_path = {
        row.get("path"): row for row in files if isinstance(row, dict)
    }
    manifest = by_path.get("manifest.sha256")
    if (
        not isinstance(manifest, dict)
        or manifest.get("sha256") != SEGMENT2_CHECKPOINT_MANIFEST_SHA256
    ):
        raise ValueError("Segment-2 handoff manifest binding changed")
    return dict(handoff)


def build_segment3_continuation_plan(
    *,
    base_plan_path: str | Path,
    checkpoint_gate_report_path: str | Path,
    segment2_handoff_path: str | Path,
    config_path: str | Path,
    corpus_manifest_path: str | Path,
    checkpoint_path: str,
    source_commit: str,
) -> dict[str, Any]:
    """Build a planning-only final continuation bound to the retained instance."""

    base_plan = validate_production_plan(_load_object(base_plan_path))
    source_commit = source_commit.strip().lower()
    if len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise ValueError("Segment-3 source commit must be a full Git SHA-1")
    segment = base_plan["segments"][2]
    if (
        segment.get("index"),
        segment.get("start_step"),
        segment.get("start_tokens"),
        segment.get("stop_step"),
        segment.get("stop_tokens"),
    ) != (
        3,
        PAID_4B_SEGMENT3_START_STEP,
        PAID_4B_SEGMENT3_START_TOKENS,
        PAID_4B_SEGMENT3_STOP_STEP,
        PAID_4B_SEGMENT3_STOP_TOKENS,
    ):
        raise ValueError("base production Segment-3 boundary changed")

    gate_report = _load_object(checkpoint_gate_report_path)
    candidate = _validate_gate_report(gate_report)
    gate_file_sha256 = _sha256(checkpoint_gate_report_path)
    handoff = _validate_handoff(_load_object(segment2_handoff_path))
    handoff_file_sha256 = _sha256(segment2_handoff_path)

    checkpoint = PurePosixPath(checkpoint_path)
    if not checkpoint.is_absolute() or ".." in checkpoint.parts:
        raise ValueError("Segment-3 checkpoint path must be absolute and safe")
    if checkpoint.name != "checkpoint-00063000":
        raise ValueError("Segment-3 checkpoint path changed")
    source_checkpoint = {
        "checkpoint_gate_file_sha256": gate_file_sha256,
        "checkpoint_gate_report_sha256": gate_report["report_sha256"],
        "checkpoint_path": checkpoint.as_posix(),
        "handoff_file_sha256": handoff_file_sha256,
        "handoff_sha256": handoff["handoff_sha256"],
        "manifest_sha256": SEGMENT2_CHECKPOINT_MANIFEST_SHA256,
        "read_only": True,
        "run_fingerprint": SEGMENT2_RUN_FINGERPRINT,
        "sequence_cursor": PAID_4B_SEGMENT3_START_SEQUENCE_CURSOR,
        "step": PAID_4B_SEGMENT3_START_STEP,
        "tokens": PAID_4B_SEGMENT3_START_TOKENS,
        "world_size": 8,
    }

    config_path = Path(config_path)
    corpus_manifest_path = Path(corpus_manifest_path)
    config_bytes = config_path.read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != base_plan["model"][
        "config_sha256"
    ]:
        raise ValueError("Segment-3 model configuration changed")
    corpus_manifest_sha256 = _sha256(corpus_manifest_path)
    corpus_manifest = _load_object(corpus_manifest_path)
    if (
        corpus_manifest_sha256 != SEGMENT3_CORPUS_MANIFEST_SHA256
        or corpus_manifest_sha256 != base_plan["corpus"]["manifest_sha256"]
        or len(corpus_manifest.get("shards", [])) != SEGMENT3_CORPUS_OBJECTS
    ):
        raise ValueError("Segment-3 corpus manifest changed")

    run = {
        **base_plan["run"],
        "attention_implementation": PAID_4B_SEGMENT3_PROFILE[
            "attention_implementation"
        ],
        "gradient_accumulation_steps": PAID_4B_SEGMENT3_PROFILE[
            "gradient_accumulation_steps"
        ],
        "gradient_checkpointing": PAID_4B_SEGMENT3_PROFILE[
            "gradient_checkpointing"
        ],
        "per_rank_batch_size": PAID_4B_SEGMENT3_PROFILE[
            "per_rank_batch_size"
        ],
        "precision": PAID_4B_SEGMENT3_PROFILE["precision"],
        "strategy": PAID_4B_SEGMENT3_PROFILE["strategy"],
        "torch_compile": PAID_4B_SEGMENT3_PROFILE["torch_compile"],
    }
    run_identity = build_distributed_run_identity(
        config_bytes=config_bytes,
        manifest={
            "build_fingerprint": base_plan["corpus"][
                "adapter_build_fingerprint"
            ]
        },
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
    measured_rate = float(candidate["median_tokens_per_second"])
    conservative_rate = measured_rate * SEGMENT3_THROUGHPUT_SAFETY_FACTOR
    measured_hours = SEGMENT3_TOKENS / measured_rate / 3600
    conservative_hours = SEGMENT3_TOKENS / conservative_rate / 3600

    template = {
        "approved_from_step": PAID_4B_SEGMENT3_START_STEP,
        "approved_from_tokens": PAID_4B_SEGMENT3_START_TOKENS,
        "approved_through_step": PAID_4B_SEGMENT3_STOP_STEP,
        "approved_token_cap": PAID_4B_SEGMENT3_STOP_TOKENS,
        "continuation_source_checkpoint": source_checkpoint,
        "execution_profile": PAID_4B_SEGMENT3_PROFILE,
        "full_4b_training": True,
        "local_input_paths": SEGMENT3_LOCAL_INPUT_PATHS,
        "local_reuse": SEGMENT3_LOCAL_REUSE,
        "max_cost_usd": SEGMENT3_MAX_COST_USD,
        "max_runtime_minutes": SEGMENT3_MAX_RUNTIME_MINUTES,
        "owner_acknowledgement": PAID_4B_PRODUCTION_OWNER_ACKNOWLEDGEMENT,
        "paid_gpu": False,
        "production_contract_sha256": base_plan["contract"]["sha256"],
        "run_fingerprint": run_fingerprint,
        "run_kind": PAID_4B_PRODUCTION_RUN_KIND,
        "schema_version": DISTRIBUTED_AUTHORIZATION_SCHEMA_VERSION,
        "segment_count": 3,
        "segment_index": 3,
        "status": "planning_only",
        "strategy": "fsdp_full_shard",
        "total_token_cap": PAID_4B_PRODUCTION_TOKEN_CAP,
        "world_size": 8,
    }
    plan: dict[str, Any] = {
        "approval": {"full_4b_training": False, "paid_gpu": False},
        "authorization_template": template,
        "base_production_plan": {
            "plan_sha256": base_plan["plan_sha256"],
            "run_fingerprint": base_plan["run_fingerprint"],
            "source_commit": base_plan["source_commit"],
        },
        "budget": {
            "conservative_tokens_per_second": conservative_rate,
            "estimated_conservative_compute_cost_usd": (
                conservative_hours * SEGMENT3_HOURLY_RATE_USD
            ),
            "estimated_conservative_training_hours": conservative_hours,
            "estimated_measured_compute_cost_usd": (
                measured_hours * SEGMENT3_HOURLY_RATE_USD
            ),
            "estimated_measured_training_hours": measured_hours,
            "hourly_rate_usd": SEGMENT3_HOURLY_RATE_USD,
            "instance_type": SEGMENT3_INSTANCE_TYPE,
            "maximum_cost_usd": SEGMENT3_MAX_COST_USD,
            "maximum_runtime_minutes": SEGMENT3_MAX_RUNTIME_MINUTES,
            "provider": SEGMENT3_PROVIDER,
            "throughput_safety_factor": SEGMENT3_THROUGHPUT_SAFETY_FACTOR,
        },
        "checkpoint_gate": {
            "authorization_sha256": gate_report["authorization_sha256"],
            "file_sha256": gate_file_sha256,
            "median_throughput_gain_fraction": gate_report["comparisons"][
                "median_throughput_gain_fraction"
            ],
            "plan_sha256": gate_report["plan_sha256"],
            "report_sha256": gate_report["report_sha256"],
            "selected_profile": gate_report["selected_profile"],
        },
        "created_at": _utc_now(),
        "execution_profile": PAID_4B_SEGMENT3_PROFILE,
        "gate": SEGMENT3_CONTINUATION_GATE,
        "local_input_paths": SEGMENT3_LOCAL_INPUT_PATHS,
        "local_reuse": SEGMENT3_LOCAL_REUSE,
        "run": run,
        "run_fingerprint": run_fingerprint,
        "run_identity": run_identity,
        "schema_version": SEGMENT3_CONTINUATION_SCHEMA_VERSION,
        "segment": {
            "final_step_sequences": 242,
            "final_step_tokens": 991_232,
            "index": 3,
            "start_step": PAID_4B_SEGMENT3_START_STEP,
            "start_tokens": PAID_4B_SEGMENT3_START_TOKENS,
            "stop_step": PAID_4B_SEGMENT3_STOP_STEP,
            "stop_tokens": PAID_4B_SEGMENT3_STOP_TOKENS,
        },
        "source_checkpoint": source_checkpoint,
        "source_commit": source_commit,
        "status": "planning_only_no_paid_gpu",
    }
    plan["plan_sha256"] = _fingerprint(plan)
    return validate_segment3_continuation_plan(plan)


def validate_segment3_continuation_plan(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail if a serialized Segment-3 continuation plan was altered."""

    if (
        plan.get("schema_version") != SEGMENT3_CONTINUATION_SCHEMA_VERSION
        or plan.get("gate") != SEGMENT3_CONTINUATION_GATE
        or plan.get("status") != "planning_only_no_paid_gpu"
    ):
        raise ValueError("unsupported Segment-3 continuation plan")
    if plan.get("approval") != {
        "full_4b_training": False,
        "paid_gpu": False,
    }:
        raise ValueError("Segment-3 plan cannot authorize paid work")
    if plan.get("plan_sha256") != _fingerprint(plan, "plan_sha256"):
        raise ValueError("Segment-3 plan hash changed")
    if plan.get("execution_profile") != PAID_4B_SEGMENT3_PROFILE:
        raise ValueError("Segment-3 selected execution profile changed")
    if plan.get("segment") != {
        "final_step_sequences": 242,
        "final_step_tokens": 991_232,
        "index": 3,
        "start_step": PAID_4B_SEGMENT3_START_STEP,
        "start_tokens": PAID_4B_SEGMENT3_START_TOKENS,
        "stop_step": PAID_4B_SEGMENT3_STOP_STEP,
        "stop_tokens": PAID_4B_SEGMENT3_STOP_TOKENS,
    }:
        raise ValueError("Segment-3 continuation boundary changed")
    template = plan.get("authorization_template")
    if (
        not isinstance(template, dict)
        or template.get("paid_gpu") is not False
        or template.get("status") != "planning_only"
        or template.get("run_fingerprint") != plan.get("run_fingerprint")
        or template.get("continuation_source_checkpoint")
        != plan.get("source_checkpoint")
        or template.get("local_reuse") != SEGMENT3_LOCAL_REUSE
        or template.get("local_input_paths") != SEGMENT3_LOCAL_INPUT_PATHS
    ):
        raise ValueError("Segment-3 authorization template changed")
    if plan.get("local_reuse") != SEGMENT3_LOCAL_REUSE:
        raise ValueError("Segment-3 local reuse scope changed")
    source = plan.get("source_checkpoint")
    if (
        not isinstance(source, dict)
        or source.get("handoff_sha256") != SEGMENT2_HANDOFF_SHA256
        or source.get("manifest_sha256") != SEGMENT2_CHECKPOINT_MANIFEST_SHA256
        or source.get("run_fingerprint") != SEGMENT2_RUN_FINGERPRINT
        or source.get("read_only") is not True
    ):
        raise ValueError("Segment-3 source checkpoint changed")
    return dict(plan)


def owner_approval_text(
    plan: Mapping[str, Any],
    *,
    instance_ip: str,
) -> str:
    """Return the exact owner sentence required to start Segment 3."""

    validated = validate_segment3_continuation_plan(plan)
    if instance_ip != PAID_4B_SEGMENT3_INSTANCE_IP:
        raise ValueError("Segment-3 instance changed")
    source = validated["source_checkpoint"]
    return (
        "I approve exactly one Lumi-4B/80B production Segment 3 on existing "
        f"Lambda instance {instance_ip} ({SEGMENT3_INSTANCE_TYPE}), using source "
        f"commit {validated['source_commit']}, run fingerprint "
        f"{validated['run_fingerprint']}, sealed Segment-2 handoff "
        f"{source['handoff_sha256']}, and checkpoint manifest "
        f"{source['manifest_sha256']}; steps {PAID_4B_SEGMENT3_START_STEP}→"
        f"{PAID_4B_SEGMENT3_STOP_STEP}, tokens {PAID_4B_SEGMENT3_START_TOKENS}→"
        f"{PAID_4B_SEGMENT3_STOP_TOKENS}, maximum "
        f"{SEGMENT3_MAX_RUNTIME_MINUTES} minutes/${SEGMENT3_MAX_COST_USD:,.2f}, "
        "full_4b_training=true, using the checkpoint-verified "
        "fsdp_b2_fa3_compile profile. It may reuse only the already-verified "
        "local 160-object corpus, tokenizer, held-out data, and FA3 runtime on "
        "this instance. No S3 access, AWS credentials, other corpus objects, "
        "training beyond 80,000,000,000 tokens, new instance, or instance "
        "termination is authorized."
    )


def write_segment3_owner_authorization(
    *,
    plan_path: str | Path,
    output: str | Path,
    exact_owner_approval: str,
    instance_ip: str,
) -> dict[str, Any]:
    """Write one Segment-3 authorization only after exact owner approval."""

    plan = validate_segment3_continuation_plan(_load_object(plan_path))
    expected = owner_approval_text(plan, instance_ip=instance_ip)
    if exact_owner_approval.strip() != expected:
        raise PermissionError("exact Segment-3 owner approval text does not match")
    authorization = {
        **plan["authorization_template"],
        "approved_at": _utc_now(),
        "exact_owner_approval": expected,
        "exact_owner_approval_sha256": hashlib.sha256(
            expected.encode("utf-8")
        ).hexdigest(),
        "hourly_rate_usd": SEGMENT3_HOURLY_RATE_USD,
        "instance_ip": instance_ip,
        "instance_type": SEGMENT3_INSTANCE_TYPE,
        "max_cost_usd": SEGMENT3_MAX_COST_USD,
        "max_runtime_minutes": SEGMENT3_MAX_RUNTIME_MINUTES,
        "paid_gpu": True,
        "production_plan_sha256": plan["plan_sha256"],
        "provider": SEGMENT3_PROVIDER,
        "source_commit": plan["source_commit"],
        "status": "owner_approved",
    }
    authorization["authorization_sha256"] = _fingerprint(authorization)
    _write_json_exclusive(Path(output), authorization)
    return authorization
