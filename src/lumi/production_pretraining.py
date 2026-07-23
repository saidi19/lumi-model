"""Planning-only contract for Lumi-4B's exact 80B-token pretraining run.

The repository contract cannot authorize paid work.  This module binds the
sealed marker-aware corpus, proven 8xH100 topology, full learning-rate horizon,
checkpoint cadence, conservative budget, and checkpoint-aligned operator
segments.  A separate owner artifact is still required for each paid segment.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .configuration_lumi import LumiConfig
from .distributed_training import (
    DISTRIBUTED_AUTHORIZATION_SCHEMA_VERSION,
    PAID_4B_PRODUCTION_MAX_COST_USD,
    PAID_4B_PRODUCTION_MAX_RUNTIME_MINUTES,
    PAID_4B_PRODUCTION_MAX_STEPS,
    PAID_4B_PRODUCTION_OWNER_ACKNOWLEDGEMENT,
    PAID_4B_PRODUCTION_RUN_KIND,
    PAID_4B_PRODUCTION_TOKEN_CAP,
    PAID_4B_PRODUCTION_WORLD_SIZE,
    _validate_source_commit,
    build_distributed_run_identity,
)
from .parameter_count import parameter_breakdown
from .production_training import (
    build_account_segment_plan,
    validate_final_corpus_training_manifest,
)

PRODUCTION_PRETRAINING_SCHEMA_VERSION = 1
PRODUCTION_PRETRAINING_GATE = "lumi_4b_80b_pretraining_plan_v0.1"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _fingerprint(value: Mapping[str, Any], omitted: str | None = None) -> str:
    material = dict(value)
    if omitted is not None:
        material.pop(omitted, None)
    return hashlib.sha256(_canonical_json_bytes(material)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_number(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be positive")
    return float(value)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite owner artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _validate_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    if (
        contract.get("schema_version") != PRODUCTION_PRETRAINING_SCHEMA_VERSION
        or contract.get("status") != "planning_only"
    ):
        raise ValueError("unsupported Lumi-4B/80B production contract")
    if contract.get("approval") != {
        "full_4b_training": False,
        "paid_gpu": False,
    }:
        raise ValueError("repository production contract cannot authorize paid work")

    run = contract.get("run", {})
    required_run = {
        "attention_implementation": "flash_attention_2",
        "checkpoint_steps": 250,
        "document_isolation": True,
        "global_sequences_per_step": 256,
        "gradient_accumulation_steps": 32,
        "gradient_checkpointing": True,
        "gradient_sync_mode": "last_microbatch",
        "keep_checkpoints": 2,
        "learning_rate": 0.0002,
        "lr_schedule": "cosine",
        "max_grad_norm": 1.0,
        "maximum_steps": PAID_4B_PRODUCTION_MAX_STEPS,
        "min_learning_rate_ratio": 0.1,
        "optimizer_profile": "legacy",
        "per_rank_batch_size": 1,
        "precision": "bf16",
        "schedule_steps": PAID_4B_PRODUCTION_MAX_STEPS,
        "seed": 7,
        "strategy": "fsdp_full_shard",
        "torch_compile": False,
        "warmup_steps": 1000,
        "weight_decay": 0.1,
        "world_size": PAID_4B_PRODUCTION_WORLD_SIZE,
    }
    for key, expected in required_run.items():
        if run.get(key) != expected:
            raise ValueError(f"Lumi-4B/80B production contract changed {key}")

    corpus = contract.get("corpus", {})
    required_corpus = {
        "available_tokens": 80_000_008_364,
        "doc_token_id": 3,
        "documents": 45_768_283,
        "manifest_sha256": (
            "6de45464af33d44e1261a023ccb7d50f86346a4126bb2fa060530a91213483e9"
        ),
        "object_bytes": 160_000_016_728,
        "plan_fingerprint": (
            "aab292a514b2a7a046a3e4f0daf2fb75bb6f899ff86798bdfac729a145bc9470"
        ),
        "selection_result_sha256": (
            "b017a6fec5c6ca1685c5dcd883d8afc48d36fd18cac07d573514ad77307253ee"
        ),
        "sequence_length": 4096,
        "shards": 160,
        "training_token_limit": PAID_4B_PRODUCTION_TOKEN_CAP,
        "vocab_size": 65_536,
    }
    for key, expected in required_corpus.items():
        if corpus.get(key) != expected:
            raise ValueError(f"Lumi-4B/80B production corpus changed {key}")

    if contract.get("model") != {
        "config": "configs/lumi-4b-v0.2.json",
        "config_sha256": (
            "643e8da652577f802d959740d66a7e6bafc5b065aa45010c2d7e0340ed9509be"
        ),
        "parameters": 4_002_376_704,
    }:
        raise ValueError("Lumi-4B/80B model contract changed")

    budget = contract.get("budget", {})
    required_budget = {
        "available_credit_usd_per_segment": 7500,
        "credit_reserve_usd_per_segment": 500,
        "hourly_rate_usd": 31.92,
        "instance_type": "gpu_8x_h100_sxm5",
        "maximum_cost_usd_per_segment": PAID_4B_PRODUCTION_MAX_COST_USD,
        "provider": "Lambda Cloud",
        "provider_runtime_cap_hours": 168,
        "runtime_reserve_hours": 8,
        "throughput_safety_factor": 0.8,
    }
    for key, expected in required_budget.items():
        if budget.get(key) != expected:
            raise ValueError(f"Lumi-4B/80B production budget changed {key}")
    if (
        budget["provider_runtime_cap_hours"] * 60
        != PAID_4B_PRODUCTION_MAX_RUNTIME_MINUTES
    ):
        raise ValueError("Lumi-4B/80B runtime cap changed")

    if contract.get("checkpoint_handoff") != {
        "format": "dcp-model_rank-local-optimizer-v0.1",
        "hash_every_file": True,
        "require_exact_run_fingerprint": True,
        "require_same_world_size": True,
    }:
        raise ValueError("Lumi-4B/80B checkpoint handoff contract changed")

    hardware = contract.get("hardware", {})
    if hardware != {
        "compute_capability": [9, 0],
        "gpu_model_contains": "H100",
        "minimum_free_disk_bytes": 500_000_000_000,
        "minimum_host_memory_bytes": 128_000_000_000,
        "minimum_memory_bytes_per_gpu": 80_000_000_000,
        "world_size": PAID_4B_PRODUCTION_WORLD_SIZE,
    }:
        raise ValueError("Lumi-4B/80B hardware contract changed")

    software = contract.get("software", {})
    if software != {
        "flash_attention_version": "2.8.3",
        "torch_version": "2.7.0+cu126",
    }:
        raise ValueError("Lumi-4B/80B software contract changed")

    evaluation = contract.get("evaluation", {})
    if evaluation != {
        "batch_size": 2,
        "heldout_manifest_sha256": (
            "c5022ce7ba9ed41dd9cdf1777639278c5ed111ef679e09d9fbc12e36fd6260dd"
        ),
        "heldout_sequences": 2441,
        "heldout_tokens": 4_999_168,
        "milestone_steps": [
            1000,
            5000,
            10_000,
            20_000,
            40_000,
            60_000,
            PAID_4B_PRODUCTION_MAX_STEPS,
        ],
    }:
        raise ValueError("Lumi-4B/80B evaluation contract changed")
    milestones = evaluation.get("milestone_steps")
    if not isinstance(milestones, list) or milestones != sorted(set(milestones)):
        raise ValueError("Lumi-4B/80B evaluation milestones are invalid")
    if not milestones or milestones[-1] != PAID_4B_PRODUCTION_MAX_STEPS:
        raise ValueError("Lumi-4B/80B final evaluation milestone changed")
    if any(
        not isinstance(step, int)
        or isinstance(step, bool)
        or step <= 0
        or step > PAID_4B_PRODUCTION_MAX_STEPS
        or (step != PAID_4B_PRODUCTION_MAX_STEPS and step % run["checkpoint_steps"])
        for step in milestones
    ):
        raise ValueError("Lumi-4B/80B evaluation milestone is not a checkpoint")

    pilot = contract.get("pilot_evidence", {})
    if pilot != {
        "median_tokens_per_second": 71_935.43151677848,
        "report": "reports/4b-learning-pilot-result-v0.1.json",
        "report_sha256": (
            "d7c4ce9db462219bb4feab4dee43f46fe158f99190eb6d3b35d77216519d5130"
        ),
        "run_fingerprint": (
            "c79c8a5e5a681aa61a26439d5a2aa39798cdc3884b626550cd8ff96ac7ac6eb9"
        ),
        "source_commit": "7ae753b12f0a84d9598f12e7f9edf0f5d00f138c",
        "status": "pass",
    }:
        raise ValueError("Lumi-4B/80B pilot evidence contract changed")
    return dict(contract)


def _validate_pilot_report(contract: Mapping[str, Any], pilot_result_path: Path) -> dict[str, Any]:
    expected = contract["pilot_evidence"]
    if _sha256(pilot_result_path) != expected["report_sha256"]:
        raise ValueError("Lumi-4B learning-pilot report SHA-256 changed")
    report = json.loads(pilot_result_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("status") != "pass":
        raise ValueError("Lumi-4B learning pilot did not pass")
    if report.get("gate") != "lumi_4b_learning_pilot_v0.1":
        raise ValueError("unexpected Lumi-4B learning-pilot gate")
    checks = (
        (report.get("run_fingerprint"), expected["run_fingerprint"], "run fingerprint"),
        (report.get("source_commit"), expected["source_commit"], "source commit"),
        (report.get("model_parameters"), contract["model"]["parameters"], "model parameters"),
        (
            report.get("training", {}).get("median_tokens_per_second"),
            expected["median_tokens_per_second"],
            "median throughput",
        ),
        (
            report.get("topology", {}).get("world_size"),
            contract["run"]["world_size"],
            "world size",
        ),
        (
            report.get("topology", {}).get("gradient_accumulation_steps"),
            contract["run"]["gradient_accumulation_steps"],
            "gradient accumulation",
        ),
    )
    for observed, pinned, label in checks:
        if observed != pinned:
            raise ValueError(f"Lumi-4B learning-pilot {label} changed")
    if report.get("limitations", {}).get("full_4b_80b_training_completed") is not False:
        raise ValueError("pilot report incorrectly claims full training")
    if report.get("independent_verification", {}).get("seal_status") != "pass":
        raise ValueError("pilot evidence was not independently sealed")
    return report


def _build_segments(
    *,
    run_fingerprint: str,
    contract_sha256: str,
    run: Mapping[str, Any],
    budget: Mapping[str, Any],
    measured_tokens_per_second: float,
) -> list[dict[str, Any]]:
    training_hours = (
        float(budget["provider_runtime_cap_hours"])
        - float(budget["runtime_reserve_hours"])
    )
    if training_hours <= 0:
        raise ValueError("production runtime reserve consumes the entire segment")
    segments: list[dict[str, Any]] = []
    current_step = 0
    current_tokens = 0
    while current_tokens < PAID_4B_PRODUCTION_TOKEN_CAP:
        boundary = build_account_segment_plan(
            run_fingerprint=run_fingerprint,
            current_step=current_step,
            current_tokens=current_tokens,
            total_tokens=PAID_4B_PRODUCTION_TOKEN_CAP,
            nominal_global_tokens_per_step=(
                int(run["global_sequences_per_step"]) * 4096
            ),
            checkpoint_steps=int(run["checkpoint_steps"]),
            measured_tokens_per_second=measured_tokens_per_second,
            throughput_safety_factor=float(budget["throughput_safety_factor"]),
            hourly_rate_usd=float(budget["hourly_rate_usd"]),
            available_credit_usd=float(budget["available_credit_usd_per_segment"]),
            credit_reserve_usd=float(budget["credit_reserve_usd_per_segment"]),
            provider_runtime_limit_hours=training_hours,
        )
        stop = boundary["stop"]
        segments.append(
            {
                "boundary_plan": boundary,
                "start_step": current_step,
                "start_tokens": current_tokens,
                "stop_step": int(stop["step"]),
                "stop_tokens": int(stop["tokens"]),
            }
        )
        current_step = int(stop["step"])
        current_tokens = int(stop["tokens"])
    count = len(segments)
    if count < 1:
        raise RuntimeError("production plan produced no operator segments")
    for index, segment in enumerate(segments, start=1):
        segment["index"] = index
        segment["count"] = count
        segment["authorization_template"] = {
            "approved_from_step": segment["start_step"],
            "approved_from_tokens": segment["start_tokens"],
            "approved_through_step": segment["stop_step"],
            "approved_token_cap": segment["stop_tokens"],
            "full_4b_training": True,
            "max_cost_usd": float(budget["maximum_cost_usd_per_segment"]),
            "max_runtime_minutes": PAID_4B_PRODUCTION_MAX_RUNTIME_MINUTES,
            "owner_acknowledgement": PAID_4B_PRODUCTION_OWNER_ACKNOWLEDGEMENT,
            "paid_gpu": False,
            "production_contract_sha256": contract_sha256,
            "run_fingerprint": run_fingerprint,
            "run_kind": PAID_4B_PRODUCTION_RUN_KIND,
            "schema_version": DISTRIBUTED_AUTHORIZATION_SCHEMA_VERSION,
            "segment_count": count,
            "segment_index": index,
            "status": "planning_only",
            "strategy": run["strategy"],
            "total_token_cap": PAID_4B_PRODUCTION_TOKEN_CAP,
            "world_size": run["world_size"],
        }
    return segments


def plan_4b_80b_production(
    *,
    contract_path: str | Path,
    manifest_path: str | Path,
    config_path: str | Path,
    pilot_result_path: str | Path,
    source_commit: str,
) -> dict[str, Any]:
    """Build one immutable, planning-only full-run specification."""

    contract_path = Path(contract_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    config_path = Path(config_path).resolve()
    pilot_result_path = Path(pilot_result_path).resolve()
    contract = _validate_contract(
        json.loads(contract_path.read_text(encoding="utf-8"))
    )
    normalized_commit = _validate_source_commit(source_commit)
    assert normalized_commit is not None
    pilot = _validate_pilot_report(contract, pilot_result_path)

    corpus = contract["corpus"]
    adapter = validate_final_corpus_training_manifest(
        manifest_path,
        sequence_length=corpus["sequence_length"],
        vocab_size=corpus["vocab_size"],
        doc_token_id=corpus["doc_token_id"],
        expected_manifest_sha256=corpus["manifest_sha256"],
        expected_plan_fingerprint=corpus["plan_fingerprint"],
        expected_selection_result_sha256=corpus["selection_result_sha256"],
        expected_documents=corpus["documents"],
        expected_tokens=corpus["available_tokens"],
        expected_training_token_limit=corpus["training_token_limit"],
        expected_shards=corpus["shards"],
    )
    statistics = adapter["statistics"]
    if int(statistics["object_bytes"]) != int(corpus["object_bytes"]):
        raise ValueError("Lumi-4B/80B corpus byte total changed")
    if int(statistics["tokens"]) != PAID_4B_PRODUCTION_TOKEN_CAP:
        raise ValueError("Lumi-4B/80B trainer boundary changed")

    config_bytes = config_path.read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != contract["model"]["config_sha256"]:
        raise ValueError("Lumi-4B model config SHA-256 changed")
    model_config = LumiConfig(**json.loads(config_bytes))
    if parameter_breakdown(model_config).total != contract["model"]["parameters"]:
        raise ValueError("Lumi-4B parameter count changed")
    if model_config.initialization_strategy != "residual_scaled_v0.1":
        raise ValueError("Lumi-4B initialization strategy changed")
    if model_config.vocab_size != corpus["vocab_size"]:
        raise ValueError("Lumi-4B vocabulary and final corpus differ")
    if model_config.doc_token_id != corpus["doc_token_id"]:
        raise ValueError("Lumi-4B document token and final corpus differ")

    run = contract["run"]
    global_sequences_per_step = (
        run["world_size"]
        * run["per_rank_batch_size"]
        * run["gradient_accumulation_steps"]
    )
    if global_sequences_per_step != run["global_sequences_per_step"]:
        raise ValueError("Lumi-4B/80B global batch geometry changed")
    total_sequences = int(statistics["sequences"])
    maximum_steps = math.ceil(total_sequences / global_sequences_per_step)
    if maximum_steps != run["maximum_steps"]:
        raise ValueError("Lumi-4B/80B exact step boundary changed")
    final_step_sequences = (
        total_sequences - (maximum_steps - 1) * global_sequences_per_step
    )
    if final_step_sequences != 242:
        raise ValueError("Lumi-4B/80B exact final-step sequence count changed")

    manifest_sha256 = _sha256(manifest_path)
    identity = build_distributed_run_identity(
        config_bytes=config_bytes,
        manifest=adapter,
        manifest_sha256=manifest_sha256,
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
        source_commit=normalized_commit,
        torch_version=contract["software"]["torch_version"],
    )
    run_fingerprint = _fingerprint(identity)
    contract_sha256 = _sha256(contract_path)
    measured_rate = float(pilot["training"]["median_tokens_per_second"])
    conservative_rate = measured_rate * float(
        contract["budget"]["throughput_safety_factor"]
    )
    measured_hours = PAID_4B_PRODUCTION_TOKEN_CAP / measured_rate / 3600
    conservative_hours = PAID_4B_PRODUCTION_TOKEN_CAP / conservative_rate / 3600
    segments = _build_segments(
        run_fingerprint=run_fingerprint,
        contract_sha256=contract_sha256,
        run=run,
        budget=contract["budget"],
        measured_tokens_per_second=measured_rate,
    )

    plan: dict[str, Any] = {
        "approval": {
            "full_4b_training": False,
            "paid_gpu": False,
        },
        "budget": {
            **contract["budget"],
            "conservative_tokens_per_second": conservative_rate,
            "estimated_conservative_compute_cost_usd": (
                conservative_hours * float(contract["budget"]["hourly_rate_usd"])
            ),
            "estimated_conservative_training_hours": conservative_hours,
            "estimated_measured_compute_cost_usd": (
                measured_hours * float(contract["budget"]["hourly_rate_usd"])
            ),
            "estimated_measured_training_hours": measured_hours,
            "estimate_scope": "training_only_excludes_staging_evaluation_and_transfer",
        },
        "checkpoint_handoff": contract["checkpoint_handoff"],
        "contract": {
            "path": str(contract_path),
            "sha256": contract_sha256,
        },
        "corpus": {
            "adapter_build_fingerprint": adapter["build_fingerprint"],
            "available_tokens": statistics["available_tokens"],
            "documents": statistics["documents"],
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "object_bytes": statistics["object_bytes"],
            "sequences": total_sequences,
            "shards": len(adapter["shards"]),
            "training_tokens": statistics["tokens"],
            "trailing_tokens_not_consumed": statistics[
                "trailing_tokens_not_consumed"
            ],
        },
        "created_at": _utc_now(),
        "evaluation": contract["evaluation"],
        "gate": PRODUCTION_PRETRAINING_GATE,
        "hardware": contract["hardware"],
        "model": {
            "config_path": str(config_path),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "parameters": contract["model"]["parameters"],
            "start": "fresh_residual_scaled_initialization",
        },
        "pilot_evidence": {
            **contract["pilot_evidence"],
            "path": str(pilot_result_path),
        },
        "run": {
            **run,
            "exact_final_step_sequences": final_step_sequences,
            "exact_final_step_tokens": final_step_sequences
            * corpus["sequence_length"],
            "global_tokens_per_step": global_sequences_per_step
            * corpus["sequence_length"],
        },
        "run_fingerprint": run_fingerprint,
        "run_identity": identity,
        "schema_version": PRODUCTION_PRETRAINING_SCHEMA_VERSION,
        "segments": segments,
        "software": contract["software"],
        "source_commit": normalized_commit,
        "status": "planning_only_no_paid_gpu",
    }
    plan["plan_sha256"] = _fingerprint(plan)
    return plan


def validate_production_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Fail if a serialized planning-only production plan was altered."""

    if (
        plan.get("schema_version") != PRODUCTION_PRETRAINING_SCHEMA_VERSION
        or plan.get("gate") != PRODUCTION_PRETRAINING_GATE
    ):
        raise ValueError("unsupported Lumi-4B/80B production plan")
    if plan.get("status") != "planning_only_no_paid_gpu":
        raise ValueError("Lumi-4B/80B production plan has an unsafe status")
    if plan.get("approval") != {
        "full_4b_training": False,
        "paid_gpu": False,
    }:
        raise ValueError("Lumi-4B/80B production plan authorizes paid work")
    if plan.get("plan_sha256") != _fingerprint(plan, "plan_sha256"):
        raise ValueError("Lumi-4B/80B production plan hash changed")
    segments = plan.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("Lumi-4B/80B production plan has no segments")
    if segments[0].get("start_step") != 0 or segments[0].get("start_tokens") != 0:
        raise ValueError("Lumi-4B/80B first segment does not start fresh")
    if (
        segments[-1].get("stop_step") != PAID_4B_PRODUCTION_MAX_STEPS
        or segments[-1].get("stop_tokens") != PAID_4B_PRODUCTION_TOKEN_CAP
    ):
        raise ValueError("Lumi-4B/80B final segment boundary changed")
    for previous, current in zip(segments, segments[1:], strict=False):
        if (
            previous.get("stop_step") != current.get("start_step")
            or previous.get("stop_tokens") != current.get("start_tokens")
        ):
            raise ValueError("Lumi-4B/80B segment chain has a gap")
    if any(
        segment.get("authorization_template", {}).get("paid_gpu") is not False
        or segment.get("authorization_template", {}).get("status") != "planning_only"
        for segment in segments
    ):
        raise ValueError("Lumi-4B/80B segment template authorizes paid work")
    return dict(plan)


def _load_validated_plan(plan_path: Path) -> dict[str, Any]:
    return validate_production_plan(json.loads(plan_path.read_text(encoding="utf-8")))


def write_owner_authorization(
    *,
    plan_path: str | Path,
    output: str | Path,
    segment_index: int,
    provider: str,
    instance_type: str,
    hourly_rate_usd: float,
    max_runtime_minutes: int,
    max_cost_usd: float,
) -> dict[str, Any]:
    """Create one exact owner-approved segment artifact from a sealed plan."""

    plan = _load_validated_plan(Path(plan_path))
    if not isinstance(segment_index, int) or isinstance(segment_index, bool):
        raise ValueError("production segment index must be an integer")
    segments = plan["segments"]
    if not 1 <= segment_index <= len(segments):
        raise ValueError("production segment index is outside the plan")
    segment = segments[segment_index - 1]
    template = segment["authorization_template"]
    budget = plan["budget"]
    if provider.strip() != budget["provider"]:
        raise ValueError("production provider differs from the measured plan")
    if instance_type.strip() != budget["instance_type"]:
        raise ValueError("production instance type differs from the measured plan")
    hourly_rate = _positive_number(hourly_rate_usd, "hourly rate")
    if hourly_rate > float(budget["hourly_rate_usd"]):
        raise ValueError("production hourly rate exceeds the planned rate")
    if (
        not isinstance(max_runtime_minutes, int)
        or isinstance(max_runtime_minutes, bool)
        or not 30 <= max_runtime_minutes <= template["max_runtime_minutes"]
    ):
        raise ValueError("production runtime cap is outside the segment plan")
    maximum_cost = _positive_number(max_cost_usd, "maximum cost")
    if maximum_cost > float(template["max_cost_usd"]):
        raise ValueError("production cost cap exceeds the segment plan")
    if hourly_rate * max_runtime_minutes / 60 > maximum_cost:
        raise ValueError("production runtime and hourly rate exceed the cost cap")

    authorization = {
        **template,
        "approved_at": _utc_now(),
        "hourly_rate_usd": hourly_rate,
        "instance_type": instance_type.strip(),
        "max_cost_usd": maximum_cost,
        "max_runtime_minutes": max_runtime_minutes,
        "paid_gpu": True,
        "production_plan_sha256": plan["plan_sha256"],
        "provider": provider.strip(),
        "source_commit": plan["source_commit"],
        "status": "owner_approved",
    }
    _write_json_atomic(Path(output), authorization)
    return authorization


def validate_owner_authorization(
    *,
    plan_path: str | Path,
    authorization_path: str | Path,
) -> dict[str, Any]:
    """Validate one segment artifact against every material planning boundary."""

    plan = _load_validated_plan(Path(plan_path))
    authorization = json.loads(Path(authorization_path).read_text(encoding="utf-8"))
    index = authorization.get("segment_index")
    if not isinstance(index, int) or isinstance(index, bool) or not (
        1 <= index <= len(plan["segments"])
    ):
        raise ValueError("production authorization segment index is invalid")
    template = plan["segments"][index - 1]["authorization_template"]
    exact_fields = (
        "approved_from_step",
        "approved_from_tokens",
        "approved_through_step",
        "approved_token_cap",
        "full_4b_training",
        "owner_acknowledgement",
        "production_contract_sha256",
        "run_fingerprint",
        "run_kind",
        "schema_version",
        "segment_count",
        "segment_index",
        "strategy",
        "total_token_cap",
        "world_size",
    )
    for field in exact_fields:
        if authorization.get(field) != template.get(field):
            raise ValueError(f"production authorization changed {field}")
    if authorization.get("status") != "owner_approved":
        raise ValueError("Lumi-4B/80B segment is not owner-approved")
    if authorization.get("paid_gpu") is not True:
        raise ValueError("Lumi-4B/80B paid boundary is not explicit")
    if authorization.get("full_4b_training") is not True:
        raise ValueError("Lumi-4B/80B authorization is not a production segment")
    if authorization.get("production_plan_sha256") != plan["plan_sha256"]:
        raise ValueError("production authorization targets another plan")
    if authorization.get("source_commit") != plan["source_commit"]:
        raise ValueError("production authorization source commit changed")
    budget = plan["budget"]
    if authorization.get("provider") != budget["provider"]:
        raise ValueError("production authorization provider changed")
    if authorization.get("instance_type") != budget["instance_type"]:
        raise ValueError("production authorization instance type changed")
    hourly_rate = _positive_number(
        authorization.get("hourly_rate_usd"), "authorization hourly rate"
    )
    if hourly_rate > float(budget["hourly_rate_usd"]):
        raise ValueError("production authorization hourly rate exceeds the plan")
    runtime = authorization.get("max_runtime_minutes")
    if (
        not isinstance(runtime, int)
        or isinstance(runtime, bool)
        or not 30 <= runtime <= template["max_runtime_minutes"]
    ):
        raise ValueError("production authorization runtime cap is invalid")
    maximum_cost = _positive_number(
        authorization.get("max_cost_usd"), "authorization maximum cost"
    )
    if maximum_cost > float(template["max_cost_usd"]):
        raise ValueError("production authorization cost cap exceeds the plan")
    if hourly_rate * runtime / 60 > maximum_cost:
        raise ValueError("production authorization runtime exceeds its cost cap")
    return authorization
