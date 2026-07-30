from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

import lumi.distributed_training as distributed_training
from lumi.distributed_training import DistributedContext, _validate_paid_authorization
from lumi.segment2_continuation import (
    SEGMENT2_PROFILE,
    build_segment2_continuation_plan,
    owner_approval_text,
    validate_segment2_continuation_plan,
    write_segment2_owner_authorization,
)


def canonical_fingerprint(value: dict, omitted: str | None = None) -> str:
    material = dict(value)
    if omitted is not None:
        material.pop(omitted, None)
    payload = json.dumps(
        material,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_inputs(tmp_path: Path) -> dict[str, Path | str]:
    config = tmp_path / "config.json"
    config.write_text('{"model":"test"}\n', encoding="utf-8")
    config_sha256 = hashlib.sha256(config.read_bytes()).hexdigest()

    corpus = tmp_path / "corpus.json"
    write_json(
        corpus,
        {
            "shards": [
                {"sequence": index, "object_sha256": f"{index:064x}"}
                for index in range(160)
            ]
        },
    )
    corpus_sha256 = hashlib.sha256(corpus.read_bytes()).hexdigest()
    source_run = "4" * 64
    checkpoint_manifest = "5" * 64
    base = {
        "approval": {"full_4b_training": False, "paid_gpu": False},
        "contract": {"sha256": "6" * 64},
        "corpus": {
            "adapter_build_fingerprint": "7" * 64,
            "manifest_sha256": corpus_sha256,
        },
        "gate": "lumi_4b_80b_pretraining_plan_v0.1",
        "model": {"config_sha256": config_sha256},
        "run": {
            "attention_implementation": "flash_attention_2",
            "checkpoint_steps": 250,
            "document_isolation": True,
            "exact_final_step_sequences": 242,
            "exact_final_step_tokens": 991232,
            "global_sequences_per_step": 256,
            "global_tokens_per_step": 1_048_576,
            "gradient_accumulation_steps": 32,
            "gradient_checkpointing": True,
            "gradient_sync_mode": "last_microbatch",
            "keep_checkpoints": 2,
            "learning_rate": 0.0002,
            "lr_schedule": "cosine",
            "max_grad_norm": 1.0,
            "maximum_steps": 76_294,
            "min_learning_rate_ratio": 0.1,
            "optimizer_profile": "legacy",
            "per_rank_batch_size": 1,
            "precision": "bf16",
            "schedule_steps": 76_294,
            "seed": 7,
            "strategy": "fsdp_full_shard",
            "torch_compile": False,
            "warmup_steps": 1000,
            "weight_decay": 0.1,
            "world_size": 8,
        },
        "run_fingerprint": source_run,
        "schema_version": 1,
        "segments": [
            {
                "authorization_template": {
                    "paid_gpu": False,
                    "status": "planning_only",
                },
                "start_step": 0,
                "start_tokens": 0,
                "stop_step": 31_500,
                "stop_tokens": 33_030_144_000,
            },
            {
                "authorization_template": {
                    "paid_gpu": False,
                    "status": "planning_only",
                },
                "index": 2,
                "start_step": 31_500,
                "start_tokens": 33_030_144_000,
                "stop_step": 63_000,
                "stop_tokens": 66_060_288_000,
            },
            {
                "authorization_template": {
                    "paid_gpu": False,
                    "status": "planning_only",
                },
                "start_step": 63_000,
                "start_tokens": 66_060_288_000,
                "stop_step": 76_294,
                "stop_tokens": 80_000_000_000,
            },
        ],
        "software": {
            "flash_attention_version": "2.8.3",
            "torch_version": "2.7.0+cu126",
        },
        "source_commit": "8" * 40,
        "status": "planning_only_no_paid_gpu",
    }
    base["plan_sha256"] = canonical_fingerprint(base)
    base_path = tmp_path / "base.json"
    write_json(base_path, base)

    source_checkpoint = {
        "manifest_sha256": checkpoint_manifest,
        "path": "/old/checkpoint",
        "read_only": True,
        "run_fingerprint": source_run,
        "sequence_cursor": 8_064_000,
        "step": 31_500,
        "tokens": 33_030_144_000,
    }
    candidate = {
        "checkpoint_restore_pass": True,
        "heldout_loss": 2.18,
        "label": "batch2_no_gc",
        "median_tokens_per_second": 100_000.0,
        "p10_tokens_per_second": 99_000.0,
        "peak_memory_allocated_bytes": 61_000_000_000,
        "profile": SEGMENT2_PROFILE,
        "resume_canary_pass": True,
        "run_fingerprint": "9" * 64,
        "source_checkpoint": {
            key: source_checkpoint[key]
            for key in (
                "manifest_sha256",
                "run_fingerprint",
                "sequence_cursor",
                "step",
                "tokens",
            )
        },
        "source_checkpoint_unchanged": True,
        "status": "pass",
        "training_loss": 1.83,
    }
    gate = {
        "authorization_sha256": "a" * 64,
        "comparisons": {
            "heldout_loss_regression_fraction": 0.0,
            "median_throughput_gain_fraction": 0.42,
            "p10_throughput_gain_fraction": 0.45,
            "training_loss_regression_fraction": 0.0,
        },
        "plan_sha256": "b" * 64,
        "results": [candidate],
        "selected_profile": "batch2_no_gc",
        "source_checkpoint": source_checkpoint,
        "status": "pass",
    }
    gate["report_sha256"] = canonical_fingerprint(gate)
    gate_path = tmp_path / "gate.json"
    write_json(gate_path, gate)
    gate_file_sha256 = hashlib.sha256(gate_path.read_bytes()).hexdigest()

    import_report = {
        "archive": "/home/ubuntu/import/archive",
        "checkpoint": {
            "sequence_cursor": 8_064_000,
            "step": 31_500,
            "tokens": 33_030_144_000,
            "world_size": 8,
        },
        "gate": (
            "lumi_4b_80b_segment1_checkpoint_import_"
            "independent_verification_v0.1"
        ),
        "handoff_sha256": "c" * 64,
        "manifest_entries_verified": 18,
        "plan_sha256": "d" * 64,
        "schema_version": 1,
        "status": "pass",
        "total_bytes": 48_844_401_057,
        "total_objects": 33,
        "verified_at_utc": "2026-07-30T20:28:47+00:00",
        "verified_objects": [
            {
                "relative_path": (
                    "training/checkpoint-00031500/manifest.sha256"
                ),
                "sha256": checkpoint_manifest,
            },
            {
                "relative_path": (
                    "evidence/exact-checkpoint-gate/run-approved-001/"
                    "verification-report.json"
                ),
                "sha256": gate_file_sha256,
            },
        ],
    }
    import_path = tmp_path / "import.json"
    write_json(import_path, import_report)
    return {
        "base": base_path,
        "config": config,
        "corpus": corpus,
        "gate": gate_path,
        "import": import_path,
        "archive": "/home/ubuntu/import/archive",
    }


def build_plan(tmp_path: Path) -> dict:
    inputs = build_inputs(tmp_path)
    return build_segment2_continuation_plan(
        base_plan_path=inputs["base"],
        checkpoint_gate_report_path=inputs["gate"],
        import_verification_path=inputs["import"],
        config_path=inputs["config"],
        corpus_manifest_path=inputs["corpus"],
        imported_archive_root=str(inputs["archive"]),
        source_commit="e" * 40,
    )


def test_checkpoint_bound_plan_promotes_only_measured_profile(
    tmp_path: Path,
) -> None:
    plan = build_plan(tmp_path)

    assert plan["approval"] == {
        "full_4b_training": False,
        "paid_gpu": False,
    }
    assert plan["execution_profile"] == SEGMENT2_PROFILE
    assert plan["segment"] == {
        "index": 2,
        "start_step": 31_500,
        "start_tokens": 33_030_144_000,
        "stop_step": 63_000,
        "stop_tokens": 66_060_288_000,
    }
    assert plan["run"]["per_rank_batch_size"] == 2
    assert plan["run"]["gradient_accumulation_steps"] == 16
    assert plan["run"]["gradient_checkpointing"] is False
    assert plan["authorization_template"]["paid_gpu"] is False
    assert plan["source_checkpoint"]["manifest_sha256"] == "5" * 64
    assert plan["corpus"]["staging_status"] == "required_not_authorized"
    assert plan["budget"]["estimated_measured_training_hours"] == pytest.approx(
        91.7504
    )

    changed = copy.deepcopy(plan)
    changed["segment"]["stop_step"] += 1
    with pytest.raises(ValueError, match="plan hash changed"):
        validate_segment2_continuation_plan(changed)


def test_owner_authorization_requires_exact_dynamic_sentence(
    tmp_path: Path,
) -> None:
    plan = build_plan(tmp_path)
    plan_path = tmp_path / "segment2-plan.json"
    write_json(plan_path, plan)
    text = owner_approval_text(
        plan,
        instance_ip="192.222.52.81",
        s3_bucket="lumi-corpus-prod-27777c02b29b76efcac3",
    )
    authorization = write_segment2_owner_authorization(
        plan_path=plan_path,
        output=tmp_path / "authorization.json",
        exact_owner_approval=text,
        instance_ip="192.222.52.81",
        s3_bucket="lumi-corpus-prod-27777c02b29b76efcac3",
    )

    assert authorization["paid_gpu"] is True
    assert authorization["full_4b_training"] is True
    assert authorization["segment_index"] == 2
    assert authorization["approved_through_step"] == 63_000
    assert authorization["s3_corpus_access"]["objects"] == 160

    with pytest.raises(PermissionError, match="does not match"):
        write_segment2_owner_authorization(
            plan_path=plan_path,
            output=tmp_path / "bad-authorization.json",
            exact_owner_approval=text + " changed",
            instance_ip="192.222.52.81",
            s3_bucket="lumi-corpus-prod-27777c02b29b76efcac3",
        )


def test_distributed_guard_accepts_only_exact_segment2_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = build_plan(tmp_path)
    plan_path = tmp_path / "segment2-plan.json"
    write_json(plan_path, plan)
    text = owner_approval_text(
        plan,
        instance_ip="192.222.52.81",
        s3_bucket="lumi-corpus-prod-27777c02b29b76efcac3",
    )
    authorization_path = tmp_path / "authorization.json"
    write_segment2_owner_authorization(
        plan_path=plan_path,
        output=authorization_path,
        exact_owner_approval=text,
        instance_ip="192.222.52.81",
        s3_bucket="lumi-corpus-prod-27777c02b29b76efcac3",
    )
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=8,
        backend="nccl",
        device=torch.device("cuda", 0),
    )
    monkeypatch.setattr(
        distributed_training,
        "_validate_production_cuda_runtime",
        lambda _context: None,
    )

    validated = _validate_paid_authorization(
        authorization_path,
        run_fingerprint=plan["run_fingerprint"],
        source_commit=plan["source_commit"],
        context=context,
        strategy="fsdp_full_shard",
        steps=63_000,
        target_tokens=66_060_288_000,
        global_tokens_per_step=1_048_576,
        per_rank_batch_size=2,
        gradient_accumulation_steps=16,
        gradient_checkpointing=False,
        optimizer_profile="legacy",
    )
    assert validated["segment_index"] == 2

    with pytest.raises(ValueError, match="runtime profile changed"):
        _validate_paid_authorization(
            authorization_path,
            run_fingerprint=plan["run_fingerprint"],
            source_commit=plan["source_commit"],
            context=context,
            strategy="fsdp_full_shard",
            steps=63_000,
            target_tokens=66_060_288_000,
            global_tokens_per_step=1_048_576,
            per_rank_batch_size=1,
            gradient_accumulation_steps=32,
            gradient_checkpointing=True,
            optimizer_profile="legacy",
        )

    tampered = json.loads(authorization_path.read_text(encoding="utf-8"))
    tampered["s3_corpus_access"]["bytes"] += 1
    tampered_path = tmp_path / "tampered-authorization.json"
    write_json(tampered_path, tampered)
    with pytest.raises(ValueError, match="authorization hash changed"):
        _validate_paid_authorization(
            tampered_path,
            run_fingerprint=plan["run_fingerprint"],
            source_commit=plan["source_commit"],
            context=context,
            strategy="fsdp_full_shard",
            steps=63_000,
            target_tokens=66_060_288_000,
            global_tokens_per_step=1_048_576,
            per_rank_batch_size=2,
            gradient_accumulation_steps=16,
            gradient_checkpointing=False,
            optimizer_profile="legacy",
        )
