from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

import lumi.distributed_training as distributed_training
import lumi.segment3_continuation as segment3
from lumi.distributed_training import DistributedContext, _validate_paid_authorization
from lumi.segment3_continuation import (
    SEGMENT3_LOCAL_INPUT_PATHS,
    SEGMENT3_LOCAL_REUSE,
    build_segment3_continuation_plan,
    owner_approval_text,
    validate_segment3_continuation_plan,
    write_segment3_owner_authorization,
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


def build_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Path]:
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
    monkeypatch.setattr(
        segment3,
        "SEGMENT3_CORPUS_MANIFEST_SHA256",
        corpus_sha256,
    )
    monkeypatch.setattr(
        distributed_training,
        "PAID_4B_SEGMENT3_CORPUS_MANIFEST_SHA256",
        corpus_sha256,
    )
    monkeypatch.setitem(
        segment3.SEGMENT3_LOCAL_REUSE,
        "corpus_manifest_sha256",
        corpus_sha256,
    )

    source_run = "4" * 64
    monkeypatch.setattr(segment3, "SEGMENT2_RUN_FINGERPRINT", source_run)
    checkpoint_manifest = "5" * 64
    monkeypatch.setattr(
        segment3,
        "SEGMENT2_CHECKPOINT_MANIFEST_SHA256",
        checkpoint_manifest,
    )
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
        "run_fingerprint": "8" * 64,
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
                "index": 3,
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
        "source_commit": "9" * 40,
        "status": "planning_only_no_paid_gpu",
    }
    base["plan_sha256"] = canonical_fingerprint(base)
    base_path = tmp_path / "base.json"
    write_json(base_path, base)

    profile = distributed_training.PAID_4B_SEGMENT3_PROFILE
    source = {
        "manifest_sha256": checkpoint_manifest,
        "path": "/remote/checkpoint-00063000",
        "read_only": True,
        "run_fingerprint": source_run,
        "sequence_cursor": 16_128_000,
        "step": 63_000,
        "tokens": 66_060_288_000,
    }
    candidate = {
        "checkpoint_restore_pass": True,
        "heldout_loss": 2.0508,
        "label": "fsdp_b2_fa3_compile",
        "mean_gradient_norm": 0.0959,
        "median_tokens_per_second": 140_000.0,
        "p10_tokens_per_second": 137_000.0,
        "peak_memory_bytes": 54_000_000_000,
        "profile": profile,
        "resume_canary_pass": True,
        "run_fingerprint": "a" * 64,
        "source_checkpoint": {
            key: source[key]
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
        "training_loss": 1.7367,
    }
    gate = {
        "authorization_sha256": "b" * 64,
        "comparisons": {
            "heldout_loss_regression_fraction": 0.00014,
            "median_throughput_gain_fraction": 0.384,
            "p10_throughput_gain_fraction": 0.363,
            "training_loss_regression_fraction": -0.0001,
        },
        "fa3_parity": {"status": "pass"},
        "gate": "lumi_4b_pre_segment3_checkpoint_gate_v0.1",
        "plan_sha256": "c" * 64,
        "production_training_authorized": False,
        "reasons": [],
        "results": [candidate],
        "schema_version": 1,
        "selected_profile": "fsdp_b2_fa3_compile",
        "source_checkpoint": source,
        "status": "pass",
    }
    gate["report_sha256"] = canonical_fingerprint(gate)
    gate_path = tmp_path / "gate.json"
    write_json(gate_path, gate)

    files = [
        {"bytes": 1564, "path": "manifest.sha256", "sha256": checkpoint_manifest}
    ] + [
        {
            "bytes": index + 1,
            "path": f"part-{index:02d}",
            "sha256": f"{index + 1:064x}",
        }
        for index in range(18)
    ]
    handoff = {
        "checkpoint": {
            "directory": "checkpoint-00063000",
            "sequence_cursor": 16_128_000,
            "step": 63_000,
            "tokens_seen": 66_060_288_000,
        },
        "files": files,
        "gate": "lumi_4b_checkpoint_handoff_v0.1",
        "run_fingerprint": source_run,
        "schema_version": 1,
        "source_commit": segment3.SEGMENT2_SOURCE_COMMIT,
        "status": "sealed",
        "world_size": 8,
    }
    handoff["handoff_sha256"] = canonical_fingerprint(handoff)
    monkeypatch.setattr(
        segment3,
        "SEGMENT2_HANDOFF_SHA256",
        handoff["handoff_sha256"],
    )
    handoff_path = tmp_path / "handoff.json"
    write_json(handoff_path, handoff)
    return {
        "base": base_path,
        "config": config,
        "corpus": corpus,
        "gate": gate_path,
        "handoff": handoff_path,
    }


def build_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    inputs = build_inputs(tmp_path, monkeypatch)
    return build_segment3_continuation_plan(
        base_plan_path=inputs["base"],
        checkpoint_gate_report_path=inputs["gate"],
        segment2_handoff_path=inputs["handoff"],
        config_path=inputs["config"],
        corpus_manifest_path=inputs["corpus"],
        checkpoint_path=(
            "/home/ubuntu/lumi-runs/lumi-4b-80b-segment-2/"
            "training/checkpoint-00063000"
        ),
        source_commit="d" * 40,
    )


def test_checkpoint_bound_plan_promotes_exact_fa3_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = build_plan(tmp_path, monkeypatch)

    assert plan["approval"] == {
        "full_4b_training": False,
        "paid_gpu": False,
    }
    assert plan["execution_profile"] == distributed_training.PAID_4B_SEGMENT3_PROFILE
    assert plan["segment"] == {
        "final_step_sequences": 242,
        "final_step_tokens": 991_232,
        "index": 3,
        "start_step": 63_000,
        "start_tokens": 66_060_288_000,
        "stop_step": 76_294,
        "stop_tokens": 80_000_000_000,
    }
    assert plan["run"]["attention_implementation"] == "flash_attention_3"
    assert plan["run"]["torch_compile"] is True
    assert plan["local_reuse"]["s3_access"] is False
    assert plan["local_input_paths"] == SEGMENT3_LOCAL_INPUT_PATHS
    assert plan["budget"]["estimated_measured_training_hours"] == pytest.approx(
        27.65815873
    )

    changed = copy.deepcopy(plan)
    changed["segment"]["stop_step"] += 1
    with pytest.raises(ValueError, match="plan hash changed"):
        validate_segment3_continuation_plan(changed)


def test_owner_authorization_requires_exact_sentence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = build_plan(tmp_path, monkeypatch)
    plan_path = tmp_path / "segment3-plan.json"
    write_json(plan_path, plan)
    text = owner_approval_text(plan, instance_ip="192.222.52.81")
    authorization = write_segment3_owner_authorization(
        plan_path=plan_path,
        output=tmp_path / "authorization.json",
        exact_owner_approval=text,
        instance_ip="192.222.52.81",
    )

    assert authorization["paid_gpu"] is True
    assert authorization["full_4b_training"] is True
    assert authorization["segment_index"] == 3
    assert authorization["approved_through_step"] == 76_294
    assert authorization["approved_token_cap"] == 80_000_000_000
    assert authorization["local_reuse"] == SEGMENT3_LOCAL_REUSE

    with pytest.raises(PermissionError, match="does not match"):
        write_segment3_owner_authorization(
            plan_path=plan_path,
            output=tmp_path / "bad-authorization.json",
            exact_owner_approval=text + " changed",
            instance_ip="192.222.52.81",
        )


def test_distributed_guard_accepts_only_exact_segment3_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = build_plan(tmp_path, monkeypatch)
    plan_path = tmp_path / "segment3-plan.json"
    write_json(plan_path, plan)
    text = owner_approval_text(plan, instance_ip="192.222.52.81")
    authorization_path = tmp_path / "authorization.json"
    write_segment3_owner_authorization(
        plan_path=plan_path,
        output=authorization_path,
        exact_owner_approval=text,
        instance_ip="192.222.52.81",
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
    monkeypatch.setattr(
        distributed_training,
        "PAID_4B_SEGMENT3_PROFILE",
        segment3.PAID_4B_SEGMENT3_PROFILE,
    )

    validated = _validate_paid_authorization(
        authorization_path,
        run_fingerprint=plan["run_fingerprint"],
        source_commit=plan["source_commit"],
        context=context,
        strategy="fsdp_full_shard",
        steps=76_294,
        target_tokens=80_000_000_000,
        global_tokens_per_step=1_048_576,
        per_rank_batch_size=2,
        gradient_accumulation_steps=16,
        gradient_checkpointing=False,
        optimizer_profile="legacy",
        attention_implementation="flash_attention_3",
        precision="bf16",
        torch_compile=True,
    )
    assert validated["segment_index"] == 3

    with pytest.raises(ValueError, match="runtime profile changed"):
        _validate_paid_authorization(
            authorization_path,
            run_fingerprint=plan["run_fingerprint"],
            source_commit=plan["source_commit"],
            context=context,
            strategy="fsdp_full_shard",
            steps=76_294,
            target_tokens=80_000_000_000,
            global_tokens_per_step=1_048_576,
            per_rank_batch_size=2,
            gradient_accumulation_steps=16,
            gradient_checkpointing=False,
            optimizer_profile="legacy",
            attention_implementation="flash_attention_2",
            precision="bf16",
            torch_compile=True,
        )

    tampered = json.loads(authorization_path.read_text(encoding="utf-8"))
    tampered["local_reuse"]["s3_access"] = True
    tampered_path = tmp_path / "tampered-authorization.json"
    write_json(tampered_path, tampered)
    with pytest.raises(ValueError, match="authorization hash changed"):
        _validate_paid_authorization(
            tampered_path,
            run_fingerprint=plan["run_fingerprint"],
            source_commit=plan["source_commit"],
            context=context,
            strategy="fsdp_full_shard",
            steps=76_294,
            target_tokens=80_000_000_000,
            global_tokens_per_step=1_048_576,
            per_rank_batch_size=2,
            gradient_accumulation_steps=16,
            gradient_checkpointing=False,
            optimizer_profile="legacy",
            attention_implementation="flash_attention_3",
            precision="bf16",
            torch_compile=True,
        )
