from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch

from lumi.distributed_training import (
    DISTRIBUTED_CHECKPOINT_SCHEMA_VERSION,
    PAID_SEGMENT2_CHECKPOINT_GATE_MAX_STEPS,
    DistributedContext,
    _validate_paid_authorization,
    _write_checksum_manifest,
)
from lumi.segment2_checkpoint_verification import (
    build_checkpoint_gate_bundle,
    decide_checkpoint_gate,
    validate_checkpoint_gate_bundle,
)


def _source_checkpoint(root: Path) -> Path:
    checkpoint = root / "checkpoint-00031500"
    checkpoint.mkdir()
    state = {
        "best_loss": 1.72,
        "checkpoint_schema_version": DISTRIBUTED_CHECKPOINT_SCHEMA_VERSION,
        "last_loss": 1.85,
        "run_fingerprint": "4" * 64,
        "run_specification": {"distributed": {"world_size": 8}},
        "sequence_cursor": 8_064_000,
        "step": 31_500,
        "tokens_seen": 33_030_144_000,
    }
    (checkpoint / "distributed-training-state.json").write_text(
        json.dumps(state),
        encoding="utf-8",
    )
    (checkpoint / "model").mkdir()
    (checkpoint / "model" / "shard").write_bytes(b"model")
    for rank in range(8):
        (checkpoint / f"trainer-rank-{rank:05d}.pt").write_bytes(
            f"rank-{rank}".encode()
        )
    _write_checksum_manifest(checkpoint)
    return checkpoint


def _bundle(tmp_path: Path) -> tuple[dict, dict]:
    checkpoint = _source_checkpoint(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"build_fingerprint": "b" * 64}),
        encoding="utf-8",
    )
    heldout = tmp_path / "heldout.json"
    heldout.write_text("{}", encoding="utf-8")
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text("{}", encoding="utf-8")
    return build_checkpoint_gate_bundle(
        source_checkpoint=checkpoint,
        manifest_path=manifest,
        heldout_manifest_path=heldout,
        tokenizer_path=tokenizer,
        config_path=config,
        evaluation_plan_path=evaluation,
        source_commit="a" * 40,
        instance_boot_utc="2026-07-23T22:56:00+00:00",
        approved_at=datetime(2026, 7, 29, 18, 0, tzinfo=UTC),
    )


def test_checkpoint_gate_bundle_binds_exact_source_and_profiles(
    tmp_path: Path,
) -> None:
    plan, authorization = _bundle(tmp_path)

    validate_checkpoint_gate_bundle(
        plan=plan,
        authorization=authorization,
    )

    assert tuple(profile["label"] for profile in plan["profiles"]) == (
        "baseline_b1_gc",
        "batch2_no_gc",
    )
    assert plan["source_checkpoint"]["step"] == 31_500
    assert authorization["full_4b_training"] is False
    assert authorization["scope"]["checkpoint_modification"] is False


def test_checkpoint_gate_bundle_rejects_scope_expansion(tmp_path: Path) -> None:
    plan, authorization = _bundle(tmp_path)
    changed = copy.deepcopy(authorization)
    changed["scope"]["production_training"] = True

    with pytest.raises(ValueError, match="authorization hash"):
        validate_checkpoint_gate_bundle(plan=plan, authorization=changed)


def test_checkpoint_gate_paid_authorization_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, authorization = _bundle(tmp_path)
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(authorization), encoding="utf-8")
    fingerprint = authorization["allowed_run_fingerprints"]["batch2_no_gc"]
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=8,
        backend="nccl",
        device=torch.device("cuda"),
    )
    monkeypatch.setattr(
        "lumi.distributed_training._validate_production_cuda_runtime",
        lambda _: None,
    )

    observed = _validate_paid_authorization(
        path,
        run_fingerprint=fingerprint,
        source_commit="a" * 40,
        context=context,
        strategy="fsdp_full_shard",
        steps=PAID_SEGMENT2_CHECKPOINT_GATE_MAX_STEPS,
        target_tokens=PAID_SEGMENT2_CHECKPOINT_GATE_MAX_STEPS * 1_048_576,
        global_tokens_per_step=1_048_576,
    )

    assert observed["source_checkpoint"]["read_only"] is True
    changed = copy.deepcopy(authorization)
    changed["scope"]["off_instance_export"] = True
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="scope changed"):
        _validate_paid_authorization(
            path,
            run_fingerprint=fingerprint,
            source_commit="a" * 40,
            context=context,
            strategy="fsdp_full_shard",
            steps=PAID_SEGMENT2_CHECKPOINT_GATE_MAX_STEPS,
            target_tokens=PAID_SEGMENT2_CHECKPOINT_GATE_MAX_STEPS
            * 1_048_576,
            global_tokens_per_step=1_048_576,
        )


def test_checkpoint_gate_selects_only_safe_material_gain(tmp_path: Path) -> None:
    plan, authorization = _bundle(tmp_path)
    source = {
        key: plan["source_checkpoint"][key]
        for key in (
            "manifest_sha256",
            "run_fingerprint",
            "sequence_cursor",
            "step",
            "tokens",
        )
    }
    baseline = {
        "checkpoint_restore_pass": True,
        "heldout_loss": 2.2,
        "label": "baseline_b1_gc",
        "median_tokens_per_second": 70_000.0,
        "p10_tokens_per_second": 68_000.0,
        "peak_memory_allocated_bytes": 19_000_000_000,
        "resume_canary_pass": True,
        "source_checkpoint": source,
        "source_checkpoint_unchanged": True,
        "status": "pass",
        "training_loss": 1.9,
    }
    candidate = {
        **baseline,
        "heldout_loss": 2.201,
        "label": "batch2_no_gc",
        "median_tokens_per_second": 102_000.0,
        "p10_tokens_per_second": 99_000.0,
        "peak_memory_allocated_bytes": 38_000_000_000,
        "training_loss": 1.905,
    }

    report = decide_checkpoint_gate(
        plan=plan,
        authorization=authorization,
        results=[baseline, candidate],
        observed_runtime_minutes=30,
    )

    assert report["status"] == "pass"
    assert report["selected_profile"] == "batch2_no_gc"
