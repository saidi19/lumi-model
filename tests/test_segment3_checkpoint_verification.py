from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import torch

import lumi.distributed_training as distributed_training
import lumi.segment3_checkpoint_verification as checkpoint_gate
from lumi.distributed_training import (
    DISTRIBUTED_CHECKPOINT_SCHEMA_VERSION,
    DistributedContext,
    _validate_authorized_resume_boundary,
    _validate_paid_authorization,
    _write_checksum_manifest,
)
from lumi.production_training import FINAL_CORPUS_SHARD_MANIFEST_GATE
from scripts import run_4b_segment3_checkpoint_gate as checkpoint_gate_runner


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_checkpoint(root: Path) -> Path:
    checkpoint = root / "checkpoint-00063000"
    checkpoint.mkdir()
    state = {
        "best_loss": 1.60,
        "checkpoint_schema_version": DISTRIBUTED_CHECKPOINT_SCHEMA_VERSION,
        "last_loss": 1.69,
        "run_fingerprint": (
            checkpoint_gate.PAID_SEGMENT3_CHECKPOINT_GATE_SOURCE_RUN_FINGERPRINT
        ),
        "run_specification": {
            "distributed": {
                "backend": "nccl",
                "per_rank_batch_size": 2,
                "precision": "bf16",
                "strategy": "fsdp_full_shard",
                "torch_compile": False,
                "world_size": 8,
            },
            "optimizer": {
                "attention_implementation": "flash_attention_2",
                "batch_size": 2,
                "document_isolation": "position_reset_varlen_v0.1",
                "gradient_accumulation_steps": 16,
                "learning_rate": 0.0002,
                "lr_schedule": "cosine",
                "max_grad_norm": 1.0,
                "min_learning_rate_ratio": 0.1,
                "schedule_steps": 76_294,
                "warmup_steps": 1000,
                "weight_decay": 0.1,
            },
            "source_commit": checkpoint_gate.SEGMENT2_SOURCE_COMMIT,
        },
        "sequence_cursor": 16_128_000,
        "step": 63_000,
        "tokens_seen": 66_060_288_000,
    }
    (checkpoint / "distributed-training-state.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    (checkpoint / "model").mkdir()
    (checkpoint / "model" / "shard").write_bytes(b"model")
    for rank in range(8):
        (checkpoint / f"trainer-rank-{rank:05d}.pt").write_bytes(
            f"rank-{rank}".encode()
        )
    _write_checksum_manifest(checkpoint)
    return checkpoint


def _bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict, dict]:
    checkpoint = _source_checkpoint(tmp_path)
    source_manifest_sha256 = _sha256(checkpoint / "manifest.sha256")
    monkeypatch.setattr(
        checkpoint_gate,
        "PAID_SEGMENT3_CHECKPOINT_GATE_SOURCE_MANIFEST_SHA256",
        source_manifest_sha256,
    )
    monkeypatch.setattr(
        distributed_training,
        "PAID_SEGMENT3_CHECKPOINT_GATE_SOURCE_MANIFEST_SHA256",
        source_manifest_sha256,
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"gate": FINAL_CORPUS_SHARD_MANIFEST_GATE}),
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        checkpoint_gate,
        "FINAL_CORPUS_MANIFEST_SHA256",
        _sha256(manifest),
    )
    monkeypatch.setattr(checkpoint_gate, "MODEL_CONFIG_SHA256", _sha256(config))
    monkeypatch.setattr(
        checkpoint_gate,
        "validate_final_corpus_training_manifest",
        lambda *args, **kwargs: {"build_fingerprint": "b" * 64},
    )
    heldout = tmp_path / "heldout.json"
    heldout.write_text("{}", encoding="utf-8")
    tokenizer_root = tmp_path / "tokenizer"
    tokenizer_root.mkdir()
    tokenizer = tokenizer_root / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text("{}", encoding="utf-8")
    plan = checkpoint_gate.build_checkpoint_gate_plan(
        source_checkpoint=checkpoint,
        manifest_path=manifest,
        heldout_manifest_path=heldout,
        tokenizer_path=tokenizer,
        config_path=config,
        evaluation_plan_path=evaluation,
        source_commit="a" * 40,
        instance_boot_utc="2026-07-30T19:48:00+00:00",
        created_at=datetime(2026, 8, 4, 2, 0, tzinfo=UTC),
    )
    authorization = checkpoint_gate.build_checkpoint_gate_authorization(
        plan=plan,
        approved_at=datetime(2026, 8, 4, 2, 0, tzinfo=UTC),
    )
    return plan, authorization


def test_checkpoint_gate_binds_exact_segment2_source_and_profiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, authorization = _bundle(tmp_path, monkeypatch)

    checkpoint_gate.validate_checkpoint_gate_bundle(
        plan=plan,
        authorization=authorization,
    )

    assert tuple(profile["label"] for profile in plan["profiles"]) == (
        "segment2_baseline_fsdp_b2_fa2",
        "fsdp_b2_fa3_compile",
    )
    assert plan["source_checkpoint"]["step"] == 63_000
    assert plan["source_checkpoint"]["tokens"] == 66_060_288_000
    assert plan["source_checkpoint"]["read_only"] is True
    assert plan["profiles"][0]["attention_implementation"] == "flash_attention_2"
    assert plan["profiles"][0]["torch_compile"] is False
    assert plan["profiles"][1]["attention_implementation"] == "flash_attention_3"
    assert plan["profiles"][1]["torch_compile"] is True
    assert authorization["full_4b_training"] is False
    assert authorization["production_training"] is False
    assert authorization["scope"]["segment_3"] is False


def test_checkpoint_gate_rejects_segment2_source_profile_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _source_checkpoint(tmp_path)
    state_path = checkpoint / "distributed-training-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["run_specification"]["distributed"]["torch_compile"] = True
    state_path.write_text(json.dumps(state), encoding="utf-8")
    _write_checksum_manifest(checkpoint)
    monkeypatch.setattr(
        checkpoint_gate,
        "PAID_SEGMENT3_CHECKPOINT_GATE_SOURCE_MANIFEST_SHA256",
        _sha256(checkpoint / "manifest.sha256"),
    )

    with pytest.raises(ValueError, match="distributed profile changed"):
        checkpoint_gate._validate_source_checkpoint(checkpoint)


def test_checkpoint_gate_paid_authorization_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, authorization = _bundle(tmp_path, monkeypatch)
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(authorization), encoding="utf-8")
    fingerprint = authorization["allowed_run_fingerprints"][
        "fsdp_b2_fa3_compile"
    ]
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=8,
        backend="nccl",
        device=torch.device("cuda"),
    )
    monkeypatch.setattr(
        distributed_training,
        "_validate_production_cuda_runtime",
        lambda _: None,
    )

    observed = _validate_paid_authorization(
        path,
        run_fingerprint=fingerprint,
        source_commit="a" * 40,
        context=context,
        strategy="fsdp_full_shard",
        steps=checkpoint_gate.PAID_SEGMENT3_CHECKPOINT_GATE_MAX_STEPS,
        target_tokens=(
            checkpoint_gate.PAID_SEGMENT3_CHECKPOINT_GATE_MAX_STEPS * 1_048_576
        ),
        global_tokens_per_step=1_048_576,
        per_rank_batch_size=2,
        gradient_accumulation_steps=16,
        gradient_checkpointing=False,
        optimizer_profile="legacy",
    )

    assert observed["source_checkpoint"]["read_only"] is True
    changed = copy.deepcopy(authorization)
    changed["scope"]["segment_3"] = True
    changed["authorization_sha256"] = checkpoint_gate._fingerprint(
        changed, "authorization_sha256"
    )
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="scope changed"):
        _validate_paid_authorization(
            path,
            run_fingerprint=fingerprint,
            source_commit="a" * 40,
            context=context,
            strategy="fsdp_full_shard",
            steps=checkpoint_gate.PAID_SEGMENT3_CHECKPOINT_GATE_MAX_STEPS,
            target_tokens=(
                checkpoint_gate.PAID_SEGMENT3_CHECKPOINT_GATE_MAX_STEPS
                * 1_048_576
            ),
            global_tokens_per_step=1_048_576,
            per_rank_batch_size=2,
            gradient_accumulation_steps=16,
            gradient_checkpointing=False,
            optimizer_profile="legacy",
        )


def test_checkpoint_gate_runner_binds_compile_and_fa3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, _ = _bundle(tmp_path, monkeypatch)
    baseline, candidate = plan["profiles"]
    baseline_command = checkpoint_gate_runner._train_command(
        torchrun=Path("/torchrun"),
        plan=plan,
        authorization_path=tmp_path / "authorization.json",
        profile=baseline,
        output=tmp_path / "baseline",
        steps=63_020,
        source_checkpoint=Path(plan["source_checkpoint"]["path"]),
        resume=False,
    )
    candidate_command = checkpoint_gate_runner._train_command(
        torchrun=Path("/torchrun"),
        plan=plan,
        authorization_path=tmp_path / "authorization.json",
        profile=candidate,
        output=tmp_path / "candidate",
        steps=63_020,
        source_checkpoint=Path(plan["source_checkpoint"]["path"]),
        resume=False,
    )

    assert "--torch-compile" not in baseline_command
    assert baseline_command[baseline_command.index("--attention-implementation") + 1] == (
        "flash_attention_2"
    )
    assert "--torch-compile" in candidate_command
    assert candidate_command[
        candidate_command.index("--attention-implementation") + 1
    ] == "flash_attention_3"


def test_checkpoint_gate_resume_boundary_is_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, authorization = _bundle(tmp_path, monkeypatch)
    step = 63_020
    state = {
        "sequence_cursor": step * 256,
        "step": step,
        "tokens_seen": step * 1_048_576,
    }

    _validate_authorized_resume_boundary(
        authorization,
        state,
        global_tokens_per_step=1_048_576,
    )

    state["step"] = 63_025
    state["tokens_seen"] = 63_025 * 1_048_576
    state["sequence_cursor"] = 63_025 * 256
    with pytest.raises(ValueError, match="outside its boundary"):
        _validate_authorized_resume_boundary(
            authorization,
            state,
            global_tokens_per_step=1_048_576,
        )


def test_checkpoint_gate_evaluation_uses_verified_tokenizer_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, _ = _bundle(tmp_path, monkeypatch)
    commands: list[list[str]] = []

    def fake_run_logged(
        command: list[str],
        *,
        cwd: Path,
        log_path: Path,
        deadline: datetime,
    ) -> None:
        del cwd, log_path, deadline
        commands.append(command)

    monkeypatch.setattr(checkpoint_gate_runner, "_run_logged", fake_run_logged)
    checkpoint = tmp_path / "disposable-checkpoint"
    checkpoint.mkdir()
    candidate = tmp_path / "candidate-eval"
    candidate.mkdir()
    checkpoint_gate_runner._evaluate_candidate(
        python=Path("/python"),
        repo=tmp_path,
        plan=plan,
        checkpoint=checkpoint,
        candidate=candidate,
        deadline=datetime.now(UTC) + timedelta(minutes=1),
    )

    evaluation_command = commands[1]
    tokenizer_index = evaluation_command.index("--tokenizer") + 1
    assert evaluation_command[tokenizer_index] == plan["evaluation"][
        "tokenizer_root_path"
    ]


def test_checkpoint_gate_selects_only_safe_material_gain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, authorization = _bundle(tmp_path, monkeypatch)
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
        "heldout_loss": 2.05,
        "label": "segment2_baseline_fsdp_b2_fa2",
        "median_tokens_per_second": 102_000.0,
        "p10_tokens_per_second": 100_000.0,
        "peak_memory_bytes": 69_000_000_000,
        "resume_canary_pass": True,
        "source_checkpoint": source,
        "source_checkpoint_unchanged": True,
        "status": "pass",
        "training_loss": 1.70,
    }
    candidate = {
        **baseline,
        "heldout_loss": 2.051,
        "label": "fsdp_b2_fa3_compile",
        "median_tokens_per_second": 142_000.0,
        "p10_tokens_per_second": 140_000.0,
        "peak_memory_bytes": 54_000_000_000,
        "training_loss": 1.703,
    }

    report = checkpoint_gate.decide_checkpoint_gate(
        plan=plan,
        authorization=authorization,
        results=[baseline, candidate],
        fa3_parity={"status": "pass"},
        observed_runtime_minutes=35,
    )

    assert report["status"] == "pass"
    assert report["selected_profile"] == "fsdp_b2_fa3_compile"
    assert report["production_training_authorized"] is False


def test_checkpoint_gate_falls_back_on_loss_regression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, authorization = _bundle(tmp_path, monkeypatch)
    baseline = {
        "heldout_loss": 2.0,
        "label": "segment2_baseline_fsdp_b2_fa2",
        "median_tokens_per_second": 100_000.0,
        "p10_tokens_per_second": 98_000.0,
        "peak_memory_bytes": 69_000_000_000,
        "status": "pass",
        "training_loss": 1.7,
    }
    candidate = {
        **baseline,
        "heldout_loss": 2.1,
        "label": "fsdp_b2_fa3_compile",
        "median_tokens_per_second": 142_000.0,
        "p10_tokens_per_second": 140_000.0,
        "peak_memory_bytes": 54_000_000_000,
        "training_loss": 1.8,
    }

    report = checkpoint_gate.decide_checkpoint_gate(
        plan=plan,
        authorization=authorization,
        results=[baseline, candidate],
        fa3_parity={"status": "pass"},
        observed_runtime_minutes=35,
    )

    assert report["status"] == "safe_fallback"
    assert report["selected_profile"] == "segment2_baseline_fsdp_b2_fa2"
    assert "training_loss_regression" in report["reasons"]
    assert "heldout_loss_regression" in report["reasons"]
