from __future__ import annotations

import json

import pytest

from lumi.pilot_training import (
    TRAINING_SCHEMA_VERSION,
    TRAINING_STATE_FILENAME,
    _optimizer_identity,
    _prune_checkpoints,
    scheduled_learning_rate,
)


def test_default_optimizer_identity_remains_legacy_checkpoint_compatible() -> None:
    identity = _optimizer_identity(
        batch_size=4,
        gradient_accumulation_steps=8,
        gradient_checkpointing=False,
        keep_checkpoints=None,
        learning_rate=3e-4,
        lr_schedule="constant",
        max_grad_norm=1.0,
        min_learning_rate_ratio=0.1,
        schedule_steps=None,
        warmup_steps=0,
        weight_decay=0.1,
    )

    assert identity == {
        "batch_size": 4,
        "gradient_accumulation_steps": 8,
        "learning_rate": 3e-4,
        "max_grad_norm": 1.0,
        "weight_decay": 0.1,
    }


def test_document_isolation_and_backend_are_resume_pinned() -> None:
    identity = _optimizer_identity(
        batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        keep_checkpoints=2,
        learning_rate=2e-4,
        lr_schedule="cosine",
        max_grad_norm=1.0,
        min_learning_rate_ratio=0.1,
        schedule_steps=42_725,
        warmup_steps=500,
        weight_decay=0.1,
        attention_implementation="flash_attention_2",
        document_isolation=True,
    )

    assert identity["attention_implementation"] == "flash_attention_2"
    assert identity["document_isolation"] == "position_reset_varlen_v0.1"


def test_cosine_learning_rate_has_linear_warmup_and_pinned_floor() -> None:
    common = {
        "base_learning_rate": 1.0,
        "schedule": "cosine",
        "warmup_steps": 2,
        "schedule_steps": 6,
        "min_learning_rate_ratio": 0.1,
    }

    rates = [scheduled_learning_rate(step=step, **common) for step in range(1, 7)]

    assert rates[0] == pytest.approx(0.5)
    assert rates[1] == pytest.approx(1.0)
    assert rates[-1] == pytest.approx(0.1)
    assert rates[1:] == sorted(rates[1:], reverse=True)


def test_constant_learning_rate_can_warm_up_without_a_horizon() -> None:
    common = {
        "base_learning_rate": 2e-4,
        "schedule": "constant",
        "warmup_steps": 2,
        "schedule_steps": None,
        "min_learning_rate_ratio": 0.1,
    }

    assert scheduled_learning_rate(step=1, **common) == pytest.approx(1e-4)
    assert scheduled_learning_rate(step=2, **common) == pytest.approx(2e-4)
    assert scheduled_learning_rate(step=200, **common) == pytest.approx(2e-4)


def test_checkpoint_pruning_keeps_newest_matching_run_only(tmp_path) -> None:
    for step in range(1, 5):
        checkpoint = tmp_path / f"checkpoint-{step:08d}"
        checkpoint.mkdir()
        (checkpoint / TRAINING_STATE_FILENAME).write_text(
            json.dumps(
                {
                    "run_fingerprint": "current",
                    "schema_version": TRAINING_SCHEMA_VERSION,
                    "step": step,
                }
            ),
            encoding="utf-8",
        )
    foreign = tmp_path / "checkpoint-99999999"
    foreign.mkdir()
    (foreign / TRAINING_STATE_FILENAME).write_text(
        json.dumps(
            {
                "run_fingerprint": "foreign",
                "schema_version": TRAINING_SCHEMA_VERSION,
                "step": 99_999_999,
            }
        ),
        encoding="utf-8",
    )

    removed = _prune_checkpoints(
        tmp_path,
        run_fingerprint="current",
        keep_checkpoints=2,
    )

    assert set(removed) == {"checkpoint-00000001", "checkpoint-00000002"}
    assert not (tmp_path / "checkpoint-00000001").exists()
    assert not (tmp_path / "checkpoint-00000002").exists()
    assert (tmp_path / "checkpoint-00000003").is_dir()
    assert (tmp_path / "checkpoint-00000004").is_dir()
    assert foreign.is_dir()
