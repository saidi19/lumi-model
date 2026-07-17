from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from lumi.scaling_run import (
    ISOLATION_POLICY,
    RUN_KIND,
    expected_training_identity,
    load_isolated_scaling_run_plan,
    plan_isolated_scaling_run,
    sha256_file,
    snapshot_model_checkpoint,
    validate_launch_authorization,
)

ROOT = Path(__file__).parents[1]
PLAN = ROOT / "configs" / "isolated-scaling-run-1b-v0.1.json"


def test_canonical_isolated_run_starts_fresh_and_covers_1p4b_tokens() -> None:
    plan = load_isolated_scaling_run_plan(PLAN)

    assert plan["run_kind"] == RUN_KIND
    assert plan["policy"] == {
        "attention_implementation": "flash_attention_2",
        "document_isolation": ISOLATION_POLICY,
        "initialization": "fresh_step_zero",
        "legacy_checkpoint_resume_allowed": False,
    }
    assert [item["step"] for item in plan["milestones"]] == [1, 2, 100, 7_630, 42_725]
    assert plan["milestones"][-1]["processed_tokens"] == 1_400_012_800
    identity = expected_training_identity(plan)
    assert identity["optimizer"]["document_isolation"] == ISOLATION_POLICY
    assert identity["optimizer"]["attention_implementation"] == "flash_attention_2"


def test_projection_is_measured_but_never_authorizes_launch() -> None:
    report = plan_isolated_scaling_run(PLAN, hourly_rate_usd=2.0)
    final = report["milestones"][-1]

    expected_hours = 1_400_012_800 / 9_809.508788719386 / 3_600
    assert report["status"] == "planning_only"
    assert report["authorization"]["paid_gpu_launch"] is False
    assert report["authorization"]["launch_authorization_present"] is False
    assert final["projection"]["compute_hours"] == pytest.approx(expected_hours)
    assert final["projection"]["lifecycle_cost_with_allowance_usd"] == pytest.approx(
        (expected_hours + 2.0) * 2.0
    )


def test_tampered_milestone_is_rejected(tmp_path) -> None:
    configs = tmp_path / "configs"
    shutil.copytree(ROOT / "configs", configs)
    plan_path = configs / PLAN.name
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["milestones"][-1]["processed_tokens"] -= 1
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ValueError, match="milestone unique_corpus changed"):
        load_isolated_scaling_run_plan(plan_path)


def test_launch_authorization_pins_plan_commit_and_owner_boundary(tmp_path) -> None:
    commit = "a" * 40
    authorization = {
        "approved_at_utc": "2026-07-18T00:00:00Z",
        "approved_through_step": 42_725,
        "full_4b_training": False,
        "hourly_rate_usd": 1.99,
        "instance_type": "gpu_1x_a100_sxm4",
        "owner_acknowledgement": (
            "I approve this paid Lumi-1B run through the recorded step."
        ),
        "plan_sha256": sha256_file(PLAN),
        "provider": "lambda",
        "run_kind": RUN_KIND,
        "schema_version": 1,
        "source_archive_sha256": "c" * 64,
        "source_commit": commit,
        "status": "owner_approved",
    }
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(json.dumps(authorization), encoding="utf-8")

    validated = validate_launch_authorization(
        plan_path=PLAN,
        authorization_path=authorization_path,
        source_commit=commit,
    )
    assert validated["approved_through_step"] == 42_725

    authorization["source_commit"] = "b" * 40
    authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
    with pytest.raises(ValueError, match="source commit"):
        validate_launch_authorization(
            plan_path=PLAN,
            authorization_path=authorization_path,
            source_commit=commit,
        )


def test_model_snapshot_excludes_optimizer_and_is_atomic(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-00000100"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    (checkpoint / "trainer.pt").write_bytes(b"optimizer")
    (checkpoint / "training-state.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "snapshot"

    snapshot = snapshot_model_checkpoint(checkpoint, output)

    assert snapshot["status"] == "complete"
    assert {item["name"] for item in snapshot["files"]} == {
        "config.json",
        "model.safetensors",
    }
    assert not (output / "trainer.pt").exists()
    assert (output / "snapshot-manifest.json").is_file()
    with pytest.raises(FileExistsError):
        snapshot_model_checkpoint(checkpoint, output)
