from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from lumi.scaling_run import (
    ISOLATION_POLICY,
    RUN_KIND,
    _isolated_manifest_paths,
    expected_training_identity,
    import_isolated_scaling_resume,
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


def test_cuda_setup_detects_a100_or_h100_architecture() -> None:
    for relative_path in (
        "scripts/setup_isolated_scaling_run.sh",
        "scripts/setup_packed_attention_cuda_smoke.sh",
    ):
        script = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "torch.cuda.get_device_capability()" in script
        assert "8.0|9.0" in script
        assert 'TORCH_CUDA_ARCH_LIST="$cuda_arch"' in script
        assert "TORCH_CUDA_ARCH_LIST=8.0" not in script


def test_isolated_manifest_excludes_the_live_outer_command_log(tmp_path) -> None:
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    (tmp_path / "run-command.log").write_text("still receiving stdout", encoding="utf-8")
    (tmp_path / "manifest.sha256").write_text("old manifest", encoding="utf-8")
    (tmp_path / "seal.complete").touch()
    (tmp_path / ".temporary").write_text("partial", encoding="utf-8")

    selected = [
        path.relative_to(tmp_path).as_posix()
        for path in _isolated_manifest_paths(tmp_path)
    ]

    assert selected == ["summary.json"]


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


def test_hash_bound_step100_resume_import_records_missing_metrics_prefix(tmp_path) -> None:
    plan = load_isolated_scaling_run_plan(PLAN)
    export = tmp_path / "export"
    checkpoint = export / "checkpoint-00000100"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    (checkpoint / "trainer.pt").write_bytes(b"optimizer")

    identity = expected_training_identity(plan)
    run_fingerprint = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    checkpoint_state = {
        "best_loss": 7.5,
        "last_loss": 7.5,
        "latest_checkpoint": "checkpoint-00000100",
        "run_fingerprint": run_fingerprint,
        "run_specification": {**identity, "locations": {}},
        "schema_version": 1,
        "sequence_cursor": 800,
        "status": "checkpoint",
        "step": 100,
        "tokens_seen": 3_276_800,
    }
    (checkpoint / "training-state.json").write_text(
        json.dumps(checkpoint_state), encoding="utf-8"
    )

    snapshot_dir = export / "milestone"
    snapshot_dir.mkdir()
    snapshot = {
        "files": [
            {
                "bytes": (checkpoint / name).stat().st_size,
                "name": name,
                "sha256": sha256_file(checkpoint / name),
            }
            for name in ("config.json", "model.safetensors")
        ],
        "schema_version": 1,
        "source_checkpoint": "checkpoint-00000100",
        "status": "complete",
    }
    (snapshot_dir / "snapshot-manifest.json").write_text(
        json.dumps(snapshot), encoding="utf-8"
    )

    evidence = export / "evidence"
    evaluation = evidence / "evaluations" / "step-00000100-isolated"
    evaluation.mkdir(parents=True)
    (evaluation / "evaluation-report.json").write_text("{}", encoding="utf-8")
    (evaluation / "evaluation-state.json").write_text("{}", encoding="utf-8")
    for name in (
        "milestone-smoke_fresh.complete",
        "milestone-smoke_fresh.log",
        "milestone-smoke_resume.complete",
        "milestone-smoke_resume.log",
        "milestone-calibration.complete",
        "milestone-calibration.log",
        "evaluation-step-00000100-isolated.complete",
    ):
        (evidence / name).write_text("", encoding="utf-8")

    checksum_manifest = export / "SHA256SUMS"
    files = sorted(path for path in export.rglob("*") if path.is_file())
    checksum_manifest.write_text(
        "".join(
            f"{sha256_file(path)}  ./{path.relative_to(export)}\n" for path in files
        ),
        encoding="utf-8",
    )
    commit = "a" * 40
    resume = {
        "checkpoint_export_manifest_sha256": sha256_file(checksum_manifest),
        "checkpoint_files": {
            name: sha256_file(checkpoint / name)
            for name in (
                "config.json",
                "model.safetensors",
                "trainer.pt",
                "training-state.json",
            )
        },
        "checkpoint_step": 100,
        "evaluation_report_sha256": sha256_file(
            evaluation / "evaluation-report.json"
        ),
        "metrics_prefix_available": False,
        "run_fingerprint": run_fingerprint,
        "snapshot_manifest_sha256": sha256_file(
            snapshot_dir / "snapshot-manifest.json"
        ),
    }
    authorization = {
        "approved_at_utc": "2026-07-18T00:00:00Z",
        "approved_through_step": 42_725,
        "full_4b_training": False,
        "hourly_rate_usd": 4.29,
        "instance_type": "gpu_1x_h100_sxm5",
        "owner_acknowledgement": (
            "I approve this paid Lumi-1B run through the recorded step."
        ),
        "plan_sha256": sha256_file(PLAN),
        "provider": "lambda",
        "resume": resume,
        "run_kind": RUN_KIND,
        "schema_version": 1,
        "source_archive_sha256": "c" * 64,
        "source_commit": commit,
        "status": "owner_approved",
    }
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
    workspace = tmp_path / "workspace"

    report = import_isolated_scaling_resume(
        plan_path=PLAN,
        authorization_path=authorization_path,
        source_commit=commit,
        source_export=export,
        workspace=workspace,
    )

    assert report["metrics_first_step"] == 101
    assert (workspace / "training" / "metrics.jsonl").read_text() == ""
    assert json.loads(
        (workspace / "training" / "training-state.json").read_text()
    )["status"] == "complete"
    assert (workspace / "resume-import.complete").is_file()
    assert import_isolated_scaling_resume(
        plan_path=PLAN,
        authorization_path=authorization_path,
        source_commit=commit,
        source_export=export,
        workspace=workspace,
    ) == report
