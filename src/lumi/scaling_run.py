"""Fail-closed planning and evidence gates for the isolated Lumi-1B run."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import statistics
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .cloud_pilot import load_cloud_pilot_plan
from .pilot_training import scheduled_learning_rate
from .scaling_pilot import load_scaling_pilot_plan

ISOLATED_RUN_SCHEMA_VERSION = 1
LAUNCH_AUTHORIZATION_SCHEMA_VERSION = 1
RUN_KIND = "lumi_1b_isolated_1p4b_v0.1"
ISOLATION_POLICY = "position_reset_varlen_v0.1"
MILESTONE_NAMES = (
    "smoke_fresh",
    "smoke_resume",
    "calibration",
    "learning_gate",
    "unique_corpus",
)
MILESTONE_STEPS = (1, 2, 100, 7_630, 42_725)
MODEL_SNAPSHOT_EXCLUDES = {"trainer.pt", "training-state.json"}
MODEL_PARAMETERS = 1_050_981_888


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive number")
    return number


def _sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_resume_authorization(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("launch authorization resume must be an object")
    if value.get("checkpoint_step") != 100:
        raise ValueError("resume checkpoint_step must be the isolated calibration step")
    if value.get("metrics_prefix_available") is not False:
        raise ValueError("resume metrics_prefix_available must record the missing prefix")
    _sha256(
        value.get("checkpoint_export_manifest_sha256"),
        name="resume.checkpoint_export_manifest_sha256",
    )
    _sha256(value.get("evaluation_report_sha256"), name="resume.evaluation_report_sha256")
    _sha256(value.get("snapshot_manifest_sha256"), name="resume.snapshot_manifest_sha256")
    _sha256(value.get("run_fingerprint"), name="resume.run_fingerprint")
    checkpoint_files = value.get("checkpoint_files")
    required_files = {
        "config.json",
        "model.safetensors",
        "trainer.pt",
        "training-state.json",
    }
    if not isinstance(checkpoint_files, dict) or set(checkpoint_files) != required_files:
        raise ValueError("resume checkpoint_files must bind the exact resumable checkpoint")
    for name, digest in checkpoint_files.items():
        _sha256(digest, name=f"resume.checkpoint_files.{name}")
    return value


def _contract_path(plan_path: Path, specification: Any, *, name: str) -> Path:
    if not isinstance(specification, dict):
        raise ValueError(f"contracts.{name} must be an object")
    relative_value = specification.get("path")
    if not isinstance(relative_value, str) or not relative_value:
        raise ValueError(f"contracts.{name}.path must be a nonempty relative path")
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"contracts.{name}.path must stay inside the plan directory")
    path = plan_path.parent / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = _sha256(
        specification.get("sha256"), name=f"contracts.{name}.sha256"
    )
    if sha256_file(path) != expected:
        raise ValueError(f"contracts.{name} SHA-256 changed")
    return path


def expected_training_identity(plan: dict[str, Any]) -> dict[str, Any]:
    training = plan["training"]
    artifacts = plan["artifacts"]
    policy = plan["policy"]
    optimizer = {
        "attention_implementation": policy["attention_implementation"],
        "batch_size": training["batch_size"],
        "document_isolation": policy["document_isolation"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "gradient_checkpointing": training["gradient_checkpointing"],
        "keep_checkpoints": training["keep_checkpoints"],
        "learning_rate": training["learning_rate"],
        "lr_schedule": training["lr_schedule"],
        "max_grad_norm": training["max_grad_norm"],
        "min_learning_rate_ratio": training["min_learning_rate_ratio"],
        "schedule_steps": training["schedule_steps"],
        "warmup_steps": training["warmup_steps"],
        "weight_decay": training["weight_decay"],
    }
    return {
        "config": {"sha256": artifacts["model_config_sha256"]},
        "data": {
            "build_fingerprint": artifacts["training_build_fingerprint"],
            "manifest_sha256": artifacts["training_manifest_sha256"],
        },
        "optimizer": optimizer,
        "schema_version": 1,
        "seed": training["seed"],
    }


def load_isolated_scaling_run_plan(plan_path: str | Path) -> dict[str, Any]:
    """Load and cross-check the immutable 1B isolated-run preparation contract."""

    plan_path = Path(plan_path)
    plan = _read_json(plan_path)
    if plan.get("schema_version") != ISOLATED_RUN_SCHEMA_VERSION:
        raise ValueError("unsupported isolated scaling-run schema version")
    if plan.get("run_kind") != RUN_KIND:
        raise ValueError(f"run_kind must be {RUN_KIND}")
    expected_approval = {
        "full_1b_stage": False,
        "full_4b_training": False,
        "paid_gpu_launch": False,
    }
    if plan.get("approval") != expected_approval:
        raise ValueError("repository plan cannot pre-authorize a paid run")

    contracts = plan.get("contracts")
    if not isinstance(contracts, dict) or set(contracts) != {
        "cloud_pilot",
        "evaluation",
        "scaling_pilot",
    }:
        raise ValueError("isolated run contracts changed")
    cloud_path = _contract_path(plan_path, contracts["cloud_pilot"], name="cloud_pilot")
    scaling_path = _contract_path(
        plan_path, contracts["scaling_pilot"], name="scaling_pilot"
    )
    evaluation_path = _contract_path(
        plan_path, contracts["evaluation"], name="evaluation"
    )
    cloud = load_cloud_pilot_plan(cloud_path)
    scaling = load_scaling_pilot_plan(scaling_path)
    evaluation = _read_json(evaluation_path)

    artifacts = plan.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("artifacts must be an object")
    for field in (
        "heldout_build_fingerprint",
        "heldout_manifest_sha256",
        "model_config_sha256",
        "training_build_fingerprint",
        "training_manifest_sha256",
    ):
        _sha256(artifacts.get(field), name=f"artifacts.{field}")
    expected_artifacts = {
        "heldout_build_fingerprint": scaling["data"]["heldout"]["build_fingerprint"],
        "heldout_manifest_sha256": scaling["data"]["heldout"]["manifest_sha256"],
        "model_config_sha256": scaling["model"]["config_sha256"],
        "training_build_fingerprint": cloud["artifact"]["build_fingerprint"],
        "training_manifest_sha256": cloud["artifact"]["manifest_sha256"],
    }
    if artifacts != expected_artifacts:
        raise ValueError("isolated run artifacts do not match the sealed scaling contracts")

    policy = plan.get("policy")
    expected_policy = {
        "attention_implementation": "flash_attention_2",
        "document_isolation": ISOLATION_POLICY,
        "initialization": "fresh_step_zero",
        "legacy_checkpoint_resume_allowed": False,
    }
    if policy != expected_policy:
        raise ValueError("isolated attention policy changed")

    training = plan.get("training")
    if not isinstance(training, dict):
        raise ValueError("training must be an object")
    canonical_training = scaling["training"]
    canonical_optimizer = scaling["optimizer"]
    expected_training = {
        "batch_size": canonical_training["batch_size"],
        "checkpoint_steps": canonical_training["checkpoint_steps"],
        "gradient_accumulation_steps": canonical_training[
            "gradient_accumulation_steps"
        ],
        "gradient_checkpointing": canonical_training["gradient_checkpointing"],
        "keep_checkpoints": canonical_training["keep_checkpoints"],
        "learning_rate": canonical_optimizer["learning_rate"],
        "lr_schedule": canonical_optimizer["lr_schedule"],
        "max_grad_norm": canonical_optimizer["max_grad_norm"],
        "min_learning_rate_ratio": canonical_optimizer["min_learning_rate_ratio"],
        "schedule_steps": canonical_optimizer["schedule_steps"],
        "seed": canonical_optimizer["seed"],
        "tokens_per_step": canonical_training["tokens_per_step"],
        "warmup_steps": canonical_optimizer["warmup_steps"],
        "weight_decay": canonical_optimizer["weight_decay"],
    }
    if training != expected_training:
        raise ValueError("isolated run training settings changed")

    milestones = plan.get("milestones")
    if not isinstance(milestones, list) or len(milestones) != len(MILESTONE_STEPS):
        raise ValueError("isolated run milestones changed")
    for index, (item, name, step) in enumerate(
        zip(milestones, MILESTONE_NAMES, MILESTONE_STEPS, strict=True)
    ):
        if not isinstance(item, dict):
            raise ValueError(f"milestones[{index}] must be an object")
        expected = {
            "evaluate": step in {100, 7_630, 42_725},
            "name": name,
            "processed_tokens": step * training["tokens_per_step"],
            "step": step,
        }
        if item != expected:
            raise ValueError(f"milestone {name} changed")

    final_scaling_stage = scaling["stages"][-1]
    if milestones[-1]["step"] != final_scaling_stage["steps"]:
        raise ValueError("isolated final step no longer matches the scaling plan")
    if milestones[-1]["processed_tokens"] != final_scaling_stage["processed_tokens"]:
        raise ValueError("isolated final token count no longer matches the scaling plan")

    evaluation_policy = plan.get("evaluation")
    expected_evaluation = {
        "cached_resume_check": True,
        "legacy_final_attention_implementation": "eager",
        "legacy_final_policy": "separator_only_causal",
        "milestone_steps": [100, 7_630, 42_725],
        "primary_attention_implementation": "flash_attention_2",
        "primary_policy": ISOLATION_POLICY,
    }
    if evaluation_policy != expected_evaluation:
        raise ValueError("isolated held-out evaluation policy changed")
    if evaluation.get("schema_version") != 1:
        raise ValueError("unsupported held-out evaluation contract")

    reference = plan.get("reference_measurement")
    if not isinstance(reference, dict):
        raise ValueError("reference_measurement must be an object")
    _sha256(reference.get("evidence_manifest_sha256"), name="reference evidence hash")
    _positive_number(
        reference.get("isolated_tokens_per_second"),
        name="reference isolated_tokens_per_second",
    )
    _positive_number(
        reference.get("hourly_rate_usd"), name="reference hourly_rate_usd"
    )
    _positive_number(
        reference.get("non_training_allowance_hours"),
        name="reference non_training_allowance_hours",
    )
    expected_training_identity(plan)
    return plan


def plan_isolated_scaling_run(
    plan_path: str | Path, *, hourly_rate_usd: float | None = None
) -> dict[str, Any]:
    """Return a measured-input projection without authorizing any launch."""

    plan_path = Path(plan_path)
    plan = load_isolated_scaling_run_plan(plan_path)
    reference = plan["reference_measurement"]
    throughput = float(reference["isolated_tokens_per_second"])
    rate = (
        float(reference["hourly_rate_usd"])
        if hourly_rate_usd is None
        else _positive_number(hourly_rate_usd, name="hourly_rate_usd")
    )
    allowance = float(reference["non_training_allowance_hours"])
    milestones = []
    for item in plan["milestones"]:
        compute_hours = item["processed_tokens"] / throughput / 3_600
        milestones.append(
            {
                **item,
                "projection": {
                    "compute_cost_usd": compute_hours * rate,
                    "compute_hours": compute_hours,
                    "lifecycle_cost_with_allowance_usd": (compute_hours + allowance)
                    * rate,
                    "lifecycle_hours_with_allowance": compute_hours + allowance,
                },
            }
        )
    return {
        "authorization": {
            **plan["approval"],
            "launch_authorization_present": False,
        },
        "hourly_rate_usd": rate,
        "measured_tokens_per_second": throughput,
        "milestones": milestones,
        "non_training_allowance_hours": allowance,
        "plan_sha256": sha256_file(plan_path),
        "run_kind": RUN_KIND,
        "schema_version": 1,
        "status": "planning_only",
    }


def validate_launch_authorization(
    *,
    plan_path: str | Path,
    authorization_path: str | Path,
    source_commit: str,
) -> dict[str, Any]:
    """Require a separate owner-created artifact before any paid command runs."""

    plan_path = Path(plan_path)
    load_isolated_scaling_run_plan(plan_path)
    authorization = _read_json(Path(authorization_path))
    if authorization.get("schema_version") != LAUNCH_AUTHORIZATION_SCHEMA_VERSION:
        raise ValueError("unsupported launch-authorization schema")
    if authorization.get("status") != "owner_approved":
        raise ValueError("launch authorization is not owner-approved")
    if authorization.get("run_kind") != RUN_KIND:
        raise ValueError("launch authorization targets a different run")
    if authorization.get("plan_sha256") != sha256_file(plan_path):
        raise ValueError("launch authorization does not match the run plan")
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise ValueError("source commit must be a lowercase 40-character Git SHA")
    if authorization.get("source_commit") != source_commit:
        raise ValueError("launch authorization does not match the source commit")
    _sha256(
        authorization.get("source_archive_sha256"),
        name="source_archive_sha256",
    )
    approved_step = _positive_int(
        authorization.get("approved_through_step"), name="approved_through_step"
    )
    if approved_step not in MILESTONE_STEPS:
        raise ValueError("approved_through_step must be a declared milestone")
    if authorization.get("full_4b_training") is not False:
        raise ValueError("1B authorization cannot approve 4B training")
    for field in ("approved_at_utc", "instance_type", "provider"):
        if not isinstance(authorization.get(field), str) or not authorization[field]:
            raise ValueError(f"launch authorization {field} is required")
    _positive_number(authorization.get("hourly_rate_usd"), name="hourly_rate_usd")
    if authorization.get("owner_acknowledgement") != (
        "I approve this paid Lumi-1B run through the recorded step."
    ):
        raise ValueError("launch authorization acknowledgement changed")
    _validate_resume_authorization(authorization.get("resume"))
    return authorization


def snapshot_model_checkpoint(
    checkpoint_path: str | Path, output: str | Path
) -> dict[str, Any]:
    """Atomically preserve model-only milestone files before trainer pruning."""

    checkpoint_path = Path(checkpoint_path)
    output = Path(output)
    if not checkpoint_path.is_dir():
        raise NotADirectoryError(checkpoint_path)
    if output.exists():
        raise FileExistsError(output)
    files = sorted(
        path
        for path in checkpoint_path.iterdir()
        if path.is_file() and path.name not in MODEL_SNAPSHOT_EXCLUDES
    )
    if not (checkpoint_path / "config.json").is_file():
        raise FileNotFoundError(checkpoint_path / "config.json")
    if not any(path.name.startswith("model") and path.suffix == ".safetensors" for path in files):
        raise FileNotFoundError(f"checkpoint has no safetensors weights: {checkpoint_path}")
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    try:
        manifest = []
        for source in files:
            destination = temporary / source.name
            shutil.copy2(source, destination)
            manifest.append(
                {
                    "bytes": destination.stat().st_size,
                    "name": destination.name,
                    "sha256": sha256_file(destination),
                }
            )
        snapshot = {
            "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "files": manifest,
            "schema_version": 1,
            "source_checkpoint": checkpoint_path.name,
            "status": "complete",
        }
        _write_json_atomic(temporary / "snapshot-manifest.json", snapshot)
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return snapshot


def _link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def _verify_checksum_manifest(root: Path, manifest_path: Path) -> int:
    count = 0
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            digest, relative_value = line.split("  ", 1)
        except ValueError as error:
            raise ValueError(f"invalid checksum manifest line {line_number}") from error
        _sha256(digest, name=f"checksum manifest line {line_number}")
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"checksum path escapes export at line {line_number}")
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        if sha256_file(path) != digest:
            raise ValueError(f"checksum changed: {relative_value}")
        count += 1
    if count < 1:
        raise ValueError("checkpoint export checksum manifest is empty")
    return count


def import_isolated_scaling_resume(
    *,
    plan_path: str | Path,
    authorization_path: str | Path,
    source_commit: str,
    source_export: str | Path,
    workspace: str | Path,
) -> dict[str, Any]:
    """Import a hash-bound isolated step-100 checkpoint without inventing metrics."""

    plan_path = Path(plan_path)
    source_export = Path(source_export)
    workspace = Path(workspace)
    plan = load_isolated_scaling_run_plan(plan_path)
    authorization = validate_launch_authorization(
        plan_path=plan_path,
        authorization_path=authorization_path,
        source_commit=source_commit,
    )
    resume = _validate_resume_authorization(authorization.get("resume"))
    if resume is None:
        raise ValueError("launch authorization has no resume checkpoint")
    if (workspace / "resume-import.complete").exists():
        return _validate_resume_import(workspace, resume)

    for relative in (
        "training",
        "milestones/checkpoint-00000100",
        "evaluations/step-00000100-isolated",
    ):
        if (workspace / relative).exists():
            raise FileExistsError(workspace / relative)

    checksum_manifest = source_export / "SHA256SUMS"
    if sha256_file(checksum_manifest) != resume["checkpoint_export_manifest_sha256"]:
        raise ValueError("checkpoint export manifest does not match authorization")
    checksum_entries = _verify_checksum_manifest(source_export, checksum_manifest)

    step = resume["checkpoint_step"]
    checkpoint_name = f"checkpoint-{step:08d}"
    source_checkpoint = source_export / checkpoint_name
    for name, digest in resume["checkpoint_files"].items():
        if sha256_file(source_checkpoint / name) != digest:
            raise ValueError(f"authorized resume checkpoint changed: {name}")
    checkpoint_state = _read_json(source_checkpoint / "training-state.json")
    expected_identity = expected_training_identity(plan)
    specification = checkpoint_state.get("run_specification")
    if not isinstance(specification, dict):
        raise ValueError("resume checkpoint has no run specification")
    for key, expected in expected_identity.items():
        if specification.get(key) != expected:
            raise ValueError(f"resume checkpoint identity changed at {key}")
    if checkpoint_state.get("run_fingerprint") != resume["run_fingerprint"]:
        raise ValueError("resume checkpoint run fingerprint changed")
    expected_values = {
        "latest_checkpoint": checkpoint_name,
        "sequence_cursor": step
        * plan["training"]["batch_size"]
        * plan["training"]["gradient_accumulation_steps"],
        "step": step,
        "tokens_seen": step * plan["training"]["tokens_per_step"],
    }
    for key, expected in expected_values.items():
        if checkpoint_state.get(key) != expected:
            raise ValueError(f"resume checkpoint state mismatch for {key}")

    evaluation_source = (
        source_export
        / "evidence"
        / "evaluations"
        / "step-00000100-isolated"
    )
    if sha256_file(evaluation_source / "evaluation-report.json") != resume[
        "evaluation_report_sha256"
    ]:
        raise ValueError("resume evaluation report changed")
    snapshot_manifest_source = source_export / "milestone" / "snapshot-manifest.json"
    if sha256_file(snapshot_manifest_source) != resume["snapshot_manifest_sha256"]:
        raise ValueError("resume snapshot manifest changed")
    snapshot_manifest = _read_json(snapshot_manifest_source)
    if snapshot_manifest.get("source_checkpoint") != checkpoint_name:
        raise ValueError("resume snapshot targets a different checkpoint")

    workspace.mkdir(parents=True, exist_ok=True)
    temporary = workspace / f".resume-import.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        training = temporary / "training"
        training.mkdir()
        shutil.copytree(
            source_checkpoint,
            training / checkpoint_name,
            copy_function=_link_or_copy,
        )
        root_state = {**checkpoint_state, "status": "complete"}
        _write_json_atomic(training / "training-state.json", root_state)
        (training / "metrics.jsonl").touch()

        snapshot = temporary / "milestones" / checkpoint_name
        snapshot.mkdir(parents=True)
        for item in snapshot_manifest.get("files", []):
            name = item.get("name")
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError("resume snapshot manifest has an invalid filename")
            _link_or_copy(str(source_checkpoint / name), str(snapshot / name))
        shutil.copy2(snapshot_manifest_source, snapshot / "snapshot-manifest.json")

        shutil.copytree(
            evaluation_source,
            temporary / "evaluations" / "step-00000100-isolated",
        )
        evidence = source_export / "evidence"
        for name in (
            "milestone-smoke_fresh.complete",
            "milestone-smoke_fresh.log",
            "milestone-smoke_resume.complete",
            "milestone-smoke_resume.log",
            "milestone-calibration.complete",
            "milestone-calibration.log",
            "evaluation-step-00000100-isolated.complete",
        ):
            shutil.copy2(evidence / name, temporary / name)
        prefix_evidence = temporary / "resume-prefix-evidence"
        prefix_evidence.mkdir()
        shutil.copy2(checksum_manifest, prefix_evidence / "SHA256SUMS")

        report = {
            "authorization": resume,
            "checkpoint_state_derived": True,
            "checksum_entries": checksum_entries,
            "metrics_first_step": step + 1,
            "schema_version": 1,
            "status": "pass",
        }
        _write_json_atomic(temporary / "resume-import.json", report)
        (temporary / "resume-import.complete").touch()
        for path in temporary.iterdir():
            destination = workspace / path.name
            if destination.exists():
                raise FileExistsError(destination)
            path.replace(destination)
        temporary.rmdir()
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report


def validate_training_state(
    *, plan: dict[str, Any], training_output: str | Path, expected_step: int
) -> dict[str, Any]:
    training_output = Path(training_output)
    state = _read_json(training_output / "training-state.json")
    if expected_step not in MILESTONE_STEPS:
        raise ValueError("expected step is not a declared milestone")
    training = plan["training"]
    expected_identity = expected_training_identity(plan)
    specification = state.get("run_specification")
    if not isinstance(specification, dict):
        raise ValueError("training state has no run specification")
    for key, expected in expected_identity.items():
        if specification.get(key) != expected:
            raise ValueError(f"training run identity changed at {key}")
    if state.get("run_fingerprint") != _fingerprint(expected_identity):
        raise ValueError("training run fingerprint changed")
    expected_values = {
        "latest_checkpoint": f"checkpoint-{expected_step:08d}",
        "sequence_cursor": expected_step
        * training["batch_size"]
        * training["gradient_accumulation_steps"],
        "status": "complete",
        "step": expected_step,
        "tokens_seen": expected_step * training["tokens_per_step"],
    }
    for key, expected in expected_values.items():
        if state.get(key) != expected:
            raise ValueError(f"training state mismatch for {key}")
    checkpoint = training_output / expected_values["latest_checkpoint"]
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    return state


def _read_metrics(
    path: Path, *, training: dict[str, Any], first_step: int = 1
) -> list[dict[str, Any]]:
    if first_step < 1:
        raise ValueError("metrics first step must be positive")
    tokens_per_step = training["tokens_per_step"]
    metrics: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            expected_step = first_step + len(metrics)
            if item.get("step") != expected_step:
                raise ValueError(
                    f"metrics step changed at line {line_number}: "
                    f"{item.get('step')} != {expected_step}"
                )
            if item.get("tokens_seen") != expected_step * tokens_per_step:
                raise ValueError(f"metrics token count changed at step {expected_step}")
            if item.get("device") != "cuda":
                raise ValueError(f"non-CUDA training metric at step {expected_step}")
            for field in ("elapsed_seconds", "gradient_norm", "learning_rate", "loss"):
                value = item.get(field)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise ValueError(f"non-finite {field} at step {expected_step}")
            throughput = item.get("tokens_per_second")
            if (
                isinstance(throughput, bool)
                or not isinstance(throughput, (int, float))
                or not math.isfinite(float(throughput))
                or throughput <= 0
            ):
                raise ValueError(f"invalid throughput at step {expected_step}")
            expected_throughput = tokens_per_step / float(item["elapsed_seconds"])
            if not math.isclose(
                float(throughput), expected_throughput, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(f"throughput arithmetic changed at step {expected_step}")
            expected_rate = scheduled_learning_rate(
                base_learning_rate=training["learning_rate"],
                step=expected_step,
                schedule=training["lr_schedule"],
                warmup_steps=training["warmup_steps"],
                schedule_steps=training["schedule_steps"],
                min_learning_rate_ratio=training["min_learning_rate_ratio"],
            )
            if not math.isclose(
                float(item["learning_rate"]), expected_rate, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(f"learning rate changed at step {expected_step}")
            metrics.append(item)
    return metrics


def _validate_evaluation_report(
    path: Path,
    *,
    plan: dict[str, Any],
    isolated: bool,
) -> dict[str, Any]:
    report = _read_json(path)
    if report.get("status") != "complete":
        raise ValueError(f"evaluation report is incomplete: {path}")
    expected_attention = {
        "implementation": "flash_attention_2" if isolated else "eager",
        "policy": ISOLATION_POLICY if isolated else "separator_only_causal",
    }
    if report.get("attention") != expected_attention:
        raise ValueError(f"evaluation attention policy changed: {path}")
    artifacts = report.get("artifacts", {})
    if artifacts.get("heldout_build_fingerprint") != plan["artifacts"][
        "heldout_build_fingerprint"
    ]:
        raise ValueError(f"evaluation held-out artifact changed: {path}")
    heldout = artifacts.get("heldout_manifest", {})
    if heldout.get("sha256") != plan["artifacts"]["heldout_manifest_sha256"]:
        raise ValueError(f"evaluation held-out manifest changed: {path}")
    runtime = report.get("runtime", {})
    if runtime.get("device") != "cuda":
        raise ValueError(f"evaluation did not run on CUDA: {path}")
    if runtime.get("model_parameters") != MODEL_PARAMETERS:
        raise ValueError(f"evaluation model parameter count changed: {path}")
    loss = report.get("loss", {})
    for field in ("mean", "perplexity"):
        value = loss.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"evaluation {field} is invalid: {path}")
    return report


def _validate_snapshot(path: Path, *, checkpoint_name: str) -> dict[str, Any]:
    manifest = _read_json(path / "snapshot-manifest.json")
    if manifest.get("status") != "complete" or manifest.get("source_checkpoint") != checkpoint_name:
        raise ValueError(f"invalid model snapshot manifest: {path}")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"empty model snapshot manifest: {path}")
    for item in files:
        name = item.get("name")
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise ValueError(f"invalid model snapshot filename: {path}")
        file_path = path / name
        if file_path.stat().st_size != item["bytes"] or sha256_file(file_path) != item["sha256"]:
            raise ValueError(f"model snapshot file changed: {file_path}")
    return manifest


def _validate_resume_import(workspace: Path, resume: dict[str, Any]) -> dict[str, Any]:
    report = _read_json(workspace / "resume-import.json")
    if report.get("schema_version") != 1 or report.get("status") != "pass":
        raise ValueError("resume import did not pass")
    if report.get("authorization") != resume:
        raise ValueError("resume import does not match the launch authorization")
    if report.get("metrics_first_step") != resume["checkpoint_step"] + 1:
        raise ValueError("resume import metrics boundary changed")
    if report.get("checkpoint_state_derived") is not True:
        raise ValueError("resume import did not record the derived root state")
    return report


def seal_isolated_scaling_run(
    *,
    plan_path: str | Path,
    authorization_path: str | Path,
    source_commit: str,
    workspace: str | Path,
) -> dict[str, Any]:
    """Verify a completed paid run and seal every server artifact by hash."""

    plan_path = Path(plan_path)
    workspace = Path(workspace)
    plan = load_isolated_scaling_run_plan(plan_path)
    authorization = validate_launch_authorization(
        plan_path=plan_path,
        authorization_path=authorization_path,
        source_commit=source_commit,
    )
    resume = _validate_resume_authorization(authorization.get("resume"))
    if authorization["approved_through_step"] != MILESTONE_STEPS[-1]:
        raise ValueError("final seal requires authorization through the unique-corpus step")
    if (workspace / "seal.complete").exists():
        raise FileExistsError(workspace / "seal.complete")
    required = [
        workspace / "setup.complete",
        workspace / "preflight.complete",
        workspace / "cloud-preflight.json",
        workspace / "flash-varlen-cuda-report.json",
        workspace / "tests.log",
        workspace / "ruff.log",
        workspace / "authorization-verified.json",
        workspace / "cost-projection.json",
        workspace / "gpu-telemetry.csv",
    ]
    for milestone in plan["milestones"]:
        required.append(workspace / f"milestone-{milestone['name']}.complete")
    for step in plan["evaluation"]["milestone_steps"]:
        required.append(
            workspace / f"evaluation-step-{step:08d}-isolated.complete"
        )
    required.append(workspace / "evaluation-step-00042725-legacy.complete")
    if resume is not None:
        required.extend(
            [
                workspace / "resume-import.complete",
                workspace / "resume-import.json",
            ]
        )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    cloud = _read_json(workspace / "cloud-preflight.json")
    if cloud.get("status") != "pass" or cloud.get("runtime", {}).get("runtime_ready") is not True:
        raise ValueError("cloud preflight did not pass every runtime gate")
    parity = _read_json(workspace / "flash-varlen-cuda-report.json")
    if parity.get("schema_version") != 1 or parity.get("status") != "pass":
        raise ValueError("FlashAttention varlen CUDA parity gate did not pass")
    if parity.get("isolation", {}).get("max_absolute_error") != 0.0:
        raise ValueError("FlashAttention varlen isolation changed")
    forward_parity = parity.get("forward_parity", {})
    if float(forward_parity.get("cosine_similarity", 0.0)) < 0.9999:
        raise ValueError("FlashAttention varlen cosine parity is too low")
    if float(forward_parity.get("max_absolute_error", math.inf)) > float(
        forward_parity.get("atol", 0.0)
    ):
        raise ValueError("FlashAttention varlen absolute error exceeds tolerance")
    if _read_json(workspace / "authorization-verified.json") != authorization:
        raise ValueError("verified authorization artifact changed")
    projection = _read_json(workspace / "cost-projection.json")
    if projection.get("plan_sha256") != sha256_file(plan_path):
        raise ValueError("cost projection does not match the run plan")
    if float(projection.get("hourly_rate_usd", 0.0)) != float(
        authorization["hourly_rate_usd"]
    ):
        raise ValueError("cost projection does not match the authorized rate")
    if not (workspace / "gpu-telemetry.csv").read_text(encoding="utf-8").strip():
        raise ValueError("GPU telemetry is empty")

    final_step = MILESTONE_STEPS[-1]
    training_output = workspace / "training"
    state = validate_training_state(
        plan=plan, training_output=training_output, expected_step=final_step
    )
    metrics_first_step = 1
    resume_report = None
    if resume is not None:
        resume_report = _validate_resume_import(workspace, resume)
        metrics_first_step = resume_report["metrics_first_step"]
    metrics = _read_metrics(
        training_output / "metrics.jsonl",
        training=plan["training"],
        first_step=metrics_first_step,
    )
    expected_metrics = final_step - metrics_first_step + 1
    if len(metrics) != expected_metrics or metrics[-1]["step"] != final_step:
        raise ValueError("training metrics do not end at the unique-corpus step")

    evaluation_summaries = []
    for step in plan["evaluation"]["milestone_steps"]:
        checkpoint_name = f"checkpoint-{step:08d}"
        snapshot_path = workspace / "milestones" / checkpoint_name
        _validate_snapshot(snapshot_path, checkpoint_name=checkpoint_name)
        report_path = (
            workspace
            / "evaluations"
            / f"step-{step:08d}-isolated"
            / "evaluation-report.json"
        )
        report = _validate_evaluation_report(report_path, plan=plan, isolated=True)
        evaluation_summaries.append(
            {
                "loss": report["loss"]["mean"],
                "perplexity": report["loss"]["perplexity"],
                "step": step,
            }
        )
    legacy_report = _validate_evaluation_report(
        workspace
        / "evaluations"
        / f"step-{final_step:08d}-legacy"
        / "evaluation-report.json",
        plan=plan,
        isolated=False,
    )

    steady = [
        float(item["tokens_per_second"])
        for item in metrics
        if 100 <= int(item["step"]) < final_step
    ]
    if not steady:
        raise ValueError("training metrics have no steady-state window")
    summed_training_seconds = sum(float(item["elapsed_seconds"]) for item in metrics)
    hourly_rate = float(authorization["hourly_rate_usd"])
    best_loss = state.get("best_loss")
    if (
        isinstance(best_loss, bool)
        or not isinstance(best_loss, (int, float))
        or not math.isfinite(float(best_loss))
    ):
        raise ValueError("training state best loss is invalid")
    summary = {
        "authorization": {
            "approved_through_step": authorization["approved_through_step"],
            "full_4b_training": False,
            "hourly_rate_usd": hourly_rate,
            "instance_type": authorization["instance_type"],
            "provider": authorization["provider"],
        },
        "evaluation": {
            "isolated_milestones": evaluation_summaries,
            "legacy_final": {
                "loss": legacy_report["loss"]["mean"],
                "perplexity": legacy_report["loss"]["perplexity"],
            },
        },
        "model_parameters": MODEL_PARAMETERS,
        "plan_sha256": sha256_file(plan_path),
        "run_fingerprint": state["run_fingerprint"],
        "run_kind": RUN_KIND,
        "schema_version": 1,
        "source_archive_sha256": authorization["source_archive_sha256"],
        "source_commit": source_commit,
        "status": "pass",
        "training": {
            "best_loss": float(best_loss),
            "compute_cost_scope": f"steps_{metrics_first_step}_to_{final_step}",
            "estimated_step_compute_cost_usd": summed_training_seconds / 3_600 * hourly_rate,
            "final_loss": float(metrics[-1]["loss"]),
            "metrics_first_step": metrics_first_step,
            "median_tokens_per_second_steps_100_to_42724": statistics.median(steady),
            "processed_tokens": state["tokens_seen"],
            "step": state["step"],
            "summed_step_compute_seconds": summed_training_seconds,
        },
    }
    if resume_report is not None:
        summary["resume"] = resume_report
    summary_path = workspace / "summary.json"
    _write_json_atomic(summary_path, summary)

    selected = sorted(
        path
        for path in workspace.rglob("*")
        if path.is_file()
        and path.name not in {"manifest.sha256", "seal.complete"}
        and not path.name.startswith(".")
    )
    manifest_path = workspace / "manifest.sha256"
    temporary = manifest_path.with_name(f".{manifest_path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for path in selected:
            handle.write(f"{sha256_file(path)}  {path.relative_to(workspace)}\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(manifest_path)
    (workspace / "seal.complete").touch()
    return {
        **summary,
        "manifest_entries": len(selected),
        "manifest_sha256": sha256_file(manifest_path),
    }
