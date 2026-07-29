#!/usr/bin/env python3
"""Run the bounded exact-checkpoint baseline versus batch2_no_gc gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lumi.distributed_training import distributed_checkpoint_binding
from lumi.segment2_checkpoint_verification import (
    CHECKPOINT_STEPS,
    FIRST_PHASE_STOP_STEP,
    PAID_SEGMENT2_CHECKPOINT_GATE_MAX_STEPS,
    decide_checkpoint_gate,
    summarize_candidate,
    validate_checkpoint_gate_bundle,
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_canonical_json_bytes(value) + b"\n")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _remaining_seconds(deadline: datetime) -> float:
    return (deadline - datetime.now(UTC)).total_seconds()


def _terminate_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def _run_logged(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    deadline: datetime,
) -> None:
    if _remaining_seconds(deadline) <= 0:
        raise TimeoutError("checkpoint-gate deadline already passed")
    with log_path.open("wb") as handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while process.poll() is None:
            if _remaining_seconds(deadline) <= 0:
                _terminate_group(process)
                raise TimeoutError(
                    f"checkpoint-gate deadline reached during {log_path.name}"
                )
            time.sleep(2)
        if process.returncode:
            tail = log_path.read_text(
                encoding="utf-8",
                errors="replace",
            )[-20_000:]
            raise RuntimeError(
                f"checkpoint-gate command failed ({process.returncode}):\n{tail}"
            )


def _train_command(
    *,
    torchrun: Path,
    repo: Path,
    plan: dict[str, Any],
    authorization_path: Path,
    profile: dict[str, Any],
    output: Path,
    steps: int,
    source_checkpoint: Path | None,
    resume: bool,
) -> list[str]:
    manifest = Path(plan["manifest_path"])
    command = [
        str(torchrun),
        "--standalone",
        "--nproc-per-node",
        "8",
        "scripts/train_distributed.py",
        "--manifest",
        str(manifest),
        "--config",
        str(plan["model_config_path"]),
        "--output",
        str(output),
        "--steps",
        str(steps),
        "--strategy",
        "fsdp_full_shard",
        "--per-rank-batch-size",
        str(profile["per_rank_batch_size"]),
        "--gradient-accumulation-steps",
        str(profile["gradient_accumulation_steps"]),
        "--learning-rate",
        "0.0002",
        "--weight-decay",
        "0.1",
        "--max-grad-norm",
        "1.0",
        "--checkpoint-steps",
        str(CHECKPOINT_STEPS),
        "--keep-checkpoints",
        "1",
        "--seed",
        "7",
        "--lr-schedule",
        "cosine",
        "--warmup-steps",
        "1000",
        "--schedule-steps",
        "76294",
        "--min-learning-rate-ratio",
        "0.1",
        "--attention-implementation",
        "flash_attention_2",
        "--document-isolation",
        "--optimizer-profile",
        "grouped_adamw_beta2_0.95_v0.1",
        "--precision",
        "bf16",
        "--gradient-sync-mode",
        "last_microbatch",
        "--device",
        "cuda",
        "--backend",
        "nccl",
        "--timeout-seconds",
        "1800",
        "--authorization",
        str(authorization_path),
        "--source-commit",
        plan["source_commit"],
        "--final-corpus-root",
        str(manifest.parent / "shards"),
        "--skip-final-model-digest",
    ]
    if profile["gradient_checkpointing"]:
        command.append("--gradient-checkpointing")
    if source_checkpoint is not None:
        command.extend(["--resume-from-checkpoint", str(source_checkpoint)])
    if resume:
        command.append("--resume")
    return command


def _evaluate_candidate(
    *,
    python: Path,
    repo: Path,
    plan: dict[str, Any],
    checkpoint: Path,
    candidate: Path,
    deadline: datetime,
) -> Path:
    export = candidate / "checkpoint-export"
    _run_logged(
        [
            str(python),
            "scripts/export_distributed_checkpoint_hf.py",
            "--config",
            str(plan["model_config_path"]),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(export),
            "--dtype",
            "bfloat16",
        ],
        cwd=repo,
        log_path=candidate / "export.log",
        deadline=deadline,
    )
    evaluation = candidate / "evaluation"
    try:
        _run_logged(
            [
                str(python),
                "scripts/evaluate_checkpoint.py",
                "--plan",
                plan["evaluation"]["plan_path"],
                "--checkpoint",
                str(export),
                "--manifest",
                plan["evaluation"]["heldout_manifest_path"],
                "--tokenizer",
                plan["evaluation"]["tokenizer_path"],
                "--output",
                str(evaluation),
                "--attention-implementation",
                "flash_attention_2",
                "--document-isolation",
                "--device",
                "cuda",
                "--resume",
            ],
            cwd=repo,
            log_path=candidate / "evaluation.log",
            deadline=deadline,
        )
    finally:
        if export.exists():
            shutil.rmtree(export)
    return evaluation / "evaluation-report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    authorization = json.loads(args.authorization.read_text(encoding="utf-8"))
    validate_checkpoint_gate_bundle(
        plan=plan,
        authorization=authorization,
    )
    deadline = datetime.fromisoformat(authorization["deadline_utc"]).astimezone(UTC)
    if _remaining_seconds(deadline) <= 0:
        raise TimeoutError("checkpoint-gate authorization expired")
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=False)
    repo = args.repo.resolve()
    python = Path(sys.executable)
    torchrun = python.with_name("torchrun")
    source_checkpoint = Path(plan["source_checkpoint"]["path"]).resolve()
    source_manifest = source_checkpoint / "manifest.sha256"
    source_manifest_before = _sha256_file(source_manifest)
    if source_manifest_before != plan["source_checkpoint"]["manifest_sha256"]:
        raise ValueError("sealed Segment-1 checkpoint changed before gate start")

    started = datetime.now(UTC)
    operation = {
        "authorization_sha256": authorization["authorization_sha256"],
        "deadline_utc": deadline.isoformat(),
        "plan_sha256": plan["plan_sha256"],
        "source_checkpoint_manifest_sha256_before": source_manifest_before,
        "started_at": started.isoformat(),
        "status": "running",
    }
    _write_json_atomic(workspace / "operation.json", operation)

    telemetry_path = workspace / "gpu-telemetry.csv"
    telemetry_handle = telemetry_path.open("wb")
    telemetry = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,"
            "temperature.gpu,power.draw,ecc.errors.uncorrected.volatile.total",
            "--format=csv,noheader,nounits",
            "--loop=2",
        ],
        stdout=telemetry_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    results: list[dict[str, Any]] = []
    try:
        for profile in plan["profiles"]:
            if _remaining_seconds(deadline) < 12 * 60:
                raise TimeoutError(
                    "insufficient bounded time remains for another profile"
                )
            label = profile["label"]
            candidate = workspace / "candidates" / label
            candidate.mkdir(parents=True)
            training = candidate / "training"
            print(f"PROFILE_START label={label}", flush=True)
            _run_logged(
                _train_command(
                    torchrun=torchrun,
                    repo=repo,
                    plan=plan,
                    authorization_path=args.authorization.resolve(),
                    profile=profile,
                    output=training,
                    steps=FIRST_PHASE_STOP_STEP,
                    source_checkpoint=source_checkpoint,
                    resume=False,
                ),
                cwd=repo,
                log_path=candidate / "training.log",
                deadline=deadline,
            )
            phase_one_checkpoint = (
                training / f"checkpoint-{FIRST_PHASE_STOP_STEP:08d}"
            )
            _write_json_atomic(
                candidate / "phase-one-checkpoint-binding.json",
                distributed_checkpoint_binding(phase_one_checkpoint),
            )
            _run_logged(
                _train_command(
                    torchrun=torchrun,
                    repo=repo,
                    plan=plan,
                    authorization_path=args.authorization.resolve(),
                    profile=profile,
                    output=training,
                    steps=PAID_SEGMENT2_CHECKPOINT_GATE_MAX_STEPS,
                    source_checkpoint=None,
                    resume=True,
                ),
                cwd=repo,
                log_path=candidate / "resume.log",
                deadline=deadline,
            )
            final_checkpoint = (
                training
                / f"checkpoint-{PAID_SEGMENT2_CHECKPOINT_GATE_MAX_STEPS:08d}"
            )
            _write_json_atomic(
                candidate / "final-checkpoint-binding.json",
                distributed_checkpoint_binding(final_checkpoint),
            )
            evaluation_report = _evaluate_candidate(
                python=python,
                repo=repo,
                plan=plan,
                checkpoint=final_checkpoint,
                candidate=candidate,
                deadline=deadline,
            )
            source_manifest_after = _sha256_file(source_manifest)
            result = summarize_candidate(
                plan=plan,
                profile=profile,
                metrics_path=training / "distributed-metrics.jsonl",
                state_path=training / "distributed-training-state.json",
                evaluation_report_path=evaluation_report,
                source_manifest_sha256_after=source_manifest_after,
            )
            _write_json_atomic(candidate / "result.json", result)
            results.append(result)
            print(
                f"PROFILE_PASS label={label} "
                f"median_tps={result['median_tokens_per_second']:.1f} "
                f"p10_tps={result['p10_tokens_per_second']:.1f} "
                f"train_loss={result['training_loss']:.6f} "
                f"heldout_loss={result['heldout_loss']:.6f} "
                f"peak_hbm_gb={result['peak_memory_allocated_bytes'] / 1e9:.2f}",
                flush=True,
            )

        runtime_minutes = (datetime.now(UTC) - started).total_seconds() / 60
        report = decide_checkpoint_gate(
            plan=plan,
            authorization=authorization,
            results=results,
            observed_runtime_minutes=runtime_minutes,
        )
        final_source_manifest = _sha256_file(source_manifest)
        if final_source_manifest != source_manifest_before:
            raise RuntimeError("sealed Segment-1 checkpoint changed during gate")
        _write_json_atomic(workspace / "verification-report.json", report)
        operation.update(
            {
                "completed_at": datetime.now(UTC).isoformat(),
                "report_sha256": report["report_sha256"],
                "selected_profile": report["selected_profile"],
                "source_checkpoint_manifest_sha256_after": final_source_manifest,
                "status": "complete",
            }
        )
        _write_json_atomic(workspace / "operation.json", operation)
        print(
            f"GATE_PASS selected={report['selected_profile']} "
            f"median_gain={report['comparisons']['median_throughput_gain_fraction']:.4%} "
            f"p10_gain={report['comparisons']['p10_throughput_gain_fraction']:.4%} "
            f"report={report['report_sha256']}",
            flush=True,
        )
    except BaseException as error:
        operation.update(
            {
                "error": repr(error),
                "failed_at": datetime.now(UTC).isoformat(),
                "source_checkpoint_manifest_sha256_after": _sha256_file(
                    source_manifest
                ),
                "status": "failed",
            }
        )
        _write_json_atomic(workspace / "operation.json", operation)
        raise
    finally:
        _terminate_group(telemetry)
        telemetry_handle.close()


if __name__ == "__main__":
    main()
