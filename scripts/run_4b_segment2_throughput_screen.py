#!/usr/bin/env python3
"""Execute the owner-approved five-profile pre-Segment-2 throughput screen."""

from __future__ import annotations

import argparse
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

from lumi.pilot_training import _write_json_atomic
from lumi.pre_segment2_throughput_screen import (
    BASELINE_LABEL,
    MEASURED_STEP_END,
    RESUME_CANARY_STEP,
    decide_pre_segment2_screen,
    summarize_pre_segment2_candidate,
    validate_pre_segment2_screen_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _remaining_seconds(deadline: datetime) -> int:
    return max(0, int((deadline - datetime.now(UTC)).total_seconds()))


def _run_logged(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    deadline: datetime,
) -> int:
    remaining = _remaining_seconds(deadline)
    if remaining < 60:
        raise TimeoutError("throughput-screen lifecycle deadline reached")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"command_started_utc={_utc_now()}\n")
        log.write(f"command={json.dumps(command)}\n")
        log.flush()
        wrapped_command = [
            "timeout",
            "--signal=TERM",
            "--kill-after=20s",
            f"{max(30, remaining - 20)}s",
            *command,
        ]
        process = subprocess.Popen(
            wrapped_command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert process.stdout is not None
        try:
            while True:
                line = process.stdout.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log.write(line)
                    log.flush()
                if process.poll() is not None:
                    break
                if _remaining_seconds(deadline) < 30:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                    raise TimeoutError("throughput-screen lifecycle deadline reached")
            for line in process.stdout:
                sys.stdout.write(line)
                log.write(line)
            log.write(f"command_completed_utc={_utc_now()}\n")
            log.write(f"return_code={process.returncode}\n")
            return int(process.returncode)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)


def _classify_failure(log_path: Path) -> str:
    text = log_path.read_text(encoding="utf-8", errors="replace").lower()
    if "out of memory" in text or "cuda oom" in text:
        return "oom"
    if "torch._dynamo" in text or "torchinductor" in text or "compile" in text:
        return "compile_error"
    return "failed"


def _train_command(
    *,
    torchrun: Path,
    repo: Path,
    manifest: Path,
    output: Path,
    authorization: Path,
    source_commit: str,
    profile: dict[str, Any],
    steps: int,
    resume: bool,
) -> list[str]:
    command = [
        str(torchrun),
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=8",
        "scripts/train_distributed.py",
        "--manifest",
        str(manifest),
        "--config",
        "configs/lumi-4b-v0.2.json",
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
        str(MEASURED_STEP_END),
        "--keep-checkpoints",
        "1",
        "--seed",
        "7",
        "--lr-schedule",
        "cosine",
        "--warmup-steps",
        "8",
        "--schedule-steps",
        str(RESUME_CANARY_STEP),
        "--min-learning-rate-ratio",
        "0.1",
        "--attention-implementation",
        "flash_attention_2",
        "--document-isolation",
        "--optimizer-profile",
        "legacy",
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
        "--source-commit",
        source_commit,
        "--authorization",
        str(authorization),
        "--skip-final-model-digest",
    ]
    if profile["gradient_checkpointing"]:
        command.append("--gradient-checkpointing")
    if profile["torch_compile"]:
        command.append("--torch-compile")
    if resume:
        command.append("--resume")
    return command


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    workspace = args.workspace.resolve()
    plan, authorization = validate_pre_segment2_screen_files(
        plan_path=args.plan,
        authorization_path=args.authorization,
    )
    deadline = datetime.fromisoformat(authorization["deadline_utc"]).astimezone(UTC)
    if _remaining_seconds(deadline) < 15 * 60:
        raise RuntimeError("insufficient bounded lifecycle remains to start the screen")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    workspace.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(
        workspace / "run-boundary.json",
        {
            "authorization_sha256": authorization["authorization_sha256"],
            "deadline_utc": authorization["deadline_utc"],
            "full_4b_training": False,
            "plan_sha256": plan["plan_sha256"],
            "production_training": False,
            "segment_1_touched": False,
            "started_at": _utc_now(),
        },
    )
    torchrun = repo / ".venv" / "bin" / "torchrun"
    python = repo / ".venv" / "bin" / "python"
    if not torchrun.is_file() or not python.is_file():
        raise FileNotFoundError("throughput-screen virtual environment is incomplete")

    telemetry_path = workspace / "gpu-telemetry.csv"
    telemetry_handle = telemetry_path.open("a", encoding="utf-8")
    telemetry = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=index,timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu,ecc.errors.uncorrected.volatile.total",
            "--format=csv,noheader,nounits",
            "-l",
            "2",
        ],
        stdout=telemetry_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    results: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        for profile in plan["profiles"]:
            label = profile["label"]
            candidate = workspace / "candidates" / label
            training = candidate / "training"
            if candidate.exists():
                raise FileExistsError(f"candidate output already exists: {candidate}")
            candidate.mkdir(parents=True)
            if _remaining_seconds(deadline) < 12 * 60:
                raise TimeoutError("insufficient bounded lifecycle remains for another candidate")
            train_log = candidate / "training.log"
            code = _run_logged(
                _train_command(
                    torchrun=torchrun,
                    repo=repo,
                    manifest=args.manifest.resolve(),
                    output=training,
                    authorization=args.authorization.resolve(),
                    source_commit=plan["source_commit"],
                    profile=profile,
                    steps=MEASURED_STEP_END,
                    resume=False,
                ),
                cwd=repo,
                log_path=train_log,
                deadline=deadline,
            )
            if code == 0:
                code = _run_logged(
                    _train_command(
                        torchrun=torchrun,
                        repo=repo,
                        manifest=args.manifest.resolve(),
                        output=training,
                        authorization=args.authorization.resolve(),
                        source_commit=plan["source_commit"],
                        profile=profile,
                        steps=RESUME_CANARY_STEP,
                        resume=True,
                    ),
                    cwd=repo,
                    log_path=candidate / "resume.log",
                    deadline=deadline,
                )
            if code != 0:
                failed_log = candidate / (
                    "resume.log"
                    if (candidate / "resume.log").exists()
                    else "training.log"
                )
                result = {
                    "error": failed_log.read_text(
                        encoding="utf-8", errors="replace"
                    )[-16_000:],
                    "label": label,
                    "profile": profile,
                    "status": _classify_failure(failed_log),
                }
                _write_json_atomic(candidate / "result.json", result)
                results.append(result)
                if label == BASELINE_LABEL:
                    raise RuntimeError("throughput-screen baseline failed")
                continue

            checkpoint = training / f"checkpoint-{RESUME_CANARY_STEP:08d}"
            export = candidate / "checkpoint-export"
            export_code = _run_logged(
                [
                    str(python),
                    "scripts/export_distributed_checkpoint_hf.py",
                    "--config",
                    "configs/lumi-4b-v0.2.json",
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
            evaluation_code = (
                _run_logged(
                    [
                        str(python),
                        "scripts/evaluate_checkpoint.py",
                        "--plan",
                        str(args.plan.resolve()),
                        "--checkpoint",
                        str(export),
                        "--manifest",
                        str(args.heldout_manifest.resolve()),
                        "--tokenizer",
                        str(args.tokenizer.resolve()),
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
                if export_code == 0
                else export_code
            )
            if export.exists():
                shutil.rmtree(export)
            if export_code != 0 or evaluation_code != 0:
                raise RuntimeError(f"candidate {label} evaluation failed")
            result = summarize_pre_segment2_candidate(
                profile=profile,
                metrics_path=training / "distributed-metrics.jsonl",
                state_path=training / "distributed-training-state.json",
                evaluation_report_path=evaluation / "evaluation-report.json",
            )
            _write_json_atomic(candidate / "result.json", result)
            results.append(result)
            print(
                f"CANDIDATE_PASS label={label} "
                f"median_tps={result['metrics']['median_tokens_per_second']:.1f} "
                f"p10_tps={result['metrics']['p10_tokens_per_second']:.1f}",
                flush=True,
            )

        runtime_minutes = (time.monotonic() - started) / 60
        decision = decide_pre_segment2_screen(
            contract=contract,
            results=results,
            observed_runtime_minutes=runtime_minutes,
        )
        _write_json_atomic(workspace / "decision.json", decision)
        _write_json_atomic(
            workspace / "run.complete.json",
            {
                "completed_at": _utc_now(),
                "decision_sha256": decision["decision_sha256"],
                "final_checkpoint_bound_verification_required": True,
                "production_training_authorized": False,
                "status": "pass",
                "winner": decision["winner"],
            },
        )
        print(
            f"SCREEN_PASS winner={decision['winner']} "
            f"runtime_minutes={runtime_minutes:.2f}",
            flush=True,
        )
    finally:
        if telemetry.poll() is None:
            os.killpg(telemetry.pid, signal.SIGTERM)
            try:
                telemetry.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(telemetry.pid, signal.SIGKILL)
        telemetry_handle.close()


if __name__ == "__main__":
    main()
