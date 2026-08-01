#!/usr/bin/env python3
"""Run the bounded five-profile pre-Segment-3 throughput screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lumi.pilot_training import _write_json_atomic
from lumi.pre_segment3_throughput_screen import (
    BASELINE_LABEL,
    MEASURED_STEP_END,
    RESUME_CANARY_STEP,
    decide,
    summarize_candidate,
    validate_files,
)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _remaining_seconds(deadline: datetime) -> int:
    return max(0, int((deadline - datetime.now(UTC)).total_seconds()))


def _run_logged(
    command: list[str], *, cwd: Path, log_path: Path, deadline: datetime
) -> int:
    remaining = _remaining_seconds(deadline)
    if remaining < 60:
        raise TimeoutError("pre-Segment-3 lifecycle deadline reached")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"command_started_utc={_utc_now()}\n")
        log.write(f"command={json.dumps(command)}\n")
        log.flush()
        process = subprocess.Popen(
            [
                "timeout",
                "--signal=TERM",
                "--kill-after=30s",
                f"{max(30, remaining - 20)}s",
                *command,
            ],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
                if _remaining_seconds(deadline) < 20 and process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    raise TimeoutError("pre-Segment-3 lifecycle deadline reached")
            code = int(process.wait())
            log.write(f"command_completed_utc={_utc_now()}\nreturn_code={code}\n")
            return code
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)


def _classify_failure(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    if "out of memory" in text or "cuda oom" in text:
        return "oom"
    if "flashattention-3" in text or "flash_attn_interface" in text:
        return "fa3_error"
    if "torch._dynamo" in text or "torchinductor" in text or "inductorerror" in text:
        return "compile_error"
    if "timeout" in text or "lifecycle deadline" in text:
        return "deadline"
    return "failed"


def _train_command(
    *,
    torchrun: Path,
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
        profile["strategy"],
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
        "4",
        "--schedule-steps",
        str(RESUME_CANARY_STEP),
        "--min-learning-rate-ratio",
        "0.1",
        "--attention-implementation",
        profile["attention_implementation"],
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


def _verify_inputs(
    *,
    plan: dict[str, Any],
    manifest: Path,
    heldout_manifest: Path,
    tokenizer_root: Path,
) -> None:
    if _sha256(manifest) != plan["data"]["manifest_sha256"]:
        raise ValueError("bounded training manifest changed")
    if _sha256(heldout_manifest) != plan["data"]["heldout_manifest_sha256"]:
        raise ValueError("held-out manifest changed")
    for expected in plan["data"]["tokenizer_files"]:
        path = tokenizer_root / expected["name"]
        if (
            not path.is_file()
            or path.stat().st_size != expected["bytes"]
            or _sha256(path) != expected["sha256"]
        ):
            raise ValueError(f"tokenizer changed: {expected['name']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    workspace = args.workspace.resolve()
    plan, authorization = validate_files(
        plan_path=args.plan, authorization_path=args.authorization
    )
    deadline = datetime.fromisoformat(authorization["deadline_utc"]).astimezone(UTC)
    if _remaining_seconds(deadline) < 30 * 60:
        raise RuntimeError("insufficient bounded lifecycle remains to start the screen")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    _verify_inputs(
        plan=plan,
        manifest=args.manifest.resolve(),
        heldout_manifest=args.heldout_manifest.resolve(),
        tokenizer_root=args.tokenizer_root.resolve(),
    )
    workspace.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(
        workspace / "run-boundary.json",
        {
            "authorization_sha256": authorization["authorization_sha256"],
            "deadline_utc": authorization["deadline_utc"],
            "full_4b_training": False,
            "plan_sha256": plan["plan_sha256"],
            "production_training": False,
            "running_segment_2_access": False,
            "started_at": _utc_now(),
        },
    )
    torchrun = repo / ".venv" / "bin" / "torchrun"
    if not torchrun.is_file():
        raise FileNotFoundError("pre-Segment-3 virtual environment is incomplete")
    telemetry_handle = (workspace / "gpu-telemetry.csv").open("a", encoding="utf-8")
    telemetry = subprocess.Popen(
        [
            "nvidia-smi",
            (
                "--query-gpu=index,timestamp,utilization.gpu,memory.used,power.draw,"
                "temperature.gpu,ecc.errors.uncorrected.volatile.total"
            ),
            "--format=csv,noheader,nounits",
            "-l",
            "2",
        ],
        stdout=telemetry_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    results: list[dict[str, Any]] = []
    boot = datetime.fromisoformat(authorization["instance_boot_utc"]).astimezone(UTC)
    try:
        for profile in plan["profiles"]:
            label = profile["label"]
            candidate = workspace / "candidates" / label
            training = candidate / "training"
            if candidate.exists():
                raise FileExistsError(f"candidate already exists: {candidate}")
            candidate.mkdir(parents=True)
            if _remaining_seconds(deadline) < 10 * 60:
                result = {"label": label, "profile": profile, "status": "not_run_deadline"}
                _write_json_atomic(candidate / "result.json", result)
                results.append(result)
                continue
            if (
                profile["attention_implementation"] == "flash_attention_3"
                and not Path("/home/ubuntu/lumi-pre-segment3-screen/fa3.complete").is_file()
            ):
                result = {"label": label, "profile": profile, "status": "fa3_unavailable"}
                _write_json_atomic(candidate / "result.json", result)
                results.append(result)
                continue
            train_log = candidate / "training.log"
            code = _run_logged(
                _train_command(
                    torchrun=torchrun,
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
                failed_log = (
                    candidate / "resume.log"
                    if (candidate / "resume.log").exists()
                    else train_log
                )
                result = {
                    "error": failed_log.read_text(encoding="utf-8", errors="replace")[-16000:],
                    "label": label,
                    "profile": profile,
                    "status": _classify_failure(failed_log),
                }
                _write_json_atomic(candidate / "result.json", result)
                results.append(result)
                if label == BASELINE_LABEL:
                    raise RuntimeError("pre-Segment-3 baseline failed")
                print(f"CANDIDATE_FAIL label={label} status={result['status']}", flush=True)
                continue
            result = summarize_candidate(
                profile=profile,
                metrics_path=training / "distributed-metrics.jsonl",
                state_path=training / "distributed-training-state.json",
            )
            _write_json_atomic(candidate / "result.json", result)
            results.append(result)
            print(
                f"CANDIDATE_PASS label={label} "
                f"median_tps={result['metrics']['median_tokens_per_second']:.1f} "
                f"p10_tps={result['metrics']['p10_tokens_per_second']:.1f}",
                flush=True,
            )
        runtime_minutes = (datetime.now(UTC) - boot).total_seconds() / 60
        decision = decide(contract=contract, results=results, runtime_minutes=runtime_minutes)
        _write_json_atomic(workspace / "decision.json", decision)
        _write_json_atomic(
            workspace / "run.complete.json",
            {
                "completed_at": _utc_now(),
                "decision_sha256": decision["decision_sha256"],
                "exact_segment2_checkpoint_verification_required": True,
                "production_training_authorized": False,
                "status": "pass",
                "winner": decision["winner"],
            },
        )
        print(
            f"SCREEN_PASS winner={decision['winner']} runtime_minutes={runtime_minutes:.2f}",
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
