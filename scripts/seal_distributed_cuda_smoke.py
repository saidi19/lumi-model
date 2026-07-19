#!/usr/bin/env python3
"""Validate and seal the bounded two-GPU Lumi-4B CUDA smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lumi.distributed_smoke import validate_owner_authorization
from lumi.distributed_training import verify_distributed_checkpoint

EXPECTED_STEP = 2
EXPECTED_TOKENS = 16_384
EXPECTED_CURSOR = 4
EXPECTED_PARAMETERS = 4_002_376_704


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _finite(record: dict[str, Any], key: str) -> float:
    value = float(record[key])
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite {key} at step {record.get('step')}")
    return value


def _log_value(text: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}=(.+)$", text, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError(f"run log is missing {key}")
    return match.group(1).strip()


def _telemetry(path: Path) -> list[dict[str, float | int | str]]:
    samples: list[dict[str, float | int | str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        index, timestamp, utilization, memory, power, temperature = parts
        samples.append(
            {
                "gpu_index": int(index),
                "memory_mib": float(memory),
                "power_watts": float(power),
                "temperature_c": float(temperature),
                "timestamp": timestamp,
                "utilization_percent": float(utilization),
            }
        )
    if not samples or {sample["gpu_index"] for sample in samples} != {0, 1}:
        raise RuntimeError("telemetry does not contain both smoke GPUs")
    return samples


def _write_manifest(root: Path, files: list[Path]) -> dict[str, Any]:
    entries = []
    for path in sorted(set(files)):
        if not path.is_file():
            raise FileNotFoundError(path)
        entries.append(
            {
                "bytes": path.stat().st_size,
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
            }
        )
    manifest = root / "manifest.sha256"
    with manifest.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(f"{entry['sha256']}  {entry['path']}\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "entries": len(entries),
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    required = (
        "setup.complete",
        "preflight.complete",
        "fresh.complete",
        "resume.complete",
        "plan.json",
        "tests.log",
        "ruff.log",
        "flash-varlen-cuda-report.json",
        "run.log",
        "fresh.log",
        "resume.log",
        "fresh-state.json",
        "fresh-metrics.jsonl",
        "gpu-telemetry.csv",
    )
    for name in required:
        if not (workspace / name).is_file():
            raise FileNotFoundError(f"missing distributed CUDA smoke evidence: {name}")
    if (workspace / "run.failed").exists():
        raise RuntimeError("distributed CUDA smoke has a failure marker")

    authorization = validate_owner_authorization(
        plan_path=workspace / "plan.json",
        authorization_path=args.authorization,
    )
    plan = json.loads((workspace / "plan.json").read_text(encoding="utf-8"))
    training = workspace / "training"
    state_path = training / "distributed-training-state.json"
    metrics_path = training / "distributed-metrics.jsonl"
    digest_path = training / "final-model-digest.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    expected_state = {
        "latest_checkpoint": "checkpoint-00000002",
        "sequence_cursor": EXPECTED_CURSOR,
        "status": "complete",
        "step": EXPECTED_STEP,
        "tokens_seen": EXPECTED_TOKENS,
    }
    for key, expected in expected_state.items():
        if state.get(key) != expected:
            raise RuntimeError(f"distributed CUDA state changed {key}")
    if state.get("run_fingerprint") != plan["run_fingerprint"]:
        raise RuntimeError("distributed CUDA state fingerprint differs from the plan")
    specification = state["run_specification"]
    if specification.get("source_commit") != plan["source_commit"]:
        raise RuntimeError("distributed CUDA state source commit changed")
    distributed = specification["distributed"]
    expected_distributed = {
        "backend": "nccl",
        "global_tokens_per_step": 8192,
        "gradient_sync_mode": "last_microbatch",
        "per_rank_batch_size": 1,
        "precision": "bf16",
        "strategy": "fsdp_full_shard",
        "torch_compile": False,
        "world_size": 2,
    }
    for key, expected in expected_distributed.items():
        if distributed.get(key) != expected:
            raise RuntimeError(f"distributed CUDA run changed {key}")
    scope = specification["authorization"]
    if scope.get("full_4b_training") is not False or scope.get("paid_gpu") is not True:
        raise RuntimeError("distributed CUDA state contains an unsafe authorization")
    if plan["model"]["parameters"] != EXPECTED_PARAMETERS:
        raise RuntimeError("distributed CUDA smoke did not use Lumi-4B")

    metrics = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [metric.get("step") for metric in metrics] != [1, 2]:
        raise RuntimeError("distributed CUDA metrics must contain exactly steps 1 and 2")
    for metric in metrics:
        if (
            metric.get("backend") != "nccl"
            or metric.get("device") != "cuda"
            or metric.get("strategy") != "fsdp_full_shard"
            or metric.get("world_size") != 2
        ):
            raise RuntimeError(f"invalid distributed metric topology at step {metric.get('step')}")
        for key in ("elapsed_seconds", "gradient_norm", "loss", "tokens_per_second"):
            _finite(metric, key)
        if metric.get("checkpoint_seconds") is None or _finite(metric, "checkpoint_seconds") <= 0:
            raise RuntimeError(f"missing checkpoint timing at step {metric.get('step')}")

    fresh = json.loads((workspace / "fresh-state.json").read_text(encoding="utf-8"))
    if fresh.get("step") != 1 or fresh.get("tokens_seen") != 8192:
        raise RuntimeError("fresh-process smoke boundary did not stop at step 1")
    fresh_metrics = [
        json.loads(line)
        for line in (workspace / "fresh-metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [metric.get("step") for metric in fresh_metrics] != [1]:
        raise RuntimeError("fresh-process metrics changed")

    checkpoint = training / state["latest_checkpoint"]
    checkpoint_state = verify_distributed_checkpoint(checkpoint)
    if checkpoint_state.get("step") != EXPECTED_STEP:
        raise RuntimeError("final distributed checkpoint step changed")
    digest = json.loads(digest_path.read_text(encoding="utf-8"))
    if digest.get("checkpoint") != checkpoint.name:
        raise RuntimeError("final model digest points to another checkpoint")
    if digest.get("checkpoint_manifest_sha256") != _sha256(checkpoint / "manifest.sha256"):
        raise RuntimeError("final model digest checkpoint manifest changed")

    parity = json.loads(
        (workspace / "flash-varlen-cuda-report.json").read_text(encoding="utf-8")
    )
    if parity.get("status") != "pass":
        raise RuntimeError("FlashAttention CUDA parity did not pass")
    telemetry = _telemetry(workspace / "gpu-telemetry.csv")
    run_log = (workspace / "run.log").read_text(encoding="utf-8")
    duration_seconds = int(_log_value(run_log, "run_duration_seconds"))
    if duration_seconds <= 0:
        raise RuntimeError("distributed CUDA run duration is invalid")
    estimated_cost = duration_seconds / 3600 * float(authorization["hourly_rate_usd"])

    evidence = workspace / "evidence"
    if evidence.exists():
        shutil.rmtree(evidence)
    evidence.mkdir()
    selected = [
        workspace / "setup.log",
        workspace / "preflight.log",
        workspace / "tests.log",
        workspace / "ruff.log",
        workspace / "flash-varlen-cuda-report.json",
        workspace / "plan.json",
        Path(args.authorization).resolve(),
        workspace / "run.log",
        workspace / "fresh.log",
        workspace / "resume.log",
        workspace / "fresh-state.json",
        workspace / "fresh-metrics.jsonl",
        workspace / "gpu-telemetry.csv",
        state_path,
        metrics_path,
        digest_path,
        checkpoint / "manifest.sha256",
        checkpoint / "distributed-training-state.json",
        workspace / "provenance/pip-freeze.txt",
        workspace / "provenance/source-commit.txt",
        workspace / "provenance/nvidia-smi-q.txt",
        workspace / "provenance/disk.txt",
        workspace / "provenance/memory.txt",
    ]
    copied = []
    for index, source in enumerate(selected):
        if not source.is_file():
            raise FileNotFoundError(source)
        target = evidence / f"{index:02d}-{source.name}"
        shutil.copy2(source, target)
        copied.append(target)

    per_gpu = {}
    for gpu_index in (0, 1):
        samples = [sample for sample in telemetry if sample["gpu_index"] == gpu_index]
        per_gpu[str(gpu_index)] = {
            "memory_mib_max": max(float(sample["memory_mib"]) for sample in samples),
            "samples": len(samples),
            "utilization_percent_median": statistics.median(
                float(sample["utilization_percent"]) for sample in samples
            ),
        }
    summary = {
        "authorization": {
            "full_4b_training": False,
            "max_runtime_minutes": authorization["max_runtime_minutes"],
            "paid_gpu": True,
            "plan_sha256": authorization["plan_sha256"],
            "provider": authorization["provider"],
            "instance_type": authorization["instance_type"],
        },
        "checkpoint": {
            "manifest_entries": len(
                (checkpoint / "manifest.sha256").read_text(encoding="utf-8").splitlines()
            ),
            "manifest_sha256": _sha256(checkpoint / "manifest.sha256"),
            "name": checkpoint.name,
            "verification": "all server checkpoint entries rehashed",
        },
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "final_model_digest": digest,
        "model_parameters": EXPECTED_PARAMETERS,
        "optimization": {
            "final_gradient_norm": _finite(metrics[-1], "gradient_norm"),
            "final_loss": _finite(metrics[-1], "loss"),
            "first_loss": _finite(metrics[0], "loss"),
        },
        "run_fingerprint": plan["run_fingerprint"],
        "runtime": {
            "completed_utc": _log_value(run_log, "run_completed_utc"),
            "duration_seconds": duration_seconds,
            "estimated_lifecycle_cost_usd": estimated_cost,
            "hourly_rate_usd": authorization["hourly_rate_usd"],
            "started_utc": _log_value(run_log, "run_started_utc"),
        },
        "source_commit": plan["source_commit"],
        "status": "pass",
        "telemetry": per_gpu,
        "throughput": {
            "step_1_tokens_per_second": _finite(metrics[0], "tokens_per_second"),
            "step_2_tokens_per_second": _finite(metrics[1], "tokens_per_second"),
        },
        "tokens_seen": EXPECTED_TOKENS,
        "world_size": 2,
    }
    summary_path = workspace / "summary.json"
    _write_json(summary_path, summary)
    seal = _write_manifest(workspace, [*copied, summary_path])
    (workspace / "seal.complete").touch()
    print(json.dumps({"seal": seal, "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
