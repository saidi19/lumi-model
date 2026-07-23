#!/usr/bin/env python3
"""Validate and seal the bounded 4/8-H100 Lumi-4B learning pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import statistics
from pathlib import Path
from typing import Any

from lumi.distributed_learning_pilot import validate_owner_authorization
from lumi.distributed_training import verify_distributed_checkpoint

EXPECTED_FINAL_STEP = 80
EXPECTED_FINAL_TOKENS = 83_886_080
EXPECTED_FINAL_CURSOR = 20_480
EXPECTED_FRESH_STEP = 10
EXPECTED_FRESH_TOKENS = 10_485_760
EXPECTED_FRESH_CURSOR = 2_560
EXPECTED_PARAMETERS = 4_002_376_704
EXPECTED_HELDOUT_SEQUENCES = 2_441
EXPECTED_HELDOUT_TOKENS = 4_999_168


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


def _telemetry(path: Path, *, world_size: int) -> list[dict[str, float | int | str]]:
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
    if not samples or {sample["gpu_index"] for sample in samples} != set(
        range(world_size)
    ):
        raise RuntimeError("learning-pilot telemetry does not contain every GPU")
    return samples


def _evaluation(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        raise RuntimeError(f"learning-pilot evaluation is incomplete: {path}")
    runtime = report["runtime"]
    if runtime.get("sequences") != EXPECTED_HELDOUT_SEQUENCES:
        raise RuntimeError("learning-pilot held-out sequence count changed")
    if runtime.get("input_tokens") != EXPECTED_HELDOUT_TOKENS:
        raise RuntimeError("learning-pilot held-out token count changed")
    if runtime.get("model_parameters") != EXPECTED_PARAMETERS:
        raise RuntimeError("learning-pilot evaluation model parameter count changed")
    if report.get("attention") != {
        "implementation": "flash_attention_2",
        "policy": "position_reset_varlen_v0.1",
    }:
        raise RuntimeError("learning-pilot evaluation attention policy changed")
    loss = float(report["loss"]["mean"])
    perplexity = float(report["loss"]["perplexity"])
    if not math.isfinite(loss) or not math.isfinite(perplexity):
        raise RuntimeError("learning-pilot held-out evaluation is non-finite")
    if len(report.get("generation", [])) != 4:
        raise RuntimeError("learning-pilot fixed-prompt generation count changed")
    return report


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
        "data/training-fixture/fixture-report.json",
        "tests.log",
        "ruff.log",
        "flash-varlen-cuda-report.json",
        "run.log",
        "runtime-used-seconds",
        "fresh.log",
        "resume.log",
        "fresh-state.json",
        "fresh-metrics.jsonl",
        "fresh-checkpoint-manifest.sha256",
        "fresh-checkpoint-state.json",
        "evaluation-step-10-export-report.json",
        "evaluation-step-80-export-report.json",
        "gpu-telemetry.csv",
    )
    for name in required:
        if not (workspace / name).is_file():
            raise FileNotFoundError(f"missing learning-pilot evidence: {name}")
    if (workspace / "run.failed").exists():
        raise RuntimeError("learning pilot has a failure marker")

    authorization = validate_owner_authorization(
        plan_path=workspace / "plan.json",
        authorization_path=args.authorization,
    )
    plan = json.loads((workspace / "plan.json").read_text(encoding="utf-8"))
    world_size = int(plan["selected_topology"]["world_size"])
    accumulation = int(plan["selected_topology"]["gradient_accumulation_steps"])
    if (world_size, accumulation) not in {(4, 64), (8, 32)}:
        raise RuntimeError("learning-pilot topology changed")
    runtime = plan["runtime"]
    if (
        runtime.get("cuda_available") is not True
        or runtime.get("cuda_device_count", 0) < world_size
        or runtime.get("flash_attention_version") != "2.8.3"
        or runtime.get("host_memory_bytes", 0) < 128_000_000_000
        or runtime.get("manifest_volume_free_bytes", 0) < 300_000_000_000
    ):
        raise RuntimeError("learning-pilot CUDA runtime changed")
    gpu_names = runtime.get("gpu_names")
    gpu_capabilities = runtime.get("cuda_capabilities")
    gpu_memory = runtime.get("gpu_total_memory_bytes")
    if (
        not isinstance(gpu_names, list)
        or len(gpu_names) != world_size
        or len(set(gpu_names)) != 1
        or any("H100" not in name.upper() for name in gpu_names)
        or gpu_capabilities != [[9, 0]] * world_size
        or not isinstance(gpu_memory, list)
        or len(gpu_memory) != world_size
        or any(not 80_000_000_000 <= value <= 90_000_000_000 for value in gpu_memory)
    ):
        raise RuntimeError("learning-pilot hardware differs from 80 GB H100")

    training = workspace / "training"
    state_path = training / "distributed-training-state.json"
    metrics_path = training / "distributed-metrics.jsonl"
    digest_path = training / "final-model-digest.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    expected_state = {
        "latest_checkpoint": "checkpoint-00000080",
        "sequence_cursor": EXPECTED_FINAL_CURSOR,
        "status": "complete",
        "step": EXPECTED_FINAL_STEP,
        "tokens_seen": EXPECTED_FINAL_TOKENS,
    }
    for key, expected in expected_state.items():
        if state.get(key) != expected:
            raise RuntimeError(f"learning-pilot final state changed {key}")
    if state.get("run_fingerprint") != plan["run_fingerprint"]:
        raise RuntimeError("learning-pilot state fingerprint differs from the plan")
    specification = state["run_specification"]
    if specification.get("source_commit") != plan["source_commit"]:
        raise RuntimeError("learning-pilot source commit changed")
    distributed = specification["distributed"]
    expected_distributed = {
        "backend": "nccl",
        "global_tokens_per_step": 1_048_576,
        "gradient_sync_mode": "last_microbatch",
        "per_rank_batch_size": 1,
        "precision": "bf16",
        "strategy": "fsdp_full_shard",
        "torch_compile": False,
        "world_size": world_size,
    }
    for key, expected in expected_distributed.items():
        if distributed.get(key) != expected:
            raise RuntimeError(f"learning-pilot distributed state changed {key}")
    scope = specification["authorization"]
    if scope.get("full_4b_training") is not False or scope.get("paid_gpu") is not True:
        raise RuntimeError("learning-pilot state contains an unsafe authorization")

    fresh = json.loads((workspace / "fresh-state.json").read_text(encoding="utf-8"))
    fresh_expected = {
        "sequence_cursor": EXPECTED_FRESH_CURSOR,
        "step": EXPECTED_FRESH_STEP,
        "tokens_seen": EXPECTED_FRESH_TOKENS,
    }
    for key, expected in fresh_expected.items():
        if fresh.get(key) != expected:
            raise RuntimeError(f"learning-pilot fresh boundary changed {key}")
    fresh_checkpoint = json.loads(
        (workspace / "fresh-checkpoint-state.json").read_text(encoding="utf-8")
    )
    if fresh_checkpoint.get("step") != EXPECTED_FRESH_STEP:
        raise RuntimeError("learning-pilot fresh checkpoint step changed")

    metrics = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [metric.get("step") for metric in metrics] != list(
        range(1, EXPECTED_FINAL_STEP + 1)
    ):
        raise RuntimeError("learning-pilot metrics are incomplete or out of order")
    for metric in metrics:
        if (
            metric.get("backend") != "nccl"
            or metric.get("device") != "cuda"
            or metric.get("strategy") != "fsdp_full_shard"
            or metric.get("world_size") != world_size
            or metric.get("global_tokens_per_step") != 1_048_576
        ):
            raise RuntimeError(f"invalid learning-pilot metric at step {metric.get('step')}")
        for key in ("elapsed_seconds", "gradient_norm", "loss", "tokens_per_second"):
            _finite(metric, key)
    fresh_metrics = [
        json.loads(line)
        for line in (workspace / "fresh-metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [metric.get("step") for metric in fresh_metrics] != list(
        range(1, EXPECTED_FRESH_STEP + 1)
    ):
        raise RuntimeError("learning-pilot fresh metrics changed")

    final_checkpoint = training / state["latest_checkpoint"]
    checkpoint_state = verify_distributed_checkpoint(final_checkpoint)
    if checkpoint_state.get("step") != EXPECTED_FINAL_STEP:
        raise RuntimeError("learning-pilot final checkpoint step changed")
    digest = json.loads(digest_path.read_text(encoding="utf-8"))
    if digest.get("checkpoint") != final_checkpoint.name:
        raise RuntimeError("learning-pilot final digest points to another checkpoint")
    if digest.get("checkpoint_manifest_sha256") != _sha256(
        final_checkpoint / "manifest.sha256"
    ):
        raise RuntimeError("learning-pilot final checkpoint digest changed")

    fresh_eval_path = workspace / "evaluation-step-10/evaluation-report.json"
    final_eval_path = workspace / "evaluation-step-80/evaluation-report.json"
    fresh_eval = _evaluation(fresh_eval_path)
    final_eval = _evaluation(final_eval_path)
    fresh_loss = float(fresh_eval["loss"]["mean"])
    final_loss = float(final_eval["loss"]["mean"])
    if not final_loss < fresh_loss:
        raise RuntimeError(
            f"learning-pilot held-out loss did not improve: {fresh_loss} -> {final_loss}"
        )
    first_training_loss = statistics.fmean(float(item["loss"]) for item in metrics[:10])
    last_training_loss = statistics.fmean(float(item["loss"]) for item in metrics[-10:])
    if not last_training_loss < first_training_loss:
        raise RuntimeError("learning-pilot training loss did not improve")

    parity = json.loads(
        (workspace / "flash-varlen-cuda-report.json").read_text(encoding="utf-8")
    )
    if parity.get("status") != "pass":
        raise RuntimeError("learning-pilot FlashAttention CUDA parity did not pass")
    telemetry = _telemetry(workspace / "gpu-telemetry.csv", world_size=world_size)
    run_log = (workspace / "run.log").read_text(encoding="utf-8")
    successful_duration_seconds = sum(
        int(value)
        for value in re.findall(
            r"^run_duration_seconds=(\d+)$",
            run_log,
            flags=re.MULTILINE,
        )
    )
    if successful_duration_seconds < 1:
        raise RuntimeError("learning-pilot run log has no completed attempt")
    duration_seconds = int(
        (workspace / "runtime-used-seconds").read_text(encoding="utf-8").strip()
    )
    if duration_seconds < successful_duration_seconds:
        raise RuntimeError("learning-pilot cumulative runtime accounting moved backwards")
    if duration_seconds > int(authorization["max_runtime_minutes"]) * 60:
        raise RuntimeError("learning-pilot runtime exceeded owner authorization")
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
        Path(plan["contract"]["path"]),
        Path(plan["data"]["manifest_path"]),
        Path(plan["evaluation"]["heldout_manifest_path"]),
        Path(plan["model"]["config_path"]),
        workspace / "data/training-fixture/fixture-report.json",
        Path(args.authorization).resolve(),
        workspace / "run.log",
        workspace / "runtime-used-seconds",
        workspace / "fresh.log",
        workspace / "resume.log",
        workspace / "fresh-state.json",
        workspace / "fresh-metrics.jsonl",
        workspace / "fresh-checkpoint-manifest.sha256",
        workspace / "fresh-checkpoint-state.json",
        workspace / "evaluation-step-10-export-report.json",
        workspace / "evaluation-step-80-export-report.json",
        fresh_eval_path,
        final_eval_path,
        workspace / "gpu-telemetry.csv",
        state_path,
        metrics_path,
        digest_path,
        final_checkpoint / "manifest.sha256",
        final_checkpoint / "distributed-training-state.json",
        workspace / "provenance/pip-freeze.txt",
        workspace / "provenance/source-commit.txt",
        workspace / "provenance/nvidia-smi-q.txt",
        workspace / "provenance/disk.txt",
        workspace / "provenance/memory.txt",
    ]
    copied = []
    for index, source in enumerate(selected):
        destination = evidence / f"{index:02d}-{source.name}"
        shutil.copy2(source, destination)
        copied.append(destination)

    throughput = [float(item["tokens_per_second"]) for item in metrics]
    memory = [int(item["peak_memory_allocated_bytes"]) for item in metrics]
    summary = {
        "authorization": {
            "full_4b_training": False,
            "hourly_rate_usd": authorization["hourly_rate_usd"],
            "instance_type": authorization["instance_type"],
            "max_runtime_minutes": authorization["max_runtime_minutes"],
            "provider": authorization["provider"],
        },
        "checkpoint": {
            "manifest_sha256": _sha256(final_checkpoint / "manifest.sha256"),
            "model_digest_sha256": digest["sha256"],
            "step": EXPECTED_FINAL_STEP,
        },
        "cost": {
            "duration_seconds": duration_seconds,
            "estimated_compute_usd": estimated_cost,
        },
        "evaluation": {
            "final_loss": final_loss,
            "final_perplexity": final_eval["loss"]["perplexity"],
            "fresh_loss": fresh_loss,
            "fresh_perplexity": fresh_eval["loss"]["perplexity"],
            "loss_improvement": fresh_loss - final_loss,
        },
        "gate": "lumi_4b_learning_pilot_v0.1",
        "hardware": {
            "cuda_capabilities": gpu_capabilities,
            "gpu_names": gpu_names,
            "gpu_total_memory_bytes": gpu_memory,
            "host_memory_bytes": runtime["host_memory_bytes"],
            "manifest_volume_free_bytes_at_plan": runtime[
                "manifest_volume_free_bytes"
            ],
        },
        "learning": {
            "first_10_training_loss_mean": first_training_loss,
            "last_10_training_loss_mean": last_training_loss,
        },
        "model_parameters": EXPECTED_PARAMETERS,
        "schema_version": 1,
        "source_commit": plan["source_commit"],
        "status": "pass",
        "telemetry": {
            "gpu_indices": list(range(world_size)),
            "max_memory_allocated_bytes": max(memory),
            "samples": len(telemetry),
        },
        "tokens": EXPECTED_FINAL_TOKENS,
        "topology": {
            "gradient_accumulation_steps": accumulation,
            "world_size": world_size,
        },
        "training": {
            "median_tokens_per_second": statistics.median(throughput),
            "steps": EXPECTED_FINAL_STEP,
            "tokens_per_second_max": max(throughput),
            "tokens_per_second_min": min(throughput),
        },
    }
    summary_path = workspace / "summary.json"
    manifest = _write_manifest(workspace, copied)
    summary["evidence_manifest_sha256"] = manifest["manifest_sha256"]
    _write_json(summary_path, summary)
    seal_manifest = workspace / "seal-manifest.sha256"
    with seal_manifest.open("w", encoding="utf-8") as handle:
        for path in (workspace / "manifest.sha256", summary_path):
            handle.write(f"{_sha256(path)}  {path.name}\n")
        handle.flush()
        os.fsync(handle.fileno())
    (workspace / "seal.complete").touch()
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
