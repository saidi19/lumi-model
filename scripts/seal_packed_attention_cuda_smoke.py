#!/usr/bin/env python3
"""Validate and seal the hard-capped Lumi packed-attention CUDA A/B smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, NamedTuple

EXPECTED_STEP = 100
EXPECTED_CURSOR = 800
EXPECTED_TOKENS = 3_276_800
EXPECTED_PARAMETERS = 1_050_981_888
EXPECTED_CONFIG_SHA256 = "7fd53cab43b4cde02f8f09bc33e69f4aeb3594a60d7abf0def1bfc5f496829ce"
EXPECTED_DATA_FINGERPRINT = "0bce4d9e4a51de1b831b0226e0052ea33130733eae2bb61992c6eacaf203e0c8"
EXPECTED_MANIFEST_SHA256 = "9228fa610d4a985d05e15057e8ef12619a14cbedbb5a6a5ab513d3aeca2832c8"
BASE_OPTIMIZER = {
    "attention_implementation": "flash_attention_2",
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "gradient_checkpointing": True,
    "keep_checkpoints": 2,
    "learning_rate": 0.0002,
    "lr_schedule": "cosine",
    "max_grad_norm": 1.0,
    "min_learning_rate_ratio": 0.1,
    "schedule_steps": 42_725,
    "warmup_steps": 500,
    "weight_decay": 0.1,
}


class PolicyEvidence(NamedTuple):
    name: str
    metrics: list[dict[str, Any]]
    telemetry: list[dict[str, float | str]]
    checkpoint_files: list[Path]
    summary: dict[str, Any]
    selected_files: list[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--hourly-rate-usd", type=float, default=1.99)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def read_metrics(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def finite(record: dict[str, Any], key: str) -> float:
    value = float(record[key])
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite {key} at step {record.get('step')}")
    return value


def log_value(text: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}=(.+)$", text, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError(f"missing {key} in training log")
    return match.group(1).strip()


def parse_telemetry(path: Path) -> list[dict[str, float | str]]:
    samples: list[dict[str, float | str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        timestamp, utilization, memory, power, temperature = parts
        samples.append(
            {
                "timestamp": timestamp,
                "utilization_percent": float(utilization),
                "memory_mib": float(memory),
                "power_watts": float(power),
                "temperature_c": float(temperature),
            }
        )
    if not samples:
        raise RuntimeError(f"telemetry contains no samples: {path}")
    return samples


def expected_contract(*, isolated: bool) -> dict[str, Any]:
    optimizer = dict(BASE_OPTIMIZER)
    if isolated:
        optimizer["document_isolation"] = "position_reset_varlen_v0.1"
    return {
        "config": {"sha256": EXPECTED_CONFIG_SHA256},
        "data": {
            "build_fingerprint": EXPECTED_DATA_FINGERPRINT,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        },
        "optimizer": optimizer,
        "schema_version": 1,
        "seed": 7,
    }


def validate_policy(
    workspace: Path,
    *,
    name: str,
    isolated: bool,
    hourly_rate_usd: float,
) -> PolicyEvidence:
    run = workspace / f"{name}-training"
    state_path = run / "training-state.json"
    metrics_path = run / "metrics.jsonl"
    checkpoint = run / "checkpoint-00000100"
    log_path = workspace / f"{name}-training.log"
    telemetry_path = workspace / f"{name}-telemetry.csv"
    marker = workspace / f"{name}-training.complete"
    for path in (state_path, metrics_path, log_path, telemetry_path, marker):
        if not path.is_file():
            raise RuntimeError(f"missing {name} evidence: {path}")

    state = read_json(state_path)
    expected_state = {
        "latest_checkpoint": "checkpoint-00000100",
        "schema_version": 1,
        "sequence_cursor": EXPECTED_CURSOR,
        "status": "complete",
        "step": EXPECTED_STEP,
        "tokens_seen": EXPECTED_TOKENS,
    }
    for key, expected in expected_state.items():
        if state.get(key) != expected:
            raise RuntimeError(
                f"{name} state mismatch for {key}: {state.get(key)!r} != {expected!r}"
            )

    contract = expected_contract(isolated=isolated)
    specification = state.get("run_specification")
    if not isinstance(specification, dict):
        raise RuntimeError(f"{name} state has no run specification")
    for key, expected in contract.items():
        if specification.get(key) != expected:
            raise RuntimeError(f"{name} run contract mismatch for {key}")
    if state.get("run_fingerprint") != fingerprint(contract):
        raise RuntimeError(f"{name} run fingerprint does not match its contract")

    metrics = read_metrics(metrics_path)
    if [metric.get("step") for metric in metrics] != list(range(1, EXPECTED_STEP + 1)):
        raise RuntimeError(f"{name} metrics must contain steps 1 through 100 exactly once")
    for metric in metrics:
        if metric.get("device") != "cuda":
            raise RuntimeError(f"non-CUDA {name} metric at step {metric.get('step')}")
        for key in ("elapsed_seconds", "gradient_norm", "loss", "tokens_per_second"):
            finite(metric, key)

    checkpoint_files = sorted(path for path in checkpoint.iterdir() if path.is_file())
    expected_checkpoint_files = {
        "config.json",
        "configuration_lumi.py",
        "generation_config.json",
        "model.safetensors",
        "modeling_lumi.py",
        "trainer.pt",
        "training-state.json",
    }
    if {path.name for path in checkpoint_files} != expected_checkpoint_files:
        raise RuntimeError(f"{name} checkpoint-00000100 file set changed")

    telemetry = parse_telemetry(telemetry_path)
    log_text = log_path.read_text(encoding="utf-8")
    duration_seconds = int(log_value(log_text, "smoke_duration_seconds"))
    started_utc = log_value(log_text, "smoke_started_utc")
    completed_utc = log_value(log_text, "smoke_completed_utc")
    if duration_seconds <= 0:
        raise RuntimeError(f"{name} duration must be positive")

    steady = [metric for metric in metrics if 10 <= int(metric["step"]) <= 99]
    steady_throughput = [finite(metric, "tokens_per_second") for metric in steady]
    summary = {
        "checkpoint": {
            "files": {path.name: path.stat().st_size for path in checkpoint_files},
            "name": checkpoint.name,
            "total_bytes": sum(path.stat().st_size for path in checkpoint_files),
        },
        "memory": {
            "peak_allocated_bytes": max(
                int(metric["peak_memory_allocated_bytes"]) for metric in metrics
            ),
            "peak_reserved_bytes": max(
                int(metric["peak_memory_reserved_bytes"]) for metric in metrics
            ),
        },
        "optimization": {
            "best_loss": min(finite(metric, "loss") for metric in metrics),
            "final_gradient_norm": finite(metrics[-1], "gradient_norm"),
            "final_learning_rate": finite(metrics[-1], "learning_rate"),
            "final_loss": finite(metrics[-1], "loss"),
            "first_loss": finite(metrics[0], "loss"),
        },
        "policy": (
            "position_reset_flash_varlen_document_isolated"
            if isolated
            else "separator_only_flash_causal"
        ),
        "run_fingerprint": state["run_fingerprint"],
        "runtime": {
            "completed_utc": completed_utc,
            "duration_seconds": duration_seconds,
            "estimated_training_cost_usd": duration_seconds / 3600 * hourly_rate_usd,
            "started_utc": started_utc,
            "summed_step_compute_seconds": sum(
                finite(metric, "elapsed_seconds") for metric in metrics
            ),
        },
        "steady_state": {
            "samples": len(steady),
            "tokens_per_second_max": max(steady_throughput),
            "tokens_per_second_median": statistics.median(steady_throughput),
            "tokens_per_second_min": min(steady_throughput),
            "window_steps": [10, 99],
        },
        "telemetry": {
            "gpu_utilization_percent_max": max(
                float(sample["utilization_percent"]) for sample in telemetry
            ),
            "gpu_utilization_percent_median": statistics.median(
                float(sample["utilization_percent"]) for sample in telemetry
            ),
            "memory_mib_max": max(float(sample["memory_mib"]) for sample in telemetry),
            "power_watts_median": statistics.median(
                float(sample["power_watts"]) for sample in telemetry
            ),
            "samples": len(telemetry),
            "temperature_c_max": max(
                float(sample["temperature_c"]) for sample in telemetry
            ),
        },
    }
    return PolicyEvidence(
        name=name,
        metrics=metrics,
        telemetry=telemetry,
        checkpoint_files=checkpoint_files,
        summary=summary,
        selected_files=[state_path, metrics_path, log_path, telemetry_path, *checkpoint_files],
    )


def main() -> None:
    args = parse_args()
    if args.hourly_rate_usd <= 0:
        raise ValueError("hourly rate must be positive")
    workspace = args.workspace.resolve()

    common_required = [
        workspace / "setup.complete",
        workspace / "preflight.complete",
        workspace / "cloud-preflight.json",
        workspace / "tests.log",
        workspace / "ruff.log",
        workspace / "flash-varlen-cuda-report.json",
        workspace / "baseline-calibration-summary.json",
        workspace / "baseline-calibration-metrics.jsonl",
    ]
    for path in common_required:
        if not path.is_file():
            raise RuntimeError(f"missing smoke evidence: {path}")

    parity = read_json(workspace / "flash-varlen-cuda-report.json")
    if parity.get("status") != "pass":
        raise RuntimeError("real FlashAttention CUDA parity gate did not pass")
    historical = read_json(workspace / "baseline-calibration-summary.json")
    if historical.get("stage") != "lumi_1b_a100_calibration_v0.1":
        raise RuntimeError("unexpected historical A-policy baseline summary")

    current = validate_policy(
        workspace,
        name="current",
        isolated=False,
        hourly_rate_usd=args.hourly_rate_usd,
    )
    isolated = validate_policy(
        workspace,
        name="isolated",
        isolated=True,
        hourly_rate_usd=args.hourly_rate_usd,
    )

    paired_loss_delta = {
        str(step): finite(isolated.metrics[step - 1], "loss")
        - finite(current.metrics[step - 1], "loss")
        for step in (1, 10, 50, 100)
    }
    current_speed = float(current.summary["steady_state"]["tokens_per_second_median"])
    isolated_speed = float(isolated.summary["steady_state"]["tokens_per_second_median"])
    total_training_cost = sum(
        float(policy.summary["runtime"]["estimated_training_cost_usd"])
        for policy in (current, isolated)
    )

    summary = {
        "ab_comparison": {
            "controlled_variables": [
                "code_commit",
                "cuda_runtime",
                "data_order",
                "flash_attention_backend",
                "initialization_seed",
                "optimizer_and_schedule",
            ],
            "paired_loss_delta_isolated_minus_current": paired_loss_delta,
            "steady_tokens_per_second_current": current_speed,
            "steady_tokens_per_second_isolated": isolated_speed,
            "throughput_ratio_isolated_over_current": isolated_speed / current_speed,
        },
        "authorization_boundary": "two_fresh_100_step_runs_no_250m_or_later_stage",
        "current": current.summary,
        "historical_eager_baseline": {
            "final_loss": historical["optimization"]["final_loss"],
            "steady_tokens_per_second_median": historical["steady_state"][
                "tokens_per_second_median"
            ],
        },
        "hourly_rate_usd": args.hourly_rate_usd,
        "isolated": isolated.summary,
        "model_parameters": EXPECTED_PARAMETERS,
        "parity": parity,
        "runtime": {
            "estimated_combined_training_cost_usd": total_training_cost,
            "total_training_duration_seconds": sum(
                int(policy.summary["runtime"]["duration_seconds"])
                for policy in (current, isolated)
            ),
        },
        "schema_version": 1,
        "stage": "lumi_1b_a100_packed_attention_ab_100_step_v0.1",
        "status": "pass",
    }

    summary_path = workspace / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    selected = [
        workspace / "setup.log",
        workspace / "preflight.log",
        workspace / "cloud-preflight.json",
        workspace / "tests.log",
        workspace / "ruff.log",
        workspace / "flash-varlen-cuda-report.json",
        workspace / "baseline-calibration-summary.json",
        workspace / "baseline-calibration-metrics.jsonl",
        summary_path,
        *current.selected_files,
        *isolated.selected_files,
        *sorted((workspace / "provenance").glob("*")),
    ]
    manifest_path = workspace / "manifest.sha256"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for path in sorted(selected):
            if not path.is_file():
                raise RuntimeError(f"missing selected artifact: {path}")
            handle.write(f"{sha256(path)}  {path.relative_to(workspace)}\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"manifest={manifest_path}")
    print(f"manifest_sha256={sha256(manifest_path)}")
    (workspace / "seal.complete").touch()


if __name__ == "__main__":
    main()
