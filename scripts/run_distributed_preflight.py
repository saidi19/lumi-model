#!/usr/bin/env python3
"""Run and seal Lumi's two-process DDP/FSDP exact-resume preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from lumi.distributed_training import (
    DISTRIBUTED_METRICS_FILENAME,
    DISTRIBUTED_TRAINING_STATE_FILENAME,
    verify_distributed_checkpoint,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str], *, environment: dict[str, str], log: Path) -> None:
    result = subprocess.run(
        command,
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log.write_text(result.stdout, encoding="utf-8")
    print(result.stdout, end="", flush=True)
    if result.returncode:
        raise RuntimeError(f"distributed preflight command failed ({result.returncode}): {log}")


def _torchrun_command(
    *,
    python: Path,
    manifest: Path,
    config: Path,
    output: Path,
    strategy: str,
    steps: int,
    port: int,
    resume: bool,
    torch_compile: bool = False,
) -> list[str]:
    accumulation = 2 if strategy == "ddp" else 1
    sync_mode = "last_microbatch" if strategy == "ddp" else "every_microbatch"
    command = [
        str(python),
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--node-rank=0",
        "--nproc-per-node=2",
        "--master-addr=localhost",
        f"--master-port={port}",
        "--local-addr=localhost",
        "scripts/train_distributed.py",
        "--manifest",
        str(manifest),
        "--config",
        str(config),
        "--output",
        str(output),
        "--steps",
        str(steps),
        "--strategy",
        strategy,
        "--per-rank-batch-size",
        "1",
        "--gradient-accumulation-steps",
        str(accumulation),
        "--learning-rate",
        "0.001",
        "--checkpoint-steps",
        "2",
        "--keep-checkpoints",
        "2",
        "--attention-implementation",
        "eager",
        "--document-isolation",
        "--gradient-sync-mode",
        sync_mode,
        "--device",
        "cpu",
        "--backend",
        "gloo",
    ]
    if resume:
        command.append("--resume")
    if torch_compile:
        command.append("--torch-compile")
    return command


def _inject_interruption(output: Path) -> None:
    partial = output / ".checkpoint-00000003.simulated-interruption.tmp"
    partial.mkdir()
    (partial / "orphan").write_text("not a published checkpoint\n", encoding="utf-8")
    metrics = output / DISTRIBUTED_METRICS_FILENAME
    records = _read_metrics(metrics)
    if [record["step"] for record in records] != [1, 2]:
        raise ValueError("interruption fixture expected two committed metrics")
    with metrics.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(records[0], sort_keys=True) + "\n")
        handle.write('{"simulated_uncommitted": true, "step": 3, "tokens_seen": -1}\n')
        handle.flush()
        os.fsync(handle.fileno())
    state_path = output / DISTRIBUTED_TRAINING_STATE_FILENAME
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update(
        {
            "best_loss": None,
            "last_loss": None,
            "latest_checkpoint": None,
            "sequence_cursor": 0,
            "status": "active",
            "step": 0,
            "tokens_seen": 0,
        }
    )
    _write_json(state_path, state)


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _verify_pair(workspace: Path, strategy: str) -> dict[str, Any]:
    reference = workspace / f"{strategy}-reference"
    resumed = workspace / f"{strategy}-resume"
    reference_state = json.loads(
        (reference / DISTRIBUTED_TRAINING_STATE_FILENAME).read_text(encoding="utf-8")
    )
    resumed_state = json.loads(
        (resumed / DISTRIBUTED_TRAINING_STATE_FILENAME).read_text(encoding="utf-8")
    )
    for state in (reference_state, resumed_state):
        if state.get("status") != "complete" or state.get("step") != 4:
            raise ValueError(f"{strategy} preflight did not finish at step four")
        distributed = state["run_specification"]["distributed"]
        if distributed["world_size"] != 2 or distributed["backend"] != "gloo":
            raise ValueError(f"{strategy} topology changed")
        authorization = state["run_specification"]["authorization"]
        if authorization.get("paid_gpu") is not False:
            raise ValueError(f"{strategy} preflight unexpectedly used paid compute")
        verify_distributed_checkpoint(reference / reference_state["latest_checkpoint"])
        verify_distributed_checkpoint(resumed / resumed_state["latest_checkpoint"])

    reference_metrics = _read_metrics(reference / DISTRIBUTED_METRICS_FILENAME)
    resumed_metrics = _read_metrics(resumed / DISTRIBUTED_METRICS_FILENAME)
    if len(reference_metrics) != 4 or len(resumed_metrics) != 4:
        raise ValueError(f"{strategy} metrics were not reconciled to four durable steps")
    exact_fields = (
        "global_tokens_per_step",
        "gradient_norm",
        "learning_rate",
        "loss",
        "step",
        "tokens_seen",
        "world_size",
    )
    for expected, actual in zip(reference_metrics, resumed_metrics, strict=True):
        for field in exact_fields:
            if expected[field] != actual[field]:
                raise ValueError(f"{strategy} resume changed {field} at step {expected['step']}")

    reference_digest = json.loads(
        (reference / "final-model-digest.json").read_text(encoding="utf-8")
    )
    resumed_digest = json.loads((resumed / "final-model-digest.json").read_text(encoding="utf-8"))
    if reference_digest["sha256"] != resumed_digest["sha256"]:
        raise ValueError(f"{strategy} exact-resume model digest changed")
    return {
        "final_loss": reference_metrics[-1]["loss"],
        "final_model_sha256": reference_digest["sha256"],
        "global_tokens_per_step": reference_metrics[-1]["global_tokens_per_step"],
        "metrics_exact": True,
        "resume_exact": True,
        "steps": 4,
        "world_size": 2,
    }


def _verify_compile_canary(workspace: Path) -> dict[str, Any]:
    output = workspace / "ddp-compile-canary"
    state = json.loads(
        (output / DISTRIBUTED_TRAINING_STATE_FILENAME).read_text(encoding="utf-8")
    )
    if state.get("status") != "complete" or state.get("step") != 1:
        raise ValueError("DDP torch.compile canary did not finish at step one")
    specification = state["run_specification"]
    distributed = specification["distributed"]
    if (
        distributed["strategy"] != "ddp"
        or distributed["world_size"] != 2
        or distributed["backend"] != "gloo"
        or distributed["torch_compile"] is not True
    ):
        raise ValueError("DDP torch.compile canary topology changed")
    if specification["authorization"].get("paid_gpu") is not False:
        raise ValueError("DDP torch.compile canary unexpectedly used paid compute")
    checkpoint = output / state["latest_checkpoint"]
    verify_distributed_checkpoint(checkpoint)
    metrics = _read_metrics(output / DISTRIBUTED_METRICS_FILENAME)
    if len(metrics) != 1 or metrics[0]["step"] != 1:
        raise ValueError("DDP torch.compile canary metrics are incomplete")
    digest = json.loads((output / "final-model-digest.json").read_text(encoding="utf-8"))
    return {
        "checkpoint_verified": True,
        "final_loss": metrics[0]["loss"],
        "final_model_sha256": digest["sha256"],
        "global_tokens_per_step": metrics[0]["global_tokens_per_step"],
        "steps": 1,
        "world_size": 2,
    }


def _source_hashes(repository: Path) -> dict[str, str]:
    relative_paths = (
        "scripts/prepare_distributed_preflight.py",
        "scripts/run_distributed_preflight.py",
        "scripts/train_distributed.py",
        "src/lumi/distributed_training.py",
        "src/lumi/modeling_lumi.py",
    )
    return {relative: _sha256_file(repository / relative) for relative in relative_paths}


def _write_manifest(workspace: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(workspace.rglob("*")):
        if not path.is_file() or path.name == "manifest.sha256":
            continue
        entries.append((_sha256_file(path), path.relative_to(workspace).as_posix()))
    manifest = workspace / "manifest.sha256"
    with manifest.open("w", encoding="utf-8") as handle:
        for digest, relative in entries:
            handle.write(f"{digest}  {relative}\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {"entries": len(entries), "sha256": _sha256_file(manifest)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--base-port", type=int, default=29_600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.base_port < 1_024 or args.base_port > 65_528:
        raise ValueError("base port must leave room for seven local launches")
    if args.workspace.exists() and any(args.workspace.iterdir()):
        raise FileExistsError(f"distributed preflight workspace is not empty: {args.workspace}")
    args.workspace.mkdir(parents=True, exist_ok=True)
    logs = args.workspace / "logs"
    logs.mkdir()
    environment = dict(os.environ)
    repository = Path(__file__).resolve().parents[1]
    environment["PYTHONPATH"] = str(repository / "src")
    environment.setdefault("GLOO_SOCKET_IFNAME", "lo0" if platform.system() == "Darwin" else "lo")

    _run(
        [
            str(args.python),
            "scripts/prepare_distributed_preflight.py",
            "--output",
            str(args.workspace / "fixture"),
        ],
        environment=environment,
        log=logs / "00-prepare.log",
    )
    config = args.workspace / "fixture" / "lumi-distributed-preflight.json"
    manifest = args.workspace / "fixture" / "data" / "manifest.json"
    launch_index = 0
    for strategy in ("ddp", "fsdp_full_shard"):
        reference = args.workspace / f"{strategy}-reference"
        resumed = args.workspace / f"{strategy}-resume"
        launch_index += 1
        _run(
            _torchrun_command(
                python=args.python,
                manifest=manifest,
                config=config,
                output=reference,
                strategy=strategy,
                steps=4,
                port=args.base_port + launch_index,
                resume=False,
            ),
            environment=environment,
            log=logs / f"{launch_index:02d}-{strategy}-reference.log",
        )
        launch_index += 1
        _run(
            _torchrun_command(
                python=args.python,
                manifest=manifest,
                config=config,
                output=resumed,
                strategy=strategy,
                steps=2,
                port=args.base_port + launch_index,
                resume=False,
            ),
            environment=environment,
            log=logs / f"{launch_index:02d}-{strategy}-checkpoint.log",
        )
        _inject_interruption(resumed)
        launch_index += 1
        _run(
            _torchrun_command(
                python=args.python,
                manifest=manifest,
                config=config,
                output=resumed,
                strategy=strategy,
                steps=4,
                port=args.base_port + launch_index,
                resume=True,
            ),
            environment=environment,
            log=logs / f"{launch_index:02d}-{strategy}-resume.log",
        )

    launch_index += 1
    _run(
        _torchrun_command(
            python=args.python,
            manifest=manifest,
            config=config,
            output=args.workspace / "ddp-compile-canary",
            strategy="ddp",
            steps=1,
            port=args.base_port + launch_index,
            resume=False,
            torch_compile=True,
        ),
        environment=environment,
        log=logs / f"{launch_index:02d}-ddp-compile-canary.log",
    )

    report = {
        "ddp": _verify_pair(args.workspace, "ddp"),
        "ddp_compile_canary": _verify_compile_canary(args.workspace),
        "fixture": {
            "config_sha256": _sha256_file(config),
            "manifest_sha256": _sha256_file(manifest),
        },
        "fsdp_full_shard": _verify_pair(args.workspace, "fsdp_full_shard"),
        "platform": {
            "machine": platform.machine(),
            "python": platform.python_version(),
            "system": platform.system(),
        },
        "safety": {
            "full_4b_training": False,
            "paid_gpu": False,
        },
        "schema_version": 1,
        "source_sha256": _source_hashes(repository),
        "status": "pass",
    }
    _write_json(args.workspace / "preflight-report.json", report)
    seal = _write_manifest(args.workspace)
    print(json.dumps({"report": report, "seal": seal}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
