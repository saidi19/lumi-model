"""Reproducible local initialization and optimizer A/B gate for Lumi."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import statistics
import struct
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from .configuration_lumi import LumiConfig
from .modeling_lumi import LumiForCausalLM
from .packed_data import PackedTokenDataset, validate_packed_manifest
from .pilot_training import train_packed_pilot

GATE_SCHEMA_VERSION = 1
BASELINE_LABEL = "standard_legacy"


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def selected_token_ids_sha256(
    dataset: PackedTokenDataset, indices: list[int], context_length: int
) -> str:
    digest = hashlib.sha256()
    for index in indices:
        token_ids = dataset[index][:context_length].tolist()
        digest.update(struct.pack("<I", len(token_ids)))
        digest.update(struct.pack(f"<{len(token_ids)}I", *token_ids))
    return digest.hexdigest()


def _validate_inputs(
    specification: dict[str, Any], training_manifest_path: Path, heldout_manifest_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    if specification.get("schema_version") != GATE_SCHEMA_VERSION:
        raise ValueError("unsupported initialization/optimizer gate schema")
    training_manifest = validate_packed_manifest(training_manifest_path)
    heldout_manifest = validate_packed_manifest(heldout_manifest_path)
    expected_training = specification["data"]["training"]
    expected_heldout = specification["data"]["heldout"]
    for label, path, manifest, expected in (
        ("training", training_manifest_path, training_manifest, expected_training),
        ("heldout", heldout_manifest_path, heldout_manifest, expected_heldout),
    ):
        if _sha256(path) != expected["manifest_sha256"]:
            raise ValueError(f"{label} manifest hash changed")
        if manifest["build_fingerprint"] != expected["build_fingerprint"]:
            raise ValueError(f"{label} build fingerprint changed")
    context_length = expected_heldout["context_length"]
    if training_manifest["packing"]["sequence_length"] != context_length:
        raise ValueError("training sequence length does not match the proxy context")
    if heldout_manifest["packing"]["sequence_length"] < context_length:
        raise ValueError("held-out sequences are shorter than the proxy context")
    if training_manifest["tokenizer"] != heldout_manifest["tokenizer"]:
        raise ValueError("training and held-out tokenizer contracts differ")
    training = specification["training"]
    expected_tokens = (
        training["steps"]
        * training["batch_size"]
        * training["gradient_accumulation_steps"]
        * context_length
    )
    if expected_tokens != training["token_budget_per_run"]:
        raise ValueError("configured token budget does not match the training geometry")
    labels = [variant["label"] for variant in specification["variants"]]
    if len(labels) != len(set(labels)) or BASELINE_LABEL not in labels:
        raise ValueError("A/B variants must be unique and include the baseline")
    return training_manifest, heldout_manifest


def _materialize_model_configs(specification: dict[str, Any], output: Path) -> dict[str, Path]:
    config_root = output / "model-configs"
    config_root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for variant in specification["variants"]:
        payload = {
            **specification["model"],
            "initialization_strategy": variant["initialization_strategy"],
        }
        LumiConfig(**payload)
        path = config_root / f"{variant['label']}.json"
        encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != encoded:
                raise ValueError(f"materialized model config changed: {path}")
        else:
            path.write_text(encoded, encoding="utf-8")
        paths[variant["label"]] = path
    return paths


def _resolve_device(requested: str) -> torch.device:
    if requested != "mps":
        raise ValueError("the pinned local gate requires the MPS device")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available; do not silently fall back to CPU")
    return torch.device("mps")


@torch.no_grad()
def evaluate_proxy_model(
    model: LumiForCausalLM,
    *,
    dataset: PackedTokenDataset,
    indices: list[int],
    context_length: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total_negative_log_likelihood = 0.0
    prediction_tokens = 0
    started = time.perf_counter()
    for index in indices:
        input_ids = dataset[index][:context_length].unsqueeze(0).to(device)
        loss = model(input_ids=input_ids, labels=input_ids, document_isolation=True).loss
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"non-finite held-out loss at sequence {index}: {loss}")
        sequence_prediction_tokens = input_ids.numel() - input_ids.shape[0]
        total_negative_log_likelihood += float(loss) * sequence_prediction_tokens
        prediction_tokens += sequence_prediction_tokens
    torch.mps.synchronize()
    elapsed = max(time.perf_counter() - started, 1e-9)
    mean_loss = total_negative_log_likelihood / prediction_tokens
    return {
        "elapsed_seconds": elapsed,
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss) if mean_loss < 709 else float("inf"),
        "prediction_tokens": prediction_tokens,
        "sequences": len(indices),
        "tokens_per_second": prediction_tokens / elapsed,
    }


def _training_metrics(path: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not records:
        raise ValueError(f"training metrics are empty: {path}")
    numeric_values = [
        value
        for record in records
        for key in ("loss", "gradient_norm", "tokens_per_second")
        if not math.isfinite(value := float(record[key]))
    ]
    if numeric_values:
        raise ValueError(f"training metrics contain non-finite values: {path}")
    return {
        "best_loss": min(float(record["loss"]) for record in records),
        "final_loss": float(records[-1]["loss"]),
        "max_gradient_norm": max(float(record["gradient_norm"]) for record in records),
        "median_tokens_per_second": statistics.median(
            float(record["tokens_per_second"]) for record in records
        ),
        "records": len(records),
    }


def _evaluate_initial(
    *,
    config_path: Path,
    seed: int,
    dataset: PackedTokenDataset,
    indices: list[int],
    context_length: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = LumiForCausalLM(LumiConfig(**_read_json(config_path))).to(device)
    result = evaluate_proxy_model(
        model,
        dataset=dataset,
        indices=indices,
        context_length=context_length,
        device=device,
    )
    del model
    torch.mps.empty_cache()
    return result


def _evaluate_checkpoint(
    *,
    checkpoint: Path,
    dataset: PackedTokenDataset,
    indices: list[int],
    context_length: int,
    device: torch.device,
) -> dict[str, Any]:
    model = LumiForCausalLM.from_pretrained(checkpoint, local_files_only=True).to(device)
    result = evaluate_proxy_model(
        model,
        dataset=dataset,
        indices=indices,
        context_length=context_length,
        device=device,
    )
    del model
    torch.mps.empty_cache()
    return result


def _run_resume_canary(
    *,
    specification: dict[str, Any],
    training_manifest_path: Path,
    config_path: Path,
    output: Path,
) -> dict[str, Any]:
    result_path = output / "result.json"
    if result_path.is_file():
        return _read_json(result_path)
    training = specification["training"]
    training_output = output / "training"
    common = {
        "attention_implementation": training["attention_implementation"],
        "batch_size": training["batch_size"],
        "checkpoint_steps": 2,
        "config_path": config_path,
        "device": training["device"],
        "document_isolation": training["document_isolation"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "keep_checkpoints": 1,
        "learning_rate": training["learning_rate"],
        "lr_schedule": "cosine",
        "manifest_path": training_manifest_path,
        "max_grad_norm": training["max_grad_norm"],
        "min_learning_rate_ratio": training["min_learning_rate_ratio"],
        "optimizer_profile": "grouped_adamw_beta2_0.95_v0.1",
        "output": training_output,
        "schedule_steps": 4,
        "seed": training["seeds"][0],
        "warmup_steps": 1,
        "weight_decay": training["weight_decay"],
    }
    state_path = training_output / "training-state.json"
    if not state_path.exists():
        train_packed_pilot(steps=2, resume=False, **common)
    state = train_packed_pilot(steps=4, resume=True, **common)
    result = {
        "checkpoint": state["latest_checkpoint"],
        "optimizer_profile": "grouped_adamw_beta2_0.95_v0.1",
        "status": "pass",
        "step": state["step"],
        "tokens_seen": state["tokens_seen"],
    }
    if state["step"] != 4:
        raise RuntimeError("resume canary did not reach step 4")
    output.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(result_path, result)
    return result


def evaluate_ab_decision(
    specification: dict[str, Any], run_results: list[dict[str, Any]]
) -> dict[str, Any]:
    seeds = specification["training"]["seeds"]
    variants = {variant["label"]: variant for variant in specification["variants"]}
    by_label: dict[str, dict[int, dict[str, Any]]] = {label: {} for label in variants}
    for result in run_results:
        by_label[result["variant"]][result["seed"]] = result
    if any(set(results) != set(seeds) for results in by_label.values()):
        raise ValueError("decision requires every pinned variant/seed result")

    summaries = {}
    for label, results in by_label.items():
        ordered = [results[seed] for seed in seeds]
        summaries[label] = {
            "median_final_heldout_loss": statistics.median(
                result["heldout"]["final"]["loss"] for result in ordered
            ),
            "median_final_training_loss": statistics.median(
                result["training_metrics"]["final_loss"] for result in ordered
            ),
            "median_tokens_per_second": statistics.median(
                result["training_metrics"]["median_tokens_per_second"] for result in ordered
            ),
            "per_seed_final_heldout_loss": {
                str(seed): results[seed]["heldout"]["final"]["loss"] for seed in seeds
            },
        }

    baseline = summaries[BASELINE_LABEL]
    thresholds = specification["decision"]
    eligible = []
    comparisons = {}
    for label, summary in summaries.items():
        if label == BASELINE_LABEL:
            continue
        median_improvement = 1.0 - (
            summary["median_final_heldout_loss"] / baseline["median_final_heldout_loss"]
        )
        throughput_change = (
            summary["median_tokens_per_second"] / baseline["median_tokens_per_second"] - 1.0
        )
        regressing_seeds = sum(
            summary["per_seed_final_heldout_loss"][str(seed)]
            > baseline["per_seed_final_heldout_loss"][str(seed)]
            * (1.0 + thresholds["max_seed_heldout_regression_fraction"])
            for seed in seeds
        )
        passed = (
            median_improvement >= thresholds["min_median_heldout_improvement_fraction"]
            and throughput_change >= -thresholds["max_throughput_regression_fraction"]
            and regressing_seeds <= thresholds["max_regressing_seeds"]
        )
        comparisons[label] = {
            "eligible": passed,
            "median_heldout_improvement_fraction": median_improvement,
            "regressing_seeds": regressing_seeds,
            "throughput_change_fraction": throughput_change,
        }
        if passed:
            eligible.append(label)

    winner = min(
        eligible,
        key=lambda label: summaries[label]["median_final_heldout_loss"],
        default=BASELINE_LABEL,
    )
    return {
        "comparisons_to_baseline": comparisons,
        "decision": {
            "initialization_strategy": variants[winner]["initialization_strategy"],
            "optimizer_profile": variants[winner]["optimizer_profile"],
            "reason": (
                "clear_proxy_improvement"
                if winner != BASELINE_LABEL
                else "no_candidate_cleared_conservative_thresholds"
            ),
            "winner": winner,
        },
        "status": "pass",
        "summaries": summaries,
        "thresholds": thresholds,
    }


def run_init_optimizer_ab_gate(
    *,
    specification_path: str | Path,
    training_manifest_path: str | Path,
    heldout_manifest_path: str | Path,
    output: str | Path,
    canary_only: bool = False,
) -> dict[str, Any]:
    specification_path = Path(specification_path).resolve()
    training_manifest_path = Path(training_manifest_path).resolve()
    heldout_manifest_path = Path(heldout_manifest_path).resolve()
    output = Path(output).resolve()
    specification = _read_json(specification_path)
    _validate_inputs(specification, training_manifest_path, heldout_manifest_path)
    output.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(specification["training"]["device"])
    config_paths = _materialize_model_configs(specification, output)
    heldout_dataset = PackedTokenDataset(heldout_manifest_path, verify_hashes=False)
    heldout = specification["data"]["heldout"]
    indices = heldout["middle_sequence_indices_by_stratum"]
    observed_token_hash = selected_token_ids_sha256(
        heldout_dataset, indices, heldout["context_length"]
    )
    if observed_token_hash != heldout["selected_token_ids_sha256"]:
        raise ValueError("selected held-out token IDs changed")

    canary = _run_resume_canary(
        specification=specification,
        training_manifest_path=training_manifest_path,
        config_path=config_paths["residual_grouped"],
        output=output / "resume-canary",
    )
    if canary_only:
        return {"resume_canary": canary, "status": "pass"}
    run_results = []
    training = specification["training"]
    for variant in specification["variants"]:
        label = variant["label"]
        for seed in training["seeds"]:
            run_root = output / "runs" / label / f"seed-{seed}"
            result_path = run_root / "result.json"
            if result_path.is_file():
                run_results.append(_read_json(result_path))
                continue
            run_root.mkdir(parents=True, exist_ok=True)
            print(f"ab_run variant={label} seed={seed} stage=initial_eval", flush=True)
            initial = _evaluate_initial(
                config_path=config_paths[label],
                seed=seed,
                dataset=heldout_dataset,
                indices=indices,
                context_length=heldout["context_length"],
                device=device,
            )
            print(f"ab_run variant={label} seed={seed} stage=training", flush=True)
            training_output = run_root / "training"
            state = train_packed_pilot(
                manifest_path=training_manifest_path,
                config_path=config_paths[label],
                output=training_output,
                steps=training["steps"],
                batch_size=training["batch_size"],
                gradient_accumulation_steps=training["gradient_accumulation_steps"],
                learning_rate=training["learning_rate"],
                weight_decay=training["weight_decay"],
                max_grad_norm=training["max_grad_norm"],
                checkpoint_steps=training["checkpoint_steps"],
                seed=seed,
                device=training["device"],
                resume=(training_output / "training-state.json").is_file(),
                keep_checkpoints=1,
                lr_schedule=training["lr_schedule"],
                warmup_steps=training["warmup_steps"],
                schedule_steps=training["steps"],
                min_learning_rate_ratio=training["min_learning_rate_ratio"],
                attention_implementation=training["attention_implementation"],
                document_isolation=training["document_isolation"],
                optimizer_profile=variant["optimizer_profile"],
            )
            checkpoint = training_output / state["latest_checkpoint"]
            print(f"ab_run variant={label} seed={seed} stage=final_eval", flush=True)
            final = _evaluate_checkpoint(
                checkpoint=checkpoint,
                dataset=heldout_dataset,
                indices=indices,
                context_length=heldout["context_length"],
                device=device,
            )
            result = {
                "completed_at": _utc_now(),
                "heldout": {"final": final, "initial": initial},
                "seed": seed,
                "status": "complete",
                "training": {
                    "checkpoint": state["latest_checkpoint"],
                    "run_fingerprint": state["run_fingerprint"],
                    "steps": state["step"],
                    "tokens_seen": state["tokens_seen"],
                },
                "training_metrics": _training_metrics(training_output / "metrics.jsonl"),
                "variant": label,
            }
            if state["tokens_seen"] != training["token_budget_per_run"]:
                raise RuntimeError(f"token budget changed for {label} seed {seed}")
            _write_json_atomic(result_path, result)
            run_results.append(result)

    decision_path = output / "decision.json"
    decision = evaluate_ab_decision(specification, run_results)
    decision.update(
        {
            "completed_at": _utc_now(),
            "evidence": {
                "heldout_selected_token_ids_sha256": observed_token_hash,
                "resume_canary": canary,
                "run_results": len(run_results),
                "specification_sha256": _sha256(specification_path),
            },
            "hardware": {
                "device": str(device),
                "machine": platform.machine(),
                "platform": platform.platform(),
                "torch": torch.__version__,
            },
        }
    )
    if decision_path.exists():
        existing = _read_json(decision_path)
        stable_fields = ("comparisons_to_baseline", "decision", "status", "summaries", "thresholds")
        if any(existing[field] != decision[field] for field in stable_fields):
            raise ValueError("existing A/B decision differs from recomputed evidence")
        decision = existing
    else:
        _write_json_atomic(decision_path, decision)
    return decision
