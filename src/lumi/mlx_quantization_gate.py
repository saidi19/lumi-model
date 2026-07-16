"""Stratified quality, memory, and speed gate for quantized Lumi MLX artifacts."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
import uuid
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

QUANTIZATION_GATE_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "bytes": path.stat().st_size,
        "name": path.name,
        "sha256": _sha256_file(path),
    }


def _model_artifact(path: Path) -> dict[str, Any]:
    weights = sorted(path.glob("model*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"MLX model has no safetensors: {path}")
    files = [path / "config.json", *sorted(path.glob("model*.safetensors.index.json")), *weights]
    return {
        "disk_bytes": sum(item.stat().st_size for item in path.iterdir() if item.is_file()),
        "files": [_file_artifact(item) for item in files],
    }


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _select_stratified_sequences(
    strata: list[dict[str, Any]], sequences_per_stratum: int
) -> list[dict[str, Any]]:
    if sequences_per_stratum < 1:
        raise ValueError("sequences_per_stratum must be positive")
    selections = []
    for stratum in strata:
        start = stratum["sequence_start"]
        end = stratum["sequence_end"]
        count = end - start
        if count < sequences_per_stratum:
            raise ValueError(
                f"stratum {stratum['id']} has only {count} sequences, "
                f"cannot select {sequences_per_stratum}"
            )
        indices = []
        for ordinal in range(sequences_per_stratum):
            # Select the center of equal-width bins rather than a mutable first-N prefix.
            index = start + ((2 * ordinal + 1) * count) // (2 * sequences_per_stratum)
            indices.append(min(index, end - 1))
        if len(set(indices)) != len(indices):
            raise RuntimeError(f"stratified sequence selection duplicated {stratum['id']}")
        selections.append(
            {
                "category": stratum["category"],
                "component": stratum["component"],
                "id": stratum["id"],
                "sequence_indices": indices,
            }
        )
    return selections


def _prompt_ids(tokenizer, prompt: str, bos_token_id: int | None) -> list[int]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    return [bos_token_id, *ids] if bos_token_id is not None else ids


def _array_metrics(reference, candidate) -> dict[str, float | bool]:
    import numpy as np

    if reference.shape != candidate.shape:
        raise ValueError(f"logits shapes differ: {reference.shape} != {candidate.shape}")
    reference_float = reference.astype(np.float64)
    candidate_float = candidate.astype(np.float64)
    difference = np.abs(reference_float - candidate_float)
    reference_flat = reference_float.reshape(-1)
    candidate_flat = candidate_float.reshape(-1)
    denominator = float(np.linalg.norm(reference_flat) * np.linalg.norm(candidate_flat))
    cosine = (
        float(np.dot(reference_flat, candidate_flat) / denominator)
        if denominator
        else float(np.array_equal(reference_flat, candidate_flat))
    )
    reference_rms = math.sqrt(float(np.square(reference_float).mean()))
    error_rms = math.sqrt(float(np.square(difference).mean()))
    top1 = np.argmax(reference, axis=-1) == np.argmax(candidate, axis=-1)
    return {
        "all_finite": bool(np.isfinite(candidate).all()),
        "cosine_similarity": cosine,
        "max_absolute_error": float(difference.max(initial=0.0)),
        "mean_absolute_error": float(difference.mean()),
        "normalized_root_mean_square_error": error_rms / reference_rms,
        "top1_agreement": float(top1.mean()),
        "top1_matches": int(top1.sum()),
        "top1_positions": int(top1.size),
    }


def _evaluate_model(
    *,
    model_path: Path,
    dataset,
    selections: list[dict[str, Any]],
    tokenizer,
    prompts: list[str],
    generation_tokens: int,
    benchmark_prefill_tokens: int,
    benchmark_decode_tokens: int,
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlx_lm.generate import generate_step
    from mlx_lm.utils import load_model

    mx.clear_cache()
    mx.reset_peak_memory()
    load_started = time.perf_counter()
    model, config = load_model(model_path, strict=True, lazy=False)
    load_seconds = time.perf_counter() - load_started

    first_index = selections[0]["sequence_indices"][0]
    warmup_ids = dataset[first_index][:16].numpy().astype(np.int32)
    warmup = model(mx.array(warmup_ids[None]))
    mx.eval(warmup)
    del warmup
    mx.clear_cache()

    total_negative_log_likelihood = 0.0
    prediction_tokens = 0
    heldout_seconds = 0.0
    top1_by_sequence = {}
    per_stratum = []
    for selection in selections:
        stratum_nll = 0.0
        stratum_tokens = 0
        stratum_seconds = 0.0
        for index in selection["sequence_indices"]:
            token_ids = dataset[index].numpy().astype(np.int32)
            inputs = mx.array(token_ids[None])
            started = time.perf_counter()
            logits = model(inputs)
            loss_sum = nn.losses.cross_entropy(
                logits[:, :-1, :],
                inputs[:, 1:],
                reduction="sum",
            )
            top1 = mx.argmax(logits[:, :-1, :], axis=-1)
            mx.eval(loss_sum, top1)
            elapsed = time.perf_counter() - started
            sequence_nll = float(loss_sum.item())
            sequence_tokens = len(token_ids) - 1
            stratum_nll += sequence_nll
            stratum_tokens += sequence_tokens
            stratum_seconds += elapsed
            top1_by_sequence[index] = np.array(top1).reshape(-1)
            del inputs, logits, loss_sum, top1
            mx.clear_cache()
        total_negative_log_likelihood += stratum_nll
        prediction_tokens += stratum_tokens
        heldout_seconds += stratum_seconds
        loss = stratum_nll / stratum_tokens
        per_stratum.append(
            {
                **selection,
                "forward_seconds": stratum_seconds,
                "loss": loss,
                "perplexity": math.exp(loss),
                "prediction_tokens": stratum_tokens,
                "total_negative_log_likelihood": stratum_nll,
            }
        )
    heldout_peak_memory = mx.get_peak_memory()

    prompt_logits = {}
    generations = {}
    generation_reports = []
    for prompt in prompts:
        ids = _prompt_ids(tokenizer, prompt, config.get("bos_token_id"))
        logits = model(mx.array([ids], dtype=mx.int32))
        mx.eval(logits)
        prompt_logits[prompt] = np.array(logits)
        started = time.perf_counter()
        generated = [
            int(token)
            for token, _ in generate_step(
                mx.array(ids, dtype=mx.int32),
                model,
                max_tokens=generation_tokens,
            )
        ]
        generation_seconds = time.perf_counter() - started
        generations[prompt] = generated
        generation_reports.append(
            {
                "new_token_ids": generated,
                "new_tokens": len(generated),
                "prompt": prompt,
                "seconds": generation_seconds,
                "tokens_per_second": len(generated) / generation_seconds,
            }
        )
        del logits
        mx.clear_cache()

    benchmark_ids = dataset[first_index][:benchmark_prefill_tokens].numpy().astype(np.int32)
    mx.clear_cache()
    benchmark_input = mx.array(benchmark_ids[None])

    # Warm both exact benchmark shapes so compilation is never counted as model speed.
    warmup_cache = model.make_cache()
    warmup_logits = model(benchmark_input, cache=warmup_cache)
    mx.eval(warmup_logits)
    warmup_token = int(mx.argmax(warmup_logits[0, -1]).item())
    warmup_decode = model(mx.array([[warmup_token]], dtype=mx.int32), cache=warmup_cache)
    mx.eval(warmup_decode)
    del warmup_cache, warmup_logits, warmup_decode
    mx.clear_cache()

    prefill_trials = []
    decode_trials = []
    decoded = []
    mx.reset_peak_memory()
    for _ in range(3):
        cache = model.make_cache()
        started = time.perf_counter()
        prefill_logits = model(benchmark_input, cache=cache)
        mx.eval(prefill_logits)
        prefill_trials.append(time.perf_counter() - started)
        next_token = int(mx.argmax(prefill_logits[0, -1]).item())
        del prefill_logits

        trial_decoded = []
        started = time.perf_counter()
        for _ in range(benchmark_decode_tokens):
            decode_logits = model(mx.array([[next_token]], dtype=mx.int32), cache=cache)
            mx.eval(decode_logits)
            next_token = int(mx.argmax(decode_logits[0, -1]).item())
            trial_decoded.append(next_token)
            del decode_logits
        decode_trials.append(time.perf_counter() - started)
        decoded = trial_decoded
        del cache
    prefill_seconds = statistics.median(prefill_trials)
    decode_seconds = statistics.median(decode_trials)
    benchmark_peak_memory = mx.get_peak_memory()

    mean_loss = total_negative_log_likelihood / prediction_tokens
    public = {
        "benchmark": {
            "decode_seconds": decode_seconds,
            "decode_token_ids": decoded,
            "decode_tokens": benchmark_decode_tokens,
            "decode_tokens_per_second": benchmark_decode_tokens / decode_seconds,
            "decode_trial_seconds": decode_trials,
            "peak_memory_bytes": benchmark_peak_memory,
            "prefill_seconds": prefill_seconds,
            "prefill_tokens": benchmark_prefill_tokens,
            "prefill_tokens_per_second": benchmark_prefill_tokens / prefill_seconds,
            "prefill_trial_seconds": prefill_trials,
            "timing": "median of 3 trials after exact-shape warm-up",
        },
        "config": {
            "dtype": config.get("dtype"),
            "model_file": config.get("model_file"),
            "model_type": config.get("model_type"),
            "quantization": config.get("quantization"),
        },
        "generation": generation_reports,
        "heldout": {
            "forward_seconds": heldout_seconds,
            "loss": mean_loss,
            "peak_memory_bytes": heldout_peak_memory,
            "per_stratum": per_stratum,
            "perplexity": math.exp(mean_loss),
            "prediction_tokens": prediction_tokens,
            "sequences": sum(len(item["sequence_indices"]) for item in selections),
            "tokens_per_second": (prediction_tokens + len(top1_by_sequence))
            / heldout_seconds,
            "total_negative_log_likelihood": total_negative_log_likelihood,
        },
        "load_seconds": load_seconds,
    }
    internal = {
        "generations": generations,
        "prompt_logits": prompt_logits,
        "top1_by_sequence": top1_by_sequence,
    }
    del model
    gc.collect()
    mx.clear_cache()
    return {"internal": internal, "public": public}


def run_quantization_gate(
    *,
    fp32_model_path: str | Path,
    quantized_model_path: str | Path,
    manifest_path: str | Path,
    tokenizer_path: str | Path,
    output_path: str | Path,
    prompts: list[str],
    sequences_per_stratum: int = 2,
    generation_tokens: int = 32,
    benchmark_prefill_tokens: int = 512,
    benchmark_decode_tokens: int = 64,
    max_loss_delta: float = 0.10,
    max_perplexity_ratio: float = 1.11,
    min_heldout_top1_agreement: float = 0.90,
    max_artifact_size_ratio: float = 0.25,
) -> dict[str, Any]:
    """Run a deterministic stratified FP32-versus-4-bit quality and runtime gate."""

    fp32_path = Path(fp32_model_path).resolve()
    quantized_path = Path(quantized_model_path).resolve()
    manifest = Path(manifest_path).resolve()
    tokenizer_directory = Path(tokenizer_path).resolve()
    output = Path(output_path).resolve()
    for directory in (fp32_path, quantized_path, tokenizer_directory):
        if not directory.is_dir():
            raise NotADirectoryError(directory)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite quantization report: {output}")
    if not prompts or any(not isinstance(prompt, str) or not prompt for prompt in prompts):
        raise ValueError("quantization gate requires non-empty prompts")
    if generation_tokens < 1 or benchmark_decode_tokens < 1:
        raise ValueError("generation and benchmark decode token counts must be positive")

    from transformers import AutoTokenizer

    from .evaluation_data import validate_heldout_manifest
    from .mlx_conversion import _verify_checkpoint_checksums
    from .packed_data import PackedTokenDataset

    _verify_checkpoint_checksums(fp32_path)
    _verify_checkpoint_checksums(quantized_path)
    heldout = validate_heldout_manifest(manifest)
    sequence_length = heldout["packing"]["sequence_length"]
    if not 1 <= benchmark_prefill_tokens < sequence_length:
        raise ValueError("benchmark_prefill_tokens must be inside the held-out sequence length")
    if benchmark_prefill_tokens + benchmark_decode_tokens > sequence_length:
        raise ValueError("prefill and decode benchmark exceed the Lumi context window")
    selections = _select_stratified_sequences(
        heldout["selection"]["strata"], sequences_per_stratum
    )
    dataset = PackedTokenDataset(manifest, verify_hashes=False)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_directory, local_files_only=True)

    fp32 = _evaluate_model(
        model_path=fp32_path,
        dataset=dataset,
        selections=selections,
        tokenizer=tokenizer,
        prompts=prompts,
        generation_tokens=generation_tokens,
        benchmark_prefill_tokens=benchmark_prefill_tokens,
        benchmark_decode_tokens=benchmark_decode_tokens,
    )
    quantized = _evaluate_model(
        model_path=quantized_path,
        dataset=dataset,
        selections=selections,
        tokenizer=tokenizer,
        prompts=prompts,
        generation_tokens=generation_tokens,
        benchmark_prefill_tokens=benchmark_prefill_tokens,
        benchmark_decode_tokens=benchmark_decode_tokens,
    )

    import numpy as np

    top1_matches = 0
    top1_positions = 0
    for index, reference in fp32["internal"]["top1_by_sequence"].items():
        candidate = quantized["internal"]["top1_by_sequence"][index]
        matches = reference == candidate
        top1_matches += int(matches.sum())
        top1_positions += int(matches.size)
    heldout_top1_agreement = top1_matches / top1_positions

    prompt_reports = []
    generation_matches = 0
    generation_positions = 0
    for prompt in prompts:
        metrics = _array_metrics(
            fp32["internal"]["prompt_logits"][prompt],
            quantized["internal"]["prompt_logits"][prompt],
        )
        reference_tokens = fp32["internal"]["generations"][prompt]
        candidate_tokens = quantized["internal"]["generations"][prompt]
        matches = np.array(reference_tokens) == np.array(candidate_tokens)
        generation_matches += int(matches.sum())
        generation_positions += int(matches.size)
        prefix = 0
        for reference, candidate in zip(reference_tokens, candidate_tokens, strict=True):
            if reference != candidate:
                break
            prefix += 1
        prompt_reports.append(
            {
                "generation_exact": reference_tokens == candidate_tokens,
                "generation_matching_prefix_tokens": prefix,
                "generation_token_agreement": float(matches.mean()),
                "logits": metrics,
                "prompt": prompt,
            }
        )
    generation_agreement = generation_matches / generation_positions

    fp32_loss = fp32["public"]["heldout"]["loss"]
    quantized_loss = quantized["public"]["heldout"]["loss"]
    loss_delta = quantized_loss - fp32_loss
    perplexity_ratio = math.exp(loss_delta)
    fp32_artifact = _model_artifact(fp32_path)
    quantized_artifact = _model_artifact(quantized_path)
    size_ratio = quantized_artifact["disk_bytes"] / fp32_artifact["disk_bytes"]
    fp32_config = json.loads((fp32_path / "config.json").read_text(encoding="utf-8"))
    quantized_config = json.loads(
        (quantized_path / "config.json").read_text(encoding="utf-8")
    )
    architecture_fields = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "max_position_embeddings",
        "rope_theta",
        "rms_norm_eps",
        "attention_bias",
        "tie_word_embeddings",
    )
    architecture_config_exact = all(
        fp32_config[field] == quantized_config[field] for field in architecture_fields
    )
    fp32_config_exact = bool(
        fp32_config.get("model_type") == "lumi"
        and fp32_config.get("model_file") == "lumi_mlx.py"
        and fp32_config.get("quantization") is None
        and fp32_config.get("quantization_config") is None
    )
    quantization_config = quantized["public"]["config"]["quantization"]
    expected_quantization = {"bits": 4, "group_size": 64, "mode": "affine"}
    quantization_exact = bool(
        quantized_config.get("model_type") == "lumi"
        and quantized_config.get("model_file") == "lumi_mlx.py"
        and quantization_config == expected_quantization
        and quantized_config.get("quantization_config") == expected_quantization
    )
    passed = bool(
        architecture_config_exact
        and fp32_config_exact
        and quantization_exact
        and loss_delta <= max_loss_delta
        and perplexity_ratio <= max_perplexity_ratio
        and heldout_top1_agreement >= min_heldout_top1_agreement
        and size_ratio <= max_artifact_size_ratio
    )
    report = {
        "artifacts": {"fp32": fp32_artifact, "quantized": quantized_artifact},
        "completed_at": _utc_now(),
        "comparison": {
            "architecture_config_exact": architecture_config_exact,
            "artifact_size_ratio": size_ratio,
            "artifact_size_reduction_fraction": 1.0 - size_ratio,
            "generation_token_agreement": generation_agreement,
            "generation_token_agreement_interpretation": (
                "diagnostic only: after the first free-running token differs, "
                "later positions no longer share the same context"
            ),
            "heldout_loss_delta": loss_delta,
            "heldout_perplexity_ratio": perplexity_ratio,
            "heldout_top1_agreement": heldout_top1_agreement,
            "heldout_top1_matches": top1_matches,
            "heldout_top1_positions": top1_positions,
            "fp32_config_exact": fp32_config_exact,
            "prompts": prompt_reports,
            "quantization_config_exact": quantization_exact,
        },
        "fp32": fp32["public"],
        "locations": {
            "fp32": str(fp32_path),
            "heldout_manifest": str(manifest),
            "quantized": str(quantized_path),
            "tokenizer": str(tokenizer_directory),
        },
        "quantized": quantized["public"],
        "runtime": {
            "machine": platform.machine(),
            "mlx_lm_version": version("mlx-lm"),
            "mlx_version": version("mlx"),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "schema_version": QUANTIZATION_GATE_SCHEMA_VERSION,
        "selection": {
            "heldout_build_fingerprint": heldout["build_fingerprint"],
            "sequence_length": sequence_length,
            "sequences": selections,
            "sequences_per_stratum": sequences_per_stratum,
        },
        "status": "pass" if passed else "fail",
        "thresholds": {
            "max_artifact_size_ratio": max_artifact_size_ratio,
            "max_heldout_loss_delta": max_loss_delta,
            "max_heldout_perplexity_ratio": max_perplexity_ratio,
            "generation_token_agreement": "diagnostic_only",
            "min_heldout_top1_agreement": min_heldout_top1_agreement,
            "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
        },
    }
    _write_json_atomic(output, report)
    if not passed:
        raise RuntimeError(f"4-bit MLX quantization gate failed; inspect {output}")
    return report
