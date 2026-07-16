"""Measured PyTorch-to-MLX logits and generation parity gate for Lumi."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
import uuid
from contextlib import ExitStack
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

PARITY_SCHEMA_VERSION = 1


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


def _weight_artifacts(directory: Path) -> list[dict[str, Any]]:
    config = directory / "config.json"
    weights = sorted(directory.glob("model*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"model artifact has no safetensors: {directory}")
    indexes = sorted(directory.glob("model*.safetensors.index.json"))
    return [_file_artifact(config), *map(_file_artifact, indexes), *map(_file_artifact, weights)]


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _cosine_similarity(reference, candidate) -> float:
    import numpy as np

    reference_flat = reference.reshape(-1).astype(np.float64)
    candidate_flat = candidate.reshape(-1).astype(np.float64)
    denominator = float(np.linalg.norm(reference_flat) * np.linalg.norm(candidate_flat))
    if denominator == 0:
        return 1.0 if np.array_equal(reference_flat, candidate_flat) else 0.0
    return float(np.dot(reference_flat, candidate_flat) / denominator)


def _array_metrics(reference, candidate) -> dict[str, float | bool]:
    import numpy as np

    if reference.shape != candidate.shape:
        raise ValueError(f"logits shapes differ: {reference.shape} != {candidate.shape}")
    reference_float = reference.astype(np.float64)
    candidate_float = candidate.astype(np.float64)
    difference = np.abs(reference_float - candidate_float)
    reference_rms = math.sqrt(float(np.square(reference_float).mean()))
    root_mean_square_error = math.sqrt(float(np.square(difference).mean()))
    return {
        "all_finite": bool(np.isfinite(candidate).all()),
        "cosine_similarity": _cosine_similarity(reference, candidate),
        "max_absolute_error": float(difference.max(initial=0.0)),
        "mean_absolute_error": float(difference.mean()),
        "normalized_root_mean_square_error": (
            root_mean_square_error / reference_rms if reference_rms else 0.0
        ),
        "reference_max_absolute_value": float(np.abs(reference_float).max(initial=0.0)),
        "reference_root_mean_square": reference_rms,
        "root_mean_square_error": root_mean_square_error,
    }


def _saved_tensor_parity(reference_files: list[Path], candidate_files: list[Path]) -> dict:
    import numpy as np
    from safetensors import safe_open

    with ExitStack() as stack:
        reference_readers = [
            stack.enter_context(safe_open(path, framework="numpy")) for path in reference_files
        ]
        candidate_readers = [
            stack.enter_context(safe_open(path, framework="numpy")) for path in candidate_files
        ]
        reference_locations = {
            key: reader for reader in reference_readers for key in reader.keys()
        }
        candidate_locations = {
            key: reader for reader in candidate_readers for key in reader.keys()
        }
        reference_keys = set(reference_locations)
        candidate_keys = set(candidate_locations)
        changed = []
        maximum_error = 0.0
        for key in sorted(reference_keys & candidate_keys):
            reference = reference_locations[key].get_tensor(key)
            candidate = candidate_locations[key].get_tensor(key)
            if not np.array_equal(reference, candidate):
                changed.append(key)
                if reference.shape == candidate.shape:
                    maximum_error = max(
                        maximum_error,
                        float(
                            np.max(
                                np.abs(
                                    reference.astype(np.float64)
                                    - candidate.astype(np.float64)
                                )
                            )
                        ),
                    )
                else:
                    maximum_error = math.inf
        source_only = sorted(reference_keys - candidate_keys)
        candidate_only = sorted(candidate_keys - reference_keys)
        allowed_source_only = all(
            name.endswith(".self_attn.rotary.inv_freq") for name in source_only
        )
        passed = not changed and not candidate_only and allowed_source_only
        return {
            "candidate_only": candidate_only,
            "changed_common_tensors": changed,
            "common_tensors": len(reference_keys & candidate_keys),
            "exact": passed,
            "max_absolute_error": maximum_error,
            "source_only_recomputed_buffers": source_only,
        }


def _prompt_token_ids(tokenizer, prompt: str, bos_token_id: int | None) -> list[int]:
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    return [bos_token_id, *token_ids] if bos_token_id is not None else token_ids


def _validate_thresholds(
    *,
    max_absolute_error: float,
    max_mean_absolute_error: float,
    min_cosine_similarity: float,
    max_normalized_root_mean_square_error: float,
    max_cached_logprob_error: float,
) -> None:
    if max_absolute_error <= 0 or max_mean_absolute_error <= 0:
        raise ValueError("absolute-error thresholds must be positive")
    if not 0 < min_cosine_similarity <= 1:
        raise ValueError("min_cosine_similarity must be in (0, 1]")
    if max_normalized_root_mean_square_error <= 0:
        raise ValueError("max_normalized_root_mean_square_error must be positive")
    if max_cached_logprob_error <= 0:
        raise ValueError("max_cached_logprob_error must be positive")


def check_mlx_parity(
    *,
    checkpoint_path: str | Path,
    mlx_path: str | Path,
    tokenizer_path: str | Path,
    output_path: str | Path,
    prompts: list[str],
    generation_max_new_tokens: int = 32,
    max_absolute_error: float = 2e-2,
    max_mean_absolute_error: float = 3e-3,
    min_cosine_similarity: float = 0.9999999,
    max_normalized_root_mean_square_error: float = 1e-3,
    max_cached_logprob_error: float = 2e-2,
) -> dict[str, Any]:
    """Compare an MLX artifact against the exact source PyTorch checkpoint."""

    checkpoint = Path(checkpoint_path).resolve()
    mlx_directory = Path(mlx_path).resolve()
    tokenizer_directory = Path(tokenizer_path).resolve()
    output = Path(output_path).resolve()
    for directory in (checkpoint, mlx_directory, tokenizer_directory):
        if not directory.is_dir():
            raise NotADirectoryError(directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite parity report: {output}")
    if not prompts or any(not isinstance(prompt, str) or not prompt for prompt in prompts):
        raise ValueError("parity requires at least one non-empty prompt")
    if generation_max_new_tokens < 1:
        raise ValueError("generation_max_new_tokens must be positive")
    _validate_thresholds(
        max_absolute_error=max_absolute_error,
        max_mean_absolute_error=max_mean_absolute_error,
        min_cosine_similarity=min_cosine_similarity,
        max_normalized_root_mean_square_error=max_normalized_root_mean_square_error,
        max_cached_logprob_error=max_cached_logprob_error,
    )

    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    import torch
    import torch.nn.functional as torch_functional
    from mlx_lm.generate import generate_step
    from mlx_lm.utils import load_model
    from transformers import AutoTokenizer

    from .modeling_lumi import LumiForCausalLM

    mx.reset_peak_memory()
    load_started = time.perf_counter()
    torch_model = LumiForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        dtype=torch.float32,
    )
    torch_model.eval()
    mlx_model, mlx_config = load_model(mlx_directory, strict=True, lazy=False)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_directory, local_files_only=True)
    load_seconds = time.perf_counter() - load_started

    if len(tokenizer) != torch_model.config.vocab_size:
        raise ValueError("reference checkpoint and tokenizer vocabulary sizes differ")
    if mlx_config["vocab_size"] != torch_model.config.vocab_size:
        raise ValueError("PyTorch and MLX vocabulary sizes differ")
    if mlx_config["model_type"] != "lumi" or not mlx_config.get("model_file"):
        raise ValueError("MLX artifact does not identify its explicit Lumi model file")

    source_config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
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
    config_parity = {
        field: {"mlx": mlx_config[field], "pytorch": source_config[field]}
        for field in architecture_fields
    }
    config_exact = all(
        values["mlx"] == values["pytorch"] for values in config_parity.values()
    )

    source_weight_files = sorted(checkpoint.glob("model*.safetensors"))
    mlx_weight_files = sorted(mlx_directory.glob("model*.safetensors"))
    tensor_parity = _saved_tensor_parity(source_weight_files, mlx_weight_files)

    rng = np.random.default_rng(20260716)
    rope_input = rng.standard_normal(
        (
            1,
            source_config["num_key_value_heads"],
            7,
            source_config["head_dim"],
        ),
        dtype=np.float32,
    )
    torch_rope_input = torch.from_numpy(rope_input)
    inverse_frequency = 1.0 / (
        source_config["rope_theta"]
        ** (
            torch.arange(0, source_config["head_dim"], 2, dtype=torch.float32)
            / source_config["head_dim"]
        )
    )
    positions = torch.arange(rope_input.shape[-2], dtype=torch.float32)
    frequencies = torch.einsum("t,d->td", positions, inverse_frequency)
    embeddings = torch.cat((frequencies, frequencies), dim=-1)
    first_half, second_half = torch_rope_input.chunk(2, dim=-1)
    rotated_half = torch.cat((-second_half, first_half), dim=-1)
    torch_rope = (
        torch_rope_input * embeddings.cos()[None, None]
        + rotated_half * embeddings.sin()[None, None]
    ).numpy()
    mlx_rope_array = nn.RoPE(
        source_config["head_dim"],
        traditional=False,
        base=source_config["rope_theta"],
    )(mx.array(rope_input))
    mx.eval(mlx_rope_array)
    rope_metrics = _array_metrics(torch_rope, np.array(mlx_rope_array))
    rope_pass = bool(
        rope_metrics["all_finite"]
        and rope_metrics["max_absolute_error"] <= 1e-6
        and rope_metrics["mean_absolute_error"] <= 1e-7
    )

    prompt_reports = []
    generation_reports = []
    all_prompt_gates = True
    all_generation_gates = True
    torch_forward_seconds = 0.0
    mlx_forward_seconds = 0.0

    with torch.inference_mode():
        for prompt in prompts:
            token_ids = _prompt_token_ids(tokenizer, prompt, torch_model.config.bos_token_id)
            torch_input = torch.tensor([token_ids], dtype=torch.long)

            started = time.perf_counter()
            reference_tensor = torch_model(input_ids=torch_input).logits
            torch_forward_seconds += time.perf_counter() - started
            reference_logits = reference_tensor.detach().cpu().numpy()

            started = time.perf_counter()
            candidate_array = mlx_model(mx.array([token_ids], dtype=mx.int32))
            mx.eval(candidate_array)
            mlx_forward_seconds += time.perf_counter() - started
            candidate_logits = np.array(candidate_array)

            metrics = _array_metrics(reference_logits, candidate_logits)
            reference_top = np.argmax(reference_logits, axis=-1)
            candidate_top = np.argmax(candidate_logits, axis=-1)
            top_match = reference_top == candidate_top
            metrics.update(
                {
                    "top1_agreement": float(top_match.mean()),
                    "top1_matches": int(top_match.sum()),
                    "top1_positions": int(top_match.size),
                }
            )
            prompt_pass = bool(
                metrics["all_finite"]
                and metrics["max_absolute_error"] <= max_absolute_error
                and metrics["mean_absolute_error"] <= max_mean_absolute_error
                and metrics["cosine_similarity"] >= min_cosine_similarity
                and metrics["normalized_root_mean_square_error"]
                <= max_normalized_root_mean_square_error
                and metrics["top1_agreement"] == 1.0
            )
            all_prompt_gates &= prompt_pass
            prompt_reports.append(
                {
                    "metrics": metrics,
                    "pass": prompt_pass,
                    "prompt": prompt,
                    "token_ids": token_ids,
                    "tokens": len(token_ids),
                }
            )

            reference_generated: list[int] = []
            reference_step_logprobs = []
            running_ids = list(token_ids)
            for _ in range(generation_max_new_tokens):
                running_input = torch.tensor([running_ids], dtype=torch.long)
                logits = torch_model(input_ids=running_input).logits[0, -1]
                logprobs = torch_functional.log_softmax(logits, dim=-1)
                next_token = int(torch.argmax(logprobs).item())
                reference_generated.append(next_token)
                reference_step_logprobs.append(logprobs.detach().cpu().numpy())
                running_ids.append(next_token)

            candidate_generated = []
            cached_max_errors = []
            cached_mean_errors = []
            for step, (token_id, candidate_logprobs) in enumerate(
                generate_step(
                    mx.array(token_ids, dtype=mx.int32),
                    mlx_model,
                    max_tokens=generation_max_new_tokens,
                )
            ):
                candidate_generated.append(int(token_id))
                candidate_values = np.array(candidate_logprobs)
                cached_metrics = _array_metrics(
                    reference_step_logprobs[step], candidate_values
                )
                cached_max_errors.append(cached_metrics["max_absolute_error"])
                cached_mean_errors.append(cached_metrics["mean_absolute_error"])

            exact_tokens = candidate_generated == reference_generated
            cached_max_error = max(cached_max_errors, default=0.0)
            cached_mean_error = max(cached_mean_errors, default=0.0)
            generation_pass = bool(
                exact_tokens and cached_max_error <= max_cached_logprob_error
            )
            all_generation_gates &= generation_pass
            generation_reports.append(
                {
                    "cached_logprob_max_absolute_error": cached_max_error,
                    "cached_logprob_max_mean_absolute_error": cached_mean_error,
                    "candidate_new_token_ids": candidate_generated,
                    "exact_token_match": exact_tokens,
                    "new_tokens": generation_max_new_tokens,
                    "pass": generation_pass,
                    "prompt": prompt,
                    "reference_new_token_ids": reference_generated,
                }
            )

    passed = bool(
        config_exact
        and tensor_parity["exact"]
        and rope_pass
        and all_prompt_gates
        and all_generation_gates
    )
    report = {
        "architecture": {
            "config_exact": config_exact,
            "config_fields": config_parity,
            "rope": {"metrics": rope_metrics, "pass": rope_pass},
            "saved_tensors": tensor_parity,
        },
        "artifacts": {
            "mlx": {"files": _weight_artifacts(mlx_directory)},
            "pytorch": {"files": _weight_artifacts(checkpoint)},
            "tokenizer": {
                "files": [
                    _file_artifact(tokenizer_directory / name)
                    for name in ("tokenizer.json", "tokenizer_config.json")
                ]
            },
        },
        "completed_at": _utc_now(),
        "generation": generation_reports,
        "locations": {
            "mlx": str(mlx_directory),
            "pytorch": str(checkpoint),
            "tokenizer": str(tokenizer_directory),
        },
        "prompts": prompt_reports,
        "runtime": {
            "load_seconds": load_seconds,
            "machine": platform.machine(),
            "mlx_forward_seconds": mlx_forward_seconds,
            "mlx_lm_version": version("mlx-lm"),
            "mlx_peak_memory_bytes": mx.get_peak_memory(),
            "mlx_version": version("mlx"),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch_device": "cpu",
            "torch_forward_seconds": torch_forward_seconds,
            "torch_version": torch.__version__,
        },
        "schema_version": PARITY_SCHEMA_VERSION,
        "status": "pass" if passed else "fail",
        "thresholds": {
            "cached_logprob_max_absolute_error": max_cached_logprob_error,
            "generation_exact_token_match": True,
            "logits_max_absolute_error": max_absolute_error,
            "logits_max_mean_absolute_error": max_mean_absolute_error,
            "logits_min_cosine_similarity": min_cosine_similarity,
            "logits_max_normalized_root_mean_square_error": (
                max_normalized_root_mean_square_error
            ),
            "logits_top1_agreement": 1.0,
            "rope_max_absolute_error": 1e-6,
            "rope_max_mean_absolute_error": 1e-7,
            "saved_tensor_parity": "exact",
        },
    }
    _write_json_atomic(output, report)
    if not passed:
        raise RuntimeError(f"PyTorch-to-MLX parity gate failed; inspect {output}")
    return report
