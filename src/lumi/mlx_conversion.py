"""Deterministic, provenance-tracked conversion of Lumi checkpoints to MLX."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import time
import uuid
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

MLX_FORMAT_SCHEMA_VERSION = 1
MLX_MODEL_FILENAME = "lumi_mlx.py"
CONVERSION_REPORT_FILENAME = "conversion-report.json"
CHECKSUM_FILENAME = "SHA256SUMS"
SUPPORTED_DTYPES = ("float32", "float16", "bfloat16")
TOKENIZER_FILENAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer.model",
    "spiece.model",
    "lumi_tokenizer_metadata.json",
)
REQUIRED_CONFIG_FIELDS = (
    "model_type",
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


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_source_config(config: dict[str, Any]) -> None:
    missing = [name for name in REQUIRED_CONFIG_FIELDS if name not in config]
    if missing:
        raise ValueError(f"Lumi checkpoint config is missing fields: {', '.join(missing)}")
    if config["model_type"] != "lumi":
        raise ValueError(f"expected a Lumi checkpoint, got {config['model_type']!r}")
    positive_integers = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "max_position_embeddings",
    )
    for field in positive_integers:
        value = config[field]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{field} must be a positive integer, got {value!r}")
    if config["num_attention_heads"] % config["num_key_value_heads"]:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    if config["head_dim"] % 2:
        raise ValueError("head_dim must be even for rotary embeddings")
    if not isinstance(config["rope_theta"], (int, float)) or config["rope_theta"] <= 0:
        raise ValueError("rope_theta must be positive")
    if not isinstance(config["rms_norm_eps"], (int, float)) or config["rms_norm_eps"] <= 0:
        raise ValueError("rms_norm_eps must be positive")
    if not isinstance(config["attention_bias"], bool):
        raise ValueError("attention_bias must be boolean")
    if not isinstance(config["tie_word_embeddings"], bool):
        raise ValueError("tie_word_embeddings must be boolean")


def build_mlx_config(source_config: dict[str, Any], *, dtype: str) -> dict[str, Any]:
    """Build the explicit MLX config without claiming Lumi is another model type."""

    validate_source_config(source_config)
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"dtype must be one of {', '.join(SUPPORTED_DTYPES)}")
    config = dict(source_config)
    config.pop("quantization", None)
    config.pop("quantization_config", None)
    config.update(
        {
            "dtype": dtype,
            "lumi_mlx_format_version": MLX_FORMAT_SCHEMA_VERSION,
            "model_file": MLX_MODEL_FILENAME,
            "model_type": "lumi",
            "rope_traditional": False,
        }
    )
    return config


def _weight_files(checkpoint: Path) -> list[Path]:
    files = sorted(checkpoint.glob("model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"checkpoint has no safetensors weights: {checkpoint}")
    return files


def _tokenizer_files(tokenizer_path: Path) -> list[Path]:
    files = [tokenizer_path / name for name in TOKENIZER_FILENAMES]
    files = [path for path in files if path.is_file()]
    names = {path.name for path in files}
    for required in ("tokenizer.json", "tokenizer_config.json"):
        if required not in names:
            raise FileNotFoundError(f"tokenizer artifact is missing {tokenizer_path / required}")
    return files


def _verify_checkpoint_checksums(checkpoint: Path) -> None:
    checksum_path = checkpoint / CHECKSUM_FILENAME
    if not checksum_path.is_file():
        return
    for line_number, raw_line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line:
            continue
        pieces = raw_line.split("  ", maxsplit=1)
        if len(pieces) != 2 or len(pieces[0]) != 64:
            raise ValueError(f"invalid checksum entry at {checksum_path}:{line_number}")
        digest, name = pieces
        candidate = checkpoint / name
        if candidate.parent != checkpoint or not candidate.is_file():
            raise ValueError(f"unsafe or missing checksum target: {name!r}")
        if _sha256_file(candidate) != digest:
            raise ValueError(f"checkpoint checksum mismatch: {candidate}")


def _copy_tokenizer(tokenizer_path: Path, destination: Path) -> list[Path]:
    copied = []
    for source in _tokenizer_files(tokenizer_path):
        target = destination / source.name
        shutil.copyfile(source, target)
        copied.append(target)
    return copied


def _artifact_inventory(directory: Path, *, excluded: set[str] | None = None) -> list[dict]:
    excluded = excluded or set()
    return [
        _file_artifact(path)
        for path in sorted(directory.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name not in excluded
    ]


def _write_checksums(directory: Path) -> None:
    artifacts = _artifact_inventory(directory, excluded={CHECKSUM_FILENAME})
    with (directory / CHECKSUM_FILENAME).open("x", encoding="utf-8") as handle:
        for artifact in artifacts:
            handle.write(f"{artifact['sha256']}  {artifact['name']}\n")
        handle.flush()
        os.fsync(handle.fileno())


def convert_checkpoint_to_mlx(
    *,
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    output_path: str | Path,
    dtype: str = "float32",
    quantize: bool = False,
    quantization_bits: int = 4,
    quantization_group_size: int = 64,
    quantization_mode: str = "affine",
) -> dict[str, Any]:
    """Strictly convert one local Lumi checkpoint into an atomic MLX artifact."""

    checkpoint = Path(checkpoint_path).resolve()
    tokenizer = Path(tokenizer_path).resolve()
    output = Path(output_path).resolve()
    if not checkpoint.is_dir():
        raise NotADirectoryError(checkpoint)
    if not tokenizer.is_dir():
        raise NotADirectoryError(tokenizer)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite MLX artifact: {output}")
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"dtype must be one of {', '.join(SUPPORTED_DTYPES)}")
    if quantization_bits < 2 or quantization_bits > 8:
        raise ValueError("quantization_bits must be between 2 and 8")
    if quantization_group_size < 1:
        raise ValueError("quantization_group_size must be positive")
    if quantization_mode != "affine":
        raise ValueError("the Lumi v0.1 gate currently permits affine quantization only")

    _verify_checkpoint_checksums(checkpoint)
    config_path = checkpoint / "config.json"
    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    config = build_mlx_config(source_config, dtype=dtype)
    weight_files = _weight_files(checkpoint)
    tokenizer_files = _tokenizer_files(tokenizer)

    from transformers import AutoTokenizer

    loaded_tokenizer = AutoTokenizer.from_pretrained(tokenizer, local_files_only=True)
    if len(loaded_tokenizer) != source_config["vocab_size"]:
        raise ValueError("checkpoint and tokenizer vocabulary sizes differ")
    for field in ("pad_token_id", "bos_token_id", "eos_token_id"):
        if getattr(loaded_tokenizer, field) != source_config.get(field):
            raise ValueError(f"checkpoint and tokenizer {field} values differ")
    doc_token_id = loaded_tokenizer.convert_tokens_to_ids("<|doc|>")
    if doc_token_id != source_config.get("doc_token_id"):
        raise ValueError("checkpoint and tokenizer doc_token_id values differ")

    source_model_file = Path(__file__).with_name("mlx_model.py")
    source = {
        "checkpoint": {
            "config": _file_artifact(config_path),
            "weights": [_file_artifact(path) for path in weight_files],
        },
        "tokenizer": {"files": [_file_artifact(path) for path in tokenizer_files]},
        "mlx_model_source": _file_artifact(source_model_file),
    }
    request = {
        "dtype": dtype,
        "format_schema_version": MLX_FORMAT_SCHEMA_VERSION,
        "quantization": (
            {
                "bits": quantization_bits,
                "group_size": quantization_group_size,
                "mode": quantization_mode,
            }
            if quantize
            else None
        ),
        "source": source,
    }
    conversion_fingerprint = _fingerprint(request)
    staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    staging.mkdir(parents=True)
    started = time.perf_counter()
    try:
        import mlx.core as mx
        from mlx.utils import tree_map_with_path
        from mlx_lm.utils import (
            get_total_parameters,
            load_model,
            quantize_model,
            save_config,
            save_model,
        )

        from .mlx_model import Model, ModelArgs

        weights: dict[str, mx.array] = {}
        for weight_file in weight_files:
            loaded = mx.load(str(weight_file))
            duplicates = weights.keys() & loaded.keys()
            if duplicates:
                raise ValueError(f"duplicate checkpoint weight keys: {sorted(duplicates)[:3]}")
            weights.update(loaded)

        model = Model(ModelArgs.from_dict(config))
        weights = model.sanitize(weights)
        model.load_weights(list(weights.items()), strict=True)
        del weights

        target_dtype = getattr(mx, dtype)

        def cast_floating(_path: str, value: mx.array) -> mx.array:
            if mx.issubdtype(value.dtype, mx.floating):
                return value.astype(target_dtype)
            return value

        model.update(tree_map_with_path(cast_floating, model.parameters()))
        model.eval()
        mx.eval(model.parameters())
        parameter_count = get_total_parameters(model)

        if quantize:
            model, config = quantize_model(
                model,
                config,
                group_size=quantization_group_size,
                bits=quantization_bits,
                mode=quantization_mode,
            )
            mx.eval(model.parameters())

        save_model(staging, model, donate_model=True)
        save_config(dict(config), staging / "config.json")
        shutil.copyfile(source_model_file, staging / MLX_MODEL_FILENAME)
        _copy_tokenizer(tokenizer, staging)
        generation_config = checkpoint / "generation_config.json"
        if generation_config.is_file():
            shutil.copyfile(generation_config, staging / generation_config.name)

        # A fresh strict load and forward pass prove that the finished directory,
        # not just the in-memory converter model, is independently usable.
        reloaded_model, reloaded_config = load_model(staging, strict=True, lazy=False)
        probe_ids = mx.array(
            [[source_config["bos_token_id"], source_config["doc_token_id"]]],
            dtype=mx.int32,
        )
        probe_logits = reloaded_model(probe_ids)
        mx.eval(probe_logits)
        expected_shape = (1, 2, source_config["vocab_size"])
        if probe_logits.shape != expected_shape:
            raise RuntimeError(
                f"converted MLX probe shape changed: {probe_logits.shape} != {expected_shape}"
            )
        if not bool(mx.all(mx.isfinite(probe_logits)).item()):
            raise RuntimeError("converted MLX probe produced non-finite logits")

        core_files = _artifact_inventory(staging)
        report = {
            "completed_at": _utc_now(),
            "conversion_fingerprint": conversion_fingerprint,
            "format_schema_version": MLX_FORMAT_SCHEMA_VERSION,
            "locations": {
                "checkpoint": str(checkpoint),
                "output": str(output),
                "tokenizer": str(tokenizer),
            },
            "output": {
                "config_model_file": reloaded_config["model_file"],
                "files": core_files,
                "model_parameters": parameter_count,
                "probe_logits_shape": list(probe_logits.shape),
            },
            "request": request,
            "runtime": {
                "elapsed_seconds": time.perf_counter() - started,
                "machine": platform.machine(),
                "mlx_lm_version": version("mlx-lm"),
                "mlx_version": version("mlx"),
                "platform": platform.platform(),
                "python": sys.version.split()[0],
            },
            "status": "complete",
        }
        _write_json(staging / CONVERSION_REPORT_FILENAME, report)
        _write_checksums(staging)
        os.replace(staging, output)
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
