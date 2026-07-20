"""Final-width Lumi PyTorch-to-MLX portability gate.

The production 4B model is too large for an FP32 cross-runtime parity run on a
16GB Mac.  This gate preserves every width-sensitive production dimension and
the frozen tokenizer while reducing only decoder depth from 36 layers to one.
The existing 15-layer Lumi-150M gate covers repeated-layer and long-cache
behavior; this gate closes the otherwise untested 4B projection-shape boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .mlx_conversion import convert_checkpoint_to_mlx, validate_source_config
from .mlx_parity import check_mlx_parity

GATE_SCHEMA_VERSION = 1
GATE_REPORT_FILENAME = "gate-report.json"
MANIFEST_FILENAME = "manifest.sha256"
FINAL_WIDTH_FIELDS = (
    "hidden_size",
    "intermediate_size",
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
PROVENANCE_MODULES = (
    "configuration_lumi.py",
    "mlx_conversion.py",
    "mlx_model.py",
    "mlx_parity.py",
    "mlx_topology_gate.py",
    "modeling_lumi.py",
)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_gate_inputs(plan_path: str | Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Load and hash-bind the gate plan and its final 4B source config."""

    plan_file = Path(plan_path).resolve()
    plan = _read_json(plan_file)
    if plan.get("schema_version") != GATE_SCHEMA_VERSION:
        raise ValueError(f"unsupported MLX topology gate schema: {plan.get('schema_version')!r}")

    source_spec = plan.get("source_config")
    if not isinstance(source_spec, dict):
        raise ValueError("source_config must be an object")
    relative_source = source_spec.get("path")
    if not isinstance(relative_source, str) or not relative_source:
        raise ValueError("source_config.path must be a non-empty relative path")
    source_path = (plan_file.parent / relative_source).resolve()
    if source_path.parent != plan_file.parent:
        raise ValueError("source_config.path must stay inside the plan directory")
    expected_sha256 = source_spec.get("sha256")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("source_config.sha256 must be a SHA-256 digest")
    actual_sha256 = _sha256_file(source_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"final 4B config hash mismatch: expected {expected_sha256}, got {actual_sha256}"
        )

    source_config = _read_json(source_path)
    validate_source_config(source_config)
    return plan, source_path, source_config


def build_final_width_fixture(
    source_config: Mapping[str, Any], *, num_hidden_layers: int
) -> dict[str, Any]:
    """Reduce only depth while preserving every width-sensitive 4B field."""

    validate_source_config(dict(source_config))
    fixture = dict(source_config)
    fixture["num_hidden_layers"] = _positive_int(
        num_hidden_layers, name="fixture.num_hidden_layers"
    )
    changed = {name for name in fixture if fixture.get(name) != source_config.get(name)}
    if changed != {"num_hidden_layers"}:
        raise ValueError(f"fixture may change only num_hidden_layers, got {sorted(changed)}")
    validate_source_config(fixture)
    return fixture


def _projection_shapes(config: Mapping[str, Any]) -> dict[str, list[int]]:
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["intermediate_size"])
    query_size = int(config["num_attention_heads"]) * int(config["head_dim"])
    key_value_size = int(config["num_key_value_heads"]) * int(config["head_dim"])
    return {
        "down_proj": [hidden_size, intermediate_size],
        "gate_proj": [intermediate_size, hidden_size],
        "k_proj": [key_value_size, hidden_size],
        "o_proj": [hidden_size, query_size],
        "q_proj": [query_size, hidden_size],
        "up_proj": [intermediate_size, hidden_size],
        "v_proj": [key_value_size, hidden_size],
    }


def _write_directory_checksums(directory: Path) -> None:
    files = sorted(
        path for path in directory.iterdir() if path.is_file() and path.name != "SHA256SUMS"
    )
    with (directory / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for path in files:
            handle.write(f"{_sha256_file(path)}  {path.name}\n")
        handle.flush()
        os.fsync(handle.fileno())


def _copy_provenance(plan: Path, source_config: Path, output: Path) -> dict[str, Any]:
    provenance = output / "provenance"
    provenance.mkdir()
    inputs = {
        "gate-plan.json": plan,
        "source-config.json": source_config,
    }
    module_directory = Path(__file__).parent
    implementations = {name: module_directory / name for name in PROVENANCE_MODULES}
    for name, source in (*inputs.items(), *implementations.items()):
        shutil.copyfile(source, provenance / name)

    def artifact(name: str) -> dict[str, Any]:
        path = provenance / name
        return {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}

    return {
        "implementation": {name: artifact(name) for name in sorted(implementations)},
        "inputs": {name: artifact(name) for name in sorted(inputs)},
    }


def _write_manifest(root: Path) -> Path:
    manifest = root / MANIFEST_FILENAME
    files = sorted(path for path in root.rglob("*") if path.is_file() and path != manifest)
    with manifest.open("x", encoding="utf-8") as handle:
        for path in files:
            handle.write(f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}\n")
        handle.flush()
        os.fsync(handle.fileno())
    return manifest


def verify_final_mlx_topology_gate(output_path: str | Path) -> dict[str, Any]:
    """Independently verify hashes and final-width gate invariants."""

    root = Path(output_path).resolve()
    manifest = root / MANIFEST_FILENAME
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        pieces = raw_line.split("  ", maxsplit=1)
        if len(pieces) != 2 or len(pieces[0]) != 64:
            raise ValueError(f"invalid manifest entry at line {line_number}")
        expected, relative_name = pieces
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts or relative_name in seen:
            raise ValueError(f"unsafe or duplicate manifest path: {relative_name!r}")
        seen.add(relative_name)
        candidate = root / relative
        if not candidate.is_file() or _sha256_file(candidate) != expected:
            raise ValueError(f"manifest mismatch: {relative_name}")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest
    }
    if seen != actual_files:
        raise ValueError("manifest does not cover the exact sealed file set")

    report = _read_json(root / GATE_REPORT_FILENAME)
    if report.get("schema_version") != GATE_SCHEMA_VERSION or report.get("status") != "pass":
        raise ValueError("MLX topology gate report is not a passing schema-v1 report")
    fixture = report.get("fixture", {})
    source = report.get("source", {})
    if fixture.get("num_hidden_layers") != 1 or source.get("num_hidden_layers") != 36:
        raise ValueError("expected the sealed 36-to-1 reduced-depth fixture")
    for field in FINAL_WIDTH_FIELDS:
        if fixture.get(field) != source.get(field):
            raise ValueError(f"final-width field drifted in fixture: {field}")
    if report.get("parity", {}).get("status") != "pass":
        raise ValueError("sealed MLX topology parity did not pass")
    if report.get("conversion", {}).get("status") != "complete":
        raise ValueError("sealed MLX topology conversion did not complete")
    provenance = report.get("provenance", {})
    for group in ("implementation", "inputs"):
        expected_group = provenance.get(group)
        if not isinstance(expected_group, dict) or not expected_group:
            raise ValueError(f"missing sealed provenance group: {group}")
        for name, artifact in expected_group.items():
            candidate = root / "provenance" / name
            if (
                not candidate.is_file()
                or artifact.get("bytes") != candidate.stat().st_size
                or artifact.get("sha256") != _sha256_file(candidate)
            ):
                raise ValueError(f"sealed provenance mismatch: {name}")
    parity = _read_json(root / "parity-report.json")
    if parity.get("status") != "pass":
        raise ValueError("canonical final-width parity report did not pass")
    architecture = parity.get("architecture", {})
    if not architecture.get("config_exact") or not architecture.get("saved_tensors", {}).get(
        "exact"
    ):
        raise ValueError("final-width parity lacks exact config or saved tensors")
    if not architecture.get("rope", {}).get("pass"):
        raise ValueError("final-width RoPE parity did not pass")
    if any(prompt.get("metrics", {}).get("top1_agreement") != 1.0 for prompt in parity["prompts"]):
        raise ValueError("final-width prompt top-1 agreement is not exact")
    if any(not generation.get("exact_token_match") for generation in parity["generation"]):
        raise ValueError("final-width cached generation is not exact")
    thresholds = parity.get("thresholds", {})
    strict_limits = {
        "cached_logprob_max_absolute_error": ("maximum", 0.02),
        "logits_max_absolute_error": ("maximum", 0.02),
        "logits_max_mean_absolute_error": ("maximum", 0.003),
        "logits_max_normalized_root_mean_square_error": ("maximum", 0.002),
        "logits_min_cosine_similarity": ("minimum", 0.999998),
    }
    for name, (direction, limit) in strict_limits.items():
        value = thresholds.get(name)
        if not isinstance(value, (int, float)):
            raise ValueError(f"missing numeric final-width parity threshold: {name}")
        if direction == "maximum" and value > limit:
            raise ValueError(f"final-width parity threshold is too loose: {name}")
        if direction == "minimum" and value < limit:
            raise ValueError(f"final-width parity threshold is too loose: {name}")
    return report


def run_final_mlx_topology_gate(
    *,
    plan_path: str | Path,
    tokenizer_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Create, convert, compare, and seal one final-width Lumi fixture."""

    started = time.perf_counter()
    plan_file = Path(plan_path).resolve()
    tokenizer = Path(tokenizer_path).resolve()
    output = Path(output_path).resolve()
    if not tokenizer.is_dir():
        raise NotADirectoryError(tokenizer)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite MLX topology gate: {output}")

    plan, source_path, source_config = load_gate_inputs(plan_file)
    tokenizer_spec = plan.get("tokenizer")
    if not isinstance(tokenizer_spec, dict):
        raise ValueError("tokenizer must be an object")
    tokenizer_json = tokenizer / "tokenizer.json"
    expected_tokenizer_sha256 = tokenizer_spec.get("tokenizer_json_sha256")
    actual_tokenizer_sha256 = _sha256_file(tokenizer_json)
    if actual_tokenizer_sha256 != expected_tokenizer_sha256:
        raise ValueError(
            "frozen tokenizer hash mismatch: "
            f"expected {expected_tokenizer_sha256}, got {actual_tokenizer_sha256}"
        )

    fixture_spec = plan.get("fixture")
    if not isinstance(fixture_spec, dict):
        raise ValueError("fixture must be an object")
    fixture_config = build_final_width_fixture(
        source_config,
        num_hidden_layers=_positive_int(
            fixture_spec.get("num_hidden_layers"), name="fixture.num_hidden_layers"
        ),
    )
    seed = _positive_int(fixture_spec.get("seed"), name="fixture.seed")
    parity_spec = plan.get("parity")
    if not isinstance(parity_spec, dict):
        raise ValueError("parity must be an object")
    prompts = parity_spec.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("parity.prompts must be a non-empty list")
    thresholds = parity_spec.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("parity.thresholds must be an object")

    output.mkdir(parents=True)
    provenance = _copy_provenance(plan_file, source_path, output)
    checkpoint = output / "pytorch-final-width-fixture"
    mlx_model = output / "mlx-final-width-fixture"
    parity_report_path = output / "parity-report.json"
    checkpoint.mkdir()

    import gc

    import torch

    from .configuration_lumi import LumiConfig
    from .modeling_lumi import LumiForCausalLM

    torch.manual_seed(seed)
    model = LumiForCausalLM(LumiConfig(**fixture_config)).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model.save_pretrained(checkpoint, safe_serialization=True)
    del model
    gc.collect()
    _write_directory_checksums(checkpoint)

    conversion = convert_checkpoint_to_mlx(
        checkpoint_path=checkpoint,
        tokenizer_path=tokenizer,
        output_path=mlx_model,
        dtype="float32",
    )
    parity = check_mlx_parity(
        checkpoint_path=checkpoint,
        mlx_path=mlx_model,
        tokenizer_path=tokenizer,
        output_path=parity_report_path,
        prompts=prompts,
        generation_max_new_tokens=_positive_int(
            parity_spec.get("generation_max_new_tokens"),
            name="parity.generation_max_new_tokens",
        ),
        max_absolute_error=float(thresholds["max_absolute_error"]),
        max_mean_absolute_error=float(thresholds["max_mean_absolute_error"]),
        min_cosine_similarity=float(thresholds["min_cosine_similarity"]),
        max_normalized_root_mean_square_error=float(
            thresholds["max_normalized_root_mean_square_error"]
        ),
        max_cached_logprob_error=float(thresholds["max_cached_logprob_error"]),
    )

    report = {
        "completed_at": _utc_now(),
        "conversion": {
            "conversion_fingerprint": conversion["conversion_fingerprint"],
            "model_parameters": conversion["output"]["model_parameters"],
            "status": conversion["status"],
        },
        "fixture": {
            field: fixture_config[field] for field in (*FINAL_WIDTH_FIELDS, "num_hidden_layers")
        },
        "fixture_parameter_count": parameter_count,
        "locations": {
            "output": str(output),
            "plan": str(plan_file),
            "source_config": str(source_path),
            "tokenizer": str(tokenizer),
        },
        "parity": {
            "mlx_peak_memory_bytes": parity["runtime"]["mlx_peak_memory_bytes"],
            "status": parity["status"],
        },
        "projection_shapes": _projection_shapes(fixture_config),
        "provenance": provenance,
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "schema_version": GATE_SCHEMA_VERSION,
        "source": {
            field: source_config[field] for field in (*FINAL_WIDTH_FIELDS, "num_hidden_layers")
        },
        "source_config_sha256": _sha256_file(source_path),
        "status": "pass",
        "tokenizer_json_sha256": actual_tokenizer_sha256,
    }
    _write_json(output / GATE_REPORT_FILENAME, report)
    _write_manifest(output)
    return verify_final_mlx_topology_gate(output)
