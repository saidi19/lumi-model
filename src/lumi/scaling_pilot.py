"""Fail-closed architecture, data, stage, and cost planning for Lumi-1B."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .configuration_lumi import LumiConfig
from .parameter_count import parameter_breakdown
from .pilot_training import scheduled_learning_rate

SCALING_PILOT_SCHEMA_VERSION = 1
SCALING_SELECTION_MODE = "within_source_hash_ranked_excluding_heldout"
EXPECTED_STAGE_NAMES = (
    "smoke_resume",
    "calibration",
    "learning_gate",
    "unique_corpus",
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
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a non-negative number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return converted


def _sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _relative_file(plan_path: Path, value: Any, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{name} must stay inside the plan directory")
    path = plan_path.parent / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _validated_config(
    plan_path: Path,
    specification: dict[str, Any],
    *,
    path_field: str,
    hash_field: str,
    parameters_field: str,
) -> LumiConfig:
    path = _relative_file(plan_path, specification.get(path_field), name=f"model.{path_field}")
    expected_hash = _sha256(
        specification.get(hash_field),
        name=f"model.{hash_field}",
    )
    if _sha256_file(path) != expected_hash:
        raise ValueError(f"model.{path_field} SHA-256 changed")
    config = LumiConfig.from_json_file(path)
    expected_parameters = _positive_int(
        specification.get(parameters_field),
        name=f"model.{parameters_field}",
    )
    if parameter_breakdown(config).total != expected_parameters:
        raise ValueError(f"model.{parameters_field} does not match the config")
    return config


def _validate_scaling_relationship(pilot: LumiConfig, reference: LumiConfig) -> None:
    same_fields = (
        "attention_bias",
        "bos_token_id",
        "doc_token_id",
        "eos_token_id",
        "head_dim",
        "max_position_embeddings",
        "num_hidden_layers",
        "pad_token_id",
        "rms_norm_eps",
        "rope_theta",
        "tie_word_embeddings",
        "vocab_size",
    )
    for field in same_fields:
        if getattr(pilot, field) != getattr(reference, field):
            raise ValueError(f"1B and 4B configs must share {field}")
    half_width_fields = (
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
    )
    for field in half_width_fields:
        if getattr(pilot, field) * 2 != getattr(reference, field):
            raise ValueError(f"1B config must use exactly half the 4B {field}")


def load_scaling_pilot_plan(plan_path: str | Path) -> dict[str, Any]:
    """Load and fully validate the pinned Lumi-1B planning contract."""

    plan_path = Path(plan_path)
    plan = _read_json(plan_path)
    if plan.get("schema_version") != SCALING_PILOT_SCHEMA_VERSION:
        raise ValueError("unsupported scaling-pilot schema version")

    model = plan.get("model")
    if not isinstance(model, dict):
        raise ValueError("model must be an object")
    pilot_config = _validated_config(
        plan_path,
        model,
        path_field="config",
        hash_field="config_sha256",
        parameters_field="parameters",
    )
    reference_config = _validated_config(
        plan_path,
        model,
        path_field="reference_config",
        hash_field="reference_config_sha256",
        parameters_field="reference_parameters",
    )
    _validate_scaling_relationship(pilot_config, reference_config)

    data = plan.get("data")
    if not isinstance(data, dict):
        raise ValueError("data must be an object")
    sequence_length = _positive_int(data.get("sequence_length"), name="data.sequence_length")
    if sequence_length != pilot_config.max_position_embeddings:
        raise ValueError("data sequence length must exercise the full 1B context")

    source = data.get("source_sample")
    heldout = data.get("heldout")
    selection = data.get("selection")
    if not all(isinstance(value, dict) for value in (source, heldout, selection)):
        raise ValueError("source_sample, heldout, and selection must be objects")
    assert isinstance(source, dict) and isinstance(heldout, dict) and isinstance(selection, dict)
    for field in ("manifest_sha256", "sample_sha256", "token_count_report_sha256"):
        _sha256(source.get(field), name=f"data.source_sample.{field}")
    observed_tokens = _positive_int(
        source.get("observed_tokens"), name="data.source_sample.observed_tokens"
    )
    _positive_int(source.get("accepted_text_bytes"), name="data.source_sample.accepted_text_bytes")
    _positive_int(source.get("documents"), name="data.source_sample.documents")

    for field in ("manifest_sha256", "selection_ledger_sha256", "build_fingerprint"):
        _sha256(heldout.get(field), name=f"data.heldout.{field}")
    heldout_tokens = _positive_int(heldout.get("tokens"), name="data.heldout.tokens")
    _positive_int(heldout.get("documents"), name="data.heldout.documents")

    if selection.get("mode") != SCALING_SELECTION_MODE:
        raise ValueError("1B data must be hash-ranked with permanent held-out exclusion")
    _positive_int(selection.get("seed"), name="data.selection.seed")
    _positive_int(selection.get("source_strata"), name="data.selection.source_strata")
    requested_tokens = _positive_int(
        selection.get("target_tokens_requested"),
        name="data.selection.target_tokens_requested",
    )
    effective_tokens = requested_tokens // sequence_length * sequence_length
    if selection.get("target_tokens_effective") != effective_tokens:
        raise ValueError("effective 1B target is not sequence aligned")
    if observed_tokens - heldout_tokens < effective_tokens:
        raise ValueError("observed unique-token capacity cannot cover the 1B target")

    training = plan.get("training")
    if not isinstance(training, dict):
        raise ValueError("training must be an object")
    batch_size = _positive_int(training.get("batch_size"), name="training.batch_size")
    accumulation = _positive_int(
        training.get("gradient_accumulation_steps"),
        name="training.gradient_accumulation_steps",
    )
    tokens_per_step = sequence_length * batch_size * accumulation
    if training.get("tokens_per_step") != tokens_per_step:
        raise ValueError("training.tokens_per_step does not match batch geometry")
    if training.get("gradient_checkpointing") is not True:
        raise ValueError("the 1B pilot requires gradient checkpointing")
    if training.get("precision") != "bf16_autocast_fp32_parameters":
        raise ValueError("the 1B pilot precision contract changed")
    _positive_int(training.get("checkpoint_steps"), name="training.checkpoint_steps")
    _positive_int(training.get("keep_checkpoints"), name="training.keep_checkpoints")

    stages = plan.get("stages")
    if not isinstance(stages, list) or len(stages) != len(EXPECTED_STAGE_NAMES):
        raise ValueError("scaling stages are incomplete")
    stage_names = tuple(stage.get("name") for stage in stages if isinstance(stage, dict))
    if stage_names != EXPECTED_STAGE_NAMES:
        raise ValueError("scaling stages must keep their pinned order and names")
    previous_target = 0
    for index, stage in enumerate(stages):
        assert isinstance(stage, dict)
        prefix = f"stages[{index}]"
        target_tokens = _positive_int(stage.get("target_tokens"), name=f"{prefix}.target_tokens")
        if target_tokens % sequence_length:
            raise ValueError(f"{prefix}.target_tokens must be sequence aligned")
        if target_tokens <= previous_target:
            raise ValueError("scaling-stage token targets must increase")
        previous_target = target_tokens
        steps = math.ceil(target_tokens / tokens_per_step)
        processed_tokens = steps * tokens_per_step
        repeated_tokens = processed_tokens - target_tokens
        expected = {
            "processed_tokens": processed_tokens,
            "repeated_tokens": repeated_tokens,
            "steps": steps,
        }
        for field, value in expected.items():
            if stage.get(field) != value:
                raise ValueError(f"{prefix}.{field} does not match derived value {value}")
        if repeated_tokens >= tokens_per_step:
            raise ValueError(f"{prefix} repeats at least one optimizer step")
    if stages[-1]["target_tokens"] != effective_tokens:
        raise ValueError("the final stage must cover the pinned unique-corpus target")

    optimizer = plan.get("optimizer")
    if not isinstance(optimizer, dict):
        raise ValueError("optimizer must be an object")
    learning_rate = _nonnegative_number(
        optimizer.get("learning_rate"), name="optimizer.learning_rate"
    )
    if learning_rate == 0:
        raise ValueError("optimizer.learning_rate must be positive")
    warmup_steps = _positive_int(optimizer.get("warmup_steps"), name="optimizer.warmup_steps")
    schedule_steps = _positive_int(
        optimizer.get("schedule_steps"), name="optimizer.schedule_steps"
    )
    if schedule_steps != stages[-1]["steps"]:
        raise ValueError("optimizer schedule must end at the unique-corpus stage")
    min_ratio = _nonnegative_number(
        optimizer.get("min_learning_rate_ratio"),
        name="optimizer.min_learning_rate_ratio",
    )
    if min_ratio > 1:
        raise ValueError("optimizer.min_learning_rate_ratio cannot exceed 1")
    if optimizer.get("lr_schedule") != "cosine":
        raise ValueError("the 1B pilot requires the pinned cosine schedule")
    scheduled_learning_rate(
        base_learning_rate=learning_rate,
        step=schedule_steps,
        schedule="cosine",
        warmup_steps=warmup_steps,
        schedule_steps=schedule_steps,
        min_learning_rate_ratio=min_ratio,
    )
    max_grad_norm = _nonnegative_number(
        optimizer.get("max_grad_norm"), name="optimizer.max_grad_norm"
    )
    if max_grad_norm == 0:
        raise ValueError("optimizer.max_grad_norm must be positive")
    _nonnegative_number(optimizer.get("weight_decay"), name="optimizer.weight_decay")
    _positive_int(optimizer.get("seed"), name="optimizer.seed")

    evaluation = plan.get("evaluation")
    if not isinstance(evaluation, dict) or not isinstance(evaluation.get("baseline_150m"), dict):
        raise ValueError("evaluation baseline is missing")
    baseline = evaluation["baseline_150m"]
    _sha256(baseline.get("report_sha256"), name="evaluation.baseline_150m.report_sha256")
    _positive_int(baseline.get("parameters"), name="evaluation.baseline_150m.parameters")
    _positive_int(
        baseline.get("training_tokens"),
        name="evaluation.baseline_150m.training_tokens",
    )
    if _nonnegative_number(
        baseline.get("heldout_loss"), name="evaluation.baseline_150m.heldout_loss"
    ) == 0:
        raise ValueError("baseline held-out loss must be positive")
    if evaluation.get("require_loss_below_baseline") is not True:
        raise ValueError("the 1B learning gate must beat the 150M held-out loss")
    if evaluation.get("require_zero_overlap") is not True:
        raise ValueError("the 1B pilot requires zero held-out overlap")

    approval = plan.get("approval")
    if approval != {"full_4b_training": False, "paid_gpu_launch": False}:
        raise ValueError("the design plan cannot pre-approve paid GPU or 4B training")
    return plan


def validate_scaling_pilot_inputs(
    plan: dict[str, Any],
    *,
    source_manifest_path: str | Path,
    heldout_manifest_path: str | Path,
    heldout_selection_ledger_path: str | Path,
    token_count_report_path: str | Path,
    baseline_evaluation_report_path: str | Path,
) -> dict[str, Any]:
    """Verify the current local source/evaluation artifacts against the plan."""

    source_path = Path(source_manifest_path)
    heldout_path = Path(heldout_manifest_path)
    ledger_path = Path(heldout_selection_ledger_path)
    token_count_path = Path(token_count_report_path)
    baseline_path = Path(baseline_evaluation_report_path)
    data = plan["data"]
    source_spec = data["source_sample"]
    heldout_spec = data["heldout"]

    for path, expected, label in (
        (source_path, source_spec["manifest_sha256"], "source manifest"),
        (heldout_path, heldout_spec["manifest_sha256"], "held-out manifest"),
        (ledger_path, heldout_spec["selection_ledger_sha256"], "held-out selection ledger"),
        (token_count_path, source_spec["token_count_report_sha256"], "token-count report"),
        (
            baseline_path,
            plan["evaluation"]["baseline_150m"]["report_sha256"],
            "baseline evaluation report",
        ),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        if _sha256_file(path) != expected:
            raise ValueError(f"{label} SHA-256 changed")

    source = _read_json(source_path)
    if source.get("status") != "complete":
        raise ValueError("source manifest is not complete")
    if source.get("output", {}).get("sha256") != source_spec["sample_sha256"]:
        raise ValueError("source sample SHA-256 changed")
    if source.get("output", {}).get("text_utf8_bytes") != source_spec["accepted_text_bytes"]:
        raise ValueError("source accepted-text byte count changed")
    if source.get("statistics", {}).get("documents") != source_spec["documents"]:
        raise ValueError("source document count changed")

    heldout = _read_json(heldout_path)
    if heldout.get("build_fingerprint") != heldout_spec["build_fingerprint"]:
        raise ValueError("held-out build fingerprint changed")
    if heldout.get("statistics", {}).get("tokens") != heldout_spec["tokens"]:
        raise ValueError("held-out token count changed")
    if heldout.get("leakage", {}).get("validation_documents") != heldout_spec["documents"]:
        raise ValueError("held-out document count changed")
    if len(heldout.get("selection", {}).get("strata", [])) != data["selection"]["source_strata"]:
        raise ValueError("held-out source-stratum count changed")

    with ledger_path.open(encoding="utf-8") as handle:
        ledger_documents = sum(1 for line in handle if line.strip())
    if ledger_documents != heldout_spec["documents"]:
        raise ValueError("held-out selection ledger row count changed")

    token_count = _read_json(token_count_path)
    selected = token_count.get("tokenizers", {}).get("65k", {})
    if selected.get("overall", {}).get("tokens") != source_spec["observed_tokens"]:
        raise ValueError("65K observed token count changed")
    if selected.get("artifact", {}).get("vocab_size") != 65_536:
        raise ValueError("token-count report no longer describes the 65K tokenizer")

    baseline_spec = plan["evaluation"]["baseline_150m"]
    baseline = _read_json(baseline_path)
    if baseline.get("artifacts", {}).get("heldout_build_fingerprint") != heldout_spec[
        "build_fingerprint"
    ]:
        raise ValueError("baseline evaluation used a different held-out build")
    if baseline.get("loss", {}).get("mean") != baseline_spec["heldout_loss"]:
        raise ValueError("baseline held-out loss changed")
    if baseline.get("loss", {}).get("perplexity") != baseline_spec["heldout_perplexity"]:
        raise ValueError("baseline held-out perplexity changed")
    if baseline.get("runtime", {}).get("model_parameters") != baseline_spec["parameters"]:
        raise ValueError("baseline model parameter count changed")

    return {
        "heldout_build_fingerprint": heldout_spec["build_fingerprint"],
        "heldout_documents": ledger_documents,
        "heldout_tokens": heldout_spec["tokens"],
        "baseline_heldout_loss": baseline_spec["heldout_loss"],
        "observed_source_tokens": source_spec["observed_tokens"],
        "source_documents": source_spec["documents"],
        "source_sample_sha256": source_spec["sample_sha256"],
        "status": "pass",
    }


def plan_scaling_pilot(
    *,
    plan_path: str | Path,
    tokens_per_second: float | None = None,
    hourly_rate: float | None = None,
    fixed_overhead_hours: float = 0.0,
    credit_balance: float | None = None,
    input_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return cumulative stage compute and optional measured-price projections."""

    plan_path = Path(plan_path)
    plan = load_scaling_pilot_plan(plan_path)
    if (tokens_per_second is None) != (hourly_rate is None):
        raise ValueError("tokens-per-second and hourly-rate must be supplied together")
    overhead = _nonnegative_number(fixed_overhead_hours, name="fixed_overhead_hours")
    if credit_balance is not None:
        credit = _nonnegative_number(credit_balance, name="credit_balance")
    else:
        credit = None
    priced = tokens_per_second is not None
    if priced:
        throughput = _nonnegative_number(tokens_per_second, name="tokens_per_second")
        rate = _nonnegative_number(hourly_rate, name="hourly_rate")
        if throughput == 0 or rate == 0:
            raise ValueError("tokens-per-second and hourly-rate must be positive")
    else:
        throughput = rate = None

    parameters = plan["model"]["parameters"]
    stages = []
    for stage in plan["stages"]:
        processed_tokens = stage["processed_tokens"]
        item = {
            **stage,
            "approximate_training_flops": 6 * parameters * processed_tokens,
            "approximate_training_petaflop_days": (
                6 * parameters * processed_tokens / (1e15 * 86_400)
            ),
            "processed_tokens_per_parameter": processed_tokens / parameters,
        }
        if priced:
            assert throughput is not None and rate is not None
            compute_hours = processed_tokens / throughput / 3_600
            billable_hours = compute_hours + overhead
            estimated_cost = billable_hours * rate
            item["projection"] = {
                "billable_hours": billable_hours,
                "compute_hours": compute_hours,
                "credit_remaining": None if credit is None else credit - estimated_cost,
                "estimated_cost": estimated_cost,
            }
        stages.append(item)

    pricing = None
    if priced:
        assert throughput is not None and rate is not None
        pricing = {
            "credit_balance": credit,
            "fixed_overhead_hours": overhead,
            "hourly_rate": rate,
            "projected_cost_per_billion_tokens": 1e9 / throughput / 3_600 * rate,
            "tokens_per_second": throughput,
        }
    return {
        "created_at": _utc_now(),
        "input_validation": input_validation,
        "model": {
            "parameters": parameters,
            "reference_parameters": plan["model"]["reference_parameters"],
        },
        "plan_sha256": _sha256_file(plan_path),
        "pricing": pricing,
        "schema_version": SCALING_PILOT_SCHEMA_VERSION,
        "stages": stages,
        "status": "costed" if priced else "validated_unpriced",
    }


def write_scaling_report(path: str | Path, report: dict[str, Any]) -> Path:
    """Atomically write one immutable planning report."""

    output = Path(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite scaling report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output
