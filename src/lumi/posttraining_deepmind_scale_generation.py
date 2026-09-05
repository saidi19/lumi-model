"""Bounded CPU-only DeepMind Mathematics scale-generation support."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .posttraining_deepmind_canary_run import (
    DETERMINISM_ADAPTER,
    EXPECTED_FLAT_MODULE_INVENTORY_SHA256,
    EXPECTED_FLAT_MODULES,
    OVERLAY_ROOT,
    PYTHON_EXECUTABLE,
    SOURCE_BYTES,
    SOURCE_DATASET_ID,
    SOURCE_ID,
    SOURCE_INVENTORY_SHA256,
    SOURCE_LICENSE,
    SOURCE_OBJECTS,
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    SOURCE_ROOT,
    TOP_MODULE_FILES,
    TOP_MODULES,
    apply_prompt_near_dedup,
    audit_source_runtime,
    file_sha256,
    json_file_bytes,
    jsonl_file_bytes,
    load_runtime_components,
    resolve_binding,
    sha256_value,
    verify_bound_evidence,
    write_bytes_atomic,
    write_json_atomic,
    write_jsonl_atomic,
)
from .posttraining_owner_review import build_owner_review_html

PLAN_SCHEMA_VERSION = "lumi.sft-deepmind-scale-generation-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-deepmind-scale-generation-report.v0.1"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-deepmind-scale-generation-verification.v0.1"

PRIVATE_ROOT = (
    "/Users/afnan/Documents/Flowaistudio/lumi-artifacts/"
    "4b-sft-deepmind-scale-generation-v0.1/427f45075f84/attempt-001"
)
SPLIT_NAMESPACES = {
    "train": "lumi-deepmind-math-train-scale-v0.1",
    "validation": "lumi-deepmind-math-validation-scale-v0.1",
}
SPLIT_MODULE_COUNTS = {
    "train": {
        "algebra": 29_688,
        "arithmetic": 29_688,
        "calculus": 29_688,
        "comparison": 29_688,
        "measurement": 29_687,
        "numbers": 29_687,
        "polynomials": 29_687,
        "probability": 29_687,
    },
    "validation": {
        "algebra": 1_563,
        "arithmetic": 1_563,
        "calculus": 1_563,
        "comparison": 1_563,
        "measurement": 1_562,
        "numbers": 1_562,
        "polynomials": 1_562,
        "probability": 1_562,
    },
}
EXPECTED_SPLIT_COUNTS = {"train": 237_500, "validation": 12_500}
EXPECTED_GENERATED_RECORDS = 250_000
MINIMUM_TRAIN_SUPERVISED_TOKENS = 2_150_000
MAXIMUM_TRAIN_SUPERVISED_TOKENS = 2_300_000
OWNER_REVIEW_RECORDS = 32
MAXIMUM_RUNTIME_SECONDS = 240 * 60
MAXIMUM_PRIVATE_BYTES = 1_610_612_736
MAX_QUESTION_CHARACTERS = 160
MAX_ANSWER_CHARACTERS = 30
GENERATED_RECORD_FIELDS = frozenset(
    {"answer", "module", "question", "rng_seed", "seed", "split_namespace"}
)

# This affine map is a permutation over uint32 because the multiplier is odd.
# Global ordinals are unique across train and validation, so RNG states cannot collide.
RNG_SEED_MULTIPLIER = 0x9E3779B1
RNG_SEED_OFFSET = 0x4C554D49

EXPECTED_AUTHORIZATION = {
    "candidate_registry_promotion_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_spend_authorized": False,
    "external_distribution_authorized": False,
    "gpu_authorized": False,
    "network_access_authorized": False,
    "official_test_or_test_extra_generation_authorized": False,
    "production_sft_authorized": False,
    "purpose": "bounded_deepmind_250000_record_cpu_scale_generation",
    "source_acquisition_authorized": False,
    "unrelated_data_access_authorized": False,
}
EXPECTED_LIMITS = {
    "maximum_generated_records": EXPECTED_GENERATED_RECORDS,
    "maximum_private_bytes": MAXIMUM_PRIVATE_BYTES,
    "maximum_retained_train_supervised_tokens": MAXIMUM_TRAIN_SUPERVISED_TOKENS,
    "maximum_runtime_minutes": 240,
    "minimum_target_retained_train_supervised_tokens": (
        MINIMUM_TRAIN_SUPERVISED_TOKENS
    ),
}
EXPECTED_GENERATION = {
    "assistant_response_template": "The answer is {generated_answer}.",
    "determinism_adapter": DETERMINISM_ADAPTER,
    "fresh_process_per_shard": True,
    "generated_record_fields": sorted(GENERATED_RECORD_FIELDS),
    "identity_seed": "sha256_first_u64_big_endian",
    "maximum_answer_characters": MAX_ANSWER_CHARACTERS,
    "maximum_question_characters": MAX_QUESTION_CHARACTERS,
    "network_disabled_during_generation": True,
    "official_test_or_test_extra_generation": False,
    "replay": "same_seed_cross_process_byte_exact_all_250000_records",
    "rng_seed": {
        "algorithm": "injective_uint32_affine_permutation_of_global_ordinal",
        "multiplier": RNG_SEED_MULTIPLIER,
        "offset": RNG_SEED_OFFSET,
    },
    "split_module_counts": SPLIT_MODULE_COUNTS,
    "split_namespaces": SPLIT_NAMESPACES,
    "unique_record_identities": EXPECTED_GENERATED_RECORDS,
}
EXPECTED_QUALITY = {
    "benchmark_decontamination": "frozen_hash_registry_v0.1",
    "exact_duplicate_policy": "reject_global_before_near_dedup",
    "near_duplicate": {
        "algorithm": "exact_token_shingle_jaccard_inverted_index_v0.1",
        "shingle_tokens": 5,
        "threshold": 0.9,
        "validation_precedence": True,
    },
    "owner_review": {
        "deterministic_sample_records": OWNER_REVIEW_RECORDS,
        "status_after_generation": "pending_owner_review",
    },
    "pii_and_secret_filter": "lumi_sft_basic_pattern_scan_v0.1",
    "source_registry_check": "deferred_until_separate_promotion",
    "tokenizer_4k_fit": True,
}
EXPECTED_OUTPUT = {
    "private_root": PRIVATE_ROOT,
    "raw_text_free_report_path": (
        "evidence/lumi-4b-sft-deepmind-scale-generation-v0.1.json"
    ),
    "raw_text_free_verification_path": (
        "evidence/lumi-4b-sft-deepmind-scale-generation-verification-v0.1.json"
    ),
    "raw_training_text_in_git": False,
}
EXPECTED_BINDING_PATHS = {
    "benchmark_registry": "data/benchmark-signatures-v0.1.jsonl.gz",
    "canary_implementation": "src/lumi/posttraining_deepmind_canary_run.py",
    "canary_report": "evidence/lumi-4b-sft-deepmind-canary-run-v0.1.json",
    "canary_verification": (
        "evidence/lumi-4b-sft-deepmind-canary-run-verification-v0.1.json"
    ),
    "canary_worker": "scripts/deepmind_canary_worker.py",
    "chat_template": "configs/lumi-4b-chat-template-v0.1.json",
    "data_contract": "configs/lumi-4b-sft-data-contract-v0.1.json",
    "dependency_overlay_receipt": (
        "evidence/lumi-4b-sft-deepmind-dependency-overlay-v0.1.json"
    ),
    "dependency_overlay_verification": (
        "evidence/lumi-4b-sft-deepmind-dependency-overlay-verification-v0.1.json"
    ),
    "filtering_implementation": "src/lumi/posttraining_data.py",
    "format_implementation": "src/lumi/posttraining_format.py",
    "owner_review_renderer": "src/lumi/posttraining_owner_review.py",
    "quality_config": "configs/data-quality-v0.1.json",
    "scale_implementation": "src/lumi/posttraining_deepmind_scale_generation.py",
    "scale_runner": "scripts/run_deepmind_scale_generation.py",
    "scale_verifier": "scripts/verify_deepmind_scale_generation.py",
    "scale_worker": "scripts/deepmind_scale_worker.py",
    "source_acquisition_receipt": (
        "evidence/lumi-4b-sft-deepmind-source-acquisition-v0.1.json"
    ),
    "source_acquisition_verification": (
        "evidence/lumi-4b-sft-deepmind-source-acquisition-verification-v0.1.json"
    ),
    "token_measurement_implementation": "src/lumi/posttraining_oasst1_full_run.py",
    "tokenizer_config": (
        "../lumi-artifacts/4b-segment2-throughput-gate-v0.1/8bedfbc868e8/"
        "staging/input/tokenizer/tokenizer_config.json"
    ),
    "tokenizer_json": (
        "../lumi-artifacts/4b-segment2-throughput-gate-v0.1/8bedfbc868e8/"
        "staging/input/tokenizer/tokenizer.json"
    ),
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _verify_internal_seal(value: Mapping[str, Any], field: str) -> str:
    seal = value.get(field)
    if not isinstance(seal, str) or _SHA256.fullmatch(seal) is None:
        raise ValueError(f"invalid internal SHA-256: {field}")
    unsigned = {key: item for key, item in value.items() if key != field}
    if sha256_value(unsigned) != seal:
        raise ValueError(f"internal evidence seal changed: {field}")
    return seal


def load_scale_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Load the exact scale plan and revalidate every local binding."""

    plan = _load_json(path)
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "generation": EXPECTED_GENERATION,
        "limits": EXPECTED_LIMITS,
        "observed_at": "2026-08-07",
        "output": EXPECTED_OUTPUT,
        "quality": EXPECTED_QUALITY,
        "runtime": {
            "dependency_overlay": OVERLAY_ROOT,
            "python_executable": PYTHON_EXECUTABLE,
        },
        "schema_version": PLAN_SCHEMA_VERSION,
        "source": {
            "declared_license_id": SOURCE_LICENSE,
            "exact_object_bytes": SOURCE_BYTES,
            "exact_object_count": SOURCE_OBJECTS,
            "exact_object_inventory_sha256": SOURCE_INVENTORY_SHA256,
            "local_source_root": SOURCE_ROOT,
            "repository_url": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "source_id": SOURCE_ID,
        },
        "status": "authorized_verified_source_cpu_only_scale_generation",
    }
    for field, expected_value in expected.items():
        if plan.get(field) != expected_value:
            raise ValueError(f"DeepMind scale plan changed: {field}")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(
        EXPECTED_BINDING_PATHS
    ):
        raise ValueError("DeepMind scale binding set changed")
    for name, expected_path in EXPECTED_BINDING_PATHS.items():
        binding = bindings[name]
        if not isinstance(binding, Mapping) or binding.get("path") != expected_path:
            raise ValueError(f"DeepMind scale binding path changed: {name}")
        resolve_binding(repository_root, binding, name=name)
    verify_bound_evidence(plan, repository_root=repository_root)
    canary_report = _load_json(
        resolve_binding(
            repository_root, bindings["canary_report"], name="canary_report"
        )
    )
    canary_verification = _load_json(
        resolve_binding(
            repository_root,
            bindings["canary_verification"],
            name="canary_verification",
        )
    )
    if (
        _verify_internal_seal(canary_report, "report_sha256")
        != "7d10bf58686f0ededde3e44108abec7c962be8880f8550c80498511385352a93"
        or canary_report.get("status")
        != "verified_cpu_only_nonproduction_adapter_canary"
    ):
        raise ValueError("DeepMind verified canary report changed")
    if (
        _verify_internal_seal(canary_verification, "verification_sha256")
        != "aafdecb358b41c7385cb884ce0380281ec638399519f22d3ff04cc396c32774a"
        or canary_verification.get("status")
        != "verified_cpu_only_nonproduction_deepmind_adapter_canary"
    ):
        raise ValueError("DeepMind verified canary verification changed")
    return plan


def derive_identity_seed(namespace: str, module_file: str, index: int) -> int:
    if namespace not in SPLIT_NAMESPACES.values():
        raise ValueError("DeepMind scale identity namespace is not allowed")
    if module_file not in TOP_MODULE_FILES.values():
        raise ValueError("DeepMind scale identity module is not allowed")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("DeepMind scale identity index must be non-negative")
    payload = f"{namespace}\0{module_file}\0{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def derive_rng_seed(global_ordinal: int) -> int:
    if (
        isinstance(global_ordinal, bool)
        or not isinstance(global_ordinal, int)
        or not 0 <= global_ordinal < EXPECTED_GENERATED_RECORDS
    ):
        raise ValueError("DeepMind scale global ordinal is outside the run")
    return (
        RNG_SEED_MULTIPLIER * global_ordinal + RNG_SEED_OFFSET
    ) & 0xFFFFFFFF


def generation_schedule() -> list[dict[str, Any]]:
    schedule: list[dict[str, Any]] = []
    global_offset = 0
    for split in ("train", "validation"):
        for top_module in TOP_MODULES:
            count = SPLIT_MODULE_COUNTS[split][top_module]
            schedule.append(
                {
                    "count": count,
                    "global_offset": global_offset,
                    "module_file": TOP_MODULE_FILES[top_module],
                    "namespace": SPLIT_NAMESPACES[split],
                    "split": split,
                    "top_module": top_module,
                }
            )
            global_offset += count
    if global_offset != EXPECTED_GENERATED_RECORDS:
        raise AssertionError("DeepMind scale generation schedule count changed")
    return schedule


def validate_generated_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(records) != EXPECTED_GENERATED_RECORDS:
        raise ValueError("DeepMind scale generated record count changed")
    cursor = 0
    identity_seeds: set[int] = set()
    rng_seeds: set[int] = set()
    prompt_observations: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    module_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    for shard in generation_schedule():
        for index in range(shard["count"]):
            record = records[cursor]
            global_ordinal = shard["global_offset"] + index
            cursor += 1
            if frozenset(record) != GENERATED_RECORD_FIELDS:
                raise ValueError("DeepMind scale generated record schema changed")
            if record.get("split_namespace") != shard["namespace"]:
                raise ValueError("DeepMind scale split namespace changed")
            module = record.get("module")
            if not isinstance(module, str) or not module.startswith(
                f"{shard['top_module']}__"
            ):
                raise ValueError("DeepMind scale module assignment changed")
            identity_seed = record.get("seed")
            expected_identity = derive_identity_seed(
                shard["namespace"], shard["module_file"], index
            )
            if identity_seed != expected_identity:
                raise ValueError("DeepMind scale identity seed changed")
            rng_seed = record.get("rng_seed")
            if rng_seed != derive_rng_seed(global_ordinal):
                raise ValueError("DeepMind scale RNG seed changed")
            if identity_seed in identity_seeds or rng_seed in rng_seeds:
                raise ValueError("DeepMind scale seed collision detected")
            identity_seeds.add(int(identity_seed))
            rng_seeds.add(int(rng_seed))
            question = record.get("question")
            answer = record.get("answer")
            if not isinstance(question, str) or not question.strip():
                raise ValueError("DeepMind scale question is empty")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("DeepMind scale answer is empty")
            if len(question) > MAX_QUESTION_CHARACTERS:
                raise ValueError("DeepMind scale question exceeded source limit")
            if len(answer) > MAX_ANSWER_CHARACTERS:
                raise ValueError("DeepMind scale answer exceeded source limit")
            prompt_sha256 = hashlib.sha256(question.strip().encode()).hexdigest()
            answer_sha256 = hashlib.sha256(answer.strip().encode()).hexdigest()
            prompt_observations[prompt_sha256].append(
                (str(shard["split"]), answer_sha256)
            )
            module_counts[module] += 1
            split_counts[str(shard["split"])] += 1
    if cursor != len(records) or dict(split_counts) != EXPECTED_SPLIT_COUNTS:
        raise ValueError("DeepMind scale split accounting changed")
    duplicate_groups = {
        key: values for key, values in prompt_observations.items() if len(values) > 1
    }
    conflicting_answer_groups = sum(
        len({answer for _split, answer in values}) > 1
        for values in duplicate_groups.values()
    )
    if conflicting_answer_groups:
        raise ValueError("DeepMind scale duplicate prompt gained conflicting answers")
    return {
        "conflicting_answer_prompt_groups": conflicting_answer_groups,
        "different_seed_prompt_groups": len(duplicate_groups),
        "different_seed_prompt_overlap": sum(
            len(values) - 1 for values in duplicate_groups.values()
        ),
        "flat_modules_observed": len(module_counts),
        "generated_records": len(records),
        "identity_seed_bits": 64,
        "identity_seed_inventory_sha256": sha256_value(sorted(identity_seeds)),
        "module_counts_sha256": sha256_value(dict(sorted(module_counts.items()))),
        "prompt_inventory_sha256": sha256_value(sorted(prompt_observations)),
        "rng_seed_inventory_sha256": sha256_value(sorted(rng_seeds)),
        "split_counts": dict(sorted(split_counts.items())),
        "train_validation_prompt_overlap": sum(
            len({split for split, _answer in values}) > 1
            for values in duplicate_groups.values()
        ),
        "train_validation_seed_overlap": 0,
    }


def build_candidate_records(
    generated_records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    attributions: list[dict[str, Any]] = []
    namespace_to_split = {value: key for key, value in SPLIT_NAMESPACES.items()}
    for generated in generated_records:
        namespace = str(generated["split_namespace"])
        split = namespace_to_split[namespace]
        seed = int(generated["seed"])
        rng_seed = int(generated["rng_seed"])
        module = str(generated["module"])
        top_module = module.split("__", 1)[0]
        question = str(generated["question"]).strip()
        generated_answer = str(generated["answer"]).strip()
        response = f"The answer is {generated_answer}."
        record_id = f"deepmind-scale-{split}-{seed:016x}"
        source_content = {
            "answer": generated_answer,
            "module": module,
            "question": question,
            "rng_seed": rng_seed,
            "seed": seed,
            "split_namespace": namespace,
        }
        messages = [
            {"content": question, "role": "user"},
            {"content": response, "role": "assistant"},
        ]
        content_sha256 = sha256_value(source_content)
        transform_sha256 = sha256_value(
            {"messages": messages, "transform": "deepmind_final_answer_adapter_v0.2"}
        )
        records.append(
            {
                "language": "en",
                "messages": messages,
                "provenance": {
                    "consent_basis": "source_level_open_license",
                    "content_sha256": content_sha256,
                    "transform_sha256": transform_sha256,
                },
                "quality": {
                    "pii_scan": "passed",
                    "review_method": (
                        "deepmind_fixed_seed_generator_plus_lumi_scale_filters_v0.1"
                    ),
                    "review_status": "accepted",
                    "score": 0.95,
                    "secret_scan": "passed",
                },
                "record_id": record_id,
                "safety": {
                    "policy_version": "lumi-sft-basic-pattern-scan-v0.1",
                    "risk_labels": [],
                },
                "schema_version": "lumi.sft.v0.1",
                "source": {
                    "dataset_id": SOURCE_DATASET_ID,
                    "document_id": f"{namespace}:{seed:016x}",
                    "license_id": SOURCE_LICENSE,
                    "revision": SOURCE_REVISION,
                    "source_id": SOURCE_ID,
                    "uri": (
                        f"{SOURCE_REPOSITORY}/blob/{SOURCE_REVISION}/"
                        f"mathematics_dataset/modules/{top_module}.py#seed-{seed:016x}"
                    ),
                },
                "split": split,
                "task_category": "reasoning_math_science",
            }
        )
        attributions.append(
            {
                "answer_sha256": hashlib.sha256(
                    generated_answer.encode()
                ).hexdigest(),
                "content_sha256": content_sha256,
                "generated_record_fields": sorted(GENERATED_RECORD_FIELDS),
                "module": module,
                "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
                "record_id": record_id,
                "rng_seed": rng_seed,
                "seed": seed,
                "source_module_path": TOP_MODULE_FILES[top_module],
                "split": split,
                "split_namespace": namespace,
                "target_fields": ["question", "answer"],
                "trace_fields_used_as_training_targets": False,
                "transform_sha256": transform_sha256,
            }
        )
    return records, attributions


def count_by_split(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(record["split"]) for record in records).items()))


def deterministic_owner_sample(
    records: Sequence[Mapping[str, Any]],
    attributions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select two deterministic rows from every split/module bucket."""

    attribution_by_id = {str(row["record_id"]): row for row in attributions}
    buckets: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        record_id = str(record["record_id"])
        attribution = attribution_by_id.get(record_id)
        if attribution is None:
            raise ValueError("DeepMind owner sample attribution is missing")
        top_module = str(attribution["module"]).split("__", 1)[0]
        buckets[(str(record["split"]), top_module)].append(record)
    expected_buckets = {
        (split, module) for split in ("train", "validation") for module in TOP_MODULES
    }
    if set(buckets) != expected_buckets:
        raise ValueError("DeepMind owner sample bucket coverage changed")
    selected: list[dict[str, Any]] = []
    for split, module in sorted(expected_buckets):
        rows = sorted(
            buckets[(split, module)],
            key=lambda row: hashlib.sha256(str(row["record_id"]).encode()).hexdigest(),
        )
        if len(rows) < 2:
            raise ValueError("DeepMind owner sample bucket is too small")
        for row in rows[:2]:
            selected.append(
                {
                    "language_mode": "english",
                    "messages": row["messages"],
                    "record_id": row["record_id"],
                    "skill_id": f"deepmind_math_{module}",
                    "split": row["split"],
                }
            )
    if len(selected) != OWNER_REVIEW_RECORDS:
        raise AssertionError("DeepMind owner sample size changed")
    return selected


def build_scale_owner_review_html(sample: list[Mapping[str, Any]]) -> str:
    sample_sha256 = sha256_value(sample)
    html = build_owner_review_html(sample, sample_sha256=sample_sha256)
    return (
        html.replace("Lumi owner review", "Lumi DeepMind Math scale owner review")
        .replace("Lumi owner spot-check", "Lumi DeepMind Math scale spot-check")
        .replace(
            "lumi-owner-review-decisions-v0.3.json",
            "lumi-deepmind-scale-owner-decisions-v0.1.json",
        )
    )


def artifact_inventory(
    paths: Mapping[str, Path],
    *,
    raw_text_artifacts: Iterable[str],
) -> dict[str, Any]:
    raw_names = set(raw_text_artifacts)
    objects = []
    root = Path(PRIVATE_ROOT)
    for name, path in sorted(paths.items()):
        objects.append(
            {
                "bytes": path.stat().st_size,
                "contains_raw_training_text": name in raw_names,
                "name": name,
                "path": path.relative_to(root).as_posix(),
                "sha256": file_sha256(path),
            }
        )
    return {
        "objects": objects,
        "total_bytes": sum(int(item["bytes"]) for item in objects),
    }


def validate_raw_text_free_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported DeepMind scale report")
    if report.get("status") != "verified_cpu_only_nonproduction_scale_candidate":
        raise ValueError("DeepMind scale report status changed")
    if report.get("raw_training_text_present") is not False:
        raise ValueError("DeepMind scale report contains raw training text")
    if report.get("production") != {
        "candidate_registry_promotions": 0,
        "dataset_sealed": False,
        "gpu_used": False,
        "production_eligible": False,
        "production_sft_authorized": False,
        "sft_training_started": False,
    }:
        raise ValueError("DeepMind scale production boundary changed")
    seal = report.get("report_sha256")
    if not isinstance(seal, str) or _SHA256.fullmatch(seal) is None:
        raise ValueError("DeepMind scale report seal is invalid")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if sha256_value(unsigned) != seal:
        raise ValueError("DeepMind scale report seal changed")


__all__ = [
    "DETERMINISM_ADAPTER",
    "EXPECTED_FLAT_MODULE_INVENTORY_SHA256",
    "EXPECTED_FLAT_MODULES",
    "EXPECTED_GENERATED_RECORDS",
    "EXPECTED_SPLIT_COUNTS",
    "GENERATED_RECORD_FIELDS",
    "MAXIMUM_PRIVATE_BYTES",
    "MAXIMUM_RUNTIME_SECONDS",
    "MAXIMUM_TRAIN_SUPERVISED_TOKENS",
    "MINIMUM_TRAIN_SUPERVISED_TOKENS",
    "OVERLAY_ROOT",
    "PRIVATE_ROOT",
    "PYTHON_EXECUTABLE",
    "REPORT_SCHEMA_VERSION",
    "SOURCE_BYTES",
    "SOURCE_ROOT",
    "SPLIT_MODULE_COUNTS",
    "SPLIT_NAMESPACES",
    "TOP_MODULES",
    "VERIFICATION_SCHEMA_VERSION",
    "apply_prompt_near_dedup",
    "artifact_inventory",
    "audit_source_runtime",
    "build_candidate_records",
    "build_scale_owner_review_html",
    "count_by_split",
    "derive_identity_seed",
    "derive_rng_seed",
    "deterministic_owner_sample",
    "file_sha256",
    "generation_schedule",
    "json_file_bytes",
    "jsonl_file_bytes",
    "load_runtime_components",
    "load_scale_plan",
    "sha256_value",
    "validate_generated_records",
    "validate_raw_text_free_report",
    "write_bytes_atomic",
    "write_json_atomic",
    "write_jsonl_atomic",
]
