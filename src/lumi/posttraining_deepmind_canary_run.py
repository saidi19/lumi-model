"""Bounded, CPU-only DeepMind Mathematics adapter-canary support."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .data_quality import BenchmarkRegistry, matching_tokens, validate_quality_config
from .posttraining_data import SFTPolicy, load_sft_policy
from .posttraining_deepmind_adapter_preflight import derive_seed
from .posttraining_deepmind_source_acquisition import inspect_source_root
from .posttraining_format import ChatTemplate, ChatTokenizer, load_chat_template

PLAN_SCHEMA_VERSION = "lumi.sft-deepmind-canary-run-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-deepmind-canary-run-report.v0.1"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-deepmind-canary-run-verification.v0.1"

SOURCE_ID = "deepmind_mathematics_dataset"
SOURCE_DATASET_ID = "google-deepmind/mathematics_dataset"
SOURCE_REPOSITORY = "https://github.com/google-deepmind/mathematics_dataset"
SOURCE_REVISION = "427f45075f84b8b9774950196ad63867ca20ffb3"
SOURCE_LICENSE = "apache-2.0"
SOURCE_OBJECTS = 30
SOURCE_BYTES = 234_672
SOURCE_INVENTORY_SHA256 = (
    "47e472f97f7092cf264edb727bd0821db59869133f56b12cddf3f8a4ef1a6ce9"
)
LOCAL_SOURCE_INVENTORY_SHA256 = (
    "17652ed033d1e2058c01ae9afa612c0c8ad7e284f6fbf39c5fe6af02db288fcd"
)
SOURCE_ROOT = (
    "/Users/afnan/Documents/Flowaistudio/lumi-artifacts/"
    "4b-sft-deepmind-canary-v0.1/427f45075f84/attempt-001/source"
)
PRIVATE_ROOT = (
    "/Users/afnan/Documents/Flowaistudio/lumi-artifacts/"
    "4b-sft-deepmind-canary-v0.1/427f45075f84/attempt-001/canary"
)
OVERLAY_ROOT = "/private/tmp/lumi-deepmind-runtime-v0.1"
PYTHON_EXECUTABLE = "/private/tmp/lumi-model/.venv/bin/python"

OVERLAY_RECEIPT_INTERNAL_SHA256 = (
    "192fbac2bf5cf1049ff5bcd33e84507bf55d686e296b9dfb2d67cd7b5c0bce72"
)
OVERLAY_VERIFICATION_INTERNAL_SHA256 = (
    "10ce6707e13472fcdf8641a4f9c5b5a8cb120865e0ad5a1e0b4ff51ceffabd59"
)
SOURCE_RECEIPT_INTERNAL_SHA256 = (
    "4d68a934d2d970ef6396c4786fe696f28b4b406e69e1dd7f404515237e3433a9"
)
SOURCE_VERIFICATION_INTERNAL_SHA256 = (
    "e4bf0ad734e8c718e7b0c9e0cbb490a69e7d233d917bee2d77812024e81c00ee"
)

TOP_MODULES = (
    "algebra",
    "arithmetic",
    "calculus",
    "comparison",
    "measurement",
    "numbers",
    "polynomials",
    "probability",
)
TOP_MODULE_FILES = {
    name: f"mathematics_dataset/modules/{name}.py" for name in TOP_MODULES
}
SPLIT_NAMESPACES = {
    "train": "lumi-deepmind-math-train-v0.1",
    "validation": "lumi-deepmind-math-validation-v0.1",
}
SPLIT_MODULE_COUNTS = {
    "train": {
        "algebra": 238,
        "arithmetic": 238,
        "calculus": 238,
        "comparison": 238,
        "measurement": 237,
        "numbers": 237,
        "polynomials": 237,
        "probability": 237,
    },
    "validation": {
        "algebra": 13,
        "arithmetic": 13,
        "calculus": 13,
        "comparison": 13,
        "measurement": 12,
        "numbers": 12,
        "polynomials": 12,
        "probability": 12,
    },
}
EXPECTED_SPLIT_COUNTS = {"train": 1_900, "validation": 100}
EXPECTED_GENERATED_RECORDS = 2_000
EXPECTED_FLAT_MODULES = 56
EXPECTED_FLAT_MODULE_INVENTORY_SHA256 = (
    "58870a10ba068a94aff4f32ea2b298199a8788aa1d245e4fdd901d3bf93dd618"
)
DETERMINISM_ADAPTER = [
    "numpy2_itemset_compatibility_with_stable_coefficient_index_order",
    "python313_integer_valued_randint_bounds_for_round_number",
    "stable_context_symbol_order_before_seeded_choice",
    "stable_entity_identity_dedup_before_seeded_shuffle",
    "sympy_internal_rng_seeded_from_record_seed",
]
MAX_QUESTION_CHARACTERS = 160
MAX_ANSWER_CHARACTERS = 30
GENERATED_RECORD_FIELDS = frozenset(
    {"answer", "module", "question", "seed", "split_namespace"}
)

EXPECTED_AUTHORIZATION = {
    "candidate_registry_promotion_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_spend_authorized": False,
    "external_distribution_authorized": False,
    "gpu_authorized": False,
    "network_access_authorized": True,
    "official_test_or_evaluation_generation_authorized": False,
    "package_installation_authorized": False,
    "production_sft_authorized": False,
    "purpose": "bounded_deepmind_exact_source_and_2000_record_adapter_canary_retry",
    "source_acquisition_authorized": True,
    "unrelated_data_access_authorized": False,
}
EXPECTED_LIMITS = {
    "exact_source_bytes": SOURCE_BYTES,
    "exact_source_objects": SOURCE_OBJECTS,
    "maximum_canary_generated_records": EXPECTED_GENERATED_RECORDS,
    "maximum_decoded_bytes": 8_388_608,
    "maximum_network_bytes": 1_048_576,
    "maximum_runtime_minutes": 180,
}
EXPECTED_GENERATION = {
    "assistant_response_template": "The answer is {generated_answer}.",
    "determinism_adapter": DETERMINISM_ADAPTER,
    "fresh_process_per_shard": True,
    "generated_record_fields": sorted(GENERATED_RECORD_FIELDS),
    "maximum_answer_characters": MAX_ANSWER_CHARACTERS,
    "maximum_question_characters": MAX_QUESTION_CHARACTERS,
    "network_disabled_during_generation": True,
    "numpy_and_python_rng_seed_match": True,
    "official_test_or_evaluation_generation": False,
    "replay": "same_seed_cross_process_byte_exact",
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
    "pii_and_secret_filter": "lumi_sft_basic_pattern_scan_v0.1",
    "source_registry_check": "deferred_until_separate_promotion",
    "tokenizer_4k_fit": True,
}
EXPECTED_OUTPUT = {
    "private_root": PRIVATE_ROOT,
    "raw_text_free_report_path": "evidence/lumi-4b-sft-deepmind-canary-run-v0.1.json",
    "raw_text_free_verification_path": (
        "evidence/lumi-4b-sft-deepmind-canary-run-verification-v0.1.json"
    ),
    "raw_training_text_in_git": False,
}
EXPECTED_BINDING_PATHS = {
    "adapter_preflight_plan": "configs/lumi-4b-sft-deepmind-adapter-preflight-v0.1.json",
    "benchmark_registry": "data/benchmark-signatures-v0.1.jsonl.gz",
    "canary_implementation": "src/lumi/posttraining_deepmind_canary_run.py",
    "canary_runner": "scripts/run_deepmind_canary.py",
    "canary_verifier": "scripts/verify_deepmind_canary.py",
    "canary_worker": "scripts/deepmind_canary_worker.py",
    "chat_template": "configs/lumi-4b-chat-template-v0.1.json",
    "data_contract": "configs/lumi-4b-sft-data-contract-v0.1.json",
    "dependency_overlay_receipt": "evidence/lumi-4b-sft-deepmind-dependency-overlay-v0.1.json",
    "dependency_overlay_verification": (
        "evidence/lumi-4b-sft-deepmind-dependency-overlay-verification-v0.1.json"
    ),
    "filtering_implementation": "src/lumi/posttraining_data.py",
    "format_implementation": "src/lumi/posttraining_format.py",
    "quality_config": "configs/data-quality-v0.1.json",
    "source_acquisition_receipt": "evidence/lumi-4b-sft-deepmind-source-acquisition-v0.1.json",
    "source_acquisition_verification": (
        "evidence/lumi-4b-sft-deepmind-source-acquisition-verification-v0.1.json"
    ),
    "source_inventory_implementation": "src/lumi/posttraining_deepmind_source_acquisition.py",
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
_RUNTIME_EXCLUDED_SOURCE_FILES = {
    "mathematics_dataset/generate.py",
    "mathematics_dataset/generate_settings.py",
    "mathematics_dataset/generate_to_file.py",
    "setup.py",
}
_ALLOWED_IMPORT_ROOTS = {
    "__future__",
    "abc",
    "absl",
    "collections",
    "decimal",
    "functools",
    "hashlib",
    "itertools",
    "math",
    "mathematics_dataset",
    "numpy",
    "os",
    "random",
    "setuptools",
    "six",
    "string",
    "sympy",
    "textwrap",
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_file_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def jsonl_file_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        canonical_json_bytes(dict(row)) + b"\n"
        for row in rows
    )


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a SHA-256")
    return value


def _verify_internal_seal(value: Mapping[str, Any], field: str) -> str:
    seal = _require_sha256(value.get(field), field)
    unsigned = {key: item for key, item in value.items() if key != field}
    if sha256_value(unsigned) != seal:
        raise ValueError(f"DeepMind bound evidence seal changed: {field}")
    return seal


def resolve_binding(
    repository_root: str | Path,
    binding: Mapping[str, Any],
    *,
    name: str,
) -> Path:
    root = Path(repository_root).resolve()
    relative = binding.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"DeepMind canary binding path is invalid: {name}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.parent):
        raise ValueError(f"DeepMind canary binding escapes workspace: {name}")
    if not resolved.is_file():
        raise FileNotFoundError(f"DeepMind canary binding is missing: {name}")
    expected = _require_sha256(binding.get("sha256"), f"bindings.{name}.sha256")
    if file_sha256(resolved) != expected:
        raise ValueError(f"DeepMind canary binding changed: {name}")
    return resolved


def load_canary_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    plan = _load_json(path)
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "generation": EXPECTED_GENERATION,
        "limits": EXPECTED_LIMITS,
        "observed_at": "2026-08-06",
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
        "status": "authorized_exact_source_cpu_only_adapter_canary_retry",
    }
    for field, expected_value in expected.items():
        if plan.get(field) != expected_value:
            raise ValueError(f"DeepMind canary plan changed: {field}")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(EXPECTED_BINDING_PATHS):
        raise ValueError("DeepMind canary binding set changed")
    for name, expected_path in EXPECTED_BINDING_PATHS.items():
        binding = bindings[name]
        if not isinstance(binding, Mapping) or binding.get("path") != expected_path:
            raise ValueError(f"DeepMind canary binding path changed: {name}")
        resolve_binding(repository_root, binding, name=name)
    verify_bound_evidence(plan, repository_root=repository_root)
    return plan


def verify_bound_evidence(
    plan: Mapping[str, Any],
    *,
    repository_root: str | Path,
) -> None:
    bindings = plan["bindings"]
    overlay_receipt = _load_json(
        resolve_binding(
            repository_root,
            bindings["dependency_overlay_receipt"],
            name="dependency_overlay_receipt",
        )
    )
    overlay_verification = _load_json(
        resolve_binding(
            repository_root,
            bindings["dependency_overlay_verification"],
            name="dependency_overlay_verification",
        )
    )
    source_receipt = _load_json(
        resolve_binding(
            repository_root,
            bindings["source_acquisition_receipt"],
            name="source_acquisition_receipt",
        )
    )
    source_verification = _load_json(
        resolve_binding(
            repository_root,
            bindings["source_acquisition_verification"],
            name="source_acquisition_verification",
        )
    )
    checks = (
        (
            overlay_receipt,
            "receipt_sha256",
            OVERLAY_RECEIPT_INTERNAL_SHA256,
            "verified_isolated_dependency_overlay_ready_for_canary_gate",
        ),
        (
            overlay_verification,
            "verification_sha256",
            OVERLAY_VERIFICATION_INTERNAL_SHA256,
            "verified_exact_deepmind_dependency_overlay",
        ),
        (
            source_receipt,
            "receipt_sha256",
            SOURCE_RECEIPT_INTERNAL_SHA256,
            "verified_exact_deepmind_source_acquired_nonproduction",
        ),
        (
            source_verification,
            "verification_sha256",
            SOURCE_VERIFICATION_INTERNAL_SHA256,
            "verified_exact_deepmind_source_acquisition_nonproduction",
        ),
    )
    for value, field, expected_seal, expected_status in checks:
        if _verify_internal_seal(value, field) != expected_seal:
            raise ValueError(f"DeepMind bound evidence is not authorized: {field}")
        if value.get("status") != expected_status:
            raise ValueError(f"DeepMind bound evidence status changed: {expected_status}")
    source = source_receipt.get("source")
    if not isinstance(source, Mapping) or source.get("root") != SOURCE_ROOT:
        raise ValueError("DeepMind acquired source root changed")


def _assignment_literal(tree: ast.Module, name: str) -> Any:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise ValueError(f"DeepMind source assignment is missing: {name}")


def audit_source_runtime(source_root: str | Path) -> dict[str, Any]:
    """Audit the exact source without importing the excluded CLI generators."""

    source = Path(source_root).resolve()
    inventory = inspect_source_root(source)
    if inventory["objects"] != SOURCE_OBJECTS:
        raise ValueError("DeepMind source object count changed")
    if inventory["network_payload_bytes"] != SOURCE_BYTES:
        raise ValueError("DeepMind source byte count changed")
    if inventory["inventory_sha256"] != LOCAL_SOURCE_INVENTORY_SHA256:
        raise ValueError("DeepMind local source inventory changed")

    python_files = sorted(source.rglob("*.py"))
    imports_by_file: dict[str, list[str]] = {}
    mutable_globals_by_file: dict[str, list[str]] = {}
    rng_files: set[str] = set()
    for path in python_files:
        relative = path.relative_to(source).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        imports: set[str] = set()
        mutable_globals: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if not isinstance(value, (ast.Dict, ast.List, ast.Set)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            mutable_globals.update(
                target.id for target in targets if isinstance(target, ast.Name)
            )
        unknown = imports - _ALLOWED_IMPORT_ROOTS
        if unknown:
            raise ValueError(f"DeepMind source gained imports in {relative}: {sorted(unknown)}")
        if {"random", "numpy"}.intersection(imports):
            rng_files.add(relative)
        imports_by_file[relative] = sorted(imports)
        if mutable_globals:
            mutable_globals_by_file[relative] = sorted(mutable_globals)

    settings_path = source / "mathematics_dataset/generate_settings.py"
    settings_tree = ast.parse(settings_path.read_text(encoding="utf-8"))
    if _assignment_literal(settings_tree, "MAX_QUESTION_LENGTH") != MAX_QUESTION_CHARACTERS:
        raise ValueError("DeepMind question-length limit changed")
    if _assignment_literal(settings_tree, "MAX_ANSWER_LENGTH") != MAX_ANSWER_CHARACTERS:
        raise ValueError("DeepMind answer-length limit changed")
    if len(python_files) != 28:
        raise ValueError("DeepMind Python source file count changed")
    external_import_roots = sorted(
        {
            name
            for names in imports_by_file.values()
            for name in names
            if name in {"absl", "numpy", "setuptools", "six", "sympy"}
        }
    )
    return {
        "all_source_objects_reverified": True,
        "external_import_roots": external_import_roots,
        "full_source_import_inventory_sha256": sha256_value(imports_by_file),
        "local_source_inventory_sha256": inventory["inventory_sha256"],
        "mutable_global_names_by_file": mutable_globals_by_file,
        "python_source_files": len(python_files),
        "question_answer_length_limits": {
            "answer": MAX_ANSWER_CHARACTERS,
            "question": MAX_QUESTION_CHARACTERS,
        },
        "rng_files": sorted(rng_files),
        "rng_providers": ["numpy.random", "python.random"],
        "runtime_excluded_source_files": sorted(_RUNTIME_EXCLUDED_SOURCE_FILES),
        "source_bytes": inventory["network_payload_bytes"],
        "source_objects": inventory["objects"],
        "upstream_generate_cli_imported_by_adapter": False,
    }


def generation_schedule() -> list[dict[str, Any]]:
    schedule: list[dict[str, Any]] = []
    for split in ("train", "validation"):
        for top_module in TOP_MODULES:
            schedule.append(
                {
                    "count": SPLIT_MODULE_COUNTS[split][top_module],
                    "module_file": TOP_MODULE_FILES[top_module],
                    "namespace": SPLIT_NAMESPACES[split],
                    "split": split,
                    "top_module": top_module,
                }
            )
    if sum(item["count"] for item in schedule) != EXPECTED_GENERATED_RECORDS:
        raise AssertionError("DeepMind generation schedule count changed")
    return schedule


def validate_generated_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(records) != EXPECTED_GENERATED_RECORDS:
        raise ValueError("DeepMind generated record count changed")
    cursor = 0
    seed_by_split: dict[str, set[int]] = {"train": set(), "validation": set()}
    prompt_hashes: dict[str, int] = {}
    prompt_observations: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    module_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    for shard in generation_schedule():
        for index in range(shard["count"]):
            record = records[cursor]
            cursor += 1
            if frozenset(record) != GENERATED_RECORD_FIELDS:
                raise ValueError("DeepMind generated record schema changed")
            namespace = record.get("split_namespace")
            if namespace != shard["namespace"]:
                raise ValueError("DeepMind generated split namespace changed")
            module = record.get("module")
            if not isinstance(module, str) or not module.startswith(
                f"{shard['top_module']}__"
            ):
                raise ValueError("DeepMind generated module assignment changed")
            expected_seed = derive_seed(shard["namespace"], shard["module_file"], index)
            seed = record.get("seed")
            if seed != expected_seed:
                raise ValueError("DeepMind generated seed derivation changed")
            if seed in seed_by_split[shard["split"]]:
                raise ValueError("DeepMind generated seed collided within a split")
            seed_by_split[shard["split"]].add(seed)
            question = record.get("question")
            answer = record.get("answer")
            if not isinstance(question, str) or not question.strip():
                raise ValueError("DeepMind generated question is empty")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("DeepMind generated answer is empty")
            if len(question) > MAX_QUESTION_CHARACTERS:
                raise ValueError("DeepMind generated question exceeded source limit")
            if len(answer) > MAX_ANSWER_CHARACTERS:
                raise ValueError("DeepMind generated answer exceeded source limit")
            prompt_sha256 = hashlib.sha256(question.strip().encode("utf-8")).hexdigest()
            prompt_hashes[prompt_sha256] = prompt_hashes.get(prompt_sha256, 0) + 1
            prompt_observations[prompt_sha256].append(
                (
                    str(shard["split"]),
                    hashlib.sha256(answer.strip().encode("utf-8")).hexdigest(),
                )
            )
            module_counts[module] += 1
            split_counts[shard["split"]] += 1
    if cursor != len(records):
        raise AssertionError("DeepMind generated record cursor changed")
    if dict(split_counts) != EXPECTED_SPLIT_COUNTS:
        raise ValueError("DeepMind generated split counts changed")
    if seed_by_split["train"] & seed_by_split["validation"]:
        raise ValueError("DeepMind train and validation seeds overlap")
    duplicate_groups = {
        prompt_sha256: observations
        for prompt_sha256, observations in prompt_observations.items()
        if len(observations) > 1
    }
    duplicate_prompts = sum(len(items) - 1 for items in duplicate_groups.values())
    conflicting_answer_groups = sum(
        len({answer_sha256 for _split, answer_sha256 in items}) > 1
        for items in duplicate_groups.values()
    )
    train_validation_prompt_overlap = sum(
        len({split for split, _answer_sha256 in items}) > 1
        for items in duplicate_groups.values()
    )
    if conflicting_answer_groups:
        raise ValueError("DeepMind duplicate prompt gained conflicting answers")
    if train_validation_prompt_overlap:
        raise ValueError("DeepMind train and validation prompts overlap")
    return {
        "conflicting_answer_prompt_groups": conflicting_answer_groups,
        "different_seed_prompt_groups": len(duplicate_groups),
        "different_seed_prompt_overlap": duplicate_prompts,
        "flat_modules_observed": len(module_counts),
        "generated_records": len(records),
        "module_counts_sha256": sha256_value(dict(sorted(module_counts.items()))),
        "prompt_inventory_sha256": sha256_value(sorted(prompt_hashes)),
        "seed_inventory_sha256": sha256_value(
            {
                split: sorted(seeds) for split, seeds in sorted(seed_by_split.items())
            }
        ),
        "split_counts": dict(sorted(split_counts.items())),
        "train_validation_prompt_overlap": train_validation_prompt_overlap,
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
        module = str(generated["module"])
        top_module = module.split("__", 1)[0]
        question = str(generated["question"]).strip()
        generated_answer = str(generated["answer"]).strip()
        response = f"The answer is {generated_answer}."
        record_id = f"deepmind-{split}-{seed:010d}"
        source_content = {
            "answer": generated_answer,
            "module": module,
            "question": question,
            "seed": seed,
            "split_namespace": namespace,
        }
        messages = [
            {"content": question, "role": "user"},
            {"content": response, "role": "assistant"},
        ]
        content_sha256 = sha256_value(source_content)
        transform_sha256 = sha256_value(
            {
                "messages": messages,
                "transform": "deepmind_final_answer_adapter_v0.1",
            }
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
                        "deepmind_fixed_seed_generator_plus_lumi_canary_filters_v0.1"
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
                    "document_id": f"{namespace}:{seed}",
                    "license_id": SOURCE_LICENSE,
                    "revision": SOURCE_REVISION,
                    "source_id": SOURCE_ID,
                    "uri": (
                        f"{SOURCE_REPOSITORY}/blob/{SOURCE_REVISION}/"
                        f"mathematics_dataset/modules/{top_module}.py#seed-{seed}"
                    ),
                },
                "split": split,
                "task_category": "reasoning_math_science",
            }
        )
        attributions.append(
            {
                "answer_sha256": hashlib.sha256(
                    generated_answer.encode("utf-8")
                ).hexdigest(),
                "content_sha256": content_sha256,
                "generated_record_fields": sorted(GENERATED_RECORD_FIELDS),
                "module": module,
                "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
                "record_id": record_id,
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


def apply_prompt_near_dedup(
    records: Sequence[Mapping[str, Any]],
    *,
    shingle_tokens: int = 5,
    threshold: float = 0.9,
) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Apply deterministic global prompt dedup with validation precedence."""

    if shingle_tokens != 5 or threshold != 0.9:
        raise ValueError("DeepMind near-duplicate policy changed")
    ordered = sorted(
        records,
        key=lambda row: (
            0 if row["split"] == "validation" else 1,
            str(row["record_id"]),
        ),
    )
    accepted_signatures: dict[str, frozenset[str]] = {}
    accepted_split: dict[str, str] = {}
    inverted: defaultdict[str, set[str]] = defaultdict(set)
    accepted: list[Mapping[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    cross_split_rejected = 0
    for record in ordered:
        prompt = str(record["messages"][0]["content"])
        tokens = matching_tokens(prompt)
        shingles = (
            frozenset({" ".join(tokens)})
            if len(tokens) < shingle_tokens
            else frozenset(
                " ".join(tokens[index : index + shingle_tokens])
                for index in range(len(tokens) - shingle_tokens + 1)
            )
        )
        candidate_ids: set[str] = set()
        for shingle in shingles:
            candidate_ids.update(inverted[shingle])
        duplicate_of = None
        duplicate_similarity = 0.0
        for candidate_id in sorted(candidate_ids):
            candidate = accepted_signatures[candidate_id]
            union_size = len(shingles | candidate)
            similarity = len(shingles & candidate) / union_size if union_size else 1.0
            if similarity >= threshold:
                duplicate_of = candidate_id
                duplicate_similarity = similarity
                break
        record_id = str(record["record_id"])
        accepted_row = duplicate_of is None
        if accepted_row:
            accepted_signatures[record_id] = shingles
            accepted_split[record_id] = str(record["split"])
            for shingle in shingles:
                inverted[shingle].add(record_id)
            accepted.append(record)
        elif accepted_split[duplicate_of] != record["split"]:
            cross_split_rejected += 1
        ledger.append(
            {
                "accepted": accepted_row,
                "duplicate_of": duplicate_of,
                "duplicate_similarity": round(duplicate_similarity, 6),
                "record_id": record_id,
                "signature_sha256": sha256_value(sorted(shingles)),
                "split": record["split"],
            }
        )
    accepted.sort(key=lambda row: (str(row["split"]), str(row["record_id"])))
    ledger.sort(key=lambda row: str(row["record_id"]))
    metrics = {
        "accepted": len(accepted),
        "algorithm": "exact_token_shingle_jaccard_inverted_index_v0.1",
        "cross_split_rejected": cross_split_rejected,
        "global_scope": True,
        "rejected": len(records) - len(accepted),
        "shingle_tokens": shingle_tokens,
        "threshold": threshold,
        "validation_precedence": True,
    }
    return accepted, ledger, metrics


def load_runtime_components(
    plan: Mapping[str, Any],
    *,
    repository_root: Path,
) -> tuple[SFTPolicy, BenchmarkRegistry, ChatTemplate, ChatTokenizer]:
    from transformers import AutoTokenizer

    bindings = plan["bindings"]
    contract_path = resolve_binding(
        repository_root, bindings["data_contract"], name="data_contract"
    )
    quality_path = resolve_binding(
        repository_root, bindings["quality_config"], name="quality_config"
    )
    benchmark_path = resolve_binding(
        repository_root, bindings["benchmark_registry"], name="benchmark_registry"
    )
    template_path = resolve_binding(
        repository_root, bindings["chat_template"], name="chat_template"
    )
    tokenizer_json = resolve_binding(
        repository_root, bindings["tokenizer_json"], name="tokenizer_json"
    )
    tokenizer_config = resolve_binding(
        repository_root, bindings["tokenizer_config"], name="tokenizer_config"
    )
    if tokenizer_json.parent != tokenizer_config.parent:
        raise ValueError("DeepMind frozen tokenizer files are not colocated")
    quality = _load_json(quality_path)
    validate_quality_config(quality)
    policy = load_sft_policy(contract_path)
    benchmark = BenchmarkRegistry(policy.benchmark_decontamination, repository_root)
    if benchmark.path.resolve() != benchmark_path:
        raise ValueError("DeepMind benchmark registry resolved to another path")
    template = load_chat_template(template_path)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_json.parent, local_files_only=True)
    return policy, benchmark, template, tokenizer


def count_by_split(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(record["split"]) for record in records).items()))


def artifact_inventory(
    paths: Mapping[str, Path],
    *,
    raw_text_artifacts: Iterable[str],
) -> dict[str, Any]:
    raw_names = set(raw_text_artifacts)
    objects = []
    for name, path in sorted(paths.items()):
        objects.append(
            {
                "bytes": path.stat().st_size,
                "contains_raw_training_text": name in raw_names,
                "name": name,
                "path": path.relative_to(Path(PRIVATE_ROOT)).as_posix(),
                "sha256": file_sha256(path),
            }
        )
    return {
        "objects": objects,
        "total_bytes": sum(item["bytes"] for item in objects),
    }


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite DeepMind artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_json_atomic(path: Path, value: Any) -> None:
    write_bytes_atomic(path, json_file_bytes(value))


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    write_bytes_atomic(path, jsonl_file_bytes(rows))


def validate_raw_text_free_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported DeepMind canary report")
    if report.get("status") != "verified_cpu_only_nonproduction_adapter_canary":
        raise ValueError("DeepMind canary status changed")
    if report.get("raw_training_text_present") is not False:
        raise ValueError("DeepMind canary report contains raw training text")
    production = report.get("production")
    if production != {
        "candidate_registry_promotions": 0,
        "dataset_sealed": False,
        "gpu_used": False,
        "production_eligible": False,
        "production_sft_authorized": False,
        "sft_training_started": False,
    }:
        raise ValueError("DeepMind canary production boundary changed")
    seal = _require_sha256(report.get("report_sha256"), "report_sha256")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if sha256_value(unsigned) != seal:
        raise ValueError("DeepMind canary report seal changed")
