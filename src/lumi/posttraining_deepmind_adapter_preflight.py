"""No-download adapter preflight for the pinned DeepMind Mathematics source."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .posttraining_math_replacement_preflight import (
    DEEPMIND_OBJECTS,
    DEEPMIND_REVISION,
)

PLAN_SCHEMA_VERSION = "lumi.sft-deepmind-adapter-preflight-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-deepmind-adapter-preflight-report.v0.1"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-deepmind-adapter-preflight-verification.v0.1"

SOURCE_ID = "deepmind_mathematics_dataset"
SOURCE_REPOSITORY = "https://github.com/google-deepmind/mathematics_dataset"
SOURCE_LICENSE = "apache-2.0"
SOURCE_OBJECTS = 30
SOURCE_BYTES = 234_672
SOURCE_INVENTORY_SHA256 = "47e472f97f7092cf264edb727bd0821db59869133f56b12cddf3f8a4ef1a6ce9"
DMATH_REPORT_INTERNAL_SHA256 = "c00b88c977a95f6bd8f49350a210dce0954c9255ca5b35730ab84001df3415c4"
DMATH_VERIFICATION_INTERNAL_SHA256 = (
    "05a22df0cb0d84483a8bb9f0e6f395228ef3bc706c1ec57da253321a5eef2132"
)

EXPECTED_AUTHORIZATION = {
    "candidate_registry_promotion_authorized": False,
    "dataset_acquisition_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_spend_authorized": False,
    "generator_execution_authorized": False,
    "local_preflight_authorized": True,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "purpose": "local_no_download_deepmind_adapter_preflight_only",
    "test_or_evaluation_generation_authorized": False,
    "unrelated_dataset_access_authorized": False,
}
EXPECTED_LIMITS = {
    "maximum_network_bytes": 0,
    "maximum_runtime_minutes": 60,
    "source_bytes": SOURCE_BYTES,
    "source_objects": SOURCE_OBJECTS,
}
EXPECTED_OUTPUT = {
    "report_path": "evidence/lumi-4b-sft-deepmind-adapter-preflight-v0.1.json",
    "verification_path": ("evidence/lumi-4b-sft-deepmind-adapter-preflight-verification-v0.1.json"),
    "raw_training_text_in_report": False,
}
EXPECTED_BINDING_PATHS = {
    "adapter_preflight_checker": "scripts/check_deepmind_adapter_preflight.py",
    "adapter_preflight_implementation": ("src/lumi/posttraining_deepmind_adapter_preflight.py"),
    "adapter_preflight_verifier": "scripts/verify_deepmind_adapter_preflight.py",
    "dmath_curation_report": "evidence/lumi-4b-sft-dmath-curation-v0.1.json",
    "dmath_curation_verification": ("evidence/lumi-4b-sft-dmath-curation-verification-v0.1.json"),
    "math_replacement_plan": "configs/lumi-4b-sft-math-replacement-preflight-v0.1.json",
    "math_replacement_report": ("evidence/lumi-4b-sft-math-replacement-preflight-v0.1.json"),
    "math_replacement_verification": (
        "evidence/lumi-4b-sft-math-replacement-preflight-verification-v0.1.json"
    ),
}

GENERATOR_MODULE_PATHS = [
    "mathematics_dataset/modules/algebra.py",
    "mathematics_dataset/modules/arithmetic.py",
    "mathematics_dataset/modules/calculus.py",
    "mathematics_dataset/modules/comparison.py",
    "mathematics_dataset/modules/measurement.py",
    "mathematics_dataset/modules/numbers.py",
    "mathematics_dataset/modules/polynomials.py",
    "mathematics_dataset/modules/probability.py",
]
SEED_CONTRACT = {
    "algorithm": "sha256_first_u32_big_endian",
    "derivation": "SHA256(namespace + NUL + module + NUL + decimal_index)[0:4]",
    "fresh_process_per_shard_required": True,
    "numpy_and_python_rng_seed_must_match": True,
    "split_namespaces": {
        "train": "lumi-deepmind-math-train-v0.1",
        "validation": "lumi-deepmind-math-validation-v0.1",
    },
    "test_namespace_defined": False,
}
ADAPTER_CONTRACT = {
    "assistant_response_template": "The answer is {generated_answer}.",
    "benchmark_decontamination_required": True,
    "dependency_lock_required_before_execution": True,
    "deterministic_same_seed_byte_replay_required": True,
    "different_seed_prompt_overlap_maximum": 0,
    "generated_record_fields": [
        "answer",
        "module",
        "question",
        "seed",
        "split_namespace",
    ],
    "generator_modules": GENERATOR_MODULE_PATHS,
    "network_disabled_during_generation": True,
    "official_test_or_evaluation_generation_permitted": False,
    "per_record_attribution_required": True,
    "pii_secret_filter_required": True,
    "question_and_answer_only_training_target": True,
    "raw_generator_trace_training_target_permitted": False,
    "runtime_ast_import_rng_audit_required": True,
    "seed_contract": SEED_CONTRACT,
    "source_registry_check": "deferred_until_separate_promotion",
    "tokenizer_4k_fit_required": True,
    "train_validation_exact_and_near_dedup_required": True,
}
NEXT_GATE_LIMITS = {
    "exact_source_bytes": SOURCE_BYTES,
    "exact_source_objects": SOURCE_OBJECTS,
    "maximum_canary_generated_records": 2_000,
    "maximum_decoded_bytes": 8_388_608,
    "maximum_network_bytes": 1_048_576,
    "maximum_runtime_minutes": 180,
}
NEXT_AUTHORIZATION_STATEMENT = (
    "I approve one bounded CPU-only DeepMind Mathematics source acquisition and "
    f"adapter canary at revision {DEEPMIND_REVISION}, limited to the exact 30 pinned "
    "source/license/documentation objects (234,672 bytes total), maximum 1 MiB network / "
    "8 MiB decoded / 180 minutes and at most 2,000 locally generated canary records. It "
    "may audit imports, dependencies, RNG/global state, build the fixed-seed train and "
    "validation adapter, run deterministic replay, schema, PII/secret, deduplication, "
    "benchmark-decontamination, 4K-fit and frozen-tokenizer measurement, and write only "
    "private non-production canary data plus raw-text-free evidence. No pre-generated "
    "bulk data, official test/evaluation generation, external API, package installation, "
    "GPU, source-registry promotion, dataset seal, SFT training, unrelated data, or "
    "external distribution. Stop fail-closed if the pinned runtime dependencies are not "
    "already available."
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


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


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a SHA-256")
    return value


def resolve_binding(
    repository_root: str | Path,
    binding: Mapping[str, Any],
    *,
    name: str,
) -> Path:
    root = Path(repository_root).resolve()
    workspace = root.parent.resolve()
    relative = binding.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"DeepMind adapter binding path is invalid: {name}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError(f"DeepMind adapter binding escapes workspace: {name}")
    if not resolved.is_file():
        raise FileNotFoundError(f"DeepMind adapter binding is missing: {name}")
    expected = _require_sha256(binding.get("sha256"), f"bindings.{name}.sha256")
    if file_sha256(resolved) != expected:
        raise ValueError(f"DeepMind adapter binding changed: {name}")
    return resolved


def load_adapter_preflight_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    plan = _load_json(path)
    expected = {
        "adapter_contract": ADAPTER_CONTRACT,
        "authorization": EXPECTED_AUTHORIZATION,
        "limits": EXPECTED_LIMITS,
        "next_gate": {
            "authorization_required": True,
            "exact_authorization_statement": NEXT_AUTHORIZATION_STATEMENT,
            "limits": NEXT_GATE_LIMITS,
        },
        "observed_at": "2026-08-06",
        "output": EXPECTED_OUTPUT,
        "schema_version": PLAN_SCHEMA_VERSION,
        "source": {
            "declared_license_id": SOURCE_LICENSE,
            "exact_object_bytes": SOURCE_BYTES,
            "exact_object_count": SOURCE_OBJECTS,
            "exact_object_inventory_sha256": SOURCE_INVENTORY_SHA256,
            "repository_url": SOURCE_REPOSITORY,
            "revision": DEEPMIND_REVISION,
            "source_id": SOURCE_ID,
        },
        "status": "local_no_download_deepmind_adapter_preflight_authorized",
    }
    for field, value in expected.items():
        if plan.get(field) != value:
            raise ValueError(f"DeepMind adapter preflight plan changed: {field}")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(EXPECTED_BINDING_PATHS):
        raise ValueError("DeepMind adapter preflight binding set changed")
    for name, expected_path in EXPECTED_BINDING_PATHS.items():
        binding = bindings[name]
        if not isinstance(binding, Mapping) or binding.get("path") != expected_path:
            raise ValueError(f"DeepMind adapter preflight binding path changed: {name}")
        resolve_binding(repository_root, binding, name=name)
    return plan


def validate_source_inventory(objects: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if list(objects) != DEEPMIND_OBJECTS:
        raise ValueError("DeepMind exact source inventory changed")
    if len(objects) != SOURCE_OBJECTS:
        raise ValueError("DeepMind exact source object count changed")
    if sum(int(item["size_bytes"]) for item in objects) != SOURCE_BYTES:
        raise ValueError("DeepMind exact source byte count changed")
    if sha256_value(list(objects)) != SOURCE_INVENTORY_SHA256:
        raise ValueError("DeepMind exact source inventory SHA-256 changed")
    paths = [str(item["path"]) for item in objects]
    if len(paths) != len(set(paths)):
        raise ValueError("DeepMind exact source paths are duplicated")
    if not set(GENERATOR_MODULE_PATHS).issubset(paths):
        raise ValueError("DeepMind expected generator modules changed")
    return {
        "bytes": SOURCE_BYTES,
        "generator_module_count": len(GENERATOR_MODULE_PATHS),
        "inventory_sha256": SOURCE_INVENTORY_SHA256,
        "objects": SOURCE_OBJECTS,
    }


def derive_seed(namespace: str, module: str, index: int) -> int:
    if namespace not in SEED_CONTRACT["split_namespaces"].values():
        raise ValueError("DeepMind seed namespace is not allowed")
    if module not in GENERATOR_MODULE_PATHS:
        raise ValueError("DeepMind seed module is not allowed")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("DeepMind seed index must be non-negative")
    payload = f"{namespace}\0{module}\0{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def seed_contract_evidence(sample_count: int = 64) -> dict[str, Any]:
    if sample_count < 1:
        raise ValueError("DeepMind seed sample count must be positive")
    by_split: dict[str, list[int]] = {}
    for split, namespace in SEED_CONTRACT["split_namespaces"].items():
        seeds = [
            derive_seed(namespace, module, index)
            for module in GENERATOR_MODULE_PATHS
            for index in range(sample_count)
        ]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"DeepMind sampled seeds collide within {split}")
        by_split[split] = seeds
    overlap = set(by_split["train"]) & set(by_split["validation"])
    if overlap:
        raise ValueError("DeepMind sampled train/validation seeds overlap")
    return {
        "algorithm": SEED_CONTRACT["algorithm"],
        "sampled_seeds_per_split": len(by_split["train"]),
        "sampled_train_validation_seed_overlap": 0,
        "train_seed_sample_sha256": sha256_value(by_split["train"]),
        "validation_seed_sample_sha256": sha256_value(by_split["validation"]),
    }


def load_bound_evidence(
    plan: Mapping[str, Any],
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    bindings = plan["bindings"]
    math_report = _load_json(
        resolve_binding(
            repository_root,
            bindings["math_replacement_report"],
            name="math_replacement_report",
        )
    )
    math_verification = _load_json(
        resolve_binding(
            repository_root,
            bindings["math_replacement_verification"],
            name="math_replacement_verification",
        )
    )
    dmath_report = _load_json(
        resolve_binding(
            repository_root,
            bindings["dmath_curation_report"],
            name="dmath_curation_report",
        )
    )
    dmath_verification = _load_json(
        resolve_binding(
            repository_root,
            bindings["dmath_curation_verification"],
            name="dmath_curation_verification",
        )
    )
    if math_report.get("report_sha256") != (
        "aaba91c90ea8d99a42cdeeb677df5826041d4e9de39ffa0012ae26b764a24ba4"
    ):
        raise ValueError("math replacement report changed")
    if math_verification.get("verification_sha256") != (
        "3a21d15ae37097e76c48c5522a914ced9360369d104594f2a15e088f1a92d69e"
    ):
        raise ValueError("math replacement verification changed")
    if dmath_report.get("report_sha256") != DMATH_REPORT_INTERNAL_SHA256:
        raise ValueError("DMath curation report changed")
    if dmath_verification.get("verification_sha256") != (DMATH_VERIFICATION_INTERNAL_SHA256):
        raise ValueError("DMath curation verification changed")
    if dmath_report.get("owner_review", {}).get("complete") is not False:
        raise ValueError("DMath owner-review state changed")
    return {
        "dmath_candidate_train_supervised_tokens": dmath_report["mix_projection"][
            "candidate_train_supervised_tokens"
        ],
        "dmath_owner_review_complete": False,
        "math_minimum_gap_before_dmath": math_report["mix_gap"][
            "reasoning_math_minimum_gap_after_reviewed_draft"
        ],
        "planning_train_supervised_tokens_before_dmath": math_report["mix_gap"][
            "planning_train_supervised_tokens"
        ],
        "projected_10m_floor_gap_after_dmath": dmath_report["mix_projection"][
            "projected_10m_floor_gap_after_review_and_promotion"
        ],
        "projected_planning_train_supervised_tokens_after_dmath": dmath_report["mix_projection"][
            "projected_planning_train_supervised_tokens"
        ],
    }


def build_preflight_report(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    bound_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    dmath_tokens = int(bound_evidence["dmath_candidate_train_supervised_tokens"])
    math_gap = max(
        0,
        int(bound_evidence["math_minimum_gap_before_dmath"]) - dmath_tokens,
    )
    report = {
        "adapter_contract": dict(ADAPTER_CONTRACT),
        "authorization": dict(EXPECTED_AUTHORIZATION),
        "bound_evidence": dict(bound_evidence),
        "decision": "adapter_contract_ready_source_acquisition_and_canary_not_authorized",
        "inventory": validate_source_inventory(DEEPMIND_OBJECTS),
        "next_gate": dict(plan["next_gate"]),
        "observed_at": plan["observed_at"],
        "plan_file_sha256": _require_sha256(plan_file_sha256, "plan_file_sha256"),
        "production": {
            "candidate_registry_promotions": 0,
            "dataset_sealed": False,
            "generated_records": 0,
            "production_eligible": False,
            "production_sft_authorized": False,
        },
        "raw_training_text_present": False,
        "remaining_gap": {
            "projected_10m_floor_gap_after_dmath": int(
                bound_evidence["projected_10m_floor_gap_after_dmath"]
            ),
            "projected_reasoning_math_minimum_gap_after_dmath": math_gap,
            "token_yield_pilot_required_before_scale_count": True,
        },
        "schema_version": REPORT_SCHEMA_VERSION,
        "seed_contract_evidence": seed_contract_evidence(),
        "source": {
            "declared_license_id": SOURCE_LICENSE,
            "repository_url": SOURCE_REPOSITORY,
            "revision": DEEPMIND_REVISION,
            "source_id": SOURCE_ID,
        },
        "status": "verified_local_no_download_adapter_preflight",
    }
    report["report_sha256"] = sha256_value(report)
    return report


def validate_preflight_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported DeepMind adapter preflight report")
    if report.get("status") != "verified_local_no_download_adapter_preflight":
        raise ValueError("DeepMind adapter preflight status changed")
    if report.get("raw_training_text_present") is not False:
        raise ValueError("DeepMind adapter preflight report contains raw text")
    if report.get("production") != {
        "candidate_registry_promotions": 0,
        "dataset_sealed": False,
        "generated_records": 0,
        "production_eligible": False,
        "production_sft_authorized": False,
    }:
        raise ValueError("DeepMind adapter preflight production boundary changed")
    seal = _require_sha256(report.get("report_sha256"), "report_sha256")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if sha256_value(unsigned) != seal:
        raise ValueError("DeepMind adapter preflight report seal changed")
