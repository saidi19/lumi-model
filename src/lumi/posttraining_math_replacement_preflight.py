"""Fail-closed, no-download preflight for rights-clear math SFT replacements."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .posttraining_format import encode_sft_messages, load_chat_template

PLAN_SCHEMA_VERSION = "lumi.sft-math-replacement-preflight-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-math-replacement-preflight-report.v0.1"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-math-replacement-preflight-verification.v0.1"

DMATH_REVISION = "149a7403619ac1bd6ed10f50694ad82699d5cc03"
DEEPMIND_REVISION = "427f45075f84b8b9774950196ad63867ca20ffb3"

DMATH_OBJECTS = [
    {
        "git_blob_sha1": "cde816142c26a3c795a3dcb570a7132de77784bc",
        "kind": "license",
        "path": "LICENSE",
        "size_bytes": 1_066,
    },
    {
        "git_blob_sha1": "4e8b363e51a54a926cf4d9264e020a70e89b4ac2",
        "kind": "documentation",
        "path": "README.md",
        "size_bytes": 6_992,
    },
    {
        "git_blob_sha1": "be400482da3c53d61c1601ed701efba7d8e2efda",
        "kind": "data",
        "path": "dataset/train.json",
        "size_bytes": 12_198_829,
        "split": "train",
    },
]

_DEEPMIND_OBJECT_ROWS = [
    ("LICENSE", "d645695673349e3947e8e5ae42332d0ac3164cd7", 11_358),
    ("README.md", "d12161e88da78b17985457f1ff92da02c52a822e", 5_996),
    ("mathematics_dataset/__init__.py", "d780829a1f5101333577a1ca9adcb960529213f1", 595),
    ("mathematics_dataset/example.py", "44344698f2f1fa9c8e32893b8e00f5aee44542ef", 1_952),
    ("mathematics_dataset/generate.py", "41bdf658a0e09775c83d6fdefc9dd73a5639f99b", 5_654),
    (
        "mathematics_dataset/generate_settings.py",
        "29e0c69bf58dc8c15ac5e70b71afe031a7311b2f",
        1_136,
    ),
    (
        "mathematics_dataset/generate_to_file.py",
        "f8fd5ffcf79ca28695f45aba155c97c17236bb45",
        2_538,
    ),
    (
        "mathematics_dataset/modules/__init__.py",
        "d780829a1f5101333577a1ca9adcb960529213f1",
        595,
    ),
    (
        "mathematics_dataset/modules/algebra.py",
        "409a7a9fa992ec7f47dcdd54282d89da4394429e",
        12_823,
    ),
    (
        "mathematics_dataset/modules/arithmetic.py",
        "efacf6298c09a7f707bad0284b86462d22536994",
        18_960,
    ),
    (
        "mathematics_dataset/modules/calculus.py",
        "e1c363713b30b8052cdaa86fb3ce473fe3d2e111",
        6_895,
    ),
    (
        "mathematics_dataset/modules/comparison.py",
        "05f4a6d59820f19260eb046d99d8e465c5abdebc",
        15_015,
    ),
    (
        "mathematics_dataset/modules/measurement.py",
        "828e5b83b00b5d35a93e817aa35f462ec619268d",
        8_957,
    ),
    (
        "mathematics_dataset/modules/modules.py",
        "2e86478d5812f384f3559098bfde9b16b69e83a0",
        1_840,
    ),
    (
        "mathematics_dataset/modules/numbers.py",
        "7bab2c79da289f723b0a6d03e4269c5290789c97",
        16_985,
    ),
    (
        "mathematics_dataset/modules/polynomials.py",
        "eb46a96bad9159fc8089ddd5d39b49eed990802f",
        12_689,
    ),
    (
        "mathematics_dataset/modules/probability.py",
        "b69a0c3d1420f27db6d755b63b470e97e599aa29",
        8_670,
    ),
    (
        "mathematics_dataset/modules/train_test_split.py",
        "e85d3d991a480d0e8ef6cddd31bd0ed49aff9d3a",
        987,
    ),
    (
        "mathematics_dataset/sample/__init__.py",
        "d780829a1f5101333577a1ca9adcb960529213f1",
        595,
    ),
    (
        "mathematics_dataset/sample/arithmetic.py",
        "1620bcb13bdc26034c9ab829b7842afe60f17683",
        9_010,
    ),
    (
        "mathematics_dataset/sample/linear_system.py",
        "ff42349c797a44065b17ddf21fea8fc7a341745e",
        4_750,
    ),
    (
        "mathematics_dataset/sample/number.py",
        "0b024d9feaabaccab8116f16401a9c177116af48",
        5_071,
    ),
    (
        "mathematics_dataset/sample/ops.py",
        "1a4687b70da72cc10cde66a5d116d998a45541cb",
        12_565,
    ),
    (
        "mathematics_dataset/sample/polynomials.py",
        "4ccc0d93b4203c95e9fc5f51208ac5ca1c4178ea",
        20_367,
    ),
    (
        "mathematics_dataset/util/__init__.py",
        "d780829a1f5101333577a1ca9adcb960529213f1",
        595,
    ),
    (
        "mathematics_dataset/util/combinatorics.py",
        "92eb8288622c3fd9b6f9516619674e1b7db1c082",
        1_976,
    ),
    (
        "mathematics_dataset/util/composition.py",
        "7dbf779b8612356afa91bc7fbfafe3d964079082",
        19_332,
    ),
    (
        "mathematics_dataset/util/display.py",
        "c4732902404c7f9aa988ee280cebaef5301fcbc3",
        11_813,
    ),
    (
        "mathematics_dataset/util/probability.py",
        "3016d0523306bebdd9671a104be8c6da51a61170",
        12_495,
    ),
    ("setup.py", "150dd3f3599ea68a56767f1c669585a765e5d914", 2_458),
]
DEEPMIND_OBJECTS = [
    {
        "git_blob_sha1": sha1,
        "kind": "license" if path == "LICENSE" else "source_or_documentation",
        "path": path,
        "size_bytes": size,
    }
    for path, sha1, size in _DEEPMIND_OBJECT_ROWS
]

EXPECTED_SOURCES = [
    {
        "decision": "immediate_bounded_acquisition_candidate",
        "declared_license_id": "mit",
        "exact_objects": DMATH_OBJECTS,
        "expected_train_rows": 7_943,
        "official_test_split_policy": "reserved_for_evaluation_not_acquired",
        "repository_url": "https://github.com/JiwooKimAR/dmath",
        "revision": DMATH_REVISION,
        "source_id": "dmath",
    },
    {
        "decision": "fallback_scale_candidate_requires_adapter_preflight",
        "declared_license_id": "apache-2.0",
        "exact_object_bytes": 234_672,
        "exact_object_count": 30,
        "exact_object_inventory_sha256": (
            "47e472f97f7092cf264edb727bd0821db59869133f56b12cddf3f8a4ef1a6ce9"
        ),
        "generation_policy": (
            "source_only_fixed_seed_disjoint_splits_and_runtime_trace_binding_required"
        ),
        "pre_generated_bulk_data_authorized": False,
        "repository_url": "https://github.com/google-deepmind/mathematics_dataset",
        "revision": DEEPMIND_REVISION,
        "source_id": "deepmind_mathematics_dataset",
    },
    {
        "decision": "quarantined_pending_per_row_seed_rights_map",
        "declared_license_id": "mit",
        "repository_url": (
            "https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k"
        ),
        "revision": "c38ff6678fd7f1e7c52e94bc35ea453d36cbea57",
        "source_id": "orca_math_word_problems_200k",
    },
    {
        "decision": "excluded_under_current_allowed_license_policy",
        "declared_license_id": "nvidia-license",
        "repository_url": "https://huggingface.co/datasets/nvidia/OpenMathInstruct-1",
        "revision": "unbound_not_acquisition_candidate",
        "source_id": "nvidia_openmathinstruct_1",
    },
    {
        "decision": "reserved_for_evaluation_and_decontamination_only",
        "declared_license_id": "mit",
        "repository_url": "https://github.com/openai/grade-school-math",
        "revision": "not_acquired_in_this_gate",
        "source_id": "gsm8k",
    },
]

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
    "purpose": "local_raw_text_free_math_replacement_preflight_only",
    "unrelated_dataset_access_authorized": False,
}
EXPECTED_LIMITS = {
    "dmath_expected_train_rows": 7_943,
    "dmath_exact_object_bytes": 12_206_887,
    "dmath_exact_objects": 3,
    "deepmind_exact_object_bytes": 234_672,
    "deepmind_exact_objects": 30,
    "maximum_network_bytes": 0,
    "maximum_runtime_minutes": 60,
    "shortlisted_sources": 5,
}
EXPECTED_ACQUISITION_POLICY = {
    "deepmind_fallback_requires_separate_adapter_preflight": True,
    "exact_object_hashes_are_git_blob_sha1_until_fetch": True,
    "fetch_must_record_content_sha256": True,
    "immediate_source_id": "dmath",
    "one_source_at_a_time": True,
    "source_head_before_and_after_required": True,
    "test_or_benchmark_split_fetch_authorized": False,
}
EXPECTED_OUTPUT = {
    "preflight_report_path": "evidence/lumi-4b-sft-math-replacement-preflight-v0.1.json",
    "production_promotion_permitted": False,
    "raw_training_text_in_report": False,
    "verification_path": ("evidence/lumi-4b-sft-math-replacement-preflight-verification-v0.1.json"),
}
EXPECTED_BINDING_PATHS = {
    "ai_assisted_candidate": (
        "artifacts/ai-assisted-sft-pilot-v0.3/56ff1a9f093c/attempt-004/sft-filter-accepted.jsonl"
    ),
    "ai_assisted_owner_receipt": (
        "artifacts/ai-assisted-sft-pilot-v0.3/56ff1a9f093c/attempt-004/"
        "owner-review-receipt-v0.3-attempt-002.json"
    ),
    "aya_split_verification": (
        "../lumi-artifacts/4b-sft-aya-bengali-validation-materialization-v0.1/"
        "48b21e9f5b1c/attempt-001/verification-receipt.json"
    ),
    "chat_template": "configs/lumi-4b-chat-template-v0.1.json",
    "data_contract": "configs/lumi-4b-sft-data-contract-v0.1.json",
    "data_scale_gate": "configs/lumi-4b-sft-data-scale-gate-v0.1.json",
    "missing_mix_proposal": "configs/lumi-4b-sft-missing-mix-proposal-v0.1.json",
    "oasst1_promotion": "evidence/lumi-4b-sft-oasst1-source-promotion-v0.1.json",
    "openr1_rights_report": "evidence/lumi-4b-sft-openr1-rights-map-v0.1.json",
    "openr1_rights_verification": ("evidence/lumi-4b-sft-openr1-rights-map-verification-v0.1.json"),
    "preflight_checker": "scripts/check_math_replacement_preflight.py",
    "preflight_implementation": "src/lumi/posttraining_math_replacement_preflight.py",
    "preflight_verifier": "scripts/verify_math_replacement_preflight.py",
    "safety_promotion": "evidence/lumi-4b-safety-tool-source-promotion-v0.1.json",
    "safety_token_audit": "evidence/lumi-4b-safety-tool-sft-token-audit-v0.1.json",
    "source_registry": "configs/lumi-4b-sft-source-registry-v0.4.json",
    "tokenizer_config": (
        "../lumi-artifacts/4b-segment2-throughput-gate-v0.1/8bedfbc868e8/"
        "staging/input/tokenizer/tokenizer_config.json"
    ),
    "tokenizer_json": (
        "../lumi-artifacts/4b-segment2-throughput-gate-v0.1/8bedfbc868e8/"
        "staging/input/tokenizer/tokenizer.json"
    ),
}

EXPECTED_AI_V3_METRICS = {
    "by_split": {
        "train": {"records": 240, "supervised_tokens": 9_314},
        "validation": {"records": 80, "supervised_tokens": 2_138},
    },
    "train_by_task": {
        "conversation": 3_667,
        "multilingual_bengali": 4_329,
        "reasoning_math_science": 120,
        "tool_api": 1_198,
    },
}
EXPECTED_REGISTERED_TRAIN_TOKENS = {
    "aya_bengali": 520_269,
    "oasst1": 453_981,
    "safety_tool": 18_636,
}

NEXT_AUTHORIZATION_STATEMENT = (
    "I approve one bounded CPU-only DMath train-source acquisition and curation at "
    f"revision {DMATH_REVISION}, maximum 3 exact objects / 7,943 train rows / "
    "16 MiB network / 64 MiB decoded / 120 minutes. It may fetch only LICENSE, "
    "README.md, and dataset/train.json, write private raw data, a non-production "
    "candidate, attribution evidence, raw-text-free reports, and a deterministic "
    "32-row owner-review packet outside Git. Reserve the official test split; use "
    "question_en and answer_en, treat expression/Python solutions only as verifier "
    "inputs under a fail-closed sandbox, and run schema, PII/secret, deduplication, "
    "benchmark decontamination, 4K-fit, and frozen-tokenizer measurement. No external "
    "API, GPU, source-registry promotion, dataset seal, SFT training, unrelated data, "
    "or external distribution."
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")


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
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row must be an object: {path}")
        rows.append(value)
    return rows


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a SHA-256")
    return value


def _verify_internal_seal(value: Mapping[str, Any], field: str) -> str:
    seal = _require_sha256(value.get(field), field)
    payload = {key: item for key, item in value.items() if key != field}
    if sha256_value(payload) != seal:
        raise ValueError(f"{field} changed")
    return seal


def _validate_source_inventory() -> None:
    if len(DMATH_OBJECTS) != 3 or sum(item["size_bytes"] for item in DMATH_OBJECTS) != 12_206_887:
        raise ValueError("DMath exact object inventory changed")
    if (
        len(DEEPMIND_OBJECTS) != 30
        or sum(item["size_bytes"] for item in DEEPMIND_OBJECTS) != 234_672
    ):
        raise ValueError("DeepMind exact object inventory changed")
    for source_id, objects in (
        ("dmath", DMATH_OBJECTS),
        ("deepmind_mathematics_dataset", DEEPMIND_OBJECTS),
    ):
        if len({item["path"] for item in objects}) != len(objects):
            raise ValueError(f"duplicate exact object path: {source_id}")
        for item in objects:
            if _SHA1.fullmatch(str(item.get("git_blob_sha1"))) is None:
                raise ValueError(f"invalid git blob SHA-1: {source_id}")
            if not isinstance(item.get("size_bytes"), int) or item["size_bytes"] <= 0:
                raise ValueError(f"invalid exact object size: {source_id}")


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
        raise ValueError(f"math replacement binding path is invalid: {name}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError(f"math replacement binding escapes workspace: {name}")
    if not resolved.is_file():
        raise ValueError(f"math replacement binding is missing: {name}")
    expected = _require_sha256(binding.get("sha256"), f"bindings.{name}.sha256")
    if file_sha256(resolved) != expected:
        raise ValueError(f"math replacement binding changed: {name}")
    return resolved


def load_math_replacement_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    _validate_source_inventory()
    plan = _load_json(path)
    expected = {
        "acquisition_policy": EXPECTED_ACQUISITION_POLICY,
        "authorization": EXPECTED_AUTHORIZATION,
        "limits": EXPECTED_LIMITS,
        "observed_at": "2026-08-06",
        "output": EXPECTED_OUTPUT,
        "schema_version": PLAN_SCHEMA_VERSION,
        "shortlist": EXPECTED_SOURCES,
        "status": "local_no_download_preflight_authorized",
    }
    for field, value in expected.items():
        if plan.get(field) != value:
            raise ValueError(f"math replacement plan changed: {field}")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(EXPECTED_BINDING_PATHS):
        raise ValueError("math replacement binding set changed")
    for name, expected_path in EXPECTED_BINDING_PATHS.items():
        binding = bindings[name]
        if not isinstance(binding, Mapping) or binding.get("path") != expected_path:
            raise ValueError(f"math replacement binding path changed: {name}")
        resolve_binding(repository_root, binding, name=name)
    return plan


def measure_ai_assisted_v3(
    rows: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    template: Any,
) -> dict[str, Any]:
    record_ids: set[str] = set()
    split_records: Counter[str] = Counter()
    split_tokens: Counter[str] = Counter()
    train_task_tokens: Counter[str] = Counter()
    for row in rows:
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id or record_id in record_ids:
            raise ValueError("AI-assisted v0.3 record IDs changed")
        record_ids.add(record_id)
        split = row.get("split")
        task = row.get("task_category")
        messages = row.get("messages")
        if split not in {"train", "validation"} or not isinstance(task, str):
            raise ValueError("AI-assisted v0.3 split or task changed")
        if not isinstance(messages, list):
            raise ValueError("AI-assisted v0.3 messages changed")
        encoded = encode_sft_messages(messages, tokenizer=tokenizer, template=template)
        split_records[split] += 1
        split_tokens[split] += encoded.supervised_tokens
        if split == "train":
            train_task_tokens[task] += encoded.supervised_tokens
    return {
        "by_split": {
            split: {
                "records": split_records[split],
                "supervised_tokens": split_tokens[split],
            }
            for split in ("train", "validation")
        },
        "train_by_task": dict(sorted(train_task_tokens.items())),
    }


def _verify_candidate_receipts(
    *,
    ai_owner: Mapping[str, Any],
    aya_split: Mapping[str, Any],
    oasst1_promotion: Mapping[str, Any],
    openr1_report: Mapping[str, Any],
    openr1_verification: Mapping[str, Any],
    safety_audit: Mapping[str, Any],
    safety_promotion: Mapping[str, Any],
) -> dict[str, int]:
    _verify_internal_seal(ai_owner, "receipt_sha256")
    if ai_owner.get("status") != "owner_spot_check_passed_pilot_still_not_production_dataset":
        raise ValueError("AI-assisted v0.3 owner-review status changed")
    if ai_owner.get("accepted_records") != 32 or ai_owner.get("rejected_records") != 0:
        raise ValueError("AI-assisted v0.3 owner-review result changed")

    _verify_internal_seal(aya_split, "receipt_sha256")
    aya_train = aya_split.get("projection", {}).get("train", {})
    if aya_train != {
        "record_ids_sha256": "dcc6f7329c6e3eb08744326d52422ab1e8c408e5dfa1340680b03f07b5ad3bca",
        "records": 1_330,
        "supervised_tokens": 520_269,
    }:
        raise ValueError("Aya split train metrics changed")

    _verify_internal_seal(oasst1_promotion, "receipt_sha256")
    oasst_train = oasst1_promotion.get("curated_metrics", {}).get("by_split", {}).get("train")
    if oasst_train != {"records": 1_652, "supervised_tokens": 453_981}:
        raise ValueError("OASST1 promoted train metrics changed")

    _verify_internal_seal(safety_audit, "report_sha256")
    safety_train = safety_audit.get("metrics", {}).get("by_split", {}).get("train")
    if safety_train != {
        "masked_tokens": 10_867,
        "records": 240,
        "serialized_tokens": 29_503,
        "supervised_tokens": 18_636,
    }:
        raise ValueError("safety/tool train metrics changed")
    _verify_internal_seal(safety_promotion, "receipt_sha256")
    if (
        safety_promotion.get("candidate_source_registry_promotions") != 1
        or safety_promotion.get("production_dataset_promotions") != 0
    ):
        raise ValueError("safety/tool candidate promotion boundary changed")

    _verify_internal_seal(openr1_report, "report_sha256")
    _verify_internal_seal(openr1_verification, "verification_sha256")
    if (
        openr1_report.get("source_audit", {}).get("rights_cleared_records") != 0
        or openr1_verification.get("promotion_eligible_records") != 0
        or openr1_verification.get("unresolved_records") != 2_656
    ):
        raise ValueError("OpenR1 rights quarantine boundary changed")
    return dict(EXPECTED_REGISTERED_TRAIN_TOKENS)


def _verify_mix_contracts(
    *,
    data_contract: Mapping[str, Any],
    missing_mix: Mapping[str, Any],
) -> dict[str, Any]:
    allowed_licenses = data_contract.get("allowed_licenses")
    if not isinstance(allowed_licenses, list) or "mit" not in allowed_licenses:
        raise ValueError("current SFT license policy no longer allows MIT")
    if "nvidia-license" in allowed_licenses:
        raise ValueError("NVIDIA license unexpectedly entered the allowed license set")
    learning = missing_mix.get("learning_candidate")
    if not isinstance(learning, Mapping):
        raise ValueError("learning-candidate mix contract changed")
    total = learning.get("accepted_train_supervised_tokens")
    math = learning.get("task_targets", {}).get("reasoning_math_science")
    if total != {"maximum": 30_000_000, "minimum": 10_000_000}:
        raise ValueError("learning-candidate total token target changed")
    if math != {"fraction": 0.22, "maximum_tokens": 6_600_000, "minimum_tokens": 2_200_000}:
        raise ValueError("reasoning/math token target changed")
    if missing_mix.get("diversity_policy", {}).get("maximum_fraction_from_one_source") != 0.35:
        raise ValueError("source concentration limit changed")
    return {
        "maximum_source_fraction": 0.35,
        "reasoning_math_science": {"maximum": 6_600_000, "minimum": 2_200_000},
        "total": {"maximum": 30_000_000, "minimum": 10_000_000},
    }


def rebuild_math_replacement_preflight(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    root = Path(repository_root).resolve()
    bindings = plan["bindings"]
    resolved = {
        name: resolve_binding(root, binding, name=name) for name, binding in bindings.items()
    }
    template = load_chat_template(resolved["chat_template"])
    if resolved["tokenizer_json"].parent != resolved["tokenizer_config"].parent:
        raise ValueError("frozen tokenizer files are not colocated")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer = AutoTokenizer.from_pretrained(
        resolved["tokenizer_json"].parent,
        local_files_only=True,
    )
    ai_metrics = measure_ai_assisted_v3(
        _load_jsonl(resolved["ai_assisted_candidate"]),
        tokenizer=tokenizer,
        template=template,
    )
    if ai_metrics != EXPECTED_AI_V3_METRICS:
        raise ValueError("AI-assisted v0.3 frozen-token measurement changed")
    registered = _verify_candidate_receipts(
        ai_owner=_load_json(resolved["ai_assisted_owner_receipt"]),
        aya_split=_load_json(resolved["aya_split_verification"]),
        oasst1_promotion=_load_json(resolved["oasst1_promotion"]),
        openr1_report=_load_json(resolved["openr1_rights_report"]),
        openr1_verification=_load_json(resolved["openr1_rights_verification"]),
        safety_audit=_load_json(resolved["safety_token_audit"]),
        safety_promotion=_load_json(resolved["safety_promotion"]),
    )
    targets = _verify_mix_contracts(
        data_contract=_load_json(resolved["data_contract"]),
        missing_mix=_load_json(resolved["missing_mix_proposal"]),
    )
    registered_total = sum(registered.values())
    reviewed_unregistered = ai_metrics["by_split"]["train"]["supervised_tokens"]
    planning_total = registered_total + reviewed_unregistered
    reviewed_math = ai_metrics["train_by_task"]["reasoning_math_science"]
    source_summaries = []
    for source in plan["shortlist"]:
        exact_objects = source.get("exact_objects", [])
        source_summaries.append(
            {
                "decision": source["decision"],
                "declared_license_id": source["declared_license_id"],
                "exact_object_bytes": source.get(
                    "exact_object_bytes",
                    sum(item["size_bytes"] for item in exact_objects),
                ),
                "exact_object_count": source.get(
                    "exact_object_count",
                    len(exact_objects),
                ),
                "exact_object_inventory_sha256": source.get(
                    "exact_object_inventory_sha256",
                    sha256_value(exact_objects),
                ),
                "revision": source["revision"],
                "source_id": source["source_id"],
            }
        )
    report: dict[str, Any] = {
        "acquisition_authorized": False,
        "authorization": dict(plan["authorization"]),
        "decision": "dmath_is_next_single_source_gate_deepmind_is_later_fallback",
        "mix_gap": {
            "learning_candidate_floor_gap_after_reviewed_draft": (
                targets["total"]["minimum"] - planning_total
            ),
            "planning_train_supervised_tokens": planning_total,
            "reasoning_math_maximum_gap_after_reviewed_draft": (
                targets["reasoning_math_science"]["maximum"] - reviewed_math
            ),
            "reasoning_math_minimum_gap_after_reviewed_draft": (
                targets["reasoning_math_science"]["minimum"] - reviewed_math
            ),
            "registered_reviewed_train_supervised_tokens": registered_total,
            "reviewed_unregistered_ai_v3_train_supervised_tokens": reviewed_unregistered,
        },
        "next_gate": {
            "authorization_required": True,
            "exact_authorization_statement": NEXT_AUTHORIZATION_STATEMENT,
            "source_id": "dmath",
        },
        "observed_at": plan["observed_at"],
        "plan_file_sha256": _require_sha256(plan_file_sha256, "plan_file_sha256"),
        "production_dataset_promotions": 0,
        "production_eligible": False,
        "raw_training_text_retained": False,
        "shortlist_summary": source_summaries,
        "source_token_evidence": {
            "ai_assisted_v0_3": ai_metrics,
            "openr1_rights_clear_train_supervised_tokens": 0,
            "registered_train_supervised_tokens": registered,
        },
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "verified_local_preflight_no_download_no_promotion",
        "target_contract": targets,
    }
    report["report_sha256"] = sha256_value(report)
    return report


def validate_math_replacement_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported math replacement preflight report")
    if report.get("status") != "verified_local_preflight_no_download_no_promotion":
        raise ValueError("math replacement preflight status changed")
    if report.get("acquisition_authorized") is not False:
        raise ValueError("math replacement preflight cannot authorize acquisition")
    if report.get("production_eligible") is not False:
        raise ValueError("math replacement preflight cannot establish production eligibility")
    if report.get("production_dataset_promotions") != 0:
        raise ValueError("math replacement preflight cannot promote records")
    if report.get("raw_training_text_retained") is not False:
        raise ValueError("math replacement preflight must remain raw-text-free")
    gap = report.get("mix_gap")
    if gap != {
        "learning_candidate_floor_gap_after_reviewed_draft": 8_997_800,
        "planning_train_supervised_tokens": 1_002_200,
        "reasoning_math_maximum_gap_after_reviewed_draft": 6_599_880,
        "reasoning_math_minimum_gap_after_reviewed_draft": 2_199_880,
        "registered_reviewed_train_supervised_tokens": 992_886,
        "reviewed_unregistered_ai_v3_train_supervised_tokens": 9_314,
    }:
        raise ValueError("math replacement token gap changed")
    if report.get("next_gate", {}).get("exact_authorization_statement") != (
        NEXT_AUTHORIZATION_STATEMENT
    ):
        raise ValueError("math replacement next authorization changed")
    seal = _require_sha256(report.get("report_sha256"), "report_sha256")
    payload = {key: item for key, item in report.items() if key != "report_sha256"}
    if sha256_value(payload) != seal:
        raise ValueError("math replacement preflight report seal mismatch")
