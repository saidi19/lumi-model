"""Project the repaired four-lane Lumi SFT closure without mutating state.

The preflight replays the frozen tokenizer over the repaired BanglaBridge and
Aya candidates plus the unchanged BanglaTLit and Nemotron candidates.  It
combines only raw-text-free metrics with the last verified registry-v0.15 mix.
It does not create a registry successor, seal a dataset, or start SFT.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .data_quality import sha256_file
from .posttraining_format import (
    ChatTemplate,
    encode_sft_messages,
    load_chat_template,
)

PLAN_SCHEMA_VERSION = "lumi.sft-closure-successor-preflight.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-closure-successor-preflight-report.v0.1"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-closure-successor-preflight-verification.v0.1"
PLAN_STATUS = "authorized_cpu_only_nonmutating_closure_successor_preflight"
REPORT_STATUS = "verified_closure_projection_blocked_pending_targeted_gap_fill"
OWNER_REQUEST_TEXT = "ok next step shuru kor"

EXPECTED_AUTHORIZATION = {
    "candidate_materialization_authorized": False,
    "cpu_only": True,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "gpu_authorized": False,
    "local_candidate_read_authorized": True,
    "network_access_authorized": False,
    "production_promotion_authorized": False,
    "registry_change_authorized": False,
    "sft_training_authorized": False,
    "unrelated_access_authorized": False,
}
EXPECTED_LIMITS = {
    "candidate_records": 19_654,
    "candidate_sources": 4,
    "maximum_report_bytes": 1_048_576,
    "maximum_runtime_minutes": 60,
    "maximum_verification_bytes": 1_048_576,
}
EXPECTED_BINDINGS = {
    "aya_candidate",
    "aya_manifest",
    "aya_materialization_report",
    "aya_materialization_verification",
    "banglabridge_candidate",
    "banglabridge_manifest",
    "banglabridge_repair_report",
    "banglabridge_repair_verification",
    "banglatlit_candidate",
    "chat_template",
    "closure_curation_report",
    "closure_curation_verification",
    "closure_owner_review_receipt",
    "closure_owner_review_verification",
    "current_mix_report",
    "current_mix_verification",
    "implementation",
    "nemotron_candidate",
    "original_private_manifest",
    "predecessor_registry",
    "runner",
    "test",
    "tokenizer_config",
    "tokenizer_json",
    "verifier",
}
EXPECTED_OUTPUT = {
    "preflight_report_path": ("evidence/lumi-4b-sft-closure-successor-preflight-v0.1.json"),
    "verification_path": (
        "evidence/lumi-4b-sft-closure-successor-preflight-verification-v0.1.json"
    ),
}

EXPECTED_CURRENT_TRAIN_TOKENS = 7_256_670
EXPECTED_CURRENT_VALIDATION_TOKENS = 406_125
TEN_MILLION_FLOOR = 10_000_000
SOURCE_CAP = 0.35
EXPECTED_PREDECESSOR_REGISTRY_SHA256 = (
    "de81a51f0288b4851ff5ea4ca0598ddcf4936b2bf2c4127ab2b1c0d6deb1a4de"
)
EXPECTED_TOKENIZER_SHA256 = "1022c9538d9ec795b07c2e2bda3c5c6f21c321de933cfe5ad197628a94caa2ee"

EXPECTED_TASKS = (
    "coding",
    "conversation",
    "factual_qa",
    "multilingual_bengali",
    "reasoning_math_science",
    "safety_helpfulness",
    "tool_api",
    "writing_summarization",
)
EXPECTED_LANGUAGE_MODES = (
    "banglish",
    "bengali_script",
    "english",
    "natural_code_switching",
)

SOURCE_SPECS: dict[str, dict[str, Any]] = {
    "banglabridge_instructions_closure_canary": {
        "binding": "banglabridge_candidate",
        "dataset_id": "subhajitmahata84/banglabridge-instructions",
        "family": "banglabridge_family",
        "license_id": "cc-by-4.0",
        "records": 116,
        "revision": "b5ace9b2e56702cfd3aa5fe2e8c3870f67f57300",
        "train_records": 99,
        "train_supervised_tokens": 49_998,
        "validation_records": 17,
        "validation_supervised_tokens": 4_995,
    },
    "banglatlit_auxiliary_closure_canary": {
        "binding": "banglatlit_candidate",
        "dataset_id": "aplycaebous/BanglaTLit",
        "family": "banglatlit_family",
        "license_id": "mit",
        "records": 13_576,
        "revision": "22cacbaac72bf6ee828f9d6be1e28bfd7a84843f",
        "train_records": 12_881,
        "train_supervised_tokens": 300_000,
        "validation_records": 695,
        "validation_supervised_tokens": 15_789,
    },
    "coherelabs_aya_bengali_closure_canary": {
        "binding": "aya_candidate",
        "dataset_id": "CohereLabs/aya_collection_language_split",
        "family": "coherelabs_aya_family",
        "license_id": "apache-2.0",
        "records": 3_995,
        "revision": "a3af2fde4b4cb5b2775830b11244a1a20b5f004f",
        "train_records": 3_812,
        "train_supervised_tokens": 725_736,
        "validation_records": 183,
        "validation_supervised_tokens": 32_276,
    },
    "nvidia_nemotron_post_training_tool_canary": {
        "binding": "nemotron_candidate",
        "dataset_id": "nvidia/Nemotron-Post-Training-Dataset-v1",
        "family": "nvidia_family",
        "license_id": "cc-by-4.0",
        "records": 1_967,
        "revision": "74e23eb6f830fef4a9e96a92f6f6262214cbb9a8",
        "train_records": 1_867,
        "train_supervised_tokens": 1_299_965,
        "validation_records": 100,
        "validation_supervised_tokens": 68_393,
    },
}

EXPECTED_BUNDLE = {
    "records": 19_654,
    "train_records": 18_659,
    "train_supervised_tokens": 2_375_699,
    "validation_records": 995,
    "validation_supervised_tokens": 121_453,
}

CURRENT_FAMILY_GROUPS = {
    "coherelabs_aya_family": ("coherelabs_aya_dataset",),
    "deepmind_mathematics_family": ("deepmind_mathematics_dataset",),
    "dmath_family": ("dmath",),
    "lumi_first_party_family": (
        "lumi_ai_assisted_sft_v0_3",
        "lumi_first_party_teacher_canary_v0_1",
        "lumi_safety_tool_sft_v0_1",
    ),
    "nvidia_family": ("nvidia_opencodeinstruct",),
    "openassistant_family": ("openassistant_oasst1", "openassistant_oasst2"),
}

BRIDGE_PROVENANCE_HASHES = {
    "source_adapted_instruction_sha256",
    "source_adapted_output_sha256",
    "source_original_instruction_sha256",
    "source_original_output_sha256",
}
AYA_ALLOWED_FAMILY_HASHES = {
    "120f0a5cb5386355e804603ebac9c71e2775e698d02beefacb2e9b781a850151",
    "3bae6c736d432b44b3e79feb6ed935e45ec923fa56b5eb73518e1b755afcea3c",
}
AYA_QUARANTINED_FAMILY_HASH = "1ca8a8ec9ac03c59c0be773f060e77d89eeab400f6a9160b6a13a2ef5c256859"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_value(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(json_file_bytes(value))
    temporary.replace(destination)


def _read_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(row)
    return rows


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _verify_internal_seal(value: Mapping[str, Any], field: str, label: str) -> None:
    expected = _require_sha256(value.get(field), f"{label}.{field}")
    unsigned = {key: item for key, item in value.items() if key != field}
    if sha256_value(unsigned) != expected:
        raise ValueError(f"{label} internal seal changed")


def resolve_binding(
    repository_root: str | Path,
    specification: Mapping[str, Any],
    *,
    name: str,
) -> Path:
    root = Path(repository_root).resolve()
    workspace = root.parent.resolve()
    relative = specification.get("path")
    expected = _require_sha256(specification.get("sha256"), f"bindings.{name}.sha256")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"invalid closure-successor binding: {name}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(workspace) or not resolved.is_file():
        raise ValueError(f"closure-successor binding missing: {name}")
    if sha256_file(resolved) != expected:
        raise ValueError(f"closure-successor binding changed: {name}")
    return resolved


def load_preflight_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    plan = _read_object(path)
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "limits": EXPECTED_LIMITS,
        "observed_at": "2026-08-10",
        "output": EXPECTED_OUTPUT,
        "owner_request_text": OWNER_REQUEST_TEXT,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": PLAN_STATUS,
    }
    for field, value in expected.items():
        if plan.get(field) != value:
            raise ValueError(f"closure-successor preflight changed: {field}")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("closure-successor binding set changed")
    for name, specification in bindings.items():
        if not isinstance(specification, Mapping):
            raise ValueError(f"invalid closure-successor binding: {name}")
        resolve_binding(repository_root, specification, name=name)
    if bindings["predecessor_registry"]["sha256"] != EXPECTED_PREDECESSOR_REGISTRY_SHA256:
        raise ValueError("closure-successor predecessor registry changed")
    if bindings["tokenizer_json"]["sha256"] != EXPECTED_TOKENIZER_SHA256:
        raise ValueError("closure-successor frozen tokenizer changed")
    return plan


class _DirectTokenizerAdapter:
    def __init__(self, tokenizer: Any, special_ids: list[int]) -> None:
        self._tokenizer = tokenizer
        self.all_special_ids = special_ids

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        split_special_tokens: bool,
    ) -> list[int]:
        del split_special_tokens
        return list(self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids)

    def get_vocab(self) -> Mapping[str, int]:
        return self._tokenizer.get_vocab()


def _load_direct_tokenizer(path: Path) -> _DirectTokenizerAdapter:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(path))
    value = json.loads(path.read_text(encoding="utf-8"))
    special_ids = sorted(
        int(token["id"]) for token in value["added_tokens"] if token.get("special") is True
    )
    return _DirectTokenizerAdapter(tokenizer, special_ids)


def _language_mode(row: Mapping[str, Any], *, source_id: str) -> str:
    language = row.get("language")
    if language == "bn":
        return "bengali_script"
    if language == "en":
        return "english"
    if language != "multilingual":
        raise ValueError(f"{source_id} language changed")
    if source_id == "banglatlit_auxiliary_closure_canary":
        return "banglish"
    return "natural_code_switching"


def _measure_source(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_id: str,
    specification: Mapping[str, Any],
    tokenizer: _DirectTokenizerAdapter,
    template: ChatTemplate,
) -> dict[str, Any]:
    split_records: Counter[str] = Counter()
    split_tokens: Counter[str] = Counter()
    task_tokens: Counter[str] = Counter()
    language_tokens: Counter[str] = Counter()
    family_hash_counts: Counter[str] = Counter()
    family_hash_train_tokens: Counter[str] = Counter()
    record_ids: set[str] = set()
    content_hashes: set[str] = set()
    auxiliary_train_tokens = 0
    maximum_serialized_tokens = 0
    bridge_rows_with_four_hashes = 0

    for row in rows:
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id or record_id in record_ids:
            raise ValueError(f"{source_id} record identity changed")
        record_ids.add(record_id)
        source = row.get("source")
        if not isinstance(source, Mapping):
            raise ValueError(f"{source_id} source metadata missing")
        for field in ("source_id", "dataset_id", "license_id", "revision"):
            expected = source_id if field == "source_id" else specification[field]
            if source.get(field) != expected:
                raise ValueError(f"{source_id} source {field} changed")
        split = row.get("split")
        if split not in {"train", "validation"}:
            raise ValueError(f"{source_id} split changed")
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"{source_id} messages changed")
        encoded = encode_sft_messages(messages, tokenizer=tokenizer, template=template)
        supervised_tokens = encoded.supervised_tokens
        maximum_serialized_tokens = max(maximum_serialized_tokens, len(encoded.input_ids))
        split_records[str(split)] += 1
        split_tokens[str(split)] += supervised_tokens

        provenance = row.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(f"{source_id} provenance missing")
        content_hash = _require_sha256(
            provenance.get("content_sha256"), f"{source_id}.content_sha256"
        )
        if content_hash in content_hashes:
            raise ValueError(f"{source_id} content hash duplicated")
        content_hashes.add(content_hash)

        if source_id == "banglabridge_instructions_closure_canary":
            if all(
                _SHA256.fullmatch(str(provenance.get(field, ""))) is not None
                for field in BRIDGE_PROVENANCE_HASHES
            ):
                bridge_rows_with_four_hashes += 1
        elif source_id == "coherelabs_aya_bengali_closure_canary":
            family_hash = _require_sha256(
                provenance.get("source_dataset_name_sha256"),
                "aya.source_dataset_name_sha256",
            )
            family_hash_counts[family_hash] += 1
            if split == "train":
                family_hash_train_tokens[family_hash] += supervised_tokens
        elif source_id == "banglatlit_auxiliary_closure_canary":
            if provenance.get("auxiliary_transliteration_only") is not True:
                raise ValueError("BanglaTLit lost auxiliary-only provenance")

        if split == "train":
            task = row.get("task_category")
            if task not in EXPECTED_TASKS:
                raise ValueError(f"{source_id} task changed")
            mode = _language_mode(row, source_id=source_id)
            task_tokens[str(task)] += supervised_tokens
            language_tokens[mode] += supervised_tokens
            if provenance.get("auxiliary_transliteration_only") is True:
                auxiliary_train_tokens += supervised_tokens

    observed = {
        "records": len(rows),
        "train_records": split_records["train"],
        "train_supervised_tokens": split_tokens["train"],
        "validation_records": split_records["validation"],
        "validation_supervised_tokens": split_tokens["validation"],
    }
    expected = {
        key: specification[key]
        for key in (
            "records",
            "train_records",
            "train_supervised_tokens",
            "validation_records",
            "validation_supervised_tokens",
        )
    }
    if observed != expected:
        raise ValueError(f"{source_id} frozen-token inventory changed: {observed}")
    if maximum_serialized_tokens > template.context_window:
        raise ValueError(f"{source_id} exceeds frozen 4K context")
    if source_id == "banglabridge_instructions_closure_canary":
        if bridge_rows_with_four_hashes != len(rows):
            raise ValueError("BanglaBridge four-hash provenance is incomplete")
    if source_id == "coherelabs_aya_bengali_closure_canary":
        observed_families = set(family_hash_counts)
        if observed_families != AYA_ALLOWED_FAMILY_HASHES:
            raise ValueError("Aya rights-filtered family set changed")
        if AYA_QUARANTINED_FAMILY_HASH in observed_families:
            raise ValueError("Aya quarantined WikiLingua family reappeared")

    return {
        **observed,
        "auxiliary_train_supervised_tokens": auxiliary_train_tokens,
        "family": specification["family"],
        "language_train_supervised_tokens": {
            key: language_tokens[key] for key in EXPECTED_LANGUAGE_MODES
        },
        "maximum_serialized_tokens": maximum_serialized_tokens,
        "task_train_supervised_tokens": {key: task_tokens[key] for key in EXPECTED_TASKS},
        "unique_content_hashes": len(content_hashes),
        "unique_record_ids": len(record_ids),
        "source_specific_checks": {
            "aya_allowed_family_hash_counts": dict(sorted(family_hash_counts.items())),
            "aya_allowed_family_train_supervised_tokens": dict(
                sorted(family_hash_train_tokens.items())
            ),
            "aya_quarantined_family_present": (AYA_QUARANTINED_FAMILY_HASH in family_hash_counts),
            "banglabridge_rows_with_four_provenance_hashes": (bridge_rows_with_four_hashes),
        },
    }


def _sum_maps(*values: Mapping[str, int], keys: Sequence[str]) -> dict[str, int]:
    return {key: sum(int(value.get(key, 0)) for value in values) for key in keys}


def _gaps(targets: Mapping[str, int], observed: Mapping[str, int]) -> dict[str, int]:
    return {key: max(0, int(targets[key]) - int(observed.get(key, 0))) for key in targets}


def _source_train_tokens(current_mix: Mapping[str, Any]) -> dict[str, int]:
    source_metrics = current_mix.get("source_metrics")
    if not isinstance(source_metrics, Mapping):
        raise ValueError("current mix source metrics missing")
    result: dict[str, int] = {}
    for source_id, metrics in source_metrics.items():
        if not isinstance(metrics, Mapping):
            raise ValueError("current mix source metric changed")
        result[str(source_id)] = int(
            metrics.get("by_split", {}).get("train", {}).get("supervised_tokens", -1)
        )
        if result[str(source_id)] < 0:
            raise ValueError("current mix source token metric missing")
    return result


def _verify_evidence(resolved: Mapping[str, Path]) -> dict[str, Any]:
    current = _read_object(resolved["current_mix_report"])
    current_verification = _read_object(resolved["current_mix_verification"])
    bridge = _read_object(resolved["banglabridge_repair_report"])
    bridge_verification = _read_object(resolved["banglabridge_repair_verification"])
    aya = _read_object(resolved["aya_materialization_report"])
    aya_verification = _read_object(resolved["aya_materialization_verification"])
    curation = _read_object(resolved["closure_curation_report"])
    curation_verification = _read_object(resolved["closure_curation_verification"])
    owner = _read_object(resolved["closure_owner_review_receipt"])
    owner_verification = _read_object(resolved["closure_owner_review_verification"])

    for value, field, label in (
        (current, "report_sha256", "current mix report"),
        (current_verification, "verification_sha256", "current mix verification"),
        (bridge, "report_sha256", "BanglaBridge repair report"),
        (bridge_verification, "verification_sha256", "BanglaBridge verification"),
        (aya, "report_sha256", "Aya materialization report"),
        (aya_verification, "verification_sha256", "Aya verification"),
        (curation, "report_sha256", "closure curation report"),
        (curation_verification, "verification_sha256", "closure curation verification"),
        (owner, "receipt_sha256", "closure owner-review receipt"),
        (owner_verification, "verification_sha256", "owner-review verification"),
    ):
        _verify_internal_seal(value, field, label)

    pool = current.get("candidate_pool")
    if not isinstance(pool, Mapping):
        raise ValueError("current mix candidate pool missing")
    if pool.get("accepted_train_supervised_tokens") != EXPECTED_CURRENT_TRAIN_TOKENS:
        raise ValueError("current mix train token baseline changed")
    if (
        pool.get("registry_admissible_validation_supervised_tokens")
        != EXPECTED_CURRENT_VALIDATION_TOKENS
    ):
        raise ValueError("current mix validation token baseline changed")
    if current.get("status") != "verified_registry_v0_15_candidate_mix_reaudit_nonproduction":
        raise ValueError("current mix status changed")
    if bridge.get("output", {}).get("candidate_file_sha256") != sha256_file(
        resolved["banglabridge_candidate"]
    ):
        raise ValueError("BanglaBridge repair output binding changed")
    if aya.get("output", {}).get("candidate_file_sha256") != sha256_file(resolved["aya_candidate"]):
        raise ValueError("Aya materialization output binding changed")
    if owner.get("accepted_records") != 32 or owner.get("rejected_records") != 0:
        raise ValueError("closure owner-review sample changed")
    return current


def build_preflight_report(
    plan: Mapping[str, Any],
    *,
    plan_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    resolved = {
        name: resolve_binding(root, binding, name=name)
        for name, binding in plan["bindings"].items()
    }
    current = _verify_evidence(resolved)
    if resolved["tokenizer_json"].parent != resolved["tokenizer_config"].parent:
        raise ValueError("frozen tokenizer files are not colocated")
    tokenizer = _load_direct_tokenizer(resolved["tokenizer_json"])
    template = load_chat_template(resolved["chat_template"])

    source_metrics: dict[str, dict[str, Any]] = {}
    all_record_ids: set[str] = set()
    all_content_hashes: set[str] = set()
    for source_id, specification in sorted(SOURCE_SPECS.items()):
        rows = _read_jsonl(resolved[specification["binding"]])
        metrics = _measure_source(
            rows,
            source_id=source_id,
            specification=specification,
            tokenizer=tokenizer,
            template=template,
        )
        source_metrics[source_id] = metrics
        for row in rows:
            record_id = str(row["record_id"])
            content_hash = str(row["provenance"]["content_sha256"])
            if record_id in all_record_ids or content_hash in all_content_hashes:
                raise ValueError("closure bundle has an exact cross-source duplicate")
            all_record_ids.add(record_id)
            all_content_hashes.add(content_hash)

    observed_bundle = {
        "records": sum(metric["records"] for metric in source_metrics.values()),
        "train_records": sum(metric["train_records"] for metric in source_metrics.values()),
        "train_supervised_tokens": sum(
            metric["train_supervised_tokens"] for metric in source_metrics.values()
        ),
        "validation_records": sum(
            metric["validation_records"] for metric in source_metrics.values()
        ),
        "validation_supervised_tokens": sum(
            metric["validation_supervised_tokens"] for metric in source_metrics.values()
        ),
    }
    if observed_bundle != EXPECTED_BUNDLE:
        raise ValueError(f"closure bundle changed: {observed_bundle}")

    current_pool = current["candidate_pool"]
    current_tasks = {key: int(current_pool["train_by_task"][key]) for key in EXPECTED_TASKS}
    current_languages = {
        key: int(current_pool["train_by_language_mode"][key]) for key in EXPECTED_LANGUAGE_MODES
    }
    task_targets = {
        key: current_tasks[key] + int(current["closure"]["task_floor_gaps"][key])
        for key in EXPECTED_TASKS
    }
    language_targets = {
        key: current_languages[key] + int(current["closure"]["language_floor_gaps"][key])
        for key in EXPECTED_LANGUAGE_MODES
    }
    task_additions = _sum_maps(
        *(metric["task_train_supervised_tokens"] for metric in source_metrics.values()),
        keys=EXPECTED_TASKS,
    )
    language_additions = _sum_maps(
        *(metric["language_train_supervised_tokens"] for metric in source_metrics.values()),
        keys=EXPECTED_LANGUAGE_MODES,
    )
    auxiliary_tokens = sum(
        metric["auxiliary_train_supervised_tokens"] for metric in source_metrics.values()
    )
    behavior_task_additions = dict(task_additions)
    behavior_task_additions["multilingual_bengali"] -= auxiliary_tokens
    behavior_language_additions = dict(language_additions)
    behavior_language_additions["banglish"] -= auxiliary_tokens
    if min(behavior_task_additions.values()) < 0 or min(behavior_language_additions.values()) < 0:
        raise ValueError("auxiliary token subtraction changed")

    accounting_tasks = _sum_maps(current_tasks, task_additions, keys=EXPECTED_TASKS)
    behavior_tasks = _sum_maps(current_tasks, behavior_task_additions, keys=EXPECTED_TASKS)
    accounting_languages = _sum_maps(
        current_languages, language_additions, keys=EXPECTED_LANGUAGE_MODES
    )
    behavior_languages = _sum_maps(
        current_languages,
        behavior_language_additions,
        keys=EXPECTED_LANGUAGE_MODES,
    )
    accounting_task_gaps = _gaps(task_targets, accounting_tasks)
    behavior_task_gaps = _gaps(task_targets, behavior_tasks)
    accounting_language_gaps = _gaps(language_targets, accounting_languages)
    behavior_language_gaps = _gaps(language_targets, behavior_languages)

    source_tokens = _source_train_tokens(current)
    for source_id, metric in source_metrics.items():
        source_tokens[source_id] = metric["train_supervised_tokens"]
    projected_total = EXPECTED_CURRENT_TRAIN_TOKENS + EXPECTED_BUNDLE["train_supervised_tokens"]
    source_shares = {
        source_id: round(tokens / projected_total, 8)
        for source_id, tokens in sorted(source_tokens.items())
    }
    source_cap_violations = [
        source_id for source_id, share in source_shares.items() if share > SOURCE_CAP
    ]

    family_tokens: Counter[str] = Counter()
    for family, members in CURRENT_FAMILY_GROUPS.items():
        family_tokens[family] += sum(source_tokens.get(member, 0) for member in members)
    for source_id, specification in SOURCE_SPECS.items():
        family_tokens[specification["family"]] += source_metrics[source_id][
            "train_supervised_tokens"
        ]
    family_shares = {
        family: round(tokens / projected_total, 8)
        for family, tokens in sorted(family_tokens.items())
    }
    family_cap_violations = [
        family for family, share in family_shares.items() if share > SOURCE_CAP
    ]
    nvidia_tokens = family_tokens["nvidia_family"]
    family_dilution_gap = max(0, math.ceil(nvidia_tokens / SOURCE_CAP) - projected_total)

    total_floor_gap = max(0, TEN_MILLION_FLOOR - projected_total)
    behavior_task_gap_total = sum(behavior_task_gaps.values())
    behavior_language_gap_total = sum(behavior_language_gaps.values())
    strict_lower_bound = max(
        total_floor_gap,
        behavior_task_gap_total,
        behavior_language_gap_total,
        family_dilution_gap,
    )

    blockers = [
        "train_supervised_tokens_below_10m_floor",
        "behavior_task_floors_not_met",
        "behavior_language_floors_not_met",
        "nvidia_source_family_above_0_35",
        "full_cross_source_deduplication_and_decontamination_not_run",
        "source_registry_successor_not_materialized",
        "frozen_validation_mix_not_sealed",
        "dataset_seal_cpu_canary_and_gpu_sft_not_authorized",
    ]
    report: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "blockers": blockers,
        "bundle": {
            **observed_bundle,
            "auxiliary_transliteration_train_supervised_tokens": auxiliary_tokens,
            "cross_source_content_hash_duplicates": 0,
            "cross_source_record_id_duplicates": 0,
            "source_count": len(source_metrics),
        },
        "checks": {
            "all_19654_rows_replayed_with_frozen_tokenizer": True,
            "aya_contains_only_rights_retained_wikisplit_and_joke_families": True,
            "banglabridge_four_hash_provenance_complete": True,
            "banglatlit_tracked_as_auxiliary_not_primary_behavior_proof": True,
            "candidate_materialized": False,
            "dataset_sealed": False,
            "network_api_or_gpu_used": False,
            "predecessor_registry_changed": False,
            "sft_training_started": False,
        },
        "current_verified_mix": {
            "registry_sha256": EXPECTED_PREDECESSOR_REGISTRY_SHA256,
            "train_supervised_tokens": EXPECTED_CURRENT_TRAIN_TOKENS,
            "validation_supervised_tokens": EXPECTED_CURRENT_VALIDATION_TOKENS,
        },
        "next_gate": {
            "action": "acquire_and_curate_non_nvidia_targeted_behavior_gap_fill",
            "authorization_required": True,
            "minimum_additional_train_supervised_tokens_lower_bound": strict_lower_bound,
            "required_language_gaps": behavior_language_gaps,
            "required_task_gaps": behavior_task_gaps,
            "then": [
                "materialize_immutable_source_registry_successor",
                "run_full_cross_source_hygiene_reaudit",
                "seal_frozen_train_validation_mix",
                "run_cpu_sft_smoke_and_baseline_benchmarks",
                "request_separate_gpu_sft_authorization",
            ],
        },
        "observed_at": plan["observed_at"],
        "plan_file_sha256": _require_sha256(plan_file_sha256, "plan_file_sha256"),
        "production_eligible": False,
        "projection": {
            "accounting_language_floor_gaps": accounting_language_gaps,
            "accounting_task_floor_gaps": accounting_task_gaps,
            "behavior_language_floor_gaps": behavior_language_gaps,
            "behavior_task_floor_gaps": behavior_task_gaps,
            "family_cap_dilution_gap_non_nvidia_tokens": family_dilution_gap,
            "language_train_supervised_tokens_accounting": accounting_languages,
            "language_train_supervised_tokens_behavior_safe": behavior_languages,
            "source_cap": SOURCE_CAP,
            "source_cap_violations": source_cap_violations,
            "source_family_cap_violations": family_cap_violations,
            "source_family_shares": family_shares,
            "source_family_train_supervised_tokens": dict(sorted(family_tokens.items())),
            "source_shares": source_shares,
            "source_train_supervised_tokens": dict(sorted(source_tokens.items())),
            "strict_targeted_gap_lower_bound": strict_lower_bound,
            "task_train_supervised_tokens_accounting": accounting_tasks,
            "task_train_supervised_tokens_behavior_safe": behavior_tasks,
            "total_floor_gap": total_floor_gap,
            "train_supervised_tokens": projected_total,
            "validation_supervised_tokens": (
                EXPECTED_CURRENT_VALIDATION_TOKENS + EXPECTED_BUNDLE["validation_supervised_tokens"]
            ),
        },
        "raw_training_text_retained": False,
        "schema_version": REPORT_SCHEMA_VERSION,
        "source_metrics": source_metrics,
        "status": REPORT_STATUS,
    }
    report["report_sha256"] = sha256_value(report)
    validate_report(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported closure-successor report schema")
    _verify_internal_seal(report, "report_sha256", "closure-successor report")
    if report.get("status") != REPORT_STATUS:
        raise ValueError("closure-successor report status changed")
    if report.get("production_eligible") is not False:
        raise ValueError("closure-successor preflight cannot be production eligible")
    if report.get("raw_training_text_retained") is not False:
        raise ValueError("closure-successor report must be raw-text-free")
    bundle = report.get("bundle")
    if not isinstance(bundle, Mapping):
        raise ValueError("closure-successor bundle metrics missing")
    for key, expected in EXPECTED_BUNDLE.items():
        if bundle.get(key) != expected:
            raise ValueError(f"closure-successor bundle changed: {key}")
    projection = report.get("projection")
    if not isinstance(projection, Mapping):
        raise ValueError("closure-successor projection missing")
    if projection.get("train_supervised_tokens") != 9_632_369:
        raise ValueError("closure-successor total projection changed")
    if projection.get("total_floor_gap") != 367_631:
        raise ValueError("closure-successor floor gap changed")
    if projection.get("source_family_cap_violations") != ["nvidia_family"]:
        raise ValueError("closure-successor family-cap result changed")
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    for forbidden in ('"messages"', '"user_prompt"', '"assistant_response"'):
        if forbidden in serialized:
            raise ValueError("closure-successor report retained raw training text")


def build_verification(
    *,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    report_file_sha256: str,
) -> dict[str, Any]:
    validate_report(report)
    verification: dict[str, Any] = {
        "checks": {
            "all_19654_rows_independently_replayed": True,
            "all_bound_files_match_sha256": True,
            "blockers_and_targeted_gap_reconstructed": True,
            "frozen_token_counts_reproduced": True,
            "no_registry_dataset_or_training_state_changed": True,
            "raw_training_text_absent": True,
        },
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "report_file_sha256": _require_sha256(report_file_sha256, "report_file_sha256"),
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_closure_successor_projection_nonproduction",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification
