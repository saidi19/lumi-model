"""Fail-closed planning and selection for the pinned OASST1 SFT source."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PLAN_SCHEMA_VERSION = "lumi.sft-oasst1-full-curation-plan.v0.1"
PREFLIGHT_SCHEMA_VERSION = "lumi.sft-oasst1-full-curation-preflight.v0.1"
ATTRIBUTION_SCHEMA_VERSION = "lumi.sft-oasst1-attribution.v0.1"

SOURCE_REVISION = "fdf72ae0827c1cda404aff25b6603abec9e3399b"
SOURCE_DATASET_ID = "OpenAssistant/oasst1"
SOURCE_ID = "openassistant_oasst1"
SOURCE_LICENSE = "apache-2.0"
ADAPTER_ID = "oasst1_root_direct_rank0_quality_v0.3"

EXPECTED_AUTHORIZATION = {
    "bulk_data_download_authorized": False,
    "dataset_promotion_authorized": False,
    "external_api_spend_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_dataset_seal_authorized": False,
    "production_sft_authorized": False,
    "purpose": "local_oasst1_full_curation_preflight_only",
}
EXPECTED_LIMITS = {
    "maximum_data_objects": 2,
    "maximum_decoded_bytes": 134_217_728,
    "maximum_network_bytes": 67_108_864,
    "maximum_objects": 4,
    "maximum_source_rows": 88_838,
    "maximum_train_rows": 84_437,
    "maximum_validation_rows": 4_401,
    "preflight_sample_rows": 100,
}
EXPECTED_SOURCE = {
    "adapter": ADAPTER_ID,
    "dataset_id": SOURCE_DATASET_ID,
    "dataset_size_bytes": 105_611_404,
    "download_size_bytes": 41_596_430,
    "license_id": SOURCE_LICENSE,
    "revision": SOURCE_REVISION,
    "source_id": SOURCE_ID,
    "splits": {
        "train": {"decoded_bytes": 100_367_999, "rows": 84_437},
        "validation": {"decoded_bytes": 5_243_405, "rows": 4_401},
    },
}
EXPECTED_OBJECTS = [
    {
        "decoded_bytes": 100_367_999,
        "kind": "data",
        "path": "data/train-00000-of-00001-b42a775f407cee45.parquet",
        "rows": 84_437,
        "split": "train",
    },
    {
        "decoded_bytes": 5_243_405,
        "kind": "data",
        "path": "data/validation-00000-of-00001-134b8fd0c89408b6.parquet",
        "rows": 4_401,
        "split": "validation",
    },
    {"kind": "license", "path": "LICENSE"},
    {"kind": "documentation", "path": "README.md"},
]
EXPECTED_ACQUISITION = {
    "exact_objects": EXPECTED_OBJECTS,
    "mode": "future_revision_pinned_fetch_requires_separate_authorization",
    "reject_redirect_to_different_revision": True,
    "reject_unlisted_objects": True,
}
EXPECTED_POLICY = {
    "allowed_languages": ["en"],
    "answer_quality_minimum": 0.75,
    "assistant_rank": 0,
    "cross_split_normalized_deduplication_required": True,
    "maximum_final_train_supervised_token_share": 0.35,
    "minimum_answer_review_count": 3,
    "minimum_prompt_review_count": 3,
    "model_name_must_be_null": True,
    "one_turn_root_direct_only": True,
    "post_selection_gates": [
        "lumi_schema_validation",
        "pii_and_secret_scan",
        "exact_and_normalized_deduplication",
        "benchmark_decontamination",
        "frozen_4096_token_fit",
        "assistant_only_loss_encoding",
        "deterministic_owner_quality_sample",
    ],
    "preserve_official_split": True,
    "prompt_quality_minimum": 0.5,
    "required_tree_state": "ready_for_export",
    "review_result_must_pass": True,
    "same_prompt_answer_language_required": True,
    "synthetic_rows_allowed": False,
    "task_categories": [
        "coding",
        "conversation",
        "factual_qa",
        "writing_summarization",
    ],
}
EXPECTED_ATTRIBUTION = {
    "dataset_card_url": (
        f"https://huggingface.co/datasets/{SOURCE_DATASET_ID}/tree/{SOURCE_REVISION}"
    ),
    "excluded_fields": ["raw_text", "user_id"],
    "license_url": (
        f"https://huggingface.co/datasets/{SOURCE_DATASET_ID}/blob/"
        f"{SOURCE_REVISION}/LICENSE"
    ),
    "manifest_schema_version": ATTRIBUTION_SCHEMA_VERSION,
    "paper_url": "https://arxiv.org/abs/2304.07327",
    "required_record_fields": [
        "answer_message_id",
        "answer_row_index",
        "content_sha256",
        "dataset_id",
        "license_id",
        "prompt_message_id",
        "prompt_row_index",
        "record_id",
        "revision",
        "source_id",
        "source_object_path",
        "source_object_sha256",
        "split",
        "transform_sha256",
    ],
}
EXPECTED_OUTPUT = {
    "attribution_manifest_contains_raw_training_text": False,
    "preflight_report_contains_raw_training_text": False,
    "preflight_report_path": "evidence/lumi-4b-sft-oasst1-full-curation-preflight-v0.1.json",
    "production_promotion_permitted": False,
    "raw_dataset_in_git_permitted": False,
}
EXPECTED_BINDING_PATHS = {
    "curation_implementation": "src/lumi/posttraining_oasst1_curation.py",
    "curation_runner": "scripts/check_oasst1_full_curation_plan.py",
    "data_contract": "configs/lumi-4b-sft-data-contract-v0.1.json",
    "missing_mix_proposal": "configs/lumi-4b-sft-missing-mix-proposal-v0.1.json",
    "pinned_metadata": (
        "../lumi-artifacts/4b-sft-source-canary-v0.1/"
        "5d1b59ab15bd03f23f2fd3d4e7ef931041d22a93ac0887a2ad77a289d5908f02/"
        "attempt-004/raw/openassistant_oasst1-metadata-post.json"
    ),
    "prior_reaudit_plan": "configs/lumi-4b-sft-oasst1-repair-reaudit-v0.1.json",
    "prior_reaudit_receipt": "evidence/lumi-4b-sft-oasst1-repair-reaudit-v0.1.json",
    "raw_sample": (
        "../lumi-artifacts/4b-sft-source-canary-v0.1/"
        "5d1b59ab15bd03f23f2fd3d4e7ef931041d22a93ac0887a2ad77a289d5908f02/"
        "attempt-004/raw/openassistant_oasst1-rows.json"
    ),
    "successor_source_registry": "configs/lumi-4b-sft-source-registry-v0.2.json",
}
EXPECTED_REMAINING_BLOCKERS = [
    "full_dataset_not_acquired_or_curated",
    "full_filter_deduplication_and_decontamination_not_run",
    "deterministic_owner_quality_sample_not_reviewed",
    "source_registry_production_promotion_not_authorized",
    "production_dataset_seal_not_authorized",
]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class OASST1Selection:
    """Private candidate rows plus a raw-text-free selection summary."""

    candidates: tuple[dict[str, Any], ...]
    metrics: dict[str, Any]


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


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a SHA-256")
    return value


def resolve_curation_binding(
    repository_root: str | Path,
    binding: Mapping[str, Any],
    *,
    name: str,
) -> Path:
    """Resolve one exact repository/workspace binding without path escape."""

    root = Path(repository_root).resolve()
    relative = binding.get("path")
    if relative != EXPECTED_BINDING_PATHS.get(name):
        raise ValueError(f"OASST1 curation binding path changed: {name}")
    resolved = (root / str(relative)).resolve()
    if not resolved.is_relative_to(root.parent):
        raise ValueError(f"OASST1 curation binding escapes workspace: {name}")
    return resolved


def load_oasst1_curation_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Load and validate the exact local-only OASST1 full-curation plan."""

    plan = _load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported OASST1 full-curation plan")
    if plan.get("status") != "local_preflight_ready_acquisition_requires_authorization":
        raise ValueError("OASST1 full-curation status changed")
    if plan.get("observed_at") != "2026-08-06":
        raise ValueError("OASST1 full-curation observation date changed")
    exact_fields = {
        "acquisition": EXPECTED_ACQUISITION,
        "attribution": EXPECTED_ATTRIBUTION,
        "authorization": EXPECTED_AUTHORIZATION,
        "limits": EXPECTED_LIMITS,
        "output": EXPECTED_OUTPUT,
        "policy": EXPECTED_POLICY,
        "source": EXPECTED_SOURCE,
    }
    for field, expected in exact_fields.items():
        if plan.get(field) != expected:
            raise ValueError(f"OASST1 full-curation {field} changed")

    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(EXPECTED_BINDING_PATHS):
        raise ValueError("OASST1 full-curation binding set changed")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"OASST1 curation binding must be an object: {name}")
        expected_sha256 = _require_sha256(
            binding.get("sha256"), f"bindings.{name}.sha256"
        )
        resolved = resolve_curation_binding(repository_root, binding, name=name)
        if not resolved.is_file() or file_sha256(resolved) != expected_sha256:
            raise ValueError(f"OASST1 curation binding changed: {name}")
    return plan


def verify_prior_reaudit(
    prior_plan: Mapping[str, Any],
    prior_receipt: Mapping[str, Any],
) -> None:
    """Require the already sealed structural repair before broader curation."""

    if prior_plan.get("schema_version") != "lumi.sft-oasst1-repair-reaudit-plan.v0.1":
        raise ValueError("OASST1 prior repair plan schema changed")
    source = prior_plan.get("source")
    if not isinstance(source, Mapping) or source.get("revision") != SOURCE_REVISION:
        raise ValueError("OASST1 prior repair plan revision changed")
    if prior_plan.get("authorization", {}).get("network_access_authorized") is not False:
        raise ValueError("OASST1 prior repair authorization changed")

    if prior_receipt.get("schema_version") != (
        "lumi.sft-oasst1-repair-audit-report.v0.1"
    ):
        raise ValueError("OASST1 prior repair receipt schema changed")
    seal = _require_sha256(prior_receipt.get("report_sha256"), "report_sha256")
    unsealed = {key: value for key, value in prior_receipt.items() if key != "report_sha256"}
    if sha256_value(unsealed) != seal:
        raise ValueError("OASST1 prior repair receipt seal changed")
    if (
        prior_receipt.get("decision")
        != "structural_repair_passed_source_still_blocked"
        or prior_receipt.get("closed_findings")
        != [
            "current_pair_adapter_drops_required_ancestor_history",
            "unranked_replies_are_admitted_by_current_adapter",
        ]
        or prior_receipt.get("remaining_production_blockers")
        != [
            "full_thread_selection_and_quality_policy_not_sealed",
            "attribution_manifest_not_built",
        ]
        or prior_receipt.get("production_eligible") is not False
        or prior_receipt.get("production_promotions") != 0
    ):
        raise ValueError("OASST1 prior repair receipt did not preserve blockers")


def verify_pinned_metadata(metadata: Mapping[str, Any]) -> None:
    """Verify exact source identity, split sizes, objects, and license metadata."""

    if metadata.get("id") != SOURCE_DATASET_ID or metadata.get("sha") != SOURCE_REVISION:
        raise ValueError("OASST1 pinned metadata identity changed")
    if metadata.get("private") is not False or metadata.get("gated") is not False:
        raise ValueError("OASST1 source access state changed")
    card = metadata.get("cardData")
    if not isinstance(card, Mapping) or card.get("license") != SOURCE_LICENSE:
        raise ValueError("OASST1 pinned license metadata changed")
    info = card.get("dataset_info")
    if not isinstance(info, Mapping):
        raise ValueError("OASST1 dataset_info is missing")
    if (
        info.get("download_size") != EXPECTED_SOURCE["download_size_bytes"]
        or info.get("dataset_size") != EXPECTED_SOURCE["dataset_size_bytes"]
    ):
        raise ValueError("OASST1 byte totals changed")
    split_rows = info.get("splits")
    if not isinstance(split_rows, list):
        raise ValueError("OASST1 split metadata is missing")
    observed_splits = {
        item.get("name"): {
            "decoded_bytes": item.get("num_bytes"),
            "rows": item.get("num_examples"),
        }
        for item in split_rows
        if isinstance(item, Mapping)
    }
    if observed_splits != EXPECTED_SOURCE["splits"]:
        raise ValueError("OASST1 split bounds changed")
    siblings = metadata.get("siblings")
    if not isinstance(siblings, list):
        raise ValueError("OASST1 object inventory is missing")
    observed_objects = {
        item.get("rfilename") for item in siblings if isinstance(item, Mapping)
    }
    required = {item["path"] for item in EXPECTED_OBJECTS}
    if not required.issubset(observed_objects):
        raise ValueError("OASST1 required source object is missing")
    tags = metadata.get("tags")
    if not isinstance(tags, list) or f"license:{SOURCE_LICENSE}" not in tags:
        raise ValueError("OASST1 license tag changed")


def label_measurement(row: Mapping[str, Any], name: str) -> tuple[float, int] | None:
    """Read one OASST1 label from either dict-of-lists or list-of-dicts form."""

    labels = row.get("labels")
    candidates: list[tuple[Any, Any]] = []
    if isinstance(labels, Mapping):
        names = labels.get("name")
        values = labels.get("value")
        counts = labels.get("count")
        if (
            isinstance(names, Sequence)
            and not isinstance(names, (str, bytes))
            and isinstance(values, Sequence)
            and not isinstance(values, (str, bytes))
            and isinstance(counts, Sequence)
            and not isinstance(counts, (str, bytes))
            and len(names) == len(values) == len(counts)
        ):
            candidates.extend(
                (value, count)
                for label_name, value, count in zip(names, values, counts, strict=True)
                if label_name == name
            )
    elif isinstance(labels, Sequence) and not isinstance(labels, (str, bytes)):
        candidates.extend(
            (item.get("value"), item.get("count"))
            for item in labels
            if isinstance(item, Mapping) and item.get("name") == name
        )
    if len(candidates) != 1:
        return None
    value, count = candidates[0]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 <= float(value) <= 1
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
    ):
        return None
    return float(value), count


def _task_category(prompt: str) -> str:
    normalized = f" {prompt.casefold()} "
    if any(
        marker in normalized
        for marker in ("```", " code ", " function ", "javascript", "programming", "python", "sql")
    ):
        return "coding"
    if any(
        marker in normalized
        for marker in ("essay", "rewrite", "story", "summar", "write a", "write an")
    ):
        return "writing_summarization"
    stripped = normalized.strip()
    if stripped.startswith(("define ", "explain ", "how ", "what ", "when ", "where ", "why ")):
        return "factual_qa"
    return "conversation"


def _valid_count(value: Any, minimum: int) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= minimum


def _attribution_entry(
    *,
    answer: Mapping[str, Any],
    answer_row_index: int,
    messages: Sequence[Mapping[str, str]],
    object_path: str,
    prompt: Mapping[str, Any],
    prompt_row_index: int,
    record_id: str,
    revision: str,
    split: str,
) -> dict[str, Any]:
    content_sha256 = sha256_value(list(messages))
    transform_sha256 = sha256_value(
        {
            "adapter": ADAPTER_ID,
            "content_sha256": content_sha256,
            "object_path": object_path,
            "output_schema": "lumi.sft.v0.1",
            "revision": revision,
            "split": split,
        }
    )
    return {
        "answer_message_id": answer["message_id"],
        "answer_row_index": answer_row_index,
        "content_sha256": content_sha256,
        "dataset_id": SOURCE_DATASET_ID,
        "license_id": SOURCE_LICENSE,
        "prompt_message_id": prompt["message_id"],
        "prompt_row_index": prompt_row_index,
        "record_id": record_id,
        "revision": revision,
        "schema_version": ATTRIBUTION_SCHEMA_VERSION,
        "source_id": SOURCE_ID,
        "source_object_path": object_path,
        "source_object_sha256": None,
        "split": split,
        "transform_sha256": transform_sha256,
    }


def select_oasst1_pairs(
    rows: Sequence[Mapping[str, Any]],
    *,
    object_path: str,
    revision: str,
    split: str,
) -> OASST1Selection:
    """Select strict one-turn English human pairs without writing their text."""

    expected_data_objects = {
        item["split"]: item["path"]
        for item in EXPECTED_OBJECTS
        if item["kind"] == "data"
    }
    if revision != SOURCE_REVISION:
        raise ValueError("OASST1 selection revision changed")
    if split not in expected_data_objects or object_path != expected_data_objects[split]:
        raise ValueError("OASST1 selection object or split changed")

    normalized: list[dict[str, Any]] = []
    invalid_wrappers = 0
    for wrapped in rows:
        row = wrapped.get("row") if isinstance(wrapped, Mapping) else None
        row_index = wrapped.get("row_idx") if isinstance(wrapped, Mapping) else None
        if (
            not isinstance(row, Mapping)
            or isinstance(row_index, bool)
            or not isinstance(row_index, int)
        ):
            invalid_wrappers += 1
            continue
        normalized.append({**row, "_row_index": row_index})

    id_counts = Counter(
        row.get("message_id")
        for row in normalized
        if isinstance(row.get("message_id"), str) and row["message_id"]
    )
    by_id = {
        row["message_id"]: row
        for row in normalized
        if isinstance(row.get("message_id"), str)
        and row["message_id"]
        and id_counts[row["message_id"]] == 1
    }

    rejections: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    assistant_rows = 0
    root_direct_rank0_pairs = 0
    for answer in normalized:
        if answer.get("role") != "assistant":
            continue
        assistant_rows += 1
        reasons: list[str] = []
        answer_id = answer.get("message_id")
        parent_id = answer.get("parent_id")
        if not isinstance(answer_id, str) or not answer_id:
            reasons.append("missing_answer_message_id")
        elif id_counts[answer_id] != 1:
            reasons.append("duplicate_answer_message_id")
        if not isinstance(parent_id, str) or not parent_id:
            reasons.append("missing_parent_id")
            prompt = None
        else:
            prompt = by_id.get(parent_id)
            if prompt is None:
                reasons.append("missing_or_duplicate_parent")
        if isinstance(prompt, Mapping):
            if prompt.get("role") != "prompter":
                reasons.append("parent_not_prompter")
            if prompt.get("parent_id") is not None:
                reasons.append("followup_context_excluded")
        if answer.get("rank") != EXPECTED_POLICY["assistant_rank"]:
            reasons.append("assistant_not_rank_zero")
        if reasons:
            rejections.update(dict.fromkeys(reasons).keys())
            continue
        assert isinstance(prompt, Mapping)
        root_direct_rank0_pairs += 1

        for prefix, row, minimum_reviews in (
            ("answer", answer, EXPECTED_POLICY["minimum_answer_review_count"]),
            ("prompt", prompt, EXPECTED_POLICY["minimum_prompt_review_count"]),
        ):
            if row.get("review_result") is not True:
                reasons.append(f"{prefix}_review_failed")
            if row.get("deleted") is not False:
                reasons.append(f"{prefix}_deleted_or_unknown")
            if row.get("synthetic") is not False:
                reasons.append(f"{prefix}_synthetic_or_unknown")
            if row.get("model_name") is not None:
                reasons.append(f"{prefix}_model_name_present")
            if row.get("tree_state") != EXPECTED_POLICY["required_tree_state"]:
                reasons.append(f"{prefix}_tree_not_ready")
            if not _valid_count(row.get("review_count"), minimum_reviews):
                reasons.append(f"{prefix}_review_count_below_floor")
        language = answer.get("lang")
        if language != "en" or prompt.get("lang") != language:
            reasons.append("language_not_allowed_or_mismatched")
        prompt_text = prompt.get("text")
        answer_text = answer.get("text")
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            reasons.append("empty_prompt")
        if not isinstance(answer_text, str) or not answer_text.strip():
            reasons.append("empty_answer")
        prompt_quality = label_measurement(prompt, "quality")
        answer_quality = label_measurement(answer, "quality")
        if prompt_quality is None:
            reasons.append("prompt_quality_missing_or_invalid")
        elif prompt_quality[0] < EXPECTED_POLICY["prompt_quality_minimum"]:
            reasons.append("prompt_quality_below_floor")
        if answer_quality is None:
            reasons.append("answer_quality_missing_or_invalid")
        elif answer_quality[0] < EXPECTED_POLICY["answer_quality_minimum"]:
            reasons.append("answer_quality_below_floor")
        if reasons:
            rejections.update(dict.fromkeys(reasons).keys())
            continue

        assert isinstance(prompt_text, str)
        assert isinstance(answer_text, str)
        assert isinstance(prompt_quality, tuple)
        assert isinstance(answer_quality, tuple)
        messages = (
            {"content": prompt_text, "role": "user"},
            {"content": answer_text, "role": "assistant"},
        )
        record_digest = sha256_value(
            {
                "answer_message_id": answer_id,
                "dataset_id": SOURCE_DATASET_ID,
                "revision": revision,
                "split": split,
            }
        )
        record_id = f"{SOURCE_ID}-{record_digest[:24]}"
        attribution = _attribution_entry(
            answer=answer,
            answer_row_index=answer["_row_index"],
            messages=messages,
            object_path=object_path,
            prompt=prompt,
            prompt_row_index=prompt["_row_index"],
            record_id=record_id,
            revision=revision,
            split=split,
        )
        candidates.append(
            {
                "answer_quality": answer_quality[0],
                "attribution": attribution,
                "language": "en",
                "messages": messages,
                "prompt_quality": prompt_quality[0],
                "record_id": record_id,
                "split": split,
                "task_category": _task_category(prompt_text),
            }
        )

    metrics = {
        "accepted_pairs": len(candidates),
        "assistant_rows": assistant_rows,
        "input_rows": len(rows),
        "invalid_wrappers": invalid_wrappers,
        "rejection_counts": dict(sorted(rejections.items())),
        "root_direct_rank0_pairs": root_direct_rank0_pairs,
    }
    return OASST1Selection(candidates=tuple(candidates), metrics=metrics)


def build_preflight_report(
    *,
    metadata: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    prior_plan: Mapping[str, Any],
    prior_receipt: Mapping[str, Any],
    raw_sample: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the deterministic, raw-text-free preflight receipt."""

    verify_pinned_metadata(metadata)
    verify_prior_reaudit(prior_plan, prior_receipt)
    _require_sha256(plan_file_sha256, "plan_file_sha256")
    rows = raw_sample.get("rows")
    if (
        not isinstance(rows, list)
        or len(rows) != EXPECTED_LIMITS["preflight_sample_rows"]
        or raw_sample.get("num_rows_total") != EXPECTED_SOURCE["splits"]["train"]["rows"]
        or raw_sample.get("partial") is not False
    ):
        raise ValueError("OASST1 pinned preflight sample boundary changed")
    train_object = next(
        item for item in EXPECTED_OBJECTS if item.get("split") == "train"
    )
    selection = select_oasst1_pairs(
        rows,
        object_path=train_object["path"],
        revision=SOURCE_REVISION,
        split="train",
    )
    record_ids = sorted(item["record_id"] for item in selection.candidates)
    attributions = sorted(
        (item["attribution"] for item in selection.candidates),
        key=lambda item: item["record_id"],
    )
    tasks = Counter(item["task_category"] for item in selection.candidates)
    answer_quality = [item["answer_quality"] for item in selection.candidates]
    prompt_quality = [item["prompt_quality"] for item in selection.candidates]
    if not record_ids:
        raise ValueError("OASST1 preflight policy selected zero rows")

    report = {
        "acquisition_authorized": False,
        "attribution_preflight": {
            "entry_count": len(attributions),
            "manifest_schema_version": ATTRIBUTION_SCHEMA_VERSION,
            "manifest_sha256": sha256_value(attributions),
            "raw_text_retained": False,
            "required_fields": EXPECTED_ATTRIBUTION["required_record_fields"],
            "source_object_sha256_status": "required_after_future_fetch",
        },
        "authorization": {
            **EXPECTED_AUTHORIZATION,
            "network_access_used": False,
        },
        "bindings": {
            name: binding["sha256"] for name, binding in sorted(plan["bindings"].items())
        }
        | {"plan_file_sha256": plan_file_sha256},
        "decision": "full_curation_plan_passed_acquisition_requires_authorization",
        "distributions": {
            "language": {"en": len(record_ids)},
            "split": {"train": len(record_ids)},
            "task": dict(sorted(tasks.items())),
        },
        "limits": EXPECTED_LIMITS,
        "observed_at": plan["observed_at"],
        "policy": {
            "adapter": ADAPTER_ID,
            "answer_quality_minimum": EXPECTED_POLICY["answer_quality_minimum"],
            "maximum_final_train_supervised_token_share": EXPECTED_POLICY[
                "maximum_final_train_supervised_token_share"
            ],
            "one_turn_root_direct_only": True,
            "prompt_quality_minimum": EXPECTED_POLICY["prompt_quality_minimum"],
            "selected_language": "en",
        },
        "preflight_sample": {
            "accepted_record_ids_sha256": sha256_value(record_ids),
            "answer_quality_maximum": max(answer_quality),
            "answer_quality_minimum": min(answer_quality),
            "prompt_quality_maximum": max(prompt_quality),
            "prompt_quality_minimum": min(prompt_quality),
            **selection.metrics,
        },
        "production_eligible": False,
        "production_promotions": 0,
        "raw_text_retained": False,
        "remaining_production_blockers": EXPECTED_REMAINING_BLOCKERS,
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "source": EXPECTED_SOURCE,
        "status": "local_preflight_passed_no_download_no_promotion",
    }
    report["report_sha256"] = sha256_value(report)
    verify_preflight_report(report)
    return report


def verify_preflight_report(report: Mapping[str, Any]) -> None:
    """Verify the preflight receipt and reject raw text or authority expansion."""

    if report.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        raise ValueError("unsupported OASST1 preflight receipt")
    seal = _require_sha256(report.get("report_sha256"), "report_sha256")
    unsealed = {key: value for key, value in report.items() if key != "report_sha256"}
    if sha256_value(unsealed) != seal:
        raise ValueError("OASST1 preflight receipt seal changed")
    if (
        report.get("status") != "local_preflight_passed_no_download_no_promotion"
        or report.get("decision")
        != "full_curation_plan_passed_acquisition_requires_authorization"
        or report.get("acquisition_authorized") is not False
        or report.get("production_eligible") is not False
        or report.get("production_promotions") != 0
        or report.get("raw_text_retained") is not False
        or report.get("remaining_production_blockers") != EXPECTED_REMAINING_BLOCKERS
    ):
        raise ValueError("OASST1 preflight boundary changed")
    authorization = report.get("authorization")
    if not isinstance(authorization, Mapping):
        raise ValueError("OASST1 preflight authorization is missing")
    if authorization.get("network_access_used") is not False:
        raise ValueError("OASST1 preflight unexpectedly used network access")
    for key, value in EXPECTED_AUTHORIZATION.items():
        if authorization.get(key) != value:
            raise ValueError("OASST1 preflight authorization changed")
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    for forbidden_key in ('"messages"', '"prompt"', '"response"', '"text"', '"user_id"'):
        if forbidden_key in serialized:
            raise ValueError("OASST1 preflight retained a forbidden raw-data field")
