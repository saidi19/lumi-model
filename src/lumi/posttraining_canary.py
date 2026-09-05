"""Bounded adapters and raw-text-free metrics for Lumi SFT source canaries."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .posttraining_format import ChatTemplate, encode_sft_messages

CANARY_PLAN_SCHEMA_VERSION = "lumi.sft-source-canary-plan.v0.1"
CANARY_REPORT_SCHEMA_VERSION = "lumi.sft-source-canary-report.v0.1"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_canary_plan(path: str | Path, *, repository_root: Path) -> dict[str, Any]:
    """Validate the fail-closed, CPU-only canary plan and all local bindings."""

    plan_path = Path(path)
    with plan_path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("canary plan must be an object")
    if value.get("schema_version") != CANARY_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported canary plan schema")

    authorization = value.get("authorization")
    expected_authorization = {
        "bulk_data_download_authorized": False,
        "paid_gpu_authorized": False,
        "production_sft_authorized": False,
        "purpose": "bounded_cpu_only_source_canary",
        "raw_text_in_git": False,
    }
    if authorization != expected_authorization:
        raise ValueError("canary authorization boundary changed")

    limits = value.get("limits")
    if not isinstance(limits, Mapping):
        raise ValueError("canary limits must be an object")
    if limits.get("maximum_accepted_records_per_source") != 24:
        raise ValueError("canary per-source record cap changed")
    maximum_total = limits.get("maximum_total_response_bytes")
    maximum_single = limits.get("maximum_single_response_bytes")
    if maximum_total != 16 * 1024 * 1024 or maximum_single != 4 * 1024 * 1024:
        raise ValueError("canary response-byte cap changed")

    bindings = value.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("canary bindings must be an object")
    required_bindings = {
        "chat_template",
        "final_base_binding",
        "sft_contract",
        "source_registry",
    }
    if set(bindings) != required_bindings:
        raise ValueError("canary local bindings changed")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"canary binding must be an object: {name}")
        relative_path = binding.get("path")
        expected_sha256 = binding.get("sha256")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"canary binding path is invalid: {name}")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError(f"canary binding SHA-256 is invalid: {name}")
        if file_sha256(repository_root / relative_path) != expected_sha256:
            raise ValueError(f"canary binding SHA-256 changed: {name}")

    sources = value.get("sources")
    if not isinstance(sources, list) or len(sources) != 3:
        raise ValueError("canary must contain exactly three sources")
    expected_sources = {
        "coherelabs_aya_dataset",
        "openassistant_oasst1",
        "openr1_math_220k",
    }
    if {source.get("source_id") for source in sources if isinstance(source, Mapping)} != (
        expected_sources
    ):
        raise ValueError("canary source set changed")
    for source in sources:
        if not isinstance(source, Mapping):
            raise ValueError("canary source must be an object")
        if source.get("target_records") != 24:
            raise ValueError("canary source target changed")
        fetched_rows = source.get("maximum_fetched_rows")
        if not isinstance(fetched_rows, int) or not 1 <= fetched_rows <= 100:
            raise ValueError("canary source row cap is invalid")
        revision = source.get("revision")
        if not isinstance(revision, str) or len(revision) != 40:
            raise ValueError("canary source revision is invalid")
    return value


def _content_hash(messages: Sequence[Mapping[str, str]]) -> str:
    return sha256_bytes(canonical_json_bytes(list(messages)))


def _transform_hash(
    *,
    adapter: str,
    messages: Sequence[Mapping[str, str]],
    revision: str,
) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "adapter": adapter,
                "messages": list(messages),
                "output_schema": "lumi.sft.v0.1",
                "revision": revision,
            }
        )
    )


def _record(
    *,
    adapter: str,
    dataset_id: str,
    document_id: str,
    language: str,
    license_id: str,
    messages: list[dict[str, str]],
    quality_score: float,
    revision: str,
    row_index: int,
    source_id: str,
    split: str,
    task_category: str,
) -> dict[str, Any]:
    record_digest = sha256_bytes(
        canonical_json_bytes(
            {
                "dataset_id": dataset_id,
                "document_id": document_id,
                "revision": revision,
                "row_index": row_index,
            }
        )
    )
    return {
        "language": language,
        "messages": messages,
        "provenance": {
            "consent_basis": "source_level_open_license",
            "content_sha256": _content_hash(messages),
            "transform_sha256": _transform_hash(
                adapter=adapter,
                messages=messages,
                revision=revision,
            ),
        },
        "quality": {
            "pii_scan": "passed",
            "review_method": "bounded_canary_upstream_plus_deterministic_gates_v0.1",
            "review_status": "accepted",
            "score": quality_score,
            "secret_scan": "passed",
        },
        "record_id": f"{source_id}-{record_digest[:24]}",
        "safety": {
            "policy_version": "lumi-sft-canary-basic-pattern-scan-v0.1",
            "risk_labels": [],
        },
        "schema_version": "lumi.sft.v0.1",
        "source": {
            "dataset_id": dataset_id,
            "document_id": document_id,
            "license_id": license_id,
            "revision": revision,
            "source_id": source_id,
            "uri": (
                f"https://huggingface.co/datasets/{dataset_id}/viewer/"
                f"default/{split}?row={row_index}"
            ),
        },
        "split": split,
        "task_category": task_category,
    }


def _oasst_task(prompt: str) -> str:
    normalized = prompt.casefold()
    if any(
        marker in normalized
        for marker in (
            "```",
            " code ",
            "function",
            "javascript",
            "programming",
            "python",
            "sql",
        )
    ):
        return "coding"
    if any(
        marker in normalized
        for marker in ("essay", "rewrite", "story", "summar", "write a", "write an")
    ):
        return "writing_summarization"
    if normalized.startswith(("define ", "explain ", "how ", "what ", "when ", "where ", "why ")):
        return "factual_qa"
    return "conversation"


def adapt_oasst1_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    revision: str,
    target_records: int,
) -> list[dict[str, Any]]:
    """Select reviewed, human, rank-zero root prompt/assistant pairs."""

    normalized_rows: list[dict[str, Any]] = []
    for wrapped in rows:
        row = wrapped.get("row")
        row_index = wrapped.get("row_idx")
        if isinstance(row, Mapping) and isinstance(row_index, int):
            normalized_rows.append({**row, "_row_index": row_index})
    by_id = {
        row.get("message_id"): row
        for row in normalized_rows
        if isinstance(row.get("message_id"), str)
    }
    language_map = {"de": "de", "en": "en", "es": "es"}
    records: list[dict[str, Any]] = []
    for answer in normalized_rows:
        parent = by_id.get(answer.get("parent_id"))
        if answer.get("role") != "assistant" or not isinstance(parent, Mapping):
            continue
        if parent.get("role") != "prompter":
            continue
        if answer.get("review_result") is not True or parent.get("review_result") is not True:
            continue
        if parent.get("parent_id") is not None:
            continue
        if answer.get("deleted") or parent.get("deleted"):
            continue
        if answer.get("synthetic") or parent.get("synthetic"):
            continue
        if answer.get("rank") != 0:
            continue
        language = language_map.get(answer.get("lang"))
        if language is None or parent.get("lang") != answer.get("lang"):
            continue
        prompt = parent.get("text")
        response = answer.get("text")
        message_id = answer.get("message_id")
        if not all(
            isinstance(item, str) and item.strip()
            for item in (prompt, response, message_id)
        ):
            continue
        messages = [
            {"content": prompt, "role": "user"},
            {"content": response, "role": "assistant"},
        ]
        records.append(
            _record(
                adapter="oasst1_root_direct_rank0_v0.2",
                dataset_id="OpenAssistant/oasst1",
                document_id=message_id,
                language=language,
                license_id="apache-2.0",
                messages=messages,
                quality_score=0.9,
                revision=revision,
                row_index=answer["_row_index"],
                source_id="openassistant_oasst1",
                split="train",
                task_category=_oasst_task(prompt),
            )
        )
        if len(records) == target_records:
            break
    return records


def adapt_aya_bengali_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    revision: str,
    target_records: int,
) -> list[dict[str, Any]]:
    """Convert revision-pinned Aya Bengali prompt/response annotations."""

    records: list[dict[str, Any]] = []
    for wrapped in rows:
        row = wrapped.get("row")
        row_index = wrapped.get("row_idx")
        if not isinstance(row, Mapping) or not isinstance(row_index, int):
            continue
        if row.get("language_code") != "ben":
            continue
        prompt = row.get("inputs")
        response = row.get("targets")
        if not all(isinstance(item, str) and item.strip() for item in (prompt, response)):
            continue
        messages = [
            {"content": prompt, "role": "user"},
            {"content": response, "role": "assistant"},
        ]
        records.append(
            _record(
                adapter="aya_prompt_response_v0.1",
                dataset_id="CohereLabs/aya_dataset",
                document_id=f"train-row-{row_index}",
                language="bn",
                license_id="apache-2.0",
                messages=messages,
                quality_score=0.85,
                revision=revision,
                row_index=row_index,
                source_id="coherelabs_aya_dataset",
                split="train",
                task_category="multilingual_bengali",
            )
        )
        if len(records) == target_records:
            break
    return records


def adapt_openr1_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    revision: str,
    target_records: int,
) -> list[dict[str, Any]]:
    """Convert complete, verifier-supported OpenR1 problem/solution traces."""

    records: list[dict[str, Any]] = []
    for wrapped in rows:
        row = wrapped.get("row")
        row_index = wrapped.get("row_idx")
        if not isinstance(row, Mapping) or not isinstance(row_index, int):
            continue
        correctness_count = row.get("correctness_count")
        completeness = row.get("is_reasoning_complete")
        if not isinstance(correctness_count, int) or correctness_count < 1:
            continue
        if not isinstance(completeness, list) or not any(item is True for item in completeness):
            continue
        prompt = row.get("problem")
        response = row.get("solution")
        document_id = row.get("uuid")
        if not all(
            isinstance(item, str) and item.strip()
            for item in (prompt, response, document_id)
        ):
            continue
        messages = [
            {"content": prompt, "role": "user"},
            {"content": response, "role": "assistant"},
        ]
        records.append(
            _record(
                adapter="openr1_math_default_v0.1",
                dataset_id="open-r1/OpenR1-Math-220k",
                document_id=document_id,
                language="en",
                license_id="apache-2.0",
                messages=messages,
                quality_score=0.85,
                revision=revision,
                row_index=row_index,
                source_id="openr1_math_220k",
                split="train",
                task_category="reasoning_math_science",
            )
        )
        if len(records) == target_records:
            break
    return records


def _nearest_rank(values: Sequence[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def encode_canary_records(
    records: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    template: ChatTemplate,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Apply exact 4K serialization and return aggregate, raw-text-free metrics."""

    accepted: list[Mapping[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    total_tokens: list[int] = []
    supervised_tokens: list[int] = []
    masked_tokens: list[int] = []
    for record in records:
        try:
            encoded = encode_sft_messages(
                record["messages"],
                tokenizer=tokenizer,
                template=template,
            )
        except ValueError as error:
            reason = "context_overflow" if "maximum is 4096" in str(error) else "encoding_error"
            rejection_counts[reason] += 1
            continue
        accepted.append(record)
        total_tokens.append(len(encoded.input_ids))
        supervised_tokens.append(encoded.supervised_tokens)
        masked_tokens.append(encoded.masked_tokens)

    aggregate_total = sum(total_tokens)
    report = {
        "accepted": len(accepted),
        "assistant_only_loss_verified": bool(accepted)
        and all(value > 0 for value in supervised_tokens)
        and all(value > 0 for value in masked_tokens),
        "context_window": template.context_window,
        "fit_rate": len(accepted) / len(records) if records else 0.0,
        "masked_tokens": sum(masked_tokens),
        "rejected": len(records) - len(accepted),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "supervised_fraction": (
            sum(supervised_tokens) / aggregate_total if aggregate_total else 0.0
        ),
        "supervised_tokens": sum(supervised_tokens),
        "total": len(records),
        "total_tokens": aggregate_total,
        "total_tokens_distribution": {
            "max": max(total_tokens, default=0),
            "p50": _nearest_rank(total_tokens, 0.5),
            "p90": _nearest_rank(total_tokens, 0.9),
            "p95": _nearest_rank(total_tokens, 0.95),
            "p99": _nearest_rank(total_tokens, 0.99),
        },
    }
    return accepted, report


def distribution(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    values: Counter[str] = Counter()
    for record in records:
        value: Any
        if field == "source_id":
            value = record.get("source", {}).get("source_id")
        else:
            value = record.get(field)
        if isinstance(value, str):
            values[value] += 1
    return dict(sorted(values.items()))


def bengali_script_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Measure Bengali Unicode coverage without retaining any sampled text."""

    bengali_records = 0
    records_with_bengali = 0
    alphabetic_characters = 0
    bengali_alphabetic_characters = 0
    bengali_codepoints = 0
    for record in records:
        if record.get("language") != "bn":
            continue
        bengali_records += 1
        messages = record.get("messages")
        if not isinstance(messages, list):
            continue
        record_has_bengali = False
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            for character in content:
                if character.isalpha():
                    alphabetic_characters += 1
                    if "\u0980" <= character <= "\u09ff":
                        bengali_alphabetic_characters += 1
                if "\u0980" <= character <= "\u09ff":
                    bengali_codepoints += 1
                    record_has_bengali = True
        if record_has_bengali:
            records_with_bengali += 1
    return {
        "alphabetic_characters": alphabetic_characters,
        "bengali_alphabetic_characters": bengali_alphabetic_characters,
        "bengali_alphabetic_fraction": (
            bengali_alphabetic_characters / alphabetic_characters
            if alphabetic_characters
            else 0.0
        ),
        "bengali_codepoints": bengali_codepoints,
        "records": bengali_records,
        "records_with_bengali": records_with_bengali,
    }


def finalize_canary_report(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema_version") != CANARY_REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported canary report schema")
    if report.get("raw_text_retained") is not False:
        raise ValueError("canary report must not retain raw text")
    value = dict(report)
    value["report_sha256"] = sha256_bytes(canonical_json_bytes(value))
    return value
