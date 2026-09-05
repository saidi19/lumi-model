"""Deterministic, fail-closed sealing for a production Lumi SFT dataset."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from .posttraining_data import (
    BenchmarkMatcher,
    SFTPolicy,
    filter_sft_records,
)
from .posttraining_format import ChatTemplate, encode_sft_messages
from .posttraining_readiness import validate_final_base_binding
from .posttraining_sources import SFTSourceRegistry

SFT_SEAL_SCHEMA_VERSION = "lumi.sft-seal.v0.1"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def build_sft_dataset_seal(
    records: Iterable[Mapping[str, Any]],
    *,
    base_binding: Mapping[str, Any],
    benchmark_registry: BenchmarkMatcher,
    policy: SFTPolicy,
    policy_sha256: str,
    source_registry: SFTSourceRegistry,
    template: ChatTemplate,
    tokenizer: Any,
) -> dict[str, Any]:
    """Validate, encode, and seal records without retaining text in the manifest."""

    bound_base = validate_final_base_binding(base_binding)
    materialized = list(records)
    accepted, validation_report = filter_sft_records(
        materialized,
        policy=policy,
        benchmark_registry=benchmark_registry,
        source_registry=source_registry,
    )
    if len(accepted) != len(materialized):
        raise ValueError("SFT seal input contains rejected records")
    if not accepted:
        raise ValueError("SFT seal input is empty")

    used_source_ids = sorted({record["source"]["source_id"] for record in accepted})
    source_blockers: list[str] = []
    for source_id in used_source_ids:
        source = source_registry.by_id[source_id]
        if not source.production_eligible:
            source_blockers.extend(
                f"{source_id}:{blocker}" for blocker in source.production_blockers
            )
    if source_blockers:
        raise ValueError(
            "SFT source promotion blockers remain: " + ",".join(sorted(source_blockers))
        )

    record_hasher = hashlib.sha256()
    encoded_hasher = hashlib.sha256()
    input_tokens = 0
    supervised_tokens = 0
    masked_tokens = 0
    language_records: Counter[str] = Counter()
    source_records: Counter[str] = Counter()
    split_records: Counter[str] = Counter()
    task_records: Counter[str] = Counter()

    for record in accepted:
        record_bytes = _canonical_json(record)
        record_hasher.update(len(record_bytes).to_bytes(8, "big"))
        record_hasher.update(record_bytes)
        encoded = encode_sft_messages(
            record["messages"],
            tokenizer=tokenizer,
            template=template,
        )
        encoded_payload = {
            "input_ids": list(encoded.input_ids),
            "labels": list(encoded.labels),
            "record_id": record["record_id"],
        }
        encoded_bytes = _canonical_json(encoded_payload)
        encoded_hasher.update(len(encoded_bytes).to_bytes(8, "big"))
        encoded_hasher.update(encoded_bytes)
        input_tokens += len(encoded.input_ids)
        supervised_tokens += encoded.supervised_tokens
        masked_tokens += encoded.masked_tokens
        language_records[record["language"]] += 1
        source_records[record["source"]["source_id"]] += 1
        split_records[record["split"]] += 1
        task_records[record["task_category"]] += 1

    manifest = {
        "base_checkpoint_manifest_sha256": bound_base["checkpoint"]["manifest_sha256"],
        "base_handoff_sha256": bound_base["handoff"]["sha256"],
        "base_run_fingerprint": bound_base["run_fingerprint"],
        "encoded_dataset_sha256": encoded_hasher.hexdigest(),
        "input_tokens": input_tokens,
        "language_records": dict(sorted(language_records.items())),
        "masked_tokens": masked_tokens,
        "policy_sha256": policy_sha256,
        "raw_text_retained_in_manifest": False,
        "record_jsonl_sha256": record_hasher.hexdigest(),
        "records": len(accepted),
        "schema_version": SFT_SEAL_SCHEMA_VERSION,
        "source_records": dict(sorted(source_records.items())),
        "source_registry_sha256": source_registry.registry_sha256,
        "split_records": dict(sorted(split_records.items())),
        "supervised_tokens": supervised_tokens,
        "task_records": dict(sorted(task_records.items())),
        "validation_report_sha256": validation_report["report_sha256"],
    }
    manifest["seal_sha256"] = _sha256(manifest)
    return manifest
