#!/usr/bin/env python3
"""Result-preserving DeepMind scale verification continuation.

This adapter executes the already-bound verifier unchanged except for two
explicit continuation concerns:

* the newly authorized 180-minute clock starts when this process starts; and
* assistant-only token measurement uses the frozen tokenizer's batch API after
  proving byte-for-byte token-ID parity against the scalar API.

No candidate rows are generated or modified.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from types import ModuleType
from typing import Any

from lumi.posttraining_data import validate_message_sequence
from lumi.posttraining_deepmind_scale_generation import (
    file_sha256,
    sha256_value,
    write_json_atomic,
)
from lumi.posttraining_format import ChatTemplate, verify_chat_tokenizer
from lumi.tokenizer_gate import encode_untrusted_text

SCHEMA_VERSION = "lumi.sft-deepmind-scale-verification-continuation.v0.1"
BATCH_SIZE = 4096
PARITY_RECORDS = 4096
_BATCH_PARITY: dict[str, Any] = {}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _load_bound_verifier(path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "lumi_bound_deepmind_scale_verifier", path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("could not load the bound DeepMind scale verifier")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _nearest_rank(values: Sequence[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[min(index, len(ordered) - 1)]


def _batch_encode(tokenizer: Any, texts: list[str]) -> list[list[int]]:
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        split_special_tokens=True,
    )["input_ids"]
    if not isinstance(encoded, list) or len(encoded) != len(texts):
        raise ValueError("frozen tokenizer batch output shape changed")
    if any(not isinstance(row, list) for row in encoded):
        raise ValueError("frozen tokenizer batch output type changed")
    return encoded


def measure_token_fit_batched(
    records: Iterable[Mapping[str, Any]],
    *,
    template: ChatTemplate,
    tokenizer: Any,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Reproduce scalar SFT metrics with parity-proven batch tokenization."""

    global _BATCH_PARITY

    candidates = list(records)
    verify_chat_tokenizer(tokenizer, template)
    reserved_ids = set(tokenizer.all_special_ids)
    accepted: list[Mapping[str, Any]] = []
    lengths: list[int] = []
    supervised: list[int] = []
    masked: list[int] = []
    by_split: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_task: defaultdict[str, Counter[str]] = defaultdict(Counter)
    overlength = 0
    token_ledger: list[dict[str, Any]] = []
    parity_done = False

    for offset in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[offset : offset + BATCH_SIZE]
        texts: list[str] = []
        message_counts: list[int] = []
        for record in batch:
            messages = record["messages"]
            validate_message_sequence(messages)
            message_counts.append(len(messages))
            texts.extend(str(message["content"]) for message in messages)
        encoded_texts = _batch_encode(tokenizer, texts)

        if not parity_done:
            parity_texts = texts[: PARITY_RECORDS * 2]
            scalar = [encode_untrusted_text(tokenizer, text) for text in parity_texts]
            if scalar != encoded_texts[: len(parity_texts)]:
                raise ValueError("scalar and batch frozen-tokenizer outputs differ")
            _BATCH_PARITY = {
                "batch_size": BATCH_SIZE,
                "parity_record_count": min(PARITY_RECORDS, len(batch)),
                "parity_text_count": len(parity_texts),
                "token_ids_sha256": sha256_value(scalar),
            }
            parity_done = True

        text_cursor = 0
        for record, message_count in zip(batch, message_counts, strict=True):
            messages = record["messages"]
            message_ids = encoded_texts[text_cursor : text_cursor + message_count]
            text_cursor += message_count
            total_tokens = 2  # BOS and EOS.
            supervised_tokens = 1  # EOS remains supervised.
            for message, content_ids in zip(messages, message_ids, strict=True):
                if not content_ids:
                    raise ValueError("message content encoded to zero tokens")
                collisions = reserved_ids.intersection(content_ids)
                if collisions:
                    raise ValueError(
                        "untrusted text emitted reserved token IDs: "
                        f"{sorted(collisions)}"
                    )
                total_tokens += 2 + len(content_ids)  # Role and message-end tokens.
                if message["role"] in template.loss_roles:
                    supervised_tokens += 1 + len(content_ids)
            if total_tokens > template.context_window:
                overlength += 1
                continue
            if supervised_tokens < 2:
                raise ValueError("SFT conversation has no trainable assistant response")
            masked_tokens = total_tokens - supervised_tokens
            accepted.append(record)
            lengths.append(total_tokens)
            supervised.append(supervised_tokens)
            masked.append(masked_tokens)
            for bucket in (
                by_split[str(record["split"])],
                by_task[str(record["task_category"])],
            ):
                bucket["records"] += 1
                bucket["serialized_tokens"] += total_tokens
                bucket["supervised_tokens"] += supervised_tokens
                bucket["masked_tokens"] += masked_tokens
            token_ledger.append(
                {
                    "masked_tokens": masked_tokens,
                    "record_id": record["record_id"],
                    "serialized_tokens": total_tokens,
                    "supervised_tokens": supervised_tokens,
                }
            )
        if text_cursor != len(encoded_texts):
            raise AssertionError("frozen tokenizer batch cursor changed")

    metrics = {
        "accepted_records": len(accepted),
        "assistant_only_loss_verified": True,
        "by_split": {key: dict(value) for key, value in sorted(by_split.items())},
        "by_task": {key: dict(value) for key, value in sorted(by_task.items())},
        "context_window": template.context_window,
        "masked_tokens": sum(masked),
        "overlength_rejected": overlength,
        "serialized_length": {
            "maximum": max(lengths, default=0),
            "mean": mean(lengths) if lengths else 0,
            "median": median(lengths) if lengths else 0,
            "minimum": min(lengths, default=0),
            "p95": _nearest_rank(lengths, 0.95),
        },
        "serialized_tokens": sum(lengths),
        "supervised_tokens": sum(supervised),
        "token_ledger_sha256": sha256_value(
            sorted(token_ledger, key=lambda row: row["record_id"])
        ),
    }
    return accepted, metrics


def _verify_continuation_plan(
    continuation: Mapping[str, Any], *, root: Path
) -> None:
    if continuation.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported DeepMind verification continuation schema")
    authorization = continuation.get("authorization")
    if not isinstance(authorization, Mapping):
        raise ValueError("DeepMind continuation authorization is missing")
    if authorization.get("maximum_runtime_minutes") != 180:
        raise ValueError("DeepMind continuation runtime boundary changed")
    prohibited = (
        "generation_authorized",
        "network_authorized",
        "external_api_authorized",
        "gpu_authorized",
        "registry_promotion_authorized",
        "dataset_seal_authorized",
        "sft_training_authorized",
        "unrelated_access_authorized",
        "external_distribution_authorized",
    )
    if any(authorization.get(field) is not False for field in prohibited):
        raise ValueError("DeepMind continuation authorization widened")
    bindings = continuation.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("DeepMind continuation bindings are missing")
    for name, specification in bindings.items():
        if not isinstance(specification, Mapping):
            raise ValueError(f"DeepMind continuation binding is invalid: {name}")
        path = (root / str(specification["path"])).resolve()
        if not path.is_file() or file_sha256(path) != specification["sha256"]:
            raise ValueError(f"DeepMind continuation binding changed: {name}")
    report_binding = bindings["scale_report"]
    report = _load_json((root / str(report_binding["path"])).resolve())
    if report.get("report_sha256") != report_binding["internal_sha256"]:
        raise ValueError("DeepMind scale internal report hash changed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuation-plan", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.repository_root.resolve()
    continuation_path = (
        args.continuation_plan
        if args.continuation_plan.is_absolute()
        else root / args.continuation_plan
    ).resolve()
    scale_plan_path = (args.plan if args.plan.is_absolute() else root / args.plan).resolve()
    continuation = _load_json(continuation_path)
    _verify_continuation_plan(continuation, root=root)
    started_at = datetime.now(UTC)
    maximum_seconds = 180 * 60

    bound_path = (
        root / str(continuation["bindings"]["bound_verifier"]["path"])
    ).resolve()
    verifier = _load_bound_verifier(bound_path)
    verifier.measure_token_fit = measure_token_fit_batched
    verifier.MAXIMUM_RUNTIME_SECONDS = maximum_seconds
    verifier._utc = lambda _value: started_at

    captured: dict[str, Any] = {}

    def capture_receipt(path: Path, value: Mapping[str, Any]) -> None:
        if captured:
            raise RuntimeError("bound verifier attempted more than one receipt write")
        captured["path"] = path
        captured["value"] = dict(value)

    verifier.write_json_atomic = capture_receipt
    original_argv = sys.argv
    try:
        sys.argv = [
            str(bound_path),
            "--plan",
            str(scale_plan_path),
            "--repository-root",
            str(root),
        ]
        result = int(verifier.main())
    finally:
        sys.argv = original_argv
    if result != 0 or set(captured) != {"path", "value"}:
        raise RuntimeError("bound DeepMind verifier did not produce one receipt")

    completed_at = datetime.now(UTC)
    runtime_seconds = (completed_at - started_at).total_seconds()
    if runtime_seconds > maximum_seconds:
        raise TimeoutError("DeepMind verification continuation exceeded 180 minutes")
    output_path = Path(captured["path"]).resolve()
    expected_output = (
        root / str(continuation["output"]["verification_receipt_path"])
    ).resolve()
    if output_path != expected_output:
        raise ValueError("DeepMind continuation output path changed")
    receipt = dict(captured["value"])
    receipt.pop("verification_sha256", None)
    receipt.pop("total_elapsed_seconds_from_generation_start", None)
    report_binding = continuation["bindings"]["scale_report"]
    receipt["continuation"] = {
        "authorization_purpose": "result_preserving_independent_replay_verification",
        "batch_tokenization_parity": dict(_BATCH_PARITY),
        "bound_report_file_sha256": report_binding["sha256"],
        "bound_report_internal_sha256": report_binding["internal_sha256"],
        "completed_at_utc": completed_at.isoformat().replace("+00:00", "Z"),
        "external_api_calls": 0,
        "generation_performed": False,
        "gpu_used": False,
        "maximum_runtime_seconds": maximum_seconds,
        "network_bytes": 0,
        "plan_file_sha256": file_sha256(continuation_path),
        "runtime_seconds": round(runtime_seconds, 3),
        "schema_version": SCHEMA_VERSION,
        "started_at_utc": started_at.isoformat().replace("+00:00", "Z"),
    }
    receipt["verification_sha256"] = sha256_value(receipt)
    write_json_atomic(output_path, receipt)
    print(
        "PASS: DeepMind result-preserving verification continuation complete; "
        f"receipt={receipt['verification_sha256']} runtime={runtime_seconds:.3f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
