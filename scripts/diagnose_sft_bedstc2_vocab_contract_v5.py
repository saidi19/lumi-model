#!/usr/bin/env python3
"""Run the authorized one-open, aggregate-only Be-DSTC2 vocab diagnostic."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

AUTHORIZATION_TEXT = (
    "I approve one bounded local-only Be-DSTC2 vocab-contract diagnostic and conditional "
    "curation recovery successor. Bind v0.5 plan SHA-256 "
    "0f777235bce28b958dacd0df5d3bbcb2d567510ca497432945c53c86399e4995, "
    "implementation SHA-256 "
    "db46a8babe0c4653d00f327725bc9ee339d997b79d4dcad6b261a1e064373822, runner "
    "SHA-256 f233e6889df770139a4bc239cf43299045fc27ca644af9061922833857381e9f, "
    "verifier SHA-256 "
    "20101fc6603faeb5f3f9762f89dba523fdf2038b0a7a164bc6b19f10242c8262, test "
    "SHA-256 ec01aa08f5e02db5c7f52018a22c64dc68a7661aab53d4e9fee2d844b2559ccd, "
    "and exact 47-byte attempt-004 marker SHA-256 "
    "a0f713fcb3ad48e68c857a3481a94e7cfdde83f3a49a3a761c8ce927aa070142. "
    "Preserve all prior evidence byte-for-byte. Read exactly once only the bound local "
    "19,767-byte vocab_splits.json SHA-256 "
    "f55a71418e699ece420b0206c74422f1889fbc9da24be4aea19c15d6912a6870 and retain "
    "only schema/type/count/overlap cardinalities, never lexical values or raw text. Use "
    "zero train/dev source-line, candidate, or baseline reads for the diagnostic. If and "
    "only if it confirms a load_vocab schema/key/overlap-contract mismatch, repair only "
    "that loader contract with synthetic tests, create a fresh v0.6/v5 attempt-005 "
    "successor, and execute exactly two sequential passes capped at 227,436 source-line "
    "reads, 629,432 existing-candidate reads, 40,000 new-candidate reads, 669,432 "
    "candidate-audit reads, 65 runtime-minutes, 134,283,264 private-write bytes, and zero "
    "network/GPU/inference. If the existing loader contract passes, stop fail-closed. No "
    "retry after a started pass or unrelated action."
)
AUTHORIZATION_TEXT_SHA256 = (
    "b9f2d20769d3c247954d4eeea32bf81114f0a18c6d2185f4839730c4542b7f64"
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = "scripts/diagnose_sft_bedstc2_vocab_contract_v5.py"
TEST_PATH = "tests/test_bedstc2_vocab_contract_diagnostic_v5.py"
REPORT_PATH = "evidence/lumi-4b-sft-bedstc2-vocab-contract-diagnostic-v0.8.json"
VERIFICATION_PATH = (
    "evidence/lumi-4b-sft-bedstc2-vocab-contract-diagnostic-verification-v0.8.json"
)
VOCAB_PATH = (
    "artifacts/lumi-sft-bedstc2-banglish-conversation-v0.1/attempt-001/private/raw/"
    "data/bengali/vocab_splits.json"
)
VOCAB_BYTES = 19_767
VOCAB_SHA256 = "f55a71418e699ece420b0206c74422f1889fbc9da24be4aea19c15d6912a6870"
VOCAB_GIT_BLOB_SHA1 = "39bba927098a8792ff3b4f611bf35362eb0b8e4f"
REPORT_MARKER = b'{"started":"bedstc2_vocab_contract_diagnostic_v0.8"}\n'
MAXIMUM_RUNTIME_SECONDS = 300
MAXIMUM_WRITE_BYTES = 131_072

PREDECESSOR_BINDINGS = {
    "implementation": (
        "src/lumi/posttraining_bedstc2_banglish_conversation_curation_v4.py",
        "db46a8babe0c4653d00f327725bc9ee339d997b79d4dcad6b261a1e064373822",
    ),
    "plan": (
        "configs/lumi-4b-sft-bedstc2-banglish-conversation-curation-recovery-v0.5.json",
        "0f777235bce28b958dacd0df5d3bbcb2d567510ca497432945c53c86399e4995",
    ),
    "runner": (
        "scripts/curate_sft_bedstc2_banglish_conversation_v4.py",
        "f233e6889df770139a4bc239cf43299045fc27ca644af9061922833857381e9f",
    ),
    "runner_marker": (
        "evidence/lumi-4b-sft-bedstc2-banglish-conversation-curation-v0.5.json",
        "a0f713fcb3ad48e68c857a3481a94e7cfdde83f3a49a3a761c8ce927aa070142",
    ),
    "test": (
        "tests/test_bedstc2_banglish_conversation_curation_v4.py",
        "ec01aa08f5e02db5c7f52018a22c64dc68a7661aab53d4e9fee2d844b2559ccd",
    ),
    "verifier": (
        "scripts/verify_sft_bedstc2_banglish_conversation_v4.py",
        "20101fc6603faeb5f3f9762f89dba523fdf2038b0a7a164bc6b19f10242c8262",
    ),
}
PREDECESSOR_ABSENT = (
    "artifacts/lumi-sft-bedstc2-banglish-conversation-v0.1/attempt-004/private/"
    "bedstc2-banglish-conversation-candidate.jsonl",
    "artifacts/lumi-sft-bedstc2-banglish-conversation-v0.1/attempt-004/private/"
    "owner-review-32.html",
    "artifacts/lumi-sft-bedstc2-banglish-conversation-v0.1/attempt-004/private/"
    "owner-review-32.json",
    "artifacts/lumi-sft-bedstc2-banglish-conversation-v0.1/attempt-004/private/"
    "rejection-ledger.jsonl",
    "artifacts/lumi-sft-bedstc2-banglish-conversation-v0.1/attempt-004/private/"
    "telemetry.jsonl",
    "evidence/lumi-4b-sft-bedstc2-banglish-conversation-curation-verification-v0.5.json",
)

EXPECTED_KEYS = (
    "english_language_vocab",
    "native_language_vocab",
    "others_vocab",
)
LOGICAL_ROLES = ("english", "native", "others")
# Closed, pre-read schema aliases only. Recovery is never inferred from lexical values.
ROLE_KEY_ALIASES = {
    "english": frozenset(
        {
            "en_vocab",
            "eng_vocab",
            "english_language_vocab",
            "english_vocab",
        }
    ),
    "native": frozenset(
        {
            "bangla_vocab",
            "ben_vocab",
            "bengali_language_vocab",
            "bengali_vocab",
            "bn_vocab",
            "native_language_vocab",
            "native_vocab",
        }
    ),
    "others": frozenset(
        {
            "common_vocab",
            "other_language_vocab",
            "other_vocab",
            "others_vocab",
            "rest_vocab",
            "shared_vocab",
            "univ_vocab",
            "universal_vocab",
        }
    ),
}
WORD = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*", re.IGNORECASE)
FORBIDDEN_RAW_KEYS = {
    "assistant",
    "assistant_response",
    "content",
    "messages",
    "token",
    "tokens",
    "user",
    "user_prompt",
    "value",
    "values",
}


class ObjectPairs(list[tuple[str, Any]]):
    """Preserve object pairs so duplicate top-level keys remain observable."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _internal_sha(value: dict[str, Any], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field, None)
    payload = json.dumps(
        unsigned, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return _sha256_bytes(payload)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _finish_marker(path: Path, payload: bytes) -> None:
    if path.read_bytes() != REPORT_MARKER:
        raise ValueError("diagnostic marker changed")
    with path.open("r+b") as handle:
        handle.seek(0)
        handle.truncate()
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _json_type(value: Any) -> str:
    if isinstance(value, ObjectPairs):
        return "object"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return "unknown"


def _raw_key_scan(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).casefold() in FORBIDDEN_RAW_KEYS:
                raise ValueError("raw-text key entered vocab diagnostic evidence")
            _raw_key_scan(nested)
    elif isinstance(value, list):
        for nested in value:
            _raw_key_scan(nested)


def _logical_role_mapping(schema_keys: set[str]) -> tuple[dict[str, str | None], bool]:
    candidates = {
        role: sorted(schema_keys & ROLE_KEY_ALIASES[role]) for role in LOGICAL_ROLES
    }
    mapping = {
        role: values[0] if len(values) == 1 else None
        for role, values in candidates.items()
    }
    selected = {key for key in mapping.values() if key is not None}
    unambiguous = (
        len(schema_keys) == 3
        and all(len(candidates[role]) == 1 for role in LOGICAL_ROLES)
        and selected == schema_keys
    )
    return mapping, unambiguous


def _derive_loader_contract(
    *,
    root_type: str,
    schema_keys: set[str],
    duplicate_key_count: int,
    clean_list_fields: set[str],
    exclusive: dict[str, int],
    normalized_cardinalities: dict[str, int],
) -> tuple[dict[str, Any], dict[str, str | None]]:
    mapping, mapping_unambiguous = _logical_role_mapping(schema_keys)
    key_set_matches = schema_keys == set(EXPECTED_KEYS)
    current_shape_supported = (
        root_type == "object"
        and duplicate_key_count == 0
        and key_set_matches
        and clean_list_fields == set(EXPECTED_KEYS)
    )
    current_contract_passes = (
        current_shape_supported
        and exclusive.get("english_language_vocab", 0) > 0
        and exclusive.get("native_language_vocab", 0) > 0
    )
    mapped_shape_supported = (
        root_type == "object"
        and duplicate_key_count == 0
        and mapping_unambiguous
        and clean_list_fields == schema_keys
    )
    mapped_anchor_supported = False
    if mapped_shape_supported:
        english_key = mapping["english"]
        native_key = mapping["native"]
        others_key = mapping["others"]
        if english_key is None or native_key is None or others_key is None:
            raise AssertionError("unambiguous role mapping contained a null key")
        mapped_anchor_supported = (
            exclusive.get(english_key, 0) > 0
            and exclusive.get(native_key, 0) > 0
            and normalized_cardinalities.get(others_key, 0) > 0
        )

    mismatch_codes: list[str] = []
    if root_type != "object":
        mismatch_codes.append("root_type_mismatch")
    if duplicate_key_count:
        mismatch_codes.append("duplicate_top_level_keys")
    if not key_set_matches:
        mismatch_codes.append("top_level_keys_mismatch")
    if root_type == "object" and not mapping_unambiguous:
        mismatch_codes.append("logical_role_mapping_ambiguous_or_unknown")
    if clean_list_fields != schema_keys:
        mismatch_codes.append("field_shape_mismatch")
    if mapped_shape_supported and not mapped_anchor_supported:
        mismatch_codes.append("exclusive_anchor_empty")

    repairable_key_mismatch = (
        mapped_shape_supported
        and mapped_anchor_supported
        and not key_set_matches
    )
    loader = {
        "expected_top_level_keys": list(EXPECTED_KEYS),
        "logical_role_mapping_unambiguous": mapping_unambiguous,
        "mismatch_codes": sorted(set(mismatch_codes)),
        "passes": current_contract_passes,
        "repairable_key_mismatch": repairable_key_mismatch,
    }
    return loader, mapping


def analyze_payload(payload: bytes) -> dict[str, Any]:
    try:
        decoded = payload.decode("utf-8", errors="strict")
        parsed = json.loads(decoded, object_pairs_hook=ObjectPairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("vocab JSON framing changed") from error

    root_type = _json_type(parsed)
    root_pairs = list(parsed) if isinstance(parsed, ObjectPairs) else []
    root_keys = [key for key, _value in root_pairs]
    duplicate_key_count = len(root_keys) - len(set(root_keys))
    mapping = {key: value for key, value in root_pairs}
    field_evidence: list[dict[str, Any]] = []
    token_sets: dict[str, frozenset[str]] = {}
    clean_list_fields: set[str] = set()
    for index, key in enumerate(sorted(mapping)):
        field = mapping[key]
        evidence: dict[str, Any] = {
            "field_index": index,
            "json_type": _json_type(field),
            "schema_key": key,
        }
        if isinstance(field, list) and not isinstance(field, ObjectPairs):
            type_counts = Counter(_json_type(item) for item in field)
            string_items = [item for item in field if isinstance(item, str)]
            normalized_occurrences: list[str] = []
            no_lexical = 0
            for item in string_items:
                lexical = [token.casefold() for token in WORD.findall(item)]
                normalized_occurrences.extend(lexical)
                if not lexical:
                    no_lexical += 1
            normalized = frozenset(normalized_occurrences)
            token_sets[key] = normalized
            empty_strings = sum(item == "" for item in string_items)
            whitespace_strings = sum(item != item.strip() for item in string_items)
            evidence.update(
                {
                    "duplicate_normalized_token_occurrences": (
                        len(normalized_occurrences) - len(normalized)
                    ),
                    "empty_string_items": empty_strings,
                    "item_count": len(field),
                    "item_type_counts": dict(sorted(type_counts.items())),
                    "no_lexical_token_string_items": no_lexical,
                    "normalized_token_occurrences": len(normalized_occurrences),
                    "normalized_unique_token_cardinality": len(normalized),
                    "outer_whitespace_string_items": whitespace_strings,
                }
            )
            if (
                field
                and len(string_items) == len(field)
                and empty_strings == 0
                and whitespace_strings == 0
                and no_lexical == 0
                and normalized
            ):
                clean_list_fields.add(key)
        field_evidence.append(evidence)

    pairwise = [
        {
            "intersection_cardinality": len(token_sets[left] & token_sets[right]),
            "left_schema_key": left,
            "right_schema_key": right,
        }
        for left, right in combinations(sorted(token_sets), 2)
    ]
    all_field_intersection = 0
    if token_sets:
        iterator = iter(token_sets.values())
        intersection = set(next(iterator))
        for values in iterator:
            intersection.intersection_update(values)
        all_field_intersection = len(intersection)
    exclusive = {
        key: len(
            values
            - frozenset().union(
                *(other for other_key, other in token_sets.items() if other_key != key)
            )
        )
        for key, values in sorted(token_sets.items())
    }

    normalized_cardinalities = {
        field["schema_key"]: field["normalized_unique_token_cardinality"]
        for field in field_evidence
        if field["json_type"] == "array"
    }
    loader, logical_mapping = _derive_loader_contract(
        root_type=root_type,
        schema_keys=set(root_keys),
        duplicate_key_count=duplicate_key_count,
        clean_list_fields=clean_list_fields,
        exclusive=exclusive,
        normalized_cardinalities=normalized_cardinalities,
    )
    result = {
        "all_field_intersection_cardinality": all_field_intersection,
        "current_loader_contract": loader,
        "duplicate_top_level_key_count": duplicate_key_count,
        "exclusive_normalized_token_cardinalities": exclusive,
        "field_evidence": field_evidence,
        "logical_role_mapping": logical_mapping,
        "pairwise_intersections": pairwise,
        "root_json_type": root_type,
        "top_level_key_count": len(root_keys),
        "top_level_keys": sorted(set(root_keys)),
        "unique_top_level_key_count": len(set(root_keys)),
    }
    _raw_key_scan(result)
    return result


def validate_report(report: dict[str, Any]) -> None:
    if set(report) != {
        "authorization_text_sha256",
        "bindings",
        "conditional_recovery_authorized",
        "diagnostic",
        "limits",
        "prohibited_actions",
        "raw_text_retained",
        "report_sha256",
        "schema_version",
        "status",
        "vocab_object",
    }:
        raise ValueError("vocab diagnostic report inventory changed")
    if (
        report.get("schema_version") != "lumi.sft-bedstc2-vocab-contract-diagnostic.v0.8"
        or report.get("authorization_text_sha256") != AUTHORIZATION_TEXT_SHA256
        or report.get("raw_text_retained") is not False
        or report.get("vocab_object")
        != {
            "bytes": VOCAB_BYTES,
            "git_blob_sha1": VOCAB_GIT_BLOB_SHA1,
            "path": "data/bengali/vocab_splits.json",
            "sha256": VOCAB_SHA256,
        }
        or report.get("limits")
        != {
            "baseline_row_reads": 0,
            "candidate_row_reads": 0,
            "maximum_runtime_seconds": MAXIMUM_RUNTIME_SECONDS,
            "maximum_write_bytes": MAXIMUM_WRITE_BYTES,
            "source_line_reads": 0,
            "vocab_content_opens": 1,
            "vocab_payload_bytes": VOCAB_BYTES,
        }
        or report.get("prohibited_actions")
        != {
            "gpu_used": False,
            "inference_called": False,
            "network_used": False,
            "registry_mutated": False,
            "source_mutated": False,
            "test_object_accessed": False,
            "train_dev_opened": False,
        }
        or report.get("report_sha256") != _internal_sha(report, "report_sha256")
    ):
        raise ValueError("vocab diagnostic report boundary changed")
    diagnostic = report.get("diagnostic")
    if not isinstance(diagnostic, dict) or set(diagnostic) != {
        "all_field_intersection_cardinality",
        "current_loader_contract",
        "duplicate_top_level_key_count",
        "exclusive_normalized_token_cardinalities",
        "field_evidence",
        "logical_role_mapping",
        "pairwise_intersections",
        "root_json_type",
        "top_level_key_count",
        "top_level_keys",
        "unique_top_level_key_count",
    }:
        raise ValueError("vocab diagnostic aggregate inventory changed")
    bindings = report.get("bindings")
    if (
        not isinstance(bindings, dict)
        or set(bindings) != {"diagnostic_code", "predecessor"}
        or not isinstance(bindings.get("diagnostic_code"), dict)
        or set(bindings["diagnostic_code"]) != {"script", "test"}
        or not isinstance(bindings.get("predecessor"), dict)
        or set(bindings["predecessor"]) != set(PREDECESSOR_BINDINGS)
    ):
        raise ValueError("vocab diagnostic binding inventory changed")
    for item in [*bindings["diagnostic_code"].values(), *bindings["predecessor"].values()]:
        if (
            not isinstance(item, dict)
            or set(item) != {"bytes", "path", "sha256"}
            or not isinstance(item.get("bytes"), int)
            or item["bytes"] <= 0
            or not isinstance(item.get("path"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256")))
        ):
            raise ValueError("vocab diagnostic binding fields changed")
    expected_code_paths = {"script": SCRIPT_PATH, "test": TEST_PATH}
    for name, expected_path in expected_code_paths.items():
        item = bindings["diagnostic_code"][name]
        live_path = ROOT / expected_path
        if (
            item["path"] != expected_path
            or item["bytes"] != live_path.stat().st_size
            or item["sha256"] != _sha256_file(live_path)
        ):
            raise ValueError("vocab diagnostic code binding changed")
    for name, (expected_path, expected_sha256) in PREDECESSOR_BINDINGS.items():
        item = bindings["predecessor"][name]
        if (
            item["path"] != expected_path
            or item["sha256"] != expected_sha256
            or item["bytes"] != (ROOT / expected_path).stat().st_size
            or (name == "runner_marker" and item["bytes"] != 47)
        ):
            raise ValueError("vocab diagnostic predecessor binding changed")
    root_keys = diagnostic.get("top_level_keys")
    fields = diagnostic.get("field_evidence")
    exclusive = diagnostic.get("exclusive_normalized_token_cardinalities")
    pairwise = diagnostic.get("pairwise_intersections")
    if (
        not isinstance(root_keys, list)
        or root_keys != sorted(root_keys)
        or len(root_keys) != len(set(root_keys))
        or any(not isinstance(key, str) or not key for key in root_keys)
        or diagnostic.get("root_json_type")
        not in {"array", "boolean", "integer", "null", "number", "object", "string"}
        or not isinstance(diagnostic.get("top_level_key_count"), int)
        or diagnostic["top_level_key_count"] < len(root_keys)
        or diagnostic.get("unique_top_level_key_count") != len(root_keys)
        or diagnostic.get("duplicate_top_level_key_count")
        != diagnostic["top_level_key_count"] - len(root_keys)
        or not isinstance(fields, list)
        or len(fields) != len(root_keys)
        or not isinstance(exclusive, dict)
        or not isinstance(pairwise, list)
        or not isinstance(diagnostic.get("all_field_intersection_cardinality"), int)
        or diagnostic["all_field_intersection_cardinality"] < 0
    ):
        raise ValueError("vocab diagnostic aggregate arithmetic changed")
    array_keys: set[str] = set()
    clean_list_fields: set[str] = set()
    normalized_cardinalities: dict[str, int] = {}
    for index, field in enumerate(fields):
        if (
            not isinstance(field, dict)
            or field.get("field_index") != index
            or field.get("schema_key") != root_keys[index]
            or field.get("json_type")
            not in {"array", "boolean", "integer", "null", "number", "object", "string"}
        ):
            raise ValueError("vocab diagnostic field descriptor changed")
        base_keys = {"field_index", "json_type", "schema_key"}
        if field["json_type"] == "array":
            array_keys.add(field["schema_key"])
            expected = base_keys | {
                "duplicate_normalized_token_occurrences",
                "empty_string_items",
                "item_count",
                "item_type_counts",
                "no_lexical_token_string_items",
                "normalized_token_occurrences",
                "normalized_unique_token_cardinality",
                "outer_whitespace_string_items",
            }
            if set(field) != expected or any(
                not isinstance(field[key], int) or field[key] < 0
                for key in expected - base_keys - {"item_type_counts"}
            ):
                raise ValueError("vocab diagnostic array descriptor changed")
            item_types = field["item_type_counts"]
            if (
                not isinstance(item_types, dict)
                or any(
                    key
                    not in {
                        "array",
                        "boolean",
                        "integer",
                        "null",
                        "number",
                        "object",
                        "string",
                    }
                    or not isinstance(count, int)
                    or count < 0
                    for key, count in item_types.items()
                )
                or sum(item_types.values()) != field["item_count"]
                or field["normalized_unique_token_cardinality"]
                > field["normalized_token_occurrences"]
                or field["duplicate_normalized_token_occurrences"]
                != field["normalized_token_occurrences"]
                - field["normalized_unique_token_cardinality"]
                or field["empty_string_items"] > item_types.get("string", 0)
                or field["outer_whitespace_string_items"] > item_types.get("string", 0)
                or field["no_lexical_token_string_items"] > item_types.get("string", 0)
            ):
                raise ValueError("vocab diagnostic array arithmetic changed")
            normalized_cardinalities[field["schema_key"]] = field[
                "normalized_unique_token_cardinality"
            ]
            if (
                field["item_count"] > 0
                and item_types == {"string": field["item_count"]}
                and field["empty_string_items"] == 0
                and field["outer_whitespace_string_items"] == 0
                and field["no_lexical_token_string_items"] == 0
                and field["normalized_unique_token_cardinality"] > 0
            ):
                clean_list_fields.add(field["schema_key"])
        elif set(field) != base_keys:
            raise ValueError("vocab diagnostic non-array descriptor changed")
    if set(exclusive) != array_keys or any(
        not isinstance(count, int)
        or count < 0
        or count > normalized_cardinalities[key]
        for key, count in exclusive.items()
    ):
        raise ValueError("vocab diagnostic exclusive-cardinality inventory changed")
    expected_pairs = {tuple(pair) for pair in combinations(sorted(array_keys), 2)}
    observed_pairs: set[tuple[str, str]] = set()
    pair_cardinalities: dict[tuple[str, str], int] = {}
    for item in pairwise:
        if (
            not isinstance(item, dict)
            or set(item)
            != {"intersection_cardinality", "left_schema_key", "right_schema_key"}
            or not isinstance(item.get("intersection_cardinality"), int)
            or item["intersection_cardinality"] < 0
            or not isinstance(item.get("left_schema_key"), str)
            or not isinstance(item.get("right_schema_key"), str)
            or item["left_schema_key"] >= item["right_schema_key"]
            or item["left_schema_key"] not in array_keys
            or item["right_schema_key"] not in array_keys
            or item["intersection_cardinality"]
            > min(
                normalized_cardinalities[item["left_schema_key"]],
                normalized_cardinalities[item["right_schema_key"]],
            )
        ):
            raise ValueError("vocab diagnostic pairwise evidence changed")
        pair = (item["left_schema_key"], item["right_schema_key"])
        observed_pairs.add(pair)
        pair_cardinalities[pair] = item["intersection_cardinality"]
    if observed_pairs != expected_pairs or len(observed_pairs) != len(pairwise):
        raise ValueError("vocab diagnostic pairwise inventory changed")
    all_intersection = diagnostic["all_field_intersection_cardinality"]
    if all_intersection > (
        min(normalized_cardinalities.values()) if normalized_cardinalities else 0
    ) or any(all_intersection > overlap for overlap in pair_cardinalities.values()):
        raise ValueError("vocab diagnostic all-field intersection changed")
    ordered_arrays = sorted(array_keys)
    if len(ordered_arrays) == 0 and all_intersection != 0:
        raise ValueError("vocab diagnostic empty intersection changed")
    if len(ordered_arrays) == 1:
        only = ordered_arrays[0]
        if (
            all_intersection != normalized_cardinalities[only]
            or exclusive[only] != normalized_cardinalities[only]
        ):
            raise ValueError("vocab diagnostic singleton cardinality changed")
    if len(ordered_arrays) == 2:
        left, right = ordered_arrays
        overlap = pair_cardinalities[(left, right)]
        if (
            all_intersection != overlap
            or exclusive[left] != normalized_cardinalities[left] - overlap
            or exclusive[right] != normalized_cardinalities[right] - overlap
        ):
            raise ValueError("vocab diagnostic two-field cardinality changed")
    if len(ordered_arrays) == 3:
        for key in ordered_arrays:
            others = [other for other in ordered_arrays if other != key]
            first_pair = tuple(sorted((key, others[0])))
            second_pair = tuple(sorted((key, others[1])))
            expected_exclusive = (
                normalized_cardinalities[key]
                - pair_cardinalities[first_pair]
                - pair_cardinalities[second_pair]
                + all_intersection
            )
            if exclusive[key] != expected_exclusive:
                raise ValueError("vocab diagnostic three-field cardinality changed")
    observed_mapping = diagnostic.get("logical_role_mapping")
    expected_loader, expected_mapping = _derive_loader_contract(
        root_type=diagnostic["root_json_type"],
        schema_keys=set(root_keys),
        duplicate_key_count=diagnostic["duplicate_top_level_key_count"],
        clean_list_fields=clean_list_fields,
        exclusive=exclusive,
        normalized_cardinalities=normalized_cardinalities,
    )
    if observed_mapping != expected_mapping:
        raise ValueError("vocab diagnostic logical-role mapping changed")
    loader = diagnostic.get("current_loader_contract")
    if not isinstance(loader, dict) or set(loader) != {
        "expected_top_level_keys",
        "logical_role_mapping_unambiguous",
        "mismatch_codes",
        "passes",
        "repairable_key_mismatch",
    } or loader != expected_loader:
        raise ValueError("vocab diagnostic loader evidence changed")
    conditional = report.get("conditional_recovery_authorized")
    expected_conditional = (
        loader["passes"] is False and loader["repairable_key_mismatch"] is True
    )
    if not isinstance(conditional, bool) or conditional is not expected_conditional:
        raise ValueError("vocab diagnostic conditional gate changed")
    expected_status = (
        "PASS_repairable_vocab_key_contract_mismatch"
        if conditional
        else "STOP_existing_or_unsupported_vocab_contract"
    )
    if report.get("status") != expected_status:
        raise ValueError("vocab diagnostic status changed")
    _raw_key_scan(report)


def build_bindings() -> dict[str, Any]:
    return {
        "diagnostic_code": {
            "script": {
                "bytes": (ROOT / SCRIPT_PATH).stat().st_size,
                "path": SCRIPT_PATH,
                "sha256": _sha256_file(ROOT / SCRIPT_PATH),
            },
            "test": {
                "bytes": (ROOT / TEST_PATH).stat().st_size,
                "path": TEST_PATH,
                "sha256": _sha256_file(ROOT / TEST_PATH),
            },
        },
        "predecessor": {
            name: {
                "bytes": (ROOT / relative).stat().st_size,
                "path": relative,
                "sha256": expected,
            }
            for name, (relative, expected) in sorted(PREDECESSOR_BINDINGS.items())
        },
    }


def build_report(diagnostic: dict[str, Any], bindings: dict[str, Any]) -> dict[str, Any]:
    conditional = (
        diagnostic["current_loader_contract"]["passes"] is False
        and diagnostic["current_loader_contract"]["repairable_key_mismatch"] is True
    )
    report: dict[str, Any] = {
        "authorization_text_sha256": AUTHORIZATION_TEXT_SHA256,
        "bindings": bindings,
        "conditional_recovery_authorized": conditional,
        "diagnostic": diagnostic,
        "limits": {
            "baseline_row_reads": 0,
            "candidate_row_reads": 0,
            "maximum_runtime_seconds": MAXIMUM_RUNTIME_SECONDS,
            "maximum_write_bytes": MAXIMUM_WRITE_BYTES,
            "source_line_reads": 0,
            "vocab_content_opens": 1,
            "vocab_payload_bytes": VOCAB_BYTES,
        },
        "prohibited_actions": {
            "gpu_used": False,
            "inference_called": False,
            "network_used": False,
            "registry_mutated": False,
            "source_mutated": False,
            "test_object_accessed": False,
            "train_dev_opened": False,
        },
        "raw_text_retained": False,
        "schema_version": "lumi.sft-bedstc2-vocab-contract-diagnostic.v0.8",
        "status": (
            "PASS_repairable_vocab_key_contract_mismatch"
            if conditional
            else "STOP_existing_or_unsupported_vocab_contract"
        ),
        "vocab_object": {
            "bytes": VOCAB_BYTES,
            "git_blob_sha1": VOCAB_GIT_BLOB_SHA1,
            "path": "data/bengali/vocab_splits.json",
            "sha256": VOCAB_SHA256,
        },
    }
    report["report_sha256"] = _internal_sha(report, "report_sha256")
    validate_report(report)
    return report


def build_verification(report: dict[str, Any], report_file_sha256: str) -> dict[str, Any]:
    validate_report(report)
    value: dict[str, Any] = {
        "authorization_text_sha256": AUTHORIZATION_TEXT_SHA256,
        "checks": {
            "aggregate_only": True,
            "bindings_reproduced": True,
            "candidate_baseline_and_train_dev_reads_zero": True,
            "raw_text_absent": True,
            "schema_and_cardinality_arithmetic_verified": True,
            "source_replay_not_performed": True,
            "vocab_open_count_exactly_one": True,
        },
        "conditional_recovery_authorized": report["conditional_recovery_authorized"],
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": report["report_sha256"],
        "schema_version": "lumi.sft-bedstc2-vocab-contract-diagnostic-verification.v0.8",
        "status": (
            "PASS_structurally_verified_repairable_vocab_key_mismatch"
            if report["conditional_recovery_authorized"]
            else "STOP_structurally_verified_no_conditional_recovery"
        ),
    }
    value["verification_sha256"] = _internal_sha(value, "verification_sha256")
    validate_verification(value, report, report_file_sha256)
    return value


def validate_verification(
    value: dict[str, Any], report: dict[str, Any], report_file_sha256: str
) -> None:
    validate_report(report)
    if set(value) != {
        "authorization_text_sha256",
        "checks",
        "conditional_recovery_authorized",
        "report_file_sha256",
        "report_internal_sha256",
        "schema_version",
        "status",
        "verification_sha256",
    }:
        raise ValueError("vocab diagnostic verification inventory changed")
    if (
        value.get("authorization_text_sha256") != AUTHORIZATION_TEXT_SHA256
        or value.get("schema_version")
        != "lumi.sft-bedstc2-vocab-contract-diagnostic-verification.v0.8"
        or not isinstance(value.get("conditional_recovery_authorized"), bool)
        or value["conditional_recovery_authorized"]
        is not report["conditional_recovery_authorized"]
        or not re.fullmatch(r"[0-9a-f]{64}", str(value.get("report_file_sha256")))
        or not re.fullmatch(r"[0-9a-f]{64}", str(value.get("report_internal_sha256")))
        or not re.fullmatch(r"[0-9a-f]{64}", str(value.get("verification_sha256")))
        or report_file_sha256 != _sha256_bytes(_json_bytes(report))
        or value.get("report_file_sha256") != report_file_sha256
        or value.get("report_internal_sha256") != report.get("report_sha256")
        or value.get("verification_sha256")
        != _internal_sha(value, "verification_sha256")
        or not isinstance(value.get("checks"), dict)
        or set(value["checks"])
        != {
            "aggregate_only",
            "bindings_reproduced",
            "candidate_baseline_and_train_dev_reads_zero",
            "raw_text_absent",
            "schema_and_cardinality_arithmetic_verified",
            "source_replay_not_performed",
            "vocab_open_count_exactly_one",
        }
        or not all(check is True for check in value["checks"].values())
    ):
        raise ValueError("vocab diagnostic verification boundary changed")
    expected_status = (
        "PASS_structurally_verified_repairable_vocab_key_mismatch"
        if value.get("conditional_recovery_authorized") is True
        else "STOP_structurally_verified_no_conditional_recovery"
    )
    if value.get("status") != expected_status:
        raise ValueError("vocab diagnostic verification status changed")
    _raw_key_scan(value)


def _preflight() -> None:
    if _sha256_bytes(AUTHORIZATION_TEXT.encode()) != AUTHORIZATION_TEXT_SHA256:
        raise ValueError("authorization text hash changed")
    for relative, expected in PREDECESSOR_BINDINGS.values():
        path = ROOT / relative
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError("predecessor binding changed")
    if any((ROOT / relative).exists() for relative in PREDECESSOR_ABSENT):
        raise ValueError("predecessor absent-output inventory changed")
    if (ROOT / REPORT_PATH).exists() or (ROOT / VERIFICATION_PATH).exists():
        raise FileExistsError("vocab diagnostic output already exists")
    vocab = ROOT / VOCAB_PATH
    if not vocab.is_file() or vocab.stat().st_size != VOCAB_BYTES:
        raise ValueError("vocab object metadata changed")
    if not (ROOT / TEST_PATH).is_file():
        raise ValueError("vocab diagnostic synthetic test is absent")


def _read_vocab_once(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read()


def _runtime_alarm(_signum: int, _frame: object) -> None:
    raise TimeoutError("vocab diagnostic runtime exceeded")


def main() -> int:
    started = time.monotonic()
    signal.signal(signal.SIGALRM, _runtime_alarm)
    signal.setitimer(signal.ITIMER_REAL, MAXIMUM_RUNTIME_SECONDS)
    _preflight()
    report_path = ROOT / REPORT_PATH
    verification_path = ROOT / VERIFICATION_PATH
    _write_exclusive(report_path, REPORT_MARKER)

    vocab_payload = _read_vocab_once(ROOT / VOCAB_PATH)
    if (
        len(vocab_payload) != VOCAB_BYTES
        or _sha256_bytes(vocab_payload) != VOCAB_SHA256
        or hashlib.sha1(f"blob {len(vocab_payload)}\0".encode() + vocab_payload).hexdigest()
        != VOCAB_GIT_BLOB_SHA1
    ):
        raise ValueError("vocab object content binding changed")
    diagnostic = analyze_payload(vocab_payload)
    conditional = (
        diagnostic["current_loader_contract"]["passes"] is False
        and diagnostic["current_loader_contract"]["repairable_key_mismatch"] is True
    )
    report = build_report(diagnostic, build_bindings())
    report_payload = _json_bytes(report)
    verification = build_verification(report, _sha256_bytes(report_payload))
    verification_payload = _json_bytes(verification)
    total_writes = len(REPORT_MARKER) + len(report_payload) + len(verification_payload)
    if total_writes > MAXIMUM_WRITE_BYTES or time.monotonic() - started > MAXIMUM_RUNTIME_SECONDS:
        raise ValueError("vocab diagnostic resource cap exceeded")
    _finish_marker(report_path, report_payload)
    _write_exclusive(verification_path, verification_payload)
    signal.setitimer(signal.ITIMER_REAL, 0)
    print(
        json.dumps(
            {
                "conditional_recovery_authorized": conditional,
                "report_file_sha256": _sha256_bytes(report_payload),
                "status": report["status"],
                "verification_file_sha256": _sha256_bytes(verification_payload),
            },
            sort_keys=True,
        )
    )
    return 0 if conditional else 2


def _entrypoint() -> int:
    try:
        return main()
    except Exception as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        print(
            json.dumps(
                {"exception_type": type(error).__name__, "status": "FAIL_CLOSED"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
