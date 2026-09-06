"""Pure two-input URI delta proof, without file access or old pipeline execution.

The caller authenticates and bounds the original input bytes. This checker
compares actual parsed rows but does not load a tokenizer, rerun generic data
filters, authenticate owner approval, or authorize execution or production.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

SCHEMA_VERSION = "lumi.sft-safety-uri-row-delta-audit-report.v0.1"
SAFETY_SOURCE_ID = "lumi_safety_tool_sft_v0_1"
SAFETY_DATASET_ID = "Afnan-Technologies-Inc/lumi-safety-tool-sft-v0.1"
_RECORD_ID = re.compile(r"[A-Za-z0-9._-]+\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ERROR = "invalid safety URI row-delta contract"


def _require(condition: bool) -> None:
    if not condition:
        raise ValueError(_ERROR)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in items:
        _require(key not in result)
        result[key] = value
    return result


def _nonfinite(_value: str) -> None:
    raise ValueError(_ERROR)


def _rows(payload: bytes) -> list[dict[str, Any]]:
    _require(type(payload) is bytes)
    _require(payload.endswith(b"\n") and b"\r" not in payload)
    lines = payload[:-1].split(b"\n")
    _require(len(lines) == 320)
    result = []
    for line in lines:
        _require(bool(line.strip()))
        row = json.loads(line.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_nonfinite)
        _require(type(row) is dict)
        # Reject overflow-to-infinity floats and unpaired Unicode surrogates,
        # which json.loads can otherwise accept despite strict parse_constant.
        _canonical(row)
        result.append(row)
    return result


def _check_rows(before_rows: list[dict], after_rows: list[dict]) -> None:
    seen_ids: set[str] = set()
    seen_uris: set[str] = set()
    for before, after in zip(before_rows, after_rows, strict=True):
        record_id = before.get("record_id")
        _require(type(record_id) is str and _RECORD_ID.fullmatch(record_id) is not None)
        _require(record_id not in seen_ids)
        _require(type(after.get("record_id")) is str and after["record_id"] == record_id)
        seen_ids.add(record_id)

        before_source, after_source = before.get("source"), after.get("source")
        before_provenance, after_provenance = before.get("provenance"), after.get("provenance")
        _require(type(before_source) is dict and type(after_source) is dict)
        _require(type(before_provenance) is dict and type(after_provenance) is dict)
        for source in (before_source, after_source):
            _require(source.get("source_id") == SAFETY_SOURCE_ID)
            _require(source.get("dataset_id") == SAFETY_DATASET_ID)
            _require(source.get("document_id") == record_id)
        _require("uri" not in before_source)
        uri = f"private://{SAFETY_DATASET_ID}/{record_id}"
        _require(after_source.get("uri") == uri and uri not in seen_uris)
        seen_uris.add(uri)

        prior_transform = before_provenance.get("transform_sha256")
        _require(type(prior_transform) is str and _SHA256.fullmatch(prior_transform) is not None)
        # Exact frozen operation from posttraining_safety_uri_repair.py:350-355.
        transform_binding = {
            "operation": "add_deterministic_first_party_source_uri_v0.1",
            "prior_transform_sha256": prior_transform,
            "source_uri": uri,
        }
        expected_transform = hashlib.sha256(_canonical(transform_binding)).hexdigest()
        _require(after_provenance.get("transform_sha256") == expected_transform)

        comparison = dict(after)
        comparison["source"] = dict(after_source)
        del comparison["source"]["uri"]
        comparison["provenance"] = dict(after_provenance)
        comparison["provenance"]["transform_sha256"] = prior_transform
        # Canonical JSON preserves bool/int/float distinctions, unlike dict ==.
        _require(_canonical(comparison) == _canonical(before))


def audit_pairs(before_bytes: bytes, after_bytes: bytes) -> dict[str, Any]:
    """Check exactly 320 pinned-by-caller row pairs; emit only aggregate proof.

    Records must be nonblank, LF-terminated UTF-8 JSON objects without duplicate
    keys or nonfinite numbers. Field order and JSON whitespace are immaterial;
    every parsed value except the two precisely checked metadata deltas must
    remain identical, including its JSON primitive type.
    """
    try:
        _check_rows(_rows(before_bytes), _rows(after_bytes))
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError, OverflowError):
        raise ValueError(_ERROR) from None
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_ROW_DELTA_ONLY",
        "row_delta_verified": True,
        "rows_compared": 320,
        "unique_record_ids": 320,
        "unique_source_uris": 320,
        "tokenizer_recomputed": False,
        "generic_filtering_replayed": False,
        "full_private_integration_passed": False,
        "execution_authorized": False,
        "production_eligible": False,
    }
