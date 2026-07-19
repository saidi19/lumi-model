#!/usr/bin/env python3
"""Extract a tiny sealed real-token fixture for the distributed CUDA smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from array import array
from pathlib import Path
from typing import Any

from lumi.packed_data import validate_packed_manifest

EXPECTED_PARENT_BUILD_FINGERPRINT = (
    "0bce4d9e4a51de1b831b0226e0052ea33130733eae2bb61992c6eacaf203e0c8"
)
EXPECTED_PARENT_MANIFEST_SHA256 = (
    "9228fa610d4a985d05e15057e8ef12619a14cbedbb5a6a5ab513d3aeca2832c8"
)
EXPECTED_SEQUENCE_LENGTH = 4096
EXPECTED_VOCAB_SIZE = 65536
EXPECTED_DOC_TOKEN_ID = 3


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequences", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sequences < 8 or args.sequences > 256:
        raise ValueError("CUDA smoke fixture requires 8 through 256 sequences")
    source_manifest_path = args.source_manifest.resolve()
    source_manifest_bytes = source_manifest_path.read_bytes()
    source_manifest_sha256 = hashlib.sha256(source_manifest_bytes).hexdigest()
    if source_manifest_sha256 != EXPECTED_PARENT_MANIFEST_SHA256:
        raise ValueError("source manifest is not the sealed Lumi 1.4B-token pack")
    source = json.loads(source_manifest_bytes)
    if source.get("build_fingerprint") != EXPECTED_PARENT_BUILD_FINGERPRINT:
        raise ValueError("source packed-data fingerprint changed")
    if source.get("status") != "complete" or source.get("schema_version") != 1:
        raise ValueError("source packed-data manifest is not complete schema v1")
    if source.get("packing", {}).get("sequence_length") != EXPECTED_SEQUENCE_LENGTH:
        raise ValueError("source packed-data sequence length changed")
    if source.get("tokenizer", {}).get("vocab_size") != EXPECTED_VOCAB_SIZE:
        raise ValueError("source tokenizer vocabulary changed")
    if source.get("tokenizer", {}).get("doc_token_id") != EXPECTED_DOC_TOKEN_ID:
        raise ValueError("source tokenizer document token changed")

    token_format = source.get("format", {})
    dtype = token_format.get("dtype")
    token_bytes = token_format.get("token_bytes")
    if dtype not in {"uint16", "uint32"} or token_bytes not in {2, 4}:
        raise ValueError("source packed-data token format is unsupported")
    if token_format.get("endianness") != "little":
        raise ValueError("source packed-data must be little-endian")

    first = source["shards"][0]
    source_shard = source_manifest_path.parent / first["name"]
    if source_shard.stat().st_size != first["bytes"]:
        raise ValueError("source shard size changed")
    if _sha256(source_shard) != first["sha256"]:
        raise ValueError("source shard hash changed")
    if args.sequences > first["sequences"]:
        raise ValueError("requested smoke fixture exceeds the first source shard")

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"smoke fixture output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    output_shard = args.output / "shard-00000.bin"
    output_bytes = args.sequences * EXPECTED_SEQUENCE_LENGTH * token_bytes
    with source_shard.open("rb") as source_handle, output_shard.open("wb") as output_handle:
        remaining = output_bytes
        while remaining:
            block = source_handle.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise RuntimeError("source shard ended before the requested fixture")
            output_handle.write(block)
            remaining -= len(block)
        output_handle.flush()
        os.fsync(output_handle.fileno())
    output_sha256 = _sha256(output_shard)

    typecode = "H" if dtype == "uint16" else "I"
    values = array(typecode)
    values.frombytes(output_shard.read_bytes())
    if sys.byteorder != "little":
        values.byteswap()
    doc_markers = sum(value == EXPECTED_DOC_TOKEN_ID for value in values)
    if doc_markers < 1:
        raise RuntimeError("real-token smoke fixture contains no document markers")

    build_specification = {
        "format": token_format,
        "parent": {
            "build_fingerprint": EXPECTED_PARENT_BUILD_FINGERPRINT,
            "manifest_sha256": source_manifest_sha256,
            "shard_name": first["name"],
            "shard_sha256": first["sha256"],
            "start_sequence": 0,
        },
        "packing": {
            "sequence_length": EXPECTED_SEQUENCE_LENGTH,
            "sequences": args.sequences,
        },
        "tokenizer": {
            "doc_token_id": EXPECTED_DOC_TOKEN_ID,
            "vocab_size": EXPECTED_VOCAB_SIZE,
        },
    }
    build_fingerprint = hashlib.sha256(_canonical_bytes(build_specification)).hexdigest()
    tokens = args.sequences * EXPECTED_SEQUENCE_LENGTH
    manifest = {
        "build_fingerprint": build_fingerprint,
        "format": token_format,
        "packing": {
            "purpose": "distributed_cuda_smoke_v0.1",
            "sequence_length": EXPECTED_SEQUENCE_LENGTH,
        },
        "provenance": build_specification["parent"],
        "schema_version": 1,
        "shards": [
            {
                "bytes": output_bytes,
                "name": output_shard.name,
                "sequences": args.sequences,
                "sha256": output_sha256,
                "tokens": tokens,
            }
        ],
        "statistics": {
            "document_markers": doc_markers,
            "sequences": args.sequences,
            "tokens": tokens,
        },
        "status": "complete",
        "tokenizer": {
            "doc_token": "<|doc|>",
            "doc_token_id": EXPECTED_DOC_TOKEN_ID,
            "vocab_size": EXPECTED_VOCAB_SIZE,
        },
    }
    manifest_path = args.output / "manifest.json"
    _write_json(manifest_path, manifest)
    validate_packed_manifest(manifest_path, verify_hashes=True)
    report = {
        "build_fingerprint": build_fingerprint,
        "document_markers": doc_markers,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "parent_manifest_sha256": source_manifest_sha256,
        "sequences": args.sequences,
        "shard_sha256": output_sha256,
        "status": "pass",
        "tokens": tokens,
    }
    _write_json(args.output / "fixture-report.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
