"""Deterministic, resumable fixed-length token packing for Lumi pilots."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from array import array
from bisect import bisect_right
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

PACKED_DATA_SCHEMA_VERSION = 1
TOKEN_BYTES = 4
TOKEN_DTYPE = "uint32"
TOKEN_ENDIANNESS = "little"
STATE_FILENAME = "packing-state.json"
MANIFEST_FILENAME = "manifest.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _created_at() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _tokenizer_artifact(tokenizer_path: Path) -> dict[str, Any]:
    required = ("tokenizer.json", "tokenizer_config.json")
    optional = ("lumi_tokenizer_metadata.json",)
    files = []
    for name in (*required, *optional):
        path = tokenizer_path / name
        if name in required and not path.is_file():
            raise FileNotFoundError(f"tokenizer artifact is missing {path}")
        if path.is_file():
            files.append(
                {
                    "bytes": path.stat().st_size,
                    "name": name,
                    "sha256": _sha256_file(path),
                }
            )
    return {"files": files, "path": str(tokenizer_path.resolve())}


def _input_artifact(input_path: Path) -> dict[str, Any]:
    if not input_path.is_file():
        raise FileNotFoundError(f"packing input does not exist: {input_path}")
    stat = input_path.stat()
    return {
        "bytes": stat.st_size,
        "path": str(input_path.resolve()),
        "sha256": _sha256_file(input_path),
    }


def _shard_path(output: Path, shard_index: int) -> Path:
    return output / f"shard-{shard_index:05d}.bin"


def _write_token_ids(handle, token_ids: list[int]) -> None:
    values = array("I", token_ids)
    if values.itemsize != TOKEN_BYTES:
        raise RuntimeError(
            f"platform unsigned-int width is {values.itemsize}, expected {TOKEN_BYTES}"
        )
    if sys.byteorder != TOKEN_ENDIANNESS:
        values.byteswap()
    handle.write(values.tobytes())


def _reconcile_shards(output: Path, committed_tokens: int, shard_capacity: int) -> None:
    expected_shards = (committed_tokens + shard_capacity - 1) // shard_capacity
    existing = {path.name: path for path in output.glob("shard-*.bin")}
    for shard_index in range(expected_shards):
        path = _shard_path(output, shard_index)
        expected_tokens = min(shard_capacity, committed_tokens - shard_index * shard_capacity)
        expected_bytes = expected_tokens * TOKEN_BYTES
        if not path.is_file():
            raise RuntimeError(f"checkpoint references missing shard: {path}")
        actual_bytes = path.stat().st_size
        if actual_bytes < expected_bytes:
            raise RuntimeError(
                f"checkpoint shard is shorter than committed state: {path} "
                f"({actual_bytes} < {expected_bytes})"
            )
        if actual_bytes > expected_bytes:
            with path.open("r+b") as handle:
                handle.truncate(expected_bytes)
                handle.flush()
                os.fsync(handle.fileno())
        existing.pop(path.name, None)
    for path in existing.values():
        path.unlink()


def _flush_and_sync(handle) -> None:
    if handle is None:
        return
    handle.flush()
    os.fsync(handle.fileno())


def _verify_cached_manifest(
    manifest_path: Path, expected_fingerprint: str, verify_hashes: bool = True
) -> dict[str, Any]:
    manifest = validate_packed_manifest(manifest_path, verify_hashes=verify_hashes)
    if manifest["build_fingerprint"] != expected_fingerprint:
        raise ValueError("existing packed manifest does not match the requested build")
    return manifest


def validate_packed_manifest(
    manifest_path: str | Path, *, verify_hashes: bool = True
) -> dict[str, Any]:
    """Validate one complete packed-data manifest and its local shard files."""

    manifest_path = Path(manifest_path)
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != PACKED_DATA_SCHEMA_VERSION:
        raise ValueError("unsupported packed-data schema version")
    if manifest.get("status") != "complete":
        raise ValueError("packed-data manifest is not complete")
    packing = manifest["packing"]
    sequence_length = packing["sequence_length"]
    if not isinstance(sequence_length, int) or sequence_length < 2:
        raise ValueError("packed sequence length must be at least 2")
    if manifest["format"] != {
        "dtype": TOKEN_DTYPE,
        "endianness": TOKEN_ENDIANNESS,
        "layout": "flat_fixed_sequences",
        "token_bytes": TOKEN_BYTES,
    }:
        raise ValueError("unsupported packed token format")

    total_tokens = 0
    total_sequences = 0
    names: set[str] = set()
    for shard in manifest["shards"]:
        name = shard["name"]
        if name in names or Path(name).name != name:
            raise ValueError(f"invalid or duplicate shard name: {name!r}")
        names.add(name)
        path = manifest_path.parent / name
        if not path.is_file():
            raise FileNotFoundError(f"packed shard does not exist: {path}")
        if path.stat().st_size != shard["bytes"]:
            raise ValueError(f"packed shard size changed: {path}")
        if shard["bytes"] != shard["tokens"] * TOKEN_BYTES:
            raise ValueError(f"packed shard byte/token mismatch: {path}")
        if shard["tokens"] != shard["sequences"] * sequence_length:
            raise ValueError(f"packed shard is not sequence aligned: {path}")
        if verify_hashes and _sha256_file(path) != shard["sha256"]:
            raise ValueError(f"packed shard SHA-256 changed: {path}")
        total_tokens += shard["tokens"]
        total_sequences += shard["sequences"]

    statistics = manifest["statistics"]
    if total_tokens != statistics["tokens"]:
        raise ValueError("packed manifest token total does not match its shards")
    if total_sequences != statistics["sequences"]:
        raise ValueError("packed manifest sequence total does not match its shards")
    if total_tokens != total_sequences * sequence_length:
        raise ValueError("packed manifest totals are not sequence aligned")
    return manifest


def build_packed_dataset(
    *,
    input_path: str | Path,
    tokenizer_path: str | Path,
    output: str | Path,
    sequence_length: int,
    sequences_per_shard: int = 512,
    target_tokens: int | None = None,
    text_key: str = "text",
    max_documents: int | None = None,
    checkpoint_documents: int = 1_000,
    expected_vocab_size: int | None = None,
    expected_doc_token_id: int = 3,
    resume: bool = False,
) -> dict[str, Any]:
    """Pack JSONL documents into deterministic fixed-length uint32 shards.

    State commits only after shard bytes are flushed. A resumed build truncates
    uncommitted bytes back to the last durable input offset before replaying.
    """

    input_path = Path(input_path)
    tokenizer_path = Path(tokenizer_path)
    output = Path(output)
    if sequence_length < 2:
        raise ValueError("sequence-length must be at least 2")
    if sequences_per_shard < 1:
        raise ValueError("sequences-per-shard must be positive")
    if checkpoint_documents < 1:
        raise ValueError("checkpoint-documents must be positive")
    if max_documents is not None and max_documents < 1:
        raise ValueError("max-documents must be positive")
    if target_tokens is not None and target_tokens < sequence_length:
        raise ValueError("target-tokens must cover at least one full sequence")

    effective_target = (
        None if target_tokens is None else (target_tokens // sequence_length) * sequence_length
    )
    input_artifact = _input_artifact(input_path)
    tokenizer_artifact = _tokenizer_artifact(tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    vocab_size = len(tokenizer)
    if expected_vocab_size is not None and vocab_size != expected_vocab_size:
        raise ValueError(
            f"tokenizer vocabulary mismatch: expected {expected_vocab_size}, got {vocab_size}"
        )
    vocabulary = tokenizer.get_vocab()
    doc_token = "<|doc|>"
    if vocabulary.get(doc_token) != expected_doc_token_id:
        raise ValueError(
            f"tokenizer {doc_token} ID must be {expected_doc_token_id}, "
            f"got {vocabulary.get(doc_token)!r}"
        )

    build_specification = {
        "format": {
            "dtype": TOKEN_DTYPE,
            "endianness": TOKEN_ENDIANNESS,
            "layout": "flat_fixed_sequences",
            "token_bytes": TOKEN_BYTES,
        },
        "input": input_artifact,
        "packing": {
            "checkpoint_documents": checkpoint_documents,
            "max_documents": max_documents,
            "sequence_length": sequence_length,
            "sequences_per_shard": sequences_per_shard,
            "target_tokens_effective": effective_target,
            "target_tokens_requested": target_tokens,
            "text_key": text_key,
        },
        "tokenizer": {
            **tokenizer_artifact,
            "doc_token": doc_token,
            "doc_token_id": expected_doc_token_id,
            "vocab_size": vocab_size,
        },
    }
    build_fingerprint = _fingerprint(build_specification)
    manifest_path = output / MANIFEST_FILENAME
    state_path = output / STATE_FILENAME

    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"packed manifest already exists: {manifest_path}")
        return _verify_cached_manifest(manifest_path, build_fingerprint)

    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"packed output is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    if state_path.exists():
        if not resume:
            raise FileExistsError(f"packing state exists; pass --resume: {state_path}")
        with state_path.open(encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("build_fingerprint") != build_fingerprint:
            raise ValueError("packing checkpoint does not match the requested build")
        if state.get("status") != "active":
            raise ValueError("packing checkpoint is not active")
    else:
        unexpected = [path for path in output.iterdir() if path.name != STATE_FILENAME]
        if unexpected:
            raise FileExistsError(f"packed output is not empty: {output}")
        state = {
            "build_fingerprint": build_fingerprint,
            "documents_packed": 0,
            "documents_seen": 0,
            "encoded_text_bytes": 0,
            "input_offset": 0,
            "status": "active",
            "tokens_written": 0,
            "truncated_final_document": False,
        }
        _write_json_atomic(state_path, state)

    shard_capacity = sequence_length * sequences_per_shard
    _reconcile_shards(output, state["tokens_written"], shard_capacity)
    live = dict(state)
    current_handle = None
    current_shard_index = -1

    def ensure_shard(shard_index: int):
        nonlocal current_handle, current_shard_index
        if current_handle is not None and current_shard_index == shard_index:
            return current_handle
        if current_handle is not None:
            _flush_and_sync(current_handle)
            current_handle.close()
        path = _shard_path(output, shard_index)
        current_handle = path.open("ab")
        current_shard_index = shard_index
        return current_handle

    def append_tokens(token_ids: list[int]) -> int:
        written = 0
        while written < len(token_ids):
            shard_index = live["tokens_written"] // shard_capacity
            shard_offset = live["tokens_written"] % shard_capacity
            take = min(len(token_ids) - written, shard_capacity - shard_offset)
            handle = ensure_shard(shard_index)
            _write_token_ids(handle, token_ids[written : written + take])
            live["tokens_written"] += take
            written += take
        return written

    def checkpoint() -> None:
        _flush_and_sync(current_handle)
        _write_json_atomic(state_path, live)

    end_reason = "input_exhausted"
    try:
        with input_path.open("rb") as input_handle:
            input_handle.seek(live["input_offset"])
            while True:
                if max_documents is not None and live["documents_seen"] >= max_documents:
                    end_reason = "max_documents"
                    break
                raw_line = input_handle.readline()
                if not raw_line:
                    break
                next_offset = input_handle.tell()
                record = json.loads(raw_line)
                if text_key not in record:
                    raise KeyError(f"record is missing text key {text_key!r}")
                text = record[text_key]
                if not isinstance(text, str):
                    raise TypeError(f"record text must be a string, got {type(text).__name__}")

                live["documents_seen"] += 1
                live["input_offset"] = next_offset
                if text:
                    encoded = tokenizer.backend_tokenizer.encode(
                        text, add_special_tokens=False
                    ).ids
                    token_ids = [expected_doc_token_id, *encoded]
                    if any(token_id < 0 or token_id >= vocab_size for token_id in token_ids):
                        raise ValueError("tokenizer emitted an ID outside its vocabulary")
                    if effective_target is not None:
                        remaining = effective_target - live["tokens_written"]
                        if len(token_ids) > remaining:
                            token_ids = token_ids[:remaining]
                            live["truncated_final_document"] = True
                    if token_ids:
                        append_tokens(token_ids)
                        live["documents_packed"] += 1
                        live["encoded_text_bytes"] += len(text.encode("utf-8"))

                should_checkpoint = live["documents_seen"] % checkpoint_documents == 0
                if effective_target is not None and live["tokens_written"] >= effective_target:
                    end_reason = "target_tokens"
                    checkpoint()
                    break
                if should_checkpoint:
                    checkpoint()
        checkpoint()
    finally:
        if current_handle is not None:
            _flush_and_sync(current_handle)
            current_handle.close()
            current_handle = None

    tail_tokens = live["tokens_written"] % sequence_length
    if tail_tokens:
        if live["tokens_written"] < sequence_length:
            raise ValueError("packing input did not produce one complete sequence")
        live["tokens_written"] -= tail_tokens
        _reconcile_shards(output, live["tokens_written"], shard_capacity)
    if live["tokens_written"] == 0:
        raise ValueError("packing input did not produce one complete sequence")
    live["tail_tokens_dropped"] = tail_tokens
    checkpoint()

    shards = []
    total_sequences = 0
    for path in sorted(output.glob("shard-*.bin")):
        size = path.stat().st_size
        if size == 0:
            path.unlink()
            continue
        tokens = size // TOKEN_BYTES
        if size % TOKEN_BYTES or tokens % sequence_length:
            raise RuntimeError(f"packed shard is not sequence aligned: {path}")
        sequences = tokens // sequence_length
        shards.append(
            {
                "bytes": size,
                "name": path.name,
                "sequences": sequences,
                "sha256": _sha256_file(path),
                "tokens": tokens,
            }
        )
        total_sequences += sequences

    manifest = {
        **build_specification,
        "build_fingerprint": build_fingerprint,
        "completed_at": _created_at(),
        "end_reason": end_reason,
        "schema_version": PACKED_DATA_SCHEMA_VERSION,
        "shards": shards,
        "statistics": {
            "documents_packed": live["documents_packed"],
            "documents_seen": live["documents_seen"],
            "encoded_text_bytes": live["encoded_text_bytes"],
            "sequences": total_sequences,
            "tail_tokens_dropped": tail_tokens,
            "tokens": live["tokens_written"],
            "truncated_final_document": live["truncated_final_document"],
        },
        "status": "complete",
    }
    _write_json_atomic(manifest_path, manifest)
    live["status"] = "complete"
    _write_json_atomic(state_path, live)
    return validate_packed_manifest(manifest_path)


class PackedTokenDataset(Dataset[torch.Tensor]):
    """Memory-mapped fixed-length token sequences described by a Lumi manifest."""

    def __init__(self, manifest_path: str | Path, *, verify_hashes: bool = True) -> None:
        if sys.byteorder != TOKEN_ENDIANNESS:
            raise RuntimeError("PackedTokenDataset currently requires a little-endian host")
        self.manifest_path = Path(manifest_path)
        self.manifest = validate_packed_manifest(
            self.manifest_path, verify_hashes=verify_hashes
        )
        self.sequence_length = self.manifest["packing"]["sequence_length"]
        self._shards: list[torch.Tensor] = []
        self._sequence_ends: list[int] = []
        sequence_total = 0
        for shard in self.manifest["shards"]:
            path = self.manifest_path.parent / shard["name"]
            mapped = torch.from_file(
                str(path), shared=False, size=shard["tokens"], dtype=torch.int32
            ).view(shard["sequences"], self.sequence_length)
            self._shards.append(mapped)
            sequence_total += shard["sequences"]
            self._sequence_ends.append(sequence_total)

    def __len__(self) -> int:
        return self._sequence_ends[-1] if self._sequence_ends else 0

    def __getitem__(self, index: int) -> torch.Tensor:
        length = len(self)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError(index)
        shard_index = bisect_right(self._sequence_ends, index)
        shard_start = 0 if shard_index == 0 else self._sequence_ends[shard_index - 1]
        return self._shards[shard_index][index - shard_start].to(dtype=torch.long)
