"""Deterministic raw-text-free shard materialization for Lumi SFT records."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import uuid
from array import array
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, TextIO

from .posttraining_format import ChatTemplate, EncodedSFTConversation, encode_sft_messages
from .posttraining_seal import SFT_SEAL_SCHEMA_VERSION

SFT_SHARDS_SCHEMA_VERSION = "lumi.sft-encoded-shards.v0.1"
SUPPORTED_SPLITS = frozenset({"train", "validation"})


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(_canonical_json(value) + b"\n")
    temporary.replace(path)


def _write_le_int32(handle: BinaryIO, values: Sequence[int], *, signed: bool) -> None:
    typecode = "i" if signed else "I"
    encoded = array(typecode, values)
    if encoded.itemsize != 4:
        raise RuntimeError("platform does not expose four-byte integer arrays")
    if sys.byteorder != "little":
        encoded.byteswap()
    handle.write(encoded.tobytes())


def _validate_seal(seal: Mapping[str, Any]) -> None:
    if seal.get("schema_version") != SFT_SEAL_SCHEMA_VERSION:
        raise ValueError("unsupported SFT dataset seal")
    expected = seal.get("seal_sha256")
    unsigned = dict(seal)
    unsigned.pop("seal_sha256", None)
    if not isinstance(expected, str) or expected != _sha256(unsigned):
        raise ValueError("SFT dataset seal hash changed")
    if seal.get("raw_text_retained_in_manifest") is not False:
        raise ValueError("SFT dataset seal must remain raw-text-free")


@dataclass
class _ShardWriter:
    root: Path
    split: str
    index: int
    tokens_handle: BinaryIO
    labels_handle: BinaryIO
    records_handle: TextIO
    records: int = 0
    tokens: int = 0
    supervised_tokens: int = 0

    @classmethod
    def open(cls, root: Path, *, split: str, index: int) -> _ShardWriter:
        prefix = f"{split}-{index:05d}"
        return cls(
            root=root,
            split=split,
            index=index,
            tokens_handle=(root / f"{prefix}.tokens.u32le").open("xb"),
            labels_handle=(root / f"{prefix}.labels.i32le").open("xb"),
            records_handle=(root / f"{prefix}.index.jsonl").open("x", encoding="utf-8"),
        )

    @property
    def prefix(self) -> str:
        return f"{self.split}-{self.index:05d}"

    def add(self, *, record_id: str, encoded: EncodedSFTConversation) -> None:
        if any(token < 0 or token > 0xFFFFFFFF for token in encoded.input_ids):
            raise ValueError("SFT input token is outside uint32 range")
        if any(label < -100 or label > 0x7FFFFFFF for label in encoded.labels):
            raise ValueError("SFT label is outside int32 range")
        if any(
            label != -100 and label != token
            for token, label in zip(encoded.input_ids, encoded.labels, strict=True)
        ):
            raise ValueError("unmasked SFT labels must equal their input token")

        offset = self.tokens
        _write_le_int32(self.tokens_handle, encoded.input_ids, signed=False)
        _write_le_int32(self.labels_handle, encoded.labels, signed=True)
        entry = {
            "offset_tokens": offset,
            "record_id_sha256": hashlib.sha256(record_id.encode("utf-8")).hexdigest(),
            "record_ordinal": self.records,
            "supervised_tokens": encoded.supervised_tokens,
            "tokens": len(encoded.input_ids),
        }
        self.records_handle.write(_canonical_json(entry).decode("utf-8") + "\n")
        self.records += 1
        self.tokens += len(encoded.input_ids)
        self.supervised_tokens += encoded.supervised_tokens

    def close(self) -> dict[str, Any]:
        self.tokens_handle.close()
        self.labels_handle.close()
        self.records_handle.close()
        files = []
        for suffix in ("tokens.u32le", "labels.i32le", "index.jsonl"):
            path = self.root / f"{self.prefix}.{suffix}"
            files.append(
                {
                    "bytes": path.stat().st_size,
                    "path": path.name,
                    "sha256": _sha256_file(path),
                }
            )
        return {
            "files": files,
            "index": self.index,
            "records": self.records,
            "split": self.split,
            "supervised_tokens": self.supervised_tokens,
            "tokens": self.tokens,
        }


def materialize_sft_shards(
    records: Iterable[Mapping[str, Any]],
    *,
    output: str | Path,
    seal: Mapping[str, Any],
    template: ChatTemplate,
    tokenizer: Any,
    maximum_tokens_per_shard: int = 4_194_304,
) -> dict[str, Any]:
    """Encode a sealed dataset into atomic integer shards without retaining text."""

    _validate_seal(seal)
    if (
        isinstance(maximum_tokens_per_shard, bool)
        or not isinstance(maximum_tokens_per_shard, int)
        or maximum_tokens_per_shard < template.context_window
    ):
        raise ValueError("maximum_tokens_per_shard must cover at least one full context")

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite SFT shards: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir()

    record_hasher = hashlib.sha256()
    encoded_hasher = hashlib.sha256()
    split_counts: Counter[str] = Counter()
    split_tokens: Counter[str] = Counter()
    split_supervised: Counter[str] = Counter()
    writers: dict[str, _ShardWriter] = {}
    next_indices: Counter[str] = Counter()
    inventories: list[dict[str, Any]] = []
    total_records = 0
    total_tokens = 0
    total_supervised = 0

    def close_writer(split: str) -> None:
        writer = writers.pop(split, None)
        if writer is not None:
            inventories.append(writer.close())

    try:
        for record in records:
            split = record.get("split")
            record_id = record.get("record_id")
            messages = record.get("messages")
            if split not in SUPPORTED_SPLITS:
                raise ValueError("SFT shard record has unsupported split")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError("SFT shard record_id must be a non-empty string")
            if not isinstance(messages, list):
                raise ValueError("SFT shard messages must be an array")

            record_bytes = _canonical_json(record)
            record_hasher.update(len(record_bytes).to_bytes(8, "big"))
            record_hasher.update(record_bytes)
            encoded = encode_sft_messages(
                messages,
                tokenizer=tokenizer,
                template=template,
            )
            encoded_payload = {
                "input_ids": list(encoded.input_ids),
                "labels": list(encoded.labels),
                "record_id": record_id,
            }
            encoded_bytes = _canonical_json(encoded_payload)
            encoded_hasher.update(len(encoded_bytes).to_bytes(8, "big"))
            encoded_hasher.update(encoded_bytes)

            writer = writers.get(split)
            if writer is not None and (
                writer.tokens + len(encoded.input_ids) > maximum_tokens_per_shard
            ):
                close_writer(split)
                writer = None
            if writer is None:
                writer = _ShardWriter.open(
                    temporary,
                    split=split,
                    index=next_indices[split],
                )
                next_indices[split] += 1
                writers[split] = writer
            writer.add(record_id=record_id, encoded=encoded)

            total_records += 1
            total_tokens += len(encoded.input_ids)
            total_supervised += encoded.supervised_tokens
            split_counts[split] += 1
            split_tokens[split] += len(encoded.input_ids)
            split_supervised[split] += encoded.supervised_tokens

        for split in sorted(tuple(writers)):
            close_writer(split)
        if total_records == 0:
            raise ValueError("cannot materialize an empty SFT dataset")

        observed = {
            "encoded_dataset_sha256": encoded_hasher.hexdigest(),
            "input_tokens": total_tokens,
            "record_jsonl_sha256": record_hasher.hexdigest(),
            "records": total_records,
            "split_records": dict(sorted(split_counts.items())),
            "supervised_tokens": total_supervised,
        }
        expected = {name: seal.get(name) for name in observed}
        if observed != expected:
            raise ValueError("materialized SFT shards do not match the sealed dataset")

        manifest = {
            "context_window": template.context_window,
            "encoded_dataset_sha256": observed["encoded_dataset_sha256"],
            "format": {
                "index": "canonical-jsonl-v0.1",
                "labels": "little-endian-int32-ignore-index-minus-100",
                "raw_text_retained": False,
                "tokens": "little-endian-uint32",
            },
            "maximum_tokens_per_shard": maximum_tokens_per_shard,
            "records": total_records,
            "schema_version": SFT_SHARDS_SCHEMA_VERSION,
            "seal_sha256": seal["seal_sha256"],
            "shards": sorted(inventories, key=lambda item: (item["split"], item["index"])),
            "splits": {
                split: {
                    "records": split_counts[split],
                    "supervised_tokens": split_supervised[split],
                    "tokens": split_tokens[split],
                }
                for split in sorted(split_counts)
            },
            "supervised_tokens": total_supervised,
            "tokens": total_tokens,
        }
        manifest["manifest_sha256"] = _sha256(manifest)
        _write_json_atomic(temporary / "manifest.json", manifest)
        verify_sft_shards(temporary)
        temporary.replace(output)
        return manifest
    except Exception:
        for writer in tuple(writers.values()):
            writer.tokens_handle.close()
            writer.labels_handle.close()
            writer.records_handle.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_sft_shards(root: str | Path) -> dict[str, Any]:
    """Verify exact file hashes, index continuity, counts, and manifest identity."""

    root = Path(root)
    manifest_path = root / "manifest.json"
    if not root.is_dir() or not manifest_path.is_file():
        raise ValueError("encoded SFT shard directory is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SFT_SHARDS_SCHEMA_VERSION:
        raise ValueError("unsupported encoded SFT shard schema")
    expected_manifest_sha = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if not isinstance(expected_manifest_sha, str) or expected_manifest_sha != _sha256(unsigned):
        raise ValueError("encoded SFT shard manifest hash changed")
    if manifest.get("format", {}).get("raw_text_retained") is not False:
        raise ValueError("encoded SFT shards must remain raw-text-free")

    declared_files = {"manifest.json"}
    total_records = 0
    total_tokens = 0
    total_supervised = 0
    split_records: Counter[str] = Counter()
    split_tokens: Counter[str] = Counter()
    split_supervised: Counter[str] = Counter()
    seen_shards: set[tuple[str, int]] = set()
    for shard in manifest.get("shards", []):
        split = shard.get("split")
        index = shard.get("index")
        if split not in SUPPORTED_SPLITS or not isinstance(index, int):
            raise ValueError("encoded SFT shard identity is invalid")
        if (split, index) in seen_shards:
            raise ValueError("encoded SFT shard identity is duplicated")
        seen_shards.add((split, index))
        files = shard.get("files")
        if not isinstance(files, list) or len(files) != 3:
            raise ValueError("encoded SFT shard must declare exactly three files")
        by_name = {}
        for file_record in files:
            relative = file_record.get("path")
            if not isinstance(relative, str) or Path(relative).name != relative:
                raise ValueError("encoded SFT shard file path is invalid")
            path = root / relative
            if not path.is_file():
                raise ValueError(f"encoded SFT shard file is missing: {relative}")
            if path.stat().st_size != file_record.get("bytes"):
                raise ValueError(f"encoded SFT shard size changed: {relative}")
            if _sha256_file(path) != file_record.get("sha256"):
                raise ValueError(f"encoded SFT shard hash changed: {relative}")
            by_name[path.suffixes[-2] if len(path.suffixes) > 1 else path.suffix] = path
            declared_files.add(relative)

        prefix = f"{split}-{index:05d}"
        token_path = root / f"{prefix}.tokens.u32le"
        label_path = root / f"{prefix}.labels.i32le"
        index_path = root / f"{prefix}.index.jsonl"
        if not all(path in by_name.values() for path in (token_path, label_path, index_path)):
            raise ValueError("encoded SFT shard filenames changed")
        shard_tokens = shard.get("tokens")
        shard_records = shard.get("records")
        shard_supervised = shard.get("supervised_tokens")
        if not all(isinstance(value, int) and value > 0 for value in (shard_tokens, shard_records)):
            raise ValueError("encoded SFT shard counts are invalid")
        if not isinstance(shard_supervised, int) or not 0 < shard_supervised <= shard_tokens:
            raise ValueError("encoded SFT supervised-token count is invalid")
        if token_path.stat().st_size != shard_tokens * 4:
            raise ValueError("encoded SFT token byte count changed")
        if label_path.stat().st_size != shard_tokens * 4:
            raise ValueError("encoded SFT label byte count changed")

        offset = 0
        records = 0
        supervised = 0
        for line in index_path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            if entry.get("record_ordinal") != records or entry.get("offset_tokens") != offset:
                raise ValueError("encoded SFT shard index is not contiguous")
            length = entry.get("tokens")
            entry_supervised = entry.get("supervised_tokens")
            if not isinstance(length, int) or length < 1:
                raise ValueError("encoded SFT record token count is invalid")
            if not isinstance(entry_supervised, int) or not 0 < entry_supervised <= length:
                raise ValueError("encoded SFT record supervision count is invalid")
            record_hash = entry.get("record_id_sha256")
            if not isinstance(record_hash, str) or len(record_hash) != 64:
                raise ValueError("encoded SFT record identity hash is invalid")
            offset += length
            supervised += entry_supervised
            records += 1
        if (offset, records, supervised) != (
            shard_tokens,
            shard_records,
            shard_supervised,
        ):
            raise ValueError("encoded SFT shard index totals changed")

        total_records += shard_records
        total_tokens += shard_tokens
        total_supervised += shard_supervised
        split_records[split] += shard_records
        split_tokens[split] += shard_tokens
        split_supervised[split] += shard_supervised

    actual_files = {path.name for path in root.iterdir() if path.is_file()}
    if actual_files != declared_files:
        raise ValueError("encoded SFT shard directory contains undeclared files")
    observed_splits = {
        split: {
            "records": split_records[split],
            "supervised_tokens": split_supervised[split],
            "tokens": split_tokens[split],
        }
        for split in sorted(split_records)
    }
    if (
        total_records != manifest.get("records")
        or total_tokens != manifest.get("tokens")
        or total_supervised != manifest.get("supervised_tokens")
        or observed_splits != manifest.get("splits")
    ):
        raise ValueError("encoded SFT shard manifest totals changed")
    return manifest
