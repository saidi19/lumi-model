from __future__ import annotations

import hashlib
import json
import os
import stat
import struct
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import torch

import lumi.posttraining_sft_runtime_v2 as runtime


def _write_private(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)
    path.chmod(0o600)


def _synthetic_token_consistent_root(tmp_path: Path) -> Path:
    root = tmp_path / "encoded"
    root.mkdir(mode=0o700, parents=True)
    shard_specs = (
        ("train", 0, ((10, "a" * 64), (20, "b" * 64))),
        ("train", 1, ((30, "c" * 64),)),
        ("validation", 0, ((40, "d" * 64),)),
    )
    shards: list[dict[str, Any]] = []
    split_records: Counter[str] = Counter()
    split_tokens: Counter[str] = Counter()
    split_supervised: Counter[str] = Counter()
    encoded_hasher = hashlib.sha256()
    sequence_hasher = hashlib.sha256()

    for split, index, records in shard_specs:
        prefix = f"{split}-{index:05d}"
        token_values: list[int] = []
        label_values: list[int] = []
        index_payload = bytearray()
        supervised_tokens = 0
        for ordinal, (base, record_hash) in enumerate(records):
            tokens = (base, base + 1, base + 2)
            labels = (-100, base + 1, base + 2)
            entry = {
                "offset_tokens": len(token_values),
                "record_id_sha256": record_hash,
                "record_ordinal": ordinal,
                "supervised_tokens": 2,
                "tokens": 3,
            }
            index_payload.extend(runtime._canonical_json(entry) + b"\n")
            encoded_payload = runtime._encoded_hash_payload(record_hash, tokens, labels)
            encoded_hasher.update(len(encoded_payload).to_bytes(8, "big"))
            encoded_hasher.update(encoded_payload)
            sequence_hasher.update(f"{split}\0{record_hash}\n".encode())
            token_values.extend(tokens)
            label_values.extend(labels)
            supervised_tokens += 2

        payloads = {
            f"{prefix}.tokens.u32le": struct.pack(f"<{len(token_values)}I", *token_values),
            f"{prefix}.labels.i32le": struct.pack(f"<{len(label_values)}i", *label_values),
            f"{prefix}.index.jsonl": bytes(index_payload),
        }
        files = []
        for name, payload in payloads.items():
            _write_private(root / name, payload)
            files.append(
                {
                    "bytes": len(payload),
                    "path": name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        shard_tokens = len(token_values)
        shards.append(
            {
                "files": files,
                "index": index,
                "records": len(records),
                "split": split,
                "supervised_tokens": supervised_tokens,
                "tokens": shard_tokens,
            }
        )
        split_records[split] += len(records)
        split_tokens[split] += shard_tokens
        split_supervised[split] += supervised_tokens

    manifest: dict[str, Any] = {
        "authorization_boundary": runtime._TOKEN_CONSISTENT_AUTHORIZATION_BOUNDARY,
        "context_window": 4096,
        "encoded_dataset_sha256": encoded_hasher.hexdigest(),
        "format": runtime._TOKEN_CONSISTENT_SHARD_FORMAT,
        "maximum_tokens_per_shard": 4_194_304,
        "production_eligible": False,
        "record_hash_sequence_sha256": sequence_hasher.hexdigest(),
        "records": sum(split_records.values()),
        "schema_version": "lumi.sft-encoded-shards.v0.1",
        "seal_sha256": "e" * 64,
        "shards": shards,
        "splits": {
            split: {
                "records": split_records[split],
                "supervised_tokens": split_supervised[split],
                "tokens": split_tokens[split],
            }
            for split in sorted(split_records)
        },
        "supervised_tokens": sum(split_supervised.values()),
        "tokens": sum(split_tokens.values()),
    }
    manifest["manifest_sha256"] = runtime._sha256(manifest)
    _write_private(root / "manifest.json", runtime._canonical_json(manifest) + b"\n")
    return root


def _bindings(root: Path) -> dict[str, str]:
    payload = (root / "manifest.json").read_bytes()
    manifest = json.loads(payload)
    metadata = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        info = path.lstat()
        metadata.append(
            {
                "mode": f"{stat.S_IMODE(info.st_mode):04o}",
                "relative_path": path.name,
                "st_ctime_ns": info.st_ctime_ns,
                "st_dev": info.st_dev,
                "st_ino": info.st_ino,
                "st_mtime_ns": info.st_mtime_ns,
                "st_nlink": info.st_nlink,
                "st_size": info.st_size,
                "st_uid": info.st_uid,
            }
        )
    return {
        "expected_inventory_sha256": runtime._sha256(manifest["shards"]),
        "expected_manifest_file_sha256": hashlib.sha256(payload).hexdigest(),
        "expected_manifest_sha256": manifest["manifest_sha256"],
        "expected_metadata_inventory_sha256": runtime._sha256(metadata),
        "expected_seal_sha256": manifest["seal_sha256"],
    }


def _refresh_manifest_file_record(root: Path, name: str) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    payload = (root / name).read_bytes()
    for shard in manifest["shards"]:
        for record in shard["files"]:
            if record["path"] == name:
                record["bytes"] = len(payload)
                record["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = runtime._sha256(manifest)
    _write_private(manifest_path, runtime._canonical_json(manifest) + b"\n")


def test_token_consistent_dataset_is_single_open_fused_and_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _synthetic_token_consistent_root(tmp_path)
    opened: list[str] = []
    original_open = runtime._open_secure_dataset_file

    def recording_open(
        directory_descriptor: int,
        name: str,
        *,
        require_private_mode: bool,
    ):
        opened.append(name)
        return original_open(
            directory_descriptor,
            name,
            require_private_mode=require_private_mode,
        )

    monkeypatch.setattr(runtime, "_open_secure_dataset_file", recording_open)
    monkeypatch.setattr(
        torch,
        "from_file",
        lambda *args, **kwargs: pytest.fail("path-based tensor reopen is forbidden"),
    )
    dataset = runtime.EncodedSFTDataset(root, split="train", **_bindings(root))

    assert len(dataset) == 3
    assert dataset.verification["all_inventory_file_hashes_verified"] is True
    assert dataset.verification["encoded_index_row_reads"] == 4
    assert dataset.verification["profile"] == "token_consistent_v0.2"
    assert len(opened) == len(set(opened)) == 10
    example = dataset[0]
    assert example.input_ids.tolist() == [10, 11, 12]
    assert example.labels.tolist() == [-100, 11, 12]


def test_bounded_prefix_requires_bindings_and_parses_only_requested_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _synthetic_token_consistent_root(tmp_path)
    with pytest.raises(ValueError, match="five exact"):
        runtime.EncodedSFTDataset(root, split="train", record_limit=1)

    opened: list[str] = []
    original_open = runtime._open_secure_dataset_file

    def recording_open(
        directory_descriptor: int,
        name: str,
        *,
        require_private_mode: bool,
    ):
        opened.append(name)
        return original_open(
            directory_descriptor,
            name,
            require_private_mode=require_private_mode,
        )

    monkeypatch.setattr(runtime, "_open_secure_dataset_file", recording_open)
    dataset = runtime.EncodedSFTDataset(
        root,
        split="train",
        record_limit=1,
        **_bindings(root),
    )

    assert len(dataset) == 1
    assert dataset.verification["encoded_index_row_reads"] == 1
    assert dataset.verification["binary_bytes_read"] == 24
    assert dataset.verification["index_bytes_read"] < (
        root / "train-00000.index.jsonl"
    ).stat().st_size
    assert dataset.verification["file_opens"] == 4
    assert dataset.verification["opened_shards"] == 1
    assert dataset.verification["records_loaded"] == 1
    assert len(dataset.verification["selected_prefix_sha256"]) == 64
    assert opened == [
        "manifest.json",
        "train-00000.tokens.u32le",
        "train-00000.labels.i32le",
        "train-00000.index.jsonl",
    ]
    assert dataset[0].input_ids.tolist() == [10, 11, 12]


@pytest.mark.parametrize(
    ("binding", "message"),
    [
        ("expected_manifest_sha256", "expected hash"),
        ("expected_manifest_file_sha256", "file does not match"),
        ("expected_inventory_sha256", "inventory does not match"),
        ("expected_metadata_inventory_sha256", "metadata inventory changed"),
        ("expected_seal_sha256", "expected seal"),
    ],
)
def test_exact_external_bindings_are_enforced(
    tmp_path: Path,
    binding: str,
    message: str,
) -> None:
    root = _synthetic_token_consistent_root(tmp_path)
    bindings = _bindings(root)
    bindings[binding] = "f" * 64
    with pytest.raises(ValueError, match=message):
        runtime.EncodedSFTDataset(root, split="train", **bindings)


def test_private_modes_hardlinks_and_symlinks_are_rejected(tmp_path: Path) -> None:
    root = _synthetic_token_consistent_root(tmp_path / "mode")
    root.chmod(0o755)
    with pytest.raises(ValueError, match="0700/0600"):
        runtime.EncodedSFTDataset(root, split="train", **_bindings(root))

    root = _synthetic_token_consistent_root(tmp_path / "file-mode")
    (root / "train-00000.tokens.u32le").chmod(0o644)
    with pytest.raises(ValueError, match="metadata is unsafe"):
        runtime.EncodedSFTDataset(root, split="train", **_bindings(root))

    root = _synthetic_token_consistent_root(tmp_path / "hardlink")
    os.link(
        root / "train-00000.tokens.u32le",
        root / "train-token-alias.bin",
    )
    with pytest.raises(ValueError, match="metadata is unsafe"):
        runtime.EncodedSFTDataset(root, split="train", **_bindings(root))

    real_root = _synthetic_token_consistent_root(tmp_path / "symlink")
    linked_root = tmp_path / "linked-encoded"
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(ValueError, match="real directory chain"):
        runtime.EncodedSFTDataset(
            linked_root,
            split="train",
            **_bindings(real_root),
        )


def test_file_symlink_and_path_swap_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _synthetic_token_consistent_root(tmp_path / "symlink-file")
    bindings = _bindings(root)
    victim = root / "train-00000.tokens.u32le"
    target = tmp_path / "tokens-target.bin"
    victim.rename(target)
    victim.symlink_to(target)
    with pytest.raises((OSError, ValueError)):
        runtime.EncodedSFTDataset(root, split="train", **bindings)

    root = _synthetic_token_consistent_root(tmp_path / "swap")
    bindings = _bindings(root)
    original_finish = runtime._finish_secure_dataset_file
    swapped = False

    def swapping_finish(
        handle,
        directory_descriptor: int,
        name: str,
        initial,
    ) -> None:
        nonlocal swapped
        if name == "train-00000.tokens.u32le" and not swapped:
            swapped = True
            path = root / name
            payload = path.read_bytes()
            path.rename(root / f"{name}.old")
            _write_private(path, payload)
        original_finish(handle, directory_descriptor, name, initial)

    monkeypatch.setattr(runtime, "_finish_secure_dataset_file", swapping_finish)
    with pytest.raises(ValueError, match="changed during loading"):
        runtime.EncodedSFTDataset(root, split="train", **bindings)


def test_noncanonical_index_and_same_size_tamper_are_rejected(tmp_path: Path) -> None:
    root = _synthetic_token_consistent_root(tmp_path / "noncanonical")
    index_path = root / "train-00000.index.jsonl"
    records = [json.loads(line) for line in index_path.read_text().splitlines()]
    _write_private(
        index_path,
        b"".join(json.dumps(record).encode() + b"\n" for record in records),
    )
    _refresh_manifest_file_record(root, index_path.name)
    with pytest.raises(ValueError, match="index row contract"):
        runtime.EncodedSFTDataset(root, split="train", **_bindings(root))

    root = _synthetic_token_consistent_root(tmp_path / "tamper")
    token_path = root / "train-00000.tokens.u32le"
    bindings = _bindings(root)
    payload = bytearray(token_path.read_bytes())
    payload[-1] ^= 1
    _write_private(token_path, bytes(payload))
    with pytest.raises(ValueError):
        runtime.EncodedSFTDataset(root, split="train", **bindings)


def test_manifest_duplicate_key_is_rejected(tmp_path: Path) -> None:
    root = _synthetic_token_consistent_root(tmp_path)
    manifest_path = root / "manifest.json"
    payload = manifest_path.read_bytes()
    bindings = _bindings(root)
    _write_private(manifest_path, b'{"context_window":4096,"context_window":4096}\n')
    with pytest.raises(ValueError, match="duplicate key"):
        runtime.EncodedSFTDataset(root, split="train", **bindings)
    assert payload


def test_runtime_source_has_no_generic_shard_preverification() -> None:
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    assert "verify_sft_shards" not in source
    assert "torch.from_file" not in source
    assert stat.S_IMODE(Path(runtime.__file__).stat().st_mode) in {0o644, 0o600}


def test_v2_runner_requires_every_external_shard_binding() -> None:
    source = (
        Path(__file__).parents[1] / "scripts" / "run_sft_cpu_canary_v2.py"
    ).read_text(encoding="utf-8")
    for flag in (
        "--encoded-shards-inventory-sha256",
        "--encoded-shards-manifest-file-sha256",
        "--encoded-shards-manifest-sha256",
        "--encoded-shards-metadata-inventory-sha256",
        "--encoded-shards-record-limit",
        "--encoded-shards-seal-sha256",
    ):
        assert flag in source
    assert "posttraining_sft_runtime_v2" in source


def test_v2_private_output_writers_are_mode_and_link_safe(tmp_path: Path) -> None:
    output = tmp_path / "attempt" / "private" / "uninterrupted"
    runtime._ensure_private_directory(output)
    assert stat.S_IMODE(output.stat().st_mode) == 0o700

    state = output / runtime.SFT_CANARY_STATE_FILENAME
    metrics = output / runtime.SFT_CANARY_METRICS_FILENAME
    runtime._write_json_atomic(state, {"safe": True})
    runtime._append_jsonl(metrics, {"step": 1})
    runtime._write_json_atomic(state, {"safe": False})
    marker = output / runtime.SFT_CANARY_STARTED_FILENAME
    runtime._write_json_exclusive(marker, {"status": "started"})
    with pytest.raises(FileExistsError):
        runtime._write_json_exclusive(marker, {"status": "retry"})
    for path in (state, metrics, marker):
        info = path.lstat()
        assert stat.S_IMODE(info.st_mode) == 0o600
        assert info.st_nlink == 1
        assert info.st_uid == os.geteuid()

    target = tmp_path / "target.json"
    target.write_text("untouched", encoding="utf-8")
    unsafe = output / "unsafe.json"
    unsafe.symlink_to(target)
    with pytest.raises((OSError, ValueError)):
        runtime._write_json_atomic(unsafe, {"must_not_write": True})
    assert target.read_text(encoding="utf-8") == "untouched"
