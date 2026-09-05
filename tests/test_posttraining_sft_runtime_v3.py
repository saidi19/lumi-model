from __future__ import annotations

import hashlib
import inspect
import io
import json
import os
import stat
import struct
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

import lumi.posttraining_sft_runtime_v3 as runtime


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
            f"{prefix}.tokens.u32le": struct.pack(
                f"<{len(token_values)}I", *token_values
            ),
            f"{prefix}.labels.i32le": struct.pack(
                f"<{len(label_values)}i", *label_values
            ),
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


def _minimum_canary_arguments(output: Path) -> dict[str, Any]:
    return {
        "encoded_shards": output.parent / "encoded-not-opened",
        "source_checkpoint": output.parent / "source-not-opened",
        "source_checkpoint_manifest_sha256": "a" * 64,
        "output": output,
        "target_steps": 2,
        "stop_after_step": 1,
    }


def test_secure_dataset_handles_are_unbuffered_fileio(tmp_path: Path) -> None:
    root = _synthetic_token_consistent_root(tmp_path)
    secure_root, directory_descriptor, root_identity = runtime._open_secure_root(root)
    handle = None
    try:
        handle, identity = runtime._open_secure_dataset_file(
            directory_descriptor,
            "train-00000.tokens.u32le",
            require_private_mode=True,
        )
        assert isinstance(handle, io.FileIO)
        assert not isinstance(handle, io.BufferedReader)
        assert handle.read(1) == struct.pack("<I", 10)[:1]
        assert handle.tell() == 1
        runtime._finish_secure_dataset_file(
            handle,
            directory_descriptor,
            "train-00000.tokens.u32le",
            identity,
        )
        handle = None
        runtime._finish_secure_root(secure_root, directory_descriptor, root_identity)
    finally:
        if handle is not None and not handle.closed:
            handle.close()
        os.close(directory_descriptor)

    source = inspect.getsource(runtime._open_secure_dataset_file)
    assert "buffering=0" in source
    assert "buffering=1024 * 1024" not in source


def test_metadata_digest_mismatch_precedes_selected_shard_open_and_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _synthetic_token_consistent_root(tmp_path)
    bindings = _bindings(root)
    bindings["expected_metadata_inventory_sha256"] = "f" * 64
    selected_opens: list[str] = []
    binary_read_calls = 0
    original_open = runtime._open_secure_dataset_file

    def guarded_open(
        directory_descriptor: int,
        name: str,
        *,
        require_private_mode: bool,
    ):
        if name != "manifest.json":
            selected_opens.append(name)
        return original_open(
            directory_descriptor,
            name,
            require_private_mode=require_private_mode,
        )

    def forbidden_binary_read(*args: Any, **kwargs: Any) -> bytes:
        nonlocal binary_read_calls
        binary_read_calls += 1
        pytest.fail("metadata mismatch must precede selected binary content reads")

    monkeypatch.setattr(runtime, "_open_secure_dataset_file", guarded_open)
    monkeypatch.setattr(runtime, "_read_exact_and_hash", forbidden_binary_read)

    with pytest.raises(ValueError, match="metadata inventory changed"):
        runtime.EncodedSFTDataset(
            root,
            split="train",
            record_limit=1,
            **bindings,
        )

    assert selected_opens == []
    assert binary_read_calls == 0


def test_bounded_prefix_read_audit_is_exact_and_raw_text_free(tmp_path: Path) -> None:
    root = _synthetic_token_consistent_root(tmp_path)
    dataset = runtime.EncodedSFTDataset(
        root,
        split="train",
        record_limit=1,
        **_bindings(root),
    )

    assert dataset[0].input_ids.tolist() == [10, 11, 12]
    assert dataset.verification["binary_bytes_read"] == 24
    assert dataset.verification["encoded_index_row_reads"] == 1
    assert dataset.verification["file_opens"] == 4
    assert dataset.verification["fully_hashed_shards"] == 0
    assert dataset.verification["opened_shards"] == 1
    assert dataset.verification["records_loaded"] == 1
    assert 0 < dataset.verification["index_bytes_read"] < (
        root / "train-00000.index.jsonl"
    ).stat().st_size
    assert not any(
        key in dataset.verification
        for key in ("input_ids", "labels", "record_id", "source_text", "text")
    )


def test_resume_invocation_marker_is_exclusive_and_precedes_all_input_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StopAtSourceRead(RuntimeError):
        pass

    output = tmp_path / "attempt" / "private" / "resumed"
    calls: list[str] = []
    expect_resume_marker = False

    def stop_at_source(*args: Any, **kwargs: Any) -> str:
        calls.append("source")
        marker = output / runtime.SFT_CANARY_RESUME_STARTED_FILENAME
        assert marker.is_file() is expect_resume_marker
        raise StopAtSourceRead

    class ForbiddenDataset:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls.append("dataset")
            pytest.fail("source probe must stop before dataset construction")

    monkeypatch.setattr(runtime, "_verify_checksum_manifest", stop_at_source)
    monkeypatch.setattr(runtime, "EncodedSFTDataset", ForbiddenDataset)

    with pytest.raises(StopAtSourceRead):
        runtime.run_sft_cpu_canary(**_minimum_canary_arguments(output))
    assert calls == ["source"]

    calls.clear()
    expect_resume_marker = True
    with pytest.raises(StopAtSourceRead):
        runtime.run_sft_cpu_canary(
            **_minimum_canary_arguments(output),
            resume=True,
        )
    resume_marker = output / runtime.SFT_CANARY_RESUME_STARTED_FILENAME
    assert resume_marker.is_file()
    assert stat.S_IMODE(resume_marker.stat().st_mode) == 0o600
    assert calls == ["source"]

    calls.clear()
    with pytest.raises(FileExistsError):
        runtime.run_sft_cpu_canary(
            **_minimum_canary_arguments(output),
            resume=True,
        )
    assert calls == []


def test_dataset_read_audit_is_persisted_in_state_run_specification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StopBeforeModelRead(RuntimeError):
        pass

    output = tmp_path / "attempt" / "private" / "uninterrupted"
    expected_read_audit = {
        "binary_bytes_read": 96,
        "encoded_index_row_reads": 4,
        "file_opens": 7,
        "fully_hashed_shards": 0,
        "index_bytes_read": 511,
        "metadata_inventory_verified_before_selected_content_read": True,
        "opened_shards": 2,
        "records_loaded": 4,
        "unbuffered_selected_content_reads": True,
    }

    class SyntheticDataset:
        manifest = {
            "manifest_sha256": "b" * 64,
            "seal_sha256": "c" * 64,
        }
        verification = {
            **expected_read_audit,
            "inventory_sha256": "d" * 64,
            "manifest_file_sha256": "e" * 64,
            "metadata_inventory_sha256": "f" * 64,
            "profile": "token_consistent_v0.2",
            "selected_prefix_sha256": "0" * 64,
        }

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    def stop_before_model_read(*args: Any, **kwargs: Any) -> None:
        raise StopBeforeModelRead

    monkeypatch.setattr(runtime, "_verify_checksum_manifest", lambda *a, **k: "a" * 64)
    monkeypatch.setattr(runtime, "EncodedSFTDataset", SyntheticDataset)
    monkeypatch.setattr(
        runtime.LumiForCausalLM,
        "from_pretrained",
        stop_before_model_read,
    )

    with pytest.raises(StopBeforeModelRead):
        runtime.run_sft_cpu_canary(**_minimum_canary_arguments(output))

    state_path = output / runtime.SFT_CANARY_STATE_FILENAME
    state = json.loads(state_path.read_bytes())
    assert state["run_specification"]["data"]["read_audit"] == expected_read_audit
    assert state["run_specification"]["data"]["record_limit"] is None
    assert state["run_specification"]["optimizer"]["checkpoint_steps"] == 1
    assert state["run_specification"]["optimizer"]["pad_to_multiple_of"] == 8
    assert not any(
        key in state["run_specification"]["data"]["read_audit"]
        for key in ("input_ids", "labels", "record_id", "source_text", "text")
    )


def test_v3_runtime_preserves_single_open_fused_bounded_loading(
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
    dataset = runtime.EncodedSFTDataset(
        root,
        split="train",
        record_limit=1,
        **_bindings(root),
    )

    assert len(dataset) == 1
    assert opened == [
        "manifest.json",
        "train-00000.tokens.u32le",
        "train-00000.labels.i32le",
        "train-00000.index.jsonl",
    ]


def test_v3_runner_requires_all_external_bindings() -> None:
    source = (
        Path(__file__).parents[1] / "scripts" / "run_sft_cpu_canary_v3.py"
    ).read_text(encoding="utf-8")
    for flag in (
        "--encoded-shards-inventory-sha256",
        "--encoded-shards-manifest-file-sha256",
        "--encoded-shards-manifest-sha256",
        "--encoded-shards-metadata-inventory-sha256",
        "--encoded-shards-record-limit",
        "--encoded-shards-seal-sha256",
        "--minimum-learning-rate-ratio",
        "--optimizer-profile",
        "--pad-to-multiple-of",
        "--checkpoint-steps",
    ):
        assert flag in source
    assert "posttraining_sft_runtime_v3" in source
    assert "posttraining_sft_runtime_v2" not in source
