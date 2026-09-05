from __future__ import annotations

import hashlib
import io
import os
import stat
import struct
from pathlib import Path
from typing import Any

import pytest

import lumi.posttraining_token_consistent_shards_v2 as shards_v2


def _write_private(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)
    path.chmod(0o600)


def _expected_totals() -> dict[str, Any]:
    return {
        "maximum_record_tokens": 3,
        "records": 2,
        "train": {
            "records": 1,
            "serialized_tokens": 3,
            "supervised_tokens": 2,
        },
        "validation": {
            "records": 1,
            "serialized_tokens": 3,
            "supervised_tokens": 2,
        },
    }


def _synthetic_encoded_root(tmp_path: Path) -> Path:
    root = tmp_path / "encoded"
    root.mkdir(mode=0o700, parents=True)
    shards = []
    encoded_hasher = hashlib.sha256()
    record_hash_sequence_hasher = hashlib.sha256()
    for split, index, token_base, record_hash in (
        ("train", 0, 10, "a" * 64),
        ("validation", 0, 20, "b" * 64),
    ):
        prefix = f"{split}-{index:05d}"
        tokens = (token_base, token_base + 1, token_base + 2)
        labels = (-100, token_base + 1, token_base + 2)
        encoded_payload = shards_v2._encoded_payload_bytes(record_hash, tokens, labels)
        encoded_hasher.update(len(encoded_payload).to_bytes(8, "big"))
        encoded_hasher.update(encoded_payload)
        record_hash_sequence_hasher.update(f"{split}\0{record_hash}\n".encode())
        payloads = {
            f"{prefix}.tokens.u32le": struct.pack("<3I", *tokens),
            f"{prefix}.labels.i32le": struct.pack("<3i", *labels),
            f"{prefix}.index.jsonl": shards_v2.canonical_json(
                {
                    "offset_tokens": 0,
                    "record_id_sha256": record_hash,
                    "record_ordinal": 0,
                    "supervised_tokens": 2,
                    "tokens": 3,
                }
            )
            + b"\n",
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
        shards.append(
            {
                "files": files,
                "index": index,
                "records": 1,
                "split": split,
                "supervised_tokens": 2,
                "tokens": 3,
            }
        )
    manifest: dict[str, Any] = {
        "authorization_boundary": shards_v2.EXPECTED_OUTPUT_AUTHORIZATION_BOUNDARY,
        "context_window": 4_096,
        "encoded_dataset_sha256": encoded_hasher.hexdigest(),
        "format": {
            "index": shards_v2.EXPECTED_ENCODING_CONTRACT["index_format"],
            "labels": shards_v2.EXPECTED_ENCODING_CONTRACT["label_format"],
            "raw_text_retained": False,
            "tokens": shards_v2.EXPECTED_ENCODING_CONTRACT["token_format"],
        },
        "maximum_tokens_per_shard": 4_194_304,
        "production_eligible": False,
        "record_hash_sequence_sha256": record_hash_sequence_hasher.hexdigest(),
        "records": 2,
        "schema_version": "lumi.sft-encoded-shards.v0.1",
        "seal_sha256": "d" * 64,
        "shards": shards,
        "splits": {
            "train": {"records": 1, "supervised_tokens": 2, "tokens": 3},
            "validation": {"records": 1, "supervised_tokens": 2, "tokens": 3},
        },
        "supervised_tokens": 4,
        "tokens": 6,
    }
    manifest["manifest_sha256"] = shards_v2.sha256_value(manifest)
    _write_private(root / "manifest.json", shards_v2.canonical_json(manifest) + b"\n")
    return root


def _audit(root: Path, **overrides: Any) -> dict[str, Any]:
    return shards_v2.fused_audit_encoded_shards(
        root,
        expected_totals=_expected_totals(),
        expected_shard_identities=(("train", 0), ("validation", 0)),
        maximum_index_rows=overrides.pop("maximum_index_rows", 2),
        **overrides,
    )


def _refresh_manifest_file_record(root: Path, name: str) -> None:
    manifest_path = root / "manifest.json"
    manifest = shards_v2.strict_json_bytes(manifest_path.read_bytes())
    payload = (root / name).read_bytes()
    for shard in manifest["shards"]:
        for file_record in shard["files"]:
            if file_record["path"] == name:
                file_record["bytes"] = len(payload)
                file_record["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = shards_v2.sha256_value(manifest)
    _write_private(manifest_path, shards_v2.canonical_json(manifest) + b"\n")


def test_strict_json_rejects_duplicate_and_noncanonical_objects() -> None:
    with pytest.raises(ValueError, match="duplicate JSON key"):
        shards_v2.strict_json_bytes(b'{"a":1,"a":2}\n')

    decoded = shards_v2.strict_json_bytes(b'{"b":2, "a":1}\n')
    assert decoded == {"a": 1, "b": 2}
    assert shards_v2.json_file_bytes(decoded) != b'{"b":2, "a":1}\n'


def test_exact_markers_are_bound_and_have_the_authorized_size() -> None:
    assert len(shards_v2.PASS_ONE_MARKER) == 59
    assert len(shards_v2.PASS_TWO_MARKER) == 59
    assert shards_v2.PASS_ONE_MARKER.endswith(b"\n")
    assert shards_v2.PASS_TWO_MARKER.endswith(b"\n")


def test_durable_markers_are_private_same_inode_and_nonretryable(tmp_path: Path) -> None:
    marker = b'{"started":"synthetic_v0.2_pass"}\n'
    target = tmp_path / "receipt.json"
    identity = shards_v2.start_marker(target, marker)
    before = target.stat()
    shards_v2.finish_marker(target, marker, b"{}\n", identity)
    after = target.stat()

    assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)
    assert stat.S_IMODE(after.st_mode) == 0o600
    with pytest.raises(FileExistsError):
        shards_v2.start_marker(target, marker)


def test_marker_refuses_symlink_target(tmp_path: Path) -> None:
    real = tmp_path / "real.json"
    real.write_bytes(b"untouched")
    link = tmp_path / "receipt.json"
    link.symlink_to(real)
    with pytest.raises(FileExistsError):
        shards_v2.start_marker(link, b'{"started":"synthetic"}\n')
    assert real.read_bytes() == b"untouched"


def test_finish_marker_rejects_hardlink_and_mode_drift(tmp_path: Path) -> None:
    marker = b'{"started":"synthetic"}\n'
    hardlinked = tmp_path / "hardlinked.json"
    identity = shards_v2.start_marker(hardlinked, marker)
    os.link(hardlinked, tmp_path / "alias.json")
    with pytest.raises(ValueError, match="marker"):
        shards_v2.finish_marker(hardlinked, marker, b"{}\n", identity)

    private = tmp_path / "mode.json"
    identity = shards_v2.start_marker(private, marker)
    private.chmod(0o644)
    with pytest.raises(ValueError, match="marker"):
        shards_v2.finish_marker(private, marker, b"{}\n", identity)


def test_secure_evidence_write_is_private_and_exclusive(tmp_path: Path) -> None:
    target = tmp_path / "seal.json"
    written = shards_v2.secure_write_json_exclusive(target, {"safe": True})
    assert written == len(shards_v2.json_file_bytes({"safe": True}))
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert target.stat().st_nlink == 1
    with pytest.raises(FileExistsError):
        shards_v2.secure_write_json_exclusive(target, {"safe": True})


def test_private_input_open_rejects_mode_link_and_symlink(tmp_path: Path) -> None:
    path = tmp_path / "input.jsonl"
    path.write_bytes(b"{}\n")
    specification = {"bytes": 3, "path": path.name}
    with pytest.raises(ValueError, match="input metadata"):
        shards_v2._open_private_input_once(tmp_path, specification)

    path.chmod(0o600)
    alias = tmp_path / "input-alias.jsonl"
    os.link(path, alias)
    with pytest.raises(ValueError, match="input metadata"):
        shards_v2._open_private_input_once(tmp_path, specification)

    alias.unlink()
    real = tmp_path / "real.jsonl"
    path.rename(real)
    path.symlink_to(real)
    with pytest.raises((OSError, ValueError)):
        shards_v2._open_private_input_once(tmp_path, specification)


def test_private_input_finish_detects_post_read_identity_drift(tmp_path: Path) -> None:
    path = tmp_path / "input.jsonl"
    _write_private(path, b"{}\n")
    handle, descriptor, bound_path, identity = shards_v2._open_private_input_once(
        tmp_path,
        {"bytes": 3, "path": path.name},
    )
    try:
        assert handle.read() == b"{}\n"
        path.chmod(0o400)
        with pytest.raises(ValueError, match="changed during encoding"):
            shards_v2._finish_private_input(handle, descriptor, bound_path, identity)
    finally:
        if not handle.closed:
            handle.close()


def test_encoded_file_finish_detects_post_read_identity_drift(tmp_path: Path) -> None:
    root = tmp_path / "encoded"
    root.mkdir(mode=0o700)
    path = root / "synthetic.bin"
    _write_private(path, b"safe")
    directory_descriptor = os.open(root, os.O_RDONLY)
    try:
        handle, identity = shards_v2._open_audit_file(
            directory_descriptor,
            path.name,
        )
        assert handle.read() == b"safe"
        path.chmod(0o400)
        with pytest.raises(ValueError, match="changed during audit"):
            shards_v2._finish_audit_file(
                handle,
                directory_descriptor,
                path.name,
                identity,
            )
    finally:
        os.close(directory_descriptor)


def test_exact_bounded_lines_never_reads_a_row_beyond_the_declared_range() -> None:
    class TrackingBytesIO(io.BytesIO):
        readline_calls = 0

        def readline(self, size: int = -1) -> bytes:
            self.readline_calls += 1
            return super().readline(size)

    first_line = b'{"synthetic":1}\n'
    forbidden_line = b'{"must_not_be_read":2}\n'
    handle = TrackingBytesIO(first_line + forbidden_line)
    with pytest.raises(ValueError, match="bytes beyond its row bound"):
        list(
            shards_v2.iter_exact_bounded_lines(
                handle,
                expected_rows=1,
                expected_bytes=len(first_line) + len(forbidden_line),
                maximum_rows=1,
                label="synthetic input",
            )
        )
    assert handle.readline_calls == 1


def test_verifier_compares_receipt_to_the_current_bound_plan_hash() -> None:
    plan = shards_v2.BoundTokenConsistentShardsPlan({}, file_sha256="a" * 64)
    shards_v2.validate_receipt_plan_hash(
        {"plan_file_sha256": "a" * 64},
        plan,
        "a" * 64,
    )
    with pytest.raises(ValueError, match="receipt plan hash changed"):
        shards_v2.validate_receipt_plan_hash(
            {"plan_file_sha256": "b" * 64},
            plan,
            "a" * 64,
        )


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        (
            {
                "offset_tokens": 0,
                "record_id": "raw-id",
                "record_id_sha256": "a" * 64,
                "record_ordinal": 0,
                "supervised_tokens": 1,
                "tokens": 2,
            },
            "exactly five",
        ),
        (
            {
                "offset_tokens": 0,
                "record_id_sha256": "a" * 64,
                "record_ordinal": 1,
                "supervised_tokens": 1,
                "tokens": 2,
            },
            "not contiguous",
        ),
        (
            {
                "offset_tokens": 1,
                "record_id_sha256": "a" * 64,
                "record_ordinal": 0,
                "supervised_tokens": 1,
                "tokens": 2,
            },
            "not contiguous",
        ),
    ],
)
def test_index_contract_rejects_raw_ids_and_discontinuity(
    entry: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        shards_v2.validate_index_entry(entry, expected_ordinal=0, expected_offset=0)


def test_fused_audit_accepts_hash_only_indices_and_reads_each_file_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _synthetic_encoded_root(tmp_path)
    opened: list[str] = []
    original = shards_v2._open_audit_file

    def recording_open(directory_descriptor: int, name: str):
        opened.append(name)
        return original(directory_descriptor, name)

    monkeypatch.setattr(shards_v2, "_open_audit_file", recording_open)
    audit = _audit(root)
    manifest = shards_v2.strict_json_bytes((root / "manifest.json").read_bytes())

    assert audit["binary_files_read_once"] == 4
    assert audit["encoded_dataset_sha256"] == manifest["encoded_dataset_sha256"]
    assert audit["encoded_index_row_reads"] == 2
    assert audit["index_files_read_once"] == 2
    assert audit["manifest_internal_sha256"] == manifest["manifest_sha256"]
    assert audit["manifest_seal_sha256"] == "d" * 64
    assert audit["masked_tokens"] == 2
    assert audit["record_hash_sequence_sha256"] == manifest[
        "record_hash_sequence_sha256"
    ]
    assert audit["records"] == 2
    assert audit["supervised_tokens"] == 4
    assert audit["tokens"] == 6
    assert len(opened) == len(set(opened)) == 7


@pytest.mark.parametrize(
    ("field", "value"),
    [("offset_tokens", 1), ("record_ordinal", 1)],
)
def test_fused_audit_rejects_offset_or_ordinal_drift(
    tmp_path: Path, field: str, value: int
) -> None:
    root = _synthetic_encoded_root(tmp_path)
    index_path = root / "train-00000.index.jsonl"
    entry = shards_v2.strict_json_bytes(index_path.read_bytes())
    entry[field] = value
    _write_private(index_path, shards_v2.canonical_json(entry) + b"\n")
    with pytest.raises(ValueError, match="not contiguous"):
        _audit(root)


def test_fused_audit_rejects_unmasked_label_mismatch(tmp_path: Path) -> None:
    root = _synthetic_encoded_root(tmp_path)
    _write_private(
        root / "train-00000.labels.i32le",
        struct.pack("<3i", -100, 999, 12),
    )
    with pytest.raises(ValueError, match="differs from input token"):
        _audit(root)


def test_fused_audit_reproduces_encoded_and_record_hash_sequence_digests(
    tmp_path: Path,
) -> None:
    root = _synthetic_encoded_root(tmp_path)
    index_path = root / "train-00000.index.jsonl"
    entry = shards_v2.strict_json_bytes(index_path.read_bytes())
    entry["record_id_sha256"] = "e" * 64
    _write_private(index_path, shards_v2.canonical_json(entry) + b"\n")
    _refresh_manifest_file_record(root, index_path.name)

    with pytest.raises(ValueError, match="encoded aggregate totals changed"):
        _audit(root)


def test_fused_audit_rejects_duplicate_and_noncanonical_index_json(
    tmp_path: Path,
) -> None:
    root = _synthetic_encoded_root(tmp_path)
    index_path = root / "train-00000.index.jsonl"
    duplicate = (
        b'{"offset_tokens":0,"record_id_sha256":"'
        + b"a" * 64
        + b'","record_ordinal":0,"supervised_tokens":2,'
        + b'"tokens":3,"tokens":3}\n'
    )
    _write_private(index_path, duplicate)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        _audit(root)

    root = _synthetic_encoded_root(tmp_path / "second")
    index_path = root / "train-00000.index.jsonl"
    entry = shards_v2.strict_json_bytes(index_path.read_bytes())
    noncanonical = (str(entry).replace("'", '"') + "\n").encode("utf-8")
    _write_private(index_path, noncanonical)
    with pytest.raises(ValueError, match="not canonical JSON"):
        _audit(root)


def test_fused_audit_rejects_raw_text_index_field(tmp_path: Path) -> None:
    root = _synthetic_encoded_root(tmp_path)
    index_path = root / "train-00000.index.jsonl"
    entry = shards_v2.strict_json_bytes(index_path.read_bytes())
    entry["messages"] = []
    _write_private(index_path, shards_v2.canonical_json(entry) + b"\n")
    with pytest.raises(ValueError, match="exactly five"):
        _audit(root)


def test_fused_audit_rejects_raw_text_manifest_even_when_resigned(tmp_path: Path) -> None:
    root = _synthetic_encoded_root(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = shards_v2.strict_json_bytes(manifest_path.read_bytes())
    manifest.pop("manifest_sha256")
    manifest["messages"] = ["forbidden"]
    manifest["manifest_sha256"] = shards_v2.sha256_value(manifest)
    _write_private(manifest_path, shards_v2.canonical_json(manifest) + b"\n")
    with pytest.raises(ValueError, match="raw training text"):
        _audit(root)


def test_fused_audit_enforces_index_row_cap_before_a_second_row(tmp_path: Path) -> None:
    root = _synthetic_encoded_root(tmp_path)
    with pytest.raises(ValueError, match="row cap would be exceeded"):
        _audit(root, maximum_index_rows=1)


def test_fused_audit_rejects_unsafe_directory_file_mode_link_and_symlink(
    tmp_path: Path,
) -> None:
    root = _synthetic_encoded_root(tmp_path)
    root.chmod(0o755)
    with pytest.raises(ValueError, match="directory metadata"):
        _audit(root)

    root.chmod(0o700)
    tokens = root / "train-00000.tokens.u32le"
    tokens.chmod(0o644)
    with pytest.raises(ValueError, match="file metadata"):
        _audit(root)

    tokens.chmod(0o600)
    os.link(tokens, root / "hardlink.bin")
    with pytest.raises(ValueError, match="file metadata"):
        _audit(root)

    second = tmp_path / "symlink-case"
    real = _synthetic_encoded_root(second)
    link = tmp_path / "encoded-link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises((OSError, ValueError)):
        _audit(link)


@pytest.mark.parametrize(
    "raw_key",
    [
        "assistant_response",
        "content",
        "messages",
        "prompt",
        "raw_text",
        "record_id",
        "response",
        "text",
        "user_prompt",
    ],
)
def test_evidence_raw_text_keys_fail_closed(raw_key: str) -> None:
    with pytest.raises(ValueError, match="raw training text"):
        shards_v2._assert_raw_text_free({"nested": [{raw_key: "forbidden"}]})
