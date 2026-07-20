from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lumi.emr_production_shards import assemble_candidate_shards, write_anchor_segment


def _row(index: int, tokens: int, *, category: str = "science_and_education") -> dict:
    text = f"accepted document {index}"
    return {
        "category": category,
        "component": "fixture",
        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "dataset_id": "fixture/data",
        "dataset_revision": "a" * 40,
        "document_id": f"doc-{index:04d}",
        "provenance_json": json.dumps({"license": "CC0"}),
        "quality_json": json.dumps(
            {"near_duplicate": {"fingerprint_sha256": f"{index + 1:064x}"}}
        ),
        "segment_index": 0,
        "source_file": "fixture.json.gz",
        "source_line": index,
        "source_sha256": "b" * 64,
        "text": text,
        "token_bytes": bytes([index % 255, 0]) * tokens,
        "token_count": tokens,
        "transformations_json": json.dumps({"normalization": "NFC"}),
    }


def test_assembly_seals_at_document_boundary_and_keeps_partial(tmp_path: Path) -> None:
    result = assemble_candidate_shards(
        [_row(1, 4), _row(2, 4), _row(3, 4)],
        directory=tmp_path / "shards",
        category="science_and_education",
        next_sequence=2,
        source_manifest_sha256="c" * 64,
        target_tokens=10,
    )
    assert result["accepted_tokens"] == 12
    assert len(result["sealed_shards"]) == 1
    assert result["sealed_shards"][0]["tokens"] == 12
    assert result["sealed_shards"][0]["sequence"] == 2
    assert result["partial"] is None
    assert result["next_sequence"] == 3


def test_assembly_carries_an_existing_partial(tmp_path: Path) -> None:
    seed = assemble_candidate_shards(
        [_row(1, 4)],
        directory=tmp_path / "seed",
        category="science_and_education",
        next_sequence=2,
        source_manifest_sha256="c" * 64,
        target_tokens=10,
    )
    partial = seed["partial"]
    result = assemble_candidate_shards(
        [_row(2, 7)],
        directory=tmp_path / "continued",
        category="science_and_education",
        next_sequence=2,
        source_manifest_sha256="c" * 64,
        target_tokens=10,
        partial=partial,
    )
    assert len(result["sealed_shards"]) == 1
    assert result["sealed_shards"][0]["tokens"] == 11
    assert result["sealed_shards"][0]["documents"] == 2
    assert result["partial"] is None


def test_assembly_seals_under_target_partial_at_category_floor(tmp_path: Path) -> None:
    result = assemble_candidate_shards(
        [_row(1, 4)],
        directory=tmp_path / "final",
        category="science_and_education",
        next_sequence=2,
        source_manifest_sha256="c" * 64,
        target_tokens=10,
        seal_final_partial=True,
    )
    assert result["partial"] is None
    assert result["sealed_shards"][0]["tokens"] == 4
    assert result["next_sequence"] == 3


def test_assembly_rejects_cross_category_and_non_ordered_rows(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="crosses categories"):
        assemble_candidate_shards(
            [_row(1, 2, category="code")],
            directory=tmp_path / "category",
            category="science_and_education",
            next_sequence=0,
            source_manifest_sha256="c" * 64,
            target_tokens=10,
        )
    with pytest.raises(ValueError, match="strict source order"):
        assemble_candidate_shards(
            [_row(2, 2), _row(1, 2)],
            directory=tmp_path / "order",
            category="science_and_education",
            next_sequence=0,
            source_manifest_sha256="c" * 64,
            target_tokens=10,
        )


def test_assembly_rejects_token_byte_mismatch(tmp_path: Path) -> None:
    row = _row(1, 2)
    row["token_bytes"] = b"bad"
    with pytest.raises(ValueError, match="token bytes"):
        assemble_candidate_shards(
            [row],
            directory=tmp_path / "bad",
            category="science_and_education",
            next_sequence=0,
            source_manifest_sha256="c" * 64,
            target_tokens=10,
        )


def test_anchor_segment_is_strictly_ordered_and_hash_bound(tmp_path: Path) -> None:
    rows = []
    for index in (1, 2):
        row = _row(index, 2)
        row.update(
            {
                "lsh_buckets": [{"band": 0, "bucket": bytes([index])}],
                "minhash_signature": bytes([index]) * 16,
                "source_ordinal": 0,
            }
        )
        rows.append(row)
    result = write_anchor_segment(rows, destination=tmp_path / "anchors.jsonl.gz")
    assert result["documents"] == 2
    assert len(result["object_sha256"]) == 64
    with pytest.raises(ValueError, match="strict source order"):
        write_anchor_segment(
            list(reversed(rows)), destination=tmp_path / "reversed.jsonl.gz"
        )


def test_streaming_shard_callback_releases_each_local_shard(tmp_path: Path) -> None:
    published = []

    def publish(shard: dict) -> dict:
        assert all(
            Path(shard[name]).is_file()
            for name in ("clean_path", "manifest_path", "provenance_path", "token_path")
        )
        remote = {
            key: value
            for key, value in shard.items()
            if key not in {"clean_path", "manifest_path", "provenance_path", "token_path"}
        }
        remote["object_key"] = f"fixture/{shard['manifest_sha256']}"
        published.append(remote)
        return remote

    directory = tmp_path / "streamed"
    result = assemble_candidate_shards(
        [_row(index, 2) for index in range(1, 6)],
        directory=directory,
        category="science_and_education",
        next_sequence=0,
        source_manifest_sha256="c" * 64,
        target_tokens=4,
        seal_final_partial=True,
        sealed_callback=publish,
    )
    assert result["sealed_shards"] == published
    assert len(published) == 3
    assert result["partial"] is None
    assert list(directory.iterdir()) == []
