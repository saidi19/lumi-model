from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import lumi.production_corpus as production
from lumi.production_corpus import (
    _open_state,
    _process_batch,
    _seal_current_shard,
    build_round_robin_source_schedule,
)

ROOT = Path(__file__).parents[1]


class FakeTokenizer:
    def encode_batch(self, texts: list[str], *, add_special_tokens: bool):
        assert add_special_tokens is False
        return [SimpleNamespace(ids=[10, 11, 12]) for _ in texts]


class FakeQualityGate:
    def evaluate(self, text: str, category: str):
        assert text
        assert category == "science_and_education"
        metadata = {
            "language": {"label": "ENGLISH"},
            "near_duplicate": {"fingerprint_sha256": "a" * 64},
        }
        return SimpleNamespace(
            metadata=metadata,
            near_duplicate_signature=(1, 2, 3),
            rejection_reason=None,
        )

    def accept(self, document_id: str, evaluation) -> None:
        assert document_id.startswith("lumi-")
        assert evaluation.rejection_reason is None


def _source() -> dict:
    return {
        "category": "science_and_education",
        "component": "fixture",
        "dataset_id": "fixture/data",
        "ordinal": 0,
        "revision": "1" * 40,
        "source_bytes": 100,
        "source_file": "fixture.json.gz",
        "source_sha256": "b" * 64,
    }


def _filters() -> dict:
    return {
        "max_control_character_fraction": 0.01,
        "max_repeated_line_fraction": 0.9,
        "max_segment_utf8_bytes": 131072,
        "min_document_characters": 10,
        "normalization": "NFC",
        "pii_policy": "redact",
        "secret_policy": "drop_document",
    }


def _initialized_state(tmp_path: Path):
    connection = _open_state(tmp_path / "state.sqlite3")
    source = _source()
    with connection:
        connection.execute(
            "INSERT INTO categories(ordinal, category, floor_tokens) VALUES (0, ?, ?)",
            (source["category"], 1000),
        )
        connection.execute(
            "INSERT INTO sources(ordinal, category, component, dataset_id, revision, "
            "source_file, source_bytes, source_sha256) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                source["ordinal"],
                source["category"],
                source["component"],
                source["dataset_id"],
                source["revision"],
                source["source_file"],
                source["source_bytes"],
                source["source_sha256"],
            ),
        )
        production._metadata_set(connection, "next_shard_sequence", 0)
    return connection, source


def test_round_robin_schedule_covers_exact_inventory() -> None:
    source_manifest = json.loads(
        (ROOT / "configs" / "common-pile-source-files-v0.1.json").read_text(
            encoding="utf-8"
        )
    )
    mixture = json.loads(
        (ROOT / "configs" / "data-mix-v0.1.json").read_text(encoding="utf-8")
    )
    schedule, categories = build_round_robin_source_schedule(source_manifest, mixture)
    assert len(schedule) == 758
    assert sum(row["source_bytes"] for row in schedule) == 499_817_285_384
    assert [row["component"] for row in schedule[:6]] == [
        "peS2o",
        "arxiv_papers",
        "pubmed",
        "doab",
        "libretexts",
        "oercommons",
    ]
    assert sum(row["floor_tokens"] for row in categories) == 120_000_000_000


def test_batch_commit_keeps_files_and_sqlite_in_sync(tmp_path: Path) -> None:
    connection, source = _initialized_state(tmp_path)
    source_row = connection.execute("SELECT * FROM sources WHERE ordinal = 0").fetchone()
    result = _process_batch(
        connection,
        FakeQualityGate(),
        FakeTokenizer(),
        source=source,
        source_provenance={
            "required_non_null_provenance": ["upstream_id"],
            "retain_provenance": {"upstream_id": "id"},
            "static_provenance": {},
        },
        source_row=source_row,
        records=[(1, {"id": "upstream-1", "text": "A useful fixture sentence."})],
        filters=_filters(),
        state_directory=tmp_path,
        hard_max_tokens=1000,
    )
    assert result["accepted_documents"] == 1
    assert result["accepted_tokens"] == 3
    shard = connection.execute("SELECT * FROM current_shard").fetchone()
    assert shard["tokens"] == 3
    token_path = tmp_path / shard["token_path"]
    assert token_path.stat().st_size == 6
    assert token_path.read_bytes() == b"\x0a\x00\x0b\x00\x0c\x00"
    assert connection.execute("SELECT COUNT(*) FROM exact_content").fetchone()[0] == 1
    connection.close()


def test_batch_failure_rolls_back_and_truncates_files(tmp_path: Path) -> None:
    connection, source = _initialized_state(tmp_path)
    source_row = connection.execute("SELECT * FROM sources WHERE ordinal = 0").fetchone()

    class FailingTokenizer(FakeTokenizer):
        def encode_batch(self, texts: list[str], *, add_special_tokens: bool):
            raise RuntimeError("fixture tokenizer failure")

    with pytest.raises(RuntimeError, match="fixture tokenizer failure"):
        _process_batch(
            connection,
            FakeQualityGate(),
            FailingTokenizer(),
            source=source,
            source_provenance={
                "required_non_null_provenance": ["upstream_id"],
                "retain_provenance": {"upstream_id": "id"},
                "static_provenance": {},
            },
            source_row=source_row,
            records=[(1, {"id": "upstream-1", "text": "A useful fixture sentence."})],
            filters=_filters(),
            state_directory=tmp_path,
            hard_max_tokens=1000,
        )
    shard = connection.execute("SELECT * FROM current_shard").fetchone()
    assert (tmp_path / shard["clean_path"]).stat().st_size == 0
    assert (tmp_path / shard["token_path"]).stat().st_size == 0
    assert connection.execute("SELECT COUNT(*) FROM exact_content").fetchone()[0] == 0
    connection.close()


def test_seal_writes_manifest_and_removes_local_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connection, source = _initialized_state(tmp_path)
    source_row = connection.execute("SELECT * FROM sources WHERE ordinal = 0").fetchone()
    _process_batch(
        connection,
        FakeQualityGate(),
        FakeTokenizer(),
        source=source,
        source_provenance={
            "required_non_null_provenance": ["upstream_id"],
            "retain_provenance": {"upstream_id": "id"},
            "static_provenance": {},
        },
        source_row=source_row,
        records=[(1, {"id": "upstream-1", "text": "A useful fixture sentence."})],
        filters=_filters(),
        state_directory=tmp_path,
        hard_max_tokens=1000,
    )
    uploads = []

    def fake_upload(cli, *, bucket, key, path, account_id):
        uploads.append((key, path.stat().st_size))
        return {"bytes": path.stat().st_size, "key": key, "sha256": production._sha256_file(path)}

    monkeypatch.setattr(production, "_put_immutable_object", fake_upload)
    manifest = _seal_current_shard(
        connection,
        object(),
        bucket="fixture-bucket",
        account_id="123456789012",
        namespace="lumi-corpus/v0.1",
        state_directory=tmp_path,
        source_manifest_sha256="c" * 64,
    )
    assert manifest["tokens"] == 3
    assert manifest["documents"] == 1
    assert len(uploads) == 4
    assert int(production._metadata_get(connection, "logical_storage_bytes")) > 0
    assert connection.execute("SELECT COUNT(*) FROM shards").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM current_shard").fetchone()[0] == 0
    assert not list((tmp_path / "scratch").glob("science_and_education-000000*"))
    connection.close()
