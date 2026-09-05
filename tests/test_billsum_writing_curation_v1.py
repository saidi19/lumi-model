from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

from lumi.posttraining_billsum_writing_curation_v1 import (
    EXPECTED_COLUMNS,
    EXPECTED_OBJECT,
    bill_identity,
    candidate_from_source,
    custom_quality_reasons,
    download_exact_object,
    exact_object_url,
    identity_split,
)


def _source_row() -> dict[str, str]:
    return {
        "summary": "The measure authorizes $50 for two years.",
        "text": "This measure authorizes $50 for two years and establishes reporting.",
        "title": "A reporting measure",
    }


def test_source_projection_has_stable_content_identity_and_split() -> None:
    row = _source_row()
    first, first_attr = candidate_from_source(row, row_index=1)
    second, second_attr = candidate_from_source(row, row_index=99)
    assert set(row) == EXPECTED_COLUMNS
    assert first["record_id"] == second["record_id"]
    assert first["split"] == second["split"] == identity_split(
        bill_identity(row["title"], row["text"])
    )
    assert first_attr["bill_identity_sha256"] == second_attr["bill_identity_sha256"]
    assert first["source"]["license_id"] == "cc0-1.0"
    assert first["provenance"]["assistant_response_origin"].endswith(
        "authorship_not_established"
    )


def test_custom_gate_checks_punctuation_and_numeric_support() -> None:
    record, _ = candidate_from_source(_source_row(), row_index=0)
    assert custom_quality_reasons(record) == ()
    record["messages"][1]["content"] = "The measure authorizes $75"
    assert set(custom_quality_reasons(record)) == {
        "assistant_terminal_punctuation",
        "unsupported_numeric_literal",
    }


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = io.BytesIO(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def geturl(self) -> str:
        return "https://huggingface.co/safe-object"

    def read(self, size: int = -1) -> bytes:
        return self.payload.read(size)


def test_downloader_is_exact_and_removes_failed_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"exact"
    monkeypatch.setitem(EXPECTED_OBJECT, "bytes", len(payload))
    monkeypatch.setitem(EXPECTED_OBJECT, "sha256", hashlib.sha256(payload).hexdigest())
    destination = tmp_path / "train.parquet"
    result = download_exact_object(
        destination=destination,
        maximum_bytes=len(payload),
        open_url=lambda _request: _Response(payload),
    )
    assert destination.read_bytes() == payload
    assert result["sha256"] == EXPECTED_OBJECT["sha256"]
    assert "/resolve/3d8510441c06a3d9dfb32eb0d7f80151730bcc4f/" in exact_object_url()
    destination.write_bytes(b"changed")
    with pytest.raises(FileExistsError):
        download_exact_object(
            destination=destination,
            maximum_bytes=len(payload),
            open_url=lambda _request: _Response(payload),
        )


def test_empty_source_field_fails_closed() -> None:
    row = _source_row()
    row["title"] = ""
    with pytest.raises(ValueError, match="empty source field"):
        candidate_from_source(row, row_index=0)
