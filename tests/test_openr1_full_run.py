from __future__ import annotations

import io
from pathlib import Path

import pytest

from lumi.posttraining_data import SFTRecordDecision
from lumi.posttraining_openr1_full_run import (
    deterministic_owner_sample,
    deterministic_split,
    download_exact_object,
    exact_object_url,
    reconcile_openr1_pii_decision,
    select_verified_generation,
    sha256_value,
    validate_report_is_raw_text_free,
)


def _row() -> dict:
    return {
        "correctness_math_verify": [False, True],
        "generations": ["wrong", "We calculate 12 + 5 = 17."],
        "is_reasoning_complete": [True, True],
        "problem": "What is 12 + 5?",
        "solution": "The reference says 17.",
        "source": "synthetic_math",
        "uuid": "row-001",
    }


def _selected_record() -> dict:
    record, _, reason = select_verified_generation(
        _row(),
        global_row_index=5,
        object_path="data/train-00000-of-00010.parquet",
        object_row_index=5,
        object_sha256="a" * 64,
    )
    assert reason is None
    assert record is not None
    return record


def test_verified_generation_never_uses_reference_solution() -> None:
    record, attribution, reason = select_verified_generation(
        _row(),
        global_row_index=5,
        object_path="data/train-00000-of-00010.parquet",
        object_row_index=5,
        object_sha256="a" * 64,
    )

    assert reason is None
    assert record is not None and attribution is not None
    assert record["messages"][1]["content"] == "We calculate 12 + 5 = 17."
    assert record["messages"][1]["content"] != _row()["solution"]
    assert attribution["generation_index"] == 1
    assert attribution["source_object_sha256"] == "a" * 64


def test_reference_only_row_is_rejected() -> None:
    row = _row()
    row["generations"] = [row["solution"]]
    row["is_reasoning_complete"] = [True]
    row["correctness_math_verify"] = [True]

    record, attribution, reason = select_verified_generation(
        row,
        global_row_index=5,
        object_path="data/train-00000-of-00010.parquet",
        object_row_index=5,
        object_sha256="a" * 64,
    )

    assert record is None and attribution is None
    assert reason == "verified_trace_matches_reference_solution"


def test_external_dependency_is_rejected() -> None:
    row = _row()
    row["problem"] = "Read https://example.com and solve it"

    record, attribution, reason = select_verified_generation(
        row,
        global_row_index=5,
        object_path="data/train-00000-of-00010.parquet",
        object_row_index=5,
        object_sha256="a" * 64,
    )

    assert record is None and attribution is None
    assert reason == "problem_external_dependency"


def test_deterministic_split_has_exact_five_percent() -> None:
    records = []
    for index in range(100):
        record = _selected_record()
        record["record_id"] = f"openr1-{index:03d}"
        records.append(record)

    count = deterministic_split(records, 0.05)

    assert count == 5
    assert sum(record["split"] == "validation" for record in records) == 5
    original = {record["record_id"]: record["split"] for record in records}
    deterministic_split(list(reversed(records)), 0.05)
    assert {record["record_id"]: record["split"] for record in records} == original


def test_owner_sample_is_deterministic_and_bounded() -> None:
    records = []
    for index in range(40):
        record = _selected_record()
        record["record_id"] = f"openr1-{index:03d}"
        record["split"] = "validation" if index % 10 == 0 else "train"
        records.append(record)

    first = deterministic_owner_sample(records, maximum_records=32)
    second = deterministic_owner_sample(list(reversed(records)), maximum_records=32)

    assert len(first) == 32
    assert first == second
    assert sha256_value(first) == sha256_value(second)


def test_pii_reclassification_fails_closed() -> None:
    record = _selected_record()
    accepted, classification = reconcile_openr1_pii_decision(
        record,
        SFTRecordDecision(
            accepted=False,
            normalized_sha256="b" * 64,
            reasons=("pii_pattern", "secret_pattern"),
            record_id=record["record_id"],
            response_sha256="c" * 64,
        ),
    )

    assert accepted is False
    assert classification is None


class _Response(io.BytesIO):
    headers = {"Content-Length": "4"}

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def geturl(self) -> str:
        return "https://cdn-lfs-us-1.hf.co/object"


def test_exact_download_is_allowlisted_and_bounded(tmp_path: Path) -> None:
    destination = tmp_path / "README.md"
    result = download_exact_object(
        destination=destination,
        maximum_bytes=4,
        object_path="README.md",
        open_url=lambda request: _Response(b"test"),
    )

    assert destination.read_bytes() == b"test"
    assert result["bytes"] == 4
    assert result["requested_url"] == exact_object_url("README.md")
    with pytest.raises(ValueError, match="unlisted"):
        exact_object_url("LICENSE")


def test_raw_text_free_report_guard() -> None:
    validate_report_is_raw_text_free({"counts": {"records": 1}})
    with pytest.raises(ValueError, match="raw-text"):
        validate_report_is_raw_text_free({"messages": []})
