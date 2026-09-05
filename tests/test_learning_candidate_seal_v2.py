from __future__ import annotations

import json
from pathlib import Path

from lumi.posttraining_learning_candidate_seal_v2 import (
    _materialize_sources_with_token_ledger,
)


def _row(record_id: str, source_id: str, split: str) -> dict:
    return {
        "messages": [
            {"content": "question", "role": "user"},
            {"content": "answer", "role": "assistant"},
        ],
        "record_id": record_id,
        "source": {"source_id": source_id},
        "split": split,
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_materializer_binds_token_ledger_without_retokenizing(tmp_path: Path) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _write_rows(
        a,
        [_row("a-train", "a", "train"), _row("a-validation", "a", "validation")],
    )
    _write_rows(b, [_row("b-train", "b", "train")])
    expected_metrics = {
        "a": {
            "train": {"records": 1, "supervised_tokens": 10},
            "validation": {"records": 1, "supervised_tokens": 3},
        },
        "b": {
            "train": {"records": 1, "supervised_tokens": 20},
            "validation": {"records": 0, "supervised_tokens": 0},
        },
    }
    result = _materialize_sources_with_token_ledger(
        source_inputs=[("a", a), ("b", b)],
        expected_source_metrics=expected_metrics,
        expected_input={
            "train_records": 2,
            "train_supervised_tokens": 30,
            "validation_records": 1,
            "validation_supervised_tokens": 3,
        },
        train_path=tmp_path / "train.jsonl",
        validation_path=tmp_path / "validation.jsonl",
    )
    assert result["source_metrics"] == expected_metrics
    assert result["splits"]["train"]["records"] == 2
    assert result["splits"]["train"]["supervised_tokens"] == 30
    train_ids = [json.loads(line)["record_id"] for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    assert train_ids == ["a-train", "b-train"]
