from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_learning_candidate_seal_v1 import (
    _materialize_sources,
    validate_learning_candidate_manifest,
)


def _row(record_id: str, source_id: str, split: str, tokens: int) -> dict:
    return {
        "messages": [
            {"content": "question", "role": "user"},
            {"content": "answer", "role": "assistant"},
        ],
        "record_id": record_id,
        "source": {"source_id": source_id},
        "split": split,
        "test_tokens": tokens,
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_materializer_requires_lexical_sources(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_rows(first, [_row("b-1", "b", "train", 1)])
    _write_rows(second, [_row("a-1", "a", "validation", 1)])
    with pytest.raises(ValueError, match="lexical order"):
        _materialize_sources(
            source_inputs=[("b", first), ("a", second)],
            train_path=tmp_path / "train.jsonl",
            validation_path=tmp_path / "validation.jsonl",
            measure=lambda row: row["test_tokens"],
        )


def test_manifest_rejects_production_boundary() -> None:
    manifest = {
        "authorization_boundary": {
            "closure_supplement_included": False,
            "production_dataset_seal": True,
            "production_sft_started": False,
            "smoke_only": True,
        },
        "output": {
            "splits": {
                "train": {"records": 251176, "supervised_tokens": 7341582},
                "validation": {"records": 13805, "supervised_tokens": 427005},
            }
        },
        "raw_training_text_retained_in_manifest": False,
        "schema_version": "lumi.sft-learning-candidate-seal.v0.1",
        "status": "PASS_nonproduction_learning_candidate_sealed_smoke_only",
        "seal_sha256": "0" * 64,
    }
    with pytest.raises(ValueError):
        validate_learning_candidate_manifest(manifest)
