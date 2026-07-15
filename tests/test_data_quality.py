from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.data_quality import (
    BenchmarkRegistry,
    DataQualityGate,
    LinguaLanguageIdentifier,
    MinHashNearDuplicateIndex,
    benchmark_signature_entry,
    sha256_file,
)


def _write_registry(path: Path, text: str) -> str:
    entry = benchmark_signature_entry(
        benchmark="fixture_eval",
        example_id="example-1",
        split="test",
        text=text,
        anchor_ngram_tokens=7,
        max_anchors=4,
        minimum_anchor_matches=2,
    )
    metadata = {
        "_meta": {
            "anchor_ngram_tokens": 7,
            "entries": 1,
            "normalization": "nfkc_casefold_whitespace_v1",
            "raw_text_retained": False,
            "schema_version": 1,
        }
    }
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, sort_keys=True) + "\n")
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return sha256_file(path)


def _quality_config(registry: Path) -> dict:
    return {
        "benchmark_decontamination": {
            "anchor_ngram_tokens": 7,
            "minimum_anchor_matches": 2,
            "registry": str(registry),
            "registry_sha256": sha256_file(registry),
        },
        "language": {
            "allowed_languages": ["en"],
            "backend": "lingua",
            "bypass_categories": ["code"],
            "max_characters": 4096,
            "minimum_confidence": 0.6,
            "package_version": "2.2.0",
        },
        "near_duplicate": {
            "algorithm": "minhash_lsh",
            "num_perm": 64,
            "package_version": "2.0.0",
            "scheme": "affine32",
            "seed": 1,
            "shingle_tokens": 5,
            "threshold": 0.8,
        },
        "schema_version": 1,
    }


def test_lingua_labels_english_and_rejects_bengali() -> None:
    identifier = LinguaLanguageIdentifier(
        {
            "allowed_languages": ["en"],
            "bypass_categories": ["code"],
            "max_characters": 4096,
            "minimum_confidence": 0.6,
            "package_version": "2.2.0",
        }
    )
    english = identifier.identify(
        "This carefully written English paragraph discusses reproducible data quality.",
        "reference",
    )
    assert english.label == "en"
    assert english.decision == "accepted"

    bengali = identifier.identify(
        "আমি আজ বাংলা ভাষায় একটি নতুন ভাষা মডেলের তথ্য যাচাই করছি।",
        "reference",
    )
    assert bengali.label == "bn"
    assert bengali.decision == "disallowed"
    assert identifier.identify("def f(x): return x + 1", "code").label == "zxx"


def test_benchmark_registry_matches_exact_and_embedded_text(tmp_path: Path) -> None:
    benchmark_text = (
        "Which careful procedure prevents evaluation examples from entering the training corpus "
        "while preserving a reproducible audit trail for later inspection?"
    )
    registry_path = tmp_path / "registry.jsonl"
    registry_sha = _write_registry(registry_path, benchmark_text)
    registry = BenchmarkRegistry(
        {
            "anchor_ngram_tokens": 7,
            "minimum_anchor_matches": 2,
            "registry": str(registry_path),
            "registry_sha256": registry_sha,
        },
        tmp_path,
    )
    assert registry.match(benchmark_text) == ("fixture_eval", "example-1")
    assert registry.match(f"Unrelated introduction. {benchmark_text} Extra explanation.") == (
        "fixture_eval",
        "example-1",
    )
    assert registry.match("A different sentence about astronomy and orbital mechanics.") is None


def test_minhash_index_rejects_near_copy_but_keeps_unrelated_text() -> None:
    index = MinHashNearDuplicateIndex(
        {
            "num_perm": 128,
            "package_version": "2.0.0",
            "scheme": "affine32",
            "seed": 1,
            "shingle_tokens": 4,
            "threshold": 0.75,
        }
    )
    original = (
        "Reproducible corpus construction records source identity license revision content hash "
        "and every deterministic transformation before language model training begins."
    )
    first = index.signature(original)
    index.add("first", first)
    near_copy = index.signature(original.replace("begins", "starts"))
    assert index.duplicate_of(near_copy) == "first"
    unrelated = index.signature(
        "Saturn has a complex ring system composed of ice particles dust and rocky material."
    )
    assert index.duplicate_of(unrelated) is None


def test_composed_gate_records_metadata_and_blocks_all_three_classes(tmp_path: Path) -> None:
    benchmark_text = (
        "A held out benchmark question with enough unique words must never be copied into "
        "the model development corpus because that would invalidate evaluation results."
    )
    registry_path = tmp_path / "registry.jsonl"
    _write_registry(registry_path, benchmark_text)
    gate = DataQualityGate(_quality_config(registry_path), tmp_path)

    accepted_text = (
        "A provenance ledger connects every processed segment to its pinned upstream source, "
        "license record, revision, and deterministic transformation history for auditing. Every "
        "decision remains linked to a stable identifier so later deletion requests and source "
        "audits can be completed without guessing. Deterministic hashes also prove whether an "
        "artifact changed between pipeline runs before training starts."
    )
    accepted = gate.evaluate(accepted_text, "reference")
    assert accepted.rejection_reason is None
    assert accepted.metadata["language"]["label"] == "en"
    gate.accept("accepted-1", accepted)

    near_copy = gate.evaluate(accepted_text.replace("starts", "begins"), "reference")
    assert near_copy.rejection_reason == "near_duplicate"
    contaminated = gate.evaluate(benchmark_text, "reference")
    assert contaminated.rejection_reason == "benchmark_contamination:fixture_eval"
    disallowed = gate.evaluate(
        "আমি বাংলা ভাষায় লিখছি এবং এই অংশটি ইংরেজি-প্রধান নমুনায় রাখা হবে না।",
        "reference",
    )
    assert disallowed.rejection_reason == "language:disallowed:bn"


def test_registry_rejects_raw_text_field(tmp_path: Path) -> None:
    registry_path = tmp_path / "bad.jsonl"
    metadata = {
        "_meta": {
            "anchor_ngram_tokens": 7,
            "entries": 1,
            "normalization": "nfkc_casefold_whitespace_v1",
            "schema_version": 1,
        }
    }
    entry = {
        "anchor_sha256": [],
        "benchmark": "bad",
        "example_id": "1",
        "exact_sha256": "a" * 64,
        "minimum_anchor_matches": 0,
        "split": "test",
        "text": "must not be retained",
    }
    registry_path.write_text(
        json.dumps(metadata) + "\n" + json.dumps(entry) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="must not retain raw"):
        BenchmarkRegistry(
            {
                "anchor_ngram_tokens": 7,
                "minimum_anchor_matches": 2,
                "registry": str(registry_path),
                "registry_sha256": sha256_file(registry_path),
            },
            tmp_path,
        )
