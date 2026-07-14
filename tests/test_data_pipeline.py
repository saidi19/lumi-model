from __future__ import annotations

import gzip
import json
from copy import deepcopy
from pathlib import Path

import pytest

from lumi.data_pipeline import build_sample, filter_document, validate_sample


def _policy() -> dict:
    return {
        "max_control_character_fraction": 0.01,
        "max_repeated_line_fraction": 0.9,
        "max_segment_utf8_bytes": 2048,
        "min_document_characters": 40,
        "normalization": "NFC",
        "pii_policy": "redact",
        "secret_policy": "drop_document",
    }


def _write_source(path: Path, label: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for index in range(30):
            text = (
                f"{label} document {index} contains useful educational prose and unique marker "
                f"{label}-{index}. Contact person{index}@example.org from 192.0.2.{index % 10}. "
                + "A varied tokenizer sample needs complete sentences and "
                "code-like_name tokens. "
                * 3
            )
            record = {
                "id": f"{label}-{index}",
                "metadata": {
                    "license": None if index == 0 else "CC-BY-4.0",
                    "provenance": f"fixture:{label}:{index}",
                    "url": f"https://example.invalid/{label}/{index}",
                },
                "source": label,
                "text": text,
            }
            handle.write(json.dumps(record) + "\n")


def _config(first_source: Path, second_source: Path) -> dict:
    return {
        "dataset": {
            "id": "test/corpus",
            "revision": "a" * 40,
            "upstream_card": "https://example.invalid/test/corpus",
        },
        "filters": _policy(),
        "mixture": [
            {
                "category": "education",
                "sources": [
                    {
                        "blob_sha1": "b" * 40,
                        "component": "education",
                        "compressed_bytes": first_source.stat().st_size,
                        "dataset_id": "test/education",
                        "file": "education/part.jsonl.gz",
                        "required_non_null_provenance": [
                            "upstream_id",
                            "upstream_source",
                            "license",
                            "url",
                            "collection_provenance",
                        ],
                        "retain_provenance": {
                            "collection_provenance": "metadata.provenance",
                            "license": "metadata.license",
                            "upstream_id": "id",
                            "upstream_source": "source",
                            "url": "metadata.url",
                        },
                        "revision": "d" * 40,
                        "url": first_source.as_uri(),
                    }
                ],
                "weight": 0.6,
            },
            {
                "category": "code",
                "sources": [
                    {
                        "blob_sha1": "c" * 40,
                        "component": "code",
                        "compressed_bytes": second_source.stat().st_size,
                        "file": "code/part.jsonl.gz",
                        "required_non_null_provenance": ["upstream_id", "license_basis"],
                        "retain_provenance": {"upstream_id": "id"},
                        "static_provenance": {
                            "license_basis": "fixture source-level open license"
                        },
                        "url": second_source.as_uri(),
                    }
                ],
                "weight": 0.4,
            },
        ],
        "purpose": "test",
        "target_uncompressed_text_bytes": 5000,
        "version": "0.1",
    }


def test_filter_redacts_pii_and_rejects_credentials() -> None:
    text = (
        "Please email afnan@example.com or connect to 192.0.2.42 for this sufficiently long "
        "educational example about language-model data preparation."
    )
    result = filter_document(text, _policy())
    assert result.rejection_reason is None
    assert result.text is not None
    assert "afnan@example.com" not in result.text
    assert "192.0.2.42" not in result.text
    assert "[EMAIL]" in result.text
    assert "[IP_ADDRESS]" in result.text
    assert result.redactions == {"email": 1, "ipv4": 1}

    secret = filter_document(
        "This document is long enough, but api_key=ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890 "
        "must never enter a model corpus.",
        _policy(),
    )
    assert secret.text is None
    assert secret.rejection_reason == "secret:assigned_credential"


def test_build_validate_and_resume_local_sample(tmp_path: Path) -> None:
    first_source = tmp_path / "education.jsonl.gz"
    second_source = tmp_path / "code.jsonl.gz"
    _write_source(first_source, "education")
    _write_source(second_source, "code")
    config = _config(first_source, second_source)
    output = tmp_path / "sample"

    manifest = build_sample(config, output)
    result = validate_sample(output)
    assert manifest["status"] == "complete"
    assert result["text_utf8_bytes"] >= 5000
    assert manifest["statistics"]["pii_redactions"]["email"] > 0
    assert manifest["statistics"]["pii_redactions"]["ipv4"] > 0
    assert manifest["statistics"]["rejections"]["missing_provenance:license"] == 1
    first_record = json.loads((output / "sample.jsonl").read_text().splitlines()[0])
    assert first_record["provenance"]["dataset_id"] == "test/education"
    assert first_record["provenance"]["dataset_revision"] == "d" * 40
    assert first_record["retained_provenance"]["license"] == "CC-BY-4.0"
    assert first_record["retained_provenance"]["upstream_id"] == "education-1"
    before = (output / "sample.jsonl").read_bytes()

    resumed = build_sample(config, output, resume=True)
    assert resumed["output"]["sha256"] == manifest["output"]["sha256"]
    assert (output / "sample.jsonl").read_bytes() == before
    with pytest.raises(FileExistsError):
        build_sample(config, output)


def test_resume_rejects_changed_target(tmp_path: Path) -> None:
    first_source = tmp_path / "education.jsonl.gz"
    second_source = tmp_path / "code.jsonl.gz"
    _write_source(first_source, "education")
    _write_source(second_source, "code")
    config = _config(first_source, second_source)
    output = tmp_path / "sample"
    build_sample(config, output)

    changed = deepcopy(config)
    changed["target_uncompressed_text_bytes"] = 6000
    with pytest.raises(ValueError, match="different config or target"):
        build_sample(changed, output, resume=True)
    with pytest.raises(ValueError, match="does not match the expected config"):
        validate_sample(output, expected_config=changed)
