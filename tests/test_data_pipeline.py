from __future__ import annotations

import gzip
import json
import sqlite3
from copy import deepcopy
from pathlib import Path

import pytest

import lumi.data_pipeline as data_pipeline
from lumi.data_pipeline import (
    build_sample,
    detect_pii,
    detect_secret,
    filter_document,
    preflight_sample_config,
    validate_sample,
)
from lumi.data_quality import benchmark_signature_entry, sha256_file


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
                f"{label}-{index}. Contact person{index}@example.org from server IP "
                f"192.0.2.{index % 10}. "
                + "A varied tokenizer sample needs complete sentences and "
                "code-like_name tokens. " * 3
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
                        "static_provenance": {"license_basis": "fixture source-level open license"},
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


def _write_benchmark_registry(path: Path) -> None:
    benchmark_text = (
        "This private evaluation fixture contains a deliberately distinctive sequence of words "
        "that must not appear inside any model training sample."
    )
    entry = benchmark_signature_entry(
        benchmark="pipeline_fixture",
        example_id="1",
        split="test",
        text=benchmark_text,
        anchor_ngram_tokens=7,
        max_anchors=4,
        minimum_anchor_matches=2,
    )
    metadata = {
        "_meta": {
            "anchor_ngram_tokens": 7,
            "entries": 1,
            "normalization": "nfkc_casefold_whitespace_v1",
            "schema_version": 1,
        }
    }
    path.write_text(
        json.dumps(metadata, sort_keys=True) + "\n" + json.dumps(entry, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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


def test_filter_redacts_contextual_phone_numbers() -> None:
    text = (
        "For account support, call our phone number +1 (212) 555-0123 during normal business "
        "hours and never include private contact details in training data."
    )
    result = filter_document(text, _policy())
    assert result.rejection_reason is None
    assert result.text is not None
    assert "+1 (212) 555-0123" not in result.text
    assert "[PHONE_NUMBER]" in result.text
    assert result.redactions == {"phone": 1}


def test_filter_redacts_labelled_ip_ranges_and_phone_fields() -> None:
    text = (
        "The two nodes use fixed IP addresses (192.0.2.10 - 192.0.2.11), while the support "
        'record stores Phone = "612-555-0123" for this intentionally realistic safety test.'
    )
    result = filter_document(text, _policy())
    assert result.rejection_reason is None
    assert result.text is not None
    assert result.text.count("[IP_ADDRESS]") == 2
    assert result.text.count("[PHONE_NUMBER]") == 1
    assert result.redactions == {"ipv4": 2, "phone": 1}


def test_filter_redacts_private_ips_only_with_config_context() -> None:
    text = (
        "The export configuration grants /u/usr 192.168.1.2 (rw, sync) and gives the remote "
        "machine 192.168.1.3 access for this realistic network-safety fixture."
    )
    result = filter_document(text, _policy())
    assert result.rejection_reason is None
    assert result.text is not None
    assert result.text.count("[IP_ADDRESS]") == 2
    assert result.redactions == {"ipv4": 2}


@pytest.mark.parametrize(
    "value",
    [
        "software version 1.2.3.4",
        "source-code revision 1.2.3.4",
        "enzyme EC 3.2.1.3",
        "ORCID 0000-0002-1825-0097",
        "DOI 10.1000/1825-0097",
        "JCPDS card 00-047-1049",
        "publication years 2019-2020",
        "Sections 2.3.2.4 and 4.1.1.12",
        "TCP/IP documentation section §2.5.4.1",
        "private-range-looking heading 10.1.1.1 Elimination of attributes",
        "TCP/IP index entry §10.1.1.1",
        "server IP 999.999.999.999",
    ],
)
def test_filter_preserves_scientific_identifiers_and_versions(value: str) -> None:
    text = (
        f"This sufficiently long research record contains {value} as ordinary scholarly "
        "metadata that must remain intact after corpus filtering."
    )
    result = filter_document(text, _policy())
    assert result.rejection_reason is None
    assert result.text is not None
    assert value in result.text
    assert result.redactions == {}
    assert detect_pii(result.text) is None


def test_filter_preserves_wavenumbers_near_scientific_cell_text() -> None:
    text = (
        "The liquid cell spectrum reports wavenumbers 3400 3200 2800 2600 2400 cm-1, and "
        "these numeric axis labels are measurements rather than telephone numbers."
    )
    result = filter_document(text, _policy())
    assert result.rejection_reason is None
    assert result.text == text
    assert result.redactions == {}
    assert detect_pii(result.text) is None


@pytest.mark.parametrize(
    ("value", "kind"),
    [
        ("sk-ant-" + "a" * 40, "anthropic_api_key"),
        ("sk-proj-" + "A" * 40, "openai_api_key"),
        ("hf_" + "a" * 40, "huggingface_token"),
        ("glpat-" + "a" * 32, "gitlab_token"),
        ("npm_" + "a" * 36, "npm_token"),
        ("postgresql://lumi:supersecretpassword@db.example.invalid/model", "credentialed_uri"),
    ],
)
def test_stronger_secret_scanner(value: str, kind: str) -> None:
    assert detect_secret(f"Never retain this credential: {value}") == kind


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


def test_builds_pinned_parquet_source_with_sha256_provenance(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_path = tmp_path / "bangla.parquet"
    rows = []
    for index in range(12):
        rows.append(
            {
                "id": f"bn-{index}",
                "text": (
                    f"বাংলা নিবন্ধ {index} একটি নির্ভরযোগ্য বহুভাষিক টোকেনাইজার নমুনা তৈরির "
                    "জন্য স্বাভাবিক বাক্য, ইতিহাস, বিজ্ঞান এবং সংস্কৃতি নিয়ে বিস্তারিত আলোচনা "
                    "করে। প্রতিটি নথিতে আলাদা বিষয় ও শব্দভাণ্ডার রাখা হয়েছে।"
                ),
                "title": f"বাংলা শিরোনাম {index}",
                "url": f"https://bn.wikipedia.org/wiki/Fixture_{index}",
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)
    config = {
        "dataset": {
            "id": "test/corpus",
            "revision": "a" * 40,
            "upstream_card": "https://example.invalid/test/corpus",
        },
        "filters": _policy(),
        "mixture": [
            {
                "category": "multilingual_reference",
                "sources": [
                    {
                        "blob_sha256": sha256_file(parquet_path),
                        "component": "wikimedia_wikipedia_bn",
                        "compressed_bytes": parquet_path.stat().st_size,
                        "dataset_id": "wikimedia/wikipedia",
                        "file": "20231101.bn/fixture.parquet",
                        "format": "parquet",
                        "required_non_null_provenance": [
                            "upstream_id",
                            "upstream_url",
                            "title",
                            "language",
                            "license_basis",
                        ],
                        "retain_provenance": {
                            "title": "title",
                            "upstream_id": "id",
                            "upstream_url": "url",
                        },
                        "revision": "d" * 40,
                        "static_provenance": {
                            "language": "bn",
                            "license_basis": "fixture CC-BY-SA source-level basis",
                        },
                        "url": parquet_path.as_uri(),
                    }
                ],
                "weight": 1.0,
            }
        ],
        "purpose": "test Parquet source",
        "target_uncompressed_text_bytes": 900,
        "version": "0.1",
    }
    output = tmp_path / "parquet-sample"

    manifest = build_sample(config, output)
    validation = validate_sample(output, expected_config=config)
    record = json.loads((output / "sample.jsonl").read_text().splitlines()[0])

    assert validation["text_utf8_bytes"] >= 900
    assert manifest["source_files"][0]["format"] == "parquet"
    assert manifest["source_files"][0]["blob_sha256"] == sha256_file(parquet_path)
    assert record["provenance"]["source_blob_sha256"] == sha256_file(parquet_path)
    assert record["retained_provenance"]["language"] == "bn"
    assert record["retained_provenance"]["upstream_url"].startswith(
        "https://bn.wikipedia.org/wiki/"
    )


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


def test_network_retry_is_persisted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first_source = tmp_path / "education.jsonl.gz"
    second_source = tmp_path / "code.jsonl.gz"
    _write_source(first_source, "education")
    _write_source(second_source, "code")
    config = _config(first_source, second_source)
    output = tmp_path / "retry-sample"
    original_iter = data_pipeline._iter_source_lines
    attempts = 0

    def flaky_iter(dataset: dict, source: dict, timeout_seconds: float):
        nonlocal attempts
        if attempts == 0:
            attempts += 1
            raise OSError("temporary network failure")
        yield from original_iter(dataset, source, timeout_seconds)

    monkeypatch.setattr(data_pipeline, "_iter_source_lines", flaky_iter)
    monkeypatch.setattr(data_pipeline.time, "sleep", lambda _: None)
    manifest = build_sample(config, output, retries=1)
    state = json.loads((output / "run-state.json").read_text())
    assert manifest["statistics"]["network_retries"] == 1
    assert state["network_retries"] == 1


def test_retry_forever_waits_through_multiple_network_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_source = tmp_path / "education.jsonl.gz"
    second_source = tmp_path / "code.jsonl.gz"
    _write_source(first_source, "education")
    _write_source(second_source, "code")
    config = _config(first_source, second_source)
    output = tmp_path / "persistent-retry-sample"
    original_iter = data_pipeline._iter_source_lines
    attempts = 0

    def repeatedly_flaky_iter(dataset: dict, source: dict, timeout_seconds: float):
        nonlocal attempts
        if attempts < 3:
            attempts += 1
            raise OSError("temporary network failure")
        yield from original_iter(dataset, source, timeout_seconds)

    monkeypatch.setattr(data_pipeline, "_iter_source_lines", repeatedly_flaky_iter)
    monkeypatch.setattr(data_pipeline.time, "sleep", lambda _: None)
    manifest = build_sample(config, output, retries=0, retry_forever=True)
    assert manifest["statistics"]["network_retries"] == 3


def test_validation_resumes_from_committed_sqlite_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_source = tmp_path / "education.jsonl.gz"
    second_source = tmp_path / "code.jsonl.gz"
    _write_source(first_source, "education")
    _write_source(second_source, "code")
    config = _config(first_source, second_source)
    config["target_uncompressed_text_bytes"] = 12000
    output = tmp_path / "checkpointed-validation"
    manifest = build_sample(config, output)
    original_detect_secret = data_pipeline.detect_secret
    calls = 0

    def interrupt_after_three_records(text: str) -> str | None:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("simulated validator interruption")
        return original_detect_secret(text)

    monkeypatch.setattr(data_pipeline, "detect_secret", interrupt_after_three_records)
    with pytest.raises(RuntimeError, match="simulated validator interruption"):
        validate_sample(
            output,
            resume_validation=False,
            checkpoint_interval=1,
        )
    with sqlite3.connect(output / "validation-state.sqlite3") as connection:
        status, line_number = connection.execute(
            "SELECT status, line_number FROM validation_progress WHERE id = 1"
        ).fetchone()
    assert status == "validating"
    assert line_number == 3

    monkeypatch.setattr(data_pipeline, "detect_secret", original_detect_secret)
    resumed = validate_sample(output, checkpoint_interval=1)
    assert resumed["validation_resumed_from_line"] == 3
    assert resumed["documents"] == manifest["statistics"]["documents"]


def test_weighted_source_quotas_force_every_source_into_sample(tmp_path: Path) -> None:
    first_source = tmp_path / "education.jsonl.gz"
    second_source = tmp_path / "code.jsonl.gz"
    _write_source(first_source, "education")
    _write_source(second_source, "code")
    config = _config(first_source, second_source)
    first = config["mixture"][0]["sources"][0]
    second = config["mixture"][1]["sources"][0]
    first["weight_within_category"] = 0.5
    second["weight_within_category"] = 0.5
    config["mixture"] = [
        {
            "category": "mixed",
            "sources": [first, second],
            "weight": 1.0,
        }
    ]
    config["target_uncompressed_text_bytes"] = 1000
    output = tmp_path / "weighted-sample"
    manifest = build_sample(config, output)
    validate_sample(output, expected_config=config)
    source_stats = [source["statistics"] for source in manifest["source_files"]]
    assert all(source["used"] for source in manifest["source_files"])
    assert [source["target_text_utf8_bytes"] for source in source_stats] == [500, 500]
    assert all(source["text_utf8_bytes"] >= 500 for source in source_stats)

    invalid = deepcopy(config)
    del invalid["mixture"][0]["sources"][1]["weight_within_category"]
    with pytest.raises(ValueError, match="weight either every source or no sources"):
        build_sample(invalid, tmp_path / "invalid-weighted")


def test_component_quota_spills_into_pinned_continuation_with_exact_provenance(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary.jsonl.gz"
    continuation = tmp_path / "continuation.jsonl.gz"
    unused = tmp_path / "unused.jsonl.gz"
    _write_source(primary, "primary")
    _write_source(continuation, "continuation")
    _write_source(unused, "unused")
    config = _config(primary, unused)
    source = config["mixture"][0]["sources"][0]
    source["weight_within_category"] = 1.0
    source["verified_capacity_text_utf8_bytes"] = 10_480
    source["verified_capacity_is_exhaustive"] = True
    source["continuation_files"] = [
        {
            "blob_sha1": "e" * 40,
            "compressed_bytes": continuation.stat().st_size,
            "file": "education/part-0001.jsonl.gz",
            "url": continuation.as_uri(),
            "verified_capacity_text_utf8_bytes": 5_000,
            "verified_capacity_is_exhaustive": False,
        }
    ]
    config["mixture"] = [config["mixture"][0]]
    config["mixture"][0]["weight"] = 1.0
    config["target_uncompressed_text_bytes"] = 15_000
    config["required_capacity_preflight_components"] = ["education"]

    preflight = preflight_sample_config(config)
    assert preflight["sources"][0]["status"] == "verified"
    output = tmp_path / "multi-shard"
    manifest = build_sample(config, output)
    validation = validate_sample(output, expected_config=config)

    group = manifest["source_files"][0]
    assert group["statistics"]["text_utf8_bytes"] >= 15_000
    assert [item["file"] for item in group["physical_files"]] == [
        "education/part.jsonl.gz",
        "education/part-0001.jsonl.gz",
    ]
    assert group["physical_files"][0]["statistics"]["text_utf8_bytes"] == 10_480
    assert group["physical_files"][1]["statistics"]["text_utf8_bytes"] > 0
    records = [json.loads(line) for line in (output / "sample.jsonl").read_text().splitlines()]
    continuation_records = [
        record
        for record in records
        if record["provenance"]["source_file"] == "education/part-0001.jsonl.gz"
    ]
    assert continuation_records
    assert all(
        record["provenance"]["source_blob_sha1"] == "e" * 40 for record in continuation_records
    )
    assert validation["text_utf8_bytes"] == group["statistics"]["text_utf8_bytes"]


def test_capacity_preflight_rejects_unverified_required_continuation(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary.jsonl.gz"
    continuation = tmp_path / "continuation.jsonl.gz"
    unused = tmp_path / "unused.jsonl.gz"
    _write_source(primary, "primary")
    _write_source(continuation, "continuation")
    _write_source(unused, "unused")
    config = _config(primary, unused)
    source = config["mixture"][0]["sources"][0]
    source["weight_within_category"] = 1.0
    source["verified_capacity_text_utf8_bytes"] = 10_480
    source["verified_capacity_is_exhaustive"] = True
    source["continuation_files"] = [
        {
            "blob_sha1": "e" * 40,
            "compressed_bytes": continuation.stat().st_size,
            "file": "education/part-0001.jsonl.gz",
            "url": continuation.as_uri(),
        }
    ]
    config["mixture"] = [config["mixture"][0]]
    config["mixture"][0]["weight"] = 1.0
    config["target_uncompressed_text_bytes"] = 15_000
    config["required_capacity_preflight_components"] = ["education"]
    output = tmp_path / "unverified"

    with pytest.raises(ValueError, match="required capacity preflight"):
        build_sample(config, output)
    assert not output.exists()


def test_verified_source_capacity_rejects_an_impossible_plan_before_writing(
    tmp_path: Path,
) -> None:
    first_source = tmp_path / "education.jsonl.gz"
    second_source = tmp_path / "code.jsonl.gz"
    _write_source(first_source, "education")
    _write_source(second_source, "code")
    config = _config(first_source, second_source)
    for entry in config["mixture"]:
        entry["sources"][0]["weight_within_category"] = 1.0
    config["mixture"][0]["sources"][0]["verified_capacity_text_utf8_bytes"] = 2_999
    output = tmp_path / "impossible-capacity"

    with pytest.raises(ValueError, match="exceeding verified capacity 2999"):
        build_sample(config, output)
    assert not output.exists()


def test_failed_build_records_failure_and_resume_clears_stale_error(
    tmp_path: Path,
) -> None:
    first_source = tmp_path / "education.jsonl.gz"
    second_source = tmp_path / "code.jsonl.gz"
    _write_source(first_source, "education")
    _write_source(second_source, "code")
    config = _config(first_source, second_source)
    config["target_uncompressed_text_bytes"] = 10_000_000
    output = tmp_path / "failed-state"

    with pytest.raises(RuntimeError, match="ended at"):
        build_sample(config, output)
    state = json.loads((output / "run-state.json").read_text())
    assert state["status"] == "failed"
    assert state["failure"]["type"] == "RuntimeError"
    assert "ended at" in state["failure"]["message"]

    with pytest.raises(RuntimeError, match="ended at"):
        build_sample(config, output, resume=True)
    resumed_state = json.loads((output / "run-state.json").read_text())
    assert resumed_state["status"] == "failed"
    assert resumed_state["failure"]["type"] == "RuntimeError"


def test_pipeline_applies_and_revalidates_composed_quality_gates(tmp_path: Path) -> None:
    first_source = tmp_path / "education.jsonl.gz"
    second_source = tmp_path / "code.jsonl.gz"
    _write_source(first_source, "education")
    _write_source(second_source, "code")
    registry = tmp_path / "benchmarks.jsonl"
    _write_benchmark_registry(registry)
    config = _config(first_source, second_source)
    config["target_uncompressed_text_bytes"] = 500
    config["quality"] = {
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
    output = tmp_path / "quality-sample"
    manifest = build_sample(config, output)
    result = validate_sample(output, expected_config=config)
    assert manifest["schema_version"] == 2
    assert manifest["implementation"]["pii_detector"] == "contextual-v4"
    assert manifest["statistics"]["quality"]["language_labels"] == {"en": 1, "zxx": 1}
    assert result["documents"] == 2
    with sqlite3.connect(output / "build-quality-state.sqlite3") as connection:
        indexed = connection.execute("SELECT COUNT(*) FROM near_duplicate_signatures").fetchone()[0]
    assert indexed == result["documents"]
    records = [json.loads(line) for line in (output / "sample.jsonl").read_text().splitlines()]
    assert all(record["quality"]["benchmark_registry_sha256"] for record in records)
