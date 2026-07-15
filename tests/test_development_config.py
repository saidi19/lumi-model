from __future__ import annotations

import gzip
import json
import math
import re
from pathlib import Path

from lumi.data_quality import sha256_file
from lumi.development_config import compose_development_config


def test_development_config_composes_all_23_weighted_components() -> None:
    root = Path(__file__).parents[1]
    config = compose_development_config(
        root / "configs" / "development-sample-v0.1.json",
        repository_root=root,
    )
    assert config["target_uncompressed_text_bytes"] == 1_000_000_000
    assert math.isclose(sum(entry["weight"] for entry in config["mixture"]), 1.0)
    sources = [source for entry in config["mixture"] for source in entry["sources"]]
    assert len(sources) == 23
    assert len({source["component"] for source in sources}) == 23
    for entry in config["mixture"]:
        assert math.isclose(
            sum(source["weight_within_category"] for source in entry["sources"]),
            1.0,
        )
    for source in sources:
        assert source["dataset_id"].endswith("_filtered")
        assert re.fullmatch(r"[0-9a-f]{40}", source["revision"])
        assert re.fullmatch(r"[0-9a-f]{40}", source["blob_sha1"])
        assert source["compressed_bytes"] > 0
        assert "upstream_id" in source["required_non_null_provenance"]
        assert "collection_provenance" in source["required_non_null_provenance"]
    cccc = next(source for source in sources if source["component"] == "cccc")
    assert "license_basis" in cccc["static_provenance"]
    assert "license_basis" in cccc["required_non_null_provenance"]
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
        for item in config["composition"].values()
    )


def test_representative_tokenizer_plan_uses_minimum_six_gigabyte_target() -> None:
    root = Path(__file__).parents[1]
    config = compose_development_config(
        root / "configs" / "tokenizer-sample-v0.1.json",
        repository_root=root,
    )
    assert config["target_uncompressed_text_bytes"] == 6_000_000_000
    assert "multilingual slice is required" in config["purpose"]
    assert sum(len(entry["sources"]) for entry in config["mixture"]) == 23


def test_hash_only_benchmark_registry_is_pinned_and_contains_no_raw_text() -> None:
    root = Path(__file__).parents[1]
    quality = json.loads((root / "configs" / "data-quality-v0.1.json").read_text())
    registry_path = root / quality["benchmark_decontamination"]["registry"]
    assert sha256_file(registry_path) == quality["benchmark_decontamination"][
        "registry_sha256"
    ]
    with gzip.open(registry_path, "rt", encoding="utf-8") as handle:
        metadata = json.loads(next(handle))["_meta"]
        assert metadata["entries"] == 46_999
        assert metadata["raw_text_retained"] is False
        entries = 0
        benchmarks = set()
        for line in handle:
            entry = json.loads(line)
            assert "text" not in entry
            assert "prompt" not in entry
            assert re.fullmatch(r"[0-9a-f]{64}", entry["exact_sha256"])
            assert all(
                re.fullmatch(r"[0-9a-f]{64}", value)
                for value in entry["anchor_sha256"]
            )
            benchmarks.add(entry["benchmark"])
            entries += 1
    assert entries == metadata["entries"]
    assert benchmarks == {
        "arc_challenge",
        "arc_easy",
        "gsm8k",
        "hellaswag",
        "humaneval",
        "mbpp",
        "mmlu",
        "openbookqa",
        "truthfulqa",
    }


def test_benchmark_source_files_pin_revisions_sizes_and_hashes() -> None:
    root = Path(__file__).parents[1]
    sources = json.loads(
        (root / "configs" / "benchmark-sources-v0.1.json").read_text()
    )["sources"]
    assert len(sources) == 8
    assert {source["benchmark"] for source in sources} == {
        "arc",
        "gsm8k",
        "hellaswag",
        "humaneval",
        "mbpp",
        "mmlu",
        "openbookqa",
        "truthfulqa",
    }
    for source in sources:
        assert re.fullmatch(r"[0-9a-f]{40}", source["revision"])
        for source_file in source["files"]:
            assert source_file["bytes"] > 0
            assert re.fullmatch(r"[0-9a-f]{64}", source_file["sha256"])
            assert source_file["path"].endswith(".parquet")
