from __future__ import annotations

import json
import math
import re
from pathlib import Path


def test_candidate_data_mix_is_complete_and_sums_to_80b() -> None:
    path = Path(__file__).parents[1] / "configs" / "data-mix-v0.1.json"
    with path.open(encoding="utf-8") as handle:
        plan = json.load(handle)
    mixture = plan["mixture"]
    assert math.isclose(sum(item["weight"] for item in mixture), 1.0)
    assert sum(item["target_tokens_at_80b"] for item in mixture) == 80_000_000_000
    for item in mixture:
        assert item["source_components"]
        assert item["target_tokens_at_80b"] == round(item["weight"] * 80_000_000_000)
    revision = plan["primary_dataset"]["revision"]
    assert re.fullmatch(r"[0-9a-f]{40}", revision)
    assert revision != "main"


def test_tokenizer_canary_covers_every_candidate_domain() -> None:
    root = Path(__file__).parents[1]
    with (root / "configs" / "data-mix-v0.1.json").open(encoding="utf-8") as handle:
        candidate = json.load(handle)
    with (root / "configs" / "tokenizer-canary-v0.1.json").open(encoding="utf-8") as handle:
        canary = json.load(handle)
    assert canary["dataset"]["id"] == candidate["primary_dataset"]["id"]
    assert canary["dataset"]["revision"] == candidate["primary_dataset"]["revision"]
    assert {item["category"] for item in canary["mixture"]} == {
        item["category"] for item in candidate["mixture"]
    }
    assert math.isclose(sum(item["weight"] for item in canary["mixture"]), 1.0)


def test_component_inventory_covers_mix_and_marks_missing_license_metadata() -> None:
    root = Path(__file__).parents[1]
    with (root / "configs" / "data-mix-v0.1.json").open(encoding="utf-8") as handle:
        candidate = json.load(handle)
    with (root / "configs" / "component-sources-v0.1.json").open(
        encoding="utf-8"
    ) as handle:
        inventory = json.load(handle)
    expected = {
        component for item in candidate["mixture"] for component in item["source_components"]
    }
    sources = inventory["sources"]
    assert {source["component"] for source in sources} == expected
    assert len({source["dataset_id"] for source in sources}) == len(sources)
    for source in sources:
        assert source["dataset_id"].endswith("_filtered")
        assert re.fullmatch(r"[0-9a-f]{40}", source["revision"])
        assert source["data_file_count"] > 0
        assert source["schema_anchor"].endswith(".json.gz")
    cccc = next(source for source in sources if source["component"] == "cccc")
    assert cccc["license_paths"] == []
    assert cccc["license_basis"]["kind"] == "source_level_domain_audit"
    assert "schema_mismatch_caveat" in cccc["status"]

    audit_path = root / inventory["bounded_audit_report"]
    with audit_path.open(encoding="utf-8") as handle:
        audit = json.load(handle)
    assert {row[0] for row in audit["results"]} == expected
    summary = audit["summary"]
    assert summary["rows_audited"] == 44_061
    assert summary["license_or_documented_basis_non_null"] == summary["rows_audited"]
    assert summary["upstream_id_non_null"] == summary["rows_audited"]
    assert summary["provenance_non_null"] == summary["rows_audited"]
    assert summary["url_non_null"] < summary["rows_audited"]


def test_provenance_canary_uses_pinned_component_repositories() -> None:
    root = Path(__file__).parents[1]
    with (root / "configs" / "provenance-canary-v0.2.json").open(encoding="utf-8") as handle:
        canary = json.load(handle)
    assert math.isclose(sum(item["weight"] for item in canary["mixture"]), 1.0)
    for entry in canary["mixture"]:
        for source in entry["sources"]:
            assert source["dataset_id"].endswith("_filtered")
            assert re.fullmatch(r"[0-9a-f]{40}", source["revision"])
            assert "upstream_id" in source["required_non_null_provenance"]
            assert "collection_provenance" in source["required_non_null_provenance"]
            assert "license" in source["required_non_null_provenance"] or (
                "license_basis" in source["required_non_null_provenance"]
            )
