from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from lumi.corpus_capacity import build_corpus_capacity_report
from lumi.corpus_storage import (
    build_corpus_storage_report,
    evaluate_adaptive_progress,
    evaluate_corpus_storage_plan,
    read_simulation_progress,
    record_simulated_removal,
    record_simulated_shard,
    removal_event_id,
    validate_clean_shard_manifest,
    validate_removal_event,
    validate_source_object_manifest,
    verify_corpus_storage_report,
)

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "configs" / "corpus-object-storage-v0.1.json"


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _inputs() -> tuple[dict, dict, dict, dict, dict, dict]:
    return (
        _load("configs/corpus-object-storage-v0.1.json"),
        _load("configs/data-mix-v0.1.json"),
        _load("configs/component-sources-v0.1.json"),
        build_corpus_capacity_report(ROOT / "configs" / "corpus-capacity-gate-v0.1.json"),
        _load("configs/corpus-retention-snapshot-v0.1.json"),
        _load("configs/tokenizer-final-ab-v0.1.json"),
    )


def _report() -> dict:
    return build_corpus_storage_report(CONTRACT, repository_root=ROOT)


def _progress(report: dict, *, fraction: float = 0.0) -> dict:
    return {
        "categories": {
            row["category"]: {
                "sealed_unique_candidate_tokens": int(row["minimum_candidate_tokens"] * fraction)
            }
            for row in report["categories"]
        },
        "corrupt_shards": 0,
        "logical_storage_bytes": 0,
        "pending_removal_events": 0,
        "unsealed_shards": 0,
    }


def _clean_shard(report: dict, *, digest: str = "a" * 64) -> dict:
    category = "open_web"
    key = report["object_store"]["key_layout"]["clean_shard"].format(
        category=category,
        object_sha256=digest,
    )
    return {
        "accepted_text_utf8_bytes": 400_000_000,
        "category": category,
        "documents": 100_000,
        "object_key": key,
        "object_sha256": digest,
        "provenance_index_sha256": "b" * 64,
        "removal_ledger_generation": "c" * 64,
        "schema_version": 1,
        "shard_id": digest,
        "source_manifest_sha256": "d" * 64,
        "tokens": 100_000_000,
    }


def _removal_event(shard_id: str) -> dict:
    event = {
        "affected_shards": [shard_id],
        "created_at": "2026-07-19T12:00:00Z",
        "event_id": "",
        "lookup": {
            "content_sha256": "f" * 64,
            "dataset_id": "common-pile/cccc_filtered",
            "dataset_revision": "03a3de5713a0bb23267d26724346508af0f25327",
            "source_file": "dolma-cccc-filtered-0000.json.gz",
            "upstream_document_id": "fixture-document",
        },
        "reason": "upstream deletion request",
        "schema_version": 1,
    }
    event["event_id"] = removal_event_id(event)
    return event


def test_storage_design_passes_without_authorizing_resources_or_data() -> None:
    report = _report()
    assert report["decision"] == {
        "adaptive_acquisition_design": "pass",
        "bulk_corpus_materialization": "not_authorized",
        "cccc_license_basis": "owner_accepted_source_level_domain_audit",
        "cloud_resource_creation": "not_authorized",
        "full_4b_training": "not_authorized",
        "next_action": "provider_selection_pricing_and_bounded_io_canary",
        "paid_compute": "not_authorized",
        "provider_readiness": "blocked_unselected_unpriced_unapproved",
    }
    assert report["summary"] == {
        "candidate_hard_maximum_tokens": 150_000_000_000,
        "final_training_tokens": 80_000_000_000,
        "measured_retention_categories": 3,
        "minimum_candidate_tokens": 120_000_000_000,
        "provider": None,
        "sources": 23,
    }
    assert report["shards"]["estimated_clean_candidate_shards"] == 1_200
    assert report["shards"]["estimated_final_packed_shards"] == 160
    forecast = report["storage_forecast"]
    assert forecast["raw_source_payload_bytes"] == 0
    assert forecast["forecast_logical_bytes"] == 1_037_500_000_000
    assert forecast["hard_logical_storage_bytes"] == 2_000_000_000_000
    assert forecast["forecast_headroom_fraction"] == pytest.approx(0.48125)
    assert report["cost_policy"]["provider"] is None
    assert report["cost_policy"]["approved_monthly_cost_cap_usd"] is None
    measured = {
        row["category"]
        for row in report["categories"]
        if row["capacity_basis"] == "measured_retention_canary"
    }
    assert measured == {"open_web", "questions_and_discussion", "reference"}


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("create_cloud_resources", True, "must not authorize"),
        ("transfer_bulk_source_data", True, "must not authorize"),
        ("paid_compute", True, "must not authorize"),
    ],
)
def test_design_authorization_fails_closed(field: str, value: bool, match: str) -> None:
    specification, mixture, inventory, capacity, retention, tokenizer = _inputs()
    specification["authorization"][field] = value
    with pytest.raises(ValueError, match=match):
        evaluate_corpus_storage_plan(
            specification, mixture, inventory, capacity, retention, tokenizer
        )


def test_unapproved_provider_price_or_small_storage_cap_fails_closed() -> None:
    specification, mixture, inventory, capacity, retention, tokenizer = _inputs()
    priced = copy.deepcopy(specification)
    priced["cost_policy"]["provider"] = "invented-provider"
    priced["cost_policy"]["storage_usd_per_gb_month"] = 0.01
    with pytest.raises(ValueError, match="must not invent"):
        evaluate_corpus_storage_plan(priced, mixture, inventory, capacity, retention, tokenizer)

    too_small = copy.deepcopy(specification)
    too_small["storage_budget"]["hard_logical_storage_bytes"] = 1_000_000_000_000
    with pytest.raises(ValueError, match="forecast exceeds"):
        evaluate_corpus_storage_plan(too_small, mixture, inventory, capacity, retention, tokenizer)


def test_retention_snapshot_and_no_raw_payload_contract_are_pinned() -> None:
    specification, mixture, inventory, capacity, retention, tokenizer = _inputs()
    changed = copy.deepcopy(retention)
    changed["summary"]["tokenizer_json_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differs from the frozen tokenizer"):
        evaluate_corpus_storage_plan(
            specification, mixture, inventory, capacity, changed, tokenizer
        )

    raw_enabled = copy.deepcopy(specification)
    raw_enabled["object_store"]["raw_source_payload_persistence"] = True
    with pytest.raises(ValueError, match="persistence must stay disabled"):
        evaluate_corpus_storage_plan(
            raw_enabled, mixture, inventory, capacity, retention, tokenizer
        )


def test_future_source_clean_shard_and_removal_manifests_validate() -> None:
    report = _report()
    source = {
        "component": "cccc",
        "dataset_id": "common-pile/cccc_filtered",
        "dataset_revision": "03a3de5713a0bb23267d26724346508af0f25327",
        "license_policy": "owner_accepted_source_level_domain_audit",
        "provenance_paths": ["id", "source", "metadata.provenance", "metadata.url"],
        "schema_version": 1,
        "source_bytes": 490_876_041,
        "source_digest": {"algorithm": "sha256", "value": "1" * 64},
        "source_file": "dolma-cccc-filtered-0000.json.gz",
    }
    validate_source_object_manifest(source, report)
    pending_report = copy.deepcopy(report)
    pending_cccc = next(
        row for row in pending_report["sources"] if row["component"] == "cccc"
    )
    pending_cccc["license_basis_resolution"] = "pending_owner_decision"
    with pytest.raises(ValueError, match="pending explicit owner decision"):
        validate_source_object_manifest(source, pending_report)
    unsafe_source = {**source, "text": "must never enter a control manifest"}
    with pytest.raises(ValueError, match="forbidden field text"):
        validate_source_object_manifest(unsafe_source, report)
    wrong_license = {**source, "license_policy": "document_license_required"}
    with pytest.raises(ValueError, match="differs from its inventory"):
        validate_source_object_manifest(wrong_license, report)

    shard = _clean_shard(report)
    validate_clean_shard_manifest(shard, report)
    oversized = {**shard, "tokens": 200_000_001}
    with pytest.raises(ValueError, match="overshoot"):
        validate_clean_shard_manifest(oversized, report)
    wrong_key = {**shard, "object_key": "somewhere/else.bin"}
    with pytest.raises(ValueError, match="not canonical"):
        validate_clean_shard_manifest(wrong_key, report)

    event = _removal_event(shard["shard_id"])
    validate_removal_event(event, report)
    missing_lookup = copy.deepcopy(event)
    missing_lookup["lookup"].pop("content_sha256")
    missing_lookup["event_id"] = removal_event_id(missing_lookup)
    with pytest.raises(ValueError, match="lookup keys"):
        validate_removal_event(missing_lookup, report)
    wrong_event_id = {**event, "reason": "changed after hashing"}
    with pytest.raises(ValueError, match="canonical content digest"):
        validate_removal_event(wrong_event_id, report)


def test_adaptive_progress_stops_at_floors_and_halts_on_safety_events() -> None:
    report = _report()
    empty = evaluate_adaptive_progress(report, _progress(report))
    assert empty["state"] == "continue"
    assert empty["next_category"] == "books"
    assert empty["total_sealed_unique_candidate_tokens"] == 0

    ready_progress = _progress(report, fraction=1.0)
    ready = evaluate_adaptive_progress(report, ready_progress)
    assert ready["state"] == "candidate_ready"
    assert ready["total_sealed_unique_candidate_tokens"] == 120_000_000_000
    assert ready["authorization"] == "simulation_only"

    over = copy.deepcopy(ready_progress)
    open_web_floor = next(
        row["minimum_candidate_tokens"]
        for row in report["categories"]
        if row["category"] == "open_web"
    )
    over["categories"]["open_web"]["sealed_unique_candidate_tokens"] = (
        open_web_floor + report["shards"]["maximum_category_overshoot_tokens"] + 1
    )
    halted = evaluate_adaptive_progress(report, over)
    assert halted["state"] == "halted"
    assert halted["halt_reason"] == "category_overshoot_exceeds_one_shard"

    pending = copy.deepcopy(ready_progress)
    pending["pending_removal_events"] = 1
    assert evaluate_adaptive_progress(report, pending)["halt_reason"] == ("pending_removal_event")


def test_metadata_only_simulation_state_is_idempotent_resumable_and_removal_safe(
    tmp_path: Path,
) -> None:
    report = _report()
    database = tmp_path / "acquisition-simulation.sqlite3"
    shard = _clean_shard(report)
    assert record_simulated_shard(database, report, shard, logical_bytes=250_000_000)
    assert not record_simulated_shard(database, report, shard, logical_bytes=250_000_000)
    resumed = read_simulation_progress(database, report)
    open_web = next(row for row in resumed["categories"] if row["category"] == "open_web")
    assert open_web["sealed_unique_candidate_tokens"] == 100_000_000
    assert resumed["logical_storage_bytes"] == 250_000_000

    conflicting = copy.deepcopy(shard)
    conflicting["accepted_text_utf8_bytes"] += 1
    with pytest.raises(ValueError, match="reused with different metadata"):
        record_simulated_shard(database, report, conflicting, logical_bytes=250_000_000)

    event = _removal_event(shard["shard_id"])
    assert record_simulated_removal(database, report, event)
    assert not record_simulated_removal(database, report, event)
    blocked = read_simulation_progress(database, report)
    assert blocked["state"] == "halted"
    assert blocked["halt_reason"] == "pending_removal_event"
    blocked_open_web = next(row for row in blocked["categories"] if row["category"] == "open_web")
    assert blocked_open_web["sealed_unique_candidate_tokens"] == 0

    raw_database = database.read_bytes()
    assert b"must never enter a control manifest" not in raw_database


def test_sealed_report_reproduces_and_tampering_is_detected(tmp_path: Path) -> None:
    report = _report()
    report_path = tmp_path / "report.json"
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    report_path.write_text(encoded, encoding="utf-8")
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    manifest_path = tmp_path / "manifest.sha256"
    manifest_path.write_text(f"{digest}  report.json\n", encoding="utf-8")
    assert (
        verify_corpus_storage_report(
            CONTRACT,
            report_path,
            manifest_path,
            repository_root=ROOT,
        )
        == report
    )

    report_path.write_text(encoded + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="hash does not match"):
        verify_corpus_storage_report(
            CONTRACT,
            report_path,
            manifest_path,
            repository_root=ROOT,
        )
