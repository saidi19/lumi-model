from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_aya_materialization import (
    DECISIONS_SCHEMA_VERSION,
    EXPECTED_AUTHORIZATION,
    EXPECTED_OWNER_STATEMENT,
    EXPECTED_PREFLIGHT,
    build_materialization_report,
    load_aya_materialization_plan,
    render_owner_review_html,
    validate_materialization_report,
    verify_exact_preflight,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "lumi-4b-sft-aya-bengali-materialization-v0.1.json"
PREFLIGHT = ROOT / "evidence" / "lumi-4b-sft-aya-bengali-repair-preflight-v0.1.json"


def test_aya_materialization_plan_exactly_matches_owner_boundary() -> None:
    plan = load_aya_materialization_plan(PLAN, repository_root=ROOT)
    assert plan["owner_authorization"]["statement"] == EXPECTED_OWNER_STATEMENT
    assert plan["authorization"] == EXPECTED_AUTHORIZATION
    assert plan["limits"]["exact_private_artifacts"] == 3
    assert plan["limits"]["maximum_candidate_records"] == 1_478
    assert plan["limits"]["maximum_supervised_tokens"] == 577_453
    assert plan["authorization"]["network_access_authorized"] is False
    assert plan["authorization"]["paid_gpu_authorized"] is False
    assert plan["authorization"]["source_registry_promotion_authorized"] is False
    assert plan["authorization"]["production_sft_authorized"] is False


def test_aya_materialization_binds_exact_preflight() -> None:
    report = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    verify_exact_preflight(report)
    assert report["candidate"]["records"] == EXPECTED_PREFLIGHT["candidate_records"]
    assert report["candidate"]["supervised_tokens"] == EXPECTED_PREFLIGHT["supervised_tokens"]
    assert report["candidate"]["materialized"] is False


def test_owner_review_html_has_no_preselected_decisions() -> None:
    sample = [
        {
            "messages": [
                {"content": "<unsafe-looking prompt>", "role": "user"},
                {"content": "বাংলা উত্তর", "role": "assistant"},
            ],
            "record_id": "aya-ben-example",
        }
    ]
    rendered = render_owner_review_html(
        sample,
        attribution_file_sha256="a" * 64,
        candidate_file_sha256="b" * 64,
        sample_record_ids_sha256="c" * 64,
    )
    assert "&lt;unsafe-looking prompt&gt;" in rendered
    assert '<article class="card" data-record-id="aya-ben-example">' in rendered
    assert '<article class="card" data-decision=' not in rendered
    assert DECISIONS_SCHEMA_VERSION in rendered
    assert "Review all ${recordIds.length} rows before exporting." in rendered
    assert "JSON.stringify(payload, null, 2) + '\\n'" in rendered
    assert "JSON.stringify(payload, null, 2) + '\n'" not in rendered


def test_materialization_report_is_raw_text_free_and_nonproduction() -> None:
    artifacts = [
        {
            "bytes": 10,
            "contains_raw_training_text": True,
            "name": "accepted_records",
            "path": "private/accepted-records.jsonl",
            "sha256": "a" * 64,
        },
        {
            "bytes": 10,
            "contains_raw_training_text": False,
            "name": "attribution_manifest",
            "path": "private/attribution-manifest.jsonl",
            "sha256": "b" * 64,
        },
        {
            "bytes": 10,
            "contains_raw_training_text": True,
            "name": "owner_review_packet",
            "path": "private/owner-review-32.html",
            "sha256": "c" * 64,
        },
    ]
    report = build_materialization_report(
        artifacts=artifacts,
        candidate_records=1_478,
        owner_sample_record_ids_sha256="d" * 64,
        owner_sample_records=32,
        plan_file_sha256="e" * 64,
        supervised_tokens=577_453,
    )
    validate_materialization_report(report)
    serialized = json.dumps(report, ensure_ascii=False)
    assert "private prompt" not in serialized
    assert report["raw_training_text_in_report"] is False
    assert report["production"]["production_eligible"] is False
    assert report["owner_review"]["complete"] is False


def test_materialization_plan_rejects_tampered_owner_statement(tmp_path: Path) -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    plan["owner_authorization"]["statement"] += " changed"
    tampered = tmp_path / "plan.json"
    tampered.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="owner authorization changed"):
        load_aya_materialization_plan(tampered, repository_root=ROOT)
