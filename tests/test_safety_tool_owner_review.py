from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_safety_tool_owner_review import (
    build_safety_tool_owner_review_html,
    load_owner_review_plan,
    sha256_value,
    validate_owner_decisions,
    validate_owner_sample,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs" / "lumi-4b-safety-tool-owner-review-v0.1.json"
RECEIPT_PATH = (
    ROOT
    / "artifacts"
    / "lumi-safety-tool-sft-pilot-v0.1"
    / "f7dacd1ceb30"
    / "attempt-004"
    / "owner-review-receipt-safety-tool-v0.1.json"
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _inputs() -> tuple[dict, list[dict], dict, dict]:
    plan = load_owner_review_plan(PLAN_PATH, repository_root=ROOT)
    sample = _read_jsonl((ROOT / plan["bindings"]["owner_sample"]["path"]).resolve())
    manifest = _read_json((ROOT / plan["bindings"]["pilot_manifest"]["path"]).resolve())
    evidence = _read_json(
        (ROOT / plan["bindings"]["candidate_evidence"]["path"]).resolve()
    )
    return plan, sample, manifest, evidence


def test_owner_review_sample_has_exact_skill_language_coverage() -> None:
    plan, sample, manifest, evidence = _inputs()
    coverage = validate_owner_sample(
        sample,
        manifest=manifest,
        candidate_evidence=evidence,
        plan=plan,
    )
    assert coverage == {
        "language_modes": 4,
        "records": 40,
        "records_per_skill_language_pair": 1,
        "skills": 10,
    }
    assert plan["authorization"]["ai_or_automated_owner_approval_authorized"] is False
    assert plan["authorization"]["production_sft_authorized"] is False


def test_owner_review_html_uses_dedicated_schema_and_remains_private() -> None:
    plan, sample, manifest, evidence = _inputs()
    validate_owner_sample(
        sample,
        manifest=manifest,
        candidate_evidence=evidence,
        plan=plan,
    )
    html = build_safety_tool_owner_review_html(
        sample,
        sample_sha256=plan["candidate"]["sample_internal_sha256"],
    )
    assert "Lumi safety/tool owner spot-check" in html
    assert "lumi.safety-tool-owner-decisions.v0.1" in html
    assert "lumi-safety-tool-owner-decisions-v0.1.json" in html
    assert "https://" not in html
    assert "permission to start SFT" in html


def test_human_owner_decisions_seal_non_promoting_raw_free_receipt() -> None:
    plan, sample, manifest, evidence = _inputs()
    validate_owner_sample(
        sample,
        manifest=manifest,
        candidate_evidence=evidence,
        plan=plan,
    )
    value = {
        "decisions": [
            {"accepted": True, "note": "", "record_id": row["record_id"]}
            for row in sample
        ],
        "reviewed_at": "2026-08-06T03:00:00Z",
        "reviewer_id": "owner-afnan",
        "sample_sha256": plan["candidate"]["sample_internal_sha256"],
        "schema_version": "lumi.safety-tool-owner-decisions.v0.1",
    }
    receipt = validate_owner_decisions(
        value,
        sample_rows=sample,
        sample_sha256=plan["candidate"]["sample_internal_sha256"],
        decision_file_sha256="a" * 64,
        plan_file_sha256="b" * 64,
    )
    assert receipt["accepted_records"] == 40
    assert receipt["rejected_records"] == 0
    assert receipt["production_eligible"] is False
    assert receipt["production_promotions"] == 0
    assert receipt["raw_training_text_retained"] is False
    seal = receipt.pop("receipt_sha256")
    assert seal == sha256_value(receipt)
    assert sample[0]["messages"][0]["content"] not in json.dumps(
        receipt, ensure_ascii=False
    )


def test_ai_alias_cannot_claim_owner_review() -> None:
    plan, sample, manifest, evidence = _inputs()
    validate_owner_sample(
        sample,
        manifest=manifest,
        candidate_evidence=evidence,
        plan=plan,
    )
    value = {
        "decisions": [
            {"accepted": True, "note": "", "record_id": row["record_id"]}
            for row in sample
        ],
        "reviewed_at": "2026-08-06T03:00:00Z",
        "reviewer_id": "chatgpt",
        "sample_sha256": plan["candidate"]["sample_internal_sha256"],
        "schema_version": "lumi.safety-tool-owner-decisions.v0.1",
    }
    with pytest.raises(ValueError, match="cannot claim owner review"):
        validate_owner_decisions(
            value,
            sample_rows=sample,
            sample_sha256=plan["candidate"]["sample_internal_sha256"],
            decision_file_sha256="a" * 64,
            plan_file_sha256="b" * 64,
        )


def test_actual_owner_receipt_is_complete_sealed_and_non_promoting() -> None:
    receipt = _read_json(RECEIPT_PATH)
    seal = receipt.pop("receipt_sha256")
    assert seal == sha256_value(receipt)
    assert receipt["accepted_records"] == 40
    assert receipt["rejected_records"] == 0
    assert receipt["all_sampled_rows_accepted"] is True
    assert receipt["status"] == (
        "owner_spot_check_passed_source_registry_promotion_still_separate"
    )
    assert receipt["production_eligible"] is False
    assert receipt["production_promotions"] == 0
    assert receipt["raw_training_text_retained"] is False
