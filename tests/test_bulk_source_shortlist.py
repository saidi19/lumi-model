from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lumi.posttraining_bulk_source_shortlist import (
    DEEPMIND_REVISION,
    OPENCODE_CANARY_OBJECT,
    OPENCODE_REVISION,
    file_sha256,
    load_bulk_source_shortlist_plan,
    rebuild_bulk_source_shortlist,
    validate_bulk_source_shortlist_report,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-bulk-source-shortlist-v0.1.json"
REPORT = ROOT / "evidence/lumi-4b-sft-bulk-source-shortlist-v0.1.json"
VERIFICATION = ROOT / "evidence/lumi-4b-sft-bulk-source-shortlist-verification-v0.1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_shortlist_has_no_payload_or_training_authority() -> None:
    plan = load_bulk_source_shortlist_plan(PLAN, repository_root=ROOT)
    assert plan["authorization"]["dataset_payload_download_authorized"] is False
    assert plan["authorization"]["paid_gpu_authorized"] is False
    assert plan["authorization"]["production_sft_authorized"] is False
    assert plan["limits"]["dataset_payload_bytes"] == 0


def test_exact_math_and_coding_boundaries_are_pinned() -> None:
    plan = load_bulk_source_shortlist_plan(PLAN, repository_root=ROOT)
    sources = {item["source_id"]: item for item in plan["shortlist"]}
    assert sources["deepmind_mathematics_dataset"]["revision"] == DEEPMIND_REVISION
    assert sources["nvidia_opencodeinstruct"]["revision"] == OPENCODE_REVISION
    assert sources["nvidia_opencodeinstruct"]["bounded_canary_object"] == (
        OPENCODE_CANARY_OBJECT
    )


def test_report_reproduces_and_keeps_realistic_projection() -> None:
    plan = load_bulk_source_shortlist_plan(PLAN, repository_root=ROOT)
    rebuilt = rebuild_bulk_source_shortlist(
        plan=plan,
        plan_file_sha256=file_sha256(PLAN),
        repository_root=ROOT,
    )
    observed = _load(REPORT)
    assert rebuilt == observed
    validate_bulk_source_shortlist_report(observed)
    assert observed["projections"]["deepmind_planning_generated_rows"] == 234_863
    assert observed["projections"]["remaining_gap_to_10m_after_both_priority_lanes"] == (
        4_631_515
    )


def test_outputs_are_raw_text_free_and_nonproduction() -> None:
    report = _load(REPORT)
    verification = _load(VERIFICATION)
    assert report["raw_training_text_retained"] is False
    assert report["production"]["dataset_payload_downloaded"] is False
    assert verification["production_eligible"] is False
    serialized = REPORT.read_text(encoding="utf-8")
    for forbidden in ('"messages"', '"question_en"', '"answer_en"', '"content"'):
        assert forbidden not in serialized


def test_plan_rejects_decision_drift(tmp_path: Path) -> None:
    changed = copy.deepcopy(_load(PLAN))
    changed["shortlist"][0]["decision"] = "production_eligible"
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="shortlist decision changed"):
        load_bulk_source_shortlist_plan(path, repository_root=ROOT)

