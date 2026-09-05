from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lumi.posttraining_candidate_mix_preflight import (
    build_candidate_mix_verification,
    file_sha256,
    load_candidate_mix_plan,
    rebuild_candidate_mix_preflight,
    validate_candidate_mix_report,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-candidate-mix-closure-v0.1.json"
REPORT = ROOT / "evidence/lumi-4b-sft-candidate-mix-closure-v0.1.json"
VERIFICATION = ROOT / "evidence/lumi-4b-sft-candidate-mix-closure-verification-v0.1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_candidate_mix_report_reproduces_exactly() -> None:
    plan = load_candidate_mix_plan(PLAN, repository_root=ROOT)
    rebuilt = rebuild_candidate_mix_preflight(
        plan=plan,
        plan_file_sha256=file_sha256(PLAN),
        repository_root=ROOT,
    )
    observed = _load(REPORT)
    assert rebuilt == observed
    validate_candidate_mix_report(observed)
    assert observed["candidate_pool"]["accepted_train_supervised_tokens"] == 1_059_171
    assert observed["candidate_pool"]["gap_to_10m_train_supervised_tokens"] == 8_940_829
    assert observed["candidate_pool"]["candidate_records_read"] == 12_401


def test_candidate_mix_fails_closed_on_known_registry_and_schema_gaps() -> None:
    report = _load(REPORT)
    blockers = report["closure"]["blockers"]
    assert "coherelabs_aya_dataset_validation_split_not_registry_bound" in blockers
    assert "lumi_safety_tool_sft_v0_1_requires_top_level_split_materialization" in blockers
    assert "lumi_safety_tool_sft_v0_1_row_provenance_still_marks_owner_review_pending" in blockers
    assert report["closure"]["source_cap_violations"] == [
        "coherelabs_aya_dataset",
        "openassistant_oasst1",
    ]
    excluded = {item["source_id"]: item for item in report["excluded_sources"]}
    assert excluded["openr1_math_220k"]["counted_train_supervised_tokens"] == 0
    assert excluded["lumi_ai_assisted_sft_v0_3"]["registry_status"] == "absent"


def test_verification_reproduces_and_outputs_remain_raw_text_free() -> None:
    plan = load_candidate_mix_plan(PLAN, repository_root=ROOT)
    report = _load(REPORT)
    expected = build_candidate_mix_verification(
        plan=plan,
        report=report,
        report_file_sha256=file_sha256(REPORT),
    )
    assert expected == _load(VERIFICATION)
    for path in (REPORT, VERIFICATION):
        serialized = path.read_text(encoding="utf-8")
        for forbidden in (
            '"messages"',
            '"content"',
            '"assistant_response"',
            '"user_prompt"',
        ):
            assert forbidden not in serialized


def test_plan_rejects_authorization_drift(tmp_path: Path) -> None:
    changed = copy.deepcopy(_load(PLAN))
    changed["authorization"]["production_sft_authorized"] = True
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="authorization"):
        load_candidate_mix_plan(path, repository_root=ROOT)
