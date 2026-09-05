from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lumi.posttraining_candidate_mix_reaudit import (
    build_candidate_mix_reaudit_verification,
    file_sha256,
    load_candidate_mix_reaudit_plan,
    rebuild_candidate_mix_reaudit,
    validate_candidate_mix_reaudit_report,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-candidate-mix-reaudit-v0.2.json"
REPORT = ROOT / "evidence/lumi-4b-sft-candidate-mix-reaudit-v0.2.json"
VERIFICATION = ROOT / "evidence/lumi-4b-sft-candidate-mix-reaudit-verification-v0.2.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_candidate_mix_reaudit_reproduces_exactly() -> None:
    plan = load_candidate_mix_reaudit_plan(PLAN, repository_root=ROOT)
    rebuilt = rebuild_candidate_mix_reaudit(
        plan=plan,
        plan_file_sha256=file_sha256(PLAN),
        repository_root=ROOT,
    )
    observed = _load(REPORT)
    assert rebuilt == observed
    validate_candidate_mix_reaudit_report(observed)
    assert observed["candidate_pool"]["accepted_train_supervised_tokens"] == 1_059_171
    assert observed["candidate_pool"]["gap_to_10m_train_supervised_tokens"] == 8_940_829
    assert observed["candidate_pool"]["candidate_records_read"] == 12_401
    assert observed["candidate_pool"][
        "registry_admissible_validation_supervised_tokens"
    ] == 90_519


def test_all_rows_pass_and_metadata_blockers_are_closed() -> None:
    report = _load(REPORT)
    assert report["generic_sft_validation"]["accepted"] == 12_401
    assert report["generic_sft_validation"]["rejected"] == 0
    assert report["generic_sft_validation"]["rejection_counts"] == {}
    assert all(value == 0 for value in report["exact_duplicate_audit"].values())
    near = report["near_duplicate_audit"]
    assert near["multi_record_components"] == 20
    assert near["duplicate_extra_records"] == 20
    assert near["cross_source_components"] == 0
    assert near["train_validation_components"] == 0
    blockers = report["closure"]["blockers"]
    assert "coherelabs_aya_dataset_validation_split_not_registry_bound" not in blockers
    assert "lumi_safety_tool_sft_v0_1_requires_top_level_split_materialization" not in blockers
    assert (
        "lumi_safety_tool_sft_v0_1_row_provenance_still_marks_owner_review_pending"
        not in blockers
    )


def test_verification_reproduces_and_repo_evidence_is_raw_text_free() -> None:
    plan = load_candidate_mix_reaudit_plan(PLAN, repository_root=ROOT)
    report = _load(REPORT)
    expected = build_candidate_mix_reaudit_verification(
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
        load_candidate_mix_reaudit_plan(path, repository_root=ROOT)
