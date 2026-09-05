from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_candidate_mix_reaudit_v4 import (
    EXPECTED_RECORDS,
    EXPECTED_TRAIN_TOKENS,
    EXPECTED_VALIDATION_TOKENS,
    build_candidate_mix_reaudit_v4_verification,
    load_candidate_mix_reaudit_v4_plan,
    rebuild_candidate_mix_reaudit_v4,
    validate_candidate_mix_reaudit_v4_report,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-candidate-mix-reaudit-v0.4.json"
REPORT = ROOT / "evidence/lumi-4b-sft-candidate-mix-reaudit-v0.4.json"
VERIFICATION = (
    ROOT / "evidence/lumi-4b-sft-candidate-mix-reaudit-verification-v0.4.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v4_plan_is_bounded_and_non_authorizing() -> None:
    plan = load_candidate_mix_reaudit_v4_plan(PLAN, repository_root=ROOT)
    assert plan["limits"]["maximum_candidate_records_read"] == 20_000
    assert plan["limits"]["maximum_network_bytes"] == 0
    assert all(
        value is False
        for key, value in plan["authorization"].items()
        if key.endswith("_authorized") and not key.startswith("local_")
    )
    assert plan["authorization"]["local_candidate_read_authorized"] is True


def test_v4_reaudit_reproduces_exactly() -> None:
    plan = load_candidate_mix_reaudit_v4_plan(PLAN, repository_root=ROOT)
    rebuilt = rebuild_candidate_mix_reaudit_v4(
        plan=plan,
        plan_file_sha256=file_sha256(PLAN),
        repository_root=ROOT,
    )
    observed = _load(REPORT)
    assert rebuilt == observed
    validate_candidate_mix_reaudit_v4_report(observed)
    pool = observed["candidate_pool"]
    assert pool["candidate_records_read"] == EXPECTED_RECORDS
    assert pool["accepted_train_supervised_tokens"] == EXPECTED_TRAIN_TOKENS
    assert (
        pool["registry_admissible_validation_supervised_tokens"]
        == EXPECTED_VALIDATION_TOKENS
    )


def test_v4_ai_provenance_blocker_is_truthfully_closed() -> None:
    report = _load(REPORT)
    metrics = report["source_metrics"]["lumi_ai_assisted_sft_v0_3"]
    assert metrics["records"] == 320
    assert metrics["rows_with_stale_pending_human_review_provenance"] == 0
    assert report["generic_sft_validation"]["accepted"] == EXPECTED_RECORDS
    assert report["generic_sft_validation"]["rejected"] == 0
    blockers = report["closure"]["blockers"]
    assert len(blockers) == 7
    assert not any("owner_review_provenance" in blocker for blocker in blockers)


def test_v4_verification_reproduces_and_outputs_are_raw_text_free() -> None:
    plan = load_candidate_mix_reaudit_v4_plan(PLAN, repository_root=ROOT)
    report = _load(REPORT)
    expected = build_candidate_mix_reaudit_v4_verification(
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


def test_v4_plan_rejects_authorization_drift(tmp_path: Path) -> None:
    changed = copy.deepcopy(_load(PLAN))
    changed["authorization"]["production_sft_authorized"] = True
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="authorization"):
        load_candidate_mix_reaudit_v4_plan(path, repository_root=ROOT)
