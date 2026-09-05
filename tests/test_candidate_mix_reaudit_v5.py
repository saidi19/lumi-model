from __future__ import annotations

import json
from pathlib import Path

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_candidate_mix_reaudit_v5 import (
    EXPECTED_RECORDS,
    EXPECTED_REGISTRY_SHA256,
    EXPECTED_TRAIN_TOKENS,
    EXPECTED_VALIDATION_TOKENS,
    build_candidate_mix_reaudit_v5_verification,
    load_candidate_mix_reaudit_v5_plan,
    rebuild_candidate_mix_reaudit_v5,
    validate_candidate_mix_reaudit_v5_report,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-candidate-mix-reaudit-v0.5.json"
REPORT = ROOT / "evidence/lumi-4b-sft-candidate-mix-reaudit-v0.5.json"
VERIFICATION = ROOT / "evidence/lumi-4b-sft-candidate-mix-reaudit-verification-v0.5.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v5_plan_is_bounded_and_non_authorizing() -> None:
    plan = load_candidate_mix_reaudit_v5_plan(PLAN, repository_root=ROOT)
    assert plan["limits"] == {
        "maximum_candidate_records_read": 300_000,
        "maximum_network_bytes": 0,
        "maximum_runtime_minutes": 180,
    }
    assert plan["bindings"]["source_registry"]["sha256"] == (
        EXPECTED_REGISTRY_SHA256
    )
    assert plan["authorization"]["production_sft_authorized"] is False


def test_v5_reaudit_reproduces_exactly() -> None:
    plan = load_candidate_mix_reaudit_v5_plan(PLAN, repository_root=ROOT)
    rebuilt = rebuild_candidate_mix_reaudit_v5(
        plan=plan,
        plan_file_sha256=file_sha256(PLAN),
        repository_root=ROOT,
    )
    observed = _load(REPORT)
    assert rebuilt == observed
    validate_candidate_mix_reaudit_v5_report(observed)
    pool = observed["candidate_pool"]
    assert pool["candidate_records_read"] == EXPECTED_RECORDS
    assert pool["accepted_train_supervised_tokens"] == EXPECTED_TRAIN_TOKENS
    assert pool["registry_admissible_validation_supervised_tokens"] == (
        EXPECTED_VALIDATION_TOKENS
    )


def test_v5_verification_reproduces_and_outputs_are_raw_text_free() -> None:
    plan = load_candidate_mix_reaudit_v5_plan(PLAN, repository_root=ROOT)
    report = _load(REPORT)
    expected = build_candidate_mix_reaudit_v5_verification(
        plan=plan,
        report=report,
        report_file_sha256=file_sha256(REPORT),
    )
    assert expected == _load(VERIFICATION)
    for path in (REPORT, VERIFICATION):
        serialized = path.read_text(encoding="utf-8")
        for forbidden in ('"messages"', '"content"', '"assistant_response"'):
            assert forbidden not in serialized
