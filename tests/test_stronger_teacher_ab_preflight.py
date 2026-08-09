from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_stronger_teacher_ab_preflight import (
    CANDIDATE_MODEL,
    REMOTE_OBJECT_BYTES,
    build_stronger_teacher_ab_preflight_report,
    build_stronger_teacher_ab_preflight_verification,
    load_stronger_teacher_ab_preflight_plan,
    selected_schedule,
    validate_stronger_teacher_ab_preflight_report,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-stronger-teacher-ab-preflight-v0.1.json"
REPORT = ROOT / "evidence/lumi-4b-sft-stronger-teacher-ab-preflight-v0.1.json"
VERIFICATION = ROOT / (
    "evidence/lumi-4b-sft-stronger-teacher-ab-preflight-verification-v0.1.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _plan() -> dict:
    if not PLAN.is_file():
        pytest.skip("stronger-teacher A/B preflight plan is not materialized")
    raw = _load(PLAN)
    missing = []
    for name, binding in raw.get("bindings", {}).items():
        path = Path(binding["path"])
        resolved = path if path.is_absolute() else ROOT / path
        if not resolved.is_file():
            missing.append(name)
    if missing:
        pytest.skip(f"stronger-teacher private bindings are unavailable: {missing}")
    return load_stronger_teacher_ab_preflight_plan(PLAN, repository_root=ROOT)


def test_preflight_performs_no_model_download_or_inference() -> None:
    plan = _plan()
    assert plan["limits"]["model_download_bytes"] == 0
    assert plan["limits"]["local_inference_calls"] == 0
    assert plan["authorization"]["model_download_authorized"] is False
    assert plan["authorization"]["external_api_authorized"] is False
    assert plan["authorization"]["production_sft_authorized"] is False


def test_paired_schedule_is_exactly_five_by_twenty_cells() -> None:
    schedule = selected_schedule()
    assert len(schedule) == 100
    assert sum(slot.split == "train" for slot in schedule) == 80
    assert sum(slot.split == "validation" for slot in schedule) == 20


def test_candidate_object_and_quality_gates_are_exact() -> None:
    plan = _plan()
    assert CANDIDATE_MODEL["weights_license"] == "apache-2.0"
    assert CANDIDATE_MODEL["file_bytes"] == 9_001_752_960
    assert plan["next_gate"]["maximum_network_bytes"] == REMOTE_OBJECT_BYTES
    contract = plan["ab_contract"]
    assert contract["paired_prompts"] == 100
    assert contract["expected_final_outputs"] == 200
    assert contract["candidate_minimum_machine_acceptance_rate"] == 0.90
    assert contract["candidate_minimum_advantage_over_baseline"] == 0.15


def test_report_and_verification_reproduce_when_materialized() -> None:
    plan = _plan()
    if not REPORT.is_file() or not VERIFICATION.is_file():
        pytest.skip("stronger-teacher A/B evidence is not materialized")
    observed = _load(REPORT)
    rebuilt = build_stronger_teacher_ab_preflight_report(
        plan=plan,
        plan_file_sha256=file_sha256(PLAN),
        repository_root=ROOT,
    )
    assert rebuilt == observed
    validate_stronger_teacher_ab_preflight_report(observed)
    expected = build_stronger_teacher_ab_preflight_verification(
        plan=plan,
        report=observed,
        report_file_sha256=file_sha256(REPORT),
    )
    assert expected == _load(VERIFICATION)
    serialized = REPORT.read_text(encoding="utf-8") + VERIFICATION.read_text(
        encoding="utf-8"
    )
    for forbidden in ('"messages"', '"user_prompt"', '"assistant_response"'):
        assert forbidden not in serialized
