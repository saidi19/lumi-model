from __future__ import annotations

import json
from pathlib import Path

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_gpt_oss_teacher_preflight import (
    MODEL_BYTES,
    RUNTIME_DIGEST,
    TOTAL_NETWORK_BYTES,
    build_gpt_oss_teacher_preflight_report,
    build_gpt_oss_teacher_preflight_verification,
    load_gpt_oss_teacher_preflight_plan,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-gpt-oss-teacher-preflight-v0.1.json"
REPORT = ROOT / "evidence/lumi-4b-sft-gpt-oss-teacher-preflight-v0.1.json"
VERIFICATION = ROOT / (
    "evidence/lumi-4b-sft-gpt-oss-teacher-preflight-verification-v0.1.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _plan() -> dict:
    return load_gpt_oss_teacher_preflight_plan(PLAN, repository_root=ROOT)


def test_preflight_is_nonexecuting_and_keeps_wikisplit_at_zero() -> None:
    plan = _plan()
    assert plan["authorization"]["model_or_runtime_download_authorized"] is False
    assert plan["authorization"]["paid_or_cloud_gpu_authorized"] is False
    assert plan["authorization"]["candidate_generation_authorized"] is False
    report = build_gpt_oss_teacher_preflight_report(
        plan=plan,
        plan_file_sha256=file_sha256(PLAN),
        repository_root=ROOT,
    )
    assert report["wikisplit_production_token_credit"] == 0


def test_model_runtime_cost_and_canary_boundaries_are_exact() -> None:
    plan = _plan()
    assert len(plan["model_objects"]) == 25
    assert sum(row["bytes"] for row in plan["model_objects"]) == MODEL_BYTES
    assert plan["execution_boundary"]["maximum_network_bytes"] == TOTAL_NETWORK_BYTES
    assert plan["runtime"]["container_image_digest"] == RUNTIME_DIGEST
    assert plan["execution_boundary"]["maximum_usd_before_tax"] == 17.16
    assert plan["canary_contract"]["exact_final_outputs"] == 160
    assert plan["canary_contract"]["exact_prompt_cells"] == 16


def test_report_and_verification_reproduce() -> None:
    plan = _plan()
    observed = _load(REPORT)
    rebuilt = build_gpt_oss_teacher_preflight_report(
        plan=plan,
        plan_file_sha256=file_sha256(PLAN),
        repository_root=ROOT,
    )
    assert observed == rebuilt
    expected = build_gpt_oss_teacher_preflight_verification(
        report=observed,
        report_file_sha256=file_sha256(REPORT),
    )
    assert expected == _load(VERIFICATION)
