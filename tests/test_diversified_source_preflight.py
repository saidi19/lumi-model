from __future__ import annotations

import json
from pathlib import Path

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_diversified_source_preflight import (
    build_diversified_source_preflight_verification,
    load_diversified_source_preflight_plan,
    rebuild_diversified_source_preflight,
    validate_diversified_source_preflight_report,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-diversified-source-preflight-v0.1.json"
REPORT = ROOT / "evidence/lumi-4b-sft-diversified-source-preflight-v0.1.json"
VERIFICATION = (
    ROOT / "evidence/lumi-4b-sft-diversified-source-preflight-verification-v0.1.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_plan_is_metadata_only_and_non_authorizing() -> None:
    plan = load_diversified_source_preflight_plan(PLAN, repository_root=ROOT)
    assert plan["limits"]["dataset_payload_bytes"] == 0
    assert plan["authorization"]["dataset_payload_download_authorized"] is False
    assert plan["authorization"]["paid_gpu_authorized"] is False
    assert plan["authorization"]["production_sft_authorized"] is False


def test_report_reproduces_and_closes_projected_mix() -> None:
    plan = load_diversified_source_preflight_plan(PLAN, repository_root=ROOT)
    rebuilt = rebuild_diversified_source_preflight(
        plan=plan,
        plan_file_sha256=file_sha256(PLAN),
        repository_root=ROOT,
    )
    observed = _load(REPORT)
    assert rebuilt == observed
    validate_diversified_source_preflight_report(observed)
    projected = observed["projected_mix"]
    assert projected["train_supervised_tokens"] == 10_721_793
    assert projected["all_task_floors_met"] is True
    assert projected["all_language_floors_met"] is True
    assert projected["all_source_caps_met"] is True


def test_verification_reproduces_and_outputs_are_raw_text_free() -> None:
    plan = load_diversified_source_preflight_plan(PLAN, repository_root=ROOT)
    report = _load(REPORT)
    expected = build_diversified_source_preflight_verification(
        plan=plan,
        report=report,
        report_file_sha256=file_sha256(REPORT),
    )
    assert expected == _load(VERIFICATION)
    for path in (REPORT, VERIFICATION):
        serialized = path.read_text(encoding="utf-8")
        for forbidden in ('"messages"', '"content"', '"assistant_response"'):
            assert forbidden not in serialized
