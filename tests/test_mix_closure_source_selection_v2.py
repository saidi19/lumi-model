from __future__ import annotations

import json
from pathlib import Path

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_mix_closure_source_selection_v2 import (
    build_mix_closure_source_selection_verification_v2,
    load_mix_closure_source_selection_plan_v2,
    rebuild_mix_closure_source_selection_v2,
    validate_mix_closure_source_selection_report_v2,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-10m-mix-closure-source-selection-v0.2.json"
REPORT = ROOT / "evidence/lumi-4b-sft-10m-mix-closure-source-selection-v0.2.json"
VERIFICATION = ROOT / "evidence/lumi-4b-sft-10m-mix-closure-source-selection-verification-v0.2.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v0_14_plan_is_metadata_only_and_bound_to_authorized_reaudit() -> None:
    plan = load_mix_closure_source_selection_plan_v2(PLAN, repository_root=ROOT)
    assert plan["limits"]["maximum_network_bytes"] == 16 * 1024 * 1024
    assert plan["limits"]["bulk_dataset_payload_bytes"] == 0
    assert plan["authorization"]["bulk_dataset_download_authorized"] is False
    assert plan["authorization"]["paid_gpu_authorized"] is False
    assert plan["authorization"]["production_sft_authorized"] is False
    assert (
        plan["bindings"]["candidate_mix_reaudit"]["sha256"]
        == "ab9d3575ae7de7c10858ef4abb72d667f986e1b514db49601eb044935008cd60"
    )


def test_v0_14_report_reproduces_and_closes_every_floor_and_cap() -> None:
    plan = load_mix_closure_source_selection_plan_v2(PLAN, repository_root=ROOT)
    rebuilt = rebuild_mix_closure_source_selection_v2(
        plan=plan,
        plan_file_sha256=file_sha256(PLAN),
        repository_root=ROOT,
    )
    observed = _load(REPORT)
    assert rebuilt == observed
    validate_mix_closure_source_selection_report_v2(observed)
    projected = observed["projected_mix"]
    assert projected["added_train_supervised_tokens"] == 3_200_000
    assert projected["train_supervised_tokens"] == 10_454_668
    assert projected["all_task_floors_met"] is True
    assert projected["all_language_floors_met"] is True
    assert projected["all_source_caps_met"] is True
    assert projected["all_source_family_caps_met"] is True


def test_v0_14_conditional_and_excluded_sources_do_not_close_the_mix() -> None:
    report = _load(REPORT)
    projected_sources = report["projected_mix"]["source_train_supervised_tokens"]
    assert "salesforce_xlam_function_calling_60k" not in projected_sources
    assert "nvidia_helpsteer" not in projected_sources
    for row in report["decisions"]["exclusions"]:
        assert row["counted_in_guaranteed_closure"] is False


def test_v0_14_verification_reproduces_and_outputs_are_raw_text_free() -> None:
    plan = load_mix_closure_source_selection_plan_v2(PLAN, repository_root=ROOT)
    report = _load(REPORT)
    expected = build_mix_closure_source_selection_verification_v2(
        plan=plan,
        report=report,
        report_file_sha256=file_sha256(REPORT),
    )
    assert expected == _load(VERIFICATION)
    for path in (REPORT, VERIFICATION):
        serialized = path.read_text(encoding="utf-8")
        for forbidden in ('"messages"', '"assistant_response"', '"raw_text"'):
            assert forbidden not in serialized
