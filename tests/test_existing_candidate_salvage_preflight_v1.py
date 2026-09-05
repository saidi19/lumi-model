from pathlib import Path

from lumi.posttraining_existing_candidate_salvage_preflight_v1 import (
    build_report,
    build_verification,
    file_sha256,
    load_plan,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-existing-candidate-salvage-preflight-v0.1.json"


def test_existing_candidate_salvage_preflight_is_local_and_cap_safe() -> None:
    plan = load_plan(PLAN, ROOT)
    report = build_report(plan, file_sha256(PLAN), ROOT)
    assert report["conflict_controls"]["new_download_performed"] is False
    assert report["conflict_controls"]["new_inference_performed"] is False
    assert report["conflict_controls"]["aya_wikisplit_selected"] is False
    assert report["updated_projection"]["source_family_cap_violations"] == []


def test_existing_candidate_salvage_preflight_preserves_review_and_verifies() -> None:
    plan = load_plan(PLAN, ROOT)
    report = build_report(plan, file_sha256(PLAN), ROOT)
    nemotron = report["projected_materialization"]["nemotron_tool_subset"]
    assert nemotron["prior_owner_review_rows_retained"] == 8
    assert nemotron["by_split"]["train"]["supervised_tokens"] <= 770_665
    verification = build_verification(report, "0" * 64)
    assert verification["status"].startswith("PASS")
