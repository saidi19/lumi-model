from pathlib import Path

from lumi.posttraining_existing_candidate_reconciliation_v1 import (
    build_report,
    build_verification,
    file_sha256,
    load_plan,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-existing-candidate-reconciliation-v0.1.json"


def test_existing_candidate_reconciliation_reuses_completed_work() -> None:
    plan = load_plan(PLAN, ROOT)
    report = build_report(plan, file_sha256(PLAN), ROOT)

    assert report["current_baseline"]["aya_already_included"][
        "train_supervised_tokens"
    ] == 513_379
    assert report["completed_work_ledger"]["aya_collection_rights_filtered"][
        "repeat_materialization_required"
    ] is False
    assert report["completed_work_ledger"]["banglabridge_repaired"][
        "repeat_hygiene_repair_required"
    ] is False
    assert report["completed_work_ledger"]["aya_full_local_capacity_scan"][
        "repeat_capacity_scan_required"
    ] is False


def test_existing_candidate_reconciliation_projection_and_verification() -> None:
    plan = load_plan(PLAN, ROOT)
    report = build_report(plan, file_sha256(PLAN), ROOT)
    scenarios = report["updated_v0_12_projections"]

    assert scenarios["all_existing_local_no_new_inference_policy_conditional"][
        "projected_train_supervised_tokens"
    ] == 9_717_281
    assert scenarios["exclude_unresolved_share_alike_keep_existing_synthetic"][
        "strict_targeted_gap_lower_bound"
    ] == 1_835_341
    assert scenarios[
        "exclude_unresolved_share_alike_and_preexisting_synthetic_teacher"
    ]["strict_targeted_gap_lower_bound"] == 2_606_006

    verification = build_verification(report, "0" * 64)
    assert verification["status"].startswith("PASS")
    assert verification["checks"]["already_included_aya_not_double_counted"]
