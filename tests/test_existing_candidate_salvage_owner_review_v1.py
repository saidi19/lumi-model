from pathlib import Path

from lumi.posttraining_existing_candidate_salvage_owner_review_v1 import (
    build_receipt,
    build_verification,
    file_sha256,
    load_plan,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-existing-candidate-salvage-owner-review-v0.1.json"


def test_owner_accepted_all_six_aya_mit_rows() -> None:
    plan = load_plan(PLAN, ROOT)
    receipt = build_receipt(plan, file_sha256(PLAN), ROOT)
    assert receipt["accepted_records"] == 6
    assert receipt["rejected_records"] == 0
    assert receipt["all_sampled_rows_accepted"] is True
    assert receipt["raw_training_text_retained"] is False
    verification = build_verification(receipt, "0" * 64)
    assert verification["status"].startswith("PASS")
