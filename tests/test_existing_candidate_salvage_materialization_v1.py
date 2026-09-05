import tempfile
from pathlib import Path

from lumi.posttraining_existing_candidate_salvage_materialization_v1 import (
    build_verification,
    file_sha256,
    load_plan,
    materialize,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-existing-candidate-salvage-materialization-v0.1.json"


def test_salvage_materialization_reproduces_preflight_subsets() -> None:
    plan = load_plan(PLAN, ROOT)
    with tempfile.TemporaryDirectory(prefix="lumi-salvage-test-") as directory:
        report = materialize(
            plan,
            file_sha256(PLAN),
            ROOT,
            Path(directory) / "attempt-001" / "private",
        )
    assert report["output"]["aya_mit_train_supervised_tokens"] == 225_756
    assert report["output"]["nemotron_train_supervised_tokens"] == 770_663
    assert report["owner_review_delta"]["sample_records"] == 6
    assert report["checks"]["predecessor_candidates_mutated"] is False
    verification = build_verification(report, "0" * 64)
    assert verification["status"].startswith("PASS")
