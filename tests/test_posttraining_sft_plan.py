from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lumi.posttraining_sft_plan import (
    build_sft_execution_readiness_report,
    load_sft_execution_plan,
)

ROOT = Path(__file__).parents[1]
PLAN_PATH = ROOT / "configs" / "lumi-4b-sft-execution-plan-v0.1.json"


def test_sft_execution_plan_is_bound_and_non_authorizing() -> None:
    plan = load_sft_execution_plan(PLAN_PATH, repository_root=ROOT)
    assert plan["authorization"] == {
        "dataset_promotion_authorized": False,
        "paid_gpu_authorized": False,
        "production_sft_authorized": False,
        "public_release_authorized": False,
    }
    assert plan["training"]["loss"]["ignore_index"] == -100
    assert plan["training"]["loss"]["supervised_roles"] == ["assistant", "tool"]
    assert plan["dataset"]["ai_assisted_candidate_records"] == 320
    assert plan["dataset"]["ai_assisted_owner_spot_checked_records"] == 0
    assert plan["dataset"]["representative_external_source_audit_complete"] is True
    assert plan["dataset"]["representative_external_source_promotions"] == 0
    assert plan["dataset"]["sealed_dataset_manifest_sha256"] is None


def test_sft_execution_plan_rejects_binding_or_authorization_drift(tmp_path: Path) -> None:
    plan = json.loads(PLAN_PATH.read_text())
    changed = copy.deepcopy(plan)
    changed["authorization"]["paid_gpu_authorized"] = True
    changed_path = tmp_path / "changed.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="authorization boundary"):
        load_sft_execution_plan(changed_path, repository_root=ROOT)

    changed = copy.deepcopy(plan)
    changed["bindings"]["chat_template"]["sha256"] = "0" * 64
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="binding changed"):
        load_sft_execution_plan(changed_path, repository_root=ROOT)


def test_sft_readiness_report_names_every_current_blocker_without_raw_text() -> None:
    plan = load_sft_execution_plan(PLAN_PATH, repository_root=ROOT)
    report = build_sft_execution_readiness_report(plan, repository_root=ROOT)
    assert report["base"]["training_tokens"] == 80_000_000_000
    assert report["chat_and_loss_contract"]["ready"] is True
    assert report["first_party_authoring"] == {
        "completed_records": 0,
        "planned_records": 320,
        "template_ready": True,
    }
    assert report["first_party_authoring_workflow"] == {
        "author_slots": 4,
        "human_identity_assignment_complete": False,
        "packet_count": 16,
        "reviewer_slots": 4,
        "rows_per_packet": 20,
        "ready": True,
    }
    assert report["ai_assisted_pilot"] == {
        "candidate_records": 320,
        "machine_validated_records": 320,
        "owner_sample_records": 32,
        "owner_spot_check_complete": False,
        "planned_records": 320,
        "production_eligible": False,
        "source_revision": (
            "d492d2fa0817d08b55941b0393cb3d95ae5572a0bcc3c6de39f902bd8491a9b0"
        ),
    }
    assert report["production_eligible_source_count"] == 0
    assert report["source_quality_audit"] == {
        "complete": True,
        "outcome": "complete_failed_closed",
        "production_promotions": 0,
        "report_sha256": (
            "0c8644ae02d6f661e3a0b43d4de07f4ad5029b0b76b2a27bb576743d2c7a37bd"
        ),
        "sparse_tasks": ["coding", "factual_qa", "writing_summarization"],
        "total_blocker_findings": 16,
        "zero_coverage_tasks": ["safety_helpfulness"],
    }
    assert report["ready_for_gpu"] is False
    assert report["paid_gpu_authorized"] is False
    assert report["cpu_canary"] == {
        "production_seal_bound": False,
        "synthetic_assistant_only_loss": True,
        "synthetic_fresh_process_resume": True,
    }
    assert report["language_routing_gate"] == {
        "core_modes": [
            "english",
            "bengali_script",
            "banglish",
            "code_switching",
        ],
        "metadata_cases": 216,
        "model_inference_executed": False,
        "ready_for_future_execution": True,
    }
    assert "dataset:production_seal_missing" in report["blockers"]
    assert (
        "dataset:representative_external_source_quality_and_provenance_audits_incomplete"
        not in report["blockers"]
    )
    assert (
        "sources:representative_external_audit_failed_closed_no_promotions"
        in report["blockers"]
    )
    assert (
        "ai_assisted_records:dataset_level_owner_spot_check_incomplete"
        in report["blockers"]
    )
    assert "first_party_records:human_authorship_incomplete" not in report["blockers"]
    assert "trainer:distributed_assistant_only_label_path_not_implemented" in report[
        "blockers"
    ]
    assert (
        "trainer:production_seal_bound_cpu_loss_and_resume_canary_not_run"
        in report["blockers"]
    )
    assert "evaluation:language_routing_not_run" in report["blockers"]
    assert "user_message" not in json.dumps(report)
    assert len(report["report_sha256"]) == 64
