from __future__ import annotations

import copy
from pathlib import Path

import pytest

from lumi.posttraining_production_source_policy_preflight_v1 import (
    SUCCESSOR_AUTHORIZATION_TEXT,
    _assert_raw_text_free,
    build_report,
    build_verification,
    file_sha256,
    load_plan,
    validate_report,
    validate_verification,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-production-source-policy-preflight-v0.1.json"


def _report() -> dict:
    plan = load_plan(PLAN, repository_root=ROOT)
    return build_report(
        plan=plan,
        plan_file_sha256=file_sha256(PLAN),
        repository_root=ROOT,
    )


def test_metadata_only_preflight_reproduces_exact_blockers() -> None:
    report = _report()
    assert report["status"] == "PASS_policy_preflight_decision_required_nonproduction"
    assert report["production_eligible"] is False
    assert report["audited_mix"]["records"] == 329_200
    assert report["audited_mix"]["strict_gap"] == 0
    assert report["registry_inventory"]["production_eligible_registry_source_ids"] == []
    assert report["unresolved_release_evidence"]["pending_owner_review_provenance"][
        "records"
    ] == 20_471
    assert report["unresolved_release_evidence"]["machine_assisted_zero_human_review"][
        "model_assisted_rows"
    ] == 14_420
    assert report["successor"]["exact_authorization_text"] == SUCCESSOR_AUTHORIZATION_TEXT


def test_report_rejects_production_activation() -> None:
    report = _report()
    changed = copy.deepcopy(report)
    changed["decision"]["production_contract_activation_permitted_now"] = True
    changed.pop("report_sha256")
    from lumi.posttraining_production_source_policy_preflight_v1 import sha256_value

    changed["report_sha256"] = sha256_value(changed)
    with pytest.raises(ValueError, match="boundary"):
        validate_report(changed)


def test_report_rejects_pending_owner_count_change() -> None:
    report = _report()
    changed = copy.deepcopy(report)
    changed["unresolved_release_evidence"]["pending_owner_review_provenance"][
        "records"
    ] -= 1
    changed.pop("report_sha256")
    from lumi.posttraining_production_source_policy_preflight_v1 import sha256_value

    changed["report_sha256"] = sha256_value(changed)
    with pytest.raises(ValueError, match="boundary"):
        validate_report(changed)


def test_verification_requires_every_check_true() -> None:
    report = _report()
    verification = build_verification(report=report, report_file_sha256="a" * 64)
    changed = copy.deepcopy(verification)
    changed["checks"]["no_registry_or_active_data_contract_mutation"] = False
    changed.pop("verification_sha256")
    from lumi.posttraining_production_source_policy_preflight_v1 import sha256_value

    changed["verification_sha256"] = sha256_value(changed)
    with pytest.raises(ValueError, match="boundary"):
        validate_verification(changed)


def test_raw_candidate_text_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="raw candidate text"):
        _assert_raw_text_free({"nested": {"messages": []}})


def test_resource_boundary_is_zero_row_and_zero_mutation() -> None:
    report = _report()
    assert report["resource_accounting"] == {
        "candidate_row_reads": 0,
        "factual_passes": 1,
        "gpu_use": 0,
        "inference_calls": 0,
        "network_bytes": 0,
        "registry_or_data_contract_writes": 0,
        "source_row_reads": 0,
        "test_object_reads": 0,
        "training_actions": 0,
    }
