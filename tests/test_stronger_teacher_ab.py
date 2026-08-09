from __future__ import annotations

import http.client
from collections import Counter
from pathlib import Path

from lumi.posttraining_stronger_teacher_ab import (
    CONTINUATION_MAXIMUM_ATTEMPTS,
    CONTINUATION_MAXIMUM_RETRIES,
    EXPECTED_FINAL_OUTPUTS,
    EXPECTED_GATES,
    EXPECTED_LIMITS,
    _continuation_reopens_local_transport_failure,
    _is_runtime_retryable,
    artifact_root,
    load_continuation_authorization,
    load_plan,
)
from lumi.posttraining_stronger_teacher_ab_preflight import selected_schedule

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-stronger-teacher-ab-v0.1.json"
CONTINUATION_AUTHORIZATION = (
    ROOT.parent
    / "lumi-artifacts"
    / "4b-sft-stronger-teacher-ab-v0.1"
    / "c52325bacdb8"
    / "attempt-001"
    / "continuation-authorization.json"
)


def test_execution_plan_preserves_exact_authorized_bounds() -> None:
    plan = load_plan(PLAN, repository_root=ROOT)
    assert plan["limits"] == EXPECTED_LIMITS
    assert plan["gates"] == EXPECTED_GATES
    assert EXPECTED_FINAL_OUTPUTS == 200
    assert artifact_root(plan, ROOT).name == "attempt-001"


def test_paired_schedule_is_exact_and_balanced() -> None:
    schedule = selected_schedule()
    assert len(schedule) == 100
    assert Counter(slot.split for slot in schedule) == Counter(
        {"train": 80, "validation": 20}
    )
    cells = Counter((slot.task, slot.language_mode) for slot in schedule)
    assert set(cells.values()) == {5}


def test_prohibited_actions_remain_false() -> None:
    plan = load_plan(PLAN, repository_root=ROOT)
    authorization = plan["authorization"]
    assert authorization["external_api_authorized"] is False
    assert authorization["cloud_or_paid_gpu_authorized"] is False
    assert authorization["package_installation_authorized"] is False
    assert authorization["candidate_registry_change_authorized"] is False
    assert authorization["dataset_seal_authorized"] is False
    assert authorization["sft_training_authorized"] is False


def test_continuation_is_exactly_bound_to_the_sealed_boundary() -> None:
    plan = load_plan(PLAN, repository_root=ROOT)
    artifact = artifact_root(plan, ROOT)
    authorization = load_continuation_authorization(
        CONTINUATION_AUTHORIZATION,
        artifact=artifact,
        repository_root=ROOT,
    )
    assert authorization["limits"]["cumulative_maximum_inference_attempts"] == 260
    assert authorization["limits"]["cumulative_maximum_runtime_retries"] == 60
    assert CONTINUATION_MAXIMUM_ATTEMPTS == 260
    assert CONTINUATION_MAXIMUM_RETRIES == 60


def test_only_no_response_local_transport_failures_are_reopened() -> None:
    disconnected = http.client.RemoteDisconnected("no response")
    assert _is_runtime_retryable(response=None, error=disconnected) is True
    assert _is_runtime_retryable(response=None, error=ValueError("quality")) is False
    event = {
        "raw_response_text": "",
        "rejection_reason": "RemoteDisconnected:no response",
        "runtime_complete": False,
    }
    assert _continuation_reopens_local_transport_failure(event) is True
    assert _continuation_reopens_local_transport_failure(
        {**event, "raw_response_text": "partial"}
    ) is False
