from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lumi.posttraining_benchmark_gates import (
    EXPECTED_GATE_IDS,
    build_benchmark_gate_readiness_report,
    load_benchmark_gate_plan,
    verify_benchmark_gate_readiness_report,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "lumi-4b-sft-benchmark-gates-v0.1.json"


def test_benchmark_schedule_is_bound_staged_and_non_authorizing() -> None:
    plan = load_benchmark_gate_plan(PLAN, repository_root=ROOT)
    assert all(
        value is False
        for key, value in plan["authorization"].items()
        if key.endswith("_authorized")
    )
    assert [gate["gate_id"] for gate in plan["gates"]] == EXPECTED_GATE_IDS
    assert plan["gates"][0]["accepted_train_supervised_tokens"] == 0
    assert plan["gates"][2]["accepted_train_supervised_tokens"] == 10_000_000
    assert plan["gates"][3]["accepted_train_supervised_tokens"] == 50_000_000
    assert plan["gates"][4]["accepted_train_supervised_tokens"] == 100_000_000
    assert plan["gates"][5]["accepted_train_supervised_tokens"] == 250_000_000
    assert plan["gates"][6]["accepted_train_supervised_tokens"] == 500_000_000
    assert plan["gates"][1]["kind"] != "full_quality_benchmark"
    assert (
        plan["decision_policy"][
            "five_hundred_million_is_a_quality_ceiling_not_a_mandatory_dose"
        ]
        is True
    )


def test_current_readiness_names_pending_adapters_and_token_gap() -> None:
    plan = load_benchmark_gate_plan(PLAN, repository_root=ROOT)
    report = build_benchmark_gate_readiness_report(plan, repository_root=ROOT)
    verify_benchmark_gate_readiness_report(report)
    assert report["adapter_build_pending"] == [
        "bengali-and-multilingual",
        "code",
        "instruction-following",
        "knowledge-and-reasoning",
        "math",
        "truthfulness-and-calibration",
    ]
    assert report["execution_pending_after_checkpoint_exists"] == [
        "basic-conversation-typo-and-arithmetic",
        "response-language-routing",
    ]
    assert report["base_full_baseline_ready"] is False
    assert report["candidate_pool"] == {
        "accepted_train_supervised_tokens": 1_059_171,
        "gap_to_10m_train_supervised_tokens": 8_940_829,
        "production_eligible": False,
    }
    assert report["runtime_smoke_is_quality_benchmark"] is False
    assert report["five_hundred_million_mandatory"] is False
    assert report["full_gpu_sft_quality_gate_ready"] is False


def test_benchmark_plan_rejects_authorization_or_binding_drift(tmp_path: Path) -> None:
    value = json.loads(PLAN.read_text(encoding="utf-8"))
    changed = copy.deepcopy(value)
    changed["authorization"]["paid_gpu_authorized"] = True
    path = tmp_path / "changed-authorization.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="authorization boundary"):
        load_benchmark_gate_plan(path, repository_root=ROOT)

    changed = copy.deepcopy(value)
    changed["bindings"]["evaluation_plan"]["sha256"] = "0" * 64
    path = tmp_path / "changed-binding.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="binding changed"):
        load_benchmark_gate_plan(path, repository_root=ROOT)
