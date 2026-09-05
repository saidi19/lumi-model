from __future__ import annotations

import json
from pathlib import Path

from lumi.posttraining_safety_tool import (
    audit_safety_tool_rows,
    expected_safety_tool_records,
    load_safety_tool_plan,
    sha256_value,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "lumi-4b-safety-tool-sft-pilot-v0.1.json"
SOURCE = (
    ROOT
    / "artifacts"
    / "lumi-safety-tool-sft-pilot-v0.1"
    / "f7dacd1ceb30"
    / "attempt-004"
    / "source-rows.jsonl"
)
EVIDENCE = ROOT / "evidence" / "lumi-4b-safety-tool-sft-pilot-v0.1.json"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_safety_tool_plan_has_exact_membership_and_local_only_boundary() -> None:
    plan = load_safety_tool_plan(PLAN, repository_root=ROOT)
    expected = expected_safety_tool_records(plan)
    assert len(expected) == 320
    assert sum(row["split"] == "train" for row in expected.values()) == 240
    assert sum(row["split"] == "validation" for row in expected.values()) == 80
    assert plan["authorization"]["network_access_authorized"] is False
    assert plan["authorization"]["paid_gpu_authorized"] is False
    assert plan["authorization"]["production_sft_authorized"] is False


def test_safety_tool_candidate_reproduces_machine_gates_without_promotion() -> None:
    plan = load_safety_tool_plan(PLAN, repository_root=ROOT)
    audit = audit_safety_tool_rows(_read_jsonl(SOURCE), plan=plan)
    assert audit["records"] == 320
    assert audit["split_counts"] == {"train": 240, "validation": 80}
    assert audit["quality_gate_results"] == {
        "exact_prompt_overlap": 0,
        "exact_response_overlap": 0,
        "normalized_conversation_unique_ratio": 1.0,
        "prompt_near_duplicate_pairs": 0,
        "response_near_duplicate_pairs": 0,
        "semantic_family_cross_split_overlap": 0,
    }
    assert audit["owner_review"]["complete"] is False
    assert audit["production_eligible"] is False
    assert audit["production_promotions"] == 0


def test_safety_tool_evidence_seal_is_raw_text_free_and_owner_pending() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    seal = evidence.pop("evidence_sha256")
    assert seal == sha256_value(evidence)
    assert evidence["owner_review_complete"] is False
    assert evidence["production_eligible"] is False
    assert evidence["production_promotions"] == 0
    assert evidence["raw_training_text_retained"] is False

