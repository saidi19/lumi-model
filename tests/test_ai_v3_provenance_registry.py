from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lumi.posttraining_ai_v3_provenance_materialization import json_file_bytes
from lumi.posttraining_ai_v3_provenance_registry import (
    REMAINING_SOURCE_BLOCKERS,
    SOURCE_ID,
    SUCCESSOR_ADAPTER,
    build_ai_v3_provenance_registry_outputs,
    build_ai_v3_provenance_registry_verification,
    load_ai_v3_provenance_registry_plan,
    resolve_output,
)
from lumi.posttraining_candidate_mix_preflight import file_sha256

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-ai-v3-provenance-registry-v0.1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_ai_v3_registry_plan_is_bounded_and_nonproduction() -> None:
    plan = load_ai_v3_provenance_registry_plan(PLAN, repository_root=ROOT)
    assert plan["limits"]["changed_source_entries"] == 1
    assert plan["limits"]["added_sources"] == 0
    assert plan["limits"]["removed_sources"] == 0
    assert plan["authorization"]["candidate_row_modification_authorized"] is False
    assert plan["authorization"]["production_sft_authorized"] is False
    assert plan["authorization"]["paid_gpu_authorized"] is False


def test_ai_v3_registry_outputs_reproduce_exactly() -> None:
    plan = load_ai_v3_provenance_registry_plan(PLAN, repository_root=ROOT)
    expected_registry, expected_report = build_ai_v3_provenance_registry_outputs(
        plan=plan,
        plan_file_sha256=file_sha256(PLAN),
        repository_root=ROOT,
    )
    registry_path = resolve_output(
        ROOT,
        plan["output"]["successor_registry_path"],
        name="successor_registry_path",
    )
    report_path = resolve_output(
        ROOT,
        plan["output"]["report_path"],
        name="report_path",
    )
    assert registry_path.read_bytes() == expected_registry
    assert report_path.read_bytes() == json_file_bytes(expected_report)


def test_ai_v3_registry_delta_is_exact() -> None:
    plan = load_ai_v3_provenance_registry_plan(PLAN, repository_root=ROOT)
    predecessor = _load(ROOT / plan["bindings"]["predecessor_registry"]["path"])
    successor = _load(ROOT / plan["output"]["successor_registry_path"])
    assert len(predecessor["sources"]) == len(successor["sources"]) == 13
    changed = []
    for before, after in zip(predecessor["sources"], successor["sources"], strict=True):
        if before != after:
            changed.append((before, after))
    assert len(changed) == 1
    before, after = changed[0]
    assert before["source_id"] == after["source_id"] == SOURCE_ID
    assert before["revision"] == after["revision"]
    assert after["adapter"] == SUCCESSOR_ADAPTER
    assert after["production_blockers"] == REMAINING_SOURCE_BLOCKERS
    assert {
        key for key in before if before.get(key) != after.get(key)
    } == {"adapter", "production_blockers"}


def test_ai_v3_registry_verification_is_raw_text_free_and_reproducible() -> None:
    plan = load_ai_v3_provenance_registry_plan(PLAN, repository_root=ROOT)
    report_path = ROOT / plan["output"]["report_path"]
    verification_path = ROOT / plan["output"]["verification_path"]
    report = _load(report_path)
    expected = build_ai_v3_provenance_registry_verification(
        plan=plan,
        report=report,
        report_file_sha256=file_sha256(report_path),
    )
    assert _load(verification_path) == expected
    assert report["candidate_validation"]["sampled_records"] == 32
    assert report["candidate_validation"]["unsampled_records"] == 288
    for path in (report_path, verification_path):
        serialized = path.read_text(encoding="utf-8")
        for forbidden in ('"messages"', '"content"', '"assistant_response"'):
            assert forbidden not in serialized


def test_ai_v3_registry_plan_rejects_scope_drift(tmp_path: Path) -> None:
    changed = copy.deepcopy(_load(PLAN))
    changed["authorization"]["production_sft_authorized"] = True
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="authorization"):
        load_ai_v3_provenance_registry_plan(path, repository_root=ROOT)
