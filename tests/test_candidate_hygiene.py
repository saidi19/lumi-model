from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lumi.posttraining_candidate_hygiene import (
    build_candidate_hygiene_verification,
    file_sha256,
    json_file_bytes,
    load_candidate_hygiene_plan,
    rebuild_candidate_hygiene_outputs,
    resolve_output,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-candidate-hygiene-repair-v0.1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_hygiene_outputs_reproduce_exactly() -> None:
    plan = load_candidate_hygiene_plan(PLAN, repository_root=ROOT)
    outputs = rebuild_candidate_hygiene_outputs(
        plan=plan,
        plan_file_sha256=file_sha256(PLAN),
        repository_root=ROOT,
    )
    expected = {
        "artifact_manifest_path": outputs["artifact_manifest_bytes"],
        "private_safety_candidate_path": outputs["private_safety_candidate_bytes"],
        "successor_registry_path": outputs["successor_registry_bytes"],
        "report_path": json_file_bytes(outputs["report"]),
    }
    for name, payload in expected.items():
        path = resolve_output(ROOT, plan["output"][name], name=name)
        assert path.read_bytes() == payload


def test_exact_aya_and_safety_deltas() -> None:
    plan = load_candidate_hygiene_plan(PLAN, repository_root=ROOT)
    report_path = resolve_output(ROOT, plan["output"]["report_path"], name="report")
    report = _load(report_path)
    assert report["aya_registry_delta"]["predecessor_splits"] == ["train"]
    assert report["aya_registry_delta"]["successor_splits"] == ["train", "validation"]
    assert report["aya_registry_delta"]["validation_records_bound"] == 148
    safety = report["safety_rematerialization"]
    assert safety["records"] == 320
    assert safety["rows_with_top_level_split"] == 320
    assert safety["rows_with_owner_review_receipt_binding"] == 320
    assert safety["by_split"] == {
        "train": {"records": 240, "supervised_tokens": 18_636},
        "validation": {"records": 80, "supervised_tokens": 6_388},
    }


def test_repo_receipts_are_raw_text_free_and_verification_reproduces() -> None:
    plan = load_candidate_hygiene_plan(PLAN, repository_root=ROOT)
    report_path = resolve_output(ROOT, plan["output"]["report_path"], name="report")
    verification_path = resolve_output(
        ROOT,
        plan["output"]["verification_path"],
        name="verification",
    )
    report = _load(report_path)
    expected = build_candidate_hygiene_verification(
        plan=plan,
        report=report,
        report_file_sha256=file_sha256(report_path),
    )
    assert _load(verification_path) == expected
    for path in (report_path, verification_path):
        serialized = path.read_text(encoding="utf-8")
        for forbidden in ('"messages"', '"content"', '"assistant_response"'):
            assert forbidden not in serialized


def test_plan_rejects_scope_expansion(tmp_path: Path) -> None:
    changed = copy.deepcopy(_load(PLAN))
    changed["authorization"]["production_sft_authorized"] = True
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="authorization"):
        load_candidate_hygiene_plan(path, repository_root=ROOT)
