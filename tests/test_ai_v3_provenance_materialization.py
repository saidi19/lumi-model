from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lumi.posttraining_ai_v3_provenance_materialization import (
    build_ai_v3_provenance_outputs,
    build_ai_v3_provenance_verification,
    load_ai_v3_provenance_plan,
    validate_ai_v3_provenance_report,
)
from lumi.posttraining_candidate_mix_preflight import file_sha256

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-ai-v3-provenance-materialization-v0.1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_ai_v3_provenance_plan_is_bounded_and_nonproduction() -> None:
    plan = load_ai_v3_provenance_plan(PLAN, repository_root=ROOT)
    assert plan["limits"]["candidate_records"] == 320
    assert plan["limits"]["owner_spot_check_records"] == 32
    assert plan["limits"]["maximum_network_bytes"] == 0
    assert plan["authorization"]["candidate_metadata_repair_authorized"] is True
    assert plan["authorization"]["candidate_message_modification_authorized"] is False
    assert plan["authorization"]["production_sft_authorized"] is False


def test_ai_v3_provenance_outputs_reproduce_exactly() -> None:
    plan = load_ai_v3_provenance_plan(PLAN, repository_root=ROOT)
    expected_candidate, expected_manifest, expected_report = (
        build_ai_v3_provenance_outputs(
            plan=plan,
            plan_file_sha256=file_sha256(PLAN),
            repository_root=ROOT,
        )
    )
    candidate_path = (ROOT / plan["output"]["candidate_path"]).resolve()
    manifest_path = (ROOT / plan["output"]["manifest_path"]).resolve()
    report_path = (ROOT / plan["output"]["report_path"]).resolve()
    assert candidate_path.read_bytes() == expected_candidate
    assert _load(manifest_path) == expected_manifest
    assert _load(report_path) == expected_report
    validate_ai_v3_provenance_report(expected_report)


def test_ai_v3_provenance_is_truthful_and_content_preserving() -> None:
    plan = load_ai_v3_provenance_plan(PLAN, repository_root=ROOT)
    source_path = ROOT / plan["bindings"]["source_candidate"]["path"]
    output_path = (ROOT / plan["output"]["candidate_path"]).resolve()
    source_rows = [json.loads(line) for line in source_path.read_text().splitlines()]
    output_rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(source_rows) == len(output_rows) == 320
    sampled = 0
    for before, after in zip(source_rows, output_rows, strict=True):
        assert before["record_id"] == after["record_id"]
        assert before["messages"] == after["messages"]
        assert before["source"] == after["source"]
        assert before["split"] == after["split"]
        assert before["quality"] == after["quality"]
        assert before["safety"] == after["safety"]
        assert before["provenance"]["content_sha256"] == (
            after["provenance"]["content_sha256"]
        )
        assert after["provenance"]["human_review_status"] == (
            "owner_spot_check_passed"
        )
        assert after["provenance"]["owner_review_sample_records"] == 32
        assert after["provenance"]["owner_review_scope"] == (
            "dataset_level_32_row_spot_check_not_individual_320_row_review"
        )
        sampled += int(after["provenance"]["owner_review_sampled_record"])
    assert sampled == 32


def test_ai_v3_provenance_verification_and_raw_text_free_evidence() -> None:
    plan = load_ai_v3_provenance_plan(PLAN, repository_root=ROOT)
    report_path = ROOT / plan["output"]["report_path"]
    verification_path = ROOT / plan["output"]["verification_path"]
    report = _load(report_path)
    expected = build_ai_v3_provenance_verification(
        plan=plan,
        report=report,
        report_file_sha256=file_sha256(report_path),
    )
    assert _load(verification_path) == expected
    for path in (report_path, verification_path):
        serialized = path.read_text(encoding="utf-8")
        for forbidden in (
            '"messages"',
            '"content"',
            '"assistant_response"',
            '"user_prompt"',
        ):
            assert forbidden not in serialized


def test_ai_v3_provenance_plan_rejects_scope_drift(tmp_path: Path) -> None:
    changed = copy.deepcopy(_load(PLAN))
    changed["authorization"]["candidate_message_modification_authorized"] = True
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="authorization"):
        load_ai_v3_provenance_plan(path, repository_root=ROOT)
