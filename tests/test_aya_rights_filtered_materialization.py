from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_aya_rights_filtered_materialization import (
    EXPECTED_OUTPUT_CANDIDATE_SHA256,
    EXPECTED_OUTPUT_RECORDS,
    build_materialization,
    canonical_json_bytes,
    load_plan,
    verify_materialized_outputs,
)
from lumi.posttraining_aya_rights_filtered_preflight import file_sha256

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-aya-rights-filtered-materialization-v0.1.json"


def _load_plan() -> dict:
    if not PLAN.is_file():
        pytest.skip("Aya rights-filtered materialization plan is not present")
    return load_plan(PLAN, repository_root=ROOT)


def test_authorization_is_bounded_and_nonproduction() -> None:
    plan = _load_plan()
    assert plan["authorization"]["candidate_materialization_authorized"] is True
    assert plan["authorization"]["network_access_authorized"] is False
    assert plan["authorization"]["gpu_authorized"] is False
    assert plan["authorization"]["registry_change_authorized"] is False
    assert plan["authorization"]["sft_training_authorized"] is False


def test_materialization_projection_is_byte_exact() -> None:
    plan = _load_plan()
    candidate, manifest, report = build_materialization(
        plan,
        plan_file_sha256=file_sha256(PLAN),
        repository_root=ROOT,
    )
    assert len(candidate) == 9_251_318
    assert report["output"]["records"] == EXPECTED_OUTPUT_RECORDS
    assert report["output"]["candidate_file_sha256"] == EXPECTED_OUTPUT_CANDIDATE_SHA256
    assert manifest["checks"]["input_lines_preserved_byte_exact"] is True
    assert manifest["checks"]["quarantined_rows_excluded"] is True


def test_materialized_outputs_reproduce_when_present() -> None:
    plan = _load_plan()
    output = plan["output"]
    candidate = (ROOT / output["candidate_path"]).resolve()
    manifest = (ROOT / output["manifest_path"]).resolve()
    report = (ROOT / output["report_path"]).resolve()
    verification = (ROOT / output["verification_path"]).resolve()
    if not all(path.is_file() for path in (candidate, manifest, report, verification)):
        pytest.skip("Aya rights-filtered materialization outputs are not complete")
    expected = verify_materialized_outputs(
        plan,
        plan_file_sha256=file_sha256(PLAN),
        repository_root=ROOT,
    )
    assert json.loads(verification.read_text(encoding="utf-8")) == expected
    assert verification.read_bytes() == canonical_json_bytes(expected)
