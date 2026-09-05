from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_candidate_mix_hygiene_materialization_v2 import (
    EXPECTED_LIMITS,
    EXPECTED_PROJECTED,
    load_hygiene_materialization_v2_plan,
    project_hygiene_candidates_v2,
    resolve_materialization_output,
    validate_hygiene_materialization_v2_manifest,
    validate_hygiene_materialization_v2_receipt,
)
from lumi.posttraining_candidate_mix_preflight import _load_json, file_sha256

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-candidate-mix-hygiene-materialization-v0.2.json"


def _load_plan() -> dict:
    if not PLAN.is_file():
        pytest.skip("candidate hygiene materialization v0.2 plan is absent")
    return load_hygiene_materialization_v2_plan(PLAN, repository_root=ROOT)


def test_materialization_v2_plan_is_exact_cpu_only_and_nonproduction() -> None:
    plan = _load_plan()
    assert plan["limits"] == EXPECTED_LIMITS
    assert plan["authorization"]["candidate_materialization_authorized"] is True
    assert plan["authorization"]["network_access_authorized"] is False
    assert plan["authorization"]["paid_gpu_authorized"] is False
    assert plan["authorization"]["production_sft_authorized"] is False


def test_materialization_v2_projection_matches_preflight() -> None:
    plan = _load_plan()
    output_bytes, manifest = project_hygiene_candidates_v2(plan=plan, repository_root=ROOT)
    validate_hygiene_materialization_v2_manifest(manifest)
    for source_id, expected in EXPECTED_PROJECTED.items():
        assert len(output_bytes[source_id]) == expected["bytes"]
        assert manifest["outputs"][source_id]["sha256"] == expected["sha256"]


def test_materialization_v2_outputs_verify_when_present() -> None:
    plan = _load_plan()
    outputs = {
        name: resolve_materialization_output(ROOT, relative, name=name)
        for name, relative in plan["output"].items()
        if name.endswith("_path")
    }
    if not all(path.is_file() for path in outputs.values()):
        pytest.skip("candidate hygiene materialization v0.2 outputs are incomplete")
    manifest = _load_json(outputs["manifest_path"])
    receipt = _load_json(outputs["receipt_path"])
    verification = _load_json(outputs["verification_path"])
    validate_hygiene_materialization_v2_manifest(manifest)
    validate_hygiene_materialization_v2_receipt(receipt)
    assert verification["manifest_file_sha256"] == file_sha256(outputs["manifest_path"])
    assert verification["receipt_file_sha256"] == file_sha256(outputs["receipt_path"])
    assert verification["production_eligible"] is False
    serialized = json.dumps(
        {"manifest": manifest, "receipt": receipt, "verification": verification},
        ensure_ascii=False,
        sort_keys=True,
    )
    for forbidden in ('"messages"', '"content"', '"assistant_response"'):
        assert forbidden not in serialized
