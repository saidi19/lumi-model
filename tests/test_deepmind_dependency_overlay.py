from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_deepmind_dependency_overlay import (
    EXPECTED_LIMITS,
    WHEEL_BYTES,
    WHEELS,
    load_overlay_plan,
    validate_overlay_receipt,
    validate_wheelhouse,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs/lumi-4b-sft-deepmind-dependency-overlay-v0.1.json"
RECEIPT_PATH = ROOT / "evidence/lumi-4b-sft-deepmind-dependency-overlay-v0.1.json"


def test_exact_wheel_ceiling_is_bounded() -> None:
    assert sum(wheel["size"] for wheel in WHEELS) == WHEEL_BYTES
    assert WHEEL_BYTES < EXPECTED_LIMITS["maximum_network_bytes"]
    assert [wheel["distribution"] for wheel in WHEELS] == ["absl-py", "six"]


def test_wheelhouse_rejects_unrelated_file(tmp_path: Path) -> None:
    (tmp_path / "unrelated.whl").write_bytes(b"not a wheel")
    with pytest.raises(ValueError, match="file set"):
        validate_wheelhouse(tmp_path)


def test_exact_plan_loads_when_materialized() -> None:
    if not PLAN_PATH.exists():
        pytest.skip("DeepMind overlay plan is not materialized")
    plan = load_overlay_plan(PLAN_PATH, repository_root=ROOT)
    assert plan["authorization"]["dependency_overlay_installation_authorized"] is True
    assert plan["authorization"]["canary_generation_authorized"] is False


def test_receipt_is_sealed_when_materialized() -> None:
    if not RECEIPT_PATH.exists():
        pytest.skip("DeepMind overlay receipt is not materialized")
    receipt = json.loads(RECEIPT_PATH.read_text())
    validate_overlay_receipt(receipt)
    assert receipt["production"]["packages_installed"] == 2
    assert receipt["production"]["canary_records_generated"] == 0
