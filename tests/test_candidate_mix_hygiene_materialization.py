from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_candidate_mix_hygiene_materialization import (
    EXPECTED_LIMITS,
    build_hygiene_materialization_verification,
    load_hygiene_materialization_plan,
    validate_hygiene_materialization_receipt,
)
from lumi.posttraining_candidate_mix_preflight import file_sha256

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-candidate-mix-hygiene-materialization-v0.1.json"
RECEIPT = (
    ROOT / "evidence/lumi-4b-sft-candidate-mix-hygiene-materialization-v0.1.json"
)
VERIFICATION = (
    ROOT
    / "evidence/lumi-4b-sft-candidate-mix-hygiene-materialization-verification-v0.1.json"
)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _load_plan() -> dict:
    if not PLAN.is_file():
        pytest.skip("candidate hygiene materialization plan is not materialized")
    return load_hygiene_materialization_plan(PLAN, repository_root=ROOT)


def test_materialization_plan_is_bounded_cpu_only_and_nonproduction() -> None:
    plan = _load_plan()
    assert plan["limits"] == EXPECTED_LIMITS
    assert plan["authorization"]["candidate_materialization_authorized"] is True
    assert plan["authorization"]["network_access_authorized"] is False
    assert plan["authorization"]["paid_gpu_authorized"] is False
    assert plan["authorization"]["candidate_registry_change_authorized"] is False
    assert plan["authorization"]["production_sft_authorized"] is False


def test_materialization_receipt_is_raw_text_free_when_present() -> None:
    _load_plan()
    if not RECEIPT.is_file():
        pytest.skip("candidate hygiene materialization receipt is not materialized")
    receipt = _load(RECEIPT)
    validate_hygiene_materialization_receipt(receipt)
    assert receipt["drop_ledger"]["total_records"] == 23
    serialized = RECEIPT.read_text(encoding="utf-8")
    for forbidden in ('"messages"', '"content"', '"assistant_response"'):
        assert forbidden not in serialized


def test_materialization_verification_replays_when_present() -> None:
    plan = _load_plan()
    if not RECEIPT.is_file() or not VERIFICATION.is_file():
        pytest.skip("candidate hygiene materialization verification is not materialized")
    receipt = _load(RECEIPT)
    expected = build_hygiene_materialization_verification(
        plan=plan,
        receipt=receipt,
        receipt_file_sha256=file_sha256(RECEIPT),
        repository_root=ROOT,
    )
    assert expected == _load(VERIFICATION)

