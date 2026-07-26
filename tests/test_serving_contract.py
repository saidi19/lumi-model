from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lumi.serving_contract import (
    FINAL_PRETRAINING_STEP,
    FINAL_PRETRAINING_TOKENS,
    load_serving_contract,
    serving_readiness,
    validate_serving_contract,
)

ROOT = Path(__file__).parents[1]
CONTRACT_PATH = ROOT / "configs" / "lumi-4b-serving-contract-v0.1.json"


def test_skeleton_is_valid_but_deliberately_blocked() -> None:
    contract = load_serving_contract(CONTRACT_PATH)
    report = serving_readiness(contract)
    assert contract["base_checkpoint"]["required_step"] == FINAL_PRETRAINING_STEP
    assert (
        contract["base_checkpoint"]["required_training_tokens"]
        == FINAL_PRETRAINING_TOKENS
    )
    assert report["ready"] is False
    assert "gate:base_pretraining_80b" in report["blockers"]
    assert "artifact:base_checkpoint_manifest_sha256" in report["blockers"]


def test_contract_becomes_ready_only_with_all_hashes_and_gates() -> None:
    contract = json.loads(CONTRACT_PATH.read_text())
    for gate in contract["gates"].values():
        gate["status"] = "passed"
        gate["evidence"] = ["evidence.json"]
    for index, name in enumerate(contract["artifacts"]):
        contract["artifacts"][name] = f"{index:x}".rjust(64, "0")
    report = serving_readiness(contract)
    assert report["ready"] is True
    assert report["blockers"] == []


def test_contract_rejects_boundary_or_public_enable_changes() -> None:
    contract = json.loads(CONTRACT_PATH.read_text())
    changed = copy.deepcopy(contract)
    changed["base_checkpoint"]["required_training_tokens"] -= 1
    with pytest.raises(ValueError, match="token boundary"):
        validate_serving_contract(changed)

    changed = copy.deepcopy(contract)
    changed["deployment"]["public_enabled"] = True
    report = serving_readiness(changed)
    assert report["ready"] is False
    assert "deployment:premature_public_enable" in report["blockers"]

    changed = copy.deepcopy(contract)
    changed["deployment"]["public_enabled"] = "yes"
    with pytest.raises(ValueError, match="boolean"):
        validate_serving_contract(changed)
