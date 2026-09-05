from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROPOSAL = ROOT / "configs" / "lumi-4b-sft-missing-mix-proposal-v0.1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_weighted_token_targets(
    targets: dict[str, dict[str, float | int]],
    *,
    minimum: int,
    maximum: int,
) -> None:
    assert sum(float(target["fraction"]) for target in targets.values()) == 1.0
    assert sum(int(target["minimum_tokens"]) for target in targets.values()) == (
        minimum
    )
    assert sum(int(target["maximum_tokens"]) for target in targets.values()) == (
        maximum
    )
    for target in targets.values():
        assert int(target["minimum_tokens"]) == round(
            minimum * float(target["fraction"])
        )
        assert int(target["maximum_tokens"]) == round(
            maximum * float(target["fraction"])
        )


def test_missing_mix_proposal_is_fail_closed_and_hash_bound() -> None:
    proposal = json.loads(PROPOSAL.read_text(encoding="utf-8"))
    assert proposal["schema_version"] == "lumi.sft-missing-mix-proposal.v0.1"
    assert proposal["status"] == (
        "planning_only_no_data_acquisition_or_training_authorized"
    )
    assert all(
        value is False
        for value in proposal["authorization"].values()
        if isinstance(value, bool)
    )
    for binding in proposal["bindings"].values():
        assert _sha256(ROOT / binding["path"]) == binding["file_sha256"]


def test_learning_candidate_targets_are_exact_and_complete() -> None:
    proposal = json.loads(PROPOSAL.read_text(encoding="utf-8"))
    candidate = proposal["learning_candidate"]
    bounds = candidate["accepted_train_supervised_tokens"]
    assert bounds == {"minimum": 10_000_000, "maximum": 30_000_000}
    _assert_weighted_token_targets(
        candidate["task_targets"],
        minimum=bounds["minimum"],
        maximum=bounds["maximum"],
    )
    _assert_weighted_token_targets(
        candidate["language_targets"],
        minimum=bounds["minimum"],
        maximum=bounds["maximum"],
    )
    assert candidate["measurement"]["assistant_and_tool_tokens_only"] is True
    assert candidate["measurement"]["validation_tokens_excluded_from_training_target"]


def test_all_source_lanes_remain_proposals_and_teacher_apis_are_disabled() -> None:
    proposal = json.loads(PROPOSAL.read_text(encoding="utf-8"))
    assert proposal["source_lanes"]
    for lane in proposal["source_lanes"]:
        assert lane["state"] == "proposal_only"
        assert lane["production_eligible"] is False
        assert lane["required_evidence"]
        assert lane["task_categories"]
    assert proposal["diversity_policy"]["maximum_fraction_from_one_source"] == 0.35
    assert proposal["diversity_policy"]["maximum_fraction_override_authorized"] is (
        False
    )
    assert all(
        status.startswith("disabled")
        for status in proposal["external_teacher_policy"].values()
    )
