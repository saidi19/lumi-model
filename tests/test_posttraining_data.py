from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lumi.posttraining_data import (
    SFTPolicy,
    evaluate_sft_record,
    filter_sft_records,
    load_sft_policy,
)

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "lumi-4b-sft-data-contract-v0.1.json"
ZERO = "0" * 64
ONE = "1" * 64


def _record(record_id: str = "record-1") -> dict:
    return {
        "language": "en",
        "messages": [
            {"content": "Explain why the sky looks blue.", "role": "user"},
            {
                "content": "Air scatters shorter blue wavelengths more strongly.",
                "role": "assistant",
            },
        ],
        "provenance": {
            "consent_basis": "document_license",
            "content_sha256": ZERO,
            "transform_sha256": ONE,
        },
        "quality": {
            "pii_scan": "passed",
            "review_status": "accepted",
            "score": 0.95,
            "secret_scan": "passed",
        },
        "record_id": record_id,
        "safety": {"policy_version": "lumi-safety-v0.1", "risk_labels": []},
        "schema_version": "lumi.sft.v0.1",
        "source": {
            "dataset_id": "owned-demo",
            "document_id": f"doc-{record_id}",
            "license_id": "cc-by-4.0",
            "uri": f"https://example.invalid/{record_id}",
        },
        "split": "train",
        "task_category": "factual_qa",
    }


def test_policy_loads_and_task_mix_is_exact() -> None:
    policy = load_sft_policy(CONFIG)
    assert isinstance(policy, SFTPolicy)
    assert sum(policy.task_mix.values()) == pytest.approx(1.0)
    assert "coding" in policy.task_mix


def test_record_passes_fail_closed_schema() -> None:
    decision = evaluate_sft_record(_record(), policy=load_sft_policy(CONFIG))
    assert decision.accepted is True
    assert decision.reasons == ()
    assert len(decision.normalized_sha256) == 64
    assert len(decision.response_sha256 or "") == 64


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda item: item["source"].update(license_id="unknown"), "license"),
        (lambda item: item["quality"].update(score=0.2), "quality_score_below_floor"),
        (
            lambda item: item["messages"][0].update(content="Email me at a@example.com"),
            "pii_pattern",
        ),
        (
            lambda item: item["messages"][1].update(content="api_key=abcdefghijklmno"),
            "secret_pattern",
        ),
        (
            lambda item: item["messages"][1].update(content="<|doc|> leaked marker"),
            "training_marker_literal",
        ),
        (
            lambda item: item["messages"].reverse(),
            "invalid_turn_order",
        ),
    ],
)
def test_record_rejects_unsafe_or_invalid_content(mutate, reason: str) -> None:
    record = _record()
    mutate(record)
    decision = evaluate_sft_record(record, policy=load_sft_policy(CONFIG))
    assert decision.accepted is False
    assert reason in decision.reasons


def test_batch_filter_rejects_normalized_duplicate_without_raw_text() -> None:
    first = _record("one")
    second = copy.deepcopy(first)
    second["record_id"] = "two"
    second["source"]["document_id"] = "doc-two"
    second["source"]["uri"] = "https://example.invalid/two"
    second["messages"][0]["content"] = "  EXPLAIN why the sky looks blue. "
    accepted, report = filter_sft_records(
        [first, second],
        policy=load_sft_policy(CONFIG),
    )
    assert len(accepted) == 1
    assert report["rejection_counts"] == {"duplicate_normalized_conversation": 1}
    assert report["raw_text_retained"] is False
    serialized = json.dumps(report)
    assert "Explain why the sky looks blue" not in serialized


def test_benchmark_match_is_rejected() -> None:
    class AlwaysMatch:
        def match(self, text: str) -> tuple[str, str]:
            assert text
            return ("heldout-demo", "example-1")

    decision = evaluate_sft_record(
        _record(),
        policy=load_sft_policy(CONFIG),
        benchmark_registry=AlwaysMatch(),
    )
    assert decision.accepted is False
    assert "benchmark_contamination:heldout-demo" in decision.reasons
