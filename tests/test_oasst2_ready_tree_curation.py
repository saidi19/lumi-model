from __future__ import annotations

import gzip
import json
from pathlib import Path

from lumi.posttraining_oasst2_curation import (
    build_oasst2_owner_review_html,
    label_measurement,
    load_tree_candidates,
    task_category,
    tree_split,
)


def _labels(*, assistant: bool) -> dict[str, dict[str, float | int]]:
    labels: dict[str, dict[str, float | int]] = {
        name: {"count": 3, "value": 0.0}
        for name in (
            "hate_speech",
            "lang_mismatch",
            "not_appropriate",
            "pii",
            "sexual_content",
            "spam",
        )
    }
    labels["quality"] = {"count": 3, "value": 0.9 if assistant else 0.7}
    if assistant:
        labels["helpfulness"] = {"count": 3, "value": 0.9}
        labels["fails_task"] = {"count": 3, "value": 0.0}
    return labels


def _node(
    *,
    message_id: str,
    parent_id: str | None,
    rank: int | None,
    role: str,
    text: str,
    tree_id: str,
) -> dict[str, object]:
    return {
        "created_date": "2023-01-01T00:00:00Z",
        "deleted": False,
        "detoxify": None,
        "emojis": None,
        "events": None,
        "labels": _labels(assistant=role == "assistant"),
        "lang": "en",
        "message_id": message_id,
        "message_tree_id": None,
        "model_name": None,
        "parent_id": parent_id,
        "rank": rank,
        "replies": [],
        "review_count": 3,
        "review_result": True,
        "role": role,
        "synthetic": False,
        "text": text,
        "tree_state": None,
        "user_id": "private-upstream-id",
    }


def test_task_categories_are_deterministic() -> None:
    assert task_category("Please rewrite this email in a warmer tone") == ("writing_summarization")
    assert task_category("Explain why the sky appears blue") == "factual_qa"
    assert task_category("Hello, can we chat about my day?") == "conversation"
    assert task_category("Write a Python function") == "coding"


def test_label_measurement_is_fail_closed() -> None:
    node = {"labels": {"quality": {"count": 3, "value": 0.8}}}
    assert label_measurement(node, "quality") == (0.8, 3)
    assert label_measurement({"labels": {}}, "quality") is None
    assert label_measurement({"labels": {"quality": {"count": 0, "value": 0.8}}}, "quality") is None


def test_tree_selection_preserves_path_and_tree_split(tmp_path: Path) -> None:
    tree_id = "tree-test-001"
    prompt = _node(
        message_id="prompt-1",
        parent_id=None,
        rank=None,
        role="prompter",
        text="Explain why rainbows form.",
        tree_id=tree_id,
    )
    answer = _node(
        message_id="answer-1",
        parent_id="prompt-1",
        rank=0,
        role="assistant",
        text="Rainbows form when light is refracted, reflected, and dispersed.",
        tree_id=tree_id,
    )
    prompt["replies"] = [answer]
    wrapper = {
        "message_tree_id": tree_id,
        "origin": None,
        "prompt": prompt,
        "tree_state": "ready_for_export",
    }
    source = tmp_path / "tree.jsonl.gz"
    with gzip.open(source, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(wrapper) + "\n")

    selection = load_tree_candidates(
        source,
        maximum_decoded_bytes=1024 * 1024,
        maximum_messages=2,
        maximum_trees=1,
    )
    assert selection.metrics["input_trees"] == 1
    assert selection.metrics["input_messages"] == 2
    assert len(selection.records) == 1
    record = selection.records[0]
    assert [message["role"] for message in record["messages"]] == [
        "user",
        "assistant",
    ]
    assert record["split"] == tree_split(tree_id)
    attribution = selection.attributions[0]
    assert attribution["message_tree_id"] == tree_id
    assert attribution["message_ids"] == ["prompt-1", "answer-1"]
    assert "private-upstream-id" not in json.dumps(attribution)


def test_owner_review_page_is_bound_to_sample() -> None:
    sample = [
        {
            "language_mode": "english",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
            "record_id": "oasst2-test-row",
            "skill_id": "conversation",
            "split": "validation",
        }
    ]
    html, sample_sha256 = build_oasst2_owner_review_html(sample)
    assert sample_sha256 in html
    assert "lumi-oasst2-owner-decisions-v0.1.json" in html
    assert "OpenAssistant conversations" in html
