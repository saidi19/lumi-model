from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from lumi.posttraining_format import encode_sft_messages, load_chat_template
from lumi.posttraining_gpt_oss_teacher_source_promotion_preflight_v16 import (
    EXPECTED_CANONICAL_FILE_SHA256,
    EXPECTED_PROJECTED_GAP,
    EXPECTED_SUCCESSOR_REGISTRY_SHA256,
    EXPECTED_TOKEN_METRICS,
    PureByteLevelBPETokenizer,
    build_preflight_report,
    build_reference_projection,
    load_preflight_plan,
)
from lumi.posttraining_candidate_mix_preflight import file_sha256

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-gpt-oss-teacher-source-promotion-preflight-v0.16.json"


def test_dependency_free_tokenizer_matches_prior_frozen_reference() -> None:
    tokenizer = PureByteLevelBPETokenizer(
        (
            ROOT.parent
            / "lumi-artifacts/4b-segment2-throughput-gate-v0.1/8bedfbc868e8/"
            "staging/input/tokenizer/tokenizer.json"
        ).resolve()
    )
    template = load_chat_template(ROOT / "configs/lumi-4b-chat-template-v0.1.json")
    rows = [
        json.loads(line)
        for line in (
            ROOT.parent
            / "lumi-artifacts/4b-sft-first-party-teacher-canary-v0.1/"
            "e80b7aa7936b/attempt-018/private/candidate-rows.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    metrics: dict[str, dict[str, int]] = defaultdict(
        lambda: {"masked": 0, "records": 0, "serialized": 0, "supervised": 0}
    )
    for row in rows:
        encoded = encode_sft_messages(row["messages"], tokenizer=tokenizer, template=template)
        split = row["split"]
        metrics[split]["masked"] += encoded.masked_tokens
        metrics[split]["records"] += 1
        metrics[split]["serialized"] += len(encoded.input_ids)
        metrics[split]["supervised"] += encoded.supervised_tokens
    assert metrics == {
        "train": {"masked": 1610, "records": 60, "serialized": 3612, "supervised": 2002},
        "validation": {"masked": 605, "records": 20, "serialized": 1404, "supervised": 799},
    }


def test_reference_projection_is_exact_and_non_materializing() -> None:
    projection = build_reference_projection(ROOT)["projection"]
    assert projection["canonical_file_sha256"] == EXPECTED_CANONICAL_FILE_SHA256
    assert projection["successor_registry_sha256"] == EXPECTED_SUCCESSOR_REGISTRY_SHA256
    assert projection["token_metrics"] == EXPECTED_TOKEN_METRICS
    assert projection["projected_gap_to_10m"] == EXPECTED_PROJECTED_GAP
    assert projection["filter_accepted"] == 160
    assert projection["filter_rejected"] == 0


def test_plan_and_report_remain_non_authorizing() -> None:
    plan = load_preflight_plan(PLAN, repository_root=ROOT)
    assert all(
        value is False
        for key, value in plan["authorization"].items()
        if key.endswith("_authorized")
    )
    report = build_preflight_report(
        plan, plan_file_sha256=file_sha256(PLAN), repository_root=ROOT
    )
    assert all(report["checks"].values())
    assert not any(report["production"].values())
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    assert '"messages"' not in serialized
    assert '"content"' not in serialized
