from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lumi.posttraining_canary import (
    adapt_aya_bengali_rows,
    adapt_oasst1_rows,
    adapt_openr1_rows,
    bengali_script_metrics,
    finalize_canary_report,
    load_canary_plan,
)

ROOT = Path(__file__).parents[1]


def test_canary_plan_is_bounded_and_locally_bound() -> None:
    plan = load_canary_plan(
        ROOT / "configs" / "lumi-4b-sft-source-canary-v0.1.json",
        repository_root=ROOT,
    )
    assert plan["authorization"]["paid_gpu_authorized"] is False
    assert plan["authorization"]["production_sft_authorized"] is False
    assert plan["limits"]["maximum_accepted_records_per_source"] == 24
    assert sum(source["maximum_fetched_rows"] for source in plan["sources"]) == 148


def test_oasst_adapter_keeps_reviewed_top_rank_direct_pairs() -> None:
    rows = [
        {
            "row_idx": 4,
            "row": {
                "deleted": False,
                "lang": "en",
                "message_id": "prompt",
                "parent_id": None,
                "review_result": True,
                "role": "prompter",
                "synthetic": False,
                "text": "Explain recursion.",
            },
        },
        {
            "row_idx": 5,
            "row": {
                "deleted": False,
                "lang": "en",
                "message_id": "answer",
                "parent_id": "prompt",
                "rank": 0,
                "review_result": True,
                "role": "assistant",
                "synthetic": False,
                "text": "Recursion is a function solving a problem using smaller instances.",
            },
        },
        {
            "row_idx": 6,
            "row": {
                "deleted": False,
                "lang": "en",
                "message_id": "lower-ranked",
                "parent_id": "prompt",
                "rank": 1,
                "review_result": True,
                "role": "assistant",
                "synthetic": False,
                "text": "This answer must be skipped.",
            },
        },
        {
            "row_idx": 7,
            "row": {
                "deleted": False,
                "lang": "en",
                "message_id": "unranked",
                "parent_id": "prompt",
                "rank": None,
                "review_result": True,
                "role": "assistant",
                "synthetic": False,
                "text": "This unranked answer must be skipped.",
            },
        },
        {
            "row_idx": 8,
            "row": {
                "deleted": False,
                "lang": "en",
                "message_id": "followup-prompt",
                "parent_id": "answer",
                "review_result": True,
                "role": "prompter",
                "synthetic": False,
                "text": "Give an example.",
            },
        },
        {
            "row_idx": 9,
            "row": {
                "deleted": False,
                "lang": "en",
                "message_id": "followup-answer",
                "parent_id": "followup-prompt",
                "rank": 0,
                "review_result": True,
                "role": "assistant",
                "synthetic": False,
                "text": "This context-free follow-up must be skipped.",
            },
        },
    ]
    records = adapt_oasst1_rows(rows, revision="a" * 40, target_records=24)
    assert len(records) == 1
    assert records[0]["task_category"] == "factual_qa"
    assert records[0]["messages"][-1]["content"].startswith("Recursion")
    assert records[0]["provenance"]["transform_sha256"]


def test_aya_and_openr1_adapters_preserve_source_bindings() -> None:
    aya = adapt_aya_bengali_rows(
        [
            {
                "row_idx": 7,
                "row": {
                    "inputs": "একটি সংক্ষিপ্ত উত্তর দাও।",
                    "language_code": "ben",
                    "targets": "এটি একটি সংক্ষিপ্ত উত্তর।",
                },
            }
        ],
        revision="b" * 40,
        target_records=24,
    )
    openr1 = adapt_openr1_rows(
        [
            {
                "row_idx": 9,
                "row": {
                    "correctness_count": 2,
                    "is_reasoning_complete": [True, True],
                    "problem": "What is 2 + 2?",
                    "solution": "Adding the terms gives 4.",
                    "uuid": "math-row",
                },
            }
        ],
        revision="c" * 40,
        target_records=24,
    )
    assert aya[0]["language"] == "bn"
    assert aya[0]["source"]["source_id"] == "coherelabs_aya_dataset"
    assert openr1[0]["task_category"] == "reasoning_math_science"
    assert openr1[0]["source"]["revision"] == "c" * 40


def test_report_seal_contains_no_raw_text() -> None:
    report = finalize_canary_report(
        {
            "raw_text_retained": False,
            "schema_version": "lumi.sft-source-canary-report.v0.1",
            "summary": {"accepted": 3},
        }
    )
    serialized = json.dumps(report)
    assert "report_sha256" in report
    assert "What is 2 + 2" not in serialized


def test_bengali_script_metrics_do_not_retain_text() -> None:
    records = [
        {
            "language": "bn",
            "messages": [
                {"content": "বাংলায় উত্তর দাও।", "role": "user"},
                {"content": "অবশ্যই।", "role": "assistant"},
            ],
        }
    ]
    metrics = bengali_script_metrics(records)
    assert metrics["records"] == 1
    assert metrics["records_with_bengali"] == 1
    assert metrics["bengali_codepoints"] > 0
    assert 0 < metrics["bengali_alphabetic_fraction"] <= 1
    assert "বাংলায়" not in json.dumps(metrics)


def test_identity_spec_is_first_party_and_not_yet_training_data() -> None:
    value = json.loads(
        (ROOT / "configs" / "lumi-4b-identity-sft-spec-v0.1.json").read_text()
    )
    assert value["claims"]["assistant_name"] == "Lumi"
    assert value["claims"]["developer_legal_name"] == "Afnan Technologies Inc."
    assert value["claims"]["pretraining_tokens"] == 80_000_000_000
    assert value["status"] == "planning_only_not_sft_data"
    assert set(value["canonical_introduction"]) == {"bn", "bn_latn", "en"}
    assert set(value["introduction_variants"]) == {
        "capabilities_and_limits",
        "first_run_warm",
        "short",
        "technical_after_posttraining",
        "warm_collaborative",
    }
    assert "বাংলা, English বা Banglish" in value["introduction_variants"]["first_run_warm"]["bn"]
    assert value["response_policy"]["natural_paraphrase_required"] is True
    assert value["response_policy"]["must_not_claim_to_be_chatgpt_or_made_by_openai"] is True
    assert value["response_policy"]["must_not_claim_to_be_claude_or_made_by_anthropic"] is True
    behavior_path = ROOT / value["response_policy"]["behavior_spec"]
    assert hashlib.sha256(behavior_path.read_bytes()).hexdigest() == value["response_policy"][
        "behavior_spec_sha256"
    ]


def test_assistant_behavior_spec_is_distinct_bounded_and_not_authorized() -> None:
    value = json.loads(
        (ROOT / "configs" / "lumi-4b-assistant-behavior-spec-v0.1.json").read_text()
    )
    assert value["schema_version"] == "lumi.assistant-behavior.v0.1"
    assert value["status"] == "planning_only_not_training_data"
    assert value["identity_binding"]["assistant_name"] == "Lumi"
    assert value["identity_binding"]["developer_legal_name"] == "Afnan Technologies Inc."
    assert value["design_intent"]["must_not_imitate_a_proprietary_assistant_persona"] is True
    assert value["authorization"] == {
        "actual_ai_assisted_candidate_records": 320,
        "actual_training_records": 0,
        "paid_gpu_authorized": False,
        "preference_optimization_authorized": False,
        "production_sft_authorized": False,
    }
    bound_files = (
        (
            value["evaluation_plan"]["basic_capability_gate"],
            value["evaluation_plan"]["basic_capability_gate_sha256"],
        ),
        (
            value["sft_behavior_slice_plan"]["ai_assisted_pilot"],
            value["sft_behavior_slice_plan"]["ai_assisted_pilot_sha256"],
        ),
        (
            value["sft_behavior_slice_plan"]["first_party_authoring_pilot"],
            value["sft_behavior_slice_plan"]["first_party_authoring_pilot_sha256"],
        ),
    )
    for relative_path, expected_sha256 in bound_files:
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == expected_sha256

    slice_plan = value["sft_behavior_slice_plan"]
    assert sum(item["train"] for item in slice_plan["records"].values()) == 6000
    assert sum(item["validation"] for item in slice_plan["records"].values()) == 920
    assert slice_plan["totals"] == {"train": 6000, "validation": 920}

    language_modes = {
        item["mode"]: item["dataset_language"]
        for item in value["conversation_contract"]["language_modes"]
    }
    assert language_modes == {
        "banglish": "multilingual",
        "bengali_script": "bn",
        "code_switching": "multilingual",
        "english": "en",
    }
    zero_tolerance = value["evaluation_plan"]["deterministic_zero_tolerance"]
    assert set(zero_tolerance.values()) == {0}
