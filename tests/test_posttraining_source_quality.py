from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_source_quality import (
    audit_aya,
    audit_oasst1,
    audit_openr1,
    build_candidate_coverage_gap,
    load_source_quality_audit_evidence,
    load_source_quality_audit_plan,
    sha256_value,
)

ROOT = Path(__file__).parents[1]


def _record(source_id: str, document_id: str, task: str = "conversation") -> dict:
    return {
        "record_id": f"{source_id}-{document_id}",
        "source": {"document_id": document_id, "source_id": source_id},
        "task_category": task,
    }


def test_source_quality_plan_is_non_authorizing_and_exact() -> None:
    plan = load_source_quality_audit_plan(
        ROOT / "configs" / "lumi-4b-sft-source-quality-audit-v0.1.json"
    )
    assert plan["authorization"]["bulk_data_download_authorized"] is False
    assert plan["authorization"]["paid_gpu_authorized"] is False
    assert plan["output"]["production_promotion_permitted"] is False
    assert plan["inputs"]["source_canary"]["accepted_records"] == 72
    assert plan["inputs"]["ai_assisted_candidate"]["file_sha256"] == (
        "8454fc0ca93be61eade069c123562357ffecf69358ba003315abe7dc11bc3845"
    )


def test_oasst_audit_detects_lost_history_and_unranked_reply() -> None:
    accepted = [_record("openassistant_oasst1", "answer")]
    raw = {
        "rows": [
            {
                "row_idx": 1,
                "row": {
                    "message_id": "prompt",
                    "message_tree_id": "tree",
                    "parent_id": "prior-answer",
                    "role": "prompter",
                },
            },
            {
                "row_idx": 2,
                "row": {
                    "message_id": "answer",
                    "message_tree_id": "tree",
                    "parent_id": "prompt",
                    "rank": None,
                    "role": "assistant",
                },
            },
        ]
    }
    result = audit_oasst1(accepted, raw)
    assert result["followup_pairs_missing_ancestor_history"] == 1
    assert result["rank_counts"] == {"null": 1}
    assert result["decision"] == "blocked_after_representative_audit"


def test_aya_audit_detects_concentration_and_format_artifact() -> None:
    accepted = [_record("coherelabs_aya_dataset", f"train-row-{index}") for index in range(4)]
    raw = {
        "rows": [
            {
                "row_idx": index,
                "row": {
                    "annotation_type": "re-annotations",
                    "targets": "answer |" if index == 0 else "answer",
                    "user_id": "dominant" if index < 3 else "other",
                },
            }
            for index in range(4)
        ]
    }
    result = audit_aya(accepted, raw)
    assert result["top_annotator_fraction"] == 0.75
    assert result["trailing_format_artifact_records"] == 1


def test_openr1_audit_rejects_reference_field_and_dependencies() -> None:
    accepted = [
        {
            **_record("openr1_math_220k", "math", "reasoning_math_science"),
            "messages": [
                {"content": "prompt", "role": "user"},
                {"content": "short", "role": "assistant"},
            ],
        }
    ]
    raw = {
        "rows": [
            {
                "row_idx": 1,
                "row": {
                    "is_reasoning_complete": [True, False],
                    "problem": "see ![](https://example.invalid/image.png)",
                    "solution": "short",
                    "source": "olympiads",
                    "uuid": "math",
                },
            }
        ]
    }
    result = audit_openr1(accepted, raw)
    assert result["accepted_assistant_matches_reference_solution"] == 1
    assert result["mixed_generation_completeness_records"] == 1
    assert result["external_image_dependency_records"] == 1
    assert result["short_reference_solution_records"] == 1


def test_candidate_gap_does_not_claim_a_trainable_mix() -> None:
    report = build_candidate_coverage_gap(
        source_canary_records=[_record("source", "one", "coding")],
        ai_candidate_records=[_record("owned", "two", "conversation")],
        target_mix={"coding": 0.5, "conversation": 0.25, "safety_helpfulness": 0.25},
    )
    assert report["diagnostic_only_not_a_trainable_mix"] is True
    assert report["zero_coverage_tasks"] == ["safety_helpfulness"]


def test_source_quality_evidence_seal_fails_closed(tmp_path: Path) -> None:
    evidence = {
        "audit_complete": True,
        "human_record_level_review_claimed": False,
        "production_promotions": 0,
        "raw_training_text_retained": False,
        "representative_audit_outcome": "complete_failed_closed",
        "schema_version": "lumi.sft-source-quality-audit-report.v0.1",
        "source_results": {
            source_id: {"decision": "blocked_after_representative_audit"}
            for source_id in (
                "coherelabs_aya_dataset",
                "openassistant_oasst1",
                "openr1_math_220k",
            )
        },
    }
    evidence["report_sha256"] = sha256_value(evidence)
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    assert load_source_quality_audit_evidence(path)["production_promotions"] == 0

    evidence["production_promotions"] = 1
    evidence["report_sha256"] = sha256_value(
        {key: value for key, value in evidence.items() if key != "report_sha256"}
    )
    path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(ValueError, match="may not promote"):
        load_source_quality_audit_evidence(path)
