from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_aya_repair import (
    EXPECTED_AUTHORIZATION,
    build_aya_repair_report,
    build_aya_sft_records,
    finalize_aya_repair_candidates,
    load_aya_repair_plan,
    select_aya_repair_candidates,
    validate_report_seal,
)
from lumi.posttraining_oasst1_full_run import sha256_value

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "lumi-4b-sft-aya-bengali-repair-preflight-v0.1.json"


def _wrapped_rows() -> list[dict[str, object]]:
    rows = []
    for index in range(1_534):
        target = f"উত্তর {index}"
        if index < 223:
            target += "|"
        prompt = f"প্রশ্ন {index}"
        if index == 400:
            prompt = "plain English prompt"
        if index == 401:
            target = "plain English target"
        if index == 402:
            prompt = "plain English prompt two"
            target = "plain English target two"
        if index >= 1_530:
            duplicate = 300 + (index - 1_530)
            prompt = f"প্রশ্ন {duplicate}"
            target = f"উত্তর {duplicate}"
        rows.append(
            {
                "row": {
                    "annotation_type": ("original-annotations" if index % 2 else "re-annotations"),
                    "inputs": prompt,
                    "language": "Bengali",
                    "language_code": "ben",
                    "targets": target,
                    "user_id": "dominant" if index < 800 else f"annotator-{index % 16}",
                },
                "row_idx": index,
                "truncated_cells": [],
            }
        )
    return rows


def test_aya_repair_plan_is_exact_local_and_non_authorizing() -> None:
    plan = load_aya_repair_plan(PLAN, repository_root=ROOT)
    assert plan["authorization"] == EXPECTED_AUTHORIZATION
    assert plan["limits"]["expected_source_rows"] == 1_534
    assert plan["limits"]["expected_page_objects"] == 23
    assert plan["output"]["candidate_records_materialized"] is False
    assert plan["output"]["production_promotion_permitted"] is False


def test_aya_repair_normalizes_deduplicates_and_caps_concentration() -> None:
    selection = select_aya_repair_candidates(_wrapped_rows())
    assert selection.metrics["terminal_pipe_artifacts_repaired"] == 223
    assert selection.metrics["rejection_counts"]["missing_bengali_script"] == 3
    assert selection.metrics["rejection_counts"]["normalized_exact_pair_duplicate"] == 4
    assert selection.metrics["deduplicated_rows"] == 1_527
    assert all(not row["target"].endswith("|") for row in selection.candidates)

    records, annotators = build_aya_sft_records(selection.candidates)
    final, concentration = finalize_aya_repair_candidates(
        records,
        annotators=annotators,
    )
    assert final
    assert concentration["maximum_fraction_after"] <= 0.5
    assert concentration["records_removed"] > 0
    assert all(record["language"] == "bn" for record in final)
    assert all(record["task_category"] == "multilingual_bengali" for record in final)


def test_aya_repair_report_is_raw_text_free_and_nonproduction() -> None:
    records = [
        {
            "language": "bn",
            "messages": [
                {"content": "private prompt", "role": "user"},
                {"content": "private response", "role": "assistant"},
            ],
            "provenance": {"content_sha256": "a" * 64},
            "record_id": "aya-ben-example",
            "split": "train",
            "task_category": "multilingual_bengali",
        }
    ]
    report = build_aya_repair_report(
        bindings={"example": "b" * 64},
        concentration_metrics={"maximum_fraction_after": 0.5},
        filter_report={
            "accepted": 1,
            "accepted_dataset_sha256": "c" * 64,
            "rejected": 0,
            "rejection_counts": {},
            "report_sha256": "d" * 64,
        },
        final_records=records,
        owner_sample_records=1,
        plan_file_sha256="e" * 64,
        selection_metrics={"input_rows": 1},
        token_metrics={"supervised_tokens": 2},
        annotators={"aya-ben-example": "f" * 64},
    )
    validate_report_seal(report)
    serialized = json.dumps(report, ensure_ascii=False)
    assert "private prompt" not in serialized
    assert "private response" not in serialized
    assert report["candidate"]["materialized"] is False
    assert report["production"]["production_eligible"] is False
    assert report["raw_training_text_present"] is False


def test_aya_repair_plan_rejects_a_tampered_binding(tmp_path: Path) -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    plan["bindings"]["candidate_source_registry"]["sha256"] = "0" * 64
    tampered = tmp_path / "plan.json"
    tampered.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="binding changed"):
        load_aya_repair_plan(tampered, repository_root=ROOT)


def test_report_seal_rejects_tampering() -> None:
    report = {
        "production": {
            "candidate_registry_promotion_authorized": False,
            "dataset_seal_authorized": False,
            "production_eligible": False,
            "sft_training_authorized": False,
        },
        "raw_training_text_present": False,
        "schema_version": "lumi.sft-aya-bengali-repair-preflight-report.v0.1",
    }
    report["report_sha256"] = sha256_value(report)
    report["raw_training_text_present"] = True
    with pytest.raises(ValueError, match="seal is invalid"):
        validate_report_seal(report)
