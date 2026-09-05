from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lumi.posttraining_oasst1_curation import (
    EXPECTED_BINDING_PATHS,
    SOURCE_REVISION,
    build_preflight_report,
    file_sha256,
    label_measurement,
    load_oasst1_curation_plan,
    resolve_curation_binding,
    select_oasst1_pairs,
    verify_preflight_report,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs" / "lumi-4b-sft-oasst1-full-curation-plan-v0.1.json"
REPORT_PATH = ROOT / "evidence" / "lumi-4b-sft-oasst1-full-curation-preflight-v0.1.json"
TRAIN_OBJECT = "data/train-00000-of-00001-b42a775f407cee45.parquet"


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _labels(quality: float) -> dict:
    return {"count": [3], "name": ["quality"], "value": [quality]}


def _row(
    row_index: int,
    *,
    message_id: str,
    parent_id: str | None,
    role: str,
    text: str,
    lang: str = "en",
    quality: float = 0.9,
    rank: int | None = None,
) -> dict:
    return {
        "row_idx": row_index,
        "row": {
            "deleted": False,
            "labels": _labels(quality),
            "lang": lang,
            "message_id": message_id,
            "model_name": None,
            "parent_id": parent_id,
            "rank": rank,
            "review_count": 3,
            "review_result": True,
            "role": role,
            "synthetic": False,
            "text": text,
            "tree_state": "ready_for_export",
            "user_id": "must-not-enter-attribution",
        },
    }


def _bound_values(plan: dict) -> dict[str, dict]:
    values: dict[str, dict] = {}
    for name in (
        "pinned_metadata",
        "prior_reaudit_plan",
        "prior_reaudit_receipt",
        "raw_sample",
    ):
        path = resolve_curation_binding(ROOT, plan["bindings"][name], name=name)
        values[name] = _load_json(path)
    return values


def test_full_curation_plan_is_exact_bounded_and_not_authorized() -> None:
    plan = load_oasst1_curation_plan(PLAN_PATH, repository_root=ROOT)
    assert plan["authorization"]["network_access_authorized"] is False
    assert plan["authorization"]["paid_gpu_authorized"] is False
    assert plan["authorization"]["production_sft_authorized"] is False
    assert plan["limits"] == {
        "maximum_data_objects": 2,
        "maximum_decoded_bytes": 134_217_728,
        "maximum_network_bytes": 67_108_864,
        "maximum_objects": 4,
        "maximum_source_rows": 88_838,
        "maximum_train_rows": 84_437,
        "maximum_validation_rows": 4_401,
        "preflight_sample_rows": 100,
    }
    assert {item["path"] for item in plan["acquisition"]["exact_objects"]} == {
        "LICENSE",
        "README.md",
        TRAIN_OBJECT,
        "data/validation-00000-of-00001-134b8fd0c89408b6.parquet",
    }
    assert set(plan["bindings"]) == set(EXPECTED_BINDING_PATHS)


def test_quality_label_reader_supports_both_upstream_shapes() -> None:
    assert label_measurement({"labels": _labels(0.875)}, "quality") == (0.875, 3)
    assert label_measurement(
        {"labels": [{"count": 4, "name": "quality", "value": 0.9375}]},
        "quality",
    ) == (0.9375, 4)
    assert label_measurement(
        {
            "labels": {
                "count": [3, 3],
                "name": ["quality", "quality"],
                "value": [0.8, 0.9],
            }
        },
        "quality",
    ) is None


def test_selector_rejects_followups_unranked_low_quality_and_non_english() -> None:
    rows = [
        _row(0, message_id="prompt", parent_id=None, role="prompter", text="Explain recursion."),
        _row(
            1,
            message_id="answer",
            parent_id="prompt",
            rank=0,
            role="assistant",
            text="Recursion solves a problem with smaller instances.",
        ),
        _row(
            2,
            message_id="low-quality",
            parent_id="prompt",
            quality=0.5,
            rank=0,
            role="assistant",
            text="Low quality.",
        ),
        _row(
            3,
            message_id="unranked",
            parent_id="prompt",
            role="assistant",
            text="Unranked.",
        ),
        _row(
            4,
            message_id="followup-prompt",
            parent_id="answer",
            role="prompter",
            text="Give an example.",
        ),
        _row(
            5,
            message_id="followup-answer",
            parent_id="followup-prompt",
            rank=0,
            role="assistant",
            text="This turn lacks its ancestor context.",
        ),
        _row(
            6,
            lang="es",
            message_id="spanish-prompt",
            parent_id=None,
            role="prompter",
            text="Explica recursion.",
        ),
        _row(
            7,
            lang="es",
            message_id="spanish-answer",
            parent_id="spanish-prompt",
            rank=0,
            role="assistant",
            text="Una respuesta.",
        ),
    ]
    result = select_oasst1_pairs(
        rows,
        object_path=TRAIN_OBJECT,
        revision=SOURCE_REVISION,
        split="train",
    )
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate["task_category"] == "factual_qa"
    assert candidate["answer_quality"] == 0.9
    assert candidate["attribution"]["source_object_sha256"] is None
    assert "user_id" not in candidate["attribution"]
    assert "text" not in candidate["attribution"]
    assert result.metrics["rejection_counts"] == {
        "answer_quality_below_floor": 1,
        "assistant_not_rank_zero": 1,
        "followup_context_excluded": 1,
        "language_not_allowed_or_mismatched": 1,
    }


def test_pinned_sample_preflight_is_reproducible_and_raw_text_free() -> None:
    plan = load_oasst1_curation_plan(PLAN_PATH, repository_root=ROOT)
    bound = _bound_values(plan)
    expected = build_preflight_report(
        metadata=bound["pinned_metadata"],
        plan=plan,
        plan_file_sha256=file_sha256(PLAN_PATH),
        prior_plan=bound["prior_reaudit_plan"],
        prior_receipt=bound["prior_reaudit_receipt"],
        raw_sample=bound["raw_sample"],
    )
    observed = _load_json(REPORT_PATH)
    assert observed == expected
    verify_preflight_report(observed)
    assert observed["preflight_sample"]["accepted_pairs"] == 4
    assert observed["preflight_sample"]["root_direct_rank0_pairs"] == 10
    assert observed["distributions"]["task"] == {
        "coding": 1,
        "conversation": 1,
        "factual_qa": 1,
        "writing_summarization": 1,
    }
    serialized = json.dumps(observed, ensure_ascii=False)
    assert '"messages"' not in serialized
    assert '"user_id"' not in serialized


def test_plan_and_receipt_fail_closed_on_authority_or_seal_change(tmp_path: Path) -> None:
    changed_plan = copy.deepcopy(_load_json(PLAN_PATH))
    changed_plan["authorization"]["network_access_authorized"] = True
    changed_plan_path = tmp_path / "changed-plan.json"
    changed_plan_path.write_text(json.dumps(changed_plan), encoding="utf-8")
    with pytest.raises(ValueError, match="authorization changed"):
        load_oasst1_curation_plan(changed_plan_path, repository_root=ROOT)

    changed_report = copy.deepcopy(_load_json(REPORT_PATH))
    changed_report["production_eligible"] = True
    with pytest.raises(ValueError, match="receipt seal changed"):
        verify_preflight_report(changed_report)
