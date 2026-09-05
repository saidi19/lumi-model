from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lumi.posttraining_aya_validation import (
    EXPECTED_PROJECTION,
    build_split_projection,
    build_validation_preflight_report,
    load_aya_validation_plan,
    resolve_validation_binding,
    validate_validation_preflight_report,
    verify_validation_upstream,
)
from lumi.posttraining_first_party import file_sha256
from lumi.posttraining_oasst1_full_run import load_runtime_components

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "lumi-4b-sft-aya-bengali-validation-preflight-v0.1.json"
REPORT = ROOT / "evidence" / "lumi-4b-sft-aya-bengali-validation-preflight-v0.1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_exact_aya_validation_preflight_is_reproducible_and_nonproduction() -> None:
    plan = load_aya_validation_plan(PLAN, repository_root=ROOT)
    bindings = plan["bindings"]
    verify_validation_upstream(
        materialization_report=_load(
            resolve_validation_binding(ROOT, bindings["materialization_report"])
        ),
        owner_review_receipt=_load(
            resolve_validation_binding(ROOT, bindings["owner_review_receipt"])
        ),
        source_promotion_receipt=_load(
            resolve_validation_binding(ROOT, bindings["source_promotion_receipt"])
        ),
    )
    _policy, _registry, _benchmark, template, tokenizer = load_runtime_components(
        plan,
        repository_root=ROOT,
    )
    projection = build_split_projection(
        _load_jsonl(resolve_validation_binding(ROOT, bindings["candidate_records"])),
        _load_jsonl(resolve_validation_binding(ROOT, bindings["attribution_manifest"])),
        plan=plan,
        template=template,
        tokenizer=tokenizer,
    )
    assert projection == EXPECTED_PROJECTION
    expected = build_validation_preflight_report(
        plan=plan,
        plan_file_sha256=file_sha256(PLAN),
        projection=projection,
    )
    observed = _load(REPORT)
    assert observed == expected
    validate_validation_preflight_report(observed)
    assert observed["candidate_split_materialized"] is False
    assert observed["production_dataset_promotions"] == 0


def test_aya_validation_projection_has_zero_cross_split_leakage() -> None:
    report = _load(REPORT)
    assert report["projection"]["leakage"] == {
        "exact_prompt_overlap": 0,
        "exact_response_overlap": 0,
        "prompt_near_duplicate_validation_records": 0,
        "response_near_duplicate_validation_records": 0,
    }
    assert report["projection"]["annotators"]["train_maximum_fraction"] == 0.5
    assert report["projection"]["annotators"]["validation_maximum_fraction"] == 0.5


def test_aya_validation_plan_rejects_authorization_drift(tmp_path: Path) -> None:
    changed = copy.deepcopy(_load(PLAN))
    changed["authorization"]["candidate_split_materialization_authorized"] = True
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="authorization changed"):
        load_aya_validation_plan(path, repository_root=ROOT)
