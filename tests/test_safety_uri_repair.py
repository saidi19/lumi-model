from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lumi.posttraining_safety_uri_repair import (
    build_safety_uri_repair_verification,
    file_sha256,
    json_file_bytes,
    load_safety_uri_repair_plan,
    rebuild_safety_uri_repair_outputs,
    resolve_output,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-safety-uri-repair-v0.1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_safety_uri_outputs_reproduce_exactly() -> None:
    plan = load_safety_uri_repair_plan(PLAN, repository_root=ROOT)
    outputs = rebuild_safety_uri_repair_outputs(
        plan=plan,
        plan_file_sha256=file_sha256(PLAN),
        repository_root=ROOT,
    )
    expected = {
        "artifact_manifest_path": outputs["artifact_manifest_bytes"],
        "private_safety_candidate_path": outputs["private_safety_candidate_bytes"],
        "report_path": json_file_bytes(outputs["report"]),
    }
    for name, payload in expected.items():
        path = resolve_output(ROOT, plan["output"][name], name=name)
        assert path.read_bytes() == payload


def test_all_rows_have_exact_unique_first_party_uri_and_only_metadata_delta() -> None:
    plan = load_safety_uri_repair_plan(PLAN, repository_root=ROOT)
    input_path = resolve_output(
        ROOT,
        plan["bindings"]["predecessor_candidate"]["path"],
        name="input",
    )
    output_path = resolve_output(
        ROOT,
        plan["output"]["private_safety_candidate_path"],
        name="output",
    )
    inputs = [json.loads(line) for line in input_path.read_text().splitlines()]
    outputs = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(inputs) == len(outputs) == 320
    uris = set()
    for before, after in zip(inputs, outputs, strict=True):
        expected_uri = (
            "private://Afnan-Technologies-Inc/lumi-safety-tool-sft-v0.1/"
            f"{before['record_id']}"
        )
        assert after["source"]["uri"] == expected_uri
        uris.add(expected_uri)
        comparison = copy.deepcopy(after)
        comparison["source"].pop("uri")
        comparison["provenance"]["transform_sha256"] = before["provenance"][
            "transform_sha256"
        ]
        assert comparison == before
    assert len(uris) == 320


def test_generic_validation_and_frozen_counts_pass() -> None:
    plan = load_safety_uri_repair_plan(PLAN, repository_root=ROOT)
    report_path = resolve_output(ROOT, plan["output"]["report_path"], name="report")
    report = _load(report_path)
    assert report["generic_sft_validation"] == {
        "accepted": 320,
        "benchmark_contamination_rejections": 0,
        "exact_normalized_duplicates": 0,
        "raw_text_retained": False,
        "rejected": 0,
        "rejection_counts": {},
        "report_sha256": report["generic_sft_validation"]["report_sha256"],
        "source_registry_rejections": 0,
        "total": 320,
    }
    assert report["frozen_token_verification"]["by_split"] == {
        "train": {"records": 240, "supervised_tokens": 18_636},
        "validation": {"records": 80, "supervised_tokens": 6_388},
    }


def test_repo_receipts_are_raw_text_free_and_verification_reproduces() -> None:
    plan = load_safety_uri_repair_plan(PLAN, repository_root=ROOT)
    report_path = resolve_output(ROOT, plan["output"]["report_path"], name="report")
    verification_path = resolve_output(
        ROOT,
        plan["output"]["verification_path"],
        name="verification",
    )
    report = _load(report_path)
    expected = build_safety_uri_repair_verification(
        plan=plan,
        report=report,
        report_file_sha256=file_sha256(report_path),
    )
    assert _load(verification_path) == expected
    for path in (report_path, verification_path):
        serialized = path.read_text(encoding="utf-8")
        for forbidden in ('"messages"', '"content"', '"assistant_response"'):
            assert forbidden not in serialized


def test_plan_rejects_scope_expansion(tmp_path: Path) -> None:
    changed = copy.deepcopy(_load(PLAN))
    changed["authorization"]["production_sft_authorized"] = True
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="authorization"):
        load_safety_uri_repair_plan(path, repository_root=ROOT)
