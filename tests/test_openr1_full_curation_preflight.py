from __future__ import annotations

import json
from pathlib import Path

from lumi.posttraining_openr1_full_curation import (
    DATA_OBJECTS,
    build_openr1_full_curation_preflight,
    file_sha256,
    load_openr1_full_curation_plan,
    resolve_binding,
    summarize_pinned_metadata,
    validate_openr1_full_curation_preflight,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-openr1-full-curation-plan-v0.1.json"
REPORT = ROOT / "evidence/lumi-4b-sft-openr1-full-curation-preflight-v0.1.json"


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_openr1_full_plan_is_hash_bound_and_fail_closed() -> None:
    plan = load_openr1_full_curation_plan(PLAN, repository_root=ROOT)
    assert plan["authorization"]["network_access_authorized"] is False
    assert plan["authorization"]["bulk_data_download_authorized"] is False
    assert plan["authorization"]["production_sft_authorized"] is False
    assert plan["output"]["candidate_materialization_permitted"] is False
    assert len(plan["acquisition"]["exact_objects"]) == 11


def test_pinned_metadata_has_exact_default_split_and_ten_shards() -> None:
    plan = load_openr1_full_curation_plan(PLAN, repository_root=ROOT)
    metadata = _load_json(
        resolve_binding(
            ROOT,
            plan["bindings"]["pinned_metadata"],
            name="pinned_metadata",
        )
    )
    summary = summarize_pinned_metadata(metadata)
    assert summary["rows"] == 93_733
    assert summary["data_object_count"] == len(DATA_OBJECTS) == 10
    assert summary["download_data_bytes"] == 2_149_897_914
    assert summary["decoded_data_bytes"] == 4_964_543_659


def test_exact_preflight_is_reproducible_raw_text_free_and_nonproduction() -> None:
    plan = load_openr1_full_curation_plan(PLAN, repository_root=ROOT)
    bindings = plan["bindings"]
    rebuilt = build_openr1_full_curation_preflight(
        metadata=_load_json(
            resolve_binding(ROOT, bindings["pinned_metadata"], name="pinned_metadata")
        ),
        plan=plan,
        plan_file_sha256=file_sha256(PLAN),
        secondary_pii_evidence=_load_json(
            resolve_binding(
                ROOT,
                bindings["secondary_pii_evidence"],
                name="secondary_pii_evidence",
            )
        ),
        verified_trace_evidence=_load_json(
            resolve_binding(
                ROOT,
                bindings["verified_trace_evidence"],
                name="verified_trace_evidence",
            )
        ),
    )
    report = _load_json(REPORT)
    assert rebuilt == report
    validate_openr1_full_curation_preflight(report)
    assert report["bounded_slice_evidence"] == {
        "encoded_4k_records": 6,
        "global_filter_modified": False,
        "normal_filter_records": 12,
        "secondary_unresolved_records": 0,
        "selected_verified_trace_records": 18,
    }
    assert report["production_promotions"] == 0
    assert report["rights_boundary"][
        "dataset_level_license_treated_as_per_row_rights_proof"
    ] is False


def test_preflight_report_contains_no_training_payload() -> None:
    serialized = REPORT.read_text(encoding="utf-8")
    for forbidden in (
        '"messages"',
        '"problem"',
        '"solution"',
        '"generations"',
        '"assistant_response"',
        '"user_prompt"',
    ):
        assert forbidden not in serialized
