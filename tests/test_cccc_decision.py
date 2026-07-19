from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from lumi.cccc_decision import (
    build_cccc_owner_decision_report,
    evaluate_cccc_owner_decision,
    verify_cccc_owner_decision_report,
)

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "configs" / "cccc-owner-decision-v0.1.json"


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _inputs() -> tuple[dict, dict, dict, dict, dict, dict]:
    return (
        _load("configs/cccc-owner-decision-v0.1.json"),
        _load("configs/component-sources-v0.1.json"),
        _load("configs/common-pile-source-files-v0.1.json"),
        _load("configs/data-mix-v0.1.json"),
        _load("configs/corpus-retention-snapshot-v0.1.json"),
        _load("reports/component-provenance-audit-v0.1.json"),
    )


def _evaluate(specification: dict) -> dict:
    _, inventory, files, mixture, retention, provenance = _inputs()
    return evaluate_cccc_owner_decision(
        specification, inventory, files, mixture, retention, provenance
    )


def _decided_specification(option: str) -> dict:
    specification, *_ = _inputs()
    required = specification["required_acknowledgements"]
    acknowledgement_names = set(required["all_options"]) | set(required[option])
    specification["decision_record"] = {
        "acknowledgements": {name: True for name in sorted(acknowledgement_names)},
        "decided_at": "2026-07-19T12:00:00Z",
        "owner": "Afnan Technologies Inc.",
        "reason": "Explicit owner choice after reviewing the pinned evidence and consequences.",
        "selected_option": option,
        "state": "decided",
    }
    return specification


def test_pinned_packet_records_explicit_include_without_authorizing_acquisition() -> None:
    report = build_cccc_owner_decision_report(CONTRACT, repository_root=ROOT)
    assert report["decision"] == {
        "acquisition_status": "blocked_pending_separate_production_authorization",
        "license_basis_resolution": "owner_accepted_source_level_domain_audit",
        "next_action": "request_separate_production_acquisition_authorization",
        "owner_decision": "decided",
        "production_acquisition": "not_authorized",
        "selected_option": "include_with_caveat",
    }
    assert report["impact"] == {
        "compressed_source_bytes": 21_497_337_171,
        "current_open_web_components": ["cccc"],
        "data_files": 53,
        "exclude_replacement_candidate_tokens_required": 12_000_000_000,
        "final_open_web_target_tokens": 8_000_000_000,
        "include_projected_candidate_tokens": 13_210_015_719,
        "include_projected_headroom_fraction": pytest.approx(0.10083464325),
        "include_projected_headroom_tokens": 1_210_015_719,
        "minimum_open_web_candidate_tokens": 12_000_000_000,
        "payload_bytes_downloaded": 0,
    }
    assert all(
        report["authorization"][name] is False
        for name in (
            "bulk_corpus_materialization",
            "create_cloud_resources",
            "full_4b_training",
            "paid_compute",
            "transfer_source_payload",
        )
    )


def test_pending_record_remains_supported_and_fail_closed() -> None:
    specification, *_ = _inputs()
    specification["decision_record"] = {
        "acknowledgements": {},
        "decided_at": None,
        "owner": None,
        "reason": None,
        "selected_option": None,
        "state": "pending_owner_decision",
    }
    report = _evaluate(specification)
    assert report["decision"]["acquisition_status"] == "blocked_pending_owner_decision"
    assert report["decision"]["selected_option"] is None


def test_include_and_exclude_require_explicit_complete_owner_record() -> None:
    include = _evaluate(_decided_specification("include_with_caveat"))
    assert include["decision"]["license_basis_resolution"] == (
        "owner_accepted_source_level_domain_audit"
    )
    assert include["decision"]["production_acquisition"] == "not_authorized"

    exclude = _evaluate(_decided_specification("exclude_and_replace"))
    assert exclude["decision"]["license_basis_resolution"] == (
        "excluded_pending_open_web_replacement"
    )
    assert exclude["decision"]["acquisition_status"] == (
        "blocked_pending_open_web_replacement"
    )

    incomplete = _decided_specification("include_with_caveat")
    incomplete["decision_record"]["acknowledgements"].pop(
        "preserve_document_and_warc_traceability"
    )
    with pytest.raises(ValueError, match="acknowledgement set"):
        _evaluate(incomplete)


def test_schema_card_and_capacity_drift_fail_closed() -> None:
    specification, inventory, files, mixture, retention, provenance = _inputs()
    changed_inventory = copy.deepcopy(inventory)
    cccc = next(row for row in changed_inventory["sources"] if row["component"] == "cccc")
    cccc["license_paths"] = ["metadata.license"]
    with pytest.raises(ValueError, match="unexpectedly claims"):
        evaluate_cccc_owner_decision(
            specification, changed_inventory, files, mixture, retention, provenance
        )

    changed_card = copy.deepcopy(specification)
    changed_card["reviewed_upstream_evidence"]["filtered_dataset_card"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 changed"):
        evaluate_cccc_owner_decision(
            changed_card, inventory, files, mixture, retention, provenance
        )

    changed_retention = copy.deepcopy(retention)
    open_web = next(row for row in changed_retention["categories"] if row["category"] == "open_web")
    open_web["projected_post_filter_tokens"] -= 1
    with pytest.raises(ValueError, match="retention evidence changed"):
        evaluate_cccc_owner_decision(
            specification, inventory, files, mixture, changed_retention, provenance
        )

    changed_acknowledgements = copy.deepcopy(specification)
    changed_acknowledgements["required_acknowledgements"]["all_options"].pop()
    with pytest.raises(ValueError, match="common acknowledgement set changed"):
        evaluate_cccc_owner_decision(
            changed_acknowledgements, inventory, files, mixture, retention, provenance
        )


def test_sealed_report_reproduces_and_tampering_is_detected(tmp_path: Path) -> None:
    report = build_cccc_owner_decision_report(CONTRACT, repository_root=ROOT)
    report_path = tmp_path / "report.json"
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    report_path.write_text(encoded, encoding="utf-8")
    manifest_path = tmp_path / "manifest.sha256"
    manifest_path.write_text(
        f"{hashlib.sha256(encoded.encode()).hexdigest()}  {report_path.name}\n",
        encoding="utf-8",
    )
    assert (
        verify_cccc_owner_decision_report(
            CONTRACT,
            report_path,
            manifest_path,
            repository_root=ROOT,
        )
        == report
    )
    report_path.write_text(encoded + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="hash does not match"):
        verify_cccc_owner_decision_report(
            CONTRACT,
            report_path,
            manifest_path,
            repository_root=ROOT,
        )
