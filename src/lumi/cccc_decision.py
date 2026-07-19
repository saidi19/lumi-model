"""Fail-closed owner-decision gate for Lumi's CCCC corpus component."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
EXPECTED_GATE = "cccc_owner_decision_v0.1"
EXPECTED_COMPONENT = "cccc"
EXPECTED_DATASET = "common-pile/cccc_filtered"
EXPECTED_REVISION = "03a3de5713a0bb23267d26724346508af0f25327"
EXPECTED_CARD_SHA256 = "13b727dd5bc1045a502705fa70ada1254586fa19b7ddc2fd987dec89763422ad"
EXPECTED_CARD_BLOB_SHA1 = "5c632ee7dcdd38e6834a731968bddcc6dbea9b62"
EXPECTED_OPTIONS = {"exclude_and_replace", "include_with_caveat"}
EXPECTED_RESOLVED_POLICIES = {
    "exclude_and_replace": "excluded_pending_open_web_replacement",
    "include_with_caveat": "owner_accepted_source_level_domain_audit",
}
EXPECTED_COMMON_ACKNOWLEDGEMENTS = {
    "local_audit_found_no_claimed_per_document_license_field",
    "source_level_domain_audit_is_not_per_document_license_metadata",
    "this_decision_does_not_authorize_data_transfer_cloud_compute_or_training",
    "this_engineering_record_is_not_legal_advice",
    "upstream_warns_about_license_laundering_and_inaccurate_metadata",
}
EXPECTED_OPTION_ACKNOWLEDGEMENTS = {
    "exclude_and_replace": {
        "open_web_floor_requires_a_pinned_measured_replacement",
        "other_categories_will_not_be_silently_reweighted_or_repeated",
    },
    "include_with_caveat": {
        "accept_source_level_basis_for_lumi_personal_research",
        "maintain_removal_ledger_and_rebuild_affected_shards",
        "preserve_document_and_warc_traceability",
    },
}
FORBIDDEN_AUTHORIZATIONS = {
    "bulk_corpus_materialization",
    "create_cloud_resources",
    "full_4b_training",
    "paid_compute",
    "transfer_source_payload",
}
IMPLEMENTATION_FILES = ("src/lumi/cccc_decision.py",)
ISO_UTC_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _resolve(root: Path, raw_path: Any, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} path is invalid")
    path = (root / raw_path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"{label} escapes the repository")
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    return path


def _unique(items: Any, key: str, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(items, list):
        raise ValueError(f"{label} must be a list")
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"{label} row must be an object")
        identity = item.get(key)
        if not isinstance(identity, str) or not identity:
            raise ValueError(f"{label} row has an invalid {key}")
        if identity in result:
            raise ValueError(f"duplicate {label} {key}: {identity}")
        result[identity] = item
    return result


def _validate_authorization(specification: Mapping[str, Any]) -> dict[str, Any]:
    authorization = specification.get("authorization")
    if not isinstance(authorization, dict):
        raise ValueError("CCCC decision authorization is missing")
    if authorization.get("local_evidence_review") is not True:
        raise ValueError("local CCCC evidence review must be explicitly authorized")
    if authorization.get("fetch_public_metadata") is not False:
        raise ValueError("the sealed CCCC decision gate must not fetch network metadata")
    if any(authorization.get(name) is not False for name in FORBIDDEN_AUTHORIZATIONS):
        raise ValueError("CCCC decision must not authorize data, cloud, compute, or training")
    return dict(authorization)


def _validate_reviewed_evidence(
    specification: Mapping[str, Any], source_snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    evidence = specification.get("reviewed_upstream_evidence")
    if not isinstance(evidence, dict):
        raise ValueError("reviewed upstream CCCC evidence is missing")
    card = evidence.get("filtered_dataset_card")
    if not isinstance(card, dict):
        raise ValueError("pinned CCCC dataset-card evidence is missing")
    expected_card_url = (
        "https://huggingface.co/datasets/common-pile/cccc_filtered/blob/"
        f"{EXPECTED_REVISION}/README.md"
    )
    if card.get("url") != expected_card_url:
        raise ValueError("CCCC dataset-card URL is not pinned to the reviewed revision")
    if card.get("sha256") != EXPECTED_CARD_SHA256 or card.get("bytes") != 4060:
        raise ValueError("CCCC dataset-card bytes or SHA-256 changed")
    if card.get("git_blob_sha1") != EXPECTED_CARD_BLOB_SHA1:
        raise ValueError("CCCC dataset-card Git blob changed")
    control_files = {
        row["path"]: row for row in source_snapshot.get("repository_control_files", [])
    }
    if control_files.get("README.md", {}).get("git_blob_sha1") != EXPECTED_CARD_BLOB_SHA1:
        raise ValueError("source-file snapshot does not contain the reviewed CCCC card")
    claims = card.get("claims")
    expected_claims = {
        "common_crawl_snapshots": 52,
        "contains_unresolved_removed_domain_placeholder": True,
        "licenses_named": ["CC BY", "CC BY-SA", "CC0"],
        "manual_domains_reviewed": 1000,
        "per_document_metadata_license_claimed": True,
        "retained_domains": 537,
        "warns_about_license_laundering_and_inaccurate_metadata": True,
    }
    if claims != expected_claims:
        raise ValueError("reviewed CCCC dataset-card claims changed")
    paper = evidence.get("paper")
    if not isinstance(paper, dict) or paper.get("arxiv_id") != "2506.05209":
        raise ValueError("reviewed Common Pile paper is not pinned")
    if paper.get("url") != "https://arxiv.org/abs/2506.05209":
        raise ValueError("reviewed Common Pile paper URL changed")
    expected_paper_claims = {
        "collection_license_does_not_license_underlying_documents": True,
        "license_laundering_cannot_be_exhaustively_eliminated": True,
        "licenses_can_change_after_collection": True,
        "top_domains_manually_verified": True,
    }
    if paper.get("claims") != expected_paper_claims:
        raise ValueError("reviewed Common Pile paper findings changed")
    return {"filtered_dataset_card": dict(card), "paper": dict(paper)}


def _validate_decision(
    specification: Mapping[str, Any], options: Mapping[str, Any]
) -> dict[str, Any]:
    record = specification.get("decision_record")
    if not isinstance(record, dict):
        raise ValueError("CCCC decision record is missing")
    state = record.get("state")
    selected = record.get("selected_option")
    if state == "pending_owner_decision":
        if selected is not None:
            raise ValueError("pending CCCC decision cannot select an option")
        for field in ("decided_at", "owner", "reason"):
            if record.get(field) is not None:
                raise ValueError(f"pending CCCC decision must leave {field} null")
        if record.get("acknowledgements") != {}:
            raise ValueError("pending CCCC decision must not pre-check acknowledgements")
        return {
            "acquisition_status": "blocked_pending_owner_decision",
            "license_basis_resolution": "pending_owner_decision",
            "next_action": "owner_select_include_with_caveat_or_exclude_and_replace",
            "owner_decision": "pending_owner_decision",
            "production_acquisition": "not_authorized",
            "selected_option": None,
        }
    if state != "decided" or selected not in EXPECTED_OPTIONS:
        raise ValueError("CCCC decision state or selected option is invalid")
    owner = record.get("owner")
    reason = record.get("reason")
    decided_at = record.get("decided_at")
    if not isinstance(owner, str) or not owner.strip():
        raise ValueError("decided CCCC record requires a named owner")
    if not isinstance(reason, str) or len(reason.strip()) < 20:
        raise ValueError("decided CCCC record requires a meaningful reason")
    if not isinstance(decided_at, str) or ISO_UTC_PATTERN.fullmatch(decided_at) is None:
        raise ValueError("decided CCCC record requires an ISO-8601 UTC timestamp")
    try:
        datetime.strptime(decided_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError as error:
        raise ValueError("decided CCCC timestamp is not a real UTC time") from error
    requirements = specification.get("required_acknowledgements")
    if not isinstance(requirements, dict):
        raise ValueError("CCCC acknowledgement requirements are missing")
    if set(requirements) != {"all_options", *EXPECTED_OPTIONS}:
        raise ValueError("CCCC acknowledgement option set changed")
    if set(requirements.get("all_options", [])) != EXPECTED_COMMON_ACKNOWLEDGEMENTS:
        raise ValueError("CCCC common acknowledgement set changed")
    for option, expected in EXPECTED_OPTION_ACKNOWLEDGEMENTS.items():
        if set(requirements.get(option, [])) != expected:
            raise ValueError(f"CCCC {option} acknowledgement set changed")
    required = EXPECTED_COMMON_ACKNOWLEDGEMENTS | EXPECTED_OPTION_ACKNOWLEDGEMENTS[selected]
    acknowledgements = record.get("acknowledgements")
    if not isinstance(acknowledgements, dict) or set(acknowledgements) != required:
        raise ValueError("CCCC decision acknowledgement set is incomplete or unexpected")
    if any(value is not True for value in acknowledgements.values()):
        raise ValueError("every required CCCC acknowledgement must be true")
    resolved_policy = options[selected]["resolved_license_policy"]
    if selected == "include_with_caveat":
        return {
            "acquisition_status": "blocked_pending_separate_production_authorization",
            "license_basis_resolution": resolved_policy,
            "next_action": "request_separate_production_acquisition_authorization",
            "owner_decision": "decided",
            "production_acquisition": "not_authorized",
            "selected_option": selected,
        }
    return {
        "acquisition_status": "blocked_pending_open_web_replacement",
        "license_basis_resolution": resolved_policy,
        "next_action": "pin_audit_and_measure_open_web_replacement",
        "owner_decision": "decided",
        "production_acquisition": "not_authorized",
        "selected_option": selected,
    }


def evaluate_cccc_owner_decision(
    specification: dict[str, Any],
    source_inventory: dict[str, Any],
    source_file_manifest: dict[str, Any],
    mixture: dict[str, Any],
    retention_snapshot: dict[str, Any],
    provenance_audit: dict[str, Any],
) -> dict[str, Any]:
    """Validate CCCC evidence and return the deterministic owner decision packet."""

    if specification.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported CCCC owner-decision schema")
    if specification.get("gate") != EXPECTED_GATE:
        raise ValueError("unexpected CCCC owner-decision gate")
    authorization = _validate_authorization(specification)
    source = specification.get("source")
    if source != {
        "category": "open_web",
        "component": EXPECTED_COMPONENT,
        "dataset_id": EXPECTED_DATASET,
        "dataset_revision": EXPECTED_REVISION,
    }:
        raise ValueError("CCCC owner-decision source pin changed")

    inventory = _unique(source_inventory.get("sources"), "component", "source inventory")
    cccc_inventory = inventory.get(EXPECTED_COMPONENT)
    if cccc_inventory is None:
        raise ValueError("CCCC is absent from the source inventory")
    if cccc_inventory.get("dataset_id") != EXPECTED_DATASET:
        raise ValueError("CCCC inventory dataset changed")
    if cccc_inventory.get("revision") != EXPECTED_REVISION:
        raise ValueError("CCCC inventory revision changed")
    if cccc_inventory.get("license_paths") != []:
        raise ValueError("CCCC unexpectedly claims per-document license paths")
    if cccc_inventory.get("status") != "source_level_basis_documented_schema_mismatch_caveat":
        raise ValueError("CCCC inventory caveat changed")
    basis = cccc_inventory.get("license_basis")
    if not isinstance(basis, dict) or basis.get("kind") != "source_level_domain_audit":
        raise ValueError("CCCC source-level license basis is missing")
    expected_warning = (
        "dataset card claims a per-document license field, but 0 of 10000 audited raw and "
        "filtered rows contained it"
    )
    if basis.get("warning") != expected_warning:
        raise ValueError("CCCC local license-field audit warning changed")

    source_snapshots = _unique(
        source_file_manifest.get("sources"), "component", "source-file manifest"
    )
    cccc_snapshot = source_snapshots.get(EXPECTED_COMPONENT)
    if cccc_snapshot is None:
        raise ValueError("CCCC is absent from the source-file manifest")
    if cccc_snapshot.get("dataset_id") != EXPECTED_DATASET:
        raise ValueError("CCCC source-file dataset changed")
    if cccc_snapshot.get("dataset_revision") != EXPECTED_REVISION:
        raise ValueError("CCCC source-file revision changed")
    if cccc_snapshot.get("data_file_count") != 53:
        raise ValueError("CCCC source-file count changed")
    if cccc_snapshot.get("compressed_source_bytes") != 21_497_337_171:
        raise ValueError("CCCC compressed source bytes changed")
    data_files = cccc_snapshot.get("data_files")
    if not isinstance(data_files, list) or len(data_files) != 53:
        raise ValueError("CCCC exact source-file records changed")
    if sum(int(row.get("source_bytes", -1)) for row in data_files) != 21_497_337_171:
        raise ValueError("CCCC source-file byte sum changed")
    if cccc_snapshot.get("license_policy_status") != (
        "source_level_basis_documented_schema_mismatch_caveat"
    ):
        raise ValueError("CCCC source-file license caveat changed")
    if cccc_snapshot.get("provenance_paths") != [
        "id",
        "metadata.provenance",
        "metadata.url",
        "metadata.warc_url",
        "source",
    ]:
        raise ValueError("CCCC source-file provenance paths changed")
    if cccc_snapshot.get("acquisition_status") != (
        "blocked_pending_owner_cccc_license_decision_and_production_authorization"
    ):
        raise ValueError("CCCC source files are not blocked pending this decision")
    if source_file_manifest.get("summary", {}).get("corpus_payload_bytes_downloaded") != 0:
        raise ValueError("CCCC decision gate requires the metadata-only payload boundary")

    categories = _unique(mixture.get("mixture"), "category", "mixture")
    open_web = categories.get("open_web")
    if open_web is None or open_web.get("source_components") != [EXPECTED_COMPONENT]:
        raise ValueError("CCCC must remain the only current open-web source")
    if open_web.get("target_tokens_at_80b") != 8_000_000_000:
        raise ValueError("open-web final target changed")
    retention = _unique(retention_snapshot.get("categories"), "category", "retention snapshot")
    open_web_retention = retention.get("open_web")
    expected_retention = {
        "accepted_text_utf8_bytes": 63_788_110,
        "accepted_tokens": 14_576_877,
        "category": "open_web",
        "minimum_candidate_tokens": 12_000_000_000,
        "projected_post_filter_tokens": 13_210_015_719,
        "raw_text_utf8_bytes": 64_001_352,
    }
    if open_web_retention != expected_retention:
        raise ValueError("CCCC open-web retention evidence changed")

    columns = provenance_audit.get("columns")
    results = provenance_audit.get("results")
    if not isinstance(columns, list) or not isinstance(results, list):
        raise ValueError("component provenance audit is malformed")
    try:
        component_index = columns.index("component")
        cccc_row = next(row for row in results if row[component_index] == EXPECTED_COMPONENT)
    except (ValueError, StopIteration) as error:
        raise ValueError("CCCC is absent from the component provenance audit") from error
    expected_audit = [EXPECTED_COMPONENT, 2000, 2000, 2000, 2000, 2000]
    if cccc_row != expected_audit:
        raise ValueError("CCCC bounded provenance audit changed")
    limitations = provenance_audit.get("limitations")
    if not isinstance(limitations, list) or not any(
        "source-level domain audit" in limitation for limitation in limitations
    ):
        raise ValueError("CCCC provenance audit lost its source-level limitation")

    reviewed_evidence = _validate_reviewed_evidence(specification, cccc_snapshot)
    options = specification.get("options")
    if not isinstance(options, dict) or set(options) != EXPECTED_OPTIONS:
        raise ValueError("CCCC decision options changed")
    for name in EXPECTED_OPTIONS:
        option = options[name]
        if not isinstance(option, dict):
            raise ValueError(f"CCCC option {name} is malformed")
        if option.get("resolved_license_policy") != EXPECTED_RESOLVED_POLICIES[name]:
            raise ValueError(f"CCCC option {name} policy changed")
        requirements = option.get("requirements")
        if not isinstance(requirements, list) or not requirements:
            raise ValueError(f"CCCC option {name} has no operational requirements")
    acknowledgements = specification.get("required_acknowledgements")
    if not isinstance(acknowledgements, dict):
        raise ValueError("CCCC acknowledgement requirements are missing")
    if set(acknowledgements) != {"all_options", *EXPECTED_OPTIONS}:
        raise ValueError("CCCC acknowledgement option set changed")
    if set(acknowledgements.get("all_options", [])) != EXPECTED_COMMON_ACKNOWLEDGEMENTS:
        raise ValueError("CCCC common acknowledgement set changed")
    for name, expected in EXPECTED_OPTION_ACKNOWLEDGEMENTS.items():
        if set(acknowledgements.get(name, [])) != expected:
            raise ValueError(f"CCCC {name} acknowledgement set changed")
    decision = _validate_decision(specification, options)
    floor = open_web_retention["minimum_candidate_tokens"]
    projected = open_web_retention["projected_post_filter_tokens"]
    return {
        "authorization": authorization,
        "decision": decision,
        "evidence": {
            "bounded_provenance_rows": cccc_row[1],
            "card_claims_per_document_license": True,
            "card_warns_residual_license_risk": True,
            "local_rows_with_claimed_license_field": 0,
            "local_rows_checked_for_claimed_license_field": 10_000,
            "source_level_domain_audit_documented": True,
            "upstream": reviewed_evidence,
        },
        "impact": {
            "compressed_source_bytes": cccc_snapshot["compressed_source_bytes"],
            "current_open_web_components": [EXPECTED_COMPONENT],
            "data_files": cccc_snapshot["data_file_count"],
            "exclude_replacement_candidate_tokens_required": floor,
            "final_open_web_target_tokens": open_web["target_tokens_at_80b"],
            "include_projected_candidate_tokens": projected,
            "include_projected_headroom_fraction": projected / floor - 1.0,
            "include_projected_headroom_tokens": projected - floor,
            "minimum_open_web_candidate_tokens": floor,
            "payload_bytes_downloaded": 0,
        },
        "options": options,
        "source": dict(source),
        "version": specification["version"],
    }


def build_cccc_owner_decision_report(
    contract_path: Path, *, repository_root: Path | None = None
) -> dict[str, Any]:
    """Load pinned inputs and bind the CCCC decision packet to exact hashes."""

    contract_path = contract_path.resolve()
    root = repository_root.resolve() if repository_root else contract_path.parents[1]
    specification = _read_json(contract_path)
    paths = {
        label: _resolve(root, raw_path, label)
        for label, raw_path in specification.get("inputs", {}).items()
    }
    expected_inputs = {
        "mixture",
        "provenance_audit",
        "retention_snapshot",
        "source_file_manifest",
        "source_inventory",
    }
    if set(paths) != expected_inputs:
        raise ValueError("CCCC decision input set changed")
    report = evaluate_cccc_owner_decision(
        specification,
        _read_json(paths["source_inventory"]),
        _read_json(paths["source_file_manifest"]),
        _read_json(paths["mixture"]),
        _read_json(paths["retention_snapshot"]),
        _read_json(paths["provenance_audit"]),
    )
    report["inputs"] = {
        "contract": {
            "path": contract_path.relative_to(root).as_posix(),
            "sha256": _sha256_file(contract_path),
        },
        **{
            label: {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256_file(path),
            }
            for label, path in sorted(paths.items())
        },
    }
    report["implementation"] = {
        path: _sha256_file(root / path) for path in IMPLEMENTATION_FILES
    }
    report["decision_fingerprint"] = _sha256_bytes(
        _canonical_json_bytes(
            {
                "authorization": report["authorization"],
                "decision": report["decision"],
                "evidence": report["evidence"],
                "implementation": report["implementation"],
                "impact": report["impact"],
                "inputs": report["inputs"],
                "options": report["options"],
                "source": report["source"],
            }
        )
    )
    return report


def verify_cccc_owner_decision_report(
    contract_path: Path,
    report_path: Path,
    manifest_path: Path,
    *,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Verify a sealed report hash and reproduce its exact contents."""

    manifest_parts = manifest_path.read_text(encoding="utf-8").strip().split()
    if len(manifest_parts) != 2 or manifest_parts[1] != report_path.name:
        raise ValueError("CCCC decision manifest is malformed")
    if manifest_parts[0] != _sha256_file(report_path):
        raise ValueError("CCCC decision report hash does not match its manifest")
    actual = _read_json(report_path)
    expected = build_cccc_owner_decision_report(
        contract_path, repository_root=repository_root
    )
    if actual != expected:
        raise ValueError("CCCC decision report does not reproduce")
    return actual
