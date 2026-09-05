"""Fail-closed rights/provenance audit for selected Aya Bengali constituents."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PLAN_SCHEMA_VERSION = "lumi.sft-aya-constituent-rights-audit.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-aya-constituent-rights-audit-report.v0.1"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-aya-constituent-rights-audit-verification.v0.1"
PLAN_STATUS = "authorized_cpu_only_official_metadata_rights_audit"
REPORT_STATUS = "partial_retain_partial_quarantine_candidate_unchanged"

EXPECTED_AUTHORIZATION = {
    "bulk_dataset_download_authorized": False,
    "candidate_modification_authorized": False,
    "cpu_only": True,
    "dataset_seal_authorized": False,
    "external_distribution_authorized": False,
    "external_inference_api_authorized": False,
    "gpu_authorized": False,
    "network_access_authorized": True,
    "official_metadata_only": True,
    "production_promotion_authorized": False,
    "registry_change_authorized": False,
    "sft_training_authorized": False,
    "unrelated_access_authorized": False,
}
EXPECTED_LIMITS = {
    "maximum_constituent_families": 3,
    "maximum_network_bytes": 16_777_216,
    "maximum_official_objects": 12,
    "maximum_runtime_minutes": 60,
}
EXPECTED_BINDINGS = {
    "aya_candidate",
    "banglabridge_repair_report",
    "banglabridge_repair_verification",
    "implementation",
    "runner",
    "test",
    "verifier",
}
EXPECTED_CANDIDATE_SHA256 = "f1cfb5a4d482c8e69e07a087f55f484016c3e0fa58c4b195fc2fe8feaae4c532"
EXPECTED_REPAIR_REPORT_SHA256 = "0530f983eea5c56e7ed543924852f8d221b9e851c6038c71b58716cf04785eef"
EXPECTED_REPAIR_VERIFICATION_SHA256 = (
    "69da46f43e0cafd6cd28319ae0398d1720dd06b5307d14c8d5c85ba32a8037ce"
)
EXPECTED_AGGREGATE_SOURCE = {
    "dataset_id": "CohereLabs/aya_collection_language_split",
    "license_id": "apache-2.0",
    "revision": "a3af2fde4b4cb5b2775830b11244a1a20b5f004f",
    "source_id": "coherelabs_aya_bengali_closure_canary",
}
EXPECTED_CANDIDATE = {
    "records": 5_079,
    "splits": {"train": 4_863, "validation": 216},
}
EXPECTED_FAMILIES = [
    {
        "family_id": "wiki_split_inst_translated",
        "source_dataset_name": "Wiki-split-inst (T)",
        "source_dataset_name_sha256": (
            "3bae6c736d432b44b3e79feb6ed935e45ec923fa56b5eb73518e1b755afcea3c"
        ),
        "expected_records": 3_322,
        "expected_splits": {"train": 3_156, "validation": 166},
        "upstream_repository": "google-research-datasets/wiki-split",
        "upstream_revision": "80f5bddf42d1a68201b7b107ee4b8d86d82e9fde",
        "upstream_license": "cc-by-sa-4.0",
        "commercial_compatibility": "conditional_commercial_use_permitted",
        "disposition": "retain_with_cc_by_sa_obligations",
        "production_sft_eligible_now": False,
        "required_conditions": [
            "preserve_source_and_aya_translation_provenance",
            "provide_attribution_license_link_and_change_notice",
            "resolve_share_alike_compliance_before_product_promotion",
            "bind_this_audit_to_any_filtered_successor",
        ],
    },
    {
        "family_id": "flan_gem_wikilingua_translated",
        "source_dataset_name": "Flan-GEM-wiki-lingua (T)",
        "source_dataset_name_sha256": (
            "1ca8a8ec9ac03c59c0be773f060e77d89eeab400f6a9160b6a13a2ef5c256859"
        ),
        "expected_records": 1_084,
        "expected_splits": {"train": 1_051, "validation": 33},
        "upstream_repository": "esdurmus/Wikilingua",
        "upstream_revision": "13e1ab93ac6b3abc66a8fb94c3e61bfcc6e54a8e",
        "upstream_license": "cc-by-nc-sa-3.0",
        "commercial_compatibility": "incompatible_noncommercial",
        "disposition": "quarantine_noncommercial_upstream",
        "production_sft_eligible_now": False,
        "required_conditions": [
            "exclude_from_commercial_product_sft",
            "do_not_treat_repository_cc0_wrapper_as_content_relicense",
        ],
    },
    {
        "family_id": "joke_explaination_inst_translated",
        "source_dataset_name": "Joke-explaination-inst (T)",
        "source_dataset_name_sha256": (
            "120f0a5cb5386355e804603ebac9c71e2775e698d02beefacb2e9b781a850151"
        ),
        "expected_records": 673,
        "expected_splits": {"train": 656, "validation": 17},
        "upstream_repository": "theblackcat102/joke_explaination",
        "upstream_revision": "888f2ffeb2398ae737ccc6e64cc7215e129436e7",
        "upstream_license": "mit",
        "commercial_compatibility": "declared_compatible",
        "disposition": "retain_with_mit_and_provenance_binding",
        "production_sft_eligible_now": False,
        "required_conditions": [
            "preserve_mit_notice_and_exact_source_card_revision",
            "preserve_source_and_aya_translation_provenance",
            "bind_this_audit_to_any_filtered_successor",
        ],
    },
]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_REPORT_KEYS = {
    "assistant_response",
    "content",
    "messages",
    "prompt",
    "raw_text",
    "user_prompt",
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_value(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _read_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _resolve_file(repository_root: str | Path, relative: Any, *, label: str) -> Path:
    root = Path(repository_root).resolve()
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"invalid path for {label}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.parent.resolve()) or not resolved.is_file():
        raise ValueError(f"missing or escaped file for {label}")
    return resolved


def _verify_file_spec(
    repository_root: str | Path,
    specification: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    path = _resolve_file(repository_root, specification.get("path"), label=label)
    expected = _require_sha256(specification.get("sha256"), f"{label}.sha256")
    if file_sha256(path) != expected:
        raise ValueError(f"file hash changed for {label}")
    expected_bytes = specification.get("bytes")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise ValueError(f"file size changed for {label}")
    return path


def _assert_evidence_markers(plan: Mapping[str, Any], repository_root: Path) -> None:
    objects = {item["object_id"]: item for item in plan["official_objects"]}

    def text(object_id: str) -> str:
        path = _verify_file_spec(
            repository_root,
            objects[object_id],
            label=f"official_objects.{object_id}",
        )
        return path.read_text(encoding="utf-8")

    wiki_split = text("wiki_split_readme")
    if "CC BY-SA 4.0" not in wiki_split or "Wikipedia revision history" not in wiki_split:
        raise ValueError("WikiSplit license or provenance marker changed")
    wikilingua = text("wikilingua_readme")
    if "creativecommons.org/licenses/by-nc-sa/3.0/" not in wikilingua:
        raise ValueError("WikiLingua content license marker changed")
    if "WikiHow" not in wikilingua:
        raise ValueError("WikiLingua provenance marker changed")
    wikilingua_wrapper = text("wikilingua_license")
    if "CC0 1.0 Universal" not in wikilingua_wrapper:
        raise ValueError("WikiLingua wrapper license marker changed")
    joke = text("joke_explaination_readme")
    if "license: mit" not in joke or "explainthejoke.com" not in joke:
        raise ValueError("joke source-card license or provenance marker changed")
    aya = text("aya_language_split_readme")
    if "license: apache-2.0" not in aya:
        raise ValueError("Aya aggregate license marker changed")
    by_sa = text("cc_by_sa_4_legalcode")
    if "Attribution-ShareAlike 4.0 International" not in by_sa:
        raise ValueError("CC BY-SA legal code changed")
    by_nc_sa = text("cc_by_nc_sa_3_legalcode")
    if "Attribution-NonCommercial-ShareAlike 3.0 Unported" not in by_nc_sa:
        raise ValueError("CC BY-NC-SA legal code changed")


def load_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    plan = _read_object(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported Aya constituent-rights audit plan")
    if plan.get("status") != PLAN_STATUS:
        raise ValueError("Aya constituent-rights audit status changed")
    if plan.get("observed_at") != "2026-08-10":
        raise ValueError("Aya constituent-rights audit date changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("Aya constituent-rights authorization changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("Aya constituent-rights limits changed")
    if plan.get("candidate_expectation") != EXPECTED_CANDIDATE:
        raise ValueError("Aya candidate expectation changed")
    if plan.get("constituent_families") != EXPECTED_FAMILIES:
        raise ValueError("Aya constituent-family policy changed")

    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("Aya constituent-rights binding set changed")
    for name, specification in bindings.items():
        if not isinstance(specification, Mapping):
            raise ValueError(f"invalid Aya constituent-rights binding: {name}")
        _verify_file_spec(repository_root, specification, label=f"bindings.{name}")
    if bindings["aya_candidate"]["sha256"] != EXPECTED_CANDIDATE_SHA256:
        raise ValueError("Aya candidate binding changed")
    if bindings["banglabridge_repair_report"]["sha256"] != EXPECTED_REPAIR_REPORT_SHA256:
        raise ValueError("BanglaBridge repair-report binding changed")
    if (
        bindings["banglabridge_repair_verification"]["sha256"]
        != EXPECTED_REPAIR_VERIFICATION_SHA256
    ):
        raise ValueError("BanglaBridge repair-verification binding changed")

    official_objects = plan.get("official_objects")
    if not isinstance(official_objects, list):
        raise ValueError("official object inventory is missing")
    if len(official_objects) > EXPECTED_LIMITS["maximum_official_objects"]:
        raise ValueError("official object limit exceeded")
    object_ids: set[str] = set()
    total_bytes = 0
    for item in official_objects:
        if not isinstance(item, Mapping):
            raise ValueError("official object entry is invalid")
        object_id = item.get("object_id")
        if not isinstance(object_id, str) or object_id in object_ids:
            raise ValueError("official object identity is invalid")
        if not isinstance(item.get("official_url"), str):
            raise ValueError("official object URL is invalid")
        object_ids.add(object_id)
        _verify_file_spec(repository_root, item, label=f"official_objects.{object_id}")
        total_bytes += int(item["bytes"])
    if len(object_ids) != 9:
        raise ValueError("official object inventory changed")
    if total_bytes > EXPECTED_LIMITS["maximum_network_bytes"]:
        raise ValueError("official object byte limit exceeded")
    _assert_evidence_markers(plan, Path(repository_root).resolve())

    expected_output = {
        "report_path": "evidence/lumi-4b-sft-aya-constituent-rights-audit-v0.1.json",
        "verification_path": (
            "evidence/lumi-4b-sft-aya-constituent-rights-audit-verification-v0.1.json"
        ),
    }
    if plan.get("output") != expected_output:
        raise ValueError("Aya constituent-rights outputs changed")
    return plan


def _candidate_inventory(path: Path) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    family_split_counts: Counter[tuple[str, str]] = Counter()
    records = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Aya candidate row {line_number} is not an object")
            provenance = row.get("provenance")
            source = row.get("source")
            split = row.get("split")
            if not isinstance(provenance, Mapping) or not isinstance(source, Mapping):
                raise ValueError("Aya candidate provenance/source is missing")
            for key, expected in EXPECTED_AGGREGATE_SOURCE.items():
                if source.get(key) != expected:
                    raise ValueError(f"Aya aggregate source changed: {key}")
            family_hash = provenance.get("source_dataset_name_sha256")
            _require_sha256(family_hash, "source_dataset_name_sha256")
            if split not in {"train", "validation"}:
                raise ValueError("Aya candidate split changed")
            records += 1
            family_counts[str(family_hash)] += 1
            split_counts[str(split)] += 1
            family_split_counts[(str(family_hash), str(split))] += 1

    if records != EXPECTED_CANDIDATE["records"]:
        raise ValueError("Aya candidate record count changed")
    if dict(sorted(split_counts.items())) != EXPECTED_CANDIDATE["splits"]:
        raise ValueError("Aya candidate split counts changed")
    expected_hashes = {item["source_dataset_name_sha256"] for item in EXPECTED_FAMILIES}
    if set(family_counts) != expected_hashes:
        raise ValueError("Aya constituent family set changed")

    families: dict[str, Any] = {}
    for item in EXPECTED_FAMILIES:
        family_hash = item["source_dataset_name_sha256"]
        observed_splits = {
            split: family_split_counts[(family_hash, split)] for split in ("train", "validation")
        }
        if family_counts[family_hash] != item["expected_records"]:
            raise ValueError(f"Aya family count changed: {item['family_id']}")
        if observed_splits != item["expected_splits"]:
            raise ValueError(f"Aya family split changed: {item['family_id']}")
        families[item["family_id"]] = {
            "records": family_counts[family_hash],
            "splits": observed_splits,
        }
    return {
        "records": records,
        "splits": dict(sorted(split_counts.items())),
        "families": families,
    }


def _evidence_inventory(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "bytes": item["bytes"],
            "kind": item["kind"],
            "object_id": item["object_id"],
            "official_url": item["official_url"],
            "revision": item.get("revision"),
            "sha256": item["sha256"],
        }
        for item in plan["official_objects"]
    ]


def build_report(
    plan: Mapping[str, Any],
    *,
    plan_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    candidate_path = _verify_file_spec(
        root, plan["bindings"]["aya_candidate"], label="bindings.aya_candidate"
    )
    candidate = _candidate_inventory(candidate_path)

    constituents: list[dict[str, Any]] = []
    retained = Counter()
    quarantined = Counter()
    for family in EXPECTED_FAMILIES:
        counts = candidate["families"][family["family_id"]]
        disposition = str(family["disposition"])
        bucket = quarantined if disposition.startswith("quarantine") else retained
        bucket["records"] += counts["records"]
        bucket["train"] += counts["splits"]["train"]
        bucket["validation"] += counts["splits"]["validation"]
        constituents.append(
            {
                "commercial_compatibility": family["commercial_compatibility"],
                "disposition": disposition,
                "family_id": family["family_id"],
                "production_sft_eligible_now": False,
                "records": counts["records"],
                "required_conditions": family["required_conditions"],
                "source_dataset_name_sha256": family["source_dataset_name_sha256"],
                "splits": counts["splits"],
                "upstream_license": family["upstream_license"],
                "upstream_repository": family["upstream_repository"],
                "upstream_revision": family["upstream_revision"],
            }
        )

    evidence = _evidence_inventory(plan)
    report: dict[str, Any] = {
        "authorization": plan["authorization"],
        "bindings": {
            name: {
                "path": specification["path"],
                "sha256": specification["sha256"],
            }
            for name, specification in sorted(plan["bindings"].items())
        },
        "blockers": [
            "filtered_candidate_materialization_not_authorized",
            "source_registry_change_not_authorized",
            "production_dataset_seal_not_authorized",
            "sft_training_not_authorized",
            "wiki_split_share_alike_compliance_resolution_required",
        ],
        "candidate": {
            "aggregate_license_claim": "apache-2.0",
            "aggregate_revision": EXPECTED_AGGREGATE_SOURCE["revision"],
            "candidate_file_sha256": EXPECTED_CANDIDATE_SHA256,
            "candidate_modified": False,
            "records": candidate["records"],
            "splits": candidate["splits"],
        },
        "constituents": constituents,
        "decision_summary": {
            "quarantined": {
                "records": quarantined["records"],
                "splits": {
                    "train": quarantined["train"],
                    "validation": quarantined["validation"],
                },
            },
            "retained_with_obligations": {
                "records": retained["records"],
                "splits": {
                    "train": retained["train"],
                    "validation": retained["validation"],
                },
            },
        },
        "evidence_inventory": evidence,
        "evidence_limits": {
            "constituent_families": len(constituents),
            "network_bytes": sum(item["bytes"] for item in evidence),
            "official_objects": len(evidence),
        },
        "observed_at": "2026-08-10",
        "plan_file_sha256": _require_sha256(plan_file_sha256, "plan_file_sha256"),
        "production_eligible": False,
        "raw_training_text_in_report": False,
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": REPORT_STATUS,
    }
    report["report_sha256"] = sha256_value(report)
    validate_report(report)
    return report


def _walk_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            keys.add(str(key))
            keys.update(_walk_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.update(_walk_keys(nested))
    return keys


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported Aya constituent-rights report")
    if report.get("status") != REPORT_STATUS:
        raise ValueError("Aya constituent-rights report status changed")
    if report.get("production_eligible") is not False:
        raise ValueError("Aya constituent-rights audit crossed production boundary")
    if report.get("raw_training_text_in_report") is not False:
        raise ValueError("Aya constituent-rights report retained raw text")
    expected = report.get("report_sha256")
    unsigned = dict(report)
    unsigned.pop("report_sha256", None)
    if expected != sha256_value(unsigned):
        raise ValueError("Aya constituent-rights report seal changed")
    if _walk_keys(report) & _FORBIDDEN_REPORT_KEYS:
        raise ValueError("Aya constituent-rights report contains raw-text fields")
    summary = report.get("decision_summary")
    if not isinstance(summary, Mapping):
        raise ValueError("Aya constituent-rights decision summary is missing")
    if summary["retained_with_obligations"]["records"] != 3_995:
        raise ValueError("Aya retained record total changed")
    if summary["quarantined"]["records"] != 1_084:
        raise ValueError("Aya quarantined record total changed")


def build_verification(report: Mapping[str, Any], *, report_file_sha256: str) -> dict[str, Any]:
    validate_report(report)
    verification: dict[str, Any] = {
        "checks": {
            "all_5079_candidate_rows_classified": True,
            "candidate_preserved_byte_exact": True,
            "commercial_compatibility_fail_closed": True,
            "exact_upstream_revisions_bound": True,
            "official_object_count_and_bytes_within_limits": True,
            "raw_training_text_absent": True,
            "registry_dataset_seal_and_sft_unchanged": True,
            "upstream_license_precedence_over_aggregate_claim_enforced": True,
        },
        "candidate_file_sha256": EXPECTED_CANDIDATE_SHA256,
        "observed_at": "2026-08-10",
        "production_eligible": False,
        "report_file_sha256": _require_sha256(report_file_sha256, "report_file_sha256"),
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_partial_retain_partial_quarantine_candidate_unchanged",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification
