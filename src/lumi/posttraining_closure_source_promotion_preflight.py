"""Fail-closed registry-promotion preflight for the reviewed SFT closure canary.

This module deliberately does not materialize a source-registry successor.  It
replays the immutable local evidence, checks the source-specific rights
contract, and emits a raw-text-free blocker/repair report when any source is
not yet admissible.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .data_quality import sha256_file

PLAN_SCHEMA_VERSION = "lumi.sft-closure-source-promotion-preflight.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-closure-source-promotion-preflight-report.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-closure-source-promotion-preflight-verification.v0.1"
)
PLAN_STATUS = "local_cpu_only_registry_successor_preflight_not_materialized"
REPORT_STATUS = "blocked_repair_required_registry_successor_not_projected"
OWNER_REQUEST_TEXT = "ok next step shuru kor"

EXPECTED_AUTHORIZATION = {
    "candidate_registry_change_authorized": False,
    "cpu_only": True,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_promotion_authorized": False,
    "production_sft_authorized": False,
    "purpose": "closure_candidate_registry_admissibility_preflight_only",
    "source_registry_promotion_authorized": False,
}
EXPECTED_LIMITS = {
    "candidate_records": 21_183,
    "candidate_sources": 4,
    "maximum_report_bytes": 1_048_576,
    "maximum_runtime_minutes": 30,
    "maximum_verification_bytes": 1_048_576,
}
EXPECTED_BINDINGS = {
    "aya_candidate",
    "aya_readme",
    "banglabridge_candidate",
    "banglabridge_readme",
    "banglabridge_source_rows",
    "banglatlit_candidate",
    "curation_report",
    "curation_verification",
    "implementation",
    "nemotron_candidate",
    "owner_review_receipt",
    "owner_review_sample",
    "owner_review_verification",
    "predecessor_registry",
    "private_manifest",
    "rights_preflight_report",
    "rights_preflight_verification",
    "runner",
    "test",
    "verifier",
}

EXPECTED_PREDECESSOR_SHA256 = (
    "de81a51f0288b4851ff5ea4ca0598ddcf4936b2bf2c4127ab2b1c0d6deb1a4de"
)
EXPECTED_PREDECESSOR_SOURCES = 16
EXPECTED_CURRENT_TRAIN_TOKENS = 7_256_670
EXPECTED_CANDIDATE_TRAIN_TOKENS = 2_977_454
EXPECTED_CANDIDATE_VALIDATION_TOKENS = 142_498
TEN_MILLION_FLOOR = 10_000_000
SOURCE_FAMILY_CAP = 0.35

SOURCE_SPECS = {
    "nvidia_nemotron_post_training_tool_canary": {
        "candidate_binding": "nemotron_candidate",
        "rights_source_id": "nvidia_nemotron_post_training_v1",
        "family": "nvidia_family",
        "expected_records": 1_967,
        "expected_train_tokens": 1_299_965,
        "expected_validation_tokens": 68_393,
        "ceiling": 1_550_000,
    },
    "coherelabs_aya_bengali_closure_canary": {
        "candidate_binding": "aya_candidate",
        "rights_source_id": "coherelabs_aya_collection_language_split_bengali",
        "family": "coherelabs_aya_family",
        "expected_records": 5_079,
        "expected_train_tokens": 1_125_733,
        "expected_validation_tokens": 53_321,
        "ceiling": 1_300_000,
    },
    "banglatlit_auxiliary_closure_canary": {
        "candidate_binding": "banglatlit_candidate",
        "rights_source_id": "aplycaebous_banglatlit",
        "family": "banglatlit_family",
        "expected_records": 13_576,
        "expected_train_tokens": 300_000,
        "expected_validation_tokens": 15_789,
        "ceiling": 300_000,
    },
    "banglabridge_instructions_closure_canary": {
        "candidate_binding": "banglabridge_candidate",
        "rights_source_id": "subhajitmahata84_banglabridge_instructions",
        "family": "banglabridge_family",
        "expected_records": 561,
        "expected_train_tokens": 251_756,
        "expected_validation_tokens": 4_995,
        "ceiling": 50_000,
    },
}

AYA_CONSTITUENTS = {
    "Wiki-split-inst (T)": "3bae6c736d432b44b3e79feb6ed935e45ec923fa56b5eb73518e1b755afcea3c",
    "Flan-GEM-wiki-lingua (T)": (
        "1ca8a8ec9ac03c59c0be773f060e77d89eeab400f6a9160b6a13a2ef5c256859"
    ),
    "Joke-explaination-inst (T)": (
        "120f0a5cb5386355e804603ebac9c71e2775e698d02beefacb2e9b781a850151"
    ),
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REGION_SPECIFIC_EMERGENCY = re.compile(r"(?<!\d)(?:108|1930|14416)(?!\d)")
_BRIDGE_REQUIRED_PROVENANCE = {
    "source_original_instruction_sha256",
    "source_original_output_sha256",
    "source_adapted_instruction_sha256",
    "source_adapted_output_sha256",
}


def sha256_value(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(json_file_bytes(value))
    temporary.replace(destination)


def _read_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(row)
    return rows


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def resolve_binding(
    repository_root: str | Path,
    specification: Mapping[str, Any],
    *,
    name: str,
) -> Path:
    root = Path(repository_root).resolve()
    relative = specification.get("path")
    expected = _require_sha256(specification.get("sha256"), f"bindings.{name}.sha256")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"invalid closure promotion-preflight binding: {name}")
    resolved = (root / relative).resolve()
    workspace = root.parent.resolve()
    if not resolved.is_relative_to(workspace) or not resolved.is_file():
        raise ValueError(f"closure promotion-preflight binding missing: {name}")
    if sha256_file(resolved) != expected:
        raise ValueError(f"closure promotion-preflight binding changed: {name}")
    return resolved


def load_preflight_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    plan = _read_object(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported closure source-promotion preflight")
    if plan.get("status") != PLAN_STATUS:
        raise ValueError("closure source-promotion preflight status changed")
    if plan.get("observed_at") != "2026-08-09":
        raise ValueError("closure source-promotion preflight date changed")
    if plan.get("owner_request_text") != OWNER_REQUEST_TEXT:
        raise ValueError("closure source-promotion owner request changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("closure source-promotion authorization changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("closure source-promotion limits changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("closure source-promotion bindings changed")
    for name, specification in bindings.items():
        if not isinstance(specification, Mapping):
            raise ValueError(f"invalid closure source-promotion binding: {name}")
        resolve_binding(repository_root, specification, name=name)
    if bindings["predecessor_registry"]["sha256"] != EXPECTED_PREDECESSOR_SHA256:
        raise ValueError("closure predecessor registry identity changed")
    if plan.get("output") != {
        "preflight_report_path": (
            "evidence/lumi-4b-sft-closure-source-promotion-preflight-v0.1.json"
        ),
        "verification_path": (
            "evidence/"
            "lumi-4b-sft-closure-source-promotion-preflight-verification-v0.1.json"
        ),
    }:
        raise ValueError("closure source-promotion outputs changed")
    return plan


def _validate_internal_seal(value: Mapping[str, Any], field: str, label: str) -> None:
    expected = value.get(field)
    unsigned = {key: item for key, item in value.items() if key != field}
    if expected != sha256_value(unsigned):
        raise ValueError(f"{label} internal seal changed")


def _candidate_inventory(
    rows: list[dict[str, Any]],
    *,
    expected_source_id: str,
) -> dict[str, Any]:
    record_ids: set[str] = set()
    split_counts: Counter[str] = Counter()
    provenance_key_sets: set[tuple[str, ...]] = set()
    task_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    for row in rows:
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"{expected_source_id} record_id is invalid")
        if record_id in record_ids:
            raise ValueError(f"{expected_source_id} record_id is duplicated")
        record_ids.add(record_id)
        source = row.get("source")
        if not isinstance(source, Mapping) or source.get("source_id") != expected_source_id:
            raise ValueError(f"{expected_source_id} source binding changed")
        split = row.get("split")
        if split not in {"train", "validation"}:
            raise ValueError(f"{expected_source_id} split changed")
        split_counts[str(split)] += 1
        provenance = row.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(f"{expected_source_id} provenance is missing")
        provenance_key_sets.add(tuple(sorted(str(key) for key in provenance)))
        task_counts[str(row.get("task_category"))] += 1
        language_counts[str(row.get("language"))] += 1
    return {
        "languages": dict(sorted(language_counts.items())),
        "provenance_key_sets": [list(keys) for keys in sorted(provenance_key_sets)],
        "records": len(rows),
        "splits": dict(sorted(split_counts.items())),
        "tasks": dict(sorted(task_counts.items())),
        "unique_record_ids": len(record_ids),
    }


def _rights_sources(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    selected = report.get("selected_sources")
    if not isinstance(selected, list):
        raise ValueError("rights preflight selected sources are missing")
    result: dict[str, Mapping[str, Any]] = {}
    for source in selected:
        if not isinstance(source, Mapping) or not isinstance(source.get("source_id"), str):
            raise ValueError("rights preflight source is invalid")
        result[str(source["source_id"])] = source
    return result


def _verify_owner_review(
    receipt: Mapping[str, Any],
    verification: Mapping[str, Any],
    sample: list[dict[str, Any]],
) -> dict[str, Any]:
    _validate_internal_seal(receipt, "receipt_sha256", "owner-review receipt")
    _validate_internal_seal(
        verification, "verification_sha256", "owner-review verification"
    )
    if (
        receipt.get("all_sampled_rows_accepted") is not True
        or receipt.get("accepted_records") != 32
        or receipt.get("rejected_records") != 0
        or receipt.get("review_scope") != "sample"
        or receipt.get("production_eligible") is not False
        or verification.get("checks", {}).get("all_sampled_rows_accepted") is not True
    ):
        raise ValueError("owner-review outcome changed")
    if len(sample) != 32 or sha256_value(sample) != receipt.get("sample_sha256"):
        raise ValueError("owner-review sample identity changed")
    by_source: Counter[str] = Counter()
    for row in sample:
        skill_id = row.get("skill_id")
        if not isinstance(skill_id, str) or ":" not in skill_id:
            raise ValueError("owner-review sample skill binding changed")
        by_source[skill_id.split(":", 1)[0]] += 1
    return {
        "accepted_records": 32,
        "by_source": dict(sorted(by_source.items())),
        "claim": receipt.get("owner_review_claim"),
        "production_eligible": False,
        "review_scope": "sample",
    }


def _verify_banglabridge_source_mapping(
    candidate_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    exact_matches = 0
    source_rows_with_original_fields = 0
    adapted_rows_distinct_from_original = 0
    candidate_rows_with_region_numbers = 0
    candidate_rows_with_required_provenance = 0
    for source_row in source_rows:
        if all(
            isinstance(source_row.get(field), str)
            for field in ("instruction", "output", "orig_instruction", "orig_output")
        ):
            source_rows_with_original_fields += 1
        if (
            source_row.get("instruction") != source_row.get("orig_instruction")
            or source_row.get("output") != source_row.get("orig_output")
        ):
            adapted_rows_distinct_from_original += 1
    for row in candidate_rows:
        document_id = row.get("source", {}).get("document_id")
        if not isinstance(document_id, str) or not document_id.startswith("row-"):
            raise ValueError("BanglaBridge document_id changed")
        try:
            source_index = int(document_id.removeprefix("row-"))
        except ValueError as error:
            raise ValueError("BanglaBridge document_id is invalid") from error
        if source_index < 0 or source_index >= len(source_rows):
            raise ValueError("BanglaBridge document_id escaped source rows")
        source_row = source_rows[source_index]
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 2:
            raise ValueError("BanglaBridge message schema changed")
        if (
            messages[0] != {"role": "user", "content": str(source_row["instruction"]).strip()}
            or messages[1]
            != {"role": "assistant", "content": str(source_row["output"]).strip()}
        ):
            raise ValueError("BanglaBridge candidate no longer matches adapted source")
        exact_matches += 1
        combined = "\n".join(str(message.get("content", "")) for message in messages)
        if _REGION_SPECIFIC_EMERGENCY.search(combined):
            candidate_rows_with_region_numbers += 1
        provenance = row.get("provenance", {})
        if isinstance(provenance, Mapping) and _BRIDGE_REQUIRED_PROVENANCE.issubset(
            provenance
        ):
            candidate_rows_with_required_provenance += 1
    return {
        "adapted_rows_distinct_from_original": adapted_rows_distinct_from_original,
        "candidate_rows_exactly_mapped": exact_matches,
        "candidate_rows_with_region_specific_emergency_numbers": (
            candidate_rows_with_region_numbers
        ),
        "candidate_rows_with_required_original_and_adapted_hash_provenance": (
            candidate_rows_with_required_provenance
        ),
        "source_rows": len(source_rows),
        "source_rows_with_original_fields": source_rows_with_original_fields,
    }


def _aya_constituent_inventory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected_hashes = set(AYA_CONSTITUENTS.values())
    counts: Counter[str] = Counter()
    rows_with_upstream_license_binding = 0
    for row in rows:
        provenance = row.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("Aya provenance is missing")
        source_hash = provenance.get("source_dataset_name_sha256")
        if source_hash not in expected_hashes:
            raise ValueError("Aya selected constituent identity changed")
        counts[str(source_hash)] += 1
        if all(
            field in provenance
            for field in (
                "constituent_license_id",
                "constituent_license_evidence_sha256",
                "constituent_source_revision",
            )
        ):
            rows_with_upstream_license_binding += 1
    return {
        "constituents": [
            {
                "dataset_name_sha256": digest,
                "records": counts[digest],
                "upstream_rights_evidence_bound": False,
            }
            for digest in sorted(expected_hashes)
        ],
        "rows": len(rows),
        "rows_with_constituent_upstream_license_binding": (
            rows_with_upstream_license_binding
        ),
    }


def build_preflight_report(
    plan: Mapping[str, Any],
    *,
    plan_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    bindings = plan["bindings"]

    registry = _read_object(
        resolve_binding(root, bindings["predecessor_registry"], name="predecessor_registry")
    )
    sources = registry.get("sources")
    if not isinstance(sources, list) or len(sources) != EXPECTED_PREDECESSOR_SOURCES:
        raise ValueError("predecessor registry source count changed")
    source_ids = [source.get("source_id") for source in sources if isinstance(source, Mapping)]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("predecessor registry source IDs are not unique")
    if any(source_id in SOURCE_SPECS for source_id in source_ids):
        raise ValueError("closure candidate source already exists in predecessor registry")

    rights = _read_object(
        resolve_binding(root, bindings["rights_preflight_report"], name="rights_preflight_report")
    )
    rights_verification = _read_object(
        resolve_binding(
            root,
            bindings["rights_preflight_verification"],
            name="rights_preflight_verification",
        )
    )
    _validate_internal_seal(rights, "report_sha256", "rights preflight")
    _validate_internal_seal(
        rights_verification, "verification_sha256", "rights preflight verification"
    )
    rights_sources = _rights_sources(rights)

    curation = _read_object(
        resolve_binding(root, bindings["curation_report"], name="curation_report")
    )
    curation_verification = _read_object(
        resolve_binding(
            root,
            bindings["curation_verification"],
            name="curation_verification",
        )
    )
    _validate_internal_seal(curation, "report_sha256", "curation report")
    _validate_internal_seal(
        curation_verification,
        "verification_sha256",
        "curation verification",
    )
    candidate_summary = curation.get("candidate")
    if not isinstance(candidate_summary, Mapping):
        raise ValueError("curation candidate summary is missing")
    if (
        candidate_summary.get("records") != EXPECTED_LIMITS["candidate_records"]
        or candidate_summary.get("train_supervised_tokens")
        != EXPECTED_CANDIDATE_TRAIN_TOKENS
        or candidate_summary.get("validation_supervised_tokens")
        != EXPECTED_CANDIDATE_VALIDATION_TOKENS
    ):
        raise ValueError("curation candidate totals changed")
    if curation_verification.get("checks", {}).get("candidate_records_replayed") != 21_183:
        raise ValueError("curation replay evidence changed")

    manifest = _read_object(
        resolve_binding(root, bindings["private_manifest"], name="private_manifest")
    )
    _validate_internal_seal(manifest, "manifest_sha256", "private manifest")
    manifest_entries = manifest.get("candidate_files")
    if not isinstance(manifest_entries, list) or len(manifest_entries) != 4:
        raise ValueError("private manifest candidate inventory changed")
    manifest_by_source = {
        entry.get("source_id"): entry
        for entry in manifest_entries
        if isinstance(entry, Mapping)
    }
    if set(manifest_by_source) != set(SOURCE_SPECS):
        raise ValueError("private manifest source set changed")

    candidate_rows: dict[str, list[dict[str, Any]]] = {}
    candidate_inventory: dict[str, Any] = {}
    rights_compliance: dict[str, Any] = {}
    for source_id, spec in SOURCE_SPECS.items():
        candidate_path = resolve_binding(
            root, bindings[str(spec["candidate_binding"])], name=str(spec["candidate_binding"])
        )
        rows = _read_jsonl(candidate_path)
        candidate_rows[source_id] = rows
        entry = manifest_by_source[source_id]
        if (
            entry.get("records") != spec["expected_records"]
            or entry.get("sha256") != bindings[str(spec["candidate_binding"])]["sha256"]
            or entry.get("splits", {}).get("train", {}).get("supervised_tokens")
            != spec["expected_train_tokens"]
            or entry.get("splits", {}).get("validation", {}).get("supervised_tokens")
            != spec["expected_validation_tokens"]
        ):
            raise ValueError(f"{source_id} manifest metrics changed")
        inventory = _candidate_inventory(rows, expected_source_id=source_id)
        if inventory["records"] != spec["expected_records"]:
            raise ValueError(f"{source_id} candidate record count changed")
        candidate_inventory[source_id] = inventory
        rights_source = rights_sources.get(str(spec["rights_source_id"]))
        if not isinstance(rights_source, Mapping):
            raise ValueError(f"{source_id} rights source is missing")
        if rights_source.get("target_retained_train_supervised_tokens_ceiling") != spec[
            "ceiling"
        ]:
            raise ValueError(f"{source_id} rights ceiling changed")
        actual = int(spec["expected_train_tokens"])
        ceiling = int(spec["ceiling"])
        rights_compliance[source_id] = {
            "actual_train_supervised_tokens": actual,
            "ceiling_train_supervised_tokens": ceiling,
            "excess_train_supervised_tokens": max(actual - ceiling, 0),
            "required_filters": rights_source.get("required_filters"),
            "rights_source_id": spec["rights_source_id"],
            "within_rights_ceiling": actual <= ceiling,
        }

    owner_receipt = _read_object(
        resolve_binding(root, bindings["owner_review_receipt"], name="owner_review_receipt")
    )
    owner_verification = _read_object(
        resolve_binding(
            root,
            bindings["owner_review_verification"],
            name="owner_review_verification",
        )
    )
    owner_sample = json.loads(
        resolve_binding(root, bindings["owner_review_sample"], name="owner_review_sample")
        .read_text(encoding="utf-8")
    )
    if not isinstance(owner_sample, list):
        raise ValueError("owner-review sample root changed")
    owner_review = _verify_owner_review(owner_receipt, owner_verification, owner_sample)

    bridge_source_rows = _read_jsonl(
        resolve_binding(
            root,
            bindings["banglabridge_source_rows"],
            name="banglabridge_source_rows",
        )
    )
    bridge_provenance = _verify_banglabridge_source_mapping(
        candidate_rows["banglabridge_instructions_closure_canary"],
        bridge_source_rows,
    )
    aya_provenance = _aya_constituent_inventory(
        candidate_rows["coherelabs_aya_bengali_closure_canary"]
    )

    bridge_ceiling = int(
        SOURCE_SPECS["banglabridge_instructions_closure_canary"]["ceiling"]
    )
    bridge_actual = int(
        SOURCE_SPECS["banglabridge_instructions_closure_canary"][
            "expected_train_tokens"
        ]
    )
    trim = bridge_actual - bridge_ceiling
    projected_candidate_train = EXPECTED_CANDIDATE_TRAIN_TOKENS - trim
    projected_total = EXPECTED_CURRENT_TRAIN_TOKENS + projected_candidate_train
    family_tokens = {
        "banglabridge_family": bridge_ceiling,
        "banglatlit_family": 300_000,
        "coherelabs_aya_family": 513_379 + 1_125_733,
        "nvidia_family": 2_100_077 + 1_299_965,
    }
    family_shares = {
        family: round(tokens / projected_total, 8)
        for family, tokens in sorted(family_tokens.items())
    }
    if max(family_shares.values()) > SOURCE_FAMILY_CAP:
        raise ValueError("safe repair projection exceeds source-family cap")

    blockers = [
        {
            "actual_train_supervised_tokens": bridge_actual,
            "ceiling_train_supervised_tokens": bridge_ceiling,
            "code": "banglabridge_train_token_ceiling_exceeded",
            "excess_train_supervised_tokens": trim,
        },
        {
            "candidate_rows": bridge_provenance["candidate_rows_exactly_mapped"],
            "code": "banglabridge_original_and_adapted_provenance_hashes_missing",
            "rows_with_required_provenance": bridge_provenance[
                "candidate_rows_with_required_original_and_adapted_hash_provenance"
            ],
        },
        {
            "code": "aya_constituent_upstream_rights_evidence_not_bound",
            "constituent_count": len(AYA_CONSTITUENTS),
            "rows_with_constituent_upstream_license_binding": aya_provenance[
                "rows_with_constituent_upstream_license_binding"
            ],
        },
    ]

    report: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "blockers": blockers,
        "candidate": {
            "by_source": {
                source_id: {
                    "inventory": candidate_inventory[source_id],
                    "train_supervised_tokens": spec["expected_train_tokens"],
                    "validation_supervised_tokens": spec["expected_validation_tokens"],
                }
                for source_id, spec in SOURCE_SPECS.items()
            },
            "records": EXPECTED_LIMITS["candidate_records"],
            "train_supervised_tokens": EXPECTED_CANDIDATE_TRAIN_TOKENS,
            "validation_supervised_tokens": EXPECTED_CANDIDATE_VALIDATION_TOKENS,
        },
        "checks": {
            "all_candidate_files_rehashed": True,
            "all_candidate_records_replayed": True,
            "banglabridge_candidate_maps_to_bound_source_rows": True,
            "candidate_registry_change_executed": False,
            "dataset_seal_executed": False,
            "owner_review_sample_verified": True,
            "predecessor_registry_changed": False,
            "raw_training_text_in_report": False,
            "source_registry_successor_projected": False,
            "sft_training_started": False,
        },
        "observed_at": plan["observed_at"],
        "owner_review": owner_review,
        "plan_file_sha256": _require_sha256(plan_file_sha256, "plan_file_sha256"),
        "predecessor_registry": {
            "file_sha256": bindings["predecessor_registry"]["sha256"],
            "source_count": EXPECTED_PREDECESSOR_SOURCES,
            "unchanged": True,
        },
        "production_eligible": False,
        "provenance_audit": {
            "aya": aya_provenance,
            "banglabridge": bridge_provenance,
        },
        "repair_projection": {
            "family_share_cap": SOURCE_FAMILY_CAP,
            "family_shares_after_banglabridge_ceiling_trim": family_shares,
            "projected_candidate_train_supervised_tokens": projected_candidate_train,
            "projected_total_train_supervised_tokens": projected_total,
            "required_actions": [
                (
                    "trim_banglabridge_train_to_at_most_50000_supervised_tokens_"
                    "while_preserving_owner_reviewed_rows_byte_exact"
                ),
                (
                    "bind_original_and_adapted_instruction_and_output_hashes_per_"
                    "retained_banglabridge_row"
                ),
                (
                    "bind_official_upstream_license_revision_and_evidence_for_each_"
                    "selected_aya_constituent_or_quarantine_it"
                ),
                (
                    "rematerialize_candidate_manifest_and_independent_verification_"
                    "before_any_registry_successor"
                ),
                (
                    "repeat_owner_review_only_if_owner_visible_content_or_review_"
                    "sample_membership_changes"
                ),
            ],
            "ten_million_floor": TEN_MILLION_FLOOR,
            "ten_million_floor_preserved": projected_total >= TEN_MILLION_FLOOR,
        },
        "result_preservation": {
            "candidate_files_modified": False,
            "dataset_sealed": False,
            "gpu_used": False,
            "network_used": False,
            "predecessor_registry_modified": False,
            "sft_training_started": False,
        },
        "rights_compliance": rights_compliance,
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": REPORT_STATUS,
    }
    report["report_sha256"] = sha256_value(report)
    validate_report(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported closure source-promotion preflight report")
    if report.get("status") != REPORT_STATUS:
        raise ValueError("closure source-promotion preflight did not fail closed")
    _validate_internal_seal(report, "report_sha256", "closure promotion preflight")
    blockers = report.get("blockers")
    if not isinstance(blockers, list):
        raise ValueError("closure source-promotion blockers are missing")
    codes = {blocker.get("code") for blocker in blockers if isinstance(blocker, Mapping)}
    if codes != {
        "aya_constituent_upstream_rights_evidence_not_bound",
        "banglabridge_original_and_adapted_provenance_hashes_missing",
        "banglabridge_train_token_ceiling_exceeded",
    }:
        raise ValueError("closure source-promotion blocker set changed")
    checks = report.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("closure source-promotion checks are missing")
    if checks.get("source_registry_successor_projected") is not False:
        raise ValueError("invalid registry successor was projected")
    if checks.get("candidate_registry_change_executed") is not False:
        raise ValueError("candidate registry changed during preflight")
    repair = report.get("repair_projection")
    if (
        not isinstance(repair, Mapping)
        or repair.get("projected_total_train_supervised_tokens") != 10_032_368
        or repair.get("ten_million_floor_preserved") is not True
        or max(repair.get("family_shares_after_banglabridge_ceiling_trim", {}).values())
        > SOURCE_FAMILY_CAP
    ):
        raise ValueError("closure safe-repair projection changed")
