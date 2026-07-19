"""Provider-neutral, fail-closed storage planning for the Lumi 80B corpus."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .corpus_capacity import build_corpus_capacity_report

SCHEMA_VERSION = 1
SIMULATION_DATABASE_SCHEMA_VERSION = 1
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
MEASURED_RETENTION_CATEGORIES = {
    "open_web",
    "questions_and_discussion",
    "reference",
}
FORBIDDEN_AUTHORIZATIONS = {
    "bulk_corpus_materialization",
    "create_cloud_resources",
    "full_4b_training",
    "paid_compute",
    "transfer_bulk_source_data",
}
REQUIRED_OBJECT_CAPABILITIES = {
    "access_audit_log",
    "abortable_multipart_upload",
    "bucket_versioning",
    "conditional_create",
    "deny_public_access",
    "lifecycle_rules",
    "object_sha256_metadata",
    "range_get",
    "server_side_encryption",
    "tls_transport",
}
SOURCE_MANIFEST_FIELDS = {
    "schema_version",
    "component",
    "dataset_id",
    "dataset_revision",
    "source_file",
    "source_bytes",
    "source_digest",
    "license_policy",
    "provenance_paths",
}
CLEAN_SHARD_MANIFEST_FIELDS = {
    "schema_version",
    "shard_id",
    "category",
    "object_key",
    "object_sha256",
    "source_manifest_sha256",
    "documents",
    "accepted_text_utf8_bytes",
    "tokens",
    "provenance_index_sha256",
    "removal_ledger_generation",
}
REMOVAL_EVENT_FIELDS = {
    "schema_version",
    "event_id",
    "created_at",
    "reason",
    "lookup",
    "affected_shards",
}
FORBIDDEN_PAYLOAD_KEYS = {
    "accepted_text",
    "access_key",
    "api_key",
    "credentials",
    "private_key",
    "raw_text",
    "secret_key",
    "session_token",
    "text",
}
IMPLEMENTATION_FILES = (
    "src/lumi/corpus_capacity.py",
    "src/lumi/corpus_storage.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _ratio(value: int, specification: Mapping[str, Any], label: str) -> int:
    numerator = _positive_int(specification["numerator"], f"{label} numerator")
    denominator = _positive_int(specification["denominator"], f"{label} denominator")
    return (value * numerator + denominator - 1) // denominator


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _resolve(repository_root: Path, raw_path: str, label: str) -> Path:
    path = (repository_root / raw_path).resolve()
    if not path.is_relative_to(repository_root.resolve()):
        raise ValueError(f"{label} escapes the repository")
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    return path


def _unique_by(items: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        value = item.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} has an invalid {key}")
        if value in result:
            raise ValueError(f"duplicate {label} {key}: {value}")
        result[value] = item
    return result


def _validate_no_payload_or_credentials(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_PAYLOAD_KEYS:
                raise ValueError(f"{label} contains forbidden field {key}")
            _validate_no_payload_or_credentials(child, label)
    elif isinstance(value, list):
        for child in value:
            _validate_no_payload_or_credentials(child, label)


def _validate_required_fields(configured: Any, expected: set[str], label: str) -> None:
    if not isinstance(configured, list) or set(configured) != expected:
        raise ValueError(f"{label} required fields changed")
    if len(configured) != len(expected):
        raise ValueError(f"{label} repeats a required field")


def _validate_key_layout(object_store: Mapping[str, Any]) -> None:
    namespace = object_store.get("namespace")
    if not isinstance(namespace, str) or not namespace or namespace.startswith("/"):
        raise ValueError("object-store namespace is invalid")
    templates = object_store.get("key_layout")
    if not isinstance(templates, dict) or not templates:
        raise ValueError("object-store key layout is missing")
    if len(set(templates.values())) != len(templates):
        raise ValueError("object-store key templates must be unique")
    required_placeholders = {
        "clean_shard": {"{category}", "{object_sha256}"},
        "control_manifest": {"{manifest_sha256}"},
        "final_packed_shard": {"{generation}", "{object_sha256}"},
        "provenance_index": {"{partition}", "{object_sha256}"},
        "tokenized_candidate_shard": {"{category}", "{object_sha256}"},
    }
    if set(templates) != set(required_placeholders):
        raise ValueError("object-store key layout has an unexpected artifact set")
    for name, template in templates.items():
        if not isinstance(template, str) or not template.startswith(f"{namespace}/"):
            raise ValueError(f"object key template escapes the namespace: {name}")
        if ".." in template or "//" in template:
            raise ValueError(f"object key template is unsafe: {name}")
        if not required_placeholders[name] <= set(re.findall(r"\{[^}]+\}", template)):
            raise ValueError(f"object key template is missing placeholders: {name}")


def _validate_retention_snapshot(
    snapshot: Mapping[str, Any], tokenizer_contract: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    if snapshot.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported retention snapshot schema")
    evidence = snapshot["evidence"]
    for name in ("database_sha256", "manifest_sha256", "report_sha256", "run_fingerprint"):
        _sha256(evidence[name], f"retention {name}")
    _positive_int(evidence["database_bytes"], "retention database bytes")
    _positive_int(evidence["report_bytes"], "retention report bytes")
    summary = snapshot["summary"]
    if summary["tokenizer_json_sha256"] != tokenizer_contract["baseline"]["tokenizer_json_sha256"]:
        raise ValueError("retention snapshot tokenizer differs from the frozen tokenizer")
    categories = _unique_by(snapshot["categories"], "category", "retention category")
    if set(categories) != MEASURED_RETENTION_CATEGORIES:
        raise ValueError("retention snapshot must cover the three measured categories")
    for category, row in categories.items():
        raw_bytes = _positive_int(row["raw_text_utf8_bytes"], f"{category} raw bytes")
        accepted_bytes = _positive_int(
            row["accepted_text_utf8_bytes"], f"{category} accepted bytes"
        )
        if accepted_bytes > raw_bytes:
            raise ValueError(f"{category} accepted bytes exceed raw bytes")
        _positive_int(row["accepted_tokens"], f"{category} accepted tokens")
        projection = _positive_int(
            row["projected_post_filter_tokens"], f"{category} projected tokens"
        )
        floor = _positive_int(row["minimum_candidate_tokens"], f"{category} candidate floor")
        if projection < floor:
            raise ValueError(f"{category} measured retention misses its floor")
    for name in (
        "accepted_text_utf8_bytes",
        "accepted_tokens",
        "raw_text_utf8_bytes",
    ):
        if summary[name] != sum(int(row[name]) for row in categories.values()):
            raise ValueError(f"retention snapshot summary mismatch: {name}")
    return categories


def evaluate_corpus_storage_plan(
    specification: dict[str, Any],
    mixture_plan: dict[str, Any],
    source_inventory: dict[str, Any],
    capacity_report: dict[str, Any],
    retention_snapshot: dict[str, Any],
    tokenizer_contract: dict[str, Any],
) -> dict[str, Any]:
    """Validate the local-only design and return a deterministic storage plan."""

    if specification.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported corpus object-storage schema")
    authorization = specification["authorization"]
    if authorization.get("local_design_and_simulation") is not True:
        raise ValueError("local design authorization must be explicit")
    if any(authorization.get(name) is not False for name in FORBIDDEN_AUTHORIZATIONS):
        raise ValueError("storage design must not authorize cloud, bulk data, or training")

    if capacity_report["decision"]["metadata_capacity_estimate"] != "pass":
        raise ValueError("the pinned metadata capacity gate has not passed")
    categories = _unique_by(mixture_plan["mixture"], "category", "mixture category")
    inventory = _unique_by(source_inventory["sources"], "component", "source inventory")
    component_categories: dict[str, str] = {}
    for category, item in categories.items():
        for component in item["source_components"]:
            if component in component_categories:
                raise ValueError(f"component appears in multiple categories: {component}")
            component_categories[component] = category
    if set(component_categories) != set(inventory):
        raise ValueError("source inventory does not exactly cover the candidate mixture")

    adaptive = specification["adaptive_acquisition"]
    final_tokens = _positive_int(adaptive["final_training_tokens"], "final tokens")
    if final_tokens != 80_000_000_000:
        raise ValueError("Lumi v0.1 final corpus target must remain 80B tokens")
    if sum(int(item["target_tokens_at_80b"]) for item in categories.values()) != final_tokens:
        raise ValueError("mixture targets do not sum to 80B tokens")
    multiplier = adaptive["candidate_multiplier"]
    numerator = _positive_int(multiplier["numerator"], "candidate multiplier numerator")
    denominator = _positive_int(multiplier["denominator"], "candidate multiplier denominator")
    if (numerator, denominator) != (3, 2):
        raise ValueError("candidate multiplier must remain exactly 1.5x")
    if adaptive.get("stop_when_every_category_floor_is_met") is not True:
        raise ValueError("adaptive acquisition must stop at verified category floors")
    if adaptive.get("forbid_repetition_to_fill_quota") is not True:
        raise ValueError("adaptive acquisition must forbid quota-filling repetition")
    if adaptive.get("holdout_reserved_before_global_deduplication") is not True:
        raise ValueError("held-out data must be reserved before global deduplication")
    hard_maximum = _positive_int(
        adaptive["candidate_hard_maximum_tokens"], "candidate hard maximum"
    )
    if hard_maximum != capacity_report["summary"]["raw_candidate_tokens"]["maximum"]:
        raise ValueError("candidate hard maximum differs from the capacity contract")

    measured = _validate_retention_snapshot(retention_snapshot, tokenizer_contract)
    capacity_categories = _unique_by(capacity_report["categories"], "category", "capacity category")
    if set(capacity_categories) != set(categories):
        raise ValueError("capacity categories do not match the candidate mixture")
    category_rows = []
    for category in sorted(categories):
        item = categories[category]
        target = _positive_int(item["target_tokens_at_80b"], f"{category} target")
        floor = _ceil_div(target * numerator, denominator)
        capacity_row = capacity_categories[category]
        if floor != capacity_row["minimum_candidate_tokens"]:
            raise ValueError(f"{category} candidate floor differs from the capacity gate")
        if category in measured:
            measured_row = measured[category]
            if floor != measured_row["minimum_candidate_tokens"]:
                raise ValueError(f"{category} measured floor differs from the candidate floor")
            projected = int(measured_row["projected_post_filter_tokens"])
            basis = "measured_retention_canary"
        else:
            projected = int(capacity_row["estimated_published_tokens"])
            basis = "pinned_metadata_tokenizer_estimate"
        category_rows.append(
            {
                "capacity_basis": basis,
                "category": category,
                "final_target_tokens": target,
                "minimum_candidate_tokens": floor,
                "projected_available_tokens": projected,
                "projection_headroom_fraction": projected / floor - 1.0,
                "projection_passes": projected >= floor,
                "source_components": list(item["source_components"]),
            }
        )
    if not all(row["projection_passes"] for row in category_rows):
        raise ValueError("at least one category misses its acquisition floor")
    minimum_candidate_tokens = sum(row["minimum_candidate_tokens"] for row in category_rows)
    if minimum_candidate_tokens != 120_000_000_000:
        raise ValueError("minimum candidate pool must remain exactly 120B tokens")
    if minimum_candidate_tokens > hard_maximum:
        raise ValueError("candidate floors exceed the acquisition hard maximum")

    object_store = specification["object_store"]
    if object_store.get("provider") is not None:
        raise ValueError("the provider-neutral base plan cannot select a provider")
    if object_store.get("raw_source_payload_persistence") is not False:
        raise ValueError("raw source payload persistence must stay disabled")
    if object_store.get("credential_material_in_manifests") is not False:
        raise ValueError("credential material must never enter manifests")
    if object_store.get("write_once_content_addressed_keys") is not True:
        raise ValueError("object keys must be write-once and content-addressed")
    capabilities = object_store["required_capabilities"]
    if set(capabilities) != REQUIRED_OBJECT_CAPABILITIES or any(
        capabilities[name] is not True for name in REQUIRED_OBJECT_CAPABILITIES
    ):
        raise ValueError("provider capability requirements changed or are incomplete")
    _validate_key_layout(object_store)

    manifest_contracts = specification["manifest_contracts"]
    _validate_required_fields(
        manifest_contracts["source_object_required_fields"],
        SOURCE_MANIFEST_FIELDS,
        "source manifest",
    )
    _validate_required_fields(
        manifest_contracts["clean_shard_required_fields"],
        CLEAN_SHARD_MANIFEST_FIELDS,
        "clean shard manifest",
    )
    _validate_required_fields(
        manifest_contracts["removal_event_required_fields"],
        REMOVAL_EVENT_FIELDS,
        "removal event",
    )

    deletion = specification["deletion_ledger"]
    if deletion.get("immutable_events") is not True:
        raise ValueError("removal events must be immutable")
    if deletion.get("rebuild_affected_shards_before_next_training_manifest") is not True:
        raise ValueError("pending removals must block the next training manifest")
    if set(deletion["required_lookup_keys"]) != {
        "content_sha256",
        "dataset_id",
        "dataset_revision",
        "source_file",
        "upstream_document_id",
    }:
        raise ValueError("removal lookup contract changed")
    partition_bits = _positive_int(deletion["lookup_partition_bits"], "partition bits")
    if not 8 <= partition_bits <= 20:
        raise ValueError("removal lookup partition bits are outside the safe range")
    event_template = deletion["event_key_template"]
    if not isinstance(event_template, str) or "{event_sha256}" not in event_template:
        raise ValueError("removal event key must be content-addressed")
    if not event_template.startswith(f"{object_store['namespace']}/"):
        raise ValueError("removal events must stay inside the corpus namespace")

    shards = specification["shards"]
    clean_shard_tokens = _positive_int(
        shards["clean_candidate_target_tokens"], "clean shard target"
    )
    final_shard_tokens = _positive_int(
        shards["final_packed_target_tokens"], "final packed shard target"
    )
    overshoot_tokens = _positive_int(
        shards["maximum_category_overshoot_tokens"], "category overshoot"
    )
    if overshoot_tokens > clean_shard_tokens:
        raise ValueError("category overshoot cannot exceed one clean shard")
    if shards.get("packed_token_dtype") != "uint16":
        raise ValueError("the frozen 65K tokenizer requires uint16 packed tokens")
    if shards.get("clean_text_codec") != "zstd":
        raise ValueError("clean candidate shards must use zstd")
    token_id_bytes = _positive_int(shards["candidate_token_id_bytes"], "token bytes")
    if token_id_bytes != 2 or tokenizer_contract["baseline"]["vocab_size"] != 65_536:
        raise ValueError("candidate token storage differs from the frozen tokenizer")
    if _positive_int(shards["maximum_inflight_shards_per_worker"], "inflight shards") != 1:
        raise ValueError("v0.1 permits only one in-flight shard per worker")
    part_bytes = _positive_int(shards["multipart_part_bytes"], "multipart part bytes")
    if part_bytes & (part_bytes - 1) or not 8 * 1024**2 <= part_bytes <= 1024**3:
        raise ValueError("multipart part size must be a safe power of two")
    local_scratch = _positive_int(
        shards["worker_local_scratch_hard_cap_bytes"], "worker scratch cap"
    )

    storage = specification["storage_budget"]
    maximum_text_bytes = _ratio(
        minimum_candidate_tokens,
        storage["maximum_uncompressed_text_bytes_per_token"],
        "uncompressed text ratio",
    )
    compressed_clean_bytes = _ratio(
        maximum_text_bytes,
        storage["clean_text_compression_fraction"],
        "clean compression fraction",
    )
    candidate_token_bytes = minimum_candidate_tokens * token_id_bytes
    final_packed_bytes = final_tokens * token_id_bytes
    index_bytes = _ratio(
        minimum_candidate_tokens,
        storage["candidate_index_bytes_per_token"],
        "candidate index ratio",
    )
    control_bytes = _positive_int(storage["control_plane_reserve_bytes"], "control-plane reserve")
    persistent_bytes = (
        compressed_clean_bytes
        + candidate_token_bytes
        + final_packed_bytes
        + index_bytes
        + control_bytes
    )
    transient_bytes = _ratio(
        persistent_bytes,
        storage["transient_copy_fraction"],
        "transient copy fraction",
    )
    forecast_bytes = persistent_bytes + transient_bytes
    hard_storage_bytes = _positive_int(storage["hard_logical_storage_bytes"], "logical storage cap")
    if forecast_bytes > hard_storage_bytes:
        raise ValueError("conservative storage forecast exceeds the hard cap")
    if local_scratch >= hard_storage_bytes:
        raise ValueError("worker scratch cap must stay below object-storage cap")

    lifecycle = specification["lifecycle"]
    if lifecycle.get("raw_source_payload_retention") != "never_persist":
        raise ValueError("raw source payload lifecycle must remain never-persist")
    for name in (
        "control_manifests_retention",
        "final_packed_retention",
        "removal_ledger_retention",
    ):
        if lifecycle.get(name) != "permanent":
            raise ValueError(f"{name} must remain permanent")
    if lifecycle.get("clean_candidate_retention") != "until_superseded_generation_verified":
        raise ValueError("clean candidate retention must be verification-gated")
    for name in (
        "abort_incomplete_multipart_upload_days",
        "superseded_staging_days",
        "superseded_tokenized_generation_days",
    ):
        days = _positive_int(lifecycle[name], name)
        if days > 90:
            raise ValueError(f"{name} exceeds the v0.1 lifecycle ceiling")

    cost = specification["cost_policy"]
    if cost.get("provider_overlay_required") is not True:
        raise ValueError("a priced provider overlay must be required")
    if cost.get("owner_approval_required") is not True:
        raise ValueError("owner cost approval must be required")
    unresolved_cost_fields = (
        "approved_monthly_cost_cap_usd",
        "egress_usd_per_gb",
        "provider",
        "request_pricing",
        "storage_usd_per_gb_month",
    )
    if any(cost.get(name) is not None for name in unresolved_cost_fields):
        raise ValueError("provider-neutral plan must not invent provider pricing or approval")

    blockers = specification["requirements_before_acquisition_pilot"]
    if not isinstance(blockers, list) or not blockers or len(blockers) != len(set(blockers)):
        raise ValueError("acquisition-pilot blockers must be unique and non-empty")

    source_rows = [
        {
            "category": component_categories[component],
            "component": component,
            "dataset_id": row["dataset_id"],
            "license_policy_status": row["status"],
            "revision": row["revision"],
        }
        for component, row in sorted(inventory.items())
    ]
    return {
        "authorization": dict(authorization),
        "categories": category_rows,
        "cost_policy": dict(cost),
        "decision": {
            "adaptive_acquisition_design": "pass",
            "bulk_corpus_materialization": "not_authorized",
            "cloud_resource_creation": "not_authorized",
            "full_4b_training": "not_authorized",
            "next_action": "provider_selection_pricing_and_bounded_io_canary",
            "paid_compute": "not_authorized",
            "provider_readiness": "blocked_unselected_unpriced_unapproved",
        },
        "deletion_ledger": {
            **deletion,
            "lookup_partitions": 1 << partition_bits,
        },
        "gate": specification["gate"],
        "lifecycle": dict(lifecycle),
        "manifest_contracts": dict(manifest_contracts),
        "object_store": dict(object_store),
        "requirements_before_acquisition_pilot": list(blockers),
        "schema_version": SCHEMA_VERSION,
        "shards": {
            **shards,
            "estimated_clean_candidate_shards": _ceil_div(
                minimum_candidate_tokens, clean_shard_tokens
            ),
            "estimated_final_packed_shards": _ceil_div(final_tokens, final_shard_tokens),
        },
        "sources": source_rows,
        "state_machine": {
            "current_state": "design_only",
            "halt_conditions": [
                "corrupt_shard",
                "pending_removal_event",
                "logical_storage_cap_exceeded",
                "candidate_hard_maximum_exceeded",
                "category_overshoot_exceeds_one_shard",
            ],
            "ready_condition": "every_category_candidate_floor_met_and_all_shards_sealed",
            "simulation_only": True,
        },
        "storage_forecast": {
            "candidate_index_bytes": index_bytes,
            "candidate_token_id_bytes": candidate_token_bytes,
            "clean_candidate_compressed_bytes": compressed_clean_bytes,
            "clean_candidate_uncompressed_text_bytes": maximum_text_bytes,
            "control_plane_reserve_bytes": control_bytes,
            "final_packed_token_bytes": final_packed_bytes,
            "forecast_headroom_fraction": 1.0 - forecast_bytes / hard_storage_bytes,
            "forecast_logical_bytes": forecast_bytes,
            "hard_logical_storage_bytes": hard_storage_bytes,
            "persistent_logical_bytes": persistent_bytes,
            "raw_source_payload_bytes": 0,
            "transient_logical_bytes": transient_bytes,
        },
        "summary": {
            "candidate_hard_maximum_tokens": hard_maximum,
            "final_training_tokens": final_tokens,
            "measured_retention_categories": len(measured),
            "minimum_candidate_tokens": minimum_candidate_tokens,
            "provider": None,
            "sources": len(source_rows),
        },
        "version": specification["version"],
    }


def build_corpus_storage_report(
    contract_path: Path, *, repository_root: Path | None = None
) -> dict[str, Any]:
    """Load every pinned input and bind the deterministic plan to code hashes."""

    contract_path = contract_path.resolve()
    root = repository_root.resolve() if repository_root else contract_path.parents[1]
    specification = _read_json(contract_path)
    paths = {
        label: _resolve(root, raw_path, label)
        for label, raw_path in specification["inputs"].items()
    }
    capacity_report = build_corpus_capacity_report(paths["capacity_contract"], repository_root=root)
    linked_inputs = {
        "mixture": "mixture",
        "source_inventory": "source_inventory",
        "tokenizer_contract": "tokenizer_contract",
    }
    for local_name, capacity_name in linked_inputs.items():
        if _sha256_file(paths[local_name]) != capacity_report["inputs"][capacity_name]["sha256"]:
            raise ValueError(f"storage input {local_name} differs from the capacity gate")
    report = evaluate_corpus_storage_plan(
        specification,
        _read_json(paths["mixture"]),
        _read_json(paths["source_inventory"]),
        capacity_report,
        _read_json(paths["retention_snapshot"]),
        _read_json(paths["tokenizer_contract"]),
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
    report["implementation"] = {path: _sha256_file(root / path) for path in IMPLEMENTATION_FILES}
    report["plan_fingerprint"] = _sha256_bytes(
        _canonical_json_bytes(
            {
                "authorization": report["authorization"],
                "categories": report["categories"],
                "implementation": report["implementation"],
                "inputs": report["inputs"],
                "manifest_contracts": report["manifest_contracts"],
                "object_store": report["object_store"],
                "storage_forecast": report["storage_forecast"],
            }
        )
    )
    return report


def _required_manifest_fields(manifest: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = expected - set(manifest)
    if missing:
        raise ValueError(f"{label} is missing fields: {sorted(missing)}")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported {label} schema")
    _validate_no_payload_or_credentials(manifest, label)


def validate_source_object_manifest(manifest: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    """Validate one future pinned source-object manifest without fetching it."""

    _required_manifest_fields(manifest, SOURCE_MANIFEST_FIELDS, "source object manifest")
    sources = {row["component"]: row for row in report["sources"]}
    component = manifest["component"]
    if component not in sources:
        raise ValueError("source manifest uses an unknown component")
    source = sources[component]
    if manifest["dataset_id"] != source["dataset_id"]:
        raise ValueError("source manifest dataset differs from the pinned inventory")
    if manifest["dataset_revision"] != source["revision"]:
        raise ValueError("source manifest revision differs from the pinned inventory")
    source_file = manifest["source_file"]
    if not isinstance(source_file, str) or not source_file or ".." in source_file:
        raise ValueError("source manifest has an unsafe source file")
    _positive_int(manifest["source_bytes"], "source object bytes")
    digest = manifest["source_digest"]
    if not isinstance(digest, Mapping) or digest.get("algorithm") != "sha256":
        raise ValueError("source object digest must use SHA-256")
    _sha256(digest.get("value"), "source object digest")
    if manifest["license_policy"] not in {
        "approved_source_level_basis",
        "document_license_required",
    }:
        raise ValueError("source manifest has an unsupported license policy")
    expected_license_policy = (
        "approved_source_level_basis"
        if "source_level_basis" in source["license_policy_status"]
        else "document_license_required"
    )
    if manifest["license_policy"] != expected_license_policy:
        raise ValueError("source manifest license policy differs from its inventory status")
    provenance = manifest["provenance_paths"]
    if (
        not isinstance(provenance, list)
        or not provenance
        or not all(isinstance(path, str) and path for path in provenance)
    ):
        raise ValueError("source manifest provenance paths are incomplete")


def validate_clean_shard_manifest(manifest: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    """Validate one future sealed clean-shard manifest without reading payload."""

    _required_manifest_fields(manifest, CLEAN_SHARD_MANIFEST_FIELDS, "clean shard manifest")
    object_sha256 = _sha256(manifest["object_sha256"], "clean shard object digest")
    if _sha256(manifest["shard_id"], "clean shard ID") != object_sha256:
        raise ValueError("clean shard ID must equal its content digest")
    categories = {row["category"] for row in report["categories"]}
    category = manifest["category"]
    if category not in categories:
        raise ValueError("clean shard category is unknown")
    template = report["object_store"]["key_layout"]["clean_shard"]
    expected_key = template.format(category=category, object_sha256=object_sha256)
    if manifest["object_key"] != expected_key:
        raise ValueError("clean shard object key is not canonical")
    _sha256(manifest["source_manifest_sha256"], "source manifest digest")
    _sha256(manifest["provenance_index_sha256"], "provenance index digest")
    _sha256(manifest["removal_ledger_generation"], "removal ledger generation")
    _positive_int(manifest["documents"], "clean shard documents")
    _positive_int(manifest["accepted_text_utf8_bytes"], "clean shard text bytes")
    tokens = _positive_int(manifest["tokens"], "clean shard tokens")
    if tokens > (
        report["shards"]["clean_candidate_target_tokens"]
        + report["shards"]["maximum_category_overshoot_tokens"]
    ):
        raise ValueError("clean shard exceeds the bounded whole-document overshoot")


def removal_event_id(event: Mapping[str, Any]) -> str:
    """Return the content address for a removal event, excluding its ID field."""

    payload = {key: value for key, value in event.items() if key != "event_id"}
    return _sha256_bytes(_canonical_json_bytes(payload))


def validate_removal_event(event: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    """Validate a future immutable removal-ledger event."""

    _required_manifest_fields(event, REMOVAL_EVENT_FIELDS, "removal event")
    event_id = _sha256(event["event_id"], "removal event ID")
    if event_id != removal_event_id(event):
        raise ValueError("removal event ID is not its canonical content digest")
    created_at = event["created_at"]
    if (
        not isinstance(created_at, str)
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", created_at) is None
    ):
        raise ValueError("removal event timestamp must be UTC second precision")
    if not isinstance(event["reason"], str) or not event["reason"].strip():
        raise ValueError("removal event reason is missing")
    lookup = event["lookup"]
    required_lookup = set(report["deletion_ledger"]["required_lookup_keys"])
    if not isinstance(lookup, Mapping) or set(lookup) != required_lookup:
        raise ValueError("removal event lookup keys are incomplete")
    if any(not isinstance(value, str) or not value for value in lookup.values()):
        raise ValueError("removal event lookup values are incomplete")
    _sha256(lookup["content_sha256"], "removal content digest")
    affected = event["affected_shards"]
    if not isinstance(affected, list) or len(affected) != len(set(affected)):
        raise ValueError("removal event affected shards are invalid")
    for shard_id in affected:
        _sha256(shard_id, "affected shard ID")


def evaluate_adaptive_progress(
    report: Mapping[str, Any], progress: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate a metadata-only acquisition simulation against fail-closed stops."""

    if report["state_machine"].get("simulation_only") is not True:
        raise ValueError("adaptive progress evaluation must remain simulation-only")
    category_plan = {row["category"]: row for row in report["categories"]}
    category_progress = progress.get("categories")
    if not isinstance(category_progress, Mapping) or set(category_progress) != set(category_plan):
        raise ValueError("progress categories do not match the storage plan")
    rows = []
    for category in sorted(category_plan):
        sealed = _nonnegative_int(
            category_progress[category]["sealed_unique_candidate_tokens"],
            f"{category} sealed tokens",
        )
        floor = int(category_plan[category]["minimum_candidate_tokens"])
        rows.append(
            {
                "category": category,
                "completion_fraction": sealed / floor,
                "deficit_tokens": max(0, floor - sealed),
                "floor_tokens": floor,
                "sealed_unique_candidate_tokens": sealed,
            }
        )
    total_tokens = sum(row["sealed_unique_candidate_tokens"] for row in rows)
    logical_bytes = _nonnegative_int(
        progress.get("logical_storage_bytes"), "progress logical storage bytes"
    )
    corrupt_shards = _nonnegative_int(progress.get("corrupt_shards"), "corrupt shards")
    pending_removals = _nonnegative_int(
        progress.get("pending_removal_events"), "pending removal events"
    )
    unsealed_shards = _nonnegative_int(progress.get("unsealed_shards"), "unsealed shards")
    overshoot = int(report["shards"]["maximum_category_overshoot_tokens"])
    halt_reason = None
    if corrupt_shards:
        halt_reason = "corrupt_shard"
    elif pending_removals:
        halt_reason = "pending_removal_event"
    elif logical_bytes > report["storage_forecast"]["hard_logical_storage_bytes"]:
        halt_reason = "logical_storage_cap_exceeded"
    elif total_tokens > report["summary"]["candidate_hard_maximum_tokens"]:
        halt_reason = "candidate_hard_maximum_exceeded"
    elif any(
        row["sealed_unique_candidate_tokens"] > row["floor_tokens"] + overshoot for row in rows
    ):
        halt_reason = "category_overshoot_exceeds_one_shard"
    all_floors_met = all(row["deficit_tokens"] == 0 for row in rows)
    if halt_reason is not None:
        state = "halted"
    elif all_floors_met and unsealed_shards == 0:
        state = "candidate_ready"
    elif all_floors_met:
        state = "sealing"
    else:
        state = "continue"
    next_category = None
    if state == "continue":
        next_category = min(
            (row for row in rows if row["deficit_tokens"] > 0),
            key=lambda row: (row["completion_fraction"], row["category"]),
        )["category"]
    return {
        "authorization": "simulation_only",
        "categories": rows,
        "halt_reason": halt_reason,
        "logical_storage_bytes": logical_bytes,
        "next_category": next_category,
        "state": state,
        "total_sealed_unique_candidate_tokens": total_tokens,
    }


def _open_simulation_database(path: Path, report: Mapping[str, Any]) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS simulation_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS simulated_shards ("
        "shard_id TEXT PRIMARY KEY, payload_sha256 TEXT NOT NULL, category TEXT NOT NULL, "
        "tokens INTEGER NOT NULL, logical_bytes INTEGER NOT NULL, status TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS simulated_removals ("
        "event_id TEXT PRIMARY KEY, payload_sha256 TEXT NOT NULL, status TEXT NOT NULL)"
    )
    stored_schema = connection.execute(
        "SELECT value FROM simulation_metadata WHERE key = 'schema_version'"
    ).fetchone()
    if stored_schema is not None and stored_schema[0] != str(SIMULATION_DATABASE_SCHEMA_VERSION):
        connection.close()
        raise ValueError("simulation state uses an unsupported database schema")
    fingerprint = report["plan_fingerprint"]
    existing = connection.execute(
        "SELECT value FROM simulation_metadata WHERE key = 'plan_fingerprint'"
    ).fetchone()
    if existing is not None and existing[0] != fingerprint:
        connection.close()
        raise ValueError("simulation state belongs to a different storage plan")
    with connection:
        connection.execute(
            "INSERT OR REPLACE INTO simulation_metadata (key, value) VALUES (?, ?)",
            ("plan_fingerprint", fingerprint),
        )
        connection.execute(
            "INSERT OR REPLACE INTO simulation_metadata (key, value) VALUES (?, ?)",
            ("schema_version", str(SIMULATION_DATABASE_SCHEMA_VERSION)),
        )
    return connection


def record_simulated_shard(
    database: Path,
    report: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    logical_bytes: int,
) -> bool:
    """Atomically record metadata for a sealed fake shard; never store payload text."""

    validate_clean_shard_manifest(manifest, report)
    logical_bytes = _positive_int(logical_bytes, "simulated shard logical bytes")
    payload_sha256 = _sha256_bytes(_canonical_json_bytes(manifest))
    connection = _open_simulation_database(database, report)
    try:
        existing = connection.execute(
            "SELECT payload_sha256 FROM simulated_shards WHERE shard_id = ?",
            (manifest["shard_id"],),
        ).fetchone()
        if existing is not None:
            if existing[0] != payload_sha256:
                raise ValueError("simulated shard ID was reused with different metadata")
            return False
        with connection:
            connection.execute(
                "INSERT INTO simulated_shards "
                "(shard_id, payload_sha256, category, tokens, logical_bytes, status) "
                "VALUES (?, ?, ?, ?, ?, 'sealed')",
                (
                    manifest["shard_id"],
                    payload_sha256,
                    manifest["category"],
                    manifest["tokens"],
                    logical_bytes,
                ),
            )
        return True
    finally:
        connection.close()


def record_simulated_removal(
    database: Path, report: Mapping[str, Any], event: Mapping[str, Any]
) -> bool:
    """Atomically record a fake removal and block every referenced fake shard."""

    validate_removal_event(event, report)
    payload_sha256 = _sha256_bytes(_canonical_json_bytes(event))
    connection = _open_simulation_database(database, report)
    try:
        existing = connection.execute(
            "SELECT payload_sha256 FROM simulated_removals WHERE event_id = ?",
            (event["event_id"],),
        ).fetchone()
        if existing is not None:
            if existing[0] != payload_sha256:
                raise ValueError("removal event ID was reused with different metadata")
            return False
        with connection:
            connection.execute(
                "INSERT INTO simulated_removals "
                "(event_id, payload_sha256, status) VALUES (?, ?, 'pending')",
                (event["event_id"], payload_sha256),
            )
            connection.executemany(
                "UPDATE simulated_shards SET status = 'removal_pending' WHERE shard_id = ?",
                ((shard_id,) for shard_id in event["affected_shards"]),
            )
        return True
    finally:
        connection.close()


def read_simulation_progress(database: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    """Reopen durable metadata-only state and evaluate its adaptive stop decision."""

    connection = _open_simulation_database(database, report)
    try:
        categories = {
            row["category"]: {"sealed_unique_candidate_tokens": 0} for row in report["categories"]
        }
        for row in connection.execute(
            "SELECT category, COALESCE(SUM(tokens), 0) AS tokens "
            "FROM simulated_shards WHERE status = 'sealed' GROUP BY category"
        ):
            categories[row["category"]]["sealed_unique_candidate_tokens"] = int(row["tokens"])
        logical_bytes = int(
            connection.execute(
                "SELECT COALESCE(SUM(logical_bytes), 0) FROM simulated_shards"
            ).fetchone()[0]
        )
        pending_removals = int(
            connection.execute(
                "SELECT COUNT(*) FROM simulated_removals WHERE status = 'pending'"
            ).fetchone()[0]
        )
        corrupt = int(
            connection.execute(
                "SELECT COUNT(*) FROM simulated_shards WHERE status = 'corrupt'"
            ).fetchone()[0]
        )
        unsealed = int(
            connection.execute(
                "SELECT COUNT(*) FROM simulated_shards WHERE status = 'uploading'"
            ).fetchone()[0]
        )
    finally:
        connection.close()
    return evaluate_adaptive_progress(
        report,
        {
            "categories": categories,
            "corrupt_shards": corrupt,
            "logical_storage_bytes": logical_bytes,
            "pending_removal_events": pending_removals,
            "unsealed_shards": unsealed,
        },
    )


def verify_corpus_storage_report(
    contract_path: Path,
    report_path: Path,
    manifest_path: Path,
    *,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Verify the report hash and reproduce the complete deterministic plan."""

    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        raise ValueError("storage-plan manifest must contain exactly one entry")
    digest, name = lines[0].split("  ", 1)
    _sha256(digest, "storage-plan manifest digest")
    if name != report_path.name or _sha256_file(report_path) != digest:
        raise ValueError("storage-plan report hash does not match its manifest")
    observed = _read_json(report_path)
    expected = build_corpus_storage_report(contract_path, repository_root=repository_root)
    if observed != expected:
        raise ValueError("storage-plan report cannot be reproduced from pinned inputs")
    return observed
