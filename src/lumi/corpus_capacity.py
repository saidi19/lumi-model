"""Fail-closed planning gate for the Lumi 80B-token corpus."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


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


def _resolve(repository_root: Path, raw_path: str, label: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = repository_root / path
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    return path


def evaluate_corpus_capacity(
    specification: dict[str, Any],
    mixture_plan: dict[str, Any],
    source_inventory: dict[str, Any],
    capacity_snapshot: dict[str, Any],
    tokenizer_contract: dict[str, Any],
) -> dict[str, Any]:
    """Validate pinned inputs and return deterministic planning estimates."""

    if specification.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported corpus capacity gate schema")
    if capacity_snapshot.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported published capacity snapshot schema")

    authorization = specification["authorization"]
    forbidden = ("bulk_corpus_materialization", "full_4b_training", "paid_compute")
    if any(authorization.get(name) is not False for name in forbidden):
        raise ValueError("planning gate must not authorize bulk data, paid compute, or training")
    if authorization.get("bounded_metadata_and_canary_work") is not True:
        raise ValueError("bounded metadata/canary work must be explicit")

    targets = specification["targets"]
    final_tokens = _positive_int(targets["final_training_tokens"], "final target")
    if final_tokens != 80_000_000_000:
        raise ValueError("Lumi v0.1 final corpus target must remain exactly 80B tokens")
    raw_target = targets["raw_candidate_tokens"]
    raw_minimum = _positive_int(raw_target["minimum"], "raw candidate minimum")
    raw_planning = _positive_int(raw_target["planning"], "raw candidate planning target")
    raw_maximum = _positive_int(raw_target["maximum"], "raw candidate maximum")
    if not raw_minimum <= raw_planning <= raw_maximum:
        raise ValueError("raw candidate targets must be ordered")
    multiplier = targets["minimum_category_candidate_multiplier"]
    multiplier_numerator = _positive_int(multiplier["numerator"], "multiplier numerator")
    multiplier_denominator = _positive_int(
        multiplier["denominator"], "multiplier denominator"
    )
    if multiplier_numerator <= multiplier_denominator:
        raise ValueError("category candidate multiplier must be greater than one")

    mixture = mixture_plan["mixture"]
    category_items = _unique_by(mixture, "category", "mixture category")
    if not math.isclose(sum(float(item["weight"]) for item in mixture), 1.0):
        raise ValueError("mixture weights must sum to one")
    category_target_total = sum(
        _positive_int(item["target_tokens_at_80b"], "category target") for item in mixture
    )
    if category_target_total != final_tokens:
        raise ValueError("mixture category targets do not sum to the final target")

    component_categories: dict[str, str] = {}
    for item in mixture:
        for component in item["source_components"]:
            if component in component_categories:
                raise ValueError(f"component appears in multiple categories: {component}")
            component_categories[component] = item["category"]

    inventory = _unique_by(source_inventory["sources"], "component", "source inventory")
    snapshot = _unique_by(capacity_snapshot["sources"], "component", "capacity snapshot")
    expected_components = set(component_categories)
    if set(inventory) != expected_components:
        raise ValueError("source inventory does not exactly cover the mixture")
    if set(snapshot) != expected_components:
        raise ValueError("capacity snapshot does not exactly cover the mixture")

    calibration = specification["tokenizer_calibration"]
    baseline = tokenizer_contract["baseline"]
    sample = tokenizer_contract["sample"]
    if baseline["tokenizer_json_sha256"] != calibration["tokenizer_json_sha256"]:
        raise ValueError("frozen tokenizer hash does not match the capacity calibration")
    if baseline["vocab_size"] != calibration["vocab_size"]:
        raise ValueError("frozen tokenizer vocabulary does not match the capacity calibration")
    if sample["documents"] != calibration["sample_documents"]:
        raise ValueError("tokenizer calibration document count changed")
    if sample["output_sha256"] != calibration["sample_output_sha256"]:
        raise ValueError("tokenizer calibration sample hash changed")
    if sample["text_utf8_bytes"] != calibration["sample_text_utf8_bytes"]:
        raise ValueError("tokenizer calibration byte count changed")
    category_calibration = calibration["categories"]
    if set(category_calibration) != set(category_items):
        raise ValueError("tokenizer category calibration does not match the mixture")

    category_totals = {
        category: {
            "estimated_published_tokens": 0,
            "published_documents": 0,
            "published_utf8_bytes": 0,
            "tree_compressed_bytes": 0,
        }
        for category in category_items
    }
    source_rows = []
    for component in sorted(expected_components):
        inventory_row = inventory[component]
        snapshot_row = snapshot[component]
        for field in ("dataset_id", "revision"):
            if snapshot_row[field] != inventory_row[field]:
                raise ValueError(f"{component} {field} drifted from the pinned inventory")
        if snapshot_row["tree_data_file_count"] != inventory_row["data_file_count"]:
            raise ValueError(f"{component} tree file count drifted from the pinned inventory")
        published_bytes = _positive_int(
            snapshot_row["published_utf8_bytes"], f"{component} published UTF-8 bytes"
        )
        compressed_bytes = _positive_int(
            snapshot_row["tree_compressed_bytes"], f"{component} compressed bytes"
        )
        documents = _positive_int(snapshot_row["published_documents"], f"{component} documents")
        category = component_categories[component]
        category_sample = category_calibration[category]
        estimated_tokens = (
            published_bytes
            * _positive_int(category_sample["sample_tokens"], f"{category} sample tokens")
            // _positive_int(
                category_sample["sample_text_utf8_bytes"], f"{category} sample bytes"
            )
        )
        source_rows.append(
            {
                "category": category,
                "component": component,
                "dataset_id": snapshot_row["dataset_id"],
                "estimated_published_tokens": estimated_tokens,
                "published_documents": documents,
                "published_utf8_bytes": published_bytes,
                "revision": snapshot_row["revision"],
                "tree_compressed_bytes": compressed_bytes,
                "tree_data_file_count": snapshot_row["tree_data_file_count"],
            }
        )
        totals = category_totals[category]
        totals["estimated_published_tokens"] += estimated_tokens
        totals["published_documents"] += documents
        totals["published_utf8_bytes"] += published_bytes
        totals["tree_compressed_bytes"] += compressed_bytes

    category_rows = []
    for category in sorted(category_items):
        item = category_items[category]
        final_category_tokens = item["target_tokens_at_80b"]
        minimum_candidate_tokens = (
            final_category_tokens * multiplier_numerator + multiplier_denominator - 1
        ) // multiplier_denominator
        totals = category_totals[category]
        estimate_passes = totals["estimated_published_tokens"] >= minimum_candidate_tokens
        category_rows.append(
            {
                "category": category,
                **totals,
                "final_target_tokens": final_category_tokens,
                "minimum_candidate_tokens": minimum_candidate_tokens,
                "planning_estimate_headroom_fraction": (
                    totals["estimated_published_tokens"] / minimum_candidate_tokens - 1.0
                ),
                "planning_estimate_passes": estimate_passes,
            }
        )

    total_estimated_tokens = sum(row["estimated_published_tokens"] for row in category_rows)
    minimum_category_tokens = sum(row["minimum_candidate_tokens"] for row in category_rows)
    metadata_passes = all(row["planning_estimate_passes"] for row in category_rows)
    metadata_passes = metadata_passes and total_estimated_tokens >= raw_maximum
    blockers = list(specification["requirements_before_bulk_materialization"])
    if len(blockers) != len(set(blockers)) or not blockers:
        raise ValueError("bulk-materialization blockers must be unique and non-empty")

    return {
        "authorization": authorization,
        "categories": category_rows,
        "decision": {
            "bounded_source_canaries": "ready" if metadata_passes else "blocked",
            "bulk_corpus_materialization": "blocked_pending_measured_capacity_and_owner_approval",
            "full_4b_training": "not_authorized",
            "metadata_capacity_estimate": "pass" if metadata_passes else "fail",
            "next_action": "bounded_source_acquisition_canaries",
            "paid_compute": "not_authorized",
        },
        "estimate_semantics": {
            "capacity": (
                "official dataset-card UTF-8 size converted with frozen-tokenizer "
                "category fertility"
            ),
            "not_proof_of": [
                "post_Lumi_filter_tokens",
                "post_global_dedup_unique_tokens",
                "license_acceptance",
                "final_80B_corpus_readiness",
            ],
        },
        "gate": specification["gate"],
        "requirements_before_bulk_materialization": blockers,
        "schema_version": SCHEMA_VERSION,
        "sources": source_rows,
        "summary": {
            "estimated_published_tokens": total_estimated_tokens,
            "final_training_tokens": final_tokens,
            "minimum_category_candidate_tokens": minimum_category_tokens,
            "planning_flexible_candidate_tokens": raw_planning - minimum_category_tokens,
            "published_documents": sum(row["published_documents"] for row in category_rows),
            "published_utf8_bytes": sum(row["published_utf8_bytes"] for row in category_rows),
            "raw_candidate_tokens": raw_target,
            "required_final_retention_fraction_at_planning_target": final_tokens / raw_planning,
            "tree_compressed_bytes": sum(
                row["tree_compressed_bytes"] for row in category_rows
            ),
        },
        "version": specification["version"],
    }


def build_corpus_capacity_report(
    contract_path: Path, *, repository_root: Path | None = None
) -> dict[str, Any]:
    """Load a contract and all pinned inputs, then add their file hashes."""

    contract_path = contract_path.resolve()
    specification = _read_json(contract_path)
    root = repository_root.resolve() if repository_root else contract_path.parents[1]
    paths = {
        label: _resolve(root, raw_path, label)
        for label, raw_path in specification["inputs"].items()
    }
    report = evaluate_corpus_capacity(
        specification,
        _read_json(paths["mixture"]),
        _read_json(paths["source_inventory"]),
        _read_json(paths["published_capacity_snapshot"]),
        _read_json(paths["tokenizer_contract"]),
    )
    report["inputs"] = {
        "contract": {
            "path": contract_path.relative_to(root).as_posix(),
            "sha256": _sha256(contract_path),
        },
        **{
            label: {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
            }
            for label, path in sorted(paths.items())
        },
    }
    return report
