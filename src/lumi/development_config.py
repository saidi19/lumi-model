"""Compose the all-component Lumi development-sample configuration."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from lumi.data_pipeline import validate_sample_config


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repository_path(repository_root: Path, value: str) -> Path:
    path = (repository_root / value).resolve()
    if not path.is_relative_to(repository_root.resolve()):
        raise ValueError(f"development config path escapes repository: {value}")
    if not path.is_file():
        raise FileNotFoundError(f"development config input does not exist: {path}")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be an object: {path}")
    return value


def _source_provenance(component: Mapping[str, Any]) -> tuple[dict, dict, list[str]]:
    retained = {
        "collection_provenance": "metadata.provenance",
        "upstream_id": "id",
        "upstream_source": "source",
    }
    static = {}
    required = ["upstream_id", "upstream_source", "collection_provenance"]
    license_paths = component.get("license_paths", [])
    if license_paths:
        retained["license"] = license_paths[0]
        required.append("license")
    else:
        basis = component.get("license_basis")
        if not isinstance(basis, Mapping):
            raise ValueError(f"component {component['component']} has no license path or basis")
        static["license_basis"] = "; ".join(
            str(basis[key]) for key in ("kind", "reference", "warning") if key in basis
        )
        required.append("license_basis")
    url_paths = component.get("url_paths", [])
    if url_paths:
        retained["upstream_url"] = url_paths[0]
    return retained, static, required


def compose_development_config(
    plan_path: Path, *, repository_root: Path | None = None
) -> dict[str, Any]:
    repository_root = repository_root or Path(__file__).resolve().parents[2]
    plan = _load_json(plan_path)
    if plan.get("version") not in {"0.1", "0.2"}:
        raise ValueError("development sample plan version must be 0.1 or 0.2")

    input_paths = {
        name: _repository_path(repository_root, plan[name])
        for name in (
            "component_anchor_files",
            "component_inventory",
            "mixture_plan",
            "quality_config",
        )
    }
    supplemental_path_value = plan.get("supplemental_mixture")
    if supplemental_path_value is not None:
        if not isinstance(supplemental_path_value, str):
            raise ValueError("supplemental_mixture must be a repository-relative path")
        input_paths["supplemental_mixture"] = _repository_path(
            repository_root, supplemental_path_value
        )
    capacity_path_value = plan.get("source_capacity_report")
    if capacity_path_value is not None:
        if not isinstance(capacity_path_value, str):
            raise ValueError("source_capacity_report must be a repository-relative path")
        input_paths["source_capacity_report"] = _repository_path(
            repository_root, capacity_path_value
        )
    anchors = _load_json(input_paths["component_anchor_files"])
    inventory = _load_json(input_paths["component_inventory"])
    mixture_plan = _load_json(input_paths["mixture_plan"])
    quality = _load_json(input_paths["quality_config"])
    capacity_report = None
    if "source_capacity_report" in input_paths:
        capacity_report = _load_json(input_paths["source_capacity_report"])
        if capacity_report.get("version") != "0.1":
            raise ValueError("source capacity report version must be 0.1")
        if capacity_report.get("status") != "complete":
            raise ValueError("source capacity report must be complete")
        if capacity_report.get("filters") != plan.get("filters"):
            raise ValueError("source capacity report filters do not match the plan")
        if capacity_report.get("quality_config_sha256") != _sha256_file(
            input_paths["quality_config"]
        ):
            raise ValueError("source capacity report quality config does not match")
    supplemental = {"mixture": [], "version": "0.1"}
    if "supplemental_mixture" in input_paths:
        supplemental = _load_json(input_paths["supplemental_mixture"])
        if supplemental.get("version") != "0.1":
            raise ValueError("supplemental mixture version must be 0.1")
        if not isinstance(supplemental.get("mixture"), list):
            raise ValueError("supplemental mixture must contain a mixture list")

    inventory_by_component = {
        source["component"]: source for source in inventory["sources"]
    }
    anchor_by_component = {source["component"]: source for source in anchors["sources"]}
    expected_components = {
        component
        for entry in mixture_plan["mixture"]
        for component in entry["source_components"]
    }
    if set(inventory_by_component) != expected_components:
        raise ValueError("component inventory does not exactly cover the mixture plan")
    if set(anchor_by_component) != expected_components:
        raise ValueError("component anchor files do not exactly cover the mixture plan")

    capacities_by_component: dict[str, Mapping[str, Any]] = {}
    if capacity_report is not None:
        capacity_sources = capacity_report.get("sources")
        if not isinstance(capacity_sources, list) or not capacity_sources:
            raise ValueError("source capacity report must contain a non-empty sources list")
        for capacity in capacity_sources:
            if not isinstance(capacity, Mapping):
                raise ValueError("source capacity report entries must be objects")
            component_name = capacity.get("component")
            if component_name not in expected_components:
                raise ValueError(f"capacity report contains unknown component {component_name}")
            if component_name in capacities_by_component:
                raise ValueError(f"capacity report repeats component {component_name}")
            accepted_bytes = capacity.get("accepted_text_utf8_bytes")
            if (
                not isinstance(accepted_bytes, int)
                or isinstance(accepted_bytes, bool)
                or accepted_bytes <= 0
            ):
                raise ValueError(f"capacity report has invalid bytes for {component_name}")
            capacities_by_component[str(component_name)] = capacity

    configured_weights = plan.get("source_weights")
    if not isinstance(configured_weights, Mapping):
        raise ValueError("development plan source_weights must be an object")
    supplemental_weight = sum(
        float(entry.get("weight", 0.0)) for entry in supplemental["mixture"]
    )
    if not 0.0 <= supplemental_weight < 1.0:
        raise ValueError("supplemental mixture weight must be in [0, 1)")
    base_weight_scale = 1.0 - supplemental_weight
    base_categories = {entry["category"] for entry in mixture_plan["mixture"]}
    supplemental_categories = [entry.get("category") for entry in supplemental["mixture"]]
    if len(set(supplemental_categories)) != len(supplemental_categories):
        raise ValueError("supplemental mixture repeats a category")
    if base_categories & set(supplemental_categories):
        raise ValueError("supplemental mixture category collides with the base plan")

    mixture = []
    for mixture_entry in mixture_plan["mixture"]:
        category = mixture_entry["category"]
        weights = configured_weights.get(category)
        if not isinstance(weights, Mapping):
            raise ValueError(f"development plan has no source weights for {category}")
        expected_for_category = set(mixture_entry["source_components"])
        if set(weights) != expected_for_category:
            raise ValueError(f"source weights do not exactly cover {category}")
        if not math.isclose(sum(float(value) for value in weights.values()), 1.0):
            raise ValueError(f"source weights for {category} do not sum to one")

        sources = []
        for component_name in mixture_entry["source_components"]:
            component = inventory_by_component[component_name]
            anchor = anchor_by_component[component_name]
            if anchor["file"] != component["schema_anchor"]:
                raise ValueError(f"anchor mismatch for component {component_name}")
            capacity = capacities_by_component.get(component_name)
            if capacity is not None:
                expected_capacity_pins = {
                    "blob_sha1": anchor["blob_sha1"],
                    "dataset_id": component["dataset_id"],
                    "file": anchor["file"],
                    "revision": component["revision"],
                }
                for name, expected_value in expected_capacity_pins.items():
                    if capacity.get(name) != expected_value:
                        raise ValueError(
                            f"capacity report {name} mismatch for component {component_name}"
                        )
            retained, static, required = _source_provenance(component)
            source = {
                "blob_sha1": anchor["blob_sha1"],
                "component": component_name,
                "compressed_bytes": anchor["compressed_bytes"],
                "dataset_id": component["dataset_id"],
                "file": anchor["file"],
                "required_non_null_provenance": required,
                "retain_provenance": retained,
                "revision": component["revision"],
                "weight_within_category": weights[component_name],
            }
            if capacity is not None:
                source["verified_capacity_text_utf8_bytes"] = capacity[
                    "accepted_text_utf8_bytes"
                ]
            if static:
                source["static_provenance"] = static
            sources.append(source)
        mixture.append(
            {
                "category": category,
                "sources": sources,
                "weight": float(mixture_entry["weight"]) * base_weight_scale,
            }
        )
    mixture.extend(deepcopy(supplemental["mixture"]))

    config = {
        "composition": {
            name: {
                "path": str(path.relative_to(repository_root)),
                "sha256": _sha256_file(path),
            }
            for name, path in sorted(input_paths.items())
        },
        "dataset": {
            "collection_code": "https://github.com/r-three/common-pile",
            "id": mixture_plan["primary_dataset"]["id"],
            "revision": mixture_plan["primary_dataset"]["revision"],
            "upstream_card": (
                "https://huggingface.co/datasets/"
                + mixture_plan["primary_dataset"]["id"]
            ),
        },
        "filters": plan["filters"],
        "mixture": mixture,
        "purpose": plan["purpose"],
        "quality": quality,
        "target_uncompressed_text_bytes": plan["target_uncompressed_text_bytes"],
        "version": plan["version"],
    }
    validate_sample_config(config)
    return config
