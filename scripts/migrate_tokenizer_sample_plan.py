#!/usr/bin/env python3
"""Strictly migrate a failed tokenizer sample to a compatible multi-shard plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lumi.data_pipeline import (
    PII_DETECTOR_VERSION,
    _canonical_json_bytes,
    _category_targets,
    _scan_records,
    _sha256_bytes,
    _source_key,
    _source_targets,
    _source_variants,
)
from lumi.development_config import compose_development_config

REPOSITORY_ROOT = Path(__file__).parents[1]
DEFAULT_OLD_PLAN = REPOSITORY_ROOT / "configs" / "tokenizer-sample-v0.2.json"
DEFAULT_NEW_PLAN = REPOSITORY_ROOT / "configs" / "tokenizer-sample-v0.3.json"
EXHAUSTION_PATTERN = re.compile(
    r"^pinned source (?P<file>\S+) ended at (?P<accepted>\d+) of "
    r"(?P<target>\d+) target bytes$"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_sha256(config: dict[str, Any]) -> str:
    return _sha256_bytes(_canonical_json_bytes(config))


def _run_fingerprint(config_sha256: str, target_bytes: int) -> str:
    return _sha256_bytes(
        _canonical_json_bytes(
            {
                "config_sha256": config_sha256,
                "pii_detector": PII_DETECTOR_VERSION,
                "target_bytes": target_bytes,
            }
        )
    )


def _sampling_semantics(config: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(config)
    for name in (
        "composition",
        "purpose",
        "required_capacity_preflight_components",
        "version",
    ):
        normalized.pop(name, None)
    for entry in normalized["mixture"]:
        for source in entry["sources"]:
            source.pop("continuation_files", None)
            source.pop("verified_capacity_is_exhaustive", None)
            source.pop("verified_capacity_text_utf8_bytes", None)
    return normalized


def _find_exhausted_source(
    config: dict[str, Any], stats: dict[str, Any], failure_message: str
) -> tuple[tuple[str, str, str, str], dict[str, Any], int]:
    match = EXHAUSTION_PATTERN.fullmatch(failure_message)
    if match is None:
        raise ValueError("old run failure is not a recognized single-source exhaustion")
    failed_file = match.group("file")
    claimed_accepted = int(match.group("accepted"))
    claimed_target = int(match.group("target"))
    candidates = []
    category_targets = _category_targets(config, int(config["target_uncompressed_text_bytes"]))
    source_targets = _source_targets(config, category_targets)
    for entry in config["mixture"]:
        category = entry["category"]
        for source in entry["sources"]:
            logical_key = _source_key(config["dataset"], source, category)
            for variant in _source_variants(source):
                if variant["file"] == failed_file:
                    candidates.append((category, source, variant, logical_key))
    if len(candidates) != 1:
        raise ValueError("failed physical source is not uniquely pinned in the new plan")
    category, source, variant, logical_key = candidates[0]
    physical_key = _source_key(config["dataset"], variant, category)
    accepted = int(stats["physical_source_bytes"].get(physical_key, 0))
    verified_capacity = variant.get("verified_capacity_text_utf8_bytes")
    if accepted != claimed_accepted:
        raise ValueError(
            f"sample contains {accepted} bytes from failed file, not claimed {claimed_accepted}"
        )
    if verified_capacity != accepted or not bool(
        variant.get("verified_capacity_is_exhaustive", True)
    ):
        raise ValueError("failed file does not have a matching exhaustive capacity audit")
    if source_targets.get(logical_key) != claimed_target:
        raise ValueError("new plan changes the failed component quota")
    return physical_key, source, claimed_target


def inspect_migration(
    old_plan: Path, new_plan: Path, output_directory: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    old_config = compose_development_config(old_plan, repository_root=REPOSITORY_ROOT)
    new_config = compose_development_config(new_plan, repository_root=REPOSITORY_ROOT)
    if _sampling_semantics(old_config) != _sampling_semantics(new_config):
        raise ValueError(
            "new plan changes sampling semantics beyond continuation/capacity metadata"
        )
    old_targets = _source_targets(
        old_config,
        _category_targets(old_config, int(old_config["target_uncompressed_text_bytes"])),
    )
    new_targets = _source_targets(
        new_config,
        _category_targets(new_config, int(new_config["target_uncompressed_text_bytes"])),
    )
    if old_targets != new_targets:
        raise ValueError("new plan changes one or more logical source quotas")

    sample_path = output_directory / "sample.jsonl"
    state_path = output_directory / "run-state.json"
    manifest_path = output_directory / "manifest.json"
    if not sample_path.is_file() or not state_path.is_file():
        raise FileNotFoundError("migration requires sample.jsonl and run-state.json")
    if manifest_path.exists():
        raise ValueError("migration refuses an output that already has a manifest")
    with state_path.open(encoding="utf-8") as handle:
        state = json.load(handle)
    if state.get("status") != "failed":
        raise ValueError("migration requires a safely failed run state")
    if state.get("failure", {}).get("type") != "RuntimeError":
        raise ValueError("migration requires a RuntimeError exhaustion failure")

    old_config_sha256 = _config_sha256(old_config)
    target_bytes = int(old_config["target_uncompressed_text_bytes"])
    if state.get("config_sha256") != old_config_sha256:
        raise ValueError("old run-state config hash does not match the old plan")
    if state.get("run_fingerprint") != _run_fingerprint(old_config_sha256, target_bytes):
        raise ValueError("old run-state fingerprint does not match the old plan")
    if state.get("target_uncompressed_text_bytes") != target_bytes:
        raise ValueError("old run-state target does not match the old plan")

    stats = _scan_records(sample_path, new_config)
    physical_key, failed_source, failed_target = _find_exhausted_source(
        new_config, stats, str(state["failure"]["message"])
    )
    new_config_sha256 = _config_sha256(new_config)
    sample_sha256 = _sha256_file(sample_path)
    migrated_state = deepcopy(state)
    exhausted = {
        tuple(item)
        for item in migrated_state.get("exhausted_source_files", [])
        if isinstance(item, list) and len(item) == 4
    }
    exhausted.add(physical_key)
    migrated_state["config_sha256"] = new_config_sha256
    migrated_state["run_fingerprint"] = _run_fingerprint(new_config_sha256, target_bytes)
    migrated_state["exhausted_source_files"] = [list(key) for key in sorted(exhausted)]
    migrated_state["progress"] = {
        "current_category": physical_key[3],
        "current_component": failed_source["component"],
        "current_physical_source_file": physical_key[2],
        "current_physical_source_text_bytes": int(stats["physical_source_bytes"][physical_key]),
        "current_source_target_bytes": failed_target,
        "current_source_text_bytes": int(
            stats["source_bytes"][
                _source_key(new_config["dataset"], failed_source, physical_key[3])
            ]
        ),
        "documents": int(stats["documents"]),
        "text_utf8_bytes": int(stats["text_bytes"]),
    }
    migrated_state.setdefault("migrations", []).append(
        {
            "from_config_sha256": old_config_sha256,
            "reason": (
                "add pinned component-level continuation files after verified shard exhaustion"
            ),
            "sample_sha256": sample_sha256,
            "to_config_sha256": new_config_sha256,
            "verified_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        }
    )
    report = {
        "documents": int(stats["documents"]),
        "exhausted_physical_source": list(physical_key),
        "from_config_sha256": old_config_sha256,
        "sample_sha256": sample_sha256,
        "status": "compatible",
        "text_utf8_bytes": int(stats["text_bytes"]),
        "to_config_sha256": new_config_sha256,
    }
    return report, migrated_state


def _atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-plan", type=Path, default=DEFAULT_OLD_PLAN)
    parser.add_argument("--new-plan", type=Path, default=DEFAULT_NEW_PLAN)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the migrated run state after all compatibility checks pass",
    )
    args = parser.parse_args()
    report, migrated_state = inspect_migration(args.old_plan, args.new_plan, args.output)
    if args.apply:
        state_path = args.output / "run-state.json"
        backup_path = args.output / "run-state.pre-v0.3.json"
        if backup_path.exists():
            raise FileExistsError(f"migration backup already exists: {backup_path}")
        shutil.copy2(state_path, backup_path)
        _atomic_json_write(state_path, migrated_state)
        report["backup"] = str(backup_path)
        report["status"] = "applied"
    else:
        report["status"] = "dry_run_pass"
    print("PASS: tokenizer sample plan migration is compatible")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
