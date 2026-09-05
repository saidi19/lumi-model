#!/usr/bin/env python3
"""Run the normal SFT filter against an in-memory v0.3 candidate source entry."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from lumi.data_quality import BenchmarkRegistry
from lumi.posttraining_ai_assisted_v3 import load_ai_assisted_v3_plan
from lumi.posttraining_data import filter_sft_records, load_sft_policy
from lumi.posttraining_sources import SFTSource, SFTSourceRegistry, load_sft_source_registry


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        values = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(value, dict) for value in values):
        raise ValueError("candidate JSONL must contain objects")
    return values


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        for value in values:
            handle.write(
                json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _private(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to((root / "artifacts").resolve()):
        raise ValueError(f"candidate filter output must remain under artifacts/: {path}")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--accepted-output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    report_path = _private(args.report, root)
    accepted_path = _private(args.accepted_output, root)
    if report_path.exists() or accepted_path.exists():
        raise FileExistsError("refusing to overwrite candidate filter outputs")

    plan, _ = load_ai_assisted_v3_plan(args.plan, repository_root=root)
    audit = _read_json(args.audit)
    revision = audit.get("candidate_source_revision")
    if not isinstance(revision, str) or len(revision) != 64:
        raise ValueError("candidate source revision is invalid")
    policy = load_sft_policy(args.config)
    registry = load_sft_source_registry(
        root / policy.source_registry_path,
        allowed_licenses=policy.allowed_licenses,
        allowed_languages=policy.allowed_languages,
        allowed_splits=policy.allowed_splits,
        allowed_tasks=frozenset(policy.task_mix),
        expected_sha256=policy.source_registry_sha256,
    )
    source = SFTSource.from_mapping(
        {
            "access": "first_party",
            "adapter": "lumi_ai_assisted_chat_v0.3",
            "blockers": [],
            "dataset_id": plan["source_binding"]["dataset_id"],
            "evidence_urls": [],
            "languages": ["bn", "en", "multilingual"],
            "license_id": plan["source_binding"]["license_id"],
            "origin": "synthetic",
            "production_blockers": [
                "dataset_level_owner_spot_check_not_complete",
                "candidate_is_a_small_behavior_pilot_not_the_complete_sft_mix",
            ],
            "production_eligible": False,
            "repository_url": None,
            "revision": revision,
            "source_id": plan["source_binding"]["source_id"],
            "splits": ["train", "validation"],
            "status": "candidate_eligible",
            "task_categories": [
                "conversation",
                "multilingual_bengali",
                "reasoning_math_science",
                "tool_api",
            ],
        },
        allowed_licenses=policy.allowed_licenses,
        allowed_languages=policy.allowed_languages,
        allowed_splits=policy.allowed_splits,
        allowed_tasks=frozenset(policy.task_mix),
    )
    if source.source_id in registry.by_id:
        raise ValueError("v0.3 candidate source unexpectedly exists in the production registry")
    candidate_registry = SFTSourceRegistry(
        observed_at=registry.observed_at,
        registry_sha256=registry.registry_sha256,
        sources=(*registry.sources, source),
    )
    accepted, report = filter_sft_records(
        _read_jsonl(args.input),
        policy=policy,
        benchmark_registry=BenchmarkRegistry(policy.benchmark_decontamination, root),
        source_registry=candidate_registry,
    )
    _write_jsonl(accepted_path, accepted)
    _write_json(report_path, report)
    print(
        f"PASS: accepted={report['accepted']}/{report['total']} "
        f"rejected={report['rejected']} report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
