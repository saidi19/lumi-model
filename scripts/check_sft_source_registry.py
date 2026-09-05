#!/usr/bin/env python3
"""Validate Lumi's SFT source registry and emit a raw-data-free report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.posttraining_data import load_sft_policy
from lumi.posttraining_sources import (
    build_sft_source_readiness_report,
    load_sft_source_registry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.report.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.report}")
    policy = load_sft_policy(args.config)
    registry = load_sft_source_registry(
        args.repository_root / policy.source_registry_path,
        allowed_licenses=policy.allowed_licenses,
        allowed_languages=policy.allowed_languages,
        allowed_splits=policy.allowed_splits,
        allowed_tasks=frozenset(policy.task_mix),
        expected_sha256=policy.source_registry_sha256,
    )
    report = build_sft_source_readiness_report(
        registry,
        required_tasks=frozenset(policy.task_mix),
    )
    _write_json_atomic(args.report, report)
    print(
        "PASS: source metadata registry validated; "
        f"production_seal_ready={str(report['production_seal_ready']).lower()}; "
        f"report={report['report_sha256']}"
    )


if __name__ == "__main__":
    main()
