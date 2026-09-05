#!/usr/bin/env python3
"""Validate/filter a JSONL SFT candidate set without logging raw text."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.data_quality import BenchmarkRegistry
from lumi.posttraining_data import filter_sft_records, load_sft_policy
from lumi.posttraining_sources import load_sft_source_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--accepted-output", type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at line {line_number}") from error
            records.append(value)
    return records


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
    if args.accepted_output is not None and args.accepted_output.exists():
        raise FileExistsError(f"refusing to overwrite accepted output: {args.accepted_output}")

    policy = load_sft_policy(args.config)
    registry = BenchmarkRegistry(policy.benchmark_decontamination, args.repository_root)
    source_registry = load_sft_source_registry(
        args.repository_root / policy.source_registry_path,
        allowed_licenses=policy.allowed_licenses,
        allowed_languages=policy.allowed_languages,
        allowed_splits=policy.allowed_splits,
        allowed_tasks=frozenset(policy.task_mix),
        expected_sha256=policy.source_registry_sha256,
    )
    accepted, report = filter_sft_records(
        _read_jsonl(args.input),
        policy=policy,
        benchmark_registry=registry,
        source_registry=source_registry,
    )
    if args.accepted_output is not None:
        args.accepted_output.parent.mkdir(parents=True, exist_ok=True)
        with args.accepted_output.open("x", encoding="utf-8") as handle:
            for record in accepted:
                handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
    _write_json_atomic(args.report, report)
    print(
        f"PASS: accepted {report['accepted']}/{report['total']} records; "
        f"report={report['report_sha256']}"
    )


if __name__ == "__main__":
    main()
