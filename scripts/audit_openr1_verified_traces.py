#!/usr/bin/env python3
"""Re-audit the exact pinned OpenR1 slice using verified generated traces."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from lumi.data_quality import BenchmarkRegistry
from lumi.posttraining_canary import encode_canary_records
from lumi.posttraining_data import filter_sft_records, load_sft_policy
from lumi.posttraining_format import load_chat_template, verify_chat_tokenizer
from lumi.posttraining_openr1 import (
    build_openr1_audit_report,
    file_sha256,
    load_openr1_audit_plan,
    select_verified_openr1_records,
)
from lumi.posttraining_sources import load_sft_source_registry


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _resolve(root: Path, binding: dict[str, Any]) -> Path:
    return (root / str(binding["path"])).resolve()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    output = args.output if args.output.is_absolute() else root / args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite OpenR1 evidence: {output}")

    plan = load_openr1_audit_plan(plan_path, repository_root=root)
    bindings = plan["bindings"]
    metadata_pre = _load_json(_resolve(root, bindings["metadata_pre"]))
    metadata_post = _load_json(_resolve(root, bindings["metadata_post"]))
    source = plan["source"]
    for metadata in (metadata_pre, metadata_post):
        if metadata.get("id") != source["dataset_id"]:
            raise ValueError("OpenR1 metadata dataset identity changed")
        if metadata.get("sha") != source["revision"]:
            raise ValueError("OpenR1 metadata source revision changed")
        if metadata.get("private") is not False or metadata.get("gated") not in {
            False,
            None,
        }:
            raise ValueError("OpenR1 metadata access boundary changed")
    if metadata_pre != metadata_post:
        raise ValueError("OpenR1 source head changed during the original retrieval")

    raw_path = _resolve(root, bindings["raw_rows"])
    raw_payload = _load_json(raw_path)
    if raw_payload.get("partial") is not False:
        raise ValueError("OpenR1 pinned response is partial")
    wrapped_rows = raw_payload.get("rows")
    if not isinstance(wrapped_rows, list):
        raise ValueError("OpenR1 pinned response omitted rows")
    records, selection_metrics = select_verified_openr1_records(
        wrapped_rows,
        revision=source["revision"],
        maximum_selected_records=plan["limits"]["maximum_selected_records"],
    )

    sft_contract_path = _resolve(root, bindings["sft_contract"])
    policy = load_sft_policy(sft_contract_path)
    source_registry = load_sft_source_registry(
        _resolve(root, bindings["source_registry"]),
        allowed_licenses=policy.allowed_licenses,
        allowed_languages=policy.allowed_languages,
        allowed_splits=policy.allowed_splits,
        allowed_tasks=frozenset(policy.task_mix),
        expected_sha256=policy.source_registry_sha256,
    )
    accepted, filter_report = filter_sft_records(
        records,
        policy=policy,
        benchmark_registry=BenchmarkRegistry(policy.benchmark_decontamination, root),
        source_registry=source_registry,
    )

    tokenizer_json = _resolve(root, bindings["tokenizer_json"])
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_json.parent,
        local_files_only=True,
    )
    template = load_chat_template(_resolve(root, bindings["chat_template"]))
    verify_chat_tokenizer(tokenizer, template)
    _, encoding_report = encode_canary_records(
        accepted,
        tokenizer=tokenizer,
        template=template,
    )

    report = build_openr1_audit_report(
        encoding_report=encoding_report,
        filter_report=filter_report,
        metadata_sha256=bindings["metadata_pre"]["sha256"],
        observed_at=plan["observed_at"],
        plan_sha256=file_sha256(plan_path),
        raw_rows_sha256=bindings["raw_rows"]["sha256"],
        selection_metrics=selection_metrics,
        source_revision=source["revision"],
    )
    _write_json_atomic(output, report)
    print(
        "PASS: OpenR1 verified-trace bounded re-audit completed; "
        f"selected={selection_metrics['selected_records']} "
        f"encoded={encoding_report['accepted']} "
        f"blockers={len(report['remaining_production_blockers'])} "
        f"report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
