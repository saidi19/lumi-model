#!/usr/bin/env python3
"""Measure the private safety/tool candidate with the frozen Lumi tokenizer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from lumi.posttraining_first_party import file_sha256
from lumi.posttraining_format import load_chat_template
from lumi.posttraining_safety_tool_tokens import (
    audit_candidate_token_counts,
    build_token_audit_report,
    load_token_audit_plan,
    resolve_plan_binding,
    verify_candidate_evidence,
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path, *, maximum_records: int) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if len(rows) >= maximum_records:
                raise ValueError("safety/tool token-audit maximum record count exceeded")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSON object required at line {line_number}")
            rows.append(value)
    return rows


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
    evidence_root = (root / "evidence").resolve()
    if not output.resolve().is_relative_to(evidence_root):
        raise ValueError("safety/tool token-audit report must stay under evidence/")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite safety/tool token audit: {output}")

    plan = load_token_audit_plan(plan_path, repository_root=root)
    bindings = plan["bindings"]
    evidence_path = resolve_plan_binding(root, bindings["candidate_evidence"])
    candidate_path = resolve_plan_binding(root, bindings["candidate_records"])
    template_path = resolve_plan_binding(root, bindings["chat_template"])
    tokenizer_json = resolve_plan_binding(root, bindings["tokenizer_json"])
    tokenizer_config = resolve_plan_binding(root, bindings["tokenizer_config"])
    if tokenizer_json.parent != tokenizer_config.parent:
        raise ValueError("frozen Lumi tokenizer files are not in one directory")

    evidence = _load_json(evidence_path)
    verify_candidate_evidence(evidence, plan=plan)
    rows = _read_jsonl(candidate_path, maximum_records=plan["limits"]["maximum_records"])
    template = load_chat_template(template_path)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_json.parent,
        local_files_only=True,
    )
    metrics = audit_candidate_token_counts(rows, tokenizer=tokenizer, template=template)
    report = build_token_audit_report(
        metrics=metrics,
        observed_at=plan["observed_at"],
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
    )
    _write_json_atomic(output, report)
    print(
        "PASS: safety/tool Lumi-tokenizer audit "
        f"records={metrics['accepted_records']} "
        f"train_supervised={metrics['by_split']['train']['supervised_tokens']} "
        f"total_supervised={metrics['supervised_tokens']} "
        f"max_serialized={metrics['serialized_length']['maximum']} "
        f"report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
