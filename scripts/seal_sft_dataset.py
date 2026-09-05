#!/usr/bin/env python3
"""Seal an exact Lumi SFT JSONL after every source and record gate passes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from transformers import PreTrainedTokenizerFast

from lumi.data_quality import BenchmarkRegistry
from lumi.posttraining_data import load_sft_policy
from lumi.posttraining_format import load_chat_template, verify_chat_tokenizer
from lumi.posttraining_seal import build_sft_dataset_seal
from lumi.posttraining_sources import load_sft_source_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-binding", type=Path, required=True)
    parser.add_argument("--chat-template", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--tokenizer", type=Path, required=True)
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSON object required at line {line_number}")
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
    if args.manifest.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {args.manifest}")
    policy = load_sft_policy(args.config)
    source_registry = load_sft_source_registry(
        args.repository_root / policy.source_registry_path,
        allowed_licenses=policy.allowed_licenses,
        allowed_languages=policy.allowed_languages,
        allowed_splits=policy.allowed_splits,
        allowed_tasks=frozenset(policy.task_mix),
        expected_sha256=policy.source_registry_sha256,
    )
    template = load_chat_template(args.chat_template)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(args.tokenizer))
    verify_chat_tokenizer(tokenizer, template)
    benchmark_registry = BenchmarkRegistry(
        policy.benchmark_decontamination,
        args.repository_root,
    )
    manifest = build_sft_dataset_seal(
        _load_jsonl(args.input),
        base_binding=_load_json(args.base_binding),
        benchmark_registry=benchmark_registry,
        policy=policy,
        policy_sha256=_file_sha256(args.config),
        source_registry=source_registry,
        template=template,
        tokenizer=tokenizer,
    )
    _write_json_atomic(args.manifest, manifest)
    print(
        f"PASS: sealed {manifest['records']} records / {manifest['input_tokens']} tokens; "
        f"seal={manifest['seal_sha256']}"
    )


if __name__ == "__main__":
    main()
