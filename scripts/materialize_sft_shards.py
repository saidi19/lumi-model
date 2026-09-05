#!/usr/bin/env python3
"""Materialize an exact sealed Lumi SFT JSONL into raw-text-free integer shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import PreTrainedTokenizerFast

from lumi.posttraining_format import load_chat_template, verify_chat_tokenizer
from lumi.posttraining_shards import materialize_sft_shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat-template", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--maximum-tokens-per-shard", type=int, default=4_194_304)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    template = load_chat_template(args.chat_template)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(args.tokenizer))
    verify_chat_tokenizer(tokenizer, template)
    manifest = materialize_sft_shards(
        _load_jsonl(args.input),
        output=args.output,
        seal=_load_json(args.seal),
        template=template,
        tokenizer=tokenizer,
        maximum_tokens_per_shard=args.maximum_tokens_per_shard,
    )
    print(
        f"PASS: materialized {manifest['records']} records / {manifest['tokens']} tokens; "
        f"manifest={manifest['manifest_sha256']}"
    )


if __name__ == "__main__":
    main()
