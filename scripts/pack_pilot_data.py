#!/usr/bin/env python3
"""Pack validated real JSONL text into fixed-length Lumi pilot shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi import LumiConfig
from lumi.packed_data import build_packed_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=Path("configs/lumi-150m.json"))
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--sequences-per-shard", type=int, default=512)
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--checkpoint-documents", type=int, default=1_000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.model_config.open(encoding="utf-8") as handle:
        config = LumiConfig(**json.load(handle))
    sequence_length = args.sequence_length or config.max_position_embeddings
    if sequence_length > config.max_position_embeddings:
        raise ValueError(
            f"sequence length {sequence_length} exceeds model maximum "
            f"{config.max_position_embeddings}"
        )
    manifest = build_packed_dataset(
        input_path=args.input,
        tokenizer_path=args.tokenizer,
        output=args.output,
        sequence_length=sequence_length,
        sequences_per_shard=args.sequences_per_shard,
        target_tokens=args.target_tokens,
        text_key=args.text_key,
        max_documents=args.max_documents,
        checkpoint_documents=args.checkpoint_documents,
        expected_vocab_size=config.vocab_size,
        expected_doc_token_id=config.doc_token_id,
        resume=args.resume,
    )
    statistics = manifest["statistics"]
    print(
        "PASS: packed "
        f"{statistics['tokens']} tokens in {statistics['sequences']} sequences "
        f"across {len(manifest['shards'])} shards"
    )


if __name__ == "__main__":
    main()
