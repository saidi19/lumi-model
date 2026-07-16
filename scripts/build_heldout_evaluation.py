#!/usr/bin/env python3
"""Build a deterministic, leak-resistant Lumi held-out evaluation artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi import build_heldout_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    data = plan["data"]
    selection = plan["selection"]
    manifest = build_heldout_evaluation(
        input_path=args.input,
        tokenizer_path=args.tokenizer,
        training_manifest_path=args.training_manifest,
        output=args.output,
        target_tokens=data["target_tokens"],
        sequence_length=data["sequence_length"],
        sequences_per_shard=data["sequences_per_shard"],
        seed=selection["seed"],
        candidate_fraction_denominator=selection["candidate_fraction_denominator"],
        checkpoint_documents=data["checkpoint_documents"],
        expected_vocab_size=plan["model"]["vocab_size"],
        expected_doc_token_id=plan["model"]["doc_token_id"],
        resume=args.resume,
    )
    stats = manifest["statistics"]
    print(
        f"PASS: held out {stats['tokens']} tokens in {stats['sequences']} sequences "
        f"from {stats['documents_selected']} documents with zero training overlap"
    )


if __name__ == "__main__":
    main()
