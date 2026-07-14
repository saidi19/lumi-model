#!/usr/bin/env python3
"""Validate a trained tokenizer against Lumi's frozen ID and round-trip contract."""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

from transformers import AutoTokenizer

REPOSITORY_ROOT = Path(__file__).parents[1]
DEFAULT_CONTRACT = REPOSITORY_ROOT / "configs" / "tokenizer-v0.1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer", type=Path, help="local trained-tokenizer directory")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--expected-vocab-size",
        type=int,
        help="override the contract size when checking a comparison prototype",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.contract.open(encoding="utf-8") as handle:
        contract = json.load(handle)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

    expected_vocab_size = args.expected_vocab_size or contract["vocab_size"]
    if len(tokenizer) != expected_vocab_size:
        raise ValueError(
            f"vocabulary mismatch: expected {expected_vocab_size}, got {len(tokenizer)}"
        )
    vocabulary = tokenizer.get_vocab()
    for name, specification in contract["special_tokens"].items():
        token = specification["token"]
        expected_id = specification["id"]
        actual_id = vocabulary.get(token)
        if actual_id != expected_id:
            raise ValueError(
                f"{name} token mismatch: expected {token!r}={expected_id}, got {actual_id}"
            )

    standard_ids = {
        "pad": tokenizer.pad_token_id,
        "bos": tokenizer.bos_token_id,
        "eos": tokenizer.eos_token_id,
        "unk": tokenizer.unk_token_id,
    }
    for name, actual_id in standard_ids.items():
        expected_id = contract["special_tokens"][name]["id"]
        if actual_id != expected_id:
            raise ValueError(f"tokenizer.{name}_token_id must be {expected_id}, got {actual_id}")

    probes = [
        "Hello, Lumi!",
        "Numbers: 123.45 and code: print('ok')",
        "বাংলা পরীক্ষা",
        "日本語のテスト — café",
    ]
    unknown_id = contract["special_tokens"]["unk"]["id"]
    for probe in probes:
        normalized = unicodedata.normalize(contract["normalization"], probe)
        token_ids = tokenizer.encode(normalized, add_special_tokens=False)
        if not token_ids or unknown_id in token_ids:
            raise ValueError(f"tokenizer did not provide byte-complete coverage for {probe!r}")
        decoded = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if decoded != normalized:
            raise ValueError(f"round-trip mismatch: {normalized!r} -> {decoded!r}")

    print("PASS: tokenizer vocabulary, IDs, byte coverage, and round trips")


if __name__ == "__main__":
    main()
