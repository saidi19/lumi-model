#!/usr/bin/env python3
"""Compare Lumi tokenizer prototypes on one validated data sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from lumi.data_pipeline import validate_sample

PROBES = {
    "bangla": "বাংলা ভাষায় একটি ছোট পরীক্ষা: আমি আজ একটি মডেল বানাচ্ছি।",
    "code": "def fibonacci(n: int) -> list[int]:\n    return [0, 1] if n <= 2 else []",
    "english": "Lumi should answer carefully, cite evidence, and admit uncertainty.",
    "japanese": "日本語のトークナイザー効率を確認します。",
    "math": "For x ∈ ℝ, solve x² − 5x + 6 = 0; therefore x ∈ {2, 3}.",
    "structured": '{"model":"Lumi","parameters":4000000000,"active":true}',
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_tokenizer(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("tokenizer must use LABEL=/path/to/tokenizer")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("tokenizer label and path cannot be empty")
    return label, Path(raw_path)


def _metrics(counter: dict[str, int]) -> dict[str, float | int]:
    tokens = counter["tokens"]
    text_bytes = counter["text_bytes"]
    words = counter["words"]
    return {
        "bytes_per_token": round(text_bytes / tokens, 6),
        "documents": counter["documents"],
        "text_utf8_bytes": text_bytes,
        "tokens": tokens,
        "tokens_per_word": round(tokens / words, 6),
        "words": words,
    }


def _count_tokens(tokenizer: Any, text: str) -> int:
    return len(tokenizer.backend_tokenizer.encode(text, add_special_tokens=False).ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        action="append",
        required=True,
        type=_parse_tokenizer,
        help="repeat LABEL=/path/to/tokenizer for every prototype",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-documents", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite comparison output: {args.output}")
    if args.max_documents is not None and args.max_documents < 1:
        raise ValueError("max-documents must be positive")
    sample_validation = validate_sample(args.sample)
    sample_path = args.sample / "sample.jsonl"

    tokenizers = {}
    tokenizer_results = {}
    for label, path in args.tokenizer:
        if label in tokenizers:
            raise ValueError(f"duplicate tokenizer label: {label}")
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        tokenizer_file = path / "tokenizer.json"
        tokenizers[label] = tokenizer
        tokenizer_results[label] = {
            "artifact": {
                "path": str(path.resolve()),
                "tokenizer_json_sha256": _sha256_file(tokenizer_file),
                "vocab_size": len(tokenizer),
            },
            "category_counters": defaultdict(
                lambda: {"documents": 0, "text_bytes": 0, "tokens": 0, "words": 0}
            ),
            "overall_counter": {"documents": 0, "text_bytes": 0, "tokens": 0, "words": 0},
        }

    with sample_path.open(encoding="utf-8") as handle:
        for document_index, line in enumerate(handle):
            if args.max_documents is not None and document_index >= args.max_documents:
                break
            record = json.loads(line)
            text = record["text"]
            category = record["provenance"]["category"]
            text_bytes = len(text.encode("utf-8"))
            words = len(re.findall(r"\S+", text))
            for label, tokenizer in tokenizers.items():
                tokens = _count_tokens(tokenizer, text)
                for counter in (
                    tokenizer_results[label]["overall_counter"],
                    tokenizer_results[label]["category_counters"][category],
                ):
                    counter["documents"] += 1
                    counter["text_bytes"] += text_bytes
                    counter["tokens"] += tokens
                    counter["words"] += words

    output_tokenizers = {}
    for label, result in tokenizer_results.items():
        probe_metrics = {}
        for probe_name, text in PROBES.items():
            token_count = _count_tokens(tokenizers[label], text)
            probe_metrics[probe_name] = {
                "text_utf8_bytes": len(text.encode("utf-8")),
                "tokens": token_count,
            }
        output_tokenizers[label] = {
            "artifact": result["artifact"],
            "categories": {
                category: _metrics(counter)
                for category, counter in sorted(result["category_counters"].items())
            },
            "overall": _metrics(result["overall_counter"]),
            "probes": probe_metrics,
        }

    report = {
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "sample": {
            "documents": sample_validation["documents"],
            "output_sha256": sample_validation["sha256"],
            "path": str(args.sample.resolve()),
            "text_utf8_bytes": sample_validation["text_utf8_bytes"],
        },
        "schema_version": 1,
        "tokenizers": output_tokenizers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        f"PASS: compared {len(tokenizers)} tokenizers on "
        f"{sample_validation['documents']} documents"
    )
    for label, result in output_tokenizers.items():
        print(
            f"{label}: vocab={result['artifact']['vocab_size']} "
            f"bytes/token={result['overall']['bytes_per_token']} "
            f"tokens/word={result['overall']['tokens_per_word']}"
        )


if __name__ == "__main__":
    main()
