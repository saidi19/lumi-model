#!/usr/bin/env python3
"""Build a raw-text-free benchmark decontamination registry.

Input files are private/local JSONL records with ``benchmark``, ``example_id``,
``split``, and ``text`` fields.  The output contains only SHA-256 signatures and
is safe to review in Git; it cannot reconstruct the benchmark prompts.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
from contextlib import ExitStack
from pathlib import Path

from lumi.data_quality import benchmark_signature_entry, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor-ngram-tokens", type=int, default=13)
    parser.add_argument("--max-anchors", type=int, default=4)
    parser.add_argument("--minimum-anchor-matches", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite benchmark registry: {args.output}")
    if not 5 <= args.anchor_ngram_tokens <= 64:
        raise ValueError("anchor-ngram-tokens must be between 5 and 64")
    if args.max_anchors < 1 or args.minimum_anchor_matches < 1:
        raise ValueError("anchor counts must be positive")

    entries: dict[tuple[str, str], dict] = {}
    input_files = []
    for path in args.input:
        if not path.is_file():
            raise FileNotFoundError(f"benchmark input does not exist: {path}")
        input_files.append(
            {
                "file": path.name,
                "sha256": sha256_file(path),
            }
        )
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                    benchmark = record["benchmark"]
                    example_id = record["example_id"]
                    split = record["split"]
                    text = record["text"]
                except (json.JSONDecodeError, KeyError, TypeError) as error:
                    raise ValueError(f"invalid input at {path}:{line_number}: {error}") from error
                if not all(isinstance(value, str) and value for value in (benchmark, example_id)):
                    raise ValueError(f"invalid benchmark identity at {path}:{line_number}")
                if not isinstance(split, str) or not split:
                    raise ValueError(f"invalid benchmark split at {path}:{line_number}")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"invalid benchmark text at {path}:{line_number}")
                key = (benchmark, example_id)
                if key in entries:
                    raise ValueError(f"duplicate benchmark example: {benchmark}/{example_id}")
                entries[key] = benchmark_signature_entry(
                    benchmark=benchmark,
                    example_id=example_id,
                    split=split,
                    text=text,
                    anchor_ngram_tokens=args.anchor_ngram_tokens,
                    max_anchors=args.max_anchors,
                    minimum_anchor_matches=args.minimum_anchor_matches,
                )

    metadata = {
        "_meta": {
            "anchor_ngram_tokens": args.anchor_ngram_tokens,
            "entries": len(entries),
            "input_files": sorted(input_files, key=lambda item: item["file"]),
            "max_anchors_per_example": args.max_anchors,
            "minimum_anchor_matches": args.minimum_anchor_matches,
            "normalization": "nfkc_casefold_whitespace_v1",
            "raw_text_retained": False,
            "schema_version": 1,
        }
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        if args.output.name.endswith(".jsonl.gz"):
            raw_handle = stack.enter_context(args.output.open("xb"))
            gzip_handle = stack.enter_context(
                gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0)
            )
            handle = stack.enter_context(io.TextIOWrapper(gzip_handle, encoding="utf-8"))
        elif args.output.name.endswith(".jsonl"):
            handle = stack.enter_context(args.output.open("x", encoding="utf-8"))
        else:
            raise ValueError("benchmark registry output must end in .jsonl or .jsonl.gz")
        handle.write(json.dumps(metadata, separators=(",", ":"), sort_keys=True) + "\n")
        for key in sorted(entries):
            handle.write(json.dumps(entries[key], separators=(",", ":"), sort_keys=True) + "\n")
    print(f"PASS: wrote {len(entries)} hashed benchmark signatures to {args.output}")
    print(f"SHA-256: {sha256_file(args.output)}")


if __name__ == "__main__":
    main()
