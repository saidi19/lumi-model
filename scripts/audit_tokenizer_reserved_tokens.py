#!/usr/bin/env python3
"""Audit a sealed tokenizer sample for literal Lumi control-token collisions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from lumi.data_pipeline import validate_sample
from lumi.tokenizer_gate import count_reserved_literals, reserved_token_literals

REPOSITORY_ROOT = Path(__file__).parents[1]
DEFAULT_CONTRACT = REPOSITORY_ROOT / "configs" / "tokenizer-v0.1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-documents", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit output: {args.output}")
    if args.max_documents is not None and args.max_documents < 1:
        raise ValueError("max-documents must be positive")
    sample = validate_sample(args.sample)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    literals = reserved_token_literals(contract)
    counts: Counter[str] = Counter()
    affected_documents = 0
    documents = 0
    with (args.sample / "sample.jsonl").open(encoding="utf-8") as handle:
        for document_index, line in enumerate(handle):
            if args.max_documents is not None and document_index >= args.max_documents:
                break
            record = json.loads(line)
            document_counts = count_reserved_literals(record["text"], literals)
            if document_counts:
                affected_documents += 1
                counts.update(document_counts)
            documents += 1
    status = "pass" if not counts else "fail"
    report = {
        "affected_documents": affected_documents,
        "contract": str(args.contract.resolve()),
        "counts": {name: counts.get(name, 0) for name in literals},
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "documents_scanned": documents,
        "max_documents": args.max_documents,
        "sample": {
            "documents": sample["documents"],
            "output_sha256": sample["sha256"],
            "text_utf8_bytes": sample["text_utf8_bytes"],
        },
        "schema_version": 1,
        "status": status,
        "total_literal_occurrences": sum(counts.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        f"{status.upper()}: scanned {documents} documents; "
        f"reserved literal occurrences={sum(counts.values())}"
    )
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
