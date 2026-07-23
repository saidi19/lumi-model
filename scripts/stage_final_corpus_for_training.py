#!/usr/bin/env python3
"""Stage the sealed final corpus from S3 into a resumable local shard cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.production_training import aws_cli_fetch, stage_final_corpus_shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--vocab-size", type=int, default=65_536)
    parser.add_argument("--doc-token-id", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = stage_final_corpus_shards(
        args.manifest,
        args.output,
        fetch=aws_cli_fetch,
        sequence_length=args.sequence_length,
        vocab_size=args.vocab_size,
        doc_token_id=args.doc_token_id,
        expected_manifest_sha256=args.expected_manifest_sha256,
    )
    print(
        json.dumps(
            {
                "build_fingerprint": adapter["build_fingerprint"],
                "manifest_sha256": adapter["source"]["manifest_sha256"],
                "sequences": adapter["statistics"]["sequences"],
                "shards": len(adapter["shards"]),
                "status": "complete",
                "tokens": adapter["statistics"]["tokens"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
