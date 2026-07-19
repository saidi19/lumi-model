#!/usr/bin/env python3
"""Build the local-only Lumi object-storage and adaptive-acquisition plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from lumi.corpus_storage import build_corpus_storage_report

ROOT = Path(__file__).parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "corpus-object-storage-v0.1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_corpus_storage_report(args.contract, repository_root=ROOT)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(f"{digest}  {args.output.name}\n", encoding="utf-8")
    print("PASS: provider-neutral storage and adaptive-acquisition design is sealed")
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "plan_fingerprint": report["plan_fingerprint"],
                "storage_forecast": report["storage_forecast"],
                "summary": report["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
