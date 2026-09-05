#!/usr/bin/env python3
"""Run factual pass two for the immutable v0.23 candidate registry successor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_ai_gap_source_registration_v1 import verify_registration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--plan", type=Path, default=Path("configs/lumi-4b-sft-ai-gap-source-registration-v0.1.json"))
    parser.add_argument("--successor", type=Path, default=Path("configs/lumi-4b-sft-source-registry-v0.23.json"))
    parser.add_argument("--receipt", type=Path, default=Path("evidence/lumi-4b-sft-ai-gap-source-registration-v0.1.json"))
    parser.add_argument("--verification", type=Path, default=Path("evidence/lumi-4b-sft-ai-gap-source-registration-verification-v0.1.json"))
    args = parser.parse_args()
    root = args.repository_root.resolve()
    resolve = lambda path: path if path.is_absolute() else root / path
    output = resolve(args.verification)
    if output.exists():
        raise FileExistsError("versioned verification exists; overwrite forbidden")
    result = verify_registration(
        plan_path=resolve(args.plan), repository_root=root,
        successor_path=resolve(args.successor), receipt_path=resolve(args.receipt),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

