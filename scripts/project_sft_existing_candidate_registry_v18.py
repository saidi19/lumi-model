#!/usr/bin/env python3
"""Project immutable source registry v0.18 and its non-production receipt."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.posttraining_existing_candidate_registry_v18 import (
    build_receipt,
    build_registry,
    json_bytes,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    args = parser.parse_args()
    root = Path(args.repository_root).resolve()
    predecessor = root / "configs/lumi-4b-sft-source-registry-v0.17.json"
    successor = root / "configs/lumi-4b-sft-source-registry-v0.18.json"
    owner = root / "evidence/lumi-4b-sft-existing-candidate-salvage-owner-review-v0.1.json"
    receipt_path = root / "evidence/lumi-4b-sft-existing-candidate-registry-projection-v0.18.json"
    if successor.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite registry v0.18 projection")
    successor.write_bytes(json_bytes(build_registry(predecessor)))
    receipt_path.write_bytes(
        json_bytes(
            build_receipt(
                predecessor_path=predecessor,
                successor_path=successor,
                owner_receipt_path=owner,
            )
        )
    )


if __name__ == "__main__":
    main()

