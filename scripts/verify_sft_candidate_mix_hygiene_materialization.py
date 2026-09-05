#!/usr/bin/env python3
"""Independently verify exact private SFT hygiene materialization outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_candidate_mix_hygiene_materialization import (
    build_hygiene_materialization_verification,
    json_file_bytes,
    load_hygiene_materialization_plan,
)
from lumi.posttraining_candidate_mix_preflight import file_sha256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan_path = args.plan.resolve()
    plan = load_hygiene_materialization_plan(plan_path, repository_root=root)
    receipt_path = root / plan["output"]["receipt_path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    verification = build_hygiene_materialization_verification(
        plan=plan,
        receipt=receipt,
        receipt_file_sha256=file_sha256(receipt_path),
        repository_root=root,
    )
    output = root / plan["output"]["verification_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output_bytes = json_file_bytes(verification)
    if output.exists() and output.read_bytes() != output_bytes:
        raise ValueError("refusing to overwrite changed verification output")
    output.write_bytes(output_bytes)
    print(output)
    print(verification["verification_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

