#!/usr/bin/env python3
"""Build and seal Lumi's CCCC owner-decision packet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from lumi.cccc_decision import build_cccc_owner_decision_report

ROOT = Path(__file__).parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "cccc-owner-decision-v0.1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    report = build_cccc_owner_decision_report(args.contract, repository_root=ROOT)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(f"{digest}  {args.output.name}\n", encoding="utf-8")
    print("PASS: CCCC evidence and owner-decision boundary are sealed")
    print(json.dumps({"decision": report["decision"], "impact": report["impact"]}, indent=2))


if __name__ == "__main__":
    main()
