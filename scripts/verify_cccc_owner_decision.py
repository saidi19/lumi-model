#!/usr/bin/env python3
"""Independently verify Lumi's sealed CCCC owner-decision packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.cccc_decision import verify_cccc_owner_decision_report

ROOT = Path(__file__).parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "cccc-owner-decision-v0.1.json",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    report = verify_cccc_owner_decision_report(
        args.contract,
        args.report,
        args.manifest,
        repository_root=ROOT,
    )
    print("PASS: CCCC decision report hash and deterministic reproduction agree")
    print(json.dumps({"decision": report["decision"], "impact": report["impact"]}, indent=2))


if __name__ == "__main__":
    main()
