#!/usr/bin/env python3
"""Independently verify the BillSum writing-closure candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_billsum_writing_curation_v1 import verify_curation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    verification = verify_curation(
        plan_path=args.plan,
        report_path=args.report,
        repository_root=args.repository_root,
        write_output=False,
    )
    print(json.dumps(verification, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
