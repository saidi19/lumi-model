#!/usr/bin/env python3
"""Run the authorized 100-row gpt-oss systemic repair pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_gpt_oss_scale_canary_systemic_pilot_v3 import run_systemic_pilot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--ssh-key", type=Path, required=True)
    parser.add_argument("--known-hosts", type=Path, required=True)
    args = parser.parse_args()
    report = run_systemic_pilot(
        plan_path=args.plan.resolve(),
        authorization_path=args.authorization.resolve(),
        runtime_receipt_path=args.runtime_receipt.resolve(),
        repository_root=args.repository_root.resolve(),
        artifact_root=args.artifact_root.resolve(),
        ssh_key=args.ssh_key.resolve(),
        known_hosts=args.known_hosts.resolve(),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

