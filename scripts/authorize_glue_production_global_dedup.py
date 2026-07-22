#!/usr/bin/env python3
"""Bind one exact owner approval to a Glue global-dedup run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.glue_production_dedup import write_glue_global_dedup_authorization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--mode", choices=("canary", "production"), required=True)
    parser.add_argument("--approval-text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    authorization = write_glue_global_dedup_authorization(
        args.release,
        args.output,
        mode=args.mode,
        approval_text=args.approval_text,
    )
    print(
        json.dumps(
            {
                "authorization_sha256": authorization["authorization_sha256"],
                "expires_at": authorization["expires_at"],
                "hard_maximum_dpu": authorization["bounds"]["hard_maximum_dpu"],
                "maximum_runtime_minutes": authorization["bounds"][
                    "maximum_runtime_minutes"
                ],
                "maximum_usd": authorization["bounds"]["maximum_usd"],
                "mode": authorization["mode"],
                "status": f"authorized_one_{authorization['mode']}_global_dedup_run",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
