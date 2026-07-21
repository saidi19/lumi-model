#!/usr/bin/env python3
"""Bind the recorded owner approval to one exact Glue staging release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.glue_production import write_glue_production_stage_authorization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    authorization = write_glue_production_stage_authorization(
        args.release, args.output
    )
    print(
        json.dumps(
            {
                "authorization_sha256": authorization["authorization_sha256"],
                "expires_at": authorization["expires_at"],
                "maximum_usd": authorization["bounds"]["maximum_usd"],
                "status": "authorized_one_stage_run",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
