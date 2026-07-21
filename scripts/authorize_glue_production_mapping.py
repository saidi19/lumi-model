#!/usr/bin/env python3
"""Bind the recorded owner approval to one exact 100-DPU Glue mapping run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.glue_production_mapping import write_glue_production_mapping_authorization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    authorization = write_glue_production_mapping_authorization(
        args.release, args.output, checkpoint_path=args.checkpoint
    )
    print(
        json.dumps(
            {
                "authorization_sha256": authorization["authorization_sha256"],
                "expires_at": authorization["expires_at"],
                "hard_maximum_dpu": authorization["bounds"]["hard_maximum_dpu"],
                "maximum_usd": authorization["bounds"]["maximum_usd"],
                "starting_checkpoint_sequence": authorization["bounds"][
                    "starting_checkpoint_sequence"
                ],
                "starting_checkpoint_sha256": authorization["bounds"][
                    "starting_checkpoint_sha256"
                ],
                "status": "authorized_one_100_dpu_mapping_run",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
