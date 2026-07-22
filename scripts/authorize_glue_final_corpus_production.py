#!/usr/bin/env python3
"""Bind one exact owner approval to the production final-80B Glue release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.glue_final_corpus_production import (
    write_glue_final_corpus_production_authorization,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approval-text", required=True)
    args = parser.parse_args()
    authorization = write_glue_final_corpus_production_authorization(
        args.release,
        args.output,
        approval_text=args.approval_text,
    )
    print(
        json.dumps(
            {
                "authorization_sha256": authorization["authorization_sha256"],
                "maximum_usd": authorization["bounds"]["maximum_usd"],
                "status": "authorized_one_production_final_corpus_run",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
