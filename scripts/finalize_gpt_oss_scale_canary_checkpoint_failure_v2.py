#!/usr/bin/env python3
"""Write the bound v0.2 aggregate checkpoint-failure evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_gpt_oss_scale_canary_checkpoint_failure_v2 import (
    finalize_checkpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    report, verification = finalize_checkpoint(args.artifact_root)
    print(
        json.dumps(
            {"report": report, "verification": verification},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

