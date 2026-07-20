#!/usr/bin/env python3
"""Build Lumi's local-only production EMR resume plan and checkpoint zero."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.emr_serverless_production import (
    build_emr_production_plan,
    initial_production_checkpoint,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    return parser.parse_args()


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = _args()
    plan = build_emr_production_plan(
        args.contract,
        repository_root=args.repository_root,
        source_commit=args.source_commit,
    )
    checkpoint = initial_production_checkpoint(plan)
    _write(args.output, plan)
    _write(args.checkpoint, checkpoint)
    print(
        json.dumps(
            {
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "plan_fingerprint": plan["plan_fingerprint"],
                "production_tokens_remaining": 120_000_000_000
                - checkpoint["total_accepted_tokens"],
                "status": plan["decision"]["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
