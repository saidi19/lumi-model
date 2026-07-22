#!/usr/bin/env python3
"""Build the non-authorizing Lumi Glue global-dedup package."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from lumi.glue_production_dedup import (
    build_glue_production_global_dedup_package,
    expected_glue_global_dedup_approval,
)

ROOT = Path(__file__).parents[1]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "glue-production-global-dedup-v0.1.json",
    )
    parser.add_argument("--map-release", type=Path, required=True)
    parser.add_argument("--map-checkpoint", type=Path, required=True)
    parser.add_argument("--wave-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--source-commit")
    return parser.parse_args()


def _source_commit(value: str | None) -> str:
    if value:
        return value
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> None:
    args = _args()
    release = build_glue_production_global_dedup_package(
        contract_path=args.contract,
        repository_root=ROOT,
        source_commit=_source_commit(args.source_commit),
        map_release_path=args.map_release,
        map_checkpoint_path=args.map_checkpoint,
        wave_manifest_paths=args.wave_manifest,
        output_directory=args.output_directory,
    )
    print(
        json.dumps(
            {
                "canary_approval_text": expected_glue_global_dedup_approval(
                    release, mode="canary"
                ),
                "canary_maximum_usd": release["jobs"]["canary"][
                    "maximum_compute_usd"
                ],
                "input_index_sha256": release["input_index_sha256"],
                "map_checkpoint_sha256": release["map"]["checkpoint_sha256"],
                "production_approval_text": expected_glue_global_dedup_approval(
                    release, mode="production"
                ),
                "production_maximum_usd": release["jobs"]["production"][
                    "maximum_compute_usd"
                ],
                "release_fingerprint": release["release_fingerprint"],
                "status": "global_dedup_package_ready_not_authorized",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
