#!/usr/bin/env python3
"""Build a non-authorizing adaptive Glue mapping recovery package."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from lumi.glue_production_mapping import build_glue_production_mapping_recovery_package

ROOT = Path(__file__).parents[1]


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-package", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--recovered-wave-manifest", type=Path, required=True)
    parser.add_argument("--failure-evidence", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    package = build_glue_production_mapping_recovery_package(
        previous_package=args.previous_package,
        parent_checkpoint_path=args.parent_checkpoint,
        recovered_wave_manifest_path=args.recovered_wave_manifest,
        failure_evidence_path=args.failure_evidence,
        repository_root=ROOT,
        source_commit=_source_commit(args.source_commit),
        output_directory=args.output_directory,
    )
    release = package["release"]
    checkpoint = package["checkpoint"]
    print(
        json.dumps(
            {
                "checkpoint_sequence": checkpoint["checkpoint_sequence"],
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "hard_maximum_dpu": release["job"]["hard_maximum_dpu"],
                "recovered_total_candidate_tokens": release["recovery"][
                    "recovered_total_candidate_tokens"
                ],
                "release_fingerprint": release["release_fingerprint"],
                "status": "verified_recovery_package_not_authorized",
                "target_total_candidate_tokens": release["limits"][
                    "target_total_candidate_tokens"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
