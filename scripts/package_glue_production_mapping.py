#!/usr/bin/env python3
"""Build the non-authorizing Lumi Glue 100-DPU mapping package."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from lumi.glue_production_mapping import build_glue_production_mapping_package

ROOT = Path(__file__).parents[1]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "glue-production-150b-v0.1.json",
    )
    parser.add_argument("--lingua-wheel-pin", type=Path, required=True)
    parser.add_argument("--stage-release", type=Path, required=True)
    parser.add_argument("--stage-result", type=Path, required=True)
    parser.add_argument("--stage-verification", type=Path, required=True)
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
    release = build_glue_production_mapping_package(
        contract_path=args.contract,
        repository_root=ROOT,
        source_commit=_source_commit(args.source_commit),
        lingua_wheel_pin_path=args.lingua_wheel_pin,
        stage_release_path=args.stage_release,
        stage_result_path=args.stage_result,
        stage_verification_path=args.stage_verification,
        output_directory=args.output_directory,
    )
    print(
        json.dumps(
            {
                "hard_maximum_dpu": release["job"]["hard_maximum_dpu"],
                "maximum_compute_usd": release["limits"]["maximum_compute_usd"],
                "release_fingerprint": release["release_fingerprint"],
                "status": "mapping_package_ready_not_authorized",
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
