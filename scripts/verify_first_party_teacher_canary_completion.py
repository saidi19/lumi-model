#!/usr/bin/env python3
"""Independently verify the result-preserving teacher canary completion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_first_party_teacher_canary_v3 import verify_canary_completion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    arguments = parser.parse_args()
    verification = verify_canary_completion(
        plan_path=arguments.plan,
        repository_root=arguments.repository_root,
        artifact_root=arguments.artifact_root,
    )
    print(
        json.dumps(
            {
                "status": verification["status"],
                "report_file_sha256": verification["report_file_sha256"],
                "candidate_internal_sha256": verification["candidate_internal_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
