#!/usr/bin/env python3
"""Run the authorized schema-constrained local teacher canary completion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_first_party_teacher_canary_v4 import run_schema_completion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    arguments = parser.parse_args()
    report = run_schema_completion(
        plan_path=arguments.plan,
        repository_root=arguments.repository_root,
        artifact_root=arguments.artifact_root,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "records": report["audit"]["counts"]["records"],
                "supervised_tokens": report["audit"]["tokens"]["retained_supervised_tokens"],
                "additional_calls": report["schema_completion"]["additional_calls_used"],
                "owner_review_records": report["owner_review"]["records"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
