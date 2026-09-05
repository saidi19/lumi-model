#!/usr/bin/env python3
"""Run the authorized zero-inference deterministic multilingual completion."""

from __future__ import annotations

import argparse
import json

from lumi.posttraining_first_party_teacher_canary_v18 import (
    run_deterministic_multilingual_completion,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--artifact-root", required=True)
    arguments = parser.parse_args()
    report = run_deterministic_multilingual_completion(
        plan_path=arguments.plan,
        repository_root=arguments.repository_root,
        artifact_root=arguments.artifact_root,
    )
    print(
        json.dumps(
            {
                "deterministic_materializations": report["generation"][
                    "deterministic_materializations"
                ],
                "local_inference_calls": report["generation"]["calls"],
                "owner_review_records": report["owner_review"]["records"],
                "records": report["audit"]["counts"]["records"],
                "status": report["status"],
                "supervised_tokens": report["audit"]["tokens"][
                    "retained_supervised_tokens"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
