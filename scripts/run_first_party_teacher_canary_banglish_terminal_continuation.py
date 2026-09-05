#!/usr/bin/env python3
"""Run the authorized Banglish-terminal-normalized local teacher continuation."""

from __future__ import annotations

import argparse
import json

from lumi.posttraining_first_party_teacher_canary_v11 import (
    run_banglish_terminal_continuation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--artifact-root", required=True)
    arguments = parser.parse_args()
    report = run_banglish_terminal_continuation(
        plan_path=arguments.plan,
        repository_root=arguments.repository_root,
        artifact_root=arguments.artifact_root,
    )
    print(
        json.dumps(
            {
                "additional_calls": report["schema_completion"]["additional_calls_used"],
                "banglish_terminal_appends": report["schema_completion"][
                    "banglish_terminal_appends"
                ],
                "owner_review_records": report["owner_review"]["records"],
                "records": report["audit"]["counts"]["records"],
                "status": report["status"],
                "supervised_tokens": report["audit"]["tokens"]["retained_supervised_tokens"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
