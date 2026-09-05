#!/usr/bin/env python3
"""Verify the v0.13 one-row Bengali-script exact-field repair plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_gpt_oss_teacher_bengali_script_field_repair_v13 import build_plan_verification


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    verification = build_plan_verification(report=report, report_file_sha256=file_sha256(args.report))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
