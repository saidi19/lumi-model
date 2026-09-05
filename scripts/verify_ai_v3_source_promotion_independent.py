#!/usr/bin/env python3
"""Run the corrected independent AI v0.3 registry verification."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.posttraining_ai_v3_source_promotion import (
    load_ai_v3_source_promotion_plan,
)
from lumi.posttraining_ai_v3_source_promotion_independent import (
    verify_independent_materialization,
    verify_independent_verification_report,
)


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite AI v0.3 verification: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_ai_v3_source_promotion_plan(plan_path, repository_root=root)
    verification = verify_independent_materialization(
        plan_path,
        repository_root=root,
    )
    verify_independent_verification_report(verification)
    output = (root / plan["output"]["verification_path"]).resolve()
    if not output.is_relative_to(root / "evidence"):
        raise ValueError("AI v0.3 independent verification must stay under evidence/")
    _write_json_atomic(output, verification)
    print(
        "PASS: corrected independent AI v0.3 registry verification; "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
