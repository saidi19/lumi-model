#!/usr/bin/env python3
"""Execute the authorized CPU-only Lumi SFT candidate-hygiene repair."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from lumi.posttraining_candidate_hygiene import (
    file_sha256,
    json_file_bytes,
    load_candidate_hygiene_plan,
    rebuild_candidate_hygiene_outputs,
    resolve_output,
)


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite candidate-hygiene output: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
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
    plan = load_candidate_hygiene_plan(plan_path, repository_root=root)
    outputs = rebuild_candidate_hygiene_outputs(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        repository_root=root,
    )
    payloads = {
        "artifact_manifest_path": outputs["artifact_manifest_bytes"],
        "private_safety_candidate_path": outputs["private_safety_candidate_bytes"],
        "successor_registry_path": outputs["successor_registry_bytes"],
        "report_path": json_file_bytes(outputs["report"]),
    }
    for name, payload in payloads.items():
        _write_atomic(resolve_output(root, plan["output"][name], name=name), payload)
    print(
        "PASS: scoped candidate hygiene repaired; "
        f"registry={outputs['report']['bindings']['successor_registry_sha256']} "
        f"candidate={outputs['report']['bindings']['private_safety_candidate_file_sha256']} "
        f"report={outputs['report']['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
