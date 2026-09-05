#!/usr/bin/env python3
"""Materialize the private AI v0.3 owner spot-check provenance candidate."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from lumi.posttraining_ai_v3_provenance_materialization import (
    build_ai_v3_provenance_outputs,
    json_file_bytes,
    load_ai_v3_provenance_plan,
)
from lumi.posttraining_candidate_mix_preflight import file_sha256


def _write_bundle(outputs: list[tuple[Path, bytes]]) -> None:
    if any(path.exists() for path, _payload in outputs):
        existing = [str(path) for path, _payload in outputs if path.exists()]
        raise FileExistsError(f"refusing to overwrite AI v0.3 provenance outputs: {existing}")
    temporary: list[tuple[Path, Path]] = []
    try:
        for path, payload in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.tmp")
            with tmp.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.append((tmp, path))
        for tmp, path in temporary:
            tmp.replace(path)
    finally:
        for tmp, _path in temporary:
            if tmp.exists():
                tmp.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_ai_v3_provenance_plan(plan_path, repository_root=root)
    candidate, manifest, report = build_ai_v3_provenance_outputs(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        repository_root=root,
    )
    outputs = [
        ((root / plan["output"]["candidate_path"]).resolve(), candidate),
        (
            (root / plan["output"]["manifest_path"]).resolve(),
            json_file_bytes(manifest),
        ),
        ((root / plan["output"]["report_path"]).resolve(), json_file_bytes(report)),
    ]
    _write_bundle(outputs)
    metrics = report["provenance_materialization"]
    print(
        "PASS: AI v0.3 provenance materialized truthfully; "
        f"records={metrics['records']} sampled={metrics['sampled_records']} "
        f"unsampled={metrics['unsampled_records']} report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
