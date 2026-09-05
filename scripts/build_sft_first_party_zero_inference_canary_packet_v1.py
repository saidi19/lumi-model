#!/usr/bin/env python3
"""Materialize exactly 1,000 blank private canary authoring slots."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.posttraining_first_party_zero_inference_canary_packet_v1 import (
    build_blank_rows,
    build_manifest,
    build_packets,
    build_report,
    load_plan,
    packet_jsonl_bytes,
    sha256_file,
    validate_bound_predecessors,
    write_json_atomic,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_plan(plan_path.resolve(), repository_root=root)
    validate_bound_predecessors(plan, repository_root=root)

    output_dir = (root / plan["output"]["private_packet_directory"]).resolve()
    artifact_root = (root / "artifacts").resolve()
    if not output_dir.is_relative_to(artifact_root):
        raise ValueError("private blank packets escaped artifacts/")
    report_path = (root / plan["output"]["report_path"]).resolve()
    if not report_path.is_relative_to((root / "evidence").resolve()):
        raise ValueError("blank packet report escaped evidence/")
    if output_dir.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite blank canary packet attempt")

    rows = build_blank_rows(plan)
    packets = build_packets(rows, plan=plan)
    manifest = build_manifest(packets, plan=plan)
    packet_dir = output_dir / "packets"
    packet_dir.mkdir(parents=True, exist_ok=False)
    for packet in packets:
        (packet_dir / f"{packet['packet_id']}.jsonl").write_bytes(
            packet_jsonl_bytes(packet)
        )
    manifest_path = output_dir / "blank-packet-manifest.json"
    write_json_atomic(manifest_path, manifest)
    report = build_report(
        plan=plan,
        plan_file_sha256=sha256_file(plan_path),
        manifest=manifest,
        manifest_file_sha256=sha256_file(manifest_path),
    )
    write_json_atomic(report_path, report)

    private_bytes = sum(path.stat().st_size for path in output_dir.rglob("*" ) if path.is_file())
    total_bytes = private_bytes + report_path.stat().st_size
    if total_bytes > plan["limits"]["maximum_private_write_bytes"]:
        raise ValueError("blank packet preparation exceeded private-write bound")
    print(
        "PASS: materialized 1,000 blank zero-inference canary slots "
        f"packets={manifest['packet_count']} bytes={total_bytes} "
        f"report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
