#!/usr/bin/env python3
"""Verify every blank canary packet without treating it as candidate data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from lumi.posttraining_first_party_zero_inference_canary_packet_v1 import (
    build_blank_rows,
    build_manifest,
    build_packets,
    build_report,
    build_verification,
    load_plan,
    packet_jsonl_bytes,
    sha256_file,
    validate_bound_predecessors,
    validate_manifest,
    validate_report,
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
    report_path = (root / plan["output"]["report_path"]).resolve()
    verification_path = (root / plan["output"]["verification_path"]).resolve()
    if verification_path.exists():
        raise FileExistsError("refusing to overwrite blank packet verification")

    manifest_path = output_dir / "blank-packet-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    rows = build_blank_rows(plan)
    packets = build_packets(rows, plan=plan)
    replay_manifest = build_manifest(packets, plan=plan)
    if manifest != replay_manifest:
        raise ValueError("blank packet manifest replay changed")
    expected_files = {
        f"{packet['packet_id']}.jsonl" for packet in packets
    }
    packet_dir = output_dir / "packets"
    observed_files = {path.name for path in packet_dir.glob("*.jsonl")}
    if observed_files != expected_files:
        raise ValueError("blank packet file inventory changed")
    for packet in packets:
        expected = packet_jsonl_bytes(packet)
        observed = (packet_dir / f"{packet['packet_id']}.jsonl").read_bytes()
        if observed != expected:
            raise ValueError("blank packet bytes changed")
        entry = next(
            item for item in manifest["packets"] if item["packet_id"] == packet["packet_id"]
        )
        if hashlib.sha256(observed).hexdigest() != entry["jsonl_sha256"]:
            raise ValueError("blank packet digest changed")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_report(report)
    replay_report = build_report(
        plan=plan,
        plan_file_sha256=sha256_file(plan_path),
        manifest=manifest,
        manifest_file_sha256=sha256_file(manifest_path),
    )
    if report != replay_report:
        raise ValueError("blank packet report replay changed")
    verification = build_verification(
        plan=plan,
        report=report,
        report_file_sha256=sha256_file(report_path),
    )
    write_json_atomic(verification_path, verification)
    total_bytes = sum(
        path.stat().st_size
        for path in (output_dir, report_path, verification_path)
        if path.is_file()
    ) + sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())
    if total_bytes > plan["limits"]["maximum_private_write_bytes"]:
        raise ValueError("verified blank packet outputs exceed write bound")
    print(
        "PASS: verified 1,000 blank zero-inference canary slots "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
