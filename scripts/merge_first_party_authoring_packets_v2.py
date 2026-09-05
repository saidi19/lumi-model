#!/usr/bin/env python3
"""Merge private v0.2 packets and emit a raw-text-free progress audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_authoring_workflow_v2 import (
    validate_authoring_workflow_v2_manifest,
    validate_packet_rows_against_v2_manifest,
)
from lumi.posttraining_first_party import canonical_json_bytes
from lumi.posttraining_first_party_v2 import (
    audit_first_party_authoring_rows_v2,
    load_first_party_authoring_v2_plan,
)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"packet row {line_number} must be an object: {path.name}")
            rows.append(value)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authoring-plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--workflow-manifest", type=Path, required=True)
    parser.add_argument("--packet-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    authoring_plan, coverage = load_first_party_authoring_v2_plan(
        args.authoring_plan,
        repository_root=args.repository_root,
    )
    manifest = validate_authoring_workflow_v2_manifest(
        json.loads(args.workflow_manifest.read_text(encoding="utf-8"))
    )
    expected_files = {
        f"{entry['packet_id']}.jsonl" for entry in manifest["packets"]
    }
    observed_files = {path.name for path in args.packet_dir.glob("*.jsonl")}
    if observed_files != expected_files:
        raise ValueError("first-party v0.2 packet file inventory changed")

    all_rows: list[dict] = []
    for entry in manifest["packets"]:
        packet_id = entry["packet_id"]
        rows = _load_jsonl(args.packet_dir / f"{packet_id}.jsonl")
        validate_packet_rows_against_v2_manifest(
            manifest=manifest,
            packet_id=packet_id,
            rows=rows,
        )
        all_rows.extend(rows)
    all_rows.sort(key=lambda row: row["record_id"])
    audit = audit_first_party_authoring_rows_v2(
        all_rows,
        plan=authoring_plan,
        coverage=coverage,
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows_payload = b"".join(canonical_json_bytes(row) + b"\n" for row in all_rows)
    (args.output_dir / "merged-authoring-rows.jsonl").write_bytes(rows_payload)
    (args.output_dir / "authoring-progress-audit.json").write_bytes(
        canonical_json_bytes(audit) + b"\n"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
