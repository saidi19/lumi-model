#!/usr/bin/env python3
"""Merge private human-authoring packets and emit a raw-text-free progress audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_authoring_workflow import (
    validate_authoring_workflow_manifest,
    validate_packet_rows_against_manifest,
)
from lumi.posttraining_first_party import (
    audit_first_party_authoring_rows,
    canonical_json_bytes,
    load_first_party_authoring_plan,
)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
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
    parser.add_argument("--workflow-manifest", type=Path, required=True)
    parser.add_argument("--packet-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    authoring_plan = load_first_party_authoring_plan(args.authoring_plan)
    manifest = validate_authoring_workflow_manifest(
        json.loads(args.workflow_manifest.read_text(encoding="utf-8"))
    )
    all_rows = []
    for entry in manifest["packets"]:
        packet_id = entry["packet_id"]
        rows = _load_jsonl(args.packet_dir / f"{packet_id}.jsonl")
        validate_packet_rows_against_manifest(
            manifest=manifest,
            packet_id=packet_id,
            rows=rows,
        )
        all_rows.extend(rows)
    all_rows.sort(key=lambda row: row["record_id"])
    audit = audit_first_party_authoring_rows(all_rows, plan=authoring_plan)

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
