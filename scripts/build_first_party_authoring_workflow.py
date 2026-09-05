#!/usr/bin/env python3
"""Create private blank human-authoring packets and a raw-text-free manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_authoring_workflow import (
    build_authoring_packets,
    build_authoring_workflow_manifest,
    load_authoring_workflow_plan,
)
from lumi.posttraining_first_party import (
    canonical_json_bytes,
    load_first_party_authoring_plan,
)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("authoring template row must be an object")
                rows.append(value)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authoring-plan", type=Path, required=True)
    parser.add_argument("--workflow-plan", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    authoring_plan = load_first_party_authoring_plan(args.authoring_plan)
    workflow_plan = load_authoring_workflow_plan(
        args.workflow_plan,
        repository_root=args.repository_root,
    )
    packets = build_authoring_packets(
        authoring_plan=authoring_plan,
        blank_rows=_load_jsonl(args.template),
        workflow_plan=workflow_plan,
    )
    manifest = build_authoring_workflow_manifest(
        authoring_plan=authoring_plan,
        packets=packets,
        workflow_plan=workflow_plan,
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    packet_dir = args.output_dir / "packets"
    packet_dir.mkdir()
    for packet in packets:
        (packet_dir / f"{packet.packet_id}.jsonl").write_bytes(packet.jsonl_payload)
    (args.output_dir / "workflow-manifest.json").write_bytes(
        canonical_json_bytes(manifest) + b"\n"
    )
    print(
        json.dumps(
            {
                "manifest_sha256": manifest["manifest_sha256"],
                "output_dir": str(args.output_dir),
                "packet_count": manifest["packet_count"],
                "records": manifest["records"],
                "status": manifest["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
