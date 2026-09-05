#!/usr/bin/env python3
"""Create private message-native human-authoring packets for Lumi SFT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_authoring_workflow_v2 import (
    build_authoring_packets_v2,
    build_authoring_workflow_v2_manifest,
    load_authoring_workflow_v2_plan,
)
from lumi.posttraining_first_party import canonical_json_bytes
from lumi.posttraining_first_party_v2 import (
    build_blank_authoring_template_v2,
    load_first_party_authoring_v2_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authoring-plan", type=Path, required=True)
    parser.add_argument("--workflow-plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    authoring_plan, coverage = load_first_party_authoring_v2_plan(
        args.authoring_plan,
        repository_root=args.repository_root,
    )
    workflow_plan = load_authoring_workflow_v2_plan(
        args.workflow_plan,
        repository_root=args.repository_root,
    )
    packets = build_authoring_packets_v2(
        authoring_plan=authoring_plan,
        coverage=coverage,
        blank_rows=build_blank_authoring_template_v2(authoring_plan, coverage),
        workflow_plan=workflow_plan,
    )
    manifest = build_authoring_workflow_v2_manifest(
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
