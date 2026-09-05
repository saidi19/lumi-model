#!/usr/bin/env python3
"""Build a blank, metadata-only template for bounded human SFT authoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from lumi.posttraining_first_party import (
    build_blank_authoring_template,
    canonical_json_bytes,
    file_sha256,
    load_first_party_authoring_plan,
)


def _write_template(rows: list[dict], path: Path) -> str:
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    plan = load_first_party_authoring_plan(args.config)
    rows = build_blank_authoring_template(plan)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    template_path = args.output_dir / "authoring-template.jsonl"
    template_sha256 = _write_template(rows, template_path)
    manifest = {
        "authorization": {
            "paid_gpu_authorized": False,
            "production_sft_authorized": False,
            "raw_training_text_present": False,
        },
        "plan_sha256": file_sha256(args.config),
        "rows": len(rows),
        "schema_version": "lumi.first-party-authoring-template-manifest.v0.1",
        "skill_counts": dict(sorted(Counter(row["skill_id"] for row in rows).items())),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "template_sha256": template_sha256,
    }
    manifest["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    (args.output_dir / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
