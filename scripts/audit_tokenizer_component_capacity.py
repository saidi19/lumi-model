#!/usr/bin/env python3
"""Measure a pinned component file's accepted-text lower bound."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from lumi.data_pipeline import build_sample
from lumi.development_config import compose_development_config

REPOSITORY_ROOT = Path(__file__).parents[1]
DEFAULT_PLAN = REPOSITORY_ROOT / "configs" / "tokenizer-sample-v0.3.json"
PHYSICAL_FIELDS = {
    "blob_sha1",
    "blob_sha256",
    "compressed_bytes",
    "file",
    "format",
    "url",
    "verified_capacity_is_exhaustive",
    "verified_capacity_text_utf8_bytes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--component", required=True)
    parser.add_argument("--physical-file")
    parser.add_argument("--target-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-forever", action="store_true")
    return parser.parse_args()


def _select_source(config: dict, component: str, physical_file: str | None) -> dict:
    matches = [
        (entry, source)
        for entry in config["mixture"]
        for source in entry["sources"]
        if source["component"] == component
    ]
    if len(matches) != 1:
        raise ValueError(f"component must match exactly one source: {component}")
    entry, source = matches[0]
    selected = deepcopy(source)
    requested_file = physical_file or selected["file"]
    if requested_file != selected["file"]:
        continuations = {
            continuation["file"]: continuation
            for continuation in selected.get("continuation_files", [])
        }
        if requested_file not in continuations:
            raise ValueError(f"physical file is not pinned for {component}: {requested_file}")
        continuation = continuations[requested_file]
        inherited_format = selected.get("format")
        for name in PHYSICAL_FIELDS:
            selected.pop(name, None)
        if inherited_format is not None and "format" not in continuation:
            selected["format"] = inherited_format
        selected.update(continuation)
    selected.pop("continuation_files", None)
    selected.pop("verified_capacity_text_utf8_bytes", None)
    selected.pop("verified_capacity_is_exhaustive", None)
    selected["weight_within_category"] = 1.0
    return {
        "category": entry["category"],
        "source": selected,
    }


def main() -> None:
    args = parse_args()
    if args.target_bytes <= 0:
        raise ValueError("--target-bytes must be positive")
    config = compose_development_config(args.plan, repository_root=REPOSITORY_ROOT)
    selected = _select_source(config, args.component, args.physical_file)
    config["mixture"] = [
        {
            "category": selected["category"],
            "sources": [selected["source"]],
            "weight": 1.0,
        }
    ]
    config.pop("required_capacity_preflight_components", None)
    config["purpose"] += (
        f"; isolated capacity lower-bound audit for {args.component} {selected['source']['file']}"
    )
    config["target_uncompressed_text_bytes"] = args.target_bytes
    manifest = build_sample(
        config,
        args.output,
        resume=args.resume,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
        retry_forever=args.retry_forever,
    )
    physical = manifest["source_files"][0]["physical_files"][0]
    result = {
        "accepted_documents": physical["statistics"]["documents"],
        "accepted_text_utf8_bytes": physical["statistics"]["text_utf8_bytes"],
        "audit_mode": "isolated_physical_source_lower_bound",
        "audited_target_text_utf8_bytes": args.target_bytes,
        "capacity_kind": "lower_bound",
        "component": args.component,
        "dataset_id": physical["dataset_id"],
        "file": physical["file"],
        "revision": physical["revision"],
    }
    for digest_name in ("blob_sha1", "blob_sha256"):
        if digest_name in physical:
            result[digest_name] = physical[digest_name]
    print("PASS: pinned physical-source capacity lower bound reached")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
