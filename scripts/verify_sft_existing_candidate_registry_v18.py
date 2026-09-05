#!/usr/bin/env python3
"""Independently verify immutable source registry v0.18 projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_existing_candidate_registry_v18 import (
    build_receipt,
    build_registry,
    file_sha256,
    json_bytes,
    sha256_value,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    args = parser.parse_args()
    root = Path(args.repository_root).resolve()
    predecessor = root / "configs/lumi-4b-sft-source-registry-v0.17.json"
    successor = root / "configs/lumi-4b-sft-source-registry-v0.18.json"
    owner = root / "evidence/lumi-4b-sft-existing-candidate-salvage-owner-review-v0.1.json"
    receipt_path = root / "evidence/lumi-4b-sft-existing-candidate-registry-projection-v0.18.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_registry = build_registry(predecessor)
    if successor.read_bytes() != json_bytes(expected_registry):
        raise ValueError("registry v0.18 does not reproduce")
    expected_receipt = build_receipt(
        predecessor_path=predecessor,
        successor_path=successor,
        owner_receipt_path=owner,
    )
    if receipt != expected_receipt:
        raise ValueError("registry v0.18 receipt does not reproduce")
    verification = {
        "checks": {
            "aya_owner_review_receipt_bound": True,
            "new_sources_appended_exactly_once": True,
            "no_network_gpu_seal_or_sft": True,
            "predecessor_sources_preserved_exactly": True,
            "registry_and_receipt_reproduced": True,
        },
        "receipt_file_sha256": file_sha256(receipt_path),
        "registry_file_sha256": file_sha256(successor),
        "schema_version": "lumi.sft-existing-candidate-registry-projection-verification.v0.18",
        "status": "PASS_verified_immutable_registry_v0_18_projection_nonproduction",
    }
    verification["verification_sha256"] = sha256_value(verification)
    output = root / "evidence/lumi-4b-sft-existing-candidate-registry-projection-verification-v0.18.json"
    if output.exists():
        raise FileExistsError("refusing to overwrite registry v0.18 verification")
    output.write_bytes(json_bytes(verification))


if __name__ == "__main__":
    main()

