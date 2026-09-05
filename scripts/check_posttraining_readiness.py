#!/usr/bin/env python3
"""Bind local post-training contracts to the independently verified final base backup."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lumi.posttraining_readiness import (
    build_posttraining_readiness_report,
    load_final_base_binding,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--backup-plan", type=Path, required=True)
    parser.add_argument("--backup-receipt", type=Path, required=True)
    parser.add_argument("--backup-verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return dict(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_external_evidence(
    *,
    binding: Mapping[str, Any],
    plan_path: Path,
    receipt_path: Path,
    verification_path: Path,
) -> None:
    backup = binding["s3_backup"]
    expected_hashes = {
        plan_path: backup["plan_sha256"],
        receipt_path: backup["receipt_sha256"],
        verification_path: backup["verification_report_sha256"],
    }
    for path, expected in expected_hashes.items():
        if _file_sha256(path) != expected:
            raise ValueError(f"external evidence SHA-256 changed: {path.name}")

    checkpoint = binding["checkpoint"]
    handoff = binding["handoff"]
    plan = _load_json(plan_path)
    expected_plan = {
        "checkpoint_manifest_sha256": checkpoint["manifest_sha256"],
        "handoff_file_sha256": handoff["file_sha256"],
        "handoff_sha256": handoff["sha256"],
        "run_fingerprint": binding["run_fingerprint"],
        "sequence_cursor": checkpoint["sequence_cursor"],
        "source_commit": binding["source_commit"],
        "source_step": checkpoint["step"],
        "source_tokens": checkpoint["training_tokens"],
        "total_bytes": backup["archive_bytes"],
        "total_objects": backup["archive_objects"],
        "world_size": checkpoint["world_size"],
    }
    if any(plan.get(name) != value for name, value in expected_plan.items()):
        raise ValueError("backup plan does not match the final base binding")
    plan_objects = plan.get("objects")
    if not isinstance(plan_objects, list) or len(plan_objects) != backup["archive_objects"]:
        raise ValueError("backup plan object inventory changed")
    if sum(item.get("bytes", -1) for item in plan_objects if isinstance(item, Mapping)) != backup[
        "archive_bytes"
    ]:
        raise ValueError("backup plan byte inventory changed")
    plan_tokenizer = sorted(
        (
            {
                "bytes": item["bytes"],
                "relative_path": item["relative_path"],
                "sha256": item["sha256"],
            }
            for item in plan_objects
            if isinstance(item, Mapping)
            and str(item.get("relative_path", "")).startswith("tokenizer/")
        ),
        key=lambda item: item["relative_path"],
    )
    if plan_tokenizer != binding["tokenizer"]["files"]:
        raise ValueError("backup plan tokenizer inventory changed")

    receipt = _load_json(receipt_path)
    if receipt.get("status") != "pass" or receipt.get("source_checkpoint_unchanged") is not True:
        raise ValueError("backup receipt did not pass")
    if receipt.get("total_objects") != backup["archive_objects"] or receipt.get(
        "total_bytes"
    ) != backup["archive_bytes"]:
        raise ValueError("backup receipt inventory changed")
    storage = receipt.get("storage")
    if not isinstance(storage, Mapping) or any(
        storage.get(name) != expected
        for name, expected in {
            "bucket": backup["bucket"],
            "server_side_encryption": "AES256",
            "versioning": "Enabled",
        }.items()
    ):
        raise ValueError("backup receipt storage controls changed")
    receipt_bindings = receipt.get("bindings")
    if not isinstance(receipt_bindings, Mapping) or any(
        receipt_bindings.get(name) != value
        for name, value in expected_plan.items()
        if name not in {"total_bytes", "total_objects"}
    ):
        raise ValueError("backup receipt bindings changed")

    verification = _load_json(verification_path)
    expected_verification = {
        "archive_bytes": backup["archive_bytes"],
        "archive_objects": backup["archive_objects"],
        "bucket": backup["bucket"],
        "delete_markers": 0,
        "handoff_sha256": handoff["sha256"],
        "incomplete_multipart_uploads": 0,
        "plan_sha256": backup["plan_sha256"],
        "prefix": backup["prefix"],
        "run_fingerprint": binding["run_fingerprint"],
        "s3_inventory_bytes": backup["inventory_bytes"],
        "s3_inventory_objects": backup["inventory_objects"],
        "s3_versions": backup["inventory_objects"],
        "source_checkpoint_unchanged": True,
        "status": "pass",
    }
    if any(verification.get(name) != value for name, value in expected_verification.items()):
        raise ValueError("independent S3 verification changed")
    verification_receipt = verification.get("receipt")
    if not isinstance(verification_receipt, Mapping) or verification_receipt.get(
        "sha256"
    ) != backup["receipt_sha256"]:
        raise ValueError("independent S3 receipt binding changed")


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite readiness report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    root = args.repository_root.resolve()
    binding = load_final_base_binding(root / "configs/lumi-4b-final-base-binding-v0.1.json")
    _verify_external_evidence(
        binding=binding,
        plan_path=args.backup_plan,
        receipt_path=args.backup_receipt,
        verification_path=args.backup_verification,
    )
    report = build_posttraining_readiness_report(
        base_binding=binding,
        chat_template=_load_json(root / "configs/lumi-4b-chat-template-v0.1.json"),
        evaluation_plan=_load_json(root / "configs/lumi-4b-posttraining-eval-v0.1.json"),
        sft_policy=_load_json(root / "configs/lumi-4b-sft-data-contract-v0.1.json"),
    )
    report["external_evidence"] = {
        "backup_plan_sha256": _file_sha256(args.backup_plan),
        "backup_receipt_sha256": _file_sha256(args.backup_receipt),
        "backup_verification_sha256": _file_sha256(args.backup_verification),
    }
    report_without_hash = dict(report)
    report_without_hash.pop("report_sha256")
    report["report_sha256"] = hashlib.sha256(
        json.dumps(
            report_without_hash,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    _write_json_atomic(args.output, report)
    print(
        "PASS: final 80B base is bound; SFT contracts are ready; "
        f"next_gate={report['next_gate']}; report={report['report_sha256']}"
    )


if __name__ == "__main__":
    main()
