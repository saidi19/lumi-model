#!/usr/bin/env python3
"""Create the immutable successor registry and safety/tool promotion receipt."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from lumi.posttraining_first_party import file_sha256
from lumi.posttraining_source_promotion import (
    build_source_promotion_receipt,
    build_successor_registry,
    json_file_bytes,
    load_promotion_policy_and_predecessor,
    load_source_promotion_plan,
    resolve_promotion_binding,
    validate_successor_registry,
    verify_source_promotion_prerequisites,
    verify_source_promotion_receipt,
)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_source_promotion_plan(plan_path, repository_root=root)
    successor_path = (root / plan["output"]["successor_registry_path"]).resolve()
    receipt_path = (root / plan["output"]["receipt_path"]).resolve()
    if not successor_path.is_relative_to(root / "configs"):
        raise ValueError("successor registry must stay under configs/")
    if not receipt_path.is_relative_to(root / "evidence"):
        raise ValueError("promotion receipt must stay under evidence/")
    for output in (successor_path, receipt_path):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite source-promotion output: {output}")

    bindings = plan["bindings"]
    candidate_evidence = resolve_promotion_binding(
        root, bindings["candidate_evidence"]
    )
    token_audit_path = resolve_promotion_binding(root, bindings["token_audit"])
    owner_receipt_path = resolve_promotion_binding(
        root, bindings["owner_review_receipt"]
    )
    import json

    evidence_value = json.loads(candidate_evidence.read_text(encoding="utf-8"))
    token_value = json.loads(token_audit_path.read_text(encoding="utf-8"))
    owner_value = json.loads(owner_receipt_path.read_text(encoding="utf-8"))
    verify_source_promotion_prerequisites(
        candidate_evidence=evidence_value,
        owner_review_receipt=owner_value,
        plan=plan,
        token_audit=token_value,
    )

    policy, predecessor, predecessor_value = load_promotion_policy_and_predecessor(
        plan=plan,
        repository_root=root,
    )
    successor_value = build_successor_registry(predecessor_value, plan=plan)
    successor_payload = json_file_bytes(successor_value)
    successor_sha256 = hashlib.sha256(successor_payload).hexdigest()
    successor = validate_successor_registry(
        policy=policy,
        predecessor_value=predecessor_value,
        successor_value=successor_value,
        successor_file_sha256=successor_sha256,
    )
    receipt = build_source_promotion_receipt(
        owner_review_receipt=owner_value,
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        predecessor=predecessor,
        successor=successor,
        token_audit=token_value,
    )
    verify_source_promotion_receipt(receipt)

    _write_bytes_atomic(successor_path, successor_payload)
    _write_bytes_atomic(receipt_path, json_file_bytes(receipt))
    print(
        "PASS: safety/tool source registered in immutable candidate registry; "
        f"successor={successor_sha256}; receipt={receipt['receipt_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
