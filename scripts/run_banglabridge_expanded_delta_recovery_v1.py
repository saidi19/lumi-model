#!/usr/bin/env python3
"""Run the bounded local BanglaBridge expanded-delta recovery."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.posttraining_banglabridge_expanded_delta_recovery_v1 import (
    build_recovery,
    json_bytes,
    load_plan,
)
from lumi.posttraining_candidate_mix_preflight import file_sha256


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite expanded-delta output: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_plan(plan_path.resolve(), root)
    output = plan["output"]
    destinations = {
        "candidate": root / output["candidate_path"],
        "delta_ledger": root / output["delta_ledger_path"],
        "owner_review_html": root / output["owner_review_html_path"],
        "owner_review_packet": root / output["owner_review_packet_path"],
        "report": root / output["report_path"],
    }
    temporary_paths = {
        path.with_name(f".{path.name}.tmp") for path in destinations.values()
    }
    if any(path.exists() for path in (*destinations.values(), *temporary_paths)):
        raise FileExistsError("refusing to overwrite expanded-delta output set")
    artifacts, report = build_recovery(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        repository_root=root,
    )
    report_payload = json_bytes(report)
    total = sum(len(value) for value in artifacts.values()) + len(report_payload)
    if total > plan["limits"]["maximum_private_write_bytes"]:
        raise ValueError("expanded-delta private-write cap exceeded")
    _write_atomic(destinations["candidate"], artifacts["candidate"])
    _write_atomic(destinations["delta_ledger"], artifacts["delta_ledger"])
    _write_atomic(destinations["owner_review_packet"], artifacts["owner_review_packet"])
    _write_atomic(destinations["owner_review_html"], artifacts["owner_review_html"])
    _write_atomic(destinations["report"], report_payload)
    print(
        json.dumps(
            {
                "accepted_records": report["audit"]["accepted_records"],
                "candidate_sha256": report["successor"]["sha256"],
                "delta_records": report["delta"]["records"],
                "owner_review_records": report["owner_review"]["packet_records"],
                "report_sha256": report["report_sha256"],
                "status": report["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
