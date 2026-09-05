#!/usr/bin/env python3
"""Create the private-research non-production SFT data contract v0.2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from lumi.posttraining_private_research_data_contract_v2 import (
    _load_object,
    _write_exclusive,
    build_contract,
    build_report,
    file_sha256,
    json_file_bytes,
    load_plan,
    output_paths,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--repository-root", default=".")
    args = parser.parse_args()
    root = Path(args.repository_root).resolve()
    plan_path = Path(args.plan).resolve()
    try:
        plan = load_plan(plan_path, repository_root=root)
        contract_path, report_path, _ = output_paths(plan, root)
        predecessor_path = (root / plan["bindings"]["predecessor_data_contract"]["path"]).resolve()
        predecessor = _load_object(predecessor_path)
        contract = build_contract(predecessor)
        contract_payload = json_file_bytes(contract)
        projected_contract_sha256 = hashlib.sha256(contract_payload).hexdigest()
        if projected_contract_sha256 == file_sha256(predecessor_path):
            raise ValueError("private-research successor did not change")
        report = build_report(
            plan=plan,
            plan_file_sha256=file_sha256(plan_path),
            contract=contract,
            contract_file_sha256=projected_contract_sha256,
            repository_root=root,
        )
        report_payload = json_file_bytes(report)
        if (
            len(contract_payload) + len(report_payload)
            > plan["limits"]["maximum_private_write_bytes"]
        ):
            raise ValueError("private-research contract outputs exceed write ceiling")
        _write_exclusive(contract_path, contract_payload)
        if file_sha256(contract_path) != projected_contract_sha256:
            raise ValueError("private-research contract write verification failed")
        _write_exclusive(report_path, report_payload)
        print(
            json.dumps(
                {
                    "candidate_row_reads": 0,
                    "data_contract_file_sha256": projected_contract_sha256,
                    "production_eligible": False,
                    "report_file_sha256": file_sha256(report_path),
                    "report_internal_sha256": report["report_sha256"],
                    "status": report["status"],
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException as exc:
        print(json.dumps({"error_type": type(exc).__name__, "status": "STOP"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
