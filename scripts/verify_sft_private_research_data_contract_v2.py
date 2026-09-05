#!/usr/bin/env python3
"""Independently verify the private-research data contract v0.2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_private_research_data_contract_v2 import (
    _load_object,
    _write_exclusive,
    build_contract,
    build_report,
    build_verification,
    file_sha256,
    json_file_bytes,
    load_plan,
    validate_contract,
    validate_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--repository-root", default=".")
    args = parser.parse_args()
    root = Path(args.repository_root).resolve()
    plan_path = Path(args.plan).resolve()
    report_path = Path(args.report).resolve()
    try:
        plan = load_plan(plan_path, repository_root=root)
        contract_path = (root / plan["output"]["data_contract_path"]).resolve()
        expected_report_path = (root / plan["output"]["report_path"]).resolve()
        verification_path = (root / plan["output"]["verification_path"]).resolve()
        if report_path != expected_report_path or not report_path.is_file():
            raise ValueError("private-research report path changed")
        if not contract_path.is_file() or verification_path.exists():
            raise FileExistsError("private-research verification boundary changed")
        predecessor = _load_object(
            (root / plan["bindings"]["predecessor_data_contract"]["path"]).resolve()
        )
        contract = _load_object(contract_path)
        validate_contract(contract, predecessor=predecessor)
        if build_contract(predecessor) != contract:
            raise ValueError("private-research contract independent replay changed")
        report = _load_object(report_path)
        validate_report(report)
        replay = build_report(
            plan=plan,
            plan_file_sha256=file_sha256(plan_path),
            contract=contract,
            contract_file_sha256=file_sha256(contract_path),
            repository_root=root,
        )
        if replay != report:
            raise ValueError("private-research report independent replay changed")
        verification = build_verification(
            report=report,
            report_file_sha256=file_sha256(report_path),
        )
        payload = json_file_bytes(verification)
        total_bytes = contract_path.stat().st_size + report_path.stat().st_size + len(payload)
        if total_bytes > plan["limits"]["maximum_private_write_bytes"]:
            raise ValueError("private-research evidence exceeds write ceiling")
        _write_exclusive(verification_path, payload)
        print(
            json.dumps(
                {
                    "candidate_row_reads": 0,
                    "production_eligible": False,
                    "status": verification["status"],
                    "verification_file_sha256": file_sha256(verification_path),
                    "verification_internal_sha256": verification["verification_sha256"],
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
