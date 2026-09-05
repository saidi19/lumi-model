#!/usr/bin/env python3
"""First zero-network factual pass over acquired Be-DSTC2 rights documents."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

import lumi.posttraining_bedstc2_authoritative_rights_acquisition_v1 as acquisition
import lumi.posttraining_bedstc2_authoritative_rights_factual_review_v1 as review


def _alarm(_signum: int, _frame: object) -> None:
    raise TimeoutError("rights factual-review runtime cap reached")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    marker_started = False
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(review.EXPECTED_LIMITS["maximum_seconds_per_pass"])
    try:
        plan = review.load_plan(plan_path, root)
        acquisition_plan = acquisition.load_plan(
            root / acquisition.EXPECTED_PLAN_PATH, root
        )
        report_path = review.resolve_repo_path(root, plan["output"]["report_path"])
        final_verification_path = review.resolve_repo_path(
            root, plan["output"]["verification_path"]
        )
        acquisition_verification_path = acquisition.resolve_repo_path(
            root, acquisition_plan["output"]["verification_path"]
        )
        if any(
            path.exists()
            for path in (
                report_path,
                final_verification_path,
                acquisition_verification_path,
            )
        ):
            raise FileExistsError("rights factual review already started")
        report_identity = review.start_marker(report_path, review.REPORT_MARKER)
        acquisition_verification_identity = acquisition.start_marker(
            acquisition_verification_path, acquisition.VERIFICATION_MARKER
        )
        marker_started = True
        with review.network_disabled():
            manifest, receipt, payloads = acquisition.replay_acquired_inventory(
                acquisition_plan, root
            )
            acquisition_verification = acquisition.assemble_acquisition_verification(
                manifest, receipt, payloads
            )
            acquisition_verification_payload = acquisition.canonical_json_file_bytes(
                acquisition_verification
            )
            assessment = review.assess_payloads(payloads)
        acquisition.finish_marker(
            acquisition_verification_path,
            acquisition.VERIFICATION_MARKER,
            acquisition_verification_payload,
            acquisition_verification_identity,
        )
        manifest_path = acquisition.resolve_repo_path(
            root, acquisition_plan["output"]["private_manifest_path"]
        )
        receipt_path = acquisition.resolve_repo_path(
            root, acquisition_plan["output"]["receipt_path"]
        )
        report = review.assemble_report(
            assessment,
            acquisition_plan_sha256=review.file_sha256(
                root / acquisition.EXPECTED_PLAN_PATH
            ),
            manifest_file_sha256=review.file_sha256(manifest_path),
            manifest_internal_sha256=manifest["manifest_sha256"],
            receipt_file_sha256=review.file_sha256(receipt_path),
            receipt_internal_sha256=receipt["receipt_sha256"],
            acquisition_verification_file_sha256=review.sha256_bytes(
                acquisition_verification_payload
            ),
            acquisition_verification_internal_sha256=acquisition_verification[
                "verification_sha256"
            ],
            document_bytes=sum(len(item) for item in payloads),
        )
        report_payload = review.canonical_json_file_bytes(report)
        combined_writes = (
            sum(len(item) for item in payloads)
            + manifest_path.stat().st_size
            + receipt_path.stat().st_size
            + len(acquisition.ACQUISITION_MARKER)
            + len(acquisition.MANIFEST_MARKER)
            + len(acquisition.RECEIPT_MARKER)
            + len(acquisition.VERIFICATION_MARKER)
            + len(acquisition_verification_payload)
            + len(review.REPORT_MARKER)
            + len(report_payload)
            + 1_048_576
        )
        if combined_writes > acquisition.EXPECTED_LIMITS["maximum_combined_write_bytes"]:
            raise ValueError("combined rights-document write cap exceeded")
        review.finish_marker(
            report_path, review.REPORT_MARKER, report_payload, report_identity
        )
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        print(
            json.dumps(
                {
                    "criteria_unresolved": 5,
                    "documents": len(payloads),
                    "production_eligible": False,
                    "status": report["status"],
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException as error:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        print(
            json.dumps(
                {
                    "exception_type": type(error).__name__,
                    "marker_started": marker_started,
                    "status": "FAIL_CLOSED",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
