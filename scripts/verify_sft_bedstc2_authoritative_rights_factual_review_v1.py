#!/usr/bin/env python3
"""Independent second zero-network Be-DSTC2 rights factual pass."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

import lumi.posttraining_bedstc2_authoritative_rights_acquisition_v1 as acquisition
import lumi.posttraining_bedstc2_authoritative_rights_factual_review_v1 as review


def _alarm(_signum: int, _frame: object) -> None:
    raise TimeoutError("rights factual verifier runtime cap reached")


def _independent_observations(payloads: list[bytes]) -> dict[str, bool]:
    by_id = {
        spec["id"]: payload
        for spec, payload in zip(acquisition.EXPECTED_DOCUMENTS, payloads, strict=True)
    }

    def text(identity: str) -> str:
        try:
            return by_id[identity].decode("utf-8", errors="strict").casefold()
        except UnicodeDecodeError as error:
            raise ValueError("independent strict UTF-8 gate changed") from error

    return {
        "official_microsoft_event_page_acquired": (
            "dialog state tracking challenge" in text("microsoft_dstc_event")
        ),
        "dstc2_acl_anthology_landing_page_acquired": (
            "second dialog state tracking challenge" in text("dstc2_acl_anthology_landing")
            or "dialog state tracking challenge" in text("dstc2_acl_anthology_landing")
        ),
        "organizer_associated_exact_revision_readme_acquired": (
            "dialog state tracking" in text("matthen_dstc_readme_exact_revision")
            and "dstc2" in text("matthen_dstc_readme_exact_revision")
        ),
        "organizer_associated_exact_revision_gpl3_text_acquired": (
            "gnu general public license" in text("matthen_dstc_license_exact_revision")
            and "version 3" in text("matthen_dstc_license_exact_revision")
        ),
        "dstc2_primary_paper_pdf_acquired": by_id["dstc2_primary_paper"].startswith(b"%PDF-"),
        "code_mixed_dialog_paper_pdf_acquired": by_id["code_mixed_dialog_paper"].startswith(b"%PDF-"),
        "code_mixed_dialog_exact_revision_readme_attributes_dstc2": (
            "dstc2" in text("code_mixed_dialog_readme_exact_revision")
            and "code-mixed" in text("code_mixed_dialog_readme_exact_revision")
        ),
        "code_mixed_dialog_exact_revision_apache2_text_acquired": (
            "apache license" in text("code_mixed_dialog_license_exact_revision")
            and "version 2.0" in text("code_mixed_dialog_license_exact_revision")
        ),
        "authoritative_corpus_specific_dstc2_grant_document_in_inventory": False,
        "rights_owner_written_confirmation_in_inventory": False,
        "qualified_legal_approval_in_inventory": False,
    }


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
        verification_path = review.resolve_repo_path(
            root, plan["output"]["verification_path"]
        )
        if not report_path.is_file() or verification_path.exists():
            raise FileExistsError("rights factual verification preflight changed")
        identity = review.start_marker(verification_path, review.VERIFICATION_MARKER)
        marker_started = True
        report, report_payload = review.read_report(report_path)
        with review.network_disabled():
            manifest, receipt, payloads = acquisition.replay_acquired_inventory(
                acquisition_plan, root
            )
            observations = _independent_observations(payloads)
        if observations != review.EXPECTED_OBSERVATIONS:
            raise ValueError("independent factual observations changed")
        review.validate_report(report)
        acquisition_verification_path = acquisition.resolve_repo_path(
            root, acquisition_plan["output"]["verification_path"]
        )
        acquisition_verification, acquisition_verification_payload = (
            acquisition.read_json_file(acquisition_verification_path)
        )
        acquisition.validate_seal(acquisition_verification, "verification_sha256")
        bindings = report.get("bindings", {})
        if bindings != {
            "acquisition_plan_file_sha256": review.file_sha256(
                root / acquisition.EXPECTED_PLAN_PATH
            ),
            "private_manifest_file_sha256": review.file_sha256(
                acquisition.resolve_repo_path(
                    root, acquisition_plan["output"]["private_manifest_path"]
                )
            ),
            "private_manifest_internal_sha256": manifest["manifest_sha256"],
            "acquisition_receipt_file_sha256": review.file_sha256(
                acquisition.resolve_repo_path(
                    root, acquisition_plan["output"]["receipt_path"]
                )
            ),
            "acquisition_receipt_internal_sha256": receipt["receipt_sha256"],
            "acquisition_verification_file_sha256": review.sha256_bytes(
                acquisition_verification_payload
            ),
            "acquisition_verification_internal_sha256": acquisition_verification[
                "verification_sha256"
            ],
        }:
            raise ValueError("independent report bindings changed")
        if report.get("document_accounting") != {
            "documents_read": len(payloads),
            "document_bytes_read": sum(len(item) for item in payloads),
            "passes_completed_by_this_report": 1,
            "network_bytes": 0,
        }:
            raise ValueError("independent factual accounting changed")
        verification = review.assemble_verification(
            report,
            report_file_sha256=review.sha256_bytes(report_payload),
            document_bytes=sum(len(item) for item in payloads),
        )
        payload = review.canonical_json_file_bytes(verification)
        review.finish_marker(
            verification_path, review.VERIFICATION_MARKER, payload, identity
        )
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        print(
            json.dumps(
                {
                    "factual_passes": 2,
                    "production_eligible": False,
                    "status": verification["status"],
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
