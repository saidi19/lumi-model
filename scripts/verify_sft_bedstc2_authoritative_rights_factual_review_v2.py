#!/usr/bin/env python3
"""Run the one remaining independent zero-network Be-DSTC2 factual pass."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

import lumi.posttraining_bedstc2_authoritative_rights_acquisition_v1 as acquisition
import lumi.posttraining_bedstc2_authoritative_rights_factual_review_recovery_v2 as recovery
import lumi.posttraining_bedstc2_authoritative_rights_factual_review_v1 as review_v1


def _alarm(_signum: int, _frame: object) -> None:
    raise TimeoutError("v0.2 factual verifier runtime cap reached")


def _independent_observations(payloads: list[bytes]) -> dict[str, bool]:
    if len(payloads) != len(acquisition.EXPECTED_DOCUMENTS):
        raise ValueError("independent v0.2 document count changed")
    by_id = {
        specification["id"]: payload
        for specification, payload in zip(
            acquisition.EXPECTED_DOCUMENTS, payloads, strict=True
        )
    }

    def text(identity: str) -> str:
        try:
            return by_id[identity].decode("utf-8", errors="strict").casefold()
        except UnicodeDecodeError as error:
            raise ValueError("independent v0.2 strict UTF-8 gate changed") from error

    return {
        "official_microsoft_event_page_acquired": (
            "dialog state tracking challenge" in text("microsoft_dstc_event")
        ),
        "dstc2_acl_anthology_landing_page_acquired": (
            "second dialog state tracking challenge"
            in text("dstc2_acl_anthology_landing")
            or "dialog state tracking challenge"
            in text("dstc2_acl_anthology_landing")
        ),
        "organizer_associated_exact_revision_readme_acquired": (
            "dialog state tracking" in text("matthen_dstc_readme_exact_revision")
            and "dstc2" in text("matthen_dstc_readme_exact_revision")
        ),
        "organizer_associated_exact_revision_gpl3_text_acquired": (
            "gnu general public license"
            in text("matthen_dstc_license_exact_revision")
            and "version 3" in text("matthen_dstc_license_exact_revision")
        ),
        "dstc2_primary_paper_pdf_acquired": by_id["dstc2_primary_paper"].startswith(
            b"%PDF-"
        ),
        "code_mixed_dialog_paper_pdf_acquired": by_id[
            "code_mixed_dialog_paper"
        ].startswith(b"%PDF-"),
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
        signal.alarm(recovery.EXPECTED_LIMITS["maximum_seconds_per_process"])
    try:
        plan = recovery.load_plan(plan_path, root)
        acquisition_plan = acquisition.load_plan(
            root / acquisition.EXPECTED_PLAN_PATH, root
        )
        report_path = recovery.resolve_repo_path(root, plan["output"]["report_path"])
        verification_path = recovery.resolve_repo_path(
            root, plan["output"]["verification_path"]
        )
        if not report_path.is_file() or verification_path.exists():
            raise FileExistsError("v0.2 independent verification preflight changed")
        with review_v1.network_disabled():
            recovery.validate_completion_witness(root)
            report, report_payload = recovery.read_report(report_path)
        if report["bindings"]["recovery_plan_file_sha256"] != recovery.file_sha256(
            root / recovery.EXPECTED_PLAN_PATH
        ):
            raise ValueError("v0.2 recovery-plan report binding changed")
        identity = recovery.start_marker(
            verification_path, recovery.VERIFICATION_MARKER
        )
        marker_started = True
        with review_v1.network_disabled():
            _manifest, _receipt, payloads = acquisition.replay_acquired_inventory(
                acquisition_plan, root
            )
            observations = _independent_observations(payloads)
        if observations != recovery.EXPECTED_OBSERVATIONS:
            raise ValueError("independent v0.2 factual observations changed")
        document_bytes = sum(len(item) for item in payloads)
        if (
            len(payloads)
            != recovery.EXPECTED_LIMITS["independent_pass2_document_reads"]
            or document_bytes
            > recovery.EXPECTED_LIMITS["maximum_independent_pass2_document_bytes"]
        ):
            raise ValueError("independent v0.2 document accounting changed")
        verification = recovery.assemble_verification(
            report,
            report_file_sha256=recovery.sha256_bytes(report_payload),
            document_bytes=document_bytes,
        )
        payload = recovery.canonical_json_file_bytes(verification)
        if (
            report_path.stat().st_size
            + len(payload)
            + len(recovery.REPORT_MARKER)
            + len(recovery.VERIFICATION_MARKER)
            > recovery.EXPECTED_LIMITS["maximum_evidence_write_bytes"]
        ):
            raise ValueError("v0.2 combined evidence write cap exceeded")
        recovery.finish_marker(
            verification_path, recovery.VERIFICATION_MARKER, payload, identity
        )
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        print(
            json.dumps(
                {
                    "factual_document_passes": 2,
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
                    "retry_authorized": False,
                    "status": "FAIL_CLOSED",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
