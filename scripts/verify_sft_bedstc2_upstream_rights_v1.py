#!/usr/bin/env python3
"""Independently reconstruct and verify the Be-DSTC2 rights STOP report."""

from __future__ import annotations

import argparse
import json
import os
import signal
import stat
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import lumi.posttraining_bedstc2_upstream_rights_review_v1 as contract


def _alarm_handler(_signum: int, _frame: object) -> None:
    raise TimeoutError("upstream-rights verifier runtime cap reached")


def _read_report_once(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or not 1 <= info.st_size <= 131_072
        ):
            raise ValueError("rights report metadata changed")
        chunks: list[bytes] = []
        remaining = info.st_size + 1
        while remaining:
            chunk = os.read(descriptor, min(131_072, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
    finally:
        os.close(descriptor)
    if len(payload) != info.st_size:
        raise ValueError("rights report changed during read")
    return payload


def _independent_json_input(root: Path, name: str) -> Mapping[str, object]:
    binding = contract.EXPECTED_INPUTS[name]
    path = contract.resolve_repo_path(root, str(binding["path"]))
    payload = contract.read_bound_bytes(
        path,
        expected_bytes=int(binding["bytes"]),
        expected_sha256=str(binding["sha256"]),
    )
    value = contract.strict_json_bytes(payload)
    contract.validate_internal_seal(name, value)
    if not isinstance(value, Mapping):
        raise ValueError("independent JSON root changed")
    return value


def _independent_text_input(root: Path, name: str) -> str:
    binding = contract.EXPECTED_INPUTS[name]
    path = contract.resolve_repo_path(root, str(binding["path"]))
    payload = contract.read_bound_bytes(
        path,
        expected_bytes=int(binding["bytes"]),
        expected_sha256=str(binding["sha256"]),
    )
    try:
        return payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("independent rights document encoding changed") from error


def _independent_expected_report(root: Path, plan_hash: str) -> dict[str, object]:
    acquisition = _independent_json_input(root, "acquisition_receipt")
    readme = _independent_text_input(root, "repository_readme")
    license_text = _independent_text_input(root, "repository_license")
    curation_report = _independent_json_input(root, "curation_report")
    curation_verification = _independent_json_input(root, "curation_verification")
    owner_receipt = _independent_json_input(root, "owner_receipt")
    owner_verification = _independent_json_input(root, "owner_verification")

    # Independent semantic gates over the already exact-hash-bound JSON.
    if (
        acquisition.get("status") != "PASS_exact_five_object_acquisition_only"
        or acquisition.get("revision") != contract.EXPECTED_SOURCE_CONTRACT["revision"]
        or acquisition.get("raw_training_text_in_receipt") is not False
        or curation_report.get("status")
        != "PASS_nonproduction_bedstc2_candidate_owner_review_pending"
        or curation_report.get("production_eligible") is not False
        or curation_verification.get("production_eligible") is not False
        or owner_receipt.get("status")
        != "owner_accepted_all_32_bedstc2_rows_nonproduction_rights_pending"
        or owner_receipt.get("remaining_blockers")
        != contract.EXPECTED_REMAINING_BLOCKERS
        or owner_receipt.get("production_eligible") is not False
        or owner_verification.get("production_eligible") is not False
    ):
        raise ValueError("independent upstream evidence changed")
    observations = contract.inspect_rights_documents(readme, license_text)
    assessment = {
        "document_observations": observations,
        "rights_criteria": dict(contract.EXPECTED_RIGHTS_CRITERIA),
        "criteria_satisfied": 0,
        "criteria_unresolved": 5,
    }

    if assessment != {
        "document_observations": contract.EXPECTED_DOCUMENT_OBSERVATIONS,
        "rights_criteria": contract.EXPECTED_RIGHTS_CRITERIA,
        "criteria_satisfied": 0,
        "criteria_unresolved": 5,
    }:
        raise ValueError("independent rights assessment changed")
    bindings: dict[str, object] = {"plan_file_sha256": plan_hash}
    for name, item in contract.EXPECTED_INPUTS.items():
        binding: dict[str, object] = {
            "bytes": item["bytes"],
            "file_sha256": item["sha256"],
        }
        if "internal_sha256" in item:
            binding["internal_sha256"] = item["internal_sha256"]
        bindings[name] = binding
    report: dict[str, object] = {
        "authorization": contract.EXPECTED_AUTHORIZATION,
        "bindings": bindings,
        "candidate": contract.EXPECTED_CANDIDATE,
        "checks": contract.EXPECTED_CHECKS,
        "document_observations": contract.EXPECTED_DOCUMENT_OBSERVATIONS,
        "document_snippets_retained": False,
        "document_text_retained": False,
        "evaluation_prohibition": contract.EXPECTED_EVALUATION_PROHIBITION,
        "input_accounting": {
            "bound_input_bytes": 54_784,
            "bound_input_file_reads": 7,
            "bound_input_files": 7,
            "source_line_reads": 0,
            "candidate_row_reads": 0,
            "baseline_row_reads": 0,
            "test_object_reads": 0,
        },
        "next_gate": contract.EXPECTED_NEXT_GATE,
        "owner_review": contract.EXPECTED_OWNER_REVIEW,
        "production_eligible": False,
        "provenance": contract.EXPECTED_PROVENANCE,
        "raw_training_text_retained": False,
        "reason_codes": contract.EXPECTED_REASON_CODES,
        "remaining_blockers": contract.EXPECTED_REMAINING_BLOCKERS,
        "report_sha256": "",
        "rights": contract.EXPECTED_RIGHTS,
        "rights_assessment": {
            "criteria": contract.EXPECTED_RIGHTS_CRITERIA,
            "criteria_satisfied": 0,
            "criteria_unresolved": 5,
        },
        "schema_version": contract.REPORT_SCHEMA_VERSION,
        "source_contract": contract.EXPECTED_SOURCE_CONTRACT,
        "status": contract.REPORT_STATUS,
    }
    report["report_sha256"] = contract.sha256_value(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    return report


def main() -> int:
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    report_arg = args.report if args.report.is_absolute() else root / args.report
    marker_started = False
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(contract.EXPECTED_LIMITS["runtime_seconds_per_process"])
    try:
        plan = contract.load_plan(plan_path, root)
        report_path, verification_path = contract.preflight_verifier(plan, root)
        if contract.resolve_repo_path(root, report_arg) != report_path:
            raise ValueError("rights report argument path changed")
        identity = contract.start_marker(
            verification_path, contract.EXPECTED_VERIFICATION_MARKER
        )
        marker_started = True
        report_payload = _read_report_once(report_path)
        actual = contract.strict_json_bytes(report_payload)
        if not isinstance(actual, Mapping) or contract.json_bytes(actual) != report_payload:
            raise ValueError("rights report canonical encoding changed")
        plan_hash = contract.file_sha256(plan_path)
        expected = _independent_expected_report(root, plan_hash)
        if actual != expected:
            raise ValueError("rights report is not independently reproducible")
        contract.validate_report(actual)
        verification = contract.build_verification(
            actual,
            report_file_sha256=contract.sha256_bytes(report_payload),
            report_file_bytes=len(report_payload),
            plan_file_sha256=plan_hash,
        )
        verification_payload = contract.json_bytes(verification)
        if (
            contract.prospective_private_write_bytes(
                plan=plan,
                root=root,
                report_bytes=report_payload,
                verification_bytes=verification_payload,
            )
            > contract.EXPECTED_LIMITS["private_write_bytes"]
        ):
            raise ValueError("upstream-rights private-write cap exceeded")
        if time.monotonic() - started > contract.EXPECTED_LIMITS["runtime_seconds_per_process"]:
            raise TimeoutError("upstream-rights verifier runtime cap reached")
        contract.finish_marker(
            verification_path,
            contract.EXPECTED_VERIFICATION_MARKER,
            verification_payload,
            identity,
        )
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        print(
            json.dumps(
                {
                    "production_eligible": False,
                    "status": verification["status"],
                    "verification_file_sha256": contract.sha256_bytes(
                        verification_payload
                    ),
                    "verification_internal_sha256": verification[
                        "verification_sha256"
                    ],
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
