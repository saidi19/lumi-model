#!/usr/bin/env python3
"""Independently reconstruct the Be-DSTC2 owner-review receipt."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import stat
import sys
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import lumi.posttraining_bedstc2_banglish_conversation_owner_review_v1 as contract
from lumi.posttraining_bedstc2_banglish_conversation_curation_v6 import (
    validate_report as validate_curation_report,
)
from lumi.posttraining_bedstc2_banglish_conversation_curation_v6 import (
    validate_verification as validate_curation_verification,
)

_RECORD_ID_RE = re.compile(r"^bedstc2-banglish-[0-9a-f]{24}$")
_REVIEWER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_AI_MARKERS = ("automated", "bot", "chatgpt", "claude", "codex", "gemini", "openai")


def _alarm_handler(_signum: int, _frame: object) -> None:
    raise TimeoutError("owner-review verifier runtime cap reached")


def _read_receipt_once(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or not 1 <= info.st_size <= 65_536:
            raise ValueError("receipt file metadata changed")
        remaining = info.st_size + 1
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
    finally:
        os.close(descriptor)
    if len(payload) != info.st_size:
        raise ValueError("receipt file size changed during read")
    return payload


def _read_input(root: Path, name: str) -> Any:
    binding = contract.EXPECTED_INPUTS[name]
    if name == "decisions":
        path = contract.EXPECTED_EXTERNAL_DECISIONS_PATH
    else:
        path = (root / str(binding["path"])).resolve()
        if not path.is_relative_to(root):
            raise ValueError("independent input path escaped")
    payload = contract.read_bound_bytes(
        path,
        expected_bytes=int(binding["bytes"]),
        expected_sha256=str(binding["sha256"]),
    )
    value = contract.strict_json_bytes(payload)
    if name in {"curation_report", "curation_verification"}:
        if not isinstance(value, Mapping):
            raise ValueError("independent sealed input root changed")
        seal_key = (
            "report_sha256" if name == "curation_report" else "verification_sha256"
        )
        unsigned = {key: item for key, item in value.items() if key != seal_key}
        if (
            value.get(seal_key) != binding["canonical_sha256"]
            or contract.sha256_value(unsigned) != binding["canonical_sha256"]
        ):
            raise ValueError("independent internal input seal changed")
    elif contract.sha256_value(value) != binding["canonical_sha256"]:
        raise ValueError("independent canonical input hash changed")
    return value


def _independent_packet_ids(packet: Any) -> list[str]:
    if not isinstance(packet, list) or len(packet) != 32:
        raise ValueError("independent packet inventory changed")
    ids: list[str] = []
    for row in packet:
        if not isinstance(row, Mapping) or set(row) != {
            "language_mode",
            "messages",
            "record_id",
            "skill_id",
            "split",
        }:
            raise ValueError("independent packet fields changed")
        record_id = row.get("record_id")
        messages = row.get("messages")
        if (
            not isinstance(record_id, str)
            or _RECORD_ID_RE.fullmatch(record_id) is None
            or row.get("language_mode") != "banglish"
            or row.get("skill_id") != "conversation"
            or row.get("split") != "train"
            or not isinstance(messages, list)
            or len(messages) != 2
        ):
            raise ValueError("independent packet row contract changed")
        for message, role in zip(messages, ("user", "assistant"), strict=True):
            if (
                not isinstance(message, Mapping)
                or set(message) != {"content", "role"}
                or message.get("role") != role
                or not isinstance(message.get("content"), str)
                or not message["content"]
            ):
                raise ValueError("independent packet message contract changed")
        ids.append(record_id)
    if len(set(ids)) != 32:
        raise ValueError("independent packet IDs changed")
    return ids


def _independent_decision_outcome(decisions: Any, ids: list[str]) -> dict[str, Any]:
    if not isinstance(decisions, Mapping) or set(decisions) != {
        "decisions",
        "reviewed_at",
        "reviewer_id",
        "sample_sha256",
        "schema_version",
    }:
        raise ValueError("independent decision root changed")
    reviewer = decisions.get("reviewer_id")
    reviewed_at = decisions.get("reviewed_at")
    if (
        decisions.get("schema_version") != contract.DECISIONS_SCHEMA_VERSION
        or decisions.get("sample_sha256")
        != contract.EXPECTED_INPUTS["owner_packet"]["canonical_sha256"]
        or not isinstance(reviewer, str)
        or _REVIEWER_RE.fullmatch(reviewer) is None
        or any(marker in reviewer.casefold() for marker in _AI_MARKERS)
        or not isinstance(reviewed_at, str)
    ):
        raise ValueError("independent decision binding changed")
    try:
        parsed = datetime.strptime(reviewed_at, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ValueError("independent decision timestamp changed") from error
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != reviewed_at:
        raise ValueError("independent decision timestamp changed")
    rows = decisions.get("decisions")
    if not isinstance(rows, list) or len(rows) != 32:
        raise ValueError("independent decision inventory changed")
    accepted = 0
    rejected = 0
    nonempty_notes = 0
    observed: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"accepted", "note", "record_id"}:
            raise ValueError("independent decision fields changed")
        record_id = row.get("record_id")
        accepted_value = row.get("accepted")
        note = row.get("note")
        if (
            not isinstance(record_id, str)
            or _RECORD_ID_RE.fullmatch(record_id) is None
            or type(accepted_value) is not bool
            or not isinstance(note, str)
            or len(note) > 2_000
        ):
            raise ValueError("independent decision row changed")
        observed.append(record_id)
        accepted += int(accepted_value)
        rejected += int(not accepted_value)
        nonempty_notes += int(bool(note))
    if observed != ids or len(set(observed)) != 32 or accepted < 29:
        raise ValueError("independent decision coverage changed")
    return {
        "accepted": accepted,
        "rejected": rejected,
        "nonempty_notes": nonempty_notes,
        "reviewed_at": reviewed_at,
    }


def _independent_upstream(root: Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    report = _read_input(root, "curation_report")
    verification = _read_input(root, "curation_verification")
    if not isinstance(report, Mapping) or not isinstance(verification, Mapping):
        raise ValueError("independent upstream root changed")
    validate_curation_report(report)
    validate_curation_verification(verification)
    if (
        report.get("report_sha256")
        != contract.EXPECTED_INPUTS["curation_report"]["canonical_sha256"]
        or verification.get("verification_sha256")
        != contract.EXPECTED_INPUTS["curation_verification"]["canonical_sha256"]
        or verification.get("report_file_sha256")
        != contract.EXPECTED_INPUTS["curation_report"]["sha256"]
        or report.get("candidate")
        != {
            key: value
            for key, value in contract.EXPECTED_CANDIDATE.items()
            if key != "file_sha256_attested_not_opened"
        }
        or report.get("artifacts", {}).get("candidate", {}).get("sha256")
        != contract.EXPECTED_CANDIDATE["file_sha256_attested_not_opened"]
        or report.get("provenance") != contract.EXPECTED_PROVENANCE
        or report.get("production_eligible") is not False
        or verification.get("production_eligible") is not False
    ):
        raise ValueError("independent upstream evidence changed")
    return report, verification


def _independent_expected_receipt(root: Path, plan_file_sha256: str) -> dict[str, Any]:
    packet = _read_input(root, "owner_packet")
    ids = _independent_packet_ids(packet)
    decisions = _read_input(root, "decisions")
    outcome = _independent_decision_outcome(decisions, ids)
    _independent_upstream(root)
    reviewed_ids_sha = contract.sha256_bytes(
        b"bedstc2-owner-reviewed-record-ids-v0.1\0"
        + contract.canonical_json_bytes(sorted(ids))
    )
    if (
        outcome
        != {
            "accepted": 32,
            "rejected": 0,
            "nonempty_notes": 0,
            "reviewed_at": contract.EXPECTED_REVIEWED_AT,
        }
        or reviewed_ids_sha != contract.EXPECTED_REVIEWED_IDS_DOMAIN_SHA256
    ):
        raise ValueError("independent owner-review outcome changed")
    receipt: dict[str, Any] = {
        "authorization": contract.EXPECTED_AUTHORIZATION,
        "bindings": {
            "authorization_sha256": contract.EXPECTED_AUTHORIZATION_SHA256,
            "curation_report_file_sha256": contract.EXPECTED_INPUTS[
                "curation_report"
            ]["sha256"],
            "curation_report_internal_sha256": contract.EXPECTED_INPUTS[
                "curation_report"
            ]["canonical_sha256"],
            "curation_verification_file_sha256": contract.EXPECTED_INPUTS[
                "curation_verification"
            ]["sha256"],
            "curation_verification_internal_sha256": contract.EXPECTED_INPUTS[
                "curation_verification"
            ]["canonical_sha256"],
            "decisions_bytes": contract.EXPECTED_INPUTS["decisions"]["bytes"],
            "decisions_canonical_sha256": contract.EXPECTED_INPUTS["decisions"][
                "canonical_sha256"
            ],
            "decisions_file_sha256": contract.EXPECTED_INPUTS["decisions"][
                "sha256"
            ],
            "owner_packet_bytes": contract.EXPECTED_INPUTS["owner_packet"]["bytes"],
            "owner_packet_file_sha256": contract.EXPECTED_INPUTS["owner_packet"][
                "sha256"
            ],
            "owner_packet_semantic_sha256": contract.EXPECTED_INPUTS[
                "owner_packet"
            ]["canonical_sha256"],
            "plan_file_sha256": plan_file_sha256,
            "reviewed_record_ids_domain_sha256": reviewed_ids_sha,
        },
        "candidate": contract.EXPECTED_CANDIDATE,
        "decision_notes_retained": False,
        "evaluation_prohibition": contract.EXPECTED_EVALUATION_PROHIBITION,
        "human_owner_review_claim_self_attested": True,
        "next_gate": contract.EXPECTED_NEXT_GATE,
        "owner_review": {
            "accepted_records": 32,
            "all_sampled_rows_accepted": True,
            "automated_reviewer_markers_absent": True,
            "rejected_records": 0,
            "reviewed_at_claimed_utc": contract.EXPECTED_REVIEWED_AT,
            "sample_records": 32,
            "threshold": {
                "minimum_accepts": 29,
                "observed_accepts": 32,
                "satisfied": True,
            },
        },
        "owner_review_claim": contract.EXPECTED_OWNER_REVIEW_CLAIM,
        "production_eligible": False,
        "provenance": contract.EXPECTED_PROVENANCE,
        "raw_training_text_retained": False,
        "receipt_sha256": "",
        "record_ids_retained": False,
        "remaining_blockers": contract.EXPECTED_REMAINING_BLOCKERS,
        "reviewer_alias_retained": False,
        "rights": contract.EXPECTED_RIGHTS,
        "schema_version": contract.RECEIPT_SCHEMA_VERSION,
        "status": contract.RECEIPT_STATUS,
    }
    receipt["receipt_sha256"] = contract.sha256_value(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    return receipt


def main() -> int:
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    receipt_path_arg = args.receipt if args.receipt.is_absolute() else root / args.receipt
    marker_started = False
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(contract.EXPECTED_LIMITS["runtime_seconds_per_process"])
    try:
        plan = contract.load_plan(plan_path, root)
        receipt_path, verification_path = contract.preflight_verifier(plan, root)
        if receipt_path_arg.resolve() != receipt_path:
            raise ValueError("receipt argument path changed")
        marker_identity = contract.start_marker(
            verification_path, contract.EXPECTED_VERIFICATION_MARKER
        )
        marker_started = True
        receipt_payload = _read_receipt_once(receipt_path)
        actual_receipt = contract.strict_json_bytes(receipt_payload)
        if (
            not isinstance(actual_receipt, Mapping)
            or contract.json_bytes(actual_receipt) != receipt_payload
        ):
            raise ValueError("receipt canonical file encoding changed")
        expected_receipt = _independent_expected_receipt(
            root, contract.file_sha256(plan_path)
        )
        if actual_receipt != expected_receipt:
            raise ValueError("owner-review receipt is not independently reproducible")
        contract.validate_receipt(actual_receipt)
        verification = contract.build_verification(
            actual_receipt,
            receipt_file_sha256=contract.sha256_bytes(receipt_payload),
            receipt_file_bytes=len(receipt_payload),
            plan_file_sha256=contract.file_sha256(plan_path),
        )
        verification_payload = contract.json_bytes(verification)
        physical_writes = contract.prospective_private_write_bytes(
            plan=plan,
            root=root,
            receipt_bytes=receipt_payload,
            verification_bytes=verification_payload,
        )
        if physical_writes > contract.EXPECTED_LIMITS["private_write_bytes"]:
            raise ValueError("owner-review private-write cap exceeded")
        if time.monotonic() - started > contract.EXPECTED_LIMITS["runtime_seconds_per_process"]:
            raise TimeoutError("owner-review verifier runtime cap reached")
        contract.finish_marker(
            verification_path,
            contract.EXPECTED_VERIFICATION_MARKER,
            verification_payload,
            marker_identity,
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
