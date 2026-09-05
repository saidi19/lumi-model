#!/usr/bin/env python3
"""Run pass two of the registry-v0.24 private learning-candidate seal."""

from __future__ import annotations

import argparse
import json
import os
import signal
import stat
import sys
import time
from pathlib import Path

import lumi.posttraining_learning_candidate_seal_v3 as contract


def _alarm_handler(_signum: int, _frame: object) -> None:
    raise TimeoutError("learning-candidate seal pass-two runtime cap reached")


def _read_manifest_once(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) & 0o077
            or not 1 <= info.st_size <= contract.EXPECTED_LIMITS["maximum_evidence_bytes"]
        ):
            raise ValueError("learning-candidate manifest metadata changed")
        payload = bytearray()
        while len(payload) < info.st_size:
            chunk = os.read(descriptor, min(65_536, info.st_size - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
    finally:
        os.close(descriptor)
    if len(payload) != info.st_size:
        raise ValueError("learning-candidate manifest changed during read")
    return bytes(payload)


def main() -> int:
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    manifest_arg = args.manifest if args.manifest.is_absolute() else root / args.manifest
    marker_started = False
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(contract.EXPECTED_LIMITS["maximum_runtime_seconds_per_pass"])
    try:
        expected_plan_path = (root / contract.EXPECTED_PLAN_PATH).resolve()
        if plan_path.resolve() != expected_plan_path:
            raise ValueError("learning-candidate plan path changed")
        early_verification = (root / contract.EXPECTED_OUTPUT["verification_path"]).resolve()
        if early_verification.exists():
            raise FileExistsError("learning-candidate verification already started")
        plan = contract.load_plan(
            plan_path,
            repository_root=root,
            check_candidate_files=False,
        )
        paths = contract.preflight_verifier(plan, root)
        if manifest_arg.resolve() != paths["manifest_path"]:
            raise ValueError("learning-candidate manifest argument changed")
        marker_identity = contract.start_marker(
            paths["verification_path"], contract.EXPECTED_VERIFICATION_MARKER
        )
        marker_started = True
        manifest_payload = _read_manifest_once(paths["manifest_path"])
        manifest = contract.strict_json_bytes(manifest_payload)
        if (
            not isinstance(manifest, dict)
            or contract.json_file_bytes(manifest) != manifest_payload
            or manifest.get("plan_file_sha256") != contract.file_sha256(plan_path)
        ):
            raise ValueError("learning-candidate manifest encoding or plan changed")
        contract.validate_manifest(manifest, plan=plan)
        audit = contract.audit_outputs(
            plan=plan,
            manifest=manifest,
            repository_root=root,
            started=started,
        )
        verification = contract.build_verification(
            plan=plan,
            manifest=manifest,
            manifest_file_sha256=contract.sha256_bytes(manifest_payload),
            audit=audit,
        )
        payload = contract.json_file_bytes(verification)
        if (
            len(manifest_payload) + len(payload)
            > contract.EXPECTED_LIMITS["maximum_evidence_bytes"]
            or contract.prospective_total_write_bytes(
                manifest_path=paths["manifest_path"],
                train_path=paths["train_path"],
                validation_path=paths["validation_path"],
                verification_payload=payload,
            )
            > contract.EXPECTED_LIMITS["maximum_private_write_bytes"]
        ):
            raise ValueError("learning-candidate total write cap exceeded")
        contract.finish_marker(
            paths["verification_path"],
            contract.EXPECTED_VERIFICATION_MARKER,
            payload,
            marker_identity,
        )
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        print(
            json.dumps(
                {
                    "candidate_files_opened": audit["candidate_files_opened"],
                    "output_audit_row_reads": audit["output_audit_row_reads"],
                    "production_eligible": False,
                    "status": verification["status"],
                    "verification_file_sha256": contract.sha256_bytes(payload),
                    "verification_internal_sha256": verification["verification_sha256"],
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
