#!/usr/bin/env python3
"""Run pass one of the registry-v0.24 private learning-candidate seal."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import lumi.posttraining_learning_candidate_seal_v3 as contract


def _alarm_handler(_signum: int, _frame: object) -> None:
    raise TimeoutError("learning-candidate seal pass-one runtime cap reached")


def main() -> int:
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    marker_started = False
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(contract.EXPECTED_LIMITS["maximum_runtime_seconds_per_pass"])
    try:
        expected_plan_path = (root / contract.EXPECTED_PLAN_PATH).resolve()
        if plan_path.resolve() != expected_plan_path:
            raise ValueError("learning-candidate plan path changed")
        early_outputs = [
            (root / contract.EXPECTED_OUTPUT[key]).resolve()
            for key in (
                "manifest_path",
                "train_path",
                "validation_path",
                "verification_path",
            )
        ]
        if any(path.exists() for path in early_outputs):
            raise FileExistsError("learning-candidate materialization already started")
        plan = contract.load_plan(plan_path, repository_root=root)
        paths = contract.preflight_runner(plan, root)
        marker_identity = contract.start_marker(
            paths["manifest_path"], contract.EXPECTED_MANIFEST_MARKER
        )
        marker_started = True
        with (
            contract.open_private_output(paths["train_path"]) as train_handle,
            contract.open_private_output(paths["validation_path"]) as validation_handle,
        ):
            manifest = contract.build_manifest(
                plan=plan,
                plan_file_sha256=contract.file_sha256(plan_path),
                repository_root=root,
                train_handle=train_handle,
                validation_handle=validation_handle,
                started=started,
            )
        payload = contract.json_file_bytes(manifest)
        pass_one_writes = (
            len(contract.EXPECTED_MANIFEST_MARKER)
            + paths["train_path"].stat().st_size
            + paths["validation_path"].stat().st_size
            + len(payload)
        )
        if (
            len(payload) > contract.EXPECTED_LIMITS["maximum_evidence_bytes"]
            or pass_one_writes > contract.EXPECTED_LIMITS["maximum_private_write_bytes"]
        ):
            raise ValueError("learning-candidate pass-one write cap exceeded")
        contract.finish_marker(
            paths["manifest_path"],
            contract.EXPECTED_MANIFEST_MARKER,
            payload,
            marker_identity,
        )
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        print(
            json.dumps(
                {
                    "candidate_row_reads": manifest["resource_accounting"]["candidate_row_reads"],
                    "manifest_file_sha256": contract.sha256_bytes(payload),
                    "manifest_internal_sha256": manifest["seal_sha256"],
                    "production_eligible": False,
                    "status": manifest["status"],
                    "train_records": manifest["output"]["splits"]["train"]["records"],
                    "validation_records": manifest["output"]["splits"]["validation"]["records"],
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
