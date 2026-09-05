#!/usr/bin/env python3
"""Run Be-DSTC2 exact-empty vocab recovery attempt 005 exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from collections.abc import Mapping
from pathlib import Path

from lumi.posttraining_bedstc2_banglish_conversation_curation_v6 import (
    EXPECTED_BINDING_PATHS,
    EXPECTED_COMBINED_SUCCESSOR_LIMITS,
    EXPECTED_OUTPUT_PATHS,
    BuildArtifacts,
    build_artifacts,
    build_verification,
    json_bytes,
    load_plan,
)
from lumi.posttraining_candidate_mix_preflight import file_sha256

RUNNER_MARKER = b'{"started":"v0.7_runner_pass_one_attempt_006"}\n'
VERIFIER_MARKER = b'{"started":"v0.7_verifier_pass_two_attempt_006"}\n'


def output_paths(root: Path) -> dict[str, Path]:
    return {name: root / relative for name, relative in EXPECTED_OUTPUT_PATHS.items()}


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, payload: bytes) -> None:
    """Publish one authorized final path without a transient side path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _start_marker(path: Path, *, payload: bytes) -> None:
    _write_exclusive(path, payload)


def _finish_marker(path: Path, *, marker: bytes, payload: bytes) -> None:
    if path.read_bytes() != marker:
        raise ValueError("Be-DSTC2 transaction marker changed")
    with path.open("r+b") as handle:
        handle.seek(0)
        handle.truncate()
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _payloads(artifacts: BuildArtifacts) -> dict[str, bytes]:
    return {
        "candidate": artifacts.candidate_bytes,
        "owner_review_html": artifacts.html_bytes,
        "owner_review_packet": artifacts.owner_packet_bytes,
        "rejection_ledger": artifacts.rejection_bytes,
        "telemetry": artifacts.telemetry_bytes,
    }


def _attempt_private_inventory(root: Path) -> set[Path]:
    private_root = (
        root / "artifacts/lumi-sft-bedstc2-banglish-conversation-v0.1/attempt-006/private"
    )
    if not private_root.is_dir():
        return set()
    observed: set[Path] = set()
    # Attempt 006 contains outputs only; raw inputs remain immutable in attempt 001.
    for path in private_root.iterdir():
        observed.add(path.resolve())
    return observed


def preflight_runner(*, root: Path, paths: Mapping[str, Path]) -> None:
    finals = set(paths.values())
    if any(path.exists() for path in finals):
        raise FileExistsError("refusing to start Be-DSTC2 pass one with existing output")
    if _attempt_private_inventory(root):
        raise FileExistsError("unexpected Be-DSTC2 private output already exists")


def _authorized_static_write_bytes(root: Path, plan_path: Path) -> int:
    paths = [plan_path]
    paths.extend(
        root / EXPECTED_BINDING_PATHS[name]
        for name in ("implementation", "runner", "verifier", "test")
    )
    return sum(path.stat().st_size for path in paths)


def _prospective_total_write_bytes(
    *,
    root: Path,
    plan_path: Path,
    payloads: Mapping[str, bytes],
    report: bytes,
    verification: bytes,
) -> int:
    return (
        _authorized_static_write_bytes(root, plan_path)
        + len(RUNNER_MARKER)
        + len(VERIFIER_MARKER)
        + sum(len(payload) for payload in payloads.values())
        + len(report)
        + len(verification)
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _runtime_alarm(_signum: int, _frame: object) -> None:
    raise TimeoutError("Be-DSTC2 process exceeded its 30-minute half-budget")


def arm_runtime_alarm() -> None:
    signal.signal(signal.SIGALRM, _runtime_alarm)
    signal.setitimer(signal.ITIMER_REAL, 30 * 60)


def cancel_runtime_alarm() -> None:
    signal.setitimer(signal.ITIMER_REAL, 0)


def main() -> int:
    started = time.monotonic()
    arm_runtime_alarm()
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_plan(plan_path, repository_root=root)
    paths = output_paths(root)
    preflight_runner(root=root, paths=paths)
    if time.monotonic() - started > 30 * 60:
        raise TimeoutError("Be-DSTC2 runner exceeded its 30-minute half-budget")

    # The authorized report path is also the durable no-retry marker.  No
    # unbound temporary path is created.  A failure leaves this marker in place.
    _start_marker(paths["report"], payload=RUNNER_MARKER)
    artifacts = build_artifacts(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        repository_root=root,
        started=started,
    )
    payloads = _payloads(artifacts)
    report_payload = json_bytes(artifacts.report)
    prospective_verification = build_verification(
        report=artifacts.report,
        report_file_sha256=_sha256(report_payload),
    )
    verification_payload = json_bytes(prospective_verification)
    total_bytes = _prospective_total_write_bytes(
        root=root,
        plan_path=plan_path,
        payloads=payloads,
        report=report_payload,
        verification=verification_payload,
    )
    if total_bytes > EXPECTED_COMBINED_SUCCESSOR_LIMITS["maximum_private_write_bytes"]:
        raise ValueError("Be-DSTC2 authorized output set exceeds the private-write cap")
    if time.monotonic() - started > 30 * 60:
        raise TimeoutError("Be-DSTC2 runner exceeded its 30-minute half-budget")

    if paths["report"].read_bytes() != RUNNER_MARKER:
        raise ValueError("Be-DSTC2 runner marker changed before publication")
    for name in (
        "candidate",
        "telemetry",
        "rejection_ledger",
        "owner_review_packet",
        "owner_review_html",
    ):
        if paths[name].exists():
            raise FileExistsError(f"Be-DSTC2 output appeared during pass one: {name}")
        _write_exclusive(paths[name], payloads[name])
        if paths[name].stat().st_size != len(payloads[name]) or file_sha256(paths[name]) != _sha256(
            payloads[name]
        ):
            raise ValueError(f"Be-DSTC2 published artifact changed: {name}")
    if paths["verification"].exists():
        raise FileExistsError("Be-DSTC2 verification appeared during pass one")
    if time.monotonic() - started > 30 * 60:
        raise TimeoutError("Be-DSTC2 runner exceeded its 30-minute half-budget")
    _finish_marker(paths["report"], marker=RUNNER_MARKER, payload=report_payload)
    cancel_runtime_alarm()
    print(
        json.dumps(
            {
                "candidate_records": artifacts.report["candidate"]["records"],
                "report_file_sha256": _sha256(report_payload),
                "report_internal_sha256": artifacts.report["report_sha256"],
                "status": artifacts.report["status"],
                "train_supervised_tokens": artifacts.report["candidate"]["train_supervised_tokens"],
            },
            sort_keys=True,
        )
    )
    return 0


def _entrypoint() -> int:
    try:
        return main()
    except Exception as error:  # fail closed without echoing source-derived text
        cancel_runtime_alarm()
        print(
            json.dumps(
                {"exception_type": type(error).__name__, "status": "FAIL_CLOSED"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
