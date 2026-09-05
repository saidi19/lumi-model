#!/usr/bin/env python3
"""Independently replay the BanglaBridge expanded-delta recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from lumi.posttraining_banglabridge_expanded_delta_recovery_v1 import (
    build_recovery,
    build_verification,
    json_bytes,
    load_plan,
    validate_owner_preflight_artifacts,
    validate_report,
    validate_verification,
)
from lumi.posttraining_candidate_mix_preflight import file_sha256


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(
            f"refusing to overwrite expanded-delta verification: {path}"
        )
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _preflight_then_replay(
    preflight: Callable[[], Any], replay: Callable[[], Any]
) -> tuple[Any, Any]:
    """Never begin the expensive independent replay until preflight returns."""

    preflight_state = preflight()
    return preflight_state, replay()


def preflight_first_pass_outputs(
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    paths: Mapping[str, Path],
    report_path: Path,
    verification_path: Path,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any], bytes]:
    required = (*paths.values(), report_path)
    temporary_paths = {
        path.with_name(f".{path.name}.tmp")
        for path in (*required, verification_path)
    }
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("expanded-delta first-pass output set is incomplete")
    if verification_path.exists() or any(path.exists() for path in temporary_paths):
        raise FileExistsError("refusing to overwrite expanded-delta verification")

    report_bytes = report_path.read_bytes()
    try:
        observed_report = json.loads(report_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("expanded-delta report is invalid JSON") from error
    validate_report(observed_report)
    if report_bytes != json_bytes(observed_report):
        raise ValueError("expanded-delta report bytes are not canonical")

    observed_sha256: dict[str, str] = {}
    for name, path in paths.items():
        observed_bytes = path.stat().st_size
        observed_hash = file_sha256(path)
        if (
            observed_hash != observed_report["outputs"][name]["sha256"]
            or observed_bytes != observed_report["outputs"][name]["bytes"]
        ):
            raise ValueError(f"expanded-delta first-pass artifact changed: {name}")
        observed_sha256[name] = observed_hash

    validate_owner_preflight_artifacts(
        report=observed_report,
        owner_packet_bytes=paths["owner_review_packet"].read_bytes(),
        owner_html_bytes=paths["owner_review_html"].read_bytes(),
        delta_ledger_bytes=paths["delta_ledger"].read_bytes(),
    )
    if (
        observed_report["plan_file_sha256"] != file_sha256(plan_path)
        or observed_report["authorization"] != plan["authorization"]
        or observed_report["runtime_recovery"] != plan["runtime_recovery"]
        or observed_report["bindings"]
        != {name: value["sha256"] for name, value in sorted(plan["bindings"].items())}
    ):
        raise ValueError("expanded-delta report/plan preflight changed")

    prospective_verification = build_verification(
        plan=plan,
        report=observed_report,
        report_file_sha256=file_sha256(report_path),
        artifact_file_sha256=observed_sha256,
    )
    prospective_payload = json_bytes(prospective_verification)
    total = sum(path.stat().st_size for path in paths.values())
    total += report_path.stat().st_size + len(prospective_payload)
    if total > plan["limits"]["maximum_private_write_bytes"]:
        raise ValueError("expanded-delta private-write cap exceeded")
    return (
        observed_report,
        observed_sha256,
        prospective_verification,
        prospective_payload,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_plan(plan_path.resolve(), root)
    output = plan["output"]
    paths = {
        "candidate": root / output["candidate_path"],
        "delta_ledger": root / output["delta_ledger_path"],
        "owner_review_html": root / output["owner_review_html_path"],
        "owner_review_packet": root / output["owner_review_packet_path"],
    }
    report_path = root / output["report_path"]
    verification_path = root / output["verification_path"]
    (
        (
            observed_report,
            observed_sha256,
            prospective_verification,
            verification_payload,
        ),
        (expected_artifacts, expected_report),
    ) = _preflight_then_replay(
        lambda: preflight_first_pass_outputs(
            plan=plan,
            plan_path=plan_path,
            paths=paths,
            report_path=report_path,
            verification_path=verification_path,
        ),
        lambda: build_recovery(
            plan=plan,
            plan_file_sha256=file_sha256(plan_path),
            repository_root=root,
        ),
    )
    for name, expected in expected_artifacts.items():
        observed = paths[name].read_bytes()
        if observed != expected:
            raise ValueError(f"expanded-delta materialized artifact changed: {name}")
        if hashlib.sha256(observed).hexdigest() != observed_sha256[name]:
            raise ValueError(f"expanded-delta artifact hash did not reproduce: {name}")
    if observed_report != expected_report or report_path.read_bytes() != json_bytes(
        expected_report
    ):
        raise ValueError("expanded-delta report did not reproduce")
    reproduced_verification = build_verification(
        plan=plan,
        report=observed_report,
        report_file_sha256=file_sha256(report_path),
        artifact_file_sha256=observed_sha256,
    )
    if reproduced_verification != prospective_verification:
        raise ValueError("expanded-delta verification did not reproduce")
    validate_verification(prospective_verification)
    _write_atomic(verification_path, verification_payload)
    print(
        json.dumps(
            {
                "report_file_sha256": prospective_verification["report_file_sha256"],
                "status": prospective_verification["status"],
                "verification_sha256": prospective_verification[
                    "verification_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
