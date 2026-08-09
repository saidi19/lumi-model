#!/usr/bin/env python3
"""Independently verify the paired teacher A/B fail-closed boundary receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from lumi import posttraining_stronger_teacher_ab as ab
from lumi import posttraining_stronger_teacher_ab_preflight as preflight
from lumi import posttraining_first_party_scale_canary as scale


REPORT_SCHEMA_VERSION = "lumi.sft-stronger-teacher-ab-boundary-stop.v0.1"
REPORT_STATUS = (
    "stopped_fail_closed_attempt_budget_insufficient_after_local_transport_failure"
)


def _read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root changed: {path}")
    return value


def verify(*, plan_path: Path, repository_root: Path, artifact_root: Path) -> dict:
    plan = ab.load_plan(plan_path, repository_root=repository_root)
    if artifact_root.resolve() != ab.artifact_root(plan, repository_root):
        raise ValueError("paired A/B artifact root changed")
    report_path = artifact_root / "evidence" / "boundary-stop-report.json"
    report = _read_object(report_path)
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256", None)
    if internal != ab.sha256_value(unsigned):
        raise ValueError("paired A/B boundary report internal hash changed")
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("paired A/B boundary report schema changed")
    if report.get("status") != REPORT_STATUS:
        raise ValueError("paired A/B boundary report status changed")

    schedule = preflight.selected_schedule()
    baseline_path = artifact_root / "private" / "baseline-call-journal.jsonl"
    baseline = ab._read_jsonl(baseline_path)  # noqa: SLF001
    starts, finishes = ab._journal_state(  # noqa: SLF001
        baseline, schedule, model_key="baseline"
    )
    final_by_id = {
        slot.record_id: ab._final_event(finishes.get(slot.record_id, []))  # noqa: SLF001
        for slot in schedule
    }
    complete = {
        record_id: event
        for record_id, event in final_by_id.items()
        if event is not None and event.get("runtime_complete") is True
    }
    incomplete = {
        record_id: event
        for record_id, event in final_by_id.items()
        if event is not None and event.get("runtime_complete") is not True
    }
    started_ids = {str(event["record_id"]) for event in starts}
    if len(baseline) != 232 or len(starts) != 116:
        raise ValueError("paired A/B baseline call boundary changed")
    if len(complete) != 95 or len(incomplete) != 1:
        raise ValueError("paired A/B output boundary changed")
    if len({slot.record_id for slot in schedule} - started_ids) != 4:
        raise ValueError("paired A/B unstarted baseline count changed")
    if sum(event.get("accepted") is True for event in complete.values()) != 40:
        raise ValueError("paired A/B accepted baseline count changed")
    failure = next(iter(incomplete.values()))
    reason = str(failure.get("rejection_reason") or "")
    if not reason.startswith("RemoteDisconnected:"):
        raise ValueError("paired A/B failure class changed")
    if failure.get("raw_response_text") not in {"", None}:
        raise ValueError("paired A/B transport failure unexpectedly retained text")

    candidate_path = artifact_root / "private" / "candidate-call-journal.jsonl"
    candidate = ab._read_jsonl(candidate_path)  # noqa: SLF001
    if any(event.get("event") == "call_started" for event in candidate):
        raise ValueError("candidate inference unexpectedly started")

    boundary = report["attempt_boundary"]
    independently_recomputed = {
        "authorized_attempts": 220,
        "attempts_remaining": 104,
        "attempts_used": 116,
        "baseline_outputs_remaining": 5,
        "candidate_outputs_remaining": 100,
        "minimum_additional_attempts_for_exact_200_outputs": 105,
        "minimum_total_attempts_for_exact_200_outputs": 221,
        "original_authorization_shortfall_attempts": 1,
        "runtime_retries_authorized": 20,
        "runtime_retries_used": 20,
    }
    if boundary != independently_recomputed:
        raise ValueError("paired A/B attempt-boundary math changed")

    bindings = report["bindings"]
    expected_paths = {
        "authorization_file_sha256": artifact_root / "authorization.json",
        "baseline_journal_file_sha256": baseline_path,
        "download_receipt_file_sha256": (
            artifact_root / "evidence" / "download-receipt.json"
        ),
        "execution_amendment_001_file_sha256": (
            artifact_root / "execution-package-amendment-001.json"
        ),
        "execution_amendment_002_file_sha256": (
            artifact_root / "execution-package-amendment-002.json"
        ),
        "execution_amendment_003_file_sha256": (
            artifact_root / "execution-package-amendment-003.json"
        ),
        "execution_package_file_sha256": artifact_root / "execution-package.json",
        "implementation_file_sha256": (
            repository_root / "src" / "lumi" / "posttraining_stronger_teacher_ab.py"
        ),
        "model_import_receipt_file_sha256": (
            artifact_root / "evidence" / "model-import-receipt.json"
        ),
        "plan_file_sha256": plan_path,
        "runner_file_sha256": repository_root / "scripts" / "run_stronger_teacher_ab.py",
    }
    for name, path in expected_paths.items():
        if bindings.get(name) != scale.file_sha256(path):
            raise ValueError(f"paired A/B boundary binding changed: {name}")
    if bindings.get("baseline_model_sha256") != plan["models"]["baseline"][
        "file_sha256"
    ]:
        raise ValueError("baseline model binding changed")
    if bindings.get("candidate_model_sha256") != plan["models"]["candidate"][
        "file_sha256"
    ]:
        raise ValueError("candidate model binding changed")
    if bindings.get("selected_record_ids_sha256") != preflight.SELECTED_IDS_SHA256:
        raise ValueError("paired record-ID binding changed")
    if (
        bindings.get("selected_schedule_internal_sha256")
        != preflight.SELECTED_SCHEDULE_SHA256
    ):
        raise ValueError("paired schedule binding changed")

    journal = report["journal_summary"]
    if journal.get("generated_output_tokens") != ab._generated_output_tokens(  # noqa: SLF001
        artifact_root
    ):
        raise ValueError("generated output token count changed")
    if journal.get("runtime_retries_used") != ab._runtime_retries_used(  # noqa: SLF001
        artifact_root
    ):
        raise ValueError("runtime retry count changed")
    if journal.get("failure", {}).get("message_sha256") != hashlib.sha256(
        reason.encode("utf-8")
    ).hexdigest():
        raise ValueError("transport failure fingerprint changed")

    tooling = report["evidence_tooling"]
    if tooling.get("finalizer_file_sha256") != scale.file_sha256(
        repository_root / "scripts" / "finalize_stronger_teacher_ab_boundary_stop.py"
    ):
        raise ValueError("boundary finalizer hash changed")
    if tooling.get("verifier_file_sha256") != scale.file_sha256(Path(__file__).resolve()):
        raise ValueError("boundary verifier hash changed")

    safety = ab.runtime_safety_snapshot(  # noqa: SLF001
        artifact_root, Path(plan["execution"]["thermal_probe_cache_path"])
    )
    if safety["thermal_state"] >= 2:
        raise ValueError("Mac thermal state is no longer safe")
    if safety["private_write_bytes"] > ab.MAXIMUM_PRIVATE_WRITE_BYTES:
        raise ValueError("private write boundary exceeded")

    forbidden_final_outputs = (
        artifact_root / "evidence" / "run-report.json",
        artifact_root / "evidence" / "verification-report.json",
        artifact_root / "private" / "owner-review-pairs-40.jsonl",
        artifact_root / "review" / "owner-review-pairs-40.html",
    )
    if any(path.exists() for path in forbidden_final_outputs):
        raise ValueError("incomplete A/B unexpectedly emitted final artifacts")

    receipt = {
        "boundary_stop_report_file_sha256": scale.file_sha256(report_path),
        "boundary_stop_report_internal_sha256": internal,
        "independent_checks": {
            "attempt_math_recomputed": True,
            "baseline_counts_recomputed": True,
            "bindings_recomputed": True,
            "candidate_inference_absent": True,
            "failure_fingerprint_recomputed": True,
            "final_artifacts_absent": True,
            "journal_hash_recomputed": True,
            "prohibitions_preserved": False,
            "report_internal_hash_recomputed": True,
            "runtime_safety_rechecked": True,
        },
        "next_action_authorized": False,
        "schema_version": "lumi.sft-stronger-teacher-ab-boundary-stop-verification.v0.1",
        "status": "PASS",
        "verifier_file_sha256": scale.file_sha256(Path(__file__).resolve()),
    }
    # Every prohibition value must remain false; spell this separately to avoid ambiguity.
    if any(report["prohibitions_preserved"].values()):
        raise ValueError("paired A/B prohibition ledger changed")
    receipt["independent_checks"]["prohibitions_preserved"] = True
    receipt["verification_sha256"] = ab.sha256_value(receipt)
    ab._write_json(  # noqa: SLF001
        artifact_root / "evidence" / "boundary-stop-verification.json",
        receipt,
        refuse_existing=True,
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    args = parser.parse_args()
    receipt = verify(
        plan_path=args.plan.resolve(),
        repository_root=args.repository_root.resolve(),
        artifact_root=args.artifact_root.resolve(),
    )
    print(f"PASS verification_sha256={receipt['verification_sha256']}")


if __name__ == "__main__":
    main()
