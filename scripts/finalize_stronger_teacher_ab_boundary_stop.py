#!/usr/bin/env python3
"""Seal a raw-text-free fail-closed boundary report for the paired teacher A/B."""

from __future__ import annotations

import argparse
import hashlib
import time
from collections import Counter
from pathlib import Path

from lumi import posttraining_stronger_teacher_ab as ab
from lumi import posttraining_stronger_teacher_ab_preflight as preflight
from lumi import posttraining_first_party_scale_canary as scale


REPORT_SCHEMA_VERSION = "lumi.sft-stronger-teacher-ab-boundary-stop.v0.1"
REPORT_STATUS = (
    "stopped_fail_closed_attempt_budget_insufficient_after_local_transport_failure"
)


def _reason_group(reason: object) -> str:
    text = str(reason or "")
    if text.startswith("RemoteDisconnected:"):
        return "local_transport_remote_disconnected"
    if "teacher_did_not_finish_with_stop" in text:
        return "runtime_incomplete_generation"
    if "language:" in text:
        return "language_contract"
    if "safety:" in text:
        return "safety_contract"
    if "grounding:" in text or "required_literal_missing" in text:
        return "grounding_contract"
    if "quality:" in text:
        return "quality_contract"
    if "punctuation" in text:
        return "punctuation_contract"
    if "dedup" in text or "near_duplicate" in text:
        return "deduplication_contract"
    if "benchmark" in text:
        return "benchmark_decontamination_contract"
    if text.startswith(("JSONDecodeError:", "TypeError:", "ValueError:schema")):
        return "schema_or_json_contract"
    return "other"


def _file_bindings(
    *, plan_path: Path, repository_root: Path, artifact_root: Path
) -> dict[str, str]:
    paths = {
        "authorization_file_sha256": artifact_root / "authorization.json",
        "baseline_journal_file_sha256": (
            artifact_root / "private" / "baseline-call-journal.jsonl"
        ),
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
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"paired A/B boundary binding missing: {missing}")
    return {name: scale.file_sha256(path) for name, path in paths.items()}


def _journal_summary(artifact_root: Path) -> dict:
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
    runtime_complete = {
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
    unstarted_ids = {slot.record_id for slot in schedule} - started_ids
    accepted = sum(event.get("accepted") is True for event in runtime_complete.values())
    rejection_groups = Counter(
        _reason_group(event.get("rejection_reason"))
        for event in runtime_complete.values()
        if event.get("accepted") is not True
    )
    rejection_groups.update(
        _reason_group(event.get("rejection_reason")) for event in incomplete.values()
    )
    candidate_path = artifact_root / "private" / "candidate-call-journal.jsonl"
    candidate = ab._read_jsonl(candidate_path)  # noqa: SLF001
    candidate_starts = [row for row in candidate if row.get("event") == "call_started"]

    if len(schedule) != 100:
        raise ValueError("paired A/B schedule count changed")
    if len(starts) != 116 or len(baseline) != 232:
        raise ValueError("paired A/B baseline journal boundary changed")
    if len(runtime_complete) != 95 or len(incomplete) != 1 or len(unstarted_ids) != 4:
        raise ValueError("paired A/B output boundary changed")
    if accepted != 40:
        raise ValueError("paired A/B accepted baseline count changed")
    if candidate_starts:
        raise ValueError("candidate inference unexpectedly started")

    incomplete_event = next(iter(incomplete.values()))
    reason = str(incomplete_event.get("rejection_reason") or "")
    if not reason.startswith("RemoteDisconnected:"):
        raise ValueError("paired A/B terminal failure changed")
    if incomplete_event.get("raw_response_text") not in {"", None}:
        raise ValueError("transport failure unexpectedly retained response text")

    return {
        "accepted_runtime_complete_outputs": accepted,
        "baseline_event_rows": len(baseline),
        "baseline_final_records": sum(event is not None for event in final_by_id.values()),
        "baseline_incomplete_final_records": len(incomplete),
        "baseline_runtime_complete_outputs": len(runtime_complete),
        "baseline_unstarted_records": len(unstarted_ids),
        "call_abandoned_events": sum(
            row.get("event") == "call_abandoned" for row in baseline
        ),
        "call_finished_events": sum(
            row.get("event") == "call_finished" for row in baseline
        ),
        "call_started_events": len(starts),
        "candidate_call_started_events": len(candidate_starts),
        "candidate_final_outputs": 0,
        "failure": {
            "classifier_result_in_v0_1": "non_retryable",
            "error_class": "RemoteDisconnected",
            "message_sha256": hashlib.sha256(reason.encode("utf-8")).hexdigest(),
            "record_id_sha256": hashlib.sha256(
                str(incomplete_event["record_id"]).encode("utf-8")
            ).hexdigest(),
            "response_bytes": 0,
        },
        "generated_output_tokens": ab._generated_output_tokens(artifact_root),  # noqa: SLF001
        "rejection_groups": dict(sorted(rejection_groups.items())),
        "runtime_retries_used": ab._runtime_retries_used(artifact_root),  # noqa: SLF001
        "unique_baseline_records_started": len(started_ids),
    }


def finalize(*, plan_path: Path, repository_root: Path, artifact_root: Path) -> dict:
    plan = ab.load_plan(plan_path, repository_root=repository_root)
    expected_artifact = ab.artifact_root(plan, repository_root)
    if artifact_root.resolve() != expected_artifact:
        raise ValueError("paired A/B artifact root changed")

    summary = _journal_summary(artifact_root)
    safety = ab.runtime_safety_snapshot(  # noqa: SLF001
        artifact_root, Path(plan["execution"]["thermal_probe_cache_path"])
    )
    attempts_used = int(summary["call_started_events"])
    attempts_remaining = ab.MAXIMUM_ATTEMPTS - attempts_used
    baseline_outputs_remaining = 100 - int(summary["baseline_runtime_complete_outputs"])
    candidate_outputs_remaining = 100 - int(summary["candidate_final_outputs"])
    minimum_additional_attempts = baseline_outputs_remaining + candidate_outputs_remaining
    minimum_total_attempts = attempts_used + minimum_additional_attempts
    shortfall = minimum_additional_attempts - attempts_remaining
    if (
        attempts_used != 116
        or attempts_remaining != 104
        or baseline_outputs_remaining != 5
        or candidate_outputs_remaining != 100
        or minimum_additional_attempts != 105
        or minimum_total_attempts != 221
        or shortfall != 1
    ):
        raise ValueError("paired A/B attempt-boundary arithmetic changed")
    if summary["runtime_retries_used"] != ab.MAXIMUM_RETRIES:
        raise ValueError("paired A/B runtime-retry boundary changed")

    report = {
        "attempt_boundary": {
            "authorized_attempts": ab.MAXIMUM_ATTEMPTS,
            "attempts_remaining": attempts_remaining,
            "attempts_used": attempts_used,
            "baseline_outputs_remaining": baseline_outputs_remaining,
            "candidate_outputs_remaining": candidate_outputs_remaining,
            "minimum_additional_attempts_for_exact_200_outputs": (
                minimum_additional_attempts
            ),
            "minimum_total_attempts_for_exact_200_outputs": minimum_total_attempts,
            "original_authorization_shortfall_attempts": shortfall,
            "runtime_retries_authorized": ab.MAXIMUM_RETRIES,
            "runtime_retries_used": summary["runtime_retries_used"],
        },
        "bindings": {
            **_file_bindings(
                plan_path=plan_path,
                repository_root=repository_root,
                artifact_root=artifact_root,
            ),
            "baseline_model_sha256": plan["models"]["baseline"]["file_sha256"],
            "candidate_model_sha256": plan["models"]["candidate"]["file_sha256"],
            "selected_record_ids_sha256": plan["execution"][
                "selected_record_ids_sha256"
            ],
            "selected_schedule_internal_sha256": plan["execution"][
                "selected_schedule_internal_sha256"
            ],
        },
        "evidence_tooling": {
            "finalizer_file_sha256": scale.file_sha256(Path(__file__).resolve()),
            "verifier_file_sha256": scale.file_sha256(
                repository_root
                / "scripts"
                / "verify_stronger_teacher_ab_boundary_stop.py"
            ),
        },
        "journal_summary": summary,
        "next_gate": {
            "action": "new_explicit_result_preserving_continuation_authorization_required",
            "candidate_inference_may_start_before_gate": False,
            "minimum_cumulative_attempt_ceiling": minimum_total_attempts,
            "recommended_cumulative_attempt_ceiling": 260,
            "recommended_cumulative_runtime_retry_ceiling": 60,
            "required_implementation_correction": (
                "classify only local no-response transport disconnects as retryable while "
                "keeping quality failures non-retryable"
            ),
        },
        "observed_unix": int(time.time()),
        "prohibitions_preserved": {
            "candidate_registry_changed": False,
            "cloud_or_paid_gpu_used": False,
            "dataset_sealed": False,
            "external_api_used": False,
            "external_distribution": False,
            "human_authorship_claimed": False,
            "package_installed": False,
            "production_promoted": False,
            "sft_training_started": False,
            "unrelated_data_accessed": False,
        },
        "reason": (
            "One local no-response transport disconnect occurred after all 20 internal "
            "runtime retries were consumed. The exact 200-output result now requires at "
            "least 105 additional inference starts, but only 104 remain under the original "
            "220-start authorization."
        ),
        "runtime_safety_snapshot": {
            **safety,
            "maximum_private_write_bytes": ab.MAXIMUM_PRIVATE_WRITE_BYTES,
        },
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": REPORT_STATUS,
    }
    report["report_sha256"] = ab.sha256_value(report)
    ab._write_json(  # noqa: SLF001
        artifact_root / "evidence" / "boundary-stop-report.json",
        report,
        refuse_existing=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    args = parser.parse_args()
    report = finalize(
        plan_path=args.plan.resolve(),
        repository_root=args.repository_root.resolve(),
        artifact_root=args.artifact_root.resolve(),
    )
    boundary = report["attempt_boundary"]
    print(
        "STOPPED_FAIL_CLOSED "
        f"used={boundary['attempts_used']}/{boundary['authorized_attempts']} "
        f"minimum_total={boundary['minimum_total_attempts_for_exact_200_outputs']}"
    )


if __name__ == "__main__":
    main()
