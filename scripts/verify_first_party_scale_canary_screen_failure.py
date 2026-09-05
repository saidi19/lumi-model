#!/usr/bin/env python3
"""Independently verify the raw-text-free 50-call quality-stop evidence."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from lumi import posttraining_first_party_scale_canary as scale


def _read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root changed: {path}")
    return value


def verify(*, plan_path: Path, repository_root: Path, artifact_root: Path) -> dict:
    plan = scale.load_plan(plan_path, repository_root=repository_root)
    report_path = artifact_root / "evidence" / "quality-screen-failure-report.json"
    report = _read_object(report_path)
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256", None)
    if internal != scale.sha256_value(unsigned):
        raise ValueError("quality-stop report internal hash changed")
    if report.get("status") != "stopped_fail_closed_quality_screen_failed":
        raise ValueError("quality-stop status changed")

    schedule = scale.build_schedule()
    journal_path = artifact_root / "private" / "call-journal.jsonl"
    journal = []
    with journal_path.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("journal row root changed")
            journal.append(value)
    starts = [event for event in journal if event.get("event") == "call_started"]
    terminals = [
        event
        for event in journal
        if event.get("event") in {"call_finished", "call_abandoned"}
    ]
    if len(starts) != 50 or len(terminals) != 50:
        raise ValueError("journal is not an exact 50-call terminal prefix")
    if [event["record_id"] for event in starts] != [slot.record_id for slot in schedule[:50]]:
        raise ValueError("journal start prefix changed")
    terminal_by_id = {event["record_id"]: event for event in terminals}
    ordered = [terminal_by_id[slot.record_id] for slot in schedule[:50]]
    if sum(event.get("accepted") is True for event in ordered) != 30:
        raise ValueError("cumulative accepted count changed")
    if sum(event.get("accepted") is True for event in ordered[30:50]) != 9:
        raise ValueError("amended-screen accepted count changed")

    supervised = Counter()
    for slot, event in zip(schedule[:50], ordered, strict=True):
        if event.get("accepted") is True:
            supervised[slot.split] += int(event["token_counts"]["supervised_tokens"])
    if dict(supervised) != report.get("supervised_tokens"):
        raise ValueError("quality-stop supervised token totals changed")
    if report["bindings"]["journal_file_sha256"] != scale.file_sha256(journal_path):
        raise ValueError("quality-stop journal binding changed")
    if report["bindings"]["plan_file_sha256"] != scale.file_sha256(plan_path):
        raise ValueError("quality-stop plan binding changed")
    if report["bindings"]["model_file_sha256"] != plan["model"]["file_sha256"]:
        raise ValueError("quality-stop model binding changed")
    if report["evidence_tooling"]["finalizer_file_sha256"] != scale.file_sha256(
        repository_root / "scripts" / "finalize_first_party_scale_canary_screen_failure.py"
    ):
        raise ValueError("quality-stop finalizer binding changed")
    if report["evidence_tooling"]["verifier_file_sha256"] != scale.file_sha256(
        Path(__file__).resolve()
    ):
        raise ValueError("quality-stop verifier binding changed")

    forbidden_final_outputs = (
        artifact_root / "private" / "candidate-rows.jsonl",
        artifact_root / "review" / "owner-review-100.html",
        artifact_root / "evidence" / "run-report.json",
    )
    if any(path.exists() for path in forbidden_final_outputs):
        raise ValueError("failed canary unexpectedly emitted a final candidate artifact")

    receipt = {
        "calls_unspent": 950,
        "independent_checks": {
            "accepted_counts_recomputed": True,
            "exact_schedule_prefix_recomputed": True,
            "evidence_tool_hashes_recomputed": True,
            "final_candidate_artifacts_absent": True,
            "journal_hash_recomputed": True,
            "model_and_plan_bindings_recomputed": True,
            "report_internal_hash_recomputed": True,
            "supervised_tokens_recomputed": True,
        },
        "no_registry_seal_sft_or_production_authorized": True,
        "quality_stop_report_file_sha256": scale.file_sha256(report_path),
        "quality_stop_report_internal_sha256": internal,
        "verifier_file_sha256": scale.file_sha256(Path(__file__).resolve()),
        "schema_version": "lumi.sft-first-party-scale-canary-quality-stop-verification.v0.1",
        "status": "PASS",
    }
    receipt["verification_sha256"] = scale.sha256_value(receipt)
    scale._write_json(  # noqa: SLF001
        artifact_root / "evidence" / "quality-screen-failure-verification.json", receipt
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
