#!/usr/bin/env python3
"""Seal a raw-text-free fail-closed report for the 50-call quality screen."""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

from lumi import posttraining_first_party_scale_canary as scale


def _reason_group(reason: object) -> str:
    text = str(reason or "")
    if "quality:user_prompt_prefix_echo" in text or "quality:substantial_user_prompt_echo" in text:
        return "quality_prompt_echo"
    if "quality:repeated_" in text:
        return "quality_repetition"
    if "teacher_did_not_finish_with_stop" in text:
        return "runtime_incomplete_generation"
    if "required_literal_missing" in text:
        return "grounding_required_literal_missing"
    if "terminal punctuation" in text:
        return "punctuation_contract"
    if "language:" in text:
        return "language_contract"
    if "Ollama request failed" in text:
        return "local_transport"
    return "other"


def finalize(*, plan_path: Path, repository_root: Path, artifact_root: Path) -> dict:
    plan = scale.load_plan(plan_path, repository_root=repository_root)
    schedule = scale.build_schedule()
    journal_path = artifact_root / "private" / "call-journal.jsonl"
    journal = scale._read_jsonl(journal_path)  # noqa: SLF001
    starts, terminals = scale._journal_state(journal, schedule)  # noqa: SLF001
    if len(starts) != 50 or len(terminals) != 50:
        raise ValueError("quality-screen finalization requires exactly 50 terminal calls")

    ordered = [terminals[slot.record_id] for slot in schedule[:50]]
    accepted = [event for event in ordered if event.get("accepted") is True]
    screen = ordered[30:50]
    screen_accepted = [event for event in screen if event.get("accepted") is True]
    if len(accepted) != 30 or len(screen_accepted) != 9:
        raise ValueError("quality-screen outcome changed")

    slot_by_id = {slot.record_id: slot for slot in schedule}
    supervised = Counter()
    accepted_cells = Counter()
    for event in accepted:
        slot = slot_by_id[event["record_id"]]
        supervised[slot.split] += int(event["token_counts"]["supervised_tokens"])
        accepted_cells[(slot.task, slot.language_mode)] += 1
    if supervised != Counter({"train": 1721, "validation": 306}):
        raise ValueError("quality-screen token totals changed")

    screen_rejections = Counter(
        _reason_group(event.get("rejection_reason"))
        for event in screen
        if event.get("accepted") is not True
    )
    report = {
        "acceptance": {
            "amended_screen_accepted": len(screen_accepted),
            "amended_screen_calls": len(screen),
            "amended_screen_rate": len(screen_accepted) / len(screen),
            "authorized_minimum_overall_rate": scale.MINIMUM_OVERALL_ACCEPTANCE,
            "cumulative_accepted": len(accepted),
            "cumulative_calls": len(ordered),
            "cumulative_rate": len(accepted) / len(ordered),
        },
        "accepted_cell_counts": {
            f"{task}|{language}": count
            for (task, language), count in sorted(accepted_cells.items())
        },
        "bindings": {
            "authorization_file_sha256": scale.file_sha256(
                scale._resolve(  # noqa: SLF001
                    repository_root, plan["bindings"]["authorization"]["path"]
                )
            ),
            "execution_amendment_001_file_sha256": scale.file_sha256(
                artifact_root / "execution-package-amendment-001.json"
            ),
            "execution_amendment_002_file_sha256": scale.file_sha256(
                artifact_root / "execution-package-amendment-002.json"
            ),
            "journal_file_sha256": scale.file_sha256(journal_path),
            "model_file_sha256": plan["model"]["file_sha256"],
            "plan_file_sha256": scale.file_sha256(plan_path),
            "schedule_internal_sha256": scale.schedule_sha256(schedule),
        },
        "evidence_tooling": {
            "finalizer_file_sha256": scale.file_sha256(Path(__file__).resolve()),
            "verifier_file_sha256": scale.file_sha256(
                repository_root
                / "scripts"
                / "verify_first_party_scale_canary_screen_failure.py"
            ),
        },
        "call_budget": {
            "authorized_calls": scale.MAXIMUM_CALLS,
            "calls_used": len(ordered),
            "calls_unspent": scale.MAXIMUM_CALLS - len(ordered),
        },
        "manual_diagnostic": {
            "claim": "operator_ai_diagnostic_not_human_owner_review",
            "findings": [
                "accepted outputs still included unnatural Banglish grammar",
                "accepted outputs still included instruction fragments or prompt-like phrasing",
                "accepted Bengali and code-switched outputs included awkward or incomplete phrasing",
                "accepted-output quality was therefore materially below the useful-data target",
            ],
        },
        "next_gate": (
            "new_explicit_teacher_strategy_preflight_and_authorization_required_before_any_"
            "additional_generation"
        ),
        "observed_unix": int(time.time()),
        "prohibitions_preserved": {
            "candidate_registry_changed": False,
            "dataset_sealed": False,
            "external_api_used": False,
            "external_distribution": False,
            "network_download_used": False,
            "paid_or_cloud_gpu_used": False,
            "production_promoted": False,
            "sft_training_started": False,
        },
        "reason": (
            "The amended 20-call screen accepted 45 percent, below the 85 percent gate, and "
            "manual diagnostic inspection found material quality defects among accepted rows."
        ),
        "rejection_groups_amended_screen": dict(sorted(screen_rejections.items())),
        "schema_version": "lumi.sft-first-party-scale-canary-quality-stop.v0.1",
        "status": "stopped_fail_closed_quality_screen_failed",
        "supervised_tokens": dict(supervised),
    }
    report["report_sha256"] = scale.sha256_value(report)
    scale._write_json(  # noqa: SLF001
        artifact_root / "evidence" / "quality-screen-failure-report.json", report
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
    print(
        f"STOPPED calls={report['acceptance']['cumulative_calls']} "
        f"accepted={report['acceptance']['cumulative_accepted']} "
        f"screen_rate={report['acceptance']['amended_screen_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
