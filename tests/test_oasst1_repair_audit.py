from __future__ import annotations

from lumi.posttraining_canary import adapt_oasst1_rows
from lumi.posttraining_source_quality import build_oasst1_repair_audit_report


def test_repaired_oasst1_report_closes_context_and_rank_findings() -> None:
    raw = {
        "num_rows_total": 3,
        "partial": False,
        "rows": [
            {
                "row_idx": 0,
                "row": {
                    "deleted": False,
                    "lang": "en",
                    "message_id": "prompt",
                    "parent_id": None,
                    "review_result": True,
                    "role": "prompter",
                    "synthetic": False,
                    "text": "Explain recursion.",
                },
            },
            {
                "row_idx": 1,
                "row": {
                    "deleted": False,
                    "lang": "en",
                    "message_id": "answer",
                    "parent_id": "prompt",
                    "rank": 0,
                    "review_result": True,
                    "role": "assistant",
                    "synthetic": False,
                    "text": "Recursion solves a problem using smaller instances.",
                },
            },
            {
                "row_idx": 2,
                "row": {
                    "deleted": False,
                    "lang": "en",
                    "message_id": "unranked",
                    "parent_id": "prompt",
                    "rank": None,
                    "review_result": True,
                    "role": "assistant",
                    "synthetic": False,
                    "text": "Skip this.",
                },
            },
        ],
    }
    accepted = adapt_oasst1_rows(raw["rows"], revision="a" * 40, target_records=24)
    report = build_oasst1_repair_audit_report(
        accepted=accepted,
        adapter_implementation_sha256="b" * 64,
        filter_report={"accepted": 1, "rejected": 0, "report_sha256": "c" * 64},
        metadata_sha256="d" * 64,
        observed_at="2026-08-06",
        raw_payload=raw,
        raw_rows_sha256="e" * 64,
        source_revision="a" * 40,
        target_records=24,
    )
    assert report["accepted_candidate_records"] == 1
    assert report["structural_metrics"]["rank_counts"] == {"0": 1}
    assert report["structural_metrics"]["followup_pairs_missing_ancestor_history"] == 0
    assert report["decision"] == "structural_repair_passed_source_still_blocked"
    assert report["production_eligible"] is False
