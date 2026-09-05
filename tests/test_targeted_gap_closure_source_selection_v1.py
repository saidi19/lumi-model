from __future__ import annotations

import copy
import unittest

from lumi.posttraining_targeted_gap_closure_source_selection_v1 import (
    REPORT_SCHEMA,
    REPORT_STATUS,
    _matrix_totals,
    sha256_value,
    validate_report,
)


MATRIX = {
    "conversation": {"banglish": 100_000, "natural_code_switching": 50_000},
    "factual_qa": {"banglish": 100_000, "natural_code_switching": 50_000},
    "tool_api": {"banglish": 100_000, "natural_code_switching": 50_000},
    "safety_helpfulness": {
        "banglish": 200_000,
        "natural_code_switching": 100_000,
    },
    "writing_summarization": {
        "banglish": 200_000,
        "natural_code_switching": 100_000,
    },
    "multilingual_bengali": {
        "banglish": 100_000,
        "natural_code_switching": 50_000,
    },
}


def _sealed_report() -> dict:
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": REPORT_STATUS,
        "source_selection": {
            "active_selected_sources": [],
            "complete_rights_clear_local_closure_set_exists": False,
            "conditional_candidate": {"active_production_credit": 0},
        },
        "resource_accounting": {
            "candidate_reads": 0,
            "gpu_actions": 0,
            "inference_calls": 0,
            "network_bytes": 0,
            "outreach_actions": 0,
            "private_write_bytes": 0,
            "source_line_reads": 0,
            "test_object_reads": 0,
        },
        "legal_conclusion_made": False,
        "production_eligible": False,
        "raw_training_text_retained": False,
    }
    report["report_sha256"] = sha256_value(report)
    return report


class TargetedGapClosureSelectionTests(unittest.TestCase):
    def test_target_matrix_covers_current_gaps(self) -> None:
        tasks, languages = _matrix_totals(MATRIX)
        self.assertEqual(
            tasks,
            {
                "conversation": 150_000,
                "factual_qa": 150_000,
                "tool_api": 150_000,
                "safety_helpfulness": 300_000,
                "writing_summarization": 300_000,
                "multilingual_bengali": 150_000,
            },
        )
        self.assertEqual(
            languages, {"banglish": 800_000, "natural_code_switching": 400_000}
        )
        self.assertEqual(sum(tasks.values()), 1_200_000)
        self.assertEqual(sum(languages.values()), 1_200_000)
        self.assertGreaterEqual(languages["banglish"], 772_114)
        self.assertGreaterEqual(languages["natural_code_switching"], 365_420)
        self.assertGreaterEqual(tasks["conversation"], 5_934)
        self.assertGreaterEqual(tasks["factual_qa"], 114_306)
        self.assertGreaterEqual(tasks["tool_api"], 2)

    def test_internal_hash_is_deterministic(self) -> None:
        self.assertEqual(
            sha256_value({"b": 2, "a": 1}), sha256_value({"a": 1, "b": 2})
        )

    def test_fail_closed_report_is_valid(self) -> None:
        validate_report(_sealed_report())

    def test_report_rejects_forbidden_drift(self) -> None:
        cases = [
        (("production_eligible",), True),
        (("legal_conclusion_made",), True),
        (("source_selection", "active_selected_sources"), ["forbidden"]),
        (("source_selection", "conditional_candidate", "active_production_credit"), 1),
        (("resource_accounting", "network_bytes"), 1),
        (("resource_accounting", "candidate_reads"), 1),
        ]
        for path, value in cases:
            with self.subTest(path=path):
                report = copy.deepcopy(_sealed_report())
                target = report
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                report["report_sha256"] = sha256_value(
                    {
                        key: item
                        for key, item in report.items()
                        if key != "report_sha256"
                    }
                )
                with self.assertRaises(ValueError):
                    validate_report(report)


if __name__ == "__main__":
    unittest.main()
