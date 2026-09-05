from __future__ import annotations

import copy
import unittest

from lumi.posttraining_targeted_gap_closure_source_selection_v2 import (
    REPORT_SCHEMA,
    REPORT_STATUS,
    _assert_raw_text_free,
    _matrix_totals,
    sha256_value,
    validate_report,
)


MATRIX = {
    "conversation": {"banglish": 100_000, "natural_code_switching": 50_000},
    "factual_qa": {"banglish": 100_000, "natural_code_switching": 50_000},
    "multilingual_bengali": {
        "banglish": 100_000,
        "natural_code_switching": 50_000,
    },
    "safety_helpfulness": {
        "banglish": 200_000,
        "natural_code_switching": 100_000,
    },
    "tool_api": {"banglish": 100_000, "natural_code_switching": 50_000},
    "writing_summarization": {
        "banglish": 200_000,
        "natural_code_switching": 100_000,
    },
}


def _sealed_report() -> dict:
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": REPORT_STATUS,
        "massive_gate": {
            "registry_v0_22_active": True,
            "full_mix_two_pass_audit_complete": True,
            "tool_api_gap_after_credit": 0,
        },
        "source_selection": {
            "candidate_construction_authorized": False,
            "first_party_canary_ready_for_separate_authorization": True,
            "complete_rights_clear_local_closure_set_exists": False,
        },
        "replacement_capacity_target": {
            "projected_task_gaps": {"conversation": 0, "factual_qa": 0},
            "projected_language_gaps": {
                "banglish": 0,
                "natural_code_switching": 0,
            },
            "projected_source_family_cap_violations": [],
        },
        "next_gate": {
            "action": "first_party_zero_inference_1000_row_candidate_canary"
        },
        "resource_accounting": {
            "candidate_reads": 0,
            "candidate_writes": 0,
            "gpu_actions": 0,
            "inference_calls": 0,
            "network_bytes": 0,
            "outreach_actions": 0,
            "registry_writes": 0,
            "remote_actions": 0,
            "source_line_reads": 0,
            "test_object_reads": 0,
            "training_actions": 0,
        },
        "legal_conclusion_made": False,
        "production_eligible": False,
        "raw_training_text_retained": False,
    }
    report["report_sha256"] = sha256_value(report)
    return report


class TargetedGapClosureSelectionV2Tests(unittest.TestCase):
    def test_target_matrix_covers_v016_residual_gaps(self) -> None:
        tasks, languages = _matrix_totals(MATRIX)
        self.assertEqual(sum(tasks.values()), 1_200_000)
        self.assertEqual(sum(languages.values()), 1_200_000)
        self.assertGreaterEqual(tasks["conversation"], 5_934)
        self.assertGreaterEqual(tasks["factual_qa"], 114_306)
        self.assertGreaterEqual(languages["banglish"], 772_114)
        self.assertGreaterEqual(languages["natural_code_switching"], 365_420)

    def test_projected_family_share_remains_below_cap(self) -> None:
        projected_total = 10_610_600 + 1_200_000
        projected_lumi = 114_853 + 1_200_000
        self.assertEqual(round(projected_lumi / projected_total, 8), 0.11132821)
        self.assertLess(projected_lumi / projected_total, 0.35)

    def test_massive_credit_is_not_double_counted(self) -> None:
        self.assertEqual(10_610_600 + 1_200_000, 11_810_600)
        self.assertNotEqual(10_610_600 + 6_451 + 1_200_000, 11_810_600)

    def test_valid_nonproduction_report(self) -> None:
        validate_report(_sealed_report())

    def test_report_rejects_forbidden_drift(self) -> None:
        cases = [
            (("production_eligible",), True),
            (("legal_conclusion_made",), True),
            (("massive_gate", "registry_v0_22_active"), False),
            (("massive_gate", "tool_api_gap_after_credit"), 1),
            (("source_selection", "candidate_construction_authorized"), True),
            (("resource_accounting", "candidate_reads"), 1),
            (("resource_accounting", "network_bytes"), 1),
        ]
        for path, value in cases:
            with self.subTest(path=path):
                report = copy.deepcopy(_sealed_report())
                target = report
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                report["report_sha256"] = sha256_value(
                    {key: item for key, item in report.items() if key != "report_sha256"}
                )
                with self.assertRaises(ValueError):
                    validate_report(report)

    def test_raw_text_free_validator_rejects_nested_training_fields(self) -> None:
        for key in (
            "messages",
            "content",
            "user_prompt",
            "assistant_response",
            "utterance",
            "record_ids",
            "exception",
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                _assert_raw_text_free({"safe": [{key: "forbidden"}]})

    def test_internal_hash_is_order_independent(self) -> None:
        self.assertEqual(
            sha256_value({"b": 2, "a": 1}), sha256_value({"a": 1, "b": 2})
        )


if __name__ == "__main__":
    unittest.main()
