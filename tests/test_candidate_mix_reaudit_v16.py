from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import lumi.posttraining_candidate_mix_reaudit_v16 as contract


def _massive_rows() -> list[dict]:
    return [
        {
            "record_id": f"massive-bn-bd-parser-{index:024x}",
            "language": "bn",
            "messages": [
                {"role": "user", "content": f"synthetic user {index}"},
                {"role": "tool", "content": "synthetic tool"},
                {"role": "tool_result", "content": "synthetic result"},
                {"role": "assistant", "content": "synthetic answer"},
            ],
            "provenance": {
                "owner_review_status": "pending_deterministic_64_row_spot_check",
                "source_registry_status": "pending_separate_promotion",
            },
            "source": {
                "dataset_id": "AmazonScience/massive",
                "license_id": "cc-by-4.0",
                "revision": contract.MASSIVE_SOURCE["revision"],
                "source_id": contract.MASSIVE_SOURCE["source_id"],
            },
            "split": "train",
            "task_category": "tool_api",
        }
        for index in range(64)
    ]


def _massive_metrics() -> dict:
    return {
        "by_split": {
            "train": {
                "records": 64,
                "serialized_tokens": 12_754,
                "supervised_tokens": 6_451,
            },
            "validation": {
                "records": 0,
                "serialized_tokens": 0,
                "supervised_tokens": 0,
            },
        },
        "candidate_file_sha256": contract.EXPECTED_INPUTS["massive_candidate"][
            "sha256"
        ],
        "historical_owner_review_pending_rows": 64,
        "historical_registry_pending_rows": 64,
        "language_mode_policy": "language_field",
        "maximum_record_tokens": 309,
        "owner_review_external_receipt_resolves_historical_pending": True,
        "records": 64,
        "registry_v0_22_receipt_resolves_historical_pending": True,
        "rows_with_pending_owner_review_provenance": 0,
        "train_by_language_mode": {"bengali_script": 6_451},
        "train_by_task": {"tool_api": 6_451},
    }


def _synthetic_report() -> dict:
    source_metrics = {f"source-{index}": {} for index in range(17)}
    source_metrics[contract.MASSIVE_SOURCE["source_id"]] = _massive_metrics()
    report: dict = {
        "authorization": contract.EXPECTED_AUTHORIZATION,
        "candidate_pool": {
            "accepted_records_after_generic_quarantine": 314_780,
            "accepted_train_supervised_tokens": 10_610_600,
            "accepted_validation_supervised_tokens": 628_604,
            "candidate_records_read": 314_780,
            "family_train_supervised_tokens": contract.EXPECTED_FAMILY_TOKENS,
            "input_train_supervised_tokens": 10_610_600,
            "input_validation_supervised_tokens": 628_604,
            "language_train_supervised_tokens_behavior_safe": (
                contract.EXPECTED_LANGUAGE_BEHAVIOR
            ),
            "numeric_gap_to_10m": 0,
            "source_train_supervised_tokens": {
                contract.MASSIVE_SOURCE["source_id"]: 6_451
            },
            "strict_targeted_gap_lower_bound": 1_137_534,
            "task_train_supervised_tokens_behavior_safe": contract.EXPECTED_TASK_BEHAVIOR,
        },
        "closure": {
            "blockers": contract.EXPECTED_CLOSURE_BLOCKERS,
            "family_cap_dilution_gap": 0,
            "language_floor_gaps_behavior_safe": contract.EXPECTED_LANGUAGE_GAPS,
            "production_dataset_ready": False,
            "source_family_cap": 0.35,
            "source_family_cap_violations": [],
            "source_family_shares": contract.EXPECTED_FAMILY_SHARES,
            "task_floor_gaps_behavior_safe": contract.EXPECTED_TASK_GAPS,
        },
        "evaluation_prohibition": contract.EXPECTED_EVALUATION_PROHIBITION,
        "exact_duplicate_audit": {
            "cross_source_duplicate_groups": 0,
            "duplicate_extra_records": 0,
            "duplicate_groups": 0,
            "train_validation_overlap_groups": 0,
        },
        "excluded_source_gate": {
            "candidate_rows_from_excluded_sources": 0,
            "predecessor_evidence_replayed": [
                {"counted_train_supervised_tokens": 0, "source_id": "openr1_math_220k"}
            ],
            "registry_sources_not_in_candidate_inventory": sorted(
                contract.EXPECTED_EXCLUDED_SOURCE_IDS
            ),
        },
        "external_gate_resolution": {
            "massive_historical_owner_review_pending_rows": 64,
            "massive_historical_registry_pending_rows": 64,
            "owner_review_receipt_accepted_rows": 64,
            "owner_review_receipt_rejected_rows": 0,
            "row_mutation_used": False,
            "source_registry_v0_22_materialized": True,
        },
        "generic_sft_validation": {
            "accepted": 314_780,
            "rejected": 0,
            "rejection_counts": {},
            "total": 314_780,
        },
        "input_projection": contract.EXPECTED_INPUT_PROJECTION,
        "near_duplicate_audit": {
            "comparison_unit": "role_tagged_full_conversation",
            "component_ledger_sha256": (
                "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
            ),
            "components": 314_780,
            "cross_source_components": 0,
            "duplicate_extra_records": 0,
            "edge_count": 0,
            "largest_component_records": 1,
            "multi_record_components": 0,
            "records": 314_780,
            "train_validation_components": 0,
            "train_validation_records": 0,
        },
        "next_gate": {
            "action": "close_residual_behavior_safe_task_and_language_gaps",
            "authorization_required": True,
            "priority": "targeted_gap_closure",
        },
        "observed_at": "2026-09-01",
        "plan_file_sha256": "1" * 64,
        "production_eligible": False,
        "quarantine": {"records": [], "total_records": 0},
        "raw_training_text_retained": False,
        "registry": {
            "file_sha256": contract.EXPECTED_INPUTS["source_registry"]["sha256"],
            "source_count": 25,
        },
        "resource_accounting": {
            "candidate_row_reads": 314_780,
            "full_audit_passes": 1,
            "registry_writes": 0,
            "source_line_reads": 0,
            "test_object_reads": 0,
        },
        "schema_version": contract.REPORT_SCHEMA_VERSION,
        "source_metrics": source_metrics,
        "status": contract.REPORT_STATUS,
    }
    report["report_sha256"] = contract.sha256_value(report)
    return report


class CandidateMixReauditV16Tests(unittest.TestCase):
    def test_exact_expected_projection_and_limits(self) -> None:
        self.assertEqual(contract.EXPECTED_INCLUDED_SOURCE_COUNT, 18)
        self.assertEqual(contract.EXPECTED_REGISTRY_SOURCE_COUNT, 25)
        self.assertEqual(contract.EXPECTED_INPUT_RECORDS, 314_716 + 64)
        self.assertEqual(contract.EXPECTED_INPUT_TRAIN_TOKENS, 10_604_149 + 6_451)
        self.assertEqual(
            contract.EXPECTED_LIMITS["maximum_total_candidate_row_reads"],
            2 * contract.EXPECTED_INPUT_RECORDS,
        )

    def test_train_only_metrics_are_normalized_for_aggregate(self) -> None:
        metrics = {
            "by_split": {
                "train": {
                    "records": 64,
                    "serialized_tokens": 12_754,
                    "supervised_tokens": 6_451,
                },
                "validation": {},
            }
        }
        contract._normalize_train_only_source_metrics(metrics)
        self.assertEqual(
            metrics["by_split"]["validation"],
            {"records": 0, "serialized_tokens": 0, "supervised_tokens": 0},
        )

    def test_massive_historical_gates_are_counted_without_row_mutation(self) -> None:
        rows = _massive_rows()
        original = copy.deepcopy(rows)
        self.assertEqual(
            contract._validate_massive_rows(rows),
            {
                "historical_owner_review_pending_rows": 64,
                "historical_registry_pending_rows": 64,
            },
        )
        self.assertEqual(rows, original)

    def test_massive_duplicate_record_id_is_rejected(self) -> None:
        rows = _massive_rows()
        rows[1]["record_id"] = rows[0]["record_id"]
        with self.assertRaisesRegex(ValueError, "row contract"):
            contract._validate_massive_rows(rows)

    def test_small_full_dedup_detects_exact_near_and_id_collision(self) -> None:
        rows = _massive_rows()[:2]
        rows[1]["messages"] = copy.deepcopy(rows[0]["messages"])
        duplicate_index: dict[str, list[tuple[str, str]]] = {}
        for row in rows:
            digest = contract._canonical_message_digest(row["messages"])
            duplicate_index.setdefault(digest, []).append(
                (row["source"]["source_id"], row["split"])
            )
        exact = contract._duplicate_snapshot(duplicate_index)
        self.assertEqual(exact["duplicate_groups"], 1)
        near = contract.v2._near_duplicate_audit(
            rows,
            policy=contract.EXPECTED_NEAR_DUPLICATE_POLICY,
        )
        self.assertEqual(near["multi_record_components"], 1)
        rows[1]["record_id"] = rows[0]["record_id"]
        with self.assertRaisesRegex(ValueError, "globally unique"):
            contract.v2._near_duplicate_audit(
                rows,
                policy=contract.EXPECTED_NEAR_DUPLICATE_POLICY,
            )

    def test_report_validator_freezes_clean_residual_gap_boundary(self) -> None:
        report = _synthetic_report()
        contract.validate_report(report)
        changed = copy.deepcopy(report)
        changed["closure"]["task_floor_gaps_behavior_safe"]["tool_api"] = 1
        changed["report_sha256"] = contract.sha256_value(
            {key: value for key, value in changed.items() if key != "report_sha256"}
        )
        with self.assertRaisesRegex(ValueError, "boundary or totals"):
            contract.validate_report(changed)

    def test_report_validator_rejects_nested_raw_text(self) -> None:
        report = _synthetic_report()
        report["nested"] = {"messages": ["synthetic"]}
        report["report_sha256"] = contract.sha256_value(
            {key: value for key, value in report.items() if key != "report_sha256"}
        )
        with self.assertRaisesRegex(ValueError, "raw training text"):
            contract.validate_report(report)

    def test_verification_accounts_for_exactly_two_passes(self) -> None:
        report = _synthetic_report()
        verification = contract.build_verification(
            plan={"observed_at": "2026-09-01"},
            report=report,
            report_file_sha256="2" * 64,
        )
        self.assertEqual(
            verification["resource_accounting"]["candidate_row_reads"],
            629_560,
        )
        self.assertIs(verification["production_eligible"], False)

    def test_marker_cannot_overwrite_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text("existing", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                contract.start_marker(path, contract.EXPECTED_RECEIPT_MARKER)


if __name__ == "__main__":
    unittest.main()
