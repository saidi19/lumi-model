from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lumi.posttraining_first_party_zero_inference_closure_preflight_v1 import (
    EXPECTED_LANGUAGE_TOTALS,
    EXPECTED_TASK_LANGUAGE_MATRIX,
    EXPECTED_TASK_TOTALS,
    build_report,
    build_verification,
    load_plan,
    sha256_file,
    sha256_value,
    validate_report,
    validate_verification,
)


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sealed(value: dict[str, object], field: str) -> dict[str, object]:
    result = dict(value)
    result[field] = sha256_value(result)
    return result


def _fixture_plan(root: Path) -> tuple[Path, dict[str, object]]:
    current_state = {
        "accepted_records_after_generic_quarantine": 314_716,
        "accepted_source_count": 17,
        "accepted_train_supervised_tokens": 10_604_149,
        "accepted_validation_supervised_tokens": 628_604,
        "language_gaps": {
            "banglish": 772_114,
            "bengali_script": 0,
            "english": 0,
            "natural_code_switching": 365_420,
        },
        "registry_semantic_sha256": "0" * 64,
        "registry_source_count": 24,
        "strict_targeted_gap_lower_bound": 1_137_534,
        "task_gaps": {
            "coding": 0,
            "conversation": 5_934,
            "factual_qa": 114_306,
            "multilingual_bengali": 0,
            "reasoning_math_science": 0,
            "safety_helpfulness": 0,
            "tool_api": 2,
            "writing_summarization": 0,
        },
    }
    selection = _sealed(
        {
            "schema_version": "lumi.sft-targeted-gap-closure-source-selection-report.v0.1",
            "status": "STOP_no_complete_rights_clear_local_closure_set",
            "current_verified_state": current_state,
            "replacement_capacity_target": {
                "language_totals": EXPECTED_LANGUAGE_TOTALS,
                "minimum_new_source_families": 2,
                "target_train_supervised_tokens": 1_200_000,
                "task_by_language_train_supervised_tokens": EXPECTED_TASK_LANGUAGE_MATRIX,
                "task_totals": EXPECTED_TASK_TOTALS,
            },
            "production_eligible": False,
        },
        "report_sha256",
    )
    selection_verification = _sealed(
        {
            "schema_version": (
                "lumi.sft-targeted-gap-closure-source-selection-verification.v0.1"
            ),
            "status": "PASS_verified_targeted_gap_snapshot_source_selection_blocked",
            "checks": {"bound": True},
            "production_eligible": False,
        },
        "verification_sha256",
    )
    public_discovery = _sealed(
        {
            "status": "PASS_massive_train_only_ready_jontro_hold_nonproduction",
            "decision_summary": {
                "new_public_families_ready_for_payload_acquisition": 1,
                "massive_may_seed_tool_api_only_after_all_later_gates": True,
                "public_source_closes_full_language_gap": False,
                "public_source_closes_factual_qa_gap": False,
            },
            "production_eligible": False,
        },
        "report_sha256",
    )
    public_discovery_verification = _sealed(
        {
            "status": "PASS_verified_public_gap_discovery_nonproduction",
            "checks": {"bound": True},
            "production_eligible": False,
        },
        "verification_sha256",
    )

    json_bindings: dict[str, dict[str, object]] = {
        "source_selection_report": selection,
        "source_selection_verification": selection_verification,
        "public_source_discovery_report": public_discovery,
        "public_source_discovery_verification": public_discovery_verification,
        "massive_acquisition_preflight": {
            "status": "PREPARED_not_executable",
            "authorization": {"execution_authorized": False},
            "bound_source": {
                "object_sha256": (
                    "947f65c28cce0d9a57c16bc1a99df08f84e9b7599dff25f47fd5b9760256c17d"
                )
            },
        },
        "human_authoring_plan_v02": {
            "status": "bounded_message_native_blank_template_ready_no_authored_records",
            "totals": {"train_records": 240, "validation_records": 80},
        },
        "human_authoring_workflow_v02": {
            "status": (
                "message_native_blank_human_packets_ready_identity_assignment_pending"
            )
        },
        "human_authoring_blank_audit": {
            "authoring_complete": False,
            "status_counts": {"accepted": 0, "blank": 320, "invalid": 0},
        },
        "scale_quality_failure_report": {
            "status": "stopped_fail_closed_quality_screen_failed",
            "acceptance": {"amended_screen_rate": 0.45, "cumulative_rate": 0.6},
        },
        "scale_quality_failure_verification": {"status": "PASS"},
        "deterministic_completion_report": {
            "production_eligible": False,
            "deterministic_completion": {
                "additional_records_retained": 15,
                "local_inference_calls": 0,
            },
            "generation": {"calls": 0},
        },
        "deterministic_completion_verification": {
            "status": "verified_private_candidate_pending_owner_review",
            "deterministic_template_records_verified": 15,
            "local_inference_calls_verified": 0,
        },
        "source_registry_v21": {
            "sources": [
                {
                    "source_id": "lumi_owned_sft_v0_1",
                    "status": "planned",
                    "production_eligible": False,
                    "blockers": [
                        "first_party_records_and_author_declarations_not_created"
                    ],
                }
            ]
        },
    }

    bindings: dict[str, dict[str, str]] = {}
    for name, value in json_bindings.items():
        path = root / "bound" / f"{name}.json"
        _write_json(path, value)
        bindings[name] = {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
        }
    for name in (
        "assistant_behavior_spec",
        "benchmark_signatures",
        "chat_template",
        "data_quality",
        "implementation",
        "runbook",
        "runner",
        "test",
        "verifier",
    ):
        path = root / "bound" / f"{name}.txt"
        path.write_text(f"synthetic binding: {name}\n", encoding="utf-8")
        bindings[name] = {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
        }

    plan: dict[str, object] = {
        "schema_version": (
            "lumi.sft-first-party-zero-inference-closure-preflight-plan.v0.1"
        ),
        "status": "authorized_plan_only_first_party_zero_inference_closure_preflight",
        "observed_at": "2026-08-29",
        "authorization": {
            "local_plan_only": True,
            "raw_text_free_aggregate_evidence_authorized": True,
            "candidate_construction_authorized": False,
            "candidate_reads_authorized": False,
            "dataset_seal_authorized": False,
            "gpu_authorized": False,
            "inference_authorized": False,
            "network_authorized": False,
            "outreach_authorized": False,
            "production_promotion_authorized": False,
            "qualified_legal_conclusion_authorized": False,
            "registry_mutation_authorized": False,
            "remote_access_authorized": False,
            "sft_training_authorized": False,
            "source_reads_authorized": False,
            "test_object_reads_authorized": False,
            "unrelated_access_authorized": False,
        },
        "bindings": bindings,
        "limits": {
            "maximum_candidate_reads": 0,
            "maximum_gpu_actions": 0,
            "maximum_inference_calls": 0,
            "maximum_network_bytes": 0,
            "maximum_outreach_actions": 0,
            "maximum_remote_actions": 0,
            "maximum_source_line_reads": 0,
            "maximum_test_object_reads": 0,
            "maximum_report_bytes": 1_048_576,
            "maximum_verification_bytes": 1_048_576,
        },
        "target": {
            "retained_train_supervised_tokens": 1_200_000,
            "language_totals": EXPECTED_LANGUAGE_TOTALS,
            "task_totals": EXPECTED_TASK_TOTALS,
            "task_by_language_train_supervised_tokens": EXPECTED_TASK_LANGUAGE_MATRIX,
        },
        "lanes": {
            "human_authored": {
                "retained_train_supervised_tokens": 900_000,
                "task_by_language_train_supervised_tokens": {
                    "conversation": {
                        "banglish": 100_000,
                        "natural_code_switching": 50_000,
                    },
                    "factual_qa": {
                        "banglish": 100_000,
                        "natural_code_switching": 50_000,
                    },
                    "multilingual_bengali": {
                        "banglish": 100_000,
                        "natural_code_switching": 50_000,
                    },
                    "safety_helpfulness": {
                        "banglish": 100_000,
                        "natural_code_switching": 50_000,
                    },
                    "writing_summarization": {
                        "banglish": 200_000,
                        "natural_code_switching": 100_000,
                    },
                },
            },
            "deterministic_grounded": {
                "retained_train_supervised_tokens": 300_000,
                "task_by_language_train_supervised_tokens": {
                    "safety_helpfulness": {
                        "banglish": 100_000,
                        "natural_code_switching": 50_000,
                    },
                    "tool_api": {
                        "banglish": 100_000,
                        "natural_code_switching": 50_000,
                    },
                },
            },
        },
        "canary": {
            "candidate_records": 1000,
            "train_records": 800,
            "validation_records": 200,
            "target_retained_train_supervised_tokens": 80_000,
            "minimum_retained_train_supervised_tokens": 60_000,
            "maximum_retained_train_supervised_tokens": 100_000,
            "minimum_overall_acceptance_rate": 0.95,
            "minimum_per_task_language_cell_acceptance_rate": 0.90,
            "critical_failures_maximum": 0,
            "final_row_independent_human_review_rate": 1.0,
            "candidate_construction_authorized": False,
        },
        "scale_bounds": {
            "maximum_candidate_train_rows": 20_000,
            "maximum_candidate_validation_rows": 4_000,
        },
        "source_family_policy": {
            "conservative_internal_family_id": "lumi_first_party_family",
            "count_internal_lanes_as_distinct_source_families": False,
            "minimum_new_source_families_in_bound_replacement_plan": 2,
            "fallback_owner_decision_required_if_massive_fails": True,
            "owner_decision_required_before_massive_gate": False,
            "source_family_share_maximum": 0.35,
        },
        "review_contract": {
            "author_and_reviewer_must_differ_per_row": True,
            "completed_contributor_rights_declarations_required": True,
            "deterministic_rows_must_not_claim_human_per_row_authorship": True,
            "final_row_independent_human_review_required": True,
            "final_row_review_rate": 1.0,
            "human_lane_model_assistance_prohibited": True,
            "owner_dataset_signoff_required": True,
            "qualified_contributor_rights_review_required_before_candidate_credit": True,
        },
        "quality_gates": {
            "critical_failures_maximum": 0,
            "cross_split_exact_or_near_duplicate_pairs_maximum": 0,
        },
        "next_gate": {
            "action": (
                "separately_authorized_massive_train_only_acquisition_then_first_party_"
                "1000_row_canary"
            ),
            "authorization_required": True,
            "candidate_construction_authorized": False,
        },
        "output": {
            "report_path": (
                "evidence/lumi-4b-sft-first-party-zero-inference-closure-preflight-v0.1.json"
            ),
            "verification_path": (
                "evidence/lumi-4b-sft-first-party-zero-inference-closure-preflight-"
                "verification-v0.1.json"
            ),
        },
    }
    plan["plan_internal_sha256"] = sha256_value(plan)
    plan_path = root / "plan.json"
    _write_json(plan_path, plan)
    return plan_path, plan


def _rewrite_plan(plan_path: Path, plan: dict[str, object]) -> None:
    plan.pop("plan_internal_sha256", None)
    plan["plan_internal_sha256"] = sha256_value(plan)
    _write_json(plan_path, plan)


class FirstPartyZeroInferenceClosurePreflightTest(unittest.TestCase):
    def test_build_report_is_plan_only_and_reproduces_exact_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path, _ = _fixture_plan(root)
            plan = load_plan(plan_path, repository_root=root)
            report = build_report(
                plan,
                plan_file_sha256=sha256_file(plan_path),
                repository_root=root,
            )
            validate_report(report)
            verification = build_verification(
                plan=plan,
                report=report,
                report_file_sha256="1" * 64,
            )
            validate_verification(verification)

            self.assertEqual(
                report["closure_design"]["target"]["language_totals"],
                {"banglish": 800_000, "natural_code_switching": 400_000},
            )
            self.assertEqual(report["resource_accounting"]["inference_calls"], 0)
            self.assertIs(report["production_eligible"], False)
            self.assertIs(verification["production_eligible"], False)

    def test_plan_rejects_inference_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path, plan = _fixture_plan(root)
            plan["authorization"]["inference_authorized"] = True
            _rewrite_plan(plan_path, plan)

            with self.assertRaisesRegex(ValueError, "forbidden action"):
                load_plan(plan_path, repository_root=root)

    def test_plan_rejects_changed_lane_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path, plan = _fixture_plan(root)
            plan["lanes"]["human_authored"][
                "task_by_language_train_supervised_tokens"
            ]["conversation"]["banglish"] = 99_999
            _rewrite_plan(plan_path, plan)

            with self.assertRaisesRegex(ValueError, "lane human_authored total changed"):
                load_plan(plan_path, repository_root=root)

    def test_report_rejects_stale_bound_internal_seal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path, plan = _fixture_plan(root)
            binding = plan["bindings"]["source_selection_report"]
            report_path = root / binding["path"]
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["current_verified_state"]["strict_targeted_gap_lower_bound"] = 1
            _write_json(report_path, report)
            binding["sha256"] = sha256_file(report_path)
            _rewrite_plan(plan_path, plan)
            loaded = load_plan(plan_path, repository_root=root)

            with self.assertRaisesRegex(ValueError, "internal seal changed"):
                build_report(
                    loaded,
                    plan_file_sha256=sha256_file(plan_path),
                    repository_root=root,
                )


if __name__ == "__main__":
    unittest.main()
