from __future__ import annotations

import copy

import pytest

from lumi.posttraining_candidate_mix_preflight import sha256_value
from lumi.posttraining_candidate_mix_reaudit_v15 import (
    BANGLABRIDGE_SOURCE,
    EXPECTED_AUTHORIZATION_TEXT_SHA256,
    EXPECTED_EVALUATION_PROHIBITION,
    EXPECTED_INCLUDED_SOURCE_COUNT,
    EXPECTED_INPUT_RECORDS,
    EXPECTED_INPUT_TRAIN_TOKENS,
    EXPECTED_INPUT_VALIDATION_TOKENS,
    EXPECTED_LIMITS,
    EXPECTED_NEAR_DUPLICATE_POLICY,
    REPORT_SCHEMA_VERSION,
    json_bytes,
    validate_report,
)


def test_v015_replacement_projection_is_exact() -> None:
    assert EXPECTED_INCLUDED_SOURCE_COUNT == 17
    assert EXPECTED_INPUT_RECORDS == 314_271 - 116 + 561
    assert EXPECTED_INPUT_TRAIN_TOKENS == 10_403_464 - 49_756 + 250_441
    assert EXPECTED_INPUT_VALIDATION_TOKENS == 628_604 - 4_960 + 4_960
    assert EXPECTED_LIMITS["maximum_total_candidate_row_reads"] == (
        2 * EXPECTED_INPUT_RECORDS
    )
    assert EXPECTED_LIMITS["maximum_inference_calls"] == 0


def test_v015_near_duplicate_and_banglabridge_binding_are_frozen() -> None:
    assert EXPECTED_NEAR_DUPLICATE_POLICY["threshold"] == 0.9
    assert EXPECTED_NEAR_DUPLICATE_POLICY["num_perm"] == 128
    assert BANGLABRIDGE_SOURCE["source_id"] == "banglabridge_instructions_closure_canary"
    assert BANGLABRIDGE_SOURCE["family"] == "banglabridge_family"
    assert len(EXPECTED_AUTHORIZATION_TEXT_SHA256) == 64


def test_v015_billsum_evaluation_prohibition_is_permanent() -> None:
    assert set(EXPECTED_EVALUATION_PROHIBITION) == {
        "billsum_ca_test_claimed_evaluation",
        "billsum_test_claimed_evaluation",
        "reason",
    }
    assert all(
        value == "permanently_prohibited"
        for key, value in EXPECTED_EVALUATION_PROHIBITION.items()
        if key != "reason"
    )


def _synthetic_report() -> dict[str, object]:
    report: dict[str, object] = {
        "authorization": {
            "active_data_contract_change_authorized": False,
            "candidate_mutation_authorized": False,
            "dataset_seal_authorized": False,
            "download_authorized": False,
            "external_api_authorized": False,
            "full_local_reaudit_authorized": True,
            "gpu_authorized": False,
            "instance_action_authorized": False,
            "network_authorized": False,
            "production_promotion_authorized": False,
            "registry_mutation_authorized": False,
            "remote_access_authorized": False,
            "s3_action_authorized": False,
            "sft_training_authorized": False,
            "source_promotion_authorized": False,
            "unrelated_access_authorized": False,
        },
        "candidate_pool": {
            "accepted_records_after_generic_quarantine": 314_716,
            "accepted_train_supervised_tokens": 10_604_149,
            "accepted_validation_supervised_tokens": 628_604,
            "candidate_records_read": 314_716,
            "input_train_supervised_tokens": 10_604_149,
            "input_validation_supervised_tokens": 628_604,
            "strict_targeted_gap_lower_bound": 1_137_534,
        },
        "closure": {
            "family_cap_dilution_gap": 0,
            "language_floor_gaps_behavior_safe": {
                "banglish": 772_114,
                "bengali_script": 0,
                "english": 0,
                "natural_code_switching": 365_420,
            },
            "source_family_cap_violations": [],
            "source_family_shares": {"banglabridge_family": 0.02361727},
            "task_floor_gaps_behavior_safe": {
                "coding": 0,
                "conversation": 5_934,
                "factual_qa": 114_306,
                "multilingual_bengali": 0,
                "reasoning_math_science": 0,
                "safety_helpfulness": 0,
                "tool_api": 2,
                "writing_summarization": 0,
            },
        },
        "evaluation_prohibition": EXPECTED_EVALUATION_PROHIBITION,
        "excluded_source_gate": {
            "candidate_rows_from_excluded_sources": 0,
            "predecessor_evidence_replayed": [
                {
                    "counted_train_supervised_tokens": 0,
                    "source_id": "openr1_math_220k",
                }
            ],
            "registry_sources_not_in_candidate_inventory": [
                "bigcode_self_oss_instruct_50k",
                "huggingfaceh4_ultrachat_200k",
                "lumi_ai_assisted_sft_v0_1",
                "lumi_ai_assisted_sft_v0_2",
                "lumi_owned_sft_v0_1",
                "openr1_math_220k",
                "salesforce_xlam_function_calling_60k",
            ],
        },
        "exact_duplicate_audit": {
            "cross_source_duplicate_groups": 0,
            "duplicate_extra_records": 0,
            "duplicate_groups": 0,
            "train_validation_overlap_groups": 0,
        },
        "generic_sft_validation": {
            "accepted": 314_716,
            "rejected": 0,
            "rejection_counts": {},
            "total": 314_716,
        },
        "near_duplicate_audit": {
            "cross_source_components": 0,
            "duplicate_extra_records": 0,
            "edge_count": 0,
            "multi_record_components": 0,
            "records": 314_716,
            "train_validation_components": 0,
            "train_validation_records": 0,
        },
        "input_projection": {
            "final_records": 314_716,
            "final_train_supervised_tokens": 10_604_149,
            "final_validation_supervised_tokens": 628_604,
            "net_added_records": 445,
            "net_added_train_supervised_tokens": 200_685,
            "net_added_validation_supervised_tokens": 0,
            "predecessor_banglabridge_records": 116,
            "predecessor_banglabridge_train_supervised_tokens": 49_756,
            "predecessor_banglabridge_validation_supervised_tokens": 4_960,
            "predecessor_records": 314_271,
            "predecessor_train_supervised_tokens": 10_403_464,
            "predecessor_validation_supervised_tokens": 628_604,
            "successor_banglabridge_records": 561,
            "successor_banglabridge_train_supervised_tokens": 250_441,
            "successor_banglabridge_validation_supervised_tokens": 4_960,
        },
        "production_eligible": False,
        "quarantine": {"records": [], "total_records": 0},
        "raw_training_text_retained": False,
        "registry": {
            "file_sha256": "daa2d81755456728c5c72bca9ee95d5424aba0fc05e9fdd13223018f4055e3a1",
            "source_count": 24,
        },
        "resource_accounting": {"candidate_row_reads": 314_716, "full_audit_passes": 1},
        "schema_version": REPORT_SCHEMA_VERSION,
        "source_metrics": {
            **{f"source-{index}": {} for index in range(16)},
            "banglabridge_instructions_closure_canary": {
                "by_split": {
                    "train": {"records": 544, "supervised_tokens": 250_441},
                    "validation": {"records": 17, "supervised_tokens": 4_960},
                },
                "candidate_file_sha256": (
                    "5cf0b5f4ad4029f7a9e25b49ac7785f43607fb3196baf8c9f5dd272da0551058"
                ),
                "records": 561,
                "rows_with_pending_owner_review_provenance": 561,
                "train_by_language_mode": {
                    "bengali_script": 239_144,
                    "natural_code_switching": 11_297,
                },
                "train_by_task": {"multilingual_bengali": 250_441},
            },
        },
        "status": "PASS_registry_v0_21_full_candidate_mix_reaudit_nonproduction",
    }
    report["report_sha256"] = sha256_value(report)
    return report


def test_v015_report_validator_closes_gap_share_and_hygiene_boundaries() -> None:
    report = _synthetic_report()
    validate_report(report)
    assert len(json_bytes(report)) < EXPECTED_LIMITS["maximum_private_write_bytes"]
    for path, value in (
        (("candidate_pool", "strict_targeted_gap_lower_bound"), 1),
        (("closure", "source_family_cap_violations"), ["banglabridge_family"]),
        (("generic_sft_validation", "rejected"), 1),
        (("near_duplicate_audit", "multi_record_components"), 1),
    ):
        tampered = copy.deepcopy(report)
        node = tampered
        for key in path[:-1]:
            node = node[key]  # type: ignore[index]
        node[path[-1]] = value  # type: ignore[index]
        tampered["report_sha256"] = sha256_value(
            {key: item for key, item in tampered.items() if key != "report_sha256"}
        )
        with pytest.raises(ValueError):
            validate_report(tampered)
