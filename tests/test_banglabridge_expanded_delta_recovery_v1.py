from __future__ import annotations

import hashlib
import json

import pytest

from lumi.posttraining_banglabridge_expanded_delta_recovery_v1 import (
    EXPECTED_AUDIT_RECORDS,
    EXPECTED_AUTHORIZATION,
    EXPECTED_AUTHORIZATION_TEXT_SHA256,
    EXPECTED_DELTA_ROWS,
    EXPECTED_DELTA_TRAIN_TOKENS,
    EXPECTED_LANGUAGE_GAPS,
    EXPECTED_LIMITS,
    EXPECTED_NEAR_DUPLICATE_POLICY,
    EXPECTED_SUCCESSOR_SHA256,
    EXPECTED_SUCCESSOR_TRAIN_TOKENS,
    EXPECTED_TASK_GAPS,
    PLAN_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    STATUS,
    _owner_artifacts,
    build_verification,
    json_bytes,
    jsonl_bytes,
    sha256_value,
    validate_owner_preflight_artifacts,
    validate_report,
    validate_verification,
)
from lumi.posttraining_candidate_mix_reaudit_v14 import EXPECTED_EXCLUDED_SOURCE_IDS
from lumi.posttraining_first_party import canonical_json_bytes
from lumi.posttraining_owner_review import DECISIONS_SCHEMA_VERSION, validate_owner_decisions
from scripts.verify_banglabridge_expanded_delta_recovery_v1 import (
    _preflight_then_replay,
)


def test_expanded_delta_authorized_arithmetic_is_exact() -> None:
    assert EXPECTED_AUDIT_RECORDS == 314_271 - 116 + 561
    assert EXPECTED_DELTA_ROWS == 561 - 116
    assert EXPECTED_DELTA_TRAIN_TOKENS == 250_441 - 49_756
    assert EXPECTED_SUCCESSOR_TRAIN_TOKENS == 250_441
    assert EXPECTED_LIMITS["maximum_full_audit_passes"] == 2
    assert EXPECTED_LIMITS["maximum_total_candidate_audit_row_reads"] == (
        2 * EXPECTED_AUDIT_RECORDS
    )
    assert EXPECTED_LIMITS["maximum_source_row_reads"] == 2 * 608
    assert EXPECTED_LIMITS["maximum_total_candidate_row_reads"] == (
        2 * EXPECTED_AUDIT_RECORDS + 2 * 561
    )


def test_expanded_delta_hash_policy_and_remaining_gaps_are_frozen() -> None:
    assert EXPECTED_SUCCESSOR_SHA256 == (
        "5cf0b5f4ad4029f7a9e25b49ac7785f43607fb3196baf8c9f5dd272da0551058"
    )
    assert EXPECTED_NEAR_DUPLICATE_POLICY["threshold"] == 0.9
    assert EXPECTED_NEAR_DUPLICATE_POLICY["num_perm"] == 128
    assert EXPECTED_LANGUAGE_GAPS == {
        "banglish": 772_114,
        "bengali_script": 0,
        "english": 0,
        "natural_code_switching": 365_420,
    }
    assert EXPECTED_TASK_GAPS["multilingual_bengali"] == 0
    assert EXPECTED_TASK_GAPS["factual_qa"] == 114_306


def test_expanded_delta_is_local_nonproduction_only() -> None:
    assert EXPECTED_LIMITS["maximum_network_bytes"] == 0
    assert EXPECTED_LIMITS["maximum_private_write_bytes"] == 16 * 1024 * 1024
    assert EXPECTED_AUTHORIZATION["network_authorized"] is False
    assert EXPECTED_AUTHORIZATION["gpu_authorized"] is False
    assert EXPECTED_AUTHORIZATION["inference_authorized"] is False
    assert EXPECTED_AUTHORIZATION["registry_mutation_authorized"] is False
    assert EXPECTED_AUTHORIZATION["sft_training_authorized"] is False


def test_exact_attached_authorization_digest_is_frozen() -> None:
    assert len(EXPECTED_AUTHORIZATION_TEXT_SHA256) == 64
    assert PLAN_SCHEMA_VERSION.endswith("v0.1")


def _minimal_report() -> dict:
    packet_content_sha256 = "1" * 64
    packet_file_sha256 = "2" * 64
    report = {
        "audit": {
            "accepted_records": EXPECTED_AUDIT_RECORDS,
            "accepted_train_supervised_tokens": 10_604_149,
            "accepted_validation_supervised_tokens": 628_604,
            "candidate_audit_row_reads": EXPECTED_AUDIT_RECORDS,
            "strict_targeted_gap_lower_bound": 1_137_534,
            "exact_duplicate_audit": {
                "cross_source_duplicate_groups": 0,
                "duplicate_extra_records": 0,
                "duplicate_groups": 0,
                "train_validation_overlap_groups": 0,
            },
            "family_cap_violations": [],
            "family_cap_dilution_gap": 0,
            "family_shares": {
                "banglabridge_family": round(250_441 / 10_604_149, 8)
            },
            "generic_sft_validation": {
                "accepted": EXPECTED_AUDIT_RECORDS,
                "accepted_dataset_sha256": "5" * 64,
                "raw_text_retained": False,
                "rejected": 0,
                "rejection_counts": {},
                "report_sha256": "6" * 64,
                "schema_version": "synthetic",
                "total": EXPECTED_AUDIT_RECORDS,
            },
            "excluded_source_gate": {
                "candidate_rows_from_excluded_sources": 0,
                "predecessor_evidence_replayed": [],
                "registry_sources_not_in_candidate_inventory": sorted(
                    EXPECTED_EXCLUDED_SOURCE_IDS
                ),
            },
            "language_floor_gaps_behavior_safe": EXPECTED_LANGUAGE_GAPS,
            "near_duplicate_audit": {
                "comparison_unit": "role_tagged_full_conversation",
                "component_ledger_sha256": sha256_value([]),
                "components": EXPECTED_AUDIT_RECORDS,
                "cross_source_components": 0,
                "duplicate_extra_records": 0,
                "edge_count": 0,
                "largest_component_records": 1,
                "multi_record_components": 0,
                "per_source_multi_record_components": {},
                "records": EXPECTED_AUDIT_RECORDS,
                "size_histogram": {},
                "train_validation_components": 0,
                "train_validation_records": 0,
            },
            "source_metrics": {
                **{
                    f"synthetic-source-{index:02d}": {}
                    for index in range(16)
                },
                "banglabridge_instructions_closure_canary": {
                    "candidate_file_sha256": EXPECTED_SUCCESSOR_SHA256
                },
            },
            "task_floor_gaps_behavior_safe": EXPECTED_TASK_GAPS,
            "token_metrics": {
                "train_supervised_tokens": 10_604_149,
                "validation_supervised_tokens": 628_604,
            },
        },
        "delta": {
            "bytes": 1_554_396,
            "language_train_supervised_tokens": {
                "bengali_script": 191_691,
                "natural_code_switching": 8_994,
            },
            "records": 445,
            "sha256": (
                "bf6617ed8e0a7a95710c1c046d99d0954cdbd2136574f15ea9909aebc5fecce2"
            ),
            "train_supervised_tokens": 200_685,
            "task": "multilingual_bengali",
        },
        "evaluation_prohibition": {
            "billsum_ca_test_claimed_evaluation": "permanently_prohibited",
            "billsum_test_claimed_evaluation": "permanently_prohibited",
            "reason": "BillSum train split used for SFT candidate construction",
        },
        "owner_review": {
            "all_recovered_rows_status": "pending_deterministic_spot_check",
            "historical_review_records_preserved": 8,
            "historical_review_reused_for_delta": False,
            "minimum_accepts_required_for_later_promotion": 29,
            "packet_records": 32,
            "packet_bytes": 1000,
            "packet_content_sha256": packet_content_sha256,
            "packet_file_sha256": packet_file_sha256,
            "status": "pending_human_owner_decision",
        },
        "materialization": {
            "delta_language_train_supervised_tokens": {
                "bengali_script": 191_691,
                "natural_code_switching": 8_994,
            },
            "delta_train_supervised_tokens": 200_685,
            "maximum_serialized_tokens": 3_872,
            "train_records": 544,
            "train_supervised_tokens": 250_441,
            "validation_records": 17,
            "validation_supervised_tokens": 4_960,
        },
        "outputs": {
            "candidate": {"bytes": 1_967_432, "sha256": EXPECTED_SUCCESSOR_SHA256},
            "delta_ledger": {"bytes": 1000, "sha256": "3" * 64},
            "owner_review_html": {"bytes": 1000, "sha256": "4" * 64},
            "owner_review_packet": {"bytes": 1000, "sha256": packet_file_sha256},
        },
        "production_eligible": False,
        "raw_text_boundary": {
            "candidate_contains_raw_training_text": True,
            "delta_ledger_contains_raw_training_text": False,
            "owner_review_html_contains_raw_training_text": True,
            "owner_review_packet_contains_raw_training_text": True,
            "report_contains_raw_training_text": False,
            "tests_or_logs_may_contain_raw_training_text": False,
            "verification_contains_raw_training_text": False,
        },
        "resource_accounting": {
            "candidate_audit_passes_completed": 1,
            "candidate_audit_row_reads": EXPECTED_AUDIT_RECORDS,
            "candidate_reconstruction_row_reads": 561,
            "candidate_total_row_reads": EXPECTED_AUDIT_RECORDS + 561,
            "source_row_reads": 608,
        },
        "authorization": EXPECTED_AUTHORIZATION,
        "bindings": {
            name: "a" * 64
            for name in __import__(
                "lumi.posttraining_banglabridge_expanded_delta_recovery_v1",
                fromlist=["EXPECTED_BINDINGS"],
            ).EXPECTED_BINDINGS
        },
        "checks": {
            "all_561_rows_have_four_bound_provenance_hashes": True,
            "all_445_delta_rows_owner_review_pending": True,
            "banglabridge_language_punctuation_and_safety_gates_passed": True,
            "blanket_row_level_human_authorship_claimed": False,
            "candidate_and_delta_prospective_hashes_match": True,
            "cross_source_minhash_0_9_passed": True,
            "no_exact_duplicate_or_split_overlap": True,
            "no_network_gpu_inference_registry_seal_sft_or_production_action": True,
            "predecessors_and_source_preserved": True,
            "schema_rights_provenance_secret_pii_benchmark_and_4k_gates_passed": True,
        },
        "observed_at": "2026-08-13",
        "plan_file_sha256": "b" * 64,
        "runtime_recovery": __import__(
            "lumi.posttraining_banglabridge_expanded_delta_recovery_v1",
            fromlist=["EXPECTED_RUNTIME_RECOVERY"],
        ).EXPECTED_RUNTIME_RECOVERY,
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": STATUS,
        "successor": {
            "bytes": 1_967_432,
            "records": 561,
            "sha256": EXPECTED_SUCCESSOR_SHA256,
            "train_records": 544,
            "train_supervised_tokens": 250_441,
            "validation_records": 17,
            "validation_supervised_tokens": 4_960,
        },
    }
    report["report_sha256"] = sha256_value(report)
    return report


def test_report_validator_fails_closed_on_raw_text_or_hash_change() -> None:
    report = _minimal_report()
    validate_report(report)
    tampered = json.loads(json.dumps(report))
    tampered["unexpected"] = {"messages": [{"content": "forbidden"}]}
    tampered["report_sha256"] = sha256_value(
        {key: value for key, value in tampered.items() if key != "report_sha256"}
    )
    with pytest.raises(ValueError, match="boundary|raw training text"):
        validate_report(tampered)
    assert hashlib.sha256(b"authorization").hexdigest() != (
        EXPECTED_AUTHORIZATION_TEXT_SHA256
    )


def test_unicode_owner_hash_domains_html_and_generic_decisions_round_trip() -> None:
    packet = [
        {
            "language_mode": "bengali_script",
            "messages": [
                {"role": "user", "content": "বাংলা <title>Lumi owner review</title>"},
                {
                    "role": "assistant",
                    "content": (
                        "ঠিক আছে & নিরাপদ। lumi.ai-assisted-owner-decisions.v0.3 "
                        "lumi-owner-review-decisions-v0.3.json"
                    ),
                },
            ],
            "record_id": "synthetic-bn-01",
            "skill_id": "multilingual_bengali",
            "split": "train",
        }
    ]
    packet_bytes, html_bytes, content_sha256, file_sha256 = _owner_artifacts(packet)
    assert content_sha256 == hashlib.sha256(canonical_json_bytes(packet)).hexdigest()
    assert file_sha256 == hashlib.sha256(packet_bytes).hexdigest()
    assert len({content_sha256, file_sha256, sha256_value(packet)}) == 3
    html = html_bytes.decode("utf-8")
    assert DECISIONS_SCHEMA_VERSION in html
    assert "lumi-owner-review-decisions-v0.3.json" in html
    assert "lumi.banglabridge-expanded-delta-owner-decisions" not in html
    assert "\\u003ctitle\\u003eLumi owner review\\u003c/title\\u003e" in html
    decisions = {
        "schema_version": DECISIONS_SCHEMA_VERSION,
        "sample_sha256": content_sha256,
        "reviewer_id": "owner-afnan",
        "reviewed_at": "2026-08-13T12:00:00Z",
        "decisions": [
            {"accepted": True, "note": "", "record_id": "synthetic-bn-01"}
        ],
    }
    receipt = validate_owner_decisions(
        decisions, sample_rows=packet, sample_sha256=content_sha256
    )
    assert receipt["accepted_records"] == 1


def test_report_and_verification_keep_content_and_file_hashes_distinct() -> None:
    report = _minimal_report()
    validate_report(report)
    swapped = json.loads(json.dumps(report))
    swapped["owner_review"]["packet_content_sha256"] = swapped["owner_review"][
        "packet_file_sha256"
    ]
    swapped["report_sha256"] = sha256_value(
        {key: value for key, value in swapped.items() if key != "report_sha256"}
    )
    with pytest.raises(ValueError, match="boundary"):
        validate_report(swapped)
    plan = {"observed_at": "2026-08-13"}
    artifact_hashes = {
        name: value["sha256"] for name, value in report["outputs"].items()
    }
    verification = build_verification(
        plan=plan,
        report=report,
        report_file_sha256="c" * 64,
        artifact_file_sha256=artifact_hashes,
    )
    validate_verification(verification)
    assert verification["owner_packet_content_sha256"] != verification[
        "owner_packet_file_sha256"
    ]


def _synthetic_owner_packet_and_ledger() -> tuple[list[dict], list[dict]]:
    packet: list[dict] = []
    ledger: list[dict] = []
    modes = ["bengali_script"] * 16 + ["natural_code_switching"] * 16
    for index, mode in enumerate(modes):
        record_id = f"synthetic-delta-{index:03d}"
        packet.append(
            {
                "language_mode": mode,
                "messages": [
                    {"role": "user", "content": f"বাংলা synthetic prompt {index}"},
                    {"role": "assistant", "content": f"উত্তর {index}।"},
                ],
                "record_id": record_id,
                "skill_id": "multilingual_bengali",
                "split": "train",
            }
        )
    for index in range(445):
        record_id = f"synthetic-delta-{index:03d}"
        ledger.append(
            {
                "language_mode": (
                    "bengali_script" if index < 332 else "natural_code_switching"
                ),
                "messages_sha256": hashlib.sha256(record_id.encode()).hexdigest(),
                "owner_review_status": "pending_deterministic_spot_check",
                "record_id": record_id,
                "serialized_tokens": 64 + index,
                "split": "train",
                "supervised_tokens": 32 + index,
            }
        )
    return packet, ledger


def test_owner_artifact_preflight_is_exact_and_stops_replay_on_failure() -> None:
    packet, ledger = _synthetic_owner_packet_and_ledger()
    packet_bytes, html_bytes, content_sha256, file_sha256 = _owner_artifacts(packet)
    ledger_bytes = jsonl_bytes(ledger)
    report = _minimal_report()
    report["owner_review"].update(
        {
            "packet_bytes": len(packet_bytes),
            "packet_content_sha256": content_sha256,
            "packet_file_sha256": file_sha256,
        }
    )
    report["outputs"].update(
        {
            "delta_ledger": {
                "bytes": len(ledger_bytes),
                "sha256": hashlib.sha256(ledger_bytes).hexdigest(),
            },
            "owner_review_html": {
                "bytes": len(html_bytes),
                "sha256": hashlib.sha256(html_bytes).hexdigest(),
            },
            "owner_review_packet": {
                "bytes": len(packet_bytes),
                "sha256": file_sha256,
            },
        }
    )
    report["report_sha256"] = sha256_value(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    validate_owner_preflight_artifacts(
        report=report,
        owner_packet_bytes=packet_bytes,
        owner_html_bytes=html_bytes,
        delta_ledger_bytes=ledger_bytes,
    )

    replay_started = False

    def malformed_preflight() -> None:
        validate_owner_preflight_artifacts(
            report=report,
            owner_packet_bytes=json_bytes(packet[:-1]),
            owner_html_bytes=html_bytes,
            delta_ledger_bytes=ledger_bytes,
        )

    def replay_sentinel() -> None:
        nonlocal replay_started
        replay_started = True

    with pytest.raises(ValueError, match="owner packet row inventory"):
        _preflight_then_replay(malformed_preflight, replay_sentinel)
    assert replay_started is False


def test_strict_report_and_verification_resource_evidence() -> None:
    report = _minimal_report()
    validate_report(report)
    for field in (
        "candidate_reconstruction_row_reads",
        "candidate_total_row_reads",
    ):
        tampered = json.loads(json.dumps(report))
        del tampered["resource_accounting"][field]
        tampered["report_sha256"] = sha256_value(
            {key: value for key, value in tampered.items() if key != "report_sha256"}
        )
        with pytest.raises(ValueError, match="boundary"):
            validate_report(tampered)

    artifact_hashes = {
        name: value["sha256"] for name, value in report["outputs"].items()
    }
    verification = build_verification(
        plan={"observed_at": "2026-08-13"},
        report=report,
        report_file_sha256="c" * 64,
        artifact_file_sha256=artifact_hashes,
    )
    assert verification["resource_accounting"]["candidate_total_row_reads"] == 630_554
    validate_verification(verification)
