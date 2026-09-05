from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "lumi-4b-sft-data-scale-gate-v0.1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_sft_data_scale_gate_is_fail_closed_and_evidence_bound() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert plan["schema_version"] == "lumi.sft-data-scale-gate.v0.1"
    assert all(
        value is False for value in plan["authorization"].values() if isinstance(value, bool)
    )
    assert plan["scale_decision"]["current_production_token_target"] is None
    assert plan["scale_decision"]["planning_bands_are_authorization"] is False
    assert plan["scale_decision"]["trillion_token_sft"]["enabled"] is False
    aya = plan["bindings"]["aya_bengali_repair_preflight"]
    assert aya["candidate_records"] == 1_478
    assert aya["supervised_tokens"] == 577_453
    assert aya["terminal_pipe_artifacts_repaired"] == 223
    assert aya["maximum_annotator_fraction"] == 0.5
    assert aya["candidate_materialized"] is False
    assert aya["owner_review_complete"] is False
    assert aya["production_promotions"] == 0
    assert aya["internal_report_sha256"] == (
        "6d7c20b19f6b8154c58f51c59521d1e0045dbcbe3e06baa10ac1766c0f6800b2"
    )
    aya_materialization = plan["bindings"]["aya_bengali_materialization"]
    assert aya_materialization["candidate_materialized"] is True
    assert aya_materialization["candidate_records"] == 1_478
    assert aya_materialization["supervised_tokens"] == 577_453
    assert aya_materialization["private_artifacts"] == 3
    assert aya_materialization["owner_sample_records"] == 32
    assert aya_materialization["owner_review_complete"] is True
    assert aya_materialization["production_promotions"] == 0
    assert aya_materialization["internal_report_sha256"] == (
        "a2f08b1594a2b6fd9def1f43a48e072cbe0f107f4bc7ea43b391d6066ddc6ade"
    )
    aya_owner_review = plan["bindings"]["aya_bengali_owner_review_receipt"]
    assert aya_owner_review["accepted_records"] == 32
    assert aya_owner_review["rejected_records"] == 0
    assert aya_owner_review["all_sampled_rows_accepted"] is True
    assert aya_owner_review["production_promotions"] == 0
    assert aya_owner_review["internal_receipt_sha256"] == (
        "e23851af5718d10acdd120ca1e0c200672fb5c794ad503ed4d1f7f1650cc685f"
    )
    aya_promotion = plan["bindings"]["aya_bengali_source_promotion"]
    assert aya_promotion["candidate_source_registry_promotions"] == 1
    assert aya_promotion["production_dataset_promotions"] == 0
    assert aya_promotion["successor_registry_sha256"] == (
        "4c81d9968ff8ecdef760ab7e78f01d4d4ddb8e75aae7d29e17dc8a349c69161f"
    )
    assert aya_promotion["status"] == (
        "reviewed_aya_registered_in_immutable_candidate_successor_production_dataset_still_blocked"
    )
    aya_validation = plan["bindings"]["aya_bengali_validation_preflight"]
    assert aya_validation["train_records"] == 1_330
    assert aya_validation["train_supervised_tokens"] == 520_269
    assert aya_validation["validation_records"] == 148
    assert aya_validation["validation_supervised_tokens"] == 57_184
    assert aya_validation["candidate_split_materialized"] is False
    assert aya_validation["production_dataset_promotions"] == 0
    assert aya_validation["internal_report_sha256"] == (
        "48b21e9f5b1ce07290beab67c3f77a33572a0ab887b871e19fcebc5b99fc8628"
    )
    aya_split = plan["bindings"]["aya_bengali_validation_materialization"]
    assert aya_split["train_records"] == 1_330
    assert aya_split["train_supervised_tokens"] == 520_269
    assert aya_split["validation_records"] == 148
    assert aya_split["validation_supervised_tokens"] == 57_184
    assert aya_split["candidate_split_materialized"] is True
    assert aya_split["total_private_bytes"] == 5_480_313
    assert aya_split["production_dataset_promotions"] == 0
    assert aya_split["internal_receipt_sha256"] == (
        "a4507bd0dbe7b1d335f4602205f4d62b1368efe9c0f8ce8f6f31606e667f8932"
    )
    assert plan["bindings"]["openr1_verified_trace_audit"]["selected_verified_trace_records"] == 18
    assert plan["bindings"]["openr1_verified_trace_audit"]["encoded_4k_records"] == 6
    assert plan["bindings"]["openr1_verified_trace_audit"]["production_promotions"] == 0
    assert (
        plan["bindings"]["openr1_pii_diagnostic"]["likely_math_numeric_false_positive_records"] == 4
    )
    assert plan["bindings"]["openr1_pii_diagnostic"]["unresolved_phone_like_records"] == 2
    assert plan["bindings"]["openr1_pii_diagnostic"]["global_pii_filter_modified"] is False
    openr1_secondary = plan["bindings"]["openr1_pii_secondary"]
    assert openr1_secondary["global_phone_detector_records"] == 6
    assert openr1_secondary["line_scoped_math_numeric_false_positive_records"] == 5
    assert openr1_secondary["cross_message_join_artifact_records"] == 1
    assert openr1_secondary["unresolved_records"] == 0
    assert openr1_secondary["global_pii_filter_modified"] is False
    assert openr1_secondary["production_promotions"] == 0
    assert openr1_secondary["internal_report_sha256"] == (
        "7761ab5800608658c0ea99d6b2a1d03e19c2c46c85b1641e647789ef1069fb6a"
    )
    openr1_full = plan["bindings"]["openr1_full_curation_preflight"]
    assert openr1_full["source_rows"] == 93_733
    assert openr1_full["data_objects"] == 10
    assert openr1_full["objects"] == 11
    assert openr1_full["download_size_bytes"] == 2_149_897_914
    assert openr1_full["dataset_size_bytes"] == 4_964_543_659
    assert openr1_full["maximum_network_bytes"] == 2_415_919_104
    assert openr1_full["maximum_retained_supervised_tokens"] == 6_600_000
    assert openr1_full["acquisition_authorized"] is False
    assert openr1_full["production_promotions"] == 0
    assert openr1_full["internal_report_sha256"] == (
        "520cadd35bd2ece1d971986f790b9cfaf0c9ceafc9c330f404e376fab13fbc78"
    )
    oasst1 = plan["bindings"]["oasst1_full_curation_preflight"]
    assert oasst1["accepted_preflight_pairs"] == 4
    assert oasst1["maximum_source_rows"] == 88_838
    assert oasst1["maximum_network_bytes"] == 64 * 1024 * 1024
    assert oasst1["acquisition_authorized"] is False
    assert oasst1["production_promotions"] == 0
    assert oasst1["internal_report_sha256"] == (
        "686d39d0024226bb3cdc60de870c74326b5d6556c4e6da7df10703fa2b9296d9"
    )
    curated = plan["bindings"]["oasst1_full_curation"]
    assert curated["source_rows"] == 88_838
    assert curated["accepted_records"] == 1_745
    assert curated["supervised_tokens"] == 477_456
    assert curated["owner_sample_records"] == 32
    assert curated["owner_sample_review_complete"] is True
    assert curated["production_promotions"] == 0
    verified = plan["bindings"]["oasst1_full_curation_verification"]
    assert verified["production_eligible"] is False
    assert verified["internal_verification_sha256"] == (
        "bdd099dae98812117db028f98a4d4aa11b23ab1e08b59ae67667b1a0484b2cc3"
    )
    owner_review = plan["bindings"]["oasst1_owner_review_receipt"]
    assert owner_review["accepted_records"] == 32
    assert owner_review["rejected_records"] == 0
    assert owner_review["internal_receipt_sha256"] == (
        "6f3f0232983695403f110d898ed59ab5d1bff503f587aa5360db6ec9e46439a4"
    )
    assert owner_review["production_promotions"] == 0
    oasst1_promotion = plan["bindings"]["oasst1_source_promotion"]
    assert oasst1_promotion["candidate_source_registry_promotions"] == 1
    assert oasst1_promotion["production_dataset_promotions"] == 0
    assert oasst1_promotion["successor_registry_sha256"] == (
        "e85f591f2a93b1384ed9ce4ee3da0e067bcfc07a1ade2e0b170eb875c7677914"
    )
    assert oasst1_promotion["status"] == (
        "reviewed_oasst1_registered_in_immutable_candidate_successor_"
        "production_dataset_still_blocked"
    )
    punctuation = plan["bindings"]["punctuation_quality_gate"]
    assert punctuation["expected_lanes"] == 72
    assert punctuation["minimum_post_sft_model_cases"] == 288
    assert punctuation["model_inference_executed"] is False
    assert punctuation["status"] == "gate_defined_not_executed"
    assert (
        plan["bindings"]["missing_mix_proposal"]["accepted_train_supervised_token_minimum"]
        == 10_000_000
    )
    assert (
        plan["bindings"]["missing_mix_proposal"]["accepted_train_supervised_token_maximum"]
        == 30_000_000
    )
    assert plan["bindings"]["missing_mix_proposal"]["production_promotions"] == 0
    assert plan["bindings"]["safety_tool_candidate"]["accepted_records"] == 320
    assert (
        plan["bindings"]["safety_tool_candidate"]["candidate_evidence_owner_review_complete"]
        is False
    )
    assert (
        plan["bindings"]["safety_tool_candidate"]["separate_owner_review_receipt_complete"] is True
    )
    assert plan["bindings"]["safety_tool_candidate"]["production_promotions"] == 0
    assert plan["bindings"]["safety_tool_owner_review_plan"]["sample_records"] == 40
    assert plan["bindings"]["safety_tool_owner_review_plan"]["status"] == (
        "human_owner_review_ready_no_promotion"
    )
    assert plan["bindings"]["safety_tool_owner_review_plan"]["production_promotions"] == 0
    assert plan["bindings"]["safety_tool_owner_review_receipt"]["accepted_records"] == 40
    assert plan["bindings"]["safety_tool_owner_review_receipt"]["rejected_records"] == 0
    assert plan["bindings"]["safety_tool_owner_review_receipt"]["status"] == (
        "owner_spot_check_passed_source_registry_promotion_still_separate"
    )
    assert plan["bindings"]["safety_tool_owner_review_receipt"]["production_promotions"] == 0
    promotion = plan["bindings"]["safety_tool_source_promotion"]
    assert promotion["candidate_source_registry_promotions"] == 1
    assert promotion["production_dataset_promotions"] == 0
    assert promotion["successor_registry_sha256"] == (
        "f686668fbb05f0d7930b5047560543ce113fcfb97c423fbe840f65d8a1058cdd"
    )
    assert promotion["status"] == (
        "candidate_source_registered_in_immutable_successor_production_dataset_still_blocked"
    )
    assert plan["bindings"]["safety_tool_token_audit"]["train_supervised_tokens"] == 18_636
    assert plan["bindings"]["safety_tool_token_audit"]["validation_supervised_tokens"] == 6_388
    assert plan["bindings"]["safety_tool_token_audit"]["maximum_serialized_tokens"] == 256
    assert plan["bindings"]["safety_tool_token_audit"]["production_promotions"] == 0

    for binding in plan["bindings"].values():
        assert _sha256(ROOT / binding["path"]) == binding["file_sha256"]


def test_external_teacher_outputs_are_not_preapproved() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    policies = plan["provider_output_policy"]
    assert policies["openai_api"]["status"].startswith("blocked")
    assert policies["azure_openai"]["status"].startswith("blocked")
    assert policies["open_source_teacher_models"]["status"] == ("source_specific_review_required")
