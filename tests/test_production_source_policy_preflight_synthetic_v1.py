"""Source-only validator tests; synthetic fixtures are never operational evidence.

These fixtures exercise exported contract constants without reading a historical
plan, private report, candidate, registry, owner packet, or dataset. They are kept
in memory only. Their schema-shaped metadata is not a factual gate result or an
approval, and does not replace the historical end-to-end tests.
"""

from __future__ import annotations

import copy
import unittest
from pathlib import Path
from unittest.mock import patch

from lumi import posttraining_production_source_policy_preflight_v1 as policy

_SYNTHETIC_MARKER = "synthetic_unit_test_only_not_operational_evidence"
_ZERO_RESOURCE_FIELDS = (
    "candidate_row_reads",
    "gpu_use",
    "inference_calls",
    "network_bytes",
    "registry_or_data_contract_writes",
    "source_row_reads",
    "test_object_reads",
    "training_actions",
)


def _reseal(value: dict, field: str) -> None:
    value.pop(field, None)
    value[field] = policy.sha256_value(value)


def _synthetic_report() -> dict:
    """Return a deliberately labelled, in-memory validator fixture, not evidence."""
    pending = copy.deepcopy(policy.EXPECTED_PENDING_OWNER)
    machine = copy.deepcopy(policy.EXPECTED_MACHINE_ASSISTED_NO_HUMAN_REVIEW)
    report = {
        "fixture_kind": _SYNTHETIC_MARKER,
        "schema_version": policy.REPORT_SCHEMA_VERSION,
        "status": policy.REPORT_STATUS,
        "plan_file_sha256": policy.sha256_value([_SYNTHETIC_MARKER, "not_a_plan"]),
        "input_bindings": {
            name: policy.sha256_value([_SYNTHETIC_MARKER, name])
            for name in sorted(policy.EXPECTED_BINDING_NAMES)
        },
        "audited_mix": {
            "records": policy.EXPECTED_INVENTORY["input_records"],
            "strict_gap": 0,
        },
        "registry_inventory": {
            "registry_source_count": policy.EXPECTED_INVENTORY["registry_sources"],
            "production_eligible_registry_source_ids": [],
        },
        "unresolved_release_evidence": {
            "pending_owner_review_provenance": {
                "by_source": pending,
                "records": sum(item["records"] for item in pending.values()),
                "train_supervised_tokens": sum(
                    item["train_supervised_tokens"] for item in pending.values()
                ),
            },
            "machine_assisted_zero_human_review": {
                "by_source": machine,
                "model_assisted_rows": sum(
                    item["model_assisted_rows"] for item in machine.values()
                ),
                "human_review_completed_rows": sum(
                    item["human_review_completed_rows"] for item in machine.values()
                ),
            },
        },
        "decision": {
            "active_data_contract_v0_2_materialization_permitted_by_this_gate": False,
            "private_research_nonproduction_successor_ready_for_explicit_authorization": True,
            "production_contract_activation_permitted_now": False,
            "registry_v0_25_materialization_permitted_by_this_gate": False,
            "recommended_path": "private_research_nonproduction_data_contract_v0_2",
        },
        "successor": {
            # This is the existing contract string, not a new user approval.
            "exact_authorization_text": policy.SUCCESSOR_AUTHORIZATION_TEXT,
        },
        "resource_accounting": {
            **dict.fromkeys(_ZERO_RESOURCE_FIELDS, 0),
            "factual_passes": 1,
        },
        "raw_training_text_retained": False,
        "production_eligible": False,
        "next_gate": {"fixture_kind": _SYNTHETIC_MARKER, "authorization_required": True},
    }
    _reseal(report, "report_sha256")
    return report


def _synthetic_verification() -> tuple[dict, dict]:
    report = _synthetic_report()
    verification = policy.build_verification(
        report=report,
        report_file_sha256=policy.sha256_value([_SYNTHETIC_MARKER, "not_a_report_file"]),
    )
    return report, verification


class ProductionSourcePolicySyntheticTests(unittest.TestCase):
    def test_synthetic_metadata_preserves_blocker_counts_without_file_access(self) -> None:
        with patch.object(Path, "open", side_effect=AssertionError("unexpected file access")):
            report = _synthetic_report()
            self.assertIsNone(policy.validate_report(report))
            self.assertEqual(report["fixture_kind"], _SYNTHETIC_MARKER)
            self.assertEqual(set(report["input_bindings"]), policy.EXPECTED_BINDING_NAMES)
            self.assertEqual(report["audited_mix"]["records"], 329_200)
            self.assertEqual(report["audited_mix"]["strict_gap"], 0)
            self.assertEqual(report["registry_inventory"]["registry_source_count"], 27)
            self.assertEqual(
                report["registry_inventory"]["production_eligible_registry_source_ids"], []
            )
            unresolved = report["unresolved_release_evidence"]
            self.assertEqual(unresolved["pending_owner_review_provenance"]["records"], 20_471)
            self.assertEqual(
                unresolved["machine_assisted_zero_human_review"]["model_assisted_rows"], 14_420
            )
            self.assertIs(report["production_eligible"], False)

    def test_resealed_activation_and_materialization_claims_are_rejected(self) -> None:
        for field in (
            "active_data_contract_v0_2_materialization_permitted_by_this_gate",
            "production_contract_activation_permitted_now",
            "registry_v0_25_materialization_permitted_by_this_gate",
        ):
            with self.subTest(field=field):
                changed = _synthetic_report()
                changed["decision"][field] = True
                _reseal(changed, "report_sha256")
                with self.assertRaisesRegex(ValueError, "report boundary changed"):
                    policy.validate_report(changed)

    def test_resealed_production_eligibility_is_rejected(self) -> None:
        changed = _synthetic_report()
        changed["production_eligible"] = True
        _reseal(changed, "report_sha256")
        with self.assertRaisesRegex(ValueError, "report boundary changed"):
            policy.validate_report(changed)

    def test_resealed_pending_owner_count_and_token_drift_are_rejected(self) -> None:
        for field in ("records", "train_supervised_tokens"):
            for delta in (-1, 1):
                with self.subTest(field=field, delta=delta):
                    changed = _synthetic_report()
                    changed["unresolved_release_evidence"]["pending_owner_review_provenance"][
                        field
                    ] += delta
                    _reseal(changed, "report_sha256")
                    with self.assertRaisesRegex(ValueError, "report boundary changed"):
                        policy.validate_report(changed)

    def test_resealed_machine_review_or_row_count_claims_are_rejected(self) -> None:
        for field in ("human_review_completed_rows", "model_assisted_rows"):
            with self.subTest(field=field):
                changed = _synthetic_report()
                changed["unresolved_release_evidence"]["machine_assisted_zero_human_review"][
                    field
                ] += 1
                _reseal(changed, "report_sha256")
                with self.assertRaisesRegex(ValueError, "report boundary changed"):
                    policy.validate_report(changed)

    def test_report_seal_rejects_unresealed_metadata_tampering(self) -> None:
        changed = _synthetic_report()
        changed["fixture_kind"] += "_changed"
        with self.assertRaisesRegex(ValueError, "report boundary changed"):
            policy.validate_report(changed)

    def test_verification_links_to_exact_in_memory_report(self) -> None:
        with patch.object(Path, "open", side_effect=AssertionError("unexpected file access")):
            report, verification = _synthetic_verification()
            self.assertIsNone(policy.validate_verification(verification))
            self.assertEqual(verification["report_internal_sha256"], report["report_sha256"])
            self.assertEqual(verification["next_gate"]["fixture_kind"], _SYNTHETIC_MARKER)
            self.assertIs(verification["production_eligible"], False)
            for field in _ZERO_RESOURCE_FIELDS:
                self.assertEqual(verification["resource_accounting"][field], 0)
            self.assertEqual(verification["resource_accounting"]["factual_passes"], 2)

    def test_verification_rejects_each_resealed_false_check(self) -> None:
        _, verification = _synthetic_verification()
        for field in verification["checks"]:
            with self.subTest(field=field):
                changed = copy.deepcopy(verification)
                changed["checks"][field] = False
                _reseal(changed, "verification_sha256")
                with self.assertRaisesRegex(ValueError, "verification boundary changed"):
                    policy.validate_verification(changed)

    def test_verification_checks_must_be_literal_true(self) -> None:
        _, verification = _synthetic_verification()
        verification["checks"]["no_registry_or_active_data_contract_mutation"] = 1
        _reseal(verification, "verification_sha256")
        with self.assertRaisesRegex(ValueError, "verification boundary changed"):
            policy.validate_verification(verification)

    def test_verification_rejects_unresealed_digest_tampering(self) -> None:
        _, verification = _synthetic_verification()
        verification["report_internal_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "verification boundary changed"):
            policy.validate_verification(verification)

    def test_verification_rejects_resealed_production_claim(self) -> None:
        _, verification = _synthetic_verification()
        verification["production_eligible"] = True
        _reseal(verification, "verification_sha256")
        with self.assertRaisesRegex(ValueError, "verification boundary changed"):
            policy.validate_verification(verification)

    def test_report_rejects_every_resealed_nonzero_resource(self) -> None:
        for field in _ZERO_RESOURCE_FIELDS:
            with self.subTest(field=field):
                changed = _synthetic_report()
                changed["resource_accounting"][field] = 1
                _reseal(changed, "report_sha256")
                with self.assertRaisesRegex(ValueError, "report boundary changed"):
                    policy.validate_report(changed)

    def test_report_rejects_resealed_factual_pass_drift(self) -> None:
        for count in (0, 2):
            with self.subTest(count=count):
                changed = _synthetic_report()
                changed["resource_accounting"]["factual_passes"] = count
                _reseal(changed, "report_sha256")
                with self.assertRaisesRegex(ValueError, "report boundary changed"):
                    policy.validate_report(changed)

    def test_resealed_nested_raw_text_keys_are_rejected(self) -> None:
        for field in ("messages", "content", "text"):
            with self.subTest(field=field):
                changed = _synthetic_report()
                changed["synthetic_nested"] = [{field: []}]
                _reseal(changed, "report_sha256")
                with self.assertRaisesRegex(ValueError, "raw candidate text key forbidden"):
                    policy.validate_report(changed)


if __name__ == "__main__":
    unittest.main()
