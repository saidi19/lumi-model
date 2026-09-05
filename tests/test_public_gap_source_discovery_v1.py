from __future__ import annotations

import copy
import unittest
from pathlib import Path

from lumi.posttraining_public_gap_source_discovery_v1 import (
    build_report,
    build_verification,
    load_acquisition_preflight,
    load_plan,
    sha256_file,
    sha256_value,
    validate_report,
    validate_verification,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-public-gap-source-discovery-v0.1.json"
ACQUISITION = (
    ROOT / "configs/lumi-4b-sft-public-gap-canary-acquisition-preflight-v0.1.json"
)


def _build() -> tuple[dict, dict, dict]:
    plan = load_plan(PLAN, repository_root=ROOT)
    acquisition = load_acquisition_preflight(ACQUISITION)
    report = build_report(
        plan,
        acquisition,
        plan_file_sha256=sha256_file(PLAN),
        acquisition_preflight_file_sha256=sha256_file(ACQUISITION),
        repository_root=ROOT,
    )
    return plan, acquisition, report


class PublicGapSourceDiscoveryTests(unittest.TestCase):
    def test_public_source_decisions_remain_fail_closed(self) -> None:
        _, _, report = _build()
        validate_report(report)
        self.assertIs(report["production_eligible"], False)
        self.assertIs(report["raw_training_text_retained"], False)
        self.assertEqual(
            report["decision_summary"],
            {
                "new_public_families_ready_for_payload_acquisition": 1,
                "new_public_families_on_hold": 1,
                "public_source_closes_full_language_gap": False,
                "public_source_closes_factual_qa_gap": False,
                "massive_may_seed_tool_api_only_after_all_later_gates": True,
                "jontro_has_zero_active_credit": True,
                "first_party_creation_remains_required": True,
            },
        )
        self.assertIs(report["prepared_successor"]["execution_authorized"], False)

    def test_massive_object_is_exact_and_jontro_stays_on_hold(self) -> None:
        _, _, report = _build()
        decisions = {item["source_id"]: item for item in report["source_decisions"]}
        self.assertEqual(
            decisions["amazon_massive_bn_bd_tool_canary"]["exact_train_object"],
            {
                "bytes": 1071668,
                "path": "bn-BD/train/0000.parquet",
                "sha256": "947f65c28cce0d9a57c16bc1a99df08f84e9b7599dff25f47fd5b9760256c17d",
            },
        )
        self.assertEqual(
            decisions["jontro_bangla_tool_use_conditional_canary"][
                "allowed_credit_before_clearance"
            ],
            {"banglish": 0, "tool_api": 0},
        )

    def test_report_rejects_premature_execution_or_credit(self) -> None:
        _, _, report = _build()
        modified = copy.deepcopy(report)
        modified["prepared_successor"]["execution_authorized"] = True
        modified["report_sha256"] = sha256_value(
            {key: value for key, value in modified.items() if key != "report_sha256"}
        )
        with self.assertRaisesRegex(ValueError, "unexpectedly executable"):
            validate_report(modified)

    def test_verification_seals_all_checks(self) -> None:
        plan, acquisition, report = _build()
        verification = build_verification(
            plan=plan,
            acquisition_preflight=acquisition,
            report=report,
            report_file_sha256="0" * 64,
        )
        validate_verification(verification)
        self.assertTrue(all(verification["checks"].values()))
        self.assertIs(verification["production_eligible"], False)


if __name__ == "__main__":
    unittest.main()
