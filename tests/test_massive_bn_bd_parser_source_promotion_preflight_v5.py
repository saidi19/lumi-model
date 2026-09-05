from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import lumi.posttraining_massive_bn_bd_parser_source_promotion_preflight_v5 as contract


def _clean_audit() -> dict:
    return {
        "baseline_candidate_row_reads": contract.EXPECTED_BASELINE["records"],
        "baseline_sources": contract.EXPECTED_BASELINE["candidate_sources"],
        "collision_ledger_sha256": contract.EXPECTED_EMPTY_COLLISION_LEDGER_SHA256,
        "cross_source_exact_collision_rows": 0,
        "cross_source_minhash_0_9_collision_rows": 0,
        "historical_owner_review_pending_rows": 64,
        "historical_registry_pending_rows": 64,
        "massive_candidate_row_reads": 64,
        "retained_massive_rows": 64,
        "total_collision_rows": 0,
    }


def _projection() -> dict:
    return {
        "added_sources": 1,
        "changed_predecessor_sources": 0,
        "materialized": False,
        "predecessor_file_sha256": contract.EXPECTED_INPUTS["source_registry"][
            "sha256"
        ],
        "predecessor_sources": 24,
        "projected_canonical_sha256": contract.EXPECTED_PROJECTED_REGISTRY[
            "canonical_sha256"
        ],
        "projected_file_bytes": contract.EXPECTED_PROJECTED_REGISTRY["bytes"],
        "projected_file_sha256": contract.EXPECTED_PROJECTED_REGISTRY["file_sha256"],
        "projected_observed_at": contract.EXPECTED_PROJECTED_REGISTRY["observed_at"],
        "projected_sources": 25,
        "removed_sources": 0,
        "source_id": contract.EXPECTED_NEW_SOURCE["source_id"],
        "successor_path": "configs/lumi-4b-sft-source-registry-v0.22.json",
        "successor_write_authorized": False,
    }


def _build_synthetic_report(*, clean: bool = True) -> dict:
    audit = _clean_audit()
    if not clean:
        audit.update(
            {
                "collision_ledger_sha256": "a" * 64,
                "cross_source_exact_collision_rows": 1,
                "retained_massive_rows": 63,
                "total_collision_rows": 1,
            }
        )
    with (
        patch.object(
            contract,
            "_validate_bound_evidence",
            return_value=({}, {}, {}),
        ),
        patch.object(contract, "_cross_source_audit", return_value=audit),
        patch.object(contract, "_project_registry", return_value=_projection()),
    ):
        return contract.build_report(
            plan={},
            plan_file_sha256="1" * 64,
            repository_root=Path("."),
        )


class MassiveSourcePromotionPreflightTests(unittest.TestCase):
    def test_strict_json_rejects_duplicate_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            contract.strict_json_bytes(b'{"a":1,"a":2}')

    def test_clean_report_selects_registry_promotion_gate(self) -> None:
        report = _build_synthetic_report()
        self.assertEqual(report["status"], contract.PASS_STATUS)
        self.assertEqual(report["cross_source_audit"]["retained_massive_rows"], 64)
        self.assertEqual(
            report["next_gate"]["action"],
            "separate_registry_v0_22_promotion_authorization",
        )
        self.assertIs(report["registry_successor_materialized"], False)
        self.assertIs(report["production_eligible"], False)

    def test_collision_report_stops_before_registry_promotion(self) -> None:
        report = _build_synthetic_report(clean=False)
        self.assertEqual(report["status"], contract.STOP_STATUS)
        self.assertEqual(
            report["next_gate"]["action"],
            "separate_collision_localization_and_candidate_repair_authorization",
        )
        self.assertIs(report["registry_projection"]["promotion_preflight_passed"], False)

    def test_report_rejects_raw_text_key(self) -> None:
        report = _build_synthetic_report()
        changed = copy.deepcopy(report)
        changed["content"] = "synthetic"
        changed["report_sha256"] = contract.sha256_value(
            {key: value for key, value in changed.items() if key != "report_sha256"}
        )
        with self.assertRaisesRegex(ValueError, "raw-text key"):
            contract.validate_report(changed)

    def test_verification_accounts_for_exactly_two_passes(self) -> None:
        report = _build_synthetic_report()
        verification = contract.build_verification(
            report,
            report_file_sha256="2" * 64,
            report_file_bytes=2_048,
            plan_file_sha256="1" * 64,
        )
        self.assertEqual(
            verification["combined_resource_accounting"],
            {
                "baseline_candidate_row_reads": 629_432,
                "full_audit_passes": 2,
                "massive_candidate_row_reads": 128,
                "source_line_reads": 0,
                "test_object_reads": 0,
                "registry_writes": 0,
            },
        )
        self.assertIs(verification["production_eligible"], False)

    def test_projected_source_remains_nonproduction(self) -> None:
        self.assertEqual(
            contract.EXPECTED_NEW_SOURCE["source_id"],
            "amazon_massive_bn_bd_tool_canary",
        )
        self.assertEqual(contract.EXPECTED_NEW_SOURCE["license_id"], "cc-by-4.0")
        self.assertIs(contract.EXPECTED_NEW_SOURCE["production_eligible"], False)
        self.assertEqual(len(contract.EXPECTED_NEW_SOURCE["production_blockers"]), 3)

    def test_marker_cannot_overwrite_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text("existing", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                contract.start_marker(path, contract.EXPECTED_RECEIPT_MARKER)

    def test_marker_finish_preserves_inode_and_replaces_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            identity = contract.start_marker(path, contract.EXPECTED_RECEIPT_MARKER)
            payload = b'{"status":"PASS"}\n'
            contract.finish_marker(
                path,
                contract.EXPECTED_RECEIPT_MARKER,
                payload,
                identity,
            )
            self.assertEqual(path.read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()
