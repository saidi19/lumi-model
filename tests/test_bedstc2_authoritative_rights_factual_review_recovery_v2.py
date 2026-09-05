from __future__ import annotations

import ast
import copy
import hashlib
import unittest
from pathlib import Path

import lumi.posttraining_bedstc2_authoritative_rights_factual_review_recovery_v2 as recovery


ROOT = Path(__file__).resolve().parents[1]


def _synthetic_witness() -> dict[str, object]:
    return {
        "call_lines": {
            "review.start_marker": 52,
            "acquisition.start_marker": 53,
            "acquisition.replay_acquired_inventory": 58,
            "review.assess_payloads": 67,
            "acquisition.finish_marker": 68,
            "review.sha256_bytes": 89,
            "review.finish_marker": 113,
        },
        "manifest": {"manifest_sha256": recovery.EXPECTED_INPUTS["private_manifest"]["internal_sha256"]},
        "receipt": {"receipt_sha256": recovery.EXPECTED_INPUTS["acquisition_receipt"]["internal_sha256"]},
        "verification": {"verification_sha256": recovery.EXPECTED_INPUTS["acquisition_verification"]["internal_sha256"]},
    }


def _call_name(node: ast.Call) -> str:
    value: ast.expr = node.func
    parts: list[str] = []
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return ".".join(reversed(parts))


class RecoveryV2Tests(unittest.TestCase):
    def test_exact_authorization_and_zero_network_contract(self) -> None:
        payload = recovery.EXPECTED_AUTHORIZATION_TEXT.encode("utf-8")
        self.assertEqual(len(payload), 49)
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            recovery.EXPECTED_AUTHORIZATION_SHA256,
        )
        self.assertEqual(recovery.EXPECTED_LIMITS["recovery_raw_document_reads"], 0)
        self.assertEqual(recovery.EXPECTED_LIMITS["network_calls"], 0)
        self.assertEqual(recovery.EXPECTED_LIMITS["independent_pass2_document_reads"], 8)
        self.assertFalse(recovery.EXPECTED_RECOVERY_CONTRACT["production_eligible"])

    def test_bound_failed_runner_proves_order_and_missing_attribute(self) -> None:
        runner = (ROOT / recovery.EXPECTED_INPUTS["failed_review_runner"]["path"]).read_bytes()
        module = (ROOT / recovery.EXPECTED_INPUTS["failed_review_implementation"]["path"]).read_bytes()
        self.assertEqual(
            recovery.validate_failed_runner_witness(runner, module),
            {
                "review.start_marker": 52,
                "acquisition.start_marker": 53,
                "acquisition.replay_acquired_inventory": 58,
                "review.assess_payloads": 67,
                "acquisition.finish_marker": 68,
                "review.sha256_bytes": 89,
                "review.finish_marker": 113,
            },
        )
        with self.assertRaises(ValueError):
            recovery.validate_failed_runner_witness(
                runner.replace(b"review.sha256_bytes", b"acquisition.sha256_bytes"),
                module,
            )

    def test_recovery_runner_never_replays_or_opens_raw_documents(self) -> None:
        source = (
            ROOT / recovery.EXPECTED_CODE_PATHS["runner"]
        ).read_text(encoding="utf-8")
        self.assertNotIn("replay_acquired_inventory", source)
        self.assertNotIn("read_exact_file", source)
        self.assertNotIn("assess_payloads", source)
        self.assertNotIn('/ "raw"', source)
        self.assertIn("review_v1.network_disabled", source)

    def test_recovery_runner_module_attributes_exist(self) -> None:
        source = (ROOT / recovery.EXPECTED_CODE_PATHS["runner"]).read_text()
        tree = ast.parse(source)
        names = {
            _call_name(node).split(".", 1)[1]
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and _call_name(node).startswith("recovery.")
        }
        for name in names:
            self.assertTrue(hasattr(recovery, name), name)

    def test_report_is_sealed_raw_text_free_and_nonproduction(self) -> None:
        report = recovery.assemble_report("a" * 64, _synthetic_witness())
        recovery.validate_report(report)
        self.assertFalse(report["production_eligible"])
        self.assertEqual(report["recovery"]["raw_documents_read_during_recovery"], 0)
        self.assertEqual(report["document_accounting"]["factual_document_passes_completed"], 1)
        encoded = recovery.canonical_json_file_bytes(report).lower()
        self.assertNotIn(b"dialog state tracking challenge</html>", encoded)

    def test_resealed_production_or_pass_tampering_fails(self) -> None:
        for key, value in (
            ("production_eligible", True),
            ("qualified_review_pending", False),
        ):
            report = recovery.assemble_report("a" * 64, _synthetic_witness())
            report[key] = value
            report["report_sha256"] = recovery.sha256_value(
                {name: item for name, item in report.items() if name != "report_sha256"}
            )
            with self.assertRaises(ValueError):
                recovery.validate_report(report)
        report = recovery.assemble_report("a" * 64, _synthetic_witness())
        report = copy.deepcopy(report)
        report["recovery"]["raw_documents_read_during_recovery"] = 1
        report["report_sha256"] = recovery.sha256_value(
            {name: item for name, item in report.items() if name != "report_sha256"}
        )
        with self.assertRaises(ValueError):
            recovery.validate_report(report)

    def test_verifier_starts_marker_then_replays_exactly_once_independently(self) -> None:
        source = (ROOT / recovery.EXPECTED_CODE_PATHS["verifier"]).read_text()
        tree = ast.parse(source)
        calls = [
            (_call_name(node), node.lineno)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        ]
        starts = [line for name, line in calls if name == "recovery.start_marker"]
        replays = [line for name, line in calls if name == "acquisition.replay_acquired_inventory"]
        self.assertEqual(len(starts), 1)
        self.assertEqual(len(replays), 1)
        self.assertLess(starts[0], replays[0])
        self.assertNotIn("review_v1.assess_payloads", {name for name, _ in calls})
        self.assertIn("_independent_observations", {name for name, _ in calls})
        self.assertIn("review_v1.network_disabled", {name for name, _ in calls})

    def test_verification_closes_exactly_two_factual_document_passes(self) -> None:
        report = recovery.assemble_report("a" * 64, _synthetic_witness())
        verification = recovery.assemble_verification(
            report, report_file_sha256="b" * 64, document_bytes=2_167_236
        )
        self.assertEqual(
            verification["document_accounting"]["total_factual_document_passes"], 2
        )
        self.assertEqual(
            verification["document_accounting"]["recovery_materialization_document_reads"], 0
        )
        self.assertFalse(verification["production_eligible"])


if __name__ == "__main__":
    unittest.main()
