from __future__ import annotations

import ast
import hashlib
import socket
import unittest
from pathlib import Path

import lumi.posttraining_bedstc2_authoritative_rights_acquisition_v1 as acquisition
import lumi.posttraining_bedstc2_authoritative_rights_factual_review_v1 as review


def synthetic_payloads() -> list[bytes]:
    values = {
        "microsoft_dstc_event": b"<html>Dialog State Tracking Challenge</html>",
        "dstc2_acl_anthology_landing": b"<html>Second Dialog State Tracking Challenge</html>",
        "matthen_dstc_readme_exact_revision": b"Dialog State Tracking Challenge DSTC2",
        "matthen_dstc_license_exact_revision": b"GNU GENERAL PUBLIC LICENSE Version 3",
        "dstc2_primary_paper": b"%PDF-1.7 synthetic",
        "code_mixed_dialog_paper": b"%PDF-1.7 synthetic",
        "code_mixed_dialog_readme_exact_revision": b"Code-Mixed-Dialog based on DSTC2",
        "code_mixed_dialog_license_exact_revision": b"Apache License Version 2.0",
    }
    return [values[item["id"]] for item in acquisition.EXPECTED_DOCUMENTS]


class FactualReviewContractTests(unittest.TestCase):
    def test_two_pass_zero_network_contract(self) -> None:
        self.assertEqual(review.EXPECTED_LIMITS["passes"], 2)
        self.assertEqual(review.EXPECTED_LIMITS["document_reads_per_pass"], 8)
        self.assertEqual(review.EXPECTED_LIMITS["network_bytes"], 0)
        self.assertEqual(review.EXPECTED_LIMITS["network_calls"], 0)

    def test_synthetic_assessment_remains_fail_closed(self) -> None:
        result = review.assess_payloads(synthetic_payloads())
        self.assertEqual(result["observations"], review.EXPECTED_OBSERVATIONS)
        self.assertEqual(result["criteria_satisfied"], 0)
        self.assertEqual(result["criteria_unresolved"], 5)
        self.assertFalse(any(result["criteria"].values()))

    def test_report_is_raw_text_free_and_nonproduction(self) -> None:
        digest = "a" * 64
        report = review.assemble_report(
            review.assess_payloads(synthetic_payloads()),
            acquisition_plan_sha256=digest,
            manifest_file_sha256=digest,
            manifest_internal_sha256=digest,
            receipt_file_sha256=digest,
            receipt_internal_sha256=digest,
            acquisition_verification_file_sha256=digest,
            acquisition_verification_internal_sha256=digest,
            document_bytes=sum(map(len, synthetic_payloads())),
        )
        review.validate_report(report)
        self.assertEqual(report["status"], review.REPORT_STATUS)
        self.assertFalse(report["production_eligible"])
        self.assertFalse(report["legal_conclusion_made"])
        encoded = review.canonical_json_file_bytes(report).lower()
        self.assertNotIn(b"dialog state tracking challenge</html>", encoded)

    def test_tampered_observation_fails(self) -> None:
        payloads = synthetic_payloads()
        payloads[3] = b"MIT License"
        with self.assertRaises(ValueError):
            review.assess_payloads(payloads)

    def test_network_disabled_poison_and_restore(self) -> None:
        original = socket.getaddrinfo
        with review.network_disabled():
            with self.assertRaises(RuntimeError):
                socket.getaddrinfo("example.com", 443)
        self.assertIs(socket.getaddrinfo, original)

    def test_verifier_does_not_call_first_pass_assessor_or_report_builder(self) -> None:
        source = Path(
            "scripts/verify_sft_bedstc2_authoritative_rights_factual_review_v1.py"
        ).read_text()
        tree = ast.parse(source)
        calls = {
            ast.unparse(node.func)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        }
        self.assertNotIn("review.assess_payloads", calls)
        self.assertNotIn("review.assemble_report", calls)
        self.assertIn("_independent_observations", calls)
        self.assertIn("review.network_disabled", calls)

    def test_markers_precede_document_replay_in_both_passes(self) -> None:
        for script, marker_call in (
            ("scripts/review_sft_bedstc2_authoritative_rights_facts_v1.py", "review.start_marker"),
            ("scripts/verify_sft_bedstc2_authoritative_rights_factual_review_v1.py", "review.start_marker"),
        ):
            source = Path(script).read_text()
            self.assertLess(source.index(marker_call), source.index("replay_acquired_inventory"))

    def test_reason_codes_and_next_gate_are_frozen(self) -> None:
        self.assertEqual(len(review.EXPECTED_REASON_CODES), 5)
        self.assertEqual(
            review.EXPECTED_NEXT_GATE["action"],
            "qualified_rights_owner_confirmation_or_legal_review",
        )
        self.assertFalse(review.EXPECTED_NEXT_GATE["promotion_seal_or_training_authorized"])


if __name__ == "__main__":
    unittest.main()
