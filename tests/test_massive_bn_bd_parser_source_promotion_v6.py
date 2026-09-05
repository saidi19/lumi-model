from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import lumi.posttraining_massive_bn_bd_parser_source_promotion_v6 as contract

ROOT = Path(__file__).resolve().parents[1]


def _synthetic_successor() -> dict:
    return {
        "schema_version": "lumi.sft.sources.v0.1",
        "observed_at": contract.EXPECTED_SUCCESSOR["observed_at"],
        "sources": [copy.deepcopy(contract.EXPECTED_SOURCE)],
    }


def _synthetic_receipt() -> dict:
    with (
        patch.object(
            contract,
            "_validate_preflight_evidence",
            return_value=({}, {}),
        ),
        patch.object(contract, "_predecessor_registry", return_value={}),
        patch.object(
            contract,
            "project_successor_registry",
            return_value=_synthetic_successor(),
        ),
    ):
        receipt, _ = contract.build_receipt(
            plan={},
            plan_file_sha256="1" * 64,
            repository_root=ROOT,
        )
    return receipt


class MassiveSourcePromotionTests(unittest.TestCase):
    def test_strict_json_rejects_duplicate_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            contract.strict_json_bytes(b'{"a":1,"a":2}')

    def test_registry_projection_is_exact_append_only(self) -> None:
        predecessor = contract.strict_json_bytes(
            (ROOT / contract.EXPECTED_INPUTS["predecessor_registry"]["path"])
            .read_bytes()
        )
        successor = contract.project_successor_registry(
            predecessor,
            repository_root=ROOT,
        )
        self.assertEqual(successor["sources"][:-1], predecessor["sources"])
        self.assertEqual(successor["sources"][-1], contract.EXPECTED_SOURCE)
        self.assertEqual(
            {
                key: successor[key]
                for key in successor
                if key not in {"observed_at", "sources"}
            },
            {
                key: predecessor[key]
                for key in predecessor
                if key not in {"observed_at", "sources"}
            },
        )
        payload = contract.json_bytes(successor)
        self.assertEqual(len(payload), contract.EXPECTED_SUCCESSOR["bytes"])
        self.assertEqual(
            contract.sha256_bytes(payload),
            contract.EXPECTED_SUCCESSOR["file_sha256"],
        )

    def test_registry_projection_rejects_existing_source(self) -> None:
        predecessor = contract.strict_json_bytes(
            (ROOT / contract.EXPECTED_INPUTS["predecessor_registry"]["path"])
            .read_bytes()
        )
        changed = copy.deepcopy(predecessor)
        changed["sources"][0]["dataset_id"] = contract.EXPECTED_SOURCE["dataset_id"]
        with self.assertRaisesRegex(ValueError, "already exists"):
            contract.project_successor_registry(changed, repository_root=ROOT)

    def test_receipt_is_raw_text_free_and_nonproduction(self) -> None:
        receipt = _synthetic_receipt()
        rendered = contract.json_bytes(receipt)
        for key in (b'"messages"', b'"content"', b'"record_id"'):
            self.assertNotIn(key, rendered)
        self.assertIs(receipt["production_eligible"], False)
        self.assertEqual(receipt["resource_accounting"]["candidate_row_reads"], 0)
        self.assertEqual(
            receipt["next_gate"]["action"],
            "full_registry_v0_22_candidate_mix_reaudit",
        )

    def test_receipt_tamper_is_rejected(self) -> None:
        receipt = _synthetic_receipt()
        changed = copy.deepcopy(receipt)
        changed["registry"]["successor_sources"] = 26
        changed["receipt_sha256"] = contract.sha256_value(
            {key: value for key, value in changed.items() if key != "receipt_sha256"}
        )
        with self.assertRaisesRegex(ValueError, "receipt changed"):
            contract.validate_receipt(changed)

    def test_verification_accounts_for_two_passes(self) -> None:
        receipt = _synthetic_receipt()
        verification = contract.build_verification(
            receipt,
            receipt_file_sha256="2" * 64,
            receipt_file_bytes=2_048,
            plan_file_sha256="1" * 64,
        )
        accounting = verification["combined_resource_accounting"]
        self.assertEqual(accounting["materialization_passes"], 1)
        self.assertEqual(accounting["verification_passes"], 1)
        self.assertEqual(accounting["candidate_row_reads"], 0)
        self.assertIs(verification["production_eligible"], False)

    def test_new_file_writer_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            contract.write_new_file(path, b"first")
            with self.assertRaises(FileExistsError):
                contract.write_new_file(path, b"second")
            self.assertEqual(path.read_bytes(), b"first")

    def test_marker_finish_preserves_inode_and_replaces_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
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
