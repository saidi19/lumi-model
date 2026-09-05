from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import lumi.posttraining_massive_bn_bd_parser_owner_review_v4 as contract


def _record_ids() -> list[str]:
    return [f"massive-bn-bd-parser-{index:024x}" for index in range(64)]


def _decisions(*, accepted: int = 64, reviewer: str = "owner-afnan") -> dict:
    return {
        "schema_version": contract.DECISIONS_SCHEMA_VERSION,
        "sample_sha256": contract.EXPECTED_INPUTS["candidate"]["sample_sha256"],
        "reviewer_id": reviewer,
        "reviewed_at": contract.EXPECTED_REVIEWED_AT,
        "decisions": [
            {
                "record_id": record_id,
                "accepted": index < accepted,
                "note": "",
            }
            for index, record_id in enumerate(_record_ids())
        ],
    }


def _candidate_rows() -> list[dict]:
    return [
        {
            "language": {},
            "messages": [
                {"role": "user", "content": "u"},
                {"role": "tool", "content": "t"},
                {"role": "tool_result", "content": "r"},
                {"role": "assistant", "content": "a"},
            ],
            "provenance": {},
            "quality": {},
            "record_id": record_id,
            "safety": {},
            "schema_version": "synthetic",
            "source": {},
            "split": "train",
            "task_category": "tool_api",
        }
        for record_id in _record_ids()
    ]


def _candidate_payload(rows: list[dict]) -> bytes:
    return b"".join(contract.canonical_json_bytes(row) + b"\n" for row in rows)


def _synthetic_outcome() -> dict:
    return {
        "outcome": {
            "accepted": 64,
            "rejected": 0,
            "nonempty_notes": 0,
            "reviewed_at": contract.EXPECTED_REVIEWED_AT,
        },
        "reviewed_ids_sha256": contract.EXPECTED_REVIEWED_IDS_DOMAIN_SHA256,
    }


class MassiveOwnerReviewContractTests(unittest.TestCase):
    def test_strict_json_rejects_duplicate_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            contract.strict_json_bytes(b'{"a":1,"a":2}')

    def test_decisions_accept_exact_complete_review(self) -> None:
        self.assertEqual(
            contract._validate_decisions(_decisions(), _record_ids()),
            {
                "accepted": 64,
                "rejected": 0,
                "nonempty_notes": 0,
                "reviewed_at": contract.EXPECTED_REVIEWED_AT,
            },
        )

    def test_decisions_reject_automated_or_invalid_alias(self) -> None:
        for reviewer in ("codex", "openai-owner", "bad alias"):
            with self.subTest(reviewer=reviewer):
                with self.assertRaisesRegex(ValueError, "decision binding"):
                    contract._validate_decisions(
                        _decisions(reviewer=reviewer),
                        _record_ids(),
                    )

    def test_decisions_reject_below_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "coverage or threshold"):
            contract._validate_decisions(_decisions(accepted=60), _record_ids())

    def test_decisions_reject_reordered_ids(self) -> None:
        value = _decisions()
        value["decisions"][0], value["decisions"][1] = (
            value["decisions"][1],
            value["decisions"][0],
        )
        with self.assertRaisesRegex(ValueError, "coverage or threshold"):
            contract._validate_decisions(value, _record_ids())

    def test_candidate_projection_is_deterministic(self) -> None:
        rows = _candidate_rows()
        sample = [
            {
                "record_id": row["record_id"],
                "skill_id": "amazon_massive_bn_bd_tool_canary:pure_parser",
                "language_mode": "bn_bd_source_utterance",
                "split": "train",
                "messages": row["messages"],
            }
            for row in rows
        ]
        original = contract.EXPECTED_INPUTS["candidate"]["sample_sha256"]
        contract.EXPECTED_INPUTS["candidate"]["sample_sha256"] = (
            contract.sha256_value(sample)
        )
        try:
            observed, ids = contract._candidate_sample(_candidate_payload(rows))
        finally:
            contract.EXPECTED_INPUTS["candidate"]["sample_sha256"] = original
        self.assertEqual(observed, sample)
        self.assertEqual(ids, _record_ids())

    def test_candidate_projection_rejects_wrong_role(self) -> None:
        rows = _candidate_rows()
        rows[0]["messages"][1] = {"role": "assistant", "content": "t"}
        with self.assertRaisesRegex(ValueError, "message contract"):
            contract._candidate_sample(_candidate_payload(rows))

    def test_receipt_is_raw_text_free(self) -> None:
        with patch.object(
            contract,
            "_read_and_validate_inputs",
            return_value=_synthetic_outcome(),
        ):
            receipt = contract.build_receipt(
                {},
                plan_file_sha256="1" * 64,
                repository_root=Path("."),
            )
        rendered = json.dumps(receipt, sort_keys=True)
        self.assertNotIn('"messages"', rendered)
        self.assertNotIn('"record_id"', rendered)
        self.assertNotIn('"reviewer_id"', rendered)
        self.assertIs(receipt["production_eligible"], False)

    def test_receipt_rejects_changed_rights(self) -> None:
        with patch.object(
            contract,
            "_read_and_validate_inputs",
            return_value=_synthetic_outcome(),
        ):
            receipt = contract.build_receipt(
                {},
                plan_file_sha256="1" * 64,
                repository_root=Path("."),
            )
        changed = copy.deepcopy(receipt)
        changed["rights"]["license"] = "changed"
        changed["receipt_sha256"] = contract.sha256_value(
            {
                key: value
                for key, value in changed.items()
                if key != "receipt_sha256"
            }
        )
        with self.assertRaisesRegex(ValueError, "receipt changed"):
            contract.validate_receipt(changed)

    def test_verification_is_nonproduction(self) -> None:
        with patch.object(
            contract,
            "_read_and_validate_inputs",
            return_value=_synthetic_outcome(),
        ):
            receipt = contract.build_receipt(
                {},
                plan_file_sha256="1" * 64,
                repository_root=Path("."),
            )
        verification = contract.build_verification(
            receipt,
            receipt_file_sha256="2" * 64,
            receipt_file_bytes=2_048,
            plan_file_sha256="1" * 64,
        )
        self.assertIs(verification["production_eligible"], False)
        self.assertEqual(
            verification["independent_replay"]["candidate_row_reads"],
            64,
        )
        self.assertEqual(verification["zero_use"]["training_actions"], 0)

    def test_marker_cannot_overwrite_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_text("existing", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                contract.start_marker(path, contract.EXPECTED_RECEIPT_MARKER)

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
