"""Generated row pairs only: no private files, tokenizer, or old helpers."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lumi import posttraining_safety_uri_row_delta_v1 as delta  # noqa: E402


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _payload(rows: list[dict]) -> bytes:
    return b"".join(_canonical(row) + b"\n" for row in rows)


def synthetic_rows() -> tuple[list[dict], list[dict]]:
    before = []
    after = []
    for index in range(320):
        record_id = f"synthetic-{index:03d}"
        row = {
            "record_id": record_id,
            "split": "train" if index < 240 else "validation",
            "source": {
                "source_id": "lumi_safety_tool_sft_v0_1",
                "dataset_id": "Afnan-Technologies-Inc/lumi-safety-tool-sft-v0.1",
                "document_id": record_id,
            },
            "messages": [
                {"role": "user", "content": f"synthetic question {index}"},
                {"role": "assistant", "content": f"কৃত্রিম উত্তর {index}"},
            ],
            "provenance": {
                "transform_sha256": hashlib.sha256(f"prior {index}".encode()).hexdigest(),
                "human_review_status": "synthetic_not_an_approval",
            },
            "other_metadata": {"count": 0, "flag": False, "number": 1.0, "nested": [None, "x"]},
        }
        repaired = copy.deepcopy(row)
        uri = f"private://Afnan-Technologies-Inc/lumi-safety-tool-sft-v0.1/{record_id}"
        repaired["source"]["uri"] = uri
        repaired["provenance"]["transform_sha256"] = hashlib.sha256(
            _canonical(
                {
                    "operation": "add_deterministic_first_party_source_uri_v0.1",
                    "prior_transform_sha256": row["provenance"]["transform_sha256"],
                    "source_uri": uri,
                }
            )
        ).hexdigest()
        before.append(row)
        after.append(repaired)
    return before, after


def synthetic_pairs() -> tuple[bytes, bytes]:
    before, after = synthetic_rows()
    return _payload(before), _payload(after)


class SafetyUriRowDeltaTests(unittest.TestCase):
    def _reject(self, before: bytes, after: bytes) -> None:
        with self.assertRaisesRegex(ValueError, "^invalid safety URI row-delta contract$"):
            delta.audit_pairs(before, after)

    def test_success_is_limited_aggregate_proof(self) -> None:
        before, after = synthetic_pairs()
        result = delta.audit_pairs(before, after)
        self.assertEqual(
            result,
            {
                "schema_version": "lumi.sft-safety-uri-row-delta-audit-report.v0.1",
                "status": "PASS_ROW_DELTA_ONLY",
                "row_delta_verified": True,
                "rows_compared": 320,
                "unique_record_ids": 320,
                "unique_source_uris": 320,
                "tokenizer_recomputed": False,
                "generic_filtering_replayed": False,
                "full_private_integration_passed": False,
                "execution_authorized": False,
                "production_eligible": False,
            },
        )
        self.assertNotIn("synthetic-", json.dumps(result))
        self.assertNotIn("question", json.dumps(result))

    def test_no_file_access(self) -> None:
        before, after = synthetic_pairs()
        with (
            patch("builtins.open", side_effect=AssertionError("file access")),
            patch("io.open", side_effect=AssertionError("file access")),
            patch("os.open", side_effect=AssertionError("file access")),
        ):
            self.assertTrue(delta.audit_pairs(before, after)["row_delta_verified"])

    def test_json_key_order_and_whitespace_do_not_change_value(self) -> None:
        before, after = synthetic_rows()
        alternative = b"".join(json.dumps(row, ensure_ascii=True).encode() + b"\n" for row in after)
        self.assertTrue(delta.audit_pairs(_payload(before), alternative)["row_delta_verified"])

    def test_transform_drift_rejected_although_legacy_comparison_masks_it(self) -> None:
        before, after = synthetic_rows()
        after[0]["provenance"]["transform_sha256"] = "0" * 64
        legacy_comparison = copy.deepcopy(after[0])
        legacy_comparison["source"].pop("uri")
        legacy_comparison["provenance"]["transform_sha256"] = before[0]["provenance"][
            "transform_sha256"
        ]
        self.assertEqual(legacy_comparison, before[0])
        self._reject(_payload(before), _payload(after))

    def test_prior_transform_must_be_lowercase_sha256(self) -> None:
        for value in (None, True, "A" * 64, "g" * 64, "a" * 63):
            with self.subTest(value=value):
                before, after = synthetic_rows()
                before[0]["provenance"]["transform_sha256"] = value
                self._reject(_payload(before), _payload(after))

    def test_original_uri_is_rejected_even_when_identical(self) -> None:
        before, after = synthetic_rows()
        before[0]["source"]["uri"] = after[0]["source"]["uri"]
        self._reject(_payload(before), _payload(after))

    def test_exact_uri_required(self) -> None:
        for uri in (None, "private://wrong/source", "https://example.invalid/synthetic", True):
            with self.subTest(uri=uri):
                before, after = synthetic_rows()
                after[0]["source"]["uri"] = uri
                self._reject(_payload(before), _payload(after))

    def test_content_and_unrelated_metadata_must_be_unchanged(self) -> None:
        for mutation in ("message", "extra", "deleted", "source", "provenance"):
            with self.subTest(mutation=mutation):
                before, after = synthetic_rows()
                if mutation == "message":
                    after[0]["messages"][0]["content"] = "SYNTHETIC_RAW_TEXT_SENTINEL"
                elif mutation == "extra":
                    after[0]["extra"] = 1
                elif mutation == "deleted":
                    del after[0]["other_metadata"]
                elif mutation == "source":
                    after[0]["source"]["new_field"] = "SYNTHETIC_RAW_TEXT_SENTINEL"
                else:
                    after[0]["provenance"]["human_review_status"] = "changed"
                self._reject(_payload(before), _payload(after))

    def test_bool_int_float_aliases_are_not_equal(self) -> None:
        for field, value in (("count", False), ("flag", 0), ("number", 1)):
            with self.subTest(field=field):
                before, after = synthetic_rows()
                after[0]["other_metadata"][field] = value
                self._reject(_payload(before), _payload(after))

    def test_ids_must_match_order_and_be_unique(self) -> None:
        before, after = synthetic_rows()
        after[0], after[1] = after[1], after[0]
        self._reject(_payload(before), _payload(after))
        before, after = synthetic_rows()
        before[1], after[1] = copy.deepcopy(before[0]), copy.deepcopy(after[0])
        self._reject(_payload(before), _payload(after))

    def test_record_id_and_source_identity_are_validated(self) -> None:
        for value in ("", "bad/id", "line\nbreak", None, True):
            with self.subTest(value=value):
                before, after = synthetic_rows()
                before[0]["record_id"] = value
                self._reject(_payload(before), _payload(after))
        for field in ("source_id", "dataset_id", "document_id"):
            before, after = synthetic_rows()
            before[0]["source"][field] = after[0]["source"][field] = "wrong"
            self._reject(_payload(before), _payload(after))

    def test_exactly_320_records_required_on_both_sides(self) -> None:
        before, after = synthetic_rows()
        for rows in (before[:-1], before + [before[0]]):
            self._reject(_payload(rows), _payload(after))
        for rows in (after[:-1], after + [after[0]]):
            self._reject(_payload(before), _payload(rows))

    def test_line_and_encoding_contract(self) -> None:
        before, after = synthetic_pairs()
        for payload in (
            before[:-1],
            before + b"\n",
            before.replace(b"\n", b"\r\n"),
            b"\xef\xbb\xbf" + before,
            b"\xff" + before,
            b"\n" + before.split(b"\n", 1)[1],
            b" \n" + before.split(b"\n", 1)[1],
        ):
            with self.subTest(prefix=payload[:5]):
                self._reject(payload, after)
                self._reject(before, payload)

    def test_duplicate_keys_nonfinite_numbers_and_nonobjects_rejected(self) -> None:
        before, after = synthetic_pairs()
        tail = before.split(b"\n", 1)[1]
        for line in (
            b'{"a":1,"a":2}',
            b'{"a":{"b":1,"b":2}}',
            b'{"a":NaN}',
            b'{"a":Infinity}',
            b'{"a":-Infinity}',
            b'{"a":1e999}',
            b'{"a":"\\ud800"}',
            b"null",
            b"[]",
            b'"raw"',
            b"{",
        ):
            with self.subTest(line=line):
                self._reject(line + b"\n" + tail, after)

    def test_missing_or_wrong_metadata_containers(self) -> None:
        for key in ("source", "provenance"):
            for value in (None, [], True):
                before, after = synthetic_rows()
                after[0][key] = value
                self._reject(_payload(before), _payload(after))
            before, after = synthetic_rows()
            del before[0][key]
            self._reject(_payload(before), _payload(after))

    def test_only_bytes_inputs_accepted(self) -> None:
        before, after = synthetic_pairs()
        for value in (None, True, before.decode(), bytearray(before)):
            self._reject(value, after)
            self._reject(before, value)


if __name__ == "__main__":
    unittest.main()
