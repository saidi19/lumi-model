from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import lumi.posttraining_massive_bn_bd_acquisition_v1 as contract
from lumi.posttraining_massive_bn_bd_acquisition_v1 import (
    NetworkBudget,
    _validate_initial_url,
    _validate_redirect_url,
    classify_script,
    parse_annotated_utterance,
    sha256_value,
    validate_raw_text_free_report,
)

MASSIVE_TRAIN_URL = (
    "https://huggingface.co/datasets/AmazonScience/massive/resolve/"
    "d1792d0e4f32a74d43725cc1b901f600dad990dd/"
    "bn-BD/train/0000.parquet?download=true"
)
MASSIVE_NOTICE_URL = (
    "https://raw.githubusercontent.com/alexa/massive/"
    "f966f21846043aabef9b0f974fa7970027f43738/NOTICE.md"
)


class MassiveBnBdAcquisitionTests(unittest.TestCase):
    def test_initial_url_gate_is_an_exact_https_allowlist(self) -> None:
        _validate_initial_url(MASSIVE_TRAIN_URL)
        _validate_initial_url(MASSIVE_NOTICE_URL)
        for url in (
            MASSIVE_TRAIN_URL.replace("https://", "http://", 1),
            MASSIVE_TRAIN_URL.replace("/train/", "/test/", 1),
            MASSIVE_TRAIN_URL + "#fragment",
            "https://127.0.0.1/dataset.parquet",
            "https://user@huggingface.co/dataset.parquet",
            "https://example.com/dataset.parquet",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                _validate_initial_url(url)

    def test_redirect_gate_rejects_unsafe_or_unapproved_targets(self) -> None:
        for url in (
            "http://cdn-lfs.huggingface.co/object",
            "https://127.0.0.1/object",
            "https://user@cdn-lfs.huggingface.co/object",
            "https://cdn-lfs.huggingface.co:444/object",
            "https://cdn-lfs.huggingface.co/object#fragment",
            "https://example.com/object",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                _validate_redirect_url(MASSIVE_TRAIN_URL, url)

    def test_network_budget_enforces_response_and_byte_caps(self) -> None:
        budget = NetworkBudget(maximum_responses=2, maximum_bytes=5)
        for _ in range(2):
            budget.start_request()
            budget.observe_response()
        with self.assertRaises(ValueError):
            budget.start_request()
        budget.add_bytes(5)
        with self.assertRaises(ValueError):
            budget.add_bytes(1)

    def test_script_classifier_uses_aggregate_only_buckets(self) -> None:
        self.assertEqual(classify_script("বাংলা বাক্য"), "bengali_only")
        self.assertEqual(classify_script("bangla text"), "latin_only")
        self.assertEqual(classify_script("বাংলা text"), "bengali_latin_mixed")
        self.assertEqual(classify_script("123 !?"), "other")

    def test_annotation_parser_returns_no_slot_names_or_values(self) -> None:
        parsed = parse_annotated_utterance("কাল [date : tomorrow] [time : সকাল আটটা] অ্যালার্ম দাও")
        self.assertIs(parsed["malformed"], False)
        self.assertEqual(parsed["annotation_count"], 2)
        self.assertEqual(len(parsed["slot_label_digests"]), 2)
        self.assertNotIn("tomorrow", repr(parsed))
        malformed = parse_annotated_utterance("ভাঙা [date : tomorrow")
        self.assertIs(malformed["malformed"], True)
        self.assertEqual(malformed["annotation_count"], 0)

    def test_raw_text_free_guard_accepts_counts_and_rejects_payload_fields(self) -> None:
        unsigned = {
            "schema_version": "lumi.synthetic-massive-aggregate.v0.1",
            "status": "PASS_synthetic_nonproduction",
            "counts": {"rows": 3, "slot_parseable": 2},
            "production_eligible": False,
            "raw_text_retained": False,
        }
        report = dict(unsigned, report_sha256=sha256_value(unsigned))
        validate_raw_text_free_report(report)

        for field in ("utt", "annot_utt", "worker_id", "judgments", "raw_text"):
            contaminated = copy.deepcopy(report)
            contaminated[field] = "synthetic payload"
            contaminated["report_sha256"] = sha256_value(
                {key: value for key, value in contaminated.items() if key != "report_sha256"}
            )
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_raw_text_free_report(contaminated)

    def test_internal_seal_is_canonical_and_detects_mutation(self) -> None:
        first = {
            "counts": {"rows": 3},
            "status": "PASS_synthetic",
            "production_eligible": False,
            "raw_text_retained": False,
        }
        reordered = {
            "raw_text_retained": False,
            "production_eligible": False,
            "status": "PASS_synthetic",
            "counts": {"rows": 3},
        }
        self.assertEqual(sha256_value(first), sha256_value(reordered))

        sealed = dict(first, report_sha256=sha256_value(first))
        validate_raw_text_free_report(sealed)
        tampered = copy.deepcopy(sealed)
        tampered["counts"]["rows"] = 4
        with self.assertRaises(ValueError):
            validate_raw_text_free_report(tampered)

    def test_production_report_shape_uses_non_payload_rights_metadata(self) -> None:
        unsigned = {
            "schema_version": contract.REPORT_SCHEMA,
            "status": contract.PASS_STATUS,
            "source": {
                "dataset_id": "AmazonScience/massive",
                "rows": 11514,
                "sha256": "0" * 64,
            },
            "rights_documents": [
                {"document_code": "massive_notice", "bytes": 500, "sha256": "1" * 64}
            ],
            "parquet": {
                "top_level_schema": [
                    {"name": "id", "type": "string", "nullable": True},
                    {"name": "utt", "type": "string", "nullable": True},
                ]
            },
            "decision": {"candidate_rows_constructed": 0},
            "production_eligible": False,
            "raw_text_retained": False,
        }
        report = dict(unsigned, report_sha256=sha256_value(unsigned))
        validate_raw_text_free_report(report)

    def test_synthetic_parquet_is_streamed_as_aggregates_only(self) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ModuleNotFoundError:
            self.skipTest("PyArrow is supplied by the bound local audit runtime")

        table = pa.table(
            {
                "id": ["a", "b", "c"],
                "locale": ["bn-BD"] * 3,
                "partition": ["train"] * 3,
                "scenario": [0, 0, 1],
                "intent": [0, 1, 2],
                "utt": ["বাংলা বাক্য", "বাংলা text", "আরেকটি বাক্য"],
                "annot_utt": [
                    "বাংলা বাক্য",
                    "[date : tomorrow] বাংলা text",
                    "[time : সকাল আটটা] আরেকটি বাক্য",
                ],
                "worker_id": ["w1", "w2", "w3"],
                "slot_method": ["synthetic"] * 3,
                "judgments": ["synthetic"] * 3,
            }
        )
        original_rows = contract.EXPECTED_SOURCE["rows"]
        original_cap = contract.EXPECTED_LIMITS["maximum_source_row_reads"]
        try:
            contract.EXPECTED_SOURCE["rows"] = 3
            contract.EXPECTED_LIMITS["maximum_source_row_reads"] = 3
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "synthetic.parquet"
                pq.write_table(table, path, compression="zstd")
                result = contract.aggregate_parquet(path)
        finally:
            contract.EXPECTED_SOURCE["rows"] = original_rows
            contract.EXPECTED_LIMITS["maximum_source_row_reads"] = original_cap

        self.assertTrue(result["gate_pass"])
        self.assertEqual(result["aggregate_audit"]["source_rows_read"], 3)
        self.assertEqual(result["aggregate_audit"]["intent"]["cardinality"], 3)
        self.assertEqual(result["aggregate_audit"]["annotations"]["total_annotations"], 2)
        self.assertNotIn("worker_id", repr(result["aggregate_audit"]))
        self.assertNotIn("tomorrow", repr(result))


if __name__ == "__main__":
    unittest.main()
