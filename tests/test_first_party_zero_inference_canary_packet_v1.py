from __future__ import annotations

import copy
import json
import unittest

from lumi.posttraining_first_party_zero_inference_canary_packet_v1 import (
    EXPECTED_ALLOCATIONS,
    REPORT_SCHEMA,
    REPORT_STATUS,
    allocation_totals,
    build_blank_rows,
    build_manifest,
    build_packets,
    sha256_value,
    validate_blank_rows,
    validate_manifest,
    validate_report,
)


EXPECTED_TOTALS = allocation_totals(EXPECTED_ALLOCATIONS)


def _plan() -> dict:
    return {
        "allocations": list(EXPECTED_ALLOCATIONS),
        "expected_totals": EXPECTED_TOTALS,
    }


def _sealed_report() -> dict:
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": REPORT_STATUS,
        "packet_summary": {
            "blank_template_only": True,
            "packet_count": 50,
            "records": 1000,
            "train_records": 800,
            "validation_records": 200,
        },
        "next_gate": {
            "action": "real_human_identity_assignment_terms_and_private_packet_completion"
        },
        "resource_accounting": {
            "candidate_reads": 0,
            "candidate_writes": 0,
            "gpu_actions": 0,
            "inference_calls": 0,
            "network_bytes": 0,
            "registry_writes": 0,
            "source_line_reads": 0,
            "test_object_reads": 0,
            "training_actions": 0,
        },
        "candidate_construction_started": False,
        "legal_conclusion_made": False,
        "production_eligible": False,
        "raw_training_text_retained": False,
    }
    report["report_sha256"] = sha256_value(report)
    return report


class FirstPartyZeroInferenceCanaryPacketTests(unittest.TestCase):
    def test_allocation_totals_are_exact(self) -> None:
        self.assertEqual(EXPECTED_TOTALS["records"], 1000)
        self.assertEqual(
            EXPECTED_TOTALS["split_records"], {"train": 800, "validation": 200}
        )
        self.assertEqual(
            EXPECTED_TOTALS["lane_records"],
            {"deterministic_grounded": 250, "human_authored": 750},
        )
        self.assertEqual(
            EXPECTED_TOTALS["language_records"],
            {"banglish": 666, "natural_code_switching": 334},
        )
        self.assertEqual(len(EXPECTED_TOTALS["task_language_cell_records"]), 12)

    def test_builds_exact_blank_rows(self) -> None:
        rows = build_blank_rows(_plan())
        self.assertEqual(len(rows), 1000)
        self.assertEqual(len({row["record_id"] for row in rows}), 1000)
        self.assertTrue(all(row["messages"] == [] for row in rows))
        self.assertTrue(all(row["task_evidence"] == {} for row in rows))
        self.assertTrue(all(row["template_family_id"] == "" for row in rows))

    def test_packet_balance_and_placeholder_independence(self) -> None:
        packets = build_packets(build_blank_rows(_plan()), plan=_plan())
        self.assertEqual(len(packets), 50)
        for packet in packets:
            self.assertEqual(len(packet["rows"]), 20)
            self.assertEqual(
                sum(row["split"] == "train" for row in packet["rows"]), 16
            )
            self.assertEqual(
                sum(row["split"] == "validation" for row in packet["rows"]), 4
            )
            self.assertNotEqual(
                packet["author_slot"].split("-")[-1],
                packet["reviewer_slot"].split("-")[-1],
            )

    def test_manifest_is_raw_text_free_and_sealed(self) -> None:
        plan = _plan()
        manifest = build_manifest(build_packets(build_blank_rows(plan), plan=plan), plan=plan)
        self.assertNotIn("messages", json.dumps(manifest))
        self.assertNotIn("content", json.dumps(manifest))
        validate_manifest(manifest)

    def test_nonblank_row_is_rejected(self) -> None:
        rows = build_blank_rows(_plan())
        rows[0]["messages"] = [{"role": "user", "content": "not blank"}]
        with self.assertRaises(ValueError):
            validate_blank_rows(rows, expected_totals=EXPECTED_TOTALS)

    def test_manifest_rejects_hash_or_slot_drift(self) -> None:
        plan = _plan()
        manifest = build_manifest(build_packets(build_blank_rows(plan), plan=plan), plan=plan)
        changed = copy.deepcopy(manifest)
        changed["packets"][0]["author_slot"] = "author-slot-99"
        changed["manifest_sha256"] = sha256_value(
            {key: value for key, value in changed.items() if key != "manifest_sha256"}
        )
        with self.assertRaises(ValueError):
            validate_manifest(changed)

    def test_report_rejects_candidate_or_production_drift(self) -> None:
        validate_report(_sealed_report())
        for path, value in (
            (("candidate_construction_started",), True),
            (("production_eligible",), True),
            (("resource_accounting", "candidate_writes"), 1),
            (("resource_accounting", "inference_calls"), 1),
        ):
            with self.subTest(path=path):
                report = copy.deepcopy(_sealed_report())
                target = report
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                report["report_sha256"] = sha256_value(
                    {key: item for key, item in report.items() if key != "report_sha256"}
                )
                with self.assertRaises(ValueError):
                    validate_report(report)


if __name__ == "__main__":
    unittest.main()
