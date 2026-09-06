"""In-memory validator coverage, not a dataset seal or historical evidence replay.

The plan deliberately has synthetic source identities and is not an operational
load_plan input. Exported aggregate constants define validator boundaries only;
no corresponding rows, private files, approvals or output artifacts are created.
The original metadata-chain and materializer tests remain unchanged.
"""

from __future__ import annotations

import copy
import unittest
from pathlib import Path
from unittest.mock import patch

from lumi import posttraining_learning_candidate_seal_v3 as seal

FIXTURE_KIND = "synthetic_validator_only_not_operational_evidence"


def digest(label: str) -> str:
    return seal.sha256_value([FIXTURE_KIND, label])


def partition(total: int, count: int, index: int) -> int:
    quotient, remainder = divmod(total, count)
    return quotient + int(index < remainder)


def synthetic_plan() -> dict:
    count = seal.EXPECTED_INPUT["candidate_source_files"]
    sources = []
    for index in range(count):
        splits = {
            split: {
                field: partition(seal.EXPECTED_INPUT[f"{split}_{field}"], count, index)
                for field in ("records", "serialized_tokens", "supervised_tokens")
            }
            for split in ("train", "validation")
        }
        sources.append({
            "source_id": f"synthetic-seal-source-{index:02d}",
            "path": f"synthetic-unused-{index:02d}.jsonl",
            "records": sum(value["records"] for value in splits.values()),
            "bytes": partition(seal.EXPECTED_INPUT["candidate_input_bytes"], count, index),
            "candidate_sha256": digest(f"unused-candidate-{index}"),
            "by_split": splits,
        })
    return {"fixture_kind": FIXTURE_KIND, "sources": sources}


def reseal(value: dict, field: str) -> None:
    value.pop(field, None)
    value[field] = seal.sha256_value(value)


def synthetic_manifest(plan: dict) -> dict:
    value = {
        "authorization_boundary": copy.deepcopy(seal.EXPECTED_MANIFEST_AUTHORIZATION_BOUNDARY),
        "bindings": copy.deepcopy(seal.EXPECTED_MANIFEST_BINDINGS),
        "contract_boundary": copy.deepcopy(seal.EXPECTED_CONTRACT_BOUNDARY),
        "hygiene": copy.deepcopy(seal.EXPECTED_HYGIENE),
        "input": copy.deepcopy(seal.EXPECTED_INPUT),
        "next_gate": copy.deepcopy(seal.EXPECTED_MANIFEST_NEXT_GATE),
        "observed_at": seal.EXPECTED_OBSERVED_AT,
        "ordering_contract": "source_id_lexical_then_existing_file_order_v0.1",
        "output": {
            "candidate_input_bytes": seal.EXPECTED_INPUT["candidate_input_bytes"],
            "candidate_row_reads": seal.EXPECTED_INPUT["records"],
            "input_record_id_sequence_sha256": digest("unused-record-sequence"),
            "source_metrics": {
                source["source_id"]: {
                    "by_split": copy.deepcopy(source["by_split"]),
                    "candidate_bytes": source["bytes"],
                    "candidate_sha256": source["candidate_sha256"],
                    "records": source["records"],
                }
                for source in plan["sources"]
            },
            "splits": {
                split: {
                    "bytes": 1,
                    "record_id_sequence_sha256": digest(f"unused-{split}-sequence"),
                    "sha256": digest(f"unused-{split}-output"),
                    **{
                        field: seal.EXPECTED_INPUT[f"{split}_{field}"]
                        for field in ("records", "serialized_tokens", "supervised_tokens")
                    },
                }
                for split in ("train", "validation")
            },
            "unique_record_ids": seal.EXPECTED_INPUT["records"],
        },
        "plan_file_sha256": digest("not-a-plan-file"),
        "production_blockers_preserved": list(seal.EXPECTED_PRODUCTION_BLOCKERS),
        "production_eligible": False,
        "raw_training_text_retained_in_manifest": False,
        "resource_accounting": copy.deepcopy(seal.EXPECTED_PASS_ONE_RESOURCES),
        "schema_version": seal.MANIFEST_SCHEMA_VERSION,
        "status": seal.MANIFEST_STATUS,
        "token_accounting_contract": (
            "exact_candidate_sha256_plus_frozen_v0_18_per_source_token_ledger"
        ),
    }
    reseal(value, "seal_sha256")
    return value


def synthetic_audit(plan: dict, manifest: dict) -> dict:
    return {
        "candidate_files_opened": 0,
        "output_audit_row_reads": seal.EXPECTED_INPUT["records"],
        "source_split_records": {
            source["source_id"]: {
                split: source["by_split"][split]["records"] for split in ("train", "validation")
            }
            for source in plan["sources"]
        },
        "splits": {
            split: {
                field: manifest["output"]["splits"][split][field]
                for field in ("bytes", "record_id_sequence_sha256", "records", "sha256")
            }
            for split in ("train", "validation")
        },
        "unique_record_ids": seal.EXPECTED_INPUT["records"],
    }


def assign(value: dict, path: tuple[str, ...], replacement: object) -> None:
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = replacement


class LearningCandidateSealSyntheticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.enterContext(patch.object(Path, "open", side_effect=AssertionError("file access")))
        for name in ("load_plan", "materialize_sources", "audit_outputs", "start_marker"):
            self.enterContext(
                patch.object(seal, name, side_effect=AssertionError("operational call"))
            )
        self.plan = synthetic_plan()
        self.manifest = synthetic_manifest(self.plan)

    def verification(self) -> dict:
        return copy.deepcopy(seal.build_verification(
            plan=self.plan,
            manifest=self.manifest,
            manifest_file_sha256=digest("not-a-manifest-file"),
            audit=synthetic_audit(self.plan, self.manifest),
        ))

    def rejects_manifest(self, path: tuple[str, ...], replacement: object) -> None:
        changed = copy.deepcopy(self.manifest)
        assign(changed, path, replacement)
        reseal(changed, "seal_sha256")
        with self.assertRaises(ValueError):
            seal.validate_manifest(changed, plan=self.plan)

    def rejects_verification(self, path: tuple[str, ...], replacement: object) -> None:
        changed = self.verification()
        assign(changed, path, replacement)
        reseal(changed, "verification_sha256")
        with self.assertRaises(ValueError):
            seal.validate_verification(changed, plan=self.plan, manifest=self.manifest)

    def test_in_memory_fixture_validates_without_operational_calls(self) -> None:
        self.assertEqual(self.plan["fixture_kind"], FIXTURE_KIND)
        self.assertIsNone(seal.validate_manifest(self.manifest, plan=self.plan))
        self.assertIs(self.manifest["production_eligible"], False)
        self.assertNotIn("schema_version", self.plan)

    def test_synthetic_source_partition_preserves_declared_aggregate_totals(self) -> None:
        sources = self.plan["sources"]
        self.assertEqual(len(sources), 20)
        self.assertTrue(all(source["source_id"].startswith("synthetic-") for source in sources))
        self.assertEqual(sum(s["records"] for s in sources), 329_200)
        self.assertEqual(sum(s["bytes"] for s in sources), 529_207_367)
        for split in ("train", "validation"):
            for field in ("records", "serialized_tokens", "supervised_tokens"):
                self.assertEqual(
                    sum(s["by_split"][split][field] for s in sources),
                    seal.EXPECTED_INPUT[f"{split}_{field}"],
                )

    def test_resealed_original_ten_boundary_drifts_remain_rejected(self) -> None:
        for path, value in (
            (("input", "records"), 329_199),
            (("output", "unique_record_ids"), 329_199),
            (("output", "splits", "train", "serialized_tokens"), 33_596_488),
            (("output", "splits", "validation", "supervised_tokens"), 939_292),
            (("authorization_boundary", "production_dataset_seal"), True),
            (("contract_boundary", "production_sft_allowed"), True),
            (("bindings", "source_registry_sha256"), digest("wrong-registry")),
            (("next_gate", "candidate_files_reopened"), True),
            (("resource_accounting", "output_audit_row_reads"), 1),
            (("production_eligible",), True),
        ):
            with self.subTest(path=path):
                self.rejects_manifest(path, value)

    def test_each_manifest_resource_counter_is_exact(self) -> None:
        for name, expected in seal.EXPECTED_PASS_ONE_RESOURCES.items():
            with self.subTest(name=name):
                self.rejects_manifest(("resource_accounting", name), expected + 1)
        self.rejects_manifest(("resource_accounting", "full_passes"), True)

    def test_manifest_schema_and_unresealed_tampering_are_rejected(self) -> None:
        self.rejects_manifest(("unexpected_field",), FIXTURE_KIND)
        self.manifest["plan_file_sha256"] = digest("unresealed-change")
        with self.assertRaisesRegex(ValueError, "manifest boundary changed"):
            seal.validate_manifest(self.manifest, plan=self.plan)

    def test_malformed_plan_and_split_hashes_are_rejected(self) -> None:
        for path in (
            ("plan_file_sha256",),
            ("output", "input_record_id_sequence_sha256"),
            ("output", "splits", "train", "sha256"),
            ("output", "splits", "validation", "record_id_sequence_sha256"),
        ):
            for value in ("", "a" * 63, "G" * 64, None):
                with self.subTest(path=path, value=value):
                    self.rejects_manifest(path, value)

    def test_split_bytes_must_be_strict_positive_integers(self) -> None:
        for split in ("train", "validation"):
            for value in (0, -1, True, 1.5, "1"):
                with self.subTest(split=split, value=value):
                    self.rejects_manifest(("output", "splits", split, "bytes"), value)

    def test_source_ledgers_cannot_drift_from_supplied_synthetic_plan(self) -> None:
        identifier = self.plan["sources"][0]["source_id"]
        for field, replacement in (("records", 1), ("candidate_sha256", digest("wrong-source"))):
            with self.subTest(field=field):
                self.rejects_manifest(("output", "source_metrics", identifier, field), replacement)
        self.rejects_manifest(("output", "source_metrics"), {})

    def test_verification_links_exact_manifest_and_retains_nonproduction_boundary(self) -> None:
        verification = self.verification()
        self.assertIsNone(seal.validate_verification(
            verification, plan=self.plan, manifest=self.manifest
        ))
        self.assertEqual(verification["manifest_internal_sha256"], self.manifest["seal_sha256"])
        self.assertIs(verification["production_eligible"], False)

    def test_verification_rejects_reopened_candidates_and_audit_drift(self) -> None:
        for path, value in (
            (("observed_outputs", "candidate_files_opened"), 1),
            (("observed_outputs", "candidate_files_opened"), False),
            (("observed_outputs", "output_audit_row_reads"), 329_199),
            (("observed_outputs", "source_split_records"), {}),
            (("observed_outputs", "unique_record_ids"), 329_201),
            (("observed_outputs", "splits", "train", "sha256"), digest("wrong-output")),
        ):
            with self.subTest(path=path, value=value):
                self.rejects_verification(path, value)

    def test_every_verification_check_must_be_literal_true(self) -> None:
        for name in seal.EXPECTED_VERIFICATION_CHECKS:
            for value in (False, 1):
                with self.subTest(name=name, value=value):
                    self.rejects_verification(("checks", name), value)

    def test_each_verification_resource_counter_is_exact(self) -> None:
        for name, expected in seal.EXPECTED_PASS_TWO_RESOURCES.items():
            with self.subTest(name=name):
                self.rejects_verification(("resource_accounting", name), expected + 1)

    def test_verification_rejects_resealed_manifest_link_or_production_drift(self) -> None:
        for path, value in (
            (("manifest_internal_sha256",), digest("another-manifest")),
            (("production_eligible",), True),
            (("raw_training_text_retained",), True),
            (("next_gate", "production_sft"), True),
        ):
            with self.subTest(path=path):
                self.rejects_verification(path, value)

    def test_verification_requires_valid_seal_and_exact_field_inventory(self) -> None:
        verification = self.verification()
        verification["manifest_file_sha256"] = digest("unresealed-change")
        with self.assertRaisesRegex(ValueError, "verification boundary changed"):
            seal.validate_verification(verification, plan=self.plan, manifest=self.manifest)
        self.rejects_verification(("unexpected_field",), FIXTURE_KIND)

    def test_raw_text_keys_are_rejected_without_any_dataset_access(self) -> None:
        for key in ("messages", "content", "prompt", "record_id", "raw_text"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "raw candidate key"):
                seal._assert_raw_text_free({"synthetic_nested": [{key: "synthetic sentinel"}]})


if __name__ == "__main__":
    unittest.main()
