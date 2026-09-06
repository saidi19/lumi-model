"""Source-only unit tests with synthetic rows, registries and temporary bytes.

No historical plans, candidates or owner receipts are read. Projection results
remain in memory and are not evidence of owner review, rights, promotion or data
eligibility. These tests exercise pure transforms and binding rejection, not the
historical private-artifact integration or the full preflight report builder.
"""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from lumi import posttraining_teacher_canary_source_promotion_preflight as preflight

SYNTHETIC_RECEIPT_DIGEST = hashlib.sha256(b"synthetic unit-test marker, not a receipt").hexdigest()


def synthetic_digest(value) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def synthetic_row(mode: str = "english") -> dict:
    return {
        "record_id": "synthetic-fixture-" + mode,
        "messages": [
            {"role": "user", "content": "Synthetic unit-test prompt."},
            {"role": "assistant", "content": "Synthetic unit-test response."},
        ],
        "provenance": {"synthetic_fixture": True, "nested": {"sequence": [1]}},
        "rights_status": "synthetic_fixture_not_rights_evidence",
        "source_id": "synthetic_attempt_fixture",
        "language_mode": mode,
        "split": "train",
        "task": "factual_qa",
    }


def synthetic_predecessor() -> dict:
    return {
        "synthetic_fixture": True,
        "metadata": {"production_eligible": False, "nested": ["synthetic"]},
        "sources": [
            {"source_id": f"synthetic_source_{index}", "dataset_id": f"synthetic/dataset-{index}",
             "metadata": {"synthetic_fixture": True}}
            for index in range(preflight.EXPECTED_PREDECESSOR_SOURCES)
        ],
    }


def synthetic_source_after() -> dict:
    return {
        "source_id": "synthetic_addition",
        "dataset_id": "synthetic/addition",
        "production_eligible": False,
        "metadata": {"synthetic_fixture": True, "labels": ["unit-test-only"]},
    }


class TeacherCanaryPreflightSyntheticTests(unittest.TestCase):
    def test_projection_maps_all_four_language_modes_without_reordering_rows(self):
        expected = {
            "banglish": "multilingual", "bengali_script": "bn", "english": "en",
            "natural_code_switching": "multilingual",
        }
        rows = [synthetic_row(mode) for mode in expected]
        projected = preflight.project_canonical_candidate(
            rows, owner_receipt_file_sha256=SYNTHETIC_RECEIPT_DIGEST,
        )
        self.assertEqual(len(projected), 4)
        for original, canonical, language in zip(rows, projected, expected.values(), strict=True):
            self.assertEqual(canonical["language"], language)
            self.assertEqual(canonical["record_id"], original["record_id"])
            self.assertEqual(canonical["messages"], original["messages"])
            self.assertEqual(canonical["split"], original["split"])
            self.assertEqual(canonical["task_category"], original["task"])
            self.assertEqual(canonical["schema_version"], "lumi.sft.v0.1")
            self.assertEqual(canonical["source"]["source_id"], preflight.SOURCE_ID)
            self.assertEqual(canonical["source"]["dataset_id"], preflight.DATASET_ID)
            self.assertEqual(canonical["source"]["revision"], preflight.SOURCE_REVISION)

    def test_projection_binds_input_content_and_provenance_without_human_authorship(self):
        row = synthetic_row()
        projected = preflight.project_canonical_candidate(
            [row], owner_receipt_file_sha256=SYNTHETIC_RECEIPT_DIGEST,
        )[0]
        provenance = projected["provenance"]
        self.assertTrue(provenance["ai_assisted"])
        self.assertFalse(provenance["human_authored"])
        self.assertFalse(provenance["human_authorship_claimed"])
        self.assertEqual(provenance["content_sha256"], synthetic_digest(row["messages"]))
        self.assertEqual(
            provenance["original_provenance_sha256"], synthetic_digest(row["provenance"]),
        )
        self.assertEqual(provenance["owner_review_receipt_file_sha256"], SYNTHETIC_RECEIPT_DIGEST)
        transform = {
            "contract": "lumi.teacher-canary-canonical-owner-bound.v0.1",
            "original_provenance_sha256": synthetic_digest(row["provenance"]),
            "original_rights_status": row["rights_status"],
            "original_source_id": row["source_id"],
            "owner_review_receipt_file_sha256": SYNTHETIC_RECEIPT_DIGEST,
            "source_revision": preflight.SOURCE_REVISION,
        }
        self.assertEqual(provenance["transform_sha256"], synthetic_digest(transform))

    def test_projection_is_deterministic_and_deep_copies_messages(self):
        rows = [synthetic_row()]
        before = copy.deepcopy(rows)
        first = preflight.project_canonical_candidate(
            rows, owner_receipt_file_sha256=SYNTHETIC_RECEIPT_DIGEST,
        )
        second = preflight.project_canonical_candidate(
            rows, owner_receipt_file_sha256=SYNTHETIC_RECEIPT_DIGEST,
        )
        self.assertEqual(first, second)
        self.assertEqual(rows, before)
        first[0]["messages"][0]["content"] = "Changed synthetic output only."
        self.assertEqual(rows, before)
        self.assertEqual(second[0]["messages"], before[0]["messages"])

    def test_projection_rejects_unsupported_language_mode(self):
        with self.assertRaisesRegex(ValueError, "unsupported teacher-canary language mode"):
            preflight.project_canonical_candidate(
                [synthetic_row("synthetic_unsupported_mode")],
                owner_receipt_file_sha256=SYNTHETIC_RECEIPT_DIGEST,
            )

    def test_successor_adds_one_source_without_mutating_either_input(self):
        predecessor = synthetic_predecessor()
        addition = synthetic_source_after()
        before = copy.deepcopy(predecessor)
        addition_before = copy.deepcopy(addition)
        successor = preflight.build_projected_successor(predecessor, source_after=addition)
        self.assertEqual(len(successor["sources"]), 16)
        self.assertEqual(successor["sources"][:-1], predecessor["sources"])
        self.assertEqual(successor["sources"][-1], addition)
        self.assertEqual(successor["metadata"], predecessor["metadata"])
        self.assertFalse(successor["sources"][-1]["production_eligible"])
        successor["metadata"]["nested"].append("synthetic mutation")
        successor["sources"][0]["metadata"]["synthetic_fixture"] = False
        successor["sources"][-1]["metadata"]["labels"].append("synthetic mutation")
        self.assertEqual(predecessor, before)
        self.assertEqual(addition, addition_before)

    def test_successor_rejects_wrong_predecessor_count_or_container(self):
        valid = synthetic_predecessor()["sources"]
        for sources in (None, {}, (), [], valid[:-1], valid + [synthetic_source_after()]):
            with self.subTest(container=type(sources).__name__):
                with self.assertRaisesRegex(ValueError, "predecessor source count changed"):
                    preflight.build_projected_successor(
                        {"sources": sources}, source_after=synthetic_source_after(),
                    )

    def test_successor_rejects_either_existing_source_or_dataset_identity(self):
        for field, value in (("source_id", preflight.SOURCE_ID),
                             ("dataset_id", preflight.DATASET_ID)):
            predecessor = synthetic_predecessor()
            predecessor["sources"][0][field] = value
            before = copy.deepcopy(predecessor)
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "source already exists"):
                    preflight.build_projected_successor(
                        predecessor, source_after=synthetic_source_after(),
                    )
                self.assertEqual(predecessor, before)

    def test_binding_accepts_exact_synthetic_bytes_without_mutating_them(self):
        with tempfile.TemporaryDirectory(prefix="lumi-synthetic-binding-") as directory:
            root = Path(directory)
            raw = b"synthetic binding fixture; no source rows or owner receipt\n"
            path = root / "synthetic.txt"
            path.write_bytes(raw)
            result = preflight.resolve_binding(
                root, {"path": path.name, "sha256": hashlib.sha256(raw).hexdigest()},
                name="synthetic_fixture",
            )
            self.assertEqual(result, path.resolve())
            self.assertEqual(path.read_bytes(), raw)

    def test_binding_rejects_changed_bytes_and_wrong_sha256(self):
        with tempfile.TemporaryDirectory(prefix="lumi-synthetic-binding-") as directory:
            root = Path(directory)
            path = root / "synthetic.txt"
            original = b"synthetic original bytes"
            path.write_bytes(original)
            for digest, content in (
                ("0" * 64, original),
                (hashlib.sha256(original).hexdigest(), b"synthetic changed bytes"),
            ):
                path.write_bytes(content)
                with self.subTest(changed_content=content != original):
                    with self.assertRaisesRegex(ValueError, "binding changed"):
                        preflight.resolve_binding(
                            root, {"path": path.name, "sha256": digest}, name="synthetic_fixture",
                        )

    def test_binding_rejects_invalid_digest_before_reading_any_file(self):
        with tempfile.TemporaryDirectory(prefix="lumi-synthetic-binding-") as directory:
            for digest in (None, 0, "", "A" * 64, "a" * 63, "g" * 64):
                with self.subTest(digest_type=type(digest).__name__):
                    with self.assertRaisesRegex(ValueError, "must be a lowercase SHA-256"):
                        preflight.resolve_binding(
                            directory, {"path": "synthetic-missing.txt", "sha256": digest},
                            name="synthetic_fixture",
                        )

    def test_binding_rejects_missing_nonfile_and_outside_boundary_paths(self):
        with (
            tempfile.TemporaryDirectory(prefix="lumi-synthetic-binding-") as directory,
            tempfile.TemporaryDirectory(prefix="lumi-synthetic-outside-") as outside,
        ):
            root = Path(directory) / "repository"
            root.mkdir()
            outside_file = Path(outside) / "synthetic.txt"
            outside_file.write_bytes(b"synthetic outside-boundary fixture")
            digest = hashlib.sha256(outside_file.read_bytes()).hexdigest()
            for path in ("synthetic-missing.txt", ".", str(outside_file)):
                with self.subTest(path_kind="absolute" if Path(path).is_absolute() else "relative"):
                    with self.assertRaisesRegex(ValueError, "binding missing"):
                        preflight.resolve_binding(
                            root, {"path": path, "sha256": digest}, name="synthetic_fixture",
                        )


if __name__ == "__main__":
    unittest.main()
