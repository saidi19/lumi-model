"""No-skip inventory guards; no CI log, network or private fixture access."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "unresolved_inventory_guard", ROOT / "scripts/check_ci_unresolved_inventory_v1.py"
)
inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory)


class UnresolvedInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = inventory.load_inventory(ROOT)

    def changed(self) -> dict:
        return copy.deepcopy(self.manifest)

    def rejected(self, value: dict) -> None:
        with self.assertRaises(inventory.InventoryError):
            inventory.validate_inventory(value)

    def test_exact_inventory_and_all_source_hashes(self) -> None:
        result = inventory.check(ROOT)
        self.assertEqual(result["open_hosted_outcomes"], 327)
        self.assertEqual(result["unchanged_test_files"], 123)
        self.assertIs(result["hosted_ci_passed"], False)
        self.assertEqual(result["tests_executed"], 0)

    def test_missing_outcome_is_rejected(self) -> None:
        value = self.changed()
        value["entries"].pop()
        self.rejected(value)

    def test_duplicate_outcome_is_rejected(self) -> None:
        value = self.changed()
        value["entries"][0] = copy.deepcopy(value["entries"][1])
        self.rejected(value)

    def test_reordering_is_rejected(self) -> None:
        value = self.changed()
        value["entries"].reverse()
        self.rejected(value)

    def test_cannot_close_or_skip_an_original_hosted_failure(self) -> None:
        for status in ("PASS", "SKIP", "XFAIL", "CLOSED", True):
            value = self.changed()
            value["entries"][0]["hosted_status"] = status
            with self.subTest(status=status):
                self.rejected(value)

    def test_local_pass_cannot_reduce_hosted_failure_totals(self) -> None:
        value = self.changed()
        value["totals"]["failed"] -= 4
        self.rejected(value)

    def test_source_run_change_is_rejected(self) -> None:
        for key, replacement in (("head_sha", "0" * 40), ("run_id", 1)):
            value = self.changed()
            value["source_run"][key] = replacement
            with self.subTest(key=key):
                self.rejected(value)

    def test_source_inventory_cannot_lose_a_file(self) -> None:
        value = self.changed()
        value["test_files"].pop(next(iter(value["test_files"])))
        self.rejected(value)

    def test_scope_cannot_be_used_as_execution_authorization(self) -> None:
        for key in (
            "skip_or_xfail_authorized", "private_inputs_included",
            "historical_replay_authorized", "production_eligible",
        ):
            value = self.changed()
            value["scope"][key] = True
            with self.subTest(key=key):
                self.rejected(value)

    def test_bool_integer_coercion_cannot_change_scope_or_totals(self) -> None:
        for field, key, replacement in (
            ("scope", "production_eligible", 0),
            ("scope", "inventory_only", 1),
            ("totals", "failed", 306.0),
        ):
            value = self.changed()
            value[field][key] = replacement
            with self.subTest(key=key):
                self.rejected(value)

    def test_unknown_lane_and_outcome_are_rejected(self) -> None:
        for key, replacement in (("review_lane", "ignore"), ("outcome", "PASSED")):
            value = self.changed()
            value["entries"][0][key] = replacement
            with self.subTest(key=key):
                self.rejected(value)

    def test_lane_counts_cannot_be_relabelled_as_completed(self) -> None:
        value = self.changed()
        value["entries"][0]["review_lane"] = "local_vocab_metadata_unit"
        self.rejected(value)

    def test_no_exception_or_private_text_field_is_accepted(self) -> None:
        for key in ("traceback", "message", "private_fixture", "source_text"):
            value = self.changed()
            value["entries"][0][key] = "synthetic text"
            with self.subTest(key=key):
                self.rejected(value)

    def test_path_escape_is_rejected(self) -> None:
        for node in ("../tests/test_a.py::test_a", "tests/../private/a.py::test_a"):
            value = self.changed()
            value["entries"][0]["nodeid"] = node
            with self.subTest(node=node):
                self.rejected(value)

    def test_malformed_source_hash_is_rejected(self) -> None:
        value = self.changed()
        value["test_files"][next(iter(value["test_files"]))]["sha256"] = "not-a-hash"
        self.rejected(value)

    def test_duplicate_json_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(inventory.InventoryError, "duplicate_json_key"):
            json.loads('{"a":1,"a":2}', object_pairs_hook=inventory.no_duplicates)

    def test_same_shape_renamed_node_cannot_bypass_manifest_hash(self) -> None:
        value = self.changed()
        value["entries"][0]["nodeid"] += "_renamed"
        payload = json.dumps(value).encode()
        with patch.object(inventory, "read_fixed_file", return_value=payload):
            with self.assertRaisesRegex(inventory.InventoryError, "inventory_hash"):
                inventory.load_inventory(ROOT)

    def test_changed_source_bytes_cannot_pass(self) -> None:
        with patch.object(inventory, "load_inventory", return_value=self.manifest):
            with patch.object(inventory, "read_fixed_file", return_value=b"changed"):
                with self.assertRaisesRegex(inventory.InventoryError, "source_changed"):
                    inventory.check(ROOT)

    def test_symlink_is_rejected_before_content_read(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lumi-inventory-guard-") as directory:
            base = Path(directory)
            (base / "link").symlink_to(base / "missing-target")
            with self.assertRaisesRegex(inventory.InventoryError, "symlink"):
                inventory.read_fixed_file(base, "link", 64)

    def test_directory_and_multiple_links_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lumi-inventory-shape-") as directory:
            base = Path(directory)
            with self.assertRaisesRegex(inventory.InventoryError, "file_shape"):
                inventory.read_fixed_file(base, ".", 64)
            (base / "source").write_bytes(b"synthetic")
            os.link(base / "source", base / "second")
            with self.assertRaisesRegex(inventory.InventoryError, "file_shape"):
                inventory.read_fixed_file(base, "source", 64)

    def test_leaf_swapped_to_symlink_at_open_cannot_be_read(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lumi-inventory-race-") as directory:
            base = Path(directory)
            (base / "source").write_bytes(b"synthetic")
            (base / "outside").write_bytes(b"synthetic outside must not be read")
            real_open = os.open

            def swapped_open(path, flags, *args, **kwargs):
                if path == "source":
                    (base / "source").rename(base / "preserved-source")
                    (base / "source").symlink_to(base / "outside")
                return real_open(path, flags, *args, **kwargs)

            with (
                patch.object(inventory.os, "open", side_effect=swapped_open),
                patch.object(inventory.os, "read", side_effect=AssertionError("unexpected read")),
                self.assertRaisesRegex(inventory.InventoryError, "unreadable_or_symlink"),
            ):
                inventory.read_fixed_file(base, "source", 64)

    def test_parent_symlink_cannot_redirect_read(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lumi-inventory-parent-") as directory:
            base = Path(directory)
            (base / "outside").mkdir()
            (base / "outside/value").write_bytes(b"synthetic")
            (base / "link").symlink_to(base / "outside")
            with (
                patch.object(inventory.os, "read", side_effect=AssertionError("unexpected read")),
                self.assertRaisesRegex(inventory.InventoryError, "unreadable_or_symlink"),
            ):
                inventory.read_fixed_file(base, "link/value", 64)


if __name__ == "__main__":
    unittest.main()
