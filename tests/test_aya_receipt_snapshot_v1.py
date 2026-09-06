"""Tracked original source plus generated receipts; never read real Aya evidence."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from lumi import posttraining_aya_receipt_snapshot_v1 as projection

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_ROOT = "/synthetic/workspace/repo"
ERROR = "invalid Aya receipt source projection"


def _pin(payloads: dict, bindings: dict, role: str) -> None:
    bindings[role]["bytes"] = len(payloads[role])
    bindings[role]["sha256"] = hashlib.sha256(payloads[role]).hexdigest()


def _receipt(payloads: dict, bindings: dict, value: dict, *, reseal: bool = True) -> None:
    value = copy.deepcopy(value)
    if reseal:
        unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
        value["receipt_sha256"] = hashlib.sha256(json.dumps(
            unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
    payloads["receipt"] = json.dumps(value, sort_keys=True).encode()
    bindings["receipt"]["internal_sha256"] = value["receipt_sha256"]
    _pin(payloads, bindings, "receipt")


def synthetic_inputs() -> tuple[dict, dict]:
    """Read only the three fixed tracked Python sources; construct all receipt bytes."""
    payloads = {
        role: (ROOT / path).read_bytes()
        for role, path in projection.SOURCE_PATHS.items() if role != "receipt"
    }
    bindings = {role: {"path": path} for role, path in projection.SOURCE_PATHS.items()}
    for role in payloads:
        _pin(payloads, bindings, role)
    value = {
        "accepted_records": 32,
        "all_sampled_rows_accepted": True,
        "authorization": {
            **dict.fromkeys(projection._AUTHORIZATION_FLAGS, False),
            "purpose": "private_human_owner_quality_spot_check_only",
        },
        "bindings": {
            "decision_file_sha256": "1" * 64,
            "materialization_report_internal_sha256": "2" * 64,
            "plan_file_sha256": "3" * 64,
            "sample_record_ids_sha256": "4" * 64,
        },
        "owner_review_claim": "synthetic_fixture_not_a_human_review",
        "production_eligible": False,
        "production_promotions": 0,
        "raw_training_text_retained": False,
        "rejected_record_ids": [],
        "rejected_records": 0,
        "remaining_blockers": ["synthetic_fixture_non_authorizing"],
        "reviewed_at": "2026-09-06T00:00:00Z",
        "reviewer_alias": "synthetic-owner-fixture",
        "sample_records": 32,
        "schema_version": "lumi.sft-aya-bengali-owner-review-receipt.v0.1",
        "status": "owner_quality_sample_passed_source_promotion_still_separate",
    }
    _receipt(payloads, bindings, value)
    return payloads, bindings


class AyaReceiptSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.payloads, self.bindings = synthetic_inputs()

    def build(self):
        return projection.build_test_module(self.payloads, self.bindings, SYNTHETIC_ROOT)

    def reject(self):
        with self.assertRaisesRegex(ValueError, f"^{ERROR}$"):
            self.build()

    def test_exact_original_test_passes_once_with_synthetic_receipt(self):
        module, snapshot = self.build()
        function = getattr(module, projection.TEST_NAME)
        original = next(node for node in ast.parse(self.payloads["test_source"]).body
                        if isinstance(node, ast.FunctionDef) and node.name == projection.TEST_NAME)
        self.assertEqual(function.__code__.co_filename, module.__file__)
        self.assertEqual(function.__code__.co_firstlineno, original.lineno)
        self.assertEqual(module.__name__, projection.MODULE_NAME)
        self.assertEqual({name for name in vars(module) if name.startswith("test_")},
                         {projection.TEST_NAME})
        self.assertNotIn("pytest", vars(module))
        self.assertEqual(snapshot.summary()["virtual_opens"], 0)
        self.assertIsNone(function())
        summary = snapshot.summary()
        self.assertEqual(summary["bound_blob_count"], 1)
        self.assertEqual(summary["virtual_opens"], 1)
        self.assertEqual(summary["logical_read_bytes_debited"], len(self.payloads["receipt"]))
        self.assertEqual(summary["maximum_logical_read_bytes"], len(self.payloads["receipt"]))
        self.assertIs(summary["filesystem_access_performed"], False)
        self.assertIs(summary["execution_permitted"], False)
        with self.assertRaisesRegex(ValueError, "^snapshot_open_limit_exceeded$"):
            function()

    def test_original_source_ast_is_preserved(self):
        captured = []
        compile_original = projection._compile

        def capture(nodes, filename, namespace):
            captured.extend(nodes)
            compile_original(nodes, filename, namespace)

        with patch.object(projection, "_compile", side_effect=capture):
            self.build()
        actual = {node.name: ast.dump(node, include_attributes=True)
                  for node in captured if isinstance(node, ast.FunctionDef)}
        expected_names = {"_require_sha256", "validate_owner_review_receipt",
                          "canonical_json_bytes", "sha256_value", projection.TEST_NAME}
        self.assertEqual(set(actual), expected_names)
        names_by_role = {
            "review_implementation": {"_require_sha256", "validate_owner_review_receipt"},
            "hash_helper_source": {"canonical_json_bytes", "sha256_value"},
            "test_source": {projection.TEST_NAME},
        }
        for role, names in names_by_role.items():
            for node in ast.parse(self.payloads[role]).body:
                if isinstance(node, ast.FunctionDef) and node.name in names:
                    self.assertEqual(actual[node.name], ast.dump(node, include_attributes=True))

    def test_unselected_imports_and_operational_functions_never_execute(self):
        poison = (b"\nimport nonexistent_receipt_dependency\nraise RuntimeError('unselected')\n"
                  b"def unused_poison(value=1 / 0):\n    raise RuntimeError('unused')\n")
        for role in ("review_implementation", "hash_helper_source", "test_source"):
            self.payloads[role] += poison
            _pin(self.payloads, self.bindings, role)
        module, _ = self.build()
        getattr(module, projection.TEST_NAME)()
        for name in ("unused_poison", "load_exact_owner_sample", "validate_owner_decisions",
                     "load_owner_review_plan", "deterministic_owner_sample", "file_sha256"):
            self.assertNotIn(name, vars(module))

    def test_no_real_file_access_during_projection_or_test(self):
        with (
            patch("builtins.open", side_effect=AssertionError("real open")),
            patch("io.open", side_effect=AssertionError("real open")),
            patch("os.open", side_effect=AssertionError("real open")),
            patch("os.stat", side_effect=AssertionError("real stat")),
            patch("os.lstat", side_effect=AssertionError("real stat")),
        ):
            module, _ = self.build()
            getattr(module, projection.TEST_NAME)()

    def test_existing_module_registration_is_untouched(self):
        sentinel = types.ModuleType(projection.MODULE_NAME)
        with patch.dict(sys.modules, {projection.MODULE_NAME: sentinel}):
            module, _ = self.build()
            self.assertIsNot(module, sentinel)
            self.assertIs(sys.modules[projection.MODULE_NAME], sentinel)

    def test_authentication_precedes_json_and_ast_parsing(self):
        self.payloads["test_source"] += b"\n"
        with (
            patch.object(
                projection.json, "loads", side_effect=AssertionError("early JSON")
            ) as loads,
            patch("ast.parse", side_effect=AssertionError("early AST")) as parse,
        ):
            self.reject()
            loads.assert_not_called()
            parse.assert_not_called()

    def test_every_role_digest_and_byte_count_are_authenticated(self):
        original = copy.deepcopy(self.bindings)
        for role in projection.INPUT_ROLES:
            for field, value in (("sha256", "0" * 64), ("bytes", 1), ("bytes", True)):
                with self.subTest(role=role, field=field):
                    self.bindings = copy.deepcopy(original)
                    self.bindings[role][field] = value
                    self.reject()

    def test_extra_missing_roles_and_binding_fields_are_rejected(self):
        for target in ("payloads", "bindings"):
            for extra in (True, False):
                with self.subTest(target=target, extra=extra):
                    self.payloads, self.bindings = synthetic_inputs()
                    value = getattr(self, target)
                    if extra:
                        value["unexpected"] = value["receipt"]
                    else:
                        del value["receipt"]
                    self.reject()
        self.payloads, self.bindings = synthetic_inputs()
        self.bindings["test_source"]["git_revision"] = "1" * 40
        self.reject()

    def test_paths_and_root_namespace_are_exact(self):
        for role in projection.INPUT_ROLES:
            original = self.bindings[role]["path"]
            self.bindings[role]["path"] = "unexpected/../" + original
            self.reject()
            self.bindings[role]["path"] = original
        for root in (None, 1, "relative", "/", "//repo", "/repo/", "/repo/../other",
                     "/repo//other", "/repo\n", "/repo\\other", "/~user/repo", "/" + "x" * 4096):
            with self.subTest(root=root), self.assertRaisesRegex(ValueError, f"^{ERROR}$"):
                projection.build_test_module(self.payloads, self.bindings, root)

    def test_size_caps_and_non_bytes_are_rejected(self):
        for cap in ("MAXIMUM_RECEIPT_BYTES", "MAXIMUM_SOURCE_BYTES", "MAXIMUM_TOTAL_BYTES"):
            with self.subTest(cap=cap), patch.object(projection, cap, 1):
                self.reject()
        self.payloads["receipt"] = bytearray(self.payloads["receipt"])
        self.reject()

    def test_duplicate_json_keys_including_nested_keys_are_rejected(self):
        raw = self.payloads["receipt"]
        for changed in (
            b'{"production_eligible": true,' + raw[1:],
            raw.replace(b'"authorization": {', b'"authorization": {"paid_gpu_authorized": true,'),
        ):
            self.payloads["receipt"] = changed
            _pin(self.payloads, self.bindings, "receipt")
            self.reject()

    def test_json_root_constants_and_encoding_are_rejected(self):
        for raw in (b"[]", b"null", b"NaN", b"Infinity", b"\xff", b"{"):
            with self.subTest(raw=raw):
                self.payloads["receipt"] = raw
                _pin(self.payloads, self.bindings, "receipt")
                self.reject()

    def test_receipt_internal_digest_is_separately_pinned(self):
        self.bindings["receipt"]["internal_sha256"] = "0" * 64
        self.reject()
        self.bindings["receipt"]["internal_sha256"] = "invalid"
        self.reject()
        del self.bindings["receipt"]["internal_sha256"]
        self.reject()

    def test_all_operation_authorizations_remain_strictly_false(self):
        original = json.loads(self.payloads["receipt"])
        for flag in projection._AUTHORIZATION_FLAGS:
            for value in (True, 0, None):
                with self.subTest(flag=flag, value=value):
                    receipt = copy.deepcopy(original)
                    receipt["authorization"][flag] = value
                    _receipt(self.payloads, self.bindings, receipt)
                    self.reject()

    def test_extra_authorizations_and_receipt_claims_are_rejected(self):
        original = json.loads(self.payloads["receipt"])
        for mutation in ("extra_authorization", "missing_authorization", "purpose", "extra_claim"):
            with self.subTest(mutation=mutation):
                receipt = copy.deepcopy(original)
                if mutation == "extra_authorization":
                    receipt["authorization"]["training_authorized"] = True
                elif mutation == "missing_authorization":
                    del receipt["authorization"]["paid_gpu_authorized"]
                elif mutation == "purpose":
                    receipt["authorization"]["purpose"] = "production"
                else:
                    receipt["production_authorized"] = True
                _receipt(self.payloads, self.bindings, receipt)
                self.reject()

    def test_strict_nonproduction_and_no_raw_text_flags(self):
        original = json.loads(self.payloads["receipt"])
        for field, value in (("production_eligible", True), ("production_eligible", 0),
                             ("raw_training_text_retained", True),
                             ("production_promotions", 1), ("production_promotions", False)):
            with self.subTest(field=field, value=value):
                receipt = copy.deepcopy(original)
                receipt[field] = value
                _receipt(self.payloads, self.bindings, receipt)
                self.reject()

    def test_original_test_still_rejects_mutated_counts_and_status(self):
        original = json.loads(self.payloads["receipt"])
        for field, value in (("accepted_records", 31), ("rejected_records", 1),
                             ("all_sampled_rows_accepted", False), ("status", "promoted")):
            with self.subTest(field=field):
                receipt = copy.deepcopy(original)
                receipt[field] = value
                _receipt(self.payloads, self.bindings, receipt)
                module, _ = self.build()
                with self.assertRaises(AssertionError):
                    getattr(module, projection.TEST_NAME)()

    def test_original_validator_still_rejects_invalid_seal(self):
        receipt = json.loads(self.payloads["receipt"])
        receipt["reviewer_alias"] = "changed-synthetic-alias"
        _receipt(self.payloads, self.bindings, receipt, reseal=False)
        module, _ = self.build()
        with self.assertRaises(ValueError):
            getattr(module, projection.TEST_NAME)()

    def test_missing_duplicate_or_decorated_selected_body_is_rejected(self):
        original = self.payloads["test_source"]
        for changed in (
            original.replace(projection.TEST_NAME.encode(), b"unused_original_name", 1),
            original + f"\ndef {projection.TEST_NAME}():\n    pass\n".encode(),
            original.replace(f"def {projection.TEST_NAME}".encode(),
                             f"@print('must not execute')\ndef {projection.TEST_NAME}".encode()),
        ):
            self.payloads["test_source"] = changed
            _pin(self.payloads, self.bindings, "test_source")
            with patch("builtins.print", side_effect=AssertionError("side effect")) as printing:
                self.reject()
                printing.assert_not_called()

    def test_snapshot_rejects_writes_filesystem_escape_and_source_reads(self):
        module, snapshot = self.build()
        with self.assertRaisesRegex(ValueError, "^snapshot_mode_rejected$"):
            module.RECEIPT.open("w")
        with self.assertRaisesRegex(ValueError, "^snapshot_filesystem_escape_rejected$"):
            os.fspath(module.RECEIPT)
        with self.assertRaisesRegex(ValueError, "^snapshot_lookup_rejected$"):
            source_path = SYNTHETIC_ROOT + "/" + projection.SOURCE_PATHS["test_source"]
            snapshot.Path(source_path).read_bytes()
        with module.RECEIPT.open("rb") as stream:
            self.assertTrue(stream.getbuffer().readonly)
            with self.assertRaisesRegex(ValueError, "^snapshot_write_rejected$"):
                stream.write(b"mutation")

    def test_snapshot_preserves_authenticated_input_after_caller_mutation(self):
        module, _ = self.build()
        self.payloads["receipt"] = b"{}"
        self.bindings["receipt"]["path"] = "elsewhere"
        getattr(module, projection.TEST_NAME)()


if __name__ == "__main__":
    unittest.main()
