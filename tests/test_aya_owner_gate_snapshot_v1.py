"""Synthetic-only owner-tamper projection tests; no historical payload reads.

Only the materialization schema gets an explicit unavailable-preimage stand-in.
The invented owner receipt is sealed with genuine SHA-256, and its mutation is
checked by the unchanged genuine owner hashing. No prior test body is called.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import pytest
import test_aya_prerequisite_snapshot_v1 as fixtures

from lumi import posttraining_aya_owner_gate_snapshot_v1 as projection
from lumi import posttraining_aya_prerequisite_snapshot_v1 as prerequisite

ERROR = "invalid Aya owner-gate source projection"
MATERIALIZATION_SCHEMA = "lumi.sft-aya-bengali-materialization-report.v0.1"


@contextmanager
def materialization_preimage_double_only(module):
    """Keep genuine owner hashing; substitute only the invented report preimage."""
    original = module.sha256_value

    def digest(value):
        if value.get("schema_version") == MATERIALIZATION_SCHEMA:
            return fixtures.APPROVED_SEALS[MATERIALIZATION_SCHEMA]
        return original(value)

    with patch.object(module, "sha256_value", side_effect=digest) as calls:
        yield calls


class AyaOwnerGateSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.payloads, self.bindings = fixtures.synthetic_inputs()
        owner = json.loads(self.payloads["owner_review_receipt"])
        del owner["receipt_sha256"]
        owner["receipt_sha256"] = hashlib.sha256(fixtures.canonical(owner)).hexdigest()
        self.payloads["owner_review_receipt"] = fixtures.canonical(owner)
        fixtures.pin(self.payloads, self.bindings, "owner_review_receipt")
        fixtures.repin_plan(self.payloads, self.bindings)

    def build(self):
        module, snapshot = projection.build_test_module(
            self.payloads, self.bindings, fixtures.SYNTHETIC_ROOT,
        )
        self.assertNotIn("pytest", vars(module))
        module.pytest = pytest
        return module, snapshot

    def reject(self):
        with self.assertRaisesRegex(ValueError, "^" + ERROR + "$") as caught:
            self.build()
        self.assertTrue(caught.exception.__suppress_context__)

    def run_negative(self):
        module, snapshot = self.build()
        with materialization_preimage_double_only(module) as calls:
            self.assertIsNone(getattr(module, projection.TEST_NAME)())
            self.assertEqual(calls.call_count, 2)
            changed = calls.call_args_list[1].args[0]
            self.assertEqual(changed["rejected_records"], 1)
            self.assertIs(changed["all_sampled_rows_accepted"], False)
        return module, snapshot

    def test_genuine_owner_tamper_rejection_and_exact_snapshot_accounting(self):
        before = copy.deepcopy((self.payloads, self.bindings))
        module, snapshot = self.run_negative()
        expected_bytes = sum(
            self.bindings[role]["bytes"] for role in projection.PLAN_BINDING_ROLES | {"plan"}
        ) + sum(self.bindings[role]["bytes"] for role in (
            "attribution_manifest", "materialization_report", "owner_review_receipt",
        ))
        self.assertEqual(snapshot.summary(), {
            "bound_blob_count": 9, "virtual_opens": 12,
            "logical_read_bytes_debited": expected_bytes,
            "maximum_virtual_opens": 12, "maximum_logical_read_bytes": expected_bytes,
            "maximum_gzip_bytes": 1, "filesystem_access_performed": False,
            "execution_permitted": False,
        })
        self.assertEqual((self.payloads, self.bindings), before)
        with self.assertRaisesRegex(ValueError, "^snapshot_open_limit_exceeded$"):
            module.PLAN.read_bytes()

    def test_removing_both_tamper_assignments_cannot_pass(self):
        self.payloads["test_source"] = self.payloads["test_source"].replace(
            b'changed["rejected_records"] = 1', b'changed["rejected_records"] = 0',
        ).replace(
            b'changed["all_sampled_rows_accepted"] = False',
            b'changed["all_sampled_rows_accepted"] = True',
        )
        fixtures.pin(self.payloads, self.bindings, "test_source")
        module, _ = self.build()
        with materialization_preimage_double_only(module), self.assertRaises(AssertionError):
            getattr(module, projection.TEST_NAME)()

    def test_authentic_hash_helpers_and_unchanged_owner_seal(self):
        module, _ = self.build()
        owner = json.loads(self.payloads["owner_review_receipt"])
        self.assertEqual(module._verify_internal_seal(owner, "receipt_sha256"),
                         owner["receipt_sha256"])
        changed = copy.deepcopy(owner)
        changed["rejected_records"] = 1
        with self.assertRaisesRegex(ValueError, "^receipt_sha256 changed$"):
            module._verify_internal_seal(changed, "receipt_sha256")
        self.assertEqual(module.canonical_json_bytes(owner), fixtures.canonical(owner))
        self.assertEqual(module.sha256_value(owner),
                         hashlib.sha256(fixtures.canonical(owner)).hexdigest())

    def test_no_synthetic_materialization_double_cannot_fake_owner_rejection(self):
        module, snapshot = self.build()
        with self.assertRaises(AssertionError):
            getattr(module, projection.TEST_NAME)()
        self.assertEqual(snapshot.summary()["virtual_opens"], 12)

    def test_original_asts_and_locations_are_exact_and_prior_positive_is_not_selected(self):
        module, _ = self.build()
        selections = {
            "promotion_implementation": (projection.AYA_CONSTANTS, projection.AYA_FUNCTIONS),
            "hash_helper_source": (frozenset(), {"canonical_json_bytes", "file_sha256"}),
            "test_source": (frozenset(), {"_load", "_prerequisites", projection.TEST_NAME}),
        }
        for role, (constants, functions) in selections.items():
            original = ast.parse(self.payloads[role])
            nodes = projection._select(self.payloads[role], frozenset(constants),
                                       frozenset(functions))
            for node in nodes:
                name = node.name if isinstance(node, ast.FunctionDef) else node.targets[0].id
                expected = next(item for item in original.body if (
                    isinstance(item, ast.FunctionDef) and item.name == name
                    or isinstance(item, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == name
                            for target in item.targets)
                ))
                self.assertEqual(ast.dump(node, include_attributes=True),
                                 ast.dump(expected, include_attributes=True))
                if name in functions:
                    self.assertEqual(getattr(module, name).__code__.co_firstlineno, node.lineno)
                    self.assertEqual(getattr(module, name).__code__.co_filename,
                                     fixtures.SYNTHETIC_ROOT + "/" + projection.SOURCE_PATHS[role])
        self.assertEqual({name for name in vars(module) if name.startswith("test_")},
                         {projection.TEST_NAME})
        self.assertNotIn(prerequisite.TEST_NAME, vars(module))
        self.assertIs(projection._authenticate, prerequisite._authenticate)
        self.assertIs(projection._check_plan, prerequisite._check_plan)

    def test_no_real_file_operations_prior_builder_or_namespace_changes(self):
        sentinel = types.ModuleType(projection.MODULE_NAME)
        child = types.ModuleType(projection.MODULE_NAME + ".sentinel")
        prior_identity = prerequisite.TEST_NAME
        with (
            patch.dict(sys.modules, {sentinel.__name__: sentinel, child.__name__: child}),
            patch.object(prerequisite, "build_test_module", side_effect=AssertionError("prior")),
            patch("builtins.open", side_effect=AssertionError("real open")),
            patch("io.open", side_effect=AssertionError("real open")),
            patch("os.open", side_effect=AssertionError("real open")),
            patch("os.stat", side_effect=AssertionError("real stat")),
            patch("os.lstat", side_effect=AssertionError("real stat")),
        ):
            module, _ = self.run_negative()
            self.assertIsNot(module, sentinel)
            self.assertIs(sys.modules[sentinel.__name__], sentinel)
            self.assertIs(sys.modules[child.__name__], child)
            self.assertEqual(prerequisite.TEST_NAME, prior_identity)

    def test_hash_only_roles_and_unselected_source_never_parse_or_execute(self):
        poison = (b"\nimport nonexistent_synthetic_owner_gate_dependency\n"
                  b"raise RuntimeError('synthetic top-level poison')\n")
        for role in fixtures.SOURCE_ROLES:
            self.payloads[role] += poison
            fixtures.pin(self.payloads, self.bindings, role)
        fixtures.repin_plan(self.payloads, self.bindings)
        module, _ = self.run_negative()
        for name in ("build_successor_registry", "load_policy_and_predecessor",
                     "verify_source_promotion_receipt", "build_source_promotion_receipt",
                     "SFTSource", "SFTPolicy", "main", "SUCCESSOR", "RECEIPT"):
            self.assertNotIn(name, vars(module))

    def test_private_metadata_stays_opaque_until_original_test_execution(self):
        for role in ("attribution_manifest", "materialization_report", "owner_review_receipt"):
            self.payloads[role] = b"unparsable invented metadata\xff"
            fixtures.pin(self.payloads, self.bindings, role)
        fixtures.repin_plan(self.payloads, self.bindings)
        _, snapshot = self.build()
        self.assertEqual(snapshot.summary()["virtual_opens"], 0)

    def test_every_input_authenticates_before_ast_or_json_parsing(self):
        for role in prerequisite.INPUT_ROLES:
            with self.subTest(role=role):
                original = self.payloads[role]
                self.payloads[role] = original + b"x"
                with (
                    patch.object(ast, "parse", side_effect=AssertionError("early AST")) as parse,
                    patch.object(json, "loads", side_effect=AssertionError("early JSON")) as loads,
                ):
                    self.reject()
                    parse.assert_not_called()
                    loads.assert_not_called()
                self.payloads[role] = original

    def test_reused_authentication_limits_and_binding_checks_still_fail_closed(self):
        original = copy.deepcopy(self.bindings)
        for field, value in (("path", "unexpected/path"), ("bytes", True),
                             ("sha256", "0" * 64), ("extra", "SYNTHETIC_SECRET")):
            with self.subTest(field=field):
                self.bindings = copy.deepcopy(original)
                self.bindings["owner_review_receipt"][field] = value
                self.reject()
        self.bindings = original
        with patch.object(prerequisite, "MAXIMUM_TOTAL_BYTES", 1):
            self.reject()
        with patch.object(prerequisite, "MAXIMUM_SOURCE_BYTES", 1):
            self.reject()
        with patch.object(prerequisite, "MAXIMUM_METADATA_BYTES", 1):
            self.reject()
        self.payloads["unexpected"] = b"extra"
        self.reject()

    def test_snapshot_excludes_source_only_inputs_writes_and_filesystem_escape(self):
        module, snapshot = self.build()
        for role in prerequisite.INPUT_ROLES - (projection.PLAN_BINDING_ROLES | {"plan"}):
            path = module.ROOT / projection.SOURCE_PATHS[role]
            self.assertFalse(path.exists())
            with self.assertRaisesRegex(ValueError, "^snapshot_lookup_rejected$"):
                path.read_bytes()
        with self.assertRaisesRegex(ValueError, "^snapshot_mode_rejected$"):
            module.PLAN.open("w")
        with self.assertRaisesRegex(ValueError, "^snapshot_filesystem_escape_rejected$"):
            module.PLAN.__fspath__()
        self.assertEqual(snapshot.summary()["virtual_opens"], 0)

    def test_invalid_roots_and_unauthenticated_plan_references_are_raw_free(self):
        for root in ("/", "//synthetic", "relative", "/synthetic/..", "/bad\\path", "/~bad"):
            with self.subTest(root=root), self.assertRaisesRegex(ValueError, "^" + ERROR + "$"):
                projection.build_test_module(self.payloads, self.bindings, root)
        plan = json.loads(self.payloads["plan"])
        plan["bindings"]["data_contract"]["sha256"] = "0" * 64
        self.payloads["plan"] = fixtures.canonical(plan)
        fixtures.pin(self.payloads, self.bindings, "plan")
        self.reject()
