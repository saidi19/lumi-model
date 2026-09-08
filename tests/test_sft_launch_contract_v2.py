"""Synthetic metadata-budget contract checks plus isolated inherited v1 coverage."""

from __future__ import annotations

import copy
import importlib.util
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(filename):
    name = "_synthetic_metadata_" + filename.removesuffix(".py")
    spec = importlib.util.spec_from_file_location(name, ROOT / "src/lumi" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launch = load("posttraining_sft_launch_contract_v2.py")
previous = load("posttraining_sft_launch_contract_v1.py")


def inherited_tests():
    """Compile untouched historical test source into private v2-bound globals.

    Do not import lumi or modify any already-imported test/production globals.
    The only adaptations are the explicit module binding and versioned deny code.
    """
    path = ROOT / "tests/test_sft_launch_contract_v1.py"
    source = path.read_text(encoding="utf-8")
    old_import = "from lumi import posttraining_sft_launch_contract_v1 as launch"
    if source.count(old_import) != 1:
        raise AssertionError("synthetic_inherited_import_changed")
    source = source.replace(old_import, "launch = _bound_launch")
    source = source.replace("execution_disabled_in_local_development_v1",
                            "execution_disabled_in_local_development_v2")
    namespace = types.ModuleType("_synthetic_launch_v2_inherited")
    namespace.__file__ = str(path)
    namespace._bound_launch = launch
    exec(compile(source, str(path), "exec"), namespace.__dict__)
    return namespace


inherited = inherited_tests()
TestInheritedLaunchContract = inherited.LaunchContractTests


class MetadataLaunchContractTests(unittest.TestCase):
    def setUp(self):
        self.reference = inherited.synthetic_reference()
        self.contract = inherited.fill_synthetic_slots(self.reference)

    def test_all_three_schema_bindings_are_versioned_and_draft_remains_pending(self):
        self.assertEqual(launch.SCHEMA, "lumi.sft-launch-contract.v0.2")
        self.assertEqual(launch.AUTH_SCHEMA, "lumi.sft-execution-authorization.v0.2")
        self.assertEqual(launch.REPORT_SCHEMA, "lumi.sft-launch-contract-report.v0.2")
        self.assertIsNone(self.reference["resource_caps"])
        self.assertIsNone(self.reference["execution_authorization"])
        report = launch.analyze(self.reference, self.reference)
        self.assertEqual(report["schema_version"], launch.REPORT_SCHEMA)
        self.assertIn("resource_caps_unbound", report["pending_gates"])
        self.assertIn("execution_authorization_unbound", report["pending_gates"])
        self.assertFalse(report["execution_permitted"])
        self.assertFalse(report["production_eligible"])

    def test_old_contract_and_old_reference_cannot_validate_each_other(self):
        old = copy.deepcopy(self.reference)
        old["schema_version"] = previous.SCHEMA
        for contract, reference in ((old, old), (old, self.reference), (self.reference, old)):
            with self.subTest(contract=contract["schema_version"],
                              reference=reference["schema_version"]):
                with self.assertRaisesRegex(ValueError, "frozen_contract_field_changed"):
                    launch.analyze(contract, reference)

    def test_old_authorization_schema_rejected_even_with_new_subject_and_digest(self):
        approval = inherited.synthetic_approval(self.contract)
        approval["schema_version"] = previous.AUTH_SCHEMA
        self.contract["execution_authorization"] = approval
        with self.assertRaisesRegex(ValueError, "not_sft_specific_authorization"):
            launch.analyze(self.contract, self.reference,
                           expected_authorization_sha256=launch.digest(approval), now_unix=1500)

    def test_metadata_cap_is_mandatory_and_not_inferred_from_other_read_budgets(self):
        del self.contract["resource_caps"]["maximum_encoded_metadata_read_bytes"]
        with self.assertRaisesRegex(ValueError, "resource_caps_schema"):
            launch.analyze(self.contract, self.reference)

    def test_metadata_cap_must_be_positive_bounded_integer(self):
        for value in (None, False, True, 0, -1, 1.0, "1", 2**63):
            with self.subTest(value=value):
                self.contract["resource_caps"]["maximum_encoded_metadata_read_bytes"] = value
                with self.assertRaisesRegex(ValueError, "positive_integer_required"):
                    launch.analyze(self.contract, self.reference)

    def test_metadata_allowance_change_invalidates_subject_binding(self):
        approval = inherited.synthetic_approval(self.contract)
        self.contract["execution_authorization"] = approval
        self.contract["resource_caps"]["maximum_encoded_metadata_read_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "authorization_subject_mismatch"):
            launch.analyze(self.contract, self.reference,
                           expected_authorization_sha256=launch.digest(approval), now_unix=1500)

    def test_even_current_synthetic_v2_claim_is_not_execution_permission(self):
        approval = inherited.synthetic_approval(self.contract)
        self.contract["execution_authorization"] = approval
        report = launch.analyze(
            self.contract, self.reference,
            expected_authorization_sha256=launch.digest(approval), now_unix=1500)
        self.assertFalse(report["execution_permitted"])
        self.assertFalse(report["resource_caps_enforced_by_this_compiler"])
        self.assertIsNone(report["executable_argv"])
        with self.assertRaisesRegex(PermissionError, "execution_disabled_in_local_development_v2"):
            launch.deny_execution()

    def test_only_cap_and_versioned_bindings_change_draft_policy(self):
        old = previous.make_draft(*inherited.synthetic_inputs())
        current = copy.deepcopy(self.reference)
        current["schema_version"] = previous.SCHEMA
        self.assertEqual(current, old)
        self.assertEqual(launch.PROHIBITIONS, previous.PROHIBITIONS)
        self.assertEqual(launch.geometry(current["profile"]), previous.geometry(old["profile"]))
        self.assertEqual(set(launch.CAP_FIELDS) - set(previous.CAP_FIELDS),
                         {"maximum_encoded_metadata_read_bytes"})
        self.assertTrue(set(previous.CAP_FIELDS) <= set(launch.CAP_FIELDS))

    def test_isolated_inherited_binding_does_not_load_lumi(self):
        self.assertIs(inherited.launch, launch)
        self.assertNotEqual(launch.AUTH_SCHEMA, previous.AUTH_SCHEMA)
        # A full remote suite may already have ML imported; this binding must not
        # register its own implementation under the real lumi package namespace.
        self.assertIsNot(sys.modules.get("lumi.posttraining_sft_launch_contract_v1"), launch)


if __name__ == "__main__":
    unittest.main()
