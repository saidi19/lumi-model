"""Synthetic supplied-byte validation and native source AST checks only."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import unittest
from pathlib import Path
from unittest import mock

from scripts import sft_cpu_lookup_acceptance_report_v1 as contract
from scripts import sft_cpu_origin_diagnostic_report_v1 as diagnostic
from tests.test_sft_cpu_origin_diagnostic_report_v1 import controls, report, synthetic_spec


def envelope(*, passed=True):
    diag = report(synthetic_spec(), passed=passed)
    observer = None
    if passed:
        worker = {
            "schema_version": contract.WORKER_SCHEMA,
            "status": "passed",
            **dict.fromkeys(contract.WORKER_TRUE_FLAGS, True),
            **dict.fromkeys(contract.WORKER_FALSE_FLAGS, False),
            **contract.WORKER_COUNTS,
            **dict.fromkeys(contract.WORKER_COUNT_LIMITS, 6),
        }
        observer = {
            "schema": contract.OBSERVER_SCHEMA,
            "status": "passed",
            "worker_run": worker,
            "origin_counts": dict(diag["frozen_origin_counts"]),
            **controls(),
            **dict.fromkeys(contract.OBSERVER_TRUE_FLAGS, True),
            **dict.fromkeys(contract.OBSERVER_FALSE_FLAGS, False),
            **contract.OBSERVER_COUNTS,
        }
    return {"schema": contract.SCHEMA, "diagnostic": diag, "observer_report": observer}


class AcceptanceReportTests(unittest.TestCase):
    def validate(self, value):
        return self.raw(contract.canonical(value))

    def raw(self, raw):
        return contract.validate_report(raw, controls(), diagnostic, synthetic_spec())

    def rejected(self, value):
        with self.assertRaises(contract.ValidationError) as raised:
            self.validate(value)
        self.assertIn(raised.exception.code, contract.ERROR_CODES)
        self.assertEqual(str(raised.exception), raised.exception.code)

    def test_passed_and_failed_roundtrip(self):
        for passed in (True, False):
            value = envelope(passed=passed)
            self.assertEqual(self.validate(value), value)

    def test_valid_failed_report_never_grants_acceptance(self):
        value = self.validate(envelope(passed=False))
        self.assertIsNone(value["observer_report"])
        self.assertIs(value["diagnostic"]["acceptance_passed"], False)

    def test_delegates_exact_canonical_diagnostic(self):
        value = envelope()
        with mock.patch.object(
            diagnostic, "validate_report", wraps=diagnostic.validate_report
        ) as fn:
            self.validate(value)
        fn.assert_called_once_with(
            contract.canonical(value["diagnostic"]), controls(), synthetic_spec()
        )

    def test_failed_report_cannot_carry_passed_observer(self):
        value = envelope()
        value["diagnostic"] = envelope(passed=False)["diagnostic"]
        self.rejected(value)

    def test_passed_diagnostic_requires_observer_report(self):
        for invalid in (None, True, [], "passed", {}):
            value = envelope()
            value["observer_report"] = invalid
            self.rejected(value)

    def test_every_envelope_observer_worker_origin_key_exact(self):
        for path in (
            (),
            ("observer_report",),
            ("observer_report", "worker_run"),
            ("observer_report", "origin_counts"),
        ):
            source = envelope()
            selected = source
            for key in path:
                selected = selected[key]
            for key in tuple(selected):
                value = copy.deepcopy(source)
                target = value
                for part in path:
                    target = target[part]
                del target[key]
                self.rejected(value)
            selected["raw_text"] = "must not survive"
            self.rejected(source)

    def test_schemas_and_statuses_exact(self):
        for path, field in (
            ((), "schema"),
            (("observer_report",), "schema"),
            (("observer_report",), "status"),
            (("observer_report", "worker_run"), "schema_version"),
            (("observer_report", "worker_run"), "status"),
        ):
            for invalid in (None, False, 1, "wrong", "failed"):
                value = envelope()
                target = value
                for key in path:
                    target = target[key]
                target[field] = invalid
                self.rejected(value)

    def test_all_existing_true_and_false_claims_exact_boolean(self):
        for path, fields, expected in (
            (("observer_report",), contract.OBSERVER_TRUE_FLAGS, True),
            (("observer_report",), contract.OBSERVER_FALSE_FLAGS, False),
            (("observer_report", "worker_run"), contract.WORKER_TRUE_FLAGS, True),
            (("observer_report", "worker_run"), contract.WORKER_FALSE_FLAGS, False),
        ):
            for field in fields:
                for invalid in (not expected, int(expected), None, "true"):
                    value = envelope()
                    target = value
                    for key in path:
                        target = target[key]
                    target[field] = invalid
                    self.rejected(value)

    def test_all_exact_counts_reject_bool_wrong_type_and_wrong_value(self):
        for path, fields in (
            (("observer_report",), contract.OBSERVER_COUNTS),
            (("observer_report", "worker_run"), contract.WORKER_COUNTS),
        ):
            for field, expected in fields.items():
                for invalid in (True, False, None, str(expected), expected - 1, expected + 1):
                    value = envelope()
                    target = value
                    for key in path:
                        target = target[key]
                    target[field] = invalid
                    self.rejected(value)

    def test_existing_resource_count_ceilings(self):
        for field, maximum in contract.WORKER_COUNT_LIMITS.items():
            for accepted in (1, maximum):
                value = envelope()
                value["observer_report"]["worker_run"][field] = accepted
                self.validate(value)
            for invalid in (True, False, -1, 0, maximum + 1, None, "6"):
                value = envelope()
                value["observer_report"]["worker_run"][field] = invalid
                self.rejected(value)

    def test_origin_counters_bound_and_match_diagnostic(self):
        field = contract.ORIGIN_FIELDS[0]
        for invalid in (True, -1, 196_609, "0", None, 1):
            value = envelope()
            value["observer_report"]["origin_counts"][field] = invalid
            self.rejected(value)
        value = envelope()
        value["observer_report"]["origin_counts"][field] = 196_608
        value["diagnostic"]["frozen_origin_counts"][field] = 196_608
        self.validate(value)

    def test_each_observer_control_matches_expected(self):
        for field in contract.CONTROL_FIELDS:
            for invalid in (False, None, "b" * len(controls()[field])):
                value = envelope()
                value["observer_report"][field] = invalid
                self.rejected(value)

    def test_diagnostic_failure_remains_fixed_code(self):
        value = envelope()
        value["diagnostic"]["acceptance_passed"] = True
        self.rejected(value)
        with mock.patch.object(diagnostic, "validate_report", side_effect=ValueError("secret")):
            with self.assertRaisesRegex(contract.ValidationError, "^diagnostic_rejected$"):
                self.validate(envelope())

    def test_duplicate_keys_at_any_depth(self):
        raw = contract.canonical(envelope())
        for anchor in (b'{"diagnostic":', b'{"anonymous_executable_mappings":'):
            altered = raw.replace(anchor, anchor[:1] + b'"dup":0,"dup":1,' + anchor[1:], 1)
            with self.assertRaisesRegex(contract.ValidationError, "report_duplicate_key"):
                self.raw(altered)

    def test_noncanonical_json_rejected(self):
        value = envelope()
        raw = contract.canonical(value)
        for altered in (
            raw[:-1],
            raw + b"\n",
            b" " + raw,
            json.dumps(value, indent=2).encode() + b"\n",
        ):
            with self.assertRaises(contract.ValidationError):
                self.raw(altered)

    def test_float_special_number_and_large_integer_rejected(self):
        raw = contract.canonical(envelope())
        for number in (b"6.0", b"6e0", b"NaN", b"Infinity", b"-Infinity", b"1234567890123"):
            altered = raw.replace(b'"controlled_loads":6', b'"controlled_loads":' + number)
            with self.assertRaisesRegex(contract.ValidationError, "report_number_rejected"):
                self.raw(altered)

    def test_type_budget_encoding_and_malformed_json(self):
        for raw in (
            None,
            "{}",
            bytearray(b"{}"),
            b"",
            b" " * 32769,
            b"\xff",
            b"{",
            b"[]\n",
            b"null\n",
            b"[" * 2000,
        ):
            with self.assertRaises(contract.ValidationError):
                self.raw(raw)

    def test_result_is_fresh_and_native_sources_not_imported(self):
        value = envelope()
        result = self.validate(value)
        result["observer_report"]["status"] = "changed"
        self.assertEqual(value["observer_report"]["status"], "passed")


class SourceContractTests(unittest.TestCase):
    def test_real_wrapper_ast_spec_validates_both_synthetic_envelopes(self):
        root = Path(__file__).resolve().parents[1]
        raw = (root / "scripts/run_sft_cpu_lookup_acceptance_v1.py").read_bytes()
        spec = diagnostic.spec_from_source(
            raw, "6e453f0a8ab5dc84f126db1761a7c089e0a239cf588a20cdc56b6d5236af0cf4"
        )
        for passed in (True, False):
            value = envelope(passed=passed)
            self.assertEqual(
                contract.validate_report(contract.canonical(value), controls(), diagnostic, spec),
                value,
            )

    def constants(self, filename, expected):
        path = Path(__file__).resolve().parents[1] / "scripts" / filename
        raw = path.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), expected)
        tree = ast.parse(raw)
        constants = {}
        for item in tree.body:
            if isinstance(item, ast.Assign) and len(item.targets) == 1:
                if isinstance(item.targets[0], ast.Name):
                    try:
                        constants[item.targets[0].id] = ast.literal_eval(item.value)
                    except (ValueError, TypeError):
                        pass
        return constants, tree

    def test_frozen_flag_count_report_key_allowlists_match_source_ast(self):
        fixture, fixture_tree = self.constants(
            "run_sft_worker_run_torch_fixture_v1.py",
            "53c50c92965ed1cef0c9624c57367c31ed7ebfb9d7cfb34257188ad179d333d0",
        )
        probe, probe_tree = self.constants(
            "run_sft_oci_worker_probe_v2.py",
            "b45b04fc9cb93dd2301df0444f2db95666ba4f7a32e8f4d8e28c5b4077314450",
        )
        self.assertEqual(fixture["FLAGS"], contract.WORKER_TRUE_FLAGS)
        self.assertEqual(fixture["NONCLAIMS"], contract.WORKER_FALSE_FLAGS)
        self.assertEqual(probe["NONCLAIMS"], contract.OBSERVER_FALSE_FLAGS)
        self.assertEqual(probe["WITNESS_FIELDS"], contract.ORIGIN_FIELDS)
        self.assertEqual(fixture["SCHEMA"], contract.WORKER_SCHEMA)
        self.assertEqual(probe["SCHEMA"], contract.OBSERVER_SCHEMA)
        for tree, keys, expanded, exact in (
            (
                fixture_tree,
                contract.WORKER_KEYS,
                (*contract.WORKER_TRUE_FLAGS, *contract.WORKER_FALSE_FLAGS),
                contract.WORKER_COUNTS,
            ),
            (
                probe_tree,
                contract.OBSERVER_KEYS,
                contract.OBSERVER_FALSE_FLAGS,
                contract.OBSERVER_COUNTS,
            ),
        ):
            run = next(
                item
                for item in tree.body
                if isinstance(item, ast.FunctionDef) and item.name == "run"
            )
            result = next(
                item.value
                for item in ast.walk(run)
                if isinstance(item, ast.Return) and isinstance(item.value, ast.Dict)
            )
            explicit = {item.value for item in result.keys if isinstance(item, ast.Constant)}
            self.assertEqual(explicit | set(expanded), set(keys))
            literals = {
                key.value: value.value
                for key, value in zip(result.keys, result.values, strict=True)
                if isinstance(key, ast.Constant) and isinstance(value, ast.Constant)
            }
            for key, expected in exact.items():
                if key in literals:
                    self.assertEqual(literals[key], expected)
        self.assertEqual(
            contract.WORKER_COUNTS["index_bytes_read"], 12 * fixture["INDEX_ROW_BYTES"]
        )

    def test_validator_has_only_stdlib_pure_imports(self):
        tree = ast.parse(Path(contract.__file__).read_bytes())
        imports = [
            item for item in ast.walk(tree) if isinstance(item, (ast.Import, ast.ImportFrom))
        ]
        names = {
            item.module if isinstance(item, ast.ImportFrom) else item.names[0].name
            for item in imports
        }
        self.assertEqual(names, {"__future__", "json"})


if __name__ == "__main__":
    unittest.main()
