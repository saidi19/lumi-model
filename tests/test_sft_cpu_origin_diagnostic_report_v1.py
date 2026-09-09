"""Small supplied-byte tests; no native diagnostic source is imported or run."""

from __future__ import annotations

import copy
import hashlib
import json
import unittest
from unittest import mock

from scripts import sft_cpu_origin_diagnostic_report_v1 as contract


def synthetic_spec():
    flags = (*contract.BASE_FALSE_FLAGS, *contract.ADDITIONAL_FALSE_FLAGS)
    return {
        "REPORT_KEYS": (*contract.REPORT_BASE_FIELDS, *flags),
        "CONTROL_KEYS": tuple(contract.CONTROL_FIELDS),
        "DIAGNOSTIC_ENUMS": {
            "captured_check": (
                "not_reported",
                "module_origins_check_01",
                "module_origins_unexpected",
            ),
            "captured_function": ("not_reported", "module_origins"),
            "phase": ("not_reported", "inputs", "final"),
            "origin_category": ("not_available", "runtime", "source", "dependency"),
            "module_family": ("none", "torch", "lumi", "other"),
            "exception_class": ("not_reported", "ValueError", "ProbeError"),
            "probe_error_code": ("not_reported", "module_origin_rejected"),
        },
        "COUNT_LIMITS": dict(contract.HARD_COUNT_LIMITS),
        "FROZEN_STAGE_ENUMS": {
            "probe_stage": ("not_reported", "worker_run", "complete"),
            "fixture_stage": ("not_reported", "workers", "complete"),
            "child_stage": ("not_reported", "dependencies", "done"),
            "worker_stage": ("not_reported", "runtime", "complete"),
            "worker_code": ("not_reported", "runtime_rejected"),
        },
        "COMPONENT_ENUMS": {
            field: ("not_reported", "runtime_rejected") for field in contract.COMPONENT_FIELDS
        },
        "ORIGIN_COUNT_FIELDS": tuple(contract.ORIGIN_FIELDS),
        "FALSE_FLAGS": flags,
    }


def source(spec):
    return ("\n".join(key + " = " + repr(value) for key, value in spec.items()) + "\n").encode()


def pin(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("ascii")
        + b"\n"
    )


def controls():
    return {key: "a" * (40 if key == "source_commit" else 64) for key in contract.CONTROL_FIELDS}


def report(spec, *, passed=False):
    value = {
        "schema": contract.SCHEMA,
        "status": "diagnostic_completed",
        "underlying_status": "passed" if passed else "failed",
        "controls": controls(),
        "diagnostic": dict(contract.DIAGNOSTIC_DEFAULTS),
        "counts": dict.fromkeys(spec["COUNT_LIMITS"], 0),
        "frozen_stages": dict.fromkeys(spec["FROZEN_STAGE_ENUMS"], "not_reported"),
        "frozen_component_diagnostics": dict.fromkeys(spec["COMPONENT_ENUMS"], "not_reported"),
        "frozen_origin_counts": dict.fromkeys(spec["ORIGIN_COUNT_FIELDS"], 0),
        **dict.fromkeys(spec["FALSE_FLAGS"], False),
    }
    if passed:
        value["counts"].update(
            witnesses_started=12,
            witnesses_completed=12,
            need_checks_observed=1000,
            rank_workers_completed=6,
            invocations_completed=3,
            origin_observations_completed=12,
        )
    else:
        value["diagnostic"].update(
            captured_check="module_origins_check_01",
            captured_function="module_origins",
            exception_class="ProbeError",
            probe_error_code="module_origin_rejected",
        )
        value["counts"]["failure_records_retained"] = 1
    return value


class DiagnosticSpecTests(unittest.TestCase):
    def setUp(self):
        self.spec = synthetic_spec()

    def extract(self, raw):
        return contract.spec_from_source(raw, pin(raw))

    def assert_code(self, code, action):
        with self.assertRaises(contract.ValidationError) as raised:
            action()
        self.assertEqual(raised.exception.code, code)
        self.assertEqual(str(raised.exception), code)

    def test_extracts_eight_fresh_nested_literals(self):
        raw = source(self.spec)
        actual = self.extract(raw)
        self.assertEqual(actual, self.spec)
        actual["COUNT_LIMITS"]["witnesses_started"] = 1
        self.assertEqual(self.spec["COUNT_LIMITS"]["witnesses_started"], 12)
        self.assertEqual(self.extract(raw), self.spec)

    def test_native_source_is_ast_only_not_executed(self):
        raw = source(self.spec) + b"raise AssertionError('MUST_NEVER_EXECUTE')\nimport torch\n"
        self.assertEqual(self.extract(raw), self.spec)

    def test_pin_is_checked_before_ast_parsing(self):
        raw = source(self.spec)
        with mock.patch.object(contract.ast, "parse") as parser:
            self.assert_code(
                "source_hash_mismatch", lambda: contract.spec_from_source(raw, "0" * 64)
            )
        parser.assert_not_called()

    def test_exact_bytes_type_and_source_budget(self):
        raw = source(self.spec)
        for invalid in (None, raw.decode(), bytearray(raw), memoryview(raw)):
            with self.subTest(kind=type(invalid).__name__):
                self.assert_code(
                    "invalid_source_type",
                    lambda invalid=invalid: contract.spec_from_source(invalid, pin(raw)),
                )
        for invalid in (b"", b" " * (contract.MAXIMUM_SOURCE_BYTES + 1)):
            self.assert_code(
                "source_budget_exceeded", lambda invalid=invalid: self.extract(invalid)
            )

    def test_pin_must_be_lowercase_hex_64(self):
        raw = source(self.spec)
        for invalid in (None, 1, "a" * 63, "a" * 65, "A" * 64, "g" * 64):
            with self.subTest(pin=invalid):
                self.assert_code(
                    "invalid_source_sha256",
                    lambda invalid=invalid: contract.spec_from_source(raw, invalid),
                )

    def test_malformed_source_has_fixed_failure(self):
        for raw in (b"def (: invalid", b"\x00", b"\xff"):
            self.assert_code("source_syntax_rejected", lambda raw=raw: self.extract(raw))

    def test_missing_constant_rejected(self):
        for key in contract.SPEC_KEYS:
            altered = dict(self.spec)
            del altered[key]
            with self.subTest(key=key):
                self.assert_code(
                    "spec_missing_constants", lambda altered=altered: self.extract(source(altered))
                )

    def test_duplicate_and_nested_reassignment_rejected(self):
        suffixes = (
            b"REPORT_KEYS = ()\n",
            b"REPORT_KEYS += ('extra',)\n",
            b"def mutate():\n    COUNT_LIMITS = {}\n",
            b"def mutate():\n    x = (FALSE_FLAGS := ())\n",
        )
        for suffix in suffixes:
            with self.subTest(suffix=suffix):
                self.assert_code(
                    "spec_duplicate_constants",
                    lambda suffix=suffix: self.extract(source(self.spec) + suffix),
                )

    def test_computed_chained_and_annotated_constants_rejected(self):
        original = source(self.spec)
        first = original.splitlines(keepends=True)[0]
        replacements = (
            b"REPORT_KEYS = tuple()\n",
            b"REPORT_KEYS = OTHER = ()\n",
            b"REPORT_KEYS: tuple = ()\n",
            b"REPORT_KEYS, OTHER = (), ()\n",
        )
        for replacement in replacements:
            with self.subTest(replacement=replacement):
                raw = replacement + original[len(first) :]
                self.assert_code("spec_literal_rejected", lambda raw=raw: self.extract(raw))

    def test_spec_does_not_allow_missing_or_duplicate_flags(self):
        for flags in (
            self.spec["FALSE_FLAGS"][:-1],
            (*self.spec["FALSE_FLAGS"][:-1], self.spec["FALSE_FLAGS"][0]),
        ):
            altered = copy.deepcopy(self.spec)
            altered["FALSE_FLAGS"] = flags
            with self.assertRaises(contract.ValidationError):
                self.extract(source(altered))

    def test_spec_required_control_count_origin_and_diagnostic_keys(self):
        for name in ("CONTROL_KEYS", "COUNT_LIMITS", "ORIGIN_COUNT_FIELDS", "DIAGNOSTIC_ENUMS"):
            altered = copy.deepcopy(self.spec)
            if isinstance(altered[name], dict):
                altered[name].pop(next(iter(altered[name])))
            else:
                altered[name] = altered[name][:-1]
            with self.subTest(name=name):
                self.assert_code(
                    "spec_shape_rejected", lambda altered=altered: self.extract(source(altered))
                )

    def test_spec_enum_values_bounded_unique_ascii_atoms(self):
        invalid_values = (
            (),
            ("not_reported", "not_reported"),
            ("not_reported", "x" * 129),
            ("not_reported", "raw text"),
            ("not_reported", "বাংলা"),
            ("not_reported", "/private/path"),
            ("not_reported", 1),
            ("other",),
            tuple("v" + str(index) for index in range(257)),
        )
        for values in invalid_values:
            altered = copy.deepcopy(self.spec)
            altered["DIAGNOSTIC_ENUMS"]["captured_check"] = values
            with self.subTest(values=values[:2]):
                self.assert_code(
                    "spec_enum_rejected", lambda altered=altered: self.extract(source(altered))
                )

    def test_spec_count_caps_reject_bool_float_zero_and_huge(self):
        for invalid in (True, 1.0, 0, -1, 2_000_001, "10"):
            altered = copy.deepcopy(self.spec)
            altered["COUNT_LIMITS"]["need_checks_observed"] = invalid
            with self.subTest(invalid=invalid):
                self.assert_code(
                    "spec_count_rejected", lambda altered=altered: self.extract(source(altered))
                )


class DiagnosticReportTests(unittest.TestCase):
    def setUp(self):
        self.spec = synthetic_spec()
        self.failed = report(self.spec)
        self.passed = report(self.spec, passed=True)

    def validate(self, value, *, spec=None, expected=None):
        return contract.validate_report(
            encoded(value),
            controls() if expected is None else expected,
            self.spec if spec is None else spec,
        )

    def assert_code(self, code, action):
        with self.assertRaises(contract.ValidationError) as raised:
            action()
        self.assertEqual(raised.exception.code, code)
        self.assertEqual(str(raised.exception), code)

    def test_completed_failed_and_passed_reports_validate(self):
        self.assertEqual(self.validate(self.failed), self.failed)
        self.assertEqual(self.validate(self.passed), self.passed)

    def test_return_and_inputs_are_independent_fresh_values(self):
        original = copy.deepcopy(self.failed)
        actual = self.validate(self.failed)
        actual["counts"]["failure_records_retained"] = 999
        actual["controls"]["source_commit"] = "b" * 40
        self.assertEqual(self.failed, original)
        self.assertEqual(self.validate(self.failed), original)

    def test_exact_bytes_and_report_size_cap(self):
        raw = encoded(self.failed)
        for invalid in (None, raw.decode(), bytearray(raw), memoryview(raw)):
            self.assert_code(
                "invalid_report_type",
                lambda invalid=invalid: contract.validate_report(invalid, controls(), self.spec),
            )
        for invalid in (b"", b" " * (contract.MAXIMUM_REPORT_BYTES + 1)):
            self.assert_code(
                "report_budget_exceeded",
                lambda invalid=invalid: contract.validate_report(invalid, controls(), self.spec),
            )

    def test_exactly_one_lf_and_canonical_ascii_encoding(self):
        raw = encoded(self.failed)
        variations = (
            raw[:-1],
            raw + b"\n",
            b" " + raw,
            raw.replace(b":", b": ", 1),
            raw.replace(b"\n", b"\r\n"),
            json.dumps(self.failed, sort_keys=False).encode() + b"\n",
        )
        for changed in variations:
            with self.subTest(changed=changed[:20]):
                self.assert_code(
                    "report_noncanonical",
                    lambda changed=changed: contract.validate_report(
                        changed, controls(), self.spec
                    ),
                )

    def test_duplicate_json_keys_fail_even_with_equal_value(self):
        raw = encoded(self.failed).replace(b'{"', b'{"schema":"ignored","', 1)
        self.assert_code(
            "report_duplicate_key", lambda: contract.validate_report(raw, controls(), self.spec)
        )

    def test_floats_nonfinite_and_oversized_integer_literals_reject(self):
        raw = encoded(self.failed)
        marker = b'"need_checks_observed":0'
        for value in (b"0.0", b"1e0", b"NaN", b"Infinity", b"-Infinity", b"1" * 13):
            changed = raw.replace(marker, b'"need_checks_observed":' + value)
            self.assert_code(
                "report_number_rejected",
                lambda changed=changed: contract.validate_report(changed, controls(), self.spec),
            )

    def test_malformed_and_non_ascii_json_never_exposes_text(self):
        for raw in (b"PRIVATE_EXCEPTION_TEXT", b"\xff", b"[]\n", b"null\n"):
            with self.assertRaises(contract.ValidationError) as raised:
                contract.validate_report(raw, controls(), self.spec)
            self.assertNotIn("PRIVATE", str(raised.exception))
            self.assertIn(raised.exception.code, contract.ERROR_CODES)

    def test_top_level_extra_missing_and_wrong_status_rejected(self):
        for mutation in (
            lambda value: value.update(extra="sensitive"),
            lambda value: value.pop("counts"),
            lambda value: value.update(schema="other"),
            lambda value: value.update(status="diagnostic_incomplete"),
            lambda value: value.update(status="passed"),
            lambda value: value.update(underlying_status="unknown"),
        ):
            altered = copy.deepcopy(self.failed)
            mutation(altered)
            self.assert_code(
                "report_shape_rejected", lambda altered=altered: self.validate(altered)
            )

    def test_expected_controls_exact_keys_types_lengths_and_matching(self):
        for key in contract.CONTROL_FIELDS:
            for invalid in (
                None,
                True,
                "A" * len(controls()[key]),
                "a",
                "b" * len(controls()[key]),
            ):
                altered = controls()
                altered[key] = invalid
                with self.subTest(key=key, invalid=invalid):
                    self.assert_code(
                        "control_rejected",
                        lambda altered=altered: self.validate(self.failed, expected=altered),
                    )
        altered = controls()
        altered["extra"] = "a" * 64
        self.assert_code("control_rejected", lambda: self.validate(self.failed, expected=altered))

    def test_report_controls_cannot_substitute_caller_pins(self):
        altered = copy.deepcopy(self.failed)
        altered["controls"]["source_inventory_sha256"] = "b" * 64
        self.assert_code("control_rejected", lambda: self.validate(altered))

    def test_enum_objects_exact_keys_and_only_declared_values(self):
        cases = (
            ("diagnostic", "diagnostic_rejected"),
            ("frozen_stages", "stage_rejected"),
            ("frozen_component_diagnostics", "component_rejected"),
        )
        for field, code in cases:
            for action in ("missing", "extra", "value", "type"):
                altered = copy.deepcopy(self.failed)
                target = altered[field]
                if action == "missing":
                    target.pop(next(iter(target)))
                elif action == "extra":
                    target["extra"] = "not_reported"
                else:
                    target[next(iter(target))] = "SENSITIVE_RAW_PATH" if action == "value" else 1
                with self.subTest(field=field, action=action):
                    self.assert_code(code, lambda altered=altered: self.validate(altered))

    def test_counts_reject_bool_negative_float_and_over_cap(self):
        for field in ("counts", "frozen_origin_counts"):
            key = next(iter(self.failed[field]))
            maximum = self.spec["COUNT_LIMITS"][key] if field == "counts" else 196_608
            for invalid in (True, False, -1, 0.0, maximum + 1, "0"):
                altered = copy.deepcopy(self.failed)
                altered[field][key] = invalid
                with self.subTest(field=field, invalid=invalid):
                    expected = (
                        "report_number_rejected" if type(invalid) is float else "count_rejected"
                    )
                    self.assert_code(expected, lambda altered=altered: self.validate(altered))

    def test_counts_require_all_exact_nine_and_ten_fields(self):
        for field in ("counts", "frozen_origin_counts"):
            for extra in (True, False):
                altered = copy.deepcopy(self.failed)
                if extra:
                    altered[field]["unknown"] = 0
                else:
                    altered[field].pop(next(iter(altered[field])))
                self.assert_code("count_rejected", lambda altered=altered: self.validate(altered))

    def test_every_nonclaim_requires_exact_false(self):
        for flag in self.spec["FALSE_FLAGS"]:
            for invalid in (True, 0, 1, None, "false"):
                altered = copy.deepcopy(self.failed)
                altered[flag] = invalid
                with self.subTest(flag=flag, invalid=invalid):
                    self.assert_code(
                        "nonclaim_rejected", lambda altered=altered: self.validate(altered)
                    )

    def test_failed_result_requires_check_function_and_error_evidence(self):
        for fields in (
            ("captured_check",),
            ("captured_function",),
            ("probe_error_code", "exception_class"),
        ):
            altered = copy.deepcopy(self.failed)
            for field in fields:
                altered["diagnostic"][field] = "not_reported"
            self.assert_code(
                "failed_evidence_missing", lambda altered=altered: self.validate(altered)
            )
        altered = copy.deepcopy(self.failed)
        altered["diagnostic"]["exception_class"] = "not_reported"
        altered["diagnostic"]["probe_error_code"] = "module_origin_rejected"
        self.assert_code("failed_evidence_missing", lambda: self.validate(altered))

    def test_unexpected_origin_exception_is_separate_from_predicate_rejection(self):
        altered = copy.deepcopy(self.failed)
        altered["diagnostic"].update(
            captured_check="module_origins_unexpected",
            exception_class="ValueError",
            probe_error_code="not_reported",
        )
        self.assertEqual(self.validate(altered), altered)
        altered["diagnostic"]["exception_class"] = "ProbeError"
        self.assert_code("failed_evidence_missing", lambda: self.validate(altered))

    def test_passed_result_requires_exact_six_three_twelve(self):
        for key, expected in (
            ("rank_workers_completed", 6),
            ("invocations_completed", 3),
            ("origin_observations_completed", 12),
        ):
            for changed in (expected - 1, expected + 1):
                altered = copy.deepcopy(self.passed)
                altered["counts"][key] = changed
                code = "count_rejected" if changed > expected else "passed_evidence_rejected"
                self.assert_code(code, lambda altered=altered: self.validate(altered))

    def test_passed_result_cannot_contain_check_or_failure_capture(self):
        for key in ("captured_check", "captured_function", "probe_error_code", "exception_class"):
            altered = copy.deepcopy(self.passed)
            altered["diagnostic"][key] = self.spec["DIAGNOSTIC_ENUMS"][key][1]
            self.assert_code(
                "passed_evidence_rejected", lambda altered=altered: self.validate(altered)
            )
        for key in (
            "failed_outcomes_observed",
            "failure_messages_received",
            "failure_records_retained",
        ):
            altered = copy.deepcopy(self.passed)
            altered["counts"][key] = 1
            self.assert_code(
                "passed_evidence_rejected", lambda altered=altered: self.validate(altered)
            )

    def test_mutated_spec_is_revalidated_on_every_report(self):
        for change in ("cap", "required_flag", "keys", "diagnostic", "extra"):
            altered = copy.deepcopy(self.spec)
            if change == "cap":
                altered["COUNT_LIMITS"]["need_checks_observed"] = 2_000_001
            elif change == "required_flag":
                altered["FALSE_FLAGS"] = altered["FALSE_FLAGS"][1:]
            elif change == "keys":
                altered["REPORT_KEYS"] = altered["REPORT_KEYS"][1:]
            elif change == "diagnostic":
                altered["DIAGNOSTIC_ENUMS"]["captured_check"] = ("not_reported", "/private/raw")
            else:
                altered["EXTRA"] = True
            with self.subTest(change=change), self.assertRaises(contract.ValidationError):
                self.validate(self.failed, spec=altered)

    def test_validation_error_constructor_cannot_echo_unknown_input(self):
        for invalid in ("SECRET_EXCEPTION", None, [], 4):
            error = contract.ValidationError(invalid)
            self.assertEqual(error.code, "validation_failed")
            self.assertEqual(str(error), "validation_failed")


if __name__ == "__main__":
    unittest.main()
