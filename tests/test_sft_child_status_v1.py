"""Synthetic byte fixtures only: no actual procfs, children, sockets or data."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "_test_sft_child_status_v1", ROOT / "src" / "lumi" / "posttraining_sft_child_status_v1.py"
)
status = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = status
SPEC.loader.exec_module(status)

EXPECTED = {
    "expected_pid": 12345, "expected_ppid": 23456, "expected_uid": 34567, "expected_gid": 45678,
}
FIELDS = {
    b"Name": b"fixed-inert", b"State": b"S (sleeping)", b"Tgid": b"12345",
    b"Pid": b"12345", b"PPid": b"23456", b"TracerPid": b"0", b"Threads": b"1",
    b"Uid": b"34567\t34567\t34567\t34567", b"Gid": b"45678\t45678\t45678\t45678",
    b"NSpid": b"12345", b"NStgid": b"12345", b"CapInh": b"0000000000000000",
    b"CapPrm": b"0000000000000000", b"CapEff": b"0000000000000000",
    b"CapAmb": b"0000000000000000", b"CapBnd": b"ffffffffffffffff", b"NoNewPrivs": b"1",
}


def snapshot(changes=None, omitted=(), extra=b""):
    fields = dict(FIELDS)
    fields.update(changes or {})
    return b"".join(
        key + b":\t" + value + b"\n" for key, value in fields.items() if key not in omitted
    ) + extra


class ChildStatusTests(unittest.TestCase):
    def reject(self, raw, code="child_status_shape_rejected", **changes):
        expected = dict(EXPECTED)
        expected.update(changes)
        with self.assertRaises(status.StatusError) as caught:
            status.parse_status(raw, **expected)
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(str(caught.exception), code)
        self.assertEqual(caught.exception.args, (code,))
        self.assertIsNone(caught.exception.__cause__)

    def test_accepts_exact_policy_without_retaining_identifiers_or_raw_values(self):
        report = status.parse_status(snapshot(), **EXPECTED)
        self.assertEqual(report["status_fields_checked"], 17)
        self.assertEqual(report["required_fields_checked"], 15)
        self.assertTrue(report["child_status_policy_passed"])
        self.assertTrue(report["current_capability_sets_empty"])
        self.assertFalse(report["capability_bounding_set_empty"])
        self.assertFalse(report["permanent_privilege_containment"])
        self.assertFalse(report["process_generation_bound"])
        self.assertFalse(report["execution_authorized"])
        self.assertTrue(all(type(value) in (bool, int) for value in report.values()))
        encoded = json.dumps(report)
        for forbidden in ("12345", "23456", "34567", "45678", "fixed-inert", "ffffffff"):
            self.assertNotIn(forbidden, encoded)

    def test_every_required_field_is_required(self):
        for key in status.REQUIRED_FIELDS:
            with self.subTest(key=key):
                self.reject(snapshot(omitted=(key,)))

    def test_every_duplicate_key_is_rejected_including_unknown_keys(self):
        for key, value in FIELDS.items():
            with self.subTest(key=key):
                self.reject(snapshot(extra=key + b":\t" + value + b"\n"))

    def test_unknown_normal_fields_are_allowed_without_retaining_values(self):
        report = status.parse_status(
            snapshot(extra=b"Groups:\t\nFuture_kernel_key:\topaque:not-retained\n"), **EXPECTED
        )
        self.assertEqual(report["status_fields_checked"], 19)
        self.assertNotIn("opaque", json.dumps(report))
        self.assertNotIn("Future_kernel_key", json.dumps(report))

    def test_invalid_raw_types_are_opaque(self):
        for raw in (None, "private raw", bytearray(snapshot()), memoryview(snapshot()), 5, True):
            with self.subTest(kind=type(raw).__name__):
                self.reject(raw)

    def test_raw_size_bounds_and_truncated_final_line(self):
        for raw in (b"", snapshot()[:-1], b"x" * (status.MAXIMUM_STATUS_BYTES + 1)):
            with self.subTest(length=len(raw)):
                self.reject(raw)
        prefix = snapshot()
        suffix = b"Padding:\t"
        maximum = prefix + suffix + b"x" * (
            status.MAXIMUM_STATUS_BYTES - len(prefix) - len(suffix) - 1
        ) + b"\n"
        self.assertEqual(len(maximum), status.MAXIMUM_STATUS_BYTES)
        self.assertTrue(status.parse_status(maximum, **EXPECTED)["child_status_policy_passed"])
        self.reject(maximum + b"\n")

    def test_line_count_bounds(self):
        additions = b"".join(
            b"Future_" + str(index).encode() + b":\tignored\n"
            for index in range(status.MAXIMUM_STATUS_FIELDS - len(FIELDS))
        )
        report = status.parse_status(snapshot(extra=additions), **EXPECTED)
        self.assertEqual(report["status_fields_checked"], status.MAXIMUM_STATUS_FIELDS)
        self.reject(snapshot(extra=additions + b"TooMany:\tignored\n"))

    def test_control_bytes_and_non_ascii_are_rejected_everywhere(self):
        for byte in tuple(range(0, 9)) + (11, 12, 13, 14, 31, 127, 128, 255):
            with self.subTest(byte=byte):
                self.reject(snapshot({b"Name": b"private" + bytes((byte,))}))

    def test_malformed_lines_and_unknown_keys_are_rejected(self):
        for extra in (
            b"\n", b"plain text\n", b":\tvalue\n", b"Bad-Key:\tvalue\n",
            b"9Bad:\tvalue\n", b" Bad:\tvalue\n", b"Bad :\tvalue\n",
            b"Bad:value\n", b"Bad:\n", b"A" * 65 + b":\tvalue\n",
        ):
            with self.subTest(extra=extra):
                self.reject(snapshot(extra=extra))

    def test_expected_configuration_is_strict_positive_bounded_integer(self):
        for key in EXPECTED:
            maximum = status.MAXIMUM_ID if key.endswith(("uid", "gid")) else status.MAXIMUM_PID
            for value in (0, -1, True, False, None, "12345", 1.0, maximum + 1, 10 ** 100):
                with self.subTest(key=key, value_type=type(value).__name__):
                    self.reject(
                        snapshot(), "child_status_configuration_rejected", **{key: value}
                    )

    def test_identity_mismatch_is_rejected(self):
        for key in (b"Pid", b"Tgid", b"PPid", b"NSpid", b"NStgid"):
            for value in (b"0", b"1", b"54321"):
                with self.subTest(key=key, value=value):
                    self.reject(snapshot({key: value}), "child_status_identity_rejected")

    def test_identity_numbers_are_canonical_decimal_with_exact_cardinality(self):
        for key in (b"Pid", b"Tgid", b"PPid", b"NSpid", b"NStgid"):
            for value in (
                b"", b"+12345", b"-1", b"012345", b"1.0", b"0x12345", b"12345 12345",
                b"12345 text", b"2147483648", b"12345678901234567890",
            ):
                with self.subTest(key=key, value=value):
                    self.reject(snapshot({key: value}))

    def test_nested_namespace_and_wrong_single_namespace_are_rejected(self):
        for key in (b"NSpid", b"NStgid"):
            with self.subTest(key=key):
                self.reject(snapshot({key: b"12345\t1"}))
                self.reject(snapshot({key: b"1"}), "child_status_identity_rejected")

    def test_uid_gid_must_have_exactly_four_numeric_values(self):
        for key, number in ((b"Uid", b"34567"), (b"Gid", b"45678")):
            for value in (
                b"", number, b" ".join([number] * 3), b" ".join([number] * 5),
                b" ".join([number] * 3 + [b"private"]),
                b" ".join([number] * 3 + [b"4294967295"]),
            ):
                with self.subTest(key=key, value=value):
                    self.reject(snapshot({key: value}))

    def test_each_uid_gid_role_must_match_expected_nonzero_identity(self):
        for key, number in ((b"Uid", b"34567"), (b"Gid", b"45678")):
            for index in range(4):
                for changed in (b"0", b"1"):
                    values = [number] * 4
                    values[index] = changed
                    with self.subTest(key=key, index=index, changed=changed):
                        self.reject(
                            snapshot({key: b"\t".join(values)}), "child_status_identity_rejected"
                        )

    def test_every_current_capability_bit_is_rejected(self):
        for key in (b"CapInh", b"CapPrm", b"CapEff", b"CapAmb"):
            for bit in range(64):
                with self.subTest(key=key, bit=bit):
                    self.reject(
                        snapshot({key: f"{1 << bit:016x}".encode()}),
                        "child_status_capability_rejected",
                    )

    def test_capabilities_require_exact_kernel_hex_shape(self):
        for key in status.CAPABILITY_FIELDS:
            for value in (
                b"", b"0", b"0" * 15, b"0" * 17, b"0x00000000000000",
                b"000000000000000g", b"-000000000000000", b"00000000 00000000",
                b"0000000000000000 private",
            ):
                with self.subTest(key=key, value=value):
                    self.reject(snapshot({key: value}))

    def test_bounding_set_empty_and_nonempty_never_claim_permanent_containment(self):
        for mask in (b"0000000000000000", b"0000000000000001", b"FFFFFFFFFFFFFFFF"):
            with self.subTest(mask=mask):
                report = status.parse_status(snapshot({b"CapBnd": mask}), **EXPECTED)
                self.assertEqual(report["capability_bounding_set_empty"], int(mask, 16) == 0)
                self.assertFalse(report["permanent_privilege_containment"])

    def test_tracer_threads_and_no_new_privs_must_satisfy_policy(self):
        for key, values in (
            (b"TracerPid", (b"1", b"12345")), (b"Threads", (b"0", b"2", b"100")),
            (b"NoNewPrivs", (b"0", b"2")),
        ):
            for value in values:
                with self.subTest(key=key, value=value):
                    self.reject(snapshot({key: value}), "child_status_policy_rejected")

    def test_tracer_threads_and_no_new_privs_reject_extra_tokens_or_invalid_numbers(self):
        for key in (b"TracerPid", b"Threads", b"NoNewPrivs"):
            for value in (b"", b"0 1", b"-1", b"true", b"01", b"2147483648"):
                with self.subTest(key=key, value=value):
                    self.reject(snapshot({key: value}))

    def test_irregular_spaces_and_tabs_are_accepted_without_weakening_values(self):
        report = status.parse_status(
            snapshot({
                b"Uid": b" 34567\t 34567 34567\t34567  ",
                b"CapEff": b" 0000000000000000\t",
            }),
            **EXPECTED,
        )
        self.assertTrue(report["child_status_policy_passed"])

    def test_status_error_normalizes_untrusted_codes(self):
        for value in ("private secret", "", None, 1, {}, [], True):
            with self.subTest(kind=type(value).__name__):
                error = status.StatusError(value)
                self.assertEqual(error.code, "child_status_policy_rejected")
                self.assertEqual(str(error), "child_status_policy_rejected")


class ControllerStatusTests(unittest.TestCase):
    def reject(self, raw, code="child_status_shape_rejected", **changes):
        expected = dict(EXPECTED)
        expected.update(changes)
        with self.assertRaises(status.StatusError) as caught:
            status.parse_controller_status(raw, **expected)
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(str(caught.exception), code)
        self.assertEqual(caught.exception.args, (code,))
        self.assertIsNone(caught.exception.__cause__)

    def test_controller_accepts_no_new_privs_zero_or_one_with_actual_boolean(self):
        for value in (b"0", b"1"):
            with self.subTest(value=value):
                report = status.parse_controller_status(
                    snapshot({b"NoNewPrivs": value}), **EXPECTED
                )
                self.assertTrue(report["controller_status_policy_passed"])
                self.assertNotIn("child_status_policy_passed", report)
                self.assertIs(report["no_new_privs_set"], value == b"1")
                self.assertFalse(report["permanent_privilege_containment"])
                self.assertFalse(report["process_generation_bound"])
                self.assertFalse(report["execution_authorized"])
                self.assertTrue(all(type(item) in (bool, int) for item in report.values()))

    def test_child_still_requires_no_new_privs_one(self):
        raw = snapshot({b"NoNewPrivs": b"0"})
        self.assertTrue(status.parse_controller_status(raw, **EXPECTED)[
            "controller_status_policy_passed"
        ])
        with self.assertRaises(status.StatusError) as caught:
            status.parse_status(raw, **EXPECTED)
        self.assertEqual(caught.exception.code, "child_status_policy_rejected")
        child = status.parse_status(snapshot(), **EXPECTED)
        self.assertNotIn("controller_status_policy_passed", child)

    def test_controller_rejects_no_new_privs_values_other_than_zero_or_one(self):
        self.reject(snapshot({b"NoNewPrivs": b"2"}), "child_status_policy_rejected")
        for value in (b"", b"00", b"01", b"-1", b"false", b"0 1"):
            with self.subTest(value=value):
                self.reject(snapshot({b"NoNewPrivs": value}))

    def test_controller_requires_expected_identity_in_every_identity_field(self):
        for key in (b"Pid", b"Tgid", b"PPid", b"NSpid", b"NStgid"):
            with self.subTest(key=key):
                self.reject(snapshot({key: b"1"}), "child_status_identity_rejected")
        for key, number in ((b"Uid", b"34567"), (b"Gid", b"45678")):
            for index in range(4):
                values = [number] * 4
                values[index] = b"0"
                with self.subTest(key=key, index=index):
                    self.reject(
                        snapshot({key: b" ".join(values)}), "child_status_identity_rejected"
                    )

    def test_controller_requires_single_thread_and_no_visible_tracer(self):
        for key, value in ((b"Threads", b"2"), (b"TracerPid", b"1")):
            with self.subTest(key=key):
                self.reject(snapshot({key: value}), "child_status_policy_rejected")

    def test_controller_rejects_visible_nested_pid_namespaces(self):
        for key in (b"NSpid", b"NStgid"):
            with self.subTest(key=key):
                self.reject(snapshot({key: b"12345 1"}))

    def test_controller_requires_no_current_capabilities_even_with_no_new_privs_zero(self):
        for key in (b"CapInh", b"CapPrm", b"CapEff", b"CapAmb"):
            with self.subTest(key=key):
                self.reject(
                    snapshot({key: b"0000000000000001", b"NoNewPrivs": b"0"}),
                    "child_status_capability_rejected",
                )

    def test_controller_preserves_missing_duplicate_invalid_and_size_rejection(self):
        for raw in (
            snapshot(omitted=(b"NoNewPrivs",)), snapshot(extra=b"Pid:\t12345\n"),
            snapshot({b"Name": b"secret\x00"}), snapshot()[:-1],
            b"x" * (status.MAXIMUM_STATUS_BYTES + 1),
        ):
            with self.subTest(length=len(raw)):
                self.reject(raw)
        self.reject(snapshot(), "child_status_configuration_rejected", expected_uid=0)


if __name__ == "__main__":
    unittest.main()
