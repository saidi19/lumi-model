"""Synthetic v2 metadata predebits, with isolated inherited supervisor checks."""

from __future__ import annotations

import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def inherited_tests():
    path = ROOT / "tests/test_sft_supervisor_v1.py"
    source = path.read_text(encoding="utf-8")
    source = source.replace("posttraining_sft_supervisor_v1.py",
                            "posttraining_sft_supervisor_v2.py")
    source = source.replace("posttraining_sft_launch_contract_v1.py",
                            "posttraining_sft_launch_contract_v2.py")
    cap_line = '        "maximum_checkpoint_read_bytes": 4096,\n'
    if source.count(cap_line) != 1:
        raise AssertionError("synthetic_inherited_cap_changed")
    metadata_line = '        "maximum_encoded_metadata_read_bytes": 9,\n'
    source = source.replace(cap_line, cap_line + metadata_line)
    # Exact inherited full-summary expectation gains only the unreserved category.
    expected = "encoded_index_read_bytes=3, private_write_bytes=9,"
    if source.count(expected) != 1:
        raise AssertionError("synthetic_inherited_resource_expectation_changed")
    source = source.replace(expected, expected + " encoded_metadata_read_bytes=0,")
    namespace = types.ModuleType("_synthetic_supervisor_v2_inherited")
    namespace.__file__ = str(path)
    exec(compile(source, str(path), "exec"), namespace.__dict__)
    return namespace


inherited = inherited_tests()
supervision = inherited.supervision
TestInheritedSupervisor = inherited.SupervisorTests


class MetadataSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.caps, self.counts = inherited.synthetic_inputs()
        self.controller = self.make()

    def make(self):
        return supervision.Supervisor(self.caps, self.counts, world_size=2, clock_ns=lambda: 0)

    def begin(self):
        self.controller.begin_invocation("reference", 0, 2, False)

    def reject(self, code, operation, *args, **kwargs):
        with self.assertRaises(supervision.SupervisorError) as caught:
            operation(*args, **kwargs)
        self.assertEqual(caught.exception.args, (code,))

    def test_new_summary_and_required_cap_without_old_fallback(self):
        self.assertEqual(self.controller.summary()["schema_version"],
                         "lumi.sft-supervisor-summary.v0.2")
        self.assertEqual(self.controller.summary()["resources_reserved"][
            "encoded_metadata_read_bytes"], 0)
        del self.caps["maximum_encoded_metadata_read_bytes"]
        self.reject("caps_schema", self.make)

    def test_completed_cap_rejects_zero_none_bool_fraction_negative_and_overflow(self):
        for value in (0, None, True, False, -1, 1.0, "1", 2**63):
            with self.subTest(value=value):
                self.caps["maximum_encoded_metadata_read_bytes"] = value
                self.reject("integer_required", self.make)

    def test_all_ranks_share_exact_positive_metadata_cap(self):
        self.begin()
        self.controller.reserve(0, encoded_metadata_read_bytes=4)
        self.controller.reserve(1, encoded_metadata_read_bytes=5)
        self.assertEqual(self.controller.summary()["resources_reserved"][
            "encoded_metadata_read_bytes"], 9)
        self.reject("resource_exhausted", self.controller.reserve, 0, encoded_metadata_read_bytes=1)
        self.assertEqual(self.controller.summary()["resources_reserved"][
            "encoded_metadata_read_bytes"], 9)

    def test_mixed_reservation_is_atomic_when_metadata_or_old_resource_fails(self):
        for amounts in ({"encoded_metadata_read_bytes": 10, "private_write_bytes": 1},
                        {"encoded_metadata_read_bytes": 1, "private_write_bytes": 1025}):
            self.controller = self.make()
            self.begin()
            self.controller.reserve(0, encoded_metadata_read_bytes=2, checkpoint_read_bytes=3)
            before = self.controller.summary()["resources_reserved"]
            self.reject("resource_exhausted", self.controller.reserve, 1, **amounts)
            self.assertEqual(self.controller.summary()["resources_reserved"], before)

    def test_integer_addition_cannot_wrap_and_existing_charge_is_preserved(self):
        self.caps["maximum_encoded_metadata_read_bytes"] = 2**63 - 1
        self.controller = self.make()
        self.begin()
        self.controller.reserve(0, encoded_metadata_read_bytes=2**63 - 1)
        self.reject("resource_exhausted", self.controller.reserve, 1, encoded_metadata_read_bytes=1)
        self.assertEqual(self.controller.summary()["resources_reserved"][
            "encoded_metadata_read_bytes"], 2**63 - 1)

    def test_interrupt_and_invalid_negative_amount_cannot_refund(self):
        for interrupt in (True, False):
            self.controller = self.make()
            self.begin()
            self.controller.reserve(0, encoded_metadata_read_bytes=7)
            if interrupt:
                self.reject("interrupted", self.controller.interrupt)
            else:
                self.reject("resource_amount", self.controller.reserve, 0,
                            encoded_metadata_read_bytes=-7)
            self.assertEqual(self.controller.summary()["resources_reserved"][
                "encoded_metadata_read_bytes"], 7)
            self.assertEqual(self.controller.summary()["state"], "failed")

    def test_metadata_persists_across_fresh_invocation_boundaries(self):
        for index, (run, start, stop, resume) in enumerate(supervision.SEQUENCE):
            self.controller.begin_invocation(run, start, stop, resume)
            self.controller.reserve(index % 2, encoded_metadata_read_bytes=3)
            self.controller.observe_memory(1, [0, 0])
            for step in range(start, stop):
                self.controller.reserve_optimizer_step(step, step + 1)
                self.controller.finish_optimizer_step(step + 1)
            self.controller.finish_invocation(stop)
        summary = self.controller.summary()
        self.assertEqual(summary["resources_reserved"]["encoded_metadata_read_bytes"], 9)
        self.assertEqual(summary["state"], "completed")
        self.assertFalse(summary["production_eligible"])

    def test_metadata_predebits_do_not_alias_other_read_or_log_resources(self):
        self.begin()
        self.controller.reserve(0, encoded_metadata_read_bytes=4, checkpoint_read_bytes=2,
                                encoded_binary_read_bytes=3, encoded_index_read_bytes=5,
                                log_bytes=7, private_write_bytes=11)
        used = self.controller.summary()["resources_reserved"]
        self.assertEqual(used["encoded_metadata_read_bytes"], 4)
        self.assertEqual(used["checkpoint_read_bytes"], 2)
        self.assertEqual(used["encoded_binary_read_bytes"], 3)
        self.assertEqual(used["encoded_index_read_bytes"], 5)
        self.assertEqual(used["log_bytes"], 7)
        self.assertEqual(used["private_write_bytes"], 18)

    def test_caps_and_summary_copies_cannot_enlarge_metadata_budget(self):
        self.caps["maximum_encoded_metadata_read_bytes"] = 999
        self.begin()
        summary = self.controller.summary()
        summary["resources_reserved"]["encoded_metadata_read_bytes"] = -999
        self.reject("resource_exhausted", self.controller.reserve, 0,
                    encoded_metadata_read_bytes=10)
        self.assertEqual(self.controller.summary()["resources_reserved"][
            "encoded_metadata_read_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
