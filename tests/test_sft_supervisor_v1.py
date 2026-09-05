"""Pure synthetic supervisor tests; no data files, ML imports, or worker execution."""

from __future__ import annotations

import copy
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import unittest


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "src/lumi" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


supervision = _load("synthetic_supervisor", "posttraining_sft_supervisor_v1.py")
launch = _load("synthetic_launch", "posttraining_sft_launch_contract_v1.py")


class Clock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now


def synthetic_inputs():
    counts = {
        "sequential_invocations": 3,
        "global_optimizer_steps": 4,
        "rank_optimizer_steps": 8,
        "logical_sample_events": 8,
        "maximum_unique_train_records": 8,
        "rank_local_input_loads": 6,
        "maximum_encoded_index_rows": 48,
        "maximum_selected_binary_bytes": 48 * 32 * 8,
        "maximum_selected_index_bytes": 48 * 176,
        "includes_checkpoint_hashing_or_final_digest": False,
        "is_total_io_or_cost_budget": False,
    }
    caps = {
        "maximum_combined_runtime_seconds": 60,
        "maximum_cost_microusd": 1000,
        "total_host_hourly_rate_microusd": 60_000,
        "maximum_host_ram_bytes": 4096,
        "maximum_per_gpu_memory_bytes": 2048,
        "maximum_private_write_bytes": 1024,
        "maximum_checkpoint_read_bytes": 4096,
        "maximum_encoded_binary_read_bytes": counts["maximum_selected_binary_bytes"],
        "maximum_encoded_index_read_bytes": counts["maximum_selected_index_bytes"],
        "maximum_log_bytes": 256,
        "maximum_invocations": 3,
        "maximum_global_optimizer_steps": 4,
        "maximum_external_network_bytes": 0,
        "maximum_inference_calls": 0,
    }
    return caps, counts


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.caps, self.counts = synthetic_inputs()
        self.clock = Clock()
        self.controller = self.make()

    def make(self):
        return supervision.Supervisor(self.caps, self.counts, world_size=2, clock_ns=self.clock)

    def begin(self):
        self.controller.begin_invocation("reference", 0, 2, False)

    def assert_code(self, code, callback, *args, **kwargs):
        with self.assertRaises(supervision.SupervisorError) as caught:
            callback(*args, **kwargs)
        self.assertEqual(str(caught.exception), code)
        self.assertEqual(caught.exception.args, (code,))
        self.assertIn(code, supervision.ERROR_CODES)

    def steps(self, start, stop):
        for step in range(start, stop):
            self.controller.reserve_optimizer_step(step, step + 1)
            self.controller.finish_optimizer_step(step + 1)

    def complete_one(self, item):
        run, start, stop, resume = item
        self.controller.begin_invocation(run, start, stop, resume)
        self.controller.observe_memory(256, [64, 65])
        self.steps(start, stop)
        self.controller.finish_invocation(stop)

    def test_caps_match_existing_compiler(self):
        self.assertEqual(supervision.CAP_FIELDS, launch.CAP_FIELDS)
        launch.validate_caps(self.caps, self.counts)

    def test_exact_three_invocations_complete_four_global_steps(self):
        for item in supervision.SEQUENCE:
            self.complete_one(item)
        summary = self.controller.summary()
        self.assertEqual(summary["state"], "completed")
        self.assertEqual(summary["invocations_started"], 3)
        self.assertEqual(summary["invocations_completed"], 3)
        self.assertEqual(summary["global_optimizer_steps_reserved"], 4)
        self.assertEqual(summary["global_optimizer_steps_completed"], 4)
        self.assertEqual(summary["logical_sample_events_reserved"], 8)

    def test_caps_missing_or_extra_keys_rejected(self):
        for caps in ({}, {**self.caps, "unknown": 1}):
            self.assert_code("caps_schema", supervision.Supervisor, caps, self.counts, world_size=2)

    def test_counts_missing_or_extra_keys_rejected(self):
        for counts in ({}, {**self.counts, "raw_text": "SYNTHETIC-SECRET"}):
            self.assert_code("counts_schema", supervision.Supervisor, self.caps, counts, world_size=2)

    def test_every_positive_cap_rejects_bad_numeric_types(self):
        for field in supervision.CAP_FIELDS:
            for bad in (True, False, 0, -1, 1.0, float("nan"), "1", None, 2**63):
                caps = {**self.caps, field: bad}
                with self.subTest(field=field, bad=bad):
                    self.assert_code("integer_required", supervision.Supervisor, caps, self.counts, world_size=2)

    def test_count_numeric_types_rejected(self):
        for field in supervision.COUNT_FIELDS[:-2]:
            counts = {**self.counts, field: True}
            self.assert_code("integer_required", supervision.Supervisor, self.caps, counts, world_size=2)

    def test_count_boolean_flags_are_strict(self):
        for field in supervision.COUNT_FIELDS[-2:]:
            for bad in (0, True, None):
                counts = {**self.counts, field: bad}
                self.assert_code("counts_mismatch", supervision.Supervisor, self.caps, counts, world_size=2)

    def test_each_geometry_relation_is_checked(self):
        for field in supervision.COUNT_FIELDS[:-2]:
            counts = {**self.counts, field: self.counts[field] + 1}
            self.assert_code("counts_mismatch", supervision.Supervisor, self.caps, counts, world_size=2)

    def test_operation_caps_are_exact_not_boolean(self):
        for field in supervision.CAP_EXTRA_FIELDS:
            for bad in (True, False, -1, 99):
                caps = {**self.caps, field: bad}
                self.assert_code("operation_cap_changed", supervision.Supervisor, caps, self.counts, world_size=2)

    def test_world_size_bounded_and_integer(self):
        for bad in (True, 0, 1, 13, 2.0, "2"):
            self.assert_code("integer_required", supervision.Supervisor, self.caps, self.counts, world_size=bad)

    def test_cost_cap_rounds_up(self):
        self.caps["total_host_hourly_rate_microusd"] = 60_001
        self.assert_code("cost_cap_too_small", self.make)
        self.caps["maximum_cost_microusd"] = 1001
        self.make()

    def test_rank_multiplied_io_caps_enforced_at_construction(self):
        for field in ("maximum_encoded_binary_read_bytes", "maximum_encoded_index_read_bytes"):
            caps = {**self.caps, field: self.caps[field] - 1}
            self.assert_code("io_cap_too_small", supervision.Supervisor, caps, self.counts, world_size=2)

    def test_log_cap_cannot_exceed_private_cap(self):
        self.caps["maximum_log_bytes"] = 1025
        self.assert_code("log_cap_too_large", self.make)

    def test_input_mutation_cannot_expand_caps(self):
        self.caps["maximum_checkpoint_read_bytes"] = 999_999
        self.begin()
        self.assert_code("resource_exhausted", self.controller.reserve, 0, checkpoint_read_bytes=4097)

    def test_count_mutation_cannot_expand_rows(self):
        self.counts["maximum_encoded_index_rows"] = 999_999
        self.begin()
        self.assert_code("resource_exhausted", self.controller.reserve, 0, encoded_index_rows=49)

    def test_reservations_charge_all_categories_and_log_private_overlap(self):
        self.begin()
        self.controller.reserve(0, checkpoint_read_bytes=1, encoded_binary_read_bytes=2,
                                encoded_index_read_bytes=3, private_write_bytes=4,
                                log_bytes=5, encoded_index_rows=6)
        used = self.controller.summary()["resources_reserved"]
        self.assertEqual(used, dict(checkpoint_read_bytes=1, encoded_binary_read_bytes=2,
                                    encoded_index_read_bytes=3, private_write_bytes=9,
                                    log_bytes=5, encoded_index_rows=6))

    def test_reservations_share_budget_between_ranks(self):
        self.begin()
        self.controller.reserve(0, checkpoint_read_bytes=2048)
        self.controller.reserve(1, checkpoint_read_bytes=2048)
        self.assert_code("resource_exhausted", self.controller.reserve, 0, checkpoint_read_bytes=1)

    def test_atomic_failure_does_not_partially_charge(self):
        self.begin()
        before = self.controller.summary()["resources_reserved"]
        self.assert_code("resource_exhausted", self.controller.reserve, 0,
                         checkpoint_read_bytes=1, log_bytes=257)
        self.assertEqual(self.controller.summary()["resources_reserved"], before)
        self.assertEqual(self.controller.summary()["resource_reservations"], 0)

    def test_each_resource_cap_exhaustion_is_sticky(self):
        for field in supervision.RESOURCE_FIELDS:
            self.controller = self.make()
            self.begin()
            cap = (self.counts["maximum_encoded_index_rows"] if field == "encoded_index_rows"
                   else self.caps["maximum_" + field])
            self.assert_code("resource_exhausted", self.controller.reserve, 0, **{field: cap + 1})
            self.assert_code("resource_exhausted", self.controller.checkpoint)

    def test_log_bytes_can_exhaust_private_cap(self):
        self.begin()
        self.controller.reserve(0, private_write_bytes=1000)
        self.assert_code("resource_exhausted", self.controller.reserve, 0, log_bytes=25)

    def test_invalid_resource_amounts_never_refund(self):
        for bad in (-1, True, 1.0, "SYNTHETIC-SECRET", 2**63, None, 0):
            self.controller = self.make()
            self.begin()
            self.controller.reserve(0, checkpoint_read_bytes=10)
            self.assert_code("resource_amount", self.controller.reserve, 0, checkpoint_read_bytes=bad)
            self.assertEqual(self.controller.summary()["resources_reserved"]["checkpoint_read_bytes"], 10)

    def test_unknown_empty_resource_requests_rejected(self):
        self.begin()
        self.assert_code("resource_schema", self.controller.reserve, 0, network_bytes=1)
        self.controller = self.make()
        self.begin()
        self.assert_code("resource_schema", self.controller.reserve, 0)

    def test_rank_types_and_bounds_checked(self):
        for bad in (-1, 2, True, "SYNTHETIC-SECRET", 0.0):
            self.controller = self.make()
            self.begin()
            self.assert_code("rank_invalid", self.controller.reserve, bad, checkpoint_read_bytes=1)

    def test_concurrent_reservations_cannot_overdraw(self):
        self.begin()
        def reserve(_):
            try:
                self.controller.reserve(0, checkpoint_read_bytes=64)
                return True
            except supervision.SupervisorError:
                return False
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            successes = sum(executor.map(reserve, range(80)))
        self.assertEqual(successes, 64)
        self.assertEqual(self.controller.summary()["resources_reserved"]["checkpoint_read_bytes"], 4096)
        self.assertEqual(self.controller.summary()["state"], "failed")

    def test_operations_require_active_invocation(self):
        self.assert_code("no_active_invocation", self.controller.reserve, 0, checkpoint_read_bytes=1)

    def test_unexpected_first_invocation_rejected(self):
        self.assert_code("invocation_boundary", self.controller.begin_invocation, "resumed", 0, 1, False)

    def test_invocation_fields_have_exact_types(self):
        for item in (("reference", False, 2, False), ("reference", 0, 2.0, False),
                     ("reference", 0, 2, 0), ("SYNTHETIC-SECRET", 0, 2, False)):
            self.controller = self.make()
            self.assert_code("invocation_boundary", self.controller.begin_invocation, *item)

    def test_overlapping_invocations_fail_and_do_not_retry(self):
        self.begin()
        self.assert_code("invocation_overlap", self.controller.begin_invocation, "reference", 0, 2, False)
        self.assertEqual(self.controller.summary()["invocations_started"], 1)

    def test_interrupted_invocation_is_consumed_and_bytes_not_refunded(self):
        self.begin()
        self.controller.reserve(0, checkpoint_read_bytes=37)
        self.controller.reserve_optimizer_step(0, 1)
        self.assert_code("interrupted", self.controller.interrupt)
        self.assert_code("interrupted", self.controller.begin_invocation, "reference", 0, 2, False)
        state = self.controller.summary()
        self.assertEqual(state["invocations_started"], 1)
        self.assertEqual(state["invocations_completed"], 0)
        self.assertEqual(state["global_optimizer_steps_reserved"], 1)
        self.assertEqual(state["global_optimizer_steps_completed"], 0)
        self.assertEqual(state["resources_reserved"]["checkpoint_read_bytes"], 37)

    def test_completed_sequence_cannot_be_restarted(self):
        for item in supervision.SEQUENCE:
            self.complete_one(item)
        self.assert_code("already_completed", self.controller.begin_invocation, "reference", 0, 2, False)
        self.assertEqual(self.controller.summary()["invocations_started"], 3)

    def test_finish_cannot_skip_optimizer_work(self):
        self.begin()
        self.assert_code("invocation_boundary", self.controller.finish_invocation, 2)

    def test_optimizer_reserve_cannot_skip_steps_or_cross_boundary(self):
        for start, stop in ((0, 2), (1, 2), (0, True), (False, 1), (0, 3)):
            self.controller = self.make()
            self.begin()
            self.assert_code("optimizer_boundary", self.controller.reserve_optimizer_step, start, stop)

    def test_optimizer_reservation_precedes_acknowledgement(self):
        self.begin()
        self.assert_code("optimizer_not_pending", self.controller.finish_optimizer_step, 1)

    def test_pending_optimizer_reservation_blocks_second(self):
        self.begin()
        self.controller.reserve_optimizer_step(0, 1)
        self.assert_code("optimizer_pending", self.controller.reserve_optimizer_step, 0, 1)

    def test_pending_optimizer_blocks_invocation_completion(self):
        self.begin()
        self.controller.reserve_optimizer_step(0, 1)
        self.assert_code("optimizer_pending", self.controller.finish_invocation, 2)

    def test_optimizer_ack_boundary_is_exact(self):
        self.begin()
        self.controller.reserve_optimizer_step(0, 1)
        self.assert_code("optimizer_boundary", self.controller.finish_optimizer_step, True)

    def test_resumed_invocation_cannot_replace_resume_with_fresh(self):
        self.complete_one(supervision.SEQUENCE[0])
        self.complete_one(supervision.SEQUENCE[1])
        self.assert_code("invocation_boundary", self.controller.begin_invocation, "resumed", 0, 2, False)

    def test_monotonic_cost_uses_ceil_integer_arithmetic(self):
        self.clock.now = 1
        self.controller.checkpoint()
        self.assertEqual(self.controller.summary()["cost_microusd_last_observed"], 1)
        self.clock.now = 1_000_000_000
        self.controller.checkpoint()
        self.assertEqual(self.controller.summary()["cost_microusd_last_observed"], 17)
        self.clock.now = 59_999_999_999
        self.controller.checkpoint()
        self.assertEqual(self.controller.summary()["cost_microusd_last_observed"], 1000)

    def test_deadline_exhausted_at_exact_boundary_before_operation(self):
        self.begin()
        self.clock.now = 60_000_000_000
        self.assert_code("deadline_exhausted", self.controller.reserve, 0, checkpoint_read_bytes=1)
        self.assertEqual(self.controller.summary()["resource_reservations"], 0)

    def test_total_deadline_includes_inter_invocation_gap(self):
        self.complete_one(supervision.SEQUENCE[0])
        self.clock.now = 60_000_000_000
        self.assert_code("deadline_exhausted", self.controller.begin_invocation, *supervision.SEQUENCE[1])
        self.assertEqual(self.controller.summary()["invocations_started"], 1)

    def test_clock_regression_is_sticky(self):
        self.clock.now = 10
        self.controller.checkpoint()
        self.clock.now = 9
        self.assert_code("clock_regressed", self.controller.checkpoint)
        self.clock.now = 11
        self.assert_code("clock_regressed", self.controller.checkpoint)

    def test_invalid_clock_values_fail_closed(self):
        for bad in (True, -1, 0.0, float("nan"), "SYNTHETIC-SECRET", None, 2**63):
            self.clock.now = bad
            self.assert_code("clock_invalid", self.make)

    def test_clock_exception_text_is_not_copied(self):
        def broken():
            raise RuntimeError("SYNTHETIC-SECRET")
        self.assert_code("clock_failed", supervision.Supervisor, self.caps, self.counts,
                         world_size=2, clock_ns=broken)

    def test_summary_is_snapshot_and_does_not_poll_clock(self):
        self.clock.now = 99_000_000_000
        self.assertEqual(self.controller.summary()["elapsed_nanoseconds_last_observed"], 0)

    def test_memory_observations_are_required_per_invocation(self):
        self.complete_one(supervision.SEQUENCE[0])
        self.controller.begin_invocation(*supervision.SEQUENCE[1])
        self.steps(0, 1)
        self.assert_code("memory_observation_missing", self.controller.finish_invocation, 1)

    def test_host_memory_overage_is_fail_closed(self):
        self.begin()
        self.assert_code("host_memory_exhausted", self.controller.observe_memory, 4097, [0, 0])
        self.assertEqual(self.controller.summary()["maximum_host_ram_bytes_observed"], 4097)

    def test_any_gpu_memory_overage_is_fail_closed(self):
        self.begin()
        self.assert_code("gpu_memory_exhausted", self.controller.observe_memory, 0, [0, 2049])

    def test_memory_boundaries_inclusive(self):
        self.begin()
        self.controller.observe_memory(4096, [2048, 2048])
        self.assertEqual(self.controller.summary()["memory_observations"], 1)

    def test_memory_schema_requires_all_ranks_and_strict_numbers(self):
        for host, gpu in ((True, [0, 0]), (0, [0]), (0, [0, 0, 0]), (0, (0, 0)),
                          (0, [True, 0]), (0, [-1, 0]), (0, [0.0, 0])):
            self.controller = self.make()
            self.begin()
            self.assert_code("memory_schema", self.controller.observe_memory, host, gpu)

    def test_summary_is_copy_and_contains_only_aggregate_allowlisted_fields(self):
        self.begin()
        original = self.controller.summary()
        changed = copy.deepcopy(original)
        changed["resources_reserved"]["checkpoint_read_bytes"] = 99999
        self.assertEqual(self.controller.summary(), original)
        encoded = json.dumps(original)
        self.assertNotIn("SYNTHETIC-SECRET", encoded)
        self.assertFalse(original["executable_dispatch_enabled"])
        self.assertFalse(original["os_memory_enforcement"])
        self.assertFalse(original["os_network_enforcement"])
        self.assertFalse(original["multiprocess_shared_state"])
        self.assertFalse(original["production_eligible"])

    def test_exception_constructor_never_retains_arbitrary_text(self):
        self.assertEqual(str(supervision.SupervisorError("SYNTHETIC-SECRET")), "resource_schema")


if __name__ == "__main__":
    unittest.main()
