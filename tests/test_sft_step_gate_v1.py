"""Deterministic in-memory step-gate tests; no ML, data, or transport access."""

from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "_lumi_synthetic_step_gate_v1", ROOT / "src/lumi/posttraining_sft_step_gate_v1.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class StepGateTests(unittest.TestCase):
    def setUp(self):
        self.events = []

    def reserve(self, from_step, to_step):
        self.events.append(("reserve", from_step, to_step))

    def complete(self, to_step):
        self.events.append(("complete", to_step))

    def synchronize(self, stage, to_step):
        self.events.append(("sync", stage, to_step))

    def interrupt(self):
        self.events.append(("interrupt",))

    def arguments(self, **changes):
        arguments = {"rank": 0, "world_size": 2, "reserve_step": self.reserve,
                     "complete_step": self.complete, "synchronize": self.synchronize,
                     "interrupt": self.interrupt, "from_step": 0, "stop_step": 2}
        arguments.update(changes)
        return arguments

    def gate(self, **changes):
        return module.StepGate(**self.arguments(**changes))

    def run_step(self, gate, to_step):
        with gate.step(to_step):
            self.events.append(("body", to_step))

    def assert_error(self, code, function, *args, **kwargs):
        with self.assertRaises(module.StepGateError) as caught:
            function(*args, **kwargs)
        self.assertEqual(caught.exception.args, (code,))
        self.assertEqual(caught.exception.code, code)
        self.assertNotIn("SYNTHETIC-PRIVATE", repr(caught.exception))

    def test_coordinator_orders_reservation_work_update_completion(self):
        gate = self.gate()
        self.run_step(gate, 1)
        self.assertEqual(self.events, [
            ("reserve", 0, 1), ("sync", "reserved", 1), ("body", 1),
            ("sync", "updated", 1), ("complete", 1), ("sync", "completed", 1)])
        self.assertEqual(gate.current_step, 1)
        self.assertFalse(gate.failed)
        self.assertIsNone(gate.failure_stage)

    def test_noncoordinator_never_reserves_but_waits_before_work(self):
        gate = self.gate(rank=1)
        self.run_step(gate, 1)
        self.assertEqual(self.events, [
            ("sync", "reserved", 1), ("body", 1), ("sync", "updated", 1),
            ("complete", 1), ("sync", "completed", 1)])
        self.assertEqual(gate.current_step, 1)

    def test_two_sequential_steps_have_distinct_exact_reservations(self):
        gate = self.gate()
        self.run_step(gate, 1)
        self.run_step(gate, 2)
        self.assertEqual([event for event in self.events if event[0] == "reserve"],
                         [("reserve", 0, 1), ("reserve", 1, 2)])
        self.assertEqual(gate.current_step, 2)

    def test_resume_starts_at_one_without_repeating_first_step(self):
        gate = self.gate(from_step=1, stop_step=2)
        self.run_step(gate, 2)
        self.assertEqual(self.events[0], ("reserve", 1, 2))
        self.assertEqual(gate.current_step, 2)

    def test_single_rank_still_requires_all_confirmation_stages(self):
        gate = self.gate(world_size=1, stop_step=1)
        self.run_step(gate, 1)
        self.assertEqual([event[1] for event in self.events if event[0] == "sync"],
                         ["reserved", "updated", "completed"])

    def test_cursor_advances_only_after_completed_parent_confirmation(self):
        observed = []
        gate = self.gate()

        def synchronize(stage, to_step):
            observed.append((stage, gate.current_step))
            self.synchronize(stage, to_step)

        gate._synchronize = synchronize
        with gate.step(1):
            self.assertEqual(gate.current_step, 0)
        self.assertEqual(observed, [("reserved", 0), ("updated", 0), ("completed", 0)])
        self.assertEqual(gate.current_step, 1)

    def test_constructor_rejects_scope_types_and_callbacks(self):
        changes = ({"rank": True}, {"rank": -1}, {"rank": 2}, {"world_size": True},
                   {"world_size": 0}, {"world_size": 13}, {"from_step": True},
                   {"from_step": -1}, {"from_step": 2}, {"stop_step": True},
                   {"stop_step": 0}, {"stop_step": 3}, {"stop_step": 1.0},
                   {"reserve_step": None}, {"complete_step": None},
                   {"synchronize": None}, {"interrupt": None})
        for changed in changes:
            with self.subTest(changed=changed):
                self.assert_error("invalid_arguments", self.gate, **changed)
        self.assertEqual(self.events, [])

    def test_readonly_public_status_properties(self):
        gate = self.gate()
        self.assertEqual(gate.rank, 0)
        self.assertEqual(gate.world_size, 2)
        for name, value in (("rank", 1), ("world_size", 1), ("current_step", 2), ("failed", False),
                            ("failure_stage", "idle"), ("failure_code", None)):
            with self.subTest(name=name), self.assertRaises(AttributeError):
                setattr(gate, name, value)

    def test_reinitialization_poison_existing_cursor_without_reset(self):
        gate = self.gate()
        self.run_step(gate, 1)
        self.assert_error("invalid_arguments", gate.__init__, **self.arguments())
        self.assertEqual(gate.current_step, 1)
        self.assertTrue(gate.failed)
        self.assert_error("terminal_failure", self.run_step, gate, 2)
        self.assertEqual(self.events.count(("interrupt",)), 1)

    def test_step_requires_exact_next_integer(self):
        for to_step in (True, False, None, -1, 0, 2, 3, 1.0, "1"):
            with self.subTest(to_step=to_step):
                self.events.clear()
                gate = self.gate()
                self.assert_error("step_mismatch", self.run_step, gate, to_step)
                self.assertEqual(self.events, [("interrupt",)])
                self.assertEqual(gate.current_step, 0)

    def test_step_cannot_exceed_frozen_stop(self):
        gate = self.gate(stop_step=1)
        self.run_step(gate, 1)
        self.events.clear()
        self.assert_error("step_mismatch", self.run_step, gate, 2)
        self.assertEqual(self.events, [("interrupt",)])
        self.assertEqual(gate.current_step, 1)

    def test_completed_step_cannot_be_replayed(self):
        gate = self.gate()
        self.run_step(gate, 1)
        self.events.clear()
        self.assert_error("step_mismatch", self.run_step, gate, 1)
        self.assertEqual(self.events, [("interrupt",)])

    def test_reserve_failure_prevents_sync_and_body(self):
        def reserve(from_step, to_step):
            self.reserve(from_step, to_step)
            raise RuntimeError("SYNTHETIC-PRIVATE-RESERVE")

        gate = self.gate(reserve_step=reserve)
        self.assert_error("reserve_failed", self.run_step, gate, 1)
        self.assertEqual(self.events, [("reserve", 0, 1), ("interrupt",)])
        self.assertEqual(gate.failure_stage, "reserve")
        self.assertEqual(gate.current_step, 0)

    def test_reserved_sync_failure_prevents_body_on_each_rank(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                self.events.clear()

                def synchronize(stage, to_step):
                    self.synchronize(stage, to_step)
                    raise RuntimeError("SYNTHETIC-PRIVATE-PEER")

                gate = self.gate(rank=rank, synchronize=synchronize)
                self.assert_error("synchronize_failed", self.run_step, gate, 1)
                self.assertNotIn(("body", 1), self.events)
                self.assertEqual(gate.failure_stage, "reserved")
                self.assertEqual(gate.current_step, 0)

    def test_updated_sync_failure_prevents_any_completion(self):
        def synchronize(stage, to_step):
            self.synchronize(stage, to_step)
            if stage == "updated":
                raise RuntimeError("SYNTHETIC-PRIVATE-UPDATE")

        gate = self.gate(synchronize=synchronize)
        self.assert_error("synchronize_failed", self.run_step, gate, 1)
        self.assertEqual(self.events, [("reserve", 0, 1), ("sync", "reserved", 1),
                                      ("body", 1), ("sync", "updated", 1), ("interrupt",)])
        self.assertEqual(gate.current_step, 0)
        self.assertEqual(gate.failure_stage, "updated")

    def test_complete_failure_does_not_commit_local_cursor(self):
        def complete(to_step):
            self.complete(to_step)
            raise RuntimeError("SYNTHETIC-PRIVATE-COMPLETE")

        gate = self.gate(complete_step=complete)
        self.assert_error("complete_failed", self.run_step, gate, 1)
        self.assertIn(("sync", "updated", 1), self.events)
        self.assertNotIn(("sync", "completed", 1), self.events)
        self.assertEqual(gate.current_step, 0)
        self.assertEqual(gate.failure_stage, "complete")

    def test_final_parent_confirmation_failure_keeps_old_cursor_without_retry(self):
        def synchronize(stage, to_step):
            self.synchronize(stage, to_step)
            if stage == "completed":
                raise RuntimeError("SYNTHETIC-PRIVATE-PARENT")

        gate = self.gate(synchronize=synchronize)
        self.assert_error("synchronize_failed", self.run_step, gate, 1)
        self.assertIn(("complete", 1), self.events)
        self.assertEqual(gate.current_step, 0)
        self.assertEqual(gate.failure_stage, "completed")
        before = list(self.events)
        self.assert_error("terminal_failure", self.run_step, gate, 1)
        self.assertEqual(self.events, before)

    def test_body_exception_interrupts_without_completion(self):
        gate = self.gate()
        with self.assertRaises(module.StepGateError) as caught:
            with gate.step(1):
                raise RuntimeError("SYNTHETIC-PRIVATE-BODY")
        self.assertEqual(caught.exception.args, ("body_failed",))
        self.assertEqual(gate.failure_stage, "work")
        self.assertNotIn(("complete", 1), self.events)
        self.assertEqual(gate.current_step, 0)

    def test_keyboard_interrupt_is_terminal_not_success(self):
        gate = self.gate()
        with self.assertRaises(module.StepGateError) as caught:
            with gate.step(1):
                raise KeyboardInterrupt
        self.assertEqual(caught.exception.args, ("body_failed",))
        self.assertEqual(self.events.count(("interrupt",)), 1)
        self.assertEqual(gate.current_step, 0)

    def test_non_none_reservation_response_rejected(self):
        gate = self.gate(reserve_step=lambda from_step, to_step: True)
        self.assert_error("reserve_failed", self.run_step, gate, 1)
        self.assertEqual(self.events, [("interrupt",)])

    def test_non_none_sync_response_rejected_at_each_stage(self):
        for rejected in ("reserved", "updated", "completed"):
            with self.subTest(stage=rejected):
                self.events.clear()

                def synchronize(stage, to_step, rejected=rejected):
                    self.synchronize(stage, to_step)
                    return False if stage == rejected else None

                gate = self.gate(synchronize=synchronize)
                self.assert_error("synchronize_failed", self.run_step, gate, 1)
                self.assertEqual(gate.failure_stage, rejected)
                self.assertEqual(gate.current_step, 0)

    def test_non_none_complete_response_rejected(self):
        gate = self.gate(complete_step=lambda to_step: 0)
        self.assert_error("complete_failed", self.run_step, gate, 1)
        self.assertEqual(gate.current_step, 0)
        self.assertNotIn(("sync", "completed", 1), self.events)

    def test_swallowed_reservation_reentry_poison_outer_context(self):
        gate = self.gate()

        def reserve(from_step, to_step):
            self.reserve(from_step, to_step)
            self.assert_error("reentry", self.run_step, gate, to_step)

        gate._reserve_step = reserve
        self.assert_error("terminal_failure", self.run_step, gate, 1)
        self.assertEqual(gate.failure_code, "reentry")
        self.assertEqual(gate.failure_stage, "reserve")
        self.assertNotIn(("body", 1), self.events)

    def test_swallowed_body_reentry_prevents_updated_barrier(self):
        gate = self.gate()
        with self.assertRaises(module.StepGateError) as caught:
            with gate.step(1):
                self.assert_error("reentry", self.run_step, gate, 1)
        self.assertEqual(caught.exception.args, ("terminal_failure",))
        self.assertNotIn(("sync", "updated", 1), self.events)
        self.assertEqual(gate.current_step, 0)

    def test_swallowed_sync_reentry_prevents_cursor_commit(self):
        gate = self.gate()

        def synchronize(stage, to_step):
            self.synchronize(stage, to_step)
            if stage == "completed":
                self.assert_error("reentry", self.run_step, gate, to_step)

        gate._synchronize = synchronize
        self.assert_error("terminal_failure", self.run_step, gate, 1)
        self.assertEqual(gate.failure_stage, "completed")
        self.assertEqual(gate.failure_code, "reentry")
        self.assertEqual(gate.current_step, 0)

    def test_swallowed_complete_reinitialization_is_terminal(self):
        gate = self.gate()

        def complete(to_step):
            self.complete(to_step)
            self.assert_error("invalid_arguments", gate.__init__, **self.arguments())

        gate._complete_step = complete
        self.assert_error("terminal_failure", self.run_step, gate, 1)
        self.assertEqual(gate.failure_stage, "complete")
        self.assertEqual(gate.current_step, 0)

    def test_process_owner_mismatch_interrupts_before_any_work(self):
        gate = self.gate()
        with patch.object(module.os, "getpid", return_value=os.getpid() + 1):
            self.assert_error("owner_mismatch", self.run_step, gate, 1)
        self.assertEqual(self.events, [("interrupt",)])
        self.assertEqual(gate.current_step, 0)

    def test_thread_owner_mismatch_interrupts_before_any_work(self):
        gate = self.gate()
        with patch.object(module.threading, "get_ident", return_value=-1):
            self.assert_error("owner_mismatch", self.run_step, gate, 1)
        self.assertEqual(self.events, [("interrupt",)])

    def test_interrupt_failure_is_suppressed_and_not_retried(self):
        def interrupt():
            self.interrupt()
            raise RuntimeError("SYNTHETIC-PRIVATE-INTERRUPT")

        gate = self.gate(interrupt=interrupt)
        self.assert_error("step_mismatch", self.run_step, gate, 2)
        self.assert_error("terminal_failure", self.run_step, gate, 1)
        self.assertEqual(self.events, [("interrupt",)])
        self.assertEqual(gate.failure_code, "step_mismatch")

    def test_context_exit_does_not_swallow_first_failure_evidence(self):
        gate = self.gate()
        with self.assertRaises(module.StepGateError):
            with gate.step(1):
                with gate.step(1):
                    self.fail("nested body ran")
        self.assertEqual(gate.failure_code, "reentry")
        self.assertEqual(gate.failure_stage, "work")
        self.assertEqual(self.events.count(("interrupt",)), 1)

    def test_second_step_failure_retains_first_committed_cursor(self):
        gate = self.gate()
        self.run_step(gate, 1)
        with self.assertRaises(module.StepGateError):
            with gate.step(2):
                raise RuntimeError("SYNTHETIC-PRIVATE-SECOND-STEP")
        self.assertEqual(gate.current_step, 1)
        self.assertEqual(gate.failure_stage, "work")
        self.assertEqual(self.events.count(("reserve", 1, 2)), 1)
        self.assertNotIn(("complete", 2), self.events)

    def test_repr_and_unknown_error_are_raw_text_free(self):
        gate = self.gate()
        self.assertEqual(repr(gate), "<StepGate private>")
        self.assertEqual(module.StepGateError("SYNTHETIC-PRIVATE").args,
                         ("invalid_arguments",))


if __name__ == "__main__":
    unittest.main()
