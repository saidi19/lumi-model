"""Pure guard tests only: no Torch, model, data, or training execution."""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_synthetic_worker_step_guard"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
steps = importlib.import_module(PACKAGE + ".posttraining_sft_worker_step_v1")
gates = importlib.import_module(PACKAGE + ".posttraining_sft_step_gate_v1")


class WorkerStepGuardTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.gate = gates.StepGate(
            rank=0, world_size=2, from_step=0, stop_step=2,
            reserve_step=lambda a, b: self.events.append(("reserve", a, b)),
            complete_step=lambda step: self.events.append(("complete", step)),
            synchronize=lambda stage, step: self.events.append((stage, step)),
            interrupt=lambda: self.events.append(("interrupt",)),
        )

    def worker(self):
        return steps.WorkerStep(gate=self.gate, world_size=2)

    def perform(self, worker, **changes):
        options = {"to_step": 1, "learning_rate": 0.01, "max_grad_norm": 1.0}
        options.update(changes)
        with patch.dict(sys.modules, {"torch": None}):
            return worker.perform(None, None, None, **options)

    def test_constructor_has_no_runtime_or_tensor_action(self):
        self.assertIn("admitted objects only", repr(self.worker()))
        self.assertEqual(self.events, [])
        self.assertFalse(self.gate.failed)

    def test_constructor_rejects_world_and_device_mismatch(self):
        for change in ({"world_size": True}, {"world_size": 1}, {"device_type": "mps"},
                       {"gate": object()}):
            with self.subTest(change=tuple(change)):
                options = {"gate": self.gate, "world_size": 2, **change}
                with self.assertRaises(steps.WorkerStepError):
                    steps.WorkerStep(**options)
        self.assertEqual(self.events, [])

    def test_no_first_ml_import_and_no_model_touch(self):
        worker = self.worker()
        with self.assertRaises(steps.WorkerStepError) as caught:
            self.perform(worker)
        self.assertEqual(caught.exception.code, "runtime_rejected")
        self.assertEqual(self.events, [("reserve", 0, 1), ("reserved", 1), ("interrupt",)])
        self.assertTrue(self.gate.failed)
        self.assertEqual(self.gate.current_step, 0)
        self.assertEqual(worker.failure_stage, "runtime")

    def test_failure_cannot_be_retried(self):
        worker = self.worker()
        for _ in range(2):
            with self.assertRaises(steps.WorkerStepError):
                self.perform(worker)
        self.assertEqual(self.events.count(("reserve", 0, 1)), 1)
        self.assertEqual(self.events.count(("interrupt",)), 1)

    def test_reinitialization_poisoned_without_new_reservation(self):
        worker = self.worker()
        with self.assertRaises(steps.WorkerStepError):
            worker.__init__(gate=self.gate, world_size=2)
        self.assertEqual(self.events, [("interrupt",)])
        self.assertTrue(self.gate.failed)

    def test_owner_mismatch_is_terminal_before_reservation(self):
        worker = self.worker()
        with patch.object(steps.os, "getpid", return_value=-1):
            with self.assertRaises(steps.WorkerStepError):
                self.perform(worker)
        self.assertEqual(self.events, [("interrupt",)])

    def test_failed_gate_cannot_be_attached_to_a_working_step(self):
        with self.assertRaises(gates.StepGateError):
            self.gate._fail("body_failed")
        with self.assertRaises(steps.WorkerStepError):
            self.perform(self.worker())
        self.assertEqual(self.events, [("interrupt",)])

    def test_nonfinite_and_bool_hyperparameters_rejected_before_tensor_work(self):
        for options in ({"learning_rate": True}, {"learning_rate": float("nan")},
                        {"max_grad_norm": float("inf")}, {"max_grad_norm": 0}):
            with self.subTest(options=tuple(options)):
                self.setUp()
                worker = self.worker()
                with self.assertRaises(steps.WorkerStepError) as caught:
                    self.perform(worker, **options)
                self.assertEqual(caught.exception.code, "invalid_arguments")
                self.assertEqual(worker.failure_stage, "reservation")

    def test_mode_contract_has_no_silent_fallback(self):
        for options in ({"strategy": "unknown"}, {"precision": "fp16"},
                        {"gradient_sync_mode": "never"}):
            with self.subTest(options=tuple(options)):
                self.setUp()
                worker = self.worker()
                with self.assertRaises(steps.WorkerStepError) as caught:
                    self.perform(worker, **options)
                self.assertEqual(caught.exception.code, "invalid_arguments")

    def test_cuda_fp32_rejected_before_runtime_initialization(self):
        worker = steps.WorkerStep(gate=self.gate, world_size=2, device_type="cuda")
        with self.assertRaises(steps.WorkerStepError) as caught:
            self.perform(worker, precision="fp32")
        self.assertEqual(caught.exception.code, "invalid_arguments")

    def optimizer_binding(self, model_parameters, optimizer_parameters):
        class Optimizer:
            param_groups = [{"params": optimizer_parameters}]

        torch = types.SimpleNamespace(optim=types.SimpleNamespace(Optimizer=Optimizer))
        model = types.SimpleNamespace(parameters=lambda: iter(model_parameters))
        self.worker()._optimizer_binding(torch, model, Optimizer())

    def test_optimizer_binding_permits_frozen_parameter_inclusion_or_omission(self):
        trainable = types.SimpleNamespace(requires_grad=True)
        frozen = types.SimpleNamespace(requires_grad=False)
        self.optimizer_binding([trainable, frozen], [trainable])
        self.optimizer_binding([trainable, frozen], [trainable, frozen])
        self.assertEqual(self.events, [])

    def test_optimizer_binding_compares_identity_not_parameter_equality(self):
        class Parameter:
            requires_grad = True

            def __eq__(self, _other):
                raise AssertionError("implicit_tensor_equality")

        parameter = Parameter()
        self.optimizer_binding([parameter], [parameter])
        self.assertEqual(self.events, [])

    def test_optimizer_binding_rejects_duplicates(self):
        parameter = types.SimpleNamespace(requires_grad=True)
        with self.assertRaises(steps.WorkerStepError):
            self.optimizer_binding([parameter], [parameter, parameter])
        self.assertEqual(self.events, [("interrupt",)])

    def test_optimizer_binding_rejects_missing_trainable_parameter(self):
        first, second = (types.SimpleNamespace(requires_grad=True) for _ in range(2))
        with self.assertRaises(steps.WorkerStepError):
            self.optimizer_binding([first, second], [first])

    def test_optimizer_binding_rejects_foreign_parameter(self):
        first, second = (types.SimpleNamespace(requires_grad=True) for _ in range(2))
        with self.assertRaises(steps.WorkerStepError):
            self.optimizer_binding([first], [first, second])

    def test_bad_step_is_rejected_by_gate_before_runtime(self):
        worker = self.worker()
        with self.assertRaises(steps.WorkerStepError):
            self.perform(worker, to_step=2)
        self.assertEqual(self.events, [("interrupt",)])

    def test_swallowed_reentry_cannot_release_outer_work(self):
        worker = self.worker()

        def reserve(_start, _end):
            try:
                self.perform(worker)
            except steps.WorkerStepError:
                pass

        self.gate._reserve_step = reserve
        with self.assertRaises(steps.WorkerStepError):
            self.perform(worker)
        self.assertEqual(self.events, [("interrupt",)])
        self.assertTrue(self.gate.failed)
        self.assertEqual(worker.failure_code, "reentry")

    def test_unknown_error_text_is_not_retained(self):
        error = steps.WorkerStepError("SYNTHETIC-PRIVATE-TEXT")
        self.assertEqual(error.args, ("step_failed",))
        self.assertNotIn("PRIVATE", repr(error))


if __name__ == "__main__":
    unittest.main()
