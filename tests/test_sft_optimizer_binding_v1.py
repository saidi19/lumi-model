"""Metadata-only validator tests; no mocked Torch execution or tensor values.

These stdlib containers exercise only _optimizer_binding's compatibility hash.
They do not satisfy WorkerRun admission/runtime and never run its training loop.
"""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_lumi_synthetic_optimizer_binding_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
before_import = set(sys.modules)
worker_run = importlib.import_module(f"{PACKAGE}.posttraining_sft_worker_run_v1")
IMPORTED_MODULES = set(sys.modules) - before_import


class ParameterMetadata:
    def __init__(self, shape=(2, 3), dtype="torch.float32", requires_grad=True):
        self.shape, self.dtype, self.requires_grad = shape, dtype, requires_grad

    @property
    def data(self):
        raise AssertionError("tensor_values_must_not_be_read")

    def state_dict(self):
        raise AssertionError("tensor_values_must_not_be_read")


class ModelMetadata:
    def __init__(self, parameters):
        self.parameters_metadata = parameters

    def named_parameters(self):
        return iter(self.parameters_metadata)

    def state_dict(self):
        raise AssertionError("model_state_must_not_be_read")


class OptimizerMetadata:
    def __init__(self, groups):
        self.param_groups = groups

    @property
    def state(self):
        raise AssertionError("optimizer_state_must_not_be_read")

    def state_dict(self):
        raise AssertionError("optimizer_state_must_not_be_read")


class OptimizerBindingTests(unittest.TestCase):
    def setUp(self):
        self.weight, self.bias = ParameterMetadata(), ParameterMetadata(shape=(2,))
        self.model = ModelMetadata([("module.weight", self.weight), ("module.bias", self.bias)])
        self.group = {"params": [self.weight, self.bias], "lr": 0.01, "betas": (0.9, 0.95),
                      "eps": 1e-8, "weight_decay": 0.1, "amsgrad": False, "maximize": False,
                      "foreach": None, "capturable": False, "differentiable": False,
                      "fused": False, "decoupled_weight_decay": True}
        self.optimizer = OptimizerMetadata([self.group])

    def binding(self):
        return worker_run._optimizer_binding(self.model, self.optimizer)

    def assert_rejected(self):
        with self.assertRaises(worker_run.WorkerRunError) as caught:
            self.binding()
        self.assertEqual(caught.exception.args, ("runtime_rejected",))
        self.assertNotIn("SYNTHETIC-PRIVATE", repr(caught.exception))

    def test_import_does_not_import_ml(self):
        forbidden = {"torch", "transformers", "numpy", "safetensors", "tokenizers"}
        self.assertFalse({name.split(".")[0] for name in IMPORTED_MODULES} & forbidden)

    def test_opaque_stable_hash_independent_of_fresh_object_identity(self):
        first = self.binding()
        self.assertRegex(first, r"^[0-9a-f]{64}$")
        self.setUp()
        self.assertEqual(self.binding(), first)
        self.assertEqual(self.group["params"], [self.weight, self.bias])

    def test_group_parameter_order_changes_binding(self):
        first = self.binding()
        self.group["params"].reverse()
        self.assertNotEqual(self.binding(), first)

    def test_model_parameter_order_changes_binding(self):
        first = self.binding()
        self.model.parameters_metadata.reverse()
        self.assertNotEqual(self.binding(), first)

    def test_group_order_and_membership_changes_binding(self):
        first = self.binding()
        self.group["params"] = [self.weight]
        self.optimizer.param_groups.append({**self.group, "params": [self.bias]})
        second = self.binding()
        self.assertNotEqual(first, second)
        self.optimizer.param_groups.reverse()
        self.assertNotEqual(second, self.binding())

    def test_names_shapes_dtypes_and_trainability_change_binding(self):
        for change in ("name", "shape", "dtype", "trainability"):
            with self.subTest(change=change):
                self.setUp()
                before = self.binding()
                if change == "name":
                    self.model.parameters_metadata[0] = ("module.other_weight", self.weight)
                elif change == "shape":
                    self.weight.shape = (3, 2)
                elif change == "dtype":
                    self.weight.dtype = "torch.float64"
                else:
                    self.weight.requires_grad = False
                self.assertNotEqual(self.binding(), before)

    def test_all_plain_settings_are_bound_with_dict_order_irrelevant(self):
        self.group["metadata"] = {"label": "SYNTHETIC-PRIVATE", "values": [None, True, 7, 1.5]}
        before = self.binding()
        self.optimizer.param_groups = [dict(reversed(list(self.group.items())))]
        self.assertEqual(self.binding(), before)
        self.group["metadata"]["values"][-1] = 2.5
        self.assertNotEqual(self.binding(), before)

    def test_numeric_and_boolean_unknown_setting_values_are_not_collapsed(self):
        self.group["metadata"] = {"flag": True}
        first = self.binding()
        self.group["metadata"]["flag"] = 1
        self.assertNotEqual(self.binding(), first)

    def test_optimizer_hyperparameter_and_switch_changes_are_bound(self):
        changes = {"lr": 0.02, "betas": (0.8, 0.9), "eps": 1e-6, "weight_decay": 0.2,
                   "amsgrad": True, "maximize": True, "foreach": False, "capturable": True,
                   "differentiable": True, "fused": None, "decoupled_weight_decay": False}
        for key, value in changes.items():
            with self.subTest(key=key):
                self.setUp()
                first = self.binding()
                self.group[key] = value
                self.assertNotEqual(self.binding(), first)

    def test_duplicate_name_parameter_reference_and_dangling_group_rejected(self):
        for change in ("name", "model_alias", "within_group", "between_groups", "dangling"):
            with self.subTest(change=change):
                self.setUp()
                if change == "name":
                    self.model.parameters_metadata[1] = ("module.weight", self.bias)
                elif change == "model_alias":
                    self.model.parameters_metadata[1] = ("module.bias", self.weight)
                elif change == "within_group":
                    self.group["params"].append(self.weight)
                elif change == "between_groups":
                    self.optimizer.param_groups.append({"params": [self.weight]})
                else:
                    self.group["params"].append(ParameterMetadata())
                self.assert_rejected()

    def test_missing_trainable_and_no_trainable_parameters_rejected(self):
        self.group["params"] = [self.weight]
        self.assert_rejected()
        self.weight.requires_grad = self.bias.requires_grad = False
        self.assert_rejected()

    def test_optional_frozen_parameter_and_empty_group_are_bound(self):
        self.bias.requires_grad = False
        first = self.binding()
        self.group["params"] = [self.weight]
        self.assertNotEqual(self.binding(), first)
        second = self.binding()
        self.optimizer.param_groups.append({"params": [], "lr": 0.01})
        self.assertNotEqual(self.binding(), second)

    def test_malformed_model_metadata_rejected(self):
        for malformed in ([], [("weight",)], [["weight", self.weight]],
                          [(True, self.weight)], [("", self.weight)],
                          [("a" * 257, self.weight)], [("bad/name", self.weight)]):
            with self.subTest(case=str(type(malformed))):
                self.model.parameters_metadata = malformed
                self.assert_rejected()

    def test_malformed_shape_dtype_and_trainability_rejected(self):
        changes = [("shape", [2, 3]), ("shape", (True, 2)), ("shape", (-1, 2)),
                   ("shape", (2**31,)), ("shape", (1,) * 9), ("dtype", "float32"),
                   ("dtype", "torch." + "x" * 49), ("requires_grad", 1)]
        for key, value in changes:
            with self.subTest(key=key):
                self.setUp()
                setattr(self.weight, key, value)
                self.assert_rejected()

    def test_scalar_shape_valid_and_shape_dimensions_are_bound(self):
        self.weight.shape = ()
        first = self.binding()
        self.weight.shape = (0,)
        self.assertNotEqual(self.binding(), first)

    def test_boolean_numeric_settings_and_numeric_switches_rejected(self):
        changes = [(key, True) for key in ("lr", "eps", "weight_decay", "initial_lr")]
        changes += [(key, 0) for key in ("amsgrad", "maximize", "capturable", "differentiable",
                                         "fused", "foreach", "decoupled_weight_decay")]
        changes += [("betas", (True, 0.95)), ("betas", (0.9, 1.0)), ("betas", (0.9,))]
        for key, value in changes:
            with self.subTest(key=key):
                self.setUp()
                self.group[key] = value
                self.assert_rejected()

    def test_nonfinite_objects_noncanonical_strings_and_large_integers_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf"), object(), b"private",
                      {1, 2}, 2**64, "x" * 257, "é"):
            with self.subTest(kind=type(value).__name__):
                self.group["metadata"] = value
                self.assert_rejected()

    def test_group_containers_and_parameter_sequences_are_exact(self):
        class DictSubclass(dict):
            pass

        class ListSubclass(list):
            pass

        for groups in ((), [], [DictSubclass(self.group)], [None], [dict(self.group, params=())],
                       ListSubclass([self.group]), [dict(self.group, params=ListSubclass())]):
            with self.subTest(kind=type(groups).__name__):
                self.optimizer.param_groups = groups
                self.assert_rejected()

    def test_setting_subclasses_and_invalid_keys_rejected(self):
        class IntSubclass(int):
            pass

        class DictSubclass(dict):
            pass

        for value in (IntSubclass(1), DictSubclass(a=1), {True: 1}, {"": 1},
                      {"x" * 65: 1}, {"bad/key": 1}):
            with self.subTest(kind=type(value).__name__):
                self.group["metadata"] = value
                self.assert_rejected()

    def test_setting_depth_and_sequence_length_and_cycle_are_bounded(self):
        deep = 1
        for _ in range(7):
            deep = [deep]
        cycle = []
        cycle.append(cycle)
        for value in (deep, cycle, list(range(129)), {f"k{i}": i for i in range(65)}):
            with self.subTest(kind=type(value).__name__):
                self.group["metadata"] = value
                self.assert_rejected()

    def test_group_count_and_aggregate_setting_nodes_are_bounded(self):
        self.optimizer.param_groups.extend({"params": []} for _ in range(64))
        self.assert_rejected()
        extra = {f"k{i}": [0] * 128 for i in range(63)}
        self.optimizer.param_groups = [self.group] + [{"params": [], **extra} for _ in range(3)]
        self.assert_rejected()

    def test_iterator_exceptions_use_fixed_code(self):
        def broken():
            raise RuntimeError("SYNTHETIC-PRIVATE")
            yield

        self.model.parameters_metadata = broken()
        self.assert_rejected()


if __name__ == "__main__":
    unittest.main()
