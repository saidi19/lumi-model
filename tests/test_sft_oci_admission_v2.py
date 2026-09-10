"""Pure metadata and source-AST tests; no native runtime, Docker, network or ML.

Modules are imported under a fake package, never lumi.__init__. Verification
tests replace the runtime and inspect entry points with explicit inert doubles;
unchanged runtime, transport, resource and deadline predicates are AST-compared.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
V1_PATH = ROOT / "src/lumi/posttraining_sft_oci_admission_v1.py"
V2_PATH = ROOT / "src/lumi/posttraining_sft_oci_admission_v2.py"
V1_RAW, V2_RAW = V1_PATH.read_bytes(), V2_PATH.read_bytes()
V1_TREE, V2_TREE = ast.parse(V1_RAW), ast.parse(V2_RAW)
PACKAGE = "_lumi_synthetic_oci_admission_v2"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
before = set(sys.modules)
old = importlib.import_module(PACKAGE + ".posttraining_sft_oci_admission_v1")
new = importlib.import_module(PACKAGE + ".posttraining_sft_oci_admission_v2")
IMPORT_ADDITIONS = set(sys.modules) - before
PRIVATE_TEXT = "SYNTHETIC-PRIVATE-TEXT-NEVER-RETAIN"
CONTAINER_ID = "a" * 64
ABSENT = object()


def top(tree, name):
    return next(node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)


def same_ast(left, right):
    return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)


def configuration():
    environment = {
        "LUMI_RELEASE_LOCK_SHA256": old.LOCK_SHA256,
        "LUMI_RELEASE_BASE_INVENTORY_SHA256": old.BASE_SHA256,
        "LUMI_RELEASE_SOURCE_INVENTORY_SHA256": old.SOURCE_SHA256,
        "LUMI_RELEASE_SOURCE_COMMIT": old.SOURCE_COMMIT,
        "LUMI_WORKER_RUN_TORCH_CI_FIXTURE": "1", "CUDA_VISIBLE_DEVICES": "",
        "GLOO_SOCKET_IFNAME": "lo", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    }
    return {
        "User": "10001:10001", "WorkingDir": "/opt/lumi-release/source",
        "Entrypoint": [old.PYTHON, "-I", "-B"], "Cmd": [old.WRAPPER, "--synthetic-only"],
        "Env": [key + "=" + value for key, value in environment.items()],
        "Volumes": None, "Labels": None,
    }


def metadata():
    config = configuration()
    image = {"Id": old.IMAGE_ID, "Os": "linux", "Architecture": "amd64", "Config": config}
    container = {
        "Id": CONTAINER_ID, "Image": old.IMAGE_ID,
        "Config": {**copy.deepcopy(config), "Image": old.IMAGE_ID},
        "Path": old.PYTHON, "Args": ["-I", "-B", old.WRAPPER, "--synthetic-only"],
        "RestartCount": 0,
        "State": {"Status": "created", "Running": False, "Paused": False, "Restarting": False,
                  "OOMKilled": False, "Dead": False, "Pid": 0, "ExitCode": 0,
                  "StartedAt": "0001-01-01T00:00:00Z", "FinishedAt": "0001-01-01T00:00:00Z"},
        "HostConfig": {
            "ReadonlyRootfs": True, "Privileged": False, "AutoRemove": False,
            "NetworkMode": "none", "CapDrop": ["ALL"], "SecurityOpt": ["no-new-privileges"],
            "IpcMode": "private", "Tmpfs": dict(old.TMPFS),
            "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
            "Memory": 4294967296, "MemorySwap": 4294967296, "ShmSize": 67108864,
            "PidsLimit": 256, "NanoCpus": 2000000000,
            "Ulimits": [{"Name": "core", "Soft": 0, "Hard": 0},
                        {"Name": "nofile", "Soft": 1024, "Hard": 1024}],
        },
        "Mounts": [],
    }
    return image, container


def verify(module, image, container, *, exception=None):
    responses = {"image": module.canonical([image]), "container": module.canonical([container])}

    def inspect(kind, identifier, deadline):
        if identifier != (module.IMAGE_ID if kind == "image" else CONTAINER_ID) or deadline != 110:
            raise AssertionError("unexpected synthetic inspect")
        if exception is not None:
            raise exception
        return responses[kind]

    with patch.object(module, "_runtime", return_value=None), \
            patch.object(module, "_inspect", side_effect=inspect) as selected, \
            patch.object(module, "time", types.SimpleNamespace(monotonic=lambda: 100)):
        try:
            result = module.verify_created_oci(CONTAINER_ID, deadline=110)
        except module.OciAdmissionError as error:
            return error, selected.call_count
        return result, selected.call_count


class NormalizeOnlyAuthorizedEdits(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "_check_configuration_equal":
            return ast.parse(
                'need(canonical(original.get(key)) == canonical(selected.get(key)), '
                '"profile_rejected")', mode="eval",
            ).body
        if (isinstance(node.func, ast.Name) and node.func.id == "OciAdmissionError"
                and len(node.args) == 2 and isinstance(node.args[1], ast.Name)
                and node.args[1].id == "predicate"):
            node.args = node.args[:1]
        return node

    def visit_Assign(self, node):
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "predicate"):
            return None
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple)
                and [item.id for item in node.targets[0].elts]
                == ["failure", "result", "predicate"]):
            node.targets[0].elts = node.targets[0].elts[:2]
            node.value.elts = node.value.elts[:2]
        return self.generic_visit(node)


class FrozenContractTests(unittest.TestCase):
    def test_exact_frozen_v1_hash_and_size(self):
        self.assertEqual(len(V1_RAW), 15603)
        self.assertEqual(hashlib.sha256(V1_RAW).hexdigest(), new.FROZEN_V1_SHA256)
        self.assertEqual(new.FROZEN_V1_SHA256,
                         "a2f26771b22fedfe8a003bc3104c40f46477465b7f0ab2c50bee7ad3abb945f0")

    def test_all_original_functions_unchanged_except_authorized_edits(self):
        originals = {node.name: node for node in V1_TREE.body if isinstance(node, ast.FunctionDef)}
        changed = {"_configuration", "verify_created_oci"}
        for name, original in originals.items():
            with self.subTest(function=name):
                selected = copy.deepcopy(top(V2_TREE, name))
                if name in changed:
                    selected = NormalizeOnlyAuthorizedEdits().visit(selected)
                self.assertTrue(same_ast(original, selected))

    def test_original_software_binding_and_handoff_classes_are_ast_identical(self):
        for name in ("SoftwareBinding", "VerifiedOciHandoff"):
            with self.subTest(class_name=name):
                self.assertTrue(same_ast(top(V1_TREE, name), top(V2_TREE, name)))

    def test_every_existing_constant_ast_is_identical(self):
        original = {node.targets[0].id: node for node in V1_TREE.body
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
        selected = {node.targets[0].id: node for node in V2_TREE.body
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
        for name, node in original.items():
            with self.subTest(constant=name):
                self.assertTrue(same_ast(node, selected[name]))

    def test_software_binding_values_domain_and_identity_are_identical(self):
        previous, selected = old.SoftwareBinding(), new.SoftwareBinding()
        self.assertEqual(old.SCHEMA, new.SCHEMA)
        self.assertEqual(previous.__slots__, selected.__slots__)
        self.assertEqual({key: getattr(previous, key) for key in previous.__slots__},
                         {key: getattr(selected, key) for key in selected.__slots__})
        self.assertEqual(previous.source_identity_sha256(), selected.source_identity_sha256())

    def test_exact_runtime_transport_and_budget_controls_unchanged(self):
        for name in ("_runtime", "_remaining", "_inspect", "_json", "_container"):
            with self.subTest(function=name):
                self.assertTrue(same_ast(top(V1_TREE, name), top(V2_TREE, name)))
        self.assertEqual((new.MAXIMUM_INSPECT_BYTES, new.MAXIMUM_INSPECT_SECONDS,
                          new.MAXIMUM_HANDOFF_SECONDS), (65536, 5, 15))

    def test_original_comparison_is_first_and_ast_identical(self):
        original = next(node for node in ast.walk(top(V1_TREE, "_configuration"))
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == "need" and isinstance(node.args[0], ast.Compare)
                        and isinstance(node.args[0].left, ast.Call)
                        and isinstance(node.args[0].left.func, ast.Name)
                        and node.args[0].left.func.id == "canonical")
        helper = top(V2_TREE, "_configuration_equal")
        self.assertIsInstance(helper.body[0], ast.If)
        self.assertTrue(same_ast(original.args[0], helper.body[0].test))
        self.assertEqual(len(helper.body), 2)

    def test_exception_ast_has_both_presence_checks_and_exact_direction(self):
        helper = top(V2_TREE, "_labels_null_empty_exception")
        expected = ast.parse(
            'key == "Labels" and "Labels" in original and "Labels" in selected '
            'and original["Labels"] is None and type(selected["Labels"]) is dict '
            'and len(selected["Labels"]) == 0', mode="eval",
        ).body
        self.assertTrue(same_ast(helper.body[-1].value, expected))

    def test_equality_loop_order_and_fixed_predicate_map_are_exact(self):
        previous = next(node for node in ast.walk(top(V1_TREE, "_configuration"))
                        if isinstance(node, ast.For))
        selected = next(node for node in ast.walk(top(V2_TREE, "_configuration"))
                        if isinstance(node, ast.For))
        keys = ast.literal_eval(previous.iter)
        self.assertTrue(same_ast(previous.iter, selected.iter))
        self.assertEqual(tuple(new.CONFIGURATION_PREDICATES), keys)
        self.assertEqual(new.PREDICATE_CODES,
                         frozenset({"not_localized", *new.CONFIGURATION_PREDICATES.values()}))

    def test_no_mutation_operations_in_new_helpers(self):
        for name in ("_labels_null_empty_exception", "_configuration_equal",
                     "_check_configuration_equal"):
            node = top(V2_TREE, name)
            self.assertFalse(any(isinstance(part, (ast.Assign, ast.AugAssign, ast.Delete))
                                 for part in ast.walk(node)))
            self.assertFalse(any(isinstance(part, ast.Attribute) and part.attr in
                                 {"update", "pop", "clear", "setdefault", "append", "extend"}
                                 for part in ast.walk(node)))

    def test_imports_do_not_initialize_lumi_or_ml(self):
        forbidden = {"lumi", "torch", "transformers", "numpy", "cuda", "cupy", "jax", "tensorflow"}
        self.assertFalse(any(name.split(".")[0] in forbidden for name in IMPORT_ADDITIONS))


class ExactExceptionTests(unittest.TestCase):
    def test_exhaustive_neighboring_labels_shapes_widen_only_one_exact_pair(self):
        values = [ABSENT, None, {}, {"key": "value"}, [], [None], "", "text", 0, 1, False, True]
        added = []
        for left_index, left in enumerate(values):
            for right_index, right in enumerate(values):
                with self.subTest(image_shape=left_index, container_shape=right_index):
                    image = {} if left is ABSENT else {"Labels": left}
                    container = {} if right is ABSENT else {"Labels": right}
                    previous = old.canonical(image.get("Labels")) == old.canonical(
                        container.get("Labels"))
                    exception = left is None and type(right) is dict and len(right) == 0
                    self.assertEqual(new._labels_null_empty_exception(image, container, "Labels"),
                                     exception)
                    selected = new._configuration_equal(image, container, "Labels")
                    self.assertEqual(selected, previous or exception)
                    if selected and not previous:
                        added.append((left_index, right_index))
        self.assertEqual(added, [(1, 2)])

    def test_exception_does_not_apply_to_absent_keys_or_reverse_direction(self):
        cases = [({}, {"Labels": {}}), ({"Labels": None}, {}), ({}, {}),
                 ({"Labels": {}}, {"Labels": None})]
        for image, container in cases:
            with self.subTest(image=image, container=container):
                self.assertFalse(new._labels_null_empty_exception(image, container, "Labels"))

    def test_old_missing_or_null_equal_behavior_is_retained(self):
        cases = [({}, {}), ({"Labels": None}, {}), ({}, {"Labels": None}),
                 ({"Labels": None}, {"Labels": None})]
        for image, container in cases:
            with self.subTest(image=image, container=container):
                self.assertTrue(new._configuration_equal(image, container, "Labels"))

    def test_nonempty_equal_values_keep_old_behavior_without_exception(self):
        for value in ({"label": "value"}, ["value"], "value", 1, True):
            image, container = {"Labels": value}, {"Labels": copy.deepcopy(value)}
            self.assertTrue(new._configuration_equal(image, container, "Labels"))
            self.assertFalse(new._labels_null_empty_exception(image, container, "Labels"))

    def test_exact_dict_type_required_for_new_exception(self):
        class DictionarySubclass(dict):
            pass

        self.assertFalse(new._labels_null_empty_exception(
            {"Labels": None}, {"Labels": DictionarySubclass()}, "Labels"))
        self.assertFalse(new._configuration_equal(
            {"Labels": None}, {"Labels": DictionarySubclass()}, "Labels"))

    def test_no_other_configuration_field_gets_null_to_empty_exception(self):
        for key in (*new.CONFIGURATION_PREDICATES, "Image", "labels"):
            if key == "Labels":
                continue
            with self.subTest(field=key):
                self.assertFalse(new._configuration_equal({key: None}, {key: {}}, key))
                self.assertFalse(new._labels_null_empty_exception({key: None}, {key: {}}, key))

    def test_original_comparison_short_circuits_exception_helper(self):
        with patch.object(new, "_labels_null_empty_exception",
                          side_effect=AssertionError) as helper:
            self.assertTrue(new._configuration_equal({"Labels": {}}, {"Labels": {}}, "Labels"))
        helper.assert_not_called()

    def test_canonical_exceptions_and_left_to_right_order_are_unchanged(self):
        selected = Mock(side_effect=RuntimeError(PRIVATE_TEXT))
        with patch.object(new, "canonical", selected), \
                patch.object(new, "_labels_null_empty_exception",
                             side_effect=AssertionError) as helper:
            with self.assertRaises(RuntimeError):
                new._configuration_equal({"Labels": None}, {"Labels": {}}, "Labels")
        self.assertEqual(selected.call_args_list[0].args, (None,))
        self.assertEqual(selected.call_count, 1)
        helper.assert_not_called()

    def test_input_metadata_not_mutated_on_success_or_rejection(self):
        for selected_labels in ({}, [], {"key": "value"}, ""):
            image, container = metadata()
            container["Config"]["Labels"] = selected_labels
            before_image, before_container = copy.deepcopy(image), copy.deepcopy(container)
            references = id(image["Config"]), id(container["Config"]), id(selected_labels)
            try:
                new._configuration(image, container)
            except new.OciAdmissionError:
                pass
            self.assertEqual(image, before_image)
            self.assertEqual(container, before_container)
            self.assertEqual(references,
                             (id(image["Config"]), id(container["Config"]),
                              id(container["Config"]["Labels"])))


class TypedApiTests(unittest.TestCase):
    def test_exact_observed_pair_fails_v1_and_returns_same_binding_in_v2(self):
        image, container = metadata()
        container["Config"]["Labels"] = {}
        previous, old_count = verify(old, image, container)
        selected, new_count = verify(new, image, container)
        self.assertIs(type(previous), old.OciAdmissionError)
        self.assertEqual(previous.code, "profile_rejected")
        self.assertIs(type(selected), new.SoftwareBinding)
        self.assertEqual(selected, new.SoftwareBinding())
        self.assertEqual(selected.source_identity_sha256(),
                         old.SoftwareBinding().source_identity_sha256())
        self.assertEqual((old_count, new_count), (2, 2))

    def test_preexisting_equal_configuration_passes_both_versions(self):
        for module in (old, new):
            with self.subTest(module=module.__name__):
                result, count = verify(module, *metadata())
                self.assertIs(type(result), module.SoftwareBinding)
                self.assertEqual(count, 2)

    def test_every_configuration_equality_failure_retains_fixed_predicate(self):
        for key in new.CONFIGURATION_PREDICATES:
            with self.subTest(field=key):
                image, container = metadata()
                container["Config"][key] = PRIVATE_TEXT
                result, count = verify(new, image, container)
                self.assertIs(type(result), new.OciAdmissionError)
                self.assertEqual(result.code, "profile_rejected")
                self.assertEqual(result.predicate, new.CONFIGURATION_PREDICATES[key])
                self.assertEqual(result.args, ("profile_rejected",))
                self.assertNotIn(PRIVATE_TEXT, str(result))
                self.assertEqual(count, 2)

    def test_exact_labels_exception_does_not_skip_later_equality(self):
        image, container = metadata()
        container["Config"]["Labels"] = {}
        container["Config"]["StopSignal"] = ""
        result, _count = verify(new, image, container)
        self.assertEqual(result.code, "profile_rejected")
        self.assertEqual(result.predicate, "configuration_stopsignal_equal")

    def test_exact_labels_exception_does_not_skip_required_environment(self):
        image, container = metadata()
        container["Config"]["Labels"] = {}
        env = [item for item in image["Config"]["Env"] if not item.startswith("HF_HUB_OFFLINE=")]
        image["Config"]["Env"], container["Config"]["Env"] = env, list(env)
        result, _count = verify(new, image, container)
        self.assertEqual(result.code, "profile_rejected")
        self.assertEqual(result.predicate, "not_localized")

    def test_exact_labels_exception_does_not_skip_original_fixed_profile(self):
        image, container = metadata()
        container["Config"]["Labels"] = {}
        image["Config"]["User"] = container["Config"]["User"] = "20000:20000"
        result, _count = verify(new, image, container)
        self.assertEqual(result.code, "profile_rejected")
        self.assertEqual(result.predicate, "not_localized")

    def test_earlier_host_policy_rejections_are_unchanged(self):
        for key, value in (("ReadonlyRootfs", False), ("Memory", 1), ("DeviceRequests", [{}]),
                           ("NetworkMode", "host"), ("SecurityOpt", [])):
            with self.subTest(field=key):
                image, container = metadata()
                container["Config"]["Labels"] = {}
                container["HostConfig"][key] = value
                selected, _count = verify(new, image, container)
                self.assertEqual(selected.code, "profile_rejected")
                self.assertEqual(selected.predicate, "not_localized")

    def test_unknown_exception_remains_fixed_inspect_failed(self):
        result, count = verify(new, *metadata(), exception=RuntimeError(PRIVATE_TEXT))
        self.assertIs(type(result), new.OciAdmissionError)
        self.assertEqual(result.code, "inspect_failed")
        self.assertEqual(result.predicate, "not_localized")
        self.assertEqual(result.args, ("inspect_failed",))
        self.assertNotIn(PRIVATE_TEXT, str(result))
        self.assertEqual(count, 1)

    def test_error_constructor_closes_codes_predicates_and_pairs(self):
        class StringSubclass(str):
            pass

        cases = [
            ("profile_rejected", "configuration_labels_equal", "profile_rejected",
             "configuration_labels_equal"),
            ("profile_rejected", PRIVATE_TEXT, "profile_rejected", "not_localized"),
            ("inspect_failed", "configuration_labels_equal", "inspect_failed", "not_localized"),
            (PRIVATE_TEXT, "configuration_labels_equal", "internal_failed", "not_localized"),
            ("profile_rejected", StringSubclass("configuration_labels_equal"),
             "profile_rejected", "not_localized"),
            (StringSubclass("profile_rejected"), "configuration_labels_equal",
             "internal_failed", "not_localized"),
            (None, None, "internal_failed", "not_localized"),
        ]
        for code, predicate, expected_code, expected_predicate in cases:
            with self.subTest(code_type=type(code).__name__,
                              predicate_type=type(predicate).__name__):
                error = new.OciAdmissionError(code, predicate)
                self.assertEqual(error.code, expected_code)
                self.assertEqual(error.predicate, expected_predicate)
                self.assertEqual(error.args, (expected_code,))
                self.assertNotIn(PRIVATE_TEXT, str(error))

    def test_default_error_predicate_is_not_localized(self):
        for code in new.ERROR_CODES:
            self.assertEqual(new.OciAdmissionError(code).predicate, "not_localized")

    def test_success_and_failure_do_not_retain_prior_predicate_state(self):
        image, container = metadata()
        container["Config"]["Labels"] = PRIVATE_TEXT
        error, _count = verify(new, image, container)
        self.assertEqual(error.predicate, "configuration_labels_equal")
        success, _count = verify(new, *metadata())
        self.assertIs(type(success), new.SoftwareBinding)
        error, _count = verify(new, *metadata(), exception=ValueError(PRIVATE_TEXT))
        self.assertEqual(error.predicate, "not_localized")


if __name__ == "__main__":
    unittest.main()
