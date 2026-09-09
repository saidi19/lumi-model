"""Pure synthetic lookup and AST checks; NEVER import/compile/run the native probe.

The standalone expression below is bound structurally to the one changed probe
predicate. Synthetic modules are stdlib-only, not imported Torch/Transformers.
No upstream source, native runtime, image, wheel, model or data is executed/read.
"""

from __future__ import annotations

import ast
import hashlib
import sys
import unittest
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
OLD_PATH = ROOT / "scripts/run_sft_oci_worker_probe_v1.py"
NEW_PATH = ROOT / "scripts/run_sft_oci_worker_probe_v2.py"
OLD_SHA256 = "bed636e8cc08546a6be1b43c00c2f516fdc92dbf907dca674f7fb5aa7b5cb769"
OLD_LOOKUP = b'str(vars(sys.modules[name]).get("__version__"))'
NEW_LOOKUP = b'str(getattr(sys.modules[name], "__version__", None))'


def legacy_accepts(module, expected):
    return str(vars(module).get("__version__")) == expected


def synthetic_accepts(module, expected):
    return str(getattr(module, "__version__", None)) == expected


class LazyVersionModule(ModuleType):
    """Minimal synthetic attribute contract, not copied/executed upstream code."""

    def __init__(self, values):
        super().__init__("synthetic_version_module")
        self.values = values
        self.requested = []

    def __getattr__(self, name):
        self.requested.append(name)
        if name in self.values:
            return self.values[name]
        raise AttributeError(name)


class OciVersionLookupSyntheticTests(unittest.TestCase):
    def test_eager_exact_version_preserves_success(self):
        module = ModuleType("synthetic_eager")
        module.__version__ = "5.15.1"
        self.assertTrue(legacy_accepts(module, "5.15.1"))
        self.assertTrue(synthetic_accepts(module, "5.15.1"))

    def test_reproduces_direct_dictionary_false_rejection_and_corrected_lookup(self):
        module = LazyVersionModule({"__version__": "5.15.1"})
        self.assertNotIn("__version__", vars(module))
        self.assertFalse(legacy_accepts(module, "5.15.1"))
        self.assertEqual(module.requested, [])
        self.assertTrue(synthetic_accepts(module, "5.15.1"))
        self.assertEqual(module.requested, ["__version__"])

    def test_lazy_lookup_does_not_require_or_inject_a_dictionary_cache(self):
        module = LazyVersionModule({"__version__": "5.15.1"})
        for _ in range(2):
            self.assertTrue(synthetic_accepts(module, "5.15.1"))
            self.assertNotIn("__version__", vars(module))
        self.assertEqual(module.requested, ["__version__", "__version__"])

    def test_lazy_wrong_version_still_rejects(self):
        module = LazyVersionModule({"__version__": "5.15.0"})
        self.assertFalse(legacy_accepts(module, "5.15.1"))
        self.assertFalse(synthetic_accepts(module, "5.15.1"))

    def test_eager_wrong_version_still_rejects(self):
        module = ModuleType("synthetic_wrong")
        module.__version__ = "5.15.0"
        self.assertFalse(synthetic_accepts(module, "5.15.1"))

    def test_missing_eager_version_still_rejects(self):
        module = ModuleType("synthetic_missing")
        self.assertFalse(legacy_accepts(module, "5.15.1"))
        self.assertFalse(synthetic_accepts(module, "5.15.1"))

    def test_missing_lazy_version_still_rejects(self):
        module = LazyVersionModule({})
        self.assertFalse(synthetic_accepts(module, "5.15.1"))
        self.assertEqual(module.requested, ["__version__"])

    def test_none_bool_number_or_empty_does_not_match_locked_version(self):
        for value in (None, False, True, 5151, 5.15, ""):
            with self.subTest(value_type=type(value).__name__):
                module = LazyVersionModule({"__version__": value})
                self.assertFalse(synthetic_accepts(module, "5.15.1"))

    def test_no_whitespace_suffix_prefix_or_range_normalization(self):
        for value in (
            " 5.15.1",
            "5.15.1 ",
            "5.15.1\n",
            "v5.15.1",
            "5.15.1+cpu",
            "5.15.1.dev0",
            ">=5.15.1",
            "5.15.10",
            "5.15",
        ):
            with self.subTest(synthetic_version=value):
                self.assertFalse(
                    synthetic_accepts(LazyVersionModule({"__version__": value}), "5.15.1")
                )

    def test_torch_local_version_suffix_and_case_remain_exact(self):
        for value, expected in (("2.14.0+cpu", True), ("2.14.0", False), ("2.14.0+CPU", False)):
            with self.subTest(synthetic_version=value):
                module = ModuleType("synthetic_torch_version")
                module.__version__ = value
                self.assertEqual(synthetic_accepts(module, "2.14.0+cpu"), expected)

    def test_existing_str_conversion_is_preserved(self):
        class VersionString(str):
            pass

        module = ModuleType("synthetic_string_subclass")
        module.__version__ = VersionString("5.15.1")
        self.assertTrue(legacy_accepts(module, "5.15.1"))
        self.assertTrue(synthetic_accepts(module, "5.15.1"))

    def test_eager_value_takes_priority_over_lazy_extra_objects(self):
        module = LazyVersionModule({"__version__": "5.15.0"})
        module.__version__ = "5.15.1"
        self.assertTrue(synthetic_accepts(module, "5.15.1"))
        self.assertEqual(module.requested, [])

    def test_non_attribute_error_propagates_without_fallback(self):
        class BrokenModule(ModuleType):
            def __getattr__(self, name):
                raise RuntimeError("synthetic_lookup_failure")

        with self.assertRaises(RuntimeError):
            synthetic_accepts(BrokenModule("synthetic_broken"), "5.15.1")

    def test_module_level_getattr_contract_also_works(self):
        module = ModuleType("synthetic_module_getattr")
        requested = []

        def public_attribute(name):
            requested.append(name)
            if name == "__version__":
                return "5.15.1"
            raise AttributeError(name)

        module.__getattr__ = public_attribute
        self.assertFalse(legacy_accepts(module, "5.15.1"))
        self.assertTrue(synthetic_accepts(module, "5.15.1"))
        self.assertEqual(requested, ["__version__"])


class OciVersionLookupSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = OLD_PATH.read_bytes()
        cls.new = NEW_PATH.read_bytes()
        cls.tree = ast.parse(cls.new)
        cls.test_tree = ast.parse(Path(__file__).read_bytes())

    def test_frozen_old_probe_is_byte_identical(self):
        self.assertEqual(len(self.old), 22633)
        self.assertEqual(hashlib.sha256(self.old).hexdigest(), OLD_SHA256)

    def test_successor_has_exactly_one_expression_replacement_and_nothing_else(self):
        self.assertEqual(self.old.count(OLD_LOOKUP), 1)
        self.assertEqual(self.old.count(NEW_LOOKUP), 0)
        self.assertEqual(self.new.count(OLD_LOOKUP), 0)
        self.assertEqual(self.new.count(NEW_LOOKUP), 1)
        self.assertEqual(self.new, self.old.replace(OLD_LOOKUP, NEW_LOOKUP, 1))

    def test_comparison_still_uses_same_lock_and_failure_code(self):
        node = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "module_origins"
        )
        comparisons = [
            call
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and ast.unparse(call.func) == "need"
            and ast.unparse(call.args[0]).startswith("str(getattr(")
        ]
        self.assertEqual(len(comparisons), 1)
        self.assertEqual(
            ast.unparse(comparisons[0].args[0]),
            "str(getattr(sys.modules[name], '__version__', None)) == _STATE['versions'][name]",
        )
        self.assertEqual(ast.literal_eval(comparisons[0].args[1]), "distribution_rejected")

    def test_executed_synthetic_expression_is_ast_bound_to_production_predicate(self):
        function = next(
            node
            for node in self.test_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "synthetic_accepts"
        )
        self.assertEqual(len(function.body), 1)
        self.assertIsInstance(function.body[0], ast.Return)
        self.assertEqual(
            ast.dump(function.body[0].value, include_attributes=False),
            ast.dump(
                ast.parse(
                    "str(getattr(module, '__version__', None)) == expected", mode="eval"
                ).body,
                include_attributes=False,
            ),
        )

        # Only AST is transformed here; no native source or subtree is compiled/executed.
        class BindArguments(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id == "module":
                    return ast.parse("sys.modules[name]", mode="eval").body
                if node.id == "expected":
                    return ast.parse("_STATE['versions'][name]", mode="eval").body
                return node

        bound = BindArguments().visit(
            ast.parse(ast.unparse(function.body[0].value), mode="eval").body
        )
        origin = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "module_origins"
        )
        self.assertTrue(
            any(
                ast.dump(call.args[0], include_attributes=False)
                == ast.dump(bound, include_attributes=False)
                for call in ast.walk(origin)
                if isinstance(call, ast.Call) and ast.unparse(call.func) == "need"
            )
        )

    def test_legacy_replica_stays_bound_to_failed_expression(self):
        function = next(
            node
            for node in self.test_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "legacy_accepts"
        )
        self.assertEqual(
            ast.unparse(function.body[0]), "return str(vars(module).get('__version__')) == expected"
        )

    def test_probe_fixture_or_ml_not_imported(self):
        # This test owns its imports, not the monolithic suite's earlier imports.
        native = (
            "scripts.run_sft_oci_worker_probe_v1",
            "scripts.run_sft_oci_worker_probe_v2",
            "_lumi_oci_frozen_worker_v1",
        )
        self.assertFalse(any(name in sys.modules for name in native), "native_probe_imported")
        imports = {
            alias.name.split(".")[0]
            for node in ast.walk(self.test_tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imports |= {
            node.module.split(".")[0]
            for node in ast.walk(self.test_tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertEqual(
            imports, {"__future__", "ast", "hashlib", "sys", "unittest", "pathlib", "types"}
        )
        dynamic = {
            "__import__",
            "compile",
            "exec",
            "eval",
            "importlib.import_module",
            "runpy.run_path",
        }
        self.assertFalse(
            any(
                isinstance(node, ast.Call) and ast.unparse(node.func) in dynamic
                for node in ast.walk(self.test_tree)
            ),
            "dynamic_native_execution_in_test",
        )

    def test_guard_allows_preexisting_synthetic_ml_entries_without_loading_ml(self):
        from unittest import mock

        stubs = {name: ModuleType(name) for name in ("torch", "transformers")}
        with mock.patch.dict(sys.modules, stubs):
            # Reproduces the old guard's failure without importing either library.
            self.assertTrue(any(name in sys.modules for name in stubs))
            self.test_probe_fixture_or_ml_not_imported()
            self.assertTrue(all(sys.modules[name] is value for name, value in stubs.items()))

    def test_guard_still_rejects_synthetic_native_entry_without_dumping_registry(self):
        from unittest import mock

        name = "scripts.run_sft_oci_worker_probe_v2"
        with mock.patch.dict(sys.modules, {name: ModuleType(name)}):
            with self.assertRaisesRegex(AssertionError, "native_probe_imported"):
                self.test_probe_fixture_or_ml_not_imported()



if __name__ == "__main__":
    unittest.main()
