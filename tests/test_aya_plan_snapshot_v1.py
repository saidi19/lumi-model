"""Synthetic bindings exercise the selected original Aya test without private IO."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from lumi import posttraining_aya_plan_snapshot_v1 as projection

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_ROOT = "/synthetic/workspace/repo"
ERROR = "invalid Aya plan source projection"


def _pin(payloads: dict, bindings: dict, role: str) -> None:
    bindings[role]["bytes"] = len(payloads[role])
    bindings[role]["sha256"] = hashlib.sha256(payloads[role]).hexdigest()


def _plan(payloads: dict, bindings: dict, value: dict) -> None:
    payloads["plan"] = json.dumps(value, sort_keys=True).encode()
    _pin(payloads, bindings, "plan")


def _repin_source(payloads: dict, bindings: dict, role: str) -> None:
    _pin(payloads, bindings, role)
    if role in projection.PLAN_BINDING_ROLES:
        plan = json.loads(payloads["plan"])
        plan["bindings"][role]["sha256"] = bindings[role]["sha256"]
        _plan(payloads, bindings, plan)


def synthetic_inputs() -> tuple[dict, dict]:
    """Use tracked source only; every plan/data binding is generated in memory."""
    payloads = {role: ("synthetic " + role).encode() for role in projection.INPUT_ROLES}
    bindings = {
        role: {"path": projection.SOURCE_PATHS.get(role, f"synthetic/{role}.bin")}
        for role in projection.INPUT_ROLES
    }
    for role in ("repair_implementation", "hash_helper_source", "test_source"):
        payloads[role] = (ROOT / projection.SOURCE_PATHS[role]).read_bytes()
    for role in projection.INPUT_ROLES:
        _pin(payloads, bindings, role)

    # Read only literal declarations from authentic tracked source, not its imports.
    constants = [
        node
        for node in ast.parse(payloads["repair_implementation"]).body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in projection.AYA_CONSTANTS
    ]
    namespace = {"re": re}
    exec(
        compile(ast.Module(body=constants, type_ignores=[]), "<synthetic-fixture>", "exec"),
        namespace,
    )
    value = {
        "schema_version": namespace["PLAN_SCHEMA_VERSION"],
        "status": namespace["EXPECTED_STATUS"],
        "observed_at": "2026-08-06",
        **{
            field: copy.deepcopy(namespace[f"EXPECTED_{field.upper()}"])
            for field in ("authorization", "limits", "output", "policy", "source")
        },
        "bindings": {
            role: {key: bindings[role][key] for key in ("path", "sha256")}
            for role in projection.PLAN_BINDING_ROLES
        },
    }
    _plan(payloads, bindings, value)
    return payloads, bindings


class AyaPlanSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.payloads, self.bindings = synthetic_inputs()

    def build(self):
        return projection.build_test_module(self.payloads, self.bindings, SYNTHETIC_ROOT)

    def reject(self):
        with self.assertRaisesRegex(ValueError, f"^{ERROR}$"):
            self.build()

    def test_authentic_test_body_passes_with_synthetic_bindings(self):
        module, snapshot = self.build()
        self.assertEqual(module.__name__, projection.MODULE_NAME)
        self.assertEqual(
            module.__file__, SYNTHETIC_ROOT + "/" + projection.SOURCE_PATHS["test_source"]
        )
        function = getattr(module, projection.TEST_NAME)
        original = next(
            node
            for node in ast.parse(self.payloads["test_source"]).body
            if isinstance(node, ast.FunctionDef) and node.name == projection.TEST_NAME
        )
        self.assertEqual(function.__code__.co_filename, module.__file__)
        self.assertEqual(function.__code__.co_firstlineno, original.lineno)
        self.assertEqual(
            {name for name in vars(module) if name.startswith("test_")}, {projection.TEST_NAME}
        )
        self.assertIsNone(function())
        self.assertEqual(snapshot.summary()["virtual_opens"], 16)
        self.assertIs(snapshot.summary()["filesystem_access_performed"], False)

    def test_unused_imports_statements_and_functions_never_execute(self):
        poison = (
            b"\nimport nonexistent_projection_dependency\n"
            b"raise RuntimeError('unused top-level statement')\n"
            b"def unused_poison(value=1 / 0):\n    raise RuntimeError('unused function')\n"
        )
        for role in ("repair_implementation", "hash_helper_source", "test_source"):
            self.payloads[role] += poison
            _repin_source(self.payloads, self.bindings, role)
        module, _ = self.build()
        getattr(module, projection.TEST_NAME)()
        for name in ("unused_poison", "select_aya_repair_candidates", "deterministic_owner_sample"):
            self.assertNotIn(name, vars(module))

    def test_projection_and_original_body_use_no_real_file_operations(self):
        with (
            patch("builtins.open", side_effect=AssertionError("real file open")),
            patch("io.open", side_effect=AssertionError("real file open")),
            patch("os.open", side_effect=AssertionError("real file open")),
            patch("os.stat", side_effect=AssertionError("real file stat")),
            patch("os.lstat", side_effect=AssertionError("real file stat")),
        ):
            module, _ = self.build()
            getattr(module, projection.TEST_NAME)()

    def test_existing_import_namespace_is_untouched(self):
        sentinel = types.ModuleType(projection.MODULE_NAME)
        child = types.ModuleType(projection.MODULE_NAME + ".sentinel")
        entries = {projection.MODULE_NAME: sentinel, child.__name__: child}
        with patch.dict(sys.modules, entries):
            module, _ = self.build()
            self.assertIsNot(module, sentinel)
            for name, expected in entries.items():
                self.assertIs(sys.modules[name], expected)

    def test_authentication_precedes_source_parsing(self):
        self.payloads["test_source"] += b"\n"
        with patch.object(
            projection.ast, "parse", side_effect=AssertionError("parsed too early")
        ) as parse:
            self.reject()
            parse.assert_not_called()

    def test_malformed_binding_metadata_is_rejected(self):
        original = copy.deepcopy(self.bindings)
        cases = (
            ("bytes", True),
            ("bytes", 0),
            ("bytes", 1),
            ("sha256", "A" * 64),
            ("sha256", "0" * 64),
            ("git_revision", "invalid"),
            ("unexpected", "ignored"),
            ("path", "../../../escape"),
            ("path", "/absolute"),
            ("path", "bad\npath"),
            ("path", self.bindings["tokenizer_json"]["path"]),
        )
        for key, value in cases:
            with self.subTest(key=key, value=value):
                self.bindings = copy.deepcopy(original)
                self.bindings["tokenizer_config"][key] = value
                self.reject()

    def test_roles_and_source_locations_are_exact(self):
        for target in ("payloads", "bindings"):
            for extra in (False, True):
                with self.subTest(target=target, extra=extra):
                    self.payloads, self.bindings = synthetic_inputs()
                    value = getattr(self, target)
                    if extra:
                        value["unexpected"] = value["plan"]
                    else:
                        del value["plan"]
                    self.reject()
        self.payloads, self.bindings = synthetic_inputs()
        self.bindings["test_source"]["path"] = "synthetic/wrong_test.py"
        self.reject()

    def test_input_sizes_are_bounded(self):
        with patch.object(projection, "MAXIMUM_TOTAL_BYTES", 1):
            self.reject()
        with patch.object(projection, "MAXIMUM_SOURCE_BYTES", 1):
            self.reject()

    def test_repository_namespace_is_strict(self):
        for root in (
            None,
            1,
            "relative",
            "/",
            "//workspace/repo",
            "/synthetic/../repo",
            "/bad\nrepo",
        ):
            with self.subTest(root=root), self.assertRaisesRegex(ValueError, f"^{ERROR}$"):
                projection.build_test_module(self.payloads, self.bindings, root)

    def test_missing_or_duplicate_selected_function_is_rejected(self):
        original = self.payloads["test_source"]
        for changed in (
            original.replace(projection.TEST_NAME.encode(), b"unused_test_name", 1),
            original + f"\ndef {projection.TEST_NAME}():\n    pass\n".encode(),
        ):
            with self.subTest(changed=hashlib.sha256(changed).hexdigest()):
                self.payloads["test_source"] = changed
                _repin_source(self.payloads, self.bindings, "test_source")
                self.reject()

    def test_executable_selected_declarations_are_rejected(self):
        original = self.payloads["repair_implementation"]
        for changed in (
            original.replace(
                b'EXPECTED_STATUS = "local_repair_preflight_only"', b"EXPECTED_STATUS = str(1)"
            ),
            original.replace(
                b"def _load_json(", b"@print('decorator must not execute')\ndef _load_json("
            ),
            original.replace(
                b"def _load_json(path: str | Path)", b"def _load_json(path: str | Path = 1 / 0)"
            ),
        ):
            with self.subTest(changed=hashlib.sha256(changed).hexdigest()):
                self.payloads["repair_implementation"] = changed
                _repin_source(self.payloads, self.bindings, "repair_implementation")
                with patch("builtins.print", side_effect=AssertionError("side effect")) as printing:
                    self.reject()
                    printing.assert_not_called()

    def test_unchanged_original_validation_still_rejects_plan_mutations(self):
        original = json.loads(self.payloads["plan"])
        for field in ("authorization", "binding"):
            with self.subTest(field=field):
                plan = copy.deepcopy(original)
                if field == "authorization":
                    plan[field]["network_access_authorized"] = True
                else:
                    plan["bindings"]["tokenizer_json"]["sha256"] = "0" * 64
                _plan(self.payloads, self.bindings, plan)
                module, _ = self.build()
                with self.assertRaises(ValueError):
                    getattr(module, projection.TEST_NAME)()


if __name__ == "__main__":
    unittest.main()
