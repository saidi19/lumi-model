"""AST/literal-only native diagnostic checks. NEVER import the native script."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import unittest
from pathlib import Path

from scripts import sft_cpu_origin_diagnostic_report_v1 as contract

ROOT = Path(__file__).resolve().parents[1]
NATIVE_PATH = ROOT / "scripts/run_sft_cpu_origin_diagnostic_v1.py"
FROZEN_PATH = ROOT / "scripts/run_sft_oci_worker_probe_v1.py"


def function(tree, name):
    return next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def literal(tree, name):
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    )
    return ast.literal_eval(node.value)


def calls(node, name):
    return [
        item
        for item in ast.walk(node)
        if isinstance(item, ast.Call) and ast.unparse(item.func) == name
    ]


class CpuOriginDiagnosticSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = NATIVE_PATH.read_bytes()
        cls.tree = ast.parse(cls.raw)
        cls.frozen_raw = FROZEN_PATH.read_bytes()
        cls.frozen = ast.parse(cls.frozen_raw)
        cls.spec = contract.spec_from_source(cls.raw, hashlib.sha256(cls.raw).hexdigest())

    def test_all_frozen_observer_bytes_still_match(self):
        self.assertEqual(len(self.frozen_raw), literal(self.tree, "OBSERVER_SIZE"))
        self.assertEqual(
            hashlib.sha256(self.frozen_raw).hexdigest(), literal(self.tree, "OBSERVER_SHA256")
        )
        fixture = (ROOT / "scripts/run_sft_worker_run_torch_fixture_v1.py").read_bytes()
        self.assertEqual(len(fixture), literal(self.tree, "FIXTURE_SIZE"))
        self.assertEqual(hashlib.sha256(fixture).hexdigest(), literal(self.tree, "FIXTURE_SHA256"))

    def test_all_seventy_two_exact_need_sites_are_mapped_without_rewriting(self):
        found = []
        for node in self.frozen.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            for index, call in enumerate(sorted(calls(node, "need"), key=lambda c: c.lineno), 1):
                found.append(
                    (
                        node.name,
                        call.lineno,
                        call.end_lineno,
                        node.name + "_check_" + str(index).zfill(2),
                    )
                )
        expected = literal(self.tree, "CHECKPOINTS")
        self.assertEqual(len(expected), 72)
        self.assertEqual(set(found), set(expected))

    def test_every_mapped_code_is_in_the_literal_closed_vocabulary(self):
        enums = self.spec["DIAGNOSTIC_ENUMS"]
        mapped = literal(self.tree, "CHECKPOINTS")
        for name, _, _, code in mapped:
            self.assertIn(name, enums["captured_function"])
            self.assertIn(code, enums["captured_check"])
        expected = {row[3] for row in mapped} | {"not_reported"}
        expected |= {name + "_unexpected" for name in literal(self.tree, "UNEXPECTED_FUNCTIONS")}
        self.assertEqual(set(enums["captured_check"]), expected)

    def test_native_module_not_imported_by_this_suite(self):
        self.assertNotIn(literal(self.tree, "OBSERVER_NAME"), sys.modules)
        self.assertNotIn("scripts.run_sft_cpu_origin_diagnostic_v1", sys.modules)
        self.assertNotIn("scripts.run_sft_oci_worker_probe_v1", sys.modules)

    def test_only_stdlib_imports_in_native_probe(self):
        imports = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module.split(".")[0])
        self.assertEqual(
            imports,
            {"__future__", "hashlib", "importlib", "json", "os", "re", "stat", "sys", "pathlib"},
        )

    def test_linux_isolated_nonroot_guard_precedes_loading(self):
        prepare = function(self.tree, "prepare")
        self.assertLess(
            calls(prepare, "guard")[0].lineno, calls(prepare, "load_observer")[0].lineno
        )
        guard = ast.unparse(function(self.tree, "guard"))
        for assertion in (
            "sys.platform == 'linux'",
            "sys.flags.isolated == 1",
            "os.geteuid() == os.getegid() == 10001",
            "'--synthetic-only'",
            "'torch' not in sys.modules",
            "sys.dont_write_bytecode",
        ):
            self.assertIn(assertion, guard)

    def test_source_is_opened_readonly_bounded_and_hash_verified(self):
        loader = function(self.tree, "load_observer")
        text = ast.unparse(loader)
        for required in (
            "os.O_RDONLY",
            "os.O_NOFOLLOW",
            "os.O_NONBLOCK",
            "before.st_size != OBSERVER_SIZE",
            "OBSERVER_SHA256",
            "os.read(descriptor, 1)",
            "os.close(descriptor)",
        ):
            self.assertIn(required, text)
        self.assertEqual(len(calls(loader, "compile")), 1)
        self.assertEqual(len(calls(loader, "exec")), 1)

    def test_need_wrapper_always_calls_original_predicate_unchanged(self):
        node = function(self.tree, "observed_need")
        self.assertIsInstance(node.body[-1], ast.Return)
        self.assertEqual(ast.unparse(node.body[-1].value), "_ORIGINALS['need'](condition, code)")
        self.assertIn("frame.f_lineno", ast.unparse(node))
        self.assertIn("del frame", ast.unparse(node))

    def test_unexpected_wrappers_rethrow_not_suppress_rejection(self):
        node = function(self.tree, "origin_wrapper")
        raises = [item for item in ast.walk(node) if isinstance(item, ast.Raise)]
        self.assertEqual(len(raises), 1)
        self.assertIsNone(raises[0].exc)
        self.assertIn("original(*args, **kwargs)", ast.unparse(node))

    def test_frozen_rank_and_workload_functions_are_not_replaced(self):
        writes = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and "_V1" in ast.unparse(target):
                        writes.append(ast.unparse(target))
        self.assertEqual(
            set(writes),
            {
                "_V1.need",
                "_V1.observed_invocation",
                "_V1._FIXTURE.outcome",
                "_V1._FIXTURE.record_failure",
            },
        )
        self.assertNotIn("rank_worker =", self.raw.decode())

    def test_failure_handler_preserves_original_first_failure_collection(self):
        node = function(self.tree, "diagnostic_record_failure")
        self.assertEqual(ast.unparse(node.body[-1]), "return _RECORD_FAILURE(observed)")
        self.assertIn("valid_packet(value)", ast.unparse(node))
        self.assertIn("len(_RECEIVED) < 6", ast.unparse(node))

    def test_no_extra_failure_pipe_message_or_raw_value_output(self):
        node = function(self.tree, "diagnostic_outcome")
        self.assertEqual(len(calls(node, "_OUTCOME")), 1)
        self.assertFalse(calls(self.tree, "pipe.send_bytes"))
        self.assertFalse(calls(self.tree, "print"))
        self.assertFalse(calls(self.tree, "repr"))
        # The guard renders only the executable path; captured values must never stringify.
        for name in ("diagnostic_outcome", "context_categories", "capture_unexpected", "report"):
            self.assertFalse(calls(function(self.tree, name), "str"))

    def test_context_capture_is_bounded_and_only_categorizes_fixed_frame_locals(self):
        text = ast.unparse(function(self.tree, "context_categories"))
        self.assertIn("range(8)", text)
        self.assertIn("frame.f_code in _CODES", text)
        self.assertIn("del frame", text)
        node = function(self.tree, "category")
        self.assertFalse(calls(node, "os.stat"))
        self.assertFalse(calls(node, "Path"))

    def test_exception_classes_are_identity_mapped_not_messages(self):
        text = ast.unparse(function(self.tree, "exception_class"))
        self.assertIn("type(error) is kind", text)
        self.assertNotIn("__name__", text)
        self.assertNotIn("str(error)", self.raw.decode())
        self.assertNotIn("repr(error)", self.raw.decode())

    def test_failure_packet_is_bounded_and_counts_reject_bools(self):
        text = ast.unparse(function(self.tree, "valid_packet"))
        self.assertIn("type(counts[key]) is int", text)
        self.assertIn("len(canonical(value)) <= MAXIMUM_PACKET_BYTES", text)
        self.assertEqual(literal(self.tree, "MAXIMUM_PACKET_BYTES"), 4096)

    def test_exactly_one_frozen_run_and_spawn_hook_setup(self):
        self.assertEqual(len(calls(self.tree, "_V1.run")), 1)
        entry = function(self.tree, "entry")
        self.assertLess(calls(entry, "prepare")[0].lineno, calls(entry, "_V1.run")[0].lineno)
        self.assertIn("__mp_main__", ast.unparse(self.tree.body[-1]))
        self.assertIn("os._exit(2)", ast.unparse(entry))

    def test_public_output_is_suppressed_then_bounded_to_sixteen_kib(self):
        text = ast.unparse(function(self.tree, "entry"))
        self.assertIn("os.dup2(sink, 1)", text)
        self.assertIn("os.dup2(sink, 2)", text)
        self.assertIn("len(raw) > MAXIMUM_REPORT_BYTES", text)
        self.assertEqual(literal(self.tree, "MAXIMUM_REPORT_BYTES"), 16384)
        self.assertEqual(len(calls(self.tree, "os.write")), 1)

    def test_report_has_no_acceptance_or_recovery_true_path(self):
        self.assertEqual(
            set(self.spec["FALSE_FLAGS"]),
            set(contract.BASE_FALSE_FLAGS) | {"acceptance_passed", "recovery_performed"},
        )
        self.assertIn(
            "dict.fromkeys(FALSE_FLAGS, False)", ast.unparse(function(self.tree, "report"))
        )

    def test_frozen_worker_component_labels_are_visible_but_no_raw_exceptions(self):
        text = ast.unparse(function(self.tree, "frozen_stages"))
        self.assertIn("'WORKER_STAGE'", text)
        self.assertIn("'WORKER_CODE'", text)
        self.assertEqual(set(self.spec["COMPONENT_ENUMS"]), set(contract.COMPONENT_FIELDS))
        self.assertEqual(len(self.spec["COMPONENT_ENUMS"]), 7)

    def test_actual_literal_spec_roundtrips_synthetic_failure_only(self):
        expected = {
            key: "a" * (40 if key == "source_commit" else 64) for key in self.spec["CONTROL_KEYS"]
        }
        value = {
            "schema": contract.SCHEMA,
            "status": "diagnostic_completed",
            "underlying_status": "failed",
            "controls": expected,
            "diagnostic": dict(
                contract.DIAGNOSTIC_DEFAULTS,
                captured_check="native_origins_check_05",
                captured_function="native_origins",
                phase="inputs",
                exception_class="ProbeError",
                probe_error_code="native_origin_rejected",
            ),
            "counts": dict.fromkeys(self.spec["COUNT_LIMITS"], 0),
            "frozen_stages": dict.fromkeys(self.spec["FROZEN_STAGE_ENUMS"], "not_reported"),
            "frozen_component_diagnostics": dict.fromkeys(
                self.spec["COMPONENT_ENUMS"], "not_reported"
            ),
            "frozen_origin_counts": dict.fromkeys(self.spec["ORIGIN_COUNT_FIELDS"], 0),
            **dict.fromkeys(self.spec["FALSE_FLAGS"], False),
        }
        raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"
        self.assertEqual(contract.validate_report(raw, expected, self.spec), value)


if __name__ == "__main__":
    unittest.main()
