"""Synthetic boundary tests only; no real plan contents or seal execution."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "learning_seal_ci_guard", ROOT / "scripts/check_learning_seal_local_ci_v1.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class LearningSealGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = Path("/synthetic-only/code")
        self.plan = Path("/synthetic-only/private/plan.json")
        self.boundary = runner.ReadBoundary(self.plan, (self.base,))

    def kind(self, event: str, *arguments) -> str | None:
        return self.boundary.classify(event, arguments)

    def test_code_reads_are_allowed(self) -> None:
        self.assertIsNone(self.kind("open", self.base / "unit.py", "r", os.O_RDONLY))

    def test_other_private_and_neighbor_reads_are_rejected(self) -> None:
        for value in ("/synthetic-only/private/rows", "/synthetic-only/code-next/unit.py",
                      "/synthetic-only/code/../private/rows"):
            with self.subTest(value=value):
                self.assertEqual(self.kind("open", value, "r", os.O_RDONLY), "file_content")

    def test_plan_read_without_phase_is_rejected(self) -> None:
        self.assertEqual(self.kind("open", self.plan, "r", os.O_RDONLY), "plan_open_cap")
        self.assertEqual(sum(self.boundary.plan_opens.values()), 0)

    def test_exactly_one_plan_open_in_each_phase(self) -> None:
        for phase in runner.PHASES:
            with self.boundary.allow_plan(phase):
                self.assertIsNone(self.kind("open", self.plan, "r", os.O_RDONLY))
                self.assertEqual(self.kind("open", self.plan, "r", os.O_RDONLY), "plan_open_cap")
        self.assertEqual(self.boundary.plan_opens, dict.fromkeys(runner.PHASES, 1))
        self.assertIsNone(self.boundary.phase)

    def test_reentering_phase_does_not_reset_open_count(self) -> None:
        with self.boundary.allow_plan("fixture"):
            self.assertIsNone(self.kind("open", self.plan, "r", os.O_RDONLY))
        with self.boundary.allow_plan("fixture"):
            self.assertEqual(self.kind("open", self.plan, "r", os.O_RDONLY), "plan_open_cap")

    def test_unknown_or_nested_phase_is_rejected(self) -> None:
        with self.assertRaisesRegex(runner.LocalUnitError, "plan_phase"):
            with self.boundary.allow_plan("other"):
                pass
        with self.boundary.allow_plan("fixture"):
            with self.assertRaisesRegex(runner.LocalUnitError, "plan_phase"):
                with self.boundary.allow_plan("hash_after"):
                    pass

    def test_phase_is_closed_after_exception(self) -> None:
        with self.assertRaises(AssertionError):
            with self.boundary.allow_plan("fixture"):
                raise AssertionError("synthetic")
        self.assertIsNone(self.boundary.phase)

    def test_raw_descriptors_are_rejected(self) -> None:
        self.assertEqual(self.kind("open", 7, "r", os.O_RDONLY), "file_content")

    def test_all_write_flags_are_rejected_for_code_and_plan(self) -> None:
        for path in (self.base / "unit.py", self.plan):
            for flags in (os.O_WRONLY, os.O_RDWR, os.O_CREAT, os.O_TRUNC, os.O_APPEND):
                with self.subTest(path=path, flags=flags):
                    self.assertEqual(self.kind("open", path, "w", flags), "writes")

    def test_network_process_and_mutation_events_are_rejected(self) -> None:
        for event in ("socket.connect", "socket.getaddrinfo", "socket.__new__"):
            self.assertEqual(self.kind(event), "network")
        for event in runner.PROCESS_EVENTS:
            self.assertEqual(self.kind(event), "process")
        for event in runner.WRITE_EVENTS:
            self.assertEqual(self.kind(event), "writes")

    def test_exact_ml_stack_denylist_and_nested_imports(self) -> None:
        self.assertEqual(runner.ML_MODULES, {
            "torch", "transformers", "tensorflow", "jax", "huggingface_hub", "tokenizers",
            "safetensors", "numpy", "accelerate", "datasets", "sentencepiece",
        })
        for name in runner.ML_MODULES:
            self.assertEqual(self.kind("import", name), "ml_import")
            self.assertEqual(self.kind("import", name + ".internal"), "ml_import")
        self.assertIsNone(self.kind("import", "json"))

    def test_guard_increments_blocked_count_and_aborts(self) -> None:
        with self.assertRaises(runner.BoundaryViolation):
            self.boundary.guard("socket.connect", ())
        self.assertEqual(self.boundary.blocked["network"], 1)

    def test_boundary_and_deadline_cannot_satisfy_value_error_rejections(self) -> None:
        for exception in (runner.BoundaryViolation, runner.DeadlineExceeded):
            self.assertFalse(issubclass(exception, ValueError))
            with self.assertRaises(exception):
                with self.assertRaises(ValueError):
                    raise exception()

    def test_deadline_always_raises(self) -> None:
        with self.assertRaises(runner.DeadlineExceeded):
            runner.deadline(None, None)

    def test_symlink_cannot_extend_read_scope(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lumi-seal-guard-") as directory:
            base = Path(directory).resolve()
            code = base / "code"
            code.mkdir()
            (code / "escape").symlink_to(base / "private")
            boundary = runner.ReadBoundary(base / "plan", (code,))
            self.assertEqual(boundary.classify(
                "open", (code / "escape/rows", "r", os.O_RDONLY)
            ), "file_content")

    def test_discard_does_not_retain_output(self) -> None:
        sink = runner.DiscardText()
        self.assertEqual(sink.write("synthetic output"), 16)
        self.assertEqual(vars(sink), {})

    def synthetic_module(self, outcome=None):
        calls = []
        values = [(("synthetic", str(index)), index) for index in range(10)]

        def case(*args):
            calls.append(args)
            if outcome is not None:
                raise outcome()

        case.pytestmark = [types.SimpleNamespace(
            name="parametrize", args=(("path", "value"), values), kwargs={}
        )]
        manifest = {"synthetic_manifest": True}
        module = types.SimpleNamespace(**dict.fromkeys(runner.TEST_NAMES, case))
        module.manifest = types.SimpleNamespace(__wrapped__=lambda plan: manifest)
        return module, calls, values, manifest

    def test_exact_original_function_names(self) -> None:
        self.assertEqual(runner.TEST_NAMES, (
            "test_plan_binds_exact_twenty_source_paths_without_row_reads",
            "test_manifest_rejects_resigned_boundary_drift",
            "test_verification_rejects_resigned_audit_drift",
        ))

    def test_original_parameters_are_used_without_replacement(self) -> None:
        module, _, values, _ = self.synthetic_module()
        actual = runner.original_parameters(module)
        self.assertEqual(len(actual), 10)
        for original, observed in zip(values, actual, strict=True):
            self.assertIs(original, observed)

    def test_exact_twelve_calls_original_fixture_and_arguments(self) -> None:
        module, calls, values, manifest = self.synthetic_module()
        plan = {"synthetic_plan": True}
        self.assertEqual(runner.run_fixed_tests(module, plan), (12, 0))
        self.assertEqual(calls, [(plan,), *[(plan, manifest, *value) for value in values],
                                 (plan, manifest)])

    def test_actual_failures_are_never_converted_to_expected_outcomes(self) -> None:
        for exception in (AssertionError, ValueError, SystemExit):
            module, calls, _, _ = self.synthetic_module(exception)
            self.assertEqual(runner.run_fixed_tests(module, {}), (0, 12))
            self.assertEqual(len(calls), 12)

    def test_boundary_and_deadline_abort_remaining_tests(self) -> None:
        for exception in (runner.BoundaryViolation, runner.DeadlineExceeded, runner.LocalUnitError):
            module, calls, _, _ = self.synthetic_module(exception)
            with self.assertRaises(exception):
                runner.run_fixed_tests(module, {})
            self.assertEqual(len(calls), 1)

    def test_parameter_contract_drift_is_rejected(self) -> None:
        for change in ("count", "names", "kwargs", "extra_mark", "missing_mark", "tuple_shape"):
            module, _, _, _ = self.synthetic_module()
            function = getattr(module, runner.MANIFEST_TEST)
            mark = function.pytestmark[0]
            if change == "count":
                mark.args[1].pop()
            elif change == "names":
                mark.args = (("changed", "value"), mark.args[1])
            elif change == "kwargs":
                mark.kwargs = {"ids": ["fake"]}
            elif change == "extra_mark":
                function.pytestmark.append(mark)
            elif change == "missing_mark":
                function.pytestmark = []
            else:
                mark.args[1][0] = ("bad",)
            with self.subTest(change=change), self.assertRaises(runner.LocalUnitError):
                runner.original_parameters(module)

    def test_missing_original_function_is_not_a_pass(self) -> None:
        module, _, _, _ = self.synthetic_module()
        delattr(module, runner.PLAN_TEST)
        with self.assertRaises(AttributeError):
            runner.run_fixed_tests(module, {})

    def test_fixture_failure_is_not_a_pass(self) -> None:
        module, _, _, _ = self.synthetic_module()
        module.manifest.__wrapped__ = lambda _plan: (_ for _ in ()).throw(AssertionError())
        with self.assertRaises(AssertionError):
            runner.run_fixed_tests(module, {})

    def test_preload_removes_site_before_lumi_import(self) -> None:
        site = Path("/synthetic-only/runtime-site")
        original = list(sys.path)
        calls = []

        def synthetic_import(name):
            calls.append((name, list(sys.path)))

        with patch.object(runner.importlib, "import_module", side_effect=synthetic_import):
            runner.preload_pytest(site)
        self.assertEqual(calls, [("pytest", [str(site), *original])])
        self.assertEqual(sys.path, original)

    def test_preload_restores_path_on_error(self) -> None:
        original = list(sys.path)
        with (
            patch.object(runner.importlib, "import_module", side_effect=ImportError),
            self.assertRaises(ImportError),
        ):
            runner.preload_pytest(Path("/synthetic-only/runtime-site"))
        self.assertEqual(sys.path, original)

    def test_real_code_closure_hashes_without_plan_reads(self) -> None:
        sources = runner.verify_sources(ROOT)
        self.assertEqual(len(sources), 86)
        self.assertNotIn(ROOT / runner.PLAN_PATH, sources)

    def test_source_hash_mismatch_is_rejected(self) -> None:
        with patch.object(runner, "SOURCE_BINDINGS", {runner.TEST_PATH: "0" * 64}):
            with self.assertRaisesRegex(runner.LocalUnitError, "source_hash"):
                runner.verify_sources(ROOT)

    def test_closure_hash_mismatch_is_rejected(self) -> None:
        with patch.object(runner, "CLOSURE_SHA256", "0" * 64):
            with self.assertRaisesRegex(runner.LocalUnitError, "closure_hash"):
                runner.verify_sources(ROOT)

    def test_closure_limit_is_fail_closed(self) -> None:
        with patch.object(runner, "CLOSURE_COUNT", 1):
            with self.assertRaisesRegex(runner.LocalUnitError, "closure_count"):
                runner.verify_sources(ROOT)

    def test_only_module_level_lumi_imports_define_closure(self) -> None:
        payload = b"""from . import first\nfrom .second import x\nimport lumi.third\nimport json
def later():
    from . import operational
"""
        self.assertEqual(runner.local_imports(payload), {"lumi.first", "lumi.second", "lumi.third"})

    def test_unbound_loaded_lumi_is_rejected(self) -> None:
        with patch.dict(sys.modules, {"lumi.synthetic_guard": types.SimpleNamespace(
            __file__="/synthetic-only/unbound.py"
        )}):
            with self.assertRaisesRegex(runner.LocalUnitError, "unbound_lumi_import"):
                runner.verify_loaded_lumi(set())

    def test_synthetic_plan_exact_size_and_hash(self) -> None:
        payload = b'{"synthetic_only":true}'
        with tempfile.TemporaryDirectory(prefix="lumi-seal-plan-guard-") as directory:
            base = Path(directory)
            file = base / "plan.json"
            file.write_bytes(payload)
            with (
                patch.object(runner, "PLAN_PATH", "plan.json"),
                patch.object(runner, "PLAN_BYTES", len(payload)),
                patch.object(runner, "PLAN_SHA256", hashlib.sha256(payload).hexdigest()),
            ):
                self.assertEqual(runner.verify_plan(base), runner.identity(file))
                file.write_bytes(b"x" * len(payload))
                with self.assertRaisesRegex(runner.LocalUnitError, "plan_hash"):
                    runner.verify_plan(base)

    def test_synthetic_plan_wrong_size_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lumi-seal-size-guard-") as directory:
            base = Path(directory)
            (base / "plan.json").write_bytes(b"x")
            with patch.object(runner, "PLAN_PATH", "plan.json"):
                with self.assertRaisesRegex(runner.LocalUnitError, "plan_size"):
                    runner.verify_plan(base)

    def test_path_traversal_and_symlink_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lumi-seal-path-guard-") as directory:
            base = Path(directory)
            (base / "link").symlink_to(base / "private")
            for relative in ("../escape", "/absolute", "link/file"):
                with self.subTest(relative=relative), self.assertRaises(runner.LocalUnitError):
                    runner.no_link_path(base, relative)


if __name__ == "__main__":
    unittest.main()
