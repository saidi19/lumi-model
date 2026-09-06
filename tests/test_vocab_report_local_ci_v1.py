"""Synthetic guard tests; no private fixtures or operational diagnostic execution."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "local_vocab_ci_guard", ROOT / "scripts/check_vocab_report_local_ci_v1.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class LocalVocabGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = Path("/synthetic-only/code")
        self.allowed = (self.base,)

    def kind(self, event: str, *arguments) -> str | None:
        return runner.forbidden_event(event, arguments, self.allowed)

    def test_code_read_is_allowed(self) -> None:
        self.assertIsNone(self.kind("open", self.base / "unit.py", "r", os.O_RDONLY))

    def test_ordinary_python_assertions_are_required(self) -> None:
        runner.require_assertions(0)
        for optimize in (1, 2, False, 0.0):
            with self.subTest(optimize=optimize):
                with self.assertRaisesRegex(runner.LocalUnitError, "assertions_disabled"):
                    runner.require_assertions(optimize)

    def test_private_and_neighbor_prefix_reads_are_rejected(self) -> None:
        for path in (
            "/synthetic-only/private/plan.json",
            "/synthetic-only/code-neighbor/unit.py",
            "/synthetic-only/code/../private/plan.json",
        ):
            with self.subTest(path=path):
                self.assertEqual(self.kind("open", path, "r", os.O_RDONLY), "file_content")

    def test_raw_descriptor_read_is_rejected(self) -> None:
        self.assertEqual(self.kind("open", 4, "r", os.O_RDONLY), "file_content")

    def test_write_flags_are_rejected_even_inside_code(self) -> None:
        for flags in (os.O_WRONLY, os.O_RDWR, os.O_CREAT, os.O_TRUNC, os.O_APPEND):
            with self.subTest(flags=flags):
                self.assertEqual(self.kind("open", self.base / "unit.py", "w", flags), "writes")

    def test_raw_descriptor_write_is_rejected(self) -> None:
        self.assertEqual(self.kind("open", 4, "w", os.O_WRONLY), "writes")

    def test_network_events_are_rejected(self) -> None:
        for event in ("socket.connect", "socket.getaddrinfo", "socket.sendto", "socket.__new__"):
            with self.subTest(event=event):
                self.assertEqual(self.kind(event), "network")

    def test_process_events_are_rejected(self) -> None:
        for event in runner.PROCESS_EVENTS:
            with self.subTest(event=event):
                self.assertEqual(self.kind(event), "process")

    def test_mutation_events_are_rejected(self) -> None:
        for event in runner.WRITE_EVENTS:
            with self.subTest(event=event):
                self.assertEqual(self.kind(event), "writes")

    def test_ml_and_nested_imports_are_rejected(self) -> None:
        for name in runner.ML_MODULES:
            with self.subTest(name=name):
                self.assertEqual(self.kind("import", name), "ml_import")
                self.assertEqual(self.kind("import", name + ".internal"), "ml_import")

    def test_exact_ml_stack_denylist_is_preserved(self) -> None:
        self.assertEqual(runner.ML_MODULES, {
            "torch", "transformers", "tensorflow", "jax", "huggingface_hub", "tokenizers",
            "safetensors", "numpy", "accelerate", "datasets", "sentencepiece",
        })

    def test_stdlib_imports_are_allowed(self) -> None:
        self.assertIsNone(self.kind("import", "json"))

    def test_symlink_cannot_extend_read_scope(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lumi-vocab-guard-") as directory:
            base = Path(directory).resolve()
            code = base / "code"
            code.mkdir()
            (code / "escape").symlink_to(base / "private")
            actual = runner.forbidden_event(
                "open", (code / "escape/value", "r", os.O_RDONLY), (code,)
            )
            self.assertEqual(actual, "file_content")

    def test_exact_four_original_test_names(self) -> None:
        self.assertEqual(runner.TEST_NAMES, (
            "test_report_rejects_resealed_predecessor_binding_tamper",
            "test_report_recomputes_mapping_and_conditional_gate",
            "test_report_rejects_resealed_logical_mapping_tamper",
            "test_verification_is_linked_to_exact_report_payload_and_gate",
        ))

    def test_fixed_tests_really_execute_in_order(self) -> None:
        calls = []
        module = types.SimpleNamespace(**{
            name: (lambda selected=name: calls.append(selected)) for name in runner.TEST_NAMES
        })
        self.assertEqual(runner.run_fixed_tests(module), (4, 0))
        self.assertEqual(calls, list(runner.TEST_NAMES))

    def test_failures_are_not_expected_or_skipped(self) -> None:
        def fail():
            raise AssertionError("synthetic failure")

        module = types.SimpleNamespace(**dict.fromkeys(runner.TEST_NAMES, fail))
        self.assertEqual(runner.run_fixed_tests(module), (0, 4))

    def test_missing_tests_fail(self) -> None:
        self.assertEqual(runner.run_fixed_tests(types.SimpleNamespace()), (0, 4))

    def test_boundary_failure_aborts_remaining_tests(self) -> None:
        calls = []

        def fail():
            calls.append(1)
            raise runner.LocalUnitError("boundary")

        module = types.SimpleNamespace(**dict.fromkeys(runner.TEST_NAMES, fail))
        with self.assertRaises(runner.LocalUnitError):
            runner.run_fixed_tests(module)
        self.assertEqual(calls, [1])

    def test_deadline_is_a_boundary_failure(self) -> None:
        with self.assertRaises(runner.DeadlineExceeded):
            runner.deadline(None, None)

    def test_deadline_cannot_satisfy_a_value_error_rejection(self) -> None:
        self.assertFalse(issubclass(runner.DeadlineExceeded, ValueError))
        with self.assertRaises(runner.DeadlineExceeded):
            with self.assertRaises(ValueError):
                runner.deadline(None, None)

    def test_deadline_aborts_remaining_tests(self) -> None:
        calls = []

        def expire():
            calls.append(1)
            runner.deadline(None, None)

        module = types.SimpleNamespace(**dict.fromkeys(runner.TEST_NAMES, expire))
        with self.assertRaises(runner.DeadlineExceeded):
            runner.run_fixed_tests(module)
        self.assertEqual(calls, [1])

    def test_real_source_hashes_match_without_private_inputs(self) -> None:
        runner.verify_sources(ROOT)

    def test_source_mismatch_is_rejected(self) -> None:
        with patch.object(runner, "SOURCE_BINDINGS", {runner.TEST_PATH: "0" * 64}):
            with self.assertRaisesRegex(runner.LocalUnitError, "source_hash"):
                runner.verify_sources(ROOT)

    def test_source_missing_is_not_a_pass(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lumi-vocab-missing-") as directory:
            with self.assertRaises(FileNotFoundError):
                runner.verify_sources(Path(directory))

    def test_private_metadata_never_opens_contents(self) -> None:
        values = [(1, 2, 3, 1, 100, 5, 6), (1, 3, 3, 1, 47, 5, 6)]
        with (
            patch.object(runner, "identity", side_effect=values),
            patch.object(runner, "no_link_path", side_effect=lambda root, name: root / name),
            patch.object(Path, "open", side_effect=AssertionError("private read")),
        ):
            self.assertEqual(runner.private_metadata(self.base), tuple(values))

    def test_wrong_marker_size_is_rejected(self) -> None:
        for size in (0, 46, 48):
            values = [(1, 2, 3, 1, 100, 5, 6), (1, 3, 3, 1, size, 5, 6)]
            with (
                self.subTest(size=size),
                patch.object(runner, "identity", side_effect=values),
                patch.object(runner, "no_link_path", side_effect=lambda root, name: root / name),
                self.assertRaisesRegex(runner.LocalUnitError, "private_metadata_shape"),
            ):
                runner.private_metadata(self.base)

    def test_discard_does_not_accumulate_test_output(self) -> None:
        sink = runner.DiscardText()
        self.assertEqual(sink.write("synthetic output"), 16)
        self.assertEqual(vars(sink), {})


if __name__ == "__main__":
    unittest.main()
