"""Pure projection tests; never import or execute the remote Linux fixture."""

import importlib
import sys
import types
import unittest
from pathlib import Path

PACKAGE = "_synthetic_ipc_diagnostic_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(Path(__file__).resolve().parents[1] / "src/lumi")]
sys.modules[PACKAGE] = package
diagnostic = importlib.import_module(PACKAGE + ".posttraining_sft_ipc_diagnostic_v1")


class DiagnosticTests(unittest.TestCase):
    def test_errno_maps_only_to_fixed_category(self):
        for number, expected in diagnostic.ERRNO_CODES.items():
            with self.subTest(number=number):
                self.assertEqual(diagnostic.failure_code(OSError(number, "PRIVATE", "/PRIVATE")),
                                 expected)

    def test_unknown_errno_never_leaks_values(self):
        for number in (-1, 0, 999999, True, None, "PRIVATE", [], {}):
            self.assertEqual(diagnostic.failure_code(OSError(number, "PRIVATE")), "errno_other")

    def test_timeout_without_errno_has_fixed_code(self):
        self.assertEqual(diagnostic.failure_code(TimeoutError("PRIVATE")), "errno_timeout")

    def test_failed_expectations_are_not_passes(self):
        for error in (RuntimeError("PRIVATE"), AssertionError("PRIVATE")):
            self.assertEqual(diagnostic.failure_code(error), "expectation_failed")

    def test_unexpected_python_errors_have_no_details(self):
        for error in (TypeError("PRIVATE"), ValueError("PRIVATE"), KeyboardInterrupt()):
            self.assertEqual(diagnostic.failure_code(error), "python_failed")

    def test_hostile_subclass_text_and_errno_properties_are_never_called(self):
        class Hostile(OSError):
            @property
            def errno(self):
                raise AssertionError("PRIVATE")

            def __str__(self):
                raise AssertionError("PRIVATE")

        error = Hostile(13, "PRIVATE")
        self.assertEqual(diagnostic.failure_code(error), "errno_access")

    def test_stage_and_code_vocabularies_are_finite(self):
        self.assertEqual(len(diagnostic.STAGES), 12)
        self.assertEqual(len(diagnostic.ERROR_CODES), 19)
        for label in (*diagnostic.STAGES, *diagnostic.ERROR_CODES):
            self.assertRegex(label, r"^[a-z_]+$")
            self.assertLess(len(label), 55)


if __name__ == "__main__":
    unittest.main()
