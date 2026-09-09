"""In-memory structural report tests; never import the Linux fixture."""

import importlib
import sys
import types
import unittest
from pathlib import Path

PACKAGE = "_synthetic_filesystem_diagnostic_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(Path(__file__).resolve().parents[1] / "src/lumi")]
sys.modules[PACKAGE] = package
diagnostic = importlib.import_module(PACKAGE + ".posttraining_sft_filesystem_diagnostic_v1")


class DiagnosticTests(unittest.TestCase):
    def report(self, **changes):
        return {"schema_version": diagnostic.SCHEMA, "status": "failed", "stage": "ledger",
                "child_stage": "run", "worker_stage": "inputs", "worker_code": "input_rejected",
                **changes}

    def test_fixed_failure_fields_survive(self):
        self.assertEqual(diagnostic.project_failure(self.report()), self.report())
        self.assertEqual(diagnostic.component_code(self.report()),
                         "worker_run_ledger_run_inputs_input_rejected")

    def test_unknown_fields_never_retained(self):
        self.assertEqual(diagnostic.project_failure(self.report(exception="PRIVATE", pid=99)),
                         self.report())

    def test_unknown_or_nonstring_values_are_not_reported(self):
        for field in diagnostic.FIELDS:
            for value in ("PRIVATE", [], {}, 7, True, None):
                with self.subTest(field=field, value_type=type(value).__name__):
                    result = diagnostic.project_failure(self.report(**{field: value}))
                    self.assertEqual(result[field], "not_reported")
                    self.assertNotIn("PRIVATE", diagnostic.component_code(result))

    def test_wrong_schema_or_status_cannot_be_failure_evidence(self):
        for value in (None, [], {}, self.report(status="passed"), self.report(status=[]),
                      self.report(schema_version="PRIVATE"), self.report(schema_version=[])):
            self.assertIsNone(diagnostic.project_failure(value))
            self.assertIsNone(diagnostic.component_code(value))

    def test_projection_is_idempotent(self):
        result = diagnostic.project_failure(self.report(stage="PRIVATE"))
        self.assertEqual(diagnostic.project_failure(result), result)

    def test_missing_fields_cannot_be_guessed(self):
        result = diagnostic.project_failure({"schema_version": diagnostic.SCHEMA,
                                             "status": "failed"})
        self.assertTrue(all(result[name] == "not_reported" for name in diagnostic.FIELDS))


if __name__ == "__main__":
    unittest.main()
