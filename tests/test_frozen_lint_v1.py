"""Synthetic tests for fail-closed lint-debt accounting, with no model or data I/O."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import re
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_frozen_lint_v1.py"
SPEC = importlib.util.spec_from_file_location("frozen_lint_check", SCRIPT)
lint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lint)

ROOT = Path("/synthetic/repository")
NAME = "src/lumi/synthetic.py"
SOURCE = b"SYNTHETIC-NO-EXECUTION\n"


def fixture():
    finding = {
        "filename": str(ROOT / NAME),
        "code": "E501",
        "message": "Synthetic line length",
        "location": {"row": 1, "column": 101},
        "end_location": {"row": 1, "column": 102},
    }
    grouped = lint.group_findings([finding], ROOT)
    baseline = {
        "schema_version": "lumi.ci-frozen-lint.v0.1",
        "source_commit": lint.SOURCE_COMMIT,
        "ruff_version": lint.RUFF_VERSION,
        "total_findings": 1,
        "files": {
            NAME: {
                "sha256": hashlib.sha256(SOURCE).hexdigest(),
                "diagnostic_sha256": lint.digest(sorted(grouped[NAME], key=lint.canonical)),
                "count": 1,
                "rules": {"E501": 1},
            }
        },
    }
    return baseline, [finding]


class FrozenLintTests(unittest.TestCase):
    def setUp(self):
        self.baseline, self.report = fixture()

    def evaluate(self, source=SOURCE):
        return lint.evaluate(self.baseline, self.report, ROOT, lambda root, name: source)

    def fails(self, code):
        with self.assertRaisesRegex(lint.LintPolicyError, "^" + code + "$"):
            self.evaluate()

    def test_exact_debt_is_not_all_source_lint_clean(self):
        result = self.evaluate()
        self.assertEqual(result["new_findings"], 0)
        self.assertEqual(result["historical_findings_retained"], 1)
        self.assertFalse(result["all_source_lint_clean"])

    def test_new_file_finding_fails(self):
        extra = copy.deepcopy(self.report[0])
        extra["filename"] = str(ROOT / "src/lumi/new.py")
        self.report.append(extra)
        self.fails("new_lint_findings")

    def test_new_finding_in_frozen_file_fails(self):
        self.report.append(copy.deepcopy(self.report[0]))
        self.fails("historical_finding_count_changed")

    def test_removed_finding_fails_instead_of_silently_shrinking_history(self):
        self.report.clear()
        self.fails("historical_finding_count_changed")

    def test_changed_source_fails_even_if_diagnostics_identical(self):
        with self.assertRaisesRegex(lint.LintPolicyError, "historical_source_changed"):
            self.evaluate(SOURCE + b"# changed\n")

    def test_changed_rule_fails(self):
        self.report[0]["code"] = "F821"
        self.fails("historical_rules_changed")

    def test_changed_position_fails(self):
        self.report[0]["location"]["row"] = 2
        self.fails("historical_diagnostics_changed")

    def test_changed_message_fails(self):
        self.report[0]["message"] = "different warning"
        self.fails("historical_diagnostics_changed")

    def test_baseline_cannot_grandfather_unreviewed_undefined_names(self):
        self.baseline["files"][NAME]["rules"] = {"F821": 1}
        self.fails("unreviewed_rule")

    def test_baseline_count_mismatch_fails(self):
        self.baseline["total_findings"] = 2
        self.fails("baseline_count")

    def test_rule_count_mismatch_fails(self):
        self.baseline["files"][NAME]["rules"]["E501"] = 2
        self.fails("baseline_count")

    def test_boolean_count_fails(self):
        self.baseline["files"][NAME]["count"] = True
        self.fails("baseline_count")

    def test_wrong_source_commit_fails(self):
        self.baseline["source_commit"] = "a" * 40
        self.fails("baseline_source_commit")

    def test_wrong_ruff_version_fails(self):
        self.baseline["ruff_version"] = "0.0.0"
        self.fails("baseline_ruff_version")

    def test_malformed_hash_fails(self):
        self.baseline["files"][NAME]["sha256"] = "a" * 63
        self.fails("baseline_hash")

    def test_unknown_baseline_field_fails(self):
        self.baseline["ignore_everything"] = True
        self.fails("baseline_schema")

    def test_outside_diagnostic_path_fails(self):
        self.report[0]["filename"] = "/private/secret.py"
        self.fails("ruff_path")

    def test_traversal_diagnostic_path_fails(self):
        self.report[0]["filename"] = str(ROOT / "src/../secret.py")
        self.fails("ruff_path")

    def test_negative_or_boolean_location_fails(self):
        for value in (-1, 0, True, "1"):
            with self.subTest(value=value):
                self.report[0]["location"]["row"] = value
                self.fails("ruff_location")

    def test_metadata_or_archive_files_cannot_be_source_baseline(self):
        for path in (
            "",
            ".",
            "configs/example.py",
            "artifacts/example.py",
            "../src/x.py",
            "src/x.py\n",
        ):
            self.assertFalse(lint.valid_path(path))

    def test_ruff_configuration_and_file_selection_cannot_be_weakened_by_pyproject(self):
        result = mock.Mock(stdout=b"src/lumi/frozen.py\0tests/test_new.py\0")
        with mock.patch.object(lint.subprocess, "run", return_value=result) as run:
            command = lint.lint_command(ROOT)
        self.assertIn("--isolated", command)
        self.assertIn("--no-respect-gitignore", command)
        self.assertEqual(command[command.index("--select") + 1], "E,F,I,B,UP")
        self.assertEqual(command[command.index("--line-length") + 1], "100")
        self.assertEqual(command[command.index("--target-version") + 1], "py311")
        self.assertEqual(
            command[command.index("--") + 1 :], ["src/lumi/frozen.py", "tests/test_new.py"]
        )
        self.assertIn("--cached", run.call_args.args[0])

    def test_empty_file_discovery_fails(self):
        with mock.patch.object(lint.subprocess, "run", return_value=mock.Mock(stdout=b"")):
            with self.assertRaisesRegex(lint.LintPolicyError, "python_file_selection_empty"):
                lint.lint_command(ROOT)

    def test_missing_finding_fields_fail(self):
        self.report[0].pop("message")
        self.fails("ruff_finding_shape")

    def test_workflow_routes_all_control_suites_to_required_job(self):
        root = SCRIPT.parents[1]
        spec = importlib.util.spec_from_file_location(
            "control_ci_routing_contract", root / "scripts/run_sft_control_ci_v1.py"
        )
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        workflow = (root / ".github/workflows/tests.yml").read_text()
        omitted = re.findall(r"--ignore=(tests/[a-z0-9_]+\.py)", workflow)
        self.assertEqual(omitted, [name for name, _ in runner.SUITES])
        self.assertIn("python -S -B scripts/run_sft_control_ci_v1.py", workflow)
        self.assertIn("python -B -m pytest", workflow)
        self.assertNotIn("continue-on-error", workflow)


if __name__ == "__main__":
    unittest.main()
