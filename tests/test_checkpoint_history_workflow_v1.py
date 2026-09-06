"""Guard the exact BF16 CI route; no ML, private inputs or child execution.

This intentionally checks the literal reviewed test-job block. It is a routing
regression guard, not a general YAML parser or an all-tests-passed claim.
"""

from __future__ import annotations

import importlib.util
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_checkpoint_import_history_ci_v1.py"
SPEC = importlib.util.spec_from_file_location("history_workflow_contract", SCRIPT)
history = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(history)

NODES = (
    "tests/test_sft_checkpoint_import_torch_v1.py::"
    "CheckpointImportTorchTests::test_real_bfloat16_roundtrip",
    "tests/test_sft_checkpoint_import_torch_v1.py::"
    "CheckpointImportTorchTests::test_real_adapter_bfloat16_and_worker_consumer",
)
HISTORY_COMMAND = "        run: python -B scripts/run_checkpoint_import_history_ci_v1.py\n"
EXPECTED_TEST_JOB = f"""  test:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
          cache: pip
      - run: python -m pip install --upgrade pip
      - run: python -m pip install -e '.[dev,data]'
      - name: Frozen BF16 failure reproduction and current successor acceptance
        timeout-minutes: 4
{HISTORY_COMMAND}      - name: General suite (control suites run in their required isolated job)
        if: always()
        run: >-
          python -B -m pytest
          --ignore=tests/test_sft_source_bundle_v1.py
          --ignore=tests/test_sft_supervisor_v1.py
          --ignore=tests/test_sft_worker_admission_v1.py
          --ignore=tests/test_sft_launch_contract_v1.py
          --ignore=tests/test_process_watchdog_v1.py
          --ignore=tests/test_sft_guarded_bootstrap_v1.py
          --ignore=tests/test_sft_ledger_protocol_v1.py
          --deselect={NODES[0]}
          --deselect={NODES[1]}
"""


def validate_route(workflow: str) -> None:
    matches = list(re.finditer(r"^  test:\n", workflow, re.MULTILINE))
    if len(matches) != 1:
        raise ValueError("test_job_inventory")
    tail = workflow[matches[0].start():]
    next_job = re.search(r"\n  [A-Za-z0-9_-]+:\n", tail)
    block = tail[:next_job.start() + 1] if next_job is not None else tail
    if block.rstrip() != EXPECTED_TEST_JOB.rstrip():
        raise ValueError("history_route_changed")


class CheckpointHistoryWorkflowTests(unittest.TestCase):
    def test_real_workflow_preserves_exact_required_history_and_general_steps(self) -> None:
        validate_route((ROOT / ".github/workflows/tests.yml").read_text())

    def test_both_routed_nodes_are_executed_by_the_fixed_history_runner(self) -> None:
        actual = {
            "tests/test_sft_checkpoint_import_torch_v1.py::"
            + name.removeprefix("checkpoint_history_v1.").replace(".", "::")
            for name in history.EXPECTED_HISTORY
        }
        self.assertEqual(actual, set(NODES))
        self.assertEqual([item[2:] for item in history.SUITES], [(14, 2), (17, 0)])
        self.assertEqual(len(history.BINDINGS), 6)

    def test_missing_or_nonblocking_history_step_is_rejected(self) -> None:
        for changed in (
            EXPECTED_TEST_JOB.replace(HISTORY_COMMAND, ""),
            EXPECTED_TEST_JOB.replace(
                HISTORY_COMMAND, "        continue-on-error: true\n" + HISTORY_COMMAND
            ),
        ):
            with self.subTest(nonblocking="continue-on-error" in changed):
                with self.assertRaisesRegex(ValueError, "history_route_changed"):
                    validate_route(changed)

    def test_changed_history_command_or_deadline_is_rejected(self) -> None:
        for old, new in (
            (HISTORY_COMMAND, HISTORY_COMMAND.rstrip() + " || true\n"),
            ("timeout-minutes: 4", "timeout-minutes: 20"),
            ("python -B scripts/run_checkpoint", "python -S -B scripts/run_checkpoint"),
        ):
            with self.subTest(change=old), self.assertRaises(ValueError):
                validate_route(EXPECTED_TEST_JOB.replace(old, new))

    def test_general_suite_must_still_run_after_history_failure(self) -> None:
        for replacement in ("", "        if: success()\n"):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                validate_route(EXPECTED_TEST_JOB.replace("        if: always()\n", replacement))

    def test_missing_additional_or_module_wide_deselections_are_rejected(self) -> None:
        selected = f"          --deselect={NODES[0]}\n"
        for replacement in (
            "",
            selected + "          --deselect=tests/test_new.py::test_new\n",
            "          --ignore=tests/test_sft_checkpoint_import_torch_v1.py\n",
        ):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                validate_route(EXPECTED_TEST_JOB.replace(selected, replacement))

    def test_current_v2_cases_cannot_be_deselected(self) -> None:
        changed = EXPECTED_TEST_JOB.replace(
            f"--deselect={NODES[0]}", f"--deselect={NODES[0].replace('_v1.py', '_v2.py')}"
        )
        with self.assertRaisesRegex(ValueError, "history_route_changed"):
            validate_route(changed)

    def test_duplicate_missing_or_conditional_test_job_is_rejected(self) -> None:
        for changed in (
            "",
            EXPECTED_TEST_JOB + "\n" + EXPECTED_TEST_JOB,
            EXPECTED_TEST_JOB.replace("  test:\n", "  test:\n    if: false\n"),
        ):
            with self.subTest(size=len(changed)), self.assertRaises(ValueError):
                validate_route(changed)

    def test_unrelated_job_does_not_change_literal_test_job_boundary(self) -> None:
        other = "  unrelated:\n    runs-on: ubuntu-latest\n"
        validate_route(other + EXPECTED_TEST_JOB)
        validate_route(EXPECTED_TEST_JOB + other)

    def test_route_guard_is_in_the_source_only_job(self) -> None:
        workflow = (ROOT / ".github/workflows/tests.yml").read_text()
        section = workflow.split("  source-only-validators:\n", 1)[1].split("  test:\n", 1)[0]
        self.assertIn("python -S -B -m unittest -q", section)
        self.assertIn("tests/test_checkpoint_history_workflow_v1.py", section)
        self.assertNotIn("if:", section)
        self.assertNotIn("continue-on-error", section)


if __name__ == "__main__":
    unittest.main()
