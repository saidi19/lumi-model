"""Source push CI is not full/private readiness; preserve both lanes honestly."""

from __future__ import annotations

import importlib.util
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "frozen_workflow_lane_contract", ROOT / "tests/test_checkpoint_history_workflow_v1.py"
)
history_guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(history_guard)

SOURCE_HEADER = """name: Source checks

on:
  push:

permissions:
  contents: read

concurrency:
  group: source-checks-${{ github.ref }}
  cancel-in-progress: true

"""
FULL_HEADER = """name: Full integration readiness

on:
  pull_request:
  workflow_dispatch:

"""
PRIVATE = "private-integration-readiness"
SOURCE_JOBS = {"frozen-lint-integrity", "isolated-sft-controls", "source-only-validators", "test"}


def split_workflow(workflow: str) -> tuple[str, dict[str, str]]:
    if workflow.count("\njobs:\n") != 1:
        raise ValueError("jobs_boundary")
    header, body = workflow.split("jobs:\n", 1)
    matches = list(re.finditer(r"^  ([a-z][a-z0-9_-]*):\n", body, re.MULTILINE))
    if not matches or matches[0].start() != 0:
        raise ValueError("job_shape")
    blocks = {}
    for index, match in enumerate(matches):
        name = match.group(1)
        if name in blocks:
            raise ValueError("duplicate_job")
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        blocks[name] = body[match.start() : end].rstrip()
    return header, blocks


def validate_lanes(source: str, full: str) -> None:
    source_header, source_jobs = split_workflow(source)
    full_header, full_jobs = split_workflow(full)
    if source_header != SOURCE_HEADER or full_header != FULL_HEADER:
        raise ValueError("trigger_permission_or_concurrency_changed")
    if set(source_jobs) != SOURCE_JOBS or set(full_jobs) != SOURCE_JOBS | {PRIVATE}:
        raise ValueError("lane_job_inventory")
    if any(source_jobs[name] != full_jobs[name] for name in SOURCE_JOBS):
        raise ValueError("source_checks_changed_between_lanes")
    # Invoke the old exact contract, not a relaxed successor or skip/xfail.
    history_guard.validate_route(full)
    history_guard.validate_readiness(full)


class WorkflowLaneTests(unittest.TestCase):
    def setUp(self):
        self.source = (ROOT / ".github/workflows/source-checks.yml").read_text()
        self.full = (ROOT / ".github/workflows/tests.yml").read_text()

    def test_live_lanes_preserve_all_source_checks_and_private_pr_gate(self):
        validate_lanes(self.source, self.full)

    def test_full_readiness_cannot_automatically_fail_every_backup_push(self):
        with self.assertRaises(ValueError):
            validate_lanes(self.source, self.full.replace("on:\n", "on:\n  push:\n"))

    def test_private_gate_cannot_be_removed(self):
        with self.assertRaises(ValueError):
            validate_lanes(self.source, self.full.replace(history_guard.EXPECTED_READINESS_JOB, ""))

    def test_private_gate_cannot_be_ignored_or_made_successful(self):
        for addition in ("    if: false\n", "    continue-on-error: true\n"):
            with self.subTest(addition=addition), self.assertRaises(ValueError):
                validate_lanes(
                    self.source, self.full.replace(f"  {PRIVATE}:\n", f"  {PRIVATE}:\n{addition}")
                )
        with self.assertRaises(ValueError):
            validate_lanes(
                self.source, self.full.replace("--private-readiness", "--private-readiness || true")
            )

    def test_private_gate_cannot_be_reintroduced_into_source_backup_lane(self):
        with self.assertRaises(ValueError):
            validate_lanes(self.source + history_guard.EXPECTED_READINESS_JOB, self.full)

    def test_source_command_cannot_drift_from_required_full_checks(self):
        with self.assertRaises(ValueError):
            validate_lanes(self.source.replace("python -B -m pytest", "echo successful"), self.full)

    def test_path_filters_cannot_suppress_source_push_checks(self):
        with self.assertRaises(ValueError):
            validate_lanes(
                self.source.replace("  push:\n", "  push:\n    paths: ['README.md']\n"), self.full
            )

    def test_duplicate_automatic_pr_source_lane_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_lanes(self.source.replace("on:\n", "on:\n  pull_request:\n"), self.full)

    def test_source_token_permissions_cannot_be_broadened(self):
        with self.assertRaises(ValueError):
            validate_lanes(self.source.replace("contents: read", "contents: write"), self.full)

    def test_obsolete_source_runs_are_cancelled(self):
        with self.assertRaises(ValueError):
            validate_lanes(
                self.source.replace("cancel-in-progress: true", "cancel-in-progress: false"),
                self.full,
            )

    def test_duplicate_or_missing_source_jobs_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_lanes(self.source.replace("  test:\n", "  unexpected:\n"), self.full)
        with self.assertRaises(ValueError):
            validate_lanes(self.source + "  test:\n    runs-on: ubuntu-latest\n", self.full)

    def test_history_and_source_scope_cannot_be_weakened_in_both_lanes(self):
        old = "-p scripts.ci_source_partition_v1 --lumi-source-suite tests"
        with self.assertRaises(ValueError):
            validate_lanes(
                self.source.replace(old, "tests -k small"), self.full.replace(old, "tests -k small")
            )


if __name__ == "__main__":
    unittest.main()
