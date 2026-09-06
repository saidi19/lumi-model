"""Guard additive Spark coverage without rewriting either existing CI lane."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/spark-source-checks.yml"
INSTALL = "run: python -m pip install -e '.[dev,data]' pyspark==3.5.6"
COMMAND = """        run: >-
          timeout --signal=TERM --kill-after=30s 12m
          python -B scripts/run_spark_source_ci_v1.py
"""


def validate_workflow(text: str) -> None:
    required = (
        "name: Spark source checks\n",
        "on:\n  push:\n  pull_request:\n",
        "permissions:\n  contents: read\n",
        "  group: spark-source-checks-${{ github.ref }}\n  cancel-in-progress: true\n",
        "jobs:\n  synthetic-spark:\n    runs-on: ubuntu-latest\n    timeout-minutes: 20\n",
        '      PYTHONDONTWRITEBYTECODE: "1"\n',
        '      SPARK_LOCAL_IP: "127.0.0.1"\n',
        '      HF_HUB_OFFLINE: "1"\n',
        '      TRANSFORMERS_OFFLINE: "1"\n',
        '      AWS_EC2_METADATA_DISABLED: "true"\n',
        "      - uses: actions/checkout@v6\n        with:\n          persist-credentials: false\n",
        '      - uses: actions/setup-python@v6\n        with:\n          python-version: "3.12"\n',
        "      - uses: actions/setup-java@v5\n        with:\n"
        '          distribution: temurin\n          java-version: "17"\n',
        "        " + INSTALL + "\n",
        "        timeout-minutes: 15\n" + COMMAND,
    )
    if any(text.count(part) != 1 for part in required):
        raise ValueError("spark_workflow_contract")
    if any(token in text for token in (
        "continue-on-error", "if:", "||", "secrets.", "contents: write", "workflow_dispatch",
        "workflow_call", "pull_request_target", "paths:", "branches:", "-O ",
    )):
        raise ValueError("spark_workflow_contract")
    if len(re.findall(r"^  [a-z][a-z0-9_-]*:\n", text.split("jobs:\n", 1)[1], re.M)) != 1:
        raise ValueError("spark_workflow_contract")
    if len(re.findall(r"^        run:|^      - run:", text, re.M)) != 2:
        raise ValueError("spark_workflow_contract")


class SparkWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.workflow = WORKFLOW.read_text(encoding="utf-8")

    def test_real_workflow_is_additive_and_requires_three_original_tests(self):
        validate_workflow(self.workflow)

    def test_runtime_and_dependency_pins_cannot_disappear_or_drift(self):
        for old, new in (
            ("pyspark==3.5.6", "pyspark"),
            (".[dev,data]", ".[dev]"),
            ('java-version: "17"', 'java-version: "25"'),
            ('python-version: "3.12"', 'python-version: "3.13"'),
            ("persist-credentials: false", "persist-credentials: true"),
        ):
            with self.subTest(change=old), self.assertRaises(ValueError):
                validate_workflow(self.workflow.replace(old, new))

    def test_timeouts_and_failure_propagation_are_required(self):
        for old, new in (
            ("timeout-minutes: 20", "timeout-minutes: 60"),
            ("timeout-minutes: 15", "timeout-minutes: 60"),
            ("--kill-after=30s", "--kill-after=1h"),
            (" 12m\n", " 1h\n"),
            (COMMAND, COMMAND.rstrip() + " || true\n"),
            ("    steps:\n", "    continue-on-error: true\n    steps:\n"),
        ):
            with self.subTest(change=old), self.assertRaises(ValueError):
                validate_workflow(self.workflow.replace(old, new))

    def test_check_cannot_be_filtered_manual_or_secret_using(self):
        for old, new in (
            ("  push:\n", "  push:\n    paths: ['README.md']\n"),
            ("  pull_request:\n", "  workflow_dispatch:\n"),
            ("    steps:\n", "    if: false\n    steps:\n"),
            ("contents: read", "contents: write"),
            ("    env:\n", "    env:\n      TOKEN: ${{ secrets.AWS_TOKEN }}\n"),
        ):
            with self.subTest(change=old), self.assertRaises(ValueError):
                validate_workflow(self.workflow.replace(old, new))

    def test_duplicate_job_or_extra_run_is_rejected(self):
        for addition in (
            "\n  extra:\n    runs-on: ubuntu-latest\n",
            "\n      - run: echo unexpected\n",
        ):
            with self.subTest(addition=addition), self.assertRaises(ValueError):
                validate_workflow(self.workflow + addition)


if __name__ == "__main__":
    unittest.main()
