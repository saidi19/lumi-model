"""Run exactly three unchanged synthetic Spark tests in the normal project runtime.

Requires Python 3.12, PySpark 3.5.6 and Java 17. The workflow supplies the outer
process-group deadline; this runner only bounds its Java version probe. Loopback
and offline defaults are not an OS/network sandbox. Normal synthetic pytest logs
are retained, but Java probe output and command exceptions are never printed.
No private integration, model training, or production permission is granted.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import selectors
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NODES = (
    "tests/test_distributed_corpus.py::"
    "test_local_spark_dedup_matches_independent_sequential_reference",
    "tests/test_final_corpus.py::test_local_spark_hash_selection_matches_python_rank",
    "tests/test_glue_production_dedup.py::"
    "test_memory_safe_projection_streams_long_text_only_after_selection",
)
MAXIMUM_JAVA_OUTPUT_BYTES = 8192
JAVA_TIMEOUT_SECONDS = 5
CODES = frozenset({
    "arguments_rejected", "runtime_rejected", "pyspark_version_rejected",
    "java_version_rejected", "java_probe_failed", "java_probe_timeout",
    "java_probe_output_limit", "java_cleanup_failed", "test_selection_rejected",
    "tests_failed", "runner_failed",
})


class SparkCIError(ValueError):
    def __init__(self, code):
        self.code = code if type(code) is str and code in CODES else "runner_failed"
        super().__init__(self.code)


def require(condition, code):
    if not condition:
        raise SparkCIError(code) from None


def configure_environment():
    """Fixed local worker interpreter and no optional dependency/plugin discovery."""
    os.environ.update({
        "SPARK_LOCAL_IP": "127.0.0.1",
        "SPARK_LOCAL_HOSTNAME": "localhost",
        "SPARK_WORKER_HOST": "127.0.0.1",
        "PYSPARK_PYTHON": sys.executable,
        "PYSPARK_DRIVER_PYTHON": sys.executable,
        "PYSPARK_SUBMIT_ARGS": (
            "--master local[2] --conf spark.driver.host=127.0.0.1 "
            "--conf spark.driver.bindAddress=127.0.0.1 --conf spark.ui.enabled=false pyspark-shell"
        ),
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "AWS_EC2_METADATA_DISABLED": "true",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    })
    for name in (
        "PYTEST_ADDOPTS", "PYTEST_PLUGINS", "JAVA_TOOL_OPTIONS", "_JAVA_OPTIONS",
        "JDK_JAVA_OPTIONS", "PYSPARK_GATEWAY_PORT", "PYSPARK_GATEWAY_SECRET",
        "SPARK_REMOTE", "SPARK_CONNECT_MODE_ENABLED",
    ):
        os.environ.pop(name, None)


def validate_java_version(raw: bytes):
    require(type(raw) is bytes and 0 < len(raw) <= MAXIMUM_JAVA_OUTPUT_BYTES,
            "java_version_rejected")
    first = raw.splitlines()[0]
    match = re.fullmatch(rb'(?:openjdk|java) version "([^"\r\n]+)"(?:[ \t].*)?', first)
    require(match is not None, "java_version_rejected")
    require(re.fullmatch(rb"17(?:\.[0-9]+)*(?:[-+][A-Za-z0-9._+-]+)?", match[1]) is not None,
            "java_version_rejected")


def probe_java():
    """Capped read and deadline before parsing; child remains in the workflow group."""
    process = selector = None
    failure = None
    raw = bytearray()
    deadline = time.monotonic() + JAVA_TIMEOUT_SECONDS
    try:
        process = subprocess.Popen(
            ["java", "-version"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, shell=False, env={**os.environ, "LC_ALL": "C"},
        )
        selector = selectors.DefaultSelector()
        os.set_blocking(process.stdout.fileno(), False)
        selector.register(process.stdout, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            require(remaining > 0 and selector.select(remaining), "java_probe_timeout")
            block = os.read(process.stdout.fileno(), min(
                4096, MAXIMUM_JAVA_OUTPUT_BYTES + 1 - len(raw)
            ))
            if not block:
                break
            raw.extend(block)
            require(len(raw) <= MAXIMUM_JAVA_OUTPUT_BYTES, "java_probe_output_limit")
        remaining = deadline - time.monotonic()
        require(remaining > 0, "java_probe_timeout")
        require(process.wait(timeout=remaining) == 0, "java_probe_failed")
        validate_java_version(bytes(raw))
    except SparkCIError as error:
        failure = error.code
    except subprocess.TimeoutExpired:
        failure = "java_probe_timeout"
    except BaseException:
        failure = "java_probe_failed"
    finally:
        if process is not None:
            try:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=1)
            except BaseException:
                failure = "java_cleanup_failed"
            try:
                process.stdout.close()
            except BaseException:
                failure = "java_cleanup_failed"
        if selector is not None:
            try:
                selector.close()
            except BaseException:
                failure = "java_cleanup_failed"
    if failure is not None:
        raise SparkCIError(failure) from None
    require(time.monotonic() <= deadline, "java_probe_timeout")


def check_runtime():
    require(sys.version_info[:2] == (3, 12) and not sys.flags.optimize and not sys.flags.no_site,
            "runtime_rejected")
    pyspark = importlib.import_module("pyspark")
    require(getattr(pyspark, "__version__", None) == "3.5.6", "pyspark_version_rejected")
    probe_java()


class FixedSparkNodes:
    """Exact collection plus all nine successful phase reports; never rewrite outcomes."""

    def __init__(self):
        self.collected = 0
        self.phases = {}
        self.deselected = 0
        self.invalid_reports = 0

    def pytest_collection_modifyitems(self, items):
        names = [item.nodeid for item in items]
        require(len(names) == 3 and len(set(names)) == 3 and set(names) == set(NODES),
                "test_selection_rejected")
        require(self.collected == 0, "test_selection_rejected")
        self.collected = 3

    def pytest_deselected(self, items):
        self.deselected += len(items)

    def pytest_runtest_logreport(self, report):
        node, phase, outcome = report.nodeid, report.when, report.outcome
        key = (node, phase)
        if (node not in NODES or phase not in {"setup", "call", "teardown"}
                or outcome not in {"passed", "failed", "skipped"} or key in self.phases):
            self.invalid_reports += 1
            return
        self.phases[key] = outcome

    def passed(self):
        return (self.collected == 3 and self.deselected == 0 and self.invalid_reports == 0
                and len(self.phases) == 9
                and all(value == "passed" for value in self.phases.values()))

    def summary(self):
        return {
            "collected_tests": self.collected,
            "passed_tests": sum(all(self.phases.get((node, phase)) == "passed"
                                    for phase in ("setup", "call", "teardown")) for node in NODES),
            "phase_reports": len(self.phases),
            "skipped_reports": sum(value == "skipped" for value in self.phases.values()),
            "failed_reports": sum(value == "failed" for value in self.phases.values()),
            "deselected_tests": self.deselected,
            "invalid_reports": self.invalid_reports,
        }


def main(argv=None):
    plugin = FixedSparkNodes()
    output = {
        "schema_version": "lumi.spark-source-ci.v0.1",
        "status": "STOP",
        "private_integration_tested": False,
        "training_executed": False,
        "production_eligible": False,
        "os_network_sandbox_enforced": False,
    }
    try:
        require(list(sys.argv[1:] if argv is None else argv) == [], "arguments_rejected")
        configure_environment()
        check_runtime()
        pytest = importlib.import_module("pytest")
        result = pytest.main([
            "--rootdir", str(ROOT), "--noconftest", "-o", "addopts=", "-p", "no:cacheprovider",
            "-rA", *NODES,
        ], plugins=[plugin])
        require(result == 0 and plugin.passed(), "tests_failed")
        output["status"] = "PASS_fixed_synthetic_spark_nodes"
    except SparkCIError as error:
        output["failure_code"] = error.code
    except BaseException:
        output["failure_code"] = "runner_failed"
    output.update(plugin.summary())
    print(json.dumps(output, sort_keys=True))
    return 0 if output["status"] == "PASS_fixed_synthetic_spark_nodes" else 1


if __name__ == "__main__":
    raise SystemExit(main())
