"""Synthetic unit tests for the fixed Spark source-CI gate; no Spark is started."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import Mock

import pytest

EXPECTED_NODES = (
    "tests/test_distributed_corpus.py::"
    "test_local_spark_dedup_matches_independent_sequential_reference",
    "tests/test_final_corpus.py::test_local_spark_hash_selection_matches_python_rank",
    "tests/test_glue_production_dedup.py::"
    "test_memory_safe_projection_streams_long_text_only_after_selection",
)
PHASES = ("setup", "call", "teardown")


class VersionInfo(NamedTuple):
    major: int
    minor: int
    micro: int = 0
    releaselevel: str = "final"
    serial: int = 0


class MinimalReport:
    """Any access beyond the three approved report fields is a test failure."""

    __slots__ = ("nodeid", "outcome", "when")

    def __init__(self, nodeid, when, outcome):
        self.nodeid = nodeid
        self.when = when
        self.outcome = outcome

    def __getattr__(self, name):
        raise AssertionError(f"Runner inspected an unapproved report field: {name}")


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_spark_source_ci_v1.py"
    spec = importlib.util.spec_from_file_location("synthetic_spark_source_ci_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def items(nodes=EXPECTED_NODES):
    return [SimpleNamespace(nodeid=node) for node in nodes]


def collect(plugin, nodes=EXPECTED_NODES):
    plugin.pytest_collection_modifyitems(items=items(nodes))


def report(plugin, node, phase, outcome="passed"):
    plugin.pytest_runtest_logreport(MinimalReport(node, phase, outcome))


def complete_reports(plugin, *, omit=None, replace=None):
    for node in EXPECTED_NODES:
        for phase in PHASES:
            if (node, phase) == omit:
                continue
            outcome = "passed"
            if replace is not None and (node, phase) == replace[:2]:
                outcome = replace[2]
            report(plugin, node, phase, outcome)


def fake_runtime(monkeypatch, runner, *, version=(3, 12), optimize=0, pyspark="3.5.6"):
    monkeypatch.setattr(
        runner,
        "sys",
        SimpleNamespace(
            version_info=VersionInfo(*version),
            flags=SimpleNamespace(optimize=optimize, no_site=0),
            executable="/synthetic/python",
        ),
    )
    monkeypatch.setitem(sys.modules, "pyspark", SimpleNamespace(__version__=pyspark))
    probes = []
    monkeypatch.setattr(runner, "probe_java", lambda: probes.append("java"))
    return probes


def invoke_main(monkeypatch, runner, capsys, *, lifecycle=None, exit_code=0, argv=None):
    calls = []
    runtime_checks = []

    def fake_pytest_main(args, plugins):
        calls.append((list(args), list(plugins)))
        assert len(plugins) == 1
        plugin = plugins[0]
        assert isinstance(plugin, runner.FixedSparkNodes)
        print("synthetic pytest log remains visible")
        if lifecycle is None:
            collect(plugin)
            complete_reports(plugin)
        else:
            lifecycle(plugin)
        return exit_code

    def fake_import(name):
        assert name == "pytest"
        return SimpleNamespace(main=fake_pytest_main)

    monkeypatch.setattr(runner, "check_runtime", lambda: runtime_checks.append("checked"))
    monkeypatch.setattr(runner, "configure_environment", lambda: None)
    monkeypatch.setattr(runner, "importlib", SimpleNamespace(import_module=fake_import))
    result = runner.main([] if argv is None else argv)
    captured = capsys.readouterr()
    lines = captured.out.strip().splitlines()
    assert lines, "The runner must print its final aggregate JSON"
    summary = json.loads(lines[-1])
    return result, summary, captured, calls, runtime_checks


def test_fixed_node_contract_is_exact(runner):
    assert runner.NODES == EXPECTED_NODES
    assert issubclass(runner.SparkCIError, ValueError)


@pytest.mark.parametrize("nodes", [EXPECTED_NODES, EXPECTED_NODES[::-1]])
def test_exact_collection_and_all_passed_phases_succeed(runner, nodes):
    plugin = runner.FixedSparkNodes()
    collect(plugin, nodes)
    assert plugin.passed() is False
    complete_reports(plugin)
    assert plugin.passed() is True
    assert isinstance(plugin.summary(), dict)
    json.dumps(plugin.summary())


@pytest.mark.parametrize(
    "nodes",
    [
        (),
        EXPECTED_NODES[:2],
        (*EXPECTED_NODES, EXPECTED_NODES[0]),
        (*EXPECTED_NODES[:2], "tests/synthetic.py::test_unexpected"),
        (*EXPECTED_NODES, "tests/synthetic.py::test_unexpected"),
    ],
)
def test_collection_rejects_missing_duplicate_or_extra_nodes(runner, nodes):
    plugin = runner.FixedSparkNodes()
    with pytest.raises(runner.SparkCIError):
        collect(plugin, nodes)
    assert plugin.passed() is False


def test_reports_without_collection_cannot_pass(runner):
    plugin = runner.FixedSparkNodes()
    complete_reports(plugin)
    assert plugin.passed() is False


@pytest.mark.parametrize("node", EXPECTED_NODES)
@pytest.mark.parametrize("phase", PHASES)
def test_every_node_requires_every_phase(runner, node, phase):
    plugin = runner.FixedSparkNodes()
    collect(plugin)
    complete_reports(plugin, omit=(node, phase))
    assert plugin.passed() is False


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("outcome", ["failed", "skipped", "unknown"])
def test_nonpassing_phase_cannot_be_hidden_by_other_passes(runner, phase, outcome):
    plugin = runner.FixedSparkNodes()
    collect(plugin)
    complete_reports(plugin, replace=(EXPECTED_NODES[0], phase, outcome))
    assert plugin.passed() is False


@pytest.mark.parametrize("phase", PHASES)
def test_duplicate_report_invalidates_an_otherwise_complete_run(runner, phase):
    plugin = runner.FixedSparkNodes()
    collect(plugin)
    complete_reports(plugin)
    report(plugin, EXPECTED_NODES[0], phase)
    assert plugin.passed() is False


@pytest.mark.parametrize(
    ("node", "phase"),
    [("tests/synthetic.py::test_unexpected", "call"), (EXPECTED_NODES[0], "collection")],
)
def test_unknown_report_node_or_phase_invalidates_run(runner, node, phase):
    plugin = runner.FixedSparkNodes()
    collect(plugin)
    complete_reports(plugin)
    report(plugin, node, phase)
    assert plugin.passed() is False


@pytest.mark.parametrize("node", [EXPECTED_NODES[0], "tests/synthetic.py::test_unexpected"])
def test_deselection_invalidates_an_otherwise_complete_run(runner, node):
    plugin = runner.FixedSparkNodes()
    collect(plugin)
    complete_reports(plugin)
    plugin.pytest_deselected(items=items([node]))
    assert plugin.passed() is False


def test_empty_deselection_does_not_invalidate_run(runner):
    plugin = runner.FixedSparkNodes()
    collect(plugin)
    plugin.pytest_deselected(items=[])
    complete_reports(plugin)
    assert plugin.passed() is True


@pytest.mark.parametrize(
    "raw",
    [
        b'openjdk version "17.0.12" 2024-07-16\nOpenJDK Runtime Environment\n',
        b'java version "17.0.12" 2024-07-16 LTS\nJava(TM) SE Runtime Environment\n',
    ],
)
def test_java_17_version_output_is_accepted(runner, raw):
    assert runner.validate_java_version(raw) is None


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b'openjdk version "1.8.0_412"\n',
        b'openjdk version "11.0.24"\n',
        b'openjdk version "21.0.4"\n',
        b"not Java version output\n",
        b'openjdk version "17.0.12"\n' + b"x" * 1_048_576,
    ],
)
def test_invalid_or_oversized_java_output_is_rejected(runner, raw):
    with pytest.raises(runner.SparkCIError):
        runner.validate_java_version(raw)


def test_environment_is_local_and_offline_without_touching_process_environment(monkeypatch, runner):
    expected = {
        "SPARK_LOCAL_IP": "127.0.0.1",
        "SPARK_LOCAL_HOSTNAME": "localhost",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "AWS_EC2_METADATA_DISABLED": "true",
    }
    environment = {key: "untrusted-existing-value" for key in expected}
    environment["SYNTHETIC_UNRELATED_SETTING"] = "preserved"
    monkeypatch.setattr(runner, "os", SimpleNamespace(environ=environment))
    runner.configure_environment()
    for key, value in expected.items():
        assert environment[key] == value
    assert environment["SYNTHETIC_UNRELATED_SETTING"] == "preserved"


def test_required_runtime_is_accepted_without_real_imports_or_java(monkeypatch, runner):
    probes = fake_runtime(monkeypatch, runner)
    runner.check_runtime()
    assert probes == ["java"]


def test_no_site_runtime_is_rejected(monkeypatch, runner):
    probes = fake_runtime(monkeypatch, runner)
    runner.sys.flags.no_site = 1
    with pytest.raises(runner.SparkCIError, match="runtime_rejected"):
        runner.check_runtime()
    assert probes == []


def mock_java_probe(monkeypatch, runner, payload):
    process = Mock()
    process.stdout.fileno.return_value = 42
    process.wait.return_value = 0
    process.poll.return_value = 0
    launch = Mock(return_value=process)
    selector = Mock()
    selector.select.return_value = [object()]
    offset = 0
    read_sizes = []

    def read(_descriptor, maximum):
        nonlocal offset
        read_sizes.append(maximum)
        block = payload[offset:offset + maximum]
        offset += len(block)
        return block

    monkeypatch.setattr(runner, "os", SimpleNamespace(
        environ={}, set_blocking=Mock(), read=read,
    ))
    monkeypatch.setattr(runner, "subprocess", SimpleNamespace(
        Popen=launch, DEVNULL=runner.subprocess.DEVNULL, PIPE=runner.subprocess.PIPE,
        STDOUT=runner.subprocess.STDOUT, TimeoutExpired=runner.subprocess.TimeoutExpired,
    ))
    monkeypatch.setattr(runner, "selectors", SimpleNamespace(
        DefaultSelector=lambda: selector, EVENT_READ=runner.selectors.EVENT_READ,
    ))
    monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic=lambda: 0))
    return process, selector, launch, read_sizes


def test_java_probe_is_capped_and_closes_handles(monkeypatch, runner):
    process, selector, launch, sizes = mock_java_probe(
        monkeypatch, runner, b'openjdk version "17.0.12"\n'
    )
    runner.probe_java()
    assert launch.call_args.args == (["java", "-version"],)
    assert launch.call_args.kwargs["shell"] is False
    assert sizes and max(sizes) <= 4096
    process.stdout.close.assert_called_once()
    selector.close.assert_called_once()
    process.kill.assert_not_called()


@pytest.mark.parametrize("failure", ["oversize", "timeout", "nonzero", "close"])
def test_java_probe_failures_stay_closed_and_cleanup(monkeypatch, runner, failure):
    payload = b'x' * 9000 if failure == "oversize" else b'openjdk version "17.0.12"\n'
    process, selector, _, sizes = mock_java_probe(monkeypatch, runner, payload)
    expected = {
        "oversize": "java_probe_output_limit",
        "timeout": "java_probe_timeout",
        "nonzero": "java_probe_failed",
        "close": "java_cleanup_failed",
    }[failure]
    if failure == "timeout":
        selector.select.return_value = []
        process.poll.return_value = None
    elif failure == "nonzero":
        process.wait.return_value = 1
    elif failure == "close":
        process.stdout.close.side_effect = OSError("SYNTHETIC_PRIVATE_DETAIL")
    with pytest.raises(runner.SparkCIError) as caught:
        runner.probe_java()
    assert str(caught.value) == expected
    process.stdout.close.assert_called_once()
    selector.close.assert_called_once()
    if failure == "timeout":
        process.kill.assert_called_once()
    if failure == "oversize":
        assert sizes == [4096, 4096, 1]


@pytest.mark.parametrize("version", [(3, 11), (3, 13), (2, 12)])
def test_wrong_python_version_is_rejected(monkeypatch, runner, version):
    probes = fake_runtime(monkeypatch, runner, version=version)
    with pytest.raises(runner.SparkCIError):
        runner.check_runtime()
    assert probes == []


@pytest.mark.parametrize("optimize", [1, 2])
def test_optimized_python_is_rejected(monkeypatch, runner, optimize):
    probes = fake_runtime(monkeypatch, runner, optimize=optimize)
    with pytest.raises(runner.SparkCIError):
        runner.check_runtime()
    assert probes == []


@pytest.mark.parametrize("version", ["3.5.5", "3.5.7", "4.0.0", "3.5.6-local"])
def test_wrong_imported_pyspark_version_is_rejected(monkeypatch, runner, version):
    probes = fake_runtime(monkeypatch, runner, pyspark=version)
    with pytest.raises(runner.SparkCIError):
        runner.check_runtime()
    assert probes == []


def test_main_runs_only_fixed_nodes_and_preserves_pytest_logs(monkeypatch, runner, capsys):
    result, summary, captured, calls, checks = invoke_main(monkeypatch, runner, capsys)
    assert result == 0
    assert summary["status"] == "PASS_fixed_synthetic_spark_nodes"
    assert "synthetic pytest log remains visible" in captured.out
    assert checks == ["checked"]
    assert len(calls) == 1
    assert tuple(arg for arg in calls[0][0] if "::" in arg) == EXPECTED_NODES


@pytest.mark.parametrize("exit_code", [1, 2, 3, 4, 5])
def test_nonzero_pytest_exit_is_stop_even_with_all_passed_reports(
    monkeypatch, runner, capsys, exit_code
):
    result, summary, _, _, _ = invoke_main(monkeypatch, runner, capsys, exit_code=exit_code)
    assert result != 0
    assert summary["status"] == "STOP"
    assert isinstance(summary["failure_code"], str)
    assert summary["failure_code"]


def test_zero_pytest_exit_without_required_reports_is_stop(monkeypatch, runner, capsys):
    result, summary, _, _, _ = invoke_main(monkeypatch, runner, capsys, lifecycle=collect)
    assert result != 0
    assert summary["status"] == "STOP"
    assert summary["failure_code"]


@pytest.mark.parametrize("argv", [["--help"], ["-k", "synthetic"], [EXPECTED_NODES[0]]])
def test_main_rejects_arguments_before_runtime_or_pytest(monkeypatch, runner, capsys, argv):
    result, summary, _, calls, checks = invoke_main(monkeypatch, runner, capsys, argv=argv)
    assert result != 0
    assert summary["status"] == "STOP"
    assert summary["failure_code"]
    assert calls == []
    assert checks == []
