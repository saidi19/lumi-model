"""Exact source-reviewed CI routing, never a private integration pass.

The pinned manifest records manual dependency review. Hashes and witness line
bounds authenticate that review; this is not an inferred call graph. Required
private paths are descriptive strings and are never opened or inspected. Only
the two pinned manifests and their reviewed source/support bindings are read.
No exception is converted to a pass, skip or xfail. Without the explicit pytest
option this plugin is inert. Private readiness remains a separate blocking gate.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
from collections import Counter
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "configs/lumi-ci-source-partition-v0.1.json"
SOURCE_PARTITION_MANIFEST_SHA256 = (
    "4e595b654880668a4462c7b8db7c40eaf715ebd86b90eb14cdf852720bc831ae"
)
BASELINE = "configs/lumi-ci-unresolved-inventory-v0.1.json"
BASELINE_SHA256 = "441390c9c98cd4b83d224055a5af5ae421e5ef01b2e4b86101e7b17693d87665"
SCHEMA = "lumi.ci-source-partition.v0.1"
STATE = "_lumi_source_partition_v1"
HASH = re.compile(r"[0-9a-f]{64}\Z")
IGNORES = (
    "tests/test_sft_source_bundle_v1.py", "tests/test_sft_supervisor_v1.py",
    "tests/test_sft_worker_admission_v1.py", "tests/test_sft_launch_contract_v1.py",
    "tests/test_process_watchdog_v1.py", "tests/test_sft_guarded_bootstrap_v1.py",
    "tests/test_sft_ledger_protocol_v1.py",
)
HISTORY = (
    "tests/test_sft_checkpoint_import_torch_v1.py::"
    "CheckpointImportTorchTests::test_real_bfloat16_roundtrip",
    "tests/test_sft_checkpoint_import_torch_v1.py::"
    "CheckpointImportTorchTests::test_real_adapter_bfloat16_and_worker_consumer",
)


class PartitionError(ValueError):
    """Fixed public error codes, never private or exception text."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise PartitionError(code)


def require_assertions(optimize: int) -> None:
    require(type(optimize) is int and optimize == 0, "assertions_disabled")


def no_duplicates(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate_json_key")
        result[key] = value
    return result


def source_path(name: object) -> bool:
    if type(name) is not str:
        return False
    path = PurePosixPath(name)
    return (
        re.fullmatch(r"(?:src|scripts|tests)/[A-Za-z0-9_/]+\.py", name) is not None
        and str(path) == name and ".." not in path.parts
    )


def support_path(name: object) -> bool:
    return type(name) is str and re.fullmatch(r"configs/[A-Za-z0-9_.-]+\.json", name) is not None


def required_path(name: object) -> bool:
    if type(name) is not str or not re.fullmatch(r"[A-Za-z0-9_./-]+", name):
        return False
    path = PurePosixPath(name)
    parts = path.parts
    if path.is_absolute() or str(path) != name or len(parts) < 2:
        return False
    return ".." not in parts or (parts[:2] == ("..", "lumi-artifacts") and ".." not in parts[1:])


def read_bound_file(root: Path, name: str, limit: int = 2_097_152) -> bytes:
    """Descriptor-relative bounded no-follow read of reviewed files only."""
    require(name in {MANIFEST, BASELINE} or source_path(name) or support_path(name), "read_path")
    require(hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY"), "platform")
    handles = []
    try:
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY
        current = os.open(root, flags)
        handles.append(current)
        parts = PurePosixPath(name).parts
        for part in parts[:-1]:
            current = os.open(part, flags, dir_fd=current)
            handles.append(current)
        descriptor = os.open(
            parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=current,
        )
        handles.append(descriptor)
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "file_shape")
        require(0 < before.st_size <= limit, "file_size")
        chunks, remaining = [], limit + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 65_536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        named = os.stat(parts[-1], dir_fd=current, follow_symlinks=False)
        fields = (
            "st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns",
        )
        require(all(getattr(before, key) == getattr(after, key) == getattr(named, key)
                    for key in fields), "file_changed")
        require(len(payload) == before.st_size, "file_changed")
        return payload
    except OSError:
        raise PartitionError("file_unreadable_or_symlink") from None
    finally:
        for descriptor in reversed(handles):
            os.close(descriptor)


def decode(payload: bytes) -> dict:
    try:
        value = json.loads(payload, object_pairs_hook=no_duplicates)
    except (UnicodeError, ValueError):
        raise PartitionError("manifest_json") from None
    require(type(value) is dict, "manifest_schema")
    return value


def validate_manifest(value: dict, baseline: dict) -> dict:
    require(type(value) is dict and set(value) == {
        "schema_version", "baseline_inventory_sha256", "entries", "bindings",
    }, "manifest_schema")
    require(value["schema_version"] == SCHEMA, "manifest_schema")
    require(value["baseline_inventory_sha256"] == BASELINE_SHA256, "baseline_binding")
    entries, bindings = value["entries"], value["bindings"]
    require(type(entries) is list and 0 < len(entries) <= 327, "entry_count")
    require(type(bindings) is dict and 0 < len(bindings) <= 1_000, "bindings")
    known = {entry["nodeid"]: entry for entry in baseline["entries"]}
    nodes, witnesses, private_paths = [], set(), set()
    for entry in entries:
        require(type(entry) is dict and set(entry) == {
            "nodeid", "classification", "status", "witness", "reason_code",
        }, "entry_schema")
        node = entry["nodeid"]
        require(type(node) is str and node in known, "unknown_node")
        require(entry["status"] == "OPEN", "integration_not_closed")
        require(type(entry["classification"]) is str and
                entry["classification"] in {"private_fixture", "historical_source"},
                "classification")
        historical = known[node]["review_lane"] == "historical_source_and_private_integration"
        require((entry["classification"] == "historical_source") == historical,
                "historical_classification")
        require(type(entry["reason_code"]) is str and
                re.fullmatch(r"[a-z][a-z0-9_]{2,99}", entry["reason_code"]) is not None,
                "reason_code")
        witness = entry["witness"]
        require(type(witness) is dict and set(witness) == {
            "test_path", "test_line", "callee", "implementation_path", "implementation_line",
            "binding_key", "required_path",
        }, "witness_schema")
        require(witness["test_path"] == node.split("::", 1)[0], "witness_test")
        for field in ("test_path", "implementation_path"):
            require(source_path(witness[field]) and witness[field] in bindings, "witness_source")
            witnesses.add(witness[field])
        for field in ("test_line", "implementation_line"):
            require(type(witness[field]) is int and witness[field] > 0, "witness_line")
        for field in ("callee", "binding_key"):
            require(type(witness[field]) is str and
                    re.fullmatch(r"[A-Za-z0-9_.\[\]-]{1,200}", witness[field]) is not None,
                    "witness_label")
        require(required_path(witness["required_path"]), "required_path")
        private_paths.add(witness["required_path"])
        nodes.append(node)
    require(nodes == sorted(set(nodes)), "node_order_or_duplicate")
    require(not private_paths.intersection(bindings), "private_path_cannot_be_read")
    require(witnesses <= set(bindings), "missing_witness_binding")
    require(sum(item.get("bytes", 0) for item in bindings.values()
                if type(item) is dict and type(item.get("bytes")) is int) <= 33_554_432,
            "binding_byte_limit")
    for name, binding in bindings.items():
        require(source_path(name) or support_path(name), "binding_path")
        require(type(binding) is dict and set(binding) == {"bytes", "sha256"}, "binding_schema")
        require(type(binding["bytes"]) is int and 0 < binding["bytes"] <= 2_097_152, "binding_size")
        require(type(binding["sha256"]) is str and bool(HASH.fullmatch(binding["sha256"])),
                "binding_hash")
        if name in baseline["test_files"]:
            require(binding == baseline["test_files"][name], "historical_test_changed")
    return value


def check(root: Path) -> dict:
    require_assertions(sys.flags.optimize)
    payload = read_bound_file(root, BASELINE)
    require(hashlib.sha256(payload).hexdigest() == BASELINE_SHA256, "baseline_hash")
    baseline = decode(payload)
    payload = read_bound_file(root, MANIFEST)
    require(bool(HASH.fullmatch(SOURCE_PARTITION_MANIFEST_SHA256)), "manifest_not_bound")
    require(hashlib.sha256(payload).hexdigest() == SOURCE_PARTITION_MANIFEST_SHA256,
            "manifest_hash")
    manifest = validate_manifest(decode(payload), baseline)
    lines = {}
    for name, binding in manifest["bindings"].items():
        payload = read_bound_file(root, name)
        require(len(payload) == binding["bytes"] and
                hashlib.sha256(payload).hexdigest() == binding["sha256"], "bound_file_changed")
        lines[name] = len(payload.splitlines())
    for entry in manifest["entries"]:
        witness = entry["witness"]
        for prefix in ("test", "implementation"):
            require(witness[prefix + "_line"] <= lines[witness[prefix + "_path"]], "witness_line")
    return manifest


def validate_scope(arguments: list[str], environment: dict, args: list[str], ini: dict) -> None:
    expected = ["-p", "scripts.ci_source_partition_v1", "--lumi-source-suite"]
    expected += ["--ignore=" + name for name in IGNORES]
    expected += ["--deselect=" + name for name in HISTORY]
    filtered = [arg for arg in arguments if arg not in {"tests", "-q"}]
    require(Counter(filtered) == Counter(expected), "scope_arguments")
    require(arguments.count("tests") <= 1 and arguments.count("-q") <= 1, "scope_arguments")
    require(not environment.get("PYTEST_ADDOPTS"), "environment_filters")
    require(not environment.get("PYTEST_PLUGINS"), "environment_plugins")
    require(args == ["tests"], "test_subset")
    require(ini == {"addopts": ["-q"], "testpaths": ["tests"]}, "ini_scope")


def partition_nodes(nodeids: list[str], manifest: dict) -> tuple[set[str], set[str]]:
    require(all(type(node) is str for node in nodeids), "collection_node")
    require(len(nodeids) == len(set(nodeids)), "duplicate_collected_node")
    collected = set(nodeids)
    private = {entry["nodeid"] for entry in manifest["entries"]}
    require(private <= collected and set(HISTORY) <= collected, "missing_collected_node")
    require(not private.intersection(HISTORY), "conflicting_route")
    return collected - private - set(HISTORY), private


def enabled(config) -> bool:
    return bool(config.getoption("--lumi-source-suite"))


def pytest_addoption(parser) -> None:
    parser.addoption("--lumi-source-suite", action="store_true", default=False,
                     help="Run source tests; exact reviewed private tests remain OPEN, not passed.")


def pytest_configure(config) -> None:
    if not enabled(config):
        return
    import pytest

    try:
        require(Path(config.rootpath).resolve() == ROOT, "root_mismatch")
        validate_scope(list(config.invocation_params.args), os.environ, list(config.args), {
            "addopts": config.getini("addopts"), "testpaths": config.getini("testpaths"),
        })
        manifest = check(ROOT)
    except PartitionError as error:
        raise pytest.UsageError(f"source partition: {error}") from None
    setattr(config, STATE, {"manifest": manifest, "collected": [], "validated": False})


def pytest_itemcollected(item) -> None:
    if hasattr(item.config, STATE):
        getattr(item.config, STATE)["collected"].append(item.nodeid)


def pytest_collection_modifyitems(config, items) -> None:
    if not hasattr(config, STATE):
        return
    import pytest

    state = getattr(config, STATE)
    try:
        source, private = partition_nodes(state["collected"], state["manifest"])
        current = [item.nodeid for item in items]
        require(len(current) == len(set(current)), "duplicate_current_node")
        require(source | private <= set(current) <= source | private | set(HISTORY),
                "collection_changed")
    except PartitionError as error:
        raise pytest.UsageError(f"source partition: {error}") from None
    routed = [item for item in items if item.nodeid in private]
    items[:] = [item for item in items if item.nodeid not in private]
    config.hook.pytest_deselected(items=routed)
    state.update(source=source, private=private)


def pytest_collection_finish(session) -> None:
    if not hasattr(session.config, STATE):
        return
    import pytest

    state = getattr(session.config, STATE)
    actual = [item.nodeid for item in session.items]
    try:
        require("source" in state and set(actual) == state["source"] and
                len(actual) == len(state["source"]), "final_collection_accounting")
        require(session.testsfailed == 0, "collection_failed")
    except PartitionError as error:
        raise pytest.UsageError(f"source partition: {error}") from None
    state["validated"] = True


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    if not hasattr(config, STATE):
        return
    state = getattr(config, STATE)
    terminalreporter.write_line(json.dumps({
        "status": "SOURCE_SCOPE_ACCOUNTED" if state["validated"] else "STOP",
        "source_selected": len(state.get("source", ())),
        "private_NOT_EXECUTED": len(state.get("private", ())),
        "history_routed_separately": len(HISTORY),
        "private_integration_passed": False,
        "full_ci_passed": False,
    }, sort_keys=True))


def private_readiness(root: Path) -> dict:
    manifest = check(root)
    return {
        "status": "BLOCKED_PRIVATE_INTEGRATION_UNAVAILABLE",
        "open_baseline_nodes": 327,
        "reviewed_private_nodes": len(manifest["entries"]),
        "unreviewed_nodes": 327 - len(manifest["entries"]),
        "private_tests_executed": 0,
        "private_integration_passed": False,
        "full_ci_passed": False,
        "production_eligible": False,
    }


def main() -> int:
    try:
        require(sys.argv[1:] == ["--private-readiness"], "arguments")
        report = private_readiness(ROOT)
    except BaseException:
        report = {"status": "STOP", "private_integration_passed": False, "full_ci_passed": False}
    print(json.dumps(report, sort_keys=True))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
