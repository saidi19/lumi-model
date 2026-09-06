"""Check the full frozen 327-outcome inventory without running or waiving tests.

Only reviewed test sources and the pinned inventory are read. No private fixture,
Git/network command, test import, exception transcript or workflow change is used.
A successful check means the OPEN inventory is intact, not that CI has passed.
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
INVENTORY = "configs/lumi-ci-unresolved-inventory-v0.1.json"
INVENTORY_SHA256 = "441390c9c98cd4b83d224055a5af5ae421e5ef01b2e4b86101e7b17693d87665"
SCHEMA = "lumi.ci-unresolved-inventory.v0.1"
EXPECTED_RUN = {
    "repository": "saidi19/lumi-model",
    "run_id": 34011375472,
    "head_sha": "f7f81dcdf50fa6675172a0fb1e28638916282e86",
    "url": "https://github.com/saidi19/lumi-model/actions/runs/34011375472",
}
EXPECTED_SCOPE = {
    "inventory_only": True,
    "all_entries_remain_hosted_open": True,
    "skip_or_xfail_authorized": False,
    "private_inputs_included": False,
    "historical_replay_authorized": False,
    "production_eligible": False,
}
EXPECTED_TOTALS = {
    "deselected": 2, "errors": 21, "failed": 306, "passed": 1965,
    "skipped": 58, "subtests_passed": 382,
}
LANES = {
    "dependency_review_pending": 306,
    "historical_source_and_private_integration": 5,
    "local_vocab_metadata_unit": 4,
    "local_learning_plan_unit": 12,
}
HASH = re.compile(r"[0-9a-f]{64}\Z")
NODE = re.compile(r"tests/test_[a-zA-Z0-9_]+\.py::[a-zA-Z0-9_:]+(?:\[[a-zA-Z0-9_.-]+\])?\Z")


class InventoryError(ValueError):
    """Fixed error codes only."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise InventoryError(code)


def no_duplicates(pairs: list[tuple]) -> dict:
    value = {}
    for key, item in pairs:
        require(key not in value, "duplicate_json_key")
        value[key] = item
    return value


def strict_equal(left: object, right: object) -> bool:
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def validate_inventory(value: object) -> dict:
    require(type(value) is dict and set(value) == {
        "schema_version", "source_run", "scope", "totals", "test_files", "entries",
    }, "schema")
    require(value["schema_version"] == SCHEMA, "schema")
    for field, expected in (
        ("source_run", EXPECTED_RUN), ("scope", EXPECTED_SCOPE), ("totals", EXPECTED_TOTALS),
    ):
        require(strict_equal(value[field], expected), field)
    files, entries = value["test_files"], value["entries"]
    require(type(files) is dict and len(files) == 123, "source_inventory")
    for name, binding in files.items():
        path = PurePosixPath(name)
        require(
            re.fullmatch(r"tests/test_[a-zA-Z0-9_]+\.py", name) is not None
            and len(path.parts) == 2 and str(path) == name,
            "source_path",
        )
        require(type(binding) is dict and set(binding) == {"bytes", "sha256"}, "source_binding")
        require(type(binding["bytes"]) is int and 0 < binding["bytes"] <= 131_072, "source_size")
        require(type(binding["sha256"]) is str and bool(HASH.fullmatch(binding["sha256"])), "hash")
    require(type(entries) is list and len(entries) == 327, "outcome_inventory")
    nodes, source_names = [], set()
    outcomes, exceptions, lanes = Counter(), Counter(), Counter()
    for entry in entries:
        require(type(entry) is dict and set(entry) == {
            "nodeid", "outcome", "exception_type", "hosted_status", "review_lane",
        }, "entry")
        node = entry["nodeid"]
        require(type(node) is str and bool(NODE.fullmatch(node)), "nodeid")
        require(entry["hosted_status"] == "OPEN", "cannot_close_hosted_failure")
        require(entry["outcome"] in {"FAILED", "ERROR"}, "outcome")
        require(entry["exception_type"] in {
            "FileNotFoundError", "ValueError", "AssertionError",
        }, "exception_type")
        require(entry["review_lane"] in LANES, "review_lane")
        source_names.add(node.split("::", 1)[0])
        nodes.append(node)
        outcomes[entry["outcome"]] += 1
        exceptions[entry["exception_type"]] += 1
        lanes[entry["review_lane"]] += 1
    require(nodes == sorted(set(nodes)), "duplicate_or_unsorted_nodes")
    require(source_names == set(files), "source_inventory")
    require(outcomes == {"FAILED": 306, "ERROR": 21}, "outcome_totals")
    require(exceptions == {
        "FileNotFoundError": 140, "ValueError": 185, "AssertionError": 2,
    }, "exception_totals")
    require(lanes == LANES, "lane_totals")
    return value


def read_fixed_file(root: Path, name: str, limit: int) -> bytes:
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and ".." not in relative.parts, "source_path")
    require(hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY"), "platform")
    parts = relative.parts or (".",)
    handles = []
    try:
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY
        current = os.open(root, flags)
        handles.append(current)
        for part in parts[:-1]:
            current = os.open(part, flags, dir_fd=current)
            handles.append(current)
        descriptor = os.open(
            parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=current
        )
        handles.append(descriptor)
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "file_shape")
        require(0 < before.st_size <= limit, "file_size")
        chunks, remaining = [], limit + 1
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        after = os.fstat(descriptor)
        named = os.stat(parts[-1], dir_fd=current, follow_symlinks=False)
        fields = (
            "st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns",
        )
        require(all(getattr(before, key) == getattr(after, key) == getattr(named, key)
                    for key in fields), "file_changed")
        require(len(data) == before.st_size, "file_changed")
        return data
    except OSError:
        raise InventoryError("unreadable_or_symlink") from None
    finally:
        for handle in reversed(handles):
            os.close(handle)


def load_inventory(root: Path) -> dict:
    payload = read_fixed_file(root, INVENTORY, 131_072)
    require(hashlib.sha256(payload).hexdigest() == INVENTORY_SHA256, "inventory_hash")
    return validate_inventory(json.loads(payload, object_pairs_hook=no_duplicates))


def check(root: Path) -> dict:
    inventory = load_inventory(root)
    for name, binding in inventory["test_files"].items():
        data = read_fixed_file(root, name, 131_072)
        require(len(data) == binding["bytes"], "source_changed")
        require(hashlib.sha256(data).hexdigest() == binding["sha256"], "source_hash")
    return {
        "status": "PASS_OPEN_INVENTORY_INTACT",
        "hosted_ci_passed": False,
        "open_hosted_outcomes": 327,
        "unchanged_test_files": 123,
        "private_files_read": 0,
        "tests_executed": 0,
        "review_lanes": LANES,
    }


def main() -> int:
    try:
        require(sys.dont_write_bytecode and not sys.argv[1:], "startup")
        report = check(ROOT)
    except BaseException:
        print(json.dumps({"status": "STOP", "hosted_ci_passed": False}))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
