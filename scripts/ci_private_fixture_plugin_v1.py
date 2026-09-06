"""Opt-in pytest routing for four reviewed, unavailable private-fixture tests.

Load with ``-p scripts.ci_private_fixture_plugin_v1 --lumi-source-only`` from
the repository root. Without the flag, no files are inspected and no tests
are changed. With the flag, the complete four-node inventory must be collected;
an incompatible subset is a policy error, not a reason to widen the inventory.

Only the pinned manifest and test source bytes are read. Fixture checks inspect
metadata without opening fixture contents. This is not an integration-test pass,
a complete fixture inventory, a runtime sandbox, or an exception-to-skip policy.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "configs/lumi-ci-private-fixture-requirements-v0.1.json"
MANIFEST_SHA256 = "1cf7f20d5bbdf476f9b8b15c376143a5b9bd6de2aa971ef63261108d0b292d3c"
HASH = re.compile(r"[0-9a-f]{64}\Z")
RELATIVE = re.compile(r"[A-Za-z0-9_./-]+\Z")
REASON = re.compile(r"[A-Za-z0-9 ,.:/()-]+\Z")
REVIEWED_NODES = frozenset(
    {
        "tests/test_aya_bengali_materialization.py::"
        "test_aya_materialization_plan_exactly_matches_owner_boundary",
        "tests/test_aya_bengali_materialization.py::"
        "test_materialization_plan_rejects_tampered_owner_statement",
        "tests/test_aya_bengali_materialization.py::test_aya_materialization_binds_exact_preflight",
        "tests/test_ai_gap_source_registration_v1.py::"
        "test_canary_evidence_reconstructs_without_candidate_rows",
    }
)
STATE = "_lumi_private_fixture_unavailable_v1"


class FixturePolicyError(ValueError):
    """A fixed policy error code, never private content or an exception transcript."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise FixturePolicyError(code)


def valid_path(value: object, roots: set[str]) -> bool:
    if not isinstance(value, str) or not RELATIVE.fullmatch(value):
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and str(path) == value
        and len(path.parts) >= 2
        and ".." not in path.parts
        and path.parts[0] in roots
    )


def validate_manifest(value: object) -> dict:
    require(isinstance(value, dict), "manifest_schema")
    require(set(value) == {"schema_version", "entries"}, "manifest_schema")
    require(
        value["schema_version"] == "lumi.ci-private-fixture-requirements.v0.1",
        "manifest_schema",
    )
    entries = value["entries"]
    require(isinstance(entries, list) and len(entries) == 4, "manifest_inventory")
    seen: set[str] = set()
    source_hashes: dict[str, str] = {}
    for entry in entries:
        require(
            isinstance(entry, dict)
            and set(entry) == {"nodeid", "test_file_sha256", "static_reason", "fixtures"},
            "manifest_entry",
        )
        nodeid = entry["nodeid"]
        require(isinstance(nodeid, str) and nodeid in REVIEWED_NODES, "manifest_nodeid")
        require(nodeid not in seen, "manifest_duplicate_nodeid")
        seen.add(nodeid)
        source = nodeid.split("::", 1)[0]
        digest = entry["test_file_sha256"]
        require(isinstance(digest, str) and bool(HASH.fullmatch(digest)), "manifest_test_hash")
        require(source_hashes.get(source, digest) == digest, "manifest_test_hash_conflict")
        source_hashes[source] = digest
        reason = entry["static_reason"]
        require(
            isinstance(reason, str)
            and 20 <= len(reason) <= 400
            and reason == reason.strip()
            and bool(REASON.fullmatch(reason)),
            "manifest_static_reason",
        )
        fixtures = entry["fixtures"]
        require(isinstance(fixtures, list) and 1 <= len(fixtures) <= 2, "manifest_fixtures")
        names: set[str] = set()
        for fixture in fixtures:
            require(
                isinstance(fixture, dict) and set(fixture) == {"path", "kind"},
                "manifest_fixture",
            )
            name = fixture["path"]
            require(valid_path(name, {"configs", "evidence"}), "manifest_fixture_path")
            require(fixture["kind"] == "file", "manifest_fixture_kind")
            require(name not in names, "manifest_duplicate_fixture")
            names.add(name)
    require(seen == REVIEWED_NODES, "manifest_inventory")
    return value


def _directory_flags() -> int:
    require(hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY"), "platform_unsupported")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def read_reviewed_file(root: Path, name: str, maximum_bytes: int) -> bytes:
    """Bounded no-follow read for manifest or source only; never for a fixture."""

    allowed = {MANIFEST, *(nodeid.split("::", 1)[0] for nodeid in REVIEWED_NODES)}
    require(name in allowed, "reviewed_path")
    descriptors: list[int] = []
    try:
        current = os.open(root, _directory_flags())
        descriptors.append(current)
        parts = PurePosixPath(name).parts
        for part in parts[:-1]:
            current = os.open(part, _directory_flags(), dir_fd=current)
            descriptors.append(current)
        descriptor = os.open(
            parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=current
        )
        descriptors.append(descriptor)
        before = os.fstat(descriptor)
        require(
            stat.S_ISREG(before.st_mode)
            and before.st_nlink == 1
            and before.st_size <= maximum_bytes,
            "reviewed_file_shape",
        )
        chunks = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 65_536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        require(len(data) <= maximum_bytes, "reviewed_file_oversize")
        after = os.fstat(descriptor)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
        require(
            all(getattr(before, field) == getattr(after, field) for field in fields),
            "reviewed_file_changed",
        )
        return data
    except OSError:
        raise FixturePolicyError("reviewed_file_unreadable") from None
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def load_manifest(root: Path) -> dict:
    data = read_reviewed_file(root, MANIFEST, 16_384)
    require(hashlib.sha256(data).hexdigest() == MANIFEST_SHA256, "manifest_hash_mismatch")
    try:
        value = json.loads(data)
    except (ValueError, UnicodeError):
        raise FixturePolicyError("manifest_json") from None
    return validate_manifest(value)


def tracked_fixtures(root: Path, names: set[str]) -> set[str]:
    """Read only the local Git index; no network, fetch, hooks, or fixture reads."""

    executable = shutil.which("git", path="/usr/bin:/bin")
    require(executable is not None, "git_unavailable")
    command = [
        executable,
        "--no-optional-locks",
        "-C",
        str(root),
        "ls-files",
        "--cached",
        "--full-name",
        "-z",
        "--",
        *sorted(names),
    ]
    environment = {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
    }
    try:
        result = subprocess.run(
            command,
            cwd=root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
        require(result.returncode == 0, "git_index_failed")
        require(len(result.stdout) <= 16_384, "git_index_output")
        require(not result.stdout or result.stdout.endswith(b"\0"), "git_index_output")
        rows = result.stdout.decode("utf-8").split("\0")[:-1]
        require(len(rows) == len(set(rows)) and set(rows) <= names, "git_index_output")
        return set(rows)
    except (OSError, subprocess.TimeoutExpired, UnicodeError):
        raise FixturePolicyError("git_index_unavailable") from None


def fixture_present(root: Path, name: str) -> bool:
    """Inspect components and leaf metadata only; a fixture leaf is never opened."""

    require(valid_path(name, {"configs", "evidence"}), "fixture_path")
    descriptors: list[int] = []
    try:
        current = os.open(root, _directory_flags())
        descriptors.append(current)
        parts = PurePosixPath(name).parts
        for index, part in enumerate(parts):
            try:
                metadata = os.stat(part, dir_fd=current, follow_symlinks=False)
            except FileNotFoundError:
                return False
            require(not stat.S_ISLNK(metadata.st_mode), "fixture_symlink")
            if index == len(parts) - 1:
                require(stat.S_ISREG(metadata.st_mode), "fixture_wrong_type")
                require(metadata.st_nlink == 1, "fixture_hardlink")
                return True
            require(stat.S_ISDIR(metadata.st_mode), "fixture_parent_wrong_type")
            current = os.open(part, _directory_flags(), dir_fd=current)
            descriptors.append(current)
    except OSError:
        raise FixturePolicyError("fixture_metadata_unreadable") from None
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
    raise FixturePolicyError("fixture_path")


def assess_requirements(
    root: Path, manifest: dict, nodeids: list[str]
) -> dict[str, tuple[str, ...]]:
    manifest = validate_manifest(manifest)
    require(all(isinstance(nodeid, str) for nodeid in nodeids), "collection_nodeid")
    require(
        all(nodeids.count(nodeid) == 1 for nodeid in REVIEWED_NODES),
        "manifest_node_not_collected_once",
    )
    checked_sources: set[str] = set()
    for entry in manifest["entries"]:
        name = entry["nodeid"].split("::", 1)[0]
        if name not in checked_sources:
            data = read_reviewed_file(root, name, 1_048_576)
            require(
                hashlib.sha256(data).hexdigest() == entry["test_file_sha256"],
                "test_source_hash_mismatch",
            )
            checked_sources.add(name)
    names = {fixture["path"] for entry in manifest["entries"] for fixture in entry["fixtures"]}
    tracked = tracked_fixtures(root, names)
    require(isinstance(tracked, set) and tracked <= names, "git_index_output")
    present = {name: fixture_present(root, name) for name in sorted(names)}
    require(all(type(value) is bool for value in present.values()), "fixture_metadata_result")
    require(all(present[name] for name in tracked), "tracked_fixture_missing")
    unavailable = {}
    for entry in manifest["entries"]:
        missing = tuple(sorted(f["path"] for f in entry["fixtures"] if not present[f["path"]]))
        if missing:
            unavailable[entry["nodeid"]] = missing
    return unavailable


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--lumi-source-only",
        action="store_true",
        default=False,
        help="Report four exact private-fixture tests unavailable only when fixtures are absent.",
    )


def pytest_collection_modifyitems(config, items) -> None:
    if not config.getoption("--lumi-source-only"):
        return
    import pytest

    try:
        require(Path(config.rootpath).resolve() == ROOT, "repository_root_mismatch")
        unavailable = assess_requirements(
            ROOT, load_manifest(ROOT), [item.nodeid for item in items]
        )
    except FixturePolicyError as error:
        raise pytest.UsageError(f"lumi source-only fixture policy: {error}") from None
    for item in items:
        if item.nodeid in unavailable:
            reason = "lumi source-only: private fixtures unavailable (not passed): "
            item.add_marker(pytest.mark.skip(reason=reason + ", ".join(unavailable[item.nodeid])))
    setattr(config, STATE, unavailable)


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    if not config.getoption("--lumi-source-only") or not hasattr(config, STATE):
        return
    unavailable = getattr(config, STATE)
    terminalreporter.write_sep(
        "=", f"Private fixture coverage: {len(unavailable)} unavailable; not counted as passed"
    )
    for nodeid, missing in sorted(unavailable.items()):
        terminalreporter.write_line(f"UNAVAILABLE {nodeid}: {', '.join(missing)}")
    terminalreporter.write_line(
        "All other tests retain normal behavior. This is not a private integration-test pass."
    )
