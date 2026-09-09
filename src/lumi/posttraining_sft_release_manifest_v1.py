"""Pure, bounded, externally pinned SFT release-manifest validation.

This module inspects supplied bytes only. It does not read or authenticate any
filesystem object, import a release, verify its publisher, or permit execution.
The external SHA-256 pin must come from a separately trusted release decision;
matching a self-selected pin is not runtime-origin authentication.
"""

from __future__ import annotations

import hashlib
import json
import re

SCHEMA = "lumi.sft-release.v0.1"
PLATFORM = "linux-x86_64"
PYTHON_ABI = "cp312"
MAXIMUM_MANIFEST_BYTES = 8 * 1024 * 1024
MAXIMUM_ENTRIES = 32768
MAXIMUM_FILE_BYTES = 2 * 1024**3
MAXIMUM_TOTAL_BYTES = 16 * 1024**3
ALLOWED_MODES = frozenset((0o400, 0o444, 0o500, 0o555, 0o600, 0o644, 0o700, 0o755))
ROLE_PREFIXES = {"runtime": "runtime/", "dependency": "dependencies/", "source": "source/"}
MANIFEST_KEYS = frozenset((
    "schema", "platform", "python_abi", "interpreter", "entrypoint", "files",
))
FILE_KEYS = frozenset(("path", "sha256", "bytes", "mode", "role"))
ERROR_CODES = frozenset((
    "invalid_budget", "invalid_raw_type", "manifest_budget_exceeded", "invalid_expected_sha256",
    "manifest_hash_mismatch", "invalid_json", "duplicate_json_key", "invalid_json_number",
    "noncanonical_json", "invalid_manifest_shape", "unsupported_release_contract",
    "invalid_files", "entry_budget_exceeded", "invalid_file_shape", "invalid_path",
    "invalid_file_sha256", "invalid_file_size", "file_budget_exceeded", "total_budget_exceeded",
    "invalid_mode", "invalid_role", "role_path_mismatch", "files_not_sorted_unique",
    "file_directory_conflict", "missing_required_role", "invalid_interpreter",
    "invalid_entrypoint",
))
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMPONENT = re.compile(r"[A-Za-z0-9_.+-]{1,128}")


class ManifestError(ValueError):
    """A fixed code only; no input, path, JSON or underlying exception text."""

    def __init__(self, code: str):
        if type(code) is not str or code not in ERROR_CODES:
            code = "invalid_manifest_shape"
        self.code = code
        super().__init__(code)


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ManifestError(code)


def _budget(value: int, absolute_limit: int) -> None:
    _require(type(value) is int and 1 <= value <= absolute_limit, "invalid_budget")


def _is_sha256(value: object) -> bool:
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _path(value: object) -> None:
    _require(type(value) is str and 1 <= len(value) <= 1024, "invalid_path")
    parts = value.split("/")
    _require(2 <= len(parts) <= 16, "invalid_path")
    _require(
        all(part not in (".", "..") and _COMPONENT.fullmatch(part) for part in parts),
        "invalid_path",
    )
    _require(parts[0] in ("runtime", "dependencies", "source"), "invalid_path")


def _pairs(items: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in items:
        _require(key not in value, "duplicate_json_key")
        value[key] = item
    return value


def _reject_number(_value: str) -> None:
    raise ManifestError("invalid_json_number")


def _integer(value: str) -> int:
    # All valid numeric fields are at most 2 GiB. Bound conversion before int()
    # so a hostile integer cannot trigger arbitrary-size arithmetic or depend
    # on the interpreter's configurable decimal-conversion digit limit.
    _require(len(value) <= 20, "invalid_json_number")
    return int(value)


def _parse(raw: bytes) -> dict:
    try:
        value = json.loads(
            raw.decode("ascii"), object_pairs_hook=_pairs, parse_int=_integer,
            parse_float=_reject_number, parse_constant=_reject_number,
        )
    except ManifestError:
        raise
    except (ValueError, UnicodeError, RecursionError, OverflowError):
        raise ManifestError("invalid_json") from None
    _require(type(value) is dict and value.keys() == MANIFEST_KEYS, "invalid_manifest_shape")
    return value


def validate_manifest(
    raw: bytes,
    expected_sha256: str,
    *,
    maximum_manifest_bytes: int,
    maximum_entries: int,
    maximum_file_bytes: int,
    maximum_total_bytes: int,
) -> dict:
    """Return a fresh exact-schema snapshot after bounded byte-only validation.

    All four caller budgets and the external raw-byte pin are mandatory. The
    digest covers exactly the canonical ASCII JSON bytes, without a newline.
    No lexical release values are included in validation errors.
    """
    for budget, cap in (
        (maximum_manifest_bytes, MAXIMUM_MANIFEST_BYTES), (maximum_entries, MAXIMUM_ENTRIES),
        (maximum_file_bytes, MAXIMUM_FILE_BYTES), (maximum_total_bytes, MAXIMUM_TOTAL_BYTES),
    ):
        _budget(budget, cap)
    _require(type(raw) is bytes, "invalid_raw_type")
    _require(1 <= len(raw) <= maximum_manifest_bytes, "manifest_budget_exceeded")
    _require(_is_sha256(expected_sha256), "invalid_expected_sha256")
    _require(hashlib.sha256(raw).hexdigest() == expected_sha256, "manifest_hash_mismatch")
    value = _parse(raw)
    _require(
        value["schema"] == SCHEMA and value["platform"] == PLATFORM
        and value["python_abi"] == PYTHON_ABI,
        "unsupported_release_contract",
    )
    files = value["files"]
    _require(type(files) is list, "invalid_files")
    _require(1 <= len(files) <= maximum_entries, "entry_budget_exceeded")
    by_path, roles, total = {}, set(), 0
    previous = None
    for row in files:
        _require(type(row) is dict and row.keys() == FILE_KEYS, "invalid_file_shape")
        path, role = row["path"], row["role"]
        _path(path)
        _require(_is_sha256(row["sha256"]), "invalid_file_sha256")
        size = row["bytes"]
        _require(type(size) is int and size >= 0, "invalid_file_size")
        _require(size <= maximum_file_bytes, "file_budget_exceeded")
        total += size
        _require(total <= maximum_total_bytes, "total_budget_exceeded")
        mode = row["mode"]
        _require(type(mode) is int and mode in ALLOWED_MODES, "invalid_mode")
        _require(type(role) is str and role in ROLE_PREFIXES, "invalid_role")
        _require(path.startswith(ROLE_PREFIXES[role]), "role_path_mismatch")
        _require(previous is None or previous < path, "files_not_sorted_unique")
        components = path.split("/")
        _require(
            not any("/".join(components[:count]) in by_path for count in range(1, len(components))),
            "file_directory_conflict",
        )
        by_path[path] = row
        roles.add(role)
        previous = path
    _require(roles == ROLE_PREFIXES.keys(), "missing_required_role")
    interpreter, entrypoint = value["interpreter"], value["entrypoint"]
    _require(type(interpreter) is str and interpreter in by_path, "invalid_interpreter")
    _require(
        by_path[interpreter]["role"] == "runtime" and by_path[interpreter]["mode"] & 0o111 != 0,
        "invalid_interpreter",
    )
    _require(type(entrypoint) is str and entrypoint in by_path, "invalid_entrypoint")
    _require(by_path[entrypoint]["role"] == "source", "invalid_entrypoint")
    try:
        canonical = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
        ).encode("ascii")
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
        raise ManifestError("noncanonical_json") from None
    _require(canonical == raw, "noncanonical_json")
    return {**value, "files": [dict(row) for row in files]}
