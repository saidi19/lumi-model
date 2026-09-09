#!/usr/bin/env python3
"""Remote pinned-base inventory; NEVER import or execute on the local Mac.

Run inside the selected read-only official CPython 3.12 slim-bookworm image,
before installing wheels. The caller independently binds its immutable image
manifest/config; echoed digests do not authenticate the currently running image.
Only the explicit library roots and interpreter are inventoried. Regular-file
symlink leaves under the two auxiliary roots may be included, but those roots
are not recursively inventoried. This is not a whole-filesystem inventory or
proof of original ownership, complete runtime closure, or execution permission.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import select
import stat
import sys
import time
from pathlib import Path

SCHEMA = "lumi.sft-oci-base-inventory.v0.1"
ROOTS = ("/usr/lib", "/usr/local/lib")
AUXILIARY_LEAF_ROOTS = ("/etc/ssl", "/usr/share/zoneinfo")
RUNTIME = "/usr/local/bin/python3.12"
SITE_PACKAGES = "/usr/local/lib/python3.12/site-packages"
MAXIMUM_FILES = 40_000
MAXIMUM_TREE_ENTRIES = 80_000
MAXIMUM_FILE_BYTES = 268_435_456
MAXIMUM_TOTAL_READ_BYTES = 1_073_741_824
MAXIMUM_OUTPUT_BYTES = 16_777_216
MAXIMUM_INPUT_BYTES = 512
MAXIMUM_METADATA_BYTES = 2_097_152
MAXIMUM_DISTRIBUTIONS = 256
MAXIMUM_RUNTIME_SECONDS = 120
MAXIMUM_PATH_BYTES = 4096
ML_ROOTS = frozenset(("torch", "numpy", "transformers", "tensorflow", "jax", "cuda", "cupy"))
ERROR_CODES = frozenset((
    "runtime_rejected", "input_rejected", "path_rejected", "tree_rejected",
    "file_rejected", "file_changed", "read_budget_exhausted", "entry_budget_exhausted",
    "output_budget_exhausted", "distribution_rejected", "deadline_exhausted",
    "output_failed", "inventory_failed",
))


class InventoryError(ValueError):
    def __init__(self, code):
        self.code = code if type(code) is str and code in ERROR_CODES else "inventory_failed"
        super().__init__(self.code)


def need(condition, code):
    if not condition:
        raise InventoryError(code) from None


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("ascii")


def below(path, root):
    return path == root or path.startswith(root + "/")


def ordinary_path(path):
    need(type(path) is str and 1 <= len(path.encode("utf-8")) <= MAXIMUM_PATH_BYTES
         and path.startswith("/") and not path.startswith("//")
         and "\\" not in path and str(Path(path)) == path and ".." not in Path(path).parts
         and all(32 <= ord(char) < 127 for char in path), "path_rejected")


def permitted(path):
    return path == RUNTIME or any(below(path, root) for root in (*ROOTS, *AUXILIARY_LEAF_ROOTS))


def resolved(path):
    ordinary_path(path)
    result = str(Path(path).resolve(strict=True))
    ordinary_path(result)
    need(permitted(result), "path_rejected")
    return result


def signature(info):
    return tuple(getattr(info, key) for key in (
        "st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink", "st_size",
        "st_mtime_ns", "st_ctime_ns",
    ))


def runtime_guard():
    need(sys.platform == "linux" and os.uname().machine == "x86_64"
         and sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 12)
         and sys.flags.isolated == 1 and sys.dont_write_bytecode, "runtime_rejected")
    need(sys.argv[1:] == ["--base-inventory-only"]
         and os.environ.get("LUMI_OCI_BASE_INVENTORY_CI") == "1", "runtime_rejected")
    need(str(Path(sys.executable).resolve(strict=True)) == RUNTIME
         and sys.prefix == sys.base_prefix == "/usr/local", "runtime_rejected")
    need(not any(name.split(".", 1)[0] in ML_ROOTS for name in sys.modules),
         "runtime_rejected")


class Budget:
    def __init__(self):
        self.deadline = time.monotonic() + MAXIMUM_RUNTIME_SECONDS
        self.requested = self.entries = self.row_bytes = 0

    def check(self):
        need(time.monotonic() < self.deadline, "deadline_exhausted")

    def entry(self):
        self.check()
        need(self.entries < MAXIMUM_TREE_ENTRIES, "entry_budget_exhausted")
        self.entries += 1

    def reserve_read(self, amount):
        self.check()
        need(type(amount) is int and 0 < amount <= MAXIMUM_TOTAL_READ_BYTES - self.requested,
             "read_budget_exhausted")
        self.requested += amount


def unique_pairs(items):
    value = {}
    for key, item in items:
        need(key not in value, "input_rejected")
        value[key] = item
    return value


def read_input(budget):
    raw = bytearray()
    while True:
        budget.check()
        need(select.select([0], [], [], 5)[0], "input_rejected")
        part = os.read(0, MAXIMUM_INPUT_BYTES + 1 - len(raw))
        if not part:
            break
        raw.extend(part)
        need(len(raw) <= MAXIMUM_INPUT_BYTES, "input_rejected")
    value = json.loads(raw.decode("ascii"), object_pairs_hook=unique_pairs)
    need(type(value) is dict and set(value) == {"digest", "config_digest"}, "input_rejected")
    need(all(type(value[key]) is str and re.fullmatch(r"sha256:[0-9a-f]{64}", value[key])
             for key in value), "input_rejected")
    return value


def read_file(path, budget, *, keep=False):
    """One stable regular-file read, prepaid including EOF; no symlink open."""
    ordinary_path(path)
    need(permitted(path), "path_rejected")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        maximum = MAXIMUM_METADATA_BYTES if keep else MAXIMUM_FILE_BYTES
        need(stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= maximum, "file_rejected")
        budget.reserve_read(before.st_size + 1)
        digest, remaining = hashlib.sha256(), before.st_size
        chunks = [] if keep else None
        while remaining:
            budget.check()
            part = os.read(descriptor, min(65_536, remaining))
            need(bool(part), "file_changed")
            remaining -= len(part)
            digest.update(part)
            if keep:
                chunks.append(part)
        need(os.read(descriptor, 1) == b"", "file_changed")
        need(signature(os.fstat(descriptor)) == signature(before)
             == signature(os.stat(path, follow_symlinks=False)), "file_changed")
        budget.check()
        row = {"path": path, "bytes": before.st_size, "sha256": digest.hexdigest()}
        return row, b"".join(chunks) if keep else None
    finally:
        os.close(descriptor)


def files(budget):
    pending = [*ROOTS, RUNTIME]
    directories, result = set(), {}
    while pending:
        budget.entry()
        original = pending.pop()
        path = resolved(original)
        info = os.stat(path, follow_symlinks=False)
        if stat.S_ISDIR(info.st_mode):
            if any(below(path, root) for root in AUXILIARY_LEAF_ROOTS):
                # Debian SSL/zoneinfo directory aliases are not traversal roots.
                continue
            need(any(below(path, root) for root in ROOTS), "tree_rejected")
            if path in directories:
                continue
            directories.add(path)
            with os.scandir(path) as entries:
                for entry in entries:
                    budget.check()
                    need(len(pending) + budget.entries < MAXIMUM_TREE_ENTRIES,
                         "entry_budget_exhausted")
                    pending.append(path + "/" + entry.name)
            need(signature(os.stat(path, follow_symlinks=False)) == signature(info),
                 "file_changed")
        else:
            need(stat.S_ISREG(info.st_mode), "tree_rejected")
            if path in result:
                continue
            need(len(result) < MAXIMUM_FILES, "entry_budget_exhausted")
            row, _raw = read_file(path, budget)
            budget.row_bytes += len(canonical(row)) + 1
            need(budget.row_bytes <= MAXIMUM_OUTPUT_BYTES, "output_budget_exhausted")
            result[path] = row
    need(RUNTIME in result, "runtime_rejected")
    return result


class SuppliedMetadata(importlib.metadata.Distribution):
    """Parse bounded supplied metadata with stdlib; never load package code."""

    def __init__(self, raw):
        self.text = raw.decode("utf-8")

    def read_text(self, filename):
        return self.text if filename == "METADATA" else None

    def locate_file(self, path):
        raise InventoryError("distribution_rejected") from None


def base_distributions(rows, budget):
    observed = {}
    for distribution in importlib.metadata.distributions(path=[SITE_PACKAGES]):
        budget.check()
        need(len(observed) < MAXIMUM_DISTRIBUTIONS
             and type(distribution) is importlib.metadata.PathDistribution,
             "distribution_rejected")
        # Discovery only enumerates the explicit directory. Do not let metadata
        # helpers reopen arbitrary paths or unbounded METADATA/PKG-INFO content.
        directory = str(distribution._path)
        ordinary_path(directory)
        relative = directory.removeprefix(SITE_PACKAGES + "/")
        need(directory.startswith(SITE_PACKAGES + "/") and "/" not in relative
             and relative.endswith(".dist-info") and resolved(directory) == directory,
             "distribution_rejected")
        path = directory + "/METADATA"
        need(path in rows, "distribution_rejected")
        row, raw = read_file(path, budget, keep=True)
        need(row == rows[path], "file_changed")
        metadata = SuppliedMetadata(raw).metadata
        name, version = metadata.get("Name"), metadata.get("Version")
        need(type(name) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name)
             and type(version) is str
             and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.!+_-]{0,127}", version),
             "distribution_rejected")
        name = re.sub(r"[-_.]+", "-", name).lower()
        need(name not in observed, "distribution_rejected")
        observed[name] = version
    return [{"name": name, "version": observed[name]} for name in sorted(observed)]


def produce():
    runtime_guard()
    budget = Budget()
    binding = read_input(budget)
    rows = files(budget)
    distributions = base_distributions(rows, budget)
    runtime_guard()
    report = {
        "schema": SCHEMA, "base_digest": binding["digest"],
        "config_digest": binding["config_digest"],
        "files": [rows[path] for path in sorted(rows)], "base_distributions": distributions,
    }
    raw = canonical(report)
    need(len(raw) <= MAXIMUM_OUTPUT_BYTES, "output_budget_exhausted")
    budget.check()
    return raw


def write_stdout(raw):
    offset = 0
    while offset < len(raw):
        written = os.write(1, raw[offset:offset + 65_536])
        need(0 < written <= min(65_536, len(raw) - offset), "output_failed")
        offset += written


def main():
    try:
        raw = produce()
    except BaseException as error:
        code = error.code if type(error) is InventoryError else "inventory_failed"
        try:
            write_stdout(canonical({"schema": SCHEMA, "status": "failed", "code": code}))
        except BaseException:
            pass
        return 1
    try:
        write_stdout(raw)
    except BaseException:
        # Do not append a second JSON document after an ambiguous partial write.
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
