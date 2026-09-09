"""Bounded pre-ML release-tree content verification, never a launch capability.

The expected manifest digest and owner IDs must arrive from a trusted control
plane. The host, kernel, administrator and release owner are trusted. A scan
checks each object's stability while observed, not a point-in-time tree snapshot;
it cannot freeze files or authenticate a release's origin/completeness.
Only an explicitly supplied release root is read; nothing is imported from it.
"""

from __future__ import annotations

import hashlib
import os
import platform
import stat
import sys
import threading
import time
from pathlib import PurePosixPath

from . import posttraining_sft_release_manifest_v1 as manifest

SCHEMA = "lumi.sft-release-verification.v0.1"
MAXIMUM_TREE_ENTRIES = 65_536
MAXIMUM_OPENS = 131_072
MAXIMUM_RUNTIME_MS = 1_800_000
LIMIT_FIELDS = frozenset({
    "maximum_manifest_bytes", "maximum_entries", "maximum_file_bytes", "maximum_total_bytes",
    "maximum_tree_entries", "maximum_content_read_bytes", "maximum_opens", "maximum_runtime_ms",
})
ERROR_CODES = manifest.ERROR_CODES | frozenset({
    "configuration_rejected", "runtime_rejected", "root_rejected", "tree_rejected",
    "owner_rejected", "mode_rejected", "file_rejected", "file_changed", "digest_mismatch",
    "interpreter_rejected", "read_budget_exhausted", "entry_budget_exhausted",
    "open_budget_exhausted", "deadline_exhausted", "io_failed", "cleanup_failed",
})
NONCLAIMS = (
    "runtime_origin_authenticated", "dynamic_dependency_completeness_verified",
    "point_in_time_snapshot_verified", "immutable_during_execution",
    "host_binding_verified", "execution_permitted",
    "real_sft_execution_permitted", "production_eligible",
)


class ReleaseVerificationError(ValueError):
    def __init__(self, code="io_failed"):
        self.code = code if type(code) is str and code in ERROR_CODES else "io_failed"
        super().__init__(self.code)


def need(value, code):
    if not value:
        raise ReleaseVerificationError(code) from None


def _runtime():
    need(sys.platform == "linux" and platform.machine() == "x86_64"
         and sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 12)
         and sys.flags.no_site == 1 and sys.dont_write_bytecode
         and threading.current_thread() is threading.main_thread()
         and threading.active_count() == 1
         and not any(name.split(".", 1)[0] in
                     {"torch", "numpy", "transformers", "tensorflow", "jax", "cuda", "cupy"}
                     for name in sys.modules), "runtime_rejected")


def _signature(info):
    return tuple(getattr(info, name) for name in (
        "st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink", "st_size",
        "st_mtime_ns", "st_ctime_ns",
    ))


def _configuration(root, limits, uid, gid):
    need(type(root) is str and 0 < len(root) <= 4096 and root.startswith("/")
         and not root.startswith("//"),
         "root_rejected")
    parsed = PurePosixPath(root)
    need(str(parsed) == root and 2 <= len(parsed.parts) <= 32 and ".." not in parsed.parts
         and all(len(part) <= 255 and all(ord(char) >= 32 and ord(char) != 127 for char in part)
                 for part in parsed.parts) and "\\" not in root, "root_rejected")
    need(type(uid) is int and type(gid) is int and 0 <= uid < 2**31 and 0 <= gid < 2**31,
         "configuration_rejected")
    need(type(limits) is dict and set(limits) == LIMIT_FIELDS, "configuration_rejected")
    ceilings = {
        "maximum_manifest_bytes": manifest.MAXIMUM_MANIFEST_BYTES,
        "maximum_entries": manifest.MAXIMUM_ENTRIES,
        "maximum_file_bytes": manifest.MAXIMUM_FILE_BYTES,
        "maximum_total_bytes": manifest.MAXIMUM_TOTAL_BYTES,
        "maximum_tree_entries": MAXIMUM_TREE_ENTRIES,
        "maximum_content_read_bytes": manifest.MAXIMUM_TOTAL_BYTES + manifest.MAXIMUM_ENTRIES,
        "maximum_opens": MAXIMUM_OPENS, "maximum_runtime_ms": MAXIMUM_RUNTIME_MS,
    }
    need(all(type(limits[key]) is int and 1 <= limits[key] <= ceiling
             for key, ceiling in ceilings.items()), "configuration_rejected")
    return dict(limits), parsed.parts[1:]


class Reader:
    def __init__(self, limits, uid, gid):
        self.limits, self.uid, self.gid = limits, uid, gid
        self.deadline = time.monotonic_ns() + limits["maximum_runtime_ms"] * 1_000_000
        self.handles = set()
        self.open_attempts = self.opened = self.closed = 0
        self.reserved = self.read_bytes = self.entries = self.files = self.directories = 0
        self.cleanup_failed = False

    def check(self):
        need(time.monotonic_ns() < self.deadline, "deadline_exhausted")

    def reserve_open(self):
        self.check()
        need(self.open_attempts < self.limits["maximum_opens"], "open_budget_exhausted")
        self.open_attempts += 1

    def open(self, name, *, parent=None, directory=False):
        self.reserve_open()
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
        if directory:
            flags |= os.O_DIRECTORY
        descriptor = os.open(name, flags, dir_fd=parent)
        self.opened += 1
        try:
            self.handles.add(descriptor)
        except BaseException:
            try:
                os.close(descriptor)
                self.closed += 1
            except BaseException:
                self.cleanup_failed = True
            raise
        return descriptor

    def close(self, descriptor):
        self.handles.remove(descriptor)  # Never retry an ambiguous close.
        try:
            os.close(descriptor)
            self.closed += 1
        except BaseException:
            self.cleanup_failed = True
            raise ReleaseVerificationError("cleanup_failed") from None

    def finish(self):
        for descriptor in tuple(self.handles):
            try:
                self.close(descriptor)
            except BaseException:
                self.cleanup_failed = True

    def root(self, parts):
        current = self.open("/", directory=True)
        for component in parts:
            child = self.open(component, parent=current, directory=True)
            self.close(current)
            current = child
        return current

    def owner(self, info):
        need(info.st_uid == self.uid and info.st_gid == self.gid, "owner_rejected")

    def directory(self, descriptor):
        info = os.fstat(descriptor)
        need(stat.S_ISDIR(info.st_mode), "tree_rejected")
        self.owner(info)
        need(stat.S_IMODE(info.st_mode) in {0o500, 0o555, 0o700, 0o755}, "mode_rejected")
        return info

    def names(self, descriptor, expected):
        self.reserve_open()  # scandir(fd) owns a duplicate descriptor on supported CPython.
        iterator = os.scandir(descriptor)
        self.opened += 1
        seen = set()
        try:
            for entry in iterator:
                self.check()
                # One overflow directory entry may be observed before failing closed.
                need(self.entries < self.limits["maximum_tree_entries"], "entry_budget_exhausted")
                self.entries += 1
                need(entry.name in expected and entry.name not in seen, "tree_rejected")
                seen.add(entry.name)
            need(seen == expected, "tree_rejected")
        finally:
            try:
                iterator.close()
                self.closed += 1
            except BaseException:
                self.cleanup_failed = True
                raise ReleaseVerificationError("cleanup_failed") from None

    def read(self, descriptor, capacity):
        self.check()
        need(self.reserved + capacity <= self.limits["maximum_content_read_bytes"],
             "read_budget_exhausted")
        self.reserved += capacity  # Prepay capacity even on short/failed reads.
        result = os.read(descriptor, capacity)
        self.read_bytes += len(result)
        self.check()
        return result

    def file(self, descriptor, row, *, interpreter):
        before = os.fstat(descriptor)
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
             and before.st_size == row["bytes"], "file_rejected")
        self.owner(before)
        need(stat.S_IMODE(before.st_mode) == row["mode"], "mode_rejected")
        digest, remaining, first = hashlib.sha256(), row["bytes"], True
        while remaining:
            raw = self.read(descriptor, min(65_536, remaining))
            need(0 < len(raw) <= remaining, "file_changed")
            if first and interpreter:
                need(raw.startswith(b"\x7fELF"), "interpreter_rejected")
            first = False
            remaining -= len(raw)
            digest.update(raw)
        need(not interpreter or not first, "interpreter_rejected")
        need(self.read(descriptor, 1) == b"", "file_changed")
        need(_signature(os.fstat(descriptor)) == _signature(before), "file_changed")
        need(digest.hexdigest() == row["sha256"], "digest_mismatch")
        self.files += 1
        return before

    def walk(self, descriptor, prefix, children, rows, interpreter):
        self.check()
        before = self.directory(descriptor)
        self.names(descriptor, set(children[prefix]))
        self.directories += 1
        for name in sorted(children[prefix]):
            relative = name if not prefix else prefix + "/" + name
            is_directory = relative not in rows
            child = self.open(name, parent=descriptor, directory=is_directory)
            info = os.fstat(child)
            try:
                if is_directory:
                    self.walk(child, relative, children, rows, interpreter)
                else:
                    info = self.file(child, rows[relative], interpreter=relative == interpreter)
                named = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                need(_signature(os.fstat(child)) == _signature(info) == _signature(named),
                     "file_changed")
            finally:
                self.close(child)
        need(_signature(os.fstat(descriptor)) == _signature(before), "file_changed")


def verify_release(root, raw_manifest, expected_sha256, *, limits, expected_uid, expected_gid):
    """Verify an exact tree once; no retry, imports, execution or output-file writes.

    All limits and the expected manifest pin are explicit trusted inputs. Deadline
    checks cannot interrupt parsing or a blocked filesystem syscall; only a trusted
    local release filesystem is supported. An independent supervisor owns hard deadlines.
    """
    reader, failure, result = None, None, None
    try:
        _runtime()
        selected, parts = _configuration(root, limits, expected_uid, expected_gid)
        reader = Reader(selected, expected_uid, expected_gid)
        value = manifest.validate_manifest(raw_manifest, expected_sha256, **{
            key: selected[key] for key in ("maximum_manifest_bytes", "maximum_entries",
                                          "maximum_file_bytes", "maximum_total_bytes")})
        rows = {row["path"]: row for row in value["files"]}
        children = {"": set()}
        for path in rows:
            components = path.split("/")
            for index, component in enumerate(components):
                parent = "/".join(components[:index])
                children.setdefault(parent, set()).add(component)
        tree_entries = len(rows) + len(children) - 1
        need(tree_entries <= selected["maximum_tree_entries"], "entry_budget_exhausted")
        need(sum(row["bytes"] + 1 for row in rows.values())
             <= selected["maximum_content_read_bytes"], "read_budget_exhausted")
        reader.check()
        root_fd = reader.root(parts)
        before = reader.directory(root_fd)
        reader.walk(root_fd, "", children, rows, value["interpreter"])
        recheck = reader.root(parts)
        need(_signature(os.fstat(root_fd)) == _signature(before) == _signature(os.fstat(recheck)),
             "file_changed")
        reader.close(recheck)
        reader.close(root_fd)
        reader.check()
        result = {
            "schema": SCHEMA, "status": "verified", "manifest_sha256": expected_sha256,
            "files_verified": reader.files, "directories_verified": reader.directories,
            "content_read_bytes": reader.read_bytes, "reserved_read_bytes": reader.reserved,
            "tree_entries_examined": reader.entries, "open_attempts": reader.open_attempts,
            "handles_opened": reader.opened, "handles_closed": reader.closed,
            "exact_tree_matched": True, "root_identity_stable": True,
            **dict.fromkeys(NONCLAIMS, False),
        }
    except BaseException as error:
        failure = (error.code if type(error) in (ReleaseVerificationError, manifest.ManifestError)
                   else "io_failed")
    finally:
        if reader is not None:
            reader.finish()
            if reader.cleanup_failed or reader.opened != reader.closed:
                failure = "cleanup_failed"
    if failure is not None:
        raise ReleaseVerificationError(failure) from None
    return result
