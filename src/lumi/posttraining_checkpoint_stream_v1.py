"""Read-only, bounded verification of an explicitly supplied checkpoint inventory.

This kernel is not an authorization, importer, executor or origin-authentication
authority. It imports no ML and writes nothing. The caller must independently
authorize the exact local input and keep it quiescent. Metadata comparisons are
change detection, not an atomic snapshot or protection against a hostile owner.

The entire plan is validated and copied before filesystem access. maximum_files
counts regular files; directories are the bounded trie implied by their paths.
No other files or empty directories are accepted. Each file is hashed once in
bounded chunks, without an extra EOF read; exact sizes and unchanged metadata
are checked instead. A final metadata-only closed-tree walk catches changes to
earlier files. Root ancestry remains pinned until verification finishes.

Private mode requires the current effective UID/GID, directories 0700, and files
0400 or 0600 with one link. require_private=False relaxes only mode checks, not
ownership, regular-file, no-symlink or single-link checks. Ancestors above the
root are pinned no-symlink directories but need not be private/current-owned.
The monotonic deadline is cooperative; an external supervisor must bound blocked
syscalls. Only fixed error codes and aggregate reports may be exposed.
"""

from __future__ import annotations

import hashlib
import math
import os
import stat
import time
from pathlib import Path

SCHEMA = "lumi.checkpoint-stream-verification.v0.1"
MAXIMUM_FILES = 65536
MAXIMUM_DIRECTORIES = 65536
MAXIMUM_DEPTH = 64
MAXIMUM_CHUNK_BYTES = 1048576
MAXIMUM_READ_BYTES = 2**63 - 1
MAXIMUM_SECONDS = 86400
ERROR_CODES = frozenset({
    "invalid_plan", "bounds_exhausted", "deadline_exhausted", "path_rejected",
    "metadata_rejected", "inventory_mismatch", "identity_changed", "short_read",
    "hash_mismatch", "verification_failed", "close_failed",
})


class CheckpointStreamError(ValueError):
    def __init__(self, code="verification_failed"):
        self.code = code if type(code) is str and code in ERROR_CODES else "verification_failed"
        super().__init__(self.code)


def _need(value, code):
    if not value:
        raise CheckpointStreamError(code) from None


def _integer(value, low, high):
    return type(value) is int and low <= value <= high


def _parts(value):
    _need(type(value) is str and 0 < len(value) <= 4096
          and "\\" not in value and all(ord(char) >= 32 and ord(char) != 127 for char in value),
          "invalid_plan")
    encoded = value.encode("utf-8")
    _need(len(encoded) <= 4096, "invalid_plan")
    parts = tuple(value.split("/"))
    _need(len(parts) <= MAXIMUM_DEPTH and all(part not in {"", ".", ".."}
          and len(part.encode("utf-8")) <= 255 for part in parts), "invalid_plan")
    return parts


def _plan(root, inventory, maximum_read_bytes, maximum_files, maximum_seconds,
          chunk_bytes, require_private, started):
    _need(type(root) is type(Path()) and root.is_absolute() and str(root) != "/",
          "invalid_plan")
    root_parts = _parts(str(root)[1:])
    _need(_integer(maximum_read_bytes, 0, MAXIMUM_READ_BYTES)
          and _integer(maximum_files, 1, MAXIMUM_FILES)
          and _integer(chunk_bytes, 1, MAXIMUM_CHUNK_BYTES)
          and type(maximum_seconds) in {int, float}
          and 0 < maximum_seconds <= MAXIMUM_SECONDS and math.isfinite(maximum_seconds)
          and type(require_private) is bool, "invalid_plan")
    _need(type(inventory) is list and 0 < len(inventory) <= maximum_files, "invalid_plan")
    tree, total, directories = {}, 0, 1
    for row in inventory:
        _need(time.monotonic() < started + maximum_seconds, "deadline_exhausted")
        _need(type(row) is dict and len(row) == 3 and set(row) == {"path", "bytes", "sha256"}
              and all(type(key) is str for key in row), "invalid_plan")
        parts = _parts(row["path"])
        size, digest = row["bytes"], row["sha256"]
        _need(_integer(size, 0, maximum_read_bytes) and type(digest) is str
              and len(digest) == 64 and all(char in "0123456789abcdef" for char in digest),
              "invalid_plan")
        total += size
        _need(total <= maximum_read_bytes, "bounds_exhausted")
        node = tree
        for part in parts[:-1]:
            _need(time.monotonic() < started + maximum_seconds, "deadline_exhausted")
            if part not in node:
                directories += 1
                _need(directories <= MAXIMUM_DIRECTORIES, "bounds_exhausted")
                node[part] = {}
            _need(type(node[part]) is dict, "invalid_plan")
            node = node[part]
        _need(parts[-1] not in node, "invalid_plan")
        node[parts[-1]] = size, digest
    return root_parts, tree, total, directories


def _identity(info):
    return info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid


def _snapshot(info):
    # Reading/listing may change atime, which is not a content mutation.
    return (*_identity(info), info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


class _Verifier:
    def __init__(self, deadline, maximum_read_bytes, chunk_bytes, require_private):
        self.deadline, self.maximum = deadline, maximum_read_bytes
        self.chunk, self.private = chunk_bytes, require_private
        self.owner = os.geteuid(), os.getegid()
        self.owned, self.ancestry, self.snapshots = [], [], {}
        self.read_bytes = self.chunks = self.files = 0

    def check_time(self):
        _need(time.monotonic() < self.deadline, "deadline_exhausted")

    def close(self, descriptor):
        self.owned.remove(descriptor)  # An ambiguous close is never retried.
        try:
            os.close(descriptor)
        except BaseException:
            raise CheckpointStreamError("close_failed") from None

    def open(self, name, *, directory, parent=None):
        self.check_time()
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
        if directory:
            flags |= os.O_DIRECTORY
        descriptor = os.open(name, flags, dir_fd=parent)
        self.owned.append(descriptor)
        return descriptor

    def metadata(self, info, *, directory):
        _need((stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))
              and (info.st_uid, info.st_gid) == self.owner and info.st_ino > 0,
              "metadata_rejected")
        if not directory:
            _need(info.st_nlink == 1, "metadata_rejected")
        if self.private:
            modes = {0o700} if directory else {0o400, 0o600}
            _need(stat.S_IMODE(info.st_mode) in modes, "metadata_rejected")

    def root(self, parts):
        descriptor = self.open("/", directory=True)
        info = os.fstat(descriptor)
        _need(stat.S_ISDIR(info.st_mode), "path_rejected")
        self.ancestry.append((descriptor, None, None, _identity(info)))
        for part in parts:
            self.check_time()
            before = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
            _need(stat.S_ISDIR(before.st_mode), "path_rejected")
            child = self.open(part, directory=True, parent=descriptor)
            named = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
            current = os.fstat(child)
            _need(_identity(before) == _identity(current) == _identity(named), "identity_changed")
            self.ancestry.append((child, descriptor, part, _identity(current)))
            descriptor = child
        self.metadata(os.fstat(descriptor), directory=True)
        return descriptor

    def check_ancestry(self):
        for descriptor, parent, name, expected in self.ancestry:
            self.check_time()
            _need(_identity(os.fstat(descriptor)) == expected, "identity_changed")
            if parent is not None:
                _need(_identity(os.stat(name, dir_fd=parent, follow_symlinks=False)) == expected,
                      "identity_changed")

    def file(self, parent, name, leaf, trail):
        self.check_time()
        before = os.stat(name, dir_fd=parent, follow_symlinks=False)
        self.metadata(before, directory=False)
        size, expected = leaf
        _need(before.st_size == size, "inventory_mismatch")
        descriptor = self.open(name, directory=False, parent=parent)
        _need(_snapshot(os.fstat(descriptor)) == _snapshot(before), "identity_changed")
        hasher, remaining = hashlib.sha256(), size
        while remaining:
            self.check_time()
            request = min(remaining, self.chunk)
            _need(self.read_bytes + request <= self.maximum, "bounds_exhausted")
            data = os.read(descriptor, request)
            _need(type(data) is bytes and 0 < len(data) <= request, "short_read")
            self.read_bytes += len(data)
            self.chunks += 1
            remaining -= len(data)
            hasher.update(data)
        self.check_time()
        _need(_snapshot(before) == _snapshot(os.fstat(descriptor))
              == _snapshot(os.stat(name, dir_fd=parent, follow_symlinks=False)), "identity_changed")
        _need(hasher.hexdigest() == expected, "hash_mismatch")
        self.snapshots[trail] = _snapshot(before)
        self.files += 1
        self.close(descriptor)

    def walk(self, descriptor, tree, trail=(), *, final=False):
        self.check_time()
        before = os.fstat(descriptor)
        self.metadata(before, directory=True)
        if final:
            _need(_snapshot(before) == self.snapshots[trail], "identity_changed")
        else:
            self.snapshots[trail] = _snapshot(before)
        found = set()
        # scandir streams entries; refuse the first extra without traversing it.
        with os.scandir(descriptor) as entries:
            for entry in entries:
                self.check_time()
                name = entry.name
                _need(name in tree and name not in found, "inventory_mismatch")
                found.add(name)
                node, child_trail = tree[name], (*trail, name)
                if type(node) is dict:
                    named = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                    self.metadata(named, directory=True)
                    child = self.open(name, directory=True, parent=descriptor)
                    _need(_snapshot(named) == _snapshot(os.fstat(child)), "identity_changed")
                    self.walk(child, node, child_trail, final=final)
                    _need(_snapshot(named) == _snapshot(os.fstat(child))
                          == _snapshot(os.stat(name, dir_fd=descriptor, follow_symlinks=False)),
                          "identity_changed")
                    self.close(child)
                elif final:
                    current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                    self.metadata(current, directory=False)
                    _need(_snapshot(current) == self.snapshots[child_trail], "identity_changed")
                else:
                    self.file(descriptor, name, node, child_trail)
        _need(found == set(tree), "inventory_mismatch")
        self.check_time()
        _need(_snapshot(os.fstat(descriptor)) == _snapshot(before), "identity_changed")


def verify_inventory(root: Path, inventory: list[dict], *, maximum_read_bytes: int,
                     maximum_files: int, maximum_seconds: float, chunk_bytes: int = 1048576,
                     require_private: bool = True) -> dict:
    """Verify one exact closed tree, returning no file names or lexical content."""
    failure, verifier, result = None, None, None
    try:
        started = time.monotonic()
        root_parts, tree, total, directories = _plan(
            root, inventory, maximum_read_bytes, maximum_files, maximum_seconds,
            chunk_bytes, require_private, started)
        verifier = _Verifier(started + maximum_seconds, maximum_read_bytes,
                             chunk_bytes, require_private)
        descriptor = verifier.root(root_parts)
        verifier.walk(descriptor, tree)
        verifier.check_ancestry()
        verifier.walk(descriptor, tree, final=True)
        verifier.check_ancestry()
        _need(verifier.files == len(inventory) and verifier.read_bytes == total,
              "inventory_mismatch")
        result = {
            "schema": SCHEMA, "status": "verified", "files_verified": verifier.files,
            "directories_verified": directories, "bytes_hashed": verifier.read_bytes,
            "read_chunks": verifier.chunks, "inventory_closed": True,
            "metadata_unchanged": True, "private_modes_required": require_private,
            "filesystem_mutated": False, "authorization_verified": False,
            "source_authenticity_verified": False, "execution_permitted": False,
        }
    except BaseException as error:
        failure = error.code if type(error) is CheckpointStreamError else "verification_failed"
    finally:
        if verifier is not None:
            while verifier.owned:
                descriptor = verifier.owned[-1]
                try:
                    verifier.close(descriptor)
                except BaseException:
                    failure = "close_failed"
    # A final stat or descriptor close can cross the cooperative deadline.
    # Cleanup still completes, but that attempt must not publish success.
    if failure is None and verifier is not None:
        try:
            verifier.check_time()
        except BaseException as error:
            failure = error.code if type(error) is CheckpointStreamError else "verification_failed"
    if failure is not None:
        raise CheckpointStreamError(failure) from None
    return result
