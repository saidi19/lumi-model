"""Strict requested-byte checkpoint audit over the frozen inventory verifier.

Every bounded content request is charged before os.read. Any short read, EOF or
read exception is terminal: no remainder read, refund or retry occurs. Frozen
tree, path, checksum, owner/mode, ancestry and final-walk checks are preserved.
No old verifier globals or code are modified. Only fixed error arguments are
safe to report; suppressed exception contexts must not be logged.

Counts are logical requested content bytes, not physical device I/O or OS
limits. A successful audit is neither origin proof nor execution authority.
The shared callback is cooperative; an independent hard supervisor is required.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from pathlib import Path

from . import posttraining_checkpoint_stream_v1 as stream

SCHEMA = "lumi.checkpoint-strict-audit.v0.1"
ERROR_CODES = stream.ERROR_CODES | frozenset({
    "budget_check_failed", "owner_mismatch", "reentry", "read_failed",
})


class CheckpointAuditError(ValueError):
    def __init__(self, code="verification_failed"):
        self.code = code if type(code) is str and code in ERROR_CODES else "verification_failed"
        super().__init__(self.code)


class _StrictVerifier(stream._Verifier):
    def __init__(self, deadline, maximum_read_bytes, chunk_bytes, require_private, check_budget):
        super().__init__(deadline, maximum_read_bytes, chunk_bytes, require_private)
        self._callback = check_budget
        self._process_owner = os.getpid(), threading.get_ident()
        self._failed = self._in_callback = False

    def _fail(self, code):
        self._failed = True
        raise CheckpointAuditError(code) from None

    def check_time(self):
        if self._process_owner != (os.getpid(), threading.get_ident()):
            self._fail("owner_mismatch")
        if self._in_callback:
            self._fail("reentry")
        if self._failed:
            self._fail("verification_failed")
        super().check_time()
        self._in_callback = True
        try:
            if self._callback() is not None:
                self._fail("budget_check_failed")
            if self._failed:
                self._fail("verification_failed")
        except BaseException:
            self._fail("budget_check_failed")
        finally:
            self._in_callback = False
        super().check_time()

    def file(self, parent, name, leaf, trail):
        self.check_time()
        before = os.stat(name, dir_fd=parent, follow_symlinks=False)
        self.metadata(before, directory=False)
        size, expected = leaf
        stream._need(before.st_size == size, "inventory_mismatch")
        descriptor = self.open(name, directory=False, parent=parent)
        stream._need(stream._snapshot(os.fstat(descriptor)) == stream._snapshot(before),
                     "identity_changed")
        hasher, remaining = hashlib.sha256(), size
        while remaining:
            self.check_time()
            request = min(remaining, self.chunk)
            stream._need(self.read_bytes + request <= self.maximum, "bounds_exhausted")
            self.read_bytes += request  # Debit first; never refund failed/short reads.
            self.chunks += 1
            try:
                data = os.read(descriptor, request)
            except BaseException:
                self._fail("read_failed")
            if type(data) is not bytes or len(data) != request:
                self._fail("short_read")
            hasher.update(data)
            remaining -= request
        self.check_time()
        stream._need(stream._snapshot(before) == stream._snapshot(os.fstat(descriptor))
                     == stream._snapshot(os.stat(name, dir_fd=parent, follow_symlinks=False)),
                     "identity_changed")
        stream._need(hasher.hexdigest() == expected, "hash_mismatch")
        self.snapshots[trail] = stream._snapshot(before)
        self.files += 1
        self.close(descriptor)


def verify_inventory(root: Path, inventory: list[dict], *, maximum_read_bytes: int,
                     maximum_files: int, maximum_seconds: float, check_budget,
                     chunk_bytes: int = 1048576, require_private: bool = True) -> dict:
    """One strict full audit; success requires cleanup and the final shared tick."""
    if not callable(check_budget):
        raise CheckpointAuditError("invalid_plan") from None
    verifier = result = failure = None
    started = time.monotonic()
    try:
        root_parts, tree, total, directories = stream._plan(
            root, inventory, maximum_read_bytes, maximum_files, maximum_seconds,
            chunk_bytes, require_private, started,
        )
        expected_files = len(inventory)
        verifier = _StrictVerifier(started + maximum_seconds, maximum_read_bytes,
                                   chunk_bytes, require_private, check_budget)
        descriptor = verifier.root(root_parts)
        verifier.walk(descriptor, tree)
        verifier.check_ancestry()
        verifier.walk(descriptor, tree, final=True)
        verifier.check_ancestry()
        stream._need(verifier.files == expected_files and verifier.read_bytes == total,
                     "inventory_mismatch")
        result = {
            "schema": SCHEMA, "status": "verified", "files_verified": verifier.files,
            "directories_verified": directories, "bytes_hashed": verifier.read_bytes,
            "read_chunks": verifier.chunks, "inventory_closed": True,
            "metadata_unchanged": True, "private_modes_required": require_private,
            "logical_read_bytes_requested": verifier.read_bytes,
            "short_reads_are_terminal": True, "reservations_refunded": False,
            "physical_device_bytes_measured": False, "os_resource_limits_enforced": False,
            "filesystem_mutated": False, "authorization_verified": False,
            "source_authenticity_verified": False, "execution_permitted": False,
        }
    except BaseException as error:
        failure = error.code if type(error) in {
            CheckpointAuditError, stream.CheckpointStreamError
        } else "verification_failed"
    finally:
        if verifier is not None:
            while verifier.owned:
                try:
                    verifier.close(verifier.owned[-1])
                except BaseException:
                    failure = "close_failed"
    if failure is None and verifier is not None:
        try:
            verifier.check_time()
        except BaseException as error:
            failure = error.code if type(error) in {
                CheckpointAuditError, stream.CheckpointStreamError
            } else "verification_failed"
    if failure is not None:
        raise CheckpointAuditError(failure) from None
    return result
