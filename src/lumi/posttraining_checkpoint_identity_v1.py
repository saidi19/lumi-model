"""Private, metadata-only closed-tree observations for a bound checkpoint plan.

Reuse the frozen streamed verifier's path, directory, ancestry and final-walk
rules, replacing only content hashing with size/identity observations. Source
files must already be private 0600 (DCP contract), directories 0700. Nothing is
read, hashed, copied, written or chmodded. The returned mapping contains private
names/identities and must stay in memory; do not serialize or log it.

Equality detects observed mutations; it is not a checksum audit, atomic lock,
origin proof, execution permission or OS resource enforcement. A caller must
provide its shared cooperative budget and an independent hard supervisor.
Only fixed error arguments are safe to report, never suppressed exception text.
"""

from __future__ import annotations

import os
import stat
import threading
import time
from collections.abc import Mapping
from types import MappingProxyType

from . import posttraining_checkpoint_stream_v1 as stream

MAXIMUM_FILES = 129  # The frozen import plan's closed handoff inventory cap.
ERROR_CODES = stream.ERROR_CODES | frozenset({
    "invalid_arguments", "budget_check_failed", "owner_mismatch", "reentry",
})


class CheckpointIdentityError(ValueError):
    def __init__(self, code="verification_failed"):
        self.code = code if type(code) is str and code in ERROR_CODES else "verification_failed"
        super().__init__(self.code)


class _PrivateSnapshot(Mapping):
    def __init__(self, values):
        self._values = MappingProxyType(dict(values))

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __repr__(self):
        return "<CheckpointIdentitySnapshot private>"


class _Observer(stream._Verifier):
    def __init__(self, *, check_budget, started):
        super().__init__(started + stream.MAXIMUM_SECONDS, 0, 1, True)
        self._callback = check_budget
        self._process_owner = os.getpid(), threading.get_ident(), os.geteuid(), os.getegid()
        self._in_callback = self._failed = False

    def _fail(self, code):
        self._failed = True
        raise CheckpointIdentityError(code) from None

    def check_time(self):
        if self._process_owner != (
                os.getpid(), threading.get_ident(), os.geteuid(), os.getegid()):
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

    def metadata(self, info, *, directory):
        super().metadata(info, directory=directory)
        if not directory and stat.S_IMODE(info.st_mode) != 0o600:
            self._fail("metadata_rejected")

    def file(self, parent, name, leaf, trail):
        self.check_time()
        before = os.stat(name, dir_fd=parent, follow_symlinks=False)
        self.metadata(before, directory=False)
        size, _expected_digest = leaf
        stream._need(before.st_size == size, "inventory_mismatch")
        descriptor = self.open(name, directory=False, parent=parent)
        expected = stream._snapshot(before)
        stream._need(expected == stream._snapshot(os.fstat(descriptor)), "identity_changed")
        self.check_time()
        stream._need(expected == stream._snapshot(os.fstat(descriptor))
                     == stream._snapshot(os.stat(name, dir_fd=parent, follow_symlinks=False)),
                     "identity_changed")
        self.snapshots[trail] = expected
        self.files += 1
        self.close(descriptor)


def snapshot_inventory(root, inventory, *, check_budget):
    """Return a private immutable mapping after cleanup and a final budget check.

    Exactly the directories implied by the inventory are permitted; extra empty
    directories are rejected too. Digest syntax is validated as part of the
    unchanged plan parser, but no content digest is computed by this function.
    """
    if (not callable(check_budget) or type(inventory) is not list
            or not 0 < len(inventory) <= MAXIMUM_FILES):
        raise CheckpointIdentityError("invalid_arguments") from None
    observer = result = failure = None
    started = time.monotonic()
    try:
        root_parts, tree, _total, directories = stream._plan(
            root, inventory, stream.MAXIMUM_READ_BYTES, MAXIMUM_FILES, stream.MAXIMUM_SECONDS,
            1, True, started,
        )
        expected_files = len(inventory)
        observer = _Observer(check_budget=check_budget, started=started)
        descriptor = observer.root(root_parts)
        observer.walk(descriptor, tree)
        observer.check_ancestry()
        observer.walk(descriptor, tree, final=True)
        observer.check_ancestry()
        stream._need(observer.files == expected_files
                     and len(observer.snapshots) == expected_files + directories
                     and observer.read_bytes == observer.chunks == 0,
                     "inventory_mismatch")
        result = _PrivateSnapshot(observer.snapshots)
    except BaseException as error:
        failure = error.code if type(error) in {
            CheckpointIdentityError, stream.CheckpointStreamError
        } else "verification_failed"
    finally:
        if observer is not None:
            while observer.owned:
                try:
                    observer.close(observer.owned[-1])
                except BaseException:
                    failure = "close_failed"
    if failure is None and observer is not None:
        try:
            observer.check_time()  # A late close/deadline never publishes success.
        except BaseException as error:
            failure = error.code if type(error) in {
                CheckpointIdentityError, stream.CheckpointStreamError
            } else "verification_failed"
    if failure is not None:
        raise CheckpointIdentityError(failure) from None
    return result
