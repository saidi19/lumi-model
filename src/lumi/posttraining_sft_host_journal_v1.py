"""Exclusive bounded POSIX journal sink for fixed lifecycle records.

The caller supplies an already-open, trusted, stable, local, owned 0700
directory. Its namespace must not be concurrently modified by another actor.
Descriptor/path identity checks detect replacements; they do not make an
untrusted namespace safe. File and directory fsync are requested, not proof
of encryption, a particular local filesystem, or hard-power-loss durability.

This sink checks record framing, not semantic privacy or schema. Its caller
must supply only allowlisted records. An exclusive claim is never deleted,
truncated, reopened or resumed, including after partial write or crash.
"""

from __future__ import annotations

import fcntl
import os
import re
import stat
import threading

MAXIMUM_RECORD_BYTES = 4096
MAXIMUM_RECORDS = 256
MAXIMUM_CAPACITY_BYTES = 1_048_576
ERROR_CODES = frozenset({"journal_rejected"})
_ATTEMPT = re.compile(r"[0-9a-f]{64}")


class JournalError(ValueError):
    """A fixed error code without record, identifier, path or exception text."""

    def __init__(self, _code=None):
        self.code = "journal_rejected"
        super().__init__(self.code)


def _need(condition):
    if not condition:
        raise JournalError from None


class Journal:
    """One process/thread owner with an immutable exclusive attempt claim.

    The input directory descriptor is borrowed. Only its CLOEXEC duplicate and
    the newly claimed file descriptor are owned here. Forked copies may close
    their descriptors, but may never append or fsync records in the parent.
    """

    def __init__(self, root_fd: int, attempt_id: str):
        self._owner_pid = os.getpid()
        self._owner_thread = threading.current_thread()
        self._owner_uid = os.getuid()
        self._root_fd = self._file_fd = None
        self._root_identity = self._file_identity = None
        self._root_close_attempted = self._file_close_attempted = False
        self._root_closed = self._file_closed = False
        self._claimed = self._failed = False
        self._busy = False
        self._records_reserved = self._records_synced = 0
        self._capacity_reserved_bytes = self._write_calls = 0
        self._fsync_calls = self._close_calls = self._file_bytes = 0
        self._filename = None
        try:
            self._require_owner()
            _need(os.name == "posix" and type(root_fd) is int and root_fd >= 0
                  and type(attempt_id) is str and _ATTEMPT.fullmatch(attempt_id) is not None
                  and hasattr(fcntl, "F_DUPFD_CLOEXEC") and hasattr(os, "O_CLOEXEC")
                  and hasattr(os, "O_NOFOLLOW"))
            original = os.fstat(root_fd)
            self._check_directory(original)
            self._root_identity = (original.st_dev, original.st_ino)
            duplicate = fcntl.fcntl(root_fd, fcntl.F_DUPFD_CLOEXEC, 0)
            _need(type(duplicate) is int and duplicate >= 0 and duplicate != root_fd)
            self._root_fd = duplicate
            self._verify_root()
            self._filename = attempt_id + ".lifecycle.jsonl"
            self._file_fd = os.open(
                self._filename,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600, dir_fd=self._root_fd,
            )
            self._claimed = True
            created = os.fstat(self._file_fd)
            self._file_identity = (created.st_dev, created.st_ino)
            self._verify_identity()
            self._sync(self._file_fd)
            self._sync(self._root_fd)
            self._verify_identity()
        except BaseException:
            self._failed = True
            self.close()
            raise JournalError from None

    def _require_owner(self):
        _need(os.getpid() == self._owner_pid
              and threading.current_thread() is self._owner_thread)

    def _check_directory(self, observed):
        _need(stat.S_ISDIR(observed.st_mode) and observed.st_uid == self._owner_uid
              and stat.S_IMODE(observed.st_mode) == 0o700)

    def _verify_root(self):
        _need(self._root_fd is not None and not self._root_close_attempted)
        observed = os.fstat(self._root_fd)
        self._check_directory(observed)
        _need((observed.st_dev, observed.st_ino) == self._root_identity
              and not os.get_inheritable(self._root_fd))

    def _check_file(self, observed):
        _need(stat.S_ISREG(observed.st_mode) and observed.st_uid == self._owner_uid
              and stat.S_IMODE(observed.st_mode) == 0o600 and observed.st_nlink == 1
              and observed.st_size == self._file_bytes
              and (observed.st_dev, observed.st_ino) == self._file_identity)

    def _verify_identity(self):
        self._verify_root()
        _need(self._file_fd is not None and not self._file_close_attempted
              and not os.get_inheritable(self._file_fd))
        self._check_file(os.fstat(self._file_fd))
        self._check_file(os.stat(self._filename, dir_fd=self._root_fd, follow_symlinks=False))

    def _sync(self, descriptor):
        self._require_owner()
        _need(not self._failed)
        self._fsync_calls += 1
        os.fsync(descriptor)

    def append(self, record: bytes) -> None:
        """Precharge one capacity slot, issue one write, and fsync before acceptance."""
        entered = False
        try:
            self._require_owner()
            _need(not self._busy and not self._failed and self._claimed
                  and not self._root_close_attempted
                  and not self._file_close_attempted)
            self._busy = entered = True
            _need(type(record) is bytes and 0 < len(record) <= MAXIMUM_RECORD_BYTES
                  and record.endswith(b"\n") and record.count(b"\n") == 1)
            _need(self._records_reserved < MAXIMUM_RECORDS
                  and self._capacity_reserved_bytes + MAXIMUM_RECORD_BYTES
                  <= MAXIMUM_CAPACITY_BYTES)
            self._records_reserved += 1
            self._capacity_reserved_bytes += MAXIMUM_RECORD_BYTES
            self._verify_identity()
            self._require_owner()
            _need(not self._failed)
            self._write_calls += 1
            written = os.write(self._file_fd, record)
            _need(type(written) is int and written == len(record))
            self._file_bytes += written
            _need(not self._failed)
            self._sync(self._file_fd)
            self._verify_identity()
            _need(not self._failed)
            self._records_synced += 1
        except BaseException:
            self._failed = True
            raise JournalError from None
        finally:
            if entered:
                self._busy = False

    def close(self) -> bool:
        """Close each owned descriptor once; result describes cleanup, not record success.

        A foreign fork may release its copies without touching journal contents.
        A failed close is never retried, even if its descriptor number is reused.
        Wrong-thread or reentrant closes cannot clobber a live owner operation.
        """
        try:
            if os.getpid() == self._owner_pid:
                self._require_owner()
                _need(not self._busy)
            else:
                self._failed = True
        except BaseException:
            self._failed = True
            return False
        self._busy = True
        try:
            for name in ("file", "root"):
                descriptor = getattr(self, f"_{name}_fd")
                if descriptor is None or getattr(self, f"_{name}_close_attempted"):
                    continue
                setattr(self, f"_{name}_close_attempted", True)
                self._close_calls += 1
                try:
                    os.close(descriptor)
                except BaseException:
                    self._failed = True
                else:
                    setattr(self, f"_{name}_closed", True)
        finally:
            self._busy = False
        return all(getattr(self, f"_{name}_fd") is None or getattr(self, f"_{name}_closed")
                   for name in ("file", "root"))

    def summary(self) -> dict[str, bool | int]:
        """Only fixed aggregate accounting, without paths, attempt IDs or records."""
        return {
            "claimed": self._claimed, "failed": self._failed,
            "closed": all(getattr(self, f"_{name}_fd") is None or getattr(self, f"_{name}_closed")
                          for name in ("file", "root")),
            "records_reserved": self._records_reserved, "records_synced": self._records_synced,
            "capacity_reserved_bytes": self._capacity_reserved_bytes,
            "write_calls": self._write_calls, "fsync_calls": self._fsync_calls,
            "close_calls": self._close_calls,
        }
