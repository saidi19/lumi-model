"""Private, create-once metadata journal for a trusted local SFT controller.

This module never launches a process, imports ML, authorizes SFT, or resumes an
attempt. A journal that already exists is permanently consumed, including an
empty, truncated or failed journal. ``verify_bytes`` is an audit-only parser,
not a way to reopen a journal or restore execution rights.

The caller owns an existing private directory and supplies a trusted aggregate
``debit(number_of_bytes)`` callback. Every buffer offered to ``os.write`` is
precharged in full. Short writes therefore conservatively charge the remaining
buffer again; no failure refunds any charge. The configured cap bounds these
attempted bytes, including the automatically written initial record. A callback
failure is also counted conservatively and permits no write.

The hash chain detects malformed, changed and incomplete frames. It does not
authenticate an external owner, prevent same-user tampering with an entire
chain, or distinguish removal of complete trailing records from an incomplete
attempt. An external expected final digest is needed for that last distinction.
The parent must independently enforce worker deadlines and no-retry semantics;
blocking filesystem operations are not an OS execution or memory sandbox.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading

FILENAME = "attempt.journal"
RECORD_SCHEMA = "lumi.sft-attempt-journal-record.v0.1"
MAXIMUM_BYTES = 1_048_576
MAXIMUM_FRAME_BYTES = 4096
MAXIMUM_COUNTER = 2**63 - 1
ZERO_DIGEST = "0" * 64
EVENTS = frozenset(
    (
        "attempt_created",
        "epoch_started",
        "resource_reserved",
        "step_reserved",
        "step_completed",
        "epoch_completed",
        "attempt_completed",
        "attempt_failed",
    )
)
TERMINAL_EVENTS = frozenset(("attempt_completed", "attempt_failed"))
ERROR_CODES = frozenset(
    (
        "configuration_invalid",
        "directory_invalid",
        "attempt_already_exists",
        "journal_create_failed",
        "journal_identity_changed",
        "journal_byte_cap_exceeded",
        "journal_event_invalid",
        "journal_counters_invalid",
        "journal_closed",
        "journal_debit_failed",
        "journal_write_failed",
        "journal_fsync_failed",
        "journal_close_failed",
        "journal_input_invalid",
        "journal_truncated",
        "journal_record_invalid",
        "journal_noncanonical",
        "journal_chain_invalid",
    )
)


class JournalError(ValueError):
    """Only fixed error codes; never retain a path, callback error or record."""

    def __init__(self, code: str):
        self.code = code if type(code) is str and code in ERROR_CODES else "configuration_invalid"
        super().__init__(self.code)


def _raise(code: str) -> None:
    raise JournalError(code) from None


def _try_call(callback, *args, **kwargs):
    """Do not leave arbitrary exception objects in a later error's context."""
    try:
        return True, callback(*args, **kwargs)
    except BaseException:
        return False, None


def _close_descriptor(fd: int | None) -> bool:
    # Never retry close: an interrupted close can already have released the FD.
    return fd is None or _try_call(os.close, fd)[0]


def _counter(value: object) -> bool:
    return type(value) is int and 0 <= value <= MAXIMUM_COUNTER


def _valid_counters(counters: object) -> bool:
    return type(counters) is tuple and len(counters) <= 16 and all(_counter(x) for x in counters)


def _canonical(record: dict) -> bytes:
    return json.dumps(
        record, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _frame(sequence: int, previous: str, event: str, counters: tuple[int, ...]) -> bytes:
    raw = _canonical(
        {
            "schema_version": RECORD_SCHEMA,
            "sequence": sequence,
            "previous_sha256": previous,
            "event": event,
            "counters": list(counters),
        }
    )
    return len(raw).to_bytes(4, "big") + raw


def _private_directory(info) -> bool:
    return (
        stat.S_ISDIR(info.st_mode)
        and info.st_uid == os.geteuid()
        and stat.S_IMODE(info.st_mode) & ~0o700 == 0
    )


def _directory_identity(info) -> tuple:
    return info.st_dev, info.st_ino, info.st_uid, stat.S_IMODE(info.st_mode)


def _private_file(info) -> bool:
    return (
        stat.S_ISREG(info.st_mode)
        and info.st_uid == os.geteuid()
        and info.st_nlink == 1
        and stat.S_IMODE(info.st_mode) == 0o600
    )


def _file_identity(info) -> tuple:
    return (
        info.st_dev,
        info.st_ino,
        info.st_uid,
        info.st_nlink,
        stat.S_IMODE(info.st_mode),
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _open_private_directory(directory) -> int:
    """Walk absolute path components by FD; no symlink component is followed."""
    ok, path = _try_call(os.fspath, directory)
    if (
        not ok
        or type(path) is not str
        or not path.startswith("/")
        or path.startswith("//")
        or len(path) > 4096
        or os.path.normpath(path) != path
        or not all(32 <= ord(char) != 127 for char in path)
    ):
        _raise("directory_invalid")
    if os.name != "posix" or not all(
        hasattr(os, flag) for flag in ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC", "geteuid")
    ):
        _raise("configuration_invalid")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    ok, fd = _try_call(os.open, "/", flags)
    if not ok:
        _raise("directory_invalid")
    for part in path.split("/")[1:]:
        if not part:
            continue
        ok, next_fd = _try_call(os.open, part, flags, dir_fd=fd)
        closed = _close_descriptor(fd)
        fd = next_fd if ok else None
        if not ok or not closed:
            _close_descriptor(fd)
            _raise("directory_invalid")
    ok, info = _try_call(os.fstat, fd)
    if not ok or not _private_directory(info):
        _close_descriptor(fd)
        _raise("directory_invalid")
    return fd


class AttemptJournal:
    """Append-only private handle; construct with ``create_new``, never reopen.

    ``append`` returns only the committed sequence and frame digest after fsync.
    Terminal events prevent further records. Any append rejection is sticky.
    ``close`` is idempotent, retains artifacts, and never masks an earlier error;
    a close failure is exposed through ``summary`` instead of being raised.
    """

    def __init__(self, directory_fd: int, file_fd: int, *, debit, maximum_bytes: int):
        self._directory_fd = directory_fd
        self._file_fd = file_fd
        self._debit = debit
        self._maximum_bytes = maximum_bytes
        self._lock = threading.Lock()
        self._sequence = 0
        self._previous = ZERO_DIGEST
        self._attempted_bytes = 0
        self._written_bytes = 0
        self._failure = None
        self._terminal_event = None
        self._closed = False
        self._directory_pin = None
        self._file_pin = None

    @classmethod
    def create_new(cls, directory, *, debit, maximum_bytes: int) -> AttemptJournal:
        return create_new(directory, debit=debit, maximum_bytes=maximum_bytes)

    def _fail(self, code: str) -> None:
        if self._failure is None:
            self._failure = code
        _raise(self._failure)

    def _check_identity(self, *, initialize: bool = False, after_write: bool = False) -> None:
        ok_directory, directory_info = _try_call(os.fstat, self._directory_fd)
        ok_file, file_info = _try_call(os.fstat, self._file_fd)
        ok_named, named_info = _try_call(
            os.stat, FILENAME, dir_fd=self._directory_fd, follow_symlinks=False
        )
        if (
            not ok_directory
            or not ok_file
            or not ok_named
            or not _private_directory(directory_info)
            or not _private_file(file_info)
            or not _private_file(named_info)
            or _file_identity(file_info) != _file_identity(named_info)
            or file_info.st_size != self._written_bytes
        ):
            self._fail("journal_identity_changed")
        directory_pin = _directory_identity(directory_info)
        file_pin = _file_identity(file_info)
        if not initialize:
            if directory_pin != self._directory_pin or file_pin[:5] != self._file_pin[:5]:
                self._fail("journal_identity_changed")
            if not after_write and file_pin != self._file_pin:
                self._fail("journal_identity_changed")
        self._directory_pin = directory_pin
        self._file_pin = file_pin

    def _fsync(self, fd: int) -> None:
        if not _try_call(os.fsync, fd)[0]:
            self._fail("journal_fsync_failed")

    def _append(self, event: str, counters: tuple[int, ...], *, initial: bool = False) -> dict:
        if self._failure is not None:
            self._fail(self._failure)
        if self._closed or self._terminal_event is not None:
            self._fail("journal_closed")
        if type(event) is not str or event not in EVENTS or (event == "attempt_created") != initial:
            self._fail("journal_event_invalid")
        if not _valid_counters(counters):
            self._fail("journal_counters_invalid")
        if self._sequence >= MAXIMUM_COUNTER:
            self._fail("journal_record_invalid")
        frame = _frame(self._sequence + 1, self._previous, event, counters)
        self._check_identity()
        cursor = 0
        while cursor < len(frame):
            chunk = frame[cursor:]
            if len(chunk) > self._maximum_bytes - self._attempted_bytes:
                self._fail("journal_byte_cap_exceeded")
            # Count conservatively even if debit raises after a partial effect.
            self._attempted_bytes += len(chunk)
            ok, result = _try_call(self._debit, len(chunk))
            if not ok or result is not None:
                self._fail("journal_debit_failed")
            self._check_identity()
            ok, written = _try_call(os.write, self._file_fd, chunk)
            if not ok or type(written) is not int or not 0 < written <= len(chunk):
                self._fail("journal_write_failed")
            cursor += written
            self._written_bytes += written
            self._check_identity(after_write=True)
        self._fsync(self._file_fd)
        self._check_identity()
        self._sequence += 1
        self._previous = hashlib.sha256(frame).hexdigest()
        if event in TERMINAL_EVENTS:
            self._terminal_event = event
        return {"sequence": self._sequence, "record_sha256": self._previous}

    def append(self, event: str, counters: tuple[int, ...] = ()) -> dict:
        with self._lock:
            return self._append(event, counters)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            file_fd, directory_fd = self._file_fd, self._directory_fd
            self._file_fd = self._directory_fd = None
            self._closed = True
            file_closed = _close_descriptor(file_fd)
            directory_closed = _close_descriptor(directory_fd)
            if (not file_closed or not directory_closed) and self._failure is None:
                self._failure = "journal_close_failed"

    def summary(self) -> dict:
        with self._lock:
            return {
                "schema_version": "lumi.sft-attempt-journal-summary.v0.1",
                "state": (
                    "failed"
                    if self._failure
                    else "terminal"
                    if self._terminal_event
                    else "closed"
                    if self._closed
                    else "open"
                ),
                "failure_code": self._failure,
                "terminal_event": self._terminal_event,
                "records_committed": self._sequence,
                "write_bytes_attempted": self._attempted_bytes,
                "file_bytes_written": self._written_bytes,
                "maximum_bytes": self._maximum_bytes,
                "last_record_sha256": self._previous,
                "closed": self._closed,
                "resume_permitted": False,
                "execution_permitted": False,
            }


def create_new(directory, *, debit, maximum_bytes: int) -> AttemptJournal:
    """Create ``directory/attempt.journal`` exclusively; never remove on failure."""
    if (
        type(maximum_bytes) is not int
        or not 1 <= maximum_bytes <= MAXIMUM_BYTES
        or not callable(debit)
    ):
        _raise("configuration_invalid")
    if len(_frame(1, ZERO_DIGEST, "attempt_created", ())) > maximum_bytes:
        _raise("journal_byte_cap_exceeded")
    directory_fd = _open_private_directory(directory)
    file_fd = None
    code = None
    try:
        file_fd = os.open(
            FILENAME,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=directory_fd,
        )
    except FileExistsError:
        code = "attempt_already_exists"
    except BaseException:
        code = "journal_create_failed"
    if code is not None:
        _close_descriptor(directory_fd)
        _raise(code)
    journal = AttemptJournal(directory_fd, file_fd, debit=debit, maximum_bytes=maximum_bytes)
    code = None
    try:
        journal._check_identity(initialize=True)
        journal._append("attempt_created", (), initial=True)
        journal._fsync(directory_fd)
        journal._check_identity()
    except JournalError as error:
        code = error.code
    except BaseException:
        code = "journal_create_failed"
    if code is not None:
        journal.close()
        _raise(code)
    return journal


def verify_bytes(raw: bytes) -> dict:
    """Strict, read-only chain audit; a valid prefix never enables resume."""
    if type(raw) is not bytes or not 0 < len(raw) <= MAXIMUM_BYTES:
        _raise("journal_input_invalid")
    cursor, sequence, previous, terminal = 0, 0, ZERO_DIGEST, None
    while cursor < len(raw):
        if len(raw) - cursor < 4:
            _raise("journal_truncated")
        size = int.from_bytes(raw[cursor : cursor + 4], "big")
        if not 0 < size <= MAXIMUM_FRAME_BYTES:
            _raise("journal_record_invalid")
        end = cursor + 4 + size
        if end > len(raw):
            _raise("journal_truncated")
        payload = raw[cursor + 4 : end]
        ok, record = _try_call(json.loads, payload)
        if (
            not ok
            or type(record) is not dict
            or set(record) != {"schema_version", "sequence", "previous_sha256", "event", "counters"}
            or record["schema_version"] != RECORD_SCHEMA
            or type(record["event"]) is not str
            or record["event"] not in EVENTS
            or type(record["counters"]) is not list
            or not _valid_counters(tuple(record["counters"]))
            or type(record["sequence"]) is not int
            or not 1 <= record["sequence"] <= MAXIMUM_COUNTER
        ):
            _raise("journal_record_invalid")
        ok, canonical = _try_call(_canonical, record)
        if not ok or canonical != payload:
            _raise("journal_noncanonical")
        if (
            record["sequence"] != sequence + 1
            or record["previous_sha256"] != previous
            or (record["event"] == "attempt_created") != (sequence == 0)
            or (sequence == 0 and record["counters"] != [])
            or terminal is not None
        ):
            _raise("journal_chain_invalid")
        sequence += 1
        previous = hashlib.sha256(raw[cursor:end]).hexdigest()
        if record["event"] in TERMINAL_EVENTS:
            terminal = record["event"]
        cursor = end
    return {
        "schema_version": "lumi.sft-attempt-journal-audit.v0.1",
        "records_verified": sequence,
        "bytes_verified": len(raw),
        "last_record_sha256": previous,
        "terminal_event": terminal,
        "complete_attempt": terminal is not None,
        "resume_permitted": False,
        "execution_permitted": False,
    }
