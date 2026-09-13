"""One verified, metered DCP reader lease; no real-import or host admission.

Reuse the frozen checkpoint session, strict serializer views and instance-local
DCP adapter. Torch is imported lazily only after the complete checksum audit.
All content requests share one nonrefundable logical-byte cap; these are not
physical disk bytes, OS quotas, allocation limits or safe-unpickling guarantees.
Only already trusted, hash-bound DCP metadata may be deserialized by the caller.

Input files must already be private 0600, directories 0700; no chmod is done.
One argument-free initial DCP reset is a consumed no-op handshake, never a
counter/source/deadline reset. No retry, source mutation or global patch exists.
The caller owns a quiescent source and must provide an independent hard guardian.
Only fixed error arguments are safe to report, never suppressed exception text.
"""

from __future__ import annotations

import importlib
import os
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType

from .posttraining_sft_checkpoint_reader_v1 import CheckpointReadSession
from .posttraining_sft_checkpoint_worker_v1 import _dcp_reader

MAXIMUM_READ_CALL_BYTES = 1024**3
MAXIMUM_BYTES = 2**63 - 1
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
ERROR_CODES = frozenset({
    "invalid_arguments", "path_rejected", "owner_mismatch", "reentry", "closed",
    "terminal_failure", "read_cap_exceeded", "budget_check_failed", "audit_failed",
    "dependencies_unavailable", "storage_rejected", "reset_rejected", "reader_failed",
    "restore_already_started",
    "close_failed", "operation_failed",
})


class DCPReadError(RuntimeError):
    def __init__(self, code="operation_failed"):
        self.code = code if type(code) is str and code in ERROR_CODES else "operation_failed"
        super().__init__(self.code)


class _Lease:
    def __init__(self, maximum, maximum_call, callback):
        self._owner = os.getpid(), threading.get_ident()
        self._maximum, self._callback = maximum, callback
        self.limits = MappingProxyType({"maximum_read_call_bytes": maximum_call})
        self._session = self._reader = None
        self._spent = self._audit_bytes = self._files = 0
        self._failed = self._closed = self._success = self._in_callback = False

    def __repr__(self):
        return "<DCPReadLease private>"

    def _fail(self, code):
        self._failed = True
        if self._session is not None:
            self._session._failed = True
        raise DCPReadError(code) from None

    def _owner_check(self):
        if self._owner != (os.getpid(), threading.get_ident()):
            self._fail("owner_mismatch")
        if self._in_callback:
            self._fail("reentry")

    def _tick(self):
        self._owner_check()
        if self._failed:
            self._fail("terminal_failure")
        self._in_callback = True
        try:
            if self._callback() is not None:
                self._fail("budget_check_failed")
            if self._failed:
                self._fail("terminal_failure")
        except BaseException:
            self._fail("budget_check_failed")
        finally:
            self._in_callback = False

    def _check(self):
        self._owner_check()
        if self._closed:
            self._fail("closed")
        if self._failed or (self._session is not None and self._session._failed):
            self._fail("terminal_failure")
        self._tick()

    def _reserve(self, rank, *, checkpoint_read_bytes):
        self._check()
        amount = checkpoint_read_bytes
        if type(rank) is not int or rank != 0 or type(amount) is not int or amount < 0:
            self._fail("invalid_arguments")
        if self._spent + amount > self._maximum:
            self._fail("read_cap_exceeded")
        self._spent += amount  # Never refunded after a short/failed read.
        self._tick()

    @property
    def reader(self):
        self._check()
        if self._reader is None:
            self._fail("reader_failed")
        return self._reader

    def summary(self):
        self._owner_check()
        if self._failed or (self._session is not None and self._session._failed):
            self._fail("terminal_failure")
        if self._closed:
            if not self._success:
                self._fail("closed")
        else:
            self._check()
        return {
            "dcp_read_bytes_reserved": self._spent,
            "dcp_input_audit_bytes": self._audit_bytes,
            "model_payload_read_bytes_reserved": self._spent - self._audit_bytes,
            "hash_verified_files": self._files,
            "dcp_reads_metered": True,
            "physical_disk_bytes_measured": False,
            "os_resource_limits_enforced": False,
            "real_4b_import": False,
            "training_authorized": False,
            "production_eligible": False,
        }


def _open_root(root, lease):
    if (type(root) is not type(Path()) or not root.is_absolute() or ".." in root.parts
            or not 3 <= len(root.parts) <= 128 or len(str(root)) > 4096):
        lease._fail("path_rejected")
    descriptor = None
    try:
        descriptor = os.open("/", _DIR_FLAGS)
        for component in root.parts[1:]:
            lease._check()
            child = os.open(component, _DIR_FLAGS, dir_fd=descriptor)
            old, descriptor = descriptor, child
            os.close(old)  # An ambiguous close is never retried.
        result, descriptor = descriptor, None
        return result
    except BaseException:
        lease._fail("path_rejected")
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except BaseException:
                lease._fail("close_failed")


def _bind_reader(session, lease, filesystem):
    reader = _dcp_reader(session, lease, filesystem)
    reset_seen = data_started = False

    def initial_reset(*args, **kwargs):
        nonlocal reset_seen
        lease._check()
        if args or kwargs or reset_seen or data_started:
            lease._fail("reset_rejected")
        # Torch load_state_dict calls reset once before its internal metadata
        # reread. Accept that handshake only; preserve every resource and pin.
        reset_seen = True
        lease._check()

    def guarded(method, *, restore=False):
        def call(*args, **kwargs):
            nonlocal data_started
            lease._check()
            if restore:
                if data_started:
                    lease._fail("restore_already_started")
                data_started = True  # Consumed before entering native code.
            try:
                result = method(*args, **kwargs)
                lease._check()
                return result
            except BaseException:
                lease._fail("reader_failed")
        return call

    # Only this fresh adapter instance changes; frozen classes/globals do not.
    reader.reset = initial_reset
    reader.read_metadata = guarded(reader.read_metadata)
    reader.read_data = guarded(reader.read_data, restore=True)
    return reader


@contextmanager
def verified_dcp_reader(root: Path, *, expected_manifest_sha256: str,
                        maximum_file_bytes: int, maximum_read_bytes: int,
                        maximum_read_call_bytes: int, check_budget):
    """Yield one reader; success includes explicit session close and a final tick.

    Summary remains available after a successful exit, never after any failure.
    Repeated DCP metadata/tensor reads are charged in addition to the initial
    full-manifest audit. File integrity is neither origin proof nor admission.
    """
    if (not callable(check_budget) or type(expected_manifest_sha256) is not str
            or re.fullmatch(r"[a-f0-9]{64}", expected_manifest_sha256) is None
            or any(type(value) is not int or not 0 < value <= MAXIMUM_BYTES
                   for value in (maximum_file_bytes, maximum_read_bytes))
            or type(maximum_read_call_bytes) is not int
            or not 0 < maximum_read_call_bytes <= MAXIMUM_READ_CALL_BYTES):
        raise DCPReadError("invalid_arguments") from None
    lease = _Lease(maximum_read_bytes, maximum_read_call_bytes, check_budget)
    descriptor = None
    try:
        lease._check()
        descriptor = _open_root(root, lease)
        try:
            session = CheckpointReadSession(
                descriptor, rank=0, reserve=lease._reserve,
                expected_manifest_sha256=expected_manifest_sha256,
                maximum_file_bytes=maximum_file_bytes, maximum_read_bytes=maximum_read_bytes,
            )
            lease._session, descriptor = session, None  # Session now owns the FD.
            audit = session.verify()
        except BaseException:
            lease._fail("audit_failed")
        lease._audit_bytes = audit["reserved_read_bytes"]
        lease._files = audit["payload_files"]
        if lease._audit_bytes != lease._spent:
            lease._fail("audit_failed")
        lease._check()
        try:
            filesystem = importlib.import_module("torch.distributed.checkpoint.filesystem")
        except BaseException:
            lease._fail("dependencies_unavailable")
        lease._check()
        lease._reader = _bind_reader(session, lease, filesystem)
        lease._check()
        yield lease
        lease._check()
    except DCPReadError:
        lease._failed = True
        raise
    except BaseException:
        lease._fail("operation_failed")
    finally:
        lease._closed = True  # Revoke before any ambiguous close.
        try:
            if lease._session is not None:
                lease._session.close()
            elif descriptor is not None:
                owned, descriptor = descriptor, None
                os.close(owned)
        except BaseException:
            lease._fail("close_failed")
        if not lease._failed:
            lease._tick()  # Even post-close deadline failure cannot claim success.
            lease._success = True
