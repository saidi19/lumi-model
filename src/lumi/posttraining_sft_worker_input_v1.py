"""One-shot frozen-v4 train-prefix loading through explicit v2 input accounting.

The caller has already admitted the exact root/five pins, runtime, memory and
parent channel. This module makes no first ML import and grants no real-data or
execution approval. Only the frozen loader's secure opener/finisher are replaced
in an isolated globals dictionary; its validation and tensor code are unchanged.
The Dataset initializer also uses a private globals copy to reach that loader.
Original module globals/classes are never modified. Validation/unselected content
opens are denied, while complete inventory metadata checks remain mandatory.

The v2 parent cap is required independently: no category remapping/default cap.
One object loads once, never retries or refunds. A failure must retire all workers.
This is not the full SFT launcher, OS sandbox or production dataset seal. Only
summary() and fixed errors are safe to publish; Dataset/tracebacks remain private.
"""

from __future__ import annotations

import os
import re
import stat
import sys
import threading
from pathlib import Path
from types import FunctionType, ModuleType

from .posttraining_sft_input_reader_v1 import InputReadHandle

RUNTIME_MODULE = "lumi.posttraining_sft_runtime_v4"
ERROR_CODES = frozenset({
    "invalid_arguments", "runtime_rejected", "owner_mismatch", "already_consumed",
    "reentry", "reserve_failed", "load_failed", "input_rejected", "terminal_failure",
})
PIN_FIELDS = (
    "expected_manifest_sha256", "expected_manifest_file_sha256", "expected_inventory_sha256",
    "expected_metadata_inventory_sha256", "expected_seal_sha256",
)


class WorkerInputError(RuntimeError):
    def __init__(self, code="load_failed"):
        self.code = code if type(code) is str and code in ERROR_CODES else "load_failed"
        super().__init__(self.code)


def _clone(function, replacements):
    copied = dict(function.__globals__)
    copied.update(replacements)
    result = FunctionType(function.__code__, copied, function.__name__,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = (dict(function.__kwdefaults__)
                             if function.__kwdefaults__ is not None else None)
    return result


class _LoaderHandle:
    """Nonthrowing cleanup seam so frozen finally still closes its root FD.

    Primitive close failures poison the owner before returning. They are never
    retried or accepted as a successful load. No borrowed root descriptor is
    closed by this proxy; the frozen loader remains its sole close authority.
    """

    def __init__(self, reader, owner):
        self._reader, self._owner = reader, owner

    def __repr__(self):
        return "<LoaderHandle private>"

    @property
    def closed(self):
        return self._reader.closed

    def _call(self, method, *arguments):
        self._owner._loading()
        result = getattr(self._reader, method)(*arguments)
        self._owner._check()
        return result

    def read(self, size):
        return self._call("read", size)

    def readline(self, size):
        return self._call("readline", size)

    def tell(self):
        return self._call("tell")

    def close(self):
        try:
            self._reader.close()
        except BaseException:
            self._owner._poison("load_failed")


class WorkerInput:
    def __init__(self, *, rank, reserve, interrupt):
        if hasattr(self, "_owner"):
            self._fail("already_consumed")
        if (type(rank) is not int or not 0 <= rank < 12
                or not callable(reserve) or not callable(interrupt)):
            raise WorkerInputError("invalid_arguments") from None
        self._rank, self._parent_reserve, self._interrupt = rank, reserve, interrupt
        self._owner = os.getpid(), threading.get_ident()
        self._state, self._failure, self._interrupted = "idle", None, False
        self._handles, self._names = [], set()
        self._root_fd, self._runtime, self._records = None, None, 0

    def __repr__(self):
        return "<WorkerInput private>"

    def _interrupt_once(self):
        if not self._interrupted:
            self._interrupted = True
            try:
                self._interrupt()
            except BaseException:
                pass

    def _poison(self, code):
        if self._state == "complete":
            return  # A retired loader cannot interrupt a later shared epoch.
        if self._failure is None:
            self._failure = code
        self._state = "failed"
        self._interrupt_once()

    def _fail(self, code):
        if self._state == "complete":
            raise WorkerInputError("already_consumed") from None
        self._poison(code)
        raise WorkerInputError(self._failure) from None

    def _check(self):
        if self._owner != (os.getpid(), threading.get_ident()):
            self._fail("owner_mismatch")
        if self._state == "failed" or self._interrupted:
            self._fail("terminal_failure")

    def _loading(self):
        self._check()
        if self._state != "loading":
            self._fail("already_consumed")

    def _reserve(self, rank, **amounts):
        self._loading()
        if type(rank) is not int or rank != self._rank:
            self._fail("reserve_failed")
        try:
            result = self._parent_reserve(rank, **amounts)
        except BaseException:
            self._fail("reserve_failed")
        self._loading()
        if result is not None:
            self._fail("reserve_failed")

    def _open(self, directory_descriptor, name, *, require_private_mode):
        self._loading()
        if type(name) is not str or type(require_private_mode) is not bool or name in self._names:
            self._fail("input_rejected")
        if name == "manifest.json":
            resource = "encoded_metadata_read_bytes"
        elif re.fullmatch(r"train-[0-9]{5,}\.index\.jsonl", name):
            resource = "encoded_index_read_bytes"
        elif re.fullmatch(r"train-[0-9]{5,}\.(tokens\.u32le|labels\.i32le)", name):
            resource = "encoded_binary_read_bytes"
        else:
            self._fail("input_rejected")
        if self._root_fd is None:
            self._root_fd = directory_descriptor
        if directory_descriptor != self._root_fd:
            self._fail("input_rejected")
        root_info = os.fstat(directory_descriptor)
        if (not stat.S_ISDIR(root_info.st_mode) or stat.S_IMODE(root_info.st_mode) != 0o700
                or root_info.st_uid != os.geteuid()):
            self._fail("input_rejected")
        # Nonblocking additionally prevents a raced FIFO from hanging before fstat.
        flags = self._runtime._secure_open_flags() | os.O_NONBLOCK
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
        transferred = False
        try:
            opened = os.fstat(descriptor)
            identity = self._runtime._secure_file_identity(opened)
            path_info = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
            if (not stat.S_ISREG(opened.st_mode) or identity.links != 1
                    or identity.owner != os.geteuid() or identity.mode != 0o600
                    or self._runtime._secure_file_identity(path_info) != identity):
                self._fail("input_rejected")
            reader = InputReadHandle(descriptor, rank=self._rank, resource=resource,
                                     reserve=self._reserve, interrupt=self._interrupt_once)
            transferred = True
            try:
                handle = _LoaderHandle(reader, self)
                self._handles.append((handle, name, identity))
                self._names.add(name)
            except BaseException:
                try:
                    reader.close()
                except BaseException:
                    self._poison("load_failed")
                raise
            return handle, identity
        finally:
            if not transferred:
                os.close(descriptor)

    def _finish(self, handle, directory_descriptor, name, initial):
        self._loading()
        if not any(handle is item and name == known and initial == identity
                   for item, known, identity in self._handles):
            self._fail("input_rejected")
        try:
            handle._reader._verify()
            after = self._runtime._secure_file_identity(os.fstat(handle._reader._fd))
            final_info = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
            final = self._runtime._secure_file_identity(final_info)
            if (directory_descriptor != self._root_fd or not stat.S_ISREG(final_info.st_mode)
                    or after != initial or final != initial):
                self._fail("input_rejected")
        finally:
            handle.close()
        self._check()

    def load(self, root, *, record_limit, expected_manifest_sha256,
             expected_manifest_file_sha256, expected_inventory_sha256,
             expected_metadata_inventory_sha256, expected_seal_sha256):
        self._check()
        if self._state != "idle":
            self._fail("reentry" if self._state == "loading" else "already_consumed")
        self._state = "loading"
        pins = dict(zip(PIN_FIELDS, (
            expected_manifest_sha256, expected_manifest_file_sha256, expected_inventory_sha256,
            expected_metadata_inventory_sha256, expected_seal_sha256,
        ), strict=True))
        try:
            if (type(record_limit) is not int or not 1 <= record_limit <= 12
                    or type(root) not in (str, type(Path(".")))
                    or any(type(pin) is not str or re.fullmatch(r"[a-f0-9]{64}", pin) is None
                           for pin in pins.values())):
                self._fail("invalid_arguments")
            path = Path(root)
            if (not path.is_absolute() or ".." in path.parts or "\0" in str(path)
                    or path == Path("/")):
                self._fail("invalid_arguments")
            runtime, torch = sys.modules.get(RUNTIME_MODULE), sys.modules.get("torch")
            if (not isinstance(runtime, ModuleType) or not isinstance(torch, ModuleType)
                    or re.match(r"2\.(10|14)\.", getattr(torch, "__version__", "")) is None):
                self._fail("runtime_rejected")
            original = runtime._load_verified_sft_split
            initializer = runtime.EncodedSFTDataset.__init__
            if (type(original) is not FunctionType or type(initializer) is not FunctionType
                    or original.__globals__ is not runtime.__dict__
                    or initializer.__globals__ is not runtime.__dict__):
                self._fail("runtime_rejected")
            self._runtime = runtime
            loader = _clone(original, {"_open_secure_dataset_file": self._open,
                                       "_finish_secure_dataset_file": self._finish})
            initialize = _clone(initializer, {"_load_verified_sft_split": loader})
            dataset = object.__new__(runtime.EncodedSFTDataset)
            initialize(dataset, path, split="train", record_limit=record_limit, **pins)
            self._check()
            audit = dataset.verification
            if (audit["profile"] != "token_consistent_v0.2"
                    or audit["records_loaded"] != record_limit or len(dataset) != record_limit
                    or audit["encoded_index_row_reads"] != record_limit
                    or audit["fully_hashed_shards"] != 0
                    or audit["full_content_hashes_verified"] is not False
                    or audit["all_inventory_file_hashes_verified"] is not False
                    or audit["metadata_inventory_verified_before_selected_content_read"] is not True
                    or any(not handle.closed or handle._reader._failed
                           for handle, _, _ in self._handles)):
                self._fail("input_rejected")
            totals = self.summary()
            if (totals["binary_bytes_read"] != audit["binary_bytes_read"]
                    or totals["index_bytes_read"] != audit["index_bytes_read"]
                    or totals["encoded_index_rows"] != record_limit
                    or totals["file_opens"] != audit["file_opens"]
                    or sum(name == "manifest.json" for _, name, _ in self._handles) != 1
                    or totals["metadata_bytes_read"] != sum(
                        identity.size for _, name, identity in self._handles
                        if name == "manifest.json")):
                self._fail("input_rejected")
            self._records, self._state = record_limit, "complete"
            return dataset
        except BaseException:
            self._fail("load_failed")
        finally:
            for handle, _, _ in self._handles:
                if not handle.closed:
                    try:
                        handle.close()
                    except BaseException:
                        self._interrupt_once()

    def summary(self):
        def total(resource):
            return sum(handle._reader.summary()["bytes_read"] for handle, _, _ in self._handles
                       if handle._reader._resource == resource)

        return {
            "schema_version": "lumi.sft-worker-input-summary.v0.1", "state": self._state,
            "failure_code": self._failure,
            "metadata_bytes_read": total("encoded_metadata_read_bytes"),
            "binary_bytes_read": total("encoded_binary_read_bytes"),
            "index_bytes_read": total("encoded_index_read_bytes"),
            "encoded_index_rows": sum(handle._reader.summary()["index_rows_reserved"]
                                      for handle, _, _ in self._handles),
            "file_opens": len(self._handles), "records_loaded": self._records,
            "full_content_hashes_verified": False, "production_eligible": False,
            "real_input_execution_permitted": False,
        }
