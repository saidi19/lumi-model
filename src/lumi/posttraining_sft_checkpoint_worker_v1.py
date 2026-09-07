"""Explicit worker checkpoint save/restore through metered DCP storage.

No eager ML import or execution entrypoint. The admitted caller must already
have imported Torch, own the private output, provide one shared reservation and
interrupt service, establish its bounded process group, and enforce resource /
parent-death policies. This is not real-data/GPU admission or an OS sandbox.

Only trusted, own-generated DCP metadata may be restored: DCP uses pickle.
Hash integrity is NOT safe-unpickling or origin authentication. Tensor allocation,
collective transport, and native library logging outside the scoped Python DCP
loggers remain the caller's responsibility. No stock worker globals are patched.
Failure interrupts once, retains evidence, and permits cleanup only; no retry or
pruning. Fixed exception args are safe; never log suppressed Python tracebacks.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import sys
import threading
import uuid
import warnings
from contextlib import contextmanager
from pathlib import PurePosixPath
from types import MappingProxyType

from .posttraining_sft_checkpoint_publication_v1 import (
    CheckpointPublisher,
    _native_rename_noreplace,
    _rename_binding,
)
from .posttraining_sft_checkpoint_reader_v1 import CheckpointReadSession, _metadata, _private
from .posttraining_sft_checkpoint_saver_v1 import CheckpointStagingFinalizer, CheckpointWriteStream
from .posttraining_sft_checkpoint_view_v1 import CheckpointReadView
from .posttraining_sft_metered_io_v1 import MAX_FILE_BYTES

OPAQUE_ROOT = "lumi-owned-checkpoint"
STATE_NAME = "distributed-training-state.json"
LIMIT_KEYS = frozenset({"maximum_file_bytes", "maximum_read_bytes", "maximum_read_call_bytes",
                        "maximum_write_call_bytes", "maximum_sync_operations"})
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
_FILE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
_SERIALIZER_LOCK = threading.Lock()
ERROR_CODES = frozenset({"invalid_arguments", "invalid_descriptor", "descriptor_changed",
                         "owner_mismatch", "reentry", "terminal_failure", "runtime_rejected",
                         "storage_rejected", "read_cap_exceeded", "reserve_failed",
                         "collective_failed", "operation_failed", "close_failed"})
STAGES = frozenset({"idle", "runtime", "prepare", "writer_setup", "model_save", "trainer_save",
                    "state_save", "staging_verify", "publication", "checkpoint_verify",
                    "reader_setup", "model_restore", "trainer_restore", "complete"})


class CheckpointWorkerError(RuntimeError):
    def __init__(self, code="operation_failed"):
        self.code = code if type(code) is str and code in ERROR_CODES else "operation_failed"
        super().__init__(self.code)


def _directory_info(fd):
    if type(fd) is not int or fd < 3:
        raise CheckpointWorkerError("invalid_descriptor") from None
    try:
        info = os.fstat(fd)
        flags = fcntl.fcntl(fd, fcntl.F_GETFL), fcntl.fcntl(fd, fcntl.F_GETFD)
        if (not _private(info, directory=True) or flags[0] & os.O_ACCMODE != os.O_RDONLY
                or flags[0] & os.O_APPEND or not flags[1] & fcntl.FD_CLOEXEC):
            raise ValueError
        return _metadata(info), flags
    except BaseException:
        raise CheckpointWorkerError("invalid_descriptor") from None


@contextmanager
def _child_directory(parent_fd, name):
    _directory_info(parent_fd)
    if type(name) is not str or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", name):
        raise CheckpointWorkerError("storage_rejected") from None
    if name in {".", ".."}:
        raise CheckpointWorkerError("storage_rejected") from None
    info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not _private(info, directory=True):
        raise CheckpointWorkerError("storage_rejected") from None
    fd = os.open(name, _DIR_FLAGS, dir_fd=parent_fd)
    try:
        if _metadata(os.fstat(fd)) != _metadata(info):
            raise CheckpointWorkerError("descriptor_changed") from None
        yield fd
    finally:
        os.close(fd)


class _WriteDirectory:
    """Borrowed private directory; DCP creates distinct rank files concurrently."""

    def __init__(self, fd, worker, *, model_directory):
        self.fd, self.worker, self.model_directory = fd, worker, model_directory
        self.identity, self.flags = _directory_info(fd)
        self.receipts, self.identities = {}, {}

    def check(self):
        self.worker._check()
        identity, flags = _directory_info(self.fd)
        # Other admitted ranks legitimately create siblings in the model directory.
        if any(identity[i] != self.identity[i] for i in (0, 1, 2, 4)) or flags != self.flags:
            self.worker._fail("descriptor_changed")

    def _leaf(self, path):
        parts = PurePosixPath(path).parts
        if len(parts) != 2 or parts[0] != OPAQUE_ROOT:
            self.worker._fail("storage_rejected")
        name = parts[1]
        if self.model_directory:
            allowed = (name == f"__{self.worker.rank}_0.distcp"
                       or name in {".metadata.tmp", ".metadata"})
        else:
            allowed = name in {f"trainer-rank-{self.worker.rank:05d}.pt", STATE_NAME}
        if not allowed:
            self.worker._fail("storage_rejected")
        return name

    def init_path(self, path):
        if str(path) != OPAQUE_ROOT:
            self.worker._fail("storage_rejected")
        return PurePosixPath(OPAQUE_ROOT)

    def concat_path(self, path, suffix):
        self.init_path(path)
        value = PurePosixPath(OPAQUE_ROOT) / suffix
        self._leaf(value)
        return value

    def mkdir(self, path):
        self.init_path(path)
        self.check()  # Already created by the coordinator, never mkdir parents here.

    def exists(self, path):
        self.check()
        name = self._leaf(path)
        try:
            os.stat(name, dir_fd=self.fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True

    @contextmanager
    def create_stream(self, path, mode):
        self.check()
        name = self._leaf(path)
        if mode != "wb" or name in self.receipts or name == ".metadata":
            self.worker._fail("storage_rejected")
        if name in {".metadata.tmp", STATE_NAME} and self.worker.rank != 0:
            self.worker._fail("storage_rejected")
        fd, stream = os.open(name, _FILE_FLAGS, 0o600, dir_fd=self.fd), None
        try:
            opened = _metadata(os.fstat(fd))
            self.check()
            stream = CheckpointWriteStream(fd, rank=self.worker.rank, reserve=self.worker._reserve,
                                           maximum_file_bytes=self.worker.limits["maximum_file_bytes"],
                                           maximum_write_call_bytes=self.worker.limits[
                                               "maximum_write_call_bytes"],
                                           maximum_sync_operations=self.worker.limits[
                                               "maximum_sync_operations"])
            yield stream
            stream.close()
            self.check()
            info = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
            observed = _metadata(info)
            if (not _private(info, directory=False)
                    or any(observed[i] != opened[i] for i in (0, 1, 2, 3, 4))):
                self.worker._fail("descriptor_changed")
            receipt = stream.receipt()
            if info.st_size != receipt["bytes"]:
                self.worker._fail("descriptor_changed")
            self.receipts[name], self.identities[name] = receipt, _metadata(info)
        except BaseException:
            if stream is not None:
                stream._failed = True
            raise
        finally:
            if stream is None:
                os.close(fd)
            else:
                stream.close()

    def rename(self, path, new_path):
        self.check()
        old, new = self._leaf(path), self._leaf(new_path)
        if (old, new) != (".metadata.tmp", ".metadata") or old not in self.receipts:
            self.worker._fail("storage_rejected")
        if _metadata(os.stat(old, dir_fd=self.fd, follow_symlinks=False)) != self.identities[old]:
            self.worker._fail("descriptor_changed")
        _native_rename_noreplace(_rename_binding(), self.fd, old, new)
        after = _metadata(os.stat(new, dir_fd=self.fd, follow_symlinks=False))
        if after[:7] != self.identities[old][:7]:
            self.worker._fail("descriptor_changed")
        self.receipts[new] = self.receipts.pop(old)
        self.identities[new] = after
        del self.identities[old]

    def rm_file(self, path):
        self.worker._fail("storage_rejected")


class _ReadDirectory:
    def __init__(self, session, worker):
        self.session, self.worker = session, worker

    def concat_path(self, path, suffix):
        if str(path) != OPAQUE_ROOT or type(suffix) is not str:
            self.worker._fail("storage_rejected")
        if not re.fullmatch(r"(?:__[0-9]+_[0-9]+\.distcp|\.metadata)", suffix):
            self.worker._fail("storage_rejected")
        return PurePosixPath(OPAQUE_ROOT) / suffix

    @contextmanager
    def create_stream(self, path, mode):
        self.worker._check()
        parts = PurePosixPath(path).parts
        if mode != "rb" or len(parts) != 2 or parts[0] != OPAQUE_ROOT:
            self.worker._fail("storage_rejected")
        self.concat_path(OPAQUE_ROOT, parts[1])
        view = CheckpointReadView(self.session, "model/" + parts[1],
                                  maximum_read_call_bytes=self.worker.limits["maximum_read_call_bytes"])
        try:
            yield view
        finally:
            view.close()


class _NoReadTransforms:
    def transform_load_stream(self, read_item, transform_descriptors, raw_stream):
        if transform_descriptors:
            raise CheckpointWorkerError("storage_rejected") from None
        return raw_stream


def _dcp_writer(fd, worker, filesystem_module):
    backend = _WriteDirectory(fd, worker, model_directory=True)
    writer = filesystem_module.FileSystemWriter(
        OPAQUE_ROOT, single_file_per_rank=True, thread_count=1, per_thread_copy_ahead=0,
        sync_files=False, overwrite=False, _extensions=(),
    )
    if (writer.sync_files is not False or writer.overwrite is not False
            or writer.thread_count != 1 or writer.serialization_format.name != "TORCH_SAVE"):
        worker._fail("runtime_rejected")
    writer.fs, writer.path = backend, PurePosixPath(OPAQUE_ROOT)
    return writer, backend


def _dcp_reader(session, worker, filesystem_module):
    class Reader(filesystem_module.FileSystemReader):
        def _slice_file(self, file, sinfo):
            if getattr(sinfo, "transform_descriptors", None):
                worker._fail("storage_rejected")
            return file.slice(sinfo.offset, sinfo.length)

    reader = Reader(OPAQUE_ROOT)
    reader.fs, reader.path = _ReadDirectory(session, worker), PurePosixPath(OPAQUE_ROOT)
    reader.transforms = _NoReadTransforms()
    return reader


class CheckpointWorker:
    """Checkpoint hooks for admitted model/optimizer objects, not a train launcher.

    Parent FDs are borrowed; caller must keep them immutable/uniquely controlled
    for each call. Read caps accumulate over this instance. Shared write/cost/
    runtime quotas are supplied by reserve, not inferred from retained file size.
    A failed collective still needs the caller's bounded group/guardian teardown.
    """

    def __init__(self, *, rank, world_size, reserve, interrupt, limits):
        if hasattr(self, "_owner"):
            self._fail("invalid_arguments")
        if (type(rank) is not int or type(world_size) is not int
                or not 1 <= world_size <= 12 or not 0 <= rank < world_size
                or not callable(reserve) or not callable(interrupt)
                or type(limits) is not dict or set(limits) != LIMIT_KEYS
                or any(type(n) is not int or not 0 < n <= MAX_FILE_BYTES for n in limits.values())):
            raise CheckpointWorkerError("invalid_arguments") from None
        self.rank, self.world_size = rank, world_size
        self.limits = MappingProxyType(dict(limits))
        self._reservation, self._interrupt = reserve, interrupt
        self._owner = os.getpid(), threading.get_ident()
        self._failed, self._busy, self._interrupted, self._reads = False, False, False, 0
        self._reserving = False
        self._stage, self.failure_stage, self.failure_code = "idle", None, None

    def __repr__(self):
        return "<CheckpointWorker private>"

    def _fail(self, code):
        self._failed = True
        if self.failure_stage is None:
            self.failure_stage = self._stage if self._stage in STAGES else "idle"
            self.failure_code = code if code in ERROR_CODES else "operation_failed"
        if not self._interrupted:
            self._interrupted = True
            try:
                self._interrupt()
            except BaseException:
                pass  # A fail-closed Supervisor interrupt intentionally raises its fixed code.
        raise CheckpointWorkerError(code) from None

    def _check(self):
        if self._owner != (os.getpid(), threading.get_ident()):
            self._fail("owner_mismatch")
        if self._failed:
            self._fail("terminal_failure")

    def _reserve(self, rank, **amounts):
        self._check()
        if self._reserving:
            self._fail("reentry")
        if (type(rank) is not int or rank != self.rank or not amounts
                or set(amounts) - {"checkpoint_read_bytes", "private_write_bytes"}
                or any(type(n) is not int or not 0 < n <= MAX_FILE_BYTES
                       for n in amounts.values())):
            self._fail("invalid_arguments")
        reads = amounts.get("checkpoint_read_bytes", 0)
        if self._reads + reads > self.limits["maximum_read_bytes"]:
            self._fail("read_cap_exceeded")
        self._reads += reads
        self._reserving = True
        try:
            result = self._reservation(rank, **amounts)
        except BaseException:
            self._fail("reserve_failed")
        finally:
            self._reserving = False
        self._check()
        if result is not None:
            self._fail("reserve_failed")

    @contextmanager
    def _operation(self):
        self._check()
        if self._busy or not _SERIALIZER_LOCK.acquire(blocking=False):
            self._fail("reentry")
        self._busy = True
        try:
            yield
            self._check()
        except BaseException:
            self._fail("operation_failed")
        finally:
            self._busy = False
            _SERIALIZER_LOCK.release()

    @contextmanager
    def _runtime(self):
        # No first ML import here: only an already-admitted caller may initialize it.
        torch = sys.modules.get("torch")
        if torch is None or not re.match(r"2\.(10|14)\.", getattr(torch, "__version__", "")):
            self._fail("runtime_rejected")
        import torch.distributed.checkpoint as dcp
        import torch.distributed.checkpoint.filesystem as filesystem
        import torch.distributed.checkpoint.logger as dcp_logging
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict,
            set_model_state_dict,
        )

        if self.world_size > 1:
            if (not torch.distributed.is_initialized()
                    or torch.distributed.get_world_size() != self.world_size
                    or torch.distributed.get_rank() != self.rank):
                self._fail("runtime_rejected")
        elif torch.distributed.is_initialized():
            if torch.distributed.get_world_size() != 1 or torch.distributed.get_rank() != 0:
                self._fail("runtime_rejected")
        loggers = [dcp_logging._dcp_logger]
        c10d = sys.modules.get("torch.distributed.c10d_logger")
        if c10d is not None and hasattr(c10d, "_c10d_logger"):
            loggers.append(c10d._c10d_logger)
        if any(not isinstance(logger, logging.Logger) for logger in loggers):
            self._fail("runtime_rejected")
        previous = [logger.disabled for logger in loggers]
        try:
            for logger in loggers:
                logger.disabled = True
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield torch, dcp, filesystem, get_model_state_dict, set_model_state_dict
        finally:
            for logger, disabled in zip(loggers, previous, strict=True):
                logger.disabled = disabled

    def _rank_zero(self, torch, action):
        result = None
        if self.rank == 0:
            try:
                result = {"ok": True, "value": action()}
            except BaseException:
                result = {"ok": False}
        values = [result]
        if self.world_size > 1:
            torch.distributed.broadcast_object_list(values, src=0)
        if not values[0]["ok"]:
            self._fail("collective_failed")
        return values[0]["value"]

    def _all_local(self, torch, action):
        try:
            result = {"ok": True, "value": action()}
        except BaseException:
            result = {"ok": False}
        values = [None] * self.world_size
        if self.world_size > 1:
            torch.distributed.all_gather_object(values, result)
        else:
            values[0] = result
        if not all(value["ok"] for value in values):
            self._fail("collective_failed")
        return [value["value"] for value in values]

    def save_checkpoint(self, parent_fd, *, model, optimizer, rng_state, training_state, step):
        self._stage = "runtime"
        with self._operation(), self._runtime() as runtime:
            torch, dcp, filesystem, get_state, _ = runtime
            _directory_info(parent_fd)
            if (type(step) is not int or not 0 <= step < 100_000_000
                    or type(training_state) is not dict):
                self._fail("invalid_arguments")
            checkpoint_name = f"checkpoint-{step:08d}"

            def prepare():
                try:
                    os.stat(checkpoint_name, dir_fd=parent_fd, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    self._fail("storage_rejected")
                staging = f".{checkpoint_name}.{uuid.uuid4().hex}.tmp"
                os.mkdir(staging, mode=0o700, dir_fd=parent_fd)
                with _child_directory(parent_fd, staging) as root:
                    os.mkdir("model", mode=0o700, dir_fd=root)
                return staging

            self._stage = "prepare"
            staging = self._rank_zero(torch, prepare)
            with _child_directory(parent_fd, staging) as root:
                with _child_directory(root, "model") as model_root:
                    self._stage = "writer_setup"
                    writer, backend = _dcp_writer(model_root, self, filesystem)
                    self._stage = "model_save"
                    dcp.save({"model": get_state(model)}, storage_writer=writer,
                             no_dist=self.world_size == 1)

                    def local_finish():
                        self._stage = "trainer_save"
                        private = _WriteDirectory(root, self, model_directory=False)
                        name = f"trainer-rank-{self.rank:05d}.pt"
                        destination = private.concat_path(OPAQUE_ROOT, name)
                        with private.create_stream(destination, "wb") as out:
                            torch.save({"optimizer": optimizer.state_dict(),
                                        "rng_state": rng_state}, out)
                        receipts = {"model/" + name: value
                                    for name, value in backend.receipts.items()}
                        receipts.update(private.receipts)
                        return receipts

                    all_receipts = self._all_local(torch, local_finish)

                def finish():
                    self._stage = "state_save"
                    expected = {}
                    for receipts in all_receipts:
                        if set(expected) & set(receipts):
                            self._fail("storage_rejected")
                        expected.update(receipts)
                    state = {**training_state, "step": step, "latest_checkpoint": checkpoint_name,
                             "status": "checkpoint"}
                    payload = (json.dumps(state, sort_keys=True, ensure_ascii=True,
                                          allow_nan=False, separators=(",", ":")) + "\n").encode()
                    if len(payload) > 65_536:
                        self._fail("storage_rejected")
                    private = _WriteDirectory(root, self, model_directory=False)
                    destination = private.concat_path(OPAQUE_ROOT, STATE_NAME)
                    with private.create_stream(destination, "wb") as out:
                        out.write(payload)
                    expected.update(private.receipts)
                    self._stage = "staging_verify"
                    fd, finalizer = os.dup(root), None
                    try:
                        finalizer = CheckpointStagingFinalizer(
                            fd, rank=self.rank, reserve=self._reserve, expected_files=expected,
                            maximum_file_bytes=self.limits["maximum_file_bytes"],
                            maximum_read_bytes=self.limits["maximum_read_bytes"],
                            maximum_manifest_write_bytes=65_536,
                            maximum_sync_operations=self.limits["maximum_sync_operations"])
                        receipt = finalizer.finish()
                    finally:
                        finalizer.close() if finalizer is not None else os.close(fd)
                    fd, publisher = os.dup(parent_fd), None
                    self._stage = "publication"
                    try:
                        publisher = CheckpointPublisher(
                            fd, rank=self.rank, reserve=self._reserve, staging_name=staging,
                            checkpoint_name=checkpoint_name, staging_receipt=receipt,
                            maximum_file_bytes=self.limits["maximum_file_bytes"],
                            maximum_read_bytes=self.limits["maximum_read_bytes"])
                        return {**publisher.publish(), "checkpoint_name": checkpoint_name}
                    finally:
                        publisher.close() if publisher is not None else os.close(fd)

                result = self._rank_zero(torch, finish)
                self._stage = "complete"
                return result

    def restore_checkpoint(self, parent_fd, *, checkpoint_name, manifest_sha256, model, optimizer):
        self._stage = "runtime"
        with self._operation(), self._runtime() as runtime:
            torch, dcp, filesystem, get_state, set_state = runtime
            _directory_info(parent_fd)
            if (type(checkpoint_name) is not str
                    or not re.fullmatch(r"checkpoint-[0-9]{8}", checkpoint_name, re.ASCII)):
                self._fail("invalid_arguments")
            with _child_directory(parent_fd, checkpoint_name) as root:
                self._stage = "checkpoint_verify"
                fd, session = os.dup(root), None
                try:
                    session = CheckpointReadSession(
                        fd, rank=self.rank, reserve=self._reserve,
                        expected_manifest_sha256=manifest_sha256,
                        maximum_file_bytes=self.limits["maximum_file_bytes"],
                        maximum_read_bytes=self.limits["maximum_read_bytes"])
                    self._all_local(torch, session.verify)
                    self._stage = "reader_setup"
                    model_state = get_state(model)
                    reader = _dcp_reader(session, self, filesystem)
                    self._stage = "model_restore"
                    dcp.load({"model": model_state}, storage_reader=reader,
                             no_dist=self.world_size == 1)
                    incompatible = set_state(model, model_state)
                    if incompatible.missing_keys or incompatible.unexpected_keys:
                        self._fail("storage_rejected")

                    def restore_trainer():
                        self._stage = "trainer_restore"
                        view = CheckpointReadView(
                            session, f"trainer-rank-{self.rank:05d}.pt",
                            maximum_read_call_bytes=self.limits["maximum_read_call_bytes"])
                        try:
                            trainer = torch.load(view, map_location="cpu", weights_only=True)
                        finally:
                            view.close()
                        if type(trainer) is not dict or set(trainer) != {"optimizer", "rng_state"}:
                            self._fail("storage_rejected")
                        optimizer.load_state_dict(trainer["optimizer"])
                        return trainer["rng_state"]

                    # RNG values stay rank-local; collective result carries only success.
                    rng = []
                    def local_restore():
                        rng.append(restore_trainer())
                        return True
                    self._all_local(torch, local_restore)
                    self._stage = "complete"
                    return rng[0]
                finally:
                    session.close() if session is not None else os.close(fd)
