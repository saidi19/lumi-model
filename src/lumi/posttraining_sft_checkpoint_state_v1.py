"""Read private training state from an exactly pinned, own-generated checkpoint.

This is a metered preflight for the existing checkpoint worker, not a restore
entrypoint or an authorization decision. The admitted caller owns the borrowed
immutable parent FD and validates state identity/counters before any model,
optimizer, or RNG restore. DCP metadata remains trusted-only: a matching hash
does not make pickle safe or authenticate its origin. No ML library is imported.

Verification and state rereads use the worker's existing cumulative reservation
ledger. No extra allowance is created. Failure poisons that worker and interrupts
its lifecycle once; only fixed exception arguments are safe to report, never
suppressed exception contexts or the returned private payload.
"""

from __future__ import annotations

import json
import os
import re

from .posttraining_sft_checkpoint_reader_v1 import CheckpointReadSession
from .posttraining_sft_checkpoint_view_v1 import CheckpointReadView
from .posttraining_sft_checkpoint_worker_v1 import (
    STATE_NAME,
    CheckpointWorker,
    CheckpointWorkerError,
    _child_directory,
    _directory_info,
)

MAX_STATE_BYTES = 65_536


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError from None
        result[key] = value
    return result


def _reject_constant(_value):
    raise ValueError from None


def _read_state(session, worker):
    view = CheckpointReadView(
        session, STATE_NAME,
        maximum_read_call_bytes=worker.limits["maximum_read_call_bytes"],
    )
    try:
        remaining = view.seek(0, os.SEEK_END)
        if not 0 < remaining <= MAX_STATE_BYTES:
            worker._fail("storage_rejected")
        view.seek(0)
        chunks = []
        while remaining:
            amount = min(remaining, worker.limits["maximum_read_call_bytes"])
            chunk = view.read(amount)
            if type(chunk) is not bytes or len(chunk) != amount:
                worker._fail("storage_rejected")
            chunks.append(chunk)
            remaining -= amount
        payload = b"".join(chunks)
    finally:
        view.close()
    state = json.loads(payload.decode("ascii"), object_pairs_hook=_unique_object,
                       parse_constant=_reject_constant)
    if type(state) is not dict:
        worker._fail("storage_rejected")
    canonical = (json.dumps(state, sort_keys=True, ensure_ascii=True, allow_nan=False,
                            separators=(",", ":")) + "\n").encode("ascii")
    if canonical != payload:
        worker._fail("storage_rejected")
    return state


def read_training_state(parent_fd, *, checkpoint_name, manifest_sha256, checkpoint_worker):
    """Return private canonical JSON only; all byte reservations stay cumulative.

    The parent FD is borrowed and never closed. Child/session descriptors are
    closed exactly once, including constructor failures. The worker's existing
    operation scope provides owner/reentry checks and terminal error semantics.
    """
    if type(checkpoint_worker) is not CheckpointWorker:
        raise CheckpointWorkerError("invalid_arguments") from None
    worker = checkpoint_worker
    with worker._operation():
        worker._stage = "checkpoint_verify"
        try:
            if (type(checkpoint_name) is not str
                    or not re.fullmatch(r"checkpoint-[0-9]{8}", checkpoint_name, re.ASCII)
                    or type(manifest_sha256) is not str
                    or not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256, re.ASCII)):
                worker._fail("invalid_arguments")
            parent_identity = _directory_info(parent_fd)
            with _child_directory(parent_fd, checkpoint_name) as root:
                fd, session = os.dup(root), None
                try:
                    session = CheckpointReadSession(
                        fd, rank=worker.rank, reserve=worker._reserve,
                        expected_manifest_sha256=manifest_sha256,
                        maximum_file_bytes=worker.limits["maximum_file_bytes"],
                        maximum_read_bytes=worker.limits["maximum_read_bytes"],
                    )
                    session.verify()
                    state = _read_state(session, worker)
                finally:
                    session.close() if session is not None else os.close(fd)
            if _directory_info(parent_fd) != parent_identity:
                worker._fail("descriptor_changed")
            worker._stage = "complete"
            return state
        except CheckpointWorkerError as error:
            worker._fail(error.code)
        except BaseException:
            worker._fail("storage_rejected")
