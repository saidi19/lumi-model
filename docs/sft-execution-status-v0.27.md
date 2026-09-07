# SFT execution status v0.27

Lightweight source-engineering checkpoint, 2026-09-07. The user's Mac must not
run training, model conversion, large downloads or heavy tests. This successor
uses only source inspection/editing, static lint and stdlib tests over tiny
new synthetic files. No Torch import, optimizer step, model/dataset read, GPU,
SSH, cloud-account operation or S3 restore is part of its local tests. H100 is
still a proposed hardware family, not an observed target or rental request.

## Previous step and exact scope

Commit `0bf4a4f441fffcfa7ed80f9b58fef2853f101cf6` published the v0.26 tiny
CPU training/resume integration. Its exact Source and Spark CI passed in the
preceding turn. That completed CPU fixture is not permission to repeat ML work
on the Mac after the user's newer restriction, or to run a real GPU worker.

Only these three new files are added here:

- `src/lumi/posttraining_sft_metered_io_v1.py`
- `tests/test_sft_metered_io_v1.py`
- This checkpoint.

All earlier worker/runtime implementations, tests, plans, contracts, evidence,
CI workflows and private inventories remain unchanged. The new primitive is
not inserted into a frozen worker by monkeypatching `open` or permitting an old
entrypoint. It does not grant admission or complete the GPU launcher.

## Implemented: debit bytes before the operation

`MeteredReader` and `MeteredWriter` take ownership of an already-open private,
unique regular file descriptor only after successful validation. They accept no
path and never open a model or dataset. Descriptor access mode, private mode,
owner, close-on-exec and observed identity are checked. The trusted caller must
already have authorized and safely opened the exact file; a descriptor and a
callable are not evidence of owner approval.

Every nonempty read/write calls the supplied `reserve(rank, **amounts)` before
performing I/O. That callback must synchronously complete the shared ledger's
reservation/acknowledgement and return exactly `None`. A local dictionary or a
separate Supervisor copy in each rank does not implement a shared ledger.

Reads are bounded to at most 1 MiB per call and to the captured file extent.
Seeking back and reading again reserves the bytes again. Known EOF needs no
content syscall; an unexpected short read fails. The reader rechecks metadata
before and after reading, including after the reservation callback. `readinto`
uses only a bounded bytearray. Unbounded reads and out-of-range seeks reject.

Writers require a fresh empty file and append at their internal cursor, with a
bounded per-file limit and at most 1 MiB per call. They neither overwrite existing
content nor retry short/failed writes. A `log_bytes` reservation is sent alone:
the frozen Supervisor already debits those bytes from private writes as well.

Reservations are never refunded. A failure makes that handle unusable except
for explicit cleanup; the future worker must also interrupt the shared lifecycle
and retire its ranks. Handles are bound to the owning process/thread and reject
reentry. Cleanup revokes ownership before close, never retries an ambiguous
close and refuses to close an observed different file at a reused FD number.
Public error arguments and representations contain fixed codes, not paths,
payloads or callback text. Python may retain a suppressed underlying exception
context; the future worker must never log that context or traceback.

These are cooperative logical-byte controls, not physical-disk metering,
filesystem locking, authentication, a cgroup, a wall-clock killer or protection
against arbitrary code bypassing the wrapper. Filesystem metadata operations,
directory creation, rename, fsync and system-level storage effects are outside
the byte counter. The caller must reserve those separately where applicable.
Snapshot checks also cannot prove exclusive open-file-description ownership or
detect every replacement by a new descriptor referring to the same inode.

## Why an outer counter is insufficient

Read-only review located the following operations in the frozen worker. Their
costs must be accounted for separately, not reduced to unique artifact size:

| Worker stage | Operations requiring the later integrated adapter |
| --- | --- |
| Source load | Every rank verifies the complete source checksum tree, rereads config, then opens model weights again. Encoded manifest/index/token/label reads have separate byte and row counts. |
| Checkpoint save | DCP model shards, per-rank optimizer/RNG serialization and JSON writes; checksum creation rereads payloads, and post-publication verification reads them again. |
| Resume | Rank-zero checkpoint discovery verifies candidates; each rank verifies the selected checkpoint and loads DCP plus optimizer/RNG state again. |
| Training step | Reserve one global optimizer step before its work, not once per rank; complete only after every rank succeeds. |
| Logs and terminal work | Count encoded bytes and reconcile atomic state/metrics writes. Pruning and an unbudgeted final digest must not silently run. |

The primitive is deliberately not advertised as a drop-in Torch/DCP file object.
The inspected serialization paths need operations it does not supply: bounded
`readline`, slice reads, synchronization and controlled storage publication.
Exposing `fileno()` can enable native reads that bypass reservations. The
inspected DCP writer can fall back to host-wide `os.sync()` without `fileno()`;
simply substituting this object is therefore unsafe. HF/safetensors native path
APIs need their own bounded adapter, not a full-4B bytes buffer in RAM.

## Verification and next handoff

The 40 new tests and 94 existing Supervisor/protocol tests passed together:
134 tests, zero failures/errors/skips, in 0.022 seconds. Exact isolated Ruff on
both new Python files passed. Tests use only short synthetic byte strings in
new temporary files. The existing Supervisor is exercised directly for
shared-rank budget exhaustion and to verify logs are not double charged. This
is an in-process shared-controller test, not a multiprocess transport claim.
No ML test is run on the Mac. No future GitHub result is preclaimed.

The first tiny-file tests caught a Darwin-specific flag transition after a
successful write. The local public Kernel SDK header defines `FWASWRITTEN`
(`0x00010000`) as a descriptor-written flag. Only that bit's absent-to-present
transition immediately after our own successful write is accepted on Darwin.
Other flags, access mode, FD_CLOEXEC and file identity remain exact; pre-operation
and post-reservation checks are unchanged. Deterministic tests reject the same
transition on Linux, a reverse transition and other flag changes. No general
flag masking or historical source change was used.

The next substantive connection remains the versioned storage/worker adapter:
route every content read/write/hash and global optimizer transition through the
shared controller, preserve checkpoint semantics, and verify it on a separate
CPU test host/CI. Linux resource enforcement, real-size CPU import/relocation and
the complete runtime release remain the other pre-GPU deliverables from v0.26.
GPU-only acceptance follows on the dedicated H100 after exact host/resources and
execution scope are fixed. This checkpoint adds no roadmap gate and closes no
production clearance. `production_eligible=false` remains.
