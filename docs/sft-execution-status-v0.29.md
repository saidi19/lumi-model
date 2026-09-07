# SFT execution status v0.29

Checkpoint stream and private staging finalization, 2026-09-07. Local work is
restricted to source editing, lint and tiny stdlib synthetic-file tests. No ML
import, model/data access, training, conversion, large download, GPU, SSH, S3
restore or cloud-account action is authorized or performed on the Mac.

## Scope and preceding evidence

The preceding checkpoint-reader commit is
`669c99a72710f926525b783edcd6eda875cab611`. Its exact
[Source CI](https://github.com/saidi19/lumi-model/actions/runs/34106762999) and
[Spark CI](https://github.com/saidi19/lumi-model/actions/runs/34106762940) completed
successfully. Detailed logs were not retrieved at that turn's credit limit; no
unobserved main-suite counts are asserted here. Its local tiny suite passed 181
tests with no ML imports.

Only three new successor files belong to this step:

- `src/lumi/posttraining_sft_checkpoint_saver_v1.py`
- `tests/test_sft_checkpoint_saver_v1.py`
- This status document.

All earlier source, evidence, plans, tests, workflows and private inventories
remain unchanged. The saver explicitly composes the frozen v1 reader and metered
I/O internals; targeted checked fsync is their only FD use outside those helpers.
No raw descriptor or native loader path is exported to a serializer.

## Implemented connection

`CheckpointWriteStream` takes ownership of an already-authorized, uniquely owned
empty 0600 write-only file FD after constructor validation. It supplies
write/tell/flush/close and a post-close byte-count/SHA-256 receipt. Its write API
accepts exact bytes, bytearray or one-dimensional contiguous byte memoryviews.
It checks the complete call and remaining per-file caps before writing any of
that call, then makes at most 1 MiB immutable chunks and reserves each chunk's
`private_write_bytes` before the actual write. It does not allocate a full-model
buffer or account for a serializer's pre-existing tensor/buffer allocation.

Flush is targeted file fsync, not host-wide sync. Unchanged repeated flushes are
coalesced; initially empty files still receive one sync. Sync attempts have a
separate cap and are not retried after failure. Successful close syncs any dirty
position and revokes the handle. Receipt creation requires a successful close;
failure cleanup cannot manufacture a valid receipt.

`CheckpointStagingFinalizer` starts only AFTER every payload writer has closed
and the trusted caller has established its rank barrier. It takes a private 0700
directory FD containing those newly written payloads and copies the exact private
filename-to-receipt mapping. It does not create that barrier or independently
prove that prior writers used a controller. The callback must be one shared
controller/acknowledged IPC reservation, not a separate ledger in every rank.

Finalization checks exact safe inventory, streams metered payload hash readbacks
and compares them with the writer receipts, then syncs the payload files. It
creates `manifest.sha256` exclusively with 0600 permissions, meters its write and
syncs it. A fresh pinned `CheckpointReadSession` then verifies the manifest and
all payloads. Directories are synced children-before-parents, including the root,
and the complete metadata inventory is checked again before a fixed aggregate
`verified_staging` receipt can be returned. No payload is deserialized.

For payload size P and manifest size M, earlier payload streams charge P writes;
the finalizer charges M writes and 2P+M reads. These are logical content bytes,
not unique artifact size, disk traffic or an OS I/O quota. The finalizer's sync
cap covers each payload, one manifest sync and every directory. Earlier stream
sync caps are separate; a future worker must compose all resource limits.

The canonical manifest keeps frozen Path ordering, digest/two-space/LF format,
the v1 reader's path grammar, 256-payload and 64 KiB manifest bounds, and support
for empty payloads/empty private directories. Prior manifests, undeclared files,
unsafe metadata, wrong receipts, short I/O and quota failures stop terminally.
No reservation is refunded and no existing file is overwritten or deleted.
Partial new staging files remain as failure evidence. The caller must interrupt
the shared lifecycle and retire its ranks; the module cannot do that itself.

The tiny APFS tests observed a +1 directory link-count transition when creating
the new regular manifest. Only immediately after our own exclusive create is
that +1 accepted on Darwin; stable device/inode/mode/owner must match. Linux does
not receive this exception. Exact final inventory and callback mutation checks
remain in force. This is not a blanket metadata refresh, filesystem lock or
protection against every hostile race/same-inode descriptor reuse.

Only fixed error args/repr are safe to report. Suppressed exception contexts and
tracebacks may retain private data and must never be logged.

## Verification and next boundary

The 56 new tests and 181 existing reader/metered-I/O/Supervisor/protocol tests
passed together: 237 tests, no failures/errors/skips, 0.093 seconds. A fresh
`python -S -B` process confirmed no Torch, Transformers, NumPy, safetensors or
tokenizers modules loaded. Exact isolated Ruff on both new Python files passed.
All fixtures contain only tiny synthetic byte strings. An independently authored
test joins actual streams/receipts/finalizer with one real in-process Supervisor;
another exhausts that ledger and verifies later separate writers/readers cannot
issue content syscalls. This is not a multiprocess-IPC or real-model claim. This
successor's exact GitHub CI results remain to be observed after publication.

This completes metered serialization-facing streams and verified private staging,
not atomic checkpoint publication or full distributed worker admission. There is
no rename/replace/pruning, promotion, dataset seal, production clearance or real
SFT execution. `published=false` and `production_eligible=false` remain.

Unmodified Torch DCP is not a drop-in consumer: its writer expects additional
storage operations and can fall back to host-wide `os.sync()` when fileno is
absent. The next worker/storage connection must provide controlled no-replace
publication and integrate the admitted serializer/distributed lifecycle without
native-I/O bypasses. That is still the existing worker-integration deliverable,
not a new roadmap gate. Linux resource enforcement, real-size CPU import/data
relocation and the complete runtime release remain the other pre-GPU deliverables.
No H100 rental is needed for this source-engineering step.
