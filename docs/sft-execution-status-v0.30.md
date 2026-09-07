# SFT execution status v0.30

Atomic no-replace checkpoint publication, 2026-09-07. The user's Mac restriction
continues: source editing, static lint and tiny stdlib synthetic-file tests only.
No ML import, model/data access, training, conversion, large download, GPU, SSH,
S3 restore, instance management or cloud-account action is performed locally.

## Prior completion and exact scope

The preceding saver/staging commit is
`4ad5e53191c5c638527025a2a17102a168076aa1`. Its exact
[Source CI](https://github.com/saidi19/lumi-model/actions/runs/34109988958) and
[Spark CI](https://github.com/saidi19/lumi-model/actions/runs/34109989042) passed.
The Source summary accounted for 3780 selected tests and 327 private integrations
NOT_EXECUTED, with private/full-CI claims false. Spark ran three synthetic tests,
zero skips, in 106.62 seconds. The local tiny regression suite passed 237 tests.

Only three new files are added in this successor:

- `src/lumi/posttraining_sft_checkpoint_publication_v1.py`
- `tests/test_sft_checkpoint_publication_v1.py`
- This status document.

Earlier versions, evidence, tests, plans, workflows and private inventories remain
byte-for-byte unchanged. Publishing a synthetic checkpoint directory is not
dataset sealing, registry mutation, production promotion or actual training.

## Implemented publication boundary

`CheckpointPublisher` owns an already-authorized private parent directory FD
after successful validation. The caller supplies the frozen worker's exact
checkpoint name, its hidden UUID-named staging sibling, a copied v1 finalizer
receipt, explicit byte caps and one shared-controller reservation callback. The
caller must have finished/closed every staging writer and finalizer and established
its rank barrier. A receipt dictionary is a binding, not independent proof of
admission, file durability or synchronization between ranks.

The publisher validates the private parent/staging descriptors, same filesystem,
bounded parent inventory and absent destination. It fully verifies staging using
the pinned metered reader and matches payload count/bytes against the finalizer
receipt. It checks the observed inventory again, attempts exactly one same-parent
native no-replace rename, observes the precise name/inode transition, syncs the
parent once, then fully verifies the destination through a fresh metered reader.
The before/after payload and descendant-directory metadata must match; only the
moved root's rename-related ctime and the parent's own rename-related metadata
are allowed to change. Unrelated sibling metadata must remain unchanged.

Existing destination directories (including empty ones), files and symlinks must
not be overwritten. The operation uses Linux `renameat2(RENAME_NOREPLACE)` or
Darwin `renameatx_np(RENAME_EXCL)`, resolved only from the current process's libc.
The [Linux API documentation](https://man7.org/linux/man-pages/man2/rename.2.html)
distinguishes no-replace semantics from ordinary rename; the flag value is bound
to the [kernel UAPI](https://github.com/torvalds/linux/blob/master/include/uapi/linux/fs.h).
Darwin's installed SDK `sys/stdio.h` declares `renameatx_np` and flag `0x4`.
Missing OS/libc/filesystem support stops without an ordinary rename, replace,
copy or deletion fallback. Native filesystem support is a real target prerequisite.

For payload bytes P and manifest bytes M, a successful publication charges
2*(P+M) additional `checkpoint_read_bytes`, including full verification both
before and after rename. The complete expected read cost is checked before the
first content read; every actual chunk also requires the shared reservation.
There are zero content writes, one rename attempt and one targeted parent-sync
attempt. Namespace/journal writes are not zero physical disk traffic: byte caps
measure logical content, while namespace/sync attempts are counted separately.
Earlier payload/finalizer writes and syncs remain separate shared-run costs.

At most 1024 parent entries are inspected via bounded metadata traversal. Sibling
contents are never read. The staging tree retains all frozen v1 reader bounds,
private modes, canonical paths, exact inventory and SHA-256 checks. Descriptor
ownership, process/thread/reentry checks and nonrefundable failure semantics are
preserved. Underlying Python exception contexts/tracebacks can retain private
data and must not be logged; public error args/repr and `state()` are fixed-code
or aggregate-only.

## Failures are not automatically reversible

The native namespace change, parent sync and post-verification are distinct
operations. A failed or interrupted call after rename was attempted can leave
the destination visible without a successful publication receipt. No automatic
retry, rollback, overwrite, deletion or checkpoint pruning occurs.

`state()` distinguishes ready, verifying_staging, rename_attempted, renamed and
complete, alongside failure, actual attempt counts and read charges. It does not
authorize recovery. A rename-attempted error can have an uncertain outcome; a
renamed error means the native call reported success but later checks did not
finish. Only a successful final receipt reports `published=true`, always with
`production_eligible=false`. The shared lifecycle must stop and ranks retire on
failure; finding a directory at the destination is not sufficient to resume it.

The trusted parent must remain exclusively controlled. Native name-based rename
cannot atomically prove that the source inode equals a previous observation;
postchecks detect observed changes but cannot undo an unauthorized concurrent
mutation. This is not a hostile-filesystem lock, an OS resource quota or protection
against every same-inode FD reuse. Target-local crash durability and execution
enforcement require their separate acceptance work.

## Verification and remaining integration

The independently authored 43 new tests plus 237 existing reader/saver/metered-I/O/
Supervisor/protocol tests passed together: 280 tests, zero failures/errors/skips,
in 0.155 seconds. A fresh `python -S -B` process confirmed no Torch, Transformers,
NumPy, safetensors or tokenizers imports. Isolated Ruff on both new Python files
passed. Tiny tests exercise real Darwin native publication and late destination
collisions, fixed Linux/Darwin ABI/flag bindings, constructor/FD ownership,
ambiguous rename/sync failures, metered post-verification and evidence retention.
Linux native execution is left to remote CI, not claimed from mocked ABI tests.

Two joining tests use the real in-process Supervisor: success charges exactly
2*(P+M) reads and zero content writes; a shared cap expiring after rename fails the
same controller and blocks another rank's independent reader before its content
syscall. This does not claim multiprocess IPC or real checkpoint acceptance.
This successor's exact GitHub CI results remain to be observed after publication.

The storage sequence now has metered byte streams, verified staging and an
exclusive publication operation. The next connection remains the versioned actual
training worker: admitted serializer/DCP adapters, every real source/resume/save
I/O path and shared optimizer transitions. Unmodified DCP must not be given these
objects if it can expose native I/O or invoke host-wide sync. Linux resource
enforcement, real-size CPU import/data relocation and the complete runtime release
remain the other pre-GPU deliverables. No new roadmap gate is introduced and no
H100 rental is needed for this source-engineering step.
