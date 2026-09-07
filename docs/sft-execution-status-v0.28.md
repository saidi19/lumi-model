# SFT execution status v0.28

Lightweight checkpoint-reader integration, 2026-09-07. The Mac restriction is
unchanged: source editing, static lint, and tiny stdlib synthetic-file tests only.
No model/data read, ML import, deserialization, training, conversion, large
download, GPU, SSH, S3 restore or cloud-account action occurs locally. Remote
source CI is verification, not authorization for a real training run.

## Previous evidence and bounded successor

The preceding metered-I/O commit was
`c3a59824d950b68b4c3c5b2392c213afdf8b4ad1`. Its exact Source CI
[34103410649](https://github.com/saidi19/lumi-model/actions/runs/34103410649)
and Spark CI
[34103410650](https://github.com/saidi19/lumi-model/actions/runs/34103410650)
passed. The Source main suite reported 3619 passed, 58 skipped and 329 deselected;
327 private tests remained NOT_EXECUTED. Those exclusions are not new passes or
production clearance.

Only three successor files are added:

- `src/lumi/posttraining_sft_checkpoint_reader_v1.py`
- `tests/test_sft_checkpoint_reader_v1.py`
- This status checkpoint.

All prior source, worker/runtime versions, tests, contracts, CI workflows, private
inventories and evidence remain byte-for-byte unchanged. No old entrypoint is
enabled and no global `open` monkeypatch is inserted into a worker.

## Implemented: checksum verification and checkpoint slice reads

`CheckpointReadSession` owns an already-authorized private directory descriptor
after successful validation. It accepts a mandatory caller-pinned SHA-256 for
`manifest.sha256`, a per-payload size cap, a cumulative read cap, and a trusted
shared-controller reservation callback. Neither a descriptor nor a hash proves
owner approval. The caller still supplies admission and shared IPC semantics.

One `verify()` enumerates the bounded private tree, reads the manifest through
`MeteredReader`, verifies its external pin before using its entries, checks exact
file inventory and streams every declared payload through the same metered
reader to verify its SHA-256. It rechecks the complete metadata inventory before
returning a fixed aggregate receipt. A subsequent `read_file(name, offset, size)`
can return only a declared payload slice of at most 1 MiB within the verified
extent. Its bytes are charged again. The manifest cannot be read through this
payload API; there is no unbounded read, file mapping, raw FD export or loader
path handoff. Empty payloads and known empty slices need no content syscall.

Every nonempty manifest/hash/slice read reserves `checkpoint_read_bytes` before
the content syscall. The session cap also applies to verification plus rereads,
not only unique artifact size. Reservations are not refunded after callback,
checksum or read failure. Every failure is sticky; repeated verification is
rejected and cannot silently restart a scan. The owning process/thread and
reentry guards match the cooperative primitive. The caller must interrupt the
shared lifecycle and retire all ranks on failure; this session cannot retire
other processes itself.

The format matches the frozen checksum producer's lowercase digest, two-space
separator and LF records. This version deliberately accepts only canonical ASCII
relative names: no traversal, absolute paths, duplicate entries, self-manifest
entries, symlinks, hardlinks or special files. Payloads must be private 0600,
directories private 0700, all owned by the current user. The exact inventory may
contain bounded empty private directories, as the prior verifier allowed. Empty
payload files are valid. It imposes no plain-string sort requirement: the frozen
producer sorts `pathlib` paths, whose ordering must not be substituted silently.

Structural bounds are separate from caller resource caps: manifest at most
65,536 bytes, 256 payload files, 2048 total tree entries, eight path components,
128 ASCII characters per component and 256 per relative path. Descriptor-relative
opens use no-follow, close-on-exec and nonblocking flags. Metadata is checked
before and after opening, around reads, and along each opened directory chain.
Intermediate descriptors are explicitly closed on success or failure. Root
ownership is revoked before close, with no retry of an ambiguous close or close
of an observed different directory at a reused descriptor number.

These are observed metadata checks, not an atomic filesystem lock or protection
from every hostile race/same-inode FD reuse. The trusted caller must keep the
artifact immutable and retain exclusive FD ownership. Post-verification slice
reads validate their requested file and directory chain, not a fresh full-tree
hash on every slice. Directory/stat/open/close calls are structurally bounded,
but are not content-byte charges or physical I/O measurements. Fixed error args
and repr contain no private names or bytes; suppressed exception contexts and
tracebacks may retain private data and must never be logged.

## Local verification and remaining integration

The 47 new tests plus the 134 existing metered-I/O, Supervisor and ledger-protocol
tests passed together: 181 tests, no failures/errors/skips, 0.060 seconds. A fresh
`python -S -B` process confirmed no Torch, Transformers, NumPy, safetensors or
tokenizers modules were loaded. Fixtures contain only short synthetic strings,
including explicitly non-pickle bytes. Tests cover exact pins and hashes,
unsafe/inexact inventory, bounds, repeat charging, shared two-rank Supervisor
exhaustion, mutation, reentry, failed/short I/O, cleanup and FD ownership. Isolated
Ruff on the two new Python files is the static check. Exact successor GitHub CI
results must be observed separately; this document does not preclaim them.

This completes a real metered checksum-and-slice reader, not Torch/DCP
deserialization or the full worker connection. The next storage connection must
handle controlled serialization/saving, synchronization and checkpoint publication
without exposing native-I/O bypasses or DCP's host-wide sync fallback. Then the
versioned worker must route all actual source/resume/save reads and writes plus
global optimizer transitions through the admitted shared lifecycle. That remains
the same worker-integration deliverable, not an additional roadmap gate.

The other pre-GPU deliverables remain the training-capable Linux resource
enforcement, real-size CPU import/data relocation successor and complete runtime
release/run/stop verification from v0.26. No H100 rental is needed for this source
step. Production rights/owner review remain separate, and
`production_eligible=false`. No roadmap gate or actual SFT run is newly complete.
