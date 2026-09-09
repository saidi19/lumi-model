# SFT execution status v0.62

Implementation handoff, 2026-09-09. This is the direct successor of
`a0296d4dd80463023a154b5eb51bd4384158f513`. It composes the accepted CPU software
with the real Linux guardian and generated WorkerRun. It does not start an H100,
read the pretrained S3 checkpoint, use real training data, seal a dataset, or
grant real-SFT/production eligibility. Native acceptance is **not yet claimed**
by this source record.

## Actual composition implemented

The accepted artifact remains available: artifact `10101633554`, accepted run
`34345663337`, source `c4272e36bc98354067dd23a83931a3db58823d76`. Its current
authenticated GitHub metadata was checked again before publication: not expired,
same name, ZIP byte count and digest. No large artifact was downloaded on the Mac.

The new bounded acquisition makes three HTTPS GET attempts without retry:
metadata, a non-followed archive redirect, and the signed storage URL without
the GitHub token. Proxies and further redirects are disabled. It validates the
1,062,113,436-byte ZIP against
`98d395f35af2b23874b6d5d7a56a13c969fd72fa237065fec74d81634927ac1c`, then extracts
only the 1,062,113,280-byte software tar against
`99d21fc60a54a65f28ceb4780f603f7ee48e49ac7b6d00a053dcdabe9ebc9ff4`.
It does not rebuild, pull, publish, or acquire a model/data archive.

The root fixture copies and rehashes that tar into a fresh root-owned directory.
A successful Docker inventory must prove the exact image and unique container
name absent before ownership is assumed. Docker loads the accepted software,
creates one never-started fixed-profile container, verifies it with the existing
admission module, exports it under a byte cap, and verifies the same created
state again. The container is retired without ever being started. Ambiguous
daemon writes/retirements cannot be retried or reported as confirmed cleanup.

The export is hashed before bounded dirfd/no-follow extraction. File contents
and valid symlink targets are preserved; ownership/set-id/directory metadata is
normalized explicitly, not represented as byte-identical metadata. Only two
new, separately pinned root-owned `0444` files are installed:

- `/opt/lumi-release/guardian/worker_entry.py`.
- `/opt/lumi-release/guardian/admission.json`.

The privileged hierarchy keeps the frozen controller/pidfd/subreaper lifecycle.
The actual blocked workload attaches to the independently owned cgroup before
network/mount/Landlock/IPC setup. The mounted root is the exported image, not host
`/usr`, `/etc`, Python or source substitutions. It is read-only, nosuid/nodev;
scratch, shared memory, minimal devices and procfs are fresh kernel overlays.
All root identities/groups/capabilities are dropped and authority descriptors
closed before the guardian releases the accepted CPython executable.

The old Landlock names-only root rule authenticated a specific 1 MiB synthetic
tmpfs. Reusing it for an OCI bind root would deterministically reject. A separate
OCI-specific successor instead authenticates the exact exported root's
device/inode/type/owner/group and read-only flags, while granting only `READ_DIR`
at root. The old helper and historical tests are unchanged.

The worker bridge separately checks its own bytes and control before ML, then
wraps the unchanged accepted probe/fixture. The real accepted software identity
replaces the fixture's placeholder at WorkerRun construction. Existing plan
hashing, collective agreement, checkpoint/resume checks, source inventories,
module/native origin checks and owner lifecycle remain active. Each of six rank
workers contributes pre-ML, constructor, admission and controller-agreement
proofs. The one extra adapter origin is explicitly separately bound, never
silently inserted into the old image's accepted inventory.

## One native attempt, bounded evidence

The workflow runs only on the first private direct-successor push whose `before`
is `a0296d4dd80463023a154b5eb51bd4384158f513` and whose `after` equals its checkout
SHA. Reruns and future ordinary source edits skip it. The preparer independently
checks the parent, checkout, event, branch and attempt before network activity.
The three old native workflows remain byte-for-byte unchanged and continue to
skip successors.

Native cases are sequential: wrong control pin, wrong adapter pin, then one
positive generated WorkerRun. Each rejection must exit normally with code2,
the exact expected fixed rejection code, no ML import and zero worker admission
counts. The positive case must validate the full existing three-invocation,
six-rank, eight-rank-step/four-global-update and resume-equality observer report.
The first unexpected failure stops the sequence; no native case is retried.

Bounds and scope:

- Acquisition logical deadline300s, per-request timeout at most30s; external
  timeout6m plus15s termination allowance. This is not a claim that an already
  blocked I/O syscall returns within300s.
- Native logical budget900s; each workload uses the frozen220s deadline with
  independently bounded guardian cleanup. External timeout17m plus30s allowance;
  complete hosted job28m. Kernel-uninterruptible work is not claimed strictly
  preemptible by userspace deadlines.
- Export at most2GiB, extracted file content at most4GiB, at most65,536 physical
  tar headers/owned entries, individual extracted file at most1GiB; minimum12GiB
  free host storage before root copy. These are logical file-content limits,
  not a claim of exact filesystem metadata or Docker-cache disk quotas.
- Actual workload cgroup:4GiB memory, zero swap,256pids and two CPU equivalents.
  Writable tmpfs:512MiB scratch plus64MiB shared memory; no GPU devices.
- Worker report at most32KiB; validated guardian receipt below the frozen4KiB
  channel cap. Uploaded evidence consists only of small aggregate JSON reports.
  No raw native logs, dataset text, credentials, signed URLs or large archive.
- Private cleanup removes only identity-bound objects created by this attempt.
  Rootfs deletion waits for all owned/adopted descendants to be reaped. The
  downloaded software copy is retired separately by exact file/parent identities.

The trusted ephemeral CI host and exclusive Docker daemon remain premises.
Complete dynamic code closure, publisher authenticity, hostile-root resistance,
guardian-death containment, private-integration acceptance and full-CI success
are not inferred from this composition.

## Source verification and handoff

Before publication, **292 new source/synthetic/mock tests passed with zero
failures and zero skips**, using stdlib-only `-S -B` invocation. Exact Ruff
`E,F,I,B,UP`, line100, targetpy311 passes on every new Python file. No Linux
mount/cgroup/native fixture or ML workload was executed on the Mac.
The combined run with70 existing admission/transport/source regression tests
also passed: **362 tests, zero failures/skips**. The host additionally compares
reported adapter/control pins with its independently computed expected hashes.

Independent implementation reviews caught and corrected the old-root Landlock
assumption, false atime change detection, untracked partial copies, ambiguous
Docker absence/ownership, launch-after-deadline, rootfs lifetime mismatch and
descendant cleanup ordering before any native workflow was published.

All changes are new versioned files; existing source, prior artifacts, historical
markers and evidence remain unchanged. The v0.61 result-only receipt remains
local and unstaged. Publish only this exact implementation/test/workflow/status
set, then read the one native result together with normal Source/Spark results.
Do not manufacture success by rerunning a failed case, rebuilding the image or
relaxing its acceptance contract.

Gate5's full guardian/OCI composition is pending that native result. Dedicated
host/S3/data binding, approved H100/CUDA validation and actual SFT are still
downstream; this source record does not mark them complete.
