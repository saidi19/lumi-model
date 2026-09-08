# SFT execution status v0.37

Filesystem-view and content-policy successor, 2026-09-08. This step composes
private mount/root isolation and mandatory Landlock with the accepted cgroup,
loopback namespace and generated WorkerRun. It does not authorize real model,
dataset, GPU, S3, training, host admission or production access.

## Accepted predecessor and exact scope

Predecessor `a05200a12d5d7fd88573767b41aa26b2d18223ab` passed all four jobs in
[Source CI](https://github.com/saidi19/lumi-model/actions/runs/34225149633):
4498 passed, 58 skipped, 329 deselected, 36 warnings and 8466 subtests in
284.18 seconds. Source selection was 4556 tests. The 327 private integrations
remained NOT_EXECUTED; two historical failures were routed separately and
private/full-CI claims remained false. [Spark CI](https://github.com/saidi19/lumi-model/actions/runs/34225149463)
passed three tests with zero skips in 99.60 seconds.

This successor is limited to seven new files:

- `src/lumi/posttraining_sft_mount_root_v1.py`
- `tests/test_sft_mount_root_v1.py`
- `src/lumi/posttraining_sft_landlock_v1.py`
- `tests/test_sft_landlock_v1.py`
- `scripts/run_sft_filesystem_fixture_v1.py`
- `tests/test_sft_filesystem_linux_v1.py`
- This document.

Prior tracked files, evidence, workflows, policy floors and budgets remain
unchanged. Unrelated local files are excluded. Local work is only source edits,
small stdlib syscall-double tests, exact Ruff and AST parsing. Neither Linux
fixture nor Linux test wrapper is executed on the Mac, including skip paths.

The composed network fixture remains exactly 16067 bytes with SHA-256
`70400a1428d0e54834721a5163db7724940a271bbeeeaec50714f28108f20bfc`.
The nested cgroup fixture remains bound to
`f91ed0c353a20c309ff3e1bfaab8dbe990b28c36626fdeeb0ac5131e0cf75fad`;
the generated WorkerRun remains bound to
`53c50c92965ed1cef0c9624c57367c31ed7ebfb9d7cfb34257188ad179d333d0`.

## Filesystem contract

The root bootstrap is attached to its owned workload cgroup and fresh loopback
namespace before filesystem setup. A one-shot private mount namespace removes
mount propagation, builds a small tmpfs root, pins the selected software paths,
and requires recursive read-only mount attributes for each exposed bind. It
pivots into that root, changes the current directory, detaches/removes the old
root entry, closes its owned descriptors and seals the root read-only.

The exposed read-only paths are the fixed system software tree, selected CI
Python runtime, repository source directory, exact pinned generated WorkerRun
script, a small set of system configuration files and procfs needed for the
accepted owned-rank RSS/status checks. There is no general home, host `/tmp`,
credentials, dataset, sysfs or cgroup path mount. Device access is limited to
new null/zero/random/urandom nodes and descriptor aliases. Private `/tmp` has an
explicit tmpfs size/inode cap; `/dev/shm` uses a directory within that capacity.
The guardian remains outside the workload mount namespace and cgroup limits.

Mandatory Landlock ABI 5 or newer adds filesystem content-access rules and an
implicit ptrace-domain boundary. Unsupported kernels fail, without reducing
handled rights. Read-only software/proc paths, the writable temporary tree and
the four device nodes have separate roles. Setup is one-shot and owned ruleset
and path descriptors close once. No-new-privileges is set before restriction;
the accepted UID/GID/groups/capability drop and inherited-FD closure still occur
before release and ML work. A failed setup must exit, not retry or restore the
host namespace.

The [pivot-root contract](https://man7.org/linux/man-pages/man2/pivot_root.2.html)
and [recursive mount attributes](https://man7.org/linux/man-pages/man2/mount_setattr.2.html)
support the namespace/read-only boundary. Landlock is complementary, not a
replacement: its documented [access-control semantics](https://docs.kernel.org/userspace-api/landlock.html)
leave preopened descriptors and several metadata operations outside its basic
file-content controls. Those require the mount and descriptor boundaries here.
Likewise, a tmpfs capacity is resident storage capacity, not a lifetime write
counter or total cloud bill; the prior cooperative accounting is retained.

## Important limits

Landlock ABI 5 does not deny pathname UNIX connections within an exposed
read-only tree. These software trees remain trusted host-provided inputs; the
fixture does not prove that their host-side contents are immutable or free of
all possible service sockets. Full pathname-IPC isolation and complete hostile
process containment therefore remain false. Hidden host paths and proc aliases
must be tested separately from this residual exposed-tree authority.

Procfs is not presented as a host-metadata confidentiality boundary. The
accepted WorkerRun still observes its owned ranks by their ordinary host PIDs;
no PID namespace or frozen parser rule is changed. Same-UID outside-domain
process access must be denied by the Landlock/ptrace boundary, not merely by
root ownership of the guardian. GPU memory, billing, runtime origin and real
host admission remain outside this generated proof.

## Verification and remote acceptance

The new mount-root and Landlock suites pass 26 and 33 tests respectively.
The exact combined 914-test stdlib regression passed with zero failures,
errors or skips in 0.587 seconds, reporting `ML_MODULES_LOADED=False`.
Isolated Ruff passes for all six new Python files; both Linux-only files pass
AST parsing without execution. Remote-only acceptance must be observed on the
exact published commit; these local tests cannot establish kernel/Torch acceptance.

The integrated generated fixture retains four cases: normal filesystem/network
checks, deadline cleanup, parent-loss/escaped-descendant cleanup and the pinned
WorkerRun's three invocations/six fresh ranks. It uses only newly owned empty
mountpoints, canary files and sentinel processes. Mutation probes target its
own generated read-only canary, never an existing source or runtime file.
Temporary storage exhaustion is bounded and cleaned inside the private view.
Exact-target cgroup kill, empty state, reaping and mountpoint removal must all
be accounted for; neither process exit nor an empty leaf alone proves success.

Reports are bounded fixed codes/counts/booleans without private file contents,
paths, PIDs or exception text. No namespace/controller/global host settings,
existing instance, registry, promotion, seal, training or real data is changed.

## What remains

Gate 5 remains partial until admission/host binding, trusted runtime/filesystem
handoff and residual host IPC constraints are resolved and tested together.
Real-size source import/relocation and actual H100 checks are separate. Private
research preparation does not grant production source/owner rights. No GPU is
needed for this source and remote-generated acceptance step.
