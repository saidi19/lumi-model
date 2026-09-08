# SFT execution status v0.35

Owned Linux cgroup resource/parent-death successor, 2026-09-08. This work adds
kernel resource controls around the accepted generated WorkerRun lifecycle. It
does not rent a GPU, authorize real model/data access or open the public training
entrypoint. All local verification remains small stdlib tests and exact lint/AST;
neither Linux-only wrapper nor fixture is executed on the Mac, even in skip mode.

## Exact predecessor and new scope

Predecessor `0923d34ec8fe382f5251fb3a74331a239d153ebf` passed all four jobs in
[Source CI](https://github.com/saidi19/lumi-model/actions/runs/34196982880):
4325 passed, 58 skipped, 329 deselected and 8458 subtests in 280.58 seconds.
The source partition selected 4383 tests. Its 327 private integrations remained
NOT_EXECUTED; two historical failures remained separately routed. Private/full
CI claims stayed false. [Spark CI](https://github.com/saidi19/lumi-model/actions/runs/34196982968)
passed three tests with zero skips in 101.39 seconds.

Only seven new files belong to this successor:

- `src/lumi/posttraining_sft_cgroup_v1.py`
- `src/lumi/posttraining_sft_cgroup_guardian_v1.py`
- `tests/test_sft_cgroup_v1.py`
- `tests/test_sft_cgroup_guardian_v1.py`
- `scripts/run_sft_cgroup_fixture_v1.py`
- `tests/test_sft_cgroup_linux_v1.py`
- This document.

All earlier sources, tests, workflows, budgets and evidence remain unchanged.
Unrelated untracked files are excluded. The accepted WorkerRun fixture remains
bound to SHA-256 `53c50c92965ed1cef0c9624c57367c31ed7ebfb9d7cfb34257188ad179d333d0`.
No retry or fallback to process-group-only cleanup substitutes for a failed
cgroup attempt. No parent controller, existing cgroup, registry, dataset seal,
S3 object, instance or production eligibility is modified by this work.

## Lease and cleanup authority

`CgroupLease` accepts a borrowed, authenticated root-owned cgroup2 parent FD and
creates one exclusive, specifically named child. It requires the needed kernel
interfaces before configuring six exact controls: memory maximum, zero swap,
group OOM policy, task maximum, CPU bandwidth and zero CPU burst. It reads those
settings back. It never enables controllers or changes parent ownership/limits.
Missing facilities fail before a workload can be admitted.

Reads/writes are bounded and descriptor-relative. Root ownership, modes,
filesystem magic and parent/leaf identity are rechecked. Short/ambiguous writes
are not retried. Failure or a consumed kill attempt denies further admission while
retaining separately
guarded cleanup reads and one kill attempt for the same authenticated leaf.
Changed identity cannot gain cleanup authority. Removal is empty-only `rmdir`
of that one owned leaf; no recursive removal or directory scan selects targets.

The guardian lives outside workload limits. It checks the actual parent around
pidfd binding, verifies a populated configured lease, then invokes the trusted
release callback. Parent loss or the monotonic deadline takes precedence over
apparent completion. Every completion path retires the cgroup and checks empty;
the caller separately verifies exit results, reaps children and closes the lease.
An empty cgroup alone is not a successful training result.

Before workload execution, the root bootstrap attaches itself, drops all real,
effective and saved UID/GID values and supplementary groups, and sets
`no_new_privs`. A bounded genuine self-procfs snapshot verifies identities,
empty current capability sets, no visible tracer, one thread and the flag before
ML/descendants. All cgroup/pidfd authority descriptors are then closed. No ML
library is imported in the privileged guardian or outer controller.

These mechanisms were checked against the official
[cgroup v2 interface](https://docs.kernel.org/admin-guide/cgroup-v2.html) and
[no-new-privileges semantics](https://docs.kernel.org/userspace-api/no_new_privs.html).
Cgroup killing covers descendants despite `setsid`; CPU bandwidth is not total
CPU time or billing. Task limits count kernel tasks, not just model ranks.
Memory maximum enforcement can include transient kernel overshoot; the test
checks actual OOM events instead of asserting that observed peak never exceeds
the configured number. `no_new_privs` constrains exec privilege gains, not all
possible privilege-changing operations or arbitrary external authority.

## Local verification and remote-only acceptance

The new lease and guardian suites pass 35 and 60 tests respectively. Their
kernel/privilege operations are syscall doubles only. The final exact 779-test
regression run, including predecessor self-status/host-parser and WorkerRun
tests, passed with zero failures/errors/skips in 0.539 seconds.
Isolated lint passes for all six new
Python files, and both Linux-only files pass AST parsing without execution.
Fresh no-site/no-bytecode local runs confirm `ML_MODULES_LOADED=False`.

The Linux acceptance fixture requires a genuine writable cgroup2 host and
noninteractive sudo; missing prerequisites fail, not skip. It changes only
seven newly created root-owned workload leaves. Its hierarchy is an outer
subreaper, a controller watching that outer by pidfd, an independent guardian
watching the controller, and a permanently unprivileged workload. The guardian
creates and owns its lease; the forked bootstrap receives only a borrowed FD
for attachment, never inherited lease-object ownership.

The seven cases are normal completion; a bounded 96-MiB allocation against a
64-MiB memory limit; four-task fork rejection with at most eight attempts; CPU
throttling; deadline cleanup; actual controller SIGKILL after a double-fork and
`setsid` descendant is confirmed alive; and the exact previously accepted
WorkerRun fixture under a 4-GiB, 256-task, two-CPU-bandwidth cap. These are tiny
synthetic allowances, not actual H100 or real-data budgets.

The positive ML case uses the CI-provided Python 3.12 executable's bounded
path/ELF hash binding and the fixed script hash above. The runtime executes only
after privilege drop, with fixed offline/CPU/loopback settings. No packages are
installed by the privileged fixture. It must again complete three invocations,
six fresh ranks and four global/eight rank updates with model/Adam/RNG/cursor
resume equivalence. This is an additional controlled generated test, not real SFT.

Reports require parent binding, release, exactly one kill attempt, empty state,
owned-process reaping and removal receipts. Snapshots are refreshed after
reaping. Resource event counters, not mere interface existence, establish each
fault case. Public output is at most 4096 fixed-code/count/boolean bytes; no
PIDs, paths, model values or third-party exception text escape.

The root hierarchy has its own non-resetting deadline. The outer wrapper's
300-second timeout does not claim that an unprivileged killpg cleaned root
processes; it explicitly reports unconfirmed cleanup. Guardian death, hostile
root intervention and uninterruptible kernel calls remain outside this proof.
Remote acceptance must be observed on the published commit; static/local
results alone do not establish cgroup or Torch acceptance.

## What remains

This addresses owned descendant/resource enforcement and parent-death behavior,
subject to remote acceptance. It is not the complete training-capable sandbox:
network policy compatible with distributed communication, private filesystem
access/write enforcement, admission/host binding and their integrated fault
tests remain. Cooperative cumulative byte accounting is retained, not relabelled
as an OS disk quota. GPU memory and cloud billing are not controlled by cgroups.

Real-size source import/relocation, reproducible runtime handoff and actual
H100-only verification also remain separate. Production source/owner rights
remain separate from private research preparation; production stays false.
No GPU needs to be started for this source/remote-synthetic step, and no calendar
completion estimate follows from its test count.
