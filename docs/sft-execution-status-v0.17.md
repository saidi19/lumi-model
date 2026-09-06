# SFT execution status v0.17

Local engineering checkpoint, 2026-09-06. This successor implements a fixed
inert-child Linux syscall policy and hard per-process resource limits. It does
not implement target-host cgroup/network-namespace isolation, admit arbitrary
workers, integrate the new policy into the epoch ledger, or authorize SFT.
Roadmap Gate 5 remains partial; this is one of its containment prerequisites.

## Previous step completed

Authenticated-ledger commit `38ac8f97aafc768e197400a805c7b76e9437af5f` was pushed.
[Source checks](https://github.com/saidi19/lumi-model/actions/runs/34037035882)
passed all four jobs: 2,886 tests passed, 58 existing skips, 329 separately
routed/deselected nodes and 2,680 subtests. The 327 private nodes remained NOT
EXECUTED. [Spark source checks](https://github.com/saidi19/lumi-model/actions/runs/34037035877)
passed all three exact synthetic cases with no skips. These are source-scope
results, not full private integration or acceptance on the GPU host.

## New files and unchanged boundaries

Seven new files only: standalone kernel policy, fixed worker, parent probe,
three synthetic test modules and this checkpoint. All historical source,
tests, workflow definitions, inventories, reports and attempt markers remain
unchanged. No private-data/model files or existing cloud instance are involved.

The parent authenticates two public source files independently, then compiles
those exact bytes in separate module namespaces inside one fresh child. This
avoids global-name collisions and does not reopen either source path for worker
execution. The child gets only one explicitly inherited private packet socket;
standard streams are discarded and the environment is fixed. No caller-supplied
executable, profile, alternate allowlist or training dispatch is accepted.

- Policy: 10,521 bytes; SHA-256
  `33bf272bd1b97f3ce74a9ecec528c88adce972dd8f31873317f3fe912f2c8c85`.
- Worker: 9,092 bytes; SHA-256
  `3fe2376fd09ce7f30fdd141f90a8d2e973285375d6e89a762fb50610f6a0855b`.

Only the child changes limits and installs the filter. The parent's policy ABI
check is read-only. Supported execution is nonroot Linux x86-64, little-endian
LP64, CPython 3.12/3.13 with `-S -B`, one trusted controller thread, uniform
UID/GID roles, genuine procfs and the unchanged owned-child lifecycle contract.
Unsupported ABI/runtime or stricter incompatible inherited limits deny; there
is no fallback that weakens the profile.

## Fixed child policy

| Per-process resource | Exact soft and hard limits |
| --- | --- |
| Virtual address space | 128 MiB |
| CPU time | 2 seconds |
| Core-file size | 0 bytes |
| File size | 0 bytes |
| Open-descriptor limit | 64 |
| Process/thread creation limit | 1 |

All inherited limits are checked before changing any of them. The child only
lowers limits, reads every pair back exactly, sets and verifies NoNewPrivs, then
installs its own compiled seccomp filter and verifies filter mode/NoNewPrivs.
Any partial failure requires exit; there is no rollback, retry or READY message.

These limits have distinct semantics. Address-space limits are not aggregate
resident-memory limits. CPU time is not elapsed time. Linux's process limit is
UID-accounted and is not a cgroup tree limit; lowering the descriptor limit does
not close inherited high-numbered descriptors. The fresh descriptor allowlist
and syscall filter are separate controls. The resource semantics follow
[getrlimit](https://man7.org/linux/man-pages/man2/getrlimit.2.html).

The filter first checks architecture and rejects the x32 syscall convention.
All unlisted syscalls return EPERM; mismatched architecture/x32 terminate the
process. Syscall numbers are bound to the native
[Linux x86-64 table](https://raw.githubusercontent.com/torvalds/linux/v6.8/arch/x86/entry/syscalls/syscall_64.tbl).
Basic memory, time, signal and exit operations remain available. `sendmsg` and
`recvmsg` are restricted to the inherited descriptor; `ioctl` is limited to that
socket's FIONBIO operation, and socket-option reads to a small fixed set.
Only the two no-argument policy getters remain permitted through `prctl`.

New sockets, connections, file opening, descriptor duplication, process/thread
creation, exec, namespace changes, ptrace, resource-limit changes, new filters,
BPF and io_uring entry points are denied. Pointer contents are not inspected by
classic seccomp; the trusted IPC peer and existing received-rights cleanup remain
required. Seccomp is one mechanism, not a complete sandbox. Installation,
architecture checks and these limitations follow the
[kernel seccomp documentation](https://docs.kernel.org/userspace-api/seccomp_filter.html).

New mappings must be private anonymous mappings with zero offset and no EXEC
protection; `mprotect` also cannot request EXEC. Existing interpreter/library
executable mappings are not removed. For the anonymous `fd=-1` sentinel, the
filter accepts only its canonical zero-extended or sign-extended integer forms.
It still rejects every other descriptor representation, flags or offset. This
handles the native integer argument without granting file-backed mapping; see
[anonymous mmap semantics](https://man7.org/linux/man-pages/man2/mmap.2.html)
and [glibc's argument lowering](https://raw.githubusercontent.com/bminor/glibc/glibc-2.39/sysdeps/unix/sysv/linux/x86_64/sysdep.h).
The fixed CPython timeout path's bound-FD ioctl/poll needs were checked against
[CPython's socket implementation](https://raw.githubusercontent.com/python/cpython/3.12/Modules/socketmodule.c).

## Bounded kernel challenges and parent verification

After installation, the fixed child sends READY and waits. Before granting
permission, the parent requires live owned-pidfd evidence, per-message kernel
credentials, the unchanged capability/identity policy, and Seccomp=2 with a
positive bounded filter-count snapshot. That metadata alone does not identify
our filter; it is combined with authenticated source, successful own installation
and the fixed challenges. It is not independent hostile-worker attestation.

The child then checks exactly 17 denied syscall cases, one successful 4 KiB
anonymous mapping/unmapping, and one rejected 129 MiB mapping. No memory pages
are touched. Denial arguments use empty paths, invalid descriptors or otherwise
inert parameters; there are no real connection destinations or successful
descendant-creation challenges. Any unexpectedly created socket is closed before
failure. Actual fork/vfork/clone denial decisions are covered by filter-interpreter
tests; the live child challenges clone3 with null/zero arguments instead.

Only after all challenges pass does the child send its fixed proof and wait
again. The parent repeats credential/liveness/status checks before the final
acknowledgment, requires real successful kernel reap, then verifies strict EOF.
Even a queued empty credential packet is not EOF. Every delivered descriptor is
closed before packet/flag/deadline rejection. Cleanup/close failures and a late
final deadline prevent PASS; failed closes are never retried.

The probe permits one child, two authenticated receives, one terminal receive,
three prepaid status reads, and at most ten parent-send bytes. Each status read
reserves 16,385 bytes (49,155 maximum). Receive capacity includes the unchanged
transport's payload and ancillary buffers. Public-source authentication reads
at most 32,769 plus 16,385 bytes. These are application capacities, excluding
interpreter/library loading, kernel overhead and total host I/O.

The ordinary absolute deadline is six seconds; the stalled negative fixture uses
one second. Failure retirement may use the unchanged three-second cleanup grace.
Blocking OS operations and provider billing are not bounded by these timers.
No started probe is automatically retried. Reports contain aggregate counters
and fixed codes, not source/status/message text, identities or exceptions.

## Local verification

| Check | Result |
| --- | --- |
| Policy compiler/ABI/installation synthetic cases | 40 passed |
| Fixed worker synthetic cases | 28 passed |
| Parent snapshots/orchestration/platform cases | 29 passed |
| New stdlib suites | 97 passed |
| New suites under ordinary pytest | 97 passed, 1,386 subtests passed |
| New suites plus preceding ledger/credential/lifecycle suites | 280 passed |
| Frozen safety/control suites | 273 passed, no skips |
| Exact Ruff on all six new Python files | Zero findings |
| Frozen lint | 815 historical findings/163 files unchanged; zero new |
| Private readiness | 327 OPEN; zero private tests executed |

Repeated environments are not additional unique cases. Targeted coverage totals
553 unique tests including the frozen controls. The compiler tests exercise
architecture/x32 rejection, forward jumps, unknown syscalls, argument high words,
exact IPC restrictions and both valid mmap sentinel forms. Other tests cover
partial installation/readback failures, cleanup, source binding, separate source
namespaces and late-success denial. No test installs a filter or changes limits
in the parent/test-runner process.

Three platform cases cover success, early exit and stalled proof. On this Mac,
they verify non-Linux rejection before source reads or child creation. On the
existing Linux CI they must execute real fixed children; unsupported Linux is
a failed case, not a skip. This is the local pre-publication checkpoint, not a
claim that its new commit has already passed those Linux cases. Exact CI results
must be checked after publication. Source publication/CI and documentation
research are separate from the probe's zero-external-network contract.

## Still required

Next, compose this fixed policy with the authenticated durable-ledger successor,
then verify target-host aggregate resource/descendant controls and the network
boundary needed by the real worker. This profile deliberately cannot run ML,
load datasets or create training threads; it must not be silently widened into a
GPU profile. A separately reviewed real-worker profile is required.

Per-process limits do not replace the hierarchical resource and lifecycle
controls described by [cgroup v2](https://docs.kernel.org/admin-guide/cgroup-v2.html).
Target-host delegation, cgroup memory/process/cleanup evidence, network policy,
authentic S3/DCP import, private-data relocation, CUDA acceptance and the complete
execution ticket remain open. The existing busy Lambda instance and S3-backed
Lumi-4B pretrained base are untouched.

Roadmap: gates 1-4 complete, Gate 5 partial, gates 6-8 pending. No real SFT,
inference, GPU, model restore, registry, instance, promotion or production action
occurred. `production_eligible=false` and `full_private_integration_passed=false`.
The 327 private OPEN nodes are not a count of remaining roadmap steps.

- Inventory SHA-256: `441390c9c98cd4b83d224055a5af5ae421e5ef01b2e4b86101e7b17693d87665`.
- Partition SHA-256: `4e595b654880668a4462c7b8db7c40eaf715ebd86b90eb14cdf852720bc831ae`.
