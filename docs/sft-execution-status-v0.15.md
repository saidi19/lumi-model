# SFT execution status v0.15

Local engineering checkpoint, 2026-09-06. This successor adds an owned pidfd
lifecycle and capability-snapshot policy to a new fixed inert probe. No real SFT
worker is admitted. Gate 5 and all private integration gates remain open.

## Previous step completed

Credential-probe commit `9fbea56b5454aa6a9b9d38274715fb6f2c71ad0b` was pushed
on the existing branch. [Source checks](https://github.com/saidi19/lumi-model/actions/runs/34034359566)
passed all four jobs, and [Spark source checks](https://github.com/saidi19/lumi-model/actions/runs/34034359604)
passed their three exact synthetic tests with no skips. The source suite reported
2,760 passes, 58 existing skips and 329 separately routed/deselected nodes;
327 private nodes were explicitly NOT EXECUTED. Those are source-scope results,
not a full private-integration or GPU-host acceptance result. The preceding
v0.14 document remains an unchanged pre-publication checkpoint.

## New implementation

- `posttraining_sft_child_status_v1.py`: bounded pure parsers for controller and
  owned-child status snapshots; aggregate outputs and allowlisted errors only.
- `posttraining_sft_peer_lifecycle_v1.py`: one fixed-child lifecycle using pidfds,
  real kernel exit/reap evidence, and the unchanged credential transport.
- `sft_peer_lifecycle_inert_worker_v1.py`: hash-pinned stdlib child that sets
  NoNewPrivs, exchanges two fixed messages and exits; no training or data access.
- Two new test modules and this status document; no historical files changed.

The controller requires trusted Linux CPython 3.12/3.13 with native sigaction,
pidfd signaling/polling/waitid support, nonzero uniform UID/GID roles, one Python
thread and one kernel-visible thread. A trusted genuine procfs mount and absence
of foreign reapers or native signal mutations are explicit prerequisites.

The existing SIGCHLD handler must be default. Before spawning, the controller
normalizes that default using the supported CPython signal setter; its native
implementation replaces flags without SA_NOCLDWAIT. This is normalization with
handler-metadata readback, not independent native-flag attestation. See the
[CPython 3.12 implementation](https://raw.githubusercontent.com/python/cpython/3.12/Python/pylifecycle.c)
and [CPython 3.13 implementation](https://raw.githubusercontent.com/python/cpython/3.13/Python/pylifecycle.c).

Immediately after spawn, the controller obtains a close-on-exec pidfd before
any Popen poll or wait. A non-reaping waitid call establishes owned-child status,
and polling rejects an already-exited child. Live checks bracket message receipt
and each child status read. Retirement requires matching observed and consumed
kernel exit records. Only then is Popen's cached return code set; its wait result
alone is never accepted as proof of a successful reap. The acquisition constraints
and polling behavior follow [pidfd_open](https://man7.org/linux/man-pages/man2/pidfd_open.2.html)
and [Python waitid](https://docs.python.org/3/library/os.html#os.waitid).

Failure cleanup closes IPC, signals only an owned pidfd when necessary, obtains
real reap evidence and closes the pidfd. There is no numeric-PID signaling
fallback. If binding failed, closing IPC lets the fixed worker exit; a bounded
Popen wait is best effort only and cleanup remains unverified. Every received
descriptor, status descriptor, endpoint and pidfd close is attempted once;
earlier close failures cannot be erased by successful later cleanup.
[Pidfd signaling](https://man7.org/linux/man-pages/man2/pidfd_send_signal.2.html)
avoids targeting an unrelated process after numeric PID reuse.

## Snapshot contract and limits

One controller status read precedes spawn. The child is checked after its READY
message while waiting for permission, then after its inert message while waiting
for the final ACK. Each snapshot requires exact process/parent identities,
uniform nonzero UID/GID roles, one thread, no visible tracer, singleton matching
namespace PID entries, and zero inheritable/permitted/effective/ambient
capabilities. Child NoNewPrivs must equal one; the separate controller parser
allows zero or one without changing it. Missing/duplicate/malformed fields deny.

The bounding capability set may be nonempty; the report counts how many child
snapshots had an empty bounding set. NoNewPrivs constrains exec-based privilege
gains but does not prevent every privilege-changing operation. Namespace and
tracer observations are relative to procfs visibility, not proof that an outer
namespace or invisible tracer is absent. See [process status fields](https://man7.org/linux/man-pages/man5/proc_pid_status.5.html)
and [kernel NoNewPrivs limits](https://docs.kernel.org/userspace-api/no_new_privs.html).
Continuous capability exclusion, namespace/ptrace containment and arbitrary
untrusted-worker safety remain explicitly false.

The probe permits at most one child, two credential receives, three status reads
and one pidfd signal attempt. Each status attempt prepays 16,385 bytes of capacity
(49,155 maximum), including failed opens/reads; the parser accepts at most 16,384
bytes and 256 fields. Each credential receive prepays the unchanged transport's
data-plus-ancillary capacity. Public worker-source authentication reads at most
8,193 bytes. No status, source-message or identifier text is retained in reports.
These application counters exclude interpreter/library loading and kernel I/O;
they are not total host resource measurements or secure-erasure claims.

The ordinary success deadline is six seconds; the stalled negative fixture uses
one second. Failure retirement can use a separate three-second cleanup deadline.
Deadlines do not bound blocking filesystem/kernel calls or provider billing.
There is no automatic retry of a started probe. Child privilege changes apply
only to the new inert process, not the user's shell or existing workers.

## Local verification

| Check | Result |
| --- | --- |
| Controller/child status-policy tests | 30 passed |
| Lifecycle primitives, orchestration and platform cases | 37 passed |
| New suites under ordinary pytest | 67 passed, 569 subtests passed |
| New suites plus previous credential/probe suites, stdlib unittest | 124 passed |
| Frozen seven safety/control suites | 273 passed, no skips |
| Exact Ruff on five new Python files | Zero findings |
| Frozen lint | 815 findings in 163 historical files unchanged, zero new |

These are repeated environments for the same cases, not additional unique test
counts. Including the frozen suites, targeted coverage is 397 unique tests.
Tests cover pidfd acquisition/ownership, exited/nonchild/mismatched
records, exit during a snapshot, ECHILD, nonzero exits, signal/reap/close failures,
capacity/deadline failures, source hash rejection and cleanup report integrity.

Four platform cases cover success, wrong-parent policy rejection, early child
exit, and a stalled child. On this Mac they verify non-Linux denial before procfs
reads or child creation. On the unchanged GitHub Linux source suite, they must
execute actual kernel/child cases: missing Linux support is a failure, not a
skip. This checkpoint does not itself claim the new commit is published or has
passed Linux CI; that requires the actual workflow results for its commit.

## Still open

This completes source implementation of a trusted inert lifecycle/snapshot
component, not the guarded real-worker boundary. Next: bind this lifecycle and
credential path to the existing durable epoch ledger, then establish and verify
Linux resource/network/descendant containment for the target host. Continuous
privilege exclusion and untrusted-code execution need that stronger boundary.

Authentic S3/DCP base import, private-data relocation, CUDA acceptance and a
complete execution ticket remain required before the bounded GPU smoke/resume.
The S3-backed Lumi-4B base pretrained on 80B tokens is unchanged. The original
roadmap remains gates 1-4 complete, Gate 5 partial, and gates 6-8 pending.
The frozen admission/bootstrap real-worker entries are unchanged and closed.
No model/data reads, private-evidence replay, registry mutation, GPU/instance
action, training, inference or promotion occurred in this local verification.
`production_eligible=false` and `full_private_integration_passed=false` remain.

The 327-node OPEN inventory and source partition remain byte-for-byte unchanged:

- Inventory SHA-256: `441390c9c98cd4b83d224055a5af5ae421e5ef01b2e4b86101e7b17693d87665`.
- Partition SHA-256: `4e595b654880668a4462c7b8db7c40eaf715ebd86b90eb14cdf852720bc831ae`.
