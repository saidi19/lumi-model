# SFT execution status v0.16

Local engineering checkpoint, 2026-09-06. This successor joins the owned-child
lifecycle and per-message credential checks to the existing durable epoch ledger.
It runs only one fixed, hash-authenticated inert worker program. It does not admit
the real training worker or complete roadmap Gate 5.

## Previous step completed

Lifecycle/snapshot commit `898a50a58da9df51b1565c24eb5c5561b76e2040` was pushed.
[Source checks](https://github.com/saidi19/lumi-model/actions/runs/34035864116)
passed all four jobs, reporting 2,827 passes, 58 existing skips, 329 separately
routed/deselected nodes and 2,648 subtests. The source partition explicitly left
327 private nodes NOT EXECUTED. [Spark checks](https://github.com/saidi19/lumi-model/actions/runs/34035864102)
passed their three exact synthetic tests without skips. These were source-scope
checks, not target-GPU acceptance or full private integration.

## New integration

Only five new files are added: the authenticated-ledger module, the fixed worker,
two synthetic test modules and this checkpoint. Historical implementations,
tests, workflow definitions, inventories and private evidence are unchanged.
The worker source SHA-256 is
`c5b3e6e58afaa6a385dc14460e2466873b83792f6ec88589222e29376b020e9e`.

The parent authenticates at most 16,385 bytes from the fixed public script path,
then executes those exact bytes; it never reopens the path for execution. The
script is 10,555 bytes. Epoch/rank tokens travel only over the inherited private
socket, not argv, environment, logs or reports. Standard streams are discarded.
The child has no command for files, model/data access, descendants or training.

The successful fixed schedule remains two reference steps, one resumed-run step,
then the second resumed-run step: three fresh two-rank epochs, six owned children,
four global synthetic steps, 18 requests and 18 confirmed acknowledgements.
One cumulative supervisor and one exclusive attempt journal cover all epochs.
The attempt marker is consumed before spawning; even an existing empty or failed
journal denies another attempt. There is no worker replacement or crash-resume.

The admission order for every request and confirmation is:

1. Assert that the owned pidfd still names the live, unreaped child.
2. Receive one credential-checked packet with the expected PID/UID/GID tuple.
3. Recheck liveness and obtain the unchanged bracketed child-policy snapshot.
4. Feed the message through the permanently bound rank/session adapter.
5. For a request, persist and fsync its debit before sending the complete ACK.
6. Validate the confirmation, then send explicit permission to continue.

The continue handshake keeps the fixed child waiting while its confirmation is
checked. READY is credential/policy checked before releasing its epoch token.
Any failure interrupts the attempt; accepted debits remain charged. Descriptor
or status cleanup failures remain visible even if the ACK transaction also fails.

After protocol seal, each child sends bye and waits for release. The parent
requires actual matching kernel exit/reap evidence, then checks exactly one
terminal receive per peer. EOF requires empty data, empty ancillary data and no
unexpected flags. A queued zero-length packet carrying credentials is rejected,
as is ordinary trailing traffic. All delivered descriptor rights are closed
before validating packet data, flags or the post-receive deadline. This follows
the Linux sequence-packet receive path in the
[kernel implementation](https://raw.githubusercontent.com/torvalds/linux/v6.8/net/unix/af_unix.c)
and the distinction between EOF and zero-length messages in
[recvmsg](https://man7.org/linux/man-pages/man2/recvmsg.2.html).

Only after successful reap, EOF and socket/pidfd retirement may the durable epoch
completion be recorded. Cleanup attempts every handle, closes IPC before child
retirement, and runs before failure-journal append/fsync. Failed descriptor closes
are never retried. An unbound child receives only a bounded best-effort wait,
never numeric-PID signaling or a claimed verified reap. Failed pidfd acquisition
does not count as a closed pidfd. Journal close errors and final deadline overruns
deny overall PASS even if the journal already contains attempt_completed.

## Explicit bounds and nonclaims

| Application accounting | Hard limit | Fixed successful schedule |
| --- | --- | --- |
| Combined receive capacity reservations | 64 | 60: 54 authenticated packets and 6 terminal reads |
| Status capacity reservations | 64 | 55: controller plus 54 child snapshots |
| Parent send bytes precharged | 131,072 | Actual framed bytes charged before each send attempt |
| Journal capacity reserved once | 65,536 bytes | 22 committed aggregate-only records |
| Epochs / children / synthetic global steps | 3 / 6 / 4 | 3 / 6 / 4 |
| Synthetic checkpoint bytes reserved | Frozen existing cap | 6; no checkpoint is read |

Receive reservations include the unchanged transport's data and ancillary
capacity, cumulatively across all ranks and epochs, including terminal reads.
Each status reservation covers 16,385 bytes; the maximum is 1,048,640 bytes.
Creating a one-read frozen helper does not reset the shared 64-read limit.
Reservations are not refunded after failed reads/sends. Actual status attempts
and successful descriptor closes are reported separately from reserved capacity.

The combined ordinary deadline is 20 seconds, never reset per epoch; the stalled
negative fixture uses one second. Failure cleanup has one shared three-second
grace, not three seconds per child. Blocking OS/filesystem calls, interpreter
loading, library I/O, kernel overhead and provider billing are not bounded by
these application counters. These are not total host resource measurements.

The unchanged supported Linux CPython, trusted procfs, default SIGCHLD
normalization, nonroot uniform identities, one-thread and no-foreign-reaper
prerequisites remain. Capability/tracer/namespace checks are snapshots only.
No continuous privilege exclusion, arbitrary hostile-worker safety, cgroup
memory enforcement, network/namespace/descendant containment or target-GPU
verification is claimed. See the unchanged v0.15 contract for those limitations.

No private dataset, model, checkpoint, source object, GPU, external network,
inference, registry, instance, training, promotion or production action occurs
inside the probe. Public-source Git publication and ordinary source CI are
separate engineering operations, not a zero-network claim for this whole task.

## Local verification

| Check | Result |
| --- | --- |
| New integration/orchestration/platform cases | 27 passed |
| New primitive and failure-order cases | 32 passed |
| New suites under ordinary pytest | 59 passed, 32 subtests passed |
| New suites plus preceding credential/lifecycle/status suites | 183 passed |
| Frozen seven safety/control suites | 273 passed, no skips |
| Exact Ruff on all four new Python files | Zero findings |
| Frozen lint | 815 findings in 163 historical files unchanged; zero new |
| Private readiness inspection | 327 OPEN, zero private tests executed |

The environments repeat the same cases; counts are not additive across repeated
runs. Including frozen controls, targeted coverage is 456 unique tests. Tests
include a full mock transcript through the real durable journal, actual injected
fsync failure, early authentication/policy failure, no-refund ACK errors,
constructor ownership failures, zero-length/trailing/rights packet rejection,
close/reap failures, cumulative budgets and late-completion denial.

Nine platform cases cover success and eight fixed worker/transport failures.
On this Mac they verify non-Linux denial before procfs reads, attempt creation or
worker spawn. On Linux they must run actual inert workers and kernel checks;
missing Linux support is a failure, not a skip. This document is the local
pre-publication checkpoint, not a claim that its new commit passed Linux CI.
The final workflow result must be checked for the exact published commit.

## Roadmap and next boundary

Gates 1-4 remain complete; Gate 5 is partial; actual SFT, evaluation and serving
gates 6-8 remain pending. This is the durable authenticated inert-ledger portion
of Gate 5, not another completed top-level gate.

Next is the stronger real-worker containment boundary: establish and test Linux
resource, network and descendant controls for the target host before untrusted
training-code admission. Authentic S3/DCP base import, private-data relocation,
CUDA acceptance and the complete execution ticket remain before GPU smoke/resume.
The S3-backed Lumi-4B base pretrained on 80B tokens is untouched. Frozen real-worker
admission/bootstrap entries remain closed. `production_eligible=false` and
`full_private_integration_passed=false` remain unchanged.

The 327-node OPEN inventory is not a count of remaining roadmap steps:

- Inventory SHA-256: `441390c9c98cd4b83d224055a5af5ae421e5ef01b2e4b86101e7b17693d87665`.
- Partition SHA-256: `4e595b654880668a4462c7b8db7c40eaf715ebd86b90eb14cdf852720bc831ae`.
