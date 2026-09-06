# SFT execution status v0.14

Engineering checkpoint, 2026-09-06. This adds a Linux per-message credential
component and fixed inert probe. It does not admit a real SFT worker, close Gate 5,
authorize execution or change production eligibility.

## Completed source work

The next critical-path item in v0.13 was the guarded-worker boundary with real
process identity and target-Linux containment. This step addresses only the
per-message credential part, independently of the existing durable ledger.
Historical private-checker follow-ups are not prerequisites for implementing this
isolated source component; none were replayed or reclassified here.

The new transport uses unnamed AF_UNIX SOCK_SEQPACKET endpoints with SO_PASSCRED
enabled before sending. Each received packet must carry exactly one correctly
sized SCM_CREDENTIALS tuple matching the expected PID, real UID and real GID.
SO_PEERCRED is not used: socketpair creation-time credentials cannot establish
which later inherited child sent a message. Privileged senders can override
applicable credential fields, so this is not proof of capability exclusion.
These semantics follow the [Linux UNIX socket documentation](https://man7.org/linux/man-pages/man7/unix.7.html).

The transport returns only a validated opaque length-prefixed frame. It rejects
missing, duplicate, malformed or mismatched credentials; unexpected ancillary
data; truncated or oversized messages; expired deadlines; and exhausted budgets.
Every delivered SCM_RIGHTS descriptor is closed before packet acceptance checks,
including on rejection. Descriptors are received with MSG_CMSG_CLOEXEC; a failed
close is not retried. See [Python recvmsg documentation](https://docs.python.org/3/library/socket.html#socket.socket.recvmsg).

Each application receive attempt prepays its entire data-plus-ancillary capacity,
including errors, without refunds. Payload capacity is 4,096 bytes; the receive
data buffer is 4,101 bytes so an oversized frame can be detected. The ancillary
capacity is calculated for the host ABI. These counters do not represent all
kernel allocations, syscalls or physical host I/O. There is no application-level
receive retry, stream accumulation, deadline reset or protocol decoding.

## Fixed inert probe and boundaries

The probe authenticates at most 8,193 bytes of one hash-pinned public worker
script and executes the authenticated bytes directly, avoiding a second mutable
path read between authentication and execution. It uses the trusted current
interpreter with `-S -B`, a fixed environment, one explicitly inherited endpoint
and disabled standard streams. No caller-provided command or worker source is
accepted. The source checkout and interpreter are trusted, not OS-isolated.

One probe call starts at most one fixed inert child and makes at most one channel
receive attempt. A positive result requires the exact fixed packet, successful
child exit and reap, completed socket cleanup, and the original five-second
deadline. Failure cleanup has at most a further one-second reap wait; these are
application deadlines, not a blocking-syscall or provider-billing guarantee.

Five fixtures cover a valid sender, a child incorrectly compared with the
socketpair creator, passed descriptor rights, an oversized packet and a stalled
child. Negative fixtures must report rejection, not production success. On
non-Linux hosts, the kernel gate denies before reading worker source, creating
sockets or launching a child. Tests explicitly exercise that denial; they do not
count it as Linux acceptance.

The returned report contains allowlisted status codes, booleans and aggregate
counts, never payload, PID/UID/GID values, descriptors, paths or exception text.
Its zero data/model/private-write/network/GPU/inference counters describe only
the fixed probe, not surrounding CI installation or Git operations.

New files are limited to two modules, one fixed worker, two test modules and this
document. No existing ledger, admission, bootstrap, launch contract, workflow,
source partition or historical evidence was changed.

## Verification at this checkpoint

| Check | Result |
| --- | --- |
| Synthetic credential transport cases | 42 passed locally |
| Probe lifecycle, source binding and platform cases | 15 passed locally |
| Combined `python -S -B -m unittest` | 57 passed, no skips |
| Same cases under ordinary pytest | 57 passed, 86 subtests passed |
| Frozen seven safety/control suites | 273 passed, no skips |
| Exact Ruff check on five new Python files | Zero findings |
| Frozen lint debt | 815 findings across 163 files unchanged; zero new |
| Original unresolved inventory | 327 OPEN retained, hash unchanged |
| Source/private partition | Hash unchanged |

The two local runs are the same 57 tests, not 114 unique cases. Including the
273 frozen controls, targeted coverage is 330 unique cases. Local host is
macOS; its five platform cases verified non-Linux denial, not actual Linux
credential delivery. Existing GitHub source-suite routing automatically runs the
new unclassified tests. On Linux the five live cases must exercise actual local
kernel sockets and children; unsupported Linux features fail, not skip. Hosted
results must be read from the commit's workflow run, not inferred from this table.
This document is written before that run and does not claim a Linux CI pass.

The initial publication attempt was blocked when the approval service reported
exhausted workspace credits; that staging command was not executed. On the
user-requested resumption, scoped staging received approval and succeeded. No
permission workaround was used. Hosted acceptance is still pending at the time
this document is written and must be checked against the published commit.

Pre-publication review found and corrected an early endpoint-close error that
could otherwise be lost when later cleanup succeeded. Such errors, including
received-descriptor cleanup failures, now keep `worker_cleanup_complete=false`
and prevent a pass without retrying the close. The added regression covers both
cases. No tests, checks or workflow steps were removed to obtain these results.

Preserved inventory file SHA-256:
`441390c9c98cd4b83d224055a5af5ae421e5ef01b2e4b86101e7b17693d87665`.
Preserved source partition file SHA-256:
`4e595b654880668a4462c7b8db7c40eaf715ebd86b90eb14cdf852720bc831ae`.

## Still required on the critical path

The full roadmap remains gates 1-4 complete, Gate 5 partial, and gates 6-8
(actual SFT, post-SFT evaluation, serving) pending. This component does not close
the worker-boundary category. Next engineering work must establish a separately
trusted process-generation/lifecycle binding and capability policy, then connect
credential-checked messages to the existing durable epoch ledger without
weakening it. Linux resource/network/descendant containment and validation on the
actual target host remain required.

Authentic S3/DCP import of the unchanged Lumi-4B base pretrained on 80B tokens,
private-data relocation, CUDA acceptance and a complete execution ticket are
also still required before the bounded GPU smoke/resume. The frozen real-worker
admission and bootstrap entry points remain closed. This step does not restore
the model, read training/private data, launch an instance, run inference/SFT,
promote a dataset/model or close any of the 327 historical private nodes.
`production_eligible=false` and `full_private_integration_passed=false` remain
mandatory.
