# SFT execution status v0.24

Local engineering checkpoint, 2026-09-07. A bounded service loop and fixed separate
controller-process wrapper are implemented. Real local child processes, inherited
pipes, stop/EOF and signals were exercised with synthetic provider responses.
No live Lambda request, credential acquisition, account lookup, instance action,
GPU use, training or private input occurred. The existing busy instance and the
S3-backed pretrained Lumi base are untouched. Gate 5 remains partial.

## Previous step verified

Provider-bridge commit `0750ce33423b8eb6fbc597ec77dd51f2eb03b861` was pushed after
the earlier approval-service credit block cleared. The v0.23 note retains that
earlier blocked snapshot unchanged; it is not the current publication status.
[Source CI](https://github.com/saidi19/lumi-model/actions/runs/34087979417) passed
all four jobs: 3,405 passed, 58 existing skips, 329 separately routed/deselected
nodes and 7,008 subtests. Source selection accounted for 3,463 nodes, with 327
private nodes NOT EXECUTED.
[Spark CI](https://github.com/saidi19/lumi-model/actions/runs/34087979399) passed
all three synthetic tests with zero skips. Exact commit and successful conclusions
were verified in the preceding publication turn. These are not full private CI.

## Scope and executable boundary

Five new files only: controller service, process/FD wrapper, their two test
modules and this checkpoint. Frozen provider/authority/lifecycle/journal/HTTP/TLS
sources and tests, launch contract, budgets, workflows and evidence are unchanged.
No real bootstrap, credential, MAC key, execution ticket or owner approval is made.

IMPORTANT: the new fixed child entrypoint CAN perform actual provider I/O if
given a valid separately authorized bootstrap. Source completion is not permission
to do so. There is no GPU/worker entrypoint, remote login, cloud deployment,
credential loader, arbitrary child command or general operation/target RPC.
The production entrypoint was exercised with an invalid bootstrap in a fresh
interpreter, and with synthetic valid authority plus pre-create EOF in a network-
denied local test. Successful provider paths use a separate synthetic test child
with every network/CA constructor denied and transport exchange replaced.

## Stop-only service

The service consumes exactly one prepared, already-owned ProviderController.
It never constructs one before fork, loads a key or changes its frozen plan.
Owner PID/thread and single-use checks reject reuse. Controller reports are
projected and type/state/count checked before choosing the next operation.

The service checks stop/EOF/signal/fault before create and between synchronous
actions. An observed stop before create sends no provider request. Stop during
an in-flight create does not interrupt or retry that consumed sequence: cleanup
can follow only if its exact ID is accepted and durably bound. A stop event
arriving between checks is not an atomic cancellation guarantee.

Bound state is observed periodically. Stop or termination-due state selects one
targeted terminate; termination-observing state selects only exact-ID GETs.
No target comes from the parent. Failed observe can lead to terminate; failed
terminate can lead to GET, never a second terminate. Ambiguous/protected create,
poisoned controller, missing/invalid state or expired hard bounds end with
reconciliation required, not guessed ownership or name-based search.

Observations are at least 15 seconds after the previous observation completes,
including recoverable early-poll denials. Local control waits are at most 250 ms.
They do not call durable tick or create fictitious finish_work records. The
service permits at most 60,000 control polls and 128 provider-method actions;
the frozen plan still limits actual request reservations to at most 64 and
reserves the final cleanup requests. A maximum-request test uses 65 actions
(one termination-required denial) and 131 journal records, below the frozen
256-record ceiling. Local waiting does not spend journal records.

The service's monotonic lifetime starts before create and is no longer than the
plan's 14,400-second maximum. Shutdown starts at the plan's earlier reserve
cutoff. Clock regression, cap exhaustion, invalid reports and failed closure
deny successful completion. Blocking DNS/TLS/fsync/scheduler calls can overrun
checks; there is no hard wall-time, physical packet or invoice guarantee.

Stop only selects cleanup; it never starts training or releases a worker lease.
There is no intermediate active-work admission channel in this version.
Final close releases the journal, not the cloud host. Termination acknowledgement
is still not terminal proof: the unchanged model requires a separate exact-ID
terminal observation, and actual provider/billing flags remain false even then.

## Separate process and descriptor protocol

The fixed spawner runs this source file in a fresh isolated, stdlib-only Python
interpreter (`-I -S -B`), with a new session, DEVNULL standard streams, sanitized
environment, close_fds and exactly four inherited descriptors. The controller
is constructed inside that child after exec; no live controller/journal object
is copied across a process boundary.

The caller supplies an already-open owned directory with exact mode 0700 and a
readonly nonblocking bootstrap pipe. The other inherited descriptors are the
child's stop reader and result writer. The parent retains only stop writer and
result reader; unused endpoints are closed. Root/bootstrap descriptors passed
to the spawner are borrowed. Child-owned descriptors are taken only after all
four roles validate and are closed once, without descriptor-number retries.

Bootstrap is one canonical ASCII JSON object, at most 32,768 bytes, requiring
EOF within a five-second checked deadline and at most 128 read calls. It carries
only fixed plan/authority/launch/token fields and digest/key bindings. Duplicate,
nonfinite, extra, oversized and noncanonical values reject before controller
construction. The existing authority verifier then checks all MAC and plan
bindings; pipe possession and supplied key pins are not rights-owner identity.
No secret is put in argv, environment, stdout or an exception report.

The stop protocol accepts one byte `S`, EOF, or locally latched signals. Unknown
or multiple bytes cause a control fault and cleanup, not an operation request.
A readonly nonblocking pipe is borrowed through a CLOEXEC duplicate. Stop/EOF
is absorbing. Once latched, a timed poll waits without rereading the permanently
ready pipe, preventing a busy loop from exhausting the control-poll budget.
Both caller endpoint privacy and correct closure of unused copies are required.

Temporary main-thread SIGTERM/SIGINT/SIGHUP handlers only set a stop flag. They
perform no I/O or provider operation and remain installed through final report
and descriptor cleanup. Repeated signals do not repeat termination. The generic
worker watchdog is not reused here: killing this controller on parent EOF would
prevent its cleanup. SIGKILL, process crash, host failure and blocking kernel
calls remain outside this cooperative handling.

The wrapper fixes the program path and sanitizes environment but does not attest
the interpreter, imported source tree, host, kernel, process privileges, code
integrity or executable filesystem. The external runtime and bootstrap producer
remain trusted. A new session is process separation, not independently surviving
infrastructure or containment of a hostile guest/owner.

## Result, closure and reconciliation

The child offers one canonical newline-terminated result of at most 4,096 bytes
with one nonblocking write. A short/failed write is not retried. The parent
requires strict schema/codes/types, complete pipe EOF and a successful owned
child reap before returning a report. Unknown data, oversized/truncated frames,
noncanonical JSON, nonzero exit and close failure cannot become successful cleanup.
EOF alone is not an exit receipt. Invalid output is never surfaced as raw text.

Result collection uses a caller-selected checked timeout of at most 30 seconds
per call and a nonresetting 128-read/4,096-byte capacity. Timeout leaves the same
child running for cleanup; later collection may finish that same result. There
is no kill, restart, resubmit, journal reopen or provider-operation method on the
parent handle. Dropping the handle is not a background-monitoring guarantee.

Reports contain only fixed codes, aggregate counts and modeled outcome booleans.
When bootstrap/process failure prevents trustworthy accounting, all counts are
null and reconciliation is required, not falsely zero. A process/report failure
cannot prove that no create was sent. Signals or failure after a report write
still produce a nonzero/missing exit outcome, which the parent rejects.

Spawn failure after child creation closes owned pipe endpoints to request EOF
cleanup; it never kills or restarts the child. If no usable handle is returned,
the caller needs external reconciliation. This is not an account-wide idempotency
or recovery service. Hash-only journals still cannot reconstruct the target
after a controller crash. No independent encrypted recovery record, authoritative
provider reconciliation or persistent daemon deployment is supplied here.

## Verification checkpoint

| Check | Result |
| --- | --- |
| Stop-only service and lifecycle fault tests | 36 passed |
| Process, pipe, signal, bootstrap and result tests | 47 passed |
| Both new suites under ordinary pytest | 83 passed; 67 subtests passed |
| New plus previous source-only controller/transport/safety integration suites | 794 passed |
| Frozen safety/control suites | 273 passed; no skips |
| Exact Ruff on all four new Python files | Zero findings |
| Frozen lint | 815 historical findings/163 files unchanged; zero new |
| Private readiness | 327 OPEN; zero private tests executed |

There are 1,067 unique targeted tests including frozen controls. Repeated runs
and subtests do not add unique test cases. Real local child-session/EOF/signal
observations use synthetic provider traffic only, not live TLS/cloud evidence.
Additional tests cover single-use claims, clock/cutoff boundaries, callback
failures, no per-wakeup journal writes, descriptor cleanup, unknown failure counts,
bounded result collection, and waiting for reap after EOF. Read-only review
identified ownership, watchdog and journal-budget constraints. Final checks were
completed locally after helper credit availability ended.

This is a pre-publication checkpoint; the successor commit's Source and Spark CI
must be verified separately. No result for those future runs is preclaimed.

## Next boundary

Separate-process execution and cooperative owned-cleanup handling are implemented
and synthetically tested. Still needed before live provisioning: independently
operated control infrastructure, secure real bootstrap/authority provenance,
recovery/reconciliation strategy, real account/key/image/price/target bindings and
the full execution ticket. Do not touch the existing busy instance.

Real host/boot/subtree ownership, aggregate resource enforcement, real-worker
admission, authentic S3/DCP import, private-data relocation and CUDA acceptance
remain. Roadmap: gates 1-4 complete, Gate 5 partial, gates 6-8 pending. The 327
private OPEN nodes are not roadmap steps. Production eligibility remains false.

- Inventory SHA-256: `441390c9c98cd4b83d224055a5af5ae421e5ef01b2e4b86101e7b17693d87665`.
- Partition SHA-256: `4e595b654880668a4462c7b8db7c40eaf715ebd86b90eb14cdf852720bc831ae`.
