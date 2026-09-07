# SFT execution status v0.23

Local engineering checkpoint, 2026-09-07. The durable lifecycle and bounded HTTPS
transport now have an explicit single-attempt dispatch bridge with a separate
pinned MAC contract. The bridge was exercised only with synthetic authority,
fake DNS/sockets/TLS and owned temporary journal directories. No live Lambda
request, account lookup, instance action, credential read or GPU execution occurred.
The existing busy instance is untouched. Gate 5 remains partial.

## Previous step verified

Transport commit `0ca525c8cdd978436a3f0de1b19e449559f7bcfb` was pushed.
[Source CI](https://github.com/saidi19/lumi-model/actions/runs/34084708551) passed
all four jobs: 3,327 passed, 58 existing skips and 329 separately routed/deselected
nodes, with 6,755 subtests. Source selection accounted for 3,385 nodes; 327 private
nodes remained NOT EXECUTED.
[Spark CI](https://github.com/saidi19/lumi-model/actions/runs/34084708567) passed
all three synthetic tests with zero skips. Both exact commit/run conclusions were
verified in the previous checkpoint turn. They are not full private integration.

## Scope and authority boundary

Five new files only: pure dispatch-authority verification, provider controller,
their two synthetic test modules and this checkpoint. Frozen lifecycle, journal,
HTTP/TLS, response parser, launch contract, budget, inventories, workflows and
historical evidence are unchanged. No issuer, CLI, credential loader, execution
ticket, real authority artifact, worker launcher or remote service is added.

IMPORTANT: calling the bridge's create/observe/terminate methods with real
dependencies CAN perform provider I/O. Source tests do not authorize doing that.
The unchanged `local_decision_model_only` plan cannot by itself pass the new
constructor. A separately supplied, explicitly pinned authority contract is
required in addition to the exact plan and launch body. No current real contract,
key, token, account identity, provider target or owner approval is fabricated.

The bounded, canonical ASCII authority object contains exact schema/mode/host,
plan/account/approval/launch/credential digests, creation window, request maximum,
cleanup scope and a domain-separated HMAC-SHA256. A supplied 32-byte key and its
external SHA-256 pin are checked. Hash, key, credential and MAC comparisons use
constant-time comparison; duplicate fields, noncanonical bytes, nonfinite JSON,
unknown keys, invalid token profiles and bool-for-int substitutions reject.
Authority bytes are capped at 8,192; the frozen plan remains capped at 16,384.
Tests alone have a synthetic signing helper; production code has no issuer.

MAC verification proves consistency with the supplied key, not rights-owner
identity, legal approval, account authentication or a provider response. External
key/pin provenance and their secure delivery remain trusted prerequisites. A
caller who supplies both an arbitrary key and matching pins is not thereby an
authenticated approver. Actual execution and production flags remain false.

All authority, launch-body and initial clock/window checks precede the exclusive
journal claim. The controller retains only the transient token and pinned launch
body needed for its attempt; neither is logged or written to the journal. Closing
clears those stored references, which is not secure Python memory erasure.
Authority bytes/key are not retained as controller fields. The old journal binds
the plan, not a new authority document or its MAC. External preservation of that
authority and actual dispatch evidence is still required for a complete audit.

## Durable dispatch ordering

1. The frozen durable controller reserves one intent and synchronizes its record.
2. The bridge checks its exact kind, consecutive sequence, capacity and target.
3. Only that fresh intent supplies a non-create target; no target argument exists.
4. The fixed compiler binds the request, and one fresh HTTPS object is constructed.
5. Fresh deadline/creation-window checks precede dispatch and transport stages.
6. Exactly one outcome is consumed: durable `receive` or durable `fail_pending`.
7. Usable modeled state is withheld until receipt synchronization completes.

Returned provider data is consumed by the unchanged strict wire/model validation;
the bridge does not bind an ID by parsing a response ahead of receipt persistence.
Malformed, protected or late create responses cannot authorize targeted cleanup.
A failed create is ambiguous: no second create, name search, guessed target or
attempt replay is supplied. An existing journal, even empty or failed, consumes
that directory/attempt claim and cannot be reopened for execution.

Every HTTPS failure poisons that transport. A later DIFFERENT durably reserved
cleanup intent may use a fresh transport; the failed sequence itself never does.
A failed observation can lead to an exact-bound-ID terminate. A failed terminate
can lead to a separate exact-ID GET, not another terminate. Creation-window expiry
does not prevent this bound cleanup while the lifecycle/request limits still hold.
A terminate acknowledgement, even one saying terminated, does not replace the
separate terminal GET. No billing-stop claim is made after either observation.

Receipt handling is single-consumption: once `receive` begins, any storage/clock
failure is absorbing and can never be followed by `fail_pending` for that sequence.
Ordinary pre-intent `poll_too_soon` and `termination_required` denials are recoverable.
The same exception codes after an intent or receipt has begun are not. Other
state misuse, including an attempted second terminate, closes this bridge; no
provider retry occurs, and external reconciliation may then be needed.

CREATE authority is checked at nanosecond wall-clock resolution through the
transport's guarded clock, including around DNS, TLS and send. Monotonic time is
sampled after the wall callback so a slow callback cannot release a stale deadline.
This remains a check around calls, not hard real-time scheduling or revocation
between instructions. Wall/monotonic callbacks and runtime dependencies are trusted.
Clock regression, wrong process/thread and reentry close admission. The owning
thread can still release local descriptors after a fault.

`finish_work` and `tick` are local journaled transitions. `close` only closes the
local journal: it never sends termination or claims a cloud invoice stopped.
Closing active/nonterminal state reports reconciliation required. There is no
poll/sleep loop, independent shutdown daemon, process survival, crash recovery
or target reconstruction from the hash-only journal.

## Accounting and evidence

The bridge's cumulative dispatch count cannot reset when fresh transport objects
are created. Each exchange invocation prepays 81,921 application read-capacity
bytes and 8,192 write-capacity bytes without refunds. The exact plan maximum is
at most 64 requests: at most 5,242,944 and 524,288 bytes respectively. Each transport
retains its previous one-connect/send and bounded read/deadline profile. The
frozen model reserves two remaining requests for termination and terminal GET.

The old exclusive journal keeps its 256-record/1,048,576-byte reserved-capacity
ceiling, which may close operations before a larger model event allowance.
Actual transport counters are aggregated only from completed local transport
summaries. During an operation or after a counter-collection failure they are
null with `network_counters_complete=false`, not misleading zero totals. Prepaid
capacity and dispatch invocation counts remain known. Counter-collection failure
closes the bridge even if a modeled receipt was already synchronized.

Reports expose allowlisted aggregates, fixed codes and the last synchronized
model snapshot, not raw IDs, intent repr, credentials, authority, responses or
exceptions. Frozen synthetic zero-network/provider-call fields are deliberately
not inherited. Transport counters are volatile application observations, not
durable/provider-authenticated receipts or physical packet/invoice measurements.
The modeled work window is a snapshot, not a fresh worker admission lease;
`execution_authorized` and `real_sft_execution_permitted` always remain false.

## Verification checkpoint

| Check | Result |
| --- | --- |
| Pure authority contract | 24 tests passed |
| Fake-only provider bridge, real temporary journal and injected faults | 54 tests passed |
| Both new modules under ordinary pytest | 78 passed; 253 subtests passed |
| New plus previous transport/lifecycle/observer/guard/ledger/credential/status/IPC | 711 passed |
| Frozen safety/control suites | 273 passed; no skips |
| Exact Ruff on all four new Python files | Zero findings |
| Frozen lint | 815 historical findings/163 files unchanged; zero new |
| Private readiness | 327 OPEN; zero private tests executed |

There are 984 unique targeted tests including frozen controls. Repeated runs and
subtests do not add unique test cases. Network/CA entry points are replaced before
every test controller operation; no real socket or provider authority is used.
Tests cover reservation-fsync-before-DNS, receipt-fsync-before-window release,
post-reservation expiry, slow clock acquisition, ambiguous create, both cleanup
progressions, protected targets, request limits across fresh transports, storage
faults, single receipt consumption, reentry, owner checks and redacted counters.
Read-only review identified the clock-ordering and receipt-error classification
fixes. Final local regression was completed by the main agent after helper credit
availability ended; no additional independent review completion is claimed.

This is a pre-publication local checkpoint. The new commit's Source and Spark CI
must be verified separately; no outcome for those new runs is preclaimed here.

Publication was attempted only for the five exact successor files. The approval
service rejected `git add` before execution because workspace credits were
unavailable. That denial was not retried or bypassed. No successor staging,
commit, push or new CI run completed. Local tests above remain valid; publication
and exact-commit CI verification are pending approval-service availability.

## Next boundary

The library-level durable dispatch integration is implemented and synthetically
tested. It is not yet an independently surviving external control process with
authenticated execution authority and authoritative reconciliation. Those are
the next boundary, together with real account/key/image/price/target bindings and
the complete creation-plus-owned-cleanup authority. No GPU provisioning yet.

Real host/boot/subtree ownership, aggregate resource enforcement, the real-worker
profile, authentic S3/DCP import, private-data relocation, CUDA acceptance and
the complete execution ticket remain. The S3-backed pretrained Lumi base is
unchanged. Roadmap: gates 1-4 complete, Gate 5 partial, gates 6-8 pending. The 327
private OPEN nodes are not roadmap steps. Production eligibility remains false.

- Inventory SHA-256: `441390c9c98cd4b83d224055a5af5ae421e5ef01b2e4b86101e7b17693d87665`.
- Partition SHA-256: `4e595b654880668a4462c7b8db7c40eaf715ebd86b90eb14cdf852720bc831ae`.
