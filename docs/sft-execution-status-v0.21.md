# SFT execution status v0.21

Local engineering checkpoint, 2026-09-07. The lifecycle decision model now has an
exclusive local write-ahead journal and controller wrapper. A reservation must
be written and synchronized before its inert decision token is returned. This
closes the previous volatile-only reservation gap within one trusted local
directory/attempt. It is not a live provider controller, GPU launch or SFT approval.
The existing busy instance is untouched. Gate 5 remains partial.

## Previous step verified

Lifecycle-core commit `6dc156ad0aaea21a8566a6855761de91da06b81a` was pushed.
[Source CI](https://github.com/saidi19/lumi-model/actions/runs/34042905749) passed
all four jobs: 3,198 passed, 58 existing skips, 329 separately routed/deselected
nodes and 6,570 subtests. Source selection accounted for 3,256 nodes; 327 private
nodes remained NOT EXECUTED.
[Spark CI](https://github.com/saidi19/lumi-model/actions/runs/34042905854) passed
all three synthetic tests with zero skips. A local status-read timeout did not
fail or rerun that successful CI job. Neither result is full private integration.

## Scope and storage boundary

Five new files only: exclusive journal sink, durable lifecycle wrapper with pure
structural verifier, their two synthetic test modules and this checkpoint. The
previous lifecycle/wire sources, tests, reports, launch contract, local budget,
workflows, inventories and historical evidence remain byte-for-byte unchanged.
No real plan, account credential, execution ticket or owner approval is fabricated.

The caller supplies an already-open descriptor for a trusted, stable, local,
owned directory with exact mode `0700`. The code borrows it and owns a CLOEXEC
duplicate; it neither chooses an arbitrary path nor changes existing permissions.
One deterministic attempt filename is created using `O_CREAT|O_EXCL|O_NOFOLLOW`
and exact mode `0600`. It must be a regular, singly linked, owned file. Both the
file and containing directory are synchronized before the constructor succeeds.
The [directory sync requirement](https://man7.org/linux/man-pages/man2/fsync.2.html)
is separate from flushing the file's contents.

Before and after each append, pinned descriptor and directory-relative filename
checks compare device/inode, owner, mode, link count, expected length and CLOEXEC.
Replacement, symlink, hardlink or mode/size drift fails closed. These checks do
not make a hostile namespace safe: trusted local storage and absence of concurrent
namespace mutation remain prerequisites. The implementation does not verify the
filesystem type, encryption, extended ACL policy or physical power-loss durability.
It does not protect against privileged or same-owner evidence deletion. The
[exclusive-open semantics](https://man7.org/linux/man-pages/man2/open.2.html) are
not a global account-wide idempotency service.

Any existing attempt filename consumes that attempt, including empty, partial,
failed or successfully closed evidence. There is no overwrite, truncate, unlink,
reopen-for-append or execution-resume API. Choosing a different directory or
attempt is not authorized by this local claim. No real journal is created in the
project: all exercised journals belong to temporary synthetic test fixtures.

## Persistence before decision release

The wrapper uses the unchanged lifecycle model and its strict, externally pinned
plan parser. It serializes the owning thread/process's transition, persistence,
post-sync validity check and decision release. The sink does not accept concurrent
or reentrant writes. Another thread in the same process cannot close shared
descriptors during an append; a fork can only release its own descriptor copies.

Each offered record prepays 4,096 write-capacity bytes. At most 256 records and
1,048,576 bytes are reserved cumulatively, including initialization and closure.
This deliberately lower journal ceiling takes precedence over the frozen model's
4,096-event allowance. Exhaustion permanently closes wrapper actions; no record
is silently omitted and no failed reservation is refunded. One OS write is issued
per record. Short/zero writes, exceptions, fsync failures and identity failures
are sticky, with no partial-write retry. File fsync precedes acceptance.

Model transitions that raise a logical denial are also persisted before the
exception escapes. A normal `poll_too_soon` or `termination_required` decision
does not itself break a healthy journal. If persistence fails after the model
advanced, its result/intent is withheld and all later wrapper operations reject.
Reports expose only the last successfully synchronized model snapshot, never an
unsynchronized active or terminal observation. Work is closed during in-flight
operations and after wrapper failure/closure.

The controller samples a monotonic clock before each operation and after durable
I/O. Before releasing a create intent it also rechecks the creation approval's
half-open wall-clock window. A sync that crosses approval expiry, the intent's
15-second deadline, the whole-host deadline or an active-work shutdown cutoff
withholds the result. Creation-window expiry is not applied to owned cleanup.
Injected clocks are trusted dependencies, not independently authenticated clocks.
Blocking filesystem calls can exceed a deadline; detection cannot undo elapsed
time or provider charges. No provider request is sent by this code.

Closure writes its record before descriptor cleanup, then checks every owned
close result. Both descriptors are attempted even if the first close fails; no
failed descriptor number is retried. A failed journal receives no additional
append on close. Repeated successful wrapper close is a no-op. Closing without
modeled terminal evidence still reports reconciliation required.

## Evidence and crash semantics

Records contain only fixed event/error codes, typed aggregate model counters,
known states, intent timing/sequence and SHA-256 bindings. Instance IDs, raw API
responses, IPs, names, credentials, approval text and raw exceptions are absent.
The transient model still holds its bound ID for the current process; hashing
it in evidence is not encryption or secure erasure.

The pure verifier accepts bounded supplied bytes, checks canonical schema/types,
contiguous sequence, hash-chain/counter consistency, target-hash stability and
external plan/final-tail pins. A valid prefix cannot detect suffix removal unless
the expected tail is supplied independently. A close record proves neither that
the close syscalls succeeded nor that a decision was actually returned or sent.
Verification is structural only and always denies dispatch/resume authority.

After a crash, a reservation may have been persisted immediately before token
return, or a response may have been received but not persisted. Neither case can
be retried from this journal. Because IDs are not retained, the journal cannot
reconstruct targeted teardown. A separately authenticated external controller
and authoritative reconciliation are still needed. Never guess a target from a
reusable instance name. Actual provider termination, billing stopped, global
exclusivity, encrypted storage, controller survival, execution and production
eligibility all remain unverified/false.

## Verification checkpoint

| Check | Result |
| --- | --- |
| Exclusive journal, ownership and storage faults | 36 tests passed |
| Durable controller, clocks, persistence ordering and structural audit | 37 tests passed |
| Both new modules under ordinary pytest | 73 passed; 39 subtests passed |
| New plus previous lifecycle/observer/guard/ledger/credential/status/IPC suites | 577 passed |
| Frozen safety/control suites | 273 passed; no skips |
| Exact Ruff on all four new Python files | Zero findings |
| Frozen lint | 815 historical findings/163 files unchanged; zero new |
| Private readiness | 327 OPEN; zero private tests executed |

There are 850 unique targeted tests including frozen controls. Repeated test
environments do not add unique cases. Tests cover actual temporary-file/directory
fsync and an abrupt test-process exit, plus injected short writes, storage/close
failures, owner/reentrancy races, post-sync approval/deadline expiry, withheld
unsynchronized snapshots, consumed crash prefixes and tail-pin tampering. All
provider responses and decisions are synthetic; no live provider endpoint is used.

This is a local pre-publication checkpoint, not a preclaim of the new commit's
Linux Source or Spark outcome. Those exact CI results must be verified separately.

## Next boundary

GPU provisioning remains off. The next integration is the authenticated, bounded
provider transport and external control process, with current account/price and
fresh creation-plus-targeted-teardown authority. Durable local reservations alone
do not keep a controller alive or stop a cloud invoice.

Dedicated host/boot/subtree ownership, real aggregate memory/descendant/network
enforcement, the real-worker profile, authentic S3/DCP import, private-data
relocation, CUDA acceptance and the complete execution ticket still remain.
The S3-backed pretrained Lumi base is unchanged.

Roadmap: gates 1-4 complete, Gate 5 partial, gates 6-8 pending. The 327 private
OPEN nodes are not roadmap steps. No GPU/SFT, restore, private source input,
registry, promotion, seal, remote login or instance action occurred.

- Inventory SHA-256: `441390c9c98cd4b83d224055a5af5ae421e5ef01b2e4b86101e7b17693d87665`.
- Partition SHA-256: `4e595b654880668a4462c7b8db7c40eaf715ebd86b90eb14cdf852720bc831ae`.
