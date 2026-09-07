# SFT execution status v0.25

Local engineering checkpoint, 2026-09-07. This successor adds only a canonical
in-memory bootstrap encoder, a prefilled anonymous-pipe handoff and synthetic
tests. It does not complete secure real authority provisioning or crash recovery.
No real key/token acquisition, provider/account request, child launch, instance
action, GPU, training, model/data read or S3 restore occurred in these new tests.
The existing busy instance and S3-backed pretrained Lumi base are untouched.
Gate 5 remains partial; production eligibility remains false.

## Previous step verified

Separate-process controller commit `3a189ce62a83d88e81ba060d41279b6d7ba73b09`
was pushed and verified at that exact revision.
[Source CI](https://github.com/saidi19/lumi-model/actions/runs/34093458718) passed
all four jobs: 3,488 passed, 58 existing skips, 329 separately routed/deselected
nodes and 7,075 subtests. Selection accounted for 3,546 source nodes, while 327
private nodes were NOT EXECUTED.
[Spark CI](https://github.com/saidi19/lumi-model/actions/runs/34093458721) passed
all three synthetic tests, with no skips. These were not private integration,
training or production-eligibility results.

## Exact scope

Three new files only:

- `src/lumi/posttraining_sft_controller_bootstrap_v1.py`
- `tests/test_sft_controller_bootstrap_v1.py`
- This checkpoint.

All prior source, tests, launch contracts, plans, budgets, private inventories,
workflows and evidence remain unchanged. No real authority, approval, execution
ticket, credential store, recovery record, daemon deployment or launcher is made.

`encode_bootstrap` checks the existing frozen authority/plan/key/token bindings
and exact launch body, then encodes the existing canonical ASCII bootstrap
schema. It checks the frozen frame decoder before returning. The returned bytes
contain secrets and must never be printed, logged or put in arguments/environment.
This function is pure; it does not issue an HMAC, read credentials, check the
current creation window, claim an attempt or authorize execution. The child
retains all its existing fresh validation and durable-attempt checks.

`BootstrapPipe` performs all input validation before allocating descriptors. It
creates a nonblocking, close-on-exec anonymous pipe, makes exactly one write of
at most 32,768 bytes, and exposes its read end only after the full write and a
successful writer close. A short/zero write, EAGAIN, EINTR, EPIPE, setup fault or
close failure rejects without a retry or child launch. The schema ceiling is
not a guarantee of sufficient pipe capacity on every host: valid larger packets
can safely reject before launch. There is no blocking prefill, streaming task,
temporary plaintext file, fallback transport, resource claim or retry loop.

The handle is bound to its creating process/thread, rejects reinitialization and
has a fixed redacted representation. `fileno()` borrows its descriptor for the
frozen process-spawn API; the caller then closes the parent's copy. Alternatively
`take_fd()` transfers ownership once and the caller must close the transferred
descriptor. Neither operation launches anything. A borrowed descriptor must not
be read, closed, duplicated or modified by another consumer.

Descriptor access checks the original device/inode, FIFO type, read-only access,
nonblocking mode and close-on-exec flag. Ambiguous identity is rejected without
closing a potentially reused descriptor number. Close removes ownership before
the system call, never retries a failed close, and retains a failure result.
Explicit cleanup is required; no destructor decides later which fd to close.
A successful local close is not a child/provider/billing result.

These are cooperative trusted-runtime protections, not an adversarial-process
sandbox, fork containment or account-wide attempt exclusivity. The same payload
still supplies the key and its pin. Matching them proves consistency, not issuer
identity or key custody. Dropping Python references and closing pipes do not
prove encrypted storage, secure memory erasure or protection against privileged
inspection, core dumps or swap. No such claims are added.

## Verification checkpoint

The preceding controller/transport/safety suites were rerun: 794 passed. Frozen
control CI was rerun: 273 passed across seven suites, with no skips. Frozen lint
still reports exactly 815 historical findings in 163 unchanged files and zero
new findings. Private readiness remains 327 OPEN, zero private tests executed;
its blocked result is expected, not a claim of full CI or integration success.

The new suite passed 39 tests under stdlib-only unittest and 39 tests plus 108
subtests under ordinary isolated pytest. Exact isolated Ruff on both new Python
files passed with no findings. A separate reviewer wrote the independent tests
and identified a real Darwin portability defect: pipe endpoints can have
different device/inode identities. The fix validates each endpoint separately
and retains the reader's own identity; a deterministic regression exercises the
same condition on other platforms, including Linux CI.

Tests also cover malformed/bad-MAC/mismatched inputs before pipe allocation;
native pipe EOF with the frozen child decoder; one-write behavior; short writes
and EAGAIN/EINTR/EPIPE; configuration and closure faults; repeated initialization;
PID/thread/reentry rejection; flag tampering; and actual descriptor-number reuse.
Network, provider construction and subprocess APIs are denied in every new test.
The preceding regression suites still include their own synthetic local children;
the no-child-execution assertion applies to the new handoff suite only.

There are 1,106 unique targeted tests including frozen controls. Repeated runs
and subtests are not additional unique cases. The successor's Source and Spark CI
must be verified at its exact published commit; no future run is preclaimed by
this checkpoint.

## Operational handoff: not another readiness gate

This completes only the caller-side canonical pipe handoff. Before provisioning,
the still-missing decisions/evidence are independently operated control-host
infrastructure; trusted authority/key delivery; current account, protected busy
instance, region, image, SSH-key and price bindings; and an authoritative recovery
strategy for an ambiguous create or controller crash. A hash-only journal cannot
recover an unknown instance ID. Never replay create or infer an owned target from
its name. No existing instance may be touched to work around missing bindings.

Next obtain those operational choices and separately authorize any required
read-only account inspection. Do not ask the user to paste a credential or MAC
key into chat. Do not invent account identifiers, current prices, approvals or
recoverability from synthetic tests.

Full execution-ticket completion, real host/boot/subtree ownership, aggregate
resource enforcement, real-worker admission, authentic S3/DCP import, private-data
relocation and CUDA acceptance still remain. Roadmap gates 1-4 are complete,
Gate 5 is partial, and gates 6-8 are pending. The 327 private OPEN nodes are not
327 additional roadmap steps.
