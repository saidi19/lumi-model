# SFT execution status v0.33

Input accounting and frozen-loader connection, 2026-09-08. This successor adds a
mandatory encoded-metadata read category and routes the frozen v4 bounded train
loader through the cumulative parent ledger. It does not authorize actual inputs,
enable a real training entrypoint or close production rights/owner gates.

The user's Mac restriction is preserved: source inspection/edits, tiny stdlib
tests and isolated lint/AST only. Neither the remote-only test wrapper nor its
Torch fixture was executed locally, even in skip mode. No actual checkpoint or
dataset was read, GPU rented/used, SSH/S3 operation performed, dataset sealed,
registry changed, real inference requested or real SFT launched.

## Exact predecessor and scope

Predecessor `1cc2a67f14683c336969814bd405c164ea3f6ba0` passed all four jobs in
[Source CI](https://github.com/saidi19/lumi-model/actions/runs/34124619618):
3940 passed, 58 skipped, 329 deselected, 7749 subtests, 233.34 seconds. The source
partition selected 3998 tests, left 327 private integrations NOT_EXECUTED and
routed two historical failures separately; private/full-CI claims remained false.
Its [Spark CI](https://github.com/saidi19/lumi-model/actions/runs/34124619606)
passed three tests, zero skips, in 100.46 seconds. Earlier failed runs are retained.

Only these fifteen new files belong to this successor:

- `src/lumi/posttraining_sft_launch_contract_v2.py`
- `src/lumi/posttraining_sft_supervisor_v2.py`
- `src/lumi/posttraining_sft_epoch_protocol_v2.py`
- `src/lumi/posttraining_sft_worker_client_v2.py`
- `src/lumi/posttraining_sft_input_reader_v1.py`
- `src/lumi/posttraining_sft_worker_input_v1.py`
- The corresponding six stdlib test files under `tests/`.
- `tests/test_sft_worker_input_torch_v1.py`
- `scripts/run_sft_worker_input_torch_fixture_v1.py`
- This status document.

Earlier implementation, tests, contracts, workflows and evidence are unchanged.
Unrelated untracked work is not included. The v1 public admission entrypoint and
the legacy training launcher remain closed; this successor does not invoke them.

## Explicit metadata cap; no inherited approval

The v2 launch contract and authorization have distinct schemas. Completed cap
objects must include a positive, bounded integer
`maximum_encoded_metadata_read_bytes`. The entire contract, including that cap,
is part of the authorization subject. Old schemas/approvals cannot authorize the
new category, and changing its allowance invalidates the bound subject. Drafts
retain null resource caps and authorization; execution denial stays unconditional.

The v2 Supervisor reserves `encoded_metadata_read_bytes` in the same locked,
cumulative, nonrefundable transaction as other resources. No rank-local ledger,
separate metadata allowance, reset, refund or fallback to index/checkpoint bytes
is introduced. Mixed reservations remain atomic: an over-cap category consumes
the lifecycle but commits none of that attempted reservation. Earlier charges
remain consumed across ranks and all three planned invocations.

The v2 epoch protocol/client bind versioned request, ACK and completion-fence
schemas. Existing sequencing, exact ACK verification, confirmation, ancillary
prepayment/rights rejection, absolute deadlines, sticky failures and successful
retirement semantics remain unchanged. V1 request/ACK/fence frames reject. The
new client summary has an explicit v0.2 schema. Capability authentication is not
OS peer identity, qualified owner approval or permission to read a real dataset.

Only own-generated tiny fixture allowances were supplied. No real-data numeric
cap, production authorization, refreshed target receipt or runnable launch
contract was issued. Structural support is not execution approval.

## Exact content reads and unchanged validation

`InputReadHandle` owns an already-validated private, regular, single-link,
close-on-exec O_RDONLY descriptor. The inherited meter verifies descriptor
identity before and after operations. Every content pread is preceded by its
parent reservation. Metadata requests are bounded by the frozen 16-MiB-plus-one
read and split into at most 1-MiB operations; selected binary calls retain the
4095-token-times-four limit. These are code bounds, not default real-data caps.

Index `readline` reserves its row before the first byte, then reads at most 176
one-byte chunks until newline/EOF. It never reads the next row and seeks back or
retains an uncharged read-ahead buffer. Short reads, cap denial, mutations and
callback reentry fail terminally; no retry or refund is permitted. The handle
exposes no public fileno, raw/buffer, path or seek API.

`WorkerInput.load` requires all five exact external hashes, a train prefix from
1 through 12, an already-imported supported Torch/v4 runtime and caller-admitted
root/memory/parent capabilities. It makes no first ML import. Validation content
opens and unknown filenames reject; mandatory metadata inspection of the whole
train/validation inventory is retained. Root/file private modes are enforced even
before the manifest identifies its strict profile. Nonblocking opens additionally
prevent a raced FIFO from hanging before regular-file validation.

The loader's exact code object executes with a private globals copy replacing
only its opener/finisher. A separately copied Dataset initializer resolves that
loader; neither original function/class nor module globals are mutated. The
result is the genuine frozen `EncodedSFTDataset`, including its unchanged
in-memory indexing. The full five-hash inventory, schema, masking, token limits,
provenance and other frozen loader checks remain in force. This exact bounded
worker wrapper additionally preserves the zero-fully-hashed-shard requirement;
it is not a general substitute for all legacy v4 profiles.

File finish compares both the owned descriptor and no-follow directory-relative
path identities. A nonthrowing cleanup proxy records close failures as terminal
before interruption, allowing the frozen loader's finally to reach its sole root
descriptor close. This does not accept or retry a failed close. A failure while
registering an already-transferred reader closes that reader rather than leaking
its untracked descriptor. Successfully retired handles cannot interrupt a later
epoch. Only fixed error codes and aggregate summaries may be published; source
paths, tensors, record hashes and suppressed exception contexts remain private.

## Verification

The combined local run passed 740 tests, zero failures/errors/skips, in 0.377
seconds: 453 predecessor tests plus 287 new checks (123 launch/Supervisor, 108
epoch/client, 35 reader and 21 WorkerInput guards). A fresh `python -S -B`
process reported `ML_MODULES_LOADED=False`. All fourteen new Python files passed
isolated Ruff; both Linux-only files passed AST parsing without execution.

Coverage includes required/missing/invalid metadata caps, old authorization and
protocol rejection, cross-rank/cross-epoch accounting, atomic overflow failure,
prepayment before every pread, newline/EOF bounds, sticky short reads, descriptor
mutation/reuse, owner/reentry guards, file/root cleanup and retired-epoch safety.
Inherited tests are bound to successors in isolated test namespaces; historical
test files and imported production globals are not modified. These local tests
do not emulate a successful Torch execution.

The Linux-only acceptance fixture uses two spawned loader workers and one real
Supervisor v2/Epoch v2 parent. Each worker selects one of two records in the first
generated train shard. It checks actual CPU tensors and unchanged loader results
against two additional direct-v4 reference loads. Every controlled pread must
already be charged; validation/unselected content opens are denied; repeated
Dataset indexing runs with disk APIs denied. Expected controlled totals are two
metadata manifests, 48 binary bytes, 316 index bytes, two row reservations,
eight content opens and 324 confirmed ledger requests.

Workers stay alive until the parent processes every confirmation. The fixture
then deliberately interrupts its one active invocation, verifies failure state,
releases workers and reaps its own processes. It reserves/completes zero optimizer
steps and completes zero training invocations. Generation and the two independent
reference loads are explicitly outside the shared worker ledger. Its 120-second
inner and 150-second process-group deadlines bound fixture cleanup, not the full
training runtime. Native output is suppressed; only fixed codes/aggregates leave
the fixture. Actual Linux acceptance must be observed on the published commit;
static/local results alone are not that acceptance.

## Limits and next work

Byte reservations are not physical I/O, allocation, syscall, CPU-time or OS network
quotas. Inventory list/stat and JSON/tensor allocations still require independent
training-capable Linux enforcement. No complete three-invocation training/resume
lifecycle, real source import/relocation or target-H100 acceptance is demonstrated.

The metadata-contract gap and input-loader source connection are addressed by
this successor, subject to remote acceptance. Full worker admission, checkpoint,
input and optimizer orchestration/failure lifecycle still must be joined and
tested as one entrypoint. Training-capable Linux controls, suitable-host real-size
import/data relocation and the complete runtime/H100 handoff remain the other
pre-GPU deliverables. Their size cannot be inferred from unit-test counts.

Roadmap 5's private-research data preparation is separate from production rights
and owner approval, which remain partial. This work prepares the 5-to-6 handoff;
it is neither additional source curation nor completion of real SFT. Production
eligibility stays false. No H100 rental is needed for this source/remote-synthetic
step, and no reliable calendar completion estimate follows from this report.
