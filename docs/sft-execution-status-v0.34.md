# SFT execution status v0.34

Admitted-object three-invocation integration, 2026-09-08. This successor joins
the bounded train-prefix loader, optimizer-step gate, checkpoint storage and
versioned cursor/RNG resume under one cumulative parent controller. It does not
open the public real-training entrypoint or authorize real inputs/GPU use.

The Mac restriction remains intact: source inspection/edits, tiny stdlib tests
and exact lint/AST only. No local Torch, numpy or transformers import; no actual
model/dataset read, S3/SSH/instance operation, GPU rental/use, real SFT, inference,
registry mutation, dataset seal or production promotion. The Linux-only wrapper
and fixture are not executed locally, even in skip mode.

## Predecessor and exact scope

Predecessor `91d128561b974099e295f1f6a00e35a5652a74ab` passed all four jobs in
[Source CI](https://github.com/saidi19/lumi-model/actions/runs/34190410615):
4228 passed, 58 skipped, 329 deselected, 8277 subtests, 247.36 seconds. Source
partition selection was 4286; 327 private integrations remained NOT_EXECUTED,
with two historical failures routed separately. Private/full CI remained false.
[Spark CI](https://github.com/saidi19/lumi-model/actions/runs/34190410651) passed
three tests, zero skips, in 107.21 seconds. Earlier failed evidence is retained.

Only ten new files comprise this successor:

- `src/lumi/posttraining_sft_training_state_v1.py`
- `src/lumi/posttraining_sft_checkpoint_state_v1.py`
- `src/lumi/posttraining_sft_worker_run_v1.py`
- `tests/test_sft_training_state_v1.py`
- `tests/test_sft_checkpoint_state_v1.py`
- `tests/test_sft_worker_run_v1.py`
- `tests/test_sft_optimizer_binding_v1.py`
- `scripts/run_sft_worker_run_torch_fixture_v1.py`
- `tests/test_sft_worker_run_torch_v1.py`
- This document.

Earlier implementations, tests, workflows, contracts and evidence are unchanged.
Unrelated untracked files are excluded. There is no automatic retry after a
started invocation, no exception-to-pass conversion and no lowered CI floor.

## Connected worker lifecycle

`WorkerRun` is one-shot and accepts already-admitted objects, an exact v2 ledger
client, all five data pins, explicit checkpoint caps and trusted admission,
synchronization and interruption capabilities. It is not an OS admission proof.
Only already-loaded frozen runtime/planner/LR/RNG helpers are resolved. The public
v1 admission entrypoint and legacy top-level training launcher stay closed.

The schedule is limited to 0-to-2 steps, 0-to-1 steps or fresh 1-to-2 resume. The
parent owns the fixed reference/resumed/resumed sequence across three separate
epochs. WorkerRun does not finish the parent invocation itself. Input and
checkpoint reservations use the same client; limits are not reset between
operations or replaced with category defaults. The controller retains combined
charges across ranks and all three invocations.

Each optimizer update uses the existing StepGate/WorkerStep. The worker consumes
the explicit parent's all-rank completion fence before advancing cursor/counters
or saving the checkpoint. Checkpoint I/O therefore occurs after step completion
but before invocation completion. Input/checkpoint phase services must also
drain parent ACK confirmations before permitting the next step boundary; a rank
barrier alone does not guarantee that drain. The fixture implements explicit
parent phase-ready/release messages. Terminal failures interrupt once, never
retry or refund, and require independent peer teardown.

The frozen deterministic planner, rank-local training seed, LR schedule,
assistant-only collator, unequal-supervision weighting, accumulation, clipping
and optimizer ordering are preserved. Dataset indexing reads in-memory tensors,
not additional files. No raw metrics stream, pruning or legacy state conversion
is introduced. Returned state/receipts remain private; summaries expose fixed
codes, counts and explicit nonclaims only.

## Resume and optimizer compatibility

The new state schema has exact keys, steps 0 through 2, opaque run identity,
global cursor/record geometry, aggregate token totals and finite loss values.
It accepts only the exact checkpoint-name/status pair added by the existing
CheckpointWorker. A state advance requires one completed WorkerStep's exact
aggregate mapping; counters are not reweighted or multiplied again.

The checkpoint-state bridge verifies the pinned checkpoint and reads only its
fixed state file, at most 65536 canonical ASCII JSON bytes including newline.
Duplicate keys, nonfinite values, noncanonical bytes and unsupported shapes fail.
Verification and the subsequent read use the existing worker's cumulative read
ledger. The parent descriptor is borrowed; child/session descriptors are owned
and closed. This adds no separate allowance. The later model restore deliberately
performs its own verification, charged to the same ledger.

Identity/cursor validation precedes model/optimizer restore. The saved rank RNG
is then restored through the frozen helper. Identity binds all five input pins,
source identity, profile, world size and selected prefix, but not the invocation
split. A second compatibility digest binds ordered model parameter names,
shapes, dtypes and trainability, plus ordered optimizer groups and their plain
settings. Thus changing Adam-slot order, betas, decay or other group settings
cannot silently reuse a saved state. No tensor values or optimizer state are read
to construct that metadata digest.

This initial integration requires DDP and exact, initially empty AdamW; FSDP and
optimizer resharding are rejected, not inferred from lower-level capabilities.
The source identity and actual model-initialization provenance remain the trusted
caller's responsibility. A matching metadata digest does not authenticate model
weights, arbitrary code or DCP pickle. Restore is for trusted own-generated
checkpoints only; no safe-unpickling or external-checkpoint admission is claimed.

## Verification and remote acceptance boundary

The exact local combined suite passed 575 tests with zero failures/errors/skips
in 0.393 seconds. This includes 96 new checks: 21 state, 22 checkpoint-state,
31 WorkerRun guards and 22 optimizer-metadata checks. The remaining 479 are
related predecessor regressions. A fresh `python -S -B` process confirmed
`ML_MODULES_LOADED=False`. Local tests do not emulate successful Torch execution.
All nine new Python files passed isolated E/F/I/B/UP lint; both Linux-only files
passed AST parsing locally, without importing or executing them.

The generated Linux fixture is the first connected acceptance attempt: one
Supervisor v2, three fresh Epoch v2 sessions and six fresh rank processes. It
uses two-rank CPU DDP, AdamW and actual DCP storage, with a tiny float64 causal
model whose dropout consumes rank RNG. The reference runs 0-to-2; the second run
stops at 1; a third fresh pair restores state/model/optimizer/RNG and runs 1-to-2.
No independent extra optimizer reference is run outside the shared ledger.

Expected totals are four global/eight rank optimizer updates, three completed
invocations, six input-prefix loads, twelve selected rows, 288 binary bytes,
1896 index bytes, six manifest reads, four published checkpoints, two rank
restores and eight step fences. Prefix two of a three-row shard preserves zero
fully-hashed shards. All worker checkpoint I/O is metered. Own-generated fixture
creation and bounded private comparison IPC/CPU work are outside that ledger.

Generated model/Adam/state snapshots stay in inherited private pipes and memory;
only aggregate booleans/counts leave. Floating comparisons use relative tolerance
1e-9 and absolute tolerance 1e-11; rank RNG bytes and integer counters compare
exactly. The parent drains confirmations and validates both reports, observes
RSS, seals the epoch, releases workers and reaps them before the next invocation.
Inner 180-second and outer 210-second deadlines target only the owned fixture
process group. Fixed component diagnostics never include exception text.

Remote success must be observed on the published commit; these local/static
results alone are not multi-rank Torch/DCP acceptance. No workflow or exclusion
change is part of this successor.

## Remaining handoff work

Even a successful generated lifecycle is not full runtime admission. Cooperative
byte counters and RSS snapshots are not syscall/network, CPU, memory/cgroup,
physical-I/O or complete monetary enforcement. Training-capable Linux resource
and parent-death controls, real-size source import/relocation on a suitable host,
and the actual runtime/H100 handoff remain separate deliverables. CUDA/FSDP and
real-model correctness are not established by CPU DDP acceptance.

Roadmap 5's private-research preparation remains separate from production rights
and owner review. This advances the 5-to-6 technical handoff, not real SFT or
production eligibility. Production remains false. No H100 needs to be rented for
this source/remote-synthetic step; test counts do not justify a calendar ETA.
