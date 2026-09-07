# SFT execution status v0.32

Shared optimizer-step worker connection, 2026-09-07. This successor connects an
explicit SFT step consumer to one parent-owned ledger through authenticated
epoch messages and committed-step fences. It does not enable the old training
launcher or real encoded-input loading. The input audit found a missing encoded
manifest read-budget category; that separate gap is recorded below, not hidden
by charging another resource or removing a check.

The user's Mac restriction is unchanged: source edits, isolated lint/AST and tiny
standard-library tests only. Real Torch/DDP acceptance runs only in remote Linux
Source CI. No real model/data read, GPU use/rental, SSH, S3 restore, instance action,
conversion, dataset seal, production promotion or training launch is authorized.

## Exact predecessor and files

The predecessor is `303b7a817743ff15197b9feee2792144401ad6bb`, with exact
[Source CI](https://github.com/saidi19/lumi-model/actions/runs/34120168790) success:
3861 passed, 58 skipped, 329 deselected, 7686 subtests, 229.79 seconds. Its four
jobs passed. Source-selected tests were 3919; 327 private integration nodes were
NOT_EXECUTED and two historical nodes were routed separately. Private/full-CI
claims remained false. The exact
[Spark CI](https://github.com/saidi19/lumi-model/actions/runs/34120168763) passed
three synthetic tests with zero skips in 104.87 seconds. Local predecessor tests
passed 375 checks. Its earlier test-comparator failure remains in Git/CI history.

Only these nine new files belong to v0.32:

- `src/lumi/posttraining_sft_worker_client_v1.py`
- `src/lumi/posttraining_sft_step_gate_v1.py`
- `src/lumi/posttraining_sft_worker_step_v1.py`
- `tests/test_sft_worker_client_v1.py`
- `tests/test_sft_step_gate_v1.py`
- `tests/test_sft_worker_step_v1.py`
- `tests/test_sft_worker_step_torch_v1.py`
- `scripts/run_sft_worker_step_torch_fixture_v1.py`
- This status document.

Earlier source, workflow, contract, tests, evidence and private artifacts remain
unchanged. The public admission v1 `run_worker` remains unconditionally closed.
Nothing calls frozen top-level `train_distributed_sft` or its native-path model
load, invocation marker, metrics/state writers or checkpoint-pruning functions.

## Worker client and the completed-step boundary

`WorkerLedgerClient` takes an already-connected, exclusively owned AF_UNIX stream
socket and independently bound epoch/rank/token. It never spawns a worker or
constructs a rank-local Supervisor. Each operation prepays request, exact ACK and
confirmation data capacity before the first send. Every receive also prepays a
bounded ancillary allowance. The client rejects and closes all delivered rights,
rejects unknown/truncated control state and verifies the exact canonical ACK.
Short-I/O continuation retains one exchange deadline within an absolute session
deadline; failure cannot refund, replay or reset a transaction.

The frozen protocol's confirmation has no response. Sending it alone cannot prove
that the parent received it or completed the global step. A new versioned control
frame, `lumi.sft-worker-step-fence.v0.1`, therefore binds epoch, rank and step with
`status=committed`. It is separate from frozen epoch request frames. The trusted
parent sends it only after all rank confirmations have been processed, pending
ACKs are empty and the shared Supervisor has committed that exact step. The client
blocks further reserve/step operations after `step_complete` until this fence is
validated. A rank barrier without the parent fence is insufficient.

`StepGate` orders coordinator reservation, reserved synchronization, caller work,
all-rank updated synchronization, each-rank completion and parent-confirmed final
synchronization. Only then does its local cursor advance. It preserves the frozen
0-to-2-step invocation bounds; combined three-invocation accounting remains in
the parent. Callback exceptions, body failures, reentry, changed owner or repeated
initialization interrupt once and permanently consume the gate. Partial optimizer
updates cannot be rolled back; all affected workers must retire.

Socket capability authentication is not OS peer-PID authentication, parent honesty
or execution approval. Close is explicit; observed FD reuse cannot close a foreign
replacement descriptor. Successful retirement prevents a stale client callback
from interrupting a later epoch. Independent parent/descendant deadlines and
termination remain required; this client is not an OS sandbox or durable journal.

## Actual SFT step consumer

`WorkerStep.perform` consumes already-admitted model/optimizer objects and already
collated CPU tensors. It makes no first Torch import and performs no dataset or
checkpoint I/O. It retains frozen shifted-supervision weighting:
each microbatch loss is multiplied by world size times its shifted supervised
count divided by the global step supervised count. Accumulation, intermediate
`no_sync`, clipping, scheduled-LR assignment and optimizer-update order are retained.
Returned values are only step/loss/gradient-norm/token-count aggregates.

The consumer requires training mode, the actual DDP wrapper/default WORLD group,
or the frozen FSDP FULL_SHARD/use-orig-params configuration. CUDA FP32 is rejected;
CUDA must already be initialized and uses the frozen BF16 rule. The CPU acceptance
does not prove CUDA/FSDP execution. Torch 2.10/2.14 API-family checks are retained.
Optimizer parameter identities must be unique model members and cover every
trainable parameter; frozen parameters may be included or omitted to preserve
both historical optimizer profiles. Tensor equality is not used for membership.

Before forwards, ranks collectively compare a fixed SHA-256 settings identity
covering step, LR, clipping, strategy, precision, synchronization mode, device type
and world size. They also require equal microbatch counts. Invalid batch structure
and nonfinite returned loss/norm flags are reduced before progressing. Exceptions
inside DDP forward/backward or optimizer code can still strand peers; the independent
guardian/deadline must terminate the whole attempted operation, without retry.

Already-collated batches are not a substitute for encoded-loader provenance,
vocabulary, right-padding, prompt masking or rights checks. This consumer does not
change or certify those rules. Nor does it implement dataset planning, collation,
LR scheduling, RNG/cursor persistence, checkpoint orchestration or the complete
real-training entrypoint. The checkpoint adapter from v0.31 remains unchanged.

## Concrete input-accounting gap and preserved loader contract

Source inspection of frozen runtime v4 established that disk content is read in
`_load_verified_sft_split` during dataset construction. Later `__getitem__` calls
only slice retained tensors. Debiting file bytes again per batch would be false.
The content reads are manifest bytes, selected index lines and selected binary
token/label bytes. The existing audit and shared resource schema account for the
latter two, but not the encoded `manifest.json` content read. It is not checkpoint
data or an index row. No new input reads were performed to discover this gap.

The next input successor must bind an explicit cumulative encoded-metadata read
cap/category in the versioned launch/controller/protocol contract before actual
input dispatch. The cap is not implicitly inherited from another category. No
numeric real-data allowance or authorization is invented in this successor.

A source-preserving I/O seam exists: an isolated function object can reuse the
frozen loader's exact code with a copied globals mapping that replaces only its
secure file opener/finisher. All reads then pass through owned metered handles,
without changing stock module globals or validator code. This is a reviewed design
seam, not implemented/accepted input loading in v0.32.

That successor must preserve all five external hashes, complete metadata inventory
checks and the 1-to-12-record bounded train prefix. It must not read validation
content or claim full shard/content hashes. Index reads must stop exactly at the
newline: checkpoint-view read-ahead would cross the authorized prefix and cannot
be reused here. Setting `record_limit=None` or calling the old unmetered top-level
worker is not a workaround.

## Verification and limitations

The initial local combined run passed 452 tests, zero failures/errors/skips, in 0.247
seconds: 375 prior, 27 new socket-client, 32 new step-gate and 18 new consumer-guard
tests. A fresh `python -S -B` process reported `ML_MODULES_LOADED=False`. Isolated
Ruff on all eight new Python files passed. Both remote-only fixture files passed
AST parsing; neither their test nor Torch entrypoints were run on the Mac.

The Linux fixture uses two spawned CPU DDP ranks and one parent-owned real
Supervisor/EpochLedgerSession. It checks two controlled global optimizer steps,
four rank optimizer updates and eight controlled logical sample events. Unequal
shifted-supervision counts and both gradient-sync modes are checked against an
independent concatenated-global-batch reference at each step, using float64
tolerances. Model, Adam state, weighted loss, gradient norm and token totals must
match. Four parent commit fences are required, with all epoch ACKs confirmed.

The independent oracle additionally performs four reference optimizer updates;
that test-only reference work is not charged to the worker ledger. Whole-fixture
cost is not claimed metered. Likewise, batch generation and Gloo rendezvous are
fixture work, not encoded-data reads. Gloo uses loopback TCP plus private local
AF_UNIX control channels; external networking is not authorized, but total socket
bytes are not claimed zero. Bounded RSS snapshots cover the fixture parent and
two owned children, not continuous whole-host RAM enforcement. Only the first
of three planned invocations is exercised; there is no full-lifecycle claim.

The fixture has a 100-second inner deadline and 120-second outer process-group
deadline, owned-child cleanup, suppressed native stdout/stderr and fixed-code
structural diagnostics.

The first published commit, `49f51ee15fe3ae4d2270b9c0127b556f31c64425`, ran in
[Source CI](https://github.com/saidi19/lumi-model/actions/runs/34123478710):
3938 passed, one failed, 58 skipped, 329 deselected and 7749 subtests passed in
193.05 seconds. The real two-rank worker fixture passed; the only failure was
the synthetic client short-send/short-receive test, which reached the existing
deadline while sending its confirmation. The complete source run was therefore
not accepted. Source selection was 3997; the same 327 private nodes remained
NOT_EXECUTED and two historical nodes remained separately routed. Its exact
[Spark CI](https://github.com/saidi19/lumi-model/actions/runs/34123478697) passed
three tests with zero skips in 108.96 seconds. Both initial runs are preserved;
correcting a fixture does not turn the earlier failed run into a pass. The
corrected commit still requires its own observed remote acceptance.

The test-only correction gives the fragmented-send fixture a bounded concurrent
peer: it drains the exact request before sending ACK, then drains the exact
confirmation. Previously the fixture prequeued ACK and did not consume either
client frame until the entire transaction returned. A new finite, nonblocking
control using constrained 1-KiB socket buffers demonstrates that undrained
7-byte sends fill the queue and that draining allows sending to resume; it sends
at most 28 KiB. The corrected transaction uses those constrained buffers too and
checks every actual receive fragment's prepaid ancillary capacity. Production
client code, one-second transport limits and every accounting cap are unchanged.
After this correction, all 453 local tests passed with zero skips in 0.245 seconds
(28 client tests, the other counts unchanged), `ML_MODULES_LOADED=False`, eight
Python files passing isolated Ruff and both remote-only files passing AST checks.

The roadmap is unchanged. This advances shared optimizer integration; actual
input metering/admission, training-capable Linux resource enforcement, suitable-host
real-size import/data relocation and the full bound runtime/H100 handoff remain.
Production rights/owner gates are still partial. No H100 rental is needed for
this source and remote-synthetic step, and production eligibility stays false.
