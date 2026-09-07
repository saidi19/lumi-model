# SFT execution status v0.31

Metered Torch/DCP checkpoint-worker adapter, 2026-09-07. This successor connects
the checkpoint reader, serializer views, saver, staging verification and exclusive
publication to actual model/optimizer checkpoint hooks. It does not turn the
frozen training entrypoint into an admitted, fully metered training launcher.

The user's Mac restriction remains in force: source edits, isolated lint/AST and
tiny standard-library synthetic-file tests only. Real Torch acceptance belongs
exclusively to the existing remote Linux Source CI. No actual model/data reads,
training, checkpoint conversion, large download, GPU, SSH, S3 restore or instance
action is authorized by these checkpoint hooks. Production eligibility stays false.

## Predecessor and successor boundaries

The v0.30 predecessor is commit
`e432067b9018ce0dfcc36ac1ca1e90d9ab0cbc74`. Its exact
[Source CI](https://github.com/saidi19/lumi-model/actions/runs/34116108672) passed
all four jobs: the main suite reported 3765 passed, 58 skipped, 329 deselected and
7601 subtests. Source-selected tests were 3823; 327 private integration nodes were
NOT_EXECUTED and two historical nodes were routed separately. Private/full-CI
claims remained false. Its exact
[Spark CI](https://github.com/saidi19/lumi-model/actions/runs/34116108643) passed
three synthetic tests with zero skips. That commit's 280 local tiny tests passed.

Only seven new files belong to this successor:

- `src/lumi/posttraining_sft_checkpoint_view_v1.py`
- `src/lumi/posttraining_sft_checkpoint_worker_v1.py`
- `tests/test_sft_checkpoint_view_v1.py`
- `tests/test_sft_checkpoint_worker_v1.py`
- `tests/test_sft_checkpoint_worker_torch_v1.py`
- `scripts/run_sft_checkpoint_worker_torch_fixture_v1.py`
- This status document.

Earlier source, evidence, private artifacts, tests, workflows and dependency
declarations remain unchanged. Unrelated untracked work is not included.

## Actual checkpoint hooks

`CheckpointWorker` requires an already-admitted caller, an already-imported Torch
runtime, explicit rank/world size, immutable copied limits, one shared reservation
service and a shared interrupt service. The versioned adapter accepts the 2.10
and 2.14 Torch API families only; remote acceptance must establish compatibility
with the actual installed version. No dependency or first ML import is authorized
by constructing this pure-standard-library object. Public methods alone are not
admission, and no stock worker globals are monkey-patched.

The caller lends a private, exclusively controlled parent directory descriptor.
Rank zero creates a fresh hidden staging directory and model subdirectory. DCP
writes model state through the injected FD-relative backend, using one file per
rank and one writer thread. Each rank writes optimizer/RNG state with `torch.save`
to a metered file-like stream. Closed-writer receipts are gathered before rank
zero writes bounded state JSON, finalizes the exact payload set, and invokes the
existing no-replace publisher. All content writes, manifest creation and repeated
verification reads share the same prepaid reservation callback. Failure interrupts
once and is terminal; staging/published evidence is retained without retry,
overwrite, rollback or pruning.

DCP receives explicit storage objects, never a native `checkpoint_id`. Its path
is an opaque label. DCP's host-wide `os.sync` fallback is disabled with
`sync_files=False`; the owned streams perform targeted file sync instead. The
backend permits only its rank's exact payload names and rank-zero metadata.
The metadata temporary file can move only to its exact final name through one
native no-replace operation. Opened file identity is checked against the final
pathname before a receipt is accepted. Native rename support remains required.

Restore first fully verifies the supplied manifest pin through the existing
metered session. DCP and `torch.load(weights_only=True)` then read only through
bounded file-like views over that session. Each view has an independent cursor
and bounded nested slices; standard seek-from-end semantics are preserved.
Repeated, overlapping and read-ahead bytes are charged again. No descriptor,
native path, mmap interface or transformation/decompression extension is exposed
to the serializers. A view failure poisons the shared session. Closing a view
does not close sibling views or the owned session.

The returned RNG object remains rank-local. The training caller must restore the
appropriate CPU/CUDA RNG state; the adapter does not guess real-worker RNG layout.
Optimizer restoration and model-state installation failures are terminal and can
leave partially changed in-memory objects; the caller must retire those workers.
They must not continue training after an unsuccessful restore.

## Accounting and trust limits

The worker's read cap accumulates across save, verify, publish and repeated
restore calls. Negative, non-integer, cross-rank, unknown-resource or nested
reservations are rejected. Reinitializing an existing worker cannot reset its
failure state or counters. File/call/sync bounds are delegated to the pinned
streams and finalizer; total writes and run costs require the actual shared
controller or acknowledged IPC service, not independent per-rank copies.

Logical byte caps do not bound serializer tensor allocations, metadata object
allocations, process RAM, collective transport, namespace/journal traffic or
total physical storage I/O. State JSON has a serialized-output cap, not a
pre-allocation memory quota. Sync caps are component-local rather than a new
process-shared global sync ledger. A bounded process group and independent
guardian are necessary for rank death, collective timeouts and descendant cleanup.

Only trusted, own-generated DCP metadata is admissible: the pinned DCP reader
uses pickle for metadata. A matching digest is not origin authentication or safe
unpickling. Fixed worker failure-stage/error codes can be reported; suppressed
exception contexts and third-party traceback text must not be logged. Scoped
Python DCP/c10d loggers are disabled while the adapter runs. This does not claim
containment of every native library log or of an unrelated concurrent thread.

Directory metadata checks detect observed mutations but are not filesystem
locks. The parent is borrowed; the caller must keep it stable and exclusively
controlled. Native name-based rename and same-inode descriptor reuse retain the
predecessor's documented concurrency limits. Published checkpoint files are
synthetic/research artifacts, not production promotion or dataset sealing.

## Verification scope

Local verification uses a fresh `python -S -B` process and explicitly checks that
Torch, Transformers, NumPy, safetensors and tokenizers were not imported. It covers
the new independent views and worker backend/guard tests together with the frozen
reader, saver, publisher, metered I/O, Supervisor and ledger protocol tests.
The final local run passed 367 tests (49 new view, 38 new worker and 280 prior
tests), zero failures/errors/skips, in 0.240 seconds; `ML_MODULES_LOADED=False`.
Exact isolated Ruff on all six new Python files passed. The worker/view source
and both remote-only fixture files also passed AST parsing without ML imports.

The new Linux-only CI test launches one isolated, 90-second-bounded subprocess.
It creates only an eight-parameter Linear model, Adam state and generated CPU
inputs. It verifies real Torch/DCP save/restore, model/optimizer/RNG continuation
equality, charging of repeated restores, wrong-hash rejection, read/write caps,
and preservation of an existing checkpoint on a name collision. Spies reject
serializer native paths, stock filesystem I/O, removal and host-wide sync.
The fixture emits only structural aggregate results and fixed failure codes.

This fixture uses the real in-process Supervisor but only rank zero of its two
available schema slots. It is deliberately single-rank checkpoint acceptance,
not distributed training or a multiprocess shared-controller proof. No GPU is
enumerated or initialized. The fixture is never executed on the user's Mac;
local treatment of these two CI files is lint and AST parsing only.

This successor's exact remote CI result must be observed after publication;
source inspection and a planned test do not imply a successful Torch round trip.

## Remaining path before H100

This completes the checkpoint-adapter source connection, subject to acceptance;
it does not close the full pre-GPU worker-integration deliverable. Still required:

1. Connect admission, every encoded-input read, these checkpoint hooks and shared
   optimizer transitions inside the versioned training entrypoint; prove the
   multiprocess controller and failure lifecycle together.
2. Establish training-capable Linux process/cgroup, memory/output/deadline and
   network enforcement. The earlier inert sandbox is not Torch acceptance.
3. Prepare real-size CPU checkpoint import and data relocation with complete
   tensor/tokenizer checks on an authorized suitable host, never this Mac.
4. Bind the full source/dependency/runtime release and target-specific smoke,
   resume, fault, stop and H100 handoff procedures.

The original roadmap is unchanged: pretraining/S3 continuity remains preserved;
private-research SFT data/shards are prepared, with production rights/owner gates
still partial. Actual 4B SFT, evaluation and product serving remain ahead. These
are engineering deliverables, not hundreds of new gates derived from test counts.
No H100 rental is needed for this source/remote-synthetic checkpoint step.
