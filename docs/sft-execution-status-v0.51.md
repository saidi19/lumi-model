# SFT execution status v0.51

Pre-publication plan, 2026-09-09. The user approved one bounded CPU-only
diagnostic after the v0.50 handoff, then asked to resume the same work. This is
that diagnostic successor, not a retry of the failed acceptance and not a
conditional recovery. The previous published commit is
`edaa86e238223cf3a1ecfb6292640d8bc43e72de`.

## Scope and frozen bindings

Add only eight new files: this plan, the new CPU-origin diagnostic workflow,
the new diagnostic builder/probe/pure report validator, and their three tests.
Every previous tracked file remains byte-identical. The new builder verifies
its direct parent and exact eight added paths before snapshot/build. GitHub
checkout fetches two commits so this check can resolve the actual parent.

The new paths do not match the old OCI acceptance workflow's push filters.
Therefore publication starts one new diagnostic workflow, plus the existing
Source and Spark regression workflows, but **not** a second old OCI acceptance.
The diagnostic is restricted to the current branch and GitHub run attempt 1;
no dispatch/retry/fallback is present.

Frozen source SHA-256 bindings:

- Software builder: `458faf97c62689c9d7cc71a6eb89a4f84fef8c68bbcf9279b949dee450565044`.
- OCI observer: `bed636e8cc08546a6be1b43c00c2f516fdc92dbf907dca674f7fb5aa7b5cb769`.
- WorkerRun fixture: `53c50c92965ed1cef0c9624c57367c31ed7ebfb9d7cfb34257188ad179d333d0`.
- Base inventory: `34dc6a70b6d074808b864f53b4d511456423786aa62068ea771253fd4f8f29d1`.
- Lock validator: `9f70737db527c038d2ddd54992ce6d1ff999e9c7f204f4828a3f64fa08543ed3`.
- Software lock: `26bde6a0163e5db9d8d4564117546482601f4d0beb09f8705be5fc9183594965`.

The old failed builder/probe reports remain exactly 565/501 bytes with hashes
`a0fce383c04853fcb7173b5d560b22f2818a650e88df7cab8c5f43174bddbc21` and
`aa1e334d868214c612b6fa31f4d7cea745613c50c0642b9e9407dac8746ef7f0`.
Do not delete, overwrite, or relabel that failure. Its cause is still unknown
when this plan is committed; v0.50 records the existing evidence.

## One diagnostic execution, unchanged predicates

The frozen builder is hash-loaded under a non-main module name. Only its
`PROBE` and `SOURCE_SCRIPTS` versioned bindings change: append the diagnostic
probe and make it the image entry script. Reuse `Builder.snapshot()` and
`Builder.image()` unchanged. Never call its `main()`, `accept()`, archive path
or image-save operation. The pinned Python3.12.14 base, 34 CPU wheels totaling
248,715,637 bytes, same-commit tracked-source snapshot, offline hash-locked
binary install, `pip check` and container restrictions remain the same.

There is one separate base-inventory container during construction and at most
one **diagnostic-container start**, not literally one Docker start overall.
The new builder writes an exclusive image-ID/control-pin/software-download
receipt immediately after image construction, before diagnostic creation.
It then inspects the exact created container and writes an exclusive start
marker before the sole start. The image ID is therefore preserved on failure.

The diagnostic probe hash-loads the old observer and calls the same original
predicates. It records which frozen `need()` call rejected a value, using fixed
function/check identifiers bound to the observer's exact source lines. Unexpected
origin-check exceptions are mapped to fixed class/function identifiers, never
exception messages. Lexical origin categories and broad module families are
diagnostic labels only; they do not change allowed paths or origin decisions.
Frozen source-line mappings are checked against AST in local tests, without
importing or executing the native source on the Mac.

The rank's failed-result packet carries only validated fixed labels and bounded
counts. A metadata-only `record_failure` hook captures that packet and still
calls the original handler. The original bounded failure-collection pass,
rank implementation, inputs barrier, synchronization, ledger, optimizer and
cleanup are unchanged. The observer also exposes the frozen handler's existing
worker-stage/code and seven component diagnostics. It does not add another
training pass or modify the rejected condition.

Counts represent observed parent state, validated received packets and completed
frozen invocations. They are not proof of unseen child work. Byte-identical
failure packets may be deduplicated; counts are not independent rank identities.
The observed-check telemetry counter saturates rather than adding a new
acceptance rejection. The original maximum is still three invocations, six
rank workers, twelve origin observations and four global/eight rank updates;
an earlier failure stops the same frozen workload.

## Report validation and outcome semantics

Only a closed, canonical ASCII JSON report with exactly one LF is retained as a
completed diagnostic. A pure supplied-byte validator extracts its eight literal
schema/enum/count constants from hash-bound source **as AST only**. It never
imports native code, reads an image, or treats validation as an execution permit.
Reports require exact control pins, exact key sets, bounded genuine integer
counts (not booleans), fixed enum values and all eleven nonclaims false.

`diagnostic_completed` means either a specific causal observer rejection was
captured or the unchanged underlying observer fully passed. `underlying_status`
separately records passed/failed. An unexplained orchestration failure remains
`diagnostic_incomplete`; it must not be called a diagnosed cause. In all cases
`acceptance_passed=false`, `recovery_performed=false`, production/real SFT false,
complete dynamic closure false and generated/anonymous-code authentication false.
No admission, policy, parser, vocabulary, floor, rights or evaluation rule changes.

After its one start, the builder requires a valid completed report, exact
container exit 0, no OOM and no writable root-layer diff before recording
completion. It retires the exact owned container even after failure, using a
separate cleanup budget. It never retains an image archive or initiates a
recovery run. A later result receipt must state the actual diagnosis and limits.

## Bounds and privacy

- Work deadline 14 minutes, independent 15-second exact-owned cleanup, outer
  15-minute timeout plus 15-second kill grace, workflow cap 20 minutes.
- One pinned software acquisition/build; no wheel redirect, fallback or retry.
  Wheel cap 64 / 1 GiB, actual fixed lock 34 / 248,715,637 bytes. Base layer,
  source, file-inventory, binary-install and image-size checks are inherited.
- Diagnostic container: nonroot10001, read-only image, network=none, all caps
  dropped, no-new-privileges, no host software/source binds or device requests,
  2 CPUs / 4 GiB memory with no extra swap / 256 PIDs, 512 MiB temporary files
  and 64 MiB shared memory. Docker's generated hosts/config mounts are not
  falsely described as no mounts at all.
- Diagnostic stdout at most 16 KiB; failure packet at most 4 KiB. Only fixed
  error/stage/category labels, aggregate counts and software control hashes.
  No source text, candidate rows, model/data values, module names, paths, PIDs,
  credential text or exception messages are emitted by the probe.
- Small retained output total checked at most 32 MiB; software-only command
  failure logs at most 64 KiB each. This is not a Docker daemon-wide disk quota.
  Private GitHub evidence retention 30 days. No software-image archive,
  registry push or large download onto the Mac.
- Mac work remains source edits, pure stdlib/synthetic tests, AST/literal tests
  and exact-file lint. No native fixture/builder/observer imports or runs, even
  platform skips; no local Torch/model/data/runtime work.

No GPU/H100/instance action, S3/SSH, real checkpoint/data access, real-model
inference, real SFT, promotion, seal, registry mutation, automation or unrelated
action. Gate 5 remains partial until the underlying failure is properly fixed
and accepted, then integrated worker/guardian admission is completed. The
original pretrained S3 checkpoint and dedicated H100 work remain downstream.

## Pre-publication source verification

The three new pure/AST test modules plus the eight frozen OCI/release pure/AST
regression modules passed: **353 tests**, 0.381 seconds, Python `-S -B`.
Exact six-file Ruff checks passed with the frozen CI selection `E,F,I,B,UP`,
line length 100 and Python 3.11 target. An independent read-only source review
found no architectural blocker. No native diagnostic was imported or executed.

Final diagnostic source SHA-256:
`492c663e8c5cd0ae01d0fa44919ec46472dfb63b5f5aac74a6b1e3692d8909d0`.
Pure report validator SHA-256:
`d1ec96cd14636eb87e75c4156c7db40891f26d8764de183247434ccacfb0facf`.
Both are pinned in the new builder. Publication and native outcome remain
unclaimed until the single remote diagnostic returns fresh evidence.
