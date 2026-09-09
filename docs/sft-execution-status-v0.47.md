# SFT execution status v0.47

2026-09-09. Pre-publication release-gate ordering successor. This records source
and synthetic acceptance only; remote native acceptance is pending until the
exact successor commit's CI result is checked. Gate 5 is not yet complete.

## Accepted starting point

Commit `a0cad0f56659140c025a66080bd9d2f329c9cbb6` supplies the accepted manifest
validator and read-only release-content verifier. Its
[Source CI 34317419549](https://github.com/saidi19/lumi-model/actions/runs/34317419549)
passed all four jobs with **4726 passed, 58 skipped, 329 deselected, 36 warnings,
9191 subtests, 318.29 seconds**; source-selected 4784, private integrations
**327 NOT_EXECUTED**, two historical tests separately routed.
[Spark CI 34317419550](https://github.com/saidi19/lumi-model/actions/runs/34317419550)
passed **3 tests, zero skips, 97.53 seconds** on that same commit.
The local result handoff v0.46 and all earlier evidence are preserved.

## New gate and exact scope

`posttraining_sft_release_gate_v1.VerifiedRelease` is an internal one-shot adapter
for the trusted bootstrap, not a permission token. It calls the accepted verifier
directly with captured inputs, never a caller-authored successful receipt. It
checks owning PID, main/single thread, actual parent and cooperative deadline
before verification and immediately after it. Only a strict successful verifier
receipt, including all false nonclaims and balanced handles, allows the supplied
release callback to run once. Failed verification, callback failure, reentry or
replay cannot retry either stage. Status exposes fixed codes and integer counts,
not paths, raw text, identities or exception details.

The remote-only successor composes the **exact accepted IPC fixture bytes** and
wraps the existing guardian's release callback, whose first operation sends
`b"G"`. The unchanged contained worker waits for that byte before probes or exec.
The old guardian/watchdog, cgroup ownership, privilege drop, filesystem/Landlock,
network/IPC isolation and generated CPU WorkerRun bodies stay unchanged.

Root remains `/usr/bin/python3 -I -S -B`, with an explicit actual CPython 3.12
preflight before resource creation. The hostedtoolcache Python binding still
belongs only to the dropped-UID CPU worker. The native verifier's runtime guard
is never patched or bypassed. Nothing new runs a hosted ML runtime as root.

## Six fresh cases, with separate accounting

The original four positive lifecycle cases run first: normal, deadline,
parent-loss and generated WorkerRun. Each now verifies a fresh generated release
before the original callback can release its worker. The original acceptance
still requires exactly four installations and the old CPU totals: three
invocations, six rank workers, four global and eight rank optimizer updates.

After all four positives, two fresh **blocked WorkerRun** cases use the same
guardian/isolation path: wrong externally supplied pin, then same-size dependency
tampering. Each has one verification attempt, zero verification completions and
zero original release attempts/completions. Their required fixed codes are
`manifest_hash_mismatch` and `digest_mismatch`. The old guardian reports its own
`guardian_failed` separately; that code is not misrepresented as a successful run.

The negative outer controller path explicitly accepts only the expected failed
guardian receipt with confirmed empty/removal, reaps its exact controller with
exit status 2, and requires the guardian's one blocked worker to have been killed
with SIGKILL and reaped. One-shot retirement is counted; descriptor identities
before/after the watcher must match. The inherited `release_called` flag is not
used to claim a `G` send: it means callback consumption even on rejection.
Zero negative WorkerRun invocation follows from the zero original callback calls
and the frozen wait-for-G-before-exec sequence, not a separate direct exec counter.

Four-case counters remain positive-case counters. Additional fields report the
two negative cases, six total cgroups and generated trees removed, two rejected
workers/controllers reaped, two additional seccomp installations and four
additional exposed socket removals. The final public projection remains within
the inherited 4096-byte structural-report bound. A pre-gate failure still retains
the old failure/removal receipt; it cannot pass gate acceptance.

## Tiny generated inputs and limits

Each case uses exactly three declared stand-ins: runtime, dependency and source,
totalling **106 bytes**, with a **600-byte canonical manifest** and separately
fixed source-controlled pin
`2c83d57f86fe861979fcc74b342e0981281ab4465437989e3eae3302fa23e112`.
The expected manifest is not inferred from scanned file contents. The tamper
case changes exactly 30 generated dependency bytes without changing the manifest.

Per verification: at most 3 files, 39 bytes per file, 106 aggregate content bytes,
6 tree entries, 109 requested read bytes including EOF probes, 64 open attempts,
and a cooperative 1000 ms deadline. Overall fixture alarm remains 260 seconds;
wrapper wait remains 300 seconds. There is no retry loop. Generated tree payload
writes across six cases total 666 bytes including the one tamper. The existing
fixture's separate generated sentinels and bounded tmpfs probes remain required.
Only identity-matched, individually recorded generated paths are removed, never
recursively; file identities are registered before chmod or payload writes.

## Acceptance boundary and remaining work

This is an **integration-ordering experiment**, not a release of the runtime
actually executing WorkerRun. The ELF-looking bytes are deliberately non-executed
stand-ins. It does not authenticate runtime origin, prove complete dependencies,
provide a coherent snapshot or execution immutability, or bind a real host.
Real SFT execution and production eligibility remain false on successful reports.
Trusted host/kernel/admin/controller/software and trusted control peers remain
premises; this does not add a blanket hostile-host sandbox gate.

Local checks passed **158 stdlib tests** (the 89 accepted verifier regressions,
46 new gate tests and 23 new AST/literal tests), with exact Ruff clean.
Neither the native fixture nor its Linux wrapper is imported or executed on
the Mac, including platform-skip paths. Their runtime behavior is accepted only
after GitHub CI on the exact published commit. Independent source review checked
composition, negative cleanup/reaping, declared hashes and bounded report shape.

After native ordering acceptance, finish the independently reviewed **actual**
runtime/dependency/source release handoff and integrated worker/supervisor
admission. Real target-host binding, original S3 checkpoint restore, exact-data
relocation and explicitly approved H100/CUDA validation follow; real SFT and
downstream evaluation/rights/quality/deployment gates are not started here.

No existing implementation, fixture, tests, workflows or evidence is edited.
No GPU/instance action, S3/SSH, real model/data access, real-model inference or training,
promotion, dataset seal, registry mutation or automation is authorized by this
successor. Source CI success still leaves private/full-integration flags false.
