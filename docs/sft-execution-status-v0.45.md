# SFT execution status v0.45

2026-09-09. Pre-GPU release handoff verification successor. Gate 5 remains
partial; this change adds no launch capability and does not claim GPU readiness.

## Accepted starting point

The preceding narrow IPC correction is accepted on commit
`d974c446218325ed3e43ee20a0cdc80bbb1307c9`:

- [Source CI 34315983656](https://github.com/saidi19/lumi-model/actions/runs/34315983656):
  all four jobs passed; **4637 passed, 58 skipped, 329 deselected, 36 warnings,
  8828 subtests, 401.85 seconds**. Source-selected: 4695; private integrations
  **327 NOT_EXECUTED**, with two historical tests separately routed.
- [Spark CI 34315983631](https://github.com/saidi19/lumi-model/actions/runs/34315983631):
  **3 passed, zero skips, 82.34 seconds** on the same commit.

That native Linux acceptance includes four IPC lifecycle cases, four seccomp
installations, eight same-UID exposed-socket positive controls and eight owned
socket removals. The earlier filesystem, IP-isolation and cleanup checks remain
required. Generated CPU WorkerRun retains three invocations, six ranks, four
global and eight rank optimizer updates. The failures and all preceding evidence
remain preserved; no historical test, fixture, filter or workflow is changed here.

## Why this is the next finite step

The existing IPC test observes the current Python executable hash, while the
mount-root contract only metadata-pins runtime/source paths. Neither establishes
content agreement for a complete, independently selected runtime/release tree.
The finite workboard in v0.26 already requires that handoff before real training.
This successor implements its bounded verification component, not a new blanket
hostile-host sandbox project.

Threat-model premises are trusted host/kernel/administrator, trusted controller,
reviewed release owner/software, and trusted retained control peers. Arbitrary
hostile host administrators or writable peers are not contained by this scanner.
The accepted narrow IPC proof remains narrow: SysV IPC, process signals, host
metadata and retained-peer authority are not newly claimed closed or silently
converted into mandatory additional gates.

## New source-only component

Only four implementation/test files and this status document are added:

- `src/lumi/posttraining_sft_release_manifest_v1.py`
- `src/lumi/posttraining_sft_release_verifier_v1.py`
- `tests/test_sft_release_manifest_v1.py`
- `tests/test_sft_release_verifier_v1.py`

The pure manifest validator requires an external SHA-256 pin covering canonical
ASCII JSON with an exact schema, Linux x86-64 / CPython 3.12 contract, and sorted
file records. Every file has a path, role, byte count, mode and SHA-256. Runtime,
dependency and source roles are required, with pinned interpreter and entrypoint
members. Duplicate keys, path aliases/traversal, conflicting file/directory paths,
unknown fields, unsafe modes and caller-budget violations fail closed.

The pre-ML verifier accepts explicit caller limits, owner IDs and one local
release root. It rejects unsupported runtime context before filesystem work,
walks component-wise without following symlinks, compares exact tree membership,
checks regular-file/single-link/owner/mode/size/content bindings, and checks
per-object metadata stability while observed. Interpreter bytes need an ELF
prefix; that is a format guard, not architecture or dynamic-loader attestation.
Nothing in the release is imported or executed. The verifier writes no files.

Reads and open attempts are prepaid; EOF probes are charged. File, manifest,
entry, cumulative-byte and cooperative-deadline limits are explicit, not inferred
permission. Conservative accounting can reject short reads. A directory census
may observe one overflow entry before rejecting it. Every owned descriptor is
closed once; an ambiguous close is not retried and prevents a success receipt.
Results contain a manifest digest, fixed status/booleans and aggregate counts;
errors contain allowlisted codes, not input paths or underlying exception text.

The hard API ceilings are 8 MiB supplied manifest, 32,768 files, 2 GiB per file,
16 GiB total content, 65,536 tree entries, 131,072 open attempts and 30 minutes.
They are ceilings, **not authorization to scan a real release**. This change
exercises only tiny generated files; a real scan requires its own reviewed inputs
and lower explicit budgets. Deadline checks cannot interrupt parsing or blocking
filesystem calls: the supported filesystem is trusted and local, with hard
runtime enforcement owned by a future independent supervisor.

## Deliberate nonclaims and acceptance

An earlier observed file may change after it is closed while a later file is
being checked. Consequently this is not a coherent point-in-time snapshot,
immutability proof, publisher authentication, dynamic dependency completeness
proof, or host admission. A specific regression test demonstrates that limitation.
All eight nonclaims, including execution, real-SFT execution and production
eligibility, remain false even on a successful verification receipt.

Local validation: **89 tests passed in 0.059 seconds**, plus exact-file Ruff
`E,F,I,B,UP` with no findings. Tests use synthetic supplied bytes, tiny generated
trees and syscall/runtime doubles, with no ML imports. No Linux fixture/wrapper
was imported or executed on the Mac. Independent read-only review found the
snapshot claim issue; documentation and the regression now explicitly prevent
that claim. These are not native Linux release-handoff integration results.

This document records pre-publication local acceptance. The next exact commit's
CI result must be read separately; no future success is asserted here.
`private_integration_passed` and `full_ci_passed` remain false.

## Remaining work, without skipping dependencies

1. Build the bounded release handoff around independently reviewed runtime,
   dependency and source manifests; integrate this verifier with the existing
   generated CPU bootstrap/admission path and test it on remote Linux CI.
2. Finish the integrated training-worker/supervisor admission contract using that
   handoff. A content-match receipt alone must never authorize a launch.
3. When the non-GPU work is accepted, bind the dedicated target host, restored
   original S3 checkpoint and relocated exact data. Confirm actual H100/runtime
   properties, enforce resource caps, and run the explicitly approved CUDA test.
4. Only after those results and the exact run approval, begin real SFT. Evaluation,
   quality/rights gates and any later deployment remain separate downstream work.

No GPU/instance action, S3/SSH access, real-model/data read, inference, training,
promotion, dataset seal, registry mutation or automation is part of this change.
