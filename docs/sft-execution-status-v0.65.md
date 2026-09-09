# SFT execution status v0.65

One separately approved OCI inspection-localization diagnostic, 2026-09-09.
The user approved the preceding bounded CPU-only diagnostic request with
"ok chala". This is not a rerun or continuation of the failed guardian attempt.
Its exact parent is `11486aa570657e8c71589ced46054459972d1493`.

## Preserved stop and software

Prior run `34359447892`, job `102492361796`, attempt 1, stopped at
`inspect_container` before export/extraction/cgroups/workers. Its acquisition,
source preflight and input-copy retirement passed. The three small evidence
files remain unchanged; native report SHA-256 is
`70ae37f4e0898169e5ee0fa14d7ce123d829558ad4ca942a3391202dc876e495`.
The local v0.64 result receipt also remains unchanged and unstaged.

The accepted software is not rebuilt or pulled. Artifact `10101633554`, from
run `34345663337`, source `c4272e36bc98354067dd23a83931a3db58823d76`, is reused:

- ZIP: 1,062,113,436 bytes, SHA-256
  `98d395f35af2b23874b6d5d7a56a13c969fd72fa237065fec74d81634927ac1c`.
- Software tar: 1,062,113,280 bytes, SHA-256
  `99d21fc60a54a65f28ceb4780f603f7ee48e49ac7b6d00a053dcdabe9ebc9ff4`.
- Image ID: `sha256:321d50ae5207dee05cd58ec5436d97978becde2ee8a767f2e8768229e6a9ad43`.

Current authenticated artifact metadata still matches these pins and is not
expired. No large software archive is downloaded on the Mac.

## Narrow diagnostic, not weakened admission

The unchanged admission source is bound by SHA-256
`a2f26771b22fedfe8a003bc3104c40f46477465b7f0ab2c50bee7ad3abb945f0`.
An isolated source-bound instrumented copy preserves its predicates and
evaluation order, while adding fixed phase/predicate/error localization. Loop
keys are mapped through fixed source-defined allowlists; no arbitrary keys or
values become report text. Configuration equality failures may retain only
fixed absence/null/type/empty-category booleans. No software binding or
admission continuation is released, even when every predicate passes.

The root fixture imports only hash-bound old Docker/copy helper definitions;
it never calls the old native entry, controller, guardian or worker. Its frozen
helper source SHA-256 is
`38d64b4159bcc418ff799c14fae6cde00ec98b35b58cbfeb58fb14b2e83d8f76`.
The old create options are reused exactly. Successful inventory calls must
first prove the pinned image and fresh unique container name absent. Then one
load and one create are permitted, followed by at most one image inspection
and one container inspection. The container is never started.

The first failing predicate stops verification without retry. A completed
diagnostic may report `status=passed` with `admission_passed=false`: that means
the diagnostic obtained a validated result, **not** that native acceptance or
Gate 5 passed. Transport/source/cleanup failure remains a failed diagnostic.
No profile/parser fix, null normalization or acceptance recovery is included.

## Execution and data boundaries

- First private direct-successor push only, exact checkout/parent/branch,
  first run attempt. No manual dispatch or retry; later ordinary commits skip.
- Four bounded acquisition GETs: exact prior completed job proof; accepted
  artifact metadata; non-followed archive redirect; credential-free signed
  storage GET. Each metadata body at most 32 KiB, logical acquisition 300 seconds,
  individual timeout at most 30 seconds; outer 6 minutes plus 15 seconds.
- One fresh trusted hosted Ubuntu CPU job. Native root process: isolated,
  no-site CPython 3.12, sanitized environment, no ML imports. Logical native
  deadline 420 seconds, reserving the last 100 seconds for cleanup. All cleanup
  commands share an absolute deadline no later than the overall deadline;
  outer 8 minutes plus 30 seconds, whole job 20 minutes.
- Each image/container inspect at most 65,536 bytes and 5 seconds; total
  verification at most 15 seconds. No further inspect pass. Deadline statements
  are userspace logical/outer bounds, not guarantees about uninterruptible I/O.
- Exact software is copied once into a root-owned private file and rehashed.
  At least 12 GiB host free space is required. No export/rootfs is written.
  Docker-cache/metadata disk consumption is not claimed as a strict quota.
- Root report at most 8 KiB, fixed codes/booleans and nonclaims only. Uploaded
  evidence consists only of small aggregate JSON, never raw inspect responses,
  exception text, credentials, signed URLs, large archives or model/data text.
- Only this attempt's identity-bound input/root files and proven-owned Docker
  objects may be retired. No force/remove-all/pruning/recursive deletion. Each
  removal is attempted once; a started ambiguous daemon write is never treated
  as confirmed cleanup. Final successful inventory observations explicitly
  attest owned image/container absence. Input retirement is separately bound
  to the acquisition's exact file/parent identities.

No container start, export, rootfs extraction, workload cgroup, adapter, worker,
GPU, inference, model/S3/data access, registry mutation, promotion, seal or SFT.
No changes to the four historical native workflows, accepted software, parser,
admission predicates or prior evidence. Trusted ephemeral host/exclusive Docker
daemon remain premises; this diagnostic does not authenticate that host.

## Source verification and handoff

Only the new diagnostic module, root fixture, preparer, workflow, four synthetic
test files and this plan are publication targets. Source verification must pass
before publication and again in Linux CI before artifact acquisition. Local
checks are stdlib/synthetic only; no Docker/Linux native entry or ML is run on
the Mac. The publication binds the diagnostic module hash in the root fixture.

Final local verification: **220 tests pass, zero failures/skips**, on each of
Python 3.11.15 (0.488s), 3.13.7 (0.461s) and 3.14.6 (0.516s). This is 150 new
diagnostic tests plus 70 unchanged admission/transport/source regressions.
Exact Ruff `E,F,I,B,UP`, line length 100, target Python 3.11 also passes. Cleaned
instrumented AST matches all nine original verification helpers, except the
explicitly removed successful binding construction. New cleanup tests cover
ambiguous daemon writes, single retirements, final absence checks and deadline
reservation; the old reporting loss is not repeated for a validated failed
diagnostic result.

| New file role | SHA-256 |
| --- | --- |
| Diagnostic module | `54fb165e864db5a69d41875e014e933dcba443f8ce39ea3d8a0532656848067b` |
| Root fixture | `23df871fe7fb57719e44ebe4ac2185dfdbd178bd7ca43927cedae0adc62eb954` |
| Preparer | `b53378d70bb8f7c7b826077f5512fb4cf10f27522d0acedf43439ad2a039f53f` |
| Workflow | `c2b8c142a4bb7b04a3a98df6769e07874e195d8cab684fceecc1e3b36f21975a` |
| Diagnostic tests | `bbe0c4896766215e95a257a77a7098b1c5b80ce1b5f049d24d09f2ac4e6d2b37` |
| Fixture tests | `1db5240784ff3df5fcd1742e39954d5db39802882312b0ebcaa24006fe3f31eb` |
| Acquisition tests | `5a4723a201af6251de503dd0e17db3ae8ce55503ded35a1ac4fa0448fe2cdde2` |
| Workflow tests | `9daee4c40b24f06f17bc3e3f6e02745382f3dd48da2079a86ed10adf0ace3255` |

Native results are pending. Preserve the completed result in a fresh local
receipt without automatically publishing another workflow-triggering commit.
After this single diagnostic, stop and report the fixed failing predicate (or
that none failed); do not silently perform a curation/training/acceptance
recovery. Production, real-SFT, full-CI and private-integration claims stay false.
