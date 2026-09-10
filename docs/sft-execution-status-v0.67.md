# SFT execution status v0.67

One separately approved, Labels-corrected CPU-only guardian acceptance successor,
2026-09-10. The user answered the exact proposed Labels fix and bounded acceptance
request with "mama kaj resume kor", then resumed the same unfinished work. This is
one fresh successor, not a retry or continuation of any started native attempt.
Exact parent: `f3d8554d8999e32bb750fb8aa9020fca6818bf72`.

## Evidence and narrow correction

Prior diagnostic run `34363848352`, job `102507402716`, attempt 1 completed its
diagnosis and cleanup successfully, with **admission_passed=false**. Its first
failed predicate was `configuration_labels_equal`: both keys were present,
image `Labels=null`, created-container `Labels={}`. Later predicates were not
reached. This does not establish the cause of an earlier opaque failure or prove
that no further integration blocker exists.

Prior artifact `10109060039`: 1,531-byte ZIP, SHA-256
`64b9f11faa76f33b059f5e08a644664c3c1e6d1ea8f2732cbaff70fd592ebe42`.
Diagnostic member: 1,540 bytes, SHA-256
`e828ba303f517d217e9afba8cea389eec591523704dc9919e9a8e4104400c732`.
Local v0.66 receipt remains unstaged and byte-for-byte unchanged, SHA-256
`029233eb32afad63ebb37d96b1d668436c71ee60c0557aa7654cd5bf1b072556`.

The new v2 admission first evaluates the old canonical equality unchanged. Only
when unequal may it accept **present image Labels=None and present container
Labels of exact dict type with zero entries**. It is directional and field-local.
No normalization or input mutation occurs. Original equal cases stay accepted;
missing/null equivalence already present in the old equality is not newly added.
Empty strings/lists, nonempty labels, reverse direction, absent-key/dict pairs,
other false-like values and other fields receive no new exception. All other
admission, transport, identity, image/runtime and SoftwareBinding rules remain.
Only fixed allowlisted configuration predicates are added to typed failure
reporting; no inspect/exception text is retained.

## Unchanged accepted software

Artifact `10101633554`, run `34345663337`, source
`c4272e36bc98354067dd23a83931a3db58823d76` is reused, never rebuilt or pulled:

- ZIP 1,062,113,436 bytes, SHA-256
  `98d395f35af2b23874b6d5d7a56a13c969fd72fa237065fec74d81634927ac1c`.
- Tar 1,062,113,280 bytes, SHA-256
  `99d21fc60a54a65f28ceb4780f603f7ee48e49ac7b6d00a053dcdabe9ebc9ff4`.
- Image `sha256:321d50ae5207dee05cd58ec5436d97978becde2ee8a767f2e8768229e6a9ad43`.

Current authenticated metadata for both artifacts matches the exact pins and is
not expired. No large archive is downloaded on the Mac.

## One-shot execution and bounds

- First private push, exact branch/checkout/direct parent and attempt 1 only.
  No workflow dispatch, native retry or automatic recovery. Stop on first failed
  case. Historical workflows/sources/evidence are not edited. The old guardian
  test wildcard can match these new tests, but its historical parent guard is
  false and prevents its native job from running.
- Seven bounded acquisition GETs: prior diagnostic job proof; tiny diagnostic
  artifact metadata/redirect/signed ZIP; accepted software metadata/redirect/
  signed ZIP. Verify the tiny exact evidence and cleanup before any large
  request. Each metadata body at most 32 KiB; diagnostic ZIP exactly 1,531 bytes,
  below 64 KiB. Signed GETs carry no credentials/cookies. No redirects followed,
  retries, proxies or acquisition POSTs. Acquisition logical bound 300 seconds,
  outer 6 minutes plus 15 seconds, step 7 minutes.
- One trusted hosted Ubuntu 24.04 CPU job, whole-job bound 28 minutes. Native
  isolated/no-site root CPython 3.12 has a 900-second logical budget with the
  final 100 seconds reserved for owned cleanup. Cleanup commands share a deadline
  no later than the overall deadline. Outer native bound 17 minutes plus 30
  seconds, step 18 minutes. These are userspace/outer bounds, not guarantees for
  uninterruptible kernel I/O. No native entry runs on the Mac.
- Exactly one image load and never-started container create. Independently bound
  image/container inspect before export and again after export; each inspect at
  most 65,536 bytes/5 seconds, each verification at most 15 seconds. No Docker
  container start; actual work uses the existing guardian process architecture.
- Export at most 2 GiB, extraction at most 4 GiB, individual file at most 1 GiB,
  at most 65,536 archive entries. Require at least 12 GiB initial free space.
  Rootfs parser, cross-CPython tar handling and accepted source bytes stay frozen.
  Two separately bound adapter/control files are installed read-only; existing
  software is untouched. Docker cache/metadata is not claimed to have a strict
  write quota.
- Frozen case order: wrong control pin, wrong adapter pin, positive WorkerRun.
  The two negative controls reject before ML. The positive case uses only the
  existing tiny generated CPU fixture, including deterministic resume comparison;
  no pretrained model, real dataset, S3 object or real SFT. No case is retried.
- Existing cgroup limits: 4 GiB RAM, zero swap, 256 PIDs, two CPU equivalents;
  existing namespaces, mount policy, Landlock, IPC restriction, UID/GID 10001,
  authority-fd closure and parent-pidfd ownership/reaping remain unchanged.
  Writable scratch is 512 MiB and shared memory 64 MiB, on private overlays.
- Native evidence is bounded to 32 KiB and fixed stage/code/identity/count
  aggregates. Exact typed admission/rootfs errors and allowlisted bootstrap stages
  survive reporting. The existing packet is read once and returned unchanged;
  no new packet, release or workload permission is introduced.
- Only this attempt's exact identity-bound files/directories and proven-owned
  Docker objects can be retired. Never force/prune/remove-all. Started ambiguous
  daemon writes or cleanup failures fail closed, without retry. Successful final
  inventories must prove owned image/container absence. Rootfs deletion waits for
  descendants to be reaped. Input retirement separately binds exact acquired
  file/parent identities. Host exclusivity/trusted Docker remain premises.

Only small aggregate private JSON is uploaded. No raw logs/inspect/exception text,
credentials, signed URLs, archive, candidate or owner artifacts are published.
Git publication is an exact nine-new-file non-force successor. No GPU/H100,
instance action, S3, model/data access, real SFT, promotion, dataset seal, registry
mutation or unrelated action is authorized here.

## Source checks and publication bindings

The four new suites contain 151 tests. Together with unchanged admission,
transport, guardian transaction, rootfs, mount, Landlock and guarded-worker
regressions, **525 tests pass with zero failures/skips** on Python 3.11.15 (0.410s),
3.13.7 (0.384s), and 3.14.6 (0.384s). Exact Ruff E,F,I,B,UP, line length 100,
target Python 3.11 passes. Final AST-double linking tests cover failed bootstrap
with no release and single cleanup, unchanged ready-success receipt, main-stage
allowlisting and failing post-removal inventory without retry. All OS/socket/
cgroup/fork/exit calls in those tests are doubles, not native execution.

| New file role | SHA-256 |
| --- | --- |
| Admission module | `d57c87ecdcb31058e779deb72eba7928930d1edd5b99c0752d112edcbefc6136` |
| Root fixture | `79dcbfec678f43a39b31bb2dd2b4c9535fc162447cc591a781b72f93f079ce58` |
| Preparer | `8150fbdeb11d102b6cc8cb358f532d6504068e6c8db64cac7f29f8faac75d32d` |
| Workflow | `7cef83594341902c7b06162bdafe16afe85f6097f83f37608e9711da97858928` |
| Admission tests | `89125cf2055942e46d847983013c7e569d55bd5f5dbeac3b35bb074736154c55` |
| Fixture tests | `442bfc180137bcb232f309e25eb2d269d6e3ffb81364d3510032e6aeb8a00c5e` |
| Acquisition tests | `d7c59d1648ae76f50084c063296c467875e35806fa439934face0f8ae7c48b32` |
| Workflow tests | `9212c5ca70f67bd2cc9773bc17bc83505823637bc2c759d1faddc38ba7537428` |

All historical tracked source remains unchanged. This plan and eight new source/
workflow/test files are the only publication targets; prior local receipts and
private artifacts are excluded. Linux source-only preflight must pass before
acquisition. Native results remain pending; preserve the result in a new local
v0.68 receipt without automatically publishing another triggering commit.

Gate 5 remains partial until evidence establishes the missing composition checks.
Even a successful fixture does not admit a real host, real SFT, production, full
CI or private integrations. Dedicated-host/S3/data binding and H100/CUDA validation
remain separate downstream boundaries, not permissions granted by this run.
