# SFT execution status v0.38

Mount-source namespace-affinity correction, 2026-09-09. This is a corrective
successor to failed filesystem commit
`741639cc3eac16a4fa844c886350159f55e1e56a`, not a rerun or acceptance claim.
Gate 5 remains partial. No real model, dataset, GPU, S3, training, host admission
or production action is authorized by this generated fixture.

## Preserved failed evidence

[Source CI 34231716274](https://github.com/saidi19/lumi-model/actions/runs/34231716274)
failed its new Linux filesystem case: 1 failed, 4557 passed, 58 skipped,
329 deselected, 36 warnings and 8502 subtests in 326.00 seconds. Its other three
jobs passed. Source selection was 4616 tests; 327 private integrations remained
NOT_EXECUTED and two historical cases were routed separately. Full/private-CI
claims remain false.

The fixed failure report identified the normal case, `stage=cases`,
`component_code=filesystem_mount_mount_failed`, and `cleanup_confirmed=true`.
No successful Landlock setup, release, or WorkerRun acceptance was reported.
The original report did not identify the particular mount syscall or errno;
it must not be retrospectively described as having done so.

[Spark CI 34231716255](https://github.com/saidi19/lumi-model/actions/runs/34231716255)
passed three tests with zero skips in 105.13 seconds. Neither failed run nor
its predecessor document is deleted or overwritten.

## Concrete defect and bounded correction

The implementation pinned all ten bind sources before `unshare(CLONE_NEWNS)`
and later bound them via `/proc/self/fd`. Those descriptors retain their old
mount references. The Linux v6.14
[bind-mount implementation](https://github.com/torvalds/linux/blob/v6.14/fs/namespace.c#L2624-L2648)
rejects ordinary bind sources whose mount belongs to another namespace. This
is a concrete code defect consistent with the reported mount-stage failure;
only the corrected remote fixture can establish end-to-end acceptance.

The correction retains each original source pin, creates the private namespace,
disables propagation, and rewalks all source paths there using the existing
no-symlink component walk. Device, inode, full mode, UID and GID must match the
original pinned object. Only then is the original descriptor closed once and
the new pin admitted as a bind source. A replacement, open failure or close
failure is terminal before root construction; no fallback, rollback, setns,
retry, or source-policy expansion is added. Bind and namespace-unshare failures
also receive distinct fixed allowlisted codes without exception text.

This commit changes only the mount implementation, its syscall-double tests,
and this new status document. Landlock, the integrated Linux fixture/wrapper,
all older fixtures, workflows, budgets and policy floors remain unchanged.
Frozen netns, cgroup and WorkerRun SHA-256 bindings remain respectively:

- `70400a1428d0e54834721a5163db7724940a271bbeeeaec50714f28108f20bfc`
- `f91ed0c353a20c309ff3e1bfaab8dbe990b28c36626fdeeb0ac5131e0cf75fad`
- `53c50c92965ed1cef0c9624c57367c31ed7ebfb9d7cfb34257188ad179d333d0`

## Verification before publication

The mount suite passes 36 tests. New tests require every bind pin to have been
opened in the new namespace, require original-pin closure before binding,
reject each of the five identity-field changes, and check open/close failure
cleanup and one-shot behavior. The new namespace-affinity regression rejects
the prior committed implementation under syscall doubles.

The exact 23-suite local stdlib regression passes 924 tests with zero failures,
errors or skips in 0.578 seconds and `ML_MODULES_LOADED=False`. Exact isolated
Ruff passes for all six filesystem Python files. Both Linux-only files were
AST-parsed, never imported or executed on the Mac. Frozen hashes were rechecked.

The corrected commit still needs one observed remote Source/Spark CI result.
No failed attempt is rerun unchanged. The four-case remote fixture retains all
existing deadlines, resource caps, generated-only mutation targets, same-UID
positive/negative controls and independently confirmed cleanup requirements.

## Remaining boundaries

This correction does not resolve exposed-tree pathname UNIX IPC authority,
runtime origin, host binding/admission, real-size relocation or actual H100
checks. Trusted host software remains a premise, not an immutability proof.
Gate 5, complete hostile-process containment and real-SFT admission remain
incomplete; production eligibility remains false. No GPU is needed for this
correction's source checks or remote generated CPU acceptance.
