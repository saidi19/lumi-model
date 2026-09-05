# CI preservation policy v0.1

This is a source-backup CI recovery, not a GPU execution or production gate.
The original GitHub run for source backup `1c24ef0` installed dependencies, then
failed at `ruff check .` with 767 findings; its pytest step never ran.

## Historical lint is explicitly retained, not reported as fixed

Published historical implementations and tests are byte-bound to previous
private plans and evidence. Reformatting those original paths would invalidate
their hashes. The new lint gate therefore checks an exact immutable manifest
of the 163 affected files from commit
`1c24ef039d25dde0172fb2751d7afdba403c7975`. It pins Ruff 0.15.1, every file's
SHA-256, and each file's exact diagnostic fingerprint, count and rule counts.
It rejects changed historical bytes, added/removed/changed diagnostics, findings
in any file outside the exact manifest, and baseline changes. There is no
wildcard exclusion, category-wide ignore, `--fix`, or automatic baseline refresh.
Ruff runs with `--isolated`, explicit frozen rule/line-length/Python-target
settings and an explicit Git-tracked Python file list (plus visible untracked
Python files during local development). Nested Ruff settings and `.gitignore`
cannot hide a newly committed Python file from the check.

The old root `ruff.toml` already suppressed another 48 diagnostics in nine
historical files. The isolated check restores visibility of these too: the
pre-publication manifest therefore records **815 findings**, not only the 767
from the failed GitHub run. All 163 source files were compared byte-for-byte
with the original backup commit before creating the manifest.

The gate reports `all_source_lint_clean=false` and emits an explicit warning
for all 815 retained findings. Passing it means unchanged historical lint debt
and zero new findings, **not** that all 815 findings have been repaired. Future
changes to those implementations require reviewed successor files and their
tests/bindings, not silently growing this manifest.

The retained rule counts are E501 687, I001 70, F401 13, E731 9, UP012 6,
UP035 3, UP042 2, UP037 1, B009 2, B023 20 and F841 2. The B023 findings occur
in two synthetic test modules whose callbacks are invoked synchronously inside
the loop, not queued for later invocation. The F841 finding is an unused
intermediate attribution list in the historical Prosocial curation version;
the list's construction and possible validation side effects remain unchanged.
The other F841 is retained from the pre-existing stronger-teacher GPU experiment.
This preserves the source evidence; it is not an exhaustive semantic audit or
a waiver of undefined names, new warnings, runtime failures, or test failures.

## Control tests are moved, not removed

The seven published control suites require a genuinely clean `python -S -B`
process. Ordinary pytest imports the repository conftest and ML package before
collection, so it cannot satisfy their no-site/no-ML contract. They now run in
a separate required job, with exact per-module test counts totaling 273, zero
failures/errors/skips/expected failures, sanitized environment and a 120-second
child deadline. The matching seven paths are omitted from the general pytest
invocation only because they are required by this isolated job. New synthetic
tests exercise the CI runner and lint-debt guard themselves.

## General tests remain blocking

The existing general pytest suite still runs; no `continue-on-error`, blanket
private-test skip, arbitrary xfail, or test-count reduction is introduced.
Some historical integration cases require deliberately untracked private
plans/receipts/artifacts. Those remain local-only; they must be classified and
routed explicitly or replaced by genuine synthetic unit fixtures, not satisfied
with fabricated approvals or by uploading private datasets to GitHub.

No green full-CI claim is justified until the new Linux run passes. Local
macOS/Python 3.13 tests use a different historical ML dependency lock than the
GitHub Python 3.12 job, so local dependency/API errors are not automatically
GitHub failures. The current changes let lint, isolation and general tests
report independently instead of lint hiding all later failures.
