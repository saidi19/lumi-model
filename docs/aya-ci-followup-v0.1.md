# Aya source-test isolation and control-failure diagnostics

## Scope on 2026-09-06

The bounded, successful Aya historical-plan check is recorded separately in
`docs/aya-plan-snapshot-local-v0.1.md`. Its private invocation and the previous
safety replay were not repeated for this follow-up. Their source bindings,
private inputs and prior evidence remain unchanged.

Commit `d41a92600f11f57045f3405bf159a9d4f52ccee7` had two failed jobs in
[Source checks run 34021200453](https://github.com/saidi19/lumi-model/actions/runs/34021200453).
Frozen lint and source-only validator jobs passed. The two failures must be
distinguished; a local pass does not establish the cause of an earlier CI failure.

## Confirmed source-test defect

The newly added synthetic
`AyaPlanLocalTests.test_projection_is_removed_after_a_pytest_failure` assumed
that the original Aya test module was absent from `sys.modules`. Full-suite
collection already loads that module. The checker correctly rejected a collision
without removing the existing module; the synthetic test's absence assertion was
wrong. This was a test-isolation defect, not failure of the completed private
Aya check.

Only that new test is corrected. It exercises an isolated fresh namespace with
and without an existing outer module, verifies that the mocked pytest call
actually occurs, checks cleanup on pytest failure, and restores the original
module identity. The original Aya test and checker are unchanged.

## Unlocalized control-job failure and reporting repair

The isolated job logged only `control_child_failed`. Its first failed assertion
or error was discarded, so the original cause cannot be identified from that
log. A bounded local diagnostic and a subsequent fresh-process run both passed
the fixed 273 synthetic controls. These observations do not prove that the
historical CI failure was fixed, or establish any particular timing-race cause.

The non-frozen CI runner now retains at most 32 sanitized failure entries for
the first failing suite: its allowlisted suite path, fixed expected count,
aggregate counters, one-based execution index (zero for suite setup), fixed
result kind, and fixed error category. Counters still include every failure if
the detail cap is reached. Builtin and subprocess timeouts share the `timeout`
category. No test IDs, exception text, traceback, skip reason, command text or
private content is retained. Python test-output suppression is unchanged; this
is not a claim of native-output suppression or an OS sandbox.

The parent accepts only a size-bounded, duplicate-key-free, exact-shape failure
envelope with allowlisted values and coherent counters. Malformed child output
is discarded and remains a generic failure. Valid failure details also remain a
nonzero failure; they cannot produce a success result.

The seven suites, 273 required tests, 120-second limit, fresh `-S -B` child,
sanitized environment, unchanged PASS schema and no-skip/no-failure requirements
are preserved. No old control test, workflow, frozen lint baseline, historical
inventory, source partition or readiness gate is relaxed. This repairs missing
failure localization, not an unproven control-test cause.

## Local verification before publication

- The fresh isolated runner passed all 273 fixed synthetic control tests.
- The final targeted selection passed 154 tests and 149 subtests, including
  17 new failure-report tests, Aya tests, existing CI guards and safety lifecycle
  tests. An initial local command lacked `PYTHONPATH=src` and stopped at conftest
  import before test execution; the corrected command supplied the repository's
  source path.
- Changed Python files pass the exact CI Ruff rules; `git diff --check` passes.
- The historical inventory SHA-256 remains
  `441390c9c98cd4b83d224055a5af5ae421e5ef01b2e4b86101e7b17693d87665`;
  the source-partition SHA-256 remains
  `4e595b654880668a4462c7b8db7c40eaf715ebd86b90eb14cdf852720bc831ae`.

The follow-up commit's hosted CI outcome must be checked separately. No full
private-integration, production eligibility, dataset promotion/seal or GPU/SFT
completion is claimed here. No private replay, source/candidate processing,
model loading or training was performed by this follow-up.
