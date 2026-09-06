# Watchdog synthetic-CI storage recovery

## Evidence and limits of the diagnosis

Source checks [run 34023049805](https://github.com/saidi19/lumi-model/actions/runs/34023049805)
at commit `0f0e59ecdca22e0bbb23d9988757b0137f51f687` ran all 33 original
watchdog tests and reported two assertion failures: alphabetical indices 7 and
24, `test_child_failure_is_code_only` and `test_sleeping_child_hits_deadline`.
The other three Source checks jobs passed. The previous log did not retain the
actual result reason or assertion location; it cannot establish their exact
historical cause. The failed run is not deleted or reclassified.

The frozen watchdog starts its runtime clock before five synchronous flushes:
the parent directory, three newly written files, and the attempt directory.
Only then does it check the deadline before spawning. The two fixtures allow
800 ms and 100 ms respectively and previously used default temporary storage.
Thus disk contention can produce `deadline_exhausted_before_spawn`, which
correctly fails both original reason assertions. This is a verified failure
mechanism, not a claim that the historical run recorded that reason.

A new synthetic regression uses an injected monotonic clock with real fsync
calls to reproduce that pre-spawn result at both unchanged budgets. It verifies
that no child is spawned, the attempt is consumed, and retained child text is
zero. It changes neither the original module nor the original test bodies and
does not substitute for their real-process execution.

## Scoped correction

Only the non-frozen CI runner changes. On Linux, only the original 33-test
watchdog suite receives a fresh mode-0700 temporary directory on `/dev/shm`.
Before creation, the runner checks the no-follow directory handle, root-owned
mode-1777 parent, bounded kernel mount information for the matching tmpfs/device,
and at least 16 MiB and 64 free inodes. Missing or unsuitable storage stops the
job; there is no fallback to disk, mount action, resize, or retry. The directory
is removed and the previous tempfile setting restored after the suite.

The other six suites retain their original storage behavior. Non-Linux local
runs keep their existing temporary storage. No production watchdog path,
contract, fsync, deadline, grace/cleanup allowance, assertion, fixed count of
273 tests, 120-second suite timeout, skip rejection, workflow, or private gate
is changed. Tmpfs execution verifies process control and file/marker behavior,
not crash-persistent storage or behavior under all scheduler/storage loads.

If an original watchdog assertion still fails, the runner can retain only nine
strictly allowlisted lifecycle fields from an exact original test-code frame:
reason, child-spawn/status booleans, bounded exit codes and elapsed milliseconds,
and three cleanup booleans. It retains no exception text, test IDs, source text,
or result-file text. Other suites cannot supply these fields. Observation
failure cannot suppress a failing test. Five fixed storage-admission STOP codes
also survive parent validation without arbitrary child text.

## Verification before publication

- The original seven suites passed locally: 273 tests, zero failures, errors,
  skips, expected failures, or unexpected successes. This Mac run does not
  exercise the real Linux tmpfs mount; hosted Linux acceptance is still required.
- 116 targeted guard tests passed, including 31 new storage/observation tests
  and unchanged CI, failure-diagnostic, frozen-lint and workflow guards.
- Exact CI Ruff rules passed for changed/new Python files. Full frozen-lint
  verification retained all 815 historical findings across 163 files, with
  zero new findings and no historical source changes.
- `git diff --check` passed. The original watchdog source remains SHA-256
  `60fd833e532809b373b4a3d0f17f4ee12e70e417367c4fe2010e83de43a9e0e6`;
  its original tests remain
  `007131cede8f375609df778f5590a54523ebb15127d4303d75f3867615aa1f08`.
  Frozen lint, unresolved inventory, source partition, and original workflows
  remain byte-for-byte unchanged.

Publication should use one corrective commit and its normal hosted checks, not
a manual rerun of the failed commit. No private attempt, dataset, owner packet,
S3 object, GPU worker, SFT run, promotion or release gate is executed by this fix.
Private integration coverage and broader roadmap readiness remain separate.
