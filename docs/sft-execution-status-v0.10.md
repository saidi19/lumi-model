# SFT execution status v0.10

Local CI preparation resumed on 2026-09-06. This is a successor to
[status v0.9](sft-execution-status-v0.9.md), not a new training or data gate.
Full GitHub CI remains unresolved; no push, GPU action, model restore, private
dataset access or operational evidence publication occurred in this batch.

## Completed local work

- Added 15 learning-candidate-seal validator tests using labelled in-memory
  synthetic source metadata. They cover the original ten resealed boundary
  drifts and additional manifest, resource, audit and verification invariants.
  An independent source review and second test execution passed. The historical
  metadata-chain and materializer tests remain unchanged.
- Added 11 teacher-canary projection and binding tests using synthetic rows,
  in-memory registries and temporary synthetic bytes. They exercise the actual
  transforms and resolvers, including provenance, language mapping, deep copies,
  duplicate detection and exact binding failures. They are not owner-review,
  rights, promotion or private-integration evidence.
- Added a `source-only-validators` job to the local GitHub workflow. It runs the
  five new guard/validator modules in a no-site, no-bytecode stdlib process.
  Existing lint, isolated controls and general pytest jobs retain their behavior.
  No additional historical tests are ignored, deselected, skipped or xfailed.

The combined new suite passes **120 tests** locally. The existing 43 CI guard
tests also pass. Workflow YAML and command shape were checked. Frozen lint
integrity passes with zero new findings and all 815 historical findings retained
unchanged; this is not an all-source lint-clean claim. Historical implementation,
test, plan, evidence and marker bytes were not changed.

## Incomplete verification and resume point

An attempt to copy the reviewed new source/test files into the existing temporary
source clone was rejected by automatic approval review with a workspace-credit
error. No alternate copying or indirect execution route was used. The intended
fresh-clone 120-test replay and later commands in that rejected batch did not run.
The successful local 120-test result must not be described as a fresh-clone or
Linux GitHub result.

Restore the workspace approval/credit service and authorize retry of that exact
temporary source-copy verification. Then verify the source-only job in the clone,
finish review of the remaining missing-input and five genuine source-hash cases,
and integrate the explicit historical BF16 route without counting known failures
as passes. Only claim full CI success after a complete Linux run establishes it.

The new workflow and tests remain local and uncommitted. The last verified remote
run and roadmap counts are retained in v0.9: four of eight original roadmap gates
fully closed, six pre-GPU conditions still partly open, no real SFT launched, and
`production_eligible=false`.
