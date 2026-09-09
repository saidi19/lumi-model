# SFT execution status v0.57

Source-CI correction plan, 2026-09-09. The user asked to resume, fix properly
without wasting credits, and whether GPU would accelerate Gate 5. The present
blocker is a test-isolation assertion, not GPU execution. No GPU is needed now.

## Accepted CPU work is preserved, not repeated

Commit `c4272e36bc98354067dd23a83931a3db58823d76` already passed CPU acceptance
in run `34345663337`, with one start, six ranks/three invocations/twelve origin
observations and eight rank/four global optimizer steps. The accepted image is
`sha256:321d50ae5207dee05cd58ec5436d97978becde2ee8a767f2e8768229e6a9ad43`.
Its private artifact `10101633554` is not downloaded or rebuilt by this task.
Acceptance report SHA-256 remains
`24712f08f2522a0319da1179178aa8c73eef5969303faea73de16d89a23b386b`.
The accepted commit, its historical workflow, all runtime source, software lock,
image and prior reports remain intact. This is a test-only successor.

The same commit's Source run `34345663154` failed exactly one newly added test;
5,051 passed, 58 skipped, 329 deselected, 9,975 subtests passed. That guard
incorrectly treated Torch imported earlier by other tests as an import performed
by its own test. The test was wrong; no new CPU runtime failure was observed.
Full failed log SHA-256:
`64cf9a3a1ef51e4bea7d3c16d63c7fbea54026e362dd64f4541aea5c7c9a85a6`.

## Four-file correction, no skip hiding the failure

1. Modify `tests/test_sft_cpu_version_lookup_v2.py`: preserve its 19 other tests;
   replace the incorrect process-global ML absence assertion with an AST-bound
   stdlib-only/no-dynamic-execution assertion for this test, plus the native-probe
   absence check. Add two tiny synthetic-registry regression tests: preexisting
   ML stubs are allowed, a native probe stub is rejected. No Torch import is
   needed locally and no module-registry dump is emitted on that rejection.
2. Modify `tests/test_sft_cpu_lookup_acceptance_source_v1.py`: update the exact
   count from 20 to 22 and assert the exact one-shot workflow condition below.
3. Modify `.github/workflows/sft-cpu-lookup-acceptance.yml`: retain the private
   repository/first-attempt condition and append equality to the accepted SHA.
   Consequently this later test-repair commit cannot execute the acceptance
   job. Keep its historical accepted run untouched. No dispatch/retry is added.
4. Add this plan. Do not publish the local-only duplicate v3 test or any other
   artifact/receipt. No runtime implementation, parser, policy or floor changes.

New test SHA-256:
`21a5ce00aed73c0d64ad031ef18cb3e78293aa987d1feea4b6974938fb5d2a30`.
New source-test SHA-256:
`0f6e17cf9ec860ca6cfeb434333322d49e82d8425b5348b885c4431d988effb7`.
New workflow SHA-256:
`5e6797dc94817220fc872c11295ac82ae2ec40668ed24d547a0adf4f993c32fa`.

## Verification and publication boundary

Selected local pure/synthetic/AST suite: 416 tests passed in 0.508s. After
strengthening the exact job-condition assertion, the affected 42 tests passed
again in 0.052s. Exact two-file Ruff `E,F,I,B,UP`, line length 100 / Python 3.11
target passed. Independent read-only review found no blocker or other live
consumer requiring changes. No native script, fixture, builder, Torch or model
was imported or run on the Mac.

Publish exactly these four files once on the existing private branch. The
expected run state is Source and Spark executing, with the lookup acceptance
workflow job **skipped** due to SHA mismatch. The old OCI and old diagnostic
workflows must not start. No accepted CPU workload is repeated. Do not label
the skipped job a new acceptance pass; the accepted image remains tied to c4272e3.
Read the two existing regression results, retain a small local result receipt,
and do not retry a failed workflow automatically.

Gate 5 remains partial. After Source is clean, the next engineering work is
integrating the accepted immutable candidate with guardian/worker/supervisor
admission. Dedicated host binding, original pretrained S3 checkpoint restoration,
exact data relocation and explicitly approved H100/CUDA testing remain later.
Keep GPU rental off until an actual GPU-dependent task is ready. No S3/SSH,
instance, real checkpoint/data, real inference/SFT, registry push, seal or
production promotion is authorized or performed here.
