# CI preservation policy v0.2

This extends [v0.1](ci-preservation-policy-v0.1.md) with explicit BF16 history
execution and source-only validator coverage. It does not repair historical lint
debt, waive private integration or establish a GPU/production gate.

## Frozen BF16 tests execute under an explicit history contract

The old v1 importer globally casts configured buffers to BF16. Its two BF16
success-expectation tests therefore raise `HF reload tensor mismatch`. The
already-existing v2 successor preserves configured buffer dtype and values.
Neither historical implementation nor either original test file is modified.

The new runner executes all 14 v1 and 17 v2 tiny CPU cases. It requires 12 v1
passes, exactly two specified v1 errors and all 17 v2 passes. An accepted
historical error must have the exact built-in type, message and pinned original
raise location. A different error, missing case, skip, unexpected success or
current-v2 failure rejects the contract. Six source/test file hashes are checked
before and after execution. Output explicitly says 29 passed, two required
historical errors, `all_tests_passed=false` and no real-model acceptance.

The normal general-suite invocation deselects only these two original nodes
because they execute in the required preceding history step:

- `tests/test_sft_checkpoint_import_torch_v1.py::CheckpointImportTorchTests::test_real_bfloat16_roundtrip`
- `tests/test_sft_checkpoint_import_torch_v1.py::CheckpointImportTorchTests::test_real_adapter_bfloat16_and_worker_consumer`

The remaining v1 and all v2 cases also run in ordinary pytest. There is no
module-wide exclusion, generic xfail or `continue-on-error`. The history step has
a four-minute workflow timeout and a 180-second child deadline. General pytest
uses `always()` so a failed history step does not suppress its diagnostics; the
history failure still fails the job. Missing dependencies remain errors.

Ten new stdlib routing tests enforce the literal reviewed test-job block, exact
two-node mapping, mandatory history execution and unchanged general-suite scope.
The YAML was parsed separately as a syntax/structure check; the literal guard
does not claim to be a general YAML parser.

## Source-only unit coverage is not private replay

The added source-only job runs 130 tests across six modules using `python -S -B`.
It checks the draft fixture plugin, history runner, workflow route and synthetic
production-policy, learning-seal and teacher-canary validators. Fixtures are
explicitly synthetic and never published as approvals, rights or operational
evidence. No validator is replaced to make its rejection cases pass.

Eight additional synthetic Aya/safety binding tests use installed dependencies
and `-B` because the existing import chain requires `certifi`. They are collected
by general pytest, not by the stdlib-only job. They demonstrate that a changed
implementation does not retroactively satisfy an old hash binding.

The four-node private-fixture plugin remains opt-in and is **not enabled in the
GitHub workflow**. Its guard tests run, but existing missing-input tests retain
their normal blocking behavior. No broader exception inventory was added.

The last full Linux diagnostic found 322 missing-input-dependent outcomes, five
actual historical source-hash rejections and the two v1 BF16 errors. Only the
two BF16 cases receive a new execution route here. The other failures require
separate, reviewed treatment. A passing targeted job is not full CI success.

Historical plans, reports, started markers, owner materials and source bytes are
preserved. No private datasets or model weights are uploaded by this CI source
change. The exact 815-finding lint baseline remains unchanged, and every new
Python file must be lint-clean. `production_eligible=false` remains unchanged.
