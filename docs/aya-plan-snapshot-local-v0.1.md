# Authentic Aya historical-plan check

## Completed on 2026-09-06

One bounded local invocation passed the exact original
`test_aya_repair_plan_is_exact_local_and_non_authorizing` body under pytest in an
authenticated source projection and immutable input snapshot. Exactly one node
was collected; setup, call and teardown all passed. All 15 original plan bindings
were authenticated. No Aya source page, source row or candidate was processed.

The historical failure was an environment/source-version mismatch: the plan
requires `posttraining_data.py` digest
`df8297e753e97028999c6bcd203edc2a00a02147f7d9ecb2255569c480363970`, while
the legitimate current implementation differs. The required 21,605-byte blob
was obtained from local Git revision
`e08bddc8fbd8afe416a0d09aed6a115f06870f36`. For this test it is **hash input only**;
filtering code is not executed. Neither the current source nor the old binding
was changed to make the check pass.

## Scope and implementation

The exact specification is `configs/lumi-aya-plan-snapshot-inputs-v0.1.json`,
SHA-256 `a519b0e427077db6c39d0f94409e0def5965cdb85335d13fbece3d5ff81a020c`.
The entry point is `scripts/check_aya_plan_snapshot_local_v1.py`.

The invocation used 17 no-follow file reads totaling **10,433,360 bytes**, plus
the separate 21,605-byte historical Git blob. The 30-second deadline includes
control acquisition, Git lookup, runtime startup and cleanup. The existing
reviewed bounded reader is hash-pinned and reused with tighter limits; no new
general-purpose private reader or curation runner was introduced.

The files include the old plan, original source/test, original binding objects,
and the source containing the original file-hash helper. Private audit/inventory
and tokenizer objects are hashed as opaque bytes. No path embedded inside the
inventory is followed, and the tokenizer and benchmark registry are not decoded
or used for processing. The hash-only snapshot needs 16 virtual opens debiting
10,422,648 bytes. No compressed data is expanded.

The projection compiles only the authenticated original constants, four Aya
loader/resolver helpers, original OASST file-hash helper and selected test body.
Their statements and assertions are unchanged, with original filenames and line
locations. Imports and unrelated curation/model functions are not executed.
`Path` is explicitly bound to the existing immutable Snapshot implementation.

Pytest receives the projected module under the exact original node identity.
It uses an empty configuration, disabled plugin auto-loading, no conftest,
plain assertions, and no cache/logging/faulthandler or output capture files.
Success requires exactly the selected node and three distinct passing lifecycle
reports: a skip, failure, missing/duplicate phase or collection mismatch fails.
The temporary module registration is removed on exit without replacing an
existing namespace. The complete original module is not imported.

This is **snapshot-backed original-test execution**, not a claim that an ordinary
checkout pytest invocation now passes. The old checkout hash mismatch remains
correctly detectable. It is also not an Aya repair/materialization run or a
reproduction of the 23 source pages and 1,534 source rows named by the old plan.

## Observed result and preservation

```json
{
  "status": "PASS_AUTHENTICATED_AYA_PLAN_TEST_ONLY",
  "original_test_body_executed": true,
  "snapshot_backed_pytest_nodes_passed": 1,
  "ordinary_checkout_test_passed": false,
  "original_module_fully_imported": false,
  "bound_plan_objects_authenticated": 15,
  "bound_input_opens": 17,
  "bound_input_bytes_read": 10433360,
  "historical_source_bytes_read": 21605,
  "input_files_preserved": true,
  "blocked_operation_attempts": 0,
  "source_rows_read": 0,
  "candidate_rows_read": 0,
  "inventory_paths_followed": 0,
  "tokenizer_loaded": false,
  "input_files_written": 0,
  "network_used": false,
  "gpu_used": false,
  "inference_called": false,
  "full_private_integration_passed": false,
  "execution_authorized": false,
  "production_eligible": false
}
```

There was one actual guarded invocation, which passed; it was not retried.
Earlier runtime-only probes used a synthetic assertion with zero bound/private
reads. They found that pytest requires `isatty()` on its output sink; the new
quiet stream now implements that API. No shared/frozen reader was modified.
The final runtime-only probe passed with zero blocked operations.

Twenty new synthetic tests pass. The combined local regression selection passed
94 tests and 123 subtests, including existing reader lifecycle and CI-routing
guards. All new Python files pass the exact CI Ruff rules. Independent review
confirmed exact-node accounting and led to a post-cleanup elapsed-time check;
success is released only after preservation, descriptor cleanup and timer
teardown. Outputs contain fixed codes and aggregates, not private exception text.
Python output redirection and audit hooks are defense in depth, not native-output
suppression or an OS-sandbox claim.

The original source, plan, test, prior evidence and immutable 327-node historical
inventory remain unchanged. This result is separate local evidence for one
historical dependency check, not a subtraction from that frozen inventory or a
whole-project readiness pass. Other private integrations still need evidence;
source-suite success does not close the full-readiness gate. No dataset
promotion/seal, rights approval, remote data access, model/GPU action or SFT run
was performed. The pretrained base and S3 backup were not touched.
