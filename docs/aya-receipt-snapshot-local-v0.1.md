# Authentic Aya receipt-only check

## Completed on 2026-09-06

One bounded local invocation passed the unchanged original
`tests/test_aya_bengali_owner_review.py::test_actual_owner_receipt_is_complete_and_nonpromoting`.
Exactly one node was collected and its setup, call and teardown passed. The
existing receipt records 32 accepted and zero rejected sample rows and retains
the nonpromotion status. This is validation of that existing receipt, **not**
new owner approval, reproduction of human decisions or independent proof of
human authorship.

The preceding coverage dependency was completed at commit
`f6db0f4fceafc3751d967f9cf77fd0d29510ceec`: [Source checks 4/4 passed](https://github.com/saidi19/lumi-model/actions/runs/34023896116)
and [Spark checks passed](https://github.com/saidi19/lumi-model/actions/runs/34023896084).
That Source run reported 2,477 passed, 58 skipped and 329 deselected; its isolated
job separately passed 273 controls without failure or skip. Those historical
counts are not rewritten by this local result.

## Exact input and execution boundary

The new specification is `configs/lumi-aya-receipt-snapshot-inputs-v0.1.json`,
SHA-256 `155a4608c1188f1f7de7603893d821f7f715f53c733ff62167455d3fdeb9e4bf`.
The entry point is `scripts/check_aya_receipt_snapshot_local_v1.py`.

The receipt's file SHA-256 is
`66c28d06816e6fc69cbde022fa0ebb3820f44ad0e1daf611f1857c6ca9e243be`;
its approved internal digest is
`e23851af5718d10acdd120ca1e0c200672fb5c794ad503ed4d1f7f1650cc685f`.
These are the preexisting bindings in the Aya source-promotion plan and source,
not hashes invented from the result of this check. No promotion code is executed.

The checked invocation consumed four bound file reads totaling 45,471 bytes:
three original Python source files first, then exactly one read of the
1,667-byte receipt. It used no-follow descriptors, exact sizes and file hashes,
and preservation checks before releasing success. No candidate, source row,
owner-decision file, tokenizer, benchmark object or model was opened. The
candidate/sample-reconstruction sibling test was not selected.

The historical validator, hash helpers and selected test statements are compiled
unchanged at their original filenames and line positions. Only the receipt is
visible through the immutable snapshot: exactly one virtual open and 1,667
logical bytes. Imports, sample readers, owner finalizers and other test bodies
are excluded. Duplicate JSON keys, changed bindings and open operation flags
are rejected before the original test runs.

The checker reuses an isolated, hash-authenticated copy of the previous one-node
pytest lifecycle helper, configured with the receipt node's identity. It never
calls that helper's old main or repeats the prior Aya plan check. Its bounded
reader is likewise hash-pinned. The Snapshot, projection helpers and new receipt
core are all hash-authenticated before their exact bytes execute; the isolated
package has no source-directory import fallback. Source drift is rejected before
consuming the private receipt read. No prior file is modified by this reuse.

The 30-second deadline includes control loading, runtime startup, verification
and cleanup. Trusted interpreter/pytest and bounded control-code reads are
separate from the four bound input reads. Python audit guards and output
suppression are defense in depth, not an OS sandbox, native syscall tracer or
proof of crash-persistent storage. No operational retry, remote action, GPU,
inference, artifact publisher, registry change or training was performed.

## Observed aggregate result

```json
{
  "status": "PASS_AUTHENTICATED_AYA_RECEIPT_TEST_ONLY",
  "original_test_body_executed": true,
  "snapshot_backed_pytest_nodes_passed": 1,
  "pytest_lifecycle": {
    "collected": 1,
    "phases": {"setup": "passed", "call": "passed", "teardown": "passed"}
  },
  "bound_input_opens": 4,
  "bound_input_bytes_read": 45471,
  "receipt_claim_accepted_records": 32,
  "receipt_claim_rejected_records": 0,
  "input_files_preserved": true,
  "input_files_written": 0,
  "blocked_operation_attempts": 0,
  "source_rows_read": 0,
  "candidate_rows_read": 0,
  "candidate_sample_reconstructed": false,
  "owner_decisions_reproduced": false,
  "human_authorship_independently_verified": false,
  "original_module_fully_imported": false,
  "ordinary_checkout_test_passed": false,
  "full_private_integration_passed": false,
  "execution_authorized": false,
  "production_eligible": false,
  "network_used": false,
  "gpu_used": false,
  "inference_called": false
}
```

There was exactly one actual receipt invocation and no retry. Two preceding
runtime-only probes used synthetic assertions and zero bound/private reads.
They are not additional receipt validations. Ordinary-checkout execution was
not performed; this is authenticated snapshot-backed original-test evidence.

## Local regression and remaining work

The final local selection passed 94 tests and 176 subtests, including 49 new
core/CLI tests and existing Aya/Snapshot regressions. Exact CI Ruff rules passed.
Full frozen-lint verification retained 815 historical findings across 163 files,
with zero new findings. The original source/test, receipt, plans, workflows,
source-partition bindings and 327-node historical inventory remain unchanged.

The 55 fixture-dependent skips documented in the earlier hosted audit remain a
historical inventory; this separate local result does not silently decrement it,
waive other nodes, or turn full private integration green. The next Aya task is
the separately bounded authentic candidate/sample-reconstruction check. It must
authenticate its complete input closure before any candidate content is read;
this receipt result does not authorize an old finalizer or consumed attempt.

Roadmap Gate 5 remains partial. The guarded target-Linux worker, authentic final
base import/restore, private-data relocation, CUDA acceptance and complete
execution ticket remain separate prerequisites. No GPU/SFT action, dataset seal,
production promotion or modification of the S3-backed pretrained base occurred.

This document records local verification before publication. Hosted checks for
the new source commit must be evaluated separately and never rerun this private
receipt check automatically.
