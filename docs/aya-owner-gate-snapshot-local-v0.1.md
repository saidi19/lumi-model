# Aya owner-receipt tamper rejection: bounded check

## Result and meaning

Completed on 2026-09-06. Exactly one actual local invocation passed the unchanged
original test
`tests/test_aya_source_promotion.py::test_rejected_or_tampered_owner_gate_cannot_be_promoted`,
including one collection and all three setup/call/teardown phases. No retry was
performed. A preceding runtime-only probe used zero bound-input reads.

The original test deep-copies the existing owner receipt in memory, sets
`rejected_records=1` and `all_sampled_rows_accepted=false`, leaves its original
seal unchanged, and requires the original verifier to reject it with the matching
`receipt_sha256 changed` error. That rejection occurred. No on-disk receipt or
other input was changed, and no new owner decision or promotion was performed.

The materialization report's genuine seal, metrics and boundary are checked before
the owner schema/seal. The owner seal rejects before the later owner-policy-field
checks or attribution provenance loop. Consequently this result is a tamper/seal
check, not independent verification of all rejected-owner policy branches or a
fresh complete prerequisite/provenance pass. Attribution metadata is parsed in
memory, but no attribution-validation count is claimed. No real hash function or
original test assertion was substituted in this invocation.

The preceding two-test successor/receipt comparison is preserved at commit
`f047572dfa93b07d06adb2bf5690c3d634916a6e`, with
[Source checks 4/4 PASS](https://github.com/saidi19/lumi-model/actions/runs/34028950761)
and [Spark PASS](https://github.com/saidi19/lumi-model/actions/runs/34028950759).
Neither of those tests nor the earlier prerequisite check was rerun here.
The active-data-contract test is already in normal Source CI, not a new private
test in this invocation.

## Exact inputs and control bindings

Manifest: `configs/lumi-aya-owner-gate-snapshot-inputs-v0.1.json`, SHA-256
`77dbcbe8929bc250d51b72916322052e535586c13ad771e2baec8317626d29af`.
Entry point: `scripts/check_aya_owner_gate_snapshot_local_v1.py`, SHA-256
`dc433aeb5f246f9f39ff6f73cefa1b2c11c21484b3cecd114612d6d189be59d7`.
Projection: `src/lumi/posttraining_aya_owner_gate_snapshot_v1.py`, SHA-256
`64474df2fd53c340e6821b5099201fb84e9fc198d0b544ace40998913761015b`.

The input closure retains the same eleven exact logical bindings as the earlier
prerequisite manifest: original plan; data contract; predecessor registry;
promotion implementation/runner; historical registry source; first-party hash
source; original test source; materialization report; owner receipt; attribution
manifest. Each physical leaf is read once, totaling 1,472,969 bytes. The last
three evidence objects total 1,393,247 bytes, with attribution read last.

Only the already preserved fixture
`tests/fixtures/aya_source_registry_9f20652.py.txt` supplies the original logical
`src/lumi/posttraining_sources.py` binding: 14,927 bytes, SHA-256
`9f20652e85d0aede7b6ea75799f6e9746c5f35487b28e4bf39bdf854c80330e8`.
Unlike the preceding successor comparison, this check uses registry source only
for hashing. The registry source, promotion runner, data contract and predecessor
registry are not executed or semantically parsed. The newer active registry
implementation is unchanged and is not a fallback for the historical binding.

All eleven inputs authenticate before projection AST or JSON parsing. The
immutable nine-blob snapshot serves twelve virtual opens and 2,842,451 logical
bytes: plan plus eight binding hashes, followed by the three prerequisite reads.
Candidate JSONL, owner packets/decisions, successor registry, promotion receipt,
upstream datasets, tokenizer/model objects and embedded paths are outside this
closure. No source/candidate text, private identifiers or exception text is output.

The thirty-second cumulative deadline covers control/runtime loading,
authentication, the single test, preservation and cleanup. Bound leaf accounting
excludes trusted interpreter/pytest, directory and bounded control-source reads.
Python audit guards are defense in depth, not an OS sandbox or native syscall
tracer. Source/document authoring and Git backup are separate from this
zero-artifact-write, zero-network/GPU/inference local check.

## Reuse and fail-closed execution

The new narrow projection reuses unchanged prerequisite authentication, plan
binding checks and selected constants/functions, plus the existing AST selector,
compiler and snapshot. Only the negative test and its `_load`/`_prerequisites`
helpers are selected. Original statements, filenames and line positions remain
unchanged. The previous positive-test builder is never called.

Five control sources authenticate before any executes: snapshot, AST helper,
prerequisite helper, strict lifecycle and the new projection. Their package has
no mutable source-path import fallback. A fresh namespace of the authenticated
two-node lifecycle implementation is first checked against its original fixed
configuration, then configured to this one fixed test and its three phase events.
The lifecycle's collection, report and summary logic is length-based; its prior
file and previously installed namespaces are not changed. No new lifecycle copy
or weaker one-node runner is introduced, and no previous checker main is called.

PASS requires exactly one pytest invocation/target, one ordered collection, three
ordered passing phases, no skip/xfail, zero exit status, intact namespace identity,
exact shared accounting, preserved input metadata, zero boundary violations,
successful descriptor/alarm cleanup and an unexpired final deadline. Rejection is
sticky; partial, repeated, missing or reordered reports cannot become PASS.

## Observed aggregate result

```json
{
  "status": "PASS_AUTHENTICATED_AYA_OWNER_GATE_TEST_ONLY",
  "original_test_body_executed": true,
  "snapshot_backed_pytest_nodes_passed": 1,
  "tampered_owner_seal_rejected": true,
  "owner_receipt_copy_changed_in_memory": true,
  "materialization_prerequisite_checked": true,
  "historical_source_copy_authenticated": true,
  "bound_input_opens": 11,
  "bound_input_bytes_read": 1472969,
  "snapshot_accounting": {
    "bound_blob_count": 9,
    "virtual_opens": 12,
    "logical_read_bytes_debited": 2842451,
    "maximum_virtual_opens": 12,
    "maximum_logical_read_bytes": 2842451,
    "maximum_gzip_bytes": 1,
    "filesystem_access_performed": false,
    "execution_permitted": false
  },
  "input_files_preserved": true,
  "input_files_written": 0,
  "blocked_operation_attempts": 0,
  "candidate_rows_read": 0,
  "upstream_source_rows_read": 0,
  "prerequisite_evidence_verified": false,
  "attribution_provenance_revalidated": false,
  "owner_policy_rejection_independently_verified": false,
  "source_registry_code_executed": false,
  "current_source_replaced": false,
  "candidate_promotion_performed": false,
  "source_registry_changed": false,
  "owner_decisions_reproduced": false,
  "new_owner_approval_created": false,
  "human_authorship_independently_verified": false,
  "ordinary_checkout_test_passed": false,
  "original_module_fully_imported": false,
  "full_private_integration_passed": false,
  "execution_authorized": false,
  "production_eligible": false,
  "network_used": false,
  "gpu_used": false,
  "inference_called": false
}
```

## Verification and remaining gates

Final local regressions passed 271 tests and 1,155 subtests, including 36 new
core/CLI tests. Exact CI Ruff rules passed. Frozen-lint verification retained the
same 815 historical findings across 163 files, with zero new findings; it is not
an all-source clean lint claim. Independent source review found no blocker in the
fresh singleton lifecycle configuration before the actual invocation.

Synthetic fixtures reuse only invented prerequisite-data helpers, never prior
test bodies. The invented owner receipt is sealed using real SHA-256. Only the
unavailable historical materialization preimage receives an explicit synthetic
test double; owner hashing remains genuine. A counterfactual restoring both
changed owner fields cannot pass the original expected-error assertion, and an
unmodified synthetic materialization seal cannot fake owner rejection. No such
double or counterfactual exists in the real invocation.

The four historical private Aya source-promotion nodes now each have separate
local snapshot-backed evidence across the preserved steps. This does not close
the ordinary-checkout/full-private integration gap. The frozen 327 OPEN inventory,
source partition and skip routing remain unchanged; they are not silently
decremented by separate evidence.

The eight-gate roadmap still has four gates fully closed, Gate 5 (SFT preparation)
partial, and Gates 6 (actual SFT), 7 (quality/safety evaluation), 8 (serving) pending.
Guarded target-Linux execution, authentic final-base import/restore, private-data
relocation, CUDA acceptance and a complete execution ticket remain separate work.
The S3-backed pretrained Lumi base is untouched. No source registry, old evidence,
workflow, source/test binding, production eligibility or model state is changed.

Only the six new manifest/core/CLI/test/documentation files belong in this source
commit. Hosted Source/Spark checks for that commit must be evaluated separately
and must not rerun this private check.
