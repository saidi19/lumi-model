# Existing Aya successor and receipt: bounded comparison

## Result and scope

Completed on 2026-09-06. Exactly one actual local invocation passed these two
unchanged original test bodies, in order, including both setup/call/teardown
lifecycles:

1. `tests/test_aya_source_promotion.py::test_successor_changes_only_the_existing_aya_entry`
2. `tests/test_aya_source_promotion.py::test_promotion_receipt_is_reproducible_raw_free_and_nonproduction`

The existing successor equals the original in-memory builder's expected registry:
ten entries before and after, with only the approved Aya entry changed. The
existing promotion receipt equals its original in-memory reconstruction and
passes the unchanged receipt verifier. The attribution manifest's 1,478 entries
and prerequisite report/owner-receipt seals also pass the original prerequisite
checks. No actual invocation was retried.

This verifies an existing historical promotion; it performs no new promotion,
registry update, artifact publication, owner decision, production authorization,
dataset seal, training or inference. `production_eligible` remains false. The
receipt's historical `candidate_source_registry_promotions=1` is not a count of
actions performed by this check.

The preceding prerequisite step is preserved at commit
`9b75f933140132cca00f4fe806b3dadc1b713b84`, with
[Source checks 4/4 PASS](https://github.com/saidi19/lumi-model/actions/runs/34027674788)
and [Spark PASS](https://github.com/saidi19/lumi-model/actions/runs/34027674789).
No prior checker main, private replay or original sibling test is invoked here.

## Bindings and read boundary

The new manifest is `configs/lumi-aya-promotion-snapshot-inputs-v0.1.json`, SHA-256
`dd6d15dc9979da8f3e30da4a0dab97ae0502ae760b1a22e1bd79d2e7539baecb`.
It fixes fourteen inputs by exact logical path, byte size and hash. They are the
original plan, data contract, predecessor registry, promotion implementation and
runner, historical registry implementation, hash-helper source, original tests,
policy source, existing successor registry, materialization report, owner receipt,
promotion receipt and attribution manifest. Each physical leaf is read once.

The four evidence objects are read last, with attribution last of all. Together
they contain 1,397,494 bytes. Total bound reads are exactly fourteen files and
1,524,967 bytes. Candidate JSONL, owner packets/decisions, upstream datasets,
tokenizers, model objects and embedded benchmark/template/registry paths are not
opened. Input contents and identifiers remain inside the process; output contains
only fixed stage/test identifiers, booleans and aggregate accounting.

The original plan's eight binding hashes remain unchanged. Its registry-source
binding is satisfied by the already committed, exact historical source fixture
`tests/fixtures/aya_source_registry_9f20652.py.txt` (14,927 bytes; SHA-256
`9f20652e85d0aede7b6ea75799f6e9746c5f35487b28e4bf39bdf854c80330e8`).
Only its physical read location is mapped; its original logical path stays
`src/lumi/posttraining_sources.py`.

Unlike the preceding hash-only prerequisite check, this step executes selected
historical registry classes and validators from those authenticated fixture
bytes. The current registry implementation, SHA-256
`10aa8e195bcab2613f3941f22f5303fc1cc8264403b4d7f20d04230bfcb40e50`,
is not replaced or used as a fallback. Its newer GitHub-URL support is not added to
the old validator. The current policy implementation is separately hash-bound by
the new manifest; only its policy-loading closure is selected. The original
promotion runner remains hash-only and never executes.

All fourteen leaf inputs authenticate before projection AST or JSON parsing.
The eleven-blob immutable snapshot serves thirty-five virtual opens and
4,413,108 logical read bytes across both tests, with no budget reset between them.
The first test consumes sixteen opens and the second nineteen. Expected registry
and receipt values exist only as in-memory dictionaries/bytes, never as new files.

The thirty-second limit covers control loading, trusted runtime startup, input
authentication, both tests, preservation checks and cleanup. A separate earlier
runtime-only probe passed two inert synthetic assertions with zero bound reads.
The real invocation ran with no hash doubles or rewritten original assertions.

## Source projection and execution guards

Entry point: `scripts/check_aya_promotion_snapshot_local_v1.py`, SHA-256
`c77f2f197ba9ac23e09d1a58d391e0a4d9ff39d7682815b0778415d2a01f7b00`.
Projection: `src/lumi/posttraining_aya_promotion_snapshot_v1.py`, SHA-256
`1a02ace306d1513ea82abdec3dc910abad63610d4d23f8c2580f487bc09a1307`.
Lifecycle: `src/lumi/posttraining_aya_promotion_lifecycle_v1.py`, SHA-256
`2fccc2111a6973b5df27e8006d6a17cad2d1ef5fbdb0e673ba6de91f5695e79c`.

Original selected constants, functions, dataclass declarations and both test
bodies retain their exact AST statements, filenames and source locations. Five
isolated namespaces separate policy, registry, hash helpers, promotion helpers
and tests. Temporary dataclass registration rejects namespace collisions and
cleans up its own entries. Full original modules, ML/data-quality imports,
publisher operations and unselected tests are excluded.

A fresh authenticated helper namespace reuses only prior IO/runtime facilities,
not any old main or one-node test runner. All four projection/lifecycle control
sources authenticate before any executes, and their package has no source-path
import fallback. The new lifecycle requires one exact ordered collection, six
ordered passing phase reports, no skips/xfails, pytest exit zero and preservation
of the installed test namespace. It invokes pytest exactly once. Any missing,
extra, repeated, failed or reordered event prevents PASS.

PASS is published only after exact shared read accounting, unchanged input
metadata, zero blocked-operation attempts, descriptor cleanup, alarm cleanup and
the final unexpired deadline check. Failures expose only fixed codes, never raw
exception text or a partial-success result. There is no automatic retry.

The fourteen-leaf budget excludes trusted interpreter/pytest, directory and
bounded control-source reads. Python audit guards and output suppression are
defense in depth, not an OS sandbox or native syscall tracer. Source/document
authoring and Git backup are separate from the zero-artifact-write local check.

## Observed aggregate result

```json
{
  "status": "PASS_AUTHENTICATED_AYA_PROMOTION_TESTS_ONLY",
  "original_test_bodies_executed": true,
  "snapshot_backed_pytest_nodes_passed": 2,
  "existing_successor_matched": true,
  "existing_promotion_receipt_matched": true,
  "historical_registry_validators_executed": true,
  "predecessor_source_count_verified": 10,
  "successor_source_count_verified": 10,
  "existing_registry_changed_entries_verified": 1,
  "existing_receipt_claim_candidate_promotions": 1,
  "existing_receipt_claim_production_promotions": 0,
  "attribution_records_validated": 1478,
  "bound_input_opens": 14,
  "bound_input_bytes_read": 1524967,
  "snapshot_accounting": {
    "bound_blob_count": 11,
    "virtual_opens": 35,
    "logical_read_bytes_debited": 4413108,
    "maximum_virtual_opens": 35,
    "maximum_logical_read_bytes": 4413108,
    "maximum_gzip_bytes": 1,
    "filesystem_access_performed": false,
    "execution_permitted": false
  },
  "input_files_preserved": true,
  "input_files_written": 0,
  "blocked_operation_attempts": 0,
  "upstream_source_rows_read": 0,
  "candidate_rows_read": 0,
  "current_source_replaced": false,
  "candidate_promotion_performed": false,
  "source_registry_changed": false,
  "owner_decisions_reproduced": false,
  "new_owner_approval_created": false,
  "human_authorship_independently_verified": false,
  "ordinary_checkout_tests_passed": false,
  "original_modules_fully_imported": false,
  "full_private_integration_passed": false,
  "execution_authorized": false,
  "production_eligible": false,
  "network_used": false,
  "gpu_used": false,
  "inference_called": false
}
```

## Verification and remaining work

Local synthetic regressions passed 235 tests and 1,019 subtests, including 61 new
core/lifecycle/CLI tests. Exact CI Ruff rules passed for every new Python file.
Frozen-lint verification retained the same 815 historical findings across 163
files, with zero new findings; this is not an all-source clean lint claim.

Synthetic positive tests explicitly substitute only the original test namespace's
hardcoded predecessor-hash assertion and the two historical private-seal values,
because invented fixtures cannot match those digest preimages. Real hashing still
runs for the predecessor read and all other synthetic bindings/receipt seals.
Separate tests reject the invented inputs without those doubles, and test the
original helpers and tamper boundaries. No substitution is used in the actual run.

These are authenticated snapshot-backed tests, not an ordinary-checkout or full
private-integration pass. The unchanged historical source requirement prevents
claiming that the newer active registry source satisfied the old plan. The frozen
327 OPEN integration-node inventory and fixture-dependent skip inventory remain
unchanged; separate snapshot evidence does not silently close those entries.

The eight-gate roadmap remains four gates fully closed, Gate 5 (SFT preparation)
partial, and Gates 6 (actual SFT), 7 (quality/safety evaluation) and 8 (serving)
pending. Guarded target-Linux execution, authentic final-base import/restore,
private-data relocation, CUDA acceptance and the complete execution ticket remain
separate prerequisites. The S3-backed pretrained Lumi base is untouched.

Only the new manifest, implementation/lifecycle/CLI, synthetic tests and this
raw-text-free document are eligible for the scoped source commit. Existing
evidence, original source/tests, registries, workflows and inventories are not
edited. Hosted Source/Spark checks for that commit are evaluated separately;
they must not rerun this private evidence check.
