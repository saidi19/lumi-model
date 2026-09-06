# Authentic Aya existing-sample reconstruction

## Scope and status

Completed on 2026-09-06. Exactly one actual private invocation passed the
unchanged original test: one node collected, with passing setup, call and
teardown. All 1,478 existing candidate records were parsed and the original
32-record sample identity matched. There was no retry.

The exact target is the unchanged original
`tests/test_aya_bengali_owner_review.py::test_actual_plan_rebuilds_exact_sample`.
This checks the existing 1,478-record candidate against its original sealed
materialization report and deterministic 32-record sample identity. It does not
construct a new candidate, reproduce owner decisions, create owner approval,
independently prove human authorship, or authorize production use.

The preceding receipt-only check was published at commit
`ac2de8be62c36a1e578a1bd51a5364da2219021e`:
[Source checks 4/4 passed](https://github.com/saidi19/lumi-model/actions/runs/34024906135)
and [Spark checks passed](https://github.com/saidi19/lumi-model/actions/runs/34024906158).
The receipt, prior checkers and their evidence are not rerun or modified here.

## Exact input boundary

New manifest: `configs/lumi-aya-sample-snapshot-inputs-v0.1.json`, SHA-256
`fc08077b48e780d76813875a84efe7439f39b20dc17bafa554d091ee5f7b107d`.
Entry point: `scripts/check_aya_sample_snapshot_local_v1.py`, SHA-256
`2c857994f553fdc6d92d705395922da9ba29e8865f623b504f2472d247579cc9`.
Projection: `src/lumi/posttraining_aya_sample_snapshot_v1.py`, SHA-256
`4a6e092dec8fb6520ba3ddf7c10002238b1078e9fd53f75dd633d932324be993`.

The physical budget is ten bound leaf-file reads totaling 6,551,911 bytes:
the original plan, five Python source files, report, HTML packet, attribution
manifest, and candidate. Source/control inputs precede the private objects;
the 5,016,358-byte candidate is last. The three private objects together total
6,480,319 bytes. Each file has an exact path, length and SHA-256 binding and is
read once using no-follow descriptors. No upstream source dataset is read.

The immutable snapshot contains seven blobs: the plan and its six bindings.
The unchanged test performs nine virtual opens: one plan read, six binding
hashes, one report read and one candidate read, totaling 11,523,272 logical
bytes. The candidate is physically read once, then used twice in memory for
hashing and JSONL parsing. Private text and identifiers necessarily exist in
process memory but are never emitted or written by the checker. The HTML packet,
attribution manifest and finalizer are hash-only: they are not parsed, rendered
or executed. The packet is not a source of instructions.

All ten inputs authenticate before projection AST/JSON parsing. The selected
original constants, functions and test statements retain their filenames and
line positions. The original `file_sha256` comes from its actual first-party
import origin. The OASST module supplies only canonical JSON, value hashing and
deterministic sampling. Imports, unused test bodies, owner decision validators,
receipt validators, finalizers, materializers and training functions are excluded.

The checker uses a fresh hash-authenticated copy of the prior receipt helper,
which supplies the original one-node pytest lifecycle. Only the identity and
new core-control bindings are configured. No old main is called. Snapshot,
selector and new core bytes are authenticated before execution; the isolated
package has no source-directory import fallback. Control drift stops before
private input acquisition.

The 30-second deadline includes control loading, pytest startup, input reads,
the test, preservation checks and cleanup. Success requires exactly one node
with passing setup, call and teardown, exact accounting, unchanged input metadata,
no boundary violations, successful descriptor cleanup and an unexpired deadline.
There is no operational retry loop. Trusted runtime, directory and bounded
control-code reads are separate from the ten bound input reads. Audit guards
and Python output suppression are defense in depth, not an OS sandbox or native
syscall tracer. Source/doc authoring is outside this zero-artifact-write check.

## Observed aggregate result

```json
{
  "status": "PASS_AUTHENTICATED_AYA_SAMPLE_TEST_ONLY",
  "original_test_body_executed": true,
  "snapshot_backed_pytest_nodes_passed": 1,
  "pytest_lifecycle": {
    "collected": 1,
    "phases": {"setup": "passed", "call": "passed", "teardown": "passed"}
  },
  "bound_input_opens": 10,
  "bound_input_bytes_read": 6551911,
  "candidate_records_validated": 1478,
  "candidate_rows_parsed": 1478,
  "sample_records_validated": 32,
  "candidate_sample_reconstructed": true,
  "original_sample_identity_matched": true,
  "snapshot_accounting": {
    "bound_blob_count": 7,
    "virtual_opens": 9,
    "logical_read_bytes_debited": 11523272,
    "maximum_virtual_opens": 9,
    "maximum_logical_read_bytes": 11523272,
    "maximum_gzip_bytes": 1,
    "filesystem_access_performed": false,
    "execution_permitted": false
  },
  "input_files_preserved": true,
  "input_files_written": 0,
  "blocked_operation_attempts": 0,
  "upstream_source_rows_read": 0,
  "new_candidate_constructed": false,
  "new_owner_approval_created": false,
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

This is snapshot-backed execution of the original assertion, not execution of
the full ordinary-checkout module. No candidate, packet, attribution, report or
receipt body was printed. Prior artifacts were not regenerated.

## Verification and remaining work

One runtime-only probe passed using a synthetic assertion and zero bound/private
reads. Final local regression selection passed 132 tests and 379 subtests,
including 38 new core/CLI tests. Original function/test ASTs were also exercised
on explicitly synthetic fixtures without any private input access. Exact CI
Ruff rules passed. Independent source/control review found no blocker before
the actual invocation. Full frozen-lint verification retained 815 historical
findings across 163 files, with zero new findings; it is not an all-source clean
lint claim.

The original source/test, prior artifacts, workflows, source-partition bindings
and 327-node historical unresolved inventory remain unchanged. This local
result does not silently decrement that inventory, waive fixture-dependent
skips, or establish full ordinary-checkout private integration.

Roadmap Gate 5 remains partial. Remaining authentic private-fixture checks,
guarded target-Linux worker, final base import/restore, private-data relocation,
CUDA acceptance and the complete execution ticket remain separate prerequisites.
No registry mutation, promotion, dataset seal, GPU/SFT, remote worker operation,
or modification of the S3-backed pretrained base is part of this check.

Hosted checks for the new source commit are evaluated separately. They must
never automatically invoke the private reconstruction check.
