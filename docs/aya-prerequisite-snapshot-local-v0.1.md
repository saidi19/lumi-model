# Authentic Aya prerequisite evidence check

## Scope and status

Completed on 2026-09-06. Exactly one actual local invocation passed the unchanged
original prerequisite test, including its full setup/call/teardown lifecycle.
All 1,478 attribution identities/provenance entries, the exact report seals and
577,453-token metric, and the receipt's 32 acceptances were verified. There was
no retry. One earlier runtime-only probe used zero bound/private reads.

Target: the unchanged original
`tests/test_aya_source_promotion.py::test_exact_reviewed_aya_prerequisites_pass`.
It joins the existing attribution manifest, materialization report and owner
receipt. It does not run a promotion, reconstruct human decisions, independently
verify human authorship, or authorize production use.

The preceding original sample check passed at commit
`00df6fcaf8e78c3d9b3cae15a597d77fb8d52a7c`, followed by
[Source checks 4/4 PASS](https://github.com/saidi19/lumi-model/actions/runs/34026421630)
and [Spark PASS](https://github.com/saidi19/lumi-model/actions/runs/34026421621).
The sample and receipt checkers are not rerun by this step.

## Historical source binding recovered without changing active source

Control-only preflight found that the original promotion plan binds
`src/lumi/posttraining_sources.py` to SHA-256
`9f20652e85d0aede7b6ea75799f6e9746c5f35487b28e4bf39bdf854c80330e8`.
The current 15,025-byte source is instead
`10aa8e195bcab2613f3941f22f5303fc1cc8264403b4d7f20d04230bfcb40e50`.
The local Git path history contained only that newer source version.

An exact 14,927-byte copy was found at `build/lib/lumi/posttraining_sources.py`.
Its hash matches the original plan. The newer source adds GitHub repository-URL
support; neither version is edited or executed here. The exact old bytes are
preserved in the new source-only fixture
`tests/fixtures/aya_source_registry_9f20652.py.txt`, with the same approved hash.
This fixture can be backed up in Git without private data. No historical source
was guessed, no expected digest was relaxed, and no active source was rolled back.

The new manifest maps only this physical fixture path to the original logical
`src/lumi/posttraining_sources.py` snapshot path. The original plan and all eight
of its binding hashes remain unchanged. The fixture is hash-only, never imported
or compiled. An ordinary checkout with the newer source still does not satisfy
that historical binding; this is authenticated snapshot-backed execution only.

## Exact read and execution boundary

Manifest: `configs/lumi-aya-prerequisite-snapshot-inputs-v0.1.json`, SHA-256
`666d62dc3a5a31d895ee52defa8fcc78b1674bcd1dc5e8361617b339d7238f80`.
Entry point: `scripts/check_aya_prerequisite_snapshot_local_v1.py`, SHA-256
`f9717cc10f227c6ff26dde30f95f849775c689cb19f55e2d5f5c8051301dc620`.
Core: `src/lumi/posttraining_aya_prerequisite_snapshot_v1.py`, SHA-256
`dd4e03e676fcf6c332aca40733736cf32d7ee69ddaff0ae869ec68a094b45822`.

The bound budget is eleven physical leaf-file reads, 1,472,969 bytes, once each.
Eight source/control inputs precede the materialization report, owner receipt
and attribution manifest; attribution is last. The three evidence objects total
1,393,247 bytes. Candidate JSONL, owner packet, decisions, successor-registry
contents, promotion-receipt contents, upstream datasets and model objects are
outside the input closure.

All eleven inputs authenticate before projection AST/JSON parsing. The snapshot
contains nine blobs (plan plus eight bindings). The original test makes twelve
virtual opens: plan, eight binding hashes, and three evidence reads, totaling
2,842,451 logical bytes. Each evidence file is physically read once and used twice
in memory for hashing and parsing. The registry source, promotion runner, data
contract and predecessor registry are hash-only, not executed or semantically
loaded. Attribution identifiers/content remain in process memory only; output
contains fixed stage codes, booleans and aggregate counts, never IDs or raw text.

Only exact original constants, prerequisite helpers, first-party hash helpers,
and the selected original test/helper statements are compiled, with original
filenames and line positions. The old authorization fields are checked as data,
not treated as fresh authority to run any promotion. All operational builders,
policy/registry loaders and sibling test bodies are excluded.

The fresh hash-pinned helper namespace reuses the prior one-node pytest lifecycle;
it never calls a previous checker main. Snapshot/selector/core bytes authenticate
before execution, with no mutable source-directory import fallback. The 30-second
deadline includes control loading, runtime startup, test, preservation and cleanup.
PASS requires exact accounting, one collected node, three passing lifecycle
phases, unchanged input metadata, zero boundary violations, successful cleanup
and the final unexpired deadline. There is no automatic retry.

The bound leaf-input budget excludes trusted interpreter/pytest, directory and
bounded control-code reads. Python audit guards and output suppression are defense
in depth, not an OS sandbox or native syscall tracer. Source-fixture copying and
documentation authoring are separate from the zero-artifact-write check.

## Observed aggregate result

```json
{
  "status": "PASS_AUTHENTICATED_AYA_PREREQUISITE_TEST_ONLY",
  "original_test_body_executed": true,
  "snapshot_backed_pytest_nodes_passed": 1,
  "pytest_lifecycle": {
    "collected": 1,
    "phases": {"setup": "passed", "call": "passed", "teardown": "passed"}
  },
  "bound_input_opens": 11,
  "bound_input_bytes_read": 1472969,
  "attribution_records_parsed": 1478,
  "attribution_records_validated": 1478,
  "materialization_supervised_tokens_verified": 577453,
  "owner_receipt_accepted_records_verified": 32,
  "prerequisite_evidence_verified": true,
  "historical_source_copy_authenticated": true,
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
  "upstream_source_rows_read": 0,
  "candidate_rows_read": 0,
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

This result validates existing evidence; it does not create owner approval or
rerun materialization, tokenization, candidate sampling, curation or promotion.

## Verification and remaining gates

Final local regressions passed 174 tests and 635 subtests, including 42 new
core/CLI tests. Exact CI Ruff rules passed, and independent source/control review
found no blocker before the actual invocation. Frozen-lint verification retained
815 historical findings across 163 files with zero new findings; this is not an
all-source clean lint claim. Synthetic positive control-flow tests explicitly substitute
a schema-based seal hash double only after construction because invented evidence
cannot match the two hardcoded historical digest preimages. Separate tests verify
the original hash helpers and reject false seals without that double. No real
invocation uses a hash double, rewritten original statement or relaxed assertion.

The original eight-gate roadmap remains four gates fully closed. Gate 5 (SFT
preparation) is partial; Gate 6 (bounded actual SFT), Gate 7 (quality/safety
evaluation), and Gate 8 (serving/deployment) remain pending. This prerequisite
check is work within Gate 5, not a new top-level gate. The 327 OPEN historical
integration nodes and fixture-dependent skip inventory are not a count of product
gates and are not silently decremented by separate snapshot evidence.

Guarded target-Linux execution, authentic final-base import/restore, private-data
relocation, CUDA acceptance and the complete execution ticket remain separate
prerequisites. No existing evidence, source registry, workflow, inventory, model
or S3-backed pretrained base is modified. No seal, promotion, remote worker,
GPU/SFT or inference action is included. Hosted checks for this new source commit
must be evaluated separately and never automatically rerun this private check.
