# SFT execution status v0.6

Verified locally on 2026-09-05. **The local launch-contract checker and argument
compiler are complete and tested; the integrated GPU launcher is not complete.**
No S3 access, model restore, remote operation, GPU SFT, or data-content read was
performed. Production eligibility remains false. This adds to v0.5; its files,
the historical worker, and prior evidence were not rewritten.

## Starting model: our pretrained base in S3

SFT continues from **our own `lumi-4b-base`**, final step 76,294 at exactly
80,000,000,000 pretraining tokens. It does not restart pretraining, substitute a
third-party model, or use the tiny CPU fixture as the real base.

The new contract preserves the complete local
[final-base binding](../configs/lumi-4b-final-base-binding-v0.1.json), including
the exact S3 bucket/prefix and historical backup receipt/verification hashes.
Those records are historical evidence, not a new remote-availability check or
permission to download. The original DCP manifest and the future imported HF
checksum manifest remain distinct.

## What was implemented

- [Launch contract/compiler](../src/lumi/posttraining_sft_launch_contract_v1.py):
  strict schemas, JSON types and hashes; immutable base/data/profile identity;
  all five encoded-loader bindings; separate pretraining and SFT source commits;
  target/path/GPU/rendezvous checks; rank-multiplied read estimates; integer
  micro-USD cost arithmetic; SFT-specific approval subject and expiry checks.
- [Local CLI](../scripts/check_sft_launch_contract_v1.py): hash-bound metadata
  checks, synthetic tests, exclusive report/verification and started markers,
  a ten-minute local deadline, and bounded evidence writes. It has no transport
  or worker-dispatch path. `--execute` is unconditionally rejected before plan
  reads or evidence writes.
- [Versioned draft](../configs/lumi-4b-sft-gpu-smoke-launch-contract-v0.1.json):
  preserves the prior eight-rank BF16/FSDP, eight-record, two-step smoke/resume
  **design**. Target machine, import/relocation receipts, CUDA criteria, total
  resource limits and execution approval remain null. No GPU specification,
  price, path, approval, or receipt has been invented.
- [54 synthetic tests](../tests/test_sft_launch_contract_v1.py): incorrect scope,
  pretraining approval reuse, stale/incomplete bindings, bool/int substitution,
  unsafe paths, missing/nonportable metadata hashes, insufficient rank-scaled
  budgets, and attempted execution are rejected. Synthetic filled receipts and
  approvals are visibly test fixtures; even they cannot enable execution.

Receipt fields are structurally cross-checked, not authenticated or read from
the target host. Cost/runtime/I/O/GPU-memory limits are **validated declarations,
not enforced GPU quotas**. Worker argument previews are data only, have no
Python/torchrun/script prefix, and cannot be dispatched by this implementation.
Checkpoint pruning is absent; the legacy unbudgeted final-model digest is
explicitly excluded from previews and requires its own later verification budget.

## Why execution remains closed

Read-only inspection found that the existing v2 entry point initializes the
distributed/CUDA context before worker authorization; the worker reads inputs
before its authorization call, and that call uses generic **pretraining**
approval logic. Its numeric runtime/cost fields are not a resource supervisor.

The new compiler does not call, patch, or bypass that worker. A versioned worker
and supervisor must enforce SFT approval **before** CUDA initialization or real
input access and implement target-appropriate deadline, cost, I/O, memory, private
output and no-retry controls. A successful metadata check cannot close that gate.

Development checks also identified historical `lumi` package auto-registration
when ML packages are visible. The final checker requires Python `-S -B` before
importing it, preserving the old package while making the published passes
stdlib-only. A JSON numeric-type drift in the draft was corrected before
publication; strict comparison was retained. No published pass was retried.

## Verified results

- The first published pass and a fresh-process replay each passed **54 tests**,
  with zero failures, errors or skips. The replay uses the same implementation,
  not an independently written validation algorithm.
- All **23 bound files** were checked before and after each pass and retained
  their hashes. Only named metadata/code/evidence files were read; encoded
  records, token/index/label files, candidate rows and model weights were not.
- Published passes recorded zero network/GPU/inference, optimizer steps,
  child processes and prohibited-action attempts. The audit hook protects this
  fixed code; counters are not an OS packet-meter or general-purpose sandbox.
- New code and tests pass Ruff. The actual CLI execution request was also
  checked to reject without requiring a plan or producing output artifacts.

Evidence:

- [Development plan](../configs/lumi-4b-sft-launcher-development-v0.1.json)
- [Execution report](../evidence/lumi-4b-sft-launcher-development-v0.1.json)
- [Fresh-process verification](../evidence/lumi-4b-sft-launcher-development-verification-v0.1.json)

Plan file SHA-256:
`c98e5283c8edaf16a6a9855e72b5a81d010e635f6d62cc554e9760e1cd83e41b`

Report file SHA-256:
`8ba6a43b0ae51f8178fb631fbf084686f8421fe7c2aa46d83362c4ee1d2ddb61`

Verification file SHA-256:
`e003120be7801adf8aefe7985411301f46a75f4549b378e1397c46dd9ff0b57b`

Published artifacts and started markers are immutable. A successor is required
for changed inputs/code; do not rerun publication into those paths. For a
read-only local recheck (no output files), from the repository root:

```sh
PYTHONPATH=src artifacts/lumi-sft-import-runtime-v0.1/venv/bin/python -S -B \
  scripts/check_sft_launch_contract_v1.py \
  --expected-plan-sha256 c98e5283c8edaf16a6a9855e72b5a81d010e635f6d62cc554e9760e1cd83e41b
```

## Roadmap position and next step

The completed base/data/CPU foundations from v0.5 still stand. We remain in
**pre-GPU preparation**, not actual 4B SFT. The same six high-level prelaunch
conditions remain partially open; this work advances the local contract portion
of condition 2, not the entire integrated-launcher condition. The nine detailed
blocker codes in the report are not nine newly added roadmap stages.

1. Actual S3-backed final-base restore/import and real verification receipt.
2. SFT-specific worker integration and resource-enforcing supervisor; local
   contract/compiler and synthetic tests are now complete.
3. Exact target host/GPU/runtime and total resource/cost limits.
4. Content-verified data relocation with a new target-host metadata binding.
5. Frozen CUDA masking/precision/memory/import/resume/fault acceptance criteria.
6. Exact bounded execution approval after inputs and limits are fixed.

The next dependency is to select/bind the target GPU environment and budget,
then prepare the bounded restore/import and target-specific worker/supervisor
successor. Do not reuse a historical instance identity, infer a current S3 path
is available, or perform a download/remote/GPU action from this local approval.

After all six conditions close: GPU smoke/resume, equal-token learning-rate
screening, bounded full SFT/checkpoint verification, frozen capability/safety
evaluation, then authenticated private serving/product readiness. Preference
optimization remains conditional. Public/production use additionally needs
rights/owner/policy clearance, approved production contract/seal/shards and bound
CPU gate, and release readiness/approval. `production_eligible=false` remains.
