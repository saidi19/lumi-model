# SFT execution status v0.4

The checkpoint-import implementation is **partial, not a completed import gate**.
The preceding local metadata-readiness evidence remains unchanged. Its six
pre-launch conditions have not been reduced to five: condition 1 still needs
genuine tiny tensor tests and, later, a separately bound real-4B import receipt.

## Work completed in this local step

- Added a versioned pure-metadata lineage validator and synthetic-only package
  adapter. It binds the DCP manifest and declared state, exact step/cursor/token
  totals, run fingerprint, source commit, world size, configuration bytes and
  tokenizer manifest/files. It does not substitute shape compatibility for
  exact configuration provenance.
- Added closed-inventory, duplicate/path/symlink/hardlink, private-mode,
  tokenizer-byte and input-preservation checks; exclusive started/completed
  markers; immutable receipt and worker-format checksum-manifest publication.
  Failures leave the attempt in place and cannot silently retry it.
- Added a CPU-only backend constrained to at most one million synthetic model
  parameters. Its implemented checks include complete DCP key/shape/dtype/chunk
  coverage, finite values, tied aliases, explicit cast and exact local HF tensor
  reload. It does not perform a forward pass or optimizer step. These tensor
  checks are implemented but **not yet executed in the present environment**.
- Added 31 stdlib contract/security tests, all passing with zero skips. They
  include explicitly mocked packaging/consumer wiring, source-mutation rejection,
  receipt/manifest tampering and failure-without-completion checks. Mocked tests
  are not evidence of tensor correctness or actual worker compatibility.
- Prepared 14 genuine tiny CPU tests, including FP32/BF16 round trips, missing
  and extra DCP keys, wrong shapes/dtypes, contradictory tied aliases, nonfinite
  tensors, allocation caps and full adapter-to-frozen-worker integration.

The implementation creates a new independently inventoried package; it does
not append files to an old legacy HF export and then pretend the old export
inventory still validates. A receipt hashes the payload, and the final worker
manifest hashes the payload plus receipt, avoiding a circular self-hash.

The executable import entry point rejects the real `lumi-4b-base` identity.
This development step is not an actual 4B export/restore executor. That future
operation must separately bind source location, runtime, limits, resource
accounting and approval. Current synthetic size limits cannot be silently
expanded to perform it.

## Actual blocker and remaining order

The checked Python 3.13 environment cannot import `torch`; metadata probes also
find `transformers`, `safetensors` and `numpy` unavailable. Scoped checks of the
system, Homebrew and bundled runtimes found no usable ML stack. The genuine
test suite was invoked and failed in dependency setup **before any of its 14
tests ran**. No import success or skip-success was recorded. Existing historical
CPU evidence remains a historical completed result, not proof that this current
runtime is usable.

Next actions, without skipping the incomplete condition:

1. Restore a pinned, isolated local ML runtime using an existing suitable
   environment, or a separately bounded package-download/install operation.
   No package installation or network access was performed in this step.
2. Run all 14 genuine tiny CPU tests, fix any compatibility/import failures, and
   publish a versioned result only after actual tensor and consumer checks pass.
   Re-run the 31 contract tests in that same runtime. Keep real checkpoints,
   training datasets, inference, GPU and remote systems out of scope.
3. Complete review and bind a successor adapter. Only then proceed to the
   SFT-specific authorization/launcher work described in condition 2 of
   [status v0.3](sft-execution-status-v0.3.md). Actual final-4B export still needs
   its own exact source/runtime/resource/approval contract and real receipt.
4. Continue the remaining host/runtime/resource, shard-relocation, CUDA
   acceptance and exact GPU-approval conditions. GPU smoke, learning-rate
   screening, full SFT, evaluations and deployment remain downstream.

Production eligibility is still false. No production-policy, owner-review,
production-data or release requirement has been waived.

## Evidence and reproducibility

- [Development plan](../configs/lumi-4b-sft-checkpoint-import-development-v0.1.json)
- [Partial-state report](../evidence/lumi-4b-sft-checkpoint-import-development-v0.1.json)
- [Fresh-process structural verification](../evidence/lumi-4b-sft-checkpoint-import-development-verification-v0.1.json)

The two published passes rechecked code/input bindings and replayed all 31
stdlib tests. Verification uses the same implementation in a fresh process;
it does not claim an independent tensor audit. Test fixtures were synthetic
temporary files. No real checkpoint, dataset or tokenizer content was opened;
only existing metadata/code plus synthetic fixtures were read. There was zero
network, GPU, inference, training, remote action or runtime installation.

Published hashes:

- Plan: `dcfed6b8d54513377ca30cb8c4324db6fb50e95352c88b923e1a4844b3ee8afe`
- Report: `5e8207abe304913ae804455bf498e807a5e9894dccbeae3f139869464a635c14`
- Verification: `3e9e78d89f32918d5698a2adfda1f6a22fb99b8e8baa462b1771ae288f3ead7a`

The audit command is `scripts/check_sft_checkpoint_import_v1.py` with the exact
`--expected-plan-sha256` and optionally `--verify`. Both evidence files now
exist; the tool refuses to overwrite them. A runtime change or code fix needs
a versioned successor, not edited historical evidence.
