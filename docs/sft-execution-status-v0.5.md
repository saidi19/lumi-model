# SFT execution status v0.5

Verified locally on 2026-09-05. The isolated ML runtime is restored and the v2
synthetic import adapter passes genuine CPU tests. **No actual final-4B import,
GPU SFT, model download or training occurred.** Production eligibility remains
false. This supersedes the current-runtime blocker in status v0.4 without
changing any historical code, plan, report or verification.

## What finished

An isolated Python 3.13 environment now lives at
`artifacts/lumi-sft-import-runtime-v0.1/venv`. The approved installation fetched
Python wheels from public PyPI, without source builds or changes to the global
Python environment. Torch 2.10.0, Transformers 5.2.0, tokenizers 0.22.2,
safetensors 0.7.0, NumPy 2.2.6, pytest 9.0.2 and Ruff 0.15.1 are direct pins.
All 39 resolved package versions and exact wheel SHA-256 values are preserved
in [the platform-specific lock](../configs/lumi-sft-import-runtime-macos-arm64-v0.1.lock)
and the private installation report. `pip check` passed. Installed environment
disk use was approximately 721 MiB; no trained-model weights were downloaded.

The initial sandbox DNS failure was diagnosed before an approved package-only
network escalation completed installation. Network was used for package
acquisition, so this entire turn is **not** described as zero-network. The
subsequent published test passes ran offline with network/child-process and
real-data/checkpoint access guards.

Package references: [Torch 2.10.0](https://pypi.org/project/torch/2.10.0/) and
[Transformers 5.2.0](https://pypi.org/project/transformers/5.2.0/). The local lock
and install report, not a moving latest-version label, identify this runtime.

## Actual bug found and fixed

Restoring the runtime made the original 14 genuine tests runnable: 12 passed,
and two BF16 reload tests failed. The mismatch was one persistent rotary
buffer: the v1 exporter cast it to BF16 along with parameters, whereas the HF
loader reloaded it in the architecture's explicit FP32 dtype. Its values were
equal after FP32 conversion, but the exact dtype contract was not satisfied.

New v2 adapter/backend/test files preserve the v1 originals. The v2 contract
casts **parameters** to the requested precision while retaining buffers in
their configured dtype and preserving source values. The receipt binds that
policy. Exact key, shape, dtype, tensor-value, tied-weight and checksum checks
remain enforced; no tolerance was weakened and no buffer was recomputed from
scratch to hide a source mismatch.

## Verification now completed

- 31 contract/security tests passed, including explicitly mocked packaging and
  negative path/hash/permissions/mutation/no-retry cases.
- 17 genuine tiny CPU tests passed, including FP32 and BF16 parameter outputs,
  all-BF16 source DCP, mixed BF16-parameter/FP32-buffer source DCP, and direct
  safetensors verification of FP32 buffer-value preservation.
- Both full synthetic adapter packages passed the actual frozen worker's
  checksum verifier. This checks the source-package contract; it is **not** a
  run of the full distributed worker, an inference test or a real-4B restore.
- A fresh process repeated all 48 tests: zero failures, errors or skips.
  The stable aggregate results matched. The verification uses the same test
  implementation, not an independently written tensor-comparison algorithm.
- Ruff passed for the new v2 adapter, backend, both test files and evidence
  runner. Earlier v1 files and evidence retain their original hashes.

The published passes used explicit CPU allocation and guarded against model
forward calls, AdamW steps, network/child launches and real dataset/checkpoint
access. They recorded zero prohibited-action attempts, network bytes, GPU use,
inference calls and optimizer steps. Synthetic temporary fixtures were cleaned
up by the test framework. No actual 4B import receipt was produced.
The network counters describe the fixed offline test code and its guarded
operations, not an independent OS packet-meter measurement. The Python audit
hook is defense-in-depth, not a general security sandbox for arbitrary code.

Evidence:

- [Runtime recovery scope](../configs/lumi-4b-sft-import-runtime-recovery-v0.1.json)
- [Frozen test bindings and original failure diagnosis](../configs/lumi-4b-sft-import-runtime-test-bindings-v0.1.json)
- [48-test execution report](../evidence/lumi-4b-sft-import-runtime-test-v0.1.json)
- [48-test fresh-process verification](../evidence/lumi-4b-sft-import-runtime-test-verification-v0.1.json)

Test-plan SHA-256:
`3b506a4cb44551cec07a655e456c78c7d9cfe5c8e817c396e2df18afa596f461`

Report file SHA-256:
`1f3228fee1932c93de5779995d466580e21e064b2d01babb605603c7537995fd`

Verification file SHA-256:
`2b7c07961e77cc066d8c4232f6aed8a9477fa3b179d2a42c61f9fc0ba86d36df`

The reports are immutable; do not rerun the evidence publisher over these
paths. A changed runtime, code or input requires a versioned successor.

## Where the full roadmap stands

Completed foundations: final base pretraining and its metadata binding; private
research data contract, source registry, 329,200-record seal and encoded shards;
single-device and distributed DDP/FSDP CPU smoke/resume; local metadata
readiness; and now the tiny CPU import implementation and genuine tests.

We are still **before real-4B GPU SFT**. Six pre-launch conditions remain open
at the following level of completion, not six new equal-sized tasks:

| Condition | Current state / remaining work |
| --- | --- |
| 1. Final-base import | Synthetic adapter/tests complete. Exact real-4B source path, bounded export/import operation and verified real receipt remain pending. Current executable intentionally rejects the real model identity. |
| 2. SFT authorization and launcher | Implement/review an SFT-specific, hash-bound, resource-bounded launcher; generic pretraining approval is not sufficient. This is the next local implementation task. |
| 3. Host/runtime/resources | Bind exact target host and GPU identities, target runtime, memory headroom, runtime/cost/I/O/storage caps. The new macOS CPU environment does not qualify a CUDA host. |
| 4. Data relocation | Verify transferred content and private modes, then create the target host's new local metadata binding. Never drop the inode/mtime-based fifth loader binding. |
| 5. CUDA acceptance | Freeze masking/precision/memory/import/resume/fault-test criteria before GPU execution. CPU results do not prove CUDA parity. |
| 6. Execution approval | Obtain the exact bounded GPU/remote-operation approval after its inputs and limits are fixed. |

Local launcher preparation does not skip condition 1: **all** conditions,
including the real import receipt, must close before a GPU launch is permitted.

Then five required macro stages remain for the private-research route:

1. GPU smoke and fresh-process resume verification.
2. Equal-token learning-rate screening and held-out profile selection.
3. Bounded full SFT and checkpoint verification/seal.
4. Frozen instruction/language/reasoning/coding/safety/tool-honesty and
   base-capability-retention evaluations.
5. Serving benchmarks, authenticated private deployment and product readiness.

Preference optimization is conditional on its separate comparison gate, not
automatically another mandatory training run. Production/public release also
requires source/owner/policy clearance, an approved production contract and
production seal/shards with their bound CPU gate, plus release readiness and
approval. Private research success does not waive those obligations.

These are different-sized work packages with unknown GPU availability and
evaluation outcomes. Neither a percentage-complete estimate nor a reliable
completion date follows from simply adding the step counts.
