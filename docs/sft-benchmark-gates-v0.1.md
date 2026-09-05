# Lumi L1 staged SFT benchmark gates v0.1

Status: **schedule frozen for planning; no model inference, paid GPU, dataset
seal, SFT run, or release is authorized**

Observed: 2026-08-07

## When benchmarks run

1. Run the complete frozen suite on the exact sealed 80B-token base checkpoint
   before the first quality SFT run. This is the comparison baseline.
2. After a sealed 10M-token learning dataset exists, run the CPU canary and a
   separately authorized short GPU loss/mask/resume/HBM smoke. This is a
   runtime test, not a quality benchmark.
3. Run the complete frozen quality suite on the exact 10M SFT checkpoint.
4. Continue only if the 10M checkpoint beats the base without required safety,
   language, or base-capability regression. Repeat the complete suite at exact
   50M, 100M, and 250M accepted-supervised-token stages.
5. Run the 500M stage only if the 250M checkpoint still shows material marginal
   gain. Five hundred million tokens are a quality ceiling, not a quota.
6. Run the full release benchmark, safety red-team, blinded human comparison,
   and serving canary on the smallest stage selected by the frozen policy.

Every quality comparison uses the same revision-pinned prompt set. Benchmark
prompts must never enter training data. Reports retain hashes and aggregate
scores in Git; raw prompts and model outputs remain private for review.

## Current readiness

The exact base artifact is sealed, and its original frozen evaluation receipt
exists. The broader product benchmark is not complete: instruction following,
knowledge/reasoning, math, code, truthfulness/calibration, and
Bengali/multilingual adapters are still pending. Basic capability and language
routing gates are implemented but require an actual checkpoint; the punctuation
gate is defined and requires at least 288 post-SFT model cases.

The repaired candidate pool currently contains 1,059,171 accepted train
supervised tokens, leaving 8,940,829 tokens to the 10M learning floor. It is not
sealed or production eligible. Dataset growth and benchmark-adapter completion
can proceed in parallel, but a full quality SFT run cannot start until both the
frozen base baseline and exact dataset/runtime gates pass.

The machine-readable schedule is
`configs/lumi-4b-sft-benchmark-gates-v0.1.json`.
