# Lumi Initialization and Optimizer A/B Gate v0.1

Status: **local M5 gate passed; residual-scaled initialization is selected for
fresh 4B candidates, while legacy AdamW remains selected**

## Decision

The controlled 2x2 experiment selected:

- initialization: `residual_scaled_v0.1`;
- optimizer: existing legacy AdamW, with beta values left at PyTorch's
  `(0.9, 0.999)` and weight decay applied as before;
- fresh 4B candidate config: `configs/lumi-4b-v0.2.json`.

The candidate grouped AdamW profile, which applies weight decay only to
parameters with at least two dimensions and changes beta2 to `0.95`, did not
pass. It remains implemented as an explicit experimental option but is not the
selected production setting.

This gate does not modify the completed 150M or 1B checkpoints. Their standard
initialization and optimizer fingerprints remain historical facts and stay
resumable. It also does not authorize a paid GPU run or 4B/80B training.

## What changed

`residual_scaled_v0.1` keeps the normal standard deviation at `0.02` for
embeddings and non-residual projections. It initializes only each attention
`o_proj` and MLP `down_proj` with:

```text
0.02 / sqrt(2 * number_of_layers)
```

The eight-layer proxy therefore uses `0.005` for those residual-output
projections. The 36-layer 4B candidate uses approximately `0.002357`. This is
an explicit config field, so checkpoints record the initialization policy and
an invalid value fails closed.

The optional grouped optimizer profile:

- places matrix and embedding parameters (`ndim >= 2`) in the decay group;
- places RMSNorm and other one-dimensional parameters in the zero-decay group;
- deduplicates tied input/output embeddings;
- uses AdamW beta values `(0.9, 0.95)`;
- becomes part of the resume fingerprint only when enabled.

With the option disabled, the original optimizer construction and v1 identity
remain byte-compatible with archived checkpoints.

## Pinned experiment

The source implementation commit is
`2f02e50c5fa766fe6ecc306581739af44d18242b`. The local gate used:

| Field | Value |
|---|---:|
| Proxy parameters | 9,865,344 |
| Layers / width / context | 8 / 128 / 128 |
| Vocabulary | frozen 65,536-token Lumi tokenizer |
| Variants | 4 |
| Seeds per variant | 3 (`7`, `17`, `29`) |
| Optimizer steps per run | 128 |
| Tokens per run | 32,768 |
| Total controlled training tokens | 393,216 |
| Device | Apple M5 MPS, PyTorch 2.13.0 |

Every variant used the exact same real 8,192-token packed training artifact,
data-order algorithm, seeds, cosine schedule, learning rate, clipping,
document-isolated attention, and token budget. Reusing that small artifact for
four deterministic epochs is intentional: this is a learning-dynamics A/B,
not a model-quality benchmark or a corpus-coverage claim.

Held-out evaluation selected the middle sequence from each of the frozen
model-weight-held-out set's 24 strata, cropped each to 128 tokens, and scored
3,048 next-token predictions per evaluation. The selected token-ID stream is
pinned at SHA-256
`623247e9cb853a5272d444b4a614f8dbfbd7b60d09e19b7de7a32065567e420d`.
Initial evaluation is repeated from the same seed before each run, and the
standard/grouped pairs reproduce identical initial losses exactly.

## Verified result

| Variant | Median final held-out loss | Median final training loss | Median tokens/s | Versus baseline held-out |
|---|---:|---:|---:|---:|
| Standard init + legacy AdamW | 9.079972 | 7.387327 | 6,035.2 | baseline |
| Residual init + legacy AdamW | **8.981004** | **7.239976** | 6,012.6 | **1.08997% better** |
| Standard init + grouped AdamW | 9.190335 | 7.637200 | 5,782.3 | 1.21545% worse |
| Residual init + grouped AdamW | 9.102748 | 7.502793 | 5,989.0 | 0.25083% worse |

Promotion required at least 0.2% median held-out improvement, no more than one
seed worse than baseline by over 0.1%, and no more than 10% median-throughput
regression. Residual initialization with legacy AdamW improved every paired
seed, improved the median by 1.08997%, and reduced throughput by only 0.37374%,
so it passed. Both grouped-optimizer variants failed the quality conditions.

The checkpoint-resume canary stopped at step 2, restored the grouped optimizer
and RNG state in a fresh load, and reached step 4 and 1,024 tokens. All 12 main
runs completed without NaN, infinite loss, or OOM.

## Evidence and independent verification

Generated evidence is outside Git at
`$LUMI_DATA/init-optimizer-ab-v0.1/` and occupies about 1.4GB. It contains the
four materialized configs, resume canary, 12 final proxy checkpoints, optimizer
states, metric streams, run results, and decision report.

- `decision.json` SHA-256:
  `fca3a7395438dcdc5ae8493db02f54cf4fded00540fcdb9930cd0447a734143f`
- `manifest.sha256`: 136 entries;
- manifest-file SHA-256:
  `a2341ad0cb27d01be9b2b0d3c3b2634afa82baf84fdcc1d53c4bf895267701dc`.

Independent checks verified every manifest entry, all 12 complete states, 128
contiguous metric rows per run, exact 32,768-token budgets, finite losses and
gradient norms, held-out geometry, paired initial-loss equality, the resume
canary, and a separate recomputation of the conservative winner.

## Reproduction

The runner is idempotent and resumes matching partial training outputs:

```bash
caffeinate -dimsu scripts/run_init_optimizer_gate_local.sh
```

The test suite separately covers residual-projection targeting, unchanged
standard initialization, unique optimizer grouping for tied embeddings,
resume-identity pinning, and conservative decision behavior.

## Next gate

The frozen pre-4B choices are now the original 65K tokenizer, document-isolated
packing, residual-scaled initialization, and legacy AdamW. The next engineering
gate is the production multi-GPU trainer and a short distributed calibration
plan. Full benchmark scoring can be integrated when a meaningful 4B checkpoint
exists. Corpus acquisition, the 4B pilot, and every paid launch remain separate
future approvals.
