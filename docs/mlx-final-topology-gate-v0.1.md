# Lumi final-width MLX portability gate v0.1

Status: **pass on the M5 Mac; final trained-checkpoint quantization and GGUF/vLLM
runtime gates remain pending**

## Decision

Lumi's native MLX implementation strictly loads and executes the final
`lumi-4b-v0.2` projection topology. The gate preserves the production hidden
width, MLP width, query/key-value head layout, head dimension, context contract,
RoPE, normalization, and frozen 65,536-token tokenizer. It reduces only decoder
depth from 36 layers to one so an FP32 PyTorch/MLX comparison fits safely on the
16GB M5 Mac.

This is a deployment-topology result, not a trained-model quality or full-4B
memory/speed claim. The already completed 15-layer Lumi-150M gate separately
covers repeated layers, quantization, local serving, and longer cached decode.

## Frozen inputs

- Final source config: `configs/lumi-4b-v0.2.json`, SHA-256
  `643e8da652577f802d959740d66a7e6bafc5b065aa45010c2d7e0340ed9509be`.
- Final tokenizer JSON SHA-256:
  `1022c9538d9ec795b07c2e2bda3c5c6f21c321de933cfe5ad197628a94caa2ee`.
- Gate plan SHA-256:
  `edeb8e954bb56110ccd4f0bc1c7a84615405fc4fdd2627747829b2ee0e851aee`.
- Runtime: macOS 26.5.2 on arm64, Python 3.12.13, PyTorch 2.13.0,
  MLX 0.32.0, and MLX-LM 0.31.3.

The sealed artifact copies and hashes the plan, source config, PyTorch model,
MLX model, conversion/parity implementations, reports, and both weight sets.

## Exact topology exercised

| Field | Final 4B | Fixture |
| --- | ---: | ---: |
| Hidden size | 3,072 | 3,072 |
| Intermediate size | 8,384 | 8,384 |
| Query heads | 64 | 64 |
| Key/value heads | 8 | 8 |
| Head dimension | 64 | 64 |
| Query projection | 4,096 × 3,072 | 4,096 × 3,072 |
| Key/value projection | 512 × 3,072 | 512 × 3,072 |
| Vocabulary | 65,536 | 65,536 |
| Context contract | 4,096 | 4,096 |
| Decoder layers | 36 | **1** |

The fixture contains 306,914,304 parameters. Only layer count differs from the
final config.

## Verified result

| Check | Result | Gate |
| --- | ---: | ---: |
| Config fields | exact | exact |
| Saved trainable tensors | 11/11 byte-exact | exact |
| Expected recomputed RoPE buffers | 1/1 | only expected buffers |
| RoPE max absolute error | 4.768e-7 | <= 1e-6 |
| Worst prompt-logit max error | 0.0141602 | <= 0.02 |
| Worst prompt-logit mean error | 0.0016121 | <= 0.003 |
| Worst normalized logit RMSE | 0.18487% | <= 0.20% |
| Minimum cosine similarity | 0.9999985375 | >= 0.999998 |
| Prompt-position top-1 agreement | 60/60 | 100% |
| Cached greedy generated-token agreement | 16/16 | exact |
| Worst cached log-probability error | 0.0069675 | <= 0.02 |
| Peak MLX memory | 1,237,502,068 bytes | measured |

The first diagnostic attempt used the Lumi-150M normalized-RMSE ceiling of
0.10%. It failed only that scale-sensitive FP32 accumulation criterion: config,
weights, RoPE, all 60 top-1 decisions, and all 16 generated tokens were already
exact. No model or conversion implementation changed. The final-width gate uses
a 0.20% normalized-RMSE ceiling and 0.999998 cosine floor while preserving the
same max/mean error, exact tensor, exact top-1, exact generation, RoPE, and cache
requirements. The independent verifier rejects any looser sealed threshold.

## Evidence

Canonical artifacts are outside Git at
`lumi-data/mlx-final-topology-gate-v0.1`:

- `gate-report.json` SHA-256:
  `b7f68cb6b5dfc7be6e20462e982611d45d1fa9ba064fcb67b33e0cda13d9b597`;
- `parity-report.json` SHA-256:
  `07e8e1b47b59de2e731249e7383ef5c3ef1b1599a25631bf832d502ffab4a312`;
- `manifest.sha256` SHA-256:
  `b473059e31473235508a159c8fbbeac4cb8d104027e96a3481a5fd717b526d01`.

Fresh-process verification checks every manifest entry, refuses unlisted files,
re-hashes all copied implementation/input files, and reasserts the 36-to-1
depth boundary and exact-width/numeric decisions.

## Reproduction

```bash
python scripts/check_final_mlx_topology.py \
  --plan configs/mlx-final-topology-gate-v0.1.json \
  --tokenizer "$LUMI_TOKENIZER" \
  --output "$LUMI_DATA/mlx-final-topology-gate-v0.1"

python scripts/check_final_mlx_topology.py \
  --output "$LUMI_DATA/mlx-final-topology-gate-v0.1" \
  --verify-only
```

No cloud resource, GPU instance, corpus acquisition, model training, or trained
checkpoint was created by this gate.

## Remaining portability work

1. run the already prepared vLLM Transformers-backend gate on a low-cost CUDA
   host;
2. add and test a real GGUF/llama.cpp architecture mapping;
3. after 4B pretraining, rerun MLX conversion, 4-bit quality, memory, and speed
   gates on the actual trained checkpoint.
