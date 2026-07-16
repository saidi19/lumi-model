# Lumi MLX Local-Inference Gate v0.1

## Decision

The Lumi-150M checkpoint passes the native Apple-Silicon deployment gate. The
FP32 MLX artifact preserves the PyTorch architecture and behavior within a
measured cross-runtime FP32 tolerance, and the 4-bit affine artifact passes the
stratified quality gate while materially reducing storage, memory, and decode
latency.

Use the 4-bit artifact for local plumbing, API, and UI development. This is not
a claim that Lumi-150M has ChatGPT-like language quality: it is still a base
model with 150,625,536 parameters trained on only 50,003,968 tokens, without
instruction tuning or preference/safety post-training.

## Frozen inputs and runtime

- Source checkpoint:
  `lambda-pilot-v0.1/full/checkpoint-00000763/model.safetensors`, SHA-256
  `656edd0d2eb74eaeb1cd3e26f7c0265ce3d9a88f47e310e51041ba39e16f1431`.
- Tokenizer: frozen 65,536-token Lumi byte-level BPE.
- Held-out build fingerprint:
  `77455ce7eb2606d8f258ef13f715bbdfeb8be4f5d7910cc26348552f52c96298`.
- Runtime: Apple M5 with 16GB unified memory, macOS 26.5.2, Python 3.12.13,
  MLX 0.32.0, MLX-LM 0.31.3, and PyTorch 2.13.0.
- Canonical FP32 parity report SHA-256:
  `92d7032b759d15dd4387ecf8d878ca6d6d55b205864a6799da8acb17662dc293`.
- Canonical 4-bit gate report SHA-256:
  `9802eb46fef74ef6199a86b4743d197997b0e689a9db02e5748dd66f318e10bd`.

Generated checkpoints and reports remain outside Git under
`lumi-data/local-inference-gate-v0.1`. Both converted model directories include
their own `SHA256SUMS`; every listed file passes a fresh checksum verification.

## Explicit Lumi MLX architecture

Converted artifacts set `model_type` to `lumi` and `model_file` to
`lumi_mlx.py`. They do not pretend to be Llama or Mistral checkpoints. The
copied model file implements Lumi's exact 15-layer pilot shape: independent
64-wide head dimensions, 16 query heads, four key/value heads, GQA, non-
traditional rotate-half RoPE with theta 1,000,000, RMSNorm, SwiGLU, and tied
input/output embeddings.

Conversion is atomic and refuses to overwrite an existing directory. It
verifies the source checkpoint's checksum file, tokenizer vocabulary and
special IDs, performs strict tensor-name/shape loading, saves the artifact,
then loads that finished directory in a fresh MLX model instance and requires
finite probe logits before publishing it.

## FP32 parity result

The parity gate used the four fixed English, code, and Bengali prompts from
`configs/evaluation-pilot-v0.1.json` and 32 greedy tokens per prompt.

| Check | Result | Gate |
| --- | ---: | ---: |
| Trainable tensors | 137/137 byte-exact | exact |
| Trainable-weight max error | 0 | 0 |
| Recomputed RoPE buffers omitted | 15/15 expected | only expected buffers |
| RoPE probe max error | 4.768e-7 | <= 1e-6 |
| Worst prompt-logit max error | 0.011247 | <= 0.02 |
| Worst normalized logit RMSE | 0.06969% | <= 0.10% |
| Worst logit cosine similarity | 0.9999999441 | >= 0.9999999 |
| Prompt-position top-1 agreement | 38/38 | 100% |
| Cached greedy generated-token agreement | 128/128 | exact |
| Peak MLX memory during parity | 627,145,012 bytes | measured |

The absolute logit tolerance is intentionally paired with byte-exact weights,
an explicit RoPE probe, cosine similarity, exact top-1 decisions, and exact
cached generation. This distinguishes normal fused-kernel FP32 accumulation
differences from architecture or conversion drift.

## 4-bit quality result

The canonical gate selects two center-of-bin sequences from each of all 24
held-out source strata: 48 sequences, 98,304 input tokens, and 98,256 predicted
tokens. This is a deterministic stratified quantization probe, not a
replacement for the already completed full 4,999,168-token checkpoint
evaluation.

| Metric | FP32 | 4-bit affine | Change |
| --- | ---: | ---: | ---: |
| Stratified loss | 4.816525 | 4.823305 | +0.006780 |
| Stratified perplexity | 123.5350 | 124.3754 | +0.6803% |
| Same-context top-1 | reference | 91,517/98,256 | 93.1414% agreement |
| Artifact bytes | 607,189,860 | 98,946,546 | -83.7042% |
| Held-out peak memory | 2,370,409,584 | 1,991,046,268 | -16.00% |

Free-running fixed-prompt generations agree at 42.19%, but that value is
diagnostic rather than a gate: after the first changed token, subsequent models
receive different contexts. Same-context held-out loss and top-1 agreement are
the valid quantization comparisons. The 4-bit gate must be rerun after future
pretraining or instruction tuning because the decision margins will change.

## M5 local speed and memory

Timing uses an exact-shape warm-up followed by the median of three trials. The
prefill is 512 tokens and decode is 64 cached single-token steps.

| Metric | FP32 | 4-bit affine | 4-bit / FP32 |
| --- | ---: | ---: | ---: |
| Prefill tokens/second | 14,254.7 | 25,440.5 | 1.78x |
| Decode tokens/second | 116.08 | 522.26 | 4.50x |
| Benchmark peak memory | 1,123,646,592 bytes | 631,356,544 bytes | 0.562x |

The benchmark proves comfortable local execution on the 16GB M5 Mac. It does
not predict 1B or 4B throughput; those sizes need separate measured gates.

## Reproduction

Install the pinned Apple runtime on Apple Silicon:

```bash
python -m pip install -e '.[dev,apple]'
```

Convert and prove FP32 parity:

```bash
python scripts/convert_mlx_checkpoint.py \
  --checkpoint "$CHECKPOINT" \
  --tokenizer "$TOKENIZER" \
  --output "$LUMI_DATA/local-inference-gate-v0.1/lumi-150m-fp32-mlx" \
  --dtype float32

python scripts/check_mlx_parity.py \
  --plan configs/evaluation-pilot-v0.1.json \
  --checkpoint "$CHECKPOINT" \
  --mlx-model "$LUMI_DATA/local-inference-gate-v0.1/lumi-150m-fp32-mlx" \
  --tokenizer "$TOKENIZER" \
  --output "$LUMI_DATA/local-inference-gate-v0.1/fp32-parity-report.json"
```

Create and evaluate 4-bit MLX:

```bash
python scripts/convert_mlx_checkpoint.py \
  --checkpoint "$CHECKPOINT" \
  --tokenizer "$TOKENIZER" \
  --output "$LUMI_DATA/local-inference-gate-v0.1/lumi-150m-4bit-mlx" \
  --dtype float32 \
  --quantize \
  --quantization-bits 4 \
  --quantization-group-size 64

python scripts/check_mlx_quantization.py \
  --plan configs/evaluation-pilot-v0.1.json \
  --fp32-model "$LUMI_DATA/local-inference-gate-v0.1/lumi-150m-fp32-mlx" \
  --quantized-model "$LUMI_DATA/local-inference-gate-v0.1/lumi-150m-4bit-mlx" \
  --manifest "$HELDOUT/manifest.json" \
  --tokenizer "$TOKENIZER" \
  --output "$LUMI_DATA/local-inference-gate-v0.1/quantization-gate-report.json"
```

The next gate is an OpenAI-compatible local API and a basic chat UI backed by
the sealed 4-bit artifact. That validates product plumbing; it does not change
the model's knowledge or base-model behavior.
