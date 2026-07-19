# Lumi architecture decision record v0.1

Status: **candidate; not approved for full-scale training**

## Product target

- Dense decoder-only autoregressive language model.
- Approximately 4B trainable parameters.
- English-first general model, with multilingual-ready tokenization.
- Initial 4,096-token context.
- BF16 pretraining on H100 GPUs.
- Post-training produces a separate chat model, `Lumi-4B-Instruct`.
- Quantized MLX and GGUF releases must run on a 16GB M5 Mac.

## Candidate 4B configuration

Fresh post-gate experiments use `configs/lumi-4b-v0.2.json`. The original
`configs/lumi-4b.json` remains unchanged because it is hash-bound as the
historical reference for the completed 1B scaling evidence.

| Field | Value |
|---|---:|
| Layers | 36 |
| Residual width | 3,072 |
| Query heads | 64 |
| KV heads | 8 |
| Head dimension | 64 |
| Query projection width | 4,096 |
| SwiGLU intermediate width | 8,384 |
| Vocabulary | 65,536 |
| Context | 4,096 |
| Position encoding | RoPE |
| Normalization | Pre-RMSNorm |
| Embeddings | Input/output tied |
| Biases/dropout | Disabled |
| Initialization | Residual outputs scaled by `1 / sqrt(2L)` |
| Optimizer | Legacy AdamW; grouped beta2=0.95 candidate rejected |

The design uses public, established Transformer components. It is OpenAI-style
in the broad GPT decoder sense and in its many-small-query-head layout, but it
does not claim to reproduce any undisclosed GPT or Claude architecture.

## Controlled 1B scaling proxy

`configs/lumi-1b.json` freezes a 1,050,981,888-parameter proxy with the same 36
layers, 4,096-token context, 65,536-token vocabulary, and 64-wide heads as the
4B candidate. Its residual width, intermediate width, query heads, and KV heads
are all exactly half of 4B. This isolates width scaling and exercises the final
context instead of comparing against an unrelated 1B architecture.

The proxy's staged plan ends at only 1.4B processed tokens, or 1.332 tokens per
parameter. It is a scaling and optimization gate, not compute-optimal training.
See [`scaling-pilot-1b-v0.1.md`](scaling-pilot-1b-v0.1.md). Paid training remains
blocked on materializing and sealing the permanent-held-out-excluding data.

## Why dense for v1

A total-parameter MoE introduces routing, expert-parallel communication,
optimizer-state storage, conversion, and inference-backend risk. The project
budget is better spent validating one dense model and a high-quality corpus.

## Training target

- Candidate final corpus: 80B to 100B curated tokens.
- Candidate-source pool before filtering/deduplication: 130B to 150B tokens.
- Maximum cloud-credit envelope: USD 6,000.
- No full run is approved until a measured throughput pilot replaces estimated
  tokens/second.

The 80B-token point is a compute-balance reference, not a promise of perfect
quality. Dataset quality, coverage, optimization, and post-training remain
critical.

## Initialization and optimizer gate

A local 2x2 factorial experiment compared standard versus residual-scaled
initialization and legacy AdamW versus grouped-decay AdamW with beta2=0.95. Four
variants, three paired seeds, identical real data, and identical 32,768-token
budgets selected residual-scaled initialization with legacy AdamW. Median
held-out loss improved by 1.08997% across all paired seeds with a 0.37374%
throughput reduction. Both grouped-optimizer variants were worse than the
baseline and were rejected.

Only attention `o_proj` and MLP `down_proj` weights receive the scaled standard
deviation. Historical checkpoints and configs retain their original standard
initialization. See
[`init-optimizer-ab-v0.1.md`](init-optimizer-ab-v0.1.md) for the pinned design,
thresholds, metrics, and independent evidence checks.

## Mandatory gates

### Architecture correctness

- Exact parameter count agrees with an instantiated tiny equivalent.
- Causal masking prevents future-token leakage.
- Labels are shifted exactly once.
- PAD, EOS, BOS, and document-boundary token IDs are distinct.
- Save/load round trip preserves logits.
- Tied embeddings are actually shared.

### Portability

- Hugging Face self-contained export loads in an isolated process without the
  local `lumi` package (passed for a tiny checkpoint).
- vLLM serves the model through a Transformers-compatible backend.
- MLX conversion and generation work on M5/16GB.
- GGUF conversion and `llama.cpp` generation work.

If the 64x64 attention layout blocks MLX or GGUF support, compare it against a
portable 24x128 layout at small scale. Portability and measured validation loss,
not branding, decide the frozen architecture.

### Scaling

- Synthetic smoke model trains without NaNs.
- A 50M-150M pilot learns from a real clean corpus.
- A larger pilot establishes loss trend, memory use, MFU/throughput, checkpoint
reliability, and cost per billion tokens.

See [`portability-v0.1.md`](portability-v0.1.md) for per-backend status and
[`tokenizer-v0.1.md`](tokenizer-v0.1.md) for the tokenizer freeze gates.

## Explicit non-goals for v0.1

- Mixture-of-experts.
- Multimodal inputs.
- 128K context.
- Novel CUDA kernels.
- Training data stored in Git.
- Claiming ChatGPT/Claude-level quality from architecture alone.
