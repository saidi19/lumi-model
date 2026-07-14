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

The design uses public, established Transformer components. It is OpenAI-style
in the broad GPT decoder sense and in its many-small-query-head layout, but it
does not claim to reproduce any undisclosed GPT or Claude architecture.

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

## Mandatory gates

### Architecture correctness

- Exact parameter count agrees with an instantiated tiny equivalent.
- Causal masking prevents future-token leakage.
- Labels are shifted exactly once.
- PAD, EOS, BOS, and document-boundary token IDs are distinct.
- Save/load round trip preserves logits.
- Tied embeddings are actually shared.

### Portability

- Hugging Face `save_pretrained`/`from_pretrained` works.
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

## Explicit non-goals for v0.1

- Mixture-of-experts.
- Multimodal inputs.
- 128K context.
- Novel CUDA kernels.
- Training data stored in Git.
- Claiming ChatGPT/Claude-level quality from architecture alone.
