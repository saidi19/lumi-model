# Lumi portability gate v0.1

Status: **Hugging Face and native MLX topology pass; vLLM runtime and GGUF
remain pending; full-scale training is not approved**

## Current result

| Target | Status | Evidence / remaining work |
|---|---|---|
| Hugging Face | PASS (tiny checkpoint) | Export includes model code and `auto_map`; isolated Python loads `AutoModel` and `AutoModelForCausalLM`, preserves logits, and generates |
| vLLM | CODE READY; runtime test pending | Base model now routes attention through the Transformers attention interface and forwards backend kwargs; test on Linux/CUDA with `model_impl="transformers"` is still required |
| MLX-LM | PASS (150M trained checkpoint + final-width fixture) | Native `lumi_mlx.py` preserves Lumi identity, independent `head_dim`, wide query projection, GQA, RoPE, strict weights, cached generation, and final 4B projection shapes on the M5 Mac |
| GGUF / llama.cpp | NOT IMPLEMENTED | A custom `LumiForCausalLM` converter/architecture mapping and Metal generation test are required |

Passing Hugging Face or MLX does not imply that GGUF or vLLM passed. The final
64-head × 64-dimension layout now passes PyTorch/MLX strict conversion and
runtime parity, while the vLLM CUDA runtime and llama.cpp architecture mapping
remain separate gates.

## Why the Hugging Face test is isolated

Importing the local `lumi` package registers its classes in the current Python
process, which can hide an incomplete checkpoint. The portability test exports
the code and weights, launches Python with isolated mode, imports only
Transformers, and loads with `trust_remote_code=True`. It also checks the base
`AutoModel` mapping required by vLLM.

Run it with:

```bash
python scripts/check_isolated_hf.py
```

## vLLM runtime gate

Run only after a trained tokenizer and tiny real-data checkpoint exist:

1. start a low-cost Linux/CUDA instance;
2. install the repository and a pinned vLLM version;
3. load the local checkpoint with the Transformers backend and remote code
   explicitly trusted;
4. verify the backend class is `Transformers...`, not ordinary eager PyTorch;
5. compare prompt logits/generation with the Hugging Face reference;
6. exercise one- and multi-request serving through the OpenAI-compatible API;
7. record GPU memory, prefill/decode speed, and exact package versions.

Do not turn on the full H100 cluster for this. The first runtime gate should use
minutes, not days.

## MLX and GGUF gates

The native MLX adapter, trained 150M conversion/quantization gate, and reduced-
depth final-width 4B topology gate now pass. The latter preserves the 3,072
hidden width, 8,384 MLP width, 64 query heads, eight key/value heads, 64-wide
head dimension, 65,536-token vocabulary, and 4,096 context contract while
reducing only depth from 36 layers to one. It achieves 60/60 prompt-position
top-1 agreement and 16/16 exact cached generated tokens on the M5 Mac. See
[`mlx-final-topology-gate-v0.1.md`](mlx-final-topology-gate-v0.1.md).

GGUF remains unimplemented. It needs a real `LumiForCausalLM` converter and
llama.cpp architecture mapping before Metal generation can be claimed. The
wider-layout fallback must not change merely because GGUF support is pending.

Official implementation references:

- [Hugging Face custom models](https://huggingface.co/docs/transformers/custom_models)
- [vLLM Transformers backend and custom models](https://docs.vllm.ai/en/latest/models/supported_models/)
- [MLX-LM](https://github.com/ml-explore/mlx-lm)
- [llama.cpp model-architecture guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md)
